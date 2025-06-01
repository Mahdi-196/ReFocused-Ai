"""
Checkpoint utilities for saving model checkpoints to Google Cloud Storage
"""

import os
import torch
from google.cloud import storage
from typing import Optional, Dict, Any
import json
from datetime import datetime
import shutil


class CheckpointManager:
    """Manages model checkpoints and uploads to GCS"""
    
    def __init__(self, 
                 bucket_name: str,
                 checkpoint_path: str,
                 local_dir: str = "./checkpoints"):
        self.bucket_name = bucket_name
        self.checkpoint_path = checkpoint_path
        self.local_dir = local_dir
        os.makedirs(local_dir, exist_ok=True)
        
        # Initialize GCS client with credentials if available
        try:
            self.client = storage.Client()
        except:
            # Fall back to anonymous client if no credentials
            print("Using anonymous GCS client - uploads may fail")
            self.client = storage.Client.create_anonymous_client()
        
        self.bucket = self.client.bucket(bucket_name)
    
    def save_checkpoint(self,
                       accelerator,
                       model,
                       optimizer,
                       scheduler,
                       epoch: int,
                       step: int,
                       files_processed: int,
                       best_loss: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Save checkpoint locally and upload to GCS"""
        
        # Create checkpoint name
        checkpoint_name = f"checkpoint-epoch{epoch}-step{step}-files{files_processed}"
        local_checkpoint_dir = os.path.join(self.local_dir, checkpoint_name)
        
        # Save using accelerator (handles FSDP state dict)
        print(f"Saving checkpoint to {local_checkpoint_dir}")
        accelerator.save_state(local_checkpoint_dir)
        
        # Save additional metadata
        metadata_dict = {
            'epoch': epoch,
            'step': step,
            'files_processed': files_processed,
            'best_loss': best_loss,
            'timestamp': datetime.now().isoformat(),
            'model_config': metadata.get('model_config', {}) if metadata else {},
            'training_config': metadata.get('training_config', {}) if metadata else {}
        }
        
        metadata_path = os.path.join(local_checkpoint_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        
        # Upload to GCS if on main process
        if accelerator.is_main_process:
            self._upload_to_gcs(local_checkpoint_dir, checkpoint_name)
            
            # Clean up old local checkpoints to save space
            self._cleanup_old_checkpoints()
        
        return checkpoint_name
    
    def _upload_to_gcs(self, local_dir: str, checkpoint_name: str):
        """Upload checkpoint directory to GCS"""
        print(f"Uploading checkpoint to gs://{self.bucket_name}/{self.checkpoint_path}/{checkpoint_name}")
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # Create relative path for GCS
                relative_path = os.path.relpath(local_file_path, local_dir)
                gcs_path = f"{self.checkpoint_path}/{checkpoint_name}/{relative_path}"
                
                # Upload file
                blob = self.bucket.blob(gcs_path)
                blob.upload_from_filename(local_file_path)
        
        print(f"Checkpoint uploaded successfully")
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old local checkpoints to save disk space"""
        checkpoints = sorted([
            d for d in os.listdir(self.local_dir) 
            if os.path.isdir(os.path.join(self.local_dir, d)) and d.startswith('checkpoint-')
        ])
        
        if len(checkpoints) > keep_last:
            for checkpoint in checkpoints[:-keep_last]:
                checkpoint_path = os.path.join(self.local_dir, checkpoint)
                print(f"Removing old checkpoint: {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
    
    def load_checkpoint(self, accelerator, checkpoint_name: str):
        """Load checkpoint from GCS"""
        local_checkpoint_dir = os.path.join(self.local_dir, checkpoint_name)
        
        # Download from GCS if not exists locally
        if not os.path.exists(local_checkpoint_dir):
            print(f"Downloading checkpoint from GCS...")
            self._download_from_gcs(checkpoint_name, local_checkpoint_dir)
        
        # Load checkpoint
        print(f"Loading checkpoint from {local_checkpoint_dir}")
        accelerator.load_state(local_checkpoint_dir)
        
        # Load metadata
        metadata_path = os.path.join(local_checkpoint_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        
        return None
    
    def _download_from_gcs(self, checkpoint_name: str, local_dir: str):
        """Download checkpoint from GCS"""
        os.makedirs(local_dir, exist_ok=True)
        
        prefix = f"{self.checkpoint_path}/{checkpoint_name}/"
        blobs = self.bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            # Create local file path
            relative_path = blob.name[len(prefix):]
            local_file_path = os.path.join(local_dir, relative_path)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download file
            blob.download_to_filename(local_file_path)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in GCS"""
        prefix = f"{self.checkpoint_path}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            return None
        
        # Find checkpoint directories
        checkpoint_names = set()
        for blob in blobs:
            parts = blob.name[len(prefix):].split('/')
            if parts[0].startswith('checkpoint-'):
                checkpoint_names.add(parts[0])
        
        if not checkpoint_names:
            return None
        
        # Sort by step number (extract from checkpoint name)
        def get_step(name):
            try:
                return int(name.split('-step')[1].split('-')[0])
            except:
                return 0
        
        latest = sorted(checkpoint_names, key=get_step)[-1]
        return latest 