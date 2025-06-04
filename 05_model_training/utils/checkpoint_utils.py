"""
Checkpoint utilities for saving model checkpoints to Google Cloud Storage
"""

import os
import torch
import subprocess
import threading
from google.cloud import storage
from typing import Optional, Dict, Any
import json
from datetime import datetime
import shutil


class CheckpointManager:
    """Manages model checkpoints and uploads to GCS"""
    
    def __init__(self, 
                 bucket_name: str = "refocused-ai",
                 checkpoint_path: str = "Checkpoints",
                 local_dir: str = "./checkpoints",
                 background_upload: bool = True):
        self.bucket_name = bucket_name
        self.checkpoint_path = checkpoint_path
        self.local_dir = local_dir
        self.background_upload = background_upload
        os.makedirs(local_dir, exist_ok=True)
        
        # Track background upload processes
        self.upload_processes = []
        
        # Initialize authenticated GCS client with project/credentials
        # Ensures GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT are used
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        print(f"Initialized authenticated GCS client for bucket: {bucket_name}")
    
    def _ensure_client(self):
        """Lazy initialization of GCS client to avoid pickling issues"""
        if self.client is None:
            # Use authenticated client instead of anonymous
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
    
    def save_checkpoint(self,
                       accelerator,
                       model,
                       optimizer,
                       scheduler,
                       epoch: int,
                       step: int,
                       files_processed: int,
                       training_config=None,
                       current_loss: Optional[float] = None,
                       best_loss: Optional[float] = None,
                       loss_history: Optional[list] = None,
                       learning_rates: Optional[list] = None,
                       validation_metrics: Optional[Dict[str, Any]] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Save checkpoint locally and upload to GCS with comprehensive state"""
        
        # Create checkpoint name
        checkpoint_name = f"checkpoint-epoch{epoch}-step{step}-files{files_processed}"
        local_checkpoint_dir = os.path.join(self.local_dir, checkpoint_name)
        
        # Save using accelerator (handles FSDP state dict)
        print(f"Saving checkpoint to {local_checkpoint_dir}")
        accelerator.save_state(local_checkpoint_dir)
        
        # Explicitly save scheduler state if provided
        if scheduler is not None:
            scheduler_path = os.path.join(local_checkpoint_dir, 'scheduler_state.pt')
            if hasattr(scheduler, 'state_dict'):
                torch.save(scheduler.state_dict(), scheduler_path)
                print(f"✅ Saved scheduler state")
            
        # Save training configuration
        if training_config is not None:
            config_path = os.path.join(local_checkpoint_dir, 'training_config.json')
            # Convert config to dict if it's a dataclass
            if hasattr(training_config, '__dict__'):
                config_dict = training_config.__dict__
            else:
                config_dict = training_config
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            print(f"✅ Saved training config")
        
        # Prepare comprehensive metadata
        comprehensive_metadata = {
            # Basic checkpoint info
            'epoch': epoch,
            'step': step,
            'files_processed': files_processed,
            'timestamp': datetime.now().isoformat(),
            
            # Loss and metrics
            'current_loss': current_loss,
            'best_loss': best_loss,
            'loss_history': loss_history or [],
            'learning_rates': learning_rates or [],
            
            # Validation metrics
            'validation_metrics': validation_metrics or {},
            
            # Training progress
            'training_progress': {
                'completed_steps': step,
                'total_epochs': epoch,
                'files_processed': files_processed,
            },
            
            # System info
            'system_info': {
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'mixed_precision': str(accelerator.mixed_precision) if hasattr(accelerator, 'mixed_precision') else None,
            },
            
            # Model and training config (from metadata if provided)
            'model_config': metadata.get('model_config', {}) if metadata else {},
            'training_config_summary': metadata.get('training_config', {}) if metadata else {},
            
            # Additional metadata
            'additional_metadata': metadata or {}
        }
        
        # Save metadata
        metadata_path = os.path.join(local_checkpoint_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(comprehensive_metadata, f, indent=2, default=str)
        print(f"✅ Saved comprehensive metadata")
        
        # Save training metrics separately for easy access
        metrics_path = os.path.join(local_checkpoint_dir, 'training_metrics.json')
        training_metrics = {
            'step': step,
            'epoch': epoch,
            'current_loss': current_loss,
            'best_loss': best_loss,
            'loss_history': loss_history or [],
            'learning_rates': learning_rates or [],
            'validation_metrics': validation_metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2, default=str)
        print(f"✅ Saved training metrics")
        
        # Upload to GCS if on main process
        if accelerator.is_main_process:
            if self.background_upload:
                self._upload_to_gcs_background(local_checkpoint_dir, checkpoint_name)
            else:
                self._upload_to_gcs(local_checkpoint_dir, checkpoint_name)
            
            # Clean up old local checkpoints to save space
            self._cleanup_old_checkpoints()
        
        return checkpoint_name
    
    def _upload_to_gcs_background(self, local_dir: str, checkpoint_name: str):
        """Upload checkpoint directory to GCS in background using tar + gsutil"""
        print(f"🚀 Starting background upload for {checkpoint_name}")
        
        # Clean up completed processes
        self._cleanup_upload_processes()
        
        # Start background upload in a thread
        upload_thread = threading.Thread(
            target=self._background_upload_worker,
            args=(local_dir, checkpoint_name),
            daemon=True
        )
        upload_thread.start()
        self.upload_processes.append(upload_thread)
        
        print(f"✅ Checkpoint {checkpoint_name} queued for background upload")
    
    def _background_upload_worker(self, local_dir: str, checkpoint_name: str):
        """Background worker for uploading checkpoint to GCS"""
        try:
            # Check if gsutil is available (cross-platform)
            import shutil
            gsutil_path = shutil.which("gsutil")
            if gsutil_path is None:
                print(f"⚠️  gsutil not found, falling back to Python GCS client upload")
                self._upload_to_gcs(local_dir, checkpoint_name)
                return
            
            # Create tar.gz archive
            tar_path = f"{local_dir}.tar.gz"
            result = subprocess.run([
                "tar", "czf", tar_path, "-C", os.path.dirname(local_dir),
                os.path.basename(local_dir)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Failed to create tar archive: {result.stderr}")
                return
            
            # Upload using gsutil with multithreaded copy
            bucket_uri = f"gs://{self.bucket_name}/{self.checkpoint_path}/{checkpoint_name}.tar.gz"
            print(f"☁️  Uploading to {bucket_uri}")
            
            result = subprocess.run([
                "gsutil", "-m", "cp", tar_path, bucket_uri
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Successfully uploaded {checkpoint_name}")
                # Clean up the tar file after successful upload
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                    print(f"🗑️  Cleaned up {tar_path}")
            else:
                print(f"❌ Failed to upload {checkpoint_name}: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Background upload error for {checkpoint_name}: {e}")
    
    def _cleanup_upload_processes(self):
        """Remove completed upload threads from tracking"""
        self.upload_processes = [p for p in self.upload_processes if p.is_alive()]
    
    def wait_for_uploads(self):
        """Wait for all background uploads to complete"""
        if not self.upload_processes:
            return
            
        print(f"⏳ Waiting for {len(self.upload_processes)} background uploads to complete...")
        for process in self.upload_processes:
            process.join()
        self.upload_processes.clear()
        print("✅ All background uploads completed")
    
    def _upload_to_gcs(self, local_dir: str, checkpoint_name: str):
        """Upload checkpoint directory to GCS (synchronous fallback)"""
        print(f"Uploading checkpoint to gs://{self.bucket_name}/{self.checkpoint_path}/{checkpoint_name}")
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # Create relative path for GCS
                relative_path = os.path.relpath(local_file_path, local_dir)
                gcs_path = f"{self.checkpoint_path}/{checkpoint_name}/{relative_path}"
                
                # Upload file with proper prefix
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
                print(f"🗑️  Removing old checkpoint: {checkpoint_path}")
                shutil.rmtree(checkpoint_path)
    
    def load_checkpoint(self, accelerator, checkpoint_name: str, scheduler=None):
        """Load checkpoint from GCS with comprehensive state restoration"""
        local_checkpoint_dir = os.path.join(self.local_dir, checkpoint_name)
        
        # Download from GCS if not exists locally
        if not os.path.exists(local_checkpoint_dir):
            print(f"Downloading checkpoint from GCS...")
            self._download_from_gcs(checkpoint_name, local_checkpoint_dir)
        
        # Load checkpoint
        print(f"Loading checkpoint from {local_checkpoint_dir}")
        accelerator.load_state(local_checkpoint_dir)
        
        # Load scheduler state if available and scheduler provided
        if scheduler is not None:
            scheduler_path = os.path.join(local_checkpoint_dir, 'scheduler_state.pt')
            if os.path.exists(scheduler_path):
                scheduler_state = torch.load(scheduler_path, map_location='cpu')
                scheduler.load_state_dict(scheduler_state)
                print(f"✅ Restored scheduler state")
        
        # Load training configuration
        training_config = None
        config_path = os.path.join(local_checkpoint_dir, 'training_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                training_config = json.load(f)
            print(f"✅ Loaded training config")
        
        # Load comprehensive metadata
        metadata = None
        metadata_path = os.path.join(local_checkpoint_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        # Load training metrics
        training_metrics = None
        metrics_path = os.path.join(local_checkpoint_dir, 'training_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                training_metrics = json.load(f)
            print(f"✅ Loaded training metrics")
        
        return {
            'metadata': metadata,
            'training_config': training_config,
            'training_metrics': training_metrics,
            'checkpoint_info': {
                'epoch': metadata.get('epoch', 0) if metadata else 0,
                'step': metadata.get('step', 0) if metadata else 0,
                'files_processed': metadata.get('files_processed', 0) if metadata else 0,
                'best_loss': metadata.get('best_loss') if metadata else None,
                'current_loss': metadata.get('current_loss') if metadata else None,
            }
        }
    
    def _download_from_gcs(self, checkpoint_name: str, local_dir: str):
        """Download checkpoint from GCS (supports both tar.gz and directory format)"""
        # Try downloading tar.gz first (background upload format)
        tar_path = f"{local_dir}.tar.gz"
        bucket_uri = f"gs://{self.bucket_name}/{self.checkpoint_path}/{checkpoint_name}.tar.gz"
        
        result = subprocess.run([
            "gsutil", "cp", bucket_uri, tar_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Extract tar.gz
            print(f"📦 Extracting {tar_path}")
            subprocess.run([
                "tar", "xzf", tar_path, 
                "-C", os.path.dirname(local_dir)
            ])
            os.remove(tar_path)
            return
        
        # Fallback to old directory-based download
        os.makedirs(local_dir, exist_ok=True)
        
        # Use authenticated client for directory downloads
        if not hasattr(self, 'client') or self.client is None:
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
        
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
        if not hasattr(self, 'client') or self.client is None:
            self.client = storage.Client()
            self.bucket = self.client.bucket(self.bucket_name)
            
        prefix = f"{self.checkpoint_path}/"
        blobs = list(self.bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            return None
        
        # Find checkpoint directories and tar.gz files
        checkpoint_names = set()
        for blob in blobs:
            name = blob.name[len(prefix):]
            if name.endswith('.tar.gz') and name.startswith('checkpoint-'):
                # Remove .tar.gz extension
                checkpoint_names.add(name[:-7])
            else:
                parts = name.split('/')
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