"""
Checkpoint Management for Efficient Training
Handles saving, loading, and backup of training checkpoints
"""

import os
import shutil
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import json

from google.cloud import storage

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages training checkpoints with automatic backup and cleanup"""
    
    def __init__(
        self,
        output_dir: str,
        remote_bucket: str,
        remote_path: str,
        backup_every_n_steps: int = 1000,
        keep_last_n: int = 5,
        async_backup: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.remote_bucket = remote_bucket
        self.remote_path = remote_path.rstrip('/')
        self.backup_every_n_steps = backup_every_n_steps
        self.keep_last_n = keep_last_n
        self.async_backup = async_backup
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS client
        try:
            self.gcs_client = storage.Client()
            self.bucket = self.gcs_client.bucket(remote_bucket)
            logger.info(f"Connected to GCS bucket: {remote_bucket}")
        except Exception as e:
            logger.error(f"Failed to connect to GCS: {e}")
            raise
        
        # Background upload queue
        self.upload_queue = []
        self.upload_thread = None
        self.upload_running = False
        
        if async_backup:
            self._start_upload_thread()
    
    def _start_upload_thread(self):
        """Start background upload thread"""
        self.upload_running = True
        self.upload_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()
        logger.info("Background checkpoint upload thread started")
    
    def _upload_worker(self):
        """Background worker for uploading checkpoints"""
        while self.upload_running:
            if self.upload_queue:
                checkpoint_dir = self.upload_queue.pop(0)
                try:
                    self._upload_checkpoint_sync(checkpoint_dir)
                except Exception as e:
                    logger.error(f"Failed to upload checkpoint {checkpoint_dir}: {e}")
            else:
                time.sleep(1)  # Wait for new uploads
    
    def _upload_checkpoint_sync(self, local_checkpoint_dir: str) -> bool:
        """Synchronously upload checkpoint to GCS"""
        local_dir = Path(local_checkpoint_dir)
        
        if not local_dir.exists():
            logger.warning(f"Checkpoint directory {local_dir} does not exist")
            return False
        
        checkpoint_name = local_dir.name
        remote_checkpoint_path = f"{self.remote_path}/{checkpoint_name}"
        
        logger.info(f"Uploading checkpoint {checkpoint_name} to GCS...")
        start_time = time.time()
        uploaded_files = 0
        total_size = 0
        
        try:
            # Upload all files in the checkpoint directory
            for file_path in local_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_dir)
                    blob_name = f"{remote_checkpoint_path}/{relative_path}"
                    
                    blob = self.bucket.blob(blob_name)
                    
                    # Check if file already exists with same size
                    try:
                        blob.reload()
                        if blob.size == file_path.stat().st_size:
                            logger.debug(f"Skipping {relative_path} (already exists)")
                            continue
                    except:
                        pass  # File doesn't exist, continue with upload
                    
                    blob.upload_from_filename(str(file_path))
                    uploaded_files += 1
                    total_size += file_path.stat().st_size
                    
                    logger.debug(f"Uploaded {relative_path}")
            
            duration = time.time() - start_time
            size_mb = total_size / 1024**2
            
            logger.info(f"Checkpoint upload completed: {uploaded_files} files, "
                       f"{size_mb:.1f} MB in {duration:.1f}s "
                       f"({size_mb/duration:.1f} MB/s)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload checkpoint {checkpoint_name}: {e}")
            return False
    
    def backup_checkpoint(self, checkpoint_dir: str) -> bool:
        """Backup checkpoint to GCS (async if enabled)"""
        if self.async_backup:
            # Add to upload queue
            self.upload_queue.append(checkpoint_dir)
            logger.info(f"Queued checkpoint {Path(checkpoint_dir).name} for upload")
            return True
        else:
            # Synchronous upload
            return self._upload_checkpoint_sync(checkpoint_dir)
    
    def download_checkpoint(self, checkpoint_name: str, local_dir: Optional[str] = None) -> str:
        """Download checkpoint from GCS"""
        if local_dir is None:
            local_dir = self.output_dir / checkpoint_name
        else:
            local_dir = Path(local_dir)
        
        local_dir.mkdir(parents=True, exist_ok=True)
        
        remote_checkpoint_path = f"{self.remote_path}/{checkpoint_name}"
        
        logger.info(f"Downloading checkpoint {checkpoint_name} from GCS...")
        start_time = time.time()
        downloaded_files = 0
        total_size = 0
        
        try:
            # List all blobs with the checkpoint prefix
            blobs = list(self.bucket.list_blobs(prefix=remote_checkpoint_path))
            
            if not blobs:
                raise ValueError(f"No checkpoint found at {remote_checkpoint_path}")
            
            # Download files in parallel
            def download_blob(blob):
                try:
                    # Calculate local file path
                    relative_path = blob.name[len(remote_checkpoint_path):].lstrip('/')
                    local_file_path = local_dir / relative_path
                    
                    # Create parent directories
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file
                    blob.download_to_filename(str(local_file_path))
                    return blob.size
                except Exception as e:
                    logger.error(f"Failed to download {blob.name}: {e}")
                    return 0
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                sizes = list(executor.map(download_blob, blobs))
            
            downloaded_files = len([s for s in sizes if s > 0])
            total_size = sum(sizes)
            
            duration = time.time() - start_time
            size_mb = total_size / 1024**2
            
            logger.info(f"Checkpoint download completed: {downloaded_files} files, "
                       f"{size_mb:.1f} MB in {duration:.1f}s")
            
            return str(local_dir)
            
        except Exception as e:
            logger.error(f"Failed to download checkpoint {checkpoint_name}: {e}")
            raise
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints in GCS"""
        try:
            # List all checkpoint directories
            blobs = list(self.bucket.list_blobs(prefix=self.remote_path, delimiter='/'))
            
            checkpoints = []
            for prefix in blobs.prefixes:
                checkpoint_name = prefix.rstrip('/').split('/')[-1]
                
                # Get metadata if available
                metadata_blob_name = f"{prefix}training_metadata.json"
                try:
                    metadata_blob = self.bucket.blob(metadata_blob_name)
                    metadata_content = metadata_blob.download_as_text()
                    metadata = json.loads(metadata_content)
                except:
                    metadata = {}
                
                checkpoints.append({
                    'name': checkpoint_name,
                    'path': prefix,
                    'metadata': metadata
                })
            
            # Sort by step number if available
            checkpoints.sort(key=lambda x: x['metadata'].get('global_step', 0))
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    def cleanup_old_checkpoints(self):
        """Remove old local checkpoints to save disk space"""
        try:
            # Get all checkpoint directories
            checkpoint_dirs = []
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.startswith('checkpoint_step_'):
                    try:
                        step = int(item.name.split('_')[-1])
                        checkpoint_dirs.append((step, item))
                    except ValueError:
                        continue
            
            # Sort by step number
            checkpoint_dirs.sort(key=lambda x: x[0])
            
            # Keep only the last N checkpoints
            if len(checkpoint_dirs) > self.keep_last_n:
                to_remove = checkpoint_dirs[:-self.keep_last_n]
                
                for step, checkpoint_dir in to_remove:
                    logger.info(f"Removing old checkpoint: {checkpoint_dir.name}")
                    shutil.rmtree(checkpoint_dir)
                
                logger.info(f"Cleaned up {len(to_remove)} old checkpoints")
        
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest local checkpoint"""
        try:
            latest_step = -1
            latest_checkpoint = None
            
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.startswith('checkpoint_step_'):
                    try:
                        step = int(item.name.split('_')[-1])
                        if step > latest_step:
                            latest_step = step
                            latest_checkpoint = str(item)
                    except ValueError:
                        continue
            
            return latest_checkpoint
        
        except Exception as e:
            logger.error(f"Failed to get latest checkpoint: {e}")
            return None
    
    def get_checkpoint_info(self, checkpoint_dir: str) -> Dict[str, Any]:
        """Get information about a checkpoint"""
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            return {}
        
        info = {
            'path': str(checkpoint_path),
            'size_mb': sum(f.stat().st_size for f in checkpoint_path.rglob('*') if f.is_file()) / 1024**2,
            'created_time': checkpoint_path.stat().st_ctime,
            'file_count': len(list(checkpoint_path.rglob('*')))
        }
        
        # Load metadata if available
        metadata_file = checkpoint_path / 'training_metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    info['metadata'] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint metadata: {e}")
        
        return info
    
    def verify_checkpoint_integrity(self, checkpoint_dir: str) -> bool:
        """Verify checkpoint integrity"""
        checkpoint_path = Path(checkpoint_dir)
        
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint directory {checkpoint_path} does not exist")
            return False
        
        # Check for required DeepSpeed files
        required_files = [
            'latest',  # DeepSpeed checkpoint marker
            'zero_pp_rank_0_mp_rank_00_optim_states.pt',  # Optimizer states
        ]
        
        missing_files = []
        for file_name in required_files:
            if not (checkpoint_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            logger.error(f"Checkpoint {checkpoint_path.name} is missing files: {missing_files}")
            return False
        
        logger.info(f"Checkpoint {checkpoint_path.name} integrity verified")
        return True
    
    def stop(self):
        """Stop the checkpoint manager and cleanup"""
        if self.upload_running:
            self.upload_running = False
            
            # Wait for pending uploads to complete
            if self.upload_queue:
                logger.info(f"Waiting for {len(self.upload_queue)} pending uploads to complete...")
                while self.upload_queue and self.upload_thread.is_alive():
                    time.sleep(1)
            
            if self.upload_thread:
                self.upload_thread.join(timeout=60)  # Wait up to 1 minute
            
            logger.info("Checkpoint manager stopped")


def estimate_checkpoint_size(model_params: int, optimizer_states: bool = True) -> float:
    """Estimate checkpoint size in GB"""
    # Model parameters (fp16)
    model_size_gb = model_params * 2 / 1024**3
    
    # Optimizer states (AdamW: 2x model params for momentum + variance)
    if optimizer_states:
        optimizer_size_gb = model_params * 2 * 4 / 1024**3  # fp32
    else:
        optimizer_size_gb = 0
    
    # Add some overhead for metadata, etc.
    total_size_gb = (model_size_gb + optimizer_size_gb) * 1.2
    
    return total_size_gb


if __name__ == "__main__":
    # Test checkpoint manager
    logging.basicConfig(level=logging.INFO)
    
    # Test with dummy parameters
    checkpoint_manager = CheckpointManager(
        output_dir="/tmp/test_checkpoints",
        remote_bucket="test-bucket",
        remote_path="test_checkpoints",
        keep_last_n=3
    )
    
    # Test checkpoint info
    print(f"Estimated 1B param checkpoint size: {estimate_checkpoint_size(1_000_000_000):.1f} GB") 