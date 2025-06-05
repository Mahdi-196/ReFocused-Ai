"""
Checkpoint utilities for saving model checkpoints to Google Cloud Storage
"""

import os
import torch
import subprocess
import threading
import tempfile
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
        
        try:
            self.client = storage.Client()
            self.bucket = self.client.bucket(bucket_name)
            print(f"Initialized authenticated GCS client for bucket: {bucket_name}")
        except Exception as e:
            print(f"⚠️ Failed to initialize GCS client in __init__: {e}")
            print("   Uploads might fail. Check GCS authentication outside CheckpointManager.")
            self.client = None
            self.bucket = None
    
    def _ensure_client(self):
        """Lazy initialization or re-initialization of GCS client."""
        if self.client is None or self.bucket is None:
            try:
                print("🔄 Attempting to (re)initialize GCS client...")
                self.client = storage.Client()
                self.bucket = self.client.bucket(self.bucket_name)
                print(f"✅ GCS client (re)initialized for bucket: {self.bucket_name}")
            except Exception as e:
                print(f"❌ Failed to (re)initialize GCS client: {e}")
                # self.client and self.bucket remain None or in their previous state
    
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
        """Create tarball in the calling thread, then upload tarball in a background thread."""
        print(f"🚀 Preparing background upload for {checkpoint_name}")
        
        tar_path = f"{local_dir}.tar.gz"
        try:
            print(f"Creating tar archive: {tar_path} from {local_dir}")
            # Ensure the local_dir actually exists before tarring
            if not os.path.isdir(local_dir):
                print(f"❌ Source directory for tar does not exist: {local_dir}")
                return

            tar_result = subprocess.run(
                ["tar", "czf", tar_path, "-C", os.path.dirname(local_dir), os.path.basename(local_dir)],
                capture_output=True, text=True, check=False  # Changed to check=False to inspect result
            )
            if tar_result.returncode != 0:
                print(f"❌ Failed to create tar archive for {checkpoint_name}. Stderr: {tar_result.stderr}")
                if os.path.exists(tar_path): 
                    os.remove(tar_path)  # Clean up partial tar
                return
            print(f"✅ Tar archive created for {checkpoint_name}")

        except Exception as e:  # Catch other exceptions like FileNotFoundError for tar command
            print(f"❌ Exception during tar creation for {checkpoint_name}: {e}")
            if os.path.exists(tar_path): 
                os.remove(tar_path)
            return

        self._cleanup_upload_processes()  # Clean up list of finished threads
        
        upload_thread = threading.Thread(
            target=self._background_upload_tar_worker,  # New worker function
            args=(tar_path, checkpoint_name),          # Pass tar_path
            daemon=True
        )
        upload_thread.start()
        self.upload_processes.append(upload_thread)
        
        print(f"✅ Checkpoint tarball {checkpoint_name}.tar.gz queued for background upload")
    
    def _background_upload_tar_worker(self, tar_path: str, checkpoint_name: str):
        """Background worker to upload a pre-made tarball using gsutil and then delete the local tarball."""
        temp_boto_file = None  # Initialize
        try:
            gsutil_executable_path = shutil.which("gsutil")
            if gsutil_executable_path is None:
                print(f"⚠️ gsutil not found. Cannot perform background upload for {tar_path}.")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                    print(f"🗑️ Cleaned up unused tarball {tar_path} as gsutil is not found.")
                return

            bucket_uri = f"gs://{self.bucket_name}/{self.checkpoint_path}/{checkpoint_name}.tar.gz"
            print(f"☁️ Uploading {tar_path} to {bucket_uri} using gsutil...")

            current_env = os.environ.copy()
            gac_path = current_env.get('GOOGLE_APPLICATION_CREDENTIALS')
            gcp_project = current_env.get('GOOGLE_CLOUD_PROJECT')  # Get project from env

            if gac_path is None:
                print(f"❌ THREAD ERROR: GOOGLE_APPLICATION_CREDENTIALS not found for gsutil for {checkpoint_name}!")
                # Do not delete tar_path here as upload won't be attempted
                return

            # --- START: Create and use temporary Boto config ---
            boto_config_content = f"[Credentials]\ngs_service_key_file = {gac_path}\n\n"
            if gcp_project:  # Only add project_id if it's set
                boto_config_content += f"[GSUtil]\ndefault_project_id = {gcp_project}\n"
            
            # Create a named temporary file to store the Boto config
            # Delete=False is important on some OSes when passing the name to a subprocess
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as tf:
                tf.write(boto_config_content)
                temp_boto_file = tf.name  # Get the path to the temporary file
            
            current_env['BOTO_CONFIG'] = temp_boto_file
            print(f"🔧 Using temporary Boto config for gsutil: {temp_boto_file}")
            # --- END: Create and use temporary Boto config ---

            upload_result = subprocess.run(
                [gsutil_executable_path, "-m", "cp", tar_path, bucket_uri],  # You can keep -m if preferred
                capture_output=True, text=True, env=current_env
            )

            if upload_result.returncode == 0:
                print(f"✅ Successfully uploaded {checkpoint_name}.tar.gz")
                if os.path.exists(tar_path):
                    os.remove(tar_path)
                    print(f"🗑️ Cleaned up uploaded tarball {tar_path}")
            else:
                print(f"❌ Failed to upload {checkpoint_name}.tar.gz using gsutil. Stderr: {upload_result.stderr}")
                # Keep tar_path for debugging if upload failed

        except Exception as e:
            print(f"❌ Exception in background tarball upload worker for {checkpoint_name}: {e}")
            # Keep tar_path for debugging
        finally:
            # --- START: Clean up temporary Boto config ---
            if temp_boto_file and os.path.exists(temp_boto_file):
                os.remove(temp_boto_file)
                print(f"🗑️ Cleaned up temporary Boto config: {temp_boto_file}")
            # --- END: Clean up temporary Boto config ---
    
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
        """Upload checkpoint directory to GCS (synchronous fallback)."""
        print(f"SYNC UPLOAD: Uploading checkpoint to gs://{self.bucket_name}/{self.checkpoint_path}/{checkpoint_name}")
        self._ensure_client()  # Make sure client is initialized
        if self.client is None or self.bucket is None:
            print("❌ SYNC UPLOAD: GCS client not available. Skipping upload.")
            return

        try:
            # Test client connection
            _ = list(self.bucket.list_blobs(max_results=1))  # More reliable check than bucket.exists() for some permissions
            print("✅ SYNC UPLOAD: GCS client authenticated and bucket accessible.")
        except Exception as auth_error:
            print(f"❌ SYNC UPLOAD: GCS authentication/access failed: {auth_error}")
            return
            
        uploaded_files = 0
        total_files = sum(len(files) for _, _, files in os.walk(local_dir))
        print(f"📁 SYNC UPLOAD: Uploading {total_files} files...")
        
        for root, _, files in os.walk(local_dir):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(local_file_path, local_dir)
                gcs_path = f"{self.checkpoint_path}/{checkpoint_name}/{relative_path}"
                try:
                    blob = self.bucket.blob(gcs_path)
                    blob.upload_from_filename(local_file_path)
                    uploaded_files += 1
                    if uploaded_files % 10 == 0 or uploaded_files == total_files:
                        print(f"  📤 SYNC UPLOAD: Uploaded {uploaded_files}/{total_files} files")
                except Exception as upload_error:
                    print(f"❌ SYNC UPLOAD: Failed to upload {relative_path}: {upload_error}")
                    # Decide if you want to stop all uploads on first error or continue
                    return  # Stopping on first error for now
        print(f"✅ SYNC UPLOAD: Checkpoint uploaded successfully ({uploaded_files} files)")
    
    def _cleanup_old_checkpoints(self, keep_last: int = 3):  # Default keep_last = 3
        """Remove old local checkpoint directories to save disk space."""
        # Ensure local_dir exists
        if not os.path.isdir(self.local_dir):
            return

        checkpoints = sorted([
            d for d in os.listdir(self.local_dir) 
            if os.path.isdir(os.path.join(self.local_dir, d)) and d.startswith('checkpoint-')
        ])
        
        if len(checkpoints) > keep_last:
            checkpoints_to_delete = checkpoints[:-keep_last]  # These are the oldest ones
            for checkpoint_name_to_delete in checkpoints_to_delete:
                checkpoint_path_to_delete = os.path.join(self.local_dir, checkpoint_name_to_delete)
                print(f"🗑️ Removing old checkpoint directory: {checkpoint_path_to_delete}")
                try:
                    shutil.rmtree(checkpoint_path_to_delete)
                except Exception as e:
                    print(f"⚠️ Failed to remove {checkpoint_path_to_delete}: {e}")
    
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
        """Download checkpoint from GCS using Python client (avoids gsutil authentication issues)"""
        # Ensure GCS client is available
        self._ensure_client()
        if self.client is None or self.bucket is None:
            print("❌ GCS client not available for download. Cannot proceed.")
            return

        # Try downloading tar.gz first (background upload format)
        tar_path = f"{local_dir}.tar.gz"
        tar_blob_name = f"{self.checkpoint_path}/{checkpoint_name}.tar.gz"
        
        try:
            print(f"📥 Attempting to download {tar_blob_name} using Python GCS client...")
            tar_blob = self.bucket.blob(tar_blob_name)
            
            if tar_blob.exists():
                print(f"📦 Downloading {tar_blob_name} to {tar_path}")
                tar_blob.download_to_filename(tar_path)
                
                # Extract tar.gz
                print(f"📦 Extracting {tar_path}")
                subprocess.run([
                    "tar", "xzf", tar_path, 
                    "-C", os.path.dirname(local_dir)
                ], check=True)
                os.remove(tar_path)
                print(f"✅ Successfully downloaded and extracted {checkpoint_name}")
                return
            else:
                print(f"⚠️ Tar.gz format not found for {checkpoint_name}, trying directory format...")
        
        except Exception as e:
            print(f"⚠️ Failed to download tar.gz format: {e}. Trying directory format...")
            if os.path.exists(tar_path):
                os.remove(tar_path)
        
        # Fallback to directory-based download using Python GCS client
        print(f"📁 Downloading checkpoint directory format for {checkpoint_name}...")
        os.makedirs(local_dir, exist_ok=True)
        
        prefix = f"{self.checkpoint_path}/{checkpoint_name}/"
        try:
            blobs = list(self.bucket.list_blobs(prefix=prefix))
            
            if not blobs:
                print(f"❌ No files found for checkpoint {checkpoint_name} at {prefix}")
                return
            
            downloaded_files = 0
            print(f"📁 Found {len(blobs)} files to download...")
            
            for blob in blobs:
                # Create local file path
                relative_path = blob.name[len(prefix):]
                if not relative_path:  # Skip if it's just the prefix (directory marker)
                    continue
                    
                local_file_path = os.path.join(local_dir, relative_path)
                
                # Create directory if needed
                local_file_dir = os.path.dirname(local_file_path)
                if local_file_dir:
                    os.makedirs(local_file_dir, exist_ok=True)
                
                # Download file
                blob.download_to_filename(local_file_path)
                downloaded_files += 1
                
                if downloaded_files % 10 == 0 or downloaded_files == len(blobs):
                    print(f"  📥 Downloaded {downloaded_files}/{len(blobs)} files")
            
            print(f"✅ Successfully downloaded {checkpoint_name} directory format ({downloaded_files} files)")
            
        except Exception as e:
            print(f"❌ Failed to download checkpoint directory: {e}")
            raise
    
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