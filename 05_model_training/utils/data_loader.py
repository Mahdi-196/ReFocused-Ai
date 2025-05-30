"""
Efficient Data Loading for H100 Training
Handles GCS downloads, local caching, and optimized data pipeline
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from google.cloud import storage
import logging
from typing import List, Dict, Optional, Generator
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class TokenizedDataset(IterableDataset):
    """Efficient dataset for tokenized .npz files with streaming support"""
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int = 2048,
        cache_size: int = 10,  # Number of files to keep in memory
        prefetch_factor: int = 2,
        npz_key_priority: Optional[List[str]] = None # Added for flexible key selection
    ):
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.cache_size = cache_size
        self.prefetch_factor = prefetch_factor
        self.npz_key_priority = npz_key_priority if npz_key_priority else ['input_ids', 'arr_0', 'text', 'sequences']
        
        # Find all .npz files
        self.file_paths = list(self.data_dir.glob("*.npz"))
        self.file_paths.sort()  # Ensure consistent ordering
        
        logger.info(f"Found {len(self.file_paths)} tokenized files in {data_dir}")
        
        # Memory cache for loaded files
        self._cache = {}
        self._cache_order = []
        self._cache_lock = threading.Lock()
        
    def _load_file_data(self, file_path: Path) -> np.ndarray:
        """Load and cache file data"""
        file_key = str(file_path)
        
        with self._cache_lock:
            # Check if already in cache
            if file_key in self._cache:
                # Move to end (most recently used)
                self._cache_order.remove(file_key)
                self._cache_order.append(file_key)
                return self._cache[file_key]
            
            # Load file
            try:
                sequences = None
                # Attempt 1: Load with allow_pickle=False
                try:
                    data = np.load(file_path, allow_pickle=False)
                except ValueError as e:
                    if "allow_pickle=True" in str(e):
                        logger.warning(f"Failed to load {file_path} with allow_pickle=False, retrying with allow_pickle=True. Error: {e}")
                        data = np.load(file_path, allow_pickle=True)
                    else:
                        raise # Re-raise if it's a different ValueError
                
                # Key selection logic
                found_key = None
                for key_candidate in self.npz_key_priority:
                    if key_candidate in data:
                        found_key = key_candidate
                        break
                
                if found_key:
                    sequences = data[found_key]
                elif data.files: # Fallback to the first key if no priority keys found
                    logger.debug(f"Priority keys not found in {file_path.name}. Using first available key: {data.files[0]}")
                    sequences = data[data.files[0]]
                else:
                    logger.error(f"No data found in {file_path.name}. Skipping file.")
                    return np.array([]) # Return empty array to skip

                if not isinstance(sequences, np.ndarray):
                    logger.warning(f"Data for key '{found_key or data.files[0]}' in {file_path.name} is not a numpy array. Type: {type(sequences)}. Skipping file.")
                    return np.array([])

                # Add to cache
                self._cache[file_key] = sequences
                self._cache_order.append(file_key)
                
                # Evict old files if cache is full
                while len(self._cache_order) > self.cache_size:
                    old_key = self._cache_order.pop(0)
                    del self._cache[old_key]
                
                logger.debug(f"Loaded file {file_path.name} with {len(sequences)} sequences")
                return sequences
                
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {e}")
                return np.array([])
    
    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """Iterate through all sequences in all files"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single worker
            file_list = self.file_paths
        else:
            # Multi-worker: split files among workers
            per_worker = len(self.file_paths) // worker_info.num_workers
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            
            if worker_id == worker_info.num_workers - 1:
                # Last worker gets remaining files
                end_idx = len(self.file_paths)
            else:
                end_idx = start_idx + per_worker
            
            file_list = self.file_paths[start_idx:end_idx]
            logger.info(f"Worker {worker_id}: processing files {start_idx} to {end_idx-1}")
        
        for file_path in file_list:
            sequences = self._load_file_data(file_path)
            
            if len(sequences) == 0:
                continue
                
            # Yield individual sequences
            for sequence in sequences:
                if len(sequence) >= self.sequence_length:
                    # Truncate to exact length
                    input_ids = sequence[:self.sequence_length]
                    
                    yield {
                        'input_ids': torch.tensor(input_ids, dtype=torch.long),
                        'labels': torch.tensor(input_ids, dtype=torch.long),  # For causal LM
                        'attention_mask': torch.ones(self.sequence_length, dtype=torch.long)
                    }


class GCSDataManager:
    """Manages data synchronization with Google Cloud Storage"""
    
    def __init__(self, bucket_name: str, remote_path: str, local_path: str, use_gcs: bool = True, gcs_read_client_type: str = 'default'):
        self.bucket_name = bucket_name
        self.remote_path = remote_path.rstrip('/')
        self.local_path = Path(local_path)
        self.use_gcs = use_gcs
        
        self.read_client = None
        self.read_bucket = None
        self.write_client = None
        self.write_bucket = None
        
        if self.use_gcs:
            self.local_path.mkdir(parents=True, exist_ok=True)
            # Setup read client (can be anonymous or default)
            try:
                if gcs_read_client_type == 'anonymous':
                    logger.info("Using anonymous GCS client for read operations.")
                    self.read_client = storage.Client.create_anonymous_client()
                else:
                    logger.info("Using default GCS client for read operations (requires authentication if not public).")
                    self.read_client = storage.Client()
                
                self.read_bucket = self.read_client.bucket(bucket_name)
                logger.info(f"Read client connected to GCS bucket: {bucket_name} for {gcs_read_client_type} access.")
            except Exception as e:
                logger.error(f"Failed to initialize GCS read client (type: {gcs_read_client_type}): {e}")
                logger.warning("GCS read operations will be disabled.")
                # If read client fails, we might still want to allow writes if write client succeeds,
                # but for now, let's consider read essential for sync.
                # self.use_gcs = False # Or handle more granularly

            # Setup write client (always attempts authenticated)
            try:
                logger.info("Attempting to initialize authenticated GCS client for write operations.")
                self.write_client = storage.Client() # Standard client for writes
                self.write_bucket = self.write_client.bucket(bucket_name)
                logger.info(f"Write client connected to GCS bucket: {bucket_name} for authenticated access.")
            except Exception as e:
                logger.error(f"Failed to initialize GCS write client: {e}")
                logger.warning("GCS write operations (e.g., checkpoint uploads) will be disabled. Ensure credentials are set up if uploads are needed.")
                # self.write_client and self.write_bucket will remain None

            # If neither client could be initialized, disable GCS fully for safety
            if not self.read_client and not self.write_client:
                logger.warning("Both GCS read and write clients failed to initialize. Disabling all GCS operations.")
                self.use_gcs = False
        else:
            logger.info("GCS usage is disabled by configuration. Local data will be used.")
            self.local_path.mkdir(parents=True, exist_ok=True)
    
    def sync_data_to_local(self, max_workers: int = 8) -> bool:
        """Download data from GCS to local storage with parallel downloads using the read_client"""
        if not self.use_gcs or not self.read_client or not self.read_bucket:
            logger.info("Skipping GCS sync: GCS is not enabled or read client not initialized.")
            # If GCS is meant to be used but read client failed, this should ideally be an error or handled based on strictness.
            # For now, returning True means "no sync needed/attempted from this method's perspective if client is missing".
            return True 
            
        logger.info(f"Syncing data from gs://{self.bucket_name}/{self.remote_path} to {self.local_path} using read client.")
        
        # List remote files using read_bucket
        blobs = list(self.read_bucket.list_blobs(prefix=self.remote_path))
        npz_blobs = [b for b in blobs if b.name.endswith('.npz')]
        
        logger.info(f"Found {len(npz_blobs)} .npz files to download")
        
        def download_blob(blob):
            """Download a single blob"""
            try:
                # Get relative path from remote_path
                relative_path = blob.name[len(self.remote_path):].lstrip('/')
                local_file_path = self.local_path / relative_path
                
                # Create parent directories
                local_file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Skip if file exists and has same size
                if local_file_path.exists():
                    if local_file_path.stat().st_size == blob.size:
                        logger.debug(f"Skipping {relative_path} (already exists)")
                        return True
                
                # Download file
                logger.info(f"Downloading {relative_path} ({blob.size / 1024 / 1024:.1f} MB)")
                blob.download_to_filename(str(local_file_path))
                return True
                
            except Exception as e:
                logger.error(f"Failed to download {blob.name}: {e}")
                return False
        
        # Download files in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(download_blob, npz_blobs))
        
        success_count = sum(results)
        duration = time.time() - start_time
        
        logger.info(f"Downloaded {success_count}/{len(npz_blobs)} files in {duration:.1f}s")
        return success_count == len(npz_blobs)
    
    def upload_checkpoint(self, local_checkpoint_dir: str, remote_checkpoint_path: str):
        """Upload checkpoint to GCS using the write_client"""
        if not self.use_gcs or not self.write_client or not self.write_bucket:
            logger.warning(f"Skipping checkpoint upload to GCS: GCS is not enabled or write client not initialized for bucket {self.bucket_name}.")
            return False

        local_dir = Path(local_checkpoint_dir)
        
        if not local_dir.exists():
            logger.warning(f"Checkpoint directory {local_dir} does not exist")
            return False
        
        logger.info(f"Uploading checkpoint to gs://{self.bucket_name}/{remote_checkpoint_path}")
        
        try:
            for file_path in local_dir.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_dir)
                    blob_name = f"{remote_checkpoint_path}/{relative_path}"
                    
                    blob = self.write_bucket.blob(blob_name)
                    blob.upload_from_filename(str(file_path))
                    logger.debug(f"Uploaded {relative_path}")
            
            logger.info("Checkpoint upload completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload checkpoint: {e}")
            return False


def create_dataloader(
    data_dir: str,
    batch_size: int,
    sequence_length: int = 2048,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    npz_key_priority: Optional[List[str]] = None # Added
) -> DataLoader:
    """Create optimized DataLoader for training"""
    
    dataset = TokenizedDataset(
        data_dir=data_dir,
        sequence_length=sequence_length,
        cache_size=num_workers * 2,  # Cache more files with more workers
        prefetch_factor=prefetch_factor,
        npz_key_priority=npz_key_priority # Pass through
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True  # Ensure consistent batch sizes for distributed training
    )


def estimate_training_time(
    total_tokens: int,
    tokens_per_step: int,
    steps_per_second: float
) -> Dict[str, float]:
    """Estimate training time based on dataset size and throughput"""
    
    total_steps = total_tokens // tokens_per_step
    total_seconds = total_steps / steps_per_second
    
    return {
        'total_steps': total_steps,
        'total_hours': total_seconds / 3600,
        'total_days': total_seconds / (3600 * 24),
        'estimated_cost_8h100': (total_seconds / 3600) * 7.92  # $7.92/hour for 8xH100
    }


if __name__ == "__main__":
    # Test data loading
    logging.basicConfig(level=logging.INFO)
    
    # Test with your dataset
    data_dir = "/scratch/shards"
    
    if Path(data_dir).exists():
        dataloader = create_dataloader(
            data_dir=data_dir,
            batch_size=4,
            num_workers=2
        )
        
        print("Testing data loader...")
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: {batch['input_ids'].shape}")
            if i >= 2:  # Test first few batches
                break
    else:
        print(f"Data directory {data_dir} not found") 