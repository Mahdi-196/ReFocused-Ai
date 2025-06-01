"""
Data utilities for loading tokenized data from Google Cloud Storage
"""

import numpy as np
from google.cloud import storage
from typing import List, Optional, Iterator, Dict
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import random
import pickle
import hashlib
import json


class GCSDataLoader:
    """Handles loading tokenized data from Google Cloud Storage with preprocessing cache"""
    
    def __init__(self, bucket_name: str, cache_dir: str = "./cache", preprocess_cache_dir: str = "./preprocessed_cache"):
        self.bucket_name = bucket_name
        self.cache_dir = cache_dir
        self.preprocess_cache_dir = preprocess_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(preprocess_cache_dir, exist_ok=True)
        
        # Initialize GCS client
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Cache for flattened data to avoid repeated processing
        self.flattened_cache = {}
    
    def list_data_files(self, prefix: str = "", max_files: Optional[int] = None) -> List[str]:
        """List all tokenized .npz files in the bucket"""
        blobs = self.bucket.list_blobs(prefix=prefix)
        files = []
        
        for blob in blobs:
            if blob.name.endswith('.npz') and 'tokenized_' in blob.name:
                files.append(blob.name)
                if max_files and len(files) >= max_files:
                    break
        
        # Shuffle for better training diversity
        random.shuffle(files)
        print(f"Found {len(files)} tokenized files in bucket")
        return files
    
    def download_file(self, blob_name: str) -> str:
        """Download file from GCS to local cache"""
        local_path = os.path.join(self.cache_dir, os.path.basename(blob_name))
        
        if not os.path.exists(local_path):
            print(f"Downloading {blob_name}...")
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(local_path)
        
        return local_path
    
    def get_preprocessed_cache_path(self, file_path: str) -> str:
        """Generate cache path for preprocessed data"""
        # Create a hash of the file path for consistent naming
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:12]
        filename = f"preprocessed_{os.path.basename(file_path).replace('.npz', '')}_{file_hash}.pkl"
        return os.path.join(self.preprocess_cache_dir, filename)
    
    def load_npz_file_optimized(self, file_path: str, max_length: int = 2048, stride: int = 2048) -> Dict:
        """Load and preprocess tokenized data with caching to avoid repeated flattening"""
        
        # Check if file is already in memory cache
        if file_path in self.flattened_cache:
            return self.flattened_cache[file_path]
        
        # Check for preprocessed cache on disk
        cache_path = self.get_preprocessed_cache_path(file_path)
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    # Verify cache is still valid (check max_length and stride)
                    if (cached_data.get('max_length') == max_length and 
                        cached_data.get('stride') == stride):
                        self.flattened_cache[file_path] = cached_data
                        return cached_data
            except:
                # If cache is corrupted, remove it and reprocess
                os.remove(cache_path)
        
        # Load and preprocess the file
        print(f"Preprocessing {os.path.basename(file_path)}...")
    def load_npz_file(self, file_path: str) -> np.ndarray:
        """Load tokenized data from npz file"""
        data = np.load(file_path)
        input_ids = data['input_ids']  # Assuming tokenized data is stored as 'input_ids'
        
        # Handle the case where the NPZ file contains 2D data already
        # If the shape is [num_sequences, seq_len], flatten to 1D array
        if input_ids.ndim > 1:
            print(f"Loaded data shape before flattening: {input_ids.shape}")
            # If it's already a 2D array, keep only the necessary dimension
            # instead of squeezing which might collapse both dimensions
            if input_ids.ndim == 2:
                # If it's [num_sequences, seq_len], keep it as is
                pass
            elif input_ids.ndim == 3 and input_ids.shape[-1] == 1:
                # If it's [num_sequences, seq_len, 1], remove the last dimension
                input_ids = input_ids.squeeze(-1)
            
            # Flatten the array into a single sequence if it's 2D
            # This way we'll extract subsequences from the flattened array
            input_ids = input_ids.reshape(-1)
            print(f"Reshaped data to: {input_ids.shape}")
        
        return input_ids


class TokenizedDataset(Dataset):
    """PyTorch Dataset for tokenized data"""
    
    def __init__(self, 
                 data_loader: GCSDataLoader,
                 files: List[str],
                 max_length: int = 2048,
                 stride: int = 2048):
        self.data_loader = data_loader
        self.files = files
        self.max_length = max_length
        self.stride = stride
        
        # Build index of all sequences
        self.file_indices = []
        self.sequence_indices = []
        self.sequences_per_file = []
        
        print("Building dataset index...")
        for file_idx, file_name in enumerate(tqdm(files)):
            local_path = self.data_loader.download_file(file_name)
            tokens = self.data_loader.load_npz_file(local_path)
            
            # Create sequences with stride
            num_sequences = max(1, (len(tokens) - max_length) // stride + 1)
            self.sequences_per_file.append(num_sequences)
            
            for seq_idx in range(num_sequences):
                self.file_indices.append(file_idx)
                self.sequence_indices.append(seq_idx)
        
        print(f"Total sequences: {len(self.file_indices)}")
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        seq_idx = self.sequence_indices[idx]
        
        # Load file
        file_name = self.files[file_idx]
        local_path = self.data_loader.download_file(file_name)
        tokens = self.data_loader.load_npz_file(local_path)
        
        # Extract sequence from the flattened array
        start_idx = seq_idx * self.stride
        end_idx = start_idx + self.max_length
        
        # Ensure we don't go out of bounds
        if end_idx > len(tokens):
            print(f"Warning: Sequence end index {end_idx} exceeds token length {len(tokens)}")
            end_idx = len(tokens)
        
        sequence = tokens[start_idx:end_idx]
        
        # Pad if necessary
        if len(sequence) < self.max_length:
            sequence = np.pad(sequence, (0, self.max_length - len(sequence)), 
                            mode='constant', constant_values=0)
        
        # Ensure sequence is 1D before creating tensors
        if isinstance(sequence, np.ndarray) and sequence.ndim > 1:
            sequence = sequence.reshape(-1)
        
        # Convert to tensor with explicit dtype=torch.long
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        attention_mask = (input_ids != 0).long()  # Explicit long type
        
        # Debug shape information
        # print(f"Input IDs shape: {input_ids.shape}, dtype: {input_ids.dtype}")
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


# Custom collate function to handle 3D tensors
def collate_fn(batch):
    """Custom collate function that ensures proper tensor shapes"""
    
    # First apply the default collate to get a batch
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Check if we have a 3D tensor problem (batch, subseq, seq_len)
    if input_ids.ndim == 3:
        # Reshape: [batch_size, num_subseq, seq_len] -> [batch_size*num_subseq, seq_len]
        batch_size, num_subseq, seq_len = input_ids.shape
        input_ids = input_ids.reshape(-1, seq_len)
        labels = labels.reshape(-1, seq_len)
        attention_mask = attention_mask.reshape(-1, seq_len)
        
        print(f"Reshaped tensors from [{batch_size}, {num_subseq}, {seq_len}] to [{batch_size*num_subseq}, {seq_len}]")
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }


def create_dataloader(config, accelerator=None):
    """Create DataLoader for training"""
    # Initialize GCS data loader
    gcs_loader = GCSDataLoader(config.bucket_name, config.cache_dir)
    
    # List files
    files = gcs_loader.list_data_files(max_files=config.max_train_files)
    
    # Create dataset
    dataset = TokenizedDataset(
        gcs_loader,
        files,
        max_length=2048,  # Match model's max position embeddings
        stride=2048  # No overlap
    )
    
    # Create dataloader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn  # Use custom collate function
    )
    
    return dataloader, len(files) 