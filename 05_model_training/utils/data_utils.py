"""
Data utilities for loading tokenized data from Google Cloud Storage
"""

import numpy as np
from google.cloud import storage
from typing import List, Optional, Iterator
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import random


class GCSDataLoader:
    """Handles loading tokenized data from Google Cloud Storage"""
    
    def __init__(self, bucket_name: str, cache_dir: str = "./cache"):
        self.bucket_name = bucket_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize GCS client
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(bucket_name)
    
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
    
    def load_npz_file(self, file_path: str) -> np.ndarray:
        """Load tokenized data from npz file"""
        data = np.load(file_path)
        return data['input_ids']  # Assuming tokenized data is stored as 'input_ids'


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
        
        print("Building dataset index...")
        for file_idx, file_name in enumerate(tqdm(files)):
            local_path = self.data_loader.download_file(file_name)
            tokens = self.data_loader.load_npz_file(local_path)
            
            # Create sequences with stride
            num_sequences = max(1, (len(tokens) - max_length) // stride + 1)
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
        
        # Extract sequence
        start_idx = seq_idx * self.stride
        end_idx = start_idx + self.max_length
        
        sequence = tokens[start_idx:end_idx]
        
        # Pad if necessary
        if len(sequence) < self.max_length:
            sequence = np.pad(sequence, (0, self.max_length - len(sequence)), 
                            mode='constant', constant_values=0)
        
        # Convert to tensor
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != 0).long()
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
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=config.dataloader_num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, len(files) 