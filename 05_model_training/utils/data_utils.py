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
import platform


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
        
        # Load raw data
        data = np.load(file_path)
        input_ids = data['input_ids']
        
        # Handle the case where the NPZ file contains 2D data already
        if input_ids.ndim > 1:
            if input_ids.ndim == 2:
                pass  # Keep as is
            elif input_ids.ndim == 3 and input_ids.shape[-1] == 1:
                input_ids = input_ids.squeeze(-1)
            # Flatten the array into a single sequence
            input_ids = input_ids.reshape(-1)
        
        # Create sequences with stride
        num_sequences = max(1, (len(input_ids) - max_length) // stride + 1)
        sequences = []
        
        for seq_idx in range(num_sequences):
            start_idx = seq_idx * stride
            end_idx = start_idx + max_length
            
            if end_idx > len(input_ids):
                end_idx = len(input_ids)
            
            sequence = input_ids[start_idx:end_idx]
            
            # Pad if necessary
            if len(sequence) < max_length:
                sequence = np.pad(sequence, (0, max_length - len(sequence)), 
                                mode='constant', constant_values=0)
            
            sequences.append(sequence)
        
        # Cache the processed data
        cached_data = {
            'sequences': np.array(sequences),
            'max_length': max_length,
            'stride': stride,
            'num_sequences': len(sequences)
        }
        
        # Save to disk cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cached_data, f)
        except Exception as e:
            print(f"Warning: Could not save cache for {file_path}: {e}")
        
        # Store in memory cache
        self.flattened_cache[file_path] = cached_data
        
        return cached_data
    
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


class OptimizedTokenizedDataset(Dataset):
    """Optimized PyTorch Dataset that uses preprocessed cached data"""
    
    def __init__(self, 
                 data_loader: GCSDataLoader,
                 files: List[str],
                 max_length: int = 2048,
                 stride: int = 2048):
        self.data_loader = data_loader
        self.files = files
        self.max_length = max_length
        self.stride = stride
        
        # Build index of all sequences using cached data
        self.file_indices = []
        self.sequence_indices = []
        self.cached_data = []
        
        print("Building optimized dataset index with preprocessing cache...")
        for file_idx, file_name in enumerate(tqdm(files)):
            local_path = self.data_loader.download_file(file_name)
            cached_data = self.data_loader.load_npz_file_optimized(local_path, max_length, stride)
            
            num_sequences = cached_data['num_sequences']
            self.cached_data.append(cached_data)
            
            for seq_idx in range(num_sequences):
                self.file_indices.append(file_idx)
                self.sequence_indices.append(seq_idx)
        
        print(f"Total sequences (optimized): {len(self.file_indices)}")
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        seq_idx = self.sequence_indices[idx]
        
        # Get sequence from cached data (much faster!)
        cached_data = self.cached_data[file_idx]
        sequence = cached_data['sequences'][seq_idx]
        
        # Convert to tensor with explicit dtype=torch.long
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


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


def create_dataloader(config, accelerator):
    """Create training dataloader with optimized settings"""
    # Use 0 workers on Windows to avoid multiprocessing/pickling issues
    num_workers = 0 if platform.system() == "Windows" else config.dataloader_num_workers
    if num_workers == 0:
        print("🔧 Using single-threaded dataloader (Windows compatibility)")
    
    # Standard dataset - loads all data into memory
    print("Using standard TokenizedDataset...")
    dataset = SimpleTokenizedDataset(
        bucket_name=config.bucket_name,
        file_pattern=config.tokenized_file_pattern,
        sequence_length=config.sequence_length,
        max_files=config.max_files
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if accelerator.device.type == "cuda" else False
    )
    
    return dataloader, dataset.num_files


class SimpleTokenizedDataset(Dataset):
    """Simple tokenized dataset that works directly with bucket parameters"""
    
    def __init__(self,
                 bucket_name: str,
                 file_pattern: str,
                 sequence_length: int = 1024,
                 max_files: int = 5):
        self.bucket_name = bucket_name
        self.file_pattern = file_pattern
        self.sequence_length = sequence_length
        self.max_files = max_files
        
        # Initialize GCS client
        self.client = storage.Client.create_anonymous_client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Find and cache data files
        self._load_files()
        
        # Build sequence index
        self._build_index()
    
    def _load_files(self):
        """Load list of files matching the pattern"""
        blobs = self.bucket.list_blobs()
        self.files = []
        
        print(f"🔍 Looking for files matching pattern: {self.file_pattern}")
        
        # More flexible pattern matching
        all_npz_files = []
        tokenized_files = []
        
        for blob in blobs:
            if blob.name.endswith('.npz'):
                all_npz_files.append(blob.name)
                if 'tokenized' in blob.name and 'cleaned' in blob.name:
                    tokenized_files.append(blob.name)
                    self.files.append(blob.name)
                    if self.max_files > 0 and len(self.files) >= self.max_files:
                        break
        
        print(f"📊 Found {len(all_npz_files)} total .npz files")
        print(f"📊 Found {len(tokenized_files)} tokenized_cleaned files")
        
        self.num_files = len(self.files)
        print(f"✅ Selected {self.num_files} files for training")
        if self.num_files > 0:
            print(f"📁 Sample files: {self.files[:3]}")  # Show first 3 files for debugging
        elif len(all_npz_files) > 0:
            print(f"🔍 Available .npz files (first 5): {all_npz_files[:5]}")
        else:
            print("❌ No .npz files found in bucket")
    
    def _build_index(self):
        """Build index of all sequences"""
        self.file_indices = []
        self.start_indices = []
        self.all_tokens = []
        
        print("Building dataset index...")
        for file_idx, file_name in enumerate(tqdm(self.files)):
            # Download and load the file
            local_path = f"./cache/{os.path.basename(file_name)}"
            os.makedirs("./cache", exist_ok=True)
            
            if not os.path.exists(local_path):
                print(f"Downloading {file_name}...")
                blob = self.bucket.blob(file_name)
                blob.download_to_filename(local_path)
            
            # Load data
            data = np.load(local_path)
            input_ids = data['input_ids']
            
            # Handle different shapes
            if input_ids.ndim > 1:
                print(f"Loaded data shape before flattening: {input_ids.shape}")
                input_ids = input_ids.reshape(-1)
                print(f"Reshaped data to: {input_ids.shape}")
            
            # Store the tokens
            self.all_tokens.append(input_ids)
            
            # Create sequence indices
            num_sequences = max(1, len(input_ids) - self.sequence_length + 1)
            for start_idx in range(0, len(input_ids) - self.sequence_length + 1, self.sequence_length):
                self.file_indices.append(file_idx)
                self.start_indices.append(start_idx)
        
        print(f"Total sequences: {len(self.file_indices)}")
    
    def __len__(self):
        return len(self.file_indices)
    
    def __getitem__(self, idx):
        file_idx = self.file_indices[idx]
        start_idx = self.start_indices[idx]
        
        # Get the sequence
        tokens = self.all_tokens[file_idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract sequence
        if end_idx <= len(tokens):
            sequence = tokens[start_idx:end_idx]
        else:
            # Pad if necessary
            sequence = tokens[start_idx:]
            sequence = np.pad(sequence, (0, self.sequence_length - len(sequence)), 
                            mode='constant', constant_values=0)
        
        # Create input_ids and labels (shifted by 1)
        input_ids = torch.tensor(sequence[:-1], dtype=torch.long)
        labels = torch.tensor(sequence[1:], dtype=torch.long)
        attention_mask = (input_ids != 0).long()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        } 