"""
Data loader for streaming tokenized data from Google Cloud Storage
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, IterableDataset
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor
import io
import logging
from typing import List, Optional, Iterator
import random

logger = logging.getLogger(__name__)


class GCSTokenizedDataset(IterableDataset):
    """Iterable dataset that streams tokenized data from GCS"""
    
    def __init__(
        self,
        bucket_name: str,
        prefix: str,
        max_seq_len: int,
        num_files: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        worker_id: Optional[int] = None,
        num_workers: int = 1
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.max_seq_len = max_seq_len
        self.num_files = num_files
        self.shuffle = shuffle
        self.seed = seed
        self.worker_id = worker_id or 0
        self.num_workers = num_workers
        
        # Initialize GCS client
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Get list of files
        self.files = self._list_files()
        
    def _list_files(self) -> List[str]:
        """List all tokenized files in the bucket"""
        blobs = self.bucket.list_blobs(prefix=self.prefix)
        files = [blob.name for blob in blobs if blob.name.endswith('.npy')]
        
        if self.shuffle:
            random.Random(self.seed).shuffle(files)
        
        if self.num_files:
            files = files[:self.num_files]
            
        # Split files among workers
        if self.num_workers > 1:
            files = files[self.worker_id::self.num_workers]
            
        logger.info(f"Worker {self.worker_id}: Found {len(files)} files to process")
        return files
    
    def _download_file(self, blob_name: str) -> np.ndarray:
        """Download a file from GCS and return as numpy array"""
        blob = self.bucket.blob(blob_name)
        content = blob.download_as_bytes()
        tokens = np.load(io.BytesIO(content))
        return tokens
    
    def _create_sequences(self, tokens: np.ndarray) -> Iterator[torch.Tensor]:
        """Create sequences of max_seq_len from tokens"""
        num_sequences = len(tokens) // self.max_seq_len
        
        for i in range(num_sequences):
            start_idx = i * self.max_seq_len
            end_idx = start_idx + self.max_seq_len + 1  # +1 for labels
            
            if end_idx <= len(tokens):
                sequence = tokens[start_idx:end_idx]
                yield torch.tensor(sequence, dtype=torch.long)
    
    def __iter__(self) -> Iterator[dict]:
        """Iterate through all sequences in all files"""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # In multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = self.files[worker_id::num_workers]
        else:
            files = self.files
        
        for file_idx, file_name in enumerate(files):
            try:
                logger.info(f"Loading file {file_idx + 1}/{len(files)}: {file_name}")
                tokens = self._download_file(file_name)
                
                for sequence in self._create_sequences(tokens):
                    # Split into input and labels
                    input_ids = sequence[:-1]
                    labels = sequence[1:]
                    
                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "file_name": file_name,
                        "file_idx": file_idx
                    }
                    
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
                continue


class DataCollator:
    """Collate batches of sequences"""
    
    def __init__(self, pad_token_id: int = 1):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[dict]) -> dict:
        # Extract sequences
        input_ids = [item["input_ids"] for item in batch]
        labels = [item["labels"] for item in batch]
        
        # Pad sequences
        max_len = max(len(seq) for seq in input_ids)
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for inp, lab in zip(input_ids, labels):
            pad_len = max_len - len(inp)
            
            # Pad input_ids
            padded_inp = torch.cat([inp, torch.full((pad_len,), self.pad_token_id)])
            padded_input_ids.append(padded_inp)
            
            # Pad labels with -100 (ignored in loss)
            padded_lab = torch.cat([lab, torch.full((pad_len,), -100)])
            padded_labels.append(padded_lab)
            
            # Create attention mask
            mask = torch.cat([torch.ones(len(inp)), torch.zeros(pad_len)])
            attention_masks.append(mask)
        
        return {
            "input_ids": torch.stack(padded_input_ids),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
            "file_indices": torch.tensor([item["file_idx"] for item in batch])
        }


def create_dataloaders(
    bucket_name: str,
    prefix: str,
    max_seq_len: int,
    batch_size: int,
    num_files: Optional[int] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42
) -> DataLoader:
    """Create data loader for training"""
    
    dataset = GCSTokenizedDataset(
        bucket_name=bucket_name,
        prefix=prefix,
        max_seq_len=max_seq_len,
        num_files=num_files,
        shuffle=shuffle,
        seed=seed
    )
    
    collator = DataCollator()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    return dataloader


# Utility function to estimate dataset size
def estimate_dataset_size(bucket_name: str, prefix: str, sample_size: int = 10) -> dict:
    """Estimate total tokens and sequences in dataset"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = list(bucket.list_blobs(prefix=prefix))
    npy_files = [b for b in blobs if b.name.endswith('.npy')]
    
    if not npy_files:
        return {"total_files": 0, "estimated_tokens": 0}
    
    # Sample some files to estimate average size
    sample_files = random.sample(npy_files, min(sample_size, len(npy_files)))
    total_tokens = 0
    
    for blob in sample_files:
        content = blob.download_as_bytes()
        tokens = np.load(io.BytesIO(content))
        total_tokens += len(tokens)
    
    avg_tokens_per_file = total_tokens / len(sample_files)
    estimated_total_tokens = avg_tokens_per_file * len(npy_files)
    
    return {
        "total_files": len(npy_files),
        "sampled_files": len(sample_files),
        "average_tokens_per_file": int(avg_tokens_per_file),
        "estimated_total_tokens": int(estimated_total_tokens),
        "estimated_gb": estimated_total_tokens * 2 / 1e9  # int16 tokens
    } 