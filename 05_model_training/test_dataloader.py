"""
Test script to verify dataloader tensor shapes
"""

import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_test_config
from utils import create_dataloader


def main():
    print("Testing dataloader tensor shapes...")
    
    # Get test configuration with a small number of files
    config = get_test_config()
    config.max_train_files = 1  # Just load one file for quick testing
    config.per_device_train_batch_size = 2  # Small batch size
    config.dataloader_num_workers = 0  # No multiprocessing for debugging
    
    # Create dataloader
    dataloader, num_files = create_dataloader(config)
    
    print(f"Dataloader created with {num_files} files")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get first batch
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i+1}:")
        
        # Check input shapes
        print(f"Input shapes:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        
        # Check input dtypes
        print(f"Input dtypes:")
        print(f"  input_ids: {batch['input_ids'].dtype}")
        print(f"  attention_mask: {batch['attention_mask'].dtype}")
        print(f"  labels: {batch['labels'].dtype}")
        
        # Check for expected shape [batch_size, seq_len]
        if batch['input_ids'].ndim == 2:
            print("✅ input_ids has correct shape [batch_size, seq_len]")
            # Test unpacking to verify it will work in the model
            try:
                batch_size, seq_length = batch['input_ids'].shape
                print(f"✅ Successfully unpacked shape into batch_size={batch_size}, seq_length={seq_length}")
            except ValueError as e:
                print(f"❌ Failed to unpack shape: {e}")
        else:
            print(f"❌ input_ids has incorrect shape: {batch['input_ids'].shape}")
            if batch['input_ids'].ndim == 3:
                print("  This needs to be reshaped to [batch_size*subseq, seq_len]")
        
        # Check for expected dtype torch.long
        if batch['input_ids'].dtype == torch.long:
            print("✅ input_ids has correct dtype torch.long")
        else:
            print(f"❌ input_ids has incorrect dtype: {batch['input_ids'].dtype}")
            print("  This should be torch.long")
        
        # Try a sample forward pass simulation
        print("\nSimulating model input shape validation:")
        try:
            input_shape = batch['input_ids'].size()
            batch_size, seq_length = input_shape
            print(f"✅ Model would receive correct input_shape: {input_shape}")
            print(f"✅ Successfully unpacked as batch_size={batch_size}, seq_length={seq_length}")
        except ValueError as e:
            print(f"❌ Model would fail with error: {e}")
        
        # Only print first 3 batches
        if i >= 2:
            break
    
    print("\nDataloader test completed!")


if __name__ == "__main__":
    main() 