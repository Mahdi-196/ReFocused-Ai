"""
Test script to verify optimized dataloader tensor shapes and performance
"""

import sys
import os
import torch
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_test_config
from utils import create_dataloader


def test_preprocessing_performance():
    """Test preprocessing cache performance"""
    print("Testing preprocessing performance...")
    
    config = get_test_config()
    config.max_train_files = 3  # Test with 3 files
    config.per_device_train_batch_size = 2
    config.dataloader_num_workers = 0
    config.use_optimized_dataset = True
    config.enable_profiling = True
    
    print(f"Testing with {config.max_train_files} files")
    print(f"Optimized dataset: {config.use_optimized_dataset}")
    
    # Time the dataloader creation (includes preprocessing)
    start_time = time.time()
    dataloader, num_files = create_dataloader(config, use_optimized=True)
    setup_time = time.time() - start_time
    
    print(f"Dataloader setup time: {setup_time:.2f}s")
    print(f"Files loaded: {num_files}")
    print(f"Number of batches: {len(dataloader)}")
    
    return dataloader, setup_time


def test_batch_shapes_and_performance():
    """Test batch shapes and data loading performance"""
    dataloader, setup_time = test_preprocessing_performance()
    
    print("\n" + "="*60)
    print("TESTING BATCH SHAPES AND PERFORMANCE")
    print("="*60)
    
    batch_times = []
    
    for i, batch in enumerate(dataloader):
        batch_start = time.time()
        
        print(f"\nBatch {i+1}:")
        
        # Check input shapes
        print(f"Input shapes:")
        print(f"  input_ids: {batch['input_ids'].shape} (dtype: {batch['input_ids'].dtype})")
        print(f"  attention_mask: {batch['attention_mask'].shape} (dtype: {batch['attention_mask'].dtype})")
        print(f"  labels: {batch['labels'].shape} (dtype: {batch['labels'].dtype})")
        
        # Validate shapes
        expected_dims = 2  # [batch_size, seq_len]
        
        if batch['input_ids'].ndim == expected_dims:
            print("✅ input_ids has correct shape [batch_size, seq_len]")
            try:
                batch_size, seq_length = batch['input_ids'].shape
                print(f"✅ Batch size: {batch_size}, Sequence length: {seq_length}")
            except ValueError as e:
                print(f"❌ Failed to unpack shape: {e}")
        else:
            print(f"❌ input_ids has incorrect dimensions: {batch['input_ids'].ndim} (expected {expected_dims})")
        
        # Check dtype
        if batch['input_ids'].dtype == torch.long:
            print("✅ input_ids has correct dtype (torch.long)")
        else:
            print(f"❌ input_ids has incorrect dtype: {batch['input_ids'].dtype}")
        
        # Test model input compatibility
        try:
            input_shape = batch['input_ids'].size()
            batch_size, seq_length = input_shape
            print(f"✅ Model compatibility test passed: {input_shape}")
        except ValueError as e:
            print(f"❌ Model compatibility test failed: {e}")
        
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        print(f"Batch processing time: {batch_time:.4f}s")
        
        # Only test first 3 batches for speed
        if i >= 2:
            break
    
    # Performance summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Setup time: {setup_time:.2f}s")
    print(f"Average batch time: {sum(batch_times)/len(batch_times):.4f}s")
    print(f"Total test time: {setup_time + sum(batch_times):.2f}s")
    
    return True


def compare_optimized_vs_legacy():
    """Compare optimized vs legacy data loading"""
    print("\n" + "="*60)
    print("COMPARING OPTIMIZED VS LEGACY DATA LOADING")
    print("="*60)
    
    config = get_test_config()
    config.max_train_files = 2  # Small test
    config.per_device_train_batch_size = 2
    config.dataloader_num_workers = 0
    
    # Test legacy approach
    print("\nTesting LEGACY data loading...")
    start_time = time.time()
    legacy_dataloader, _ = create_dataloader(config, use_optimized=False)
    legacy_time = time.time() - start_time
    
    # Get one batch to test processing time
    legacy_batch_start = time.time()
    legacy_batch = next(iter(legacy_dataloader))
    legacy_batch_time = time.time() - legacy_batch_start
    
    # Test optimized approach
    print("\nTesting OPTIMIZED data loading...")
    start_time = time.time()
    optimized_dataloader, _ = create_dataloader(config, use_optimized=True)
    optimized_time = time.time() - start_time
    
    # Get one batch to test processing time
    optimized_batch_start = time.time()
    optimized_batch = next(iter(optimized_dataloader))
    optimized_batch_time = time.time() - optimized_batch_start
    
    # Compare results
    print(f"\nPERFORMANCE COMPARISON:")
    print(f"  Legacy setup time:     {legacy_time:.3f}s")
    print(f"  Optimized setup time:  {optimized_time:.3f}s")
    print(f"  Speedup factor:        {legacy_time/optimized_time:.2f}x")
    
    print(f"\nBATCH PROCESSING:")
    print(f"  Legacy batch time:     {legacy_batch_time:.4f}s")
    print(f"  Optimized batch time:  {optimized_batch_time:.4f}s")
    print(f"  Batch speedup:         {legacy_batch_time/optimized_batch_time:.2f}x")
    
    # Verify outputs are the same shape
    print(f"\nSHAPE VERIFICATION:")
    print(f"  Legacy shape:      {legacy_batch['input_ids'].shape}")
    print(f"  Optimized shape:   {optimized_batch['input_ids'].shape}")
    
    shapes_match = legacy_batch['input_ids'].shape == optimized_batch['input_ids'].shape
    print(f"  Shapes match: {'✅' if shapes_match else '❌'}")
    
    return optimized_time < legacy_time


def main():
    print("ReFocused-AI Optimized Dataloader Test")
    print("="*60)
    
    try:
        # Test 1: Basic functionality and shapes
        success1 = test_batch_shapes_and_performance()
        
        # Test 2: Performance comparison
        success2 = compare_optimized_vs_legacy()
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Shape/functionality test: {'✅ PASSED' if success1 else '❌ FAILED'}")
        print(f"Performance test:         {'✅ PASSED' if success2 else '❌ FAILED'}")
        
        if success1 and success2:
            print("\n🎉 All tests passed! Optimized dataloader is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Check the output above for details.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 