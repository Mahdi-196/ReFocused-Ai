#!/usr/bin/env python3
"""
Test tokenization on a small subset of data.
"""

import json
import time
from pathlib import Path
from tokenize_data import DataTokenizer

def test_small_batch():
    """Test tokenization on just a few files."""
    
    print("🧪 Testing tokenization on small data subset...")
    
    # Create a test tokenizer
    tokenizer = DataTokenizer(
        tokenizer_path="models/tokenizer/transformers_tokenizer",
        input_dir="data/cleaned",
        output_dir="data_tokenized_test",
        max_length=1024,
        stride=512,
        batch_size=100  # Smaller batch for testing
    )
    
    # Get just the first 2 files for testing
    all_files = tokenizer.get_input_files()
    test_files = all_files[:2]
    
    print(f"Testing with {len(test_files)} files:")
    for file in test_files:
        print(f"  - {file.name}")
    
    start_time = time.time()
    
    total_sequences = 0
    total_items = 0
    
    # Process each test file
    for file_path in test_files:
        print(f"\n📄 Processing: {file_path.name}")
        result = tokenizer.process_file(file_path)
        
        if not result.get("skipped", False):
            total_sequences += result.get("sequences", 0)
            total_items += result.get("items", 0)
            print(f"  ✅ Generated {result.get('sequences', 0)} sequences from {result.get('items', 0)} items")
        else:
            print(f"  ⏭️  Skipped (already processed)")
    
    elapsed = time.time() - start_time
    
    print(f"\n📊 Test Results:")
    print(f"  Files processed: {len(test_files)}")
    print(f"  Total sequences: {total_sequences:,}")
    print(f"  Total items: {total_items:,}")
    print(f"  Processing time: {elapsed:.2f}s")
    print(f"  Rate: {total_items/elapsed:.1f} items/sec" if elapsed > 0 else "  Rate: N/A")
    
    if total_sequences > 0:
        # Check output files
        output_dir = Path("data_tokenized_test")
        output_files = list(output_dir.glob("*.npz"))
        print(f"  Output files created: {len(output_files)}")
        
        # Load and inspect one file
        if output_files:
            import numpy as np
            sample_file = output_files[0]
            data = np.load(sample_file)
            print(f"\n📋 Sample output file: {sample_file.name}")
            print(f"  Shape: {data['input_ids'].shape}")
            print(f"  Max sequence length: {data['input_ids'].shape[1]}")
            print(f"  First sequence sample: {data['input_ids'][0][:20]}...")
            print(f"  Attention mask sample: {data['attention_mask'][0][:20]}...")
    
    return total_sequences > 0

if __name__ == "__main__":
    success = test_small_batch()
    if success:
        print("\n✅ Small batch test successful! Ready to process all data.")
    else:
        print("\n❌ Small batch test failed. Check the issues above.") 