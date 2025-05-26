#!/usr/bin/env python3
"""
Run full tokenization of all cleaned data.
"""

import os
import sys
import time
import json
from pathlib import Path
from tokenize_data import DataTokenizer

def estimate_processing_time():
    """Estimate processing time based on test results."""
    
    # Test results: 111,772 items in 94.80 seconds = ~1,179 items/sec
    test_rate = 1179  # items per second
    
    # Count total items to process
    input_dir = Path("data/cleaned")
    files = list(input_dir.glob("*.jsonl"))
    
    total_size = sum(f.stat().st_size for f in files)
    print(f"📊 Dataset Overview:")
    print(f"  Total files: {len(files)}")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    
    # Estimate items (assuming similar density as test file)
    test_file_size = Path("data/cleaned/cleaned_AcademicPsychology_comments_part001.jsonl").stat().st_size
    test_items = 111772
    items_per_byte = test_items / test_file_size
    
    estimated_total_items = int(total_size * items_per_byte)
    estimated_time_seconds = estimated_total_items / test_rate
    estimated_hours = estimated_time_seconds / 3600
    
    print(f"  Estimated total items: {estimated_total_items:,}")
    print(f"  Estimated processing time: {estimated_hours:.1f} hours")
    print(f"  Processing rate: ~{test_rate:,} items/sec")
    
    return estimated_total_items, estimated_hours

def run_full_tokenization():
    """Run the full tokenization process."""
    
    print("🚀 Starting Full Data Tokenization")
    print("=" * 50)
    
    # Show estimates
    total_items, estimated_hours = estimate_processing_time()
    
    print(f"\n⚠️  This will process ~{total_items:,} items")
    print(f"⚠️  Estimated time: ~{estimated_hours:.1f} hours")
    print(f"⚠️  Output size: ~{estimated_hours * 200:.0f} MB (estimated)")
    
    # Ask for confirmation
    response = input("\n📋 Continue with full tokenization? (y/N): ").strip().lower()
    if response != 'y':
        print("❌ Tokenization cancelled by user")
        return False
    
    # Create tokenizer with production settings
    tokenizer = DataTokenizer(
        tokenizer_path="models/tokenizer/transformers_tokenizer",
        input_dir="data/cleaned",
        output_dir="data_tokenized",
        max_length=1024,
        stride=512,
        batch_size=500,  # Slightly smaller for stability
    )
    
    print(f"\n🔧 Configuration:")
    print(f"  Input directory: {tokenizer.input_dir}")
    print(f"  Output directory: {tokenizer.output_dir}")
    print(f"  Max sequence length: {tokenizer.max_length}")
    print(f"  Stride: {tokenizer.stride}")
    print(f"  Batch size: {tokenizer.batch_size}")
    
    # Start processing
    print(f"\n🏁 Starting tokenization...")
    start_time = time.time()
    
    try:
        # Use parallel processing
        tokenizer.process_all_files(num_workers=4)
        
        elapsed = time.time() - start_time
        print(f"\n🎉 Tokenization completed in {elapsed/3600:.2f} hours!")
        
        # Show results
        summary_file = Path("data_tokenized/tokenization_summary.json")
        if summary_file.exists():
            with open(summary_file) as f:
                summary = json.load(f)
            
            print(f"\n📊 Final Results:")
            print(f"  Files processed: {summary['total_files']}")
            print(f"  Total sequences: {summary['total_sequences']:,}")
            print(f"  Total items: {summary['total_items']:,}")
            print(f"  Processing time: {summary['processing_time']/3600:.2f} hours")
            print(f"  Average rate: {summary['total_items']/summary['processing_time']:.0f} items/sec")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Tokenization interrupted by user")
        elapsed = time.time() - start_time
        print(f"   Processed for {elapsed/60:.1f} minutes")
        
        # Check partial results
        output_files = list(Path("data_tokenized").glob("*.npz"))
        print(f"   Partial files created: {len(output_files)}")
        
        return False
    
    except Exception as e:
        print(f"\n❌ Error during tokenization: {e}")
        return False

def main():
    """Main function."""
    
    # Check if test was successful
    test_files = list(Path("data_tokenized_test").glob("*.npz"))
    if not test_files:
        print("❌ Please run the test tokenization first:")
        print("   python test_small_tokenization.py")
        return
    
    print("✅ Test tokenization found - proceeding with full dataset")
    
    # Run full tokenization
    success = run_full_tokenization()
    
    if success:
        print("\n🎯 Next Steps:")
        print("  1. Your data is now tokenized and ready for model training")
        print("  2. Use the files in 'data_tokenized/' for training")
        print("  3. Each .npz file contains: input_ids, attention_mask, sequence_lengths")
        print("  4. You can now proceed to train your GPT model!")
    else:
        print("\n📝 Note:")
        print("  If interrupted, you can resume by running this script again")
        print("  Already processed files will be skipped automatically")

if __name__ == "__main__":
    main() 