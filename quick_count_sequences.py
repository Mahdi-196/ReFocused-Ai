#!/usr/bin/env python3
"""
Quick Sequence Counter - Fast version with progress
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm

def quick_count_sequences():
    output_dir = Path("data_tokenized_production")
    files = list(output_dir.glob("tokenized_cleaned_*.npz"))
    
    print("⚡ QUICK TOKENIZATION RESULTS")
    print("=" * 40)
    print(f"Found {len(files)} files to process...")
    
    total_sequences = 0
    valid_files = 0
    failed_files = 0
    total_size_mb = 0
    
    # Process with progress bar
    for npz_file in tqdm(files, desc="Counting sequences", unit="files"):
        try:
            # Quick file size check first
            file_size = npz_file.stat().st_size / (1024 * 1024)
            total_size_mb += file_size
            
            # Skip empty files
            if file_size < 0.001:  # Less than 1KB
                failed_files += 1
                continue
                
            data = np.load(npz_file, allow_pickle=True)
            if 'sequences' in data:
                sequences = data['sequences']
                seq_count = len(sequences)
                total_sequences += seq_count
                valid_files += 1
            else:
                failed_files += 1
                
        except Exception as e:
            failed_files += 1
            # Only show first few errors to avoid spam
            if failed_files <= 5:
                tqdm.write(f"⚠️ Failed: {npz_file.name}")
    
    # Calculate statistics
    avg_sequences_per_file = total_sequences // valid_files if valid_files > 0 else 0
    
    print(f"\n📊 COMPLETION STATISTICS:")
    print(f"   Valid files: {valid_files:,}")
    print(f"   Failed files: {failed_files:,}")
    print(f"   Total sequences: {total_sequences:,}")
    print(f"   Average per file: {avg_sequences_per_file:,}")
    print(f"   Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    
    if valid_files >= 700:
        print(f"\n🎉 EXCELLENT! Dataset is ready for training!")
    elif valid_files >= 500:
        print(f"\n✅ GOOD! Dataset should work well for training!")
    else:
        print(f"\n⚠️ Only {valid_files} valid files - need more data!")
    
    return {
        'valid_files': valid_files,
        'total_sequences': total_sequences,
        'failed_files': failed_files
    }

if __name__ == "__main__":
    stats = quick_count_sequences() 