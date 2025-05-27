#!/usr/bin/env python3
"""
Fixed Final Sequence Counter
Shows completion statistics and total sequences, handles object arrays
"""

import numpy as np
from pathlib import Path

def count_final_sequences():
    output_dir = Path("data_tokenized_production")
    
    print("🎉 FIXED FINAL TOKENIZATION RESULTS")
    print("=" * 40)
    
    total_sequences = 0
    valid_files = 0
    failed_files = 0
    total_size_mb = 0
    
    # Count sequences in all output files
    for npz_file in output_dir.glob("tokenized_cleaned_*.npz"):
        try:
            data = np.load(npz_file, allow_pickle=True)  # Allow pickle for object arrays
            if 'sequences' in data:
                sequences = data['sequences']
                
                # Handle both regular arrays and object arrays
                if sequences.dtype == object:
                    # Object array - each element is a list/array
                    seq_count = len(sequences)
                else:
                    # Regular array
                    seq_count = len(sequences)
                    
                total_sequences += seq_count
                valid_files += 1
                total_size_mb += npz_file.stat().st_size / (1024 * 1024)
                
        except Exception as e:
            failed_files += 1
            print(f"⚠️ Failed to read: {npz_file.name} - {str(e)[:50]}...")
    
    # Calculate statistics
    avg_sequences_per_file = total_sequences // valid_files if valid_files > 0 else 0
    
    print(f"📊 COMPLETION STATISTICS:")
    print(f"   Valid files: {valid_files:,}")
    print(f"   Failed files: {failed_files:,}")
    print(f"   Total sequences: {total_sequences:,}")
    print(f"   Average per file: {avg_sequences_per_file:,}")
    print(f"   Total size: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    
    # Estimate dataset size for training
    avg_tokens_per_sequence = 256  # Rough estimate (max 512, but most are shorter)
    total_tokens = total_sequences * avg_tokens_per_sequence
    
    print(f"\n🤖 TRAINING ESTIMATES:")
    print(f"   Estimated total tokens: {total_tokens:,}")
    print(f"   Suitable for models: GPT-2 Small to Medium")
    print(f"   Training epochs: 1-3 recommended")
    
    if valid_files >= 700:
        print(f"\n🎉 EXCELLENT! Dataset is ready for training!")
    elif valid_files >= 500:
        print(f"\n✅ GOOD! Dataset should work well for training!")
    else:
        print(f"\n⚠️ Consider processing more files for better coverage")
    
    return {
        'valid_files': valid_files,
        'total_sequences': total_sequences,
        'total_size_mb': total_size_mb,
        'failed_files': failed_files
    }

if __name__ == "__main__":
    stats = count_final_sequences() 