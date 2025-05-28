"""
Quick Dataset Size Check - Memory Efficient
Reads only metadata from NPZ files to get accurate counts
"""

import numpy as np
from pathlib import Path
import time

def quick_size_check():
    data_dir = Path(r'C:\Users\mahdi\Downloads\Documents\Desktop\data_tokenized_production')
    print('🔍 QUICK DATASET SIZE CHECK')
    print('=' * 50)
    
    total_sequences = 0
    total_files = 0
    total_size_gb = 0
    subreddit_stats = {}
    
    # Get all NPZ files
    npz_files = list(data_dir.glob('tokenized_cleaned_*.npz'))
    print(f"📁 Found {len(npz_files)} tokenized files")
    
    start_time = time.time()
    
    for i, npz_file in enumerate(npz_files):
        try:
            # Just peek at the file structure without loading arrays
            with np.load(npz_file, allow_pickle=True) as data:
                if 'input_ids' in data:
                    # Get shape without loading the array
                    shape = data['input_ids'].shape
                    seq_count = shape[0]  # First dimension is number of sequences
                    
                    total_sequences += seq_count
                    total_files += 1
                    size_gb = npz_file.stat().st_size / (1024**3)
                    total_size_gb += size_gb
                    
                    # Extract subreddit name
                    parts = npz_file.name.split('_')
                    if len(parts) >= 3:
                        subreddit = parts[2]
                        if subreddit not in subreddit_stats:
                            subreddit_stats[subreddit] = {'files': 0, 'sequences': 0, 'size_gb': 0}
                        subreddit_stats[subreddit]['files'] += 1
                        subreddit_stats[subreddit]['sequences'] += seq_count
                        subreddit_stats[subreddit]['size_gb'] += size_gb
                    
                    # Progress indicator
                    if i % 50 == 0:
                        progress = (i + 1) / len(npz_files) * 100
                        print(f"   Progress: {progress:.1f}% ({i+1}/{len(npz_files)})")
                        
        except Exception as e:
            print(f'❌ Failed: {npz_file.name} - {str(e)[:50]}...')
    
    elapsed_time = time.time() - start_time
    
    print(f'\n📊 DATASET SUMMARY:')
    print(f'   Total files processed: {total_files:,}')
    print(f'   Total sequences: {total_sequences:,}')
    print(f'   Total size: {total_size_gb:.2f} GB')
    print(f'   Average sequences per file: {total_sequences // total_files if total_files > 0 else 0:,}')
    print(f'   Processing time: {elapsed_time:.1f} seconds')
    
    # Token estimation (each sequence is 1024 tokens)
    total_tokens = total_sequences * 1024
    print(f'\n🔢 TOKEN ESTIMATES:')
    print(f'   Total tokens: {total_tokens:,}')
    print(f'   Billions of tokens: {total_tokens / 1e9:.2f}B')
    
    # Training estimates
    print(f'\n🎯 TRAINING READINESS:')
    if total_sequences >= 1_000_000:
        print(f'   ✅ EXCELLENT dataset size for training!')
        print(f'   ✅ Suitable for GPT-2 Medium/Large models')
        if total_sequences >= 10_000_000:
            print(f'   🚀 MASSIVE dataset - enterprise grade!')
    elif total_sequences >= 100_000:
        print(f'   ✅ GOOD dataset size for training')
        print(f'   ✅ Suitable for GPT-2 Small/Medium models')
    else:
        print(f'   ⚠️ Small dataset - consider gathering more data')
    
    # Top subreddits
    print(f'\n🏷️ TOP 10 SUBREDDITS BY SEQUENCES:')
    sorted_subreddits = sorted(subreddit_stats.items(), 
                              key=lambda x: x[1]['sequences'], reverse=True)
    for i, (subreddit, stats) in enumerate(sorted_subreddits[:10]):
        print(f'   {i+1:2d}. {subreddit}: {stats["sequences"]:,} sequences ({stats["files"]} files, {stats["size_gb"]:.2f} GB)')
    
    return {
        'total_files': total_files,
        'total_sequences': total_sequences,
        'total_size_gb': total_size_gb,
        'total_tokens': total_tokens,
        'subreddit_stats': subreddit_stats
    }

if __name__ == "__main__":
    stats = quick_size_check() 