import numpy as np
from pathlib import Path

def analyze_tokenized_data():
    data_dir = Path(r'C:\Users\mahdi\Downloads\Documents\Desktop\data_tokenized_production')
    print('🔍 ANALYZING YOUR TOKENIZED DATA')
    print('=' * 50)
    
    total_sequences = 0
    total_files = 0
    total_size_gb = 0
    subreddit_stats = {}
    
    # Look for files that start with "tokenized_cleaned_"
    for npz_file in data_dir.glob('tokenized_cleaned_*.npz'):
        try:
            data = np.load(npz_file, allow_pickle=True)
            if 'input_ids' in data:
                input_ids = data['input_ids']
                seq_count = len(input_ids)
                
                total_sequences += seq_count
                total_files += 1
                size_gb = npz_file.stat().st_size / (1024**3)
                total_size_gb += size_gb
                
                # Extract subreddit name - format: tokenized_cleaned_SUBREDDIT_type_partXXX.npz
                parts = npz_file.name.split('_')
                if len(parts) >= 3:
                    subreddit = parts[2]
                    if subreddit not in subreddit_stats:
                        subreddit_stats[subreddit] = {'files': 0, 'sequences': 0, 'size_gb': 0}
                    subreddit_stats[subreddit]['files'] += 1
                    subreddit_stats[subreddit]['sequences'] += seq_count
                    subreddit_stats[subreddit]['size_gb'] += size_gb
                
                if total_files <= 5:  # Show first 5 files as examples
                    print(f'✅ {npz_file.name}: {seq_count:,} sequences ({size_gb:.3f} GB)')
            
        except Exception as e:
            print(f'❌ Failed: {npz_file.name} - {str(e)[:50]}...')
    
    print(f'\n📊 TOTAL DATASET STATS:')
    print(f'   Files: {total_files:,}')
    print(f'   Sequences: {total_sequences:,}')
    print(f'   Size: {total_size_gb:.2f} GB')
    print(f'   Estimated tokens: {total_sequences * 1024:,}')  # Each sequence is 1024 tokens
    
    print(f'\n🏷️ TOP SUBREDDITS BY SEQUENCES:')
    sorted_subreddits = sorted(subreddit_stats.items(), key=lambda x: x[1]['sequences'], reverse=True)
    for i, (subreddit, stats) in enumerate(sorted_subreddits[:10]):
        print(f'   {i+1:2d}. {subreddit}: {stats["sequences"]:,} sequences ({stats["files"]} files)')
    
    return {
        'total_files': total_files,
        'total_sequences': total_sequences,
        'total_size_gb': total_size_gb,
        'subreddit_stats': subreddit_stats
    }

if __name__ == "__main__":
    stats = analyze_tokenized_data() 