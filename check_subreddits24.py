#!/usr/bin/env python3
"""
Quick analysis of subreddits24 data
Shows what Reddit data is available for processing
"""

from pathlib import Path
import sys

def analyze_subreddits24():
    """Analyze the subreddits24 directory"""
    print("📱 Analyzing subreddits24 Reddit Data")
    print("=" * 50)
    
    subreddits_dir = Path("subreddits24")
    
    if not subreddits_dir.exists():
        print("❌ subreddits24 directory not found!")
        print("💡 Make sure you're in the ReFocused-Ai directory")
        return False
    
    # Find all .zst files
    zst_files = list(subreddits_dir.glob("*.zst"))
    
    if not zst_files:
        print("❌ No .zst files found in subreddits24/")
        return False
    
    # Analyze files
    total_size = sum(f.stat().st_size for f in zst_files)
    total_size_gb = total_size / (1024**3)
    
    # Group by subreddit
    subreddits = {}
    for file in zst_files:
        name_parts = file.stem.split('_')
        if len(name_parts) >= 2:
            subreddit = name_parts[0]
            file_type = name_parts[1]
            
            if subreddit not in subreddits:
                subreddits[subreddit] = {'comments': False, 'submissions': False, 'size_mb': 0}
            
            if 'comments' in file_type:
                subreddits[subreddit]['comments'] = True
            elif 'submissions' in file_type:
                subreddits[subreddit]['submissions'] = True
            
            subreddits[subreddit]['size_mb'] += file.stat().st_size / (1024**2)
    
    # Display results
    print(f"📊 REDDIT DATA SUMMARY:")
    print(f"   Files: {len(zst_files)} .zst files")
    print(f"   Total Size: {total_size_gb:.1f}GB")
    print(f"   Subreddits: {len(subreddits)} unique")
    print()
    
    print(f"📋 AVAILABLE SUBREDDITS (Top 20):")
    sorted_subreddits = sorted(subreddits.items(), key=lambda x: x[1]['size_mb'], reverse=True)
    
    for i, (subreddit, info) in enumerate(sorted_subreddits[:20], 1):
        comments = "✅" if info['comments'] else "❌"
        submissions = "✅" if info['submissions'] else "❌"
        print(f"   {i:2d}. {subreddit:25s} | Comments: {comments} | Posts: {submissions} | {info['size_mb']:6.1f}MB")
    
    if len(sorted_subreddits) > 20:
        print(f"   ... and {len(sorted_subreddits) - 20} more subreddits")
    
    print()
    print(f"🎯 PROCESSING ESTIMATE:")
    # Conservative estimate: ~2000 records per MB compressed
    estimated_records = int(total_size / 500)
    print(f"   Estimated Records: {estimated_records:,}")
    print(f"   Estimated Processing Time: {total_size_gb * 2:.0f}-{total_size_gb * 8:.0f} hours")
    print(f"   Expected Clean Output: {total_size_gb * 0.3:.1f}-{total_size_gb * 0.6:.1f}GB")
    
    return True

def main():
    """Main function"""
    success = analyze_subreddits24()
    
    if success:
        print()
        print("🚀 READY TO PROCESS!")
        print("Next steps:")
        print("1. pip install -r requirements.txt")
        print("2. python setup_massive_dataset.py --reddit-dir subreddits24 --analyze-only")
        print("3. python setup_massive_dataset.py --reddit-dir subreddits24 --extract")
        print("4. python process_massive_dataset.py")
    else:
        print()
        print("❌ Setup needed before processing")

if __name__ == "__main__":
    main() 