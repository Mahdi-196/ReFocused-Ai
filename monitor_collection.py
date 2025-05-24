#!/usr/bin/env python3
"""
Monitor Reddit data collection progress
Shows real-time statistics about collected data
"""

import os
import time
import gzip
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

def format_size(bytes_size: int) -> str:
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def analyze_compressed_file(file_path: Path) -> Dict:
    """Analyze a compressed data file and return statistics"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            posts = []
            total_comments = 0
            for line in f:
                try:
                    post = json.loads(line.strip())
                    posts.append(post)
                    # Count comments if they exist
                    total_comments += len(post.get('comments', []))
                except json.JSONDecodeError:
                    continue
        
        if not posts:
            return {'post_count': 0, 'file_size': file_path.stat().st_size}
        
        # Calculate statistics
        scores = [post.get('score', 0) for post in posts]
        comment_counts = [post.get('num_comments', 0) for post in posts]
        
        stats = {
            'post_count': len(posts),
            'comment_count': total_comments,
            'file_size': file_path.stat().st_size,
            'avg_score': sum(scores) / len(scores) if scores else 0,
            'max_score': max(scores) if scores else 0,
            'min_score': min(scores) if scores else 0,
            'avg_comments': sum(comment_counts) / len(comment_counts) if comment_counts else 0,
            'total_comment_references': sum(comment_counts),
            'earliest_post': min(post.get('created_utc', 0) for post in posts),
            'latest_post': max(post.get('created_utc', 0) for post in posts)
        }
        
        return stats
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return {'post_count': 0, 'file_size': file_path.stat().st_size, 'error': str(e)}

def get_collection_stats(data_dirs: List[Path]) -> Dict:
    """Get overall collection statistics from multiple directories"""
    stats = {
        'total_files': 0,
        'total_size': 0,
        'total_posts': 0,
        'total_comments': 0,
        'subreddits': {},
        'collection_start': None,
        'last_update': None,
        'directories': {}
    }
    
    for data_dir in data_dirs:
        dir_name = data_dir.name
        stats['directories'][dir_name] = {'files': 0, 'size': 0, 'posts': 0}
        
        if not data_dir.exists():
            continue
        
        # Find all .gz files
        gz_files = list(data_dir.glob('*.txt.gz'))
        stats['total_files'] += len(gz_files)
        stats['directories'][dir_name]['files'] = len(gz_files)
        
        if not gz_files:
            continue
        
        # Get file timestamps
        file_times = [f.stat().st_mtime for f in gz_files]
        dir_start = datetime.fromtimestamp(min(file_times))
        dir_update = datetime.fromtimestamp(max(file_times))
        
        if not stats['collection_start'] or dir_start < stats['collection_start']:
            stats['collection_start'] = dir_start
        if not stats['last_update'] or dir_update > stats['last_update']:
            stats['last_update'] = dir_update
        
        # Analyze each file
        for file_path in gz_files:
            # Extract subreddit name from filename
            subreddit = file_path.stem.split('_')[0]
            
            if subreddit not in stats['subreddits']:
                stats['subreddits'][subreddit] = {
                    'files': 0,
                    'size': 0,
                    'posts': 0,
                    'comments': 0,
                    'total_score': 0,
                    'total_comment_refs': 0,
                    'latest_file': None,
                    'directories': set()
                }
            
            # Analyze file
            file_stats = analyze_compressed_file(file_path)
            
            # Update subreddit stats
            sub_stats = stats['subreddits'][subreddit]
            sub_stats['files'] += 1
            sub_stats['size'] += file_stats['file_size']
            sub_stats['posts'] += file_stats['post_count']
            sub_stats['comments'] += file_stats.get('comment_count', 0)
            sub_stats['directories'].add(dir_name)
            
            if 'avg_score' in file_stats:
                sub_stats['total_score'] += file_stats['avg_score'] * file_stats['post_count']
            if 'total_comment_references' in file_stats:
                sub_stats['total_comment_refs'] += file_stats['total_comment_references']
            
            # Track latest file
            if not sub_stats['latest_file'] or file_path.stat().st_mtime > sub_stats['latest_file']['timestamp']:
                sub_stats['latest_file'] = {
                    'name': file_path.name,
                    'timestamp': file_path.stat().st_mtime,
                    'posts': file_stats['post_count'],
                    'directory': dir_name
                }
            
            # Update totals
            stats['total_size'] += file_stats['file_size']
            stats['total_posts'] += file_stats['post_count']
            stats['total_comments'] += file_stats.get('comment_count', 0)
            stats['directories'][dir_name]['size'] += file_stats['file_size']
            stats['directories'][dir_name]['posts'] += file_stats['post_count']
    
    # Calculate averages and clean up subreddit data
    for subreddit, sub_stats in stats['subreddits'].items():
        if sub_stats['posts'] > 0:
            sub_stats['avg_score'] = sub_stats['total_score'] / sub_stats['posts']
            sub_stats['avg_comments'] = sub_stats['total_comment_refs'] / sub_stats['posts']
        else:
            sub_stats['avg_score'] = 0
            sub_stats['avg_comments'] = 0
        
        # Convert set to list for display
        sub_stats['directories'] = list(sub_stats['directories'])
    
    return stats

def estimate_time_to_10gb(stats: Dict) -> str:
    """Estimate time to reach 10GB based on current collection rate"""
    if not stats['collection_start'] or stats['total_size'] == 0:
        return "Unable to estimate"
    
    goal_size = 10 * 1024 * 1024 * 1024  # 10GB in bytes
    if stats['total_size'] >= goal_size:
        return "Goal reached!"
    
    elapsed_hours = (datetime.now() - stats['collection_start']).total_seconds() / 3600
    if elapsed_hours < 0.1:  # Less than 6 minutes
        return "Too early to estimate"
    
    bytes_per_hour = stats['total_size'] / elapsed_hours
    remaining_bytes = goal_size - stats['total_size']
    
    if bytes_per_hour <= 0:
        return "Collection appears to have stopped"
    
    hours_remaining = remaining_bytes / bytes_per_hour
    
    if hours_remaining > 24:
        days = hours_remaining / 24
        return f"~{days:.1f} days"
    else:
        return f"~{hours_remaining:.1f} hours"

def print_detailed_stats(stats: Dict):
    """Print detailed collection statistics"""
    print("\n" + "=" * 70)
    print("🔍 REDDIT DATA COLLECTION STATISTICS")
    print("=" * 70)
    
    # Overall stats
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Files: {stats['total_files']}")
    print(f"   Total Size: {format_size(stats['total_size'])}")
    print(f"   Total Posts: {stats['total_posts']:,}")
    print(f"   Total Comments: {stats['total_comments']:,}")
    print(f"   Total Content Items: {stats['total_posts'] + stats['total_comments']:,}")
    
    if stats['collection_start']:
        print(f"   Collection Started: {stats['collection_start'].strftime('%Y-%m-%d %H:%M:%S')}")
        duration = datetime.now() - stats['collection_start']
        print(f"   Duration: {duration}")
        
        if stats['total_posts'] > 0:
            posts_per_hour = stats['total_posts'] / (duration.total_seconds() / 3600)
            print(f"   Collection Rate: {posts_per_hour:.1f} posts/hour")
    
    if stats['last_update']:
        time_since_update = datetime.now() - stats['last_update']
        print(f"   Last Update: {time_since_update} ago")
    
    # Progress towards 10GB goal
    goal_size = 10 * 1024 * 1024 * 1024  # 10GB in bytes
    progress_percent = (stats['total_size'] / goal_size) * 100
    print(f"   Progress to 10GB: {progress_percent:.2f}%")
    print(f"   Time to 10GB: {estimate_time_to_10gb(stats)}")
    
    # Directory breakdown
    if len(stats['directories']) > 1:
        print(f"\n📁 Directory Breakdown:")
        for dir_name, dir_stats in stats['directories'].items():
            print(f"   {dir_name}: {dir_stats['files']} files, {format_size(dir_stats['size'])}, {dir_stats['posts']:,} posts")
    
    # Subreddit breakdown
    print(f"\n📋 Subreddit Breakdown:")
    print(f"{'Subreddit':<20} {'Files':<6} {'Size':<10} {'Posts':<8} {'Comments':<9} {'Avg Score':<9} {'Sources':<15}")
    print("-" * 95)
    
    # Sort subreddits by size
    sorted_subreddits = sorted(
        stats['subreddits'].items(),
        key=lambda x: x[1]['size'],
        reverse=True
    )
    
    for subreddit, sub_stats in sorted_subreddits:
        sources = "+".join(sub_stats['directories']) if sub_stats['directories'] else "unknown"
        print(f"{subreddit:<20} {sub_stats['files']:<6} {format_size(sub_stats['size']):<10} "
              f"{sub_stats['posts']:<8} {sub_stats['comments']:<9} "
              f"{sub_stats['avg_score']:<9.1f} {sources:<15}")
        
        if sub_stats['latest_file']:
            latest_time = datetime.fromtimestamp(sub_stats['latest_file']['timestamp'])
            time_ago = datetime.now() - latest_time
            print(f"   └─ Latest: {sub_stats['latest_file']['name'][:50]} "
                  f"({sub_stats['latest_file']['posts']} posts, {time_ago} ago)")

def print_simple_stats(stats: Dict):
    """Print simple collection statistics"""
    print(f"\n📊 Collection Status: {stats['total_files']} files, "
          f"{format_size(stats['total_size'])}, {stats['total_posts']:,} posts, {stats['total_comments']:,} comments")
    
    if stats['total_size'] > 0:
        goal_size = 10 * 1024 * 1024 * 1024  # 10GB
        progress = (stats['total_size'] / goal_size) * 100
        print(f"🎯 Progress: {progress:.2f}% towards 10GB goal")
        print(f"⏱️  Time to 10GB: {estimate_time_to_10gb(stats)}")
    
    # Show active subreddits
    active_subreddits = len([s for s in stats['subreddits'].values() if s['posts'] > 0])
    total_subreddits = 10  # Target subreddits
    print(f"📈 Active Subreddits: {active_subreddits}/{total_subreddits}")

def monitor_collection(data_dirs: List[Path] = None, interval: int = 30, detailed: bool = False):
    """Monitor collection progress in real-time"""
    if data_dirs is None:
        data_dirs = [Path('data/reddit_oauth'), Path('data/reddit_enhanced')]
    
    print(f"👀 Monitoring Reddit data collection in: {', '.join(str(d) for d in data_dirs)}")
    print(f"🔄 Update interval: {interval} seconds")
    print("💡 Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            stats = get_collection_stats(data_dirs)
            
            # Clear screen (works on most terminals)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            if detailed:
                print_detailed_stats(stats)
            else:
                print_simple_stats(stats)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Reddit data collection progress')
    parser.add_argument('--dir', '-d', type=str, action='append',
                      help='Data directory to monitor (can be used multiple times)')
    parser.add_argument('--interval', '-i', type=int, default=30,
                      help='Update interval in seconds (default: 30)')
    parser.add_argument('--detailed', action='store_true',
                      help='Show detailed statistics')
    parser.add_argument('--once', action='store_true',
                      help='Show statistics once and exit')
    
    args = parser.parse_args()
    
    # Set up data directories
    if args.dir:
        data_dirs = [Path(d) for d in args.dir]
    else:
        data_dirs = [Path('data/reddit_oauth'), Path('data/reddit_enhanced')]
    
    if args.once:
        # Show stats once and exit
        stats = get_collection_stats(data_dirs)
        if args.detailed:
            print_detailed_stats(stats)
        else:
            print_simple_stats(stats)
    else:
        # Monitor continuously
        monitor_collection(data_dirs, args.interval, args.detailed)

if __name__ == "__main__":
    main() 