#!/usr/bin/env python3
"""
Simple Collection Monitor
Tracks both Reddit and Multi-Source collections
"""

import time
import os
from pathlib import Path
from datetime import datetime

def get_dir_stats(directory):
    """Get directory statistics"""
    if not directory.exists():
        return {'size_gb': 0, 'files': 0, 'latest': None}
    
    total_size = 0
    file_count = 0
    latest_file = None
    latest_time = 0
    
    for file in directory.rglob('*.txt'):
        if file.is_file():
            size = file.stat().st_size
            total_size += size
            file_count += 1
            
            mtime = file.stat().st_mtime
            if mtime > latest_time:
                latest_time = mtime
                latest_file = file.name
    
    return {
        'size_gb': total_size / (1024**3),
        'files': file_count,
        'latest': latest_file,
        'latest_time': datetime.fromtimestamp(latest_time) if latest_time > 0 else None
    }

def monitor_simple():
    """Simple monitoring function"""
    start_time = datetime.now()
    
    while True:
        try:
            # Clear screen
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Get stats
            reddit_stats = get_dir_stats(Path('data/reddit_ultra_fast'))
            multi_stats = get_dir_stats(Path('data/multi_source_ultra_fast'))
            
            total_size = reddit_stats['size_gb'] + multi_stats['size_gb']
            total_files = reddit_stats['files'] + multi_stats['files']
            
            elapsed = datetime.now() - start_time
            elapsed_hours = elapsed.total_seconds() / 3600
            
            # Calculate rates
            if elapsed_hours > 0.05:  # At least 3 minutes
                rate_gb_hour = total_size / elapsed_hours
                hours_to_10gb = (10 - total_size) / rate_gb_hour if rate_gb_hour > 0 else 0
            else:
                rate_gb_hour = 0
                hours_to_10gb = 0
            
            print("🚀 DUAL COLLECTION MONITOR (SIMPLE)")
            print("=" * 50)
            print(f"⏱️  Running Time: {elapsed}")
            print(f"🕐 Current Time: {datetime.now().strftime('%H:%M:%S')}")
            print()
            
            print("📊 COLLECTION STATUS")
            print("-" * 30)
            print(f"🔴 Reddit:      {reddit_stats['size_gb']:.3f} GB ({reddit_stats['files']} files)")
            print(f"🌐 Multi-Source: {multi_stats['size_gb']:.3f} GB ({multi_stats['files']} files)")
            print(f"📈 TOTAL:       {total_size:.3f} GB ({total_files} files)")
            print()
            
            print("⚡ PERFORMANCE")
            print("-" * 30)
            if rate_gb_hour > 0:
                print(f"📊 Rate: {rate_gb_hour:.3f} GB/hour")
                if hours_to_10gb > 0:
                    print(f"🎯 ETA to 10GB: {hours_to_10gb:.1f} hours")
                print(f"🚀 24h Projection: {rate_gb_hour * 24:.1f} GB")
            else:
                print("📊 Warming up...")
            print()
            
            # Collection Health
            reddit_active = reddit_stats['latest_time'] and (datetime.now() - reddit_stats['latest_time']).seconds < 300
            multi_active = multi_stats['latest_time'] and (datetime.now() - multi_stats['latest_time']).seconds < 300
            
            print("🏥 HEALTH STATUS")
            print("-" * 30)
            print(f"🔴 Reddit:      {'🟢 ACTIVE' if reddit_active else '🔴 INACTIVE'}")
            print(f"🌐 Multi-Source: {'🟢 ACTIVE' if multi_active else '🔴 INACTIVE'}")
            print()
            
            # Progress
            if total_size >= 10:
                print("✅ 10GB TARGET: ACHIEVED!")
            elif total_size >= 1:
                print(f"🟡 10GB TARGET: {(total_size/10)*100:.1f}% Complete")
            else:
                print("🔴 10GB TARGET: Starting up...")
            
            print()
            print("🔥 BOTH COLLECTORS RUNNING SEPARATELY")
            print("🔴 Reddit = Reddit API only")
            print("🌐 Multi = WikiHow + OpenWeb + Educational")
            print("⚡ ZERO API conflicts!")
            print()
            print("Press Ctrl+C to stop monitoring")
            
            time.sleep(15)  # Update every 15 seconds
            
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")
            print(f"📊 Final Status: {total_size:.3f} GB collected")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(15)

if __name__ == "__main__":
    print("🚀 Starting Simple Dual Collection Monitor...")
    monitor_simple() 