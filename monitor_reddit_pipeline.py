#!/usr/bin/env python3
"""
Comprehensive Reddit Data Pipeline Monitor
Real-time monitoring for subreddits24 processing
"""

import os
import time
import psutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
import subprocess

class RedditPipelineMonitor:
    """Monitor Reddit data processing pipeline"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.data_dir = Path("data")
        self.subreddits_dir = Path("subreddits24")
        self.logs_dir = Path("logs")
        
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            'memory_used_gb': (memory.total - memory.available) / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'memory_percent': memory.percent,
            'disk_free_gb': disk.free / (1024**3),
            'disk_used_gb': disk.used / (1024**3),
            'disk_percent': (disk.used / disk.total) * 100,
            'cpu_percent': cpu_percent,
            'cpu_cores': psutil.cpu_count()
        }
    
    def get_reddit_data_stats(self) -> Dict:
        """Analyze subreddits24 data"""
        if not self.subreddits_dir.exists():
            return {'error': 'subreddits24 directory not found'}
        
        zst_files = list(self.subreddits_dir.glob("*.zst"))
        total_size = sum(f.stat().st_size for f in zst_files)
        
        # Group by subreddit
        subreddits = {}
        for file in zst_files:
            name_parts = file.stem.split('_')
            if len(name_parts) >= 2:
                subreddit = name_parts[0]
                file_type = name_parts[1] if len(name_parts) > 1 else 'unknown'
                
                if subreddit not in subreddits:
                    subreddits[subreddit] = {'comments': 0, 'submissions': 0, 'size_mb': 0}
                
                if 'comments' in file_type:
                    subreddits[subreddit]['comments'] = 1
                elif 'submissions' in file_type:
                    subreddits[subreddit]['submissions'] = 1
                
                subreddits[subreddit]['size_mb'] += file.stat().st_size / (1024**2)
        
        return {
            'total_files': len(zst_files),
            'total_size_gb': total_size / (1024**3),
            'unique_subreddits': len(subreddits),
            'subreddits': dict(list(subreddits.items())[:10])  # Top 10 for display
        }
    
    def get_processing_stats(self) -> Dict:
        """Get processing progress statistics"""
        stats = {}
        
        # Check unified raw data
        unified_dir = self.data_dir / "unified_raw"
        if unified_dir.exists():
            unified_files = list(unified_dir.glob("*.txt"))
            stats['unified_chunks'] = len(unified_files)
            stats['unified_size_gb'] = sum(f.stat().st_size for f in unified_files) / (1024**3)
        else:
            stats['unified_chunks'] = 0
            stats['unified_size_gb'] = 0
        
        # Check cleaned data
        cleaned_dir = self.data_dir / "cleaned"
        if cleaned_dir.exists():
            cleaned_file = cleaned_dir / "cleaned_reddit_data.jsonl"
            if cleaned_file.exists():
                stats['cleaned_size_gb'] = cleaned_file.stat().st_size / (1024**3)
                # Estimate records (rough calculation)
                stats['estimated_records'] = int(stats['cleaned_size_gb'] * 200000)  # ~200k records per GB
            else:
                stats['cleaned_size_gb'] = 0
                stats['estimated_records'] = 0
        else:
            stats['cleaned_size_gb'] = 0
            stats['estimated_records'] = 0
        
        # Check logs
        log_file = self.logs_dir / "massive_processing.log"
        if log_file.exists():
            stats['log_size_mb'] = log_file.stat().st_size / (1024**2)
            stats['log_updated'] = datetime.fromtimestamp(log_file.stat().st_mtime)
        else:
            stats['log_size_mb'] = 0
            stats['log_updated'] = None
        
        return stats
    
    def get_process_status(self) -> Dict:
        """Check if processing is running"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
            try:
                if any('process_massive_dataset' in ' '.join(proc.info['cmdline']) for cmd in [proc.info['cmdline']] if cmd):
                    processes.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'runtime_hours': (time.time() - proc.info['create_time']) / 3600,
                        'memory_mb': proc.info['memory_info'].rss / (1024**2)
                    })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return {
            'running': len(processes) > 0,
            'processes': processes
        }
    
    def get_latest_logs(self, lines: int = 5) -> List[str]:
        """Get latest log entries"""
        log_file = self.logs_dir / "massive_processing.log"
        if not log_file.exists():
            return ["No log file found"]
        
        try:
            result = subprocess.run(['tail', f'-{lines}', str(log_file)], 
                                  capture_output=True, text=True)
            return result.stdout.strip().split('\n') if result.stdout else ["Log file empty"]
        except:
            # Fallback for Windows
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines_list = f.readlines()
                    return [line.strip() for line in lines_list[-lines:]]
            except:
                return ["Error reading log file"]
    
    def display_dashboard(self):
        """Display comprehensive monitoring dashboard"""
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Header
            print("🔍 Reddit Data Pipeline Monitor - subreddits24")
            print("=" * 60)
            print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"🕐 Runtime: {datetime.now() - self.start_time}")
            print()
            
            # System Status
            sys_stats = self.get_system_stats()
            print("💻 SYSTEM RESOURCES:")
            print(f"   RAM: {sys_stats['memory_used_gb']:.1f}/{sys_stats['memory_total_gb']:.1f}GB ({sys_stats['memory_percent']:.1f}%)")
            print(f"   CPU: {sys_stats['cpu_percent']:.1f}% ({sys_stats['cpu_cores']} cores)")
            print(f"   Disk: {sys_stats['disk_used_gb']:.1f}GB used ({sys_stats['disk_percent']:.1f}%)")
            print()
            
            # Reddit Data Overview
            reddit_stats = self.get_reddit_data_stats()
            if 'error' not in reddit_stats:
                print("📱 REDDIT DATA (subreddits24):")
                print(f"   Files: {reddit_stats['total_files']} .zst files")
                print(f"   Size: {reddit_stats['total_size_gb']:.1f}GB")
                print(f"   Subreddits: {reddit_stats['unique_subreddits']} unique")
                print()
            
            # Processing Status
            proc_status = self.get_process_status()
            process_stats = self.get_processing_stats()
            
            print("⚙️ PROCESSING STATUS:")
            if proc_status['running']:
                for proc in proc_status['processes']:
                    print(f"   ✅ RUNNING (PID: {proc['pid']})")
                    print(f"      Runtime: {proc['runtime_hours']:.1f} hours")
                    print(f"      Memory: {proc['memory_mb']:.1f}MB")
            else:
                print("   ❌ NOT RUNNING")
            print()
            
            # Data Progress
            print("📊 DATA PROGRESS:")
            print(f"   Unified chunks: {process_stats['unified_chunks']} files ({process_stats['unified_size_gb']:.1f}GB)")
            print(f"   Cleaned data: {process_stats['cleaned_size_gb']:.1f}GB")
            print(f"   Estimated records: {process_stats['estimated_records']:,}")
            if process_stats['log_updated']:
                time_diff = datetime.now() - process_stats['log_updated']
                print(f"   Last activity: {time_diff.total_seconds():.0f}s ago")
            print()
            
            # Latest Logs
            print("📝 LATEST ACTIVITY:")
            latest_logs = self.get_latest_logs(3)
            for log_line in latest_logs:
                if log_line.strip():
                    print(f"   {log_line[:80]}...")
            print()
            
            # Processing Estimates
            if proc_status['running'] and process_stats['unified_chunks'] > 0:
                # Rough estimate based on typical processing rates
                remaining_gb = reddit_stats.get('total_size_gb', 0) - process_stats['cleaned_size_gb']
                if remaining_gb > 0:
                    # Estimate ~1GB per hour for typical systems
                    est_hours = remaining_gb * 2  # Conservative estimate
                    print(f"📈 ESTIMATED TIME REMAINING: {est_hours:.1f} hours")
                    print()
            
            # Controls
            print("🎮 CONTROLS:")
            print("   Ctrl+C to exit monitor")
            print("   Data location: subreddits24/")
            print("   Logs: logs/massive_processing.log")
            
            time.sleep(10)  # Update every 10 seconds

def main():
    """Main monitoring function"""
    try:
        monitor = RedditPipelineMonitor()
        monitor.display_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"❌ Monitor error: {e}")

if __name__ == "__main__":
    main() 