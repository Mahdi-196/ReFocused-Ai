#!/usr/bin/env python3
"""
Real-time training monitor for ReFocused-AI
Provides live monitoring of training progress, preprocessing status, and system metrics.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import threading
from typing import Dict, Any, Optional
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from torch.utils.tensorboard import SummaryWriter
    EventFileWriter = None
except ImportError:
    from tensorboardX import SummaryWriter
    EventFileWriter = None

try:
    import psutil
    import torch
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install psutil torch tensorboard")
    sys.exit(1)


class TrainingMonitor:
    """Monitor training progress and system performance"""
    
    def __init__(self, log_dir: str, cache_dir: str = "./cache", preprocess_cache_dir: str = "./preprocessed_cache"):
        self.log_dir = Path(log_dir)
        self.cache_dir = Path(cache_dir)
        self.preprocess_cache_dir = Path(preprocess_cache_dir)
        self.monitoring = False
        
        # Find the latest run
        self.current_run = self.find_latest_run()
        if self.current_run:
            print(f"Monitoring run: {self.current_run}")
        else:
            print("No active training runs found")
    
    def find_latest_run(self) -> Optional[str]:
        """Find the most recent training run"""
        if not self.log_dir.exists():
            return None
        
        run_dirs = [d for d in self.log_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            return None
        
        # Sort by modification time and get the latest
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        return latest_run.name
    
    def get_preprocessing_status(self) -> Dict[str, Any]:
        """Get status of preprocessing cache"""
        status = {
            'cache_exists': self.preprocess_cache_dir.exists(),
            'cached_files': 0,
            'cache_size_mb': 0,
            'raw_files': 0
        }
        
        if self.preprocess_cache_dir.exists():
            cached_files = list(self.preprocess_cache_dir.glob("*.pkl"))
            status['cached_files'] = len(cached_files)
            
            total_size = sum(f.stat().st_size for f in cached_files)
            status['cache_size_mb'] = total_size / (1024 * 1024)
        
        if self.cache_dir.exists():
            raw_files = list(self.cache_dir.glob("*.npz"))
            status['raw_files'] = len(raw_files)
        
        return status
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        metrics = {}
        
        # CPU
        metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        metrics['cpu_count'] = psutil.cpu_count()
        
        # Memory
        memory = psutil.virtual_memory()
        metrics['memory_percent'] = memory.percent
        metrics['memory_used_gb'] = memory.used / (1024**3)
        metrics['memory_total_gb'] = memory.total / (1024**3)
        
        # Disk
        if self.log_dir.parent.exists():
            disk = psutil.disk_usage(str(self.log_dir.parent))
            metrics['disk_percent'] = (disk.used / disk.total) * 100
            metrics['disk_free_gb'] = disk.free / (1024**3)
        
        # GPU if available
        if torch.cuda.is_available():
            metrics['gpu_count'] = torch.cuda.device_count()
            metrics['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            metrics['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        
        return metrics
    
    def read_tensorboard_metrics(self) -> Dict[str, Any]:
        """Read latest metrics from tensorboard logs"""
        if not self.current_run:
            return {}
        
        tb_dir = self.log_dir / self.current_run
        if not tb_dir.exists():
            return {}
        
        try:
            # TODO: EventAccumulator functionality removed - implement alternative method
            # Find the latest event file
            # event_files = list(tb_dir.glob("events.out.tfevents.*"))
            # if not event_files:
            #     return {}
            
            # latest_event = max(event_files, key=lambda x: x.stat().st_mtime)
            
            # Read tensorboard events
            # ea = EventAccumulator(str(latest_event))
            # ea.Reload()
            
            metrics = {}
            
            # Get scalar summaries
            # scalar_tags = ea.Tags()['scalars']
            # for tag in scalar_tags:
            #     try:
            #         scalar_events = ea.Scalars(tag)
            #         if scalar_events:
            #             latest_value = scalar_events[-1].value
            #             latest_step = scalar_events[-1].step
            #             metrics[tag] = {'value': latest_value, 'step': latest_step}
            #     except:
            #         continue
            
            return metrics
        
        except Exception as e:
            print(f"Error reading tensorboard logs: {e}")
            return {}
    
    def read_summary_file(self) -> Dict[str, Any]:
        """Read training summary file if it exists"""
        if not self.current_run:
            return {}
        
        summary_file = self.log_dir / f"{self.current_run}_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def print_status(self):
        """Print current training status"""
        print("\n" + "="*80)
        print(f"ReFocused-AI Training Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Preprocessing status
        preprocess_status = self.get_preprocessing_status()
        print("\n📁 PREPROCESSING STATUS:")
        print(f"  Cache Directory: {self.preprocess_cache_dir}")
        print(f"  Cached Files: {preprocess_status['cached_files']}")
        print(f"  Cache Size: {preprocess_status['cache_size_mb']:.1f} MB")
        print(f"  Raw Files: {preprocess_status['raw_files']}")
        
        # Training metrics
        tb_metrics = self.read_tensorboard_metrics()
        print("\n📊 TRAINING METRICS:")
        if tb_metrics:
            # Show key metrics
            key_metrics = ['train/loss', 'train/learning_rate', 'train/grad_norm', 'speed/samples_per_second']
            for metric in key_metrics:
                if metric in tb_metrics:
                    value = tb_metrics[metric]['value']
                    step = tb_metrics[metric]['step']
                    print(f"  {metric}: {value:.6f} (step {step})")
            
            # Show performance metrics if available
            perf_metrics = [k for k in tb_metrics.keys() if k.startswith('performance/')]
            if perf_metrics:
                print("\n⚡ PERFORMANCE METRICS:")
                for metric in perf_metrics[:5]:  # Show top 5
                    value = tb_metrics[metric]['value']
                    print(f"  {metric}: {value:.3f}")
        else:
            print("  No training metrics available yet")
        
        # System metrics
        sys_metrics = self.get_system_metrics()
        print("\n💻 SYSTEM STATUS:")
        print(f"  CPU Usage: {sys_metrics['cpu_percent']:.1f}%")
        print(f"  Memory Usage: {sys_metrics['memory_percent']:.1f}% ({sys_metrics['memory_used_gb']:.1f}/{sys_metrics['memory_total_gb']:.1f} GB)")
        if 'disk_percent' in sys_metrics:
            print(f"  Disk Usage: {sys_metrics['disk_percent']:.1f}% (Free: {sys_metrics['disk_free_gb']:.1f} GB)")
        
        if 'gpu_count' in sys_metrics:
            print(f"  GPU Count: {sys_metrics['gpu_count']}")
            print(f"  GPU Memory: {sys_metrics['gpu_memory_allocated_gb']:.1f} GB allocated, {sys_metrics['gpu_memory_reserved_gb']:.1f} GB reserved")
        
        # Training summary
        summary = self.read_summary_file()
        if summary:
            print("\n📈 TRAINING SUMMARY:")
            print(f"  Run Name: {summary.get('run_name', 'Unknown')}")
            print(f"  Total Steps: {summary.get('total_steps', 0)}")
            if summary.get('final_loss'):
                print(f"  Final Loss: {summary['final_loss']:.6f}")
            if summary.get('avg_step_time'):
                print(f"  Avg Step Time: {summary['avg_step_time']:.3f}s")
    
    def monitor_loop(self, refresh_seconds: int = 10):
        """Main monitoring loop"""
        self.monitoring = True
        print(f"Starting training monitor (refresh every {refresh_seconds}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while self.monitoring:
                # Clear screen (works on most terminals)
                os.system('cls' if os.name == 'nt' else 'clear')
                
                self.print_status()
                
                time.sleep(refresh_seconds)
        
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            print(f"\nMonitoring error: {e}")
        finally:
            self.monitoring = False
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring = False


def main():
    parser = argparse.ArgumentParser(description="Monitor ReFocused-AI training progress")
    parser.add_argument("--log-dir", type=str, default="./logs", 
                       help="Directory containing training logs")
    parser.add_argument("--cache-dir", type=str, default="./cache",
                       help="Directory containing raw cache files")
    parser.add_argument("--preprocess-cache-dir", type=str, default="./preprocessed_cache",
                       help="Directory containing preprocessed cache files")
    parser.add_argument("--refresh", type=int, default=10,
                       help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true",
                       help="Show status once and exit")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(
        log_dir=args.log_dir,
        cache_dir=args.cache_dir,
        preprocess_cache_dir=args.preprocess_cache_dir
    )
    
    if args.once:
        monitor.print_status()
    else:
        monitor.monitor_loop(args.refresh)


if __name__ == "__main__":
    main() 