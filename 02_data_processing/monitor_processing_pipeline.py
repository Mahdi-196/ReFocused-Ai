import os
import time
import json
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

from loguru import logger


class ProcessingPipelineMonitor:
    """
    Real-time monitoring for data processing pipeline
    Tracks processing progress, system resources, and output statistics
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.cleaned_dir = self.data_dir / "cleaned"
        self.real_data_dir = self.data_dir / "real_multi_source"
        
        self.start_time = datetime.now()
        self.last_update = datetime.now()
        
        # Processing stats
        self.stats = {
            'monitoring_start': self.start_time.isoformat(),
            'last_check': self.last_update.isoformat(),
            'processing_files': [],
            'completed_files': [],
            'total_processed_items': 0,
            'processing_rate_per_hour': 0.0,
            'estimated_completion': None
        }
        
    def get_system_resources(self) -> Dict:
        """Get current system resource usage"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Process count
            process_count = len(psutil.pids())
            
            # Load average (Linux/Mac)
            try:
                load_avg = os.getloadavg()
                load_1min, load_5min, load_15min = load_avg
            except (OSError, AttributeError):
                # Windows doesn't have getloadavg
                load_1min = load_5min = load_15min = cpu_percent / 100
            
            return {
                'cpu_percent': round(cpu_percent, 1),
                'cpu_cores': psutil.cpu_count(),
                'memory_percent': round(memory.percent, 1),
                'memory_used_gb': round(memory.used / (1024**3), 1),
                'memory_total_gb': round(memory.total / (1024**3), 1),
                'disk_percent': round(disk.percent, 1),
                'disk_used_gb': round(disk.used / (1024**3), 1),
                'disk_free_gb': round(disk.free / (1024**3), 1),
                'disk_total_gb': round(disk.total / (1024**3), 1),
                'process_count': process_count,
                'load_average': {
                    '1min': round(load_1min, 2),
                    '5min': round(load_5min, 2),
                    '15min': round(load_15min, 2)
                }
            }
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {}
    
    def get_processing_status(self) -> Dict:
        """Check status of processing scripts"""
        processing_status = {
            'active_processes': [],
            'python_processes': 0,
            'processing_files': []
        }
        
        try:
            # Check for active Python processing scripts
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'memory_info']):
                try:
                    pinfo = proc.info
                    if pinfo['name'] and 'python' in pinfo['name'].lower():
                        cmdline = ' '.join(pinfo['cmdline']) if pinfo['cmdline'] else ''
                        
                        # Check for our processing scripts
                        processing_scripts = [
                            'multi_source_processor.py',
                            'data_cleaner.py', 
                            'data_processor.py',
                            'process_massive_dataset.py'
                        ]
                        
                        for script in processing_scripts:
                            if script in cmdline:
                                runtime = datetime.now() - datetime.fromtimestamp(pinfo['create_time'])
                                memory_mb = pinfo['memory_info'].rss / (1024 * 1024) if pinfo['memory_info'] else 0
                                
                                processing_status['active_processes'].append({
                                    'pid': pinfo['pid'],
                                    'script': script,
                                    'runtime': str(runtime).split('.')[0],
                                    'memory_mb': round(memory_mb, 1)
                                })
                                break
                        
                        if 'python' in cmdline.lower():
                            processing_status['python_processes'] += 1
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
        except Exception as e:
            logger.error(f"Error checking processing status: {e}")
        
        return processing_status
    
    def analyze_data_directories(self) -> Dict:
        """Analyze data in various processing directories"""
        analysis = {
            'real_multi_source': self._analyze_directory(self.real_data_dir),
            'processed': self._analyze_directory(self.processed_dir),
            'cleaned': self._analyze_directory(self.cleaned_dir)
        }
        
        return analysis
    
    def _analyze_directory(self, directory: Path) -> Dict:
        """Analyze a specific data directory"""
        if not directory.exists():
            return {
                'exists': False,
                'file_count': 0,
                'total_size_gb': 0.0,
                'latest_file': None,
                'file_types': {}
            }
        
        try:
            files = list(directory.rglob('*'))
            data_files = [f for f in files if f.is_file()]
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in data_files if f.exists())
            
            # Get file types
            file_types = {}
            for f in data_files:
                ext = f.suffix.lower()
                if ext not in file_types:
                    file_types[ext] = {'count': 0, 'size_gb': 0.0}
                file_types[ext]['count'] += 1
                file_types[ext]['size_gb'] += f.stat().st_size / (1024**3)
            
            # Round file type sizes
            for ext in file_types:
                file_types[ext]['size_gb'] = round(file_types[ext]['size_gb'], 2)
            
            # Get latest file
            latest_file = None
            if data_files:
                latest_file_obj = max(data_files, key=lambda x: x.stat().st_mtime)
                latest_file = {
                    'name': latest_file_obj.name,
                    'modified': datetime.fromtimestamp(latest_file_obj.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'size_mb': round(latest_file_obj.stat().st_size / (1024**2), 2)
                }
            
            return {
                'exists': True,
                'file_count': len(data_files),
                'total_size_gb': round(total_size / (1024**3), 2),
                'latest_file': latest_file,
                'file_types': file_types
            }
            
        except Exception as e:
            logger.error(f"Error analyzing directory {directory}: {e}")
            return {
                'exists': True,
                'file_count': 0,
                'total_size_gb': 0.0,
                'latest_file': None,
                'file_types': {},
                'error': str(e)
            }
    
    def count_processed_items(self) -> Dict:
        """Count items in processed files"""
        item_counts = {
            'train': 0,
            'validation': 0,
            'test': 0,
            'total': 0
        }
        
        # Count items in processed splits
        for split in ['train', 'validation', 'test']:
            jsonl_file = self.processed_dir / f"{split}.jsonl"
            if jsonl_file.exists():
                try:
                    with open(jsonl_file, 'r', encoding='utf-8') as f:
                        count = sum(1 for line in f if line.strip())
                    item_counts[split] = count
                    item_counts['total'] += count
                except Exception as e:
                    logger.error(f"Error counting items in {jsonl_file}: {e}")
        
        return item_counts
    
    def get_processing_metadata(self) -> Dict:
        """Get processing metadata if available"""
        metadata_file = self.processed_dir / "processing_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading processing metadata: {e}")
                return {}
        else:
            return {}
    
    def display_status_dashboard(self):
        """Display comprehensive status dashboard"""
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        runtime = datetime.now() - self.start_time
        
        print("🔍 Data Processing Pipeline Monitor")
        print("=" * 60)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕐 Monitor Runtime: {str(runtime).split('.')[0]}")
        print()
        
        # System Resources
        resources = self.get_system_resources()
        if resources:
            print("💻 SYSTEM RESOURCES:")
            print(f"   CPU: {resources['cpu_percent']}% ({resources['cpu_cores']} cores)")
            print(f"   RAM: {resources['memory_used_gb']}/{resources['memory_total_gb']}GB ({resources['memory_percent']}%)")
            print(f"   Disk: {resources['disk_used_gb']}GB used ({resources['disk_percent']}%)")
            print(f"   Free Space: {resources['disk_free_gb']}GB")
            print(f"   Load Average: {resources['load_average']['1min']} (1m), {resources['load_average']['5min']} (5m)")
            print()
        
        # Processing Status
        processing = self.get_processing_status()
        print("⚙️ PROCESSING STATUS:")
        if processing['active_processes']:
            for proc in processing['active_processes']:
                print(f"   ✅ {proc['script']} (PID: {proc['pid']})")
                print(f"      Runtime: {proc['runtime']}, Memory: {proc['memory_mb']}MB")
        else:
            print("   ⏸️  No active processing scripts detected")
        
        if processing['python_processes'] > 0:
            print(f"   🐍 Python processes: {processing['python_processes']}")
        print()
        
        # Data Analysis
        data_analysis = self.analyze_data_directories()
        
        # Real-time collected data
        real_data = data_analysis['real_multi_source']
        print("📊 REAL-TIME COLLECTED DATA:")
        if real_data['exists'] and real_data['file_count'] > 0:
            print(f"   Files: {real_data['file_count']}")
            print(f"   Size: {real_data['total_size_gb']}GB")
            if real_data['file_types']:
                for ext, info in real_data['file_types'].items():
                    print(f"   {ext}: {info['count']} files ({info['size_gb']}GB)")
            if real_data['latest_file']:
                print(f"   Latest: {real_data['latest_file']['name']} ({real_data['latest_file']['modified']})")
        else:
            print("   📭 No real-time data found")
        print()
        
        # Processed data
        processed_data = data_analysis['processed']
        print("🎯 PROCESSED DATA:")
        if processed_data['exists'] and processed_data['file_count'] > 0:
            print(f"   Files: {processed_data['file_count']}")
            print(f"   Size: {processed_data['total_size_gb']}GB")
            
            # Count items in splits
            item_counts = self.count_processed_items()
            if item_counts['total'] > 0:
                print(f"   Training Items: {item_counts['train']:,}")
                print(f"   Validation Items: {item_counts['validation']:,}")
                print(f"   Test Items: {item_counts['test']:,}")
                print(f"   Total Items: {item_counts['total']:,}")
            
            if processed_data['latest_file']:
                print(f"   Latest: {processed_data['latest_file']['name']} ({processed_data['latest_file']['modified']})")
        else:
            print("   📭 No processed data found")
        print()
        
        # Processing metadata
        metadata = self.get_processing_metadata()
        if metadata:
            print("📈 PROCESSING METADATA:")
            if 'processing_info' in metadata:
                info = metadata['processing_info']
                print(f"   Processed At: {info.get('processed_at', 'Unknown')}")
                print(f"   Total Items: {info.get('total_items', 0):,}")
                
                if 'processing_stats' in info:
                    stats = info['processing_stats']
                    print(f"   Wikipedia Articles: {stats.get('wikipedia_articles', 0):,}")
                    print(f"   Reddit Posts: {stats.get('reddit_posts', 0):,}")
                    print(f"   Removed Duplicates: {stats.get('removed_duplicates', 0):,}")
                    print(f"   Processing Time: {stats.get('processing_time', 0):.1f}s")
            
            if 'source_distribution' in metadata:
                print("   Source Distribution:")
                for source, count in metadata['source_distribution'].items():
                    print(f"     {source}: {count:,}")
            print()
        
        # Cleaned data (legacy)
        cleaned_data = data_analysis['cleaned']
        if cleaned_data['exists'] and cleaned_data['file_count'] > 0:
            print("🧹 CLEANED DATA (Legacy):")
            print(f"   Files: {cleaned_data['file_count']}")
            print(f"   Size: {cleaned_data['total_size_gb']}GB")
            if cleaned_data['latest_file']:
                print(f"   Latest: {cleaned_data['latest_file']['name']} ({cleaned_data['latest_file']['modified']})")
            print()
        
        print("=" * 60)
        print("Press Ctrl+C to stop monitoring")
    
    def run_monitoring_loop(self, update_interval: int = 30):
        """Run the monitoring loop"""
        logger.info(f"🚀 Starting data processing pipeline monitor (updates every {update_interval}s)")
        
        try:
            while True:
                self.display_status_dashboard()
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
            logger.info("Data processing monitor stopped")
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            print(f"\n❌ Monitor error: {e}")


def main():
    """Main execution function"""
    monitor = ProcessingPipelineMonitor()
    
    # Display initial status
    monitor.display_status_dashboard()
    
    # Ask user if they want continuous monitoring
    try:
        response = input("\nStart continuous monitoring? (y/n): ").lower().strip()
        if response in ['y', 'yes', '']:
            monitor.run_monitoring_loop(update_interval=30)
        else:
            print("Single status check completed.")
    except KeyboardInterrupt:
        print("\n🛑 Monitoring cancelled")


if __name__ == "__main__":
    main() 