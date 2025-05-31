"""
Real-time training monitor for ReFocused-AI model
Displays GPU usage, training metrics, and estimates
"""

import time
import psutil
import gpustat
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import json
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from google.cloud import storage
import subprocess


class TrainingMonitor:
    def __init__(self, log_dir="logs", bucket_name="refocused-ai"):
        self.console = Console()
        self.log_dir = Path(log_dir)
        self.bucket_name = bucket_name
        self.start_time = datetime.now()
        
        # Initialize GCS client
        try:
            self.gcs_client = storage.Client()
            self.bucket = self.gcs_client.bucket(bucket_name)
        except:
            self.bucket = None
            self.console.print("[yellow]Warning: Could not connect to GCS[/yellow]")
    
    def get_gpu_stats(self):
        """Get current GPU statistics"""
        try:
            gpu_stats = gpustat.GPUStatCollection.new_query()
            gpu_data = []
            
            for gpu in gpu_stats:
                gpu_data.append({
                    'index': gpu.index,
                    'name': gpu.name,
                    'temperature': f"{gpu.temperature}°C",
                    'utilization': f"{gpu.utilization}%",
                    'memory_used': f"{gpu.memory_used}MB",
                    'memory_total': f"{gpu.memory_total}MB",
                    'memory_percent': f"{(gpu.memory_used/gpu.memory_total*100):.1f}%"
                })
            
            return gpu_data
        except:
            return []
    
    def get_system_stats(self):
        """Get system resource statistics"""
        return {
            'cpu_percent': f"{psutil.cpu_percent(interval=1)}%",
            'memory_percent': f"{psutil.virtual_memory().percent}%",
            'memory_used': f"{psutil.virtual_memory().used / 1e9:.1f}GB",
            'memory_total': f"{psutil.virtual_memory().total / 1e9:.1f}GB",
            'disk_usage': f"{psutil.disk_usage('/').percent}%"
        }
    
    def get_latest_metrics(self):
        """Parse latest metrics from log files"""
        latest_log = None
        latest_time = 0
        
        # Find most recent log file
        for log_file in self.log_dir.glob("*.log"):
            if log_file.stat().st_mtime > latest_time:
                latest_time = log_file.stat().st_mtime
                latest_log = log_file
        
        if not latest_log:
            return None
        
        # Parse last few lines for metrics
        metrics = {
            'loss': 'N/A',
            'learning_rate': 'N/A',
            'grad_norm': 'N/A',
            'samples_per_second': 'N/A',
            'files_processed': 0,
            'step': 0
        }
        
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
                
            for line in reversed(lines):
                if 'Step' in line and 'loss' in line:
                    # Parse metrics from log line
                    parts = line.split('|')
                    for part in parts:
                        if 'Step' in part:
                            metrics['step'] = int(part.split()[-1].rstrip(':'))
                        elif 'loss' in part:
                            metrics['loss'] = float(part.split(':')[-1].strip())
                        elif 'learning_rate' in part:
                            metrics['learning_rate'] = float(part.split(':')[-1].strip())
                        elif 'grad_norm' in part:
                            metrics['grad_norm'] = float(part.split(':')[-1].strip())
                        elif 'samples_per_second' in part:
                            metrics['samples_per_second'] = float(part.split(':')[-1].strip())
                        elif 'files_processed' in part:
                            metrics['files_processed'] = int(part.split(':')[-1].strip())
                    break
        except:
            pass
        
        return metrics
    
    def get_checkpoint_info(self):
        """Get information about latest checkpoint"""
        if not self.bucket:
            return None
        
        try:
            blobs = list(self.bucket.list_blobs(prefix="Checkpoints/"))
            checkpoints = [b.name for b in blobs if 'checkpoint_step_' in b.name]
            
            if not checkpoints:
                return None
            
            # Get latest checkpoint
            latest_checkpoint = sorted(checkpoints)[-1]
            
            # Parse checkpoint info
            parts = latest_checkpoint.split('_')
            step = int(parts[parts.index('step') + 1])
            files = int(parts[parts.index('files') + 1].rstrip('/'))
            
            return {
                'name': latest_checkpoint,
                'step': step,
                'files': files
            }
        except:
            return None
    
    def estimate_completion(self, current_step, total_steps, samples_per_second):
        """Estimate time to completion"""
        if current_step == 0 or samples_per_second == 0:
            return "N/A"
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        steps_per_second = current_step / elapsed
        remaining_steps = total_steps - current_step
        
        if steps_per_second > 0:
            eta_seconds = remaining_steps / steps_per_second
            eta = timedelta(seconds=int(eta_seconds))
            return str(eta)
        
        return "N/A"
    
    def create_display(self):
        """Create the monitoring display"""
        layout = Layout()
        
        # Get current data
        gpu_stats = self.get_gpu_stats()
        system_stats = self.get_system_stats()
        metrics = self.get_latest_metrics() or {}
        checkpoint = self.get_checkpoint_info()
        
        # Create GPU table
        gpu_table = Table(title="GPU Status", box=None)
        gpu_table.add_column("GPU", style="cyan")
        gpu_table.add_column("Temp", style="yellow")
        gpu_table.add_column("Util", style="green")
        gpu_table.add_column("Memory", style="magenta")
        
        for gpu in gpu_stats:
            gpu_table.add_row(
                f"{gpu['index']}: {gpu['name']}",
                gpu['temperature'],
                gpu['utilization'],
                f"{gpu['memory_used']}/{gpu['memory_total']} ({gpu['memory_percent']})"
            )
        
        # Create metrics table
        metrics_table = Table(title="Training Metrics", box=None)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="yellow")
        
        metrics_table.add_row("Step", str(metrics.get('step', 0)))
        metrics_table.add_row("Loss", f"{metrics.get('loss', 'N/A'):.4f}" if isinstance(metrics.get('loss'), float) else "N/A")
        metrics_table.add_row("Learning Rate", f"{metrics.get('learning_rate', 'N/A'):.2e}" if isinstance(metrics.get('learning_rate'), float) else "N/A")
        metrics_table.add_row("Grad Norm", f"{metrics.get('grad_norm', 'N/A'):.2f}" if isinstance(metrics.get('grad_norm'), float) else "N/A")
        metrics_table.add_row("Samples/sec", f"{metrics.get('samples_per_second', 'N/A'):.1f}" if isinstance(metrics.get('samples_per_second'), float) else "N/A")
        metrics_table.add_row("Files Processed", str(metrics.get('files_processed', 0)))
        
        # Create system table
        system_table = Table(title="System Resources", box=None)
        system_table.add_column("Resource", style="cyan")
        system_table.add_column("Usage", style="yellow")
        
        system_table.add_row("CPU", system_stats['cpu_percent'])
        system_table.add_row("Memory", f"{system_stats['memory_used']}/{system_stats['memory_total']} ({system_stats['memory_percent']})")
        system_table.add_row("Disk", system_stats['disk_usage'])
        
        # Create checkpoint info
        if checkpoint:
            checkpoint_text = f"Latest: {checkpoint['name']}\nStep: {checkpoint['step']}, Files: {checkpoint['files']}"
        else:
            checkpoint_text = "No checkpoints found"
        
        # Estimate completion
        eta = self.estimate_completion(
            metrics.get('step', 0),
            100000,  # Default max steps
            metrics.get('samples_per_second', 0)
        )
        
        # Create layout
        layout.split_column(
            Layout(Panel(gpu_table, title="GPU Status", border_style="blue")),
            Layout(Panel(metrics_table, title="Training Progress", border_style="green")),
            Layout(Panel(system_table, title="System Resources", border_style="yellow")),
            Layout(Panel(
                f"Checkpoint: {checkpoint_text}\n\nEstimated Time Remaining: {eta}",
                title="Status",
                border_style="cyan"
            ))
        )
        
        return layout
    
    def run(self, refresh_interval=5):
        """Run the monitor with live updates"""
        with Live(self.create_display(), refresh_per_second=1/refresh_interval) as live:
            while True:
                time.sleep(refresh_interval)
                live.update(self.create_display())


def main():
    parser = argparse.ArgumentParser(description="Monitor ReFocused-AI training")
    parser.add_argument("--log-dir", default="logs", help="Log directory")
    parser.add_argument("--bucket", default="refocused-ai", help="GCS bucket name")
    parser.add_argument("--refresh", type=int, default=5, help="Refresh interval in seconds")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.log_dir, args.bucket)
    
    try:
        monitor.run(args.refresh)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main() 