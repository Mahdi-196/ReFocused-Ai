"""
Training utilities for logging and metrics tracking with enhanced monitoring
"""

import torch
import time
from typing import Dict, Any, Optional
from collections import defaultdict, deque
import numpy as np
import os
import threading
import json

# Optional dependencies - gracefully handle if not available
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not available, metrics logging will be limited")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available, system monitoring will be disabled")


class PerformanceProfiler:
    """Profile training performance and I/O operations"""
    
    def __init__(self, enable_profiling: bool = False):
        self.enable_profiling = enable_profiling
        self.timings = defaultdict(list)
        self.data_loading_times = deque(maxlen=100)
        self.model_forward_times = deque(maxlen=100)
        self.model_backward_times = deque(maxlen=100)
        self.io_operations = []
        self.memory_usage = []
        
    def start_timer(self, operation: str):
        """Start timing an operation"""
        if not self.enable_profiling:
            return None
        return time.time()
    
    def end_timer(self, operation: str, start_time: Optional[float]):
        """End timing an operation and record duration"""
        if not self.enable_profiling or start_time is None:
            return
        
        duration = time.time() - start_time
        self.timings[operation].append(duration)
        
        # Special handling for specific operations
        if operation == 'data_loading':
            self.data_loading_times.append(duration)
        elif operation == 'model_forward':
            self.model_forward_times.append(duration)
        elif operation == 'model_backward':
            self.model_backward_times.append(duration)
    
    def record_io_operation(self, operation_type: str, file_path: str, size_mb: float, duration: float):
        """Record I/O operation details"""
        if not self.enable_profiling:
            return
        
        self.io_operations.append({
            'timestamp': time.time(),
            'operation': operation_type,
            'file': os.path.basename(file_path),
            'size_mb': size_mb,
            'duration': duration,
            'throughput_mbps': size_mb / duration if duration > 0 else 0
        })
    
    def record_memory_usage(self):
        """Record current memory usage"""
        if not self.enable_profiling:
            return
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3  # GB
        else:
            gpu_memory = gpu_memory_max = 0
        
        # CPU memory (only if psutil available)
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                cpu_memory = process.memory_info().rss / 1024**3  # GB
            except Exception:
                cpu_memory = 0
        else:
            cpu_memory = 0
        
        self.memory_usage.append({
            'timestamp': time.time(),
            'gpu_memory_gb': gpu_memory,
            'gpu_memory_max_gb': gpu_memory_max,
            'cpu_memory_gb': cpu_memory,
        })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        if not self.enable_profiling:
            return {}
        
        summary = {}
        
        # Timing statistics
        for operation, times in self.timings.items():
            if times:
                summary[f'{operation}_avg_time'] = np.mean(times)
                summary[f'{operation}_max_time'] = np.max(times)
                summary[f'{operation}_min_time'] = np.min(times)
        
        # I/O statistics
        if self.io_operations:
            total_data_mb = sum(op['size_mb'] for op in self.io_operations)
            total_io_time = sum(op['duration'] for op in self.io_operations)
            avg_throughput = np.mean([op['throughput_mbps'] for op in self.io_operations])
            
            summary['total_data_loaded_mb'] = total_data_mb
            summary['total_io_time'] = total_io_time
            summary['avg_io_throughput_mbps'] = avg_throughput
        
        # Memory statistics
        if self.memory_usage:
            latest_memory = self.memory_usage[-1]
            max_gpu_memory = max(m['gpu_memory_gb'] for m in self.memory_usage)
            avg_cpu_memory = np.mean([m['cpu_memory_gb'] for m in self.memory_usage])
            
            summary['current_gpu_memory_gb'] = latest_memory['gpu_memory_gb']
            summary['max_gpu_memory_gb'] = max_gpu_memory
            summary['avg_cpu_memory_gb'] = avg_cpu_memory
        
        return summary


class EnhancedMetricsTracker:
    """Enhanced training metrics tracker with detailed monitoring"""
    
    def __init__(self, log_dir: str, run_name: str, detailed_monitoring: bool = False, enable_profiling: bool = False):
        self.log_dir = log_dir
        self.run_name = run_name
        self.detailed_monitoring = detailed_monitoring
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tensorboard writer (if available)
        if HAS_TENSORBOARD:
            try:
                self.writer = SummaryWriter(os.path.join(log_dir, run_name))
            except Exception as e:
                print(f"Warning: Could not initialize tensorboard writer: {e}")
                self.writer = None
        else:
            self.writer = None
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.step_times = deque(maxlen=1000)  # Keep more history
        self.last_time = time.time()
        
        # Enhanced tracking
        self.loss_window = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        self.gradient_norm_history = deque(maxlen=100)
        
        # Performance profiler
        self.profiler = PerformanceProfiler(enable_profiling)
        
        # Monitoring thread for system metrics (only if psutil available)
        self.monitoring_active = detailed_monitoring and HAS_PSUTIL
        if self.monitoring_active:
            self.system_metrics = defaultdict(list)
            self.start_system_monitoring()
    
    def start_system_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while self.monitoring_active:
                try:
                    # Only proceed if psutil is available (should always be true when this method is called)
                    if not HAS_PSUTIL:
                        break
                        
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    
                    # Memory usage
                    memory = psutil.virtual_memory()
                    
                    # Disk I/O
                    disk_io = psutil.disk_io_counters()
                    
                    # Network I/O (if relevant)
                    net_io = psutil.net_io_counters()
                    
                    timestamp = time.time()
                    self.system_metrics['cpu_percent'].append((timestamp, cpu_percent))
                    self.system_metrics['memory_percent'].append((timestamp, memory.percent))
                    self.system_metrics['disk_read_mb'].append((timestamp, disk_io.read_bytes / 1024**2))
                    self.system_metrics['disk_write_mb'].append((timestamp, disk_io.write_bytes / 1024**2))
                    
                    # Log to tensorboard if available and detailed monitoring is enabled
                    if self.writer and len(self.system_metrics['cpu_percent']) % 10 == 0:  # Every 10 seconds
                        latest_step = len(self.metrics.get('loss', []))
                        if latest_step > 0:
                            self.writer.add_scalar('system/cpu_percent', cpu_percent, latest_step)
                            self.writer.add_scalar('system/memory_percent', memory.percent, latest_step)
                    
                except Exception as e:
                    print(f"Warning: System monitoring error: {e}")
                
                time.sleep(1)  # Monitor every second
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics and log to tensorboard"""
        current_time = time.time()
        step_time = current_time - self.last_time
        self.step_times.append(step_time)
        self.last_time = current_time
        
        # Update metrics
        for key, value in metrics.items():
            self.metrics[key].append(value)
            if self.writer:
                self.writer.add_scalar(f'train/{key}', value, step)
        
        # Track loss window for smoothed loss
        if 'loss' in metrics:
            self.loss_window.append(metrics['loss'])
            smoothed_loss = np.mean(self.loss_window)
            if self.writer:
                self.writer.add_scalar('train/loss_smoothed', smoothed_loss, step)
        
        # Track gradient norm
        if 'grad_norm' in metrics:
            self.gradient_norm_history.append(metrics['grad_norm'])
            avg_grad_norm = np.mean(self.gradient_norm_history)
            if self.writer:
                self.writer.add_scalar('train/grad_norm_avg', avg_grad_norm, step)
        
        # Calculate and log training speed
        if len(self.step_times) > 10:
            avg_step_time = np.mean(list(self.step_times)[-10:])
            samples_per_second = metrics.get('batch_size', 1) / avg_step_time
            self.throughput_history.append(samples_per_second)
            
            if self.writer:
                self.writer.add_scalar('speed/samples_per_second', samples_per_second, step)
                self.writer.add_scalar('speed/step_time', avg_step_time, step)
                self.writer.add_scalar('speed/avg_throughput', np.mean(self.throughput_history), step)
        
        # Record memory usage if profiling enabled
        self.profiler.record_memory_usage()
        
        # Enhanced logging for detailed monitoring
        if self.detailed_monitoring and step % 50 == 0:
            self.log_detailed_metrics(step)
    
    def log_detailed_metrics(self, step: int):
        """Log detailed performance metrics"""
        perf_summary = self.profiler.get_performance_summary()
        
        if self.writer:
            for metric_name, value in perf_summary.items():
                self.writer.add_scalar(f'performance/{metric_name}', value, step)
        
        # Log training stability metrics
        if len(self.loss_window) > 10:
            loss_variance = np.var(list(self.loss_window))
            loss_trend = np.polyfit(range(len(self.loss_window)), list(self.loss_window), 1)[0]
            if self.writer:
                self.writer.add_scalar('stability/loss_variance', loss_variance, step)
                self.writer.add_scalar('stability/loss_trend', loss_trend, step)
        
        if len(self.gradient_norm_history) > 10:
            grad_norm_variance = np.var(list(self.gradient_norm_history))
            if self.writer:
                self.writer.add_scalar('stability/grad_norm_variance', grad_norm_variance, step)
    
    def log_summary(self, step: int, epoch: int, files_processed: int):
        """Log training summary with enhanced details"""
        if len(self.metrics['loss']) > 0:
            recent_loss = np.mean(self.metrics['loss'][-100:])
            print(f"\n[Step {step}] Epoch: {epoch}, Files: {files_processed}")
            print(f"  Loss: {recent_loss:.4f}")
            
            if len(self.step_times) > 10:
                avg_time = np.mean(list(self.step_times)[-10:])
                print(f"  Speed: {1/avg_time:.2f} steps/sec")
            
            # Enhanced summary for detailed monitoring
            if self.detailed_monitoring:
                if len(self.throughput_history) > 0:
                    avg_throughput = np.mean(self.throughput_history)
                    print(f"  Avg Throughput: {avg_throughput:.2f} samples/sec")
                
                if len(self.gradient_norm_history) > 0:
                    avg_grad_norm = np.mean(list(self.gradient_norm_history)[-10:])
                    print(f"  Avg Grad Norm: {avg_grad_norm:.3f}")
                
                perf_summary = self.profiler.get_performance_summary()
                if perf_summary:
                    print(f"  Performance Summary:")
                    for key, value in perf_summary.items():
                        if isinstance(value, float):
                            print(f"    {key}: {value:.3f}")
    
    def save_metrics_summary(self, output_path: str):
        """Save detailed metrics summary to file"""
        summary = {
            'run_name': self.run_name,
            'total_steps': len(self.metrics.get('loss', [])),
            'final_loss': self.metrics['loss'][-1] if self.metrics.get('loss') else None,
            'avg_step_time': np.mean(self.step_times) if self.step_times else None,
            'performance_summary': self.profiler.get_performance_summary(),
            'system_metrics_summary': {}
        }
        
        # System metrics summary
        if self.detailed_monitoring and self.system_metrics:
            for metric_name, values in self.system_metrics.items():
                if values:
                    recent_values = [v[1] for v in values[-100:]]  # Last 100 readings
                    summary['system_metrics_summary'][metric_name] = {
                        'avg': np.mean(recent_values),
                        'max': np.max(recent_values),
                        'min': np.min(recent_values)
                    }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Metrics summary saved to {output_path}")
    
    def close(self):
        """Close tensorboard writer and stop monitoring"""
        self.monitoring_active = False
        if self.writer:
            self.writer.close()
        
        # Save final summary
        summary_path = os.path.join(self.log_dir, f"{self.run_name}_summary.json")
        self.save_metrics_summary(summary_path)


# Backward compatibility alias
MetricsTracker = EnhancedMetricsTracker


def get_grad_norm(model, norm_type: float = 2.0) -> float:
    """Calculate gradient norm for monitoring training stability"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss"""
    try:
        return np.exp(loss)
    except OverflowError:
        return float('inf')


def estimate_remaining_time(steps_done: int, total_steps: int, avg_step_time: float) -> str:
    """Estimate remaining training time"""
    if total_steps <= 0 or steps_done >= total_steps:
        return "N/A"
    
    remaining_steps = total_steps - steps_done
    remaining_seconds = remaining_steps * avg_step_time
    
    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    
    return f"{hours}h {minutes}m"


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_metrics_log(
    step: int,
    epoch: int,
    loss: float,
    learning_rate: float,
    grad_norm: Optional[float] = None,
    perplexity: Optional[float] = None,
    files_processed: int = 0,
    tokens_processed: int = 0
) -> str:
    """Format metrics for logging"""
    log_parts = [
        f"Step: {step}",
        f"Epoch: {epoch}",
        f"Files: {files_processed}",
        f"Loss: {loss:.4f}",
        f"LR: {learning_rate:.2e}",
    ]
    
    if grad_norm is not None:
        log_parts.append(f"Grad Norm: {grad_norm:.2f}")
    
    if perplexity is not None and perplexity != float('inf'):
        log_parts.append(f"PPL: {perplexity:.2f}")
    
    if tokens_processed > 0:
        billions = tokens_processed / 1e9
        log_parts.append(f"Tokens: {billions:.2f}B")
    
    return " | ".join(log_parts) 