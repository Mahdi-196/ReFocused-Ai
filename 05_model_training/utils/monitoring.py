"""
System and GPU Monitoring Utilities
Real-time monitoring for H100 training optimization
"""

import os
import time
import json
import logging
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import psutil

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    logging.warning("pynvml not available - GPU monitoring disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """GPU metrics data structure"""
    gpu_id: int
    name: str
    temperature: float
    power_usage: float
    power_limit: float
    memory_used: float
    memory_total: float
    memory_utilization: float
    gpu_utilization: float
    clock_speed: float
    max_clock_speed: float


@dataclass
class SystemMetrics:
    """System metrics data structure"""
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    network_sent_mb: float
    network_recv_mb: float


class GPUMonitor:
    """Real-time GPU monitoring for training optimization"""
    
    def __init__(self, log_interval: int = 60, log_file: Optional[str] = None):
        self.log_interval = log_interval
        self.log_file = log_file
        self.running = False
        self.thread = None
        
        if not NVML_AVAILABLE:
            logger.error("NVML not available - GPU monitoring disabled")
            return
        
        # Initialize NVML
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Initialized GPU monitoring for {self.device_count} devices")
        except Exception as e:
            logger.error(f"Failed to initialize NVML: {e}")
            self.device_count = 0
    
    def get_gpu_metrics(self) -> List[GPUMetrics]:
        """Get current GPU metrics for all devices"""
        if not NVML_AVAILABLE or self.device_count == 0:
            return []
        
        metrics = []
        
        for i in range(self.device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Basic info
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Temperature
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                
                # Power
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)[1] / 1000.0
                
                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = mem_info.used / 1024**3  # Convert to GB
                memory_total = mem_info.total / 1024**3
                memory_utilization = (mem_info.used / mem_info.total) * 100
                
                # Utilization
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util_rates.gpu
                
                # Clock speeds
                clock_speed = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                max_clock_speed = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
                
                metrics.append(GPUMetrics(
                    gpu_id=i,
                    name=name,
                    temperature=temp,
                    power_usage=power_usage,
                    power_limit=power_limit,
                    memory_used=memory_used,
                    memory_total=memory_total,
                    memory_utilization=memory_utilization,
                    gpu_utilization=gpu_utilization,
                    clock_speed=clock_speed,
                    max_clock_speed=max_clock_speed
                ))
                
            except Exception as e:
                logger.warning(f"Failed to get metrics for GPU {i}: {e}")
                continue
        
        return metrics
    
    def check_gpu_health(self, metrics: List[GPUMetrics]) -> Dict[str, Any]:
        """Check GPU health and detect potential issues"""
        issues = []
        warnings = []
        
        for gpu in metrics:
            # Temperature checks
            if gpu.temperature > 85:
                issues.append(f"GPU {gpu.gpu_id}: High temperature ({gpu.temperature}°C)")
            elif gpu.temperature > 80:
                warnings.append(f"GPU {gpu.gpu_id}: Elevated temperature ({gpu.temperature}°C)")
            
            # Power checks
            power_percent = (gpu.power_usage / gpu.power_limit) * 100
            if power_percent > 95:
                warnings.append(f"GPU {gpu.gpu_id}: Near power limit ({power_percent:.1f}%)")
            
            # Memory checks
            if gpu.memory_utilization > 95:
                warnings.append(f"GPU {gpu.gpu_id}: High memory usage ({gpu.memory_utilization:.1f}%)")
            
            # Utilization checks
            if gpu.gpu_utilization < 70:
                warnings.append(f"GPU {gpu.gpu_id}: Low utilization ({gpu.gpu_utilization}%)")
        
        return {
            'issues': issues,
            'warnings': warnings,
            'healthy': len(issues) == 0
        }
    
    def log_gpu_metrics(self):
        """Log GPU metrics to file and console"""
        metrics = self.get_gpu_metrics()
        
        if not metrics:
            return
        
        # Create summary
        total_memory_used = sum(gpu.memory_used for gpu in metrics)
        total_memory = sum(gpu.memory_total for gpu in metrics)
        avg_temperature = sum(gpu.temperature for gpu in metrics) / len(metrics)
        avg_utilization = sum(gpu.gpu_utilization for gpu in metrics) / len(metrics)
        total_power = sum(gpu.power_usage for gpu in metrics)
        
        # Check health
        health_status = self.check_gpu_health(metrics)
        
        # Log summary
        logger.info(f"GPU Summary: {len(metrics)} GPUs, "
                   f"Memory: {total_memory_used:.1f}/{total_memory:.1f} GB, "
                   f"Avg Temp: {avg_temperature:.1f}°C, "
                   f"Avg Util: {avg_utilization:.1f}%, "
                   f"Total Power: {total_power:.1f}W")
        
        # Log warnings and issues
        for warning in health_status['warnings']:
            logger.warning(warning)
        
        for issue in health_status['issues']:
            logger.error(issue)
        
        # Save detailed metrics to file
        if self.log_file:
            metrics_data = {
                'timestamp': time.time(),
                'gpus': [gpu.__dict__ for gpu in metrics],
                'summary': {
                    'total_memory_used_gb': total_memory_used,
                    'total_memory_gb': total_memory,
                    'avg_temperature': avg_temperature,
                    'avg_utilization': avg_utilization,
                    'total_power_watts': total_power
                },
                'health': health_status
            }
            
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(metrics_data) + '\n')
            except Exception as e:
                logger.error(f"Failed to write GPU metrics to file: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.log_gpu_metrics()
                time.sleep(self.log_interval)
            except Exception as e:
                logger.error(f"Error in GPU monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def start(self):
        """Start GPU monitoring"""
        if not NVML_AVAILABLE:
            logger.warning("GPU monitoring not available")
            return
        
        if self.running:
            logger.warning("GPU monitoring already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("GPU monitoring started")
    
    def stop(self):
        """Stop GPU monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("GPU monitoring stopped")


class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self, log_interval: int = 60, log_file: Optional[str] = None):
        self.log_interval = log_interval
        self.log_file = log_file
        self.running = False
        self.thread = None
        
        # Initial network stats for delta calculation
        self.last_net_io = psutil.net_io_counters()
        self.last_time = time.time()
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_used_gb = memory.used / 1024**3
        memory_total_gb = memory.total / 1024**3
        memory_percent = memory.percent
        
        # Disk (check /scratch if exists, otherwise root)
        disk_path = "/scratch" if os.path.exists("/scratch") else "/"
        disk = psutil.disk_usage(disk_path)
        disk_used_gb = disk.used / 1024**3
        disk_total_gb = disk.total / 1024**3
        disk_percent = (disk.used / disk.total) * 100
        
        # Network
        current_net_io = psutil.net_io_counters()
        current_time = time.time()
        time_delta = current_time - self.last_time
        
        if time_delta > 0:
            network_sent_mb = (current_net_io.bytes_sent - self.last_net_io.bytes_sent) / 1024**2 / time_delta
            network_recv_mb = (current_net_io.bytes_recv - self.last_net_io.bytes_recv) / 1024**2 / time_delta
        else:
            network_sent_mb = 0
            network_recv_mb = 0
        
        # Update for next calculation
        self.last_net_io = current_net_io
        self.last_time = current_time
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            memory_percent=memory_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            disk_percent=disk_percent,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )
    
    def log_system_metrics(self):
        """Log system metrics"""
        metrics = self.get_system_metrics()
        
        logger.info(f"System: CPU {metrics.cpu_percent:.1f}%, "
                   f"RAM {metrics.memory_used_gb:.1f}/{metrics.memory_total_gb:.1f} GB ({metrics.memory_percent:.1f}%), "
                   f"Disk {metrics.disk_used_gb:.1f}/{metrics.disk_total_gb:.1f} GB ({metrics.disk_percent:.1f}%), "
                   f"Net {metrics.network_sent_mb:.1f}↑/{metrics.network_recv_mb:.1f}↓ MB/s")
        
        # Check for issues
        if metrics.memory_percent > 90:
            logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.disk_percent > 90:
            logger.warning(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        if metrics.cpu_percent > 90:
            logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Save to file
        if self.log_file:
            metrics_data = {
                'timestamp': time.time(),
                'system': metrics.__dict__
            }
            
            try:
                with open(self.log_file, 'a') as f:
                    f.write(json.dumps(metrics_data) + '\n')
            except Exception as e:
                logger.error(f"Failed to write system metrics to file: {e}")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self.log_system_metrics()
                time.sleep(self.log_interval)
            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(5)
    
    def start(self):
        """Start system monitoring"""
        if self.running:
            logger.warning("System monitoring already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop system monitoring"""
        if not self.running:
            return
        
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("System monitoring stopped")


class TrainingLogger:
    """Training metrics logger"""
    
    def __init__(self, log_dir: str, log_interval: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        
        # Create log files
        self.metrics_file = self.log_dir / "training_metrics.jsonl"
        self.performance_file = self.log_dir / "performance_metrics.jsonl"
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log training metrics"""
        metrics_data = {
            'step': step,
            'timestamp': time.time(),
            **metrics
        }
        
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log training metrics: {e}")
    
    def log_performance(self, performance_data: Dict[str, Any], step: int):
        """Log performance metrics"""
        perf_data = {
            'step': step,
            'timestamp': time.time(),
            **performance_data
        }
        
        try:
            with open(self.performance_file, 'a') as f:
                f.write(json.dumps(perf_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log performance metrics: {e}")


def get_torch_memory_info() -> Dict[str, float]:
    """Get PyTorch memory information"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {}
    
    return {
        'memory_allocated_gb': torch.cuda.memory_allocated() / 1024**3,
        'memory_reserved_gb': torch.cuda.memory_reserved() / 1024**3,
        'max_memory_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
        'max_memory_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3,
    }


if __name__ == "__main__":
    # Test monitoring
    logging.basicConfig(level=logging.INFO)
    
    gpu_monitor = GPUMonitor(log_interval=5)
    system_monitor = SystemMonitor(log_interval=5)
    
    gpu_monitor.start()
    system_monitor.start()
    
    try:
        time.sleep(30)  # Monitor for 30 seconds
    except KeyboardInterrupt:
        pass
    finally:
        gpu_monitor.stop()
        system_monitor.stop() 