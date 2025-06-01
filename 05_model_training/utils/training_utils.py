"""
Training utilities for logging and metrics tracking
"""

import torch
import time
from typing import Dict, Any, Optional
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os


class MetricsTracker:
    """Track training metrics and handle logging"""
    
    def __init__(self, log_dir: str, run_name: str):
        self.log_dir = log_dir
        self.run_name = run_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(os.path.join(log_dir, run_name))
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.step_times = []
        self.last_time = time.time()
        
        # Running averages
        self.loss_window = []
        self.window_size = 100
    
    def update(self, metrics: Dict[str, float], step: int):
        """Update metrics and log to tensorboard"""
        current_time = time.time()
        step_time = current_time - self.last_time
        self.step_times.append(step_time)
        self.last_time = current_time
        
        # Update metrics
        for key, value in metrics.items():
            self.metrics[key].append(value)
            self.writer.add_scalar(f'train/{key}', value, step)
        
        # Track loss window for smoothed loss
        if 'loss' in metrics:
            self.loss_window.append(metrics['loss'])
            if len(self.loss_window) > self.window_size:
                self.loss_window.pop(0)
            
            smoothed_loss = np.mean(self.loss_window)
            self.writer.add_scalar('train/loss_smoothed', smoothed_loss, step)
        
        # Calculate and log training speed
        if len(self.step_times) > 10:
            avg_step_time = np.mean(self.step_times[-10:])
            samples_per_second = metrics.get('batch_size', 1) / avg_step_time
            self.writer.add_scalar('speed/samples_per_second', samples_per_second, step)
            self.writer.add_scalar('speed/step_time', avg_step_time, step)
    
    def log_summary(self, step: int, epoch: int, files_processed: int):
        """Log training summary"""
        if len(self.metrics['loss']) > 0:
            recent_loss = np.mean(self.metrics['loss'][-100:])
            print(f"\n[Step {step}] Epoch: {epoch}, Files: {files_processed}")
            print(f"  Loss: {recent_loss:.4f}")
            
            if len(self.step_times) > 10:
                avg_time = np.mean(self.step_times[-10:])
                print(f"  Speed: {1/avg_time:.2f} steps/sec")
    
    def close(self):
        """Close tensorboard writer"""
        self.writer.close()


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