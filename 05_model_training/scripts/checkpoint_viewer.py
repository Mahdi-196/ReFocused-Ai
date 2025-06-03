#!/usr/bin/env python3
"""
Checkpoint viewer utility for inspecting comprehensive checkpoint data
"""

import os
import sys
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs import get_training_config
from utils.checkpoint_utils import CheckpointManager


def load_checkpoint_data(checkpoint_dir: str):
    """Load all checkpoint data files"""
    data = {}
    
    # Load metadata
    metadata_path = os.path.join(checkpoint_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            data['metadata'] = json.load(f)
    
    # Load training config
    config_path = os.path.join(checkpoint_dir, 'training_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            data['training_config'] = json.load(f)
    
    # Load training metrics
    metrics_path = os.path.join(checkpoint_dir, 'training_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            data['training_metrics'] = json.load(f)
    
    return data


def print_checkpoint_summary(checkpoint_dir: str):
    """Print a comprehensive summary of checkpoint data"""
    data = load_checkpoint_data(checkpoint_dir)
    checkpoint_name = os.path.basename(checkpoint_dir)
    
    print(f"\n📊 Checkpoint Summary: {checkpoint_name}")
    print("=" * 60)
    
    # Basic info
    if 'metadata' in data:
        meta = data['metadata']
        print(f"📅 Timestamp: {meta.get('timestamp', 'N/A')}")
        print(f"🔢 Step: {meta.get('step', 'N/A')}")
        print(f"🔄 Epoch: {meta.get('epoch', 'N/A')}")
        print(f"📁 Files Processed: {meta.get('files_processed', 'N/A')}")
        
        # System info
        if 'system_info' in meta:
            sys_info = meta['system_info']
            print(f"🖥️  CUDA Available: {sys_info.get('cuda_available', 'N/A')}")
            print(f"🎮 GPU Count: {sys_info.get('device_count', 'N/A')}")
            print(f"⚡ Mixed Precision: {sys_info.get('mixed_precision', 'N/A')}")
    
    # Training metrics
    print("\n📈 Training Metrics:")
    print("-" * 30)
    
    if 'training_metrics' in data:
        metrics = data['training_metrics']
        print(f"💥 Current Loss: {metrics.get('current_loss', 'N/A'):.6f}")
        print(f"🏆 Best Loss: {metrics.get('best_loss', 'N/A'):.6f}")
        
        loss_history = metrics.get('loss_history', [])
        if loss_history:
            print(f"📊 Loss History Length: {len(loss_history)}")
            print(f"📉 Average Loss: {np.mean(loss_history):.6f}")
            print(f"📊 Loss Std Dev: {np.std(loss_history):.6f}")
            print(f"🔄 Loss Trend: {'↓' if len(loss_history) >= 2 and loss_history[-1] < loss_history[-2] else '→'}")
        
        lr_history = metrics.get('learning_rates', [])
        if lr_history:
            print(f"📈 Current LR: {lr_history[-1]:.2e}")
            print(f"📊 LR History Length: {len(lr_history)}")
    
    # Validation metrics
    if 'metadata' in data and 'validation_metrics' in data['metadata']:
        val_metrics = data['metadata']['validation_metrics']
        if val_metrics:
            print("\n🎯 Validation Metrics:")
            print("-" * 30)
            for key, value in val_metrics.items():
                print(f"  {key}: {value}")
    
    # Training config summary
    if 'training_config' in data:
        config = data['training_config']
        print("\n⚙️ Training Configuration:")
        print("-" * 30)
        print(f"🎯 Max Steps: {config.get('max_steps', 'N/A')}")
        print(f"📦 Batch Size: {config.get('per_device_train_batch_size', 'N/A')}")
        print(f"📚 Learning Rate: {config.get('learning_rate', 'N/A')}")
        print(f"💾 Save Steps: {config.get('save_steps', 'N/A')}")
        print(f"📁 Max Files: {config.get('max_files', 'N/A')}")


def plot_training_metrics(checkpoint_dir: str, save_path: str = None):
    """Plot training metrics from checkpoint data"""
    data = load_checkpoint_data(checkpoint_dir)
    
    if 'training_metrics' not in data:
        print("❌ No training metrics found in checkpoint")
        return
    
    metrics = data['training_metrics']
    loss_history = metrics.get('loss_history', [])
    lr_history = metrics.get('learning_rates', [])
    
    if not loss_history and not lr_history:
        print("❌ No loss or learning rate history found")
        return
    
    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    checkpoint_name = os.path.basename(checkpoint_dir)
    
    # Loss plot
    if loss_history:
        steps = list(range(len(loss_history)))
        axes[0].plot(steps, loss_history, 'b-', linewidth=2, label='Training Loss')
        axes[0].set_xlabel('Logging Steps')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Training Loss History - {checkpoint_name}')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Add best loss line
        best_loss = metrics.get('best_loss')
        if best_loss:
            axes[0].axhline(y=best_loss, color='r', linestyle='--', alpha=0.7, label=f'Best: {best_loss:.6f}')
            axes[0].legend()
    
    # Learning rate plot
    if lr_history:
        steps = list(range(len(lr_history)))
        axes[1].plot(steps, lr_history, 'g-', linewidth=2, label='Learning Rate')
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title(f'Learning Rate Schedule - {checkpoint_name}')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot saved to: {save_path}")
    else:
        plt.show()


def list_local_checkpoints(checkpoint_dir: str = "./checkpoints"):
    """List all local checkpoints with basic info"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = [
        d for d in os.listdir(checkpoint_dir) 
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('checkpoint-')
    ]
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    print(f"📁 Found {len(checkpoints)} checkpoints:")
    print("=" * 80)
    
    for i, checkpoint in enumerate(sorted(checkpoints), 1):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        data = load_checkpoint_data(checkpoint_path)
        
        # Extract basic info
        step = "N/A"
        loss = "N/A" 
        timestamp = "N/A"
        
        if 'metadata' in data:
            step = data['metadata'].get('step', 'N/A')
            timestamp = data['metadata'].get('timestamp', 'N/A')[:19]  # Remove milliseconds
        
        if 'training_metrics' in data:
            current_loss = data['training_metrics'].get('current_loss')
            if current_loss is not None:
                loss = f"{current_loss:.6f}"
        
        print(f"{i:2d}. {checkpoint:<40} Step: {step:<6} Loss: {loss:<10} Time: {timestamp}")


def compare_checkpoints(checkpoint_dirs: list):
    """Compare multiple checkpoints"""
    print("\n🔍 Checkpoint Comparison")
    print("=" * 80)
    
    data_list = []
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            data = load_checkpoint_data(checkpoint_dir)
            data['name'] = os.path.basename(checkpoint_dir)
            data_list.append(data)
        else:
            print(f"❌ Checkpoint not found: {checkpoint_dir}")
    
    if not data_list:
        print("No valid checkpoints to compare")
        return
    
    # Print comparison table
    print(f"{'Checkpoint':<40} {'Step':<8} {'Loss':<12} {'Best Loss':<12} {'Timestamp':<20}")
    print("-" * 100)
    
    for data in data_list:
        name = data['name']
        step = data.get('metadata', {}).get('step', 'N/A')
        current_loss = data.get('training_metrics', {}).get('current_loss', 'N/A')
        best_loss = data.get('training_metrics', {}).get('best_loss', 'N/A')
        timestamp = data.get('metadata', {}).get('timestamp', 'N/A')[:19]
        
        loss_str = f"{current_loss:.6f}" if isinstance(current_loss, (int, float)) else str(current_loss)
        best_str = f"{best_loss:.6f}" if isinstance(best_loss, (int, float)) else str(best_loss)
        
        print(f"{name:<40} {step:<8} {loss_str:<12} {best_str:<12} {timestamp:<20}")


def main():
    parser = argparse.ArgumentParser(description="View checkpoint information and metrics")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Local checkpoint directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all local checkpoints")
    
    # View command
    view_parser = subparsers.add_parser("view", help="View specific checkpoint details")
    view_parser.add_argument("checkpoint_name", help="Checkpoint directory name")
    
    # Plot command
    plot_parser = subparsers.add_parser("plot", help="Plot training metrics")
    plot_parser.add_argument("checkpoint_name", help="Checkpoint directory name")
    plot_parser.add_argument("--save", type=str, help="Save plot to file")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple checkpoints")
    compare_parser.add_argument("checkpoint_names", nargs="+", help="Checkpoint directory names")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_local_checkpoints(args.checkpoint_dir)
        
    elif args.command == "view":
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
        if os.path.exists(checkpoint_path):
            print_checkpoint_summary(checkpoint_path)
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            
    elif args.command == "plot":
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
        if os.path.exists(checkpoint_path):
            plot_training_metrics(checkpoint_path, args.save)
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            
    elif args.command == "compare":
        checkpoint_paths = [
            os.path.join(args.checkpoint_dir, name) 
            for name in args.checkpoint_names
        ]
        compare_checkpoints(checkpoint_paths)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 