#!/usr/bin/env python3
"""
Clean training script for ReFocused-AI 1.2B model
Enhanced with performance optimizations for H100 GPU utilization
"""

import os
import sys
import warnings

# Suppress FutureWarnings from transformers and torch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import signal

# Set high precision for float32 matmul (improves H100/A100 performance)
torch.set_float32_matmul_precision('high')
from torch.optim import AdamW
from transformers import GPTNeoXForCausalLM, get_scheduler, set_seed
from accelerate import Accelerator
import argparse
from tqdm import tqdm
import time
import numpy as np
import torch.backends.cudnn as cudnn

# Enable cuDNN benchmark and deterministic settings for performance
cudnn.benchmark = True
cudnn.deterministic = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_model_config, get_training_config
from utils import create_dataloader, CheckpointManager, count_parameters

# Global checkpoint manager for signal handler
checkpoint_manager = None

def signal_handler(signum, frame):
    """Handle interruption signals to complete uploads"""
    if checkpoint_manager:
        print(f"\n🛑 Training interrupted. Waiting for background uploads to complete...")
        checkpoint_manager.wait_for_uploads()
    sys.exit(0)


def check_gpu():
    """Quick GPU check with optimization recommendations"""
    print(f"🔥 GPU Status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {name}")
            if "H100" in name or "A100" in name:
                print(f"    ⚡ High-performance GPU detected - bf16 recommended")
            elif "V100" in name or "RTX" in name:
                print(f"    ⚡ Modern GPU detected - fp16/bf16 available")
    else:
        print("  ⚠️  Running on CPU - training will be slow")
    return torch.cuda.is_available()


def main():
    global checkpoint_manager
    
    parser = argparse.ArgumentParser(description="Train ReFocused-AI model")
    parser.add_argument("--config", type=str, choices=["test", "production"], 
                       default="test", help="Training configuration")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Override max steps")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--no-background-upload", action="store_true",
                       help="Disable background uploads (training will block on uploads)")
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], 
                       default=None, help="Override mixed precision setting")
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check GPU status
    cuda_available = check_gpu()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config = get_training_config(args.config)
    if args.max_steps:
        config.max_steps = args.max_steps
    
    # Determine mixed precision setting
    mixed_precision = "no"
    if cuda_available:
        if args.mixed_precision:
            mixed_precision = args.mixed_precision
        elif getattr(config, 'bf16', False):
            mixed_precision = "bf16"
        elif getattr(config, 'fp16', False):
            mixed_precision = "fp16"
    
    print(f"\n🚀 Starting {args.config.upper()} training with optimizations")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Batch size per device: {config.per_device_train_batch_size}")
    print(f"  Gradient accumulation steps: {getattr(config, 'gradient_accumulation_steps', 1)}")
    effective_batch = config.per_device_train_batch_size * getattr(config, 'gradient_accumulation_steps', 1)
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision: {mixed_precision}")
    print(f"  Background uploads: {not args.no_background_upload}")
    
    # Initialize accelerator with optimized settings
    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1),
        mixed_precision=mixed_precision,
        cpu=False,  # Ensure GPU usage
    )
    
    print(f"  Device: {accelerator.device}")
    print(f"  Process index: {accelerator.process_index}")
    print(f"  Local process index: {accelerator.local_process_index}")
    print(f"  Num processes: {accelerator.num_processes}")
    
    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # Initialize model
    model_config = get_model_config()
    model = GPTNeoXForCausalLM(model_config)
    
    if accelerator.is_main_process:
        param_count = count_parameters(model)
        print(f"  Model parameters: {param_count/1e9:.2f}B")
    
    # Create optimizer with optimized settings
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8,  # Stable epsilon for mixed precision
    )
    
    # Create optimized dataloader
    train_dataloader, num_files = create_dataloader(config, accelerator)
    print(f"  Training files: {num_files}")
    print(f"  Steps per epoch: {len(train_dataloader)}")
    
    # Create scheduler with better placement
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    
    # Prepare for training with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Apply torch.compile after accelerator.prepare() for device-aware optimization
    if getattr(config, 'compile_model', False) and hasattr(torch, 'compile'):
        print("🚀 Applying torch.compile after accelerator.prepare()...")
        try:
            model = torch.compile(model)
            print("✅ Model compilation successful")
        except Exception as e:
            print(f"⚠️  Model compilation failed: {e}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        config.bucket_name, 
        config.checkpoint_bucket_path, 
        config.output_dir,
        background_upload=not args.no_background_upload
    )
    
    # Training loop with optimizations
    print(f"\n📈 Starting optimized training loop...")
    start_time = time.time()
    model.train()
    completed_steps = 0
    total_loss = 0.0
    
    # Enhanced tracking for comprehensive checkpointing
    loss_history = []
    learning_rate_history = []
    best_loss = float('inf')
    validation_metrics = {}
    
    # Optimize progress bar
    progress_bar = tqdm(
        range(config.max_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )
    
    # Pre-allocate variables to reduce Python overhead
    logging_steps = config.logging_steps
    save_steps = getattr(config, 'save_steps', 500)
    max_grad_norm = config.max_grad_norm
    
    for epoch in range(100):  # Max 100 epochs
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(model):
                # Forward pass - batch is already on correct device via accelerator.prepare
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Optimizer step with gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress and metrics (reduced Python overhead)
            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                
                # Extract loss value efficiently
                current_loss = loss.detach().float().item()
                total_loss += current_loss
                
                # Step scheduler outside conditional for optimal placement
                lr_scheduler.step()
                
                # Update best loss
                if current_loss < best_loss:
                    best_loss = current_loss
                
                # Efficient logging (reduced frequency checks)
                if completed_steps % logging_steps == 0:
                    avg_loss = total_loss / logging_steps
                    loss_history.append(avg_loss)
                    total_loss = 0.0
                    
                    # Get learning rate efficiently
                    current_lr = lr_scheduler.get_last_lr()[0]
                    learning_rate_history.append(current_lr)
                    
                    if accelerator.is_main_process:
                        progress_bar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'best': f'{best_loss:.4f}'
                        })
                        
                        # Efficient validation metrics computation
                        validation_metrics = {
                            'current_avg_loss': avg_loss,
                            'steps_since_best': completed_steps - (loss_history.index(min(loss_history)) + 1) * logging_steps if loss_history else 0,
                            'loss_trend': 'improving' if len(loss_history) >= 2 and loss_history[-1] < loss_history[-2] else 'stable',
                        }
                
                # Optimized checkpoint saving (reduced frequency)
                if completed_steps % save_steps == 0:
                    if accelerator.is_main_process:
                        print(f"\n💾 Saving checkpoint at step {completed_steps}")
                        checkpoint_manager.save_checkpoint(
                            accelerator=accelerator,
                            model=model,
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            epoch=epoch,
                            step=completed_steps,
                            files_processed=completed_steps // len(train_dataloader),
                            training_config=config,
                            current_loss=current_loss,
                            best_loss=best_loss,
                            loss_history=loss_history.copy(),
                            learning_rates=learning_rate_history.copy(),
                            validation_metrics=validation_metrics.copy(),
                            metadata={
                                'model_config': model_config.__dict__,
                                'training_config': config.__dict__,
                                'optimization_settings': {
                                    'mixed_precision': mixed_precision,
                                    'effective_batch_size': effective_batch,
                                    'cudnn_benchmark': cudnn.benchmark,
                                }
                            }
                        )
                
                # Check if training is complete
                if completed_steps >= config.max_steps:
                    break
        
        if completed_steps >= config.max_steps:
            break
    
    # Final checkpoint with comprehensive optimization summary
    if accelerator.is_main_process:
        print("\n✅ Training complete! Saving final checkpoint...")
        
        training_time = time.time() - start_time
        steps_per_second = completed_steps / training_time if training_time > 0 else 0
        
        # Final validation metrics summary with performance stats
        final_validation_metrics = {
            **validation_metrics,
            'training_completed': True,
            'total_steps': completed_steps,
            'final_loss': current_loss,
            'best_loss_achieved': best_loss,
            'total_epochs': epoch,
            'average_loss': sum(loss_history) / len(loss_history) if loss_history else current_loss,
            'loss_std': np.std(loss_history) if len(loss_history) > 1 else 0.0,
            'training_time_seconds': training_time,
            'steps_per_second': steps_per_second,
            'effective_batch_size': effective_batch,
        }
        
        # IMPORTANT: Use unique step number for final checkpoint to prevent race conditions
        # If training ends at step 1000, the last regular save creates: checkpoint-epoch0-step1000-files0
        # We must use a different step number to avoid "File shrank" errors from concurrent directory access
        final_step = completed_steps + 1  # Creates: checkpoint-epoch0-step1001-files0 (unique directory)
        
        checkpoint_manager.save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            epoch=epoch,
            step=final_step,  # Use incremented step to create unique directory name
            files_processed=completed_steps // len(train_dataloader),
            training_config=config,
            current_loss=current_loss,
            best_loss=best_loss,
            loss_history=loss_history,
            learning_rates=learning_rate_history,
            validation_metrics=final_validation_metrics,
            metadata={
                'model_config': model_config.__dict__,
                'training_config': config.__dict__,
                'training_summary': {
                    'completed': True,
                    'total_time': training_time,
                    'actual_steps_completed': completed_steps,  # Track actual steps completed
                    'performance': {
                        'steps_per_second': steps_per_second,
                        'effective_batch_size': effective_batch,
                        'mixed_precision': mixed_precision,
                        'gradient_accumulation': getattr(config, 'gradient_accumulation_steps', 1),
                    }
                }
            }
        )
        
        # Performance summary
        print(f"\n🎯 Training Performance Summary:")
        print(f"   Total time: {training_time/3600:.2f} hours")
        print(f"   Steps per second: {steps_per_second:.2f}")
        print(f"   Effective batch size: {effective_batch}")
        print(f"   Mixed precision: {mixed_precision}")
        print(f"   Best loss achieved: {best_loss:.4f}")
        
        # Wait for all background uploads to complete
        print("\n⏳ Waiting for background uploads to complete...")
        checkpoint_manager.wait_for_uploads()
        print("🎉 All done! Training and uploads completed successfully.")


if __name__ == "__main__":
    main() 