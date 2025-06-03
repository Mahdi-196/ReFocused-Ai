#!/usr/bin/env python3
"""
Clean training script for ReFocused-AI 1.2B model
"""

import os
import sys
import torch
import signal
from torch.optim import AdamW
from transformers import GPTNeoXForCausalLM, get_scheduler, set_seed
from accelerate import Accelerator
import argparse
from tqdm import tqdm
import time
import numpy as np

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
    """Quick GPU check"""
    print(f"🔥 GPU Status:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
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
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check GPU status
    cuda_available = check_gpu()
    
    # Set seed
    set_seed(42)
    
    # Load configuration
    config = get_training_config(args.config)
    if args.max_steps:
        config.max_steps = args.max_steps
    
    print(f"\n🚀 Starting {args.config.upper()} training")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Background uploads: {not args.no_background_upload}")
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 and cuda_available else "no",
    )
    
    print(f"  Device: {accelerator.device}")
    print(f"  Mixed precision: {accelerator.mixed_precision}")
    
    # Create directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    
    # Initialize model
    model_config = get_model_config()
    model = GPTNeoXForCausalLM(model_config)
    
    if accelerator.is_main_process:
        param_count = count_parameters(model)
        print(f"  Model parameters: {param_count/1e9:.2f}B")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Create dataloader
    train_dataloader, num_files = create_dataloader(config, accelerator)
    print(f"  Training files: {num_files}")
    print(f"  Steps per epoch: {len(train_dataloader)}")
    
    # Create scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    
    # Prepare for training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        config.bucket_name, 
        config.checkpoint_bucket_path, 
        config.output_dir,
        background_upload=not args.no_background_upload
    )
    
    # Training loop
    print(f"\n📈 Starting training...")
    start_time = time.time()
    model.train()
    completed_steps = 0
    total_loss = 0.0
    
    # Enhanced tracking for comprehensive checkpointing
    loss_history = []
    learning_rate_history = []
    best_loss = float('inf')
    validation_metrics = {}
    
    progress_bar = tqdm(
        range(config.max_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(100):  # Max 100 epochs
        for step, batch in enumerate(train_dataloader):
            
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                loss = outputs.loss
                current_loss = loss.detach().float().item()
                total_loss += current_loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Optimizer step
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                
                # Track metrics
                current_lr = lr_scheduler.get_last_lr()[0]
                learning_rate_history.append(current_lr)
                
                # Update best loss
                if current_loss < best_loss:
                    best_loss = current_loss
                
                # Log progress
                if completed_steps % config.logging_steps == 0:
                    avg_loss = total_loss / config.logging_steps
                    loss_history.append(avg_loss)
                    total_loss = 0.0
                    
                    if accelerator.is_main_process:
                        print(f"Step {completed_steps}: loss={avg_loss:.4f}, lr={current_lr:.2e}, best_loss={best_loss:.4f}")
                        
                        # Add validation metrics (can be expanded)
                        validation_metrics = {
                            'current_avg_loss': avg_loss,
                            'steps_since_best': completed_steps - (loss_history.index(min(loss_history)) + 1) * config.logging_steps if loss_history else 0,
                            'loss_trend': 'improving' if len(loss_history) >= 2 and loss_history[-1] < loss_history[-2] else 'stable',
                        }
                
                # Save checkpoint with comprehensive state
                if completed_steps % config.save_steps == 0:
                    if accelerator.is_main_process:
                        print(f"💾 Saving checkpoint at step {completed_steps}")
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
                                'training_config': config.__dict__
                            }
                        )
                
                # Check if done
                if completed_steps >= config.max_steps:
                    break
        
        if completed_steps >= config.max_steps:
            break
    
    # Final checkpoint with all accumulated data
    if accelerator.is_main_process:
        print("✅ Training complete! Saving final checkpoint...")
        
        # Final validation metrics summary
        final_validation_metrics = {
            **validation_metrics,
            'training_completed': True,
            'total_steps': completed_steps,
            'final_loss': current_loss,
            'best_loss_achieved': best_loss,
            'total_epochs': epoch,
            'average_loss': sum(loss_history) / len(loss_history) if loss_history else current_loss,
            'loss_std': np.std(loss_history) if len(loss_history) > 1 else 0.0,
        }
        
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
            loss_history=loss_history,
            learning_rates=learning_rate_history,
            validation_metrics=final_validation_metrics,
            metadata={
                'model_config': model_config.__dict__,
                'training_config': config.__dict__,
                'training_summary': {
                    'completed': True,
                    'total_time': time.time() - start_time if 'start_time' in locals() else None,
                }
            }
        )
        
        # Wait for all background uploads to complete
        print("⏳ Waiting for background uploads to complete...")
        checkpoint_manager.wait_for_uploads()
        print("🎉 All done! Training and uploads completed successfully.")


if __name__ == "__main__":
    main() 