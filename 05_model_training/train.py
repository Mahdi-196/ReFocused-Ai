#!/usr/bin/env python3
"""
Clean training script for ReFocused-AI 1.2B model
"""

import os
import sys
import torch
from torch.optim import AdamW
from transformers import GPTNeoXForCausalLM, get_scheduler, set_seed
from accelerate import Accelerator
import argparse
from tqdm import tqdm
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_model_config, get_training_config
from utils import create_dataloader, CheckpointManager, count_parameters


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
    parser = argparse.ArgumentParser(description="Train ReFocused-AI model")
    parser.add_argument("--config", type=str, choices=["test", "production"], 
                       default="test", help="Training configuration")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Override max steps")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    args = parser.parse_args()
    
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
        config.output_dir
    )
    
    # Training loop
    print(f"\n📈 Starting training...")
    model.train()
    completed_steps = 0
    total_loss = 0.0
    
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
                total_loss += loss.detach().float()
                
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
                
                # Log progress
                if completed_steps % config.logging_steps == 0:
                    avg_loss = total_loss / config.logging_steps
                    total_loss = 0.0
                    
                    if accelerator.is_main_process:
                        current_lr = lr_scheduler.get_last_lr()[0]
                        print(f"Step {completed_steps}: loss={avg_loss:.4f}, lr={current_lr:.2e}")
                
                # Save checkpoint
                if completed_steps % config.save_steps == 0:
                    if accelerator.is_main_process:
                        print(f"💾 Saving checkpoint at step {completed_steps}")
                        checkpoint_manager.save_checkpoint(
                            accelerator, model, optimizer, lr_scheduler,
                            epoch, completed_steps, completed_steps // len(train_dataloader)
                        )
                
                # Check if done
                if completed_steps >= config.max_steps:
                    break
        
        if completed_steps >= config.max_steps:
            break
    
    # Final checkpoint
    if accelerator.is_main_process:
        print("✅ Training complete! Saving final checkpoint...")
        checkpoint_manager.save_checkpoint(
            accelerator, model, optimizer, lr_scheduler,
            epoch, completed_steps, completed_steps // len(train_dataloader)
        )


if __name__ == "__main__":
    main() 