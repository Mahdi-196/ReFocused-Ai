"""
Main training script for ReFocused-AI model with FSDP support
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    GPTNeoXForCausalLM,
    get_scheduler,
    set_seed
)
from accelerate import Accelerator
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import argparse
from tqdm import tqdm
import math

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_model_config, get_test_config, get_production_config
from utils import (
    create_dataloader,
    CheckpointManager,
    MetricsTracker,
    get_grad_norm,
    compute_perplexity,
    count_parameters,
    format_metrics_log
)


def setup_fsdp_plugin():
    """Configure FSDP plugin for Accelerate"""
    # Import GPTNeoXLayer to use in wrap policy
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
    
    # Create auto wrap policy for GPTNeoXLayer
    auto_wrap_policy = lambda module, *args, **kwargs: transformer_auto_wrap_policy(
        module,
        {GPTNeoXLayer},
        *args,
        **kwargs
    )
    
    fsdp_plugin = FullyShardedDataParallelPlugin(
        # Sharding strategy
        sharding_strategy="FULL_SHARD",
        
        # CPU offload for memory efficiency (disable for max speed)
        cpu_offload=False,
        
        # Backward prefetch for better overlap
        backward_prefetch="BACKWARD_PRE",
        
        # Forward prefetch
        forward_prefetch=True,
        
        # Use orig params for better optimizer compatibility  
        use_orig_params=True,
        
        # Sync module states for better checkpoint compatibility
        sync_module_states=True,
        
        # Activation checkpointing
        activation_checkpointing=True,
        
        # Auto wrap policy - using function instead of string
        auto_wrap_policy=auto_wrap_policy,
        
        # State dict config for checkpointing
        state_dict_type="FULL_STATE_DICT",
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )
    return fsdp_plugin


def main():
    parser = argparse.ArgumentParser(description="Train ReFocused-AI model")
    parser.add_argument("--mode", type=str, choices=["test", "production"], 
                       default="test", help="Training mode")
    parser.add_argument("--resume", type=str, default=None, 
                       help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get configuration based on mode
    if args.mode == "test":
        config = get_test_config()
        print("Running TEST training with 25 files...")
    else:
        config = get_production_config()
        print("Running PRODUCTION training on full dataset...")
    
    # Initialize FSDP plugin
    fsdp_plugin = setup_fsdp_plugin()
    
    # Initialize accelerator with FSDP
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision="bf16" if config.bf16 else "fp16" if config.fp16 else "no",
        log_with=config.report_to,
        project_dir=config.logging_dir,
        fsdp_plugin=fsdp_plugin,
    )
    
    # Initialize logging
    if accelerator.is_main_process:
        print(f"Accelerator state: {accelerator.state}")
        print(f"Device: {accelerator.device}")
        print(f"Distributed: {accelerator.distributed_type}")
        print(f"Num processes: {accelerator.num_processes}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Setup directories
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.logging_dir, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    
    # Initialize tracking
    metrics_tracker = MetricsTracker(config.logging_dir, config.run_name) if accelerator.is_main_process else None
    checkpoint_manager = CheckpointManager(config.bucket_name, config.checkpoint_bucket_path, config.output_dir)
    
    # Create model
    model_config = get_model_config()
    
    # Initialize model with empty weights to save memory
    with accelerator.main_process_first():
        print("Initializing model...")
        model = GPTNeoXForCausalLM(model_config)
        
        # Initialize weights
        model.apply(model._init_weights)
        
        if accelerator.is_main_process:
            param_count = count_parameters(model)
            print(f"Model initialized with {param_count/1e9:.2f}B parameters")
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_epsilon,
        weight_decay=config.weight_decay,
    )
    
    # Create dataloader
    train_dataloader, num_files = create_dataloader(config, accelerator)
    
    # Calculate training steps
    num_update_steps_per_epoch = len(train_dataloader) // config.gradient_accumulation_steps
    if config.max_steps > 0:
        max_steps = config.max_steps
    else:
        max_steps = num_update_steps_per_epoch * 100  # Default to 100 epochs
    
    # Create scheduler
    lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=max_steps,
    )
    
    # Prepare for distributed training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Resume from checkpoint if specified
    starting_epoch = 0
    completed_steps = 0
    files_processed = 0
    
    if args.resume or config.resume_from_checkpoint:
        checkpoint_name = args.resume or config.resume_from_checkpoint
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {checkpoint_name}")
        
        metadata = checkpoint_manager.load_checkpoint(accelerator, checkpoint_name)
        if metadata:
            starting_epoch = metadata.get('epoch', 0)
            completed_steps = metadata.get('step', 0)
            files_processed = metadata.get('files_processed', 0)
    
    # Training loop
    if accelerator.is_main_process:
        print(f"Starting training from epoch {starting_epoch}, step {completed_steps}")
        print(f"Total files: {num_files}, Files processed: {files_processed}")
    
    model.train()
    total_loss = 0
    tokens_processed = 0
    
    # Create progress bar
    progress_bar = tqdm(
        range(completed_steps, max_steps),
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )
    
    for epoch in range(starting_epoch, 100):  # Max 100 epochs
        for step, batch in enumerate(train_dataloader):
            # Skip completed steps when resuming
            if completed_steps > 0 and step < completed_steps:
                continue
            
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
                
                # Gradient clipping
                if accelerator.sync_gradients and config.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update metrics
            if accelerator.sync_gradients:
                completed_steps += 1
                tokens_processed += batch['input_ids'].numel() * accelerator.num_processes
                
                # Calculate file progress
                steps_per_file = len(train_dataloader) // num_files
                current_files = min(num_files, completed_steps // steps_per_file)
                
                # Log metrics
                if completed_steps % config.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / config.gradient_accumulation_steps / config.logging_steps
                    total_loss = 0
                    
                    # Get additional metrics
                    grad_norm = get_grad_norm(model) if accelerator.is_main_process else None
                    perplexity = compute_perplexity(avg_loss)
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    if accelerator.is_main_process:
                        # Update metrics tracker
                        metrics_tracker.update({
                            'loss': avg_loss,
                            'learning_rate': current_lr,
                            'grad_norm': grad_norm,
                            'perplexity': perplexity,
                            'batch_size': config.per_device_train_batch_size * accelerator.num_processes,
                        }, completed_steps)
                        
                        # Print log
                        log_str = format_metrics_log(
                            step=completed_steps,
                            epoch=epoch,
                            loss=avg_loss,
                            learning_rate=current_lr,
                            grad_norm=grad_norm,
                            perplexity=perplexity,
                            files_processed=current_files,
                            tokens_processed=tokens_processed
                        )
                        tqdm.write(log_str)
                
                # Save checkpoint
                should_save = (
                    completed_steps % config.save_steps == 0 or
                    (current_files > files_processed and current_files % config.checkpoint_every_n_files == 0)
                )
                
                if should_save:
                    if accelerator.is_main_process:
                        print(f"\nSaving checkpoint at step {completed_steps}, files {current_files}...")
                    
                    checkpoint_manager.save_checkpoint(
                        accelerator=accelerator,
                        model=model,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        epoch=epoch,
                        step=completed_steps,
                        files_processed=current_files,
                        metadata={
                            'model_config': model_config.to_dict(),
                            'training_config': config.__dict__,
                        }
                    )
                    files_processed = current_files
                
                # Update progress bar
                progress_bar.update(1)
                
                # Check if we've reached max steps
                if completed_steps >= max_steps:
                    break
        
        if completed_steps >= max_steps:
            break
    
    # Final checkpoint
    if accelerator.is_main_process:
        print("\nTraining complete! Saving final checkpoint...")
        checkpoint_manager.save_checkpoint(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            epoch=epoch,
            step=completed_steps,
            files_processed=files_processed,
            metadata={
                'model_config': model_config.to_dict(),
                'training_config': config.__dict__,
            }
        )
        
        metrics_tracker.close()
        print("Training finished successfully!")


if __name__ == "__main__":
    main() 