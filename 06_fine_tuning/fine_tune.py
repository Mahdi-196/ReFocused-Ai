#!/usr/bin/env python3
"""
Fine-tuning script for ReFocused-AI model
Supports task-specific fine-tuning with LoRA and full fine-tuning options
"""

import os
import sys
import warnings

# Suppress FutureWarnings from transformers and torch
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import signal
import json
from pathlib import Path

# Set high precision for float32 matmul (improves H100/A100 performance)
torch.set_float32_matmul_precision('high')
from torch.optim import AdamW
from transformers import (
    GPTNeoXForCausalLM, 
    GPTNeoXConfig,
    get_scheduler, 
    set_seed,
    AutoTokenizer
)
from accelerate import Accelerator
import argparse
from tqdm import tqdm
import time
import numpy as np
import torch.backends.cudnn as cudnn
from typing import Dict, Optional, List

# Enable cuDNN benchmark and deterministic settings for performance
cudnn.benchmark = True
cudnn.deterministic = False

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.fine_tuning_config import get_fine_tuning_config
from utils.data_loaders import create_fine_tuning_dataloader
from utils.checkpoint_manager import FineTuningCheckpointManager
from utils.lora import LoRAConfig, apply_lora_to_model
from utils.metrics import compute_fine_tuning_metrics

# Global checkpoint manager for signal handler
checkpoint_manager = None

def signal_handler(signum, frame):
    """Handle interruption signals to complete uploads"""
    if checkpoint_manager:
        print(f"\n🛑 Fine-tuning interrupted. Waiting for background uploads to complete...")
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
        print("  ⚠️  Running on CPU - fine-tuning will be slow")
    return torch.cuda.is_available()


def load_base_model(model_path: str, device: str = "cuda") -> GPTNeoXForCausalLM:
    """Load the base model from checkpoint or HuggingFace"""
    print(f"📦 Loading base model from: {model_path}")
    
    if Path(model_path).exists():
        # Local checkpoint
        if (Path(model_path) / "pytorch_model.bin").exists():
            model = GPTNeoXForCausalLM.from_pretrained(model_path)
        else:
            # Load from custom checkpoint format
            checkpoint = torch.load(Path(model_path) / "model.pt", map_location=device)
            config = GPTNeoXConfig.from_dict(checkpoint['config'])
            model = GPTNeoXForCausalLM(config)
            model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # HuggingFace model
        model = GPTNeoXForCausalLM.from_pretrained(model_path)
    
    print(f"✅ Model loaded successfully")
    return model


def freeze_base_layers(model: GPTNeoXForCausalLM, freeze_ratio: float = 0.9):
    """Freeze bottom layers of the model for efficient fine-tuning"""
    total_layers = len(model.gpt_neox.layers)
    freeze_layers = int(total_layers * freeze_ratio)
    
    # Freeze embeddings
    for param in model.gpt_neox.embed_in.parameters():
        param.requires_grad = False
    
    # Freeze bottom layers
    for i in range(freeze_layers):
        for param in model.gpt_neox.layers[i].parameters():
            param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"🧊 Froze {freeze_layers}/{total_layers} layers")
    print(f"📊 Trainable parameters: {trainable_params/1e6:.2f}M ({trainable_params/total_params*100:.1f}%)")
    
    return model


def main():
    global checkpoint_manager
    
    parser = argparse.ArgumentParser(description="Fine-tune ReFocused-AI model")
    parser.add_argument("--task", type=str, required=True,
                       choices=["chat", "code", "instruct", "domain", "custom"],
                       help="Fine-tuning task type")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to fine-tuning dataset or HuggingFace dataset name")
    parser.add_argument("--base-model", type=str, required=True,
                       help="Path to base model checkpoint or HuggingFace model")
    parser.add_argument("--config", type=str, default="default",
                       help="Fine-tuning configuration name")
    parser.add_argument("--output-dir", type=str, default="./fine_tuned_models",
                       help="Output directory for fine-tuned model")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Override max training steps")
    parser.add_argument("--lora", action="store_true",
                       help="Use LoRA for parameter-efficient fine-tuning")
    parser.add_argument("--lora-rank", type=int, default=8,
                       help="LoRA rank (default: 8)")
    parser.add_argument("--freeze-ratio", type=float, default=0.0,
                       help="Ratio of layers to freeze (0.0 = full fine-tuning)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")
    parser.add_argument("--mixed-precision", type=str, choices=["no", "fp16", "bf16"], 
                       default=None, help="Override mixed precision setting")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--eval-steps", type=int, default=100,
                       help="Evaluation frequency")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Save checkpoint frequency")
    parser.add_argument("--push-to-hub", action="store_true",
                       help="Push model to HuggingFace Hub")
    parser.add_argument("--hub-model-id", type=str, default=None,
                       help="HuggingFace Hub model ID")
    
    args = parser.parse_args()
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Check GPU status
    cuda_available = check_gpu()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load configuration
    config = get_fine_tuning_config(args.config, args.task)
    if args.max_steps:
        config.max_steps = args.max_steps
    
    # Update config with command line arguments
    config.eval_steps = args.eval_steps
    config.save_steps = args.save_steps
    config.gradient_checkpointing = args.gradient_checkpointing
    
    # Determine mixed precision setting
    mixed_precision = "no"
    if cuda_available:
        if args.mixed_precision:
            mixed_precision = args.mixed_precision
        elif getattr(config, 'bf16', False):
            mixed_precision = "bf16"
        elif getattr(config, 'fp16', False):
            mixed_precision = "fp16"
    
    print(f"\n🚀 Starting {args.task.upper()} fine-tuning")
    print(f"  Base model: {args.base_model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Batch size per device: {config.per_device_train_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Mixed precision: {mixed_precision}")
    print(f"  LoRA: {'Enabled (rank={})'.format(args.lora_rank) if args.lora else 'Disabled'}")
    print(f"  Freeze ratio: {args.freeze_ratio}")
    
    # Initialize accelerator with optimized settings
    accelerator = Accelerator(
        gradient_accumulation_steps=getattr(config, 'gradient_accumulation_steps', 1),
        mixed_precision=mixed_precision,
        cpu=False,
    )
    
    print(f"  Device: {accelerator.device}")
    print(f"  Num processes: {accelerator.num_processes}")
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"{args.task}_{Path(args.dataset).stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load base model
    model = load_base_model(args.base_model, device=accelerator.device)
    
    # Load tokenizer
    if Path(args.base_model).exists() and (Path(args.base_model) / "tokenizer.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    else:
        # Try to find tokenizer in standard locations
        tokenizer_paths = [
            Path("tokenizer_1B"),
            Path("03_tokenizer_training/tokenizer_750M"),
            Path("models/tokenizer")
        ]
        tokenizer_path = None
        for path in tokenizer_paths:
            if path.exists() and (path / "tokenizer.json").exists():
                tokenizer_path = path
                break
        
        if tokenizer_path:
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
        else:
            raise ValueError("Could not find tokenizer. Please specify tokenizer path.")
    
    # Apply fine-tuning strategy
    if args.lora:
        print(f"🔧 Applying LoRA with rank={args.lora_rank}")
        lora_config = LoRAConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
        )
        model = apply_lora_to_model(model, lora_config)
    elif args.freeze_ratio > 0:
        model = freeze_base_layers(model, args.freeze_ratio)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params/1e9:.2f}B")
    print(f"  Trainable parameters: {trainable_params/1e9:.3f}B ({trainable_params/total_params*100:.1f}%)")
    
    # Create optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-8,
    )
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_fine_tuning_dataloader(
        dataset_path=args.dataset,
        task_type=args.task,
        tokenizer=tokenizer,
        config=config,
        accelerator=accelerator
    )
    
    print(f"  Training samples: {len(train_dataloader.dataset)}")
    if eval_dataloader:
        print(f"  Evaluation samples: {len(eval_dataloader.dataset)}")
    
    # Create scheduler
    num_training_steps = min(config.max_steps, len(train_dataloader) * config.num_epochs)
    lr_scheduler = get_scheduler(
        config.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Prepare for training with accelerator
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    if eval_dataloader:
        eval_dataloader = accelerator.prepare(eval_dataloader)
    
    # Initialize checkpoint manager
    checkpoint_manager = FineTuningCheckpointManager(
        output_dir=output_dir,
        task_type=args.task,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id
    )
    
    # Initialize training state
    completed_steps = 0
    best_eval_loss = float('inf')
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint_data = checkpoint_manager.load_checkpoint(
            checkpoint_path=args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            accelerator=accelerator
        )
        if checkpoint_data:
            completed_steps = checkpoint_data.get('step', 0)
            best_eval_loss = checkpoint_data.get('best_eval_loss', float('inf'))
            print(f"✅ Resumed from step {completed_steps}")
    
    # Training loop
    print(f"\n📈 Starting fine-tuning loop...")
    start_time = time.time()
    model.train()
    
    progress_bar = tqdm(
        range(num_training_steps),
        desc="Fine-tuning",
        disable=not accelerator.is_local_main_process,
        initial=completed_steps,
    )
    
    for epoch in range(config.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                loss = outputs.loss
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            if accelerator.sync_gradients:
                completed_steps += 1
                progress_bar.update(1)
                
                # Logging
                if completed_steps % config.logging_steps == 0:
                    current_loss = loss.detach().float().item()
                    current_lr = lr_scheduler.get_last_lr()[0]
                    
                    if accelerator.is_main_process:
                        progress_bar.set_postfix({
                            'loss': f'{current_loss:.4f}',
                            'lr': f'{current_lr:.2e}'
                        })
                
                # Evaluation
                if eval_dataloader and completed_steps % config.eval_steps == 0:
                    eval_loss, eval_metrics = evaluate(
                        model, eval_dataloader, accelerator, args.task
                    )
                    
                    if accelerator.is_main_process:
                        print(f"\n📊 Eval at step {completed_steps}: loss={eval_loss:.4f}")
                        for metric, value in eval_metrics.items():
                            print(f"   {metric}: {value:.4f}")
                        
                        # Save best model
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            checkpoint_manager.save_best_model(
                                model=accelerator.unwrap_model(model),
                                tokenizer=tokenizer,
                                eval_loss=eval_loss,
                                step=completed_steps
                            )
                    
                    model.train()
                
                # Save checkpoint
                if completed_steps % config.save_steps == 0:
                    if accelerator.is_main_process:
                        checkpoint_manager.save_checkpoint(
                            model=accelerator.unwrap_model(model),
                            optimizer=optimizer,
                            scheduler=lr_scheduler,
                            step=completed_steps,
                            epoch=epoch,
                            best_eval_loss=best_eval_loss,
                            accelerator=accelerator
                        )
                
                # Check if done
                if completed_steps >= num_training_steps:
                    break
        
        if completed_steps >= num_training_steps:
            break
    
    # Final save
    if accelerator.is_main_process:
        print("\n✅ Fine-tuning complete! Saving final model...")
        
        # Save final model
        final_model = accelerator.unwrap_model(model)
        checkpoint_manager.save_final_model(
            model=final_model,
            tokenizer=tokenizer,
            training_args=vars(args),
            final_metrics={
                'final_loss': loss.detach().float().item(),
                'best_eval_loss': best_eval_loss,
                'total_steps': completed_steps,
                'training_time': time.time() - start_time
            }
        )
        
        print(f"\n🎯 Fine-tuning Summary:")
        print(f"   Total steps: {completed_steps}")
        print(f"   Training time: {(time.time() - start_time)/3600:.2f} hours")
        print(f"   Best eval loss: {best_eval_loss:.4f}")
        print(f"   Output directory: {output_dir}")
        
        if args.push_to_hub:
            print(f"   🤗 Model pushed to: https://huggingface.co/{args.hub_model_id}")


def evaluate(model, eval_dataloader, accelerator, task_type):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process):
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
            )
            
            loss = outputs.loss
            total_loss += loss.detach().float() * batch['input_ids'].shape[0]
            total_samples += batch['input_ids'].shape[0]
            
            # Collect predictions for metrics
            predictions = outputs.logits.argmax(dim=-1)
            all_predictions.extend(accelerator.gather(predictions).cpu().numpy())
            all_labels.extend(accelerator.gather(batch['labels']).cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / total_samples
    metrics = compute_fine_tuning_metrics(
        predictions=all_predictions,
        labels=all_labels,
        task_type=task_type
    )
    
    return avg_loss.item(), metrics


if __name__ == "__main__":
    main() 