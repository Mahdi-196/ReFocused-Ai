#!/usr/bin/env python3
"""
Modified Training Script with Standard PyTorch training support
Provides both DeepSpeed and standard PyTorch training options
"""

import os
import sys
import yaml
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import signal
import atexit
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
    set_seed
)
import deepspeed
import wandb

# Import custom utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import create_dataloader, GCSDataManager, estimate_training_time
from utils.monitoring import GPUMonitor, SystemMonitor, TrainingLogger
from utils.checkpoint_manager import CheckpointManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Flexible training class supporting both DeepSpeed and standard PyTorch"""
    
    def __init__(self, config_path: str, resume_from_checkpoint: Optional[str] = None):
        self.config = self.load_config(config_path)
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Initialize components
        self.setup_environment()
        self.setup_model_and_tokenizer()
        self.setup_data()
        self.setup_monitoring()
        self.setup_checkpoint_manager()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup graceful shutdown
        self.setup_signal_handlers()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def setup_environment(self):
        """Setup training environment and variables"""
        # Set environment variables
        env_config = self.config.get('environment', {})
        self.global_config = self.config # Store full config for later use

        if 'cuda_visible_devices' in env_config:
            os.environ['CUDA_VISIBLE_DEVICES'] = env_config['cuda_visible_devices']
        
        if 'nccl_debug' in env_config:
            os.environ['NCCL_DEBUG'] = env_config['nccl_debug']
        
        # Set random seed
        set_seed(env_config.get('seed', 42))
        
        # PyTorch optimizations
        if env_config.get('torch_backends_cudnn_benchmark', True):
            torch.backends.cudnn.benchmark = True
        
        logger.info("Environment setup completed")
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer"""
        # Load model configuration
        model_cfg = self.config['model']
        model_config_path = Path(model_cfg['config_path'])
        with open(model_config_path, 'r') as f:
            model_config_dict = json.load(f)

        self.model_config = GPT2Config(**model_config_dict)

        # Initialize tokenizer
        tokenizer_path = self.config['tokenizer']['path']
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

        # Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Determine vocab_size
        # Option 1: Use explicit vocab_size from config if specified
        if model_cfg.get('vocab_size_scan_files', 0) == 0 and 'vocab_size' in model_cfg:
             self.model_config.vocab_size = model_cfg['vocab_size']
             logger.info(f"Using explicitly configured vocab_size: {self.model_config.vocab_size} from training_config.yaml")
        else:
            # Option 2: Use tokenizer's vocab size
            self.model_config.vocab_size = len(self.tokenizer)
            logger.info(f"Using tokenizer's vocab_size: {self.model_config.vocab_size}")

        # Initialize model
        self.model = GPT2LMHeadModel(self.model_config)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model initialized with {total_params:,} total parameters")
        logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def setup_data(self):
        """Setup data loading and GCS synchronization"""
        data_config = self.config['data']
        
        # Initialize GCS data manager
        self.gcs_manager = GCSDataManager(
            bucket_name=data_config['remote_data_bucket'],
            remote_path=data_config['remote_data_path'],
            local_path=data_config['local_data_dir'],
            use_gcs=data_config.get('use_gcs', True),
            gcs_client_type=data_config.get('gcs_client_type', 'default')
        )

        # Sync data from GCS to local storage if GCS is enabled
        if self.gcs_manager.use_gcs:
            logger.info("Syncing training data from GCS...")
            sync_success = self.gcs_manager.sync_data_to_local(
                max_workers=data_config.get('preprocessing_num_workers', 8)
            )
            if not sync_success:
                logger.warning("Failed to sync all training data from GCS. Training may proceed with local data.")
        else:
            logger.info("GCS usage is disabled. Skipping data sync from GCS.")
            
        # Create data loader
        self.train_dataloader = create_dataloader(
            data_dir=data_config['local_data_dir'],
            batch_size=self.config['training']['per_device_train_batch_size'],
            sequence_length=data_config['max_seq_length'],
            num_workers=self.config['training']['dataloader_num_workers'],
            prefetch_factor=self.config['training']['dataloader_prefetch_factor'],
            npz_key_priority=data_config.get('npz_key_priority', ['input_ids', 'arr_0', 'text', 'sequences'])
        )
        
        logger.info("Data loading setup completed")
    
    def setup_monitoring(self):
        """Setup monitoring and logging"""
        monitoring_config = self.config['monitoring']
        
        # Create logs directory
        log_dir = Path(monitoring_config['logging_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if 'wandb' in monitoring_config.get('report_to', []):
            wandb_mode = "offline" if os.environ.get("WANDB_MODE") == "offline" else "online"
            wandb.init(
                project=monitoring_config['wandb_project'],
                entity=monitoring_config['wandb_entity'],
                config=self.config,
                mode=wandb_mode,
            )
        
        # Setup file logging
        log_file = log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Initialize GPU monitoring
        if monitoring_config.get('monitor_gpu_memory', True):
            num_gpus = torch.cuda.device_count()
            self.gpu_monitor = GPUMonitor(
                interval=monitoring_config.get('log_gpu_metrics_interval', 60),
                num_gpus=num_gpus
            )
            self.gpu_monitor.start()
        
        # Initialize system monitoring
        if monitoring_config.get('monitor_system_metrics', True):
            self.system_monitor = SystemMonitor(
                interval=monitoring_config.get('log_gpu_metrics_interval', 60)
            )
            self.system_monitor.start()
        
        # Initialize training logger
        self.training_logger = TrainingLogger(log_dir=log_dir)
        
        logger.info("Monitoring setup completed")
    
    def setup_checkpoint_manager(self):
        """Setup checkpoint manager"""
        checkpoint_config = self.config['checkpointing']
        
        self.checkpoint_manager = CheckpointManager(
            local_dir=checkpoint_config['output_dir'],
            remote_bucket=checkpoint_config['remote_checkpoint_bucket'],
            remote_path=checkpoint_config['remote_checkpoint_path'],
            gcs_manager=self.gcs_manager,
            keep_last_n=checkpoint_config.get('keep_last_n_checkpoints', 3),
            save_total_limit=checkpoint_config.get('save_total_limit', 10)
        )
        
        # Create output directory
        Path(checkpoint_config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        logger.info("Checkpoint management setup completed")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} received. Saving checkpoint and exiting...")
            self.cleanup()
            self.save_checkpoint(f"emergency_checkpoint_step_{self.global_step}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Also register atexit handler
        atexit.register(self.cleanup)
    
    def initialize_deepspeed(self):
        """Initialize DeepSpeed engine if DeepSpeed is enabled"""
        # Check if DeepSpeed is enabled in config
        if not self.config.get('deepspeed', {}).get('config_path'):
            logger.info("DeepSpeed configuration not found. Using standard PyTorch training.")
            
            # Initialize standard PyTorch optimizer and scheduler
            self.setup_pytorch_training()
            return False
        
        # Load DeepSpeed configuration
        deepspeed_config_path = self.config['deepspeed']['config_path']
        
        # Initialize DeepSpeed engine
        self.engine, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            config=deepspeed_config_path,
            model_parameters=self.model.parameters()
        )
        
        # Load checkpoint if resuming
        if self.resume_from_checkpoint:
            self.load_checkpoint(self.resume_from_checkpoint)
        
        logger.info("DeepSpeed initialization completed")
        logger.info(f"DeepSpeed ZeRO stage: {self.engine.zero_optimization_stage()}")
        return True
    
    def setup_pytorch_training(self):
        """Initialize standard PyTorch training components"""
        training_config = self.config['training']
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            betas=(0.9, 0.95)
        )
        
        # Setup scheduler
        total_steps = training_config['total_steps']
        warmup_steps = training_config['warmup_steps']
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Setup gradient accumulation
        self.gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        self.accumulated_steps = 0
        
        # Setup mixed precision training
        self.use_fp16 = training_config.get('fp16', False)
        self.use_bf16 = training_config.get('bf16', False)
        
        if self.use_fp16:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info("Standard PyTorch training setup completed")
        
        # Set gradient checkpointing
        if training_config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    def pytorch_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with standard PyTorch"""
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Determine if this is an accumulation step or update step
        is_update_step = (self.accumulated_steps + 1) % self.gradient_accumulation_steps == 0
        
        # Mixed precision context
        if self.use_fp16:
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            if is_update_step:
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        # BF16 precision
        elif self.use_bf16:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss / self.gradient_accumulation_steps
                
            # Backward pass (no scaler needed for bf16)
            loss.backward()
            
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        # Full precision
        else:
            # Forward pass
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            loss = outputs.loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
        
        # Update counters
        self.accumulated_steps += 1
        if is_update_step:
            self.global_step += 1
        
        return {
            'loss': loss.item() * self.gradient_accumulation_steps,  # Rescale to original loss
            'lr': self.scheduler.get_last_lr()[0],
            'step': self.global_step
        }
    
    def deepspeed_train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step with DeepSpeed"""
        # Move batch to device
        batch = {k: v.to(self.engine.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.engine(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss
        
        # Backward pass
        self.engine.backward(loss)
        self.engine.step()
        
        # Update global step
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'lr': self.scheduler.get_last_lr()[0] if self.scheduler else 0.0,
            'step': self.global_step
        }
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation (placeholder for now)"""
        # For now, just return training loss
        # In a full implementation, you'd have a separate eval dataset
        return {'eval_loss': self.best_loss}
    
    def save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint"""
        checkpoint_dir = f"{self.config['checkpointing']['output_dir']}/{checkpoint_name}"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model state differently depending on training mode
        if hasattr(self, 'engine') and self.engine is not None:
            # DeepSpeed checkpoint
            self.engine.save_checkpoint(checkpoint_dir)
        else:
            # Standard PyTorch checkpoint
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'scaler': self.scaler.state_dict() if hasattr(self, 'scaler') else None,
            }
            torch.save(checkpoint, f"{checkpoint_dir}/pytorch_model.bin")
        
        # Save additional metadata
        metadata = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_loss': self.best_loss,
            'model_config': self.model_config.to_dict(),
            'training_config': self.config
        }
        
        with open(f"{checkpoint_dir}/training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Upload to GCS
        self.checkpoint_manager.backup_checkpoint(checkpoint_dir)
        
        logger.info(f"Checkpoint saved: {checkpoint_name}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        if hasattr(self, 'engine') and self.engine is not None:
            # DeepSpeed checkpoint
            _, client_state = self.engine.load_checkpoint(checkpoint_path)
        else:
            # Standard PyTorch checkpoint
            checkpoint_file = f"{checkpoint_path}/pytorch_model.bin"
            if os.path.exists(checkpoint_file):
                checkpoint = torch.load(checkpoint_file, map_location=self.device)
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                
                if self.scheduler and 'scheduler' in checkpoint and checkpoint['scheduler']:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                
                if hasattr(self, 'scaler') and 'scaler' in checkpoint and checkpoint['scaler']:
                    self.scaler.load_state_dict(checkpoint['scaler'])
        
        # Load metadata
        metadata_path = f"{checkpoint_path}/training_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.global_step = metadata.get('global_step', 0)
            self.epoch = metadata.get('epoch', 0)
            self.best_loss = metadata.get('best_loss', float('inf'))
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'gpu_monitor') and self.gpu_monitor:
            self.gpu_monitor.stop()
        
        if wandb.run is not None:
            wandb.finish()
        
        logger.info("Training cleanup completed")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Initialize either DeepSpeed or standard PyTorch training
        use_deepspeed = self.initialize_deepspeed()
        
        if use_deepspeed:
            logger.info("Using DeepSpeed for training")
            train_step_fn = self.deepspeed_train_step
        else:
            logger.info("Using standard PyTorch training (DeepSpeed disabled)")
            train_step_fn = self.pytorch_train_step
        
        # Calculate training parameters
        total_steps = self.config['training']['total_steps']
        save_steps = self.config['training']['save_steps']
        eval_steps = self.config['training']['eval_steps']
        logging_steps = self.config['training']['logging_steps']
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config['training']['max_epochs']):
            self.epoch = epoch
            
            for batch in self.train_dataloader:
                if self.global_step >= total_steps:
                    break
                
                # Training step (either DeepSpeed or PyTorch)
                step_metrics = train_step_fn(batch)
                
                # Only proceed with logging and checkpointing on actual steps (not accumulation steps)
                if 'step' in step_metrics and step_metrics['step'] == self.global_step:
                    # Logging
                    if self.global_step % logging_steps == 0:
                        elapsed_time = time.time() - start_time
                        steps_per_second = self.global_step / elapsed_time if elapsed_time > 0 else 0
                        
                        log_metrics = {
                            **step_metrics,
                            'steps_per_second': steps_per_second,
                            'tokens_per_second': steps_per_second * self.config['training']['per_device_train_batch_size'] * self.config['data']['max_seq_length'] * self.config['hardware']['num_gpus'],
                            'memory_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                            'memory_reserved': torch.cuda.memory_reserved() / 1024**3,   # GB
                        }
                        
                        # Log to console
                        logger.info(f"Step {self.global_step}: loss={step_metrics['loss']:.4f}, lr={step_metrics['lr']:.2e}, steps/s={steps_per_second:.2f}")
                        
                        # Log to wandb
                        if wandb.run is not None:
                            wandb.log(log_metrics, step=self.global_step)
                        
                        # Log to file
                        self.training_logger.log_metrics(log_metrics, self.global_step)
                    
                    # Evaluation
                    if self.global_step % eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        if eval_metrics['eval_loss'] < self.best_loss:
                            self.best_loss = eval_metrics['eval_loss']
                            self.save_checkpoint(f"best_checkpoint_step_{self.global_step}")
                        
                        if wandb.run is not None:
                            wandb.log(eval_metrics, step=self.global_step)
                    
                    # Save checkpoint
                    if self.global_step % save_steps == 0:
                        self.save_checkpoint(f"checkpoint_step_{self.global_step}")
            
            # End of epoch
            logger.info(f"Completed epoch {self.epoch}")
        
        # Save final checkpoint
        self.save_checkpoint(f"final_checkpoint_step_{self.global_step}")
        
        # Export final model to HuggingFace format
        self.export_hf_model()
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f} seconds")
        logger.info(f"Final step: {self.global_step}")
        logger.info(f"Average steps per second: {self.global_step / total_time:.2f}")
    
    def export_hf_model(self):
        """Export final model to HuggingFace format"""
        export_dir = f"{self.config['checkpointing']['output_dir']}/hf_model"
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        
        # Save model in HuggingFace format
        # Ensure model is in eval mode for saving
        if hasattr(self, 'engine') and self.engine is not None:
            # If using DeepSpeed
            if hasattr(self.engine, 'module'):
                model_to_save = self.engine.module
            else:
                model_to_save = self.engine
            
            model_to_save.eval()
            self.engine.save_pretrained(export_dir)
        else:
            # Standard PyTorch model
            self.model.eval()
            self.model.save_pretrained(export_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(export_dir)
        logger.info(f"Model and tokenizer saved to {export_dir} locally")
        
        # Upload to GCS if enabled
        if self.gcs_manager and self.gcs_manager.use_gcs:
            logger.info(f"Attempting to upload HF model from {export_dir} to GCS...")
            success = self.checkpoint_manager.backup_checkpoint(export_dir)
            if success:
                logger.info(f"HuggingFace model from {export_dir} uploaded to GCS")
            else:
                logger.error(f"Failed to upload HuggingFace model from {export_dir} to GCS")
        else:
            logger.info("GCS usage is disabled. Skipping HF model upload to GCS.")
            
        logger.info(f"Model exported to HuggingFace format: {export_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train 1B Parameter Model")
    parser.add_argument("--config", type=str, default="config/training_config.yaml",
                       help="Path to training configuration file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    
    args = parser.parse_args()

    # Create required directories early
    temp_config = {}
    try:
        with open(args.config, 'r') as f:
            temp_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config for directory setup: {e}. Using defaults if any.")
    
    # Setup directories
    log_dir_path = Path(temp_config.get('monitoring', {}).get('logging_dir', 'logs'))
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    local_data_dir_path = Path(temp_config.get('data', {}).get('local_data_dir', 'data/shards'))
    local_data_dir_path.mkdir(parents=True, exist_ok=True)

    checkpoints_base_path = Path(temp_config.get('checkpointing', {}).get('output_dir', 'checkpoints'))
    checkpoints_base_path.mkdir(parents=True, exist_ok=True)

    nvme_offload_dir_config = temp_config.get('hardware', {}).get('nvme_offload_dir')
    if nvme_offload_dir_config:
        Path(nvme_offload_dir_config).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(
        config_path=args.config,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main() 