#!/usr/bin/env python3
"""
Main Training Script for 1B Parameter Model
Optimized for 8x H100 SXM with DeepSpeed ZeRO Stage 3
"""

import os
import sys
import yaml
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import signal
import atexit

import torch
import torch.nn.functional as F
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
    set_seed
)
import deepspeed
from deepspeed.utils.zero_to_fp32 import save_fp32_model
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
        logging.FileHandler('/scratch/logs/training.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Main training class with DeepSpeed integration"""
    
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
        model_config_path = Path(self.config['model']['config_path'])
        with open(model_config_path, 'r') as f:
            model_config_dict = json.load(f)
        
        self.model_config = GPT2Config(**model_config_dict)
        
        # Initialize tokenizer
        tokenizer_path = self.config['tokenizer']['path']
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        
        # Set special tokens
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Update vocab size in config to match tokenizer
        self.model_config.vocab_size = len(self.tokenizer)
        
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
            local_path=data_config['local_data_dir']
        )
        
        # Sync data from GCS to local storage
        logger.info("Syncing training data from GCS...")
        sync_success = self.gcs_manager.sync_data_to_local(max_workers=8)
        
        if not sync_success:
            raise RuntimeError("Failed to sync training data from GCS")
        
        # Create data loader
        self.train_dataloader = create_dataloader(
            data_dir=data_config['local_data_dir'],
            batch_size=self.config['training']['per_device_train_batch_size'],
            sequence_length=data_config['max_seq_length'],
            num_workers=self.config['training']['dataloader_num_workers'],
            prefetch_factor=self.config['training']['dataloader_prefetch_factor']
        )
        
        logger.info("Data loading setup completed")
    
    def setup_monitoring(self):
        """Setup monitoring and logging"""
        monitoring_config = self.config['monitoring']
        
        # Create logs directory
        os.makedirs(monitoring_config['logging_dir'], exist_ok=True)
        
        # Initialize Weights & Biases
        if 'wandb' in monitoring_config['report_to']:
            wandb.init(
                project=monitoring_config['wandb_project'],
                entity=monitoring_config.get('wandb_entity'),
                config=self.config,
                resume="allow" if self.resume_from_checkpoint else None
            )
        
        # Initialize system monitoring
        if monitoring_config.get('monitor_system_metrics', True):
            self.gpu_monitor = GPUMonitor(
                log_interval=monitoring_config.get('log_gpu_metrics_interval', 60)
            )
            self.gpu_monitor.start()
        
        # Initialize training logger
        self.training_logger = TrainingLogger(
            log_dir=monitoring_config['logging_dir'],
            log_interval=self.config['training']['logging_steps']
        )
        
        logger.info("Monitoring setup completed")
    
    def setup_checkpoint_manager(self):
        """Setup checkpoint management"""
        checkpoint_config = self.config['checkpointing']
        
        self.checkpoint_manager = CheckpointManager(
            output_dir=checkpoint_config['output_dir'],
            remote_bucket=checkpoint_config['remote_checkpoint_bucket'],
            remote_path=checkpoint_config['remote_checkpoint_path'],
            backup_every_n_steps=checkpoint_config['backup_every_n_steps'],
            keep_last_n=checkpoint_config['keep_last_n_checkpoints']
        )
        
        logger.info("Checkpoint management setup completed")
    
    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.save_checkpoint(f"emergency_checkpoint_step_{self.global_step}")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Also register atexit handler
        atexit.register(self.cleanup)
    
    def initialize_deepspeed(self):
        """Initialize DeepSpeed engine"""
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
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step"""
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
        
        # Save DeepSpeed checkpoint
        self.engine.save_checkpoint(checkpoint_dir)
        
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
        # Load DeepSpeed checkpoint
        _, client_state = self.engine.load_checkpoint(checkpoint_path)
        
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
        if hasattr(self, 'gpu_monitor'):
            self.gpu_monitor.stop()
        
        if wandb.run is not None:
            wandb.finish()
        
        logger.info("Training cleanup completed")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Initialize DeepSpeed
        self.initialize_deepspeed()
        
        # Calculate training parameters
        total_steps = self.config['training']['total_steps']
        save_steps = self.config['training']['save_steps']
        eval_steps = self.config['training']['eval_steps']
        logging_steps = self.config['training']['logging_steps']
        
        # Training loop
        start_time = time.time()
        
        for batch in self.train_dataloader:
            if self.global_step >= total_steps:
                break
            
            # Training step
            step_metrics = self.train_step(batch)
            
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
        
        # Save model in HuggingFace format
        self.engine.save_pretrained(export_dir)
        self.tokenizer.save_pretrained(export_dir)
        
        # Upload to GCS
        self.gcs_manager.upload_checkpoint(export_dir, "final_model")
        
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
    
    # Setup directories
    os.makedirs("/scratch/logs", exist_ok=True)
    os.makedirs("/scratch/checkpoints", exist_ok=True)
    os.makedirs("/scratch/shards", exist_ok=True)
    os.makedirs("/scratch/deepspeed_nvme", exist_ok=True)
    
    # Initialize trainer
    trainer = ModelTrainer(
        config_path=args.config,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Run training
    trainer.train()


if __name__ == "__main__":
    main() 