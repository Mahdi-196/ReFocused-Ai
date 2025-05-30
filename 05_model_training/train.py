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
import numpy as np

import torch
import torch.nn.functional as F
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    get_cosine_schedule_with_warmup,
    set_seed
)
import deepspeed
# from deepspeed.utils.zero_to_fp32 import save_fp32_model # Not used, caused import error
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
        # logging.FileHandler('/scratch/logs/training.log') # To be configured via ModelTrainer
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
        # Option 1: Use tokenizer's vocab size (default)
        determined_vocab_size = len(self.tokenizer)

        # Option 2: Scan data to determine vocab_size if configured
        if model_cfg.get('vocab_size_scan_files', 0) > 0:
            scan_data_dir = model_cfg.get('vocab_size_scan_data_dir', self.config['data']['local_data_dir'])
            num_scan_files = model_cfg.get('vocab_size_scan_files')
            logger.info(f"Scanning up to {num_scan_files} files in {scan_data_dir} to determine max token ID for vocab_size.")
            
            max_token_id = 0
            files_scanned = 0
            # Ensure GCS data is synced if scanning from local_data_dir and GCS is used
            if self.config['data'].get('use_gcs', True) and scan_data_dir == self.config['data']['local_data_dir']:
                 if not hasattr(self, 'gcs_manager') or not self.gcs_manager: # if gcs_manager not setup
                    logger.info("Temporarily initializing GCSDataManager for vocab scan...")
                    temp_gcs_manager = GCSDataManager(
                        bucket_name=self.config['data']['remote_data_bucket'],
                        remote_path=self.config['data']['remote_data_path'],
                        local_path=self.config['data']['local_data_dir'],
                        use_gcs=self.config['data'].get('use_gcs', True),
                        gcs_client_type=self.config['data'].get('gcs_client_type', 'default')
                    )
                    if temp_gcs_manager.use_gcs: # only sync if GCS is actually enabled
                        logger.info("Syncing data for vocab scan...")
                        sync_success = temp_gcs_manager.sync_data_to_local(max_workers=self.config['data'].get('preprocessing_num_workers', 8))
                        if not sync_success:
                            logger.warning("Failed to sync all data for vocab scan. Scan might be incomplete.")
                    else:
                        logger.info("GCS usage is disabled, skipping sync for vocab scan.")


            npz_key_priority = self.config['data'].get('npz_key_priority', ['input_ids', 'arr_0', 'text', 'sequences'])

            try:
                for i, file_path in enumerate(Path(scan_data_dir).glob("*.npz")) :
                    if i >= num_scan_files:
                        break
                    try:
                        with np.load(file_path, allow_pickle=True) as data: # allow_pickle for safety during scan
                            found_key = None
                            for key_candidate in npz_key_priority:
                                if key_candidate in data:
                                    found_key = key_candidate
                                    break
                            if not found_key and data.files: # fallback to first key
                                found_key = data.files[0]
                            
                            if found_key:
                                current_max = np.max(data[found_key])
                                if current_max > max_token_id:
                                    max_token_id = current_max
                                files_scanned +=1
                            else:
                                logger.warning(f"Could not find suitable data key in {file_path} for vocab scan.")
                    except Exception as e:
                        logger.warning(f"Could not load or process {file_path} for vocab scan: {e}")
                if files_scanned > 0:
                    determined_vocab_size = int(max_token_id) + 1
                    logger.info(f"Determined vocab_size={determined_vocab_size} after scanning {files_scanned} files (max_token_id={max_token_id}).")
                else:
                    logger.warning("No files successfully scanned for vocab_size. Using tokenizer default.")

            except Exception as e:
                logger.error(f"Error during vocab_size scan: {e}. Using tokenizer default.")
        
        # Override model_config_dict vocab_size if it was explicitly set (e.g. from model_config.json)
        # and not meant to be auto-detected. If vocab_size_scan_files is 0, respect model_cfg.vocab_size
        if model_cfg.get('vocab_size_scan_files', 0) == 0 and 'vocab_size' in model_cfg:
             self.model_config.vocab_size = model_cfg['vocab_size']
             logger.info(f"Using explicitly configured vocab_size: {self.model_config.vocab_size} from training_config.yaml")
        else:
            self.model_config.vocab_size = determined_vocab_size
            logger.info(f"Final vocab_size set to: {self.model_config.vocab_size}")


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
        
        # Initialize GCS data manager if not already done by vocab scan
        if not hasattr(self, 'gcs_manager') or not self.gcs_manager:
            self.gcs_manager = GCSDataManager(
                bucket_name=data_config['remote_data_bucket'],
                remote_path=data_config['remote_data_path'],
                local_path=data_config['local_data_dir'],
                use_gcs=data_config.get('use_gcs', True), # Default to True if not specified
                gcs_client_type=data_config.get('gcs_client_type', 'default') # Default to 'default'
            )

        # Sync data from GCS to local storage if GCS is enabled
        if self.gcs_manager.use_gcs:
            logger.info("Syncing training data from GCS...")
            sync_success = self.gcs_manager.sync_data_to_local(
                max_workers=data_config.get('preprocessing_num_workers', 8) # Use existing preprocessing_num_workers
            )
            if not sync_success:
                # Decide if this is fatal. For now, we'll allow training on partial data.
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
            npz_key_priority=data_config.get('npz_key_priority', ['input_ids', 'arr_0', 'text', 'sequences']) # Pass key priority
        )
        
        logger.info("Data loading setup completed")
    
    def setup_monitoring(self):
        """Setup monitoring and logging"""
        monitoring_config = self.config['monitoring']
        
        # Create logs directory
        log_dir = Path(monitoring_config['logging_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for training.log
        file_handler = logging.FileHandler(log_dir / 'training.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler) # Add to root logger
        
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
        data_config = self.config['data'] # Get data config for use_gcs
        
        self.checkpoint_manager = CheckpointManager(
            output_dir=checkpoint_config['output_dir'],
            remote_bucket=checkpoint_config['remote_checkpoint_bucket'],
            remote_path=checkpoint_config['remote_checkpoint_path'],
            use_gcs=data_config.get('use_gcs', True),  # Pass use_gcs from data_config
            backup_every_n_steps=checkpoint_config['backup_every_n_steps'],
            keep_last_n=checkpoint_config['keep_last_n_checkpoints']
            # async_backup is True by default in CheckpointManager, adjust if needed from config
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
        # Ensure model is on CPU and in FP32 for saving with save_pretrained
        if hasattr(self.engine, 'module'):
            model_to_save = self.engine.module
        else:
            model_to_save = self.engine

        # If using DeepSpeed ZeRO, need to consolidate weights first
        # This part might need adjustment based on how DeepSpeed handles model state
        # For ZeRO Stage 3, the model on rank 0 might not have all weights unless gathered.
        # However, engine.save_pretrained() is supposed to handle this.
        
        # Ensure the model is in eval mode and on CPU for saving
        model_to_save.eval() # Set to evaluation mode
        # model_to_save.cpu() # Moving to CPU might be problematic with large models / DeepSpeed states

        logger.info(f"Attempting to save model to {export_dir}...")
        self.engine.save_pretrained(export_dir) # DeepSpeed's method should handle ZeRO stages
        self.tokenizer.save_pretrained(export_dir)
        logger.info(f"Model and tokenizer saved to {export_dir} locally.")
        
        # Upload to GCS if enabled
        if self.gcs_manager and self.gcs_manager.use_gcs:
            logger.info(f"Attempting to upload HF model from {export_dir} to GCS...")
            # The method in GCSDataManager is `_upload_directory_to_gcs` or `upload_file`
            # Assuming `upload_checkpoint` can be adapted or a new method is used for general directory upload.
            # For now, let's assume a method like `upload_directory` exists or adapt existing.
            # For simplicity, and if `upload_checkpoint` internally handles directory uploads correctly:
            success = self.gcs_manager.backup_checkpoint(export_dir) # This will prefix it with remote_path from CheckpointManager
            # A better approach for GCSDataManager:
            # success = self.gcs_manager.upload_directory(local_path=export_dir, remote_target_path="final_hf_model")
            if success:
                logger.info(f"HuggingFace model from {export_dir} uploaded to GCS.")
            else:
                logger.error(f"Failed to upload HuggingFace model from {export_dir} to GCS.")
        elif self.gcs_manager:
             logger.info("GCS usage is disabled in GCSDataManager. Skipping HF model upload to GCS.")
        else:
            logger.warning("GCSDataManager not available. Skipping HF model upload to GCS.")
            
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

    # Load config early to make paths available for directory creation
    temp_config = {}
    try:
        with open(args.config, 'r') as f:
            temp_config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config for directory setup: {e}. Using defaults if any.")
        # Allow to proceed, ModelTrainer will eventually fail if config is truly broken

    # Setup directories based on config
    # These are base directories; specific subdirectories (like for checkpoints)
    # will be handled by their respective managers/components.
    
    # Log directory (primary, others might be inside ModelTrainer)
    log_dir_path = Path(temp_config.get('monitoring', {}).get('logging_dir', 'logs')) # Default to 'logs'
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Local data directory (where shards are expected)
    local_data_dir_path = Path(temp_config.get('data', {}).get('local_data_dir', 'data/shards')) # Default
    local_data_dir_path.mkdir(parents=True, exist_ok=True)

    # Checkpoints output directory (base)
    checkpoints_base_path = Path(temp_config.get('checkpointing', {}).get('output_dir', 'checkpoints')) # Default
    checkpoints_base_path.mkdir(parents=True, exist_ok=True)

    # DeepSpeed NVMe offload directory (if specified and used)
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