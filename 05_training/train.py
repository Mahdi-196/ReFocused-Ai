"""
Main training script for 1B parameter model with efficiency optimizations
"""

import os
import sys
import json
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import deepspeed
from transformers import AutoTokenizer
import wandb
from google.cloud import storage
from pathlib import Path
import logging
from datetime import datetime
import shutil

from model import GPTModel
from model_config import ModelConfig, TrainingConfig, get_test_config, get_production_config
from data_loader import create_dataloaders, estimate_dataset_size
from optimizer import create_optimizer_and_scheduler


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    """Main trainer class with all optimizations"""
    
    def __init__(self, model_config: ModelConfig, training_config: TrainingConfig):
        self.model_config = model_config
        self.training_config = training_config
        
        # Setup device and distributed training
        self.setup_distributed()
        
        # Initialize model
        self.model = GPTModel(model_config)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("../tokenizer_1B")
        
        # Setup training components
        self.setup_training()
        
        # Initialize tracking
        self.global_step = 0
        self.files_processed = 0
        self.start_time = time.time()
        
        # Setup logging and checkpointing
        if self.is_main_process:
            self.setup_logging()
            self.setup_gcs_client()
    
    def setup_distributed(self):
        """Setup distributed training"""
        if self.training_config.distributed:
            if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
                self.rank = int(os.environ['RANK'])
                self.world_size = int(os.environ['WORLD_SIZE'])
                self.local_rank = int(os.environ['LOCAL_RANK'])
            else:
                # Single GPU mode
                self.rank = 0
                self.world_size = 1
                self.local_rank = 0
            
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
            
            if self.world_size > 1:
                dist.init_process_group(backend=self.training_config.ddp_backend)
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.is_main_process = self.rank == 0
    
    def setup_training(self):
        """Setup optimizer, scheduler, and DeepSpeed"""
        # Create optimizer and scheduler
        self.optimizer, self.scheduler = create_optimizer_and_scheduler(
            self.model,
            self.training_config
        )
        
        # DeepSpeed configuration
        ds_config = {
            "train_batch_size": self.training_config.micro_batch_size * self.training_config.gradient_accumulation_steps * self.world_size,
            "train_micro_batch_size_per_gpu": self.training_config.micro_batch_size,
            "gradient_accumulation_steps": self.training_config.gradient_accumulation_steps,
            "gradient_clipping": self.training_config.grad_clip,
            "fp16": {
                "enabled": self.model_config.use_mixed_precision,
                "loss_scale": 0,
                "loss_scale_window": 1000,
                "initial_scale_power": 16,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "zero_optimization": {
                "stage": self.training_config.zero_stage,
                "contiguous_gradients": True,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e7,
                "allgather_bucket_size": 5e7
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
                "profile": False
            }
        }
        
        # Initialize DeepSpeed
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=self.model,
            optimizer=self.optimizer,
            config=ds_config,
            lr_scheduler=self.scheduler,
            dist_init_required=False
        )
        
        # Compile model if using PyTorch 2.0+
        if self.training_config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
    
    def setup_logging(self):
        """Setup logging and monitoring"""
        # Create directories
        self.log_dir = Path(f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir / "tensorboard")
        
        # Weights & Biases
        if self.training_config.use_wandb:
            wandb.init(
                project=self.training_config.wandb_project,
                name=self.training_config.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model_config": self.model_config.__dict__,
                    "training_config": self.training_config.__dict__
                }
            )
    
    def setup_gcs_client(self):
        """Setup Google Cloud Storage client"""
        self.gcs_client = storage.Client()
        self.gcs_bucket = self.gcs_client.bucket(self.training_config.gcs_bucket)
    
    def save_checkpoint(self, step: int, files_processed: int):
        """Save checkpoint to disk and upload to GCS"""
        if not self.is_main_process:
            return
        
        checkpoint_path = self.log_dir / f"checkpoint_step_{step}_files_{files_processed}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model state
        self.model.save_checkpoint(str(checkpoint_path))
        
        # Save training state
        training_state = {
            "global_step": self.global_step,
            "files_processed": self.files_processed,
            "model_config": self.model_config.__dict__,
            "training_config": self.training_config.__dict__,
        }
        
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(training_state, f, indent=2)
        
        # Upload to GCS
        logger.info(f"Uploading checkpoint to GCS...")
        gcs_checkpoint_path = f"{self.training_config.gcs_checkpoint_prefix}/checkpoint_step_{step}_files_{files_processed}"
        
        for file_path in checkpoint_path.rglob("*"):
            if file_path.is_file():
                blob_name = f"{gcs_checkpoint_path}/{file_path.relative_to(checkpoint_path)}"
                blob = self.gcs_bucket.blob(blob_name)
                blob.upload_from_filename(str(file_path))
        
        logger.info(f"Checkpoint saved to GCS: {gcs_checkpoint_path}")
        
        # Clean up local checkpoint to save space
        shutil.rmtree(checkpoint_path)
    
    def log_metrics(self, metrics: dict, step: int):
        """Log metrics to all tracking systems"""
        if not self.is_main_process:
            return
        
        # Add computed metrics
        elapsed_time = time.time() - self.start_time
        metrics['samples_per_second'] = step * self.training_config.micro_batch_size / elapsed_time
        metrics['files_processed'] = self.files_processed
        
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        
        # Weights & Biases
        if self.training_config.use_wandb:
            wandb.log(metrics, step=step)
        
        # Console
        logger.info(f"Step {step}: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()]))
    
    def train_step(self, batch: dict) -> dict:
        """Single training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"].to(self.device),
            attention_mask=batch["attention_mask"].to(self.device),
            labels=batch["labels"].to(self.device)
        )
        
        loss = outputs["loss"]
        
        # Backward pass (handled by DeepSpeed)
        self.model.backward(loss)
        self.model.step()
        
        # Get current learning rate
        lr = self.scheduler.get_last_lr()[0]
        
        return {
            "loss": loss.item(),
            "learning_rate": lr,
            "grad_norm": self.model.get_global_grad_norm()
        }
    
    def train(self):
        """Main training loop"""
        # Create data loader
        dataloader = create_dataloaders(
            bucket_name=self.training_config.gcs_bucket,
            prefix=self.training_config.gcs_data_prefix,
            max_seq_len=self.model_config.max_seq_len,
            batch_size=self.training_config.micro_batch_size,
            num_files=self.training_config.test_num_files if hasattr(self.training_config, 'test_num_files') else None,
            num_workers=self.training_config.num_workers
        )
        
        # Training loop
        self.model.train()
        running_loss = 0.0
        last_file_idx = -1
        
        for step, batch in enumerate(dataloader):
            # Track file progress
            current_file_idx = batch["file_indices"][0].item()
            if current_file_idx != last_file_idx:
                self.files_processed += 1
                last_file_idx = current_file_idx
                
                # Check if we need to save checkpoint
                if self.files_processed % self.training_config.train_files_per_checkpoint == 0:
                    self.save_checkpoint(self.global_step, self.files_processed)
            
            # Training step
            metrics = self.train_step(batch)
            running_loss += metrics["loss"]
            
            # Logging
            if self.global_step % self.training_config.log_interval == 0:
                avg_loss = running_loss / self.training_config.log_interval
                self.log_metrics({
                    "train/loss": avg_loss,
                    "train/learning_rate": metrics["learning_rate"],
                    "train/grad_norm": metrics["grad_norm"]
                }, self.global_step)
                running_loss = 0.0
            
            # Increment global step
            self.global_step += 1
            
            # Check if we've reached max steps
            if self.global_step >= self.training_config.max_steps:
                logger.info(f"Reached max steps ({self.training_config.max_steps})")
                break
        
        # Final checkpoint
        self.save_checkpoint(self.global_step, self.files_processed)
        
        # Cleanup
        if self.is_main_process:
            self.writer.close()
            if self.training_config.use_wandb:
                wandb.finish()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "production"], default="test",
                        help="Training mode: test (25 files) or production (full)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Get configuration
    if args.mode == "test":
        model_config, training_config = get_test_config()
    else:
        model_config, training_config = get_production_config()
    
    if args.resume:
        training_config.resume_from_checkpoint = args.resume
    
    # Log configuration
    logger.info(f"Model parameters: {model_config.n_params / 1e9:.2f}B")
    logger.info(f"Training mode: {args.mode}")
    
    # Estimate dataset size
    logger.info("Estimating dataset size...")
    dataset_info = estimate_dataset_size(
        training_config.gcs_bucket,
        training_config.gcs_data_prefix
    )
    logger.info(f"Dataset info: {dataset_info}")
    
    # Create trainer and start training
    trainer = Trainer(model_config, training_config)
    trainer.train()


if __name__ == "__main__":
    main() 