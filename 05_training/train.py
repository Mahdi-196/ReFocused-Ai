#!/usr/bin/env python3
"""
ReFocused-AI GPT Training Script
Trains a ~1.2B parameter GPT model from scratch using DeepSpeed
"""

import sys
import os
import json
import argparse
import shutil
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

# ========== Version & Dependency Checks ==========
print("Checking dependencies...")

# 1. NumPy must be 1.x
try:
    import numpy as np
except ImportError:
    sys.exit("ERROR: numpy is not installed. Run `pip install numpy==1.24.0`")
if not np.__version__.startswith("1."):
    sys.exit(f"ERROR: Incompatible NumPy {np.__version__}. Use 1.24.0.")

# 2. Packaging
try:
    import packaging
except ImportError:
    sys.exit("ERROR: packaging missing. Run `pip install packaging==23.2`")

# 3. PyTorch version check
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
except ImportError:
    sys.exit("ERROR: PyTorch not found. Install torch==2.1.0+cu118 first.")
if not torch.__version__.startswith("2.1.0"):
    sys.exit(f"ERROR: PyTorch {torch.__version__} unsupported. Use 2.1.0.")

# 4. Flash-Attention (optional)
FLASH_ATTN_AVAILABLE = False
try:
    import flash_attn
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
    print("✓ Flash Attention available")
except ImportError:
    print("WARNING: flash-attn not found. Using standard attention.")

# 5. TensorBoard & W&B
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    sys.exit("ERROR: TensorBoard import failed. Install tensorboard==2.15.0")

WANDB_AVAILABLE = False
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("WARNING: wandb not installed. Proceeding without W&B logs.")

# 6. Transformers (for tokenizer only)
try:
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("ERROR: transformers not installed. Run `pip install transformers==4.35.2`")

# 7. DeepSpeed
try:
    import deepspeed
    from deepspeed import comm as dist
except ImportError:
    sys.exit("ERROR: DeepSpeed not installed. Run setup_env.sh first.")

# 8. Google Cloud Storage (optional)
GCS_AVAILABLE = False
try:
    from google.cloud import storage
    from google.api_core import retry
    GCS_AVAILABLE = True
except ImportError:
    print("INFO: google-cloud-storage not installed. Local storage will be used for checkpoints.")

# 9. Other utilities
try:
    from tqdm import tqdm
except ImportError:
    sys.exit("ERROR: tqdm not installed.")

print("✓ All dependencies verified\n")

# ========== Configuration ==========
MODEL_CONFIGS = {
    "125M": {
        "n_layers": 12,
        "n_heads": 12,
        "d_model": 768,
        "d_ff": 3072,
        "vocab_size": 50257,
        "max_seq_len": 2048,
        "dropout": 0.1,
    },
    "350M": {
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 1024,
        "d_ff": 4096,
        "vocab_size": 50257,
        "max_seq_len": 2048,
        "dropout": 0.1,
    },
    "760M": {
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 1536,
        "d_ff": 6144,
        "vocab_size": 50257,
        "max_seq_len": 2048,
        "dropout": 0.1,
    },
    "1.2B": {
        "n_layers": 24,
        "n_heads": 16,
        "d_model": 2048,
        "d_ff": 8192,
        "vocab_size": 50257,
        "max_seq_len": 2048,
        "dropout": 0.1,
    },
}

# ========== Model Definition ==========
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with optional Flash Attention"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        
        if FLASH_ATTN_AVAILABLE and mask is None:
            # Use Flash Attention if available
            q = q.transpose(1, 2)  # [B, H, S, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn_output = flash_attn_func(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
            attn_output = attn_output.transpose(1, 2).contiguous()
        else:
            # Standard attention
            q = q.transpose(1, 2)  # [B, H, S, D]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous()
        
        # Reshape and project
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(self.ln2(x))
        x = x + ffn_out
        
        return x


class GPTModel(nn.Module):
    """GPT Model from scratch"""
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config["vocab_size"], config["d_model"])
        self.position_embedding = nn.Embedding(config["max_seq_len"], config["d_model"])
        self.dropout = nn.Dropout(config["dropout"])
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config["d_model"],
                config["n_heads"],
                config["d_ff"],
                config["dropout"]
            )
            for _ in range(config["n_layers"])
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config["d_model"])
        self.lm_head = nn.Linear(config["d_model"], config["vocab_size"], bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> dict:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position ids
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = self.dropout(token_embeds + position_embeds)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
        
        # Pass through transformer blocks
        for block in self.blocks:
            if deepspeed.checkpointing.is_configured():
                hidden_states = deepspeed.checkpointing.checkpoint(block, hidden_states, mask)
            else:
                hidden_states = block(hidden_states, mask)
        
        # Final layer norm
        hidden_states = self.ln_f(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {
            "loss": loss,
            "logits": logits,
        }
    
    def get_num_params(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())


# ========== Dataset ==========
class TokenDataset(Dataset):
    """Dataset for loading tokenized .npz files"""
    
    def __init__(self, file_paths: List[str], max_seq_len: int = 2048):
        self.file_paths = sorted(file_paths)
        self.max_seq_len = max_seq_len
        self.current_file_idx = 0
        self.current_data = None
        self.current_position = 0
        
        # Load first file
        if self.file_paths:
            self._load_file(0)
    
    def _load_file(self, idx: int):
        """Load a specific .npz file"""
        if idx < len(self.file_paths):
            data = np.load(self.file_paths[idx])
            self.current_data = data['tokens']
            self.current_position = 0
            self.current_file_idx = idx
    
    def __len__(self):
        # Estimate based on file size
        return len(self.file_paths) * 10000  # Rough estimate
    
    def __getitem__(self, idx):
        # Simple sequential reading
        if self.current_data is None:
            return torch.zeros(self.max_seq_len, dtype=torch.long)
        
        # Check if we need to move to next file
        if self.current_position + self.max_seq_len + 1 >= len(self.current_data):
            self.current_file_idx = (self.current_file_idx + 1) % len(self.file_paths)
            self._load_file(self.current_file_idx)
        
        # Extract sequence
        start = self.current_position
        end = start + self.max_seq_len + 1
        
        if end <= len(self.current_data):
            tokens = self.current_data[start:end]
        else:
            # Pad if necessary
            tokens = np.concatenate([
                self.current_data[start:],
                np.zeros(end - len(self.current_data), dtype=np.int64)
            ])
        
        self.current_position += self.max_seq_len
        
        return torch.tensor(tokens, dtype=torch.long)


# ========== Training Class ==========
class GPTTrainer:
    """Main trainer class"""
    
    def __init__(self, args):
        self.args = args
        self.setup_logging()
        
        # Initialize distributed training
        deepspeed.init_distributed()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = args.local_rank
        self.is_main_process = self.rank == 0
        
        # Setup device
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        # Initialize components
        self.setup_model()
        self.setup_tokenizer()
        self.setup_data()
        self.setup_logging_tools()
        
        # Training state
        self.global_step = 0
        self.files_processed = 0
        self.tokens_seen = 0
        self.best_loss = float('inf')
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = logging.INFO if self.args.local_rank in [-1, 0] else logging.WARNING
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=log_level,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.args.log_dir / "training.log")
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_model(self):
        """Initialize model"""
        config = MODEL_CONFIGS[self.args.model_size]
        self.model = GPTModel(config)
        
        if self.is_main_process:
            num_params = self.model.get_num_params()
            self.logger.info(f"Model initialized with {num_params:,} parameters")
            self.logger.info(f"Model config: {config}")
    
    def setup_tokenizer(self):
        """Load tokenizer"""
        try:
            # First try loading from tokenizer_1B folder
            tokenizer_path = Path("../tokenizer_1B")
            if tokenizer_path.exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
            else:
                # Fallback to model name
                self.tokenizer = AutoTokenizer.from_pretrained("tokenizer_1B", use_fast=True)
                
            if self.is_main_process:
                self.logger.info(f"Loaded tokenizer with vocab size: {len(self.tokenizer)}")
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            sys.exit(1)
    
    def setup_data(self):
        """Setup data loaders"""
        # Get data files
        data_files = list(Path(self.args.data_dir).glob("*.npz"))
        if self.args.max_files:
            data_files = data_files[:self.args.max_files]
        
        if not data_files:
            self.logger.error(f"No .npz files found in {self.args.data_dir}")
            sys.exit(1)
        
        if self.is_main_process:
            self.logger.info(f"Found {len(data_files)} data files")
        
        # Create dataset
        dataset = TokenDataset(
            [str(f) for f in data_files],
            max_seq_len=MODEL_CONFIGS[self.args.model_size]["max_seq_len"]
        )
        
        # Create sampler and dataloader
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        self.train_dataloader = DataLoader(
            dataset,
            batch_size=self.args.micro_batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        self.data_files = data_files
    
    def setup_logging_tools(self):
        """Setup TensorBoard and W&B"""
        if self.is_main_process:
            # TensorBoard
            self.tb_writer = SummaryWriter(self.args.log_dir / "tensorboard")
            
            # Weights & Biases
            if WANDB_AVAILABLE and self.args.wandb_project:
                wandb.init(
                    project=self.args.wandb_project,
                    name=self.args.wandb_run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config=vars(self.args)
                )
                self.logger.info(f"✓ W&B initialized: {self.args.wandb_project}")
    
    def save_checkpoint(self, step: int, files_processed: int):
        """Save checkpoint locally"""
        if not self.is_main_process:
            return
        
        try:
            ckpt_name = f"ckpt_step{step}_files{files_processed}"
            ckpt_dir = self.args.checkpoint_dir / ckpt_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model with DeepSpeed
            self.model_engine.save_checkpoint(str(ckpt_dir))
            
            # Save metadata
            metadata = {
                "step": step,
                "files_processed": files_processed,
                "tokens_seen": self.tokens_seen,
                "timestamp": datetime.now().isoformat(),
                "best_loss": self.best_loss,
                "config": MODEL_CONFIGS[self.args.model_size]
            }
            
            with open(ckpt_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"✓ Checkpoint saved to: {ckpt_dir}")
                
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
    
    def train_step(self, batch):
        """Single training step"""
        # Move batch to device
        input_ids = batch[:, :-1].to(self.device)
        labels = batch[:, 1:].to(self.device)
        
        # Forward pass
        outputs = self.model_engine(input_ids, labels=labels)
        loss = outputs["loss"]
        
        # Backward pass (handled by DeepSpeed)
        self.model_engine.backward(loss)
        self.model_engine.step()
        
        # Update metrics
        self.tokens_seen += input_ids.numel() * self.world_size
        
        return loss.item()
    
    def log_metrics(self, loss: float, step: int):
        """Log metrics to various platforms"""
        if not self.is_main_process:
            return
        
        # Calculate metrics
        if hasattr(self, 'last_log_time'):
            time_delta = time.time() - self.last_log_time
            tokens_delta = self.tokens_seen - self.last_tokens_seen
            tokens_per_sec = tokens_delta / time_delta if time_delta > 0 else 0
        else:
            tokens_per_sec = 0
        
        self.last_log_time = time.time()
        self.last_tokens_seen = self.tokens_seen
        
        # Get learning rate
        lr = self.model_engine.get_lr()[0]
        
        # TensorBoard
        self.tb_writer.add_scalar("train/loss", loss, step)
        self.tb_writer.add_scalar("train/learning_rate", lr, step)
        self.tb_writer.add_scalar("train/tokens_per_second", tokens_per_sec, step)
        self.tb_writer.add_scalar("train/tokens_seen", self.tokens_seen, step)
        
        # Weights & Biases
        if WANDB_AVAILABLE and hasattr(self, 'wandb_run'):
            wandb.log({
                "loss": loss,
                "learning_rate": lr,
                "tokens_per_second": tokens_per_sec,
                "tokens_seen": self.tokens_seen,
                "files_processed": self.files_processed,
                "step": step
            })
        
        # Console log
        self.logger.info(
            f"Step {step} | Loss: {loss:.4f} | LR: {lr:.2e} | "
            f"Tokens/s: {tokens_per_sec:.0f} | Files: {self.files_processed}"
        )
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        # Initialize DeepSpeed
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=self.args,
            model=self.model,
            model_parameters=self.model.parameters(),
            config=self.args.deepspeed_config
        )
        
        # Training loop
        self.model_engine.train()
        train_iter = iter(self.train_dataloader)
        
        with tqdm(total=len(self.data_files), desc="Files processed", disable=not self.is_main_process) as pbar:
            while self.files_processed < len(self.data_files):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_dataloader)
                    batch = next(train_iter)
                    self.files_processed += 1
                    pbar.update(1)
                
                # Training step
                loss = self.train_step(batch)
                self.global_step += 1
                
                # Update best loss
                if loss < self.best_loss:
                    self.best_loss = loss
                
                # Logging
                if self.global_step % self.args.log_interval == 0:
                    self.log_metrics(loss, self.global_step)
                
                # Checkpointing
                if self.files_processed % self.args.checkpoint_interval == 0 and self.files_processed > 0:
                    self.save_checkpoint(self.global_step, self.files_processed)
                
                # Early stopping for test mode
                if self.args.mode == "test" and self.files_processed >= 5:
                    break
        
        # Final checkpoint
        self.save_checkpoint(self.global_step, self.files_processed)
        
        if self.is_main_process:
            self.logger.info("✅ Training completed!")
            self.logger.info(f"Total steps: {self.global_step}")
            self.logger.info(f"Total tokens: {self.tokens_seen:,}")
            self.logger.info(f"Files processed: {self.files_processed}")


# ========== Main ==========
def main():
    parser = argparse.ArgumentParser(description="Train GPT model from scratch")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="1.2B",
                       choices=["125M", "350M", "760M", "1.2B"],
                       help="Model size configuration")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing tokenized .npz files")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process")
    
    # Training arguments
    parser.add_argument("--mode", type=str, choices=["test", "production"],
                       required=True, help="Training mode")
    parser.add_argument("--micro_batch_size", type=int, default=8,
                       help="Micro batch size per GPU")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"),
                       help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                       help="Save checkpoint every N files")
    
    # Logging
    parser.add_argument("--log_dir", type=Path, default=Path("logs"),
                       help="Directory for logs")
    parser.add_argument("--log_interval", type=int, default=10,
                       help="Log metrics every N steps")
    parser.add_argument("--eval_interval", type=int, default=500,
                       help="Evaluate every N steps")
    
    # W&B
    parser.add_argument("--wandb_project", type=str, default=None,
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    
    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1,
                       help="Local rank for distributed training")
    parser.add_argument("--deepspeed_config", type=str, default=None,
                       help="DeepSpeed configuration file")
    
    args = parser.parse_args()
    
    # Create directories
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load DeepSpeed config if provided
    if args.deepspeed_config and Path(args.deepspeed_config).exists():
        with open(args.deepspeed_config, 'r') as f:
            args.deepspeed_config = json.load(f)
    
    # Initialize trainer and start training
    trainer = GPTTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main() 