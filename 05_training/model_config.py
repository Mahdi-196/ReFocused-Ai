"""
Configuration for 1B parameter GPT model with efficiency optimizations
Based on latest research including HybridNorm for stable training
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Configuration for efficient 1B parameter model"""
    
    # Model architecture
    vocab_size: int = 50000  # From your tokenizer
    n_embd: int = 1536  # Hidden dimension (optimized for 1B params)
    n_layer: int = 24  # Number of transformer layers
    n_head: int = 24  # Number of attention heads
    n_kv_head: Optional[int] = 8  # Grouped-query attention for efficiency
    
    # Sequence length
    max_seq_len: int = 2048  # Context window
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Normalization - using HybridNorm for stability
    norm_type: str = "hybrid"  # "pre", "post", or "hybrid"
    norm_eps: float = 1e-5
    
    # Activation function
    activation: str = "swiglu"  # More efficient than standard GELU
    
    # Model precision
    use_mixed_precision: bool = True
    
    # Flash attention for memory efficiency
    use_flash_attention: bool = True
    
    # Rotary positional embeddings (RoPE)
    use_rope: bool = True
    rope_theta: float = 10000.0
    
    # Initialization
    init_std: float = 0.02
    
    @property
    def intermediate_size(self) -> int:
        """FFN intermediate dimension"""
        # SwiGLU uses 8/3 * n_embd for efficiency
        return int(8 * self.n_embd / 3)
    
    @property
    def n_params(self) -> int:
        """Approximate parameter count"""
        # Embedding parameters
        embedding_params = self.vocab_size * self.n_embd
        
        # Attention parameters per layer
        attention_params = 4 * self.n_embd * self.n_embd  # Q, K, V, O projections
        
        # FFN parameters per layer (with SwiGLU)
        ffn_params = 3 * self.n_embd * self.intermediate_size  # gate, up, down
        
        # Layer norm parameters
        ln_params = 2 * self.n_embd  # 2 layer norms per layer
        
        # Total
        params_per_layer = attention_params + ffn_params + ln_params
        total_params = embedding_params + (params_per_layer * self.n_layer)
        
        return total_params


@dataclass
class TrainingConfig:
    """Training configuration for efficient 1B model training"""
    
    # Optimizer settings
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    
    # Batch size settings
    micro_batch_size: int = 8  # Per GPU batch size
    gradient_accumulation_steps: int = 16  # Effective batch size = 128
    
    # Training duration
    max_steps: int = 100000  # Adjust based on data
    eval_interval: int = 500
    save_interval: int = 5000  # Save every 5 files as requested
    log_interval: int = 10
    
    # Gradient clipping
    grad_clip: float = 1.0
    
    # Data loading
    num_workers: int = 4
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    
    # Distributed training
    distributed: bool = True
    ddp_backend: str = "nccl"
    
    # Memory optimization
    gradient_checkpointing: bool = True
    zero_stage: int = 2  # DeepSpeed ZeRO stage
    
    # Monitoring
    use_wandb: bool = True
    wandb_project: str = "refocused-ai-1b"
    wandb_run_name: Optional[str] = None
    
    # GCS settings
    gcs_bucket: str = "refocused-ai"
    gcs_data_prefix: str = ""  # tokenized data location
    gcs_checkpoint_prefix: str = "Checkpoints"
    
    # Data settings
    train_files_per_checkpoint: int = 5  # Checkpoint every 5 files
    test_num_files: int = 25  # For test run
    
    # Hardware
    device: str = "cuda"
    compile_model: bool = True  # PyTorch 2.0 compile for speed


# Preset configurations
def get_test_config():
    """Configuration for test run with 25 files"""
    model_config = ModelConfig()
    training_config = TrainingConfig()
    training_config.max_steps = 1000  # Shorter for testing
    training_config.eval_interval = 100
    training_config.save_interval = 100
    training_config.distributed = False  # Single GPU for test
    return model_config, training_config


def get_production_config():
    """Configuration for full training run"""
    model_config = ModelConfig()
    training_config = TrainingConfig()
    return model_config, training_config 