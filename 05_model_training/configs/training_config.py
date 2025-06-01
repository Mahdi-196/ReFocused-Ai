"""
Training configuration for ReFocused-AI model
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Training configuration optimized for H100 GPUs"""
    
    # Model settings
    model_name: str = "refocused-ai-1.2b"
    
    # Data settings
    bucket_name: str = "refocused-ai"
    checkpoint_bucket_path: str = "Checkpoints"
    max_train_files: Optional[int] = None  # None for all files
    
    # Preprocessing optimization settings
    use_optimized_dataset: bool = True  # Use preprocessed cache
    preprocess_cache_dir: str = "./preprocessed_cache"
    max_test_steps: int = 100  # Limit steps for test runs
    enable_profiling: bool = False  # Enable performance profiling
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    warmup_steps: int = 2000
    max_steps: int = -1  # -1 for infinite
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # Effective batch size = 32
    
    # Optimization
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"
    
    # Precision and performance
    fp16: bool = False
    bf16: bool = True  # BF16 for H100
    tf32: bool = True  # Enable TF32 for better perf
    gradient_checkpointing: bool = True
    
    # Checkpointing
    save_steps: int = 1000
    save_total_limit: int = 5
    checkpoint_every_n_files: int = 5
    resume_from_checkpoint: Optional[str] = None
    
    # Logging and monitoring
    logging_steps: int = 10
    logging_first_step: bool = True
    report_to: str = "tensorboard"
    run_name: Optional[str] = None
    detailed_monitoring: bool = False  # Enable detailed performance monitoring
    
    # FSDP specific settings
    fsdp_min_num_params: int = 1e8  # Wrap layers > 100M params
    fsdp_transformer_layer_cls_to_wrap: str = "GPTNeoXLayer"
    
    # Memory optimization
    activation_checkpointing: bool = True
    
    # Evaluation
    eval_steps: int = 1000
    eval_strategy: str = "steps"
    
    # System
    dataloader_num_workers: int = 4
    preprocessing_num_workers: int = 4
    
    # Paths
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"


def get_test_config():
    """Configuration for test run with 25 files on 1 GPU"""
    config = TrainingConfig(
        max_train_files=25,
        per_device_train_batch_size=2,  # Smaller for testing
        gradient_accumulation_steps=4,   # Effective batch = 8
        save_steps=500,
        checkpoint_every_n_files=5,
        logging_steps=5,
        run_name="test_run_25_files",
        # Test-specific optimizations
        max_test_steps=100,  # Only 100 steps for quick testing
        use_optimized_dataset=True,
        enable_profiling=True,  # Enable profiling for test runs
        detailed_monitoring=True,  # Detailed monitoring for test runs
    )
    return config


def get_production_config():
    """Configuration for production training on full dataset"""
    config = TrainingConfig(
        max_train_files=None,  # Use all files
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch = 32
        save_steps=1000,
        checkpoint_every_n_files=5,
        logging_steps=10,
        run_name="production_run_full",
        # Production-specific settings
        max_test_steps=-1,  # No step limit for production
        use_optimized_dataset=True,
        enable_profiling=False,  # Disable profiling for production
        detailed_monitoring=False,  # Basic monitoring for production
    )
    return config 