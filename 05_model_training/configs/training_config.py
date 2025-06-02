"""
Training configurations for ReFocused-AI model
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Base training configuration"""
    
    # Model settings
    model_name: str = "refocused-ai-1.2b"
    
    # Data settings
    bucket_name: str = "refocused-ai"
    checkpoint_bucket_path: str = "Checkpoints"
    max_train_files: int = None  # None for all files
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    warmup_steps: int = 1000
    max_steps: int = 1000
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Mixed precision
    bf16: bool = True  # Use BF16 if GPU supports it
    
    # Checkpointing
    save_steps: int = 100
    logging_steps: int = 10
    
    # Paths
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"


def get_training_config(config_type: str = "test") -> TrainingConfig:
    """Get training configuration by type"""
    
    if config_type == "test":
        return TrainingConfig(
            max_train_files=5,  # Small number for testing
            max_steps=100,      # Quick test
            per_device_train_batch_size=1,
            save_steps=50,      # Save more frequently
            logging_steps=5,
        )
    
    elif config_type == "production":
        return TrainingConfig(
            max_train_files=None,  # All files
            max_steps=10000,       # Full training
            per_device_train_batch_size=4,
            save_steps=500,
            logging_steps=20,
        )
    
    else:
        raise ValueError(f"Unknown config type: {config_type}") 