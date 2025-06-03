"""
Training configurations for ReFocused-AI model
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Base training configuration"""
    
    # Model settings
    model_name: str = "ReFocused-AI-1.2B"
    
    # Data settings
    bucket_name: str = "refocused-ai"
    tokenized_file_pattern: str = "tokenized_cleaned_*.npz"
    sequence_length: int = 1024
    max_files: int = 5  # -1 for all files
    use_streaming: bool = False  # Use streaming dataset for very large datasets
    
    # Training hyperparameters
    learning_rate: float = 2e-4
    warmup_steps: int = 10
    max_steps: int = 100
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Data loading
    dataloader_num_workers: int = 4  # Number of parallel data loading workers
    
    # Mixed precision
    bf16: bool = True  # Use BF16 if GPU supports it
    
    # Checkpointing
    save_steps: int = 50
    logging_steps: int = 10
    
    # Paths
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    cache_dir: str = "./cache"
    preprocess_cache_dir: str = "./preprocessed_cache"
    
    # Checkpointing to GCS
    checkpoint_bucket_path: str = "Checkpoints"


def get_training_config(config_type: str = "test") -> TrainingConfig:
    """Get training configuration by type"""
    
    if config_type == "test":
        return TrainingConfig(
            max_files=5,  # Small number for testing
            max_steps=100,      # Quick test
            per_device_train_batch_size=1,
            save_steps=50,      # Save every 50 steps
            logging_steps=10,   # Log every 10 steps
        )
    
    elif config_type == "production":
        return TrainingConfig(
            max_files=-1,  # All files
            max_steps=10000,       # Full training
            per_device_train_batch_size=4,
            save_steps=500,        # Save every 500 steps
            logging_steps=100,     # Log every 100 steps
            gradient_accumulation_steps=4,
        )
    
    else:
        raise ValueError(f"Unknown config type: {config_type}") 