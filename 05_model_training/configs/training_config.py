"""
Training configurations for ReFocused-AI model
"""

from dataclasses import dataclass
import torch.backends.cudnn as cudnn

# Enable cuDNN benchmark for performance optimization
cudnn.benchmark = True


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
    max_steps: int = 11450
    per_device_train_batch_size: int = 2  # Increased from 1 to 2
    gradient_accumulation_steps: int = 4  # Increased for effective larger batch size
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    
    # Data loading optimization
    dataloader_num_workers: int = 0  # Use main thread only (avoid multiprocessing overhead)
    pin_memory: bool = True  # Pin memory for faster GPU transfers
    drop_last: bool = True  # Drop incomplete batches for consistent training
    prefetch_factor: int = 2  # Number of batches to prefetch per worker
    
    # Mixed precision
    bf16: bool = True  # Use BF16 if GPU supports it
    fp16: bool = False  # Alternative to bf16 for older GPUs
    
    # Performance optimizations
    compile_model: bool = True  # Use torch.compile for PyTorch 2.0+ (ESSENTIAL for H100 performance)
    use_flash_attention: bool = False  # Use flash attention if available
    
    # Checkpointing optimization
    save_steps: int = 1145  # Save every 1145 steps (every ~10% of training)
    logging_steps: int = 115  # Log every 115 steps (every ~1% of training)
    checkpoint_compression: bool = True  # Compress checkpoints to save space
    background_upload: bool = True  # Upload checkpoints in background
    
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
            max_steps=1000,     # Increased from 100 to 1000 for more comprehensive training
            per_device_train_batch_size=2,  # Increased from 1
            gradient_accumulation_steps=2,  # Effective batch size = 2 * 2 = 4
            save_steps=200,     # Save every 200 steps (optimized frequency)
            logging_steps=25,   # Log every 25 steps
            dataloader_num_workers=2,  # Conservative for test
        )
    
    elif config_type == "production":
        return TrainingConfig(
            max_files=10,  # All files
            max_steps=25000,       # Optimal for 51B token dataset (0.016 epochs)
            per_device_train_batch_size=4,  # Optimal for multi GPU setup
            gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 * 2 GPUs = 32k tokens/step
            save_steps=1250,       # Save every 1250 steps (every ~5% of training, 20 checkpoints)
            logging_steps=250,     # Log every 250 steps (every ~1% of training, 100 logs)
            warmup_steps=500,      # 2% of training for gradual learning rate ramp-up
            dataloader_num_workers=4,  # Full parallelism
        )
    
    elif config_type == "production_8gpu":
        return TrainingConfig(
            max_files=-1,  # All files
            max_steps=590625,      # 3 full epochs through 51B token dataset
            per_device_train_batch_size=8,  # Higher batch size per GPU (8 GPUs can handle it)
            gradient_accumulation_steps=4,  # Effective batch size = 8 * 4 * 8 GPUs = 256k tokens/step
            save_steps=20000,      # Save every 20000 steps (every ~3.4% of training, ~30 checkpoints)
            logging_steps=3000,    # Log every 3000 steps (every ~0.5% of training, ~200 logs)
            warmup_steps=11812,    # 2% of training for gradual learning rate ramp-up
            dataloader_num_workers=0,  # Use main thread to prevent I/O bottleneck
            learning_rate=3e-4,    # Slightly higher LR for larger effective batch size
        )
    
    else:
        raise ValueError(f"Unknown config type: {config_type}") 
    