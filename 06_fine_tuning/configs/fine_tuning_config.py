#!/usr/bin/env python3
"""
Fine-tuning configurations for different tasks
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class FineTuningConfig:
    """Base configuration for fine-tuning"""
    # Model settings
    model_name: str = "refocused-ai-1.2b"
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Batch sizes
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    
    # Data settings
    max_length: int = 1024
    num_workers: int = 4
    
    # Scheduler
    scheduler_type: str = "cosine"
    
    # Evaluation and saving
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = False
    
    # Gradient checkpointing
    gradient_checkpointing: bool = False
    
    # Task-specific settings
    task_type: str = "general"
    
    # Output settings
    output_dir: str = "./fine_tuned_models"
    logging_dir: str = "./logs"
    
    # Hub settings
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None
    hub_private_repo: bool = True


# Task-specific configurations
TASK_CONFIGS = {
    "chat": FineTuningConfig(
        task_type="chat",
        learning_rate=2e-5,
        num_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=200,
        eval_steps=100,
        save_steps=500,
        max_length=2048,  # Longer context for conversations
        scheduler_type="cosine_with_restarts",
    ),
    
    "code": FineTuningConfig(
        task_type="code",
        learning_rate=1e-5,
        num_epochs=5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        eval_steps=200,
        save_steps=1000,
        max_length=2048,
        gradient_checkpointing=True,  # Save memory for longer sequences
    ),
    
    "instruct": FineTuningConfig(
        task_type="instruct",
        learning_rate=3e-5,
        num_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        eval_steps=50,
        save_steps=200,
        max_length=1024,
    ),
    
    "domain": FineTuningConfig(
        task_type="domain",
        learning_rate=1e-5,
        num_epochs=10,  # More epochs for domain adaptation
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=1000,
        eval_steps=500,
        save_steps=2000,
        max_length=1024,
        weight_decay=0.05,  # Higher weight decay for domain adaptation
    ),
    
    "custom": FineTuningConfig(
        task_type="custom",
        learning_rate=2e-5,
        num_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        eval_steps=100,
        save_steps=500,
        max_length=1024,
    ),
}

# Preset configurations for different scenarios
PRESET_CONFIGS = {
    "quick_test": FineTuningConfig(
        learning_rate=5e-5,
        num_epochs=1,
        max_steps=100,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        eval_steps=20,
        save_steps=50,
        logging_steps=5,
        warmup_steps=10,
    ),
    
    "low_resource": FineTuningConfig(
        learning_rate=1e-5,
        num_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        fp16=True,  # Use mixed precision
        eval_steps=200,
        save_steps=500,
        max_length=512,  # Shorter sequences
    ),
    
    "high_quality": FineTuningConfig(
        learning_rate=5e-6,
        num_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        warmup_steps=2000,
        weight_decay=0.1,
        eval_steps=100,
        save_steps=500,
        save_total_limit=10,  # Keep more checkpoints
        load_best_model_at_end=True,
    ),
    
    "production": FineTuningConfig(
        learning_rate=2e-5,
        num_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=500,
        eval_steps=200,
        save_steps=1000,
        logging_steps=20,
        bf16=True,  # Use bfloat16 for stability
        gradient_checkpointing=False,
        save_total_limit=5,
    ),
}


def get_fine_tuning_config(preset: str = "default", task: str = "custom") -> FineTuningConfig:
    """Get fine-tuning configuration"""
    
    # Start with task-specific config
    if task in TASK_CONFIGS:
        config = TASK_CONFIGS[task]
    else:
        config = TASK_CONFIGS["custom"]
    
    # Override with preset if specified
    if preset in PRESET_CONFIGS:
        preset_config = PRESET_CONFIGS[preset]
        # Update config with preset values
        for key, value in preset_config.__dict__.items():
            setattr(config, key, value)
    
    return config


def get_lora_config(task: str = "custom") -> dict:
    """Get LoRA configuration for specific task"""
    
    lora_configs = {
        "chat": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "code": {
            "r": 32,  # Higher rank for code
            "lora_alpha": 64,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "dense"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "instruct": {
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "domain": {
            "r": 64,  # Very high rank for domain adaptation
            "lora_alpha": 128,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "dense", "dense_h_to_4h", "dense_4h_to_h"],
            "lora_dropout": 0.1,
            "bias": "lora_only",
            "task_type": "CAUSAL_LM",
        },
        "custom": {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
    }
    
    return lora_configs.get(task, lora_configs["custom"])


if __name__ == "__main__":
    # Print available configurations
    print("📋 Available Fine-tuning Configurations:")
    print("\n🎯 Task-specific configs:")
    for task in TASK_CONFIGS:
        print(f"  - {task}")
    
    print("\n⚙️  Preset configs:")
    for preset in PRESET_CONFIGS:
        print(f"  - {preset}")
    
    # Example usage
    print("\n📊 Example configuration (chat task, production preset):")
    config = get_fine_tuning_config(preset="production", task="chat")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Max length: {config.max_length}")
    print(f"  Mixed precision: bf16={config.bf16}, fp16={config.fp16}") 