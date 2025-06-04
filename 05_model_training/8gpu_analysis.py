#!/usr/bin/env python3
"""8 GPU Configuration Analysis"""

from configs import get_training_config

def analyze_8gpu_config():
    print("🚀 8 GPU PRODUCTION CONFIGURATION ANALYSIS")
    print("=" * 50)
    
    config = get_training_config('production_8gpu')
    
    # Calculate metrics
    gpus = 8
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps * gpus
    tokens_per_step = effective_batch * config.sequence_length
    total_training_tokens = config.max_steps * tokens_per_step
    dataset_tokens = 51_000_000_000  # 51B tokens from analysis
    epochs = total_training_tokens / dataset_tokens
    estimated_hours = config.max_steps / 20.0 / 3600  # 20 steps/sec for 8 GPUs
    checkpoints = config.max_steps // config.save_steps
    
    print("⚙️  CONFIGURATION PARAMETERS:")
    print(f"   Max steps: {config.max_steps:,}")
    print(f"   Save steps: {config.save_steps} (every {100/checkpoints:.1f}% of training)")
    print(f"   Logging steps: {config.logging_steps}")
    print(f"   Per GPU batch size: {config.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Warmup steps: {config.warmup_steps}")
    
    print(f"\n📊 CALCULATED METRICS:")
    print(f"   Effective batch size: {effective_batch} sequences")
    print(f"   Tokens per step: {tokens_per_step:,}")
    print(f"   Total training tokens: {total_training_tokens:,}")
    print(f"   Dataset utilization: {epochs:.3f} epochs ({epochs*100:.1f}% of your data)")
    print(f"   Estimated training time: {estimated_hours:.1f} hours")
    print(f"   Number of checkpoints: {checkpoints}")
    
    print(f"\n🎯 WHY THESE NUMBERS:")
    print(f"   • 75,000 steps with 8 GPUs = optimal training depth")
    print(f"   • See {epochs*100:.1f}% of your 51B tokens (perfect for large datasets)")
    print(f"   • {estimated_hours:.1f} hours is very reasonable for production training")
    print(f"   • {tokens_per_step:,} tokens/step = aggressive but efficient training")
    print(f"   • 30 checkpoints = excellent recovery options")
    
    print(f"\n⚡ PERFORMANCE COMPARISON:")
    configs = [
        ("2 GPU", "production", 2, 5.0),
        ("8 GPU", "production_8gpu", 8, 20.0)
    ]
    
    for name, config_type, gpu_count, steps_per_sec in configs:
        cfg = get_training_config(config_type)
        hours = cfg.max_steps / steps_per_sec / 3600
        eff_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps * gpu_count
        tokens_step = eff_batch * cfg.sequence_length
        total_tokens = cfg.max_steps * tokens_step
        epochs_used = total_tokens / dataset_tokens
        
        print(f"   {name}: {cfg.max_steps:,} steps in {hours:.1f}h, using {epochs_used:.3f} epochs")

if __name__ == "__main__":
    analyze_8gpu_config() 