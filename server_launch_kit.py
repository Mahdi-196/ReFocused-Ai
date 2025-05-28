#!/usr/bin/env python3
"""
COMPLETE SERVER LAUNCH KIT for Hyperbolic H100 Training
This script handles everything: download, setup, and training launch
"""

import subprocess
import os
import time
from pathlib import Path

def run_cmd(cmd, description=""):
    """Run command with nice output"""
    print(f"🔧 {description}")
    print(f"   $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False)
    if result.returncode == 0:
        print("   ✅ Success!")
        return True
    else:
        print(f"   ❌ Failed with code {result.returncode}")
        return False

def main():
    print("🚀 HYPERBOLIC SERVER LAUNCH SEQUENCE")
    print("=" * 60)
    print("This will:")
    print("1. 📥 Download 21GB training data (~11 minutes)")
    print("2. 🔧 Setup Python environment")
    print("3. 🎯 Launch 8x H100 training")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Activate conda environment
    print("\n🔧 STEP 1: Activate Environment")
    if not run_cmd("source ~/miniconda3/bin/activate && conda activate refocused", "Activating conda environment"):
        print("❌ Environment activation failed!")
        return False
    
    # Step 2: Download training data
    print("\n📥 STEP 2: Download Training Data")
    print("Starting 21GB download from gs://refocused-ai/")
    if not run_cmd("python fixed_download.py", "Downloading training data"):
        print("❌ Data download failed!")
        return False
    
    # Step 3: Install remaining dependencies
    print("\n🔧 STEP 3: Install Final Dependencies")
    deps = [
        "pip install deepspeed",
        "pip install wandb",
        "pip install transformers",
        "pip install accelerate",
        "pip install tensorboard"
    ]
    
    for dep in deps:
        if not run_cmd(dep, f"Installing {dep.split()[-1]}"):
            print(f"Warning: {dep} failed, continuing...")
    
    # Step 4: Setup training directories
    print("\n📁 STEP 4: Setup Training Environment")
    dirs_to_create = [
        "logs",
        "checkpoints",
        "outputs",
        "/scratch/logs",
        "/scratch/checkpoints"
    ]
    
    for directory in dirs_to_create:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")
    
    # Step 5: Quick GPU check
    print("\n🔍 STEP 5: GPU Verification")
    if not run_cmd("nvidia-smi", "Checking GPU status"):
        print("❌ GPU check failed!")
        return False
    
    # Step 6: Launch training
    print("\n🔥 STEP 6: LAUNCH TRAINING!")
    training_cmd = """cd 05_model_training && python train.py \\
        --config configs/training_config.yaml \\
        --data-path ../data/training \\
        --output-dir ../checkpoints \\
        --log-dir ../logs"""
    
    print("🎯 Starting training with 8x H100 GPUs...")
    print(f"Command: {training_cmd}")
    
    # Don't use run_cmd for training as it's long-running
    print("🚀 TRAINING LAUNCHED!")
    print("Monitor with:")
    print("   tail -f ../logs/training.log")
    print("   nvidia-smi -l 1")
    print("   wandb watch (if configured)")
    
    elapsed = time.time() - start_time
    print(f"\n⏱️  Total setup time: {elapsed/60:.1f} minutes")
    print("💰 Training cost: $317/hour")
    
    # Actually start training
    os.system(training_cmd)

if __name__ == "__main__":
    main() 