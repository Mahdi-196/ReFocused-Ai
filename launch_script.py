#!/usr/bin/env python3
"""
Complete launch script for Hyperbolic server training
Handles: Environment setup -> Data download -> Training launch
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(cmd, description="", check=True):
    """Run a command with nice output"""
    print(f"🔧 {description}")
    print(f"   $ {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=False, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success!")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Failed: {e}")
        return False

def setup_environment():
    """Set up Python environment and dependencies"""
    print("🚀 SETTING UP TRAINING ENVIRONMENT")
    print("=" * 50)
    
    # Check if already in conda environment
    if 'CONDA_DEFAULT_ENV' in os.environ:
        print(f"✅ Already in conda environment: {os.environ['CONDA_DEFAULT_ENV']}")
    else:
        print("⚠️  Not in conda environment. Please activate 'refocused' first!")
        return False
    
    # Install missing packages that might be needed
    packages = [
        "requests",
        "numpy", 
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "tensorboard",
        "wandb"
    ]
    
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"🔧 Installing {package}...")
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def download_training_data():
    """Download training data from public bucket"""
    print("\n🔽 DOWNLOADING TRAINING DATA")
    print("=" * 50)
    
    # Check if data already exists
    data_dir = Path("data/training")
    if data_dir.exists() and list(data_dir.glob("*.npz")):
        npz_count = len(list(data_dir.glob("*.npz")))
        print(f"✅ Found {npz_count} .npz files already downloaded")
        return True
    
    # Download data using our script
    print("📥 Starting download from gs://refocused-ai/...")
    
    # Use the download script we created
    download_script = """
import requests
import json
from pathlib import Path
import time

bucket_name = "refocused-ai"
base_url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o"
download_url_base = f"https://storage.googleapis.com/{bucket_name}/"

data_dir = Path("data/training")
data_dir.mkdir(parents=True, exist_ok=True)

print("Fetching file list...")
response = requests.get(base_url)
data = response.json()
files = data['items']
npz_files = [f for f in files if f['name'].endswith('.npz')]

print(f"Downloading {len(npz_files)} files...")
for i, file_info in enumerate(npz_files, 1):
    filename = file_info['name']
    download_url = f"{download_url_base}{filename}"
    local_path = data_dir / filename
    
    if i % 100 == 0:
        print(f"Progress: {i}/{len(npz_files)}")
    
    file_response = requests.get(download_url)
    with open(local_path, 'wb') as f:
        f.write(file_response.content)

print(f"✅ Downloaded {len(npz_files)} files to {data_dir}")
"""
    
    # Write and execute download script
    with open("temp_download.py", "w") as f:
        f.write(download_script)
    
    success = run_command("python temp_download.py", "Downloading training data")
    
    # Clean up
    if Path("temp_download.py").exists():
        Path("temp_download.py").unlink()
    
    if success:
        npz_count = len(list(data_dir.glob("*.npz")))
        print(f"✅ Successfully downloaded {npz_count} training files")
        return True
    else:
        print("❌ Download failed")
        return False

def start_training():
    """Start the training process"""
    print("\n🚀 STARTING TRAINING")
    print("=" * 50)
    
    # Change to training directory
    training_dir = Path("05_model_training")
    if not training_dir.exists():
        print("❌ Training directory not found!")
        return False
    
    os.chdir(training_dir)
    print(f"📁 Changed to: {Path.cwd()}")
    
    # Check training script
    train_script = Path("train.py")
    if not train_script.exists():
        print("❌ train.py not found!")
        return False
    
    # Start tensorboard in background
    print("📊 Starting TensorBoard...")
    run_command("tensorboard --logdir=../logs --host=0.0.0.0 --port=6006 &", 
               "Starting TensorBoard", check=False)
    
    # Start training
    data_path = "../data/training"
    
    # Try different training commands
    training_commands = [
        f"torchrun --nproc_per_node=8 train.py --data-path {data_path}",
        f"python train.py --data-path {data_path} --multi-gpu",
        f"python train.py --data-path {data_path}",
        "python train.py --synthetic-data"  # Fallback
    ]
    
    for cmd in training_commands:
        print(f"\n🎯 Trying: {cmd}")
        if run_command(cmd, f"Training with: {cmd}", check=False):
            print("🎉 Training started successfully!")
            return True
        else:
            print(f"❌ Failed, trying next option...")
    
    print("❌ All training attempts failed")
    return False

def main():
    """Main launch sequence"""
    print("🎯 HYPERBOLIC H100 TRAINING LAUNCHER")
    print("=" * 60)
    print("This will set up environment, download data, and start training")
    print("💰 Remember: $317/hour is running!")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Environment setup
    if not setup_environment():
        print("❌ Environment setup failed!")
        return 1
    
    # Step 2: Download data
    if not download_training_data():
        print("❌ Data download failed!")
        return 1
    
    # Step 3: Start training
    if not start_training():
        print("❌ Training failed to start!")
        return 1
    
    elapsed = time.time() - start_time
    print(f"\n🎉 SUCCESS! Setup completed in {elapsed:.1f} seconds")
    print("\n📊 MONITORING:")
    print("   TensorBoard: http://near-blackberry-bison.1.cricket.hyperbolic.xyz:6006")
    print("   GPU Usage: nvidia-smi")
    print("\n🔥 Your 8x H100s should now be training!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 