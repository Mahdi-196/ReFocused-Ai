#!/usr/bin/env python3
"""
Auto Setup Script for ReFocused-AI H100 Training Environment
Fully automated setup script that handles everything in one go

Usage:
    python auto_setup.py [--no_download] [--test_only] [--full_training]

Options:
    --no_download    Skip downloading training data
    --test_only      Only run a test training job
    --full_training  Start full training after setup
"""

import os
import sys
import subprocess
import argparse
import platform
import time
from pathlib import Path

# Configuration
BUCKET_NAME = "refocused-ai"
DATA_LOCAL_DIR = "/home/ubuntu/training_data/shards"
MAX_FILES = 25  # For test mode
WORKING_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_ROOT = "/home/ubuntu"

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f"🚀 {message}")
    print("=" * 80)

def run_command(command, check=True, shell=True, cwd=None, env=None):
    """Run a shell command and print output"""
    print(f"\n>> Running: {command}")
    try:
        # Add PATH to environment to include .local/bin
        if env is None:
            env = os.environ.copy()
            if 'PATH' in env and '/home/ubuntu/.local/bin' not in env['PATH']:
                env['PATH'] = f"/home/ubuntu/.local/bin:{env['PATH']}"
        
        # Run command
        result = subprocess.run(
            command if shell else command.split(), 
            shell=shell, 
            check=False,  # Don't raise exception, handle errors manually
            text=True, 
            capture_output=True,
            cwd=cwd,
            env=env
        )
        
        # Print output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        # Handle return code
        if result.returncode != 0:
            print(f"⚠️  Command returned non-zero exit code: {result.returncode}")
            if check:
                print(f"❌ Error executing command: {command}")
                return False
        else:
            print(f"✅ Command completed successfully")
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Exception executing command: {str(e)}")
        return False

def is_git_repo(directory):
    """Check if a directory is a Git repository"""
    return os.path.isdir(os.path.join(directory, '.git'))

def setup_directories():
    """Create all necessary directories"""
    print_header("Creating Directories")
    
    directories = [
        "/home/ubuntu/training_data/shards",
        "/home/ubuntu/training_data/checkpoints",
        "/home/ubuntu/training_data/logs",
        "/home/ubuntu/ReFocused-Ai/models/gpt_750m",
        "/home/ubuntu/ReFocused-Ai/models/tokenizer/tokenizer"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Error creating directory {directory}: {str(e)}")

def fix_environment_variables():
    """Set up environment variables for H100 training"""
    print_header("Setting Up Environment Variables")
    
    env_file = "/home/ubuntu/h100_env.sh"
    env_content = """#!/bin/bash
# Environment variables for DeepSpeed and H100
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export PATH="$HOME/.local/bin:$PATH"
"""
    
    try:
        with open(env_file, "w") as f:
            f.write(env_content)
        os.chmod(env_file, 0o755)  # Make executable
        print(f"✅ Created environment file: {env_file}")
        
        # Also add to .bashrc for permanent setup
        with open("/home/ubuntu/.bashrc", "a") as f:
            f.write('\n# Added by ReFocused-AI setup\n')
            f.write('export PATH="$HOME/.local/bin:$PATH"\n')
            f.write('export RDMAV_FORK_SAFE=1\n')
            f.write('export FI_EFA_FORK_SAFE=1\n')
            f.write('export NCCL_IB_DISABLE=1\n')
            f.write('export NCCL_P2P_DISABLE=1\n')
        
        # Source the environment file to set variables for this session
        os.environ['RDMAV_FORK_SAFE'] = '1'
        os.environ['FI_EFA_FORK_SAFE'] = '1'
        os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'
        if 'PATH' in os.environ and '/home/ubuntu/.local/bin' not in os.environ['PATH']:
            os.environ['PATH'] = f"/home/ubuntu/.local/bin:{os.environ['PATH']}"
        
        return True
    except Exception as e:
        print(f"❌ Error setting up environment variables: {str(e)}")
        return False

def install_dependencies():
    """Install all required dependencies with error handling"""
    print_header("Installing Dependencies")
    
    # Install system packages first
    print("Installing system dependencies...")
    run_command("apt-get update && apt-get install -y build-essential libopenmpi-dev", check=False)
    
    # Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install core build dependencies
    run_command(f"{sys.executable} -m pip install packaging setuptools wheel")
    
    # Fix NumPy version for compatibility
    run_command(f"{sys.executable} -m pip uninstall -y numpy")
    run_command(f"{sys.executable} -m pip install numpy==1.24.3")
    
    # Install PyTorch with CUDA support
    pytorch_cmd = f"{sys.executable} -m pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    if not run_command(pytorch_cmd):
        # Fallback to CPU version if CUDA install fails
        print("⚠️  PyTorch CUDA installation failed, falling back to CPU version...")
        run_command(f"{sys.executable} -m pip install torch==2.0.1 torchvision torchaudio")
    
    # Install mpi4py separately
    run_command(f"{sys.executable} -m pip install mpi4py")
    
    # Install key packages individually for more reliability
    key_packages = [
        "transformers==4.35.0",
        "datasets==2.14.7", 
        "accelerate==0.25.0", 
        "deepspeed==0.12.0",
        "google-cloud-storage",
        "tensorboard",
        "wandb",
        "pandas",
        "tqdm"
    ]
    
    for package in key_packages:
        run_command(f"{sys.executable} -m pip install {package}")
    
    # Verify critical packages
    print("\nVerifying key packages:")
    packages_to_verify = ["torch", "transformers", "deepspeed", "accelerate", "google.cloud.storage"]
    for package in packages_to_verify:
        module_name = package.split(".")[0]  # Handle nested modules
        run_command(f"{sys.executable} -c \"import {package}; print('✅ {module_name} is available')\"", check=False)
    
    # Check CUDA availability
    run_command(f"{sys.executable} -c \"import torch; print('CUDA available: ' + str(torch.cuda.is_available()))\"", check=False)
    run_command(f"{sys.executable} -c \"import torch; print('GPU count: ' + str(torch.cuda.device_count()))\"", check=False)
    run_command(f"{sys.executable} -c \"import torch; print('GPU name: ' + str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'No GPU')\"", check=False)

def setup_model_config():
    """Set up model configuration files"""
    print_header("Setting Up Model Configuration")
    
    config_path = "/home/ubuntu/ReFocused-Ai/models/gpt_750m/config.json"
    config_content = """{
  "architectures": ["GPTNeoXForCausalLM"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "gpt_neox",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "rotary_emb_base": 10000,
  "rotary_pct": 0.25,
  "tie_word_embeddings": false,
  "transformers_version": "4.28.1",
  "use_cache": true,
  "vocab_size": 50304
}"""
    
    try:
        # Create config file if it doesn't exist
        if not os.path.exists(config_path):
            with open(config_path, "w") as f:
                f.write(config_content)
            print(f"✅ Created model config file: {config_path}")
        else:
            print(f"✅ Model config file already exists: {config_path}")
        
        # Create empty config for deepspeed if needed
        ds_config_dir = "/home/ubuntu/ReFocused-Ai/05_model_training/config"
        os.makedirs(ds_config_dir, exist_ok=True)
        
        return True
    except Exception as e:
        print(f"❌ Error setting up model configuration: {str(e)}")
        return False

def setup_tokenizer():
    """Set up the tokenizer files"""
    print_header("Setting Up Tokenizer")
    
    tokenizer_path = "/home/ubuntu/ReFocused-Ai/models/tokenizer/tokenizer"
    
    try:
        # Check if tokenizer already exists
        if os.path.exists(os.path.join(tokenizer_path, "vocab.json")):
            print(f"✅ Tokenizer already exists at: {tokenizer_path}")
            return True
        
        # Download tokenizer using transformers
        print("Downloading tokenizer from Hugging Face...")
        cmd = f"{sys.executable} -c \"from transformers import GPT2Tokenizer; tokenizer = GPT2Tokenizer.from_pretrained('gpt2'); tokenizer.save_pretrained('{tokenizer_path}')\""
        return run_command(cmd)
    except Exception as e:
        print(f"❌ Error setting up tokenizer: {str(e)}")
        return False

def setup_monitoring():
    """Set up monitoring scripts like TensorBoard"""
    print_header("Setting Up Monitoring Tools")
    
    tensorboard_script = "/home/ubuntu/start_tensorboard.sh"
    script_content = """#!/bin/bash
mkdir -p /home/ubuntu/training_data/logs
tensorboard --logdir=/home/ubuntu/training_data/logs --port=6006 --host=0.0.0.0 &
echo "TensorBoard running at http://localhost:6006"
echo "If running on a remote server, forward this port to access TensorBoard"
"""
    
    try:
        with open(tensorboard_script, "w") as f:
            f.write(script_content)
        os.chmod(tensorboard_script, 0o755)  # Make executable
        print(f"✅ Created TensorBoard script: {tensorboard_script}")
        return True
    except Exception as e:
        print(f"❌ Error setting up monitoring tools: {str(e)}")
        return False

def download_data():
    """Download training data from GCS bucket"""
    print_header(f"Downloading Training Data from {BUCKET_NAME}")
    
    script_dir = "/home/ubuntu/ReFocused-Ai/05_model_training"
    os.makedirs(DATA_LOCAL_DIR, exist_ok=True)
    
    # Apply the download fix first
    fix_script = os.path.join(script_dir, "apply_download_fix.sh")
    if os.path.exists(fix_script):
        print("Applying download script fix...")
        run_command(f"chmod +x {fix_script} && {fix_script}", cwd=script_dir)
    
    # Run the download script
    download_script = os.path.join(script_dir, "download_data.py")
    if os.path.exists(download_script):
        cmd = f"{sys.executable} {download_script} --bucket {BUCKET_NAME} --local_dir {DATA_LOCAL_DIR} --max_files {MAX_FILES} --workers 8"
        return run_command(cmd, cwd=script_dir)
    else:
        print(f"❌ Download script not found: {download_script}")
        return False

def make_scripts_executable():
    """Make all necessary scripts executable"""
    print_header("Making Scripts Executable")
    
    scripts = [
        "/home/ubuntu/ReFocused-Ai/05_model_training/h100_runner.sh",
        "/home/ubuntu/ReFocused-Ai/05_model_training/apply_download_fix.sh",
        "/home/ubuntu/ReFocused-Ai/05_model_training/setup_fixed_env.sh"
    ]
    
    for script in scripts:
        if os.path.exists(script):
            run_command(f"chmod +x {script}")
            print(f"✅ Made executable: {script}")

def run_test_training():
    """Run a test training job"""
    print_header("Running Test Training")
    
    script_dir = "/home/ubuntu/ReFocused-Ai/05_model_training"
    runner_script = os.path.join(script_dir, "h100_runner.sh")
    
    if os.path.exists(runner_script):
        print("Starting test training job...")
        return run_command(f"bash {runner_script} test", cwd=script_dir)
    else:
        print(f"❌ Runner script not found: {runner_script}")
        return False

def run_full_training():
    """Run the full training job"""
    print_header("Starting Full Training")
    
    script_dir = "/home/ubuntu/ReFocused-Ai/05_model_training"
    runner_script = os.path.join(script_dir, "h100_runner.sh")
    
    if os.path.exists(runner_script):
        print("Starting full training job...")
        return run_command(f"bash {runner_script} full", cwd=script_dir)
    else:
        print(f"❌ Runner script not found: {runner_script}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Auto Setup for ReFocused-AI Training")
    parser.add_argument("--no_download", action="store_true", help="Skip downloading training data")
    parser.add_argument("--test_only", action="store_true", help="Only run a test training job")
    parser.add_argument("--full_training", action="store_true", help="Start full training after setup")
    
    args = parser.parse_args()
    
    print_header("ReFocused-AI Automated Setup")
    print(f"Running on: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()} at {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Execute setup steps in order
    setup_directories()
    fix_environment_variables()
    install_dependencies()
    setup_model_config()
    setup_tokenizer()
    setup_monitoring()
    make_scripts_executable()
    
    # Download data unless skipped
    if not args.no_download:
        download_data()
    
    # Run training if requested
    if args.test_only:
        run_test_training()
    elif args.full_training:
        run_full_training()
    
    print_header("Setup Complete!")
    print("""
To use your environment:
    
1. Activate environment variables:
   source /home/ubuntu/h100_env.sh
    
2. Run a test training:
   cd /home/ubuntu/ReFocused-Ai/05_model_training
   ./h100_runner.sh test
    
3. Monitor with TensorBoard:
   /home/ubuntu/start_tensorboard.sh
    
4. Start full training when ready:
   ./h100_runner.sh full
""")

if __name__ == "__main__":
    # Set working directory to script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main() 