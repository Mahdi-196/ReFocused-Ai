import os
import subprocess
import requests
import time
from pathlib import Path

# Configuration
GIT_REPO_URL = "https://github.com/Mahdi-196/ReFocused-Ai.git"
BASE_DIR = "/home/ubuntu"
REPO_DIR = os.path.join(BASE_DIR, "ReFocused-Ai")
MODEL_TRAINING_DIR = os.path.join(REPO_DIR, "05_model_training")
TRAINING_DATA_BASE_DIR = os.path.join(BASE_DIR, "training_data")
SHARDS_DIR = os.path.join(TRAINING_DATA_BASE_DIR, "shards")
CHECKPOINTS_DIR = os.path.join(TRAINING_DATA_BASE_DIR, "checkpoints")
LOGS_DIR = os.path.join(TRAINING_DATA_BASE_DIR, "logs")
CACHE_DIR = os.path.join(TRAINING_DATA_BASE_DIR, "cache")
DEEPSPEED_NVME_DIR = os.path.join(TRAINING_DATA_BASE_DIR, "deepspeed_nvme")

TENSORBOARD_LOG_DIR = LOGS_DIR
TENSORBOARD_PORT = 6006

DATA_DOWNLOAD_BUCKET = "refocused-ai"
DATA_REMOTE_PATH = ""  # Empty for root of bucket
NUM_FILES_TO_DOWNLOAD = 25

# Use the system Python - more reliable than hardcoding version
PYTHON_EXECUTABLE = "python3"

def run_command(command, working_dir=None, shell=True, check=True):
    print(f"\nExecuting: {' '.join(command) if isinstance(command, list) else command}")
    try:
        process = subprocess.run(command, cwd=working_dir, shell=shell, check=check, text=True, capture_output=True)
        if process.stdout:
            print(process.stdout)
        if process.stderr:
            print(f"Stderr: {process.stderr}")
        print(f"Successfully executed: {' '.join(command) if isinstance(command, list) else command}")
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command) if isinstance(command, list) else command}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        raise

def install_system_packages():
    print("\n--- Updating system packages and installing prerequisites ---")
    run_command("sudo apt update -y")
    run_command("sudo apt install -y git htop wget curl")
    run_command("nvidia-smi")

def clone_repository():
    print(f"\n--- Cloning repository: {GIT_REPO_URL} ---")
    if os.path.exists(REPO_DIR):
        print(f"Repository already exists at {REPO_DIR}. Skipping clone.")
        # Optional: Add logic to pull latest changes if repo exists
        run_command(["git", "pull"], working_dir=REPO_DIR)
    else:
        run_command(["git", "clone", GIT_REPO_URL, REPO_DIR], working_dir=BASE_DIR)

def install_python_dependencies():
    print("\n--- Installing Python dependencies ---")
    # Ensure pip is up-to-date
    run_command(f"{PYTHON_EXECUTABLE} -m pip install --upgrade pip")
    
    # Install dependencies from requirements file
    requirements_file = os.path.join(MODEL_TRAINING_DIR, "requirements_training.txt")
    if os.path.exists(requirements_file):
        print(f"Installing dependencies from {requirements_file}")
        try:
            run_command(f"{PYTHON_EXECUTABLE} -m pip install -r {requirements_file}")
        except subprocess.CalledProcessError:
            print("Failed to install all dependencies. Installing essential packages individually.")
            # Install PyTorch with CUDA
            run_command(f"{PYTHON_EXECUTABLE} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            # Install critical packages individually
            run_command(f"{PYTHON_EXECUTABLE} -m pip install transformers deepspeed google-cloud-storage requests")
            run_command(f"{PYTHON_EXECUTABLE} -m pip install numpy pyyaml wandb tensorboard")
    else:
        print("requirements_training.txt not found, installing packages individually.")
        run_command(f"{PYTHON_EXECUTABLE} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        run_command(f"{PYTHON_EXECUTABLE} -m pip install transformers accelerate deepspeed google-cloud-storage")
        run_command(f"{PYTHON_EXECUTABLE} -m pip install datasets wandb tensorboard pyyaml requests")
    
    print("Verifying installations...")
    run_command(f"{PYTHON_EXECUTABLE} -m pip list | grep -E 'transformers|accelerate|deepspeed|torch|google-cloud-storage|requests'")

def create_persistent_directories():
    print("\n--- Creating persistent storage directories ---")
    dirs_to_create = [SHARDS_DIR, CHECKPOINTS_DIR, LOGS_DIR, CACHE_DIR, DEEPSPEED_NVME_DIR]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")

def create_model_directory():
    print("\n--- Creating model directories and config files ---")
    # Create model directory
    model_dir = os.path.join(REPO_DIR, "models", "gpt_750m")
    tokenizer_dir = os.path.join(REPO_DIR, "models", "tokenizer", "tokenizer")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # Create model config file if it doesn't exist
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Creating model config file at {config_path}")
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
        with open(config_path, 'w') as f:
            f.write(config_content)
        print(f"Created model config file: {config_path}")
    
    # Create tokenizer files if they don't exist
    if not os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
        print("Creating tokenizer files using GPT2Tokenizer...")
        try:
            cmd = f"{PYTHON_EXECUTABLE} -c \"from transformers import GPT2Tokenizer; tokenizer = GPT2Tokenizer.from_pretrained('gpt2'); tokenizer.save_pretrained('{tokenizer_dir}')\""
            run_command(cmd)
            print(f"Created tokenizer files in {tokenizer_dir}")
        except Exception as e:
            print(f"Error creating tokenizer files: {e}")
            print("You may need to create these manually")

def update_configuration_files():
    print("\n--- Updating configuration files ---")
    # Update h100_runner.sh script
    h100_runner_path = os.path.join(MODEL_TRAINING_DIR, "h100_runner.sh")
    if os.path.exists(h100_runner_path):
        try:
            run_command(f"sed -i 's/DATA_REMOTE_PATH=\"tokenized_data\"/DATA_REMOTE_PATH=\"{DATA_REMOTE_PATH}\"/' {h100_runner_path}")
            print("Updated h100_runner.sh with correct data path")
        except:
            print("Could not update h100_runner.sh")
    
    # Update h100_multi_gpu.yaml file
    multi_gpu_config_path = os.path.join(MODEL_TRAINING_DIR, "config", "h100_multi_gpu.yaml")
    if os.path.exists(multi_gpu_config_path):
        try:
            run_command(f"sed -i 's/remote_data_path: \"tokenized_data\"/remote_data_path: \"{DATA_REMOTE_PATH}\"/' {multi_gpu_config_path}")
            print("Updated h100_multi_gpu.yaml with correct data path")
        except:
            print("Could not update h100_multi_gpu.yaml")

def download_training_data():
    print(f"\n--- Downloading training data (first {NUM_FILES_TO_DOWNLOAD} files) ---")
    os.makedirs(SHARDS_DIR, exist_ok=True)
    
    try:
        # First test bucket access and list files
        print(f"Testing access to bucket: {DATA_DOWNLOAD_BUCKET}")
        
        test_script = f"""
import sys
from google.cloud import storage

def test_bucket_access(bucket_name="{DATA_DOWNLOAD_BUCKET}"):
    try:
        # Create anonymous client for public bucket
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        
        # List top-level files/directories
        print(f"Listing files in gs://{bucket_name}/")
        blobs = list(bucket.list_blobs(max_results=10))
        
        if not blobs:
            print(f"Warning: No files found in bucket {bucket_name}")
            return False
            
        print(f"Found {len(blobs)} files in bucket. Sample files:")
        for blob in blobs[:5]:
            print(f" - {blob.name} ({blob.size/1024/1024:.2f} MB)")
            
        # Count .npz files
        npz_files = [b for b in bucket.list_blobs() if b.name.endswith('.npz')]
        print(f"Found {len(npz_files)} .npz files in the bucket")
        
        # List a few .npz files
        if npz_files:
            print("Sample .npz files:")
            for blob in npz_files[:5]:
                print(f" - {blob.name} ({blob.size/1024/1024:.2f} MB)")
        
        return True
    except Exception as e:
        print(f"Error accessing bucket: {e}")
        return False

if not test_bucket_access():
    sys.exit(1)
"""
        
        # Run the test script
        try:
            with open("test_bucket_access.py", "w") as f:
                f.write(test_script)
            
            run_command(f"{PYTHON_EXECUTABLE} test_bucket_access.py")
            print("✅ Bucket access test passed")
        except Exception as e:
            print(f"❌ Bucket access test failed: {e}")
            print("This may indicate issues with the bucket name or permissions")
            return
        
        # Check if download_data.py script exists
        download_script_path = os.path.join(MODEL_TRAINING_DIR, "download_data.py")
        if not os.path.exists(download_script_path):
            print(f"Error: {download_script_path} not found")
            return
        
        # Try to download data with proper path
        cmd = f"{PYTHON_EXECUTABLE} {download_script_path} --bucket {DATA_DOWNLOAD_BUCKET}"
        if DATA_REMOTE_PATH:
            cmd += f" --remote_path {DATA_REMOTE_PATH}"
        cmd += f" --local_dir {SHARDS_DIR} --max_files {NUM_FILES_TO_DOWNLOAD} --workers 8"
        
        print(f"Downloading data with command: {cmd}")
        try:
            run_command(cmd)
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Waiting 5 seconds and trying again...")
            time.sleep(5)
            run_command(cmd)
        
        # Check if files were downloaded
        npz_files = list(Path(SHARDS_DIR).glob("*.npz"))
        if npz_files:
            print(f"\n✅ Found {len(npz_files)} .npz files in {SHARDS_DIR}")
            # Check file sizes
            total_size_mb = sum(f.stat().st_size for f in npz_files) / (1024 * 1024)
            print(f"Total data size: {total_size_mb:.2f} MB")
            
            # Verify file integrity
            print("Verifying file integrity...")
            try:
                import numpy as np
                sample_file = npz_files[0]
                data = np.load(sample_file)
                keys = list(data.keys())
                print(f"Sample file contains keys: {keys}")
                
                # Check for expected keys
                expected_keys = ["input_ids", "arr_0", "sequences", "text"]
                found_keys = [key for key in expected_keys if key in keys]
                if found_keys:
                    key = found_keys[0]
                    shape = data[key].shape
                    print(f"✅ Data verification successful. Shape for '{key}': {shape}")
                else:
                    print(f"⚠️ Warning: None of the expected keys {expected_keys} found")
            except Exception as e:
                print(f"⚠️ Warning: Could not verify file contents: {e}")
        else:
            print(f"\n❌ No .npz files found in {SHARDS_DIR}. Download may have failed.")
            print("Please check bucket name, path, and permissions.")
            
            # Try to list local directory
            run_command(f"ls -la {SHARDS_DIR}")
    except Exception as e:
        print(f"Error in data download process: {e}")
        print("Please check the bucket name and permissions")

def start_tensorboard():
    print("\n--- Starting TensorBoard ---")
    # Check if TensorBoard is already running
    try:
        subprocess.check_output(["pgrep", "-f", f"tensorboard --logdir {TENSORBOARD_LOG_DIR}"])
        print(f"TensorBoard already running on port {TENSORBOARD_PORT}.")
        return
    except subprocess.CalledProcessError:
        # Not running, so start it
        pass # Intentional pass

    command = f"nohup tensorboard --logdir {TENSORBOARD_LOG_DIR} --host 0.0.0.0 --port {TENSORBOARD_PORT} > {BASE_DIR}/tensorboard.log 2>&1 &"
    print(f"Starting TensorBoard with command: {command}")
    subprocess.Popen(command, shell=True, preexec_fn=os.setpgrp) # Run in background, detached
    print(f"TensorBoard started in background. Log: {BASE_DIR}/tensorboard.log")
    print(f"Access TensorBoard at: http://<YOUR_INSTANCE_IP>:{TENSORBOARD_PORT} or mapped URL")

def main():
    print("🚀 Starting H100 Environment Setup Script 🚀")
    
    # 1. Clone Repository
    clone_repository()
    
    # 2. Create Persistent Directories
    create_persistent_directories()
    
    # 3. Create Model Directories and Config Files
    create_model_directory()
    
    # 4. Update Configuration Files
    update_configuration_files()
    
    # 5. Install Python Dependencies
    # Make sure to `cd` into the repo if requirements.txt is to be used from there
    os.chdir(MODEL_TRAINING_DIR) # Important for relative paths in config if any
    install_python_dependencies()
    
    # 6. Download Training Data
    download_training_data()
    
    # 7. Start TensorBoard
    start_tensorboard()
    
    print("\n🎉 Setup Complete! You should be ready to start training. 🎉")
    print("Next steps:")
    print(f"1. Navigate to the training directory: cd {MODEL_TRAINING_DIR}")
    print("2. Run a quick test: bash h100_runner.sh test")
    print("3. Start full training: bash h100_runner.sh full")
    print(f"4. Monitor on TensorBoard at port {TENSORBOARD_PORT}")

if __name__ == "__main__":
    main() 