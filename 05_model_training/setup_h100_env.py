import os
import subprocess
import requests
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
NUM_FILES_TO_DOWNLOAD = 25

PYTHON_EXECUTABLE = "python3.11" # Ensure this is the correct python version

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
        # run_command(["git", "pull"], working_dir=REPO_DIR)
    else:
        run_command(["git", "clone", GIT_REPO_URL, REPO_DIR], working_dir=BASE_DIR)

def install_python_dependencies():
    print("\n--- Installing Python dependencies ---   	")
    # Ensure pip is up-to-date for the correct python version
    run_command(f"{PYTHON_EXECUTABLE} -m pip install --upgrade pip")
    
    # Install PyTorch with CUDA 12.4 (adjust if your CUDA version differs)
    run_command(f"{PYTHON_EXECUTABLE} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
    
    # Install other dependencies from requirements_training.txt if it exists and is preferred
    # requirements_file = os.path.join(MODEL_TRAINING_DIR, "requirements_training.txt")
    # if os.path.exists(requirements_file):
    #     print(f"Installing dependencies from {requirements_file}")
    #     run_command(f"{PYTHON_EXECUTABLE} -m pip install -r {requirements_file}")
    # else:
    #     print("requirements_training.txt not found, installing packages individually.")
    run_command(f"{PYTHON_EXECUTABLE} -m pip install --user transformers accelerate deepspeed")
    run_command(f"{PYTHON_EXECUTABLE} -m pip install --user datasets wandb tensorboard pyyaml")
    run_command(f"{PYTHON_EXECUTABLE} -m pip install --user google-cloud-storage requests numpy tokenizers")
    print("Verifying installations...")
    run_command(f"{PYTHON_EXECUTABLE} -m pip list | grep -E 'transformers|accelerate|deepspeed|torch|datasets|wandb|tensorboard|PyYAML|google-cloud-storage|requests|numpy|tokenizers'")

def create_persistent_directories():
    print("\n--- Creating persistent storage directories ---")
    dirs_to_create = [SHARDS_DIR, CHECKPOINTS_DIR, LOGS_DIR, CACHE_DIR, DEEPSPEED_NVME_DIR]
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Ensured directory exists: {dir_path}")

def download_training_data():
    print(f"\n--- Downloading training data (first {NUM_FILES_TO_DOWNLOAD} files) ---")
    os.makedirs(SHARDS_DIR, exist_ok=True)

    bucket_url = f'https://storage.googleapis.com/storage/v1/b/{DATA_DOWNLOAD_BUCKET}/o'
    print(f"Fetching file list from: {bucket_url}")
    try:
        response = requests.get(bucket_url, params={'prefix': 'tokenized_data/', 'maxResults': 1000}) # Fetch more to ensure we find enough NPZs
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"Error fetching file list from GCS: {e}")
        return

    if response.status_code == 200:
        items = response.json().get('items', [])
        # Ensure we only get files from the 'tokenized_data/' prefix and not other directories like 'tokenized_data_small/'
        npz_files = [item for item in items if item['name'].endswith('.npz') and item['name'].startswith('tokenized_data/')]
        
        print(f"Found {len(npz_files)} .npz files in 'tokenized_data/' directory.")

        downloaded_count = 0
        for item in npz_files[:NUM_FILES_TO_DOWNLOAD]:
            file_name = item['name'].split('/')[-1]
            download_url = f'https://storage.googleapis.com/{DATA_DOWNLOAD_BUCKET}/{item["name"]}'
            destination_path = os.path.join(SHARDS_DIR, file_name)

            if os.path.exists(destination_path) and os.path.getsize(destination_path) == int(item['size']):
                print(f"File {file_name} already exists and size matches. Skipping download.")
                downloaded_count +=1
                continue
            
            print(f"Downloading {file_name} from {download_url}...")
            try:
                file_response = requests.get(download_url, stream=True)
                file_response.raise_for_status()
                with open(destination_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Successfully downloaded {file_name}")
                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_name}: {e}")
            
            if downloaded_count >= NUM_FILES_TO_DOWNLOAD:
                break
        
        print(f"\n✅ Downloaded/verified {downloaded_count} files to {SHARDS_DIR}")
    else:
        print(f"Failed to list files from bucket. Status code: {response.status_code}, Response: {response.text}")

def start_tensorboard():
    print("\n--- Starting TensorBoard ---   	")
    # Check if TensorBoard is already running
    try:
        subprocess.check_output(["pgrep", "-f", "tensorboard --logdir /home/ubuntu/training_data/logs"])
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
    
    # 1. System Setup (Optional, assuming image has basics. User can uncomment if needed)
    # install_system_packages() 
    
    # 2. Clone Repository
    clone_repository()
    
    # 3. Install Python Dependencies
    # Make sure to `cd` into the repo if requirements.txt is to be used from there
    os.chdir(MODEL_TRAINING_DIR) # Important for relative paths in config if any
    install_python_dependencies()
    
    # 4. Create Persistent Directories
    create_persistent_directories()
    
    # 5. Download Training Data
    download_training_data()
    
    # 6. Start TensorBoard
    start_tensorboard()
    
    print("\n🎉 Setup Complete! You should be ready to start training. 🎉")
    print("Next steps:")
    print(f"1. Navigate to the training directory: cd {MODEL_TRAINING_DIR}")
    print("2. Verify your 'config/training_config.yaml' and 'config/model_config.json' are correct.")
    print(f"3. Start training: {PYTHON_EXECUTABLE} train.py --config config/training_config.yaml")
    print(f"4. Monitor on TensorBoard (check {BASE_DIR}/tensorboard.log for any issues).")

if __name__ == "__main__":
    main() 