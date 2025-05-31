#!/usr/bin/env python3
"""
Quick Setup Script for ReFocused-AI Training Environment
Installs all dependencies and verifies access to training data

Usage:
    python quick_setup.py --test_bucket --download_data

Options:
    --test_bucket    Test access to the GCS bucket
    --download_data  Download sample training data
"""

import os
import sys
import argparse
import subprocess
import platform
from pathlib import Path

def print_separator(message):
    """Print a message with separators for visibility"""
    print("\n" + "=" * 80)
    print(f"{message}")
    print("=" * 80)

def run_command(command, check=True):
    """Run a shell command and print output"""
    print(f"\nRunning: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        print(f"Command completed with return code: {result.returncode}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error: {str(e)}")
        return False

def install_dependencies():
    """Install all required dependencies"""
    print_separator("Installing Dependencies")
    
    # Check if pip is available
    run_command(f"{sys.executable} -m pip --version", check=False)
    
    # Upgrade pip
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Install dependencies from requirements.txt
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        run_command(f"{sys.executable} -m pip install -r {requirements_file}")
    else:
        print(f"Warning: requirements.txt not found at {requirements_file.absolute()}")
        # Install critical packages individually
        run_command(f"{sys.executable} -m pip install torch numpy transformers google-cloud-storage")
    
    # Verify critical packages
    packages_to_verify = [
        "torch", "transformers", "google-cloud-storage", 
        "numpy", "deepspeed", "tensorboard"
    ]
    
    print("\nVerifying installations:")
    for package in packages_to_verify:
        run_command(f"{sys.executable} -c \"import {package}; print(f'{package} version: ' + {package}.__version__)\"", check=False)
    
    # Check CUDA availability if torch is installed
    run_command(f"{sys.executable} -c \"import torch; print('CUDA available: ' + str(torch.cuda.is_available()))\"", check=False)
    
    if run_command(f"{sys.executable} -c \"import torch; print(torch.cuda.is_available())\"", check=False):
        run_command(f"{sys.executable} -c \"import torch; print('GPU count: ' + str(torch.cuda.device_count()))\"", check=False)
        run_command(f"{sys.executable} -c \"import torch; print('GPU name: ' + torch.cuda.get_device_name(0))\"", check=False)

def create_directories():
    """Create necessary directories for training"""
    print_separator("Creating Directories")
    
    dirs = [
        "data/training",
        "logs",
        "models/gpt_750m",
        "models/tokenizer/tokenizer",
        "05_model_training/config"
    ]
    
    for dir_path in dirs:
        path = Path(dir_path)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")

def test_bucket_access(bucket_name="refocused-ai", download_sample=False):
    """Test access to GCS bucket"""
    print_separator("Testing GCS Bucket Access")
    
    # Check if we have the test script
    test_script = Path("05_model_training/test_bucket_access.py")
    if not test_script.exists():
        print(f"Warning: Bucket test script not found at {test_script}")
        return False
    
    # Run the test script
    cmd = f"{sys.executable} {test_script} --bucket {bucket_name}"
    if download_sample:
        cmd += " --download"
    
    return run_command(cmd)

def download_sample_data(bucket_name="refocused-ai", max_files=10):
    """Download sample training data using a simple, reliable approach"""
    print_separator("Downloading Sample Training Data")
    
    # Create data directory
    data_dir = Path("data/training/shards")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create client and access bucket
        run_command(f"{sys.executable} -c \"from google.cloud import storage; print('Google Cloud Storage package is available')\"")
        
        # Use a simple inline Python script for more reliable downloading
        download_script = f"""
import os
import sys
from google.cloud import storage

def download_files(bucket_name="{bucket_name}", output_dir="{data_dir}", max_files={max_files}):
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create anonymous client
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    
    # List files in bucket
    print(f"Listing files in gs://{bucket_name}/...")
    blobs = list(bucket.list_blobs(max_results=100))
    npz_blobs = [b for b in blobs if b.name.endswith('.npz')]
    
    if not npz_blobs:
        print("No .npz files found in bucket")
        return False
    
    print(f"Found {{len(npz_blobs)}} .npz files, downloading first {{min(max_files, len(npz_blobs))}}...")
    
    # Take first N files
    download_blobs = npz_blobs[:max_files]
    
    # Download each file
    success_count = 0
    for blob in download_blobs:
        # Extract filename
        filename = os.path.basename(blob.name)
        local_path = os.path.join(output_dir, filename)
        
        print(f"Downloading {{blob.name}} to {{local_path}}...")
        
        try:
            # Download file
            blob.download_to_filename(local_path)
            
            # Verify file was downloaded
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path) / (1024 * 1024)
                print(f"  Downloaded successfully: {{file_size:.2f}} MB")
                success_count += 1
            else:
                print(f"  Failed to download file")
        except Exception as e:
            print(f"  Error downloading {{blob.name}}: {{e}}")
    
    # List downloaded files
    print(f"\\nSuccessfully downloaded {{success_count}} files to {{output_dir}}")
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir) if f.endswith('.npz')) / (1024 * 1024)
    print(f"Total data size: {{total_size:.2f}} MB")
    
    return success_count > 0

# Run the download function
success = download_files()
sys.exit(0 if success else 1)
"""
        
        # Write the script to a temporary file
        temp_script_path = "temp_download_script.py"
        with open(temp_script_path, "w") as f:
            f.write(download_script)
        
        # Run the script
        success = run_command(f"{sys.executable} {temp_script_path}")
        
        # Clean up
        try:
            os.remove(temp_script_path)
        except:
            pass
        
        if success:
            print(f"\n✅ Data download successful")
        else:
            print(f"\n❌ Data download failed")
            
        return success
            
    except Exception as e:
        print(f"Error during data download: {e}")
        return False

def check_environment():
    """Check and report on the execution environment"""
    print_separator("Environment Information")
    
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable: {sys.executable}")
    print(f"Operating system: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    
    # Check for CUDA
    run_command(f"{sys.executable} -c \"import torch; print('PyTorch version: ' + torch.__version__)\"", check=False)
    run_command(f"{sys.executable} -c \"import torch; print('CUDA available: ' + str(torch.cuda.is_available()))\"", check=False)

def main():
    parser = argparse.ArgumentParser(description="Setup ReFocused-AI Training Environment")
    parser.add_argument("--test_bucket", action="store_true",
                      help="Test access to the GCS bucket")
    parser.add_argument("--download_data", action="store_true",
                      help="Download sample training data")
    parser.add_argument("--bucket", type=str, default="refocused-ai",
                      help="GCS bucket name")
    parser.add_argument("--max_files", type=int, default=10,
                      help="Maximum number of files to download")
    
    args = parser.parse_args()
    
    print_separator("Starting ReFocused-AI Environment Setup")
    
    # Check environment
    check_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Test bucket access if requested
    if args.test_bucket:
        test_bucket_access(args.bucket, download_sample=False)
    
    # Download sample data if requested
    if args.download_data:
        download_sample_data(args.bucket, args.max_files)
    
    print_separator("Setup Complete")
    print("\nNext steps:")
    print("1. Run the test script to verify the environment:")
    print(f"   python 05_model_training/quick_test.py")
    print("2. Run a single GPU test:")
    print(f"   cd 05_model_training && python h100_single_gpu_test.py")
    print("3. Start full training:")
    print(f"   cd 05_model_training && bash h100_runner.sh full")

if __name__ == "__main__":
    main() 