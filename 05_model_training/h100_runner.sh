#!/bin/bash
# H100 Training Runner Script
# Complete workflow for efficient training on H100 SXM GPUs
# 
# Usage:
#   Single GPU test: ./h100_runner.sh test
#   Full 8-GPU training: ./h100_runner.sh full
#   Download data only: ./h100_runner.sh download
#   Upload checkpoints: ./h100_runner.sh upload [checkpoint_dir]

set -e

# Configuration
BUCKET_NAME="refocused-ai"
DATA_REMOTE_PATH=""
DATA_LOCAL_DIR="/home/ubuntu/training_data/shards"
CHECKPOINT_DIR="/home/ubuntu/training_data/checkpoints"
MAX_FILES=25 # For testing only

# Critical environment variables for H100 SXM
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Create necessary directories
mkdir -p $DATA_LOCAL_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p "/home/ubuntu/training_data/logs"

# Install required packages
install_requirements() {
    echo "Installing required packages..."
    pip install google-cloud-storage deepspeed torch torchvision transformers wandb pyyaml
}

# Download data from GCS
download_data() {
    echo "Downloading training data..."
    if [ "$1" == "test" ]; then
        echo "Test mode: Downloading only $MAX_FILES files"
        if [ -z "$DATA_REMOTE_PATH" ]; then
            python3 download_data.py --bucket $BUCKET_NAME \
                --local_dir $DATA_LOCAL_DIR --max_files $MAX_FILES --workers 8
        else
            python3 download_data.py --bucket $BUCKET_NAME --remote_path "$DATA_REMOTE_PATH" \
                --local_dir $DATA_LOCAL_DIR --max_files $MAX_FILES --workers 8
        fi
    else
        echo "Full mode: Downloading all available files"
        if [ -z "$DATA_REMOTE_PATH" ]; then
            python3 download_data.py --bucket $BUCKET_NAME \
                --local_dir $DATA_LOCAL_DIR --workers 16
        else
            python3 download_data.py --bucket $BUCKET_NAME --remote_path "$DATA_REMOTE_PATH" \
                --local_dir $DATA_LOCAL_DIR --workers 16
        fi
    fi
}

# Run single GPU test
run_single_gpu_test() {
    echo "Running single GPU test on H100 SXM..."
    # Clear CUDA cache
    python3 -c "import torch; torch.cuda.empty_cache()"
    
    # Kill any existing Python processes
    pkill -9 python3 || true
    sleep 2
    
    # Run test
    python3 h100_single_gpu_test.py --output_dir "h100_test_output" \
        --data_dir $DATA_LOCAL_DIR --num_files $MAX_FILES
}

# Run full 8-GPU training
run_full_training() {
    echo "Running full 8-GPU training on H100 SXM cluster..."
    # Clear CUDA cache
    python3 -c "import torch; torch.cuda.empty_cache()"
    
    # Kill any existing Python processes
    pkill -9 python3 || true
    sleep 2
    
    # Run with DeepSpeed
    deepspeed --num_gpus=8 train_pytorch.py --config config/h100_multi_gpu.yaml
}

# Upload checkpoints to GCS
upload_checkpoints() {
    if [ -z "$1" ]; then
        echo "Error: Checkpoint directory not specified"
        echo "Usage: ./h100_runner.sh upload [checkpoint_dir]"
        exit 1
    fi
    
    echo "Uploading checkpoint $1 to GCS..."
    python3 upload_checkpoints.py --checkpoint_dir "$1" \
        --bucket $BUCKET_NAME --remote_path "checkpoints" --workers 8
}

# Main execution
if [ $# -eq 0 ]; then
    echo "No command specified. Please use one of the following:"
    echo "  ./h100_runner.sh test       - Run single GPU test with $MAX_FILES files"
    echo "  ./h100_runner.sh full       - Run full 8-GPU training"
    echo "  ./h100_runner.sh download   - Download data only"
    echo "  ./h100_runner.sh upload [checkpoint_dir] - Upload checkpoint"
    exit 1
fi

# Check if we have CUDA available
if ! python3 -c "import torch; print(torch.cuda.is_available())"; then
    echo "CUDA is not available. Please check your installation."
    exit 1
fi

# Report GPU information
echo "GPU Information:"
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}')"
python3 -c "import torch; print(f'GPU 0 Name: {torch.cuda.get_device_name(0)}')"
python3 -c "import torch; print(f'GPU 0 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# Install requirements if needed
install_requirements

# Process command
case "$1" in
    test)
        download_data "test"
        run_single_gpu_test
        ;;
    full)
        download_data "full"
        run_full_training
        ;;
    download)
        download_data "full"
        ;;
    upload)
        upload_checkpoints "$2"
        ;;
    *)
        echo "Unknown command: $1"
        echo "Please use: test, full, download, or upload"
        exit 1
        ;;
esac

echo "H100 runner script completed successfully" 