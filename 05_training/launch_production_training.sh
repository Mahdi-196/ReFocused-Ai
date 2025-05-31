#!/bin/bash
# Launch script for production training (all files, 8 H100 GPUs)

echo "=== ReFocused-AI Production Training Launch Script ==="
echo "Training on full dataset with 8x H100 GPUs"

# Activate environment
source venv/bin/activate
source .env

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.total --format=csv
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs"

if [ $NUM_GPUS -lt 8 ]; then
    echo "WARNING: Expected 8 GPUs but found $NUM_GPUS"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Configure for optimal H100 performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1

# NCCL settings for optimal multi-GPU communication
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0

# Login to Weights & Biases
echo "Setting up Weights & Biases..."
wandb login --relogin || echo "W&B login skipped"

# Create hostfile for DeepSpeed
echo "Creating hostfile..."
echo "localhost slots=8" > hostfile

# Launch distributed training with DeepSpeed
echo "Starting production training run..."
deepspeed \
    --hostfile hostfile \
    --num_nodes 1 \
    --num_gpus 8 \
    --master_port 29500 \
    train.py \
    --mode production \
    2>&1 | tee logs/production_training_$(date +%Y%m%d_%H%M%S).log

echo "Production training completed!"
echo "Check logs/ directory for training logs"
echo "Checkpoints are uploaded to GCS: gs://refocused-ai/Checkpoints/" 