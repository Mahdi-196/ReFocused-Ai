#!/bin/bash
# Launch script for test training run (25 files, single GPU)

echo "=== ReFocused-AI Test Training Launch Script ==="
echo "Training on 25 files with single H100 GPU"

# Activate environment
source venv/bin/activate
source .env

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Set single GPU mode
export CUDA_VISIBLE_DEVICES=0

# Configure for optimal H100 performance
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1

# Login to Weights & Biases (optional)
echo "Setting up Weights & Biases..."
wandb login --relogin || echo "W&B login skipped"

# Launch training
echo "Starting test training run..."
python train.py \
    --mode test \
    2>&1 | tee logs/test_training_$(date +%Y%m%d_%H%M%S).log

echo "Test training completed!"
echo "Check logs/ directory for training logs"
echo "Checkpoints are uploaded to GCS: gs://refocused-ai/Checkpoints/" 