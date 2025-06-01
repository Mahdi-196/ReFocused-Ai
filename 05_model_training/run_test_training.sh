#!/bin/bash
# Run test training on 1 GPU with 25 files

set -e

echo "=== ReFocused-AI Test Training ==="
echo "Running test training on 1 H100 GPU with 25 files..."

# Activate environment
source activate_env.sh

# Set environment variables for single GPU
export CUDA_VISIBLE_DEVICES=0
export ACCELERATE_MIXED_PRECISION=bf16

# Run training
echo "Starting test training..."
python train.py --mode test --seed 42

echo "Test training complete!"
echo "Check logs in ./logs/test_run_25_files"
echo "Check tensorboard: tensorboard --logdir=./logs" 