#!/bin/bash
# Run production training on all available GPUs

set -e

echo "=== ReFocused-AI Production Training ==="
echo "Running production training on all available GPUs..."

# Activate environment
source activate_env.sh

# For multi-GPU training with accelerate
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Adjust based on available GPUs
fi

# Count available GPUs
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPUs: $CUDA_VISIBLE_DEVICES"

# Configure accelerate for multi-GPU
export ACCELERATE_MIXED_PRECISION=bf16

# Run distributed training with accelerate
echo "Starting production training..."
accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision bf16 \
    train.py --mode production --seed 42

echo "Production training complete!"
echo "Check logs in ./logs/production_run_full"
echo "Check tensorboard: tensorboard --logdir=./logs" 