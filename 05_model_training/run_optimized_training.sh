#!/bin/bash

# ReFocused-AI Optimized Training Script
# This script demonstrates running training with all optimizations enabled

set -e  # Exit on error

echo "🚀 ReFocused-AI Optimized Training Setup"
echo "========================================"

# Configuration
MODE=${1:-"test"}  # test or production
PROFILE=${2:-"false"}  # Enable profiling
MAX_STEPS=${3:-""}  # Override max steps

echo "Training mode: $MODE"
echo "Profiling enabled: $PROFILE"
if [ ! -z "$MAX_STEPS" ]; then
    echo "Max steps override: $MAX_STEPS"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p cache
mkdir -p preprocessed_cache
mkdir -p logs
mkdir -p checkpoints

# Check system requirements
echo "🔍 Checking system requirements..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import accelerate; print(f'Accelerate version: {accelerate.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

if python -c "import torch; print('CUDA available:', torch.cuda.is_available())"; then
    python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
    python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
fi

# Test the optimized dataloader first
echo "🧪 Testing optimized dataloader..."
python test_dataloader.py

if [ $? -ne 0 ]; then
    echo "❌ Dataloader test failed. Please check the configuration."
    exit 1
fi

echo "✅ Dataloader test passed!"

# Prepare training arguments
TRAIN_ARGS="--mode $MODE"

if [ "$PROFILE" = "true" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --profile"
fi

if [ ! -z "$MAX_STEPS" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --max-steps $MAX_STEPS"
fi

echo "📋 Training arguments: $TRAIN_ARGS"

# Start monitoring in background (if requested)
START_MONITOR=${START_MONITOR:-"false"}
if [ "$START_MONITOR" = "true" ]; then
    echo "📊 Starting training monitor in background..."
    python scripts/monitor_training.py --refresh 5 &
    MONITOR_PID=$!
    echo "Monitor PID: $MONITOR_PID"
    
    # Function to cleanup monitor on exit
    cleanup() {
        echo "🛑 Stopping monitor..."
        kill $MONITOR_PID 2>/dev/null || true
    }
    trap cleanup EXIT
fi

# Run training with optimizations
echo "🏋️  Starting optimized training..."
echo "Command: python train.py $TRAIN_ARGS"
echo "========================================"

# Use accelerate launch for multi-GPU if available
if command -v accelerate &> /dev/null; then
    echo "🔀 Using Accelerate for distributed training..."
    accelerate launch train.py $TRAIN_ARGS
else
    echo "⚠️  Accelerate not found, running single process..."
    python train.py $TRAIN_ARGS
fi

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Show final status
    echo "📊 Final training status:"
    python scripts/monitor_training.py --once
    
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
fi

echo "🏁 Training script completed."
exit $TRAIN_EXIT_CODE 