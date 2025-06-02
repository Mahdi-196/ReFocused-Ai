#!/bin/bash
# Simple training script for ReFocused-AI

set -e

CONFIG=${1:-"test"}  # Default to test config
MAX_STEPS=${2:-""}   # Optional max steps override

echo "🚀 Starting ReFocused-AI Training"
echo "Configuration: $CONFIG"

# Activate environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || echo "⚠️  Could not activate venv"
fi

# Check GPU status
echo "🔍 Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  No GPU - training on CPU (will be slow)')
"

# Build command
CMD="python train.py --config $CONFIG"
if [ ! -z "$MAX_STEPS" ]; then
    CMD="$CMD --max-steps $MAX_STEPS"
fi

echo "Running: $CMD"
echo ""

# Run training
$CMD 