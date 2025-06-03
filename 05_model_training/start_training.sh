#!/bin/bash

# ReFocused-AI Training Startup Script
# Sets up authentication and starts training with background checkpoint uploading

echo "🚀 ReFocused-AI Training Startup"
echo "=================================="

# Check if we're in the training directory
if [[ ! -f "train.py" ]]; then
    echo "❌ Please run this script from the 05_model_training directory"
    exit 1
fi

# Set up authentication
echo "🔐 Setting up Google Cloud authentication..."
export GOOGLE_APPLICATION_CREDENTIALS="./credentials/black-dragon-461023-t5-93452a49f86b.json"
export GOOGLE_CLOUD_PROJECT="black-dragon-461023-t5"

# Verify credentials exist
if [[ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
    echo "❌ Credentials file not found: $GOOGLE_APPLICATION_CREDENTIALS"
    echo "   Please ensure your service account key is in the credentials folder"
    exit 1
fi

echo "✅ Authentication configured"
echo "   Project: $GOOGLE_CLOUD_PROJECT"
echo "   Credentials: $GOOGLE_APPLICATION_CREDENTIALS"

# Parse command line arguments
CONFIG_TYPE="test"
MAX_STEPS=""
RESUME_CHECKPOINT=""
NO_BACKGROUND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_TYPE="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="--max-steps $2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="--resume $2"
            shift 2
            ;;
        --no-background-upload)
            NO_BACKGROUND="--no-background-upload"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config test|production] [--max-steps N] [--resume checkpoint_name] [--no-background-upload]"
            exit 1
            ;;
    esac
done

# Display training configuration
echo ""
echo "🎯 Training Configuration:"
echo "   Config type: $CONFIG_TYPE"
echo "   Background uploads: $([ -z "$NO_BACKGROUND" ] && echo "ENABLED" || echo "DISABLED")"
echo "   Bucket: refocused-ai/Checkpoints/"

if [[ -n "$MAX_STEPS" ]]; then
    echo "   Max steps override: ${MAX_STEPS#--max-steps }"
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "   Resume from: ${RESUME_CHECKPOINT#--resume }"
fi

# Check for GPU
echo ""
echo "🔍 System Check:"
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
else
    echo "⚠️  No NVIDIA GPU detected - training will use CPU (very slow)"
fi


# Start training
echo ""
echo "🚀 Starting Training..."
echo "=================================="

# Build the training command
TRAIN_CMD="python train.py --config $CONFIG_TYPE $MAX_STEPS $RESUME_CHECKPOINT $NO_BACKGROUND"

echo "Running: $TRAIN_CMD"
echo ""

# Execute training with proper signal handling
trap 'echo "🛑 Training interrupted. Waiting for uploads to complete..."; wait; exit 130' INT

$TRAIN_CMD

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Checkpoints uploaded to: gs://refocused-ai/Checkpoints/"
else
    echo ""
    echo "❌ Training failed. Check logs above for details."
    exit 1
fi 