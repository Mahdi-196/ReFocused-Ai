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

# Activate virtual environment if it exists
if [[ -d "venv" ]]; then
    echo "🔄 Activating virtual environment..."
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found (venv directory missing)"
fi

# Set up authentication using an absolute path
# Get the absolute path to the directory where this script is located
SCRIPT_DIR_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CRED_FILE_NAME="black-dragon-461023-t5-93452a49f86b.json"
# Construct the absolute path to the credentials file
ABS_CRED_PATH="${SCRIPT_DIR_PATH}/credentials/${CRED_FILE_NAME}"

echo "🔐 Setting up Google Cloud authentication..."
export GOOGLE_APPLICATION_CREDENTIALS="${ABS_CRED_PATH}"
export GOOGLE_CLOUD_PROJECT="black-dragon-461023-t5"

# Verify credentials exist (this will now use the absolute path)
if [[ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
    echo "❌ Credentials file not found: $GOOGLE_APPLICATION_CREDENTIALS"
    echo "   (Checked absolute path based on script location)"
    echo "   Please ensure your service account key is in the 'credentials' folder relative to the script"
    exit 1
fi

echo "✅ Authentication configured"
echo "   Project: $GOOGLE_CLOUD_PROJECT"
echo "   Credentials (absolute path): $GOOGLE_APPLICATION_CREDENTIALS" # Now explicitly states absolute path

# Test GCS permissions
echo "🧪 Testing Google Cloud Storage permissions..."
python -c "
import os
from google.cloud import storage
try:
    client = storage.Client()
    bucket = client.bucket('refocused-ai')
    # Test basic read access
    list(bucket.list_blobs(max_results=1))
    print('✅ GCS read access confirmed')
    
    # Test bucket metadata access (needed for uploads)
    try:
        bucket.exists()
        print('✅ GCS bucket access confirmed')
    except Exception as e:
        print(f'⚠️  Limited GCS permissions detected: {e}')
        print('   Training will proceed but uploads may fail')
        print('   See fix_permissions.md for permission setup instructions')
        
except Exception as e:
    print(f'❌ GCS authentication failed: {e}')
    print('   Check your service account permissions')
    print('   See fix_permissions.md for troubleshooting')
    exit(1)
"

if [[ $? -ne 0 ]]; then
    echo "❌ GCS authentication test failed"
    echo "   Please check your credentials and permissions"
    echo "   See fix_permissions.md for help"
    exit 1
fi

# Parse command line arguments
CONFIG_TYPE="test"
MAX_STEPS=""
RESUME_CHECKPOINT=""
NO_BACKGROUND=""
GPU_COUNT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_TYPE="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --no-background-upload)
            NO_BACKGROUND="true"
            shift
            ;;
        --gpus)
            GPU_COUNT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config test|production] [--max-steps N] [--resume checkpoint_name] [--no-background-upload] [--gpus N]"
            echo ""
            echo "Examples:"
            echo "  # Single GPU training"
            echo "  $0 --config test"
            echo ""
            echo "  # Multi-GPU training (2 GPUs)"
            echo "  $0 --config test --gpus 2"
            echo ""
            echo "  # Production training (8 GPUs)"
            echo "  $0 --config production --gpus 8"
            exit 1
            ;;
    esac
done

# Display training configuration
echo ""
echo "🎯 Training Configuration:"
echo "   Config type: $CONFIG_TYPE"
echo "   Background uploads: $([ -z "$NO_BACKGROUND" ] && echo "ENABLED" || echo "DISABLED")"
echo "   Bucket: gs://refocused-ai/checkpoints"

if [[ -n "$GPU_COUNT" ]]; then
    echo "   GPU count: $GPU_COUNT"
else
    echo "   GPU count: 1 (single GPU)"
fi

if [[ -n "$MAX_STEPS" ]]; then
    echo "   Max steps override: $MAX_STEPS"
fi

if [[ -n "$RESUME_CHECKPOINT" ]]; then
    echo "   Resume from: $RESUME_CHECKPOINT"
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


# Check for multi-GPU requirements
if [[ -n "$GPU_COUNT" ]] && [[ "$GPU_COUNT" -gt 1 ]]; then
    echo "🔧 Checking Accelerate configuration for multi-GPU training..."
    if ! accelerate config list &> /dev/null; then
        echo "⚠️  Accelerate is not configured. Running 'accelerate config' now..."
        echo "   (Press Enter to use defaults for quick setup)"
        accelerate config
    else
        echo "✅ Accelerate configuration found"
    fi
fi

# Start training
echo ""
echo "🚀 Starting Training..."
echo "=================================="

# Build the training command arguments
ARGS=("--config" "$CONFIG_TYPE")
[[ -n "$MAX_STEPS" ]] && ARGS+=("--max-steps" "$MAX_STEPS")
[[ -n "$RESUME_CHECKPOINT" ]] && ARGS+=("--resume" "$RESUME_CHECKPOINT")
[[ -n "$NO_BACKGROUND" ]] && ARGS+=("--no-background-upload")

# Build the complete command
if [[ -n "$GPU_COUNT" ]] && [[ "$GPU_COUNT" -gt 1 ]]; then
    TRAIN_CMD=("accelerate" "launch" "--nproc_per_node=$GPU_COUNT" "train.py" "${ARGS[@]}")
else
    TRAIN_CMD=("python" "train.py" "${ARGS[@]}")
fi

echo "Running: ${TRAIN_CMD[*]}"
echo ""

# Execute training with proper signal handling
trap 'echo "🛑 Training interrupted. Exiting."; exit 130' INT

"${TRAIN_CMD[@]}"

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "🎉 Training completed successfully!"
    echo "📁 Checkpoints uploaded to: gs://refocused-ai/checkpoints"
else
    echo ""
    echo "❌ Training failed. Check logs above for details."
    exit 1
fi