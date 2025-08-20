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

 

# Parse command line arguments
# Parse command line arguments
CONFIG_TYPE="test"
MAX_STEPS=""
RESUME_CHECKPOINT=""
NO_BACKGROUND=""
GPU_COUNT=""
GCS_CREDENTIALS_PATH=""
GCP_PROJECT_ID=""

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
        --gcs-credentials)
            GCS_CREDENTIALS_PATH="$2"
            shift 2
            ;;
        --gcp-project)
            GCP_PROJECT_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--config test|production] [--max-steps N] [--resume checkpoint_name] [--no-background-upload] [--gpus N] [--gcs-credentials /abs/path/key.json] [--gcp-project PROJECT_ID]"
            echo ""
            echo "Examples:"
            echo "  # Single GPU training"
            echo "  $0 --config test --gcs-credentials /abs/key.json --gcp-project my-project"
            echo ""
            echo "  # Multi-GPU training (2 GPUs)"
            echo "  $0 --config test --gpus 2 --gcs-credentials /abs/key.json"
            echo ""
            echo "  # Production training (8 GPUs)"
            echo "  $0 --config production --gpus 8 --gcs-credentials /abs/key.json --gcp-project my-project"
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
if [[ -n "$GCS_CREDENTIALS_PATH" ]]; then
    echo "   Credentials: $GCS_CREDENTIALS_PATH"
fi
if [[ -n "$GCP_PROJECT_ID" ]]; then
    echo "   GCP Project: $GCP_PROJECT_ID"
fi

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
[[ -n "$GCS_CREDENTIALS_PATH" ]] && ARGS+=("--gcs-credentials" "$GCS_CREDENTIALS_PATH")
[[ -n "$GCP_PROJECT_ID" ]] && ARGS+=("--gcp-project" "$GCP_PROJECT_ID")

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