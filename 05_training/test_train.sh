#!/usr/bin/env bash
set -e

echo "==================================="
echo "ReFocused-AI Test Training Script"
echo "==================================="

# 1. Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo "ERROR: No virtual environment activated."
  echo "Please activate your virtualenv first: source venv/bin/activate"
  exit 1
fi

# 2. Create test data directory
DATA_DIR="data_test"
echo "Creating test data directory: $DATA_DIR"
rm -rf $DATA_DIR
mkdir -p $DATA_DIR

# 3. Copy first 25 tokenized files from local root folder
echo "Copying first 25 tokenized files from ../ ..."
LOCAL_SRC=".."
if [ ! -d "$LOCAL_SRC" ]; then
    echo "ERROR: Local folder '$LOCAL_SRC' not found."
    exit 1
fi

ls "$LOCAL_SRC"/*.npz 2>/dev/null | head -n 25 | while read -r file; do
    echo "Copying: $file"
    cp "$file" "$DATA_DIR/" || {
        echo "WARNING: failed to copy $file; continuing..."
    }
done

FILE_COUNT=$(ls "$DATA_DIR"/*.npz 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No .npz files found in '$DATA_DIR/'."
    exit 1
fi
echo "✓ Copied $FILE_COUNT files successfully"

# 4. Set up environment variables
echo "Setting up environment variables..."

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
else
    echo "WARNING: No .env file found. If you don't need GCS credentials, you can ignore this."
fi

# 5. Set CUDA device (single GPU for test)
# Use CUDA device specified by ENV var, or default to 0
export CUDA_VISIBLE_DEVICES="${TEST_GPU:-0}"
echo "Using CUDA device: $CUDA_VISIBLE_DEVICES"

# 6. Create logs directory
LOG_DIR="logs/test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR
echo "Logs will be saved to: $LOG_DIR"

# 7. Run training in test mode
echo ""
echo "Starting test training run..."
echo "==================================="

python train.py \
    --mode test \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR \
    --max_files 25 \
    --checkpoint_interval 5 \
    --log_interval 10 \
    --eval_interval 100 \
    2>&1 | tee $LOG_DIR/training.log

# 8. Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "✅ Test training completed successfully!"
    echo ""
    echo "Results:"
    echo "- Logs: $LOG_DIR"
    echo "- TensorBoard: tensorboard --logdir=$LOG_DIR"
    echo "- Checkpoints (local): $(pwd)/checkpoints/"
    echo ""
    echo "Make sure to save or inspect local checkpoints in the checkpoints directory"
    echo "==================================="
else
    echo ""
    echo "==================================="
    echo "❌ Test training failed. Check logs at: $LOG_DIR/training.log"
    echo "==================================="
    exit 1
fi 