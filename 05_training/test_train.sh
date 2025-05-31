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

# 3. Download first 25 tokenized files from GCS
echo "Downloading first 25 tokenized files from gs://refocused-ai/tokenized_data/..."

# First, check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "ERROR: gsutil not found. Install Google Cloud SDK first."
    exit 1
fi

# Download files with error handling
echo "Listing available files..."
gsutil ls "gs://refocused-ai/tokenized_data/shard_*.npz" > /tmp/available_files.txt 2>/dev/null || {
    echo "ERROR: Failed to list files from GCS. Check your credentials and bucket access."
    exit 1
}

# Download first 25 files
head -n 25 /tmp/available_files.txt | while read -r file; do
    echo "Downloading: $file"
    gsutil cp "$file" "$DATA_DIR/" || {
        echo "WARNING: Failed to download $file. Continuing..."
    }
done

# Verify we have files
FILE_COUNT=$(ls $DATA_DIR/*.npz 2>/dev/null | wc -l)
if [ "$FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No .npz files downloaded. Check GCS access and bucket contents."
    exit 1
fi
echo "✓ Downloaded $FILE_COUNT files successfully"

# 4. Set up environment variables
echo "Setting up environment variables..."

# Load .env file if it exists
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
else
    echo "WARNING: No .env file found. Make sure GOOGLE_APPLICATION_CREDENTIALS is set."
fi

# Verify Google credentials
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "WARNING: GOOGLE_APPLICATION_CREDENTIALS not set. GCS uploads may fail."
fi

# 5. Set CUDA device (single GPU for test)
export CUDA_VISIBLE_DEVICES=0
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
    echo "- Checkpoints: gs://refocused-ai/Checkpoints/"
    echo ""
    echo "To view uploaded checkpoints:"
    echo "gsutil ls -r gs://refocused-ai/Checkpoints/"
    echo "==================================="
else
    echo ""
    echo "==================================="
    echo "❌ Test training failed. Check logs at: $LOG_DIR/training.log"
    echo "==================================="
    exit 1
fi 