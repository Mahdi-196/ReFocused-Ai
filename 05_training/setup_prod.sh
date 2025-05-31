#!/usr/bin/env bash
set -e

echo "==================================="
echo "ReFocused-AI Production Setup Script"
echo "==================================="

# 1. Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo "ERROR: No virtual environment activated."
  echo "Please activate your virtualenv first: source venv/bin/activate"
  exit 1
fi

# 2. Create production data directory
DATA_DIR="data_full"
echo "Creating production data directory: $DATA_DIR"
mkdir -p $DATA_DIR

# 3. Check if gsutil is available
if ! command -v gsutil &> /dev/null; then
    echo "ERROR: gsutil not found. Install Google Cloud SDK first."
    exit 1
fi

# 4. Load environment variables
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
else
    echo "WARNING: No .env file found. Make sure GOOGLE_APPLICATION_CREDENTIALS is set."
fi

# 5. Check available space
echo "Checking available disk space..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 100 ]; then
    echo "WARNING: Less than 100GB available. You may need more space for the full dataset."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 6. Count files to download
echo "Counting files in gs://refocused-ai/tokenized_data/..."
TOTAL_FILES=$(gsutil ls "gs://refocused-ai/tokenized_data/*.npz" 2>/dev/null | wc -l) || {
    echo "ERROR: Failed to list files from GCS. Check your credentials and bucket access."
    exit 1
}

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "ERROR: No tokenized files found in gs://refocused-ai/tokenized_data/"
    exit 1
fi

echo "Found $TOTAL_FILES tokenized files to download."
echo "This represents approximately 21-22 billion tokens."

# 7. Estimate download size and time
echo ""
echo "Estimating download size..."
# Get size of first few files to estimate
SAMPLE_SIZE=$(gsutil du -s "gs://refocused-ai/tokenized_data/*.npz" 2>/dev/null | head -5 | awk '{sum+=$1} END {print sum/5/1024/1024/1024}') || SAMPLE_SIZE=1
ESTIMATED_TOTAL_SIZE=$(echo "$SAMPLE_SIZE * $TOTAL_FILES" | bc -l | xargs printf "%.1f")
echo "Estimated total size: ${ESTIMATED_TOTAL_SIZE}GB"

# 8. Confirm download
echo ""
echo "==================================="
echo "Ready to download:"
echo "- Files: $TOTAL_FILES"
echo "- Estimated size: ${ESTIMATED_TOTAL_SIZE}GB"
echo "- Destination: $DATA_DIR"
echo "==================================="
read -p "Proceed with download? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Download cancelled."
    exit 0
fi

# 9. Download all tokenized files with parallel transfers
echo ""
echo "Starting parallel download (using gsutil -m for multi-threading)..."
echo "This may take a while depending on your connection speed..."

START_TIME=$(date +%s)

# Use gsutil with multi-threading and retry
gsutil -m cp -r "gs://refocused-ai/tokenized_data/*.npz" "$DATA_DIR/" || {
    echo "ERROR: Download failed. Attempting retry..."
    # Retry with resumable uploads
    gsutil -m cp -c -r "gs://refocused-ai/tokenized_data/*.npz" "$DATA_DIR/" || {
        echo "ERROR: Download failed after retry."
        exit 1
    }
}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

# 10. Verify download
echo ""
echo "Verifying downloaded files..."
DOWNLOADED_COUNT=$(ls $DATA_DIR/*.npz 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh $DATA_DIR | cut -f1)

if [ "$DOWNLOADED_COUNT" -ne "$TOTAL_FILES" ]; then
    echo "WARNING: Downloaded $DOWNLOADED_COUNT files, expected $TOTAL_FILES"
    echo "Some files may have failed to download."
else
    echo "✓ Successfully downloaded all $DOWNLOADED_COUNT files"
fi

# 11. Create file manifest
echo "Creating file manifest..."
ls $DATA_DIR/*.npz | sort > $DATA_DIR/manifest.txt
echo "✓ Manifest created at $DATA_DIR/manifest.txt"

# 12. Summary
echo ""
echo "==================================="
echo "✅ Production data setup complete!"
echo ""
echo "Summary:"
echo "- Downloaded files: $DOWNLOADED_COUNT"
echo "- Total size: $TOTAL_SIZE"
echo "- Download time: $DURATION_MIN minutes"
echo "- Data directory: $DATA_DIR"
echo ""
echo "Next steps:"
echo "1. Review the manifest: cat $DATA_DIR/manifest.txt"
echo "2. Run production training: bash prod_train.sh"
echo "===================================" 