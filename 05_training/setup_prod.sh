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

# 3. Check local source directory
LOCAL_SRC=".."
if [ ! -d "$LOCAL_SRC" ]; then
    echo "ERROR: Local folder '$LOCAL_SRC' not found."
    exit 1
fi

# 4. Load environment variables
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
else
    echo "INFO: No .env file found. Continuing without it."
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

# 6. Count files to copy
echo "Counting files in $LOCAL_SRC..."
TOTAL_FILES=$(ls "$LOCAL_SRC"/*.npz 2>/dev/null | wc -l) 
if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "ERROR: No tokenized files found in $LOCAL_SRC"
    exit 1
fi

echo "Found $TOTAL_FILES tokenized files to copy."
echo "This represents approximately 21-22 billion tokens."

# 7. Estimate copy size
echo ""
echo "Estimating disk space required..."
TOTAL_SIZE=$(du -sh "$LOCAL_SRC"/*.npz 2>/dev/null | awk '{sum+=$1} END {print sum}')
echo "Estimated total size: ${TOTAL_SIZE}"

# 8. Confirm copy
echo ""
echo "==================================="
echo "Ready to copy files:"
echo "- Files: $TOTAL_FILES"
echo "- Estimated size: ${TOTAL_SIZE}"
echo "- Source: $LOCAL_SRC"
echo "- Destination: $DATA_DIR"
echo "==================================="
read -p "Proceed with copy? (y/N) " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Copy cancelled."
    exit 0
fi

# 9. Copy all tokenized files
echo ""
echo "Starting file copy..."
echo "This may take a while depending on your disk speed..."

START_TIME=$(date +%s)

# Use cp with wildcard
echo "Copying files from $LOCAL_SRC to $DATA_DIR..."
cp "$LOCAL_SRC"/*.npz "$DATA_DIR/" || {
    echo "ERROR: Copy failed."
    exit 1
}

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
DURATION_MIN=$((DURATION / 60))

# 10. Verify copy
echo ""
echo "Verifying copied files..."
COPIED_COUNT=$(ls $DATA_DIR/*.npz 2>/dev/null | wc -l)
COPIED_SIZE=$(du -sh $DATA_DIR | cut -f1)

if [ "$COPIED_COUNT" -ne "$TOTAL_FILES" ]; then
    echo "WARNING: Copied $COPIED_COUNT files, expected $TOTAL_FILES"
    echo "Some files may have failed to copy."
else
    echo "✓ Successfully copied all $COPIED_COUNT files"
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
echo "- Copied files: $COPIED_COUNT"
echo "- Total size: $COPIED_SIZE"
echo "- Copy time: $DURATION_MIN minutes"
echo "- Data directory: $DATA_DIR"
echo ""
echo "Next steps:"
echo "1. Review the manifest: cat $DATA_DIR/manifest.txt"
echo "2. Run production training: bash prod_train.sh"
echo "===================================" 