#!/bin/bash

# Training Parameter Analysis Launcher
# This script runs the bucket analysis to determine optimal training steps

echo "🚀 REFOCUSED-AI TRAINING PARAMETER ANALYSIS"
echo "=========================================="
echo ""

# Check if virtual environment exists and activate it
if [ -f "../venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source ../venv/bin/activate
elif [ -f "./venv/bin/activate" ]; then
    echo "📦 Activating virtual environment..."
    source ./venv/bin/activate
else
    echo "⚠️  No virtual environment found, using system Python"
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."
python -c "import google.cloud.storage, numpy, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install google-cloud-storage numpy tqdm
fi

# Run the analysis
echo "🔬 Starting bucket analysis..."
echo ""

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Run the analysis script
python analyze_training_parameters.py

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ Analysis completed successfully!"
    echo ""
    echo "📋 NEXT STEPS:"
    echo "1. Check the training_analysis_report.json for detailed results"
    echo "2. Update your configs/training_config.py with the recommended values"
    echo "3. Run ./start_training.sh --config production --gpus 2"
else
    echo "❌ Analysis failed with exit code $exit_code"
    echo "Please check the error messages above."
fi 