#!/bin/bash
# Setup script for ReFocused-AI 1B model training environment

echo "=== ReFocused-AI Training Environment Setup ==="

# Check Python version and ensure consistency
echo "Checking Python version..."
python_version=$(python3 --version)
echo "Using $python_version"

# Check if Python version is 3.10 (recommended for all dependencies)
if [[ "$python_version" != *"Python 3.10"* ]]; then
    echo "WARNING: Recommended Python version is 3.10.x"
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

# Check if running on GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU might not be available."
else
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip==23.1.2

# Install PyTorch with CUDA support - fixed version
echo "Installing PyTorch with CUDA 12.1 support..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install packaging library first (dependency resolution)
echo "Installing packaging library..."
pip install packaging==23.2

# Install NumPy 1.x explicitly first (for wandb/tensorboard compatibility)
echo "Installing NumPy 1.x (required for wandb/tensorboard)..."
pip install numpy==1.24.0

# Install core dependencies with exact versions
echo "Installing core dependencies..."
pip install -r requirements.txt

# Install Flash Attention 2 - already in requirements.txt with fixed version
echo "Verifying Flash Attention 2 installation..."
pip show flash-attn

# Install additional monitoring tools - with fixed versions
echo "Installing monitoring tools..."
pip install wandb==0.16.0 tensorboard==2.15.0 gpustat==1.1.1

# Verify NumPy version is still 1.x
echo "Verifying NumPy version (should be 1.x)..."
python -c "import numpy as np; print(f'NumPy version: {np.__version__}'); assert np.__version__.startswith('1.'), 'NumPy 2.x detected, which is incompatible with wandb/tensorboard'"

# Setup Google Cloud authentication
echo "Setting up Google Cloud authentication..."
if [ ! -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "WARNING: GOOGLE_APPLICATION_CREDENTIALS not set. You'll need to authenticate with GCS."
    echo "Run: export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json"
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p cache

# Download tokenizer if not present
if [ ! -d "../tokenizer_1B" ]; then
    echo "Tokenizer not found in ../tokenizer_1B"
    echo "Please ensure tokenizer files are in the correct location"
else
    echo "Tokenizer found at ../tokenizer_1B"
fi

# Set environment variables for optimal performance
echo "Setting environment variables..."
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

# Create environment file
cat > .env << EOF
# ReFocused-AI Training Environment Variables
export OMP_NUM_THREADS=8
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=\${PYTHONPATH}:\$(pwd)
export WANDB_PROJECT=refocused-ai-1b
export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS}
# Lock Python version to match current
export PYTHON_VERSION="$python_version"
EOF

echo "Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo "  source .env"
echo ""
echo "To start training:"
echo "  Test run (25 files): python train.py --mode test"
echo "  Full training: python train.py --mode production" 