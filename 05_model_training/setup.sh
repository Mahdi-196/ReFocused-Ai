#!/bin/bash
# ReFocused-AI Training Setup Script
# This script sets up the complete environment for training a 1-1.2B parameter model

set -e  # Exit on error

echo "=== ReFocused-AI Training Setup ==="
echo "Setting up environment for FSDP training on H100..."

# Create virtual environment with Python 3.10
echo "Creating virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (for H100)
echo "Installing PyTorch with CUDA support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face libraries
echo "Installing Hugging Face libraries..."
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install datasets==2.16.1

# Install additional dependencies
echo "Installing additional dependencies..."
pip install numpy==1.26.3
pip install scipy==1.11.4
pip install tqdm==4.66.1
pip install wandb==0.16.2
pip install tensorboard==2.15.1
pip install google-cloud-storage==2.13.0
pip install sentencepiece==0.1.99
pip install protobuf==4.25.2

# Create necessary directories
echo "Creating directory structure..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p cache
mkdir -p data

# Set environment variables
echo "Setting environment variables..."
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Configure accelerate
echo "Configuring Accelerate..."
bash configure_accelerate.sh

# Create activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
echo "Environment activated. You can now run training scripts."
EOF

chmod +x activate_env.sh

echo "=== Setup Complete ==="
echo "To activate the environment in future sessions, run: source activate_env.sh"
echo ""
echo "Next steps:"
echo "1. Run test training: bash run_test_training.sh"
echo "2. Run production training: bash run_production_training.sh" 