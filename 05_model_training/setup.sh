#!/bin/bash
# Simple setup script for ReFocused-AI training

set -e

echo "🚀 ReFocused-AI Training Setup"
echo "=============================="

# Create virtual environment
echo "📦 Creating Python environment..."
python -m venv venv
source venv/bin/activate || source venv/Scripts/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
echo "📚 Installing dependencies..."
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install datasets==2.16.1
pip install google-cloud-storage==2.13.0
pip install tqdm tensorboard

# Create directories
echo "📁 Creating directories..."
mkdir -p checkpoints logs cache

# Test installation
echo "🧪 Testing installation..."
python -c "
import torch
import transformers
import accelerate
print('✅ All packages installed successfully')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️  No GPU detected - training will run on CPU')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate environment:"
echo "  source venv/bin/activate    (Linux/Mac)"
echo "  source venv/Scripts/activate (Windows)"
echo ""
echo "To start training:"
echo "  python train.py --config test" 