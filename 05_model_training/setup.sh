#!/bin/bash
# Complete Setup Script for ReFocused-AI Training Pipeline
# This script sets up everything needed for training from scratch

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 ReFocused-AI Complete Training Setup${NC}"
echo "=============================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "mac"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo -e "${BLUE}Detected OS: $OS${NC}"

# Check if we're in the right directory
if [[ ! -f "train.py" ]]; then
    echo -e "${RED}❌ Please run this script from the 05_model_training directory${NC}"
    exit 1
fi

echo -e "\n${YELLOW}Step 1: System Prerequisites${NC}"
echo "=============================="

# Check Python
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.8+ first.${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
echo -e "✅ Python found: $PYTHON_VERSION"

# Check if Python version is 3.8+
if $PYTHON_CMD -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "✅ Python version is compatible"
else
    echo -e "${RED}❌ Python 3.8+ required. Current version: $PYTHON_VERSION${NC}"
    exit 1
fi

# Check pip
if ! command_exists pip && ! command_exists pip3; then
    echo -e "${RED}❌ pip not found. Please install pip first.${NC}"
    exit 1
fi

echo -e "✅ pip found"

# Check for CUDA
if command_exists nvidia-smi; then
    echo -e "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1
    HAS_GPU=true
else
    echo -e "${YELLOW}⚠️  No NVIDIA GPU detected - training will use CPU (very slow)${NC}"
    HAS_GPU=false
fi

echo -e "\n${YELLOW}Step 2: Virtual Environment Setup${NC}"
echo "==================================="

# Remove existing venv if it exists
if [[ -d "venv" ]]; then
    echo "🗑️  Removing existing virtual environment..."
    rm -rf venv
fi

# Create virtual environment
echo "📦 Creating Python virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
if [[ "$OS" == "windows" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo -e "✅ Virtual environment created and activated"

# Set pip command to venv pip (guaranteed to be in venv now)
PIP_CMD="pip"

# Upgrade pip
echo "📦 Upgrading pip..."
$PIP_CMD install --upgrade pip

echo -e "\n${YELLOW}Step 3: PyTorch Installation${NC}"
echo "============================="

if [[ "$HAS_GPU" == true ]]; then
    echo "🔥 Installing PyTorch with CUDA support..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
    echo "💻 Installing PyTorch CPU-only version..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "✅ PyTorch installed"

# Verify PyTorch 2.0+ for torch.compile support
echo "🔧 Verifying PyTorch version for torch.compile support..."
if $PYTHON_CMD -c "import torch; import sys; sys.exit(0 if torch.__version__ >= '2.0.0' else 1)"; then
    echo -e "✅ PyTorch 2.0+ detected - torch.compile available"
else
    TORCH_VERSION=$($PYTHON_CMD -c "import torch; print(torch.__version__)")
    echo -e "${YELLOW}⚠️  PyTorch $TORCH_VERSION detected. torch.compile requires PyTorch 2.0+${NC}"
    echo -e "${YELLOW}   Model compilation will be disabled but training will work${NC}"
fi

echo -e "\n${YELLOW}Step 4: Core Dependencies${NC}"
echo "=========================="

echo "📚 Installing core ML dependencies..."
$PIP_CMD install transformers==4.36.2
$PIP_CMD install accelerate==0.25.0
$PIP_CMD install datasets==2.16.1

echo "☁️  Installing Google Cloud dependencies..."
$PIP_CMD install google-cloud-storage==2.13.0
$PIP_CMD install google-auth==2.22.0
$PIP_CMD install google-resumable-media==2.5.0

echo "📊 Installing monitoring and utilities..."
$PIP_CMD install tqdm tensorboard wandb
$PIP_CMD install numpy>=1.24.0 pandas>=2.0.0
$PIP_CMD install requests beautifulsoup4
$PIP_CMD install psutil memory_profiler

echo -e "✅ All dependencies installed"

echo -e "\n${YELLOW}Step 5: Directory Structure${NC}"
echo "============================"

echo "📁 Creating required directories..."
mkdir -p checkpoints logs cache preprocessed_cache
mkdir -p data/training/shards data/training/simple
mkdir -p credentials

echo -e "✅ Directory structure created"

echo -e "\n${YELLOW}Step 6: Authentication Setup${NC}"
echo "=============================="

CRED_FILE="./credentials/black-dragon-461023-t5-93452a49f86b.json"
if [[ -f "$CRED_FILE" ]]; then
    echo -e "✅ Credentials file found: $CRED_FILE"
    export GOOGLE_APPLICATION_CREDENTIALS="$CRED_FILE"
    export GOOGLE_CLOUD_PROJECT="black-dragon-461023-t5"
    echo -e "✅ Environment variables set"
else
    echo -e "${YELLOW}⚠️  Credentials file not found: $CRED_FILE${NC}"
    echo "   Please place your service account key in the credentials folder"
    echo "   and name it: black-dragon-461023-t5-93452a49f86b.json"
fi

echo -e "\n${YELLOW}Step 7: System Validation${NC}"
echo "=========================="

echo "🧪 Testing installation..."
$PYTHON_CMD -c "
import sys
import torch
import transformers
import accelerate
import datasets
from google.cloud import storage

print('✅ Core packages imported successfully')
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'Accelerate: {accelerate.__version__}')
print(f'Datasets: {datasets.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    
    # Test mixed precision support
    if torch.cuda.is_bf16_supported():
        print('✅ BF16 supported (recommended)')
    else:
        print('⚠️  BF16 not supported, will use FP16')
else:
    print('⚠️  No GPU detected - training will be very slow')

print('✅ Installation validation complete')
"

echo -e "\n${YELLOW}Step 8: Configuration Optimization Check${NC}"
echo "========================================"

echo "🔧 Validating optimized configurations..."
$PYTHON_CMD -c "
from configs.training_config import get_training_config
import torch.backends.cudnn as cudnn

# Test configuration loading
test_config = get_training_config('test')
prod_config = get_training_config('production')

print('✅ Training configurations loaded')
print(f'Test config - Batch: {test_config.per_device_train_batch_size}, Grad Acc: {test_config.gradient_accumulation_steps}')
print(f'Prod config - Batch: {prod_config.per_device_train_batch_size}, Grad Acc: {prod_config.gradient_accumulation_steps}')
print(f'cuDNN benchmark enabled: {cudnn.benchmark}')
print('✅ Performance optimizations verified')
"

echo -e "\n${YELLOW}Step 8.5: Accelerate Configuration${NC}"
echo "==================================="

echo "🚀 Setting up Accelerate for multi-GPU training..."
echo -e "${BLUE}ℹ️  For multi-GPU training, you'll need to configure Accelerate:${NC}"
echo "   accelerate config    # one-time setup"
echo ""
echo "Example multi-GPU commands:"
echo "   accelerate launch --nproc_per_node=2 train.py --config test"
echo "   accelerate launch --nproc_per_node=8 train.py --config production"
echo ""
read -p "Do you want to run 'accelerate config' now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    accelerate config
    echo -e "✅ Accelerate configuration completed"
else
    echo -e "${BLUE}ℹ️  Skipping accelerate config. Run later with: accelerate config${NC}"
fi

echo -e "\n${YELLOW}Step 9: Data Download${NC}"
echo "==================="

# Check credentials before attempting download
if [[ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]]; then
    echo -e "${YELLOW}⚠️  No Google credentials found - download may fail${NC}"
    echo "   Set GOOGLE_APPLICATION_CREDENTIALS or skip download"
fi

read -p "Do you want to download training data now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Downloading training data..."
    $PYTHON_CMD download_training_data.py
    
    if [[ $? -eq 0 ]]; then
        echo -e "✅ Training data downloaded successfully"
    else
        echo -e "${YELLOW}⚠️  Data download failed - you can run it later with:${NC}"
        echo "   python download_training_data.py"
    fi
else
    echo -e "${BLUE}ℹ️  Skipping data download. Run later with:${NC}"
    echo "   python download_training_data.py"
fi

echo -e "\n${YELLOW}Step 10: Performance Tests${NC}"
echo "=========================="

read -p "Do you want to run performance validation tests? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Running CPU-safe performance tests..."
    $PYTHON_CMD test_cpu_optimizations.py
    
    if [[ "$HAS_GPU" == true ]]; then
        echo "🚀 Running GPU performance tests..."
        $PYTHON_CMD test_optimizations.py
    fi
    
    echo -e "✅ Performance tests completed"
else
    echo -e "${BLUE}ℹ️  Skipping performance tests. Run later with:${NC}"
    echo "   python test_cpu_optimizations.py  # CPU-safe tests"
    echo "   python test_optimizations.py      # GPU tests"
fi

echo -e "\n${GREEN}🎉 SETUP COMPLETE!${NC}"
echo "=================="
echo ""
echo -e "${GREEN}Next Steps:${NC}"
echo "1. To activate the environment:"
if [[ "$OS" == "windows" ]]; then
    echo "   source venv/Scripts/activate"
else
    echo "   source venv/bin/activate"
fi
echo ""
echo "2. To start training:"
echo "   # Single GPU - Quick test (5 files, 1000 steps)"
echo "   python train.py --config test"
echo ""
echo "   # Multi-GPU - Quick test with 2 GPUs"
echo "   accelerate launch --nproc_per_node=2 train.py --config test"
echo ""
echo "   # Multi-GPU - Production training with 8 GPUs"
echo "   accelerate launch --nproc_per_node=8 train.py --config production"
echo ""
echo "   # With specific mixed precision"
echo "   python train.py --config test --mixed-precision bf16"
echo "   accelerate launch --nproc_per_node=2 train.py --config test --mixed-precision bf16"
echo ""
echo "3. For interactive training guide:"
echo "   ./start_training.sh"
echo ""
echo "4. For optimization explanations:"
echo "   ./run_optimized_training.sh"
echo ""

if [[ "$HAS_GPU" == true ]]; then
    echo -e "${GREEN}Expected Performance:${NC}"
    echo "- Single GPU: 1.5-3.0 steps/second"
    echo "- Multi-GPU (2x): 2.5-5.0 steps/second"
    echo "- Multi-GPU (8x): 8-15 steps/second"
    echo "- GPU utilization: 80-95%"
    echo ""
    echo -e "${BLUE}Multi-GPU Setup Reminder:${NC}"
    echo "- Run 'accelerate config' once before multi-GPU training"
    echo "- Use 'accelerate launch --nproc_per_node=N' for N GPUs"
else
    echo -e "${YELLOW}⚠️  CPU Training Warning:${NC}"
    echo "- Training will be very slow (0.01-0.1 steps/second)"
    echo "- Consider using Google Colab or cloud GPU for actual training"
fi

echo ""
echo -e "${BLUE}📚 Documentation:${NC}"
echo "- README.md - Complete setup and usage guide"
echo "- PERFORMANCE_OPTIMIZATIONS.md - Technical details"
echo "- QUICK_START_OPTIMIZED.md - Quick reference"
echo ""
echo -e "${GREEN}🚀 Happy training!${NC}" 