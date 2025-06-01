#!/bin/bash
# ReFocused-AI Optimized Training Setup Script
# This script sets up the complete environment for training a 1-1.2B parameter model
# with advanced preprocessing optimizations and monitoring

set -e  # Exit on error

echo "🚀 ReFocused-AI Optimized Training Setup"
echo "========================================"
echo "Setting up environment for FSDP training with optimizations..."

# Create virtual environment with Python 3.10
echo "📦 Creating virtual environment..."
python3.10 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (for H100)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install Hugging Face libraries
echo "🤗 Installing Hugging Face libraries..."
pip install transformers==4.36.2
pip install accelerate==0.25.0
pip install datasets==2.16.1
pip install tokenizers==0.15.0

# Install core dependencies
echo "📚 Installing core dependencies..."
pip install numpy==1.26.3
pip install scipy==1.11.4
pip install tqdm==4.66.1

# Install monitoring and logging dependencies
echo "📊 Installing monitoring dependencies..."
pip install wandb==0.16.2
pip install tensorboard==2.15.1
pip install psutil>=5.8.0  # For system monitoring

# Install cloud and storage dependencies
echo "☁️  Installing cloud dependencies..."
pip install google-cloud-storage==2.13.0
pip install google-resumable-media>=2.5.0
pip install google-auth>=2.22.0

# Install text processing dependencies
echo "📝 Installing text processing libraries..."
pip install sentencepiece==0.1.99
pip install protobuf==4.23.4

# Install additional optimization dependencies
echo "⚡ Installing optimization dependencies..."
pip install memory-profiler>=0.60.0
pip install jsonlines>=3.1.0
pip install tabulate>=0.9.0

# Create comprehensive directory structure
echo "📁 Creating optimized directory structure..."
mkdir -p logs
mkdir -p checkpoints
mkdir -p cache
mkdir -p preprocessed_cache  # NEW: For preprocessing optimizations
mkdir -p data
mkdir -p scripts
mkdir -p utils
mkdir -p configs

# Set environment variables for optimized training
echo "🔧 Setting environment variables..."
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Configure accelerate
echo "🚀 Configuring Accelerate..."
bash configure_accelerate.sh

# Make scripts executable
echo "🔐 Making scripts executable..."
chmod +x run_optimized_training.sh
chmod +x run_test_training.sh
chmod +x run_production_training.sh
chmod +x configure_accelerate.sh

# Test optimized imports
echo "🧪 Testing optimized imports..."
python -c "
try:
    import torch
    import transformers
    import accelerate
    import psutil
    import tensorboard
    print('✅ All core dependencies imported successfully')
    print(f'   PyTorch version: {torch.__version__}')
    print(f'   Transformers version: {transformers.__version__}')
    print(f'   Accelerate version: {accelerate.__version__}')
    print(f'   CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test optimized training components
echo "🔍 Testing optimized training components..."
python -c "
try:
    from configs import get_test_config, get_production_config
    from utils import create_dataloader, EnhancedMetricsTracker
    print('✅ Optimized training components imported successfully')
except ImportError as e:
    print(f'⚠️  Warning: {e}')
    print('   This is normal if running setup before training files are ready')
"

# Create enhanced activation script
cat > activate_env.sh << 'EOF'
#!/bin/bash
echo "🚀 Activating ReFocused-AI Optimized Training Environment"
source venv/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0

# Show system info
echo "📊 System Information:"
python -c "
import torch
import psutil
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'  CPU Count: {psutil.cpu_count()}')
memory = psutil.virtual_memory()
print(f'  Memory: {memory.total/1024**3:.1f} GB total, {memory.available/1024**3:.1f} GB available')
"

echo ""
echo "✅ Environment activated successfully!"
echo ""
echo "🎯 Quick Start Commands:"
echo "  Test dataloader:     python test_dataloader.py"
echo "  Quick test run:      ./run_optimized_training.sh test"
echo "  Production run:      ./run_optimized_training.sh production"
echo "  Monitor training:    python scripts/monitor_training.py"
echo "  Show optimizations:  python scripts/show_optimizations.py"
EOF

chmod +x activate_env.sh

# Create quick start script
cat > quick_start.sh << 'EOF'
#!/bin/bash
echo "🏃 ReFocused-AI Quick Start"
echo "=========================="

# Activate environment
source activate_env.sh

echo ""
echo "🧪 Running quick tests..."

# Test dataloader
echo "Testing optimized dataloader..."
python test_dataloader.py

if [ $? -eq 0 ]; then
    echo "✅ Dataloader test passed!"
    echo ""
    echo "🚀 Starting test training (50 steps)..."
    ./run_optimized_training.sh test false 50
else
    echo "❌ Dataloader test failed. Please check the setup."
    exit 1
fi
EOF

chmod +x quick_start.sh

# Create monitoring helper script
cat > start_monitoring.sh << 'EOF'
#!/bin/bash
echo "📊 Starting ReFocused-AI Training Monitor"
source activate_env.sh
python scripts/monitor_training.py --refresh 5
EOF

chmod +x start_monitoring.sh

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "📋 What was installed:"
echo "  ✅ PyTorch 2.1.2 with CUDA support"
echo "  ✅ Transformers 4.36.2 with Accelerate 0.25.0"
echo "  ✅ Enhanced monitoring with psutil and tensorboard"
echo "  ✅ Optimized preprocessing cache system"
echo "  ✅ Real-time training monitoring"
echo "  ✅ Performance profiling capabilities"
echo ""
echo "📁 Directories created:"
echo "  ✅ logs/ checkpoints/ cache/"
echo "  ✅ preprocessed_cache/ (for optimization)"
echo "  ✅ scripts/ utils/ configs/"
echo ""
echo "🎯 Next steps:"
echo "  1. Activate environment:     source activate_env.sh"
echo "  2. Quick test:              ./quick_start.sh"
echo "  3. Or test manually:        python test_dataloader.py"
echo "  4. Start training:          ./run_optimized_training.sh test"
echo "  5. Monitor training:        ./start_monitoring.sh"
echo ""
echo "💡 Pro tips:"
echo "  • Use 'python scripts/show_optimizations.py' to see all features"
echo "  • First run will preprocess data (takes time)"
echo "  • Subsequent runs will be much faster with cached data"
echo "  • Monitor training in real-time with the monitoring script" 