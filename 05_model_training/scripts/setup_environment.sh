#!/bin/bash
# Setup Script for H100 SXM Training Environment
# Optimized for Hyperbolic Labs 8x H100 SXM setup

set -e  # Exit on any error

echo "🚀 Setting up H100 Training Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# 1. System Information
print_status "Gathering system information..."
echo "OS: $(lsb_release -d | cut -f2)"
echo "Kernel: $(uname -r)"
echo "CPUs: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "GPUs: $(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)"

# 2. GPU Check
print_status "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    print_error "nvidia-smi not found. Please install NVIDIA drivers."
    exit 1
fi

# Check for H100 GPUs
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader)

print_status "Found $gpu_count GPU(s):"
echo "$gpu_names"

if [[ $gpu_count -ne 8 ]]; then
    print_warning "Expected 8 GPUs, found $gpu_count"
fi

# 3. Create Directories
print_status "Creating training directories..."
sudo mkdir -p /scratch/{shards,checkpoints,logs,cache,deepspeed_nvme,models}
sudo chown -R $USER:$USER /scratch
chmod -R 755 /scratch

print_status "Training directories created:"
ls -la /scratch/

# 4. Set up Python Environment
print_status "Setting up Python environment..."

# Update package list
sudo apt-get update

# Install system dependencies
print_status "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    htop \
    nvtop \
    tree \
    screen \
    tmux \
    vim \
    python3-dev \
    python3-pip \
    libaio-dev

# Install Python packages
print_status "Installing Python training dependencies..."
pip3 install --upgrade pip

# Install PyTorch with CUDA support
pip3 install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training requirements
if [ -f "requirements_training.txt" ]; then
    pip3 install -r requirements_training.txt
else
    print_warning "requirements_training.txt not found, installing core packages..."
    pip3 install \
        transformers>=4.35.0 \
        accelerate>=0.25.0 \
        deepspeed>=0.12.0 \
        google-cloud-storage>=2.10.0 \
        datasets>=2.14.0 \
        wandb>=0.16.0 \
        tensorboard>=2.14.0 \
        nvidia-ml-py>=12.535.77 \
        psutil>=5.9.0 \
        pynvml>=11.5.0 \
        pyyaml>=6.0.0
fi

# 5. Configure Environment Variables
print_status "Configuring environment variables..."

# Create environment file
cat > /scratch/training_env.sh << 'EOF'
#!/bin/bash
# Training Environment Variables

# CUDA and GPU settings
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# PyTorch optimizations
export TORCH_BACKENDS_CUDNN_BENCHMARK=1
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8

# DeepSpeed settings
export DEEPSPEED_NVME_PATH=/scratch/deepspeed_nvme

# Google Cloud credentials (if available)
if [ -f "/scratch/gcp-key.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS=/scratch/gcp-key.json
fi

# Training paths
export TRAINING_DATA_DIR=/scratch/shards
export CHECKPOINT_DIR=/scratch/checkpoints
export LOGS_DIR=/scratch/logs

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
EOF

chmod +x /scratch/training_env.sh

# Source the environment
source /scratch/training_env.sh

print_status "Environment variables configured"

# 6. Optimize System Settings
print_status "Optimizing system settings..."

# Increase file limits
echo "* soft nofile 1048576" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 1048576" | sudo tee -a /etc/security/limits.conf

# Optimize networking
echo 'net.core.rmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.rmem_max = 16777216' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_default = 262144' | sudo tee -a /etc/sysctl.conf
echo 'net.core.wmem_max = 16777216' | sudo tee -a /etc/sysctl.conf

# Apply sysctl changes
sudo sysctl -p

# 7. Test GPU Communication
print_status "Testing GPU communication..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    
    # Test each GPU
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'   Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
    
    # Test multi-GPU communication
    if torch.cuda.device_count() > 1:
        print('✅ Testing multi-GPU communication...')
        try:
            # Simple multi-GPU test
            devices = list(range(torch.cuda.device_count()))
            tensors = []
            for device in devices:
                tensors.append(torch.randn(1000, 1000, device=device))
            
            # Test all-reduce operation
            for tensor in tensors:
                torch.distributed.all_reduce(tensor) if torch.distributed.is_available() else None
            
            print('✅ Multi-GPU communication test passed')
        except Exception as e:
            print(f'⚠️  Multi-GPU test warning: {e}')
else:
    print('❌ CUDA not available')
    exit(1)
"

# 8. Test DeepSpeed
print_status "Testing DeepSpeed installation..."
python3 -c "
import deepspeed
print(f'✅ DeepSpeed version: {deepspeed.__version__}')

# Test DeepSpeed CUDA ops
try:
    from deepspeed.ops.adam import FusedAdam
    print('✅ DeepSpeed CUDA ops available')
except ImportError as e:
    print(f'⚠️  DeepSpeed CUDA ops not available: {e}')
"

# 9. Setup Monitoring
print_status "Setting up monitoring..."

# Create monitoring script
cat > /scratch/monitor_training.sh << 'EOF'
#!/bin/bash
# Training Monitoring Script

echo "🔍 Training Monitoring Dashboard"
echo "==============================="

while true; do
    clear
    echo "📅 $(date)"
    echo ""
    
    # GPU Status
    echo "🖥️  GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
    awk -F',' '{printf "GPU %s: %s | Util: %s%% | Mem: %s/%s MB | Temp: %s°C | Power: %s W\n", $1, $2, $3, $4, $5, $6, $7}'
    echo ""
    
    # System Resources
    echo "💻 System Resources:"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')% usage"
    echo "Memory: $(free -h | awk '/^Mem:/ {printf "%s/%s (%.1f%%)", $3, $2, $3/$2*100}')"
    echo "Disk /scratch: $(df -h /scratch | awk 'NR==2 {printf "%s/%s (%s)", $3, $2, $5}')"
    echo ""
    
    # Training Progress (if log exists)
    if [ -f "/scratch/logs/training.log" ]; then
        echo "📊 Latest Training Progress:"
        tail -5 /scratch/logs/training.log | grep -E "(Step|loss|lr)" || echo "No recent training logs"
        echo ""
    fi
    
    # Network activity
    echo "🌐 Network Activity:"
    cat /proc/net/dev | awk 'NR>2 {print $1, $2, $10}' | head -3
    echo ""
    
    echo "Press Ctrl+C to exit"
    sleep 10
done
EOF

chmod +x /scratch/monitor_training.sh

# 10. Create launch helper script
print_status "Creating training launch script..."

cat > /scratch/launch_training.sh << 'EOF'
#!/bin/bash
# Training Launch Helper

# Source environment
source /scratch/training_env.sh

# Check if config exists
if [ ! -f "config/training_config.yaml" ]; then
    echo "❌ Training config not found at config/training_config.yaml"
    exit 1
fi

# Start monitoring in background
if [ "$1" != "--no-monitor" ]; then
    echo "🔍 Starting monitoring dashboard in background..."
    screen -dmS monitor /scratch/monitor_training.sh
fi

# Launch training with DeepSpeed
echo "🚀 Launching training..."
echo "Config: $(pwd)/config/training_config.yaml"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Logs: /scratch/logs/training.log"
echo ""

# Set up signal handlers for graceful shutdown
trap 'echo "Received interrupt signal, stopping training..."; kill $TRAINING_PID; wait $TRAINING_PID; exit 0' INT TERM

# Launch training
deepspeed --num_gpus=8 \
    --master_port=29500 \
    train.py \
    --config config/training_config.yaml \
    "$@" &

TRAINING_PID=$!

# Wait for training to complete
wait $TRAINING_PID

echo "✅ Training completed!"
EOF

chmod +x /scratch/launch_training.sh

# 11. Final System Check
print_status "Running final system check..."

# Check disk space
available_space=$(df /scratch | awk 'NR==2 {print $4}')
if [ $available_space -lt 2000000000 ]; then  # Less than 2TB
    print_warning "Available space on /scratch: $(df -h /scratch | awk 'NR==2 {print $4}')"
    print_warning "Recommend at least 2TB for training data and checkpoints"
fi

# Check memory
total_memory=$(free -m | awk 'NR==2 {print $2}')
if [ $total_memory -lt 500000 ]; then  # Less than 500GB
    print_warning "System memory: $(free -h | awk 'NR==2 {print $2}')"
    print_warning "Recommend at least 512GB RAM for optimal training"
fi

print_status "✅ Environment setup completed!"
echo ""
echo "📋 Setup Summary:"
echo "=================="
echo "• Training directories: /scratch/{shards,checkpoints,logs,cache}"
echo "• Environment script: /scratch/training_env.sh"
echo "• Launch script: /scratch/launch_training.sh"
echo "• Monitor script: /scratch/monitor_training.sh"
echo ""
echo "🚀 Next steps:"
echo "1. Upload your GCP service account key to /scratch/gcp-key.json"
echo "2. Run: source /scratch/training_env.sh"
echo "3. Navigate to your training code directory"
echo "4. Run: /scratch/launch_training.sh"
echo ""
echo "💡 To monitor training: screen -r monitor"
print_status "Happy training! 🎯" 