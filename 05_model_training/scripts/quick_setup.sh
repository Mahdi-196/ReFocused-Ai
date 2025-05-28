#!/bin/bash
# Quick Setup Script for Hyperbolic Instance
# Run this once you SSH into your H100 instance

set -e

echo "🚀 SETTING UP TRAINING ENVIRONMENT ON HYPERBOLIC"
echo "================================================"

# 1. Create directories
echo "📁 Creating training directories..."
mkdir -p /scratch/{shards,checkpoints,logs,cache,deepspeed_nvme,models}

# 2. Clone repository
echo "📦 Cloning ReFocused-Ai repository..."
if [ ! -d "/root/ReFocused-Ai" ]; then
    git clone https://github.com/yourusername/ReFocused-Ai.git /root/ReFocused-Ai
fi
cd /root/ReFocused-Ai/05_model_training

# 3. Install dependencies
echo "📦 Installing training dependencies..."
pip install --upgrade pip
pip install -r requirements_training.txt

# 4. Setup environment variables
echo "⚙️  Setting up environment..."
cat > /scratch/training_env.sh << 'EOF'
#!/bin/bash
# Training Environment Variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export DEEPSPEED_NVME_PATH=/scratch/deepspeed_nvme
export TRAINING_DATA_DIR=/scratch/shards
export CHECKPOINT_DIR=/scratch/checkpoints
export LOGS_DIR=/scratch/logs
export OMP_NUM_THREADS=8
EOF

chmod +x /scratch/training_env.sh
source /scratch/training_env.sh

# 5. Test GPU setup
echo "🔍 Testing GPU setup..."
python3 -c "
import torch
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'✅ GPU {i}: {torch.cuda.get_device_name(i)}')
"

# 6. Create launch script
echo "🎯 Creating training launch script..."
cat > /scratch/launch_training.sh << 'EOF'
#!/bin/bash
# Quick Training Launch
cd /root/ReFocused-Ai/05_model_training
source /scratch/training_env.sh

echo "🚀 Starting 1B Parameter Training..."
echo "Cost: $7.92/hour | Expected: $275-320 total"

deepspeed --num_gpus=8 --master_port=29500 train.py --config config/training_config.yaml
EOF

chmod +x /scratch/launch_training.sh

echo ""
echo "✅ SETUP COMPLETE!"
echo "=================="
echo "Next steps:"
echo "1. Upload GCP key: scp your-key.json user@instance:/scratch/gcp-key.json"
echo "2. Set credentials: export GOOGLE_APPLICATION_CREDENTIALS=/scratch/gcp-key.json"
echo "3. Launch training: /scratch/launch_training.sh"
echo ""
echo "💰 Expected cost: $275-320 for complete training" 