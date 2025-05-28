#!/bin/bash
# =============================================================================
# COMPLETE HYPERBOLIC H100 SETUP & TRAINING LAUNCH SCRIPT
# =============================================================================
# Instance Configuration:
# - GPU: 8x H100 SXM
# - HTTP Port 1: 6006 (TensorBoard)
# - HTTP Port 2: 8888 (Jupyter)
# - Storage: 2.3TB minimum
# =============================================================================

set -e  # Exit on any error

echo "🚀 STARTING COMPLETE SETUP SEQUENCE"
echo "===================================="
echo "Total estimated time: 15-20 minutes"
echo "Training cost: $275-320/hour"
echo ""

# =============================================================================
# STEP 1: ENVIRONMENT SETUP
# =============================================================================
echo "🔧 STEP 1: Setting up base environment..."

# Clone repository
if [ ! -d "ReFocused-Ai" ]; then
    git clone https://github.com/Mahdi-196/ReFocused-Ai.git
fi
cd ReFocused-Ai

# Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    echo "📦 Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    rm Miniconda3-latest-Linux-x86_64.sh
fi

# Activate conda
source ~/miniconda3/bin/activate

# Create environment
conda create -n refocused python=3.9 -y
conda activate refocused

# =============================================================================
# STEP 2: INSTALL CORE DEPENDENCIES
# =============================================================================
echo "📦 STEP 2: Installing PyTorch and core packages..."

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training dependencies
pip install \
    deepspeed>=0.12.0 \
    transformers>=4.35.0 \
    accelerate>=0.25.0 \
    google-cloud-storage>=2.10.0 \
    datasets>=2.14.0 \
    wandb>=0.16.0 \
    tensorboard>=2.14.0 \
    jupyter>=1.0.0 \
    ipywidgets>=8.0.0 \
    nvidia-ml-py>=12.535.77 \
    psutil>=5.9.0 \
    pynvml>=11.5.0 \
    pyyaml>=6.0.0 \
    requests \
    numpy

# =============================================================================
# STEP 3: CREATE DIRECTORIES AND ENVIRONMENT
# =============================================================================
echo "📁 STEP 3: Creating training directories..."

# Create essential directories
mkdir -p /scratch/{logs,checkpoints,cache,shards,deepspeed_nvme}
mkdir -p logs checkpoints training_output

# Create environment variables file
cat > /scratch/training_env.sh << 'EOF'
#!/bin/bash
# Training Environment Variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_BACKENDS_CUDNN_BENCHMARK=1
export TORCH_CUDNN_V8_API_ENABLED=1
export OMP_NUM_THREADS=8
export DEEPSPEED_NVME_PATH=/scratch/deepspeed_nvme
export TRAINING_DATA_DIR=/scratch/shards
export CHECKPOINT_DIR=/scratch/checkpoints
export LOGS_DIR=/scratch/logs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
if [ -f "/scratch/gcp-key.json" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS=/scratch/gcp-key.json
fi
EOF
chmod +x /scratch/training_env.sh

# =============================================================================
# STEP 4: SETUP MONITORING SERVICES (PORTS 6006 & 8888)
# =============================================================================
echo "📊 STEP 4: Setting up monitoring services on ports 6006 & 8888..."

# TensorBoard startup script (Port 6006)
cat > /scratch/start_tensorboard.sh << 'EOF'
#!/bin/bash
cd /scratch/logs
echo "🔥 Starting TensorBoard on port 6006..."
tensorboard --logdir=. --host=0.0.0.0 --port=6006 --reload_interval=30
EOF
chmod +x /scratch/start_tensorboard.sh

# Jupyter startup script (Port 8888)
cat > /scratch/start_jupyter.sh << 'EOF'
#!/bin/bash
cd ~/ReFocused-Ai
echo "📓 Starting Jupyter on port 8888..."
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
    --NotebookApp.token='' --NotebookApp.password='' \
    --NotebookApp.allow_origin='*' --NotebookApp.disable_check_xsrf=True
EOF
chmod +x /scratch/start_jupyter.sh

# Combined monitoring launcher
cat > /scratch/start_monitoring.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting monitoring services..."

# Start TensorBoard in background
nohup /scratch/start_tensorboard.sh > /scratch/logs/tensorboard.log 2>&1 &
TENSORBOARD_PID=$!
echo "📊 TensorBoard started on port 6006 (PID: $TENSORBOARD_PID)"

# Start Jupyter in background
nohup /scratch/start_jupyter.sh > /scratch/logs/jupyter.log 2>&1 &
JUPYTER_PID=$!
echo "📓 Jupyter started on port 8888 (PID: $JUPYTER_PID)"

# Save PIDs for later cleanup
echo $TENSORBOARD_PID > /scratch/tensorboard.pid
echo $JUPYTER_PID > /scratch/jupyter.pid

echo ""
echo "✅ Monitoring services running!"
echo "Access via Hyperbolic dashboard URLs:"
echo "  TensorBoard: http://your-instance-6006.hyperbolic.xyz"
echo "  Jupyter:     http://your-instance-8888.hyperbolic.xyz"
echo ""
echo "Logs located at:"
echo "  TensorBoard: /scratch/logs/tensorboard.log"
echo "  Jupyter:     /scratch/logs/jupyter.log"
EOF
chmod +x /scratch/start_monitoring.sh

# =============================================================================
# STEP 5: TRAINING LAUNCH SCRIPT
# =============================================================================
echo "🎯 STEP 5: Creating training launch script..."

cat > /scratch/launch_training.sh << 'EOF'
#!/bin/bash
# Source environment
source /scratch/training_env.sh
cd ~/ReFocused-Ai

echo "🔥 LAUNCHING TRAINING!"
echo "====================="
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Config: $(pwd)/05_model_training/configs/training_config.yaml"
echo "Logs: /scratch/logs/training.log"
echo ""

# Start monitoring if not already running
if ! pgrep -f tensorboard > /dev/null; then
    echo "🔍 Starting monitoring services..."
    /scratch/start_monitoring.sh
    sleep 3
fi

# Launch training with server launch kit
python server_launch_kit.py

echo "✅ Training launch completed!"
EOF
chmod +x /scratch/launch_training.sh

# =============================================================================
# STEP 6: GPU VERIFICATION
# =============================================================================
echo "🔍 STEP 6: GPU verification..."
nvidia-smi
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Count: {torch.cuda.device_count()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# =============================================================================
# STEP 7: START MONITORING SERVICES
# =============================================================================
echo "📊 STEP 7: Starting monitoring services..."
/scratch/start_monitoring.sh

# Wait for services to start
sleep 5

# =============================================================================
# STEP 8: FINAL SETUP SUMMARY
# =============================================================================
echo ""
echo "✅ SETUP COMPLETED SUCCESSFULLY!"
echo "=================================="
echo "📋 Quick Reference:"
echo "  • Environment: /scratch/training_env.sh"
echo "  • Start Training: /scratch/launch_training.sh"
echo "  • Start Monitoring: /scratch/start_monitoring.sh"
echo "  • TensorBoard: Port 6006"
echo "  • Jupyter: Port 8888"
echo "  • Logs: /scratch/logs/"
echo ""
echo "🔥 READY TO LAUNCH TRAINING!"
echo "Run: /scratch/launch_training.sh"
echo ""
echo "💰 Training cost: $275-320/hour"
echo "⏱️  Estimated duration: 40+ hours"
echo "🎯 Target: 100,000 training steps"

# =============================================================================
# AUTO-LAUNCH OPTION
# =============================================================================
read -p "🚀 Launch training now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🔥 Starting training immediately..."
    /scratch/launch_training.sh
else
    echo "✋ Setup complete. Run '/scratch/launch_training.sh' when ready."
fi 