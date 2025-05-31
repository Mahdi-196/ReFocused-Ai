#!/bin/bash
# Fixed H100 Environment Setup Script
# This script addresses NumPy compatibility issues and other setup problems

set -e  # Exit on any error

echo "=== Starting Fixed H100 Environment Setup ==="

# Create all necessary directories
echo "Creating directories..."
mkdir -p /home/ubuntu/training_data/shards
mkdir -p /home/ubuntu/training_data/checkpoints
mkdir -p /home/ubuntu/training_data/logs
mkdir -p /home/ubuntu/training_data/cache
mkdir -p /home/ubuntu/ReFocused-Ai/models/gpt_750m
mkdir -p /home/ubuntu/ReFocused-Ai/models/tokenizer/tokenizer

# Install build dependencies first
echo "Installing build dependencies..."
pip install --upgrade pip
pip install packaging setuptools wheel

# Downgrade NumPy to compatible version (critical fix)
echo "Fixing NumPy version compatibility..."
pip uninstall -y numpy
pip install numpy==1.24.3

# Install PyTorch with correct CUDA version
echo "Installing PyTorch with CUDA support..."
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install other packages from fixed requirements
echo "Installing remaining dependencies..."
cd /home/ubuntu/ReFocused-Ai/05_model_training
pip install -r fixed_requirements.txt

# Create model config file
echo "Setting up model configuration files..."
if [ ! -f /home/ubuntu/ReFocused-Ai/models/gpt_750m/config.json ]; then
    cat > /home/ubuntu/ReFocused-Ai/models/gpt_750m/config.json << 'EOL'
{
  "architectures": ["GPTNeoXForCausalLM"],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 50256,
  "eos_token_id": 50256,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "gpt_neox",
  "num_attention_heads": 16,
  "num_hidden_layers": 24,
  "rotary_emb_base": 10000,
  "rotary_pct": 0.25,
  "tie_word_embeddings": false,
  "transformers_version": "4.28.1",
  "use_cache": true,
  "vocab_size": 50304
}
EOL
    echo "Created model config file"
fi

# Create tokenizer files
echo "Setting up tokenizer..."
pip install transformers
python3 -c "from transformers import GPT2Tokenizer; tokenizer = GPT2Tokenizer.from_pretrained('gpt2'); tokenizer.save_pretrained('/home/ubuntu/ReFocused-Ai/models/tokenizer/tokenizer')"

# Set up monitoring scripts
echo "Creating monitoring scripts..."
cat > /home/ubuntu/start_tensorboard.sh << 'EOL'
#!/bin/bash
mkdir -p /home/ubuntu/training_data/logs
tensorboard --logdir=/home/ubuntu/training_data/logs --port=6006 --host=0.0.0.0 &
echo "TensorBoard running at http://kind-croton-macaw-6006.1.cricket.hyperbolic.xyz:30000"
EOL
chmod +x /home/ubuntu/start_tensorboard.sh

# Fix path
echo "Fixing PATH for user-installed packages..."
if ! grep -q "export PATH=\"\$HOME/.local/bin:\$PATH\"" ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Set up environment variables file
echo "Setting up environment variables..."
cat > /home/ubuntu/h100_env.sh << 'EOL'
#!/bin/bash
# Environment variables for DeepSpeed and H100
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=eth0
export PATH="$HOME/.local/bin:$PATH"
EOL
chmod +x /home/ubuntu/h100_env.sh

# Make h100_runner.sh executable
chmod +x /home/ubuntu/ReFocused-Ai/05_model_training/h100_runner.sh

echo "=== Environment setup complete! ==="
echo "To use the environment, run:"
echo "  source /home/ubuntu/h100_env.sh"
echo "  cd /home/ubuntu/ReFocused-Ai/05_model_training"
echo "  ./h100_runner.sh test"
echo ""
echo "To start TensorBoard, run:"
echo "  /home/ubuntu/start_tensorboard.sh"
echo ""
echo "For full training, run:"
echo "  ./h100_runner.sh full" 