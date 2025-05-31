#!/usr/bin/env bash
set -e

echo "==================================="
echo "Setting up ReFocused-AI Training Environment"
echo "==================================="

# 1. Install PyTorch + CUDA 11.8
echo "Installing PyTorch 2.1.0 with CUDA 11.8..."
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 || {
    echo "ERROR: Failed to install PyTorch"
    exit 1
}

# 2. Force NumPy 1.24.0 (required for compatibility)
echo "Installing NumPy 1.24.0..."
pip install numpy==1.24.0 || {
    echo "ERROR: Failed to install NumPy"
    exit 1
}

# 3. Install packaging
echo "Installing packaging..."
pip install packaging==23.2 || {
    echo "ERROR: Failed to install packaging"
    exit 1
}

# 4. TensorBoard and W&B for logging
echo "Installing TensorBoard and Weights & Biases..."
pip install tensorboard==2.15.0 wandb==0.16.0 || {
    echo "ERROR: Failed to install logging tools"
    exit 1
}

# 5. Transformers (only for tokenizer usage)
echo "Installing Transformers..."
pip install transformers==4.35.2 || {
    echo "ERROR: Failed to install transformers"
    exit 1
}

# 6. DeepSpeed with all operations
echo "Installing DeepSpeed..."
DS_BUILD_OPS=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 pip install deepspeed==0.12.3 || {
    echo "WARNING: DeepSpeed installation might have partial failures. Continuing..."
}

# 7. Google Cloud Storage client
echo "Installing Google Cloud Storage client..."
pip install google-cloud-storage==2.10.0 || {
    echo "ERROR: Failed to install google-cloud-storage"
    exit 1
}

# 8. Additional utilities
echo "Installing additional utilities..."
pip install tqdm==4.66.1 pytest==7.4.3 || {
    echo "ERROR: Failed to install utilities"
    exit 1
}

# 9. Optional: Flash Attention (comment out if not needed)
echo "Attempting to install Flash Attention..."
pip install flash-attn==2.3.6 --no-build-isolation || {
    echo "WARNING: Flash Attention installation failed. Using standard attention."
}

# 10. Verify critical installations
echo ""
echo "Verifying installations..."
python -c "
import sys
try:
    import numpy as np
    assert np.__version__.startswith('1.24'), f'NumPy version {np.__version__} != 1.24.0'
    print(f'✓ NumPy {np.__version__}')
except Exception as e:
    print(f'✗ NumPy verification failed: {e}')
    sys.exit(1)

try:
    import torch
    assert torch.__version__.startswith('2.1.0'), f'PyTorch version {torch.__version__} != 2.1.0'
    print(f'✓ PyTorch {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA devices: {torch.cuda.device_count()}')
except Exception as e:
    print(f'✗ PyTorch verification failed: {e}')
    sys.exit(1)

try:
    import transformers
    print(f'✓ Transformers {transformers.__version__}')
except Exception as e:
    print(f'✗ Transformers verification failed: {e}')
    sys.exit(1)

try:
    import deepspeed
    print(f'✓ DeepSpeed {deepspeed.__version__}')
except Exception as e:
    print(f'✗ DeepSpeed verification failed: {e}')
    sys.exit(1)

try:
    from torch.utils.tensorboard import SummaryWriter
    print('✓ TensorBoard')
except Exception as e:
    print(f'✗ TensorBoard verification failed: {e}')
    sys.exit(1)

print('')
print('✅ All critical packages verified successfully!')
"

echo ""
echo "==================================="
echo "✅ Environment setup complete!"
echo "===================================" 