# ReFocused-AI Training Setup

This repository contains the setup and training code for the ReFocused-AI language model.

## Quick Setup

To set up the environment and install all dependencies:

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/your-username/ReFocused-Ai.git
cd ReFocused-Ai

# Install all dependencies
python quick_setup.py

# Test bucket access and download sample data
python quick_setup.py --test_bucket --download_data
```

## Directory Structure

- **data/**: Training data directory
- **models/**: Model configurations and saved checkpoints
- **05_model_training/**: Training scripts and configurations

## Training Workflow

### 1. Test Access to Training Data

Test if you can access the training data in the Google Cloud Storage bucket:

```bash
python 05_model_training/test_bucket_access.py --download
```

### 2. Download Training Data

Download the training data from Google Cloud Storage:

```bash
cd 05_model_training
python download_data.py --bucket refocused-ai --local_dir ../data/training/shards
```

### 3. Run a Quick Test

Perform a quick test to verify everything is set up correctly:

```bash
cd 05_model_training
python quick_test.py
```

### 4. Run a Single GPU Test

Run a test on a single GPU to verify training works:

```bash
cd 05_model_training
python h100_single_gpu_test.py
```

### 5. Start Full Training

Start full 8-GPU training with DeepSpeed:

```bash
cd 05_model_training
bash h100_runner.sh full
```

## Monitoring Training

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs
```

## Configuration Files

- **model config**: `models/gpt_750m/config.json`
- **training config**: `05_model_training/config/h100_multi_gpu.yaml`
- **DeepSpeed config**: `05_model_training/config/h100_deepspeed_multi.json`

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (H100 for full training)
- All Python dependencies are listed in `requirements.txt`

## Troubleshooting

If you encounter issues:

1. Run `python check_model_setup.py` to verify model configuration
2. Ensure you have access to the Google Cloud Storage bucket
3. Check if all dependencies are installed with `pip list`
4. Verify GPU availability with `python -c "import torch; print(torch.cuda.is_available())"` 