# ReFocused-AI GPT Training

This directory contains the complete training pipeline for training a ~1.2B parameter GPT model from scratch using 21-22 billion tokens.

## Overview

The training system is designed for:
- **Efficiency**: Mixed precision (FP16), gradient accumulation, multi-GPU scaling with DeepSpeed
- **Robustness**: Version checks, automatic retries on GCS errors, checkpoint integrity
- **Quality**: Proper learning rate schedules, regularization, comprehensive logging

## Prerequisites

- Python 3.8+
- CUDA 11.8+
- NVIDIA GPUs (tested on H100 80GB)
- Google Cloud SDK (for GCS access)
- Virtual environment

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
bash setup_env.sh
```

### 2. Test Run (1 GPU)

```bash
# Downloads 25 files and runs a test
bash test_train.sh
```

### 3. Production Run (8 GPUs)

```bash
# Download all data
bash setup_prod.sh

# Start training
bash prod_train.sh
```

## Scripts

### `setup_env.sh`
- Installs all required dependencies with exact versions
- Verifies installations
- Builds Flash Attention (optional)

### `test_train.sh`
- Downloads 25 tokenized files from GCS
- Runs training on 1 GPU in test mode
- Creates checkpoints every 5 files
- Uploads to `gs://refocused-ai/Checkpoints/`

### `setup_prod.sh`
- Downloads all tokenized data (~21-22B tokens)
- Verifies disk space
- Creates file manifest

### `prod_train.sh`
- Launches training on 8×H100 GPUs
- Uses DeepSpeed ZeRO Stage 2
- Mixed precision (FP16)
- Gradient checkpointing
- Logs to TensorBoard and W&B

### `train.py`
Main training script with:
- Dependency version checking
- Custom GPT implementation from scratch
- Flash Attention support (optional)
- DeepSpeed integration
- GCS checkpointing with retries
- TensorBoard and W&B logging

## Model Configurations

| Model | Layers | Heads | d_model | d_ff | Parameters |
|-------|--------|-------|---------|------|------------|
| 125M  | 12     | 12    | 768     | 3072 | ~125M      |
| 350M  | 24     | 16    | 1024    | 4096 | ~350M      |
| 760M  | 24     | 16    | 1536    | 6144 | ~760M      |
| 1.2B  | 24     | 16    | 2048    | 8192 | ~1.2B      |

## Training Configuration

### Hyperparameters
- **Batch size**: 128 (global)
- **Micro batch size**: 8 per GPU
- **Learning rate**: 6e-4 with cosine decay
- **Warmup steps**: 2000
- **Weight decay**: 0.1
- **Gradient clipping**: 1.0
- **Dropout**: 0.1

### DeepSpeed Features
- ZeRO Stage 2 optimization
- Mixed precision (FP16)
- Gradient accumulation
- Activation checkpointing
- Overlapped communication

## Monitoring

### TensorBoard
```bash
tensorboard --logdir=logs/ --bind_all
```

### Weights & Biases
View at: https://wandb.ai/your-project

### Metrics Tracked
- Training loss
- Learning rate
- Tokens per second
- Files processed
- GPU utilization

## Checkpointing

Checkpoints are saved:
- Every 5 files processed
- Uploaded to `gs://refocused-ai/Checkpoints/`
- Include model weights, optimizer state, and metadata
- Automatic cleanup of local copies

## Troubleshooting

### CUDA Out of Memory
- Reduce `micro_batch_size` in DeepSpeed config
- Enable more aggressive activation checkpointing
- Use gradient accumulation

### GCS Authentication
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
```

### Version Conflicts
The training script performs strict version checking. If you see version errors:
1. Check the error message for the required version
2. Update `setup_env.sh` if needed
3. Reinstall dependencies

## Performance Tips

1. **Use Flash Attention** if available (2-3x speedup)
2. **Enable NCCL optimizations**:
   ```bash
   export NCCL_P2P_DISABLE=0
   export NCCL_IB_DISABLE=0
   ```
3. **Monitor GPU utilization** with `nvidia-smi`
4. **Use fast interconnect** (NVLink/InfiniBand)

## Expected Training Time

Based on benchmarks (8×H100 GPUs):
- 1B tokens: ~2-3 hours
- 10B tokens: ~20-30 hours  
- 22B tokens: ~44-66 hours

Actual time depends on:
- Hardware configuration
- Network speed (for multi-node)
- Data loading efficiency

## Safety Features

- Automatic version checking
- Graceful handling of GCS failures
- Checkpoint integrity verification
- Automatic retry with exponential backoff
- Comprehensive error logging

## License

This training code is part of the ReFocused-AI project. 