# ReFocused-AI Model Training

This directory contains the complete training setup for the ReFocused-AI 1.2B parameter language model using PyTorch FSDP (Fully Sharded Data Parallel) and Hugging Face Accelerate.

## Model Specifications

- **Parameters**: ~1.2B
- **Architecture**: GPT-NeoX
- **Hidden Size**: 2048
- **Layers**: 24
- **Attention Heads**: 16
- **Sequence Length**: 2048
- **Training Data**: 21-22B tokens from Reddit data

## Quick Start

### 1. Initial Setup (Run once on new instance)

```bash
cd 05_model_training
bash setup.sh
```

This will:
- Create Python 3.10 virtual environment
- Install PyTorch with CUDA support
- Install all required dependencies
- Create necessary directories
- Set up environment variables

### 2. Test Training (1 GPU, 25 files)

```bash
bash run_test_training.sh
```

This runs a small test to verify everything works correctly.

### 3. Production Training (Multi-GPU, full dataset)

```bash
bash run_production_training.sh
```

This runs the full training on all available GPUs.

## Directory Structure

```
05_model_training/
├── configs/            # Model and training configurations
├── utils/              # Training utilities (data loading, checkpointing, metrics)
├── scripts/            # Additional training scripts
├── logs/               # Training logs and tensorboard files
├── checkpoints/        # Local checkpoint storage
├── cache/              # Downloaded data cache
├── train.py            # Main training script
├── setup.sh            # Environment setup script
├── run_test_training.sh    # Test training launcher
├── run_production_training.sh  # Production training launcher
└── accelerate_config.yaml  # Accelerate configuration
```

## Data Source

- **Bucket**: `refocused-ai` (Google Cloud Storage, public)
- **Data Format**: Tokenized `.npz` files
- **Checkpoint Storage**: `refocused-ai/Checkpoints`

## Training Configuration

### Test Configuration (25 files)
- Batch size: 2 per GPU
- Gradient accumulation: 4 steps
- Effective batch size: 8
- Learning rate: 2e-4
- Checkpoint frequency: Every 5 files

### Production Configuration (full dataset)
- Batch size: 4 per GPU  
- Gradient accumulation: 8 steps
- Effective batch size: 32
- Learning rate: 2e-4
- Checkpoint frequency: Every 5 files

## Key Features

- **FSDP Integration**: Automatic model sharding across GPUs
- **Mixed Precision**: BF16 training for H100 optimization
- **Gradient Checkpointing**: Memory-efficient training
- **Automatic Checkpointing**: Saves to GCS every 5 files
- **Resume Support**: Can resume from any checkpoint
- **TensorBoard Logging**: Real-time metrics visualization

## Monitoring

### View Training Logs
```bash
tensorboard --logdir=./logs
```

### Track Metrics
- Loss (raw and smoothed)
- Perplexity
- Learning rate
- Gradient norms
- Training speed (samples/sec)
- Tokens processed

## Resuming Training

To resume from a checkpoint:

```bash
python train.py --mode production --resume checkpoint-epoch0-step1000-files5
```

Or use accelerate:

```bash
accelerate launch train.py --mode production --resume checkpoint-epoch0-step1000-files5
```

## Advanced Usage

### Custom Number of GPUs
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash run_production_training.sh
```

### Modify Training Parameters
Edit `configs/training_config.py` to adjust:
- Learning rate
- Batch size
- Checkpoint frequency
- Weight decay
- Warmup steps

### Debug Mode
For debugging with single process:
```bash
python train.py --mode test
```

## Troubleshooting

1. **OOM Errors**: Reduce `per_device_train_batch_size` or enable CPU offload in FSDP config
2. **Slow Training**: Ensure you're using BF16 and TF32 is enabled
3. **Checkpoint Issues**: Check GCS credentials or use local checkpoints only
4. **Data Loading**: Verify bucket access and file format

## Performance Tips

- Use BF16 mixed precision on H100s
- Keep gradient accumulation steps high for stability
- Monitor gradient norms for training stability
- Use multiple data workers for faster loading

## Requirements

- Python 3.10
- CUDA 11.8+
- PyTorch 2.1.2
- Hugging Face Transformers 4.36.2
- Accelerate 0.25.0
- 80GB+ GPU memory per device (for 1.2B model) 