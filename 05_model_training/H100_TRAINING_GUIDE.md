# H100 SXM Training Guide

This guide provides instructions for training a ~1B parameter model on H100 SXM GPUs efficiently. The setup is optimized for both single-GPU testing and full 8-GPU distributed training.

## Quick Start

```bash
# Download data and run a quick test on a single GPU with 25 files
cd 05_model_training
./h100_runner.sh test

# Download all data and run full training on 8 GPUs
./h100_runner.sh full

# Only download data
./h100_runner.sh download

# Upload a checkpoint to cloud storage
./h100_runner.sh upload /home/ubuntu/training_data/checkpoints/checkpoint_step_1000
```

## System Requirements

- NVIDIA H100 SXM GPU(s)
- Ubuntu 22.04 or later
- CUDA 12.0+
- Python 3.10+
- 500GB+ disk space

## Files Overview

- `h100_runner.sh`: Master script for the entire workflow
- `h100_single_gpu_test.py`: Single-GPU test script for 25 files
- `download_data.py`: Script to download training data from GCS
- `upload_checkpoints.py`: Script to upload checkpoints to GCS
- `config/h100_multi_gpu.yaml`: Configuration for 8-GPU training
- `config/h100_deepspeed_multi.json`: DeepSpeed ZeRO-3 configuration

## Detailed Instructions

### 1. Environment Setup

The `h100_runner.sh` script automatically installs required packages, but you can also install them manually:

```bash
pip install google-cloud-storage deepspeed torch torchvision transformers wandb pyyaml
```

### 2. Data Preparation

Data should be in `.npz` format with tokenized text in either 'input_ids', 'arr_0', 'text', or 'sequences' fields. The data download script gets these from a public GCS bucket.

```bash
python3 download_data.py --bucket refocused-ai --remote_path tokenized_data --local_dir /home/ubuntu/training_data/shards
```

### 3. Single-GPU Testing

Run a quick test on a single GPU to verify the setup:

```bash
python3 h100_single_gpu_test.py --output_dir h100_test_output --data_dir /home/ubuntu/training_data/shards --num_files 25
```

This will create a report in `h100_test_output/benchmark_results.json` with performance metrics.

### 4. Full 8-GPU Training

For production training across all 8 GPUs:

```bash
deepspeed --num_gpus=8 train_pytorch.py --config config/h100_multi_gpu.yaml
```

### 5. Checkpoint Management

Checkpoints are automatically saved during training. To upload a checkpoint to GCS:

```bash
python3 upload_checkpoints.py --checkpoint_dir /home/ubuntu/training_data/checkpoints/checkpoint_step_1000
```

## Troubleshooting

### CUDA Out of Memory

If you encounter CUDA OOM errors:

1. Reduce `per_device_train_batch_size` in the config
2. Increase `gradient_accumulation_steps`
3. Enable `gradient_checkpointing` (already enabled by default)
4. Use BF16 mixed precision (already enabled)

### Network/RDMA Issues

For network-related errors with RDMA/EFA adapters:

```bash
export RDMAV_FORK_SAFE=1
export FI_EFA_FORK_SAFE=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

These are already set in the runner script.

### DeepSpeed Issues

If DeepSpeed fails to initialize:

1. Check if `mpi4py` is installed correctly
2. Verify that all GPUs are visible with `nvidia-smi`
3. Try using PyTorch DDP as a fallback (edit config to remove deepspeed section)

## Performance Expectations

On 8x H100 SXM GPUs, expect:
- ~550 tokens/second/GPU for single-GPU testing
- ~3500-4000 tokens/second total for 8-GPU training
- ~5-7 days to train a 1B parameter model on a large dataset

## Cost Optimization

To minimize costs on expensive GPU instances:

1. Use the test script first to verify your setup
2. Set a specific `total_steps` in config to ensure training stops automatically
3. Enable checkpoint uploading to preserve progress if the instance terminates
4. Consider spot instances for non-critical testing

## Support

For issues or questions, please refer to the project documentation or create an issue in the repository. 