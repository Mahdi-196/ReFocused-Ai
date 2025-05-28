# 1B Parameter Model Training on 8x H100 SXM

Complete training infrastructure for your 1B parameter GPT model using DeepSpeed ZeRO Stage 3 on Hyperbolic Labs 8x H100 SXM setup.

## 🎯 Training Overview

- **Model**: 1B parameter GPT-2 architecture
- **Dataset**: 51.6B tokens (20GB) from Reddit/HuggingFace
- **Hardware**: 8x NVIDIA H100 SXM (80GB each)
- **Framework**: PyTorch + DeepSpeed ZeRO Stage 3
- **Storage**: Google Cloud Storage for data and checkpoints
- **Estimated Cost**: ~$316 for complete training (40 hours @ $7.92/hr)

## 📁 Directory Structure

```
05_model_training/
├── config/
│   ├── model_config.json          # 1B parameter model configuration
│   ├── deepspeed_config.json      # DeepSpeed ZeRO Stage 3 config
│   └── training_config.yaml       # Main training configuration
├── utils/
│   ├── data_loader.py             # Efficient data loading from GCS
│   ├── monitoring.py              # GPU and system monitoring
│   └── checkpoint_manager.py      # Checkpoint backup and management
├── scripts/
│   └── setup_environment.sh       # Environment setup script
├── train.py                       # Main training script
├── requirements_training.txt      # Training dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone your repository and navigate to training directory
cd 05_model_training

# Run environment setup (installs dependencies, creates directories)
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### 2. Configure Google Cloud Storage

```bash
# Upload your GCP service account key
# (Download from GCP Console -> IAM & Admin -> Service Accounts)
scp your-gcp-key.json user@instance:/scratch/gcp-key.json

# Set proper permissions
chmod 600 /scratch/gcp-key.json
```

### 3. Upload Training Data

Upload your tokenized data to Google Cloud Storage:

```bash
# Using gsutil (on your local machine)
gsutil -m cp -r /path/to/your/data_tokenized_production gs://refocused-ai/tokenized_data/

# Verify upload
gsutil ls -l gs://refocused-ai/tokenized_data/ | head -10
```

### 4. Launch Training

```bash
# Source environment variables
source /scratch/training_env.sh

# Start training (includes monitoring dashboard)
/scratch/launch_training.sh

# Or start without monitoring
/scratch/launch_training.sh --no-monitor
```

### 5. Monitor Training

```bash
# View monitoring dashboard
screen -r monitor

# Check training logs
tail -f /scratch/logs/training.log

# Check GPU utilization
watch nvidia-smi
```

## ⚙️ Configuration Details

### Model Configuration (1B Parameters)

```json
{
  "n_embd": 1536,      # Embedding dimension
  "n_layer": 24,       # Number of transformer layers
  "n_head": 24,        # Number of attention heads
  "n_positions": 2048, # Maximum sequence length
  "vocab_size": 50000  # Vocabulary size
}
```

### Training Parameters

- **Batch Size**: 512 total (4 per GPU × 16 accumulation × 8 GPUs)
- **Learning Rate**: 3e-4 with cosine decay
- **Warmup Steps**: 2,000
- **Total Steps**: 100,000
- **Mixed Precision**: FP16
- **Gradient Clipping**: 1.0

### DeepSpeed ZeRO Stage 3 Features

- **Parameter Partitioning**: Splits model across GPUs/CPU
- **CPU Offloading**: Offloads optimizer states and parameters
- **Gradient Checkpointing**: Reduces memory usage
- **Overlap Communication**: Overlaps computation and communication

## 💰 Cost Optimization

### Estimated Costs

```python
# Training time estimation
tokens_per_step = 512 * 2048  # batch_size * seq_length
total_steps = 51_600_000_000 // tokens_per_step  # ~49K steps
estimated_hours = total_steps / 350  # ~140 hours at 350 steps/hour
estimated_cost = estimated_hours * 7.92  # $1,109

# With optimizations (target: 100K steps)
optimized_hours = 100_000 / 2500  # 40 hours at 2500 steps/hour
optimized_cost = optimized_hours * 7.92  # $317
```

### Cost Reduction Strategies

1. **Efficient Batch Size**: Use largest batch size that fits in memory
2. **Mixed Precision**: FP16 reduces memory and increases speed
3. **Gradient Checkpointing**: Trade compute for memory
4. **CPU Offloading**: Use system RAM for optimizer states
5. **Frequent Checkpointing**: Save progress to avoid restart costs

### Monitoring Costs

```bash
# Check training progress and estimate remaining time
python3 -c "
import json
import glob

# Parse training logs
log_files = glob.glob('/scratch/logs/training_metrics.jsonl')
if log_files:
    with open(log_files[0], 'r') as f:
        lines = f.readlines()
        if lines:
            last_metrics = json.loads(lines[-1])
            step = last_metrics['step']
            steps_per_sec = last_metrics.get('steps_per_second', 0)
            
            remaining_steps = 100000 - step
            remaining_hours = remaining_steps / (steps_per_sec * 3600) if steps_per_sec > 0 else 0
            remaining_cost = remaining_hours * 7.92
            
            print(f'Current step: {step:,}/100,000')
            print(f'Steps/sec: {steps_per_sec:.2f}')
            print(f'Estimated remaining: {remaining_hours:.1f} hours (${remaining_cost:.2f})')
else:
    print('No training logs found')
"
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

```bash
# Reduce batch size in config/training_config.yaml
per_device_train_batch_size: 2  # Reduce from 4

# Or increase gradient accumulation
gradient_accumulation_steps: 32  # Increase from 16
```

#### 2. Slow Data Loading

```bash
# Increase data loader workers
dataloader_num_workers: 8  # Increase from 4

# Check data sync status
ls -la /scratch/shards/
```

#### 3. DeepSpeed Initialization Errors

```bash
# Check DeepSpeed installation
python3 -c "import deepspeed; print(deepspeed.__version__)"

# Verify CUDA ops
python3 -c "from deepspeed.ops.adam import FusedAdam; print('OK')"

# Check NCCL communication
export NCCL_DEBUG=INFO
```

#### 4. GCS Connection Issues

```bash
# Verify credentials
export GOOGLE_APPLICATION_CREDENTIALS=/scratch/gcp-key.json
gsutil ls gs://refocused-ai/

# Test bucket access
python3 -c "
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('refocused-ai')
print(f'Bucket accessible: {bucket.exists()}')
"
```

### Performance Optimization

#### GPU Utilization

Target: >85% GPU utilization

```bash
# Monitor GPU usage
nvidia-smi dmon -s u

# If utilization is low:
# 1. Increase batch size
# 2. Reduce data loading bottlenecks
# 3. Check CPU utilization
```

#### Memory Usage

Target: >90% GPU memory utilization

```bash
# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# If memory usage is low:
# 1. Increase batch size
# 2. Reduce CPU offloading
# 3. Disable gradient checkpointing
```

#### Training Speed

Target: >2000 steps/hour

```bash
# Monitor training speed
tail -f /scratch/logs/training.log | grep "steps/s"

# If speed is low:
# 1. Check data loading pipeline
# 2. Reduce logging frequency
# 3. Optimize data preprocessing
```

## 📊 Monitoring and Logging

### Real-time Monitoring

```bash
# GPU monitoring dashboard
screen -r monitor

# Training metrics
tail -f /scratch/logs/training.log

# System resources
htop
nvtop
```

### Weights & Biases Integration

Training automatically logs to Weights & Biases:

- **Loss curves**: Training and validation loss
- **Learning rate schedule**: Cosine decay with warmup
- **GPU metrics**: Utilization, memory, temperature
- **System metrics**: CPU, RAM, disk usage
- **Throughput**: Tokens/second, steps/second

### Log Files

```bash
/scratch/logs/
├── training.log              # Main training log
├── training_metrics.jsonl    # Structured metrics
├── performance_metrics.jsonl # Performance data
└── gpu_metrics.jsonl         # GPU monitoring data
```

## 🔄 Checkpoint Management

### Automatic Backups

- **Frequency**: Every 1,000 steps
- **Local Storage**: `/scratch/checkpoints/`
- **Remote Backup**: `gs://refocused-ai/checkpoints/`
- **Retention**: Last 5 checkpoints locally

### Manual Checkpoint Operations

```bash
# List available checkpoints
python3 -c "
from utils.checkpoint_manager import CheckpointManager
cm = CheckpointManager('/scratch/checkpoints', 'refocused-ai', 'checkpoints')
checkpoints = cm.list_checkpoints()
for cp in checkpoints[-5:]:
    print(f'{cp[\"name\"]}: step {cp[\"metadata\"].get(\"global_step\", \"unknown\")}')
"

# Download specific checkpoint
python3 -c "
from utils.checkpoint_manager import CheckpointManager
cm = CheckpointManager('/scratch/checkpoints', 'refocused-ai', 'checkpoints')
cm.download_checkpoint('checkpoint_step_50000')
"

# Resume from checkpoint
/scratch/launch_training.sh --resume_from_checkpoint /scratch/checkpoints/checkpoint_step_50000
```

## 🎯 Expected Results

### Training Metrics

- **Initial Loss**: ~10.0 (random initialization)
- **Target Loss**: ~3.5-4.0 (well-trained model)
- **Training Time**: 40-50 hours
- **Total Cost**: $320-$400

### Model Performance

After training, your model will be capable of:

- **Text Generation**: Coherent paragraph-length text
- **Context Understanding**: 2048 token context window
- **Domain Knowledge**: Reddit/HuggingFace content understanding
- **Fine-tuning Ready**: Base model for further specialization

### Output Files

```bash
/scratch/checkpoints/final_checkpoint_step_100000/  # Final checkpoint
/scratch/checkpoints/hf_model/                      # HuggingFace format
gs://refocused-ai/final_model/                      # Uploaded model
```

## 🚀 Next Steps

1. **Evaluation**: Test model on validation dataset
2. **Fine-tuning**: Adapt for specific tasks
3. **Deployment**: Serve model with FastAPI or Triton
4. **Scaling**: Train larger models or different architectures

## 📞 Support

For issues or questions:

1. Check the troubleshooting section above
2. Review DeepSpeed documentation: [deepspeed.ai](https://deepspeed.ai)
3. Check Hugging Face documentation: [huggingface.co/docs](https://huggingface.co/docs)

## 📄 License

This training infrastructure is provided as-is for educational and research purposes.

---

**Happy Training! 🎯🚀** 