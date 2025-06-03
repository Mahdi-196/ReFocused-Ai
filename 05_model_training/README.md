# ReFocused-AI Model Training

## Overview
Training pipeline for ReFocused-AI 1.2B parameter GPT-NeoX model with automated checkpointing and **background uploads** to Google Cloud Storage.

## Key Features
- **Clean, focused training script** (~150 lines)
- **GPT-NeoX 1.2B architecture** (industry standard)
- **Background checkpoint uploads** (training doesn't block)
- **Test and production configurations**
- **Real-time monitoring capabilities**
- **Automatic checkpoint management**

## 🚀 Quick Start

1. **Setup environment:**
   ```bash
   bash setup.sh
   ```

2. **Activate environment:**
   ```bash
   source venv/bin/activate      # Linux/Mac
   source venv/Scripts/activate  # Windows
   ```

3. **Start training:**
   ```bash
   python train.py --config test
   # or
   bash run.sh test
   ```

## 📁 File Structure

```
05_model_training/
├── train.py              # Main training script
├── setup.sh              # Environment setup
├── run.sh                # Training launcher
├── debug_gpu.py          # GPU diagnostics
├── configs/              # Training configurations
│   ├── model_config.py   # Model architecture (1.2B params)
│   └── training_config.py # Training hyperparameters
├── utils/                # Utility functions
│   ├── data_utils.py     # Data loading and preprocessing
│   ├── checkpoint_utils.py # Model checkpointing
│   └── training_utils.py # Training utilities
├── checkpoints/          # Saved model checkpoints
├── logs/                 # Training logs
└── cache/                # Data cache
```

## 🔧 Core Files Explained

### **train.py**
Main training script. Handles:
- Model initialization (GPT-NeoX 1.2B)
- Data loading from Google Cloud Storage
- Training loop with gradient accumulation
- Automatic checkpointing
- GPU/CPU compatibility

### **configs/model_config.py**
Defines the model architecture:
- 1.2B parameters
- 24 transformer layers
- 2048 hidden size
- 16 attention heads
- 2048 sequence length

### **configs/training_config.py**
Training configurations:
- **Test config**: 5 files, 100 steps, quick testing
- **Production config**: All files, 10K steps, full training

### **utils/data_utils.py**
Data handling:
- Downloads NPZ files from Google Cloud Storage
- Preprocesses and tokenizes text data
- Creates PyTorch DataLoaders
- Handles batching and sequence padding

### **utils/checkpoint_utils.py**
Model checkpointing:
- Saves model state to local disk
- Uploads checkpoints to Google Cloud Storage
- Handles checkpoint resuming
- Manages storage cleanup

## ⚙️ Configuration Options

### Test Configuration
```python
max_train_files=5        # Small dataset
max_steps=100           # Quick test
batch_size=1            # Small batches
save_steps=50           # Frequent saves
```

### Production Configuration  
```python
max_train_files=None    # Full dataset
max_steps=10000         # Full training
batch_size=4            # Larger batches
save_steps=500          # Less frequent saves
```

## 🖥️ Hardware Requirements

### Minimum (CPU)
- 16GB RAM
- 50GB disk space
- Training will be very slow

### Recommended (GPU)
- NVIDIA GPU with 8GB+ VRAM
- 32GB RAM
- 100GB disk space
- CUDA 12.1 compatible

## 📊 Training Commands

### Basic Training
```bash
# Test run (100 steps)
python train.py --config test

# Production run (10K steps) 
python train.py --config production

# Custom step limit
python train.py --config test --max-steps 50
```

### Using the Run Script
```bash
# Test configuration
bash run.sh test

# Production with custom steps
bash run.sh production 5000
```

## 🔍 Troubleshooting

### Check GPU Status
```bash
python debug_gpu.py
```

This will check:
- NVIDIA drivers installation
- PyTorch CUDA compatibility
- GPU memory and capabilities
- Common configuration issues

### Common Issues

**No GPU detected:**
- Install NVIDIA drivers
- Install CUDA-enabled PyTorch
- Check VM GPU passthrough (if in VM)

**Out of memory:**
- Reduce batch size in config
- Use test config instead of production
- Enable gradient checkpointing

**Data loading errors:**
- Check Google Cloud Storage access
- Verify bucket permissions
- Check internet connection

## 📈 Monitoring Training

### View Progress
Training progress is displayed in the terminal with:
- Current step and loss
- Learning rate
- Training speed

### TensorBoard (Optional)
```bash
pip install tensorboard
tensorboard --logdir logs/
```

### Checkpoints
- Saved every 50-500 steps (configurable)
- Located in `checkpoints/` directory
- Automatically uploaded to Google Cloud Storage
- Can resume training from any checkpoint

## 🛠️ Development

### Adding New Configurations
Edit `configs/training_config.py`:
```python
def get_training_config(config_type: str):
    if config_type == "my_custom_config":
        return TrainingConfig(
            max_steps=1000,
            per_device_train_batch_size=2,
            # ... other settings
        )
```

### Modifying Model Architecture
Edit `configs/model_config.py` to change:
- Model size (hidden_size, num_layers)
- Sequence length (max_position_embeddings)
- Vocabulary size

### Custom Data Processing
Modify `utils/data_utils.py` for:
- Different data formats
- Custom preprocessing
- Alternative data sources

## 📋 Dependencies

**Core Requirements:**
- Python 3.8+
- PyTorch 2.0+ with CUDA
- Transformers 4.36+
- Accelerate 0.25+

**Full list in setup.sh**

---

**Need help?** Run `python debug_gpu.py` to diagnose issues or check the troubleshooting section above.

## Enhanced Checkpointing System

### Comprehensive State Saving
Each checkpoint now includes:
- **Model state**: Weights and parameters
- **Optimizer state**: Adam optimizer state, momentum, etc.
- **Scheduler state**: Learning rate scheduler progress 
- **Training configuration**: Complete config used for training
- **Training metrics**: Loss history, learning rates, validation metrics
- **System information**: CUDA status, mixed precision, device info

### Checkpoint Contents
```
checkpoint-epoch0-step50-files0/
├── pytorch_model.bin          # Model weights (Accelerate format)
├── optimizer.bin              # Optimizer state
├── scheduler.bin              # Scheduler state  
├── scheduler_state.pt         # Explicit scheduler backup
├── training_config.json       # Complete training configuration
├── metadata.json              # Comprehensive checkpoint metadata
├── training_metrics.json      # Training progress and metrics
└── random_states_0.pkl        # Random state for reproducibility
```

### Enhanced Metadata
```json
{
  "step": 50,
  "epoch": 0, 
  "current_loss": 3.2456,
  "best_loss": 3.1234,
  "loss_history": [3.8, 3.6, 3.4, 3.2],
  "learning_rates": [0.0002, 0.00019, 0.00018],
  "validation_metrics": {
    "current_avg_loss": 3.2456,
    "steps_since_best": 10,
    "loss_trend": "improving"
  },
  "training_progress": {
    "completed_steps": 50,
    "total_epochs": 0,
    "files_processed": 0
  },
  "system_info": {
    "cuda_available": false,
    "device_count": 0,
    "mixed_precision": "no"
  }
}
```

### Checkpoint Management Tools

#### View Checkpoint Details
```bash
# List all checkpoints with summary
python scripts/checkpoint_viewer.py list

# View detailed checkpoint information
python scripts/checkpoint_viewer.py view checkpoint-epoch0-step50-files0

# Compare multiple checkpoints
python scripts/checkpoint_viewer.py compare checkpoint-epoch0-step50-files0 checkpoint-epoch0-step100-files0
```

#### Plot Training Metrics
```bash
# Plot loss and learning rate curves
python scripts/checkpoint_viewer.py plot checkpoint-epoch0-step50-files0

# Save plot to file
python scripts/checkpoint_viewer.py plot checkpoint-epoch0-step50-files0 --save training_plot.png
```

#### Resume Training
```python
# Resume from specific checkpoint with full state restoration
python train.py --config test --resume checkpoint-epoch0-step50-files0
```

### Training Metrics Tracking

The enhanced system automatically tracks:
- **Loss History**: Rolling average at each logging interval
- **Learning Rate Schedule**: Complete LR progression  
- **Best Loss**: Best loss achieved during training
- **Validation Metrics**: Extensible validation tracking
- **Training Progress**: Steps, epochs, files processed
- **System State**: Hardware and training environment

### Example Checkpoint Output
```
💾 Saving checkpoint at step 50
✅ Saved scheduler state
✅ Saved training config  
✅ Saved comprehensive metadata
✅ Saved training metrics
🚀 Starting background upload for checkpoint-epoch0-step50-files0
✅ Checkpoint checkpoint-epoch0-step50-files0 queued for background upload
```

### Checkpoint Viewer Examples

#### List Checkpoints
```bash
$ python scripts/checkpoint_viewer.py list
📁 Found 3 checkpoints:
==============================================================================
 1. checkpoint-epoch0-step50-files0        Step: 50     Loss: 3.245600   Time: 2024-01-15 14:30:25
 2. checkpoint-epoch0-step100-files0       Step: 100    Loss: 2.987543   Time: 2024-01-15 14:35:42  
 3. checkpoint-epoch0-step150-files0       Step: 150    Loss: 2.756321   Time: 2024-01-15 14:40:58
```

#### Detailed View
```bash
$ python scripts/checkpoint_viewer.py view checkpoint-epoch0-step50-files0

📊 Checkpoint Summary: checkpoint-epoch0-step50-files0
============================================================
📅 Timestamp: 2024-01-15T14:30:25.123456
🔢 Step: 50
🔄 Epoch: 0
📁 Files Processed: 0
🖥️  CUDA Available: False
🎮 GPU Count: 0
⚡ Mixed Precision: no

📈 Training Metrics:
------------------------------
💥 Current Loss: 3.245600
🏆 Best Loss: 3.123400
📊 Loss History Length: 5
📉 Average Loss: 3.456789
📊 Loss Std Dev: 0.234567
🔄 Loss Trend: ↓
📈 Current LR: 1.90e-04
📊 LR History Length: 50

🎯 Validation Metrics:
------------------------------
  current_avg_loss: 3.245600
  steps_since_best: 10
  loss_trend: improving
```

## Background Upload System

### How It Works
- **Parallel uploads**: Training continues while checkpoints upload in background threads
- **Compressed archives**: Creates tar.gz files for faster, single-file uploads
- **gsutil integration**: Uses Google's optimized upload tool with multithreading
- **Graceful shutdown**: Waits for uploads to complete if training is interrupted

### Upload Modes
```bash
# Default: Background uploads (recommended)
python train.py --config test

# Synchronous uploads (blocks training)
python train.py --config test --no-background-upload
```

### Upload Management

#### Check Upload Status
```bash
python scripts/upload_manager.py status --config test
```

#### List Local Checkpoints
```bash
python scripts/upload_manager.py list
```

#### Upload Specific Checkpoint
```bash
python scripts/upload_manager.py upload checkpoint-epoch0-step50-files0
```

#### Upload All Local Checkpoints
```bash
python scripts/upload_manager.py upload-all --config test
```

## Configuration

### Test Config (`configs/test.json`)
```json
{
  "max_files": 5,
  "max_steps": 100,
  "per_device_train_batch_size": 1,
  "save_steps": 50,
  "logging_steps": 10
}
```

### Production Config (`configs/production.json`)
```json
{
  "max_files": -1,
  "max_steps": 10000,
  "per_device_train_batch_size": 4,
  "save_steps": 500,
  "logging_steps": 100
}
```

## Model Architecture

```
ReFocused-AI 1.2B Parameters
├── 16 transformer layers
├── 2048 hidden dimensions
├── 16 attention heads
├── 50,257 vocabulary size
└── GPT-NeoX architecture
```

## File Structure

```
05_model_training/
├── train.py              # Main training script
├── run.sh                # Quick launcher
├── configs/              # Training configurations
├── utils/                # Training utilities
├── scripts/              # Monitoring and management
├── checkpoints/          # Local checkpoint storage
└── logs/                 # Training logs
```

## Command Line Options

```bash
python train.py [OPTIONS]

Options:
  --config {test,production}     Training configuration (default: test)
  --max-steps INT                Override max training steps
  --resume CHECKPOINT_NAME       Resume from specific checkpoint
  --no-background-upload         Disable background uploads (blocks training)
```

## Monitoring

### Real-time Training Monitor
```bash
# Monitor with custom refresh rate
python scripts/monitor_training.py --refresh 3

# Monitor logs only
python scripts/monitor_training.py --logs-only

# One-time status check
python scripts/monitor_training.py --once
```

### Upload Progress
Training output shows background upload status:
```
💾 Saving checkpoint at step 50
🚀 Starting background upload for checkpoint-epoch0-step50-files0
✅ Checkpoint checkpoint-epoch0-step50-files0 queued for background upload
📦 Creating archive: ./checkpoints/checkpoint-epoch0-step50-files0.tar.gz
☁️  Uploading to gs://refocused-ai/Checkpoints/checkpoint-epoch0-step50-files0.tar.gz
✅ Successfully uploaded checkpoint-epoch0-step50-files0
🗑️  Cleaned up ./checkpoints/checkpoint-epoch0-step50-files0.tar.gz
```

## Google Cloud Storage Integration

### Bucket Structure
```
gs://refocused-ai/Checkpoints/
├── checkpoint-epoch0-step50-files0.tar.gz
├── checkpoint-epoch0-step100-files0.tar.gz
└── checkpoint-epoch0-step150-files0.tar.gz
```

### Authentication
Uses anonymous GCS client to avoid credential issues. Ensure `gsutil` is configured:
```bash
gsutil version  # Should show version info
```

## Performance Benefits

### Background Uploads
- **No training interruption**: Uploads happen in parallel threads
- **Faster uploads**: tar.gz compression + gsutil multithreading
- **Reduced bandwidth**: Single compressed file vs. multiple small files
- **Graceful handling**: Automatic cleanup and error handling

### Upload Speed Comparison
```
Traditional upload: ~2-5 minutes (blocks training)
Background upload:  ~30 seconds (returns immediately)
```

## Troubleshooting

### Upload Issues
```bash
# Check gsutil configuration
gsutil version

# Test bucket access
gsutil ls gs://refocused-ai/

# Manual upload if needed
python scripts/upload_manager.py upload-all
```

### Training Issues
```bash
# Check GPU status
python train.py --config test  # Shows GPU info at startup

# Monitor resource usage
python scripts/monitor_training.py --refresh 2
```

### Signal Handling
Training handles interruption gracefully:
- `Ctrl+C`: Waits for background uploads to complete
- `SIGTERM`: Ensures uploads finish before exit

## Best Practices

1. **Use background uploads** (default) for uninterrupted training
2. **Monitor upload status** with `scripts/upload_manager.py status`
3. **Check logs regularly** with `scripts/monitor_training.py`
4. **Keep local checkpoints** limited (automatic cleanup after 3 checkpoints)
5. **Use test config** for development, production for actual training

## Integration with Main Pipeline

This training module integrates with:
- `04_data_tokenization/` → Provides tokenized training data
- `06_monitoring_validation/` → Model validation and metrics
- Google Cloud Storage → Checkpoint persistence 