# ReFocused-AI Model Training

Simple, clean training setup for the ReFocused-AI 1.2B parameter language model.

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