# ReFocused-AI Model Training

Complete training system for the ReFocused-AI 1.2B parameter language model with automatic checkpoint uploading to Google Cloud Storage.

## 🚀 Quick Start

### 1. Start Training (Recommended)
```bash
cd 05_model_training
./start_training.sh
```

### 2. Alternative: Manual Setup
```bash
cd 05_model_training
source setup_auth.sh
python train.py --config test
```

## 📋 Training Options

### Basic Usage
```bash
# Test training (1000 steps, 5 files)
./start_training.sh --config test

# Production training (10000 steps, all files)
./start_training.sh --config production

# Override steps
./start_training.sh --config test --max-steps 2000

# Resume from checkpoint
./start_training.sh --config test --resume checkpoint-epoch1-step500-files3

# Disable background uploads (slower)
./start_training.sh --config test --no-background-upload
```

### Configuration Details

#### Test Configuration
- **Max Steps**: 1000
- **Files**: 5 training files
- **Batch Size**: 1
- **Save Every**: 100 steps
- **Log Every**: 25 steps

#### Production Configuration  
- **Max Steps**: 10000
- **Files**: All available training files
- **Batch Size**: 4
- **Gradient Accumulation**: 4 (effective batch size: 16)
- **Save Every**: 500 steps
- **Log Every**: 100 steps

## 🗂️ Directory Structure

```
05_model_training/
├── train.py                 # Main training script
├── start_training.sh        # Complete training startup script
├── setup_auth.sh           # Authentication setup (Linux/Mac)
├── setup_auth.bat          # Authentication setup (Windows)
├── download_training_data.py # Download training data from GCS
├── configs/
│   ├── training_config.py   # Training configurations
│   └── model_config.py      # Model architecture config
├── utils/
│   ├── checkpoint_utils.py  # GCS checkpoint management
│   ├── data_utils.py        # Data loading utilities
│   └── training_utils.py    # Training helper functions
├── scripts/
│   └── monitor_training.py  # Training monitoring
├── credentials/
│   └── black-dragon-*.json  # GCS service account key
├── checkpoints/             # Local checkpoint storage
├── logs/                    # Training logs
└── cache/                   # Cached data
```

## ☁️ Checkpoint System

### Automatic Background Upload
- **Bucket**: `refocused-ai`
- **Path**: `Checkpoints/`
- **Format**: `checkpoint-epoch{N}-step{N}-files{N}/`
- **Upload**: Background threads (non-blocking)
- **Authentication**: Service account credentials

### Checkpoint Contents
Each checkpoint includes:
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration
- `training_args.bin` - Training arguments
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Learning rate scheduler state
- `metadata.json` - Comprehensive training metadata

### Manual Checkpoint Operations
```bash
# List available checkpoints
gsutil ls gs://refocused-ai/Checkpoints/

# Download specific checkpoint
gsutil -m cp -r gs://refocused-ai/Checkpoints/checkpoint-name ./checkpoints/

# Upload manual checkpoint
gsutil -m cp -r ./checkpoints/checkpoint-name gs://refocused-ai/Checkpoints/
```

## 🔐 Authentication Setup

Your training system uses authenticated Google Cloud Storage access:

- **Project**: `black-dragon-461023-t5`
- **Service Account**: `trainer-sa@black-dragon-461023-t5.iam.gserviceaccount.com`
- **Permissions**: Storage Object Admin
- **Credentials**: `./credentials/black-dragon-461023-t5-93452a49f86b.json`

## 📊 Monitoring Training

### Real-time Monitoring
```bash
# Watch training progress
tail -f logs/training.log

# Monitor checkpoints
watch "ls -la checkpoints/"

# Monitor GCS uploads
gsutil ls -l gs://refocused-ai/Checkpoints/ | tail -10
```

### Training Metrics
- **Loss tracking**: Every 25/100 steps
- **Learning rate**: Cosine scheduler with warmup
- **Checkpoints**: Every 100/500 steps
- **Best loss tracking**: Automatic
- **Validation metrics**: Loss trend analysis

## 🔧 Troubleshooting

### Common Issues

**Authentication Error**
```bash
# Check credentials file exists
ls -la credentials/

# Test authentication
gcloud auth application-default print-access-token
```

**Out of Memory**
```bash
# Reduce batch size in configs/training_config.py
per_device_train_batch_size: int = 1
gradient_accumulation_steps: int = 1
```

**Missing Training Data**
```bash
# Download training data
python download_training_data.py
```

**Upload Failures**
```bash
# Check network connectivity
gsutil ls gs://refocused-ai/

# Force synchronous uploads
./start_training.sh --no-background-upload
```

## 🎯 Training Tips

1. **Start with test config** to verify everything works
2. **Monitor GPU memory** usage during training
3. **Background uploads** are enabled by default for efficiency
4. **Checkpoints auto-saved** every 100/500 steps
5. **Training can be resumed** from any checkpoint
6. **Logs are saved** to `./logs/` directory

## 📈 Performance

### Expected Performance
- **Test Config**: ~10-30 minutes depending on hardware
- **Production Config**: Several hours to days
- **GPU Memory**: ~8-16GB for batch size 1
- **Checkpoint Upload**: ~1-5 minutes per checkpoint

### Optimization
- Use GPU for faster training
- Increase batch size if memory allows
- Enable mixed precision (bf16) for speed
- Background uploads prevent blocking

## 🎉 Next Steps

After training completes:
1. **Checkpoints** are automatically uploaded to GCS
2. **Model weights** can be loaded for inference
3. **Training metrics** are saved in metadata
4. **Resume training** from any checkpoint if needed

For inference and deployment, load your model from the latest checkpoint in `gs://refocused-ai/Checkpoints/`. 