# 🚀 ReFocused-AI Complete Setup Guide

Complete instructions to set up and train the ReFocused-AI 1.2B parameter language model with virtual environment and Google Cloud Storage integration.

## 📋 Prerequisites

- **Python 3.9+** (recommended: 3.11)
- **Git** for cloning the repository
- **Google Cloud Service Account** with Storage Object Admin permissions
- **NVIDIA GPU** (optional but recommended for faster training)
- **8GB+ RAM** minimum, 16GB+ recommended

## 🔧 Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/Mahdi-196/ReFocused-Ai.git
cd ReFocused-Ai
```

## 🐍 Step 2: Set Up Virtual Environment

### Option A: Using venv (Recommended)
```bash
# Create virtual environment
python -m venv refocused_env

# Activate virtual environment
# On Windows:
refocused_env\Scripts\activate
# On Mac/Linux:
source refocused_env/bin/activate

# Verify activation (should show refocused_env)
which python
```

### Option B: Using conda
```bash
# Create conda environment
conda create -n refocused_env python=3.11
conda activate refocused_env
```

## 📦 Step 3: Install Dependencies

```bash
# Make sure you're in the virtual environment
# Install required packages
pip install -r requirements.txt

# If you encounter issues, install key packages individually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate google-cloud-storage tqdm numpy
```

## 🔐 Step 4: Set Up Google Cloud Credentials

### 4.1 Create Credentials Directory
```bash
# Navigate to training folder
cd 05_model_training

# Create credentials directory
mkdir -p credentials
```

### 4.2 Add Your Service Account Key
**IMPORTANT**: You need to place your Google Cloud service account JSON key file in the credentials folder.

```bash
# Copy your service account key file to:
# 05_model_training/credentials/black-dragon-461023-t5-93452a49f86b.json

# The file should contain your Google Cloud service account credentials
# Contact the project administrator if you don't have this file
```

### 4.3 Verify Credentials Structure
Your credentials file should look like this:
```json
{
  "type": "service_account",
  "project_id": "black-dragon-461023-t5",
  "private_key_id": "...",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...",
  "client_email": "trainer-sa@black-dragon-461023-t5.iam.gserviceaccount.com",
  "client_id": "...",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token"
}
```

## 🧪 Step 5: Test Setup

```bash
# Make sure you're in 05_model_training directory
cd 05_model_training

# Run a very short training test (5 steps)
# Pass credentials explicitly (no env vars)
./start_training.sh --config test --max-steps 5 --gcs-credentials /absolute/path/to/key.json --gcp-project your-project-id
```

## 🚀 Step 6: Start Training

### Quick Start (Recommended)
```bash
# Make sure you're in the virtual environment and training directory
cd 05_model_training

# Test training (1000 steps, ~10-30 minutes)
./start_training.sh --config test

# Production training (10000 steps, several hours)
./start_training.sh --config production
```

### Advanced Training Options
```bash
# Custom number of steps
./start_training.sh --config test --max-steps 2000

# Resume from checkpoint
./start_training.sh --config test --resume checkpoint-epoch1-step500-files3

# Disable background uploads (slower but more reliable)
./start_training.sh --config test --no-background-upload
```

### Manual Setup (Alternative)
```bash
# Run training directly (pass credentials explicitly)
python train.py --config test \
  --gcs-credentials /absolute/path/to/key.json \
  --gcp-project your-project-id

# Or with custom options
python train.py --config production --max-steps 5000 \
  --gcs-credentials /absolute/path/to/key.json \
  --gcp-project your-project-id
```

## 📊 Step 7: Monitor Training

### Real-time Monitoring
```bash
# Watch training logs
tail -f logs/training.log

# Monitor local checkpoints
watch "ls -la checkpoints/"

# Monitor GCS uploads
gsutil ls -l gs://refocused-ai/Checkpoints/ | tail -10
```

### Check Training Progress
```bash
# Check if training is running
ps aux | grep train.py

# View latest logs
ls -la logs/

# Check recent checkpoints
ls -la checkpoints/
```

## 🗂️ Training Configurations

### Test Configuration (Default)
- **Max Steps**: 1000
- **Training Files**: 5 files
- **Batch Size**: 1
- **Save Every**: 100 steps
- **Duration**: ~10-30 minutes
- **Good for**: Testing setup, quick experiments

### Production Configuration
- **Max Steps**: 10000
- **Training Files**: All available
- **Batch Size**: 4 (effective: 16 with gradient accumulation)
- **Save Every**: 500 steps
- **Duration**: Several hours to days
- **Good for**: Full model training

## ☁️ Checkpoint System

### Automatic Features
- ✅ **Background uploads** to `gs://refocused-ai/Checkpoints/`
- ✅ **Authenticated access** (no public permissions needed)
- ✅ **Comprehensive metadata** saved with each checkpoint
- ✅ **Automatic cleanup** of old local checkpoints
- ✅ **Resume capability** from any checkpoint

### Checkpoint Contents
Each checkpoint includes:
- `pytorch_model.bin` - Model weights
- `config.json` - Model configuration  
- `training_args.bin` - Training arguments
- `optimizer.pt` - Optimizer state
- `scheduler.pt` - Learning rate scheduler state
- `metadata.json` - Training metrics and metadata

## 🔧 Troubleshooting

### Virtual Environment Issues
```bash
# If venv activation fails
python -m venv --clear refocused_env
source refocused_env/bin/activate  # Mac/Linux
# or
refocused_env\Scripts\activate  # Windows

# Verify Python version in venv
python --version
which python
```

### Authentication Errors
```bash
# Check credentials file exists
ls -la 05_model_training/credentials/

# start_training.sh will validate access via provided credentials path
cd 05_model_training
python -c "from google.cloud import storage; print('OK') if storage.Client().bucket('refocused-ai') else print('FAIL')"
```

### Memory Issues
```bash
# Reduce batch size in configs/training_config.py
# Edit the file and change:
per_device_train_batch_size: int = 1
gradient_accumulation_steps: int = 1
```

### Missing Training Data
```bash
# Download training data manually
cd 05_model_training
python download_training_data.py
```

### Permission Errors
```bash
# Make scripts executable
chmod +x 05_model_training/start_training.sh
```

## 🧩 Step 8: Fine‑tune (optional)

After base training, adapt the model to specific tasks using the fine‑tuning module.

```bash
# Chat fine‑tuning (full FT)
python 06_fine_tuning/fine_tune.py \
  --task chat \
  --base-model 05_model_training/checkpoints/final_model \
  --dataset ./datasets/chat_data.jsonl \
  --output-dir ./fine_tuned_models

# Instruction fine‑tuning with LoRA (parameter‑efficient)
python 06_fine_tuning/fine_tune.py \
  --task instruct \
  --base-model 05_model_training/checkpoints/final_model \
  --dataset ./datasets/instruct.jsonl \
  --lora --lora-rank 8 \
  --output-dir ./fine_tuned_models

# Useful flags: --freeze-ratio 0.7, --gradient-checkpointing, --mixed-precision bf16, --resume <checkpoint>
```

## 🧰 Utilities by Stage

- Data collection/processing:
  - utilities/data_processing/quick_dataset_size_check.py: Estimate dataset size and tokens.
  - utilities/data_processing/analyze_tokenized_data.py: Validate tokenized shards.
  - utilities/data_processing/check_missing_files.py: Detect missing/corrupt files.
  - utilities/data_processing/count_final_sequences.py: Count sequences across shards.
  - utilities/data_processing/quick_bucket_check.py: Sanity‑check GCS bucket contents.

- Training planning and monitoring:
  - utilities/analysis/analyze_training_parameters.py: Suggest steps, batch sizes, scaling.
  - utilities/analysis/8gpu_analysis.py: Recommendations for 8‑GPU runs.
  - utilities/training/resume_after_disk_full.py: Cleanup and resume after disk full.

- Deployment/cleanup:
  - utilities/deployment/cleanup_summary.py: Summarize and clean old checkpoints/models.

## 📈 Expected Performance

### Hardware Requirements
- **CPU Only**: Very slow, 10-100x slower than GPU
- **NVIDIA GPU**: Recommended, 8GB+ VRAM
- **RAM**: 16GB+ recommended for smooth operation
- **Storage**: 20GB+ free space for checkpoints and cache

### Training Times
- **Test Config (1000 steps)**:
  - GPU: 10-30 minutes
  - CPU: 2-6 hours
- **Production Config (10000 steps)**:
  - GPU: 2-8 hours  
  - CPU: 1-3 days

## 🎯 Quick Command Reference

```bash
# Complete setup sequence
git clone https://github.com/Mahdi-196/ReFocused-Ai.git
cd ReFocused-Ai
python -m venv refocused_env
source refocused_env/bin/activate  # Mac/Linux
# or refocused_env\Scripts\activate  # Windows
pip install -r requirements.txt
cd 05_model_training
# [Place credentials file in credentials/ folder]
./start_training.sh --config test

# Monitor training
tail -f logs/training.log

# Check checkpoints
gsutil ls gs://refocused-ai/Checkpoints/
```

## 🎉 Success Indicators

You'll know everything is working when you see:
- ✅ Virtual environment activated
- ✅ Dependencies installed without errors
- ✅ Authentication successful
- ✅ Training data loaded (5+ files found)
- ✅ Model initialized (~1.2B parameters)
- ✅ Training steps progressing
- ✅ Checkpoints uploading to GCS

## 📞 Support

If you encounter issues:
1. **Check this guide** for troubleshooting steps
2. **Review logs** in `05_model_training/logs/`
3. **Verify credentials** and permissions
4. **Check virtual environment** is activated
5. **Ensure dependencies** are properly installed

## 🔄 Reactivating Environment

Each time you return to work on the project:
```bash
# Navigate to project
cd ReFocused-Ai

# Activate virtual environment
source refocused_env/bin/activate  # Mac/Linux
# or refocused_env\Scripts\activate  # Windows

# Navigate to training directory
cd 05_model_training

# Start training
./start_training.sh --config test
```

Your ReFocused-AI training system is now ready! 🚀 