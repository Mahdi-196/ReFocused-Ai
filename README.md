# 🚀 ReFocused-AI: 1.2B Parameter Language Model

Complete training system for the ReFocused-AI language model with Google Cloud Storage integration and authenticated checkpoint uploading.

## 📋 Quick Start

**👉 For complete setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

### TL;DR Setup
```bash
# Clone and setup
git clone https://github.com/Mahdi-196/ReFocused-Ai.git
cd ReFocused-Ai
python -m venv refocused_env
source refocused_env/bin/activate  # Linux/Mac
# or refocused_env\Scripts\activate  # Windows
pip install -r requirements.txt

# Configure training
cd 05_model_training
# [Add your service account key to credentials/]
./start_training.sh --config test
```

## 🗂️ Project Structure

```
ReFocused-Ai/
├── 05_model_training/          # 🎯 Main training system
│   ├── train.py               # Main training script
│   ├── start_training.sh      # One-click training startup
│   ├── TRAINING_README.md     # Detailed training docs
│   ├── configs/               # Training & model configs
│   ├── utils/                 # Training utilities
│   ├── credentials/           # GCS service account keys
│   └── checkpoints/           # Local checkpoint storage
├── SETUP_GUIDE.md             # 📖 Complete setup guide
├── requirements.txt           # Python dependencies
└── [other directories...]     # Data processing, utilities
```

## ✨ Key Features

- **🔐 Authenticated GCS Access**: Secure checkpoint uploading to `refocused-ai/Checkpoints/`
- **⚡ Background Uploads**: Non-blocking checkpoint uploads during training
- **🎛️ Flexible Configurations**: Test (1000 steps) and Production (10000 steps) modes
- **📊 Comprehensive Monitoring**: Training metrics, loss tracking, and progress logging
- **🔄 Resume Capability**: Continue training from any checkpoint
- **🧪 Easy Testing**: Quick setup verification with short training runs

## 🚀 Training Quick Reference

```bash
# Test training (recommended first)
./start_training.sh --config test

# Production training
./start_training.sh --config production

# Custom steps
./start_training.sh --config test --max-steps 2000

# Resume from checkpoint
./start_training.sh --config test --resume checkpoint-name

# Monitor progress
tail -f logs/training.log
```

## 📊 Training Configurations

| Config | Steps | Files | Batch Size | Duration | Purpose |
|--------|-------|-------|------------|----------|---------|
| **test** | 1000 | 5 | 1 | ~10-30 min | Testing, experiments |
| **production** | 10000 | All | 4 | Hours-days | Full training |

## ☁️ Checkpoint System

- **Automatic uploads** to `gs://refocused-ai/Checkpoints/`
- **Comprehensive metadata** with training metrics
- **Background processing** for non-blocking uploads
- **Resume capability** from any checkpoint
- **Local cleanup** of old checkpoints

## 🔧 Requirements

- **Python 3.9+** (recommended: 3.11)
- **PyTorch 2.0+** with CUDA support (optional but recommended)
- **Google Cloud Storage** access with service account credentials
- **8GB+ RAM** (16GB+ recommended)
- **NVIDIA GPU** (optional but significantly faster)

## 📖 Documentation

- **[SETUP_GUIDE.md](SETUP_GUIDE.md)**: Complete setup instructions with virtual environment
- **[05_model_training/TRAINING_README.md](05_model_training/TRAINING_README.md)**: Detailed training documentation
- **[configs/](05_model_training/configs/)**: Training and model configuration files

## 🎯 Model Details

- **Architecture**: GPT-NeoX
- **Parameters**: ~1.4 billion
- **Context Length**: 1024 tokens  
- **Vocabulary**: 50,257 tokens
- **Training Data**: Reddit conversations (cleaned and tokenized)

## 🤝 Contributing

1. Follow the setup guide to get training working
2. Make changes in appropriate directories
3. Test with `./start_training.sh --config test --max-steps 5`
4. Submit pull requests with clear descriptions

## 📞 Support

For issues:
1. Check [SETUP_GUIDE.md](SETUP_GUIDE.md) troubleshooting section
2. Review training logs in `05_model_training/logs/`
3. Verify virtual environment and dependencies
4. Ensure credentials are properly configured

---

**Ready to train? Start with [SETUP_GUIDE.md](SETUP_GUIDE.md)! 🚀** 