# ReFocused-AI 1B Parameter Model Training

This folder contains the complete training infrastructure for the ReFocused-AI 1B parameter language model, optimized for efficiency and robustness.

## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd 05_training

# Make scripts executable
chmod +x setup_environment.sh launch_test_training.sh launch_production_training.sh

# Run setup (installs all dependencies)
./setup_environment.sh

# Activate environment
source venv/bin/activate
source .env
```

### 2. Configure Google Cloud Authentication
```bash
# Set your GCS credentials
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
```

### 3. Test Run (25 files, 1 GPU)
```bash
./launch_test_training.sh
```

### 4. Production Run (Full dataset, 8 GPUs)
```bash
./launch_production_training.sh
```

## 📊 Model Architecture

- **Parameters**: ~1.1B
- **Architecture**: GPT-style decoder-only transformer
- **Hidden Size**: 1536
- **Layers**: 24
- **Attention Heads**: 24 (with 8 KV heads for GQA)
- **Context Length**: 2048 tokens
- **Vocabulary Size**: 50,000

### Key Optimizations:
1. **HybridNorm**: Combines QKV normalization in attention with Post-Norm in FFN for stable training
2. **SwiGLU Activation**: More efficient than standard GELU
3. **Grouped Query Attention (GQA)**: Reduces memory usage with 8 KV heads
4. **Rotary Position Embeddings (RoPE)**: Better position encoding
5. **Flash Attention 2**: Faster attention computation
6. **Mixed Precision Training**: FP16 with dynamic loss scaling
7. **DeepSpeed ZeRO-2**: Optimizer state sharding across GPUs

## 📁 File Structure

```
05_training/
├── model_config.py          # Model and training configurations
├── model.py                 # Model implementation with HybridNorm
├── data_loader.py          # GCS data streaming and batching
├── optimizer.py            # AdamW optimizer with cosine schedule
├── train.py                # Main training script
├── setup_environment.sh    # Environment setup script
├── launch_test_training.sh # Test run launcher (25 files)
├── launch_production_training.sh # Full training launcher
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Model Configuration (`ModelConfig`)
- Adjust model size parameters in `model_config.py`
- Default: 1.1B parameters optimized for H100 GPUs

### Training Configuration (`TrainingConfig`)
- **Batch Size**: 8 per GPU × 16 gradient accumulation = 128 effective
- **Learning Rate**: 3e-4 with cosine decay
- **Warmup Steps**: 2000
- **Weight Decay**: 0.1
- **Gradient Clipping**: 1.0

## 💾 Data and Checkpointing

### Data Source
- **Bucket**: `gs://refocused-ai/`
- **Format**: Tokenized `.npy` files
- **Streaming**: Direct from GCS, no local storage needed

### Checkpoints
- **Frequency**: Every 5 files processed
- **Location**: `gs://refocused-ai/Checkpoints/`
- **Contents**: Model weights, optimizer state, training state

## 📈 Monitoring

### Weights & Biases
- Automatic logging of loss, learning rate, gradient norms
- Project name: `refocused-ai-1b`
- Real-time tracking at [wandb.ai](https://wandb.ai)

### TensorBoard
- Local logs in `logs/` directory
- View with: `tensorboard --logdir logs/`

### Console Output
- Real-time metrics every 10 steps
- File processing progress
- Checkpoint save notifications

## 🚨 Troubleshooting

### Out of Memory
- Reduce `micro_batch_size` in `model_config.py`
- Enable more aggressive gradient checkpointing
- Increase `gradient_accumulation_steps`

### Slow Data Loading
- Increase `num_workers` in training config
- Check GCS bandwidth and region
- Enable data prefetching

### Training Instability
- Reduce learning rate
- Increase warmup steps
- Check for gradient explosion (monitor grad_norm)

## 🔄 Resume Training

To resume from a checkpoint:
```bash
python train.py --mode production --resume gs://refocused-ai/Checkpoints/checkpoint_step_X_files_Y
```

## 🧪 Testing

Run a quick test to verify setup:
```bash
python -c "from model import GPTModel; from model_config import ModelConfig; m = GPTModel(ModelConfig()); print(f'Model parameters: {sum(p.numel() for p in m.parameters())/1e9:.2f}B')"
```

## 📝 Notes

- The model uses your custom tokenizer from `../tokenizer_1B`
- All scripts include automatic failsafes and error recovery
- Checkpoints are automatically uploaded to GCS
- Training is optimized for H100 80GB GPUs but can run on other GPUs

## 🤝 Contributing

When modifying the training code:
1. Test changes with the test run first
2. Monitor metrics carefully for regressions
3. Document any new hyperparameters
4. Update this README with significant changes

## 📊 Expected Performance

- **Training Speed**: ~500k tokens/second on 8x H100
- **Memory Usage**: ~60GB per GPU
- **Time to Train**: ~2-3 days for 22B tokens
- **Final Loss**: Expected <2.5 with proper data

## 🔗 References

- [HybridNorm Paper](https://arxiv.org/abs/2503.04598)
- [Flash Attention 2](https://github.com/Dao-AILab/flash-attention)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Efficient Training Guide](https://www.e2enetworks.com/blog/efficiently-training-transformers-a-comprehensive-guide-to-high-performance-nlp-models)

## Environment Setup

### Python Version Requirements

This training module requires Python 3.10.x for optimal compatibility with all dependencies. Using other Python versions may cause compatibility issues with certain libraries.

### Fixed Version Dependencies

To ensure reproducibility and consistency, we now use exact versions for all dependencies:

- PyTorch: 2.1.0 (with CUDA 12.1)
- Transformers: 4.35.0
- DeepSpeed: 0.12.0
- TensorBoard: 2.15.0
- Flash Attention: 2.3.0
- Accelerate: 0.25.0

The `setup_environment.sh` script checks for Python version consistency and installs all dependencies with their exact versions.

### Version Consistency

The training scripts now include version checks to ensure all dependencies are installed with the correct versions. This prevents issues that can arise from mismatched library versions.

## Version Verification

When running training scripts, they will:

1. Verify the Python version matches the one used during setup
2. Check that TensorBoard and other key dependencies have the correct versions
3. Provide warnings if any version mismatches are detected

If you encounter any issues with library compatibility, please check the logs for version mismatch warnings. 