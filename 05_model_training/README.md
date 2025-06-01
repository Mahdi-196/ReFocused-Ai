# ReFocused-AI Model Training

Optimized training pipeline for the ReFocused-AI 1.2B parameter model with advanced preprocessing, monitoring, and performance optimizations.

## 🚀 Key Optimizations

### Data Preprocessing Optimizations
- **Flatten Once Per File**: Data is flattened and cached once per file, eliminating repeated processing
- **Preprocessing Cache**: Preprocessed data is saved to disk (`./preprocessed_cache/`) to avoid recomputation
- **Optimized Dataset**: Uses `OptimizedTokenizedDataset` for faster data loading
- **Efficient Batching**: Eliminates on-the-fly reshaping with pre-shaped sequences

### Performance Monitoring
- **Enhanced Metrics Tracker**: Detailed performance monitoring with system metrics
- **Real-time Monitor**: Live training progress monitoring with `monitor_training.py`
- **Performance Profiling**: Optional I/O and computation profiling
- **System Monitoring**: CPU, memory, GPU, and disk usage tracking

### Training Optimizations
- **Step Limiting**: Configurable step limits for quick test runs
- **FSDP Support**: Fully Sharded Data Parallel for large models
- **Memory Efficiency**: Gradient checkpointing and mixed precision training
- **Smart Checkpointing**: Checkpoint on file completion and step intervals

## 📁 Directory Structure

```
05_model_training/
├── configs/               # Training configurations
│   ├── model_config.py   # Model architecture settings  
│   └── training_config.py # Training hyperparameters
├── utils/                 # Utility modules
│   ├── data_utils.py     # Optimized data loading
│   ├── training_utils.py # Enhanced metrics tracking
│   └── checkpoint_utils.py # Checkpoint management
├── scripts/              # Helper scripts
│   └── monitor_training.py # Real-time monitoring
├── train.py              # Main training script
├── test_dataloader.py    # Dataloader testing
└── run_optimized_training.sh # Training launcher
```

## 🏋️ Quick Start

### Test Run (Recommended)
```bash
# Run optimized test training (100 steps, 25 files)
./run_optimized_training.sh test

# With profiling enabled
./run_optimized_training.sh test true

# With custom step limit
./run_optimized_training.sh test false 50
```

### Production Training
```bash
# Full production training
./run_optimized_training.sh production

# With monitoring
START_MONITOR=true ./run_optimized_training.sh production
```

### Manual Training
```bash
# Test mode with all optimizations
python train.py --mode test --profile --max-steps 100

# Production mode
python train.py --mode production
```

## 📊 Monitoring

### Real-time Monitoring
```bash
# Start live monitor (refreshes every 10 seconds)
python scripts/monitor_training.py

# Custom refresh rate
python scripts/monitor_training.py --refresh 5

# One-time status check
python scripts/monitor_training.py --once
```

### Monitoring Features
- **Preprocessing Status**: Cache status and file counts
- **Training Metrics**: Loss, learning rate, gradient norms, perplexity
- **Performance Metrics**: Timing statistics, throughput, I/O performance
- **System Status**: CPU, memory, GPU usage
- **Training Summary**: Run statistics and completion status

## ⚙️ Configuration

### Test Configuration (`get_test_config()`)
- Files: 25 training files
- Batch size: 2 per device
- Steps: Limited to 100 for quick testing
- Optimizations: All enabled
- Monitoring: Detailed monitoring and profiling enabled

### Production Configuration (`get_production_config()`)
- Files: All available training files
- Batch size: 4 per device
- Steps: Unlimited (full training)
- Optimizations: Enabled but minimal monitoring overhead

### Key Settings
```python
# Enable preprocessing optimizations
use_optimized_dataset = True
preprocess_cache_dir = "./preprocessed_cache"

# Performance monitoring
enable_profiling = True  # I/O and computation profiling
detailed_monitoring = True  # System and stability metrics

# Test run limitations
max_test_steps = 100  # Limit steps for test runs
```

## 🔧 Testing

### Test Dataloader Performance
```bash
python test_dataloader.py
```

This will:
- Test preprocessing cache performance
- Validate tensor shapes and dtypes
- Compare optimized vs legacy data loading
- Measure setup and batch processing times

### Expected Performance Improvements
- **Setup Time**: 2-10x faster after first preprocessing
- **Batch Loading**: Minimal reshaping overhead
- **Memory Usage**: Reduced due to preprocessing cache
- **I/O Efficiency**: Cached preprocessed data reduces disk reads

## 📈 Performance Metrics

### Tracked Metrics
- **Training**: Loss, learning rate, gradient norms, perplexity
- **Speed**: Samples/second, step time, throughput
- **Performance**: Data loading time, forward/backward pass timing
- **System**: CPU/memory/GPU usage, disk I/O
- **Stability**: Loss variance, gradient norm variance, training trends

### TensorBoard Integration
All metrics are automatically logged to TensorBoard:
```bash
tensorboard --logdir logs/
```

## 🐛 Troubleshooting

### Common Issues

**Preprocessing Cache Issues**
```bash
# Clear preprocessing cache if corrupted
rm -rf preprocessed_cache/
```

**Memory Issues**
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing`
- Use smaller `max_train_files` for testing

**Performance Issues**
- Check if `use_optimized_dataset=True`
- Verify preprocessing cache is being used
- Monitor I/O performance with profiling

### Debug Mode
```bash
# Run with detailed debugging
python train.py --mode test --profile --max-steps 10
```

## 🔄 Preprocessing Pipeline

### First Run (Cold Start)
1. Downloads NPZ files from GCS to `./cache/`
2. Loads and flattens each NPZ file
3. Pre-creates all sequences for each file
4. Saves preprocessed data to `./preprocessed_cache/`
5. Builds optimized dataset index

### Subsequent Runs (Warm Start)
1. Checks for existing preprocessed cache
2. Loads preprocessed data directly (much faster)
3. Builds dataset index from cached data
4. Ready for training with minimal preprocessing overhead

### Cache Management
- **Location**: `./preprocessed_cache/`
- **Format**: Pickle files with metadata
- **Invalidation**: Automatic if sequence length or stride changes
- **Cleanup**: Manual removal when needed

## 📋 Command Reference

### Training Commands
```bash
# Quick test with profiling
python train.py --mode test --profile --max-steps 50

# Resume from checkpoint
python train.py --mode production --resume checkpoint_name

# Custom configuration
python train.py --mode test --max-steps 200 --profile
```

### Monitoring Commands
```bash
# Live monitoring
python scripts/monitor_training.py

# Background monitoring
python scripts/monitor_training.py --refresh 30 &

# Check preprocessing status
python scripts/monitor_training.py --once
```

### Utility Commands
```bash
# Test dataloader performance
python test_dataloader.py

# Clear caches
rm -rf cache/ preprocessed_cache/

# Check system requirements
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## 🎯 Best Practices

1. **Always test first**: Run test mode before production training
2. **Monitor actively**: Use the monitoring script during training
3. **Check preprocessing**: Verify cache is working with test script
4. **Resource management**: Monitor GPU memory and adjust batch sizes
5. **Regular checkpoints**: Use automatic checkpointing for long runs
6. **Profile periodically**: Enable profiling to identify bottlenecks

This optimized training pipeline significantly reduces preprocessing overhead while providing comprehensive monitoring and debugging capabilities. 