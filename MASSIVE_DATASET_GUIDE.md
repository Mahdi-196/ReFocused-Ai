# 📊 Dataset Processing Guide

**Comprehensive workflow for processing massive datasets (up to 150GB)**

## 🎯 Overview

Process multiple large datasets efficiently:
- **Reddit Data**: Compressed JSONL format (tested up to 13GB)
- **HuggingFace OpenWebText**: Text corpus (41GB, 8M+ documents)
- **Combined Processing**: Up to 150GB total capacity

## 🚀 Quick Workflow

### **1. System Check (2 minutes)**
```bash
python validate_setup.py
```

### **2. Analyze Your Data (5-10 minutes)**
```bash
# Single source
python setup_massive_dataset.py --reddit /path/to/data.gz --analyze-only
python setup_massive_dataset.py --huggingface --analyze-only

# Combined analysis (recommended)
python setup_massive_dataset.py --reddit /path/to/data.gz --huggingface --analyze-only
```

### **3. Extract & Prepare (2-8 hours)**
```bash
python setup_massive_dataset.py --reddit /path/to/data.gz --huggingface --extract
```

### **4. Process Everything (8-48 hours)**
```bash
python process_massive_dataset.py
```

### **5. Monitor Progress**
```bash
tail -f logs/massive_processing.log
```

## ⏱️ Time Estimates

| System Configuration | Processing Time | Expected Output |
|---------------------|----------------|-----------------|
| **8GB RAM, 4 cores** | 26-48 hours | ~7M clean samples, 25-35GB |
| **16GB RAM, 8 cores** | 16-28 hours | ~7M clean samples, 25-35GB |
| **32GB RAM, 12+ cores** | 8-16 hours | ~7M clean samples, 25-35GB |

## 🔧 System Configuration

### **Memory Optimization**

**Limited RAM (8GB):**
```python
# Edit process_massive_dataset.py
cleaner = RedditDataCleaner(n_workers=2)
cleaned_posts = cleaner.run_cleaning_pipeline(batch_size=2000)
```

**High-End (32GB+):**
```python
cleaner = RedditDataCleaner(n_workers=12)
cleaned_posts = cleaner.run_cleaning_pipeline(batch_size=15000)
```

### **Custom Datasets**
```bash
# Process different HuggingFace datasets
python setup_massive_dataset.py --hf-dataset "allenai/c4" --huggingface --extract

# Limit samples for testing
python setup_massive_dataset.py --hf-dataset "Skylion007/openwebtext" --hf-samples 100000 --extract
```

## 📁 Output Structure

```
data/
├── unified_raw/                    # Intermediate extracted chunks
├── cleaned/
│   ├── cleaned_reddit_data.jsonl  # 25-35GB high-quality dataset
│   └── cleaning_metadata.json     # Processing statistics
└── processed/                     # Training splits (after data_processor.py)
    ├── train.jsonl                # 70% training
    ├── val.jsonl                  # 15% validation
    └── test.jsonl                 # 15% testing
```

## 📊 Monitoring & Troubleshooting

### **Real-Time Monitoring**
```bash
# Processing progress
tail -f logs/massive_processing.log

# System resources
htop -p $(pgrep -f process_massive_dataset)

# Disk usage
watch -n 30 'du -sh data/*'
```

### **Common Issues**

**❌ Out of Memory**
```python
# Reduce workers and batch size
cleaner = RedditDataCleaner(n_workers=2)
cleaned_posts = cleaner.run_cleaning_pipeline(batch_size=1000)
```

**❌ Disk Space Full**
```bash
# Clean temporary files after each phase
rm -rf data/unified_raw/reddit_chunk_*.txt
rm -rf data/unified_raw/huggingface_chunk_*.txt
```

**❌ Slow Processing**
```bash
# Check bottlenecks
htop   # CPU usage
iotop  # Disk I/O
free   # Memory usage
```

**❌ HuggingFace Connection Issues**
```bash
# Use offline mode if dataset cached
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_TIMEOUT=300
```

## 🔍 Quality Metrics

### **Expected Results**
- **Retention Rate**: 65-80% of original data
- **Duplicates Removed**: 10-20% deduplication
- **Content Quality**: High across all sources
- **Format**: Consistent JSONL for training

### **Quality Features**
- Content-based deduplication across sources
- Multi-criteria quality scoring
- Spam and NSFW filtering
- Text normalization and cleaning

## 🎯 After Processing

### **Create Training Splits**
```bash
python data_processor.py
```

### **Setup Training Environment**
```bash
python training_prep.py
```

### **Train Your Model**
```bash
python train_model.py
```

## 💡 Pro Tips

1. **Test First**: Use `--analyze-only` before full processing
2. **Start Small**: Test with `--hf-samples 10000`
3. **Monitor Resources**: Watch RAM, CPU, and disk usage
4. **Use SSD**: Significantly faster for large datasets
5. **Process Overnight**: Perfect for long-running jobs
6. **Clean Progressively**: Remove temp files after each stage

## 🛠️ Technical Implementation

- **Streaming Processing**: Handles datasets larger than memory
- **Parallel Architecture**: Multi-worker processing
- **Progress Tracking**: Real-time stats and logging
- **Error Recovery**: Robust handling and resumption
- **Memory Management**: Optimized batch sizing

---

**Need help?** Run `python validate_setup.py` for system-specific guidance. 