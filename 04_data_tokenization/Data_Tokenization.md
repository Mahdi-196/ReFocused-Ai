# Data Tokenization 

Convert processed text data into tokenized sequences ready for AI model training.

## 🚀 Quick Start

### 1. Validate Setup
```bash
python setup_and_validate.py
```

### 2. Quick Test (Recommended First Step)
```bash
python test_cloud_tokenization_10min.py
```

### 3. Production Tokenization
```bash
python run_full_tokenization.py
```

## 📁 Available Scripts

| Script | Purpose | Best For | Speed | Memory |
|--------|---------|----------|-------|--------|
| `test_cloud_tokenization_10min.py` | Quick validation test | Testing setup | ~3,200 seq/sec | Low |
| `tokenize_data.py` | Multi-threaded processing | Development | ~2,800 seq/sec | Medium |
| `run_full_tokenization.py` | Production single-threaded | Large datasets | ~3,000 seq/sec | Low |
| `resume_tokenization.py` | Resume interrupted jobs | Recovery | ~3,000 seq/sec | Low |
| `fix_tokenization_single_threaded.py` | Debugging version | Troubleshooting | ~2,500 seq/sec | Low |

## 🎯 Choose the Right Script

### Dataset Size Recommendations
- **< 1GB**: `test_cloud_tokenization_10min.py` → Quick validation
- **1-10GB**: `tokenize_data.py` → Multi-threaded processing  
- **10GB+**: `run_full_tokenization.py` → Production single-threaded
- **Recovery**: `resume_tokenization.py` → Continue from interruption

### System Recommendations
- **< 4GB RAM**: Use `run_full_tokenization.py` (memory optimized)
- **4-8GB RAM**: Use `tokenize_data.py` with 2-4 workers
- **> 8GB RAM**: Use `tokenize_data.py` with 4-8 workers

## 📊 Expected Performance

Based on testing results:
- **Processing Rate**: 2,500-3,200 sequences/second
- **Memory Usage**: 1-4GB depending on script and configuration
- **Output Format**: Compressed NumPy arrays (.npz files)

## 🔧 Configuration Options

### Standard Configuration
```python
CONFIG = {
    "max_length": 1024,      # Maximum sequence length
    "stride": 512,           # Overlap for long sequences
    "batch_size": 500,       # Sequences per batch
    "save_frequency": 10000, # Save every N sequences
}
```

### Memory Optimized
```python
MEMORY_OPTIMIZED = {
    "max_length": 512,       # Shorter sequences
    "batch_size": 250,       # Smaller batches
    "save_frequency": 5000,  # Save more frequently
}
```

### Speed Optimized
```python
SPEED_OPTIMIZED = {
    "max_length": 1024,      # Full sequences
    "batch_size": 1000,      # Larger batches
    "save_frequency": 20000, # Save less frequently
}
```

## 📂 Input/Output Format

### Input Format (JSONL)
```json
{"text": "Your training text here", "title": "Optional title", "subreddit": "source"}
{"text": "Another training example", "metadata": "additional_info"}
```

### Output Format (NPZ)
```python
# Loading tokenized data
data = np.load("tokenized_file.npz")
input_ids = data['input_ids']        # Shape: (num_sequences, max_length)
attention_mask = data['attention_mask']  # Shape: (num_sequences, max_length)
```

## 🛠️ Common Commands

### Basic Usage
```bash
# Quick test and validation
python test_cloud_tokenization_10min.py

# Compare different approaches
python tokenization_comparison.py

# Setup validation
python setup_and_validate.py
```

### Production Usage
```bash
# Standard production tokenization
python run_full_tokenization.py

# Multi-threaded processing
python tokenize_data.py --workers 4

# Resume interrupted job
python resume_tokenization.py
```

### Debugging
```bash
# Single-threaded debugging
python fix_tokenization_single_threaded.py

# Check system resources
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent:.1f}%')"
```

## 🔍 Quality Checks

### Validate Output Data
```python
import numpy as np
from pathlib import Path

# Check tokenized files
for npz_file in Path("data_tokenized_production").glob("*.npz"):
    data = np.load(npz_file)
    print(f"{npz_file.name}: {data['input_ids'].shape[0]} sequences")
```

### Performance Monitoring
```bash
# Monitor system resources during processing
watch -n 5 'ps aux | grep tokenization'
```

## 🚨 Troubleshooting

### Common Issues

#### Out of Memory
```bash
# Use memory-optimized script
python run_full_tokenization.py

# Or reduce batch size in script configuration
```

#### Tokenizer Not Found
```bash
# Train tokenizer first
cd ../03_tokenizer_training
python train_tokenizer.py
cd ../04_data_tokenization
```

#### Slow Processing
```bash
# Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"

# Use multi-threaded script if sufficient resources
python tokenize_data.py --workers 4
```

#### Corrupted Output
```bash
# Resume from last checkpoint
python resume_tokenization.py

# Or delete corrupted files and restart
rm data_tokenized_production/corrupted_file.npz
python run_full_tokenization.py
```

## 📈 Integration with Training Pipeline

### Loading in PyTorch
```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TokenizedDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = list(Path(data_dir).glob("*.npz"))
        # Load all sequences (or implement streaming for large datasets)
    
    def __getitem__(self, idx):
        # Return tokenized sequence and attention mask
        pass

# Usage
dataset = TokenizedDataset("data_tokenized_production")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Loading in Transformers
```python
from transformers import Trainer, TrainingArguments

# Setup training with pre-tokenized data
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=4,
    dataloader_num_workers=4,  # Parallel data loading
    fp16=True,  # Memory optimization
)
```

## 📚 Documentation

- **[DATA_TOKENIZATION_GUIDE.md](DATA_TOKENIZATION_GUIDE.md)** - Comprehensive guide with examples
- **[tokenization_comparison.py](tokenization_comparison.py)** - Interactive comparison tool
- **[setup_and_validate.py](setup_and_validate.py)** - Environment validation

## 🔗 Pipeline Integration

**Previous Step:** `03_tokenizer_training` - Train custom tokenizer  
**Current Step:** `04_data_tokenization` - Convert data to tokens  
**Next Step:** `05_model_training` - Train language model
