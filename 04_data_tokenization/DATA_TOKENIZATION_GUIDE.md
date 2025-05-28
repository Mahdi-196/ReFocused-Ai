# Comprehensive Data Tokenization Guide

A complete guide to converting processed text data into tokenized sequences ready for AI model training.

## 📋 Table of Contents

1. [Understanding Data Tokenization](#understanding-data-tokenization)
2. [Available Tokenization Scripts](#available-tokenization-scripts)
3. [When to Use Each Script](#when-to-use-each-script)
4. [Tokenization Process Details](#tokenization-process-details)
5. [Performance Optimization](#performance-optimization)
6. [Output Formats and Storage](#output-formats-and-storage)
7. [Integration with Training Pipeline](#integration-with-training-pipeline)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## 🔍 Understanding Data Tokenization

### What is Data Tokenization?

Data tokenization is the process of converting cleaned text data into numerical sequences (token IDs) that language models can process during training. It's the crucial step that transforms human-readable text into the input format required by neural networks.

**Process Flow:**
```
Cleaned Text → Tokenizer → Token IDs → Training Data
"Hello world" → [15496, 995] → NumPy Arrays → Model Training
```

### Why Data Tokenization Matters

1. **Model Input Format** - Neural networks require numerical input, not text
2. **Memory Efficiency** - Pre-tokenized data loads faster during training
3. **Consistent Processing** - Ensures identical tokenization across training runs
4. **Batch Processing** - Enables efficient parallel processing of large datasets
5. **Storage Optimization** - Compressed token storage vs. raw text

### Pipeline Position

Data tokenization sits between data processing and model training:

```
01_data_collection → 02_data_processing → 03_tokenizer_training
                                           ↓
04_data_tokenization → 05_model_training → 06_monitoring_validation
```

## 🔧 Available Tokenization Scripts

### 1. **[tokenize_data.py](tokenize_data.py)** - Basic Multi-threaded Tokenizer
**Purpose:** Standard tokenization with parallel processing
**Best for:** Medium datasets (1-10GB), development environments

**Features:**
- Multi-threaded processing for speed
- Configurable batch sizes and sequence lengths
- Progress tracking and logging
- Memory-efficient processing

**Usage:**
```bash
python tokenize_data.py --input data/cleaned --output data_tokenized --workers 4
```

### 2. **[run_full_tokenization.py](run_full_tokenization.py)** - Production Single-threaded
**Purpose:** Reliable production tokenization based on successful testing
**Best for:** Large datasets (10GB+), production environments

**Features:**
- Single-threaded for maximum stability
- Proven approach from successful tests
- Robust error handling and recovery
- Memory-optimized for large datasets
- ~3,200 sequences/second processing rate

**Usage:**
```bash
python run_full_tokenization.py
```

### 3. **[test_cloud_tokenization_10min.py](test_cloud_tokenization_10min.py)** - Quick Test
**Purpose:** 10-minute validation test for cloud environments
**Best for:** Testing setup and performance validation

**Features:**
- Time-limited testing (10 minutes default)
- Performance benchmarking
- Error detection and reporting
- Cloud environment validation

**Usage:**
```bash
python test_cloud_tokenization_10min.py --max-minutes 10
```

### 4. **[resume_tokenization.py](resume_tokenization.py)** - Resume Interrupted Jobs
**Purpose:** Continue tokenization from where it was interrupted
**Best for:** Recovery from interruptions, large datasets

**Features:**
- Automatic progress detection
- Skip already completed files
- Seamless continuation
- Progress preservation

**Usage:**
```bash
python resume_tokenization.py
```

### 5. **[fix_tokenization_single_threaded.py](fix_tokenization_single_threaded.py)** - Debugging Version
**Purpose:** Single-threaded version for debugging issues
**Best for:** Troubleshooting, error diagnosis

**Features:**
- Single-threaded for easier debugging
- Detailed error reporting
- Step-by-step processing
- Issue identification

## 🎯 When to Use Each Script

### Quick Reference Decision Tree

```
📊 Dataset Size:
├── Small (< 1GB)     → test_cloud_tokenization_10min.py
├── Medium (1-10GB)   → tokenize_data.py  
└── Large (10GB+)     → run_full_tokenization.py

🔄 Situation:
├── First time        → test_cloud_tokenization_10min.py (validate)
├── Production run    → run_full_tokenization.py
├── Interrupted job   → resume_tokenization.py
└── Debug issues      → fix_tokenization_single_threaded.py
```

### Detailed Recommendations

#### **For Testing and Validation**
```bash
# Quick 10-minute test
python test_cloud_tokenization_10min.py

# Expected output:
# 🧪 10-MINUTE CLOUD TEST TOKENIZER
# Processing rate: ~3,200 sequences/second
# ✅ Test completed successfully
```

#### **For Development (Medium Datasets)**
```bash
# Multi-threaded processing
python tokenize_data.py \
    --input data/cleaned \
    --output data_tokenized \
    --max-length 1024 \
    --stride 512 \
    --workers 4
```

#### **For Production (Large Datasets)**
```bash
# Single-threaded reliable processing
python run_full_tokenization.py
```

#### **For Resuming Interrupted Jobs**
```bash
# Automatically detects and continues from last checkpoint
python resume_tokenization.py
```

## ⚙️ Tokenization Process Details

### Input Format

**Expected Input:** JSONL files with text content
```json
{"text": "Your training text here", "title": "Optional title", "subreddit": "source"}
{"text": "Another training example", "metadata": "additional_info"}
```

### Text Preparation Pipeline

#### 1. **Text Extraction**
```python
def prepare_text(item: dict) -> str:
    """Extract and prepare text from JSON item"""
    text_parts = []
    
    # Add title if available
    title = item.get('title', '').strip()
    if title and title not in ['', '[deleted]', '[removed]']:
        text_parts.append(title)
    
    # Add main text content
    text = item.get('text', '').strip()
    if text and text not in ['', '[deleted]', '[removed]']:
        text_parts.append(text)
    
    # Combine with newlines
    return "\n".join(text_parts)
```

#### 2. **Tokenization with Sliding Windows**
```python
def tokenize_text(text: str, max_length: int = 1024, stride: int = 512) -> List[List[int]]:
    """Tokenize with overlapping windows for long sequences"""
    
    # Tokenize full text
    encoding = tokenizer.encode(text)
    token_ids = encoding.ids
    
    # Handle short sequences
    if len(token_ids) <= max_length:
        return [token_ids]
    
    # Split into overlapping windows
    sequences = []
    start = 0
    while start < len(token_ids):
        end = min(start + max_length, len(token_ids))
        sequence = token_ids[start:end]
        sequences.append(sequence)
        
        if end == len(token_ids):
            break
        start += stride
    
    return sequences
```

#### 3. **Sequence Padding and Storage**
```python
def save_sequences(sequences: List[List[int]], output_path: Path):
    """Pad and save sequences to NumPy format"""
    
    # Pad to consistent length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    attention_masks = []
    
    for seq in sequences:
        # Pad sequence
        padded = seq + [pad_token_id] * (max_len - len(seq))
        padded_sequences.append(padded)
        
        # Create attention mask
        mask = [1] * len(seq) + [0] * (max_len - len(seq))
        attention_masks.append(mask)
    
    # Save as compressed NumPy arrays
    np.savez_compressed(
        output_path,
        input_ids=np.array(padded_sequences, dtype=np.int32),
        attention_mask=np.array(attention_masks, dtype=np.int32)
    )
```

### Special Token Handling

The tokenization process handles missing special tokens gracefully:

```python
# Token ID assignment with fallbacks
special_tokens = {
    "<|startoftext|>": tokenizer.token_to_id("<|startoftext|>") or 1,
    "<|endoftext|>": tokenizer.token_to_id("<|endoftext|>") or 2,
    "<|pad|>": tokenizer.token_to_id("<|pad|>") or 0
}
```

## 🚀 Performance Optimization

### Processing Speed Benchmarks

Based on testing results:

| Script | Processing Rate | Best For | Memory Usage |
|--------|----------------|----------|--------------|
| test_cloud_* | ~3,200 seq/sec | Testing | Low |
| tokenize_data | ~2,800 seq/sec | Development | Medium |
| run_full_* | ~3,000 seq/sec | Production | Low |
| resume_* | ~3,000 seq/sec | Recovery | Low |

### Configuration Optimization

#### **Memory Optimization**
```python
# For limited memory environments
MEMORY_OPTIMIZED = {
    "batch_size": 250,           # Smaller batches
    "max_length": 512,           # Shorter sequences
    "save_frequency": 5000,      # Save more frequently
    "num_workers": 2             # Fewer workers
}
```

#### **Speed Optimization**
```python
# For maximum processing speed
SPEED_OPTIMIZED = {
    "batch_size": 1000,          # Larger batches
    "max_length": 1024,          # Full sequences
    "save_frequency": 20000,     # Save less frequently
    "num_workers": 8             # More workers
}
```

#### **Balanced Configuration**
```python
# Recommended balanced settings
BALANCED = {
    "batch_size": 500,           # Moderate batches
    "max_length": 1024,          # Standard length
    "stride": 512,               # 50% overlap
    "save_frequency": 10000,     # Regular saves
    "num_workers": 4             # Reasonable parallelism
}
```

### Resource Monitoring

```python
def monitor_resources():
    """Monitor system resources during tokenization"""
    import psutil
    
    memory_usage = psutil.virtual_memory().percent
    cpu_usage = psutil.cpu_percent()
    disk_usage = psutil.disk_usage('.').percent
    
    logger.info(f"Resources: CPU {cpu_usage}%, Memory {memory_usage}%, Disk {disk_usage}%")
```

## 💾 Output Formats and Storage

### Output Directory Structure

```
data_tokenized_production/
├── tokenized_cleaned_data_part001.npz    # First data file
├── tokenized_cleaned_data_part002.npz    # Second data file
├── tokenized_cleaned_data_part003.npz    # Third data file
├── ...
└── tokenization_stats.json               # Processing statistics
```

### NPZ File Format

Each `.npz` file contains:

```python
# Loading tokenized data
data = np.load("tokenized_file.npz")

# Available arrays:
input_ids = data['input_ids']        # Shape: (num_sequences, max_length)
attention_mask = data['attention_mask']  # Shape: (num_sequences, max_length)

# Example usage:
print(f"Sequences: {input_ids.shape[0]}")
print(f"Max length: {input_ids.shape[1]}")
print(f"Total tokens: {np.sum(attention_mask)}")
```

### Statistics File Format

```json
{
    "total_sequences": 145623,
    "total_tokens": 156789012,
    "avg_sequence_length": 1076.3,
    "processing_time_seconds": 3456.7,
    "sequences_per_second": 3201.2,
    "files_processed": 15,
    "compression_ratio": 0.85,
    "memory_usage_mb": 2048.5
}
```

## 🔗 Integration with Training Pipeline

### Loading Tokenized Data for Training

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TokenizedDataset(Dataset):
    """Dataset for loading pre-tokenized data"""
    
    def __init__(self, data_dir: str):
        self.data_files = list(Path(data_dir).glob("*.npz"))
        self.sequences = []
        self.attention_masks = []
        
        # Load all tokenized files
        for file_path in self.data_files:
            data = np.load(file_path)
            self.sequences.extend(data['input_ids'])
            self.attention_masks.extend(data['attention_mask'])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.sequences[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long)
        }

# Usage in training
dataset = TokenizedDataset("data_tokenized_production")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for batch in dataloader:
    input_ids = batch['input_ids']      # Shape: (batch_size, seq_len)
    attention_mask = batch['attention_mask']  # Shape: (batch_size, seq_len)
    # Feed to model...
```

### Memory-Efficient Loading

```python
class StreamingTokenizedDataset(Dataset):
    """Memory-efficient streaming dataset"""
    
    def __init__(self, data_dir: str, cache_size: int = 10):
        self.data_files = list(Path(data_dir).glob("*.npz"))
        self.cache_size = cache_size
        self.file_cache = {}
        self.file_sizes = {}
        
        # Get sizes without loading
        for file_path in self.data_files:
            data = np.load(file_path)
            self.file_sizes[str(file_path)] = data['input_ids'].shape[0]
    
    def __getitem__(self, idx):
        # Efficiently load only when needed
        file_idx, sequence_idx = self._get_file_and_sequence_idx(idx)
        
        if file_idx not in self.file_cache:
            self._load_file_to_cache(file_idx)
        
        data = self.file_cache[file_idx]
        return {
            'input_ids': torch.tensor(data['input_ids'][sequence_idx], dtype=torch.long),
            'attention_mask': torch.tensor(data['attention_mask'][sequence_idx], dtype=torch.long)
        }
```

### Integration with Transformers

```python
from transformers import GPT2LMHeadModel, TrainingArguments, Trainer

# Load model with custom tokenizer vocab size
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(tokenizer.vocab_size)

# Setup training with tokenized data
training_args = TrainingArguments(
    output_dir="./model_output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=100,
    dataloader_num_workers=4,  # Parallel data loading
    fp16=True,  # Memory optimization
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()
```

## 📋 Best Practices

### 1. **Data Validation**

```python
def validate_tokenized_data(data_dir: Path):
    """Validate tokenized data integrity"""
    
    issues = []
    total_sequences = 0
    
    for npz_file in data_dir.glob("*.npz"):
        try:
            data = np.load(npz_file)
            
            # Check required arrays
            if 'input_ids' not in data:
                issues.append(f"Missing input_ids in {npz_file}")
            if 'attention_mask' not in data:
                issues.append(f"Missing attention_mask in {npz_file}")
            
            # Check shapes match
            if data['input_ids'].shape != data['attention_mask'].shape:
                issues.append(f"Shape mismatch in {npz_file}")
            
            # Check for valid token IDs
            max_token_id = np.max(data['input_ids'])
            if max_token_id >= tokenizer.vocab_size:
                issues.append(f"Invalid token ID {max_token_id} in {npz_file}")
            
            total_sequences += data['input_ids'].shape[0]
            
        except Exception as e:
            issues.append(f"Error loading {npz_file}: {e}")
    
    return issues, total_sequences
```

### 2. **Progress Tracking**

```python
def track_tokenization_progress(input_dir: Path, output_dir: Path):
    """Track and report tokenization progress"""
    
    input_files = list(input_dir.glob("*.jsonl"))
    output_files = list(output_dir.glob("*.npz"))
    
    # Calculate progress
    total_files = len(input_files)
    completed_files = len(output_files)
    progress_percent = (completed_files / total_files) * 100
    
    # Estimate completion time
    if completed_files > 0:
        # Get processing stats from logs or files
        avg_processing_time = estimate_avg_processing_time()
        remaining_files = total_files - completed_files
        estimated_time_remaining = remaining_files * avg_processing_time
        
        logger.info(f"Progress: {completed_files}/{total_files} ({progress_percent:.1f}%)")
        logger.info(f"Estimated time remaining: {estimated_time_remaining:.1f} minutes")
```

### 3. **Error Recovery**

```python
def implement_checkpointing(output_dir: Path, checkpoint_frequency: int = 1000):
    """Implement checkpointing for error recovery"""
    
    checkpoint_file = output_dir / "tokenization_checkpoint.json"
    
    def save_checkpoint(current_file: str, current_batch: int, stats: dict):
        checkpoint = {
            "current_file": current_file,
            "current_batch": current_batch,
            "timestamp": datetime.now().isoformat(),
            "stats": stats
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint():
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        return None
```

### 4. **Quality Assurance**

```python
def quality_check_tokenized_data(data_dir: Path, sample_size: int = 1000):
    """Perform quality checks on tokenized data"""
    
    npz_files = list(data_dir.glob("*.npz"))
    if not npz_files:
        return {"error": "No tokenized files found"}
    
    # Sample random sequences
    random_sequences = []
    for _ in range(sample_size):
        file_path = random.choice(npz_files)
        data = np.load(file_path)
        seq_idx = random.randint(0, data['input_ids'].shape[0] - 1)
        random_sequences.append(data['input_ids'][seq_idx])
    
    # Analyze quality metrics
    quality_metrics = {
        "avg_sequence_length": np.mean([np.sum(seq != pad_token_id) for seq in random_sequences]),
        "unique_tokens": len(set(np.concatenate(random_sequences))),
        "padding_ratio": np.mean([np.sum(seq == pad_token_id) / len(seq) for seq in random_sequences]),
        "empty_sequences": sum(1 for seq in random_sequences if np.all(seq == pad_token_id))
    }
    
    return quality_metrics
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. **Out of Memory Errors**

**Problem:** Tokenization runs out of memory
```
MemoryError: Unable to allocate array
```

**Solutions:**
```python
# Reduce batch size
MEMORY_EFFICIENT_CONFIG = {
    "batch_size": 100,           # Much smaller batches
    "save_frequency": 1000,      # Save more frequently
    "max_length": 512,           # Shorter sequences
}

# Use single-threaded processing
python fix_tokenization_single_threaded.py

# Process files one by one
python run_full_tokenization.py  # Already optimized for memory
```

#### 2. **Tokenizer Not Found**

**Problem:** Missing tokenizer files
```
FileNotFoundError: Tokenizer not found at models/tokenizer/tokenizer.json
```

**Solutions:**
```bash
# Check available tokenizer paths
find . -name "tokenizer.json" -type f

# Train tokenizer first
cd ../03_tokenizer_training
python train_tokenizer.py

# Use specific tokenizer path
python tokenize_data.py --tokenizer-path "path/to/your/tokenizer"
```

#### 3. **Slow Processing**

**Problem:** Tokenization is slower than expected

**Diagnosis:**
```python
def diagnose_performance():
    """Diagnose performance issues"""
    
    # Check system resources
    import psutil
    
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    disk_io = psutil.disk_io_counters()
    
    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory_percent}%")
    print(f"Disk Read: {disk_io.read_bytes / 1024**3:.2f} GB")
    print(f"Disk Write: {disk_io.write_bytes / 1024**3:.2f} GB")
```

**Solutions:**
```python
# Optimize for speed
SPEED_CONFIG = {
    "batch_size": 1000,          # Larger batches
    "num_workers": 8,            # More parallel workers
    "save_frequency": 50000,     # Save less frequently
}

# Use RAM disk for temporary files (Linux/Mac)
# mkdir /tmp/ramdisk
# mount -t tmpfs -o size=4G tmpfs /tmp/ramdisk
```

#### 4. **Invalid Token IDs**

**Problem:** Token IDs exceed vocabulary size
```
IndexError: Token ID 50300 is out of range for vocab size 50257
```

**Solutions:**
```python
def fix_token_ids(data_dir: Path, vocab_size: int):
    """Fix invalid token IDs in tokenized data"""
    
    for npz_file in data_dir.glob("*.npz"):
        data = np.load(npz_file)
        input_ids = data['input_ids']
        
        # Check for invalid tokens
        invalid_mask = input_ids >= vocab_size
        if np.any(invalid_mask):
            print(f"Found invalid tokens in {npz_file}")
            
            # Replace with UNK token
            input_ids[invalid_mask] = unk_token_id
            
            # Save corrected data
            np.savez_compressed(
                npz_file,
                input_ids=input_ids,
                attention_mask=data['attention_mask']
            )
```

#### 5. **Corrupted Output Files**

**Problem:** NPZ files are corrupted or incomplete

**Detection:**
```python
def check_file_integrity(data_dir: Path):
    """Check integrity of tokenized files"""
    
    corrupted_files = []
    
    for npz_file in data_dir.glob("*.npz"):
        try:
            data = np.load(npz_file)
            
            # Verify required arrays exist
            assert 'input_ids' in data
            assert 'attention_mask' in data
            
            # Verify shapes
            assert data['input_ids'].shape == data['attention_mask'].shape
            
            # Verify data types
            assert data['input_ids'].dtype in [np.int32, np.int64]
            assert data['attention_mask'].dtype in [np.int32, np.int64]
            
        except Exception as e:
            corrupted_files.append((npz_file, str(e)))
    
    return corrupted_files
```

**Recovery:**
```bash
# Re-process corrupted files
python resume_tokenization.py  # Will detect and reprocess missing/corrupted files

# Or process specific files
python tokenize_data.py --input data/cleaned/specific_file.jsonl
```

### Performance Monitoring

```python
def monitor_tokenization_performance():
    """Real-time performance monitoring"""
    
    import time
    import psutil
    from collections import deque
    
    processing_times = deque(maxlen=100)  # Last 100 batch times
    
    def log_batch_performance(batch_size: int, processing_time: float):
        processing_times.append(processing_time)
        
        if len(processing_times) >= 10:
            avg_time = sum(processing_times) / len(processing_times)
            sequences_per_second = batch_size / avg_time
            
            logger.info(f"Performance: {sequences_per_second:.0f} sequences/sec")
            logger.info(f"Memory: {psutil.virtual_memory().percent:.1f}%")
            logger.info(f"CPU: {psutil.cpu_percent():.1f}%")
```

## 🎯 Next Steps

1. **Test Setup**: Run `python test_cloud_tokenization_10min.py` to validate environment
2. **Process Data**: Use `python run_full_tokenization.py` for production tokenization
3. **Validate Output**: Check tokenized data integrity and quality
4. **Integrate Training**: Load tokenized data in your training pipeline
5. **Monitor Performance**: Track processing speed and resource usage

This comprehensive tokenization system ensures efficient conversion of your processed text data into training-ready token sequences with robust error handling and performance optimization!

---

**Pipeline Integration:**
- **Previous Step:** `03_tokenizer_training` - Train custom tokenizer
- **Current Step:** `04_data_tokenization` - Convert data to tokens
- **Next Step:** `05_model_training` - Train language model 