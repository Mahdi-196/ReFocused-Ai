# 🛠️ ReFocused-AI Utilities

This directory contains utility scripts and tools for analyzing, processing, and managing the ReFocused-AI model training pipeline.

## 📁 Directory Structure

```
utilities/
├── analysis/              # Data and training analysis tools
├── data_processing/       # Data processing and validation utilities
├── training/              # Training-related utilities
├── deployment/            # Deployment and cleanup utilities
├── tests/                 # Unit tests for all utilities
└── run_analysis.sh        # Batch analysis script
```

## 🔍 Analysis Tools

### Training Parameter Analysis
```bash
python analysis/analyze_training_parameters.py
```
Analyzes your dataset and recommends optimal training parameters including:
- Optimal number of training steps
- Batch size configurations
- Multi-GPU scaling recommendations
- Time and cost estimates

### GPU Configuration Analysis
```bash
python analysis/8gpu_analysis.py
```
Provides specific recommendations for 8-GPU training setups.

### Local Data Analysis
```bash
python analysis/local_data_analysis.py --data-dir ./data_tokenized
```
Analyzes locally stored tokenized data for quality and statistics.

## 📊 Data Processing Utilities

### Tokenized Data Analysis
```bash
python data_processing/analyze_tokenized_data.py
```
Analyzes tokenized data files for:
- Sequence length distributions
- Token frequency analysis
- Data quality metrics

### Quick Dataset Size Check
```bash
python data_processing/quick_dataset_size_check.py
```
Quickly estimates total dataset size and token count.

### Bucket Operations
```bash
python data_processing/quick_bucket_check.py --bucket refocused-ai
```
Checks Google Cloud Storage bucket for:
- File integrity
- Missing files
- Corruption detection

### Sequence Counting
```bash
python data_processing/count_final_sequences.py
```
Counts total sequences across all tokenized files.

### Missing File Detection
```bash
python data_processing/check_missing_files.py --expected-count 1000
```
Identifies missing files in the processing pipeline.

## 🚂 Training Utilities

### Resume After Disk Full
```bash
python training/resume_after_disk_full.py --checkpoint-dir ./checkpoints
```
Helps resume training after disk space issues by:
- Cleaning up temporary files
- Identifying last valid checkpoint
- Preparing for resume

## 🚀 Deployment Utilities

### Cleanup Summary
```bash
python deployment/cleanup_summary.py --model-dir ./models
```
Provides a summary of model files and helps clean up:
- Temporary checkpoints
- Intermediate files
- Old model versions

## 🧪 Running Tests

Run all tests:
```bash
python -m pytest tests/
```

Run specific test suite:
```bash
python tests/test_analysis_utils.py
python tests/test_data_processing_utils.py
```

## 📝 Batch Analysis

Run comprehensive analysis across all utilities:
```bash
./run_analysis.sh
```

This script runs multiple analysis tools in sequence and generates a comprehensive report.

## 💡 Common Use Cases

### 1. Pre-Training Analysis
Before starting training, analyze your data:
```bash
# Check dataset size and quality
python data_processing/quick_dataset_size_check.py

# Get training recommendations
python analysis/analyze_training_parameters.py

# Verify tokenized data
python data_processing/analyze_tokenized_data.py
```

### 2. During Training Monitoring
Monitor training progress:
```bash
# Check for missing files
python data_processing/check_missing_files.py

# Count processed sequences
python data_processing/count_final_sequences.py
```

### 3. Post-Training Cleanup
After training completes:
```bash
# Get cleanup recommendations
python deployment/cleanup_summary.py

# Clean up temporary files
python deployment/cleanup_summary.py --execute
```

### 4. Multi-GPU Setup
For multi-GPU training setup:
```bash
# Get GPU-specific recommendations
python analysis/8gpu_analysis.py

# Analyze with specific GPU count
python analysis/analyze_training_parameters.py --gpus 4
```

## 🔧 Configuration

Most utilities support command-line arguments. Use `--help` for options:
```bash
python analysis/analyze_training_parameters.py --help
```

Common arguments:
- `--bucket`: GCS bucket name
- `--data-dir`: Local data directory
- `--output`: Output file for results
- `--verbose`: Enable detailed logging

## 📊 Output Formats

Utilities typically output:
- Console summaries for quick viewing
- JSON files for programmatic access
- CSV files for data analysis
- Markdown reports for documentation

## 🐛 Troubleshooting

### Google Cloud Authentication
Pass credentials explicitly to training and data tools (no env vars). For example:
```
./start_training.sh --config test --gcs-credentials /abs/path/key.json --gcp-project your-project
```

### Memory Issues
For large datasets, use sampling:
```bash
python analysis/analyze_training_parameters.py --sample-size 10
```

### Missing Dependencies
Install required packages:
```bash
pip install numpy pandas google-cloud-storage tqdm
```

## 🤝 Contributing

When adding new utilities:
1. Place in appropriate subdirectory
2. Add comprehensive docstrings
3. Include command-line interface
4. Write unit tests
5. Update this README

## 📚 See Also

- [Training Guide](../05_model_training/README.md)
- [Data Processing Guide](../02_data_processing/PROCESSING_GUIDE.md)
 