# ReFocused-AI Training Tests

This directory contains test scripts for validating and debugging the ReFocused-AI training system.

## Test Files

### 🔐 `test_credentials.py`
**Google Cloud Storage Authentication Test**
- Tests GCS bucket access and permissions
- Validates service account credentials
- Tests upload/download functionality
- Run: `python test_credentials.py`

### ⚡ `test_cpu_optimizations.py`
**CPU-Safe Optimization Tests**
- Tests configuration optimizations without GPU requirements
- Validates batch size and gradient accumulation settings
- Tests DataLoader optimizations with mock data
- Safe to run on any system
- Run: `python test_cpu_optimizations.py`

### 🚀 `test_optimizations.py`
**GPU Performance Tests**
- Tests memory usage with different batch sizes
- Validates mixed precision training performance
- Tests DataLoader performance with real data
- Requires CUDA-enabled GPU
- Run: `python test_optimizations.py`

### 🔧 `debug_gpu.py`
**GPU Debug and Diagnostics**
- Comprehensive GPU and CUDA diagnostics
- Tests PyTorch CUDA functionality
- Checks NVIDIA drivers and hardware
- System environment validation
- Run: `python debug_gpu.py`

## Usage

### Quick Health Check
```bash
# Run all CPU-safe tests
cd 05_model_training/tests
python test_cpu_optimizations.py

# Check GPU setup
python debug_gpu.py

# Test GCS credentials
python test_credentials.py
```

### Before Training
```bash
# Recommended pre-training validation sequence:
python debug_gpu.py              # Check GPU setup
python test_credentials.py       # Verify GCS access  
python test_cpu_optimizations.py # Test configurations
python test_optimizations.py     # Test GPU performance (if available)
```

### Troubleshooting

#### GPU Issues
- Run `debug_gpu.py` for comprehensive diagnostics
- Check NVIDIA driver installation
- Verify CUDA version compatibility

#### GCS Issues
- Run `test_credentials.py` to validate authentication
- Check service account key placement in `../credentials/`
- Verify bucket permissions

#### Performance Issues
- Run `test_optimizations.py` to benchmark configurations
- Check memory usage with different batch sizes
- Validate mixed precision settings

## Test Structure

All tests are designed to be:
- **Independent**: Can run standalone without dependencies
- **Safe**: CPU-safe tests won't require GPU
- **Informative**: Provide clear pass/fail status with details
- **Comprehensive**: Cover major system components

## Integration

Tests automatically adjust import paths to work from the `tests/` subdirectory while maintaining access to parent module configurations and utilities. 