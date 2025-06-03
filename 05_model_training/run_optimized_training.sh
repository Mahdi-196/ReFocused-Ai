#!/bin/bash
# Optimized Training Launch Scripts for ReFocused-AI
# These scripts demonstrate how to use the performance optimizations

echo "🚀 ReFocused-AI Optimized Training Scripts"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "\n${GREEN}Available Training Configurations:${NC}"
echo "1. Quick Test (Low Memory)"
echo "2. Production Training (High Performance)"
echo "3. Memory-Constrained Training"
echo "4. Custom Configuration"

echo -e "\n${BLUE}1. Quick Test Configuration:${NC}"
echo "   - Batch size per device: 2"
echo "   - Gradient accumulation: 2 steps"
echo "   - Effective batch size: 4"
echo "   - Mixed precision: bf16 (if GPU supports)"
echo "   - Files: 5 (for testing)"

cat << 'EOF'
# Command:
python train.py --config test --mixed-precision bf16

# What this does:
# ✅ Uses optimized batch size (2) with gradient accumulation (2)
# ✅ Enables mixed precision for 2x speed + 50% memory savings
# ✅ Optimized DataLoader with workers and prefetching
# ✅ Reduced checkpoint frequency for better performance
EOF

echo -e "\n${BLUE}2. Production Training Configuration:${NC}"
echo "   - Batch size per device: 4"
echo "   - Gradient accumulation: 8 steps"
echo "   - Effective batch size: 32"
echo "   - Mixed precision: bf16"
echo "   - Files: All available"

cat << 'EOF'
# Command:
python train.py --config production --mixed-precision bf16

# What this does:
# 🚀 Maximum GPU utilization with large effective batch size
# 🚀 Optimal for H100/A100 GPUs
# 🚀 All performance optimizations enabled
# 🚀 Background checkpoint uploads
EOF

echo -e "\n${BLUE}3. Memory-Constrained Training:${NC}"
echo "   For GPUs with limited memory (8GB or less)"

cat << 'EOF'
# Command:
python train.py --config test --mixed-precision fp16

# Alternative with manual gradient accumulation:
# Edit configs/training_config.py:
# per_device_train_batch_size = 1
# gradient_accumulation_steps = 4
# Then run: python train.py --config test --mixed-precision fp16
EOF

echo -e "\n${BLUE}4. Custom Configuration Examples:${NC}"

cat << 'EOF'
# Override max steps:
python train.py --config test --max-steps 2000 --mixed-precision bf16

# Disable background uploads (for debugging):
python train.py --config test --no-background-upload

# Use fp16 instead of bf16 (for older GPUs):
python train.py --config production --mixed-precision fp16

# Force CPU mode (no mixed precision):
python train.py --config test --mixed-precision no
EOF

echo -e "\n${YELLOW}Performance Monitoring:${NC}"
echo "The training script now provides comprehensive performance metrics:"

cat << 'EOF'
📊 During training, you'll see:
   - Steps per second (training throughput)
   - Effective batch size (actual samples per update)
   - Memory usage tracking
   - Loss trends and best loss achieved
   - Real-time progress with optimized display

🎯 At the end, you'll get a performance summary:
   - Total training time
   - Average steps per second
   - Effective batch size achieved
   - Mixed precision mode used
   - Best loss achieved
EOF

echo -e "\n${YELLOW}Troubleshooting:${NC}"

cat << 'EOF'
🚨 If you get Out of Memory (OOM):
   1. Reduce per_device_train_batch_size
   2. Increase gradient_accumulation_steps proportionally
   3. Use bf16 or fp16 mixed precision
   4. Check GPU memory: nvidia-smi

🚨 If training is slow:
   1. Enable mixed precision (bf16/fp16)
   2. Increase batch size if memory allows
   3. Check DataLoader workers are being used
   4. Ensure SSD storage for cache

🚨 If loss diverges:
   1. Check effective batch size isn't too large
   2. Reduce learning rate
   3. Monitor gradient norms
   4. Ensure drop_last=True for consistent batches
EOF

echo -e "\n${GREEN}Testing the Optimizations:${NC}"
echo "Run performance tests to validate optimizations:"

cat << 'EOF'
# Test on CPU (basic validation):
python test_cpu_optimizations.py

# Test with GPU (full validation):
python test_optimizations.py
EOF

echo -e "\n${GREEN}Expected Performance Improvements:${NC}"
echo "Compared to the original configuration:"

cat << 'EOF'
📈 Batch Size (1→4):        2-4x GPU utilization
📈 Mixed Precision (bf16):  2x throughput, 50% memory
📈 DataLoader Workers:      20-50% faster data loading
📈 Gradient Accumulation:   Large effective batches without OOM
📈 Python Optimization:     5-15% faster training loop
📈 Checkpoint Optimization: 90% less I/O blocking

🎯 Overall Expected Improvement: 3-8x faster training
EOF

echo -e "\n${BLUE}Ready to start optimized training!${NC}"
echo "Choose a configuration above and run the corresponding command."
echo ""
echo "For monitoring performance, watch the steps/second metric and GPU utilization."
echo "The optimizations should significantly improve both speed and GPU efficiency." 