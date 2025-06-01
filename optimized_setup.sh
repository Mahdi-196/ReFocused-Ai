#!/bin/bash
# ReFocused-AI One-Command Setup
# Run this script immediately after cloning the repository

set -e

echo "🚀 ReFocused-AI One-Command Setup"
echo "================================="
echo "Setting up optimized training environment..."

# Navigate to training directory
cd 05_model_training

# Run the enhanced setup
echo "🔧 Running enhanced setup script..."
bash setup.sh

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Ready to start training! Try these commands:"
echo ""
echo "  # Quick start (runs tests + 50-step training):"
echo "  cd 05_model_training && ./quick_start.sh"
echo ""
echo "  # Or manual approach:"
echo "  cd 05_model_training"
echo "  source activate_env.sh"
echo "  python test_dataloader.py"
echo "  ./run_optimized_training.sh test"
echo ""
echo "  # Monitor training:"
echo "  ./start_monitoring.sh"
echo ""
echo "�� Happy training!" 