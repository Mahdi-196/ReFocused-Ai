#!/bin/bash
# Emergency fix script for NumPy and packaging issues

echo "=== ReFocused-AI Environment Fix Script ==="

# Activate virtual environment
source venv/bin/activate

# Uninstall NumPy 2.x first
echo "Removing NumPy 2.x..."
pip uninstall -y numpy

# Force installation of packaging
echo "Installing packaging..."
pip install packaging==23.2

# Force installation of NumPy 1.24.0
echo "Installing NumPy 1.24.0..."
pip install numpy==1.24.0 --force-reinstall

# Verify installations
echo "Verifying NumPy version..."
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

echo "Verifying packaging installation..."
python -c "import packaging; print(f'Packaging version: {packaging.__version__}')"

echo "Fix complete. Please run your training script again." 