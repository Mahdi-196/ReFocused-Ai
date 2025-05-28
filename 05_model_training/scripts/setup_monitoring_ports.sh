#!/bin/bash
# Setup Monitoring Services on Hyperbolic Custom Ports
# Run this after your main setup to enable web monitoring

echo "🔧 SETTING UP MONITORING SERVICES"
echo "=================================="

# Install additional monitoring dependencies
pip install tensorboard jupyter ipywidgets

# Setup TensorBoard service (Port 6006)
echo "📊 Setting up TensorBoard on port 6006..."
cat > /scratch/start_tensorboard.sh << 'EOF'
#!/bin/bash
# TensorBoard service for training monitoring
cd /scratch/logs
tensorboard --logdir=. --host=0.0.0.0 --port=6006 --reload_interval=30
EOF
chmod +x /scratch/start_tensorboard.sh

# Setup Jupyter service (Port 8888) 
echo "📓 Setting up Jupyter on port 8888..."
cat > /scratch/start_jupyter.sh << 'EOF'
#!/bin/bash
# Jupyter service for interactive debugging
cd /root/ReFocused-Ai/05_model_training
jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser --NotebookApp.token='' --NotebookApp.password=''
EOF
chmod +x /scratch/start_jupyter.sh

# Create monitoring launcher
cat > /scratch/start_monitoring.sh << 'EOF'
#!/bin/bash
# Start all monitoring services
echo "🚀 Starting monitoring services..."

# Start TensorBoard in background
nohup /scratch/start_tensorboard.sh > /scratch/logs/tensorboard.log 2>&1 &
echo "📊 TensorBoard started on port 6006"

# Start Jupyter in background  
nohup /scratch/start_jupyter.sh > /scratch/logs/jupyter.log 2>&1 &
echo "📓 Jupyter started on port 8888"

echo ""
echo "✅ Monitoring services running!"
echo "Access via your Hyperbolic dashboard URLs:"
echo "  TensorBoard: http://your-instance-6006.1.cricket.hyperbolic.xyz:30000"
echo "  Jupyter:     http://your-instance-8888.1.cricket.hyperbolic.xyz:30000"
EOF
chmod +x /scratch/start_monitoring.sh

echo ""
echo "✅ MONITORING SETUP COMPLETE!"
echo "============================="
echo "To start monitoring services:"
echo "  /scratch/start_monitoring.sh"
echo ""
echo "Recommended ports for Hyperbolic:"
echo "  Port 1: 6006 (TensorBoard)" 
echo "  Port 2: 8888 (Jupyter)" 