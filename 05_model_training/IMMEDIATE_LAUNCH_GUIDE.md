# 🚀 IMMEDIATE LAUNCH GUIDE
**Start your $275-320 training run in the next 5 minutes!**

## ⏰ RIGHT NOW - Launch Hyperbolic Instance

### 1. Configure Instance
```
GPU: 8x H100 SXM
Docker: pytorch/pytorch:latest  
HTTP Port 1: 6006 (TensorBoard)
HTTP Port 2: 8888 (Jupyter)
Storage: 2.3TB ✅
```

### 2. Get SSH Command
After launch, copy the SSH command from Hyperbolic dashboard:
```bash
# Example: 
ssh user@gpu-instance-xxx.hyperbolic.xyz -p 12345
```

## 🛠️ SETUP SEQUENCE (5 minutes)

### 3. SSH In & Quick Setup
```bash
# SSH into your instance
ssh user@YOUR_INSTANCE -p YOUR_PORT

# Clone your repo (replace with your actual GitHub repo)
git clone https://github.com/yourusername/ReFocused-Ai.git
cd ReFocused-Ai/05_model_training

# Run automated setup
bash scripts/quick_setup.sh

# Setup monitoring ports
bash scripts/setup_monitoring_ports.sh
```

### 4. Upload Your GCP Key
```bash
# From your local machine (new terminal)
scp path/to/your-gcp-key.json user@YOUR_INSTANCE:/scratch/gcp-key.json

# Back on the instance
export GOOGLE_APPLICATION_CREDENTIALS=/scratch/gcp-key.json
```

### 5. Start Monitoring Services
```bash
# Start TensorBoard and Jupyter
/scratch/start_monitoring.sh

# Your URLs will be shown - save them!
# TensorBoard: http://your-instance-6006.1.cricket.hyperbolic.xyz:30000
# Jupyter: http://your-instance-8888.1.cricket.hyperbolic.xyz:30000
```

## 🔥 LAUNCH TRAINING (The moment of truth!)

### 6. Final Launch
```bash
# Start your $275-320 training run!
/scratch/launch_training.sh
```

## 📊 MONITOR YOUR PROGRESS

### Real-time Monitoring:
- **TensorBoard**: Training loss, GPU utilization, cost tracking
- **Jupyter**: Interactive debugging and data inspection  
- **Terminal**: Live training logs and DeepSpeed metrics

### Expected Timeline:
- **Steps per hour**: 2,500 (target)
- **Total steps**: 100,000
- **Estimated duration**: 40 hours
- **Final cost**: ~$317

## 🚨 CRITICAL CHECKPOINTS

### First 30 Minutes:
- ✅ GPUs recognized (8x H100)
- ✅ Data loading from GCS
- ✅ DeepSpeed initialization
- ✅ First training step completes

### First Hour:  
- ✅ Stable loss decrease
- ✅ GPU utilization >85%
- ✅ Cost tracking active
- ✅ Checkpoints saving to GCS

### If Issues Arise:
1. Check TensorBoard for metrics
2. Monitor GPU usage: `nvidia-smi`
3. Check logs: `tail -f /scratch/logs/training.log`
4. Debug in Jupyter notebook

## 💰 COST CONTROL

**Current burn rate**: $7.92/hour
**Budget alerts**: Set at $100, $200, $300
**Auto-stop**: Implement if over budget

---

**🎯 Ready? GO LAUNCH YOUR INSTANCE NOW!** 