# 🚀 TRAINING LAUNCH CHECKLIST

## ✅ Pre-Launch Status
- [x] Training infrastructure built
- [x] 21.7GB tokenized dataset ready
- [x] 8x H100 SXM setup reserved ($7.92/hour)
- [x] DeepSpeed ZeRO Stage 3 configured
- [x] Cost monitoring ready

## 📋 5-Step Launch Process

### STEP 1: Upload Data to Google Cloud Storage
```bash
# On your local machine (where data_tokenized_production is)
# Install gsutil if not already installed
pip install google-cloud-storage

# Upload your tokenized data to GCS
gsutil -m cp -r C:/Users/mahdi/Downloads/Documents/Desktop/data_tokenized_production/* gs://refocused-ai/tokenized_data/

# Verify upload
gsutil ls -l gs://refocused-ai/tokenized_data/ | head -10
```

### STEP 2: Launch Hyperbolic Instance
1. Go to Hyperbolic Labs dashboard
2. Launch your 8x H100 SXM instance
3. Use Docker image: `pytorch/pytorch:latest`
4. Set up port forwarding if needed
5. Note the instance IP address

### STEP 3: Setup Training Environment
```bash
# SSH into your Hyperbolic instance
ssh user@your-instance-ip

# Clone your repository
git clone https://github.com/yourusername/ReFocused-Ai.git
cd ReFocused-Ai/05_model_training

# Upload your GCP service account key
# (Download from GCP Console -> IAM & Admin -> Service Accounts)
scp your-gcp-key.json user@instance:/scratch/gcp-key.json

# Run environment setup
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### STEP 4: Pre-Flight Check
```bash
# Source environment variables
source /scratch/training_env.sh

# Verify GPU setup
nvidia-smi

# Test data access
python3 -c "
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('refocused-ai')
print(f'✅ GCS bucket accessible: {bucket.exists()}')
print(f'✅ Files in bucket: {len(list(bucket.list_blobs(prefix=\"tokenized_data\")))}')
"

# Check cost estimates
python3 scripts/check_costs.py
```

### STEP 5: Launch Training! 🎯
```bash
# Start training with monitoring
/scratch/launch_training.sh

# OR start without monitoring dashboard
/scratch/launch_training.sh --no-monitor

# To monitor progress in separate terminal
screen -r monitor
# OR
tail -f /scratch/logs/training.log
```

## 🔍 **Monitoring Your Training**

### Real-Time Monitoring
```bash
# GPU utilization
watch nvidia-smi

# Training progress
tail -f /scratch/logs/training.log

# Cost monitoring  
python3 scripts/check_costs.py --monitor

# System monitoring
htop
```

### Expected Behavior
- **First 10 minutes**: Data sync from GCS to local storage
- **Training start**: Loss should start around ~10.0
- **Speed target**: 2,500-2,875 steps/hour
- **Checkpoints**: Saved every 1,000 steps
- **Total time**: ~35-40 hours
- **Total cost**: ~$275-320

## ⚠️ **Troubleshooting Quick Fixes**

### If Training Doesn't Start
```bash
# Check logs
tail -100 /scratch/logs/training.log

# Verify data sync
ls -la /scratch/shards/

# Test GPU communication
python3 -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### If Cost Exceeds Budget
```bash
# Check current spending
python3 scripts/check_costs.py

# Reduce batch size if needed (edit config/training_config.yaml)
per_device_train_batch_size: 2  # Reduce from 4
```

### If Speed is Too Slow
```bash
# Check GPU utilization
nvidia-smi

# Increase data workers if CPU/RAM allows
dataloader_num_workers: 8  # Increase from 4
```

## 📞 **Emergency Stop**
```bash
# Graceful stop (saves checkpoint)
pkill -SIGTERM python3

# Force stop (if needed)
pkill -9 python3

# Check remaining costs
python3 scripts/check_costs.py
```

## 🎉 **When Training Completes**

Your final model will be saved to:
- **Local**: `/scratch/checkpoints/hf_model/`
- **GCS**: `gs://refocused-ai/final_model/`

Expected training results:
- **Final loss**: ~3.5-4.0
- **Model size**: ~3GB
- **Ready for**: Text generation, fine-tuning, deployment

---

## 🚀 **TL;DR: Quick Start**
1. `gsutil -m cp -r data_tokenized_production/* gs://refocused-ai/tokenized_data/`
2. Launch Hyperbolic instance
3. `./scripts/setup_environment.sh`
4. Upload GCP key to `/scratch/gcp-key.json`
5. `/scratch/launch_training.sh`

**Expected total cost: $275-320 over 35-40 hours**

Good luck! 🎯 

# H100 Training Launch Checklist

Use this checklist to ensure your H100 training run is properly set up and optimized.

## 📋 Pre-Launch Checklist

- [ ] SSH into your instance:
  ```bash
  ssh ubuntu@illegal-primrose-mastodon.1.cricket.hyperbolic.xyz -p 31870
  ```

- [ ] Clone repository:
  ```bash
  git clone https://github.com/Mahdi-196/ReFocused-Ai.git
  cd ReFocused-Ai
  ```

- [ ] Run the setup script:
  ```bash
  cd 05_model_training
  python3 setup_h100_env.py
  ```

## 🔍 Verification Steps

- [ ] Check CUDA is available:
  ```bash
  python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
  ```

- [ ] Verify GPU count:
  ```bash
  python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
  ```

- [ ] Check GPU memory:
  ```bash
  nvidia-smi
  ```

- [ ] Verify directories exist:
  ```bash
  ls -la /home/ubuntu/training_data/
  ls -la ../models/gpt_750m/
  ls -la ../models/tokenizer/tokenizer/
  ```

- [ ] Verify data downloaded:
  ```bash
  ls -la /home/ubuntu/training_data/shards
  ```

## 🚀 Launch Commands

- [ ] Run single-GPU test:
  ```bash
  bash h100_runner.sh test
  ```

- [ ] Run full 8-GPU training:
  ```bash
  bash h100_runner.sh full
  ```

## 📊 Monitoring

- [ ] TensorBoard:
  ```
  http://illegal-primrose-mastodon-6006.1.cricket.hyperbolic.xyz:30000
  ```

- [ ] Jupyter (if set up):
  ```
  http://illegal-primrose-mastodon-8888.1.cricket.hyperbolic.xyz:30000
  ```

- [ ] Check training logs:
  ```bash
  tail -f /home/ubuntu/training_data/logs/latest.log
  ```

- [ ] Monitor GPU usage:
  ```bash
  watch -n 1 nvidia-smi
  ```

## 🔄 Troubleshooting

### If CUDA out of memory:
- [ ] Reduce batch size in config/h100_multi_gpu.yaml
- [ ] Increase gradient accumulation steps
- [ ] Enable gradient checkpointing

### If download fails:
- [ ] Check bucket name and prefix
- [ ] Try listing available prefixes
- [ ] Manually check Google Cloud Storage web console

### If DeepSpeed fails:
- [ ] Check config/h100_deepspeed_multi.json
- [ ] Verify that all 8 GPUs are visible
- [ ] Set environment variables:
  ```bash
  export RDMAV_FORK_SAFE=1
  export FI_EFA_FORK_SAFE=1
  export NCCL_IB_DISABLE=1
  export NCCL_P2P_DISABLE=1
  ```

## 💾 Checkpoint Management

- [ ] Upload checkpoint:
  ```bash
  bash h100_runner.sh upload /home/ubuntu/training_data/checkpoints/checkpoint_step_1000
  ```

- [ ] Check upload status:
  ```bash
  gsutil ls gs://refocused-ai/Checkpoints/
  ```

Good luck with your training run! 🚀 