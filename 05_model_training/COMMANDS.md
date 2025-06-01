# ReFocused-AI Training Commands

## Commands to Run on Hyperbolic H100 Instance

### Step 1: Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/ReFocused-Ai.git
cd ReFocused-Ai/05_model_training
```

### Step 2: Run Setup (First Time Only)
```bash
bash setup.sh
```

### Step 3: Run Test Training (Verify Everything Works)
```bash
bash run_test_training.sh
```

### Step 4: Run Production Training (Full Dataset)
```bash
bash run_production_training.sh
```

## Alternative Commands

### If you need to activate environment manually:
```bash
source activate_env.sh
```

### To run training directly with Python:
```bash
# Test mode
python train.py --mode test

# Production mode
python train.py --mode production
```

### To run with specific number of GPUs:
```bash
# For 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --num_processes 4 train.py --mode production
```

### To resume from checkpoint:
```bash
python train.py --mode production --resume checkpoint-epoch0-step1000-files5
```

### To monitor training:
```bash
# In a separate terminal
tensorboard --logdir=./logs --host 0.0.0.0
```

## GCS Authentication (If Needed for Uploads)

If checkpoint uploads fail, you may need to authenticate:

```bash
# Option 1: Use service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"

# Option 2: Use gcloud auth
gcloud auth application-default login
```

## Quick Debug Commands

### Check GPU availability:
```bash
nvidia-smi
```

### Test data loading:
```bash
python -c "from utils.data_utils import GCSDataLoader; loader = GCSDataLoader('refocused-ai'); print(loader.list_data_files(max_files=5))"
```

### Check model size:
```bash
python -c "from configs import get_model_config, calculate_params; c = get_model_config(); print(f'Parameters: {calculate_params(c)/1e9:.2f}B')"
``` 