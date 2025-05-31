#!/usr/bin/env bash
set -e

echo "==================================="
echo "ReFocused-AI Production Training Script"
echo "Starting 8×H100 GPU Training"
echo "==================================="

# 1. Check virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
  echo "ERROR: No virtual environment activated."
  echo "Please activate your virtualenv first: source venv/bin/activate"
  exit 1
fi

# 2. Verify data directory
DATA_DIR="data_full"
if [ ! -d "$DATA_DIR" ] || [ -z "$(ls $DATA_DIR/*.npz 2>/dev/null)" ]; then
  echo "ERROR: $DATA_DIR is empty or doesn't exist."
  echo "Run setup_prod.sh first to download the training data."
  exit 1
fi

FILE_COUNT=$(ls $DATA_DIR/*.npz 2>/dev/null | wc -l)
echo "✓ Found $FILE_COUNT data files in $DATA_DIR"

# 3. Load environment variables
if [ -f ".env" ]; then
    echo "Loading .env file..."
    set -a
    source .env
    set +a
fi

# 4. Set up distributed training environment
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export WORLD_SIZE=8
export RANK=0
export LOCAL_RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

echo "Distributed training configuration:"
echo "- GPUs: $CUDA_VISIBLE_DEVICES (8×H100)"
echo "- World size: $WORLD_SIZE"
echo "- Master: $MASTER_ADDR:$MASTER_PORT"

# 5. Create directories
LOG_DIR="logs/prod_$(date +%Y%m%d_%H%M%S)"
CHECKPOINT_DIR="checkpoints/prod_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR $CHECKPOINT_DIR

echo "Directories:"
echo "- Logs: $LOG_DIR"
echo "- Checkpoints: $CHECKPOINT_DIR"

# 6. Create DeepSpeed configuration
DS_CONFIG="$LOG_DIR/deepspeed_config.json"
cat > $DS_CONFIG << 'EOF'
{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 8,
  "gradient_accumulation_steps": 2,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 12,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e7,
    "allgather_bucket_size": 5e7
  },
  "gradient_clipping": 1.0,
  "wall_clock_breakdown": false,
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": false,
    "contiguous_memory_optimization": false,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 6e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },
  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 6e-4,
      "warmup_num_steps": 2000,
      "total_num_steps": 100000
    }
  },
  "tensorboard": {
    "enabled": true,
    "output_path": "./logs",
    "job_name": "refocused_gpt_1b"
  }
}
EOF

echo "✓ Created DeepSpeed configuration"

# 7. System checks
echo ""
echo "Running system checks..."

# Check GPU availability
python -c "
import torch
if not torch.cuda.is_available():
    print('ERROR: CUDA not available')
    exit(1)
print(f'✓ CUDA available with {torch.cuda.device_count()} GPUs')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)')
"

# Check memory
FREE_MEM=$(free -g | awk '/^Mem:/{print $7}')
echo "✓ Available system memory: ${FREE_MEM}GB"

# 8. Launch training with DeepSpeed
echo ""
echo "==================================="
echo "Launching production training..."
echo "Model: GPT ~1.2B parameters"
echo "Data: ~21-22B tokens"
echo "Hardware: 8×H100 80GB"
echo "==================================="

# Use DeepSpeed launcher for distributed training
deepspeed --num_gpus=8 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --mode production \
    --data_dir $DATA_DIR \
    --log_dir $LOG_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --deepspeed_config $DS_CONFIG \
    --model_size 1.2B \
    --max_seq_len 2048 \
    --checkpoint_interval 5 \
    --log_interval 10 \
    --eval_interval 500 \
    --wandb_project "refocused-ai-1b" \
    --wandb_run_name "prod_$(date +%Y%m%d_%H%M%S)" \
    2>&1 | tee $LOG_DIR/training.log &

# Get the PID of the training process
TRAIN_PID=$!
echo "Training process started with PID: $TRAIN_PID"

# 9. Monitor training
echo ""
echo "==================================="
echo "Training is running in the background."
echo ""
echo "Monitor progress:"
echo "- Logs: tail -f $LOG_DIR/training.log"
echo "- TensorBoard: tensorboard --logdir=$LOG_DIR --bind_all"
echo "- Weights & Biases: https://wandb.ai/refocused-ai-1b"
echo ""
echo "Stop training:"
echo "- kill $TRAIN_PID"
echo ""
echo "Checkpoints will be uploaded to:"
echo "- gs://refocused-ai/Checkpoints/"
echo "==================================="

# Optional: Wait for training to complete
# wait $TRAIN_PID 