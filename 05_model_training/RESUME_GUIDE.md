# Checkpoint Resume Guide

## Overview

The training script now supports resuming from checkpoints, allowing you to continue training from where you left off instead of starting from scratch.

## How to Resume Training

### Basic Usage

```bash
# Resume from a specific checkpoint
./start_training.sh --config test --resume checkpoint-epoch0-step800-files0

# Or using the Python script directly
python train.py --config test --resume checkpoint-epoch0-step800-files0
```

### Finding Available Checkpoints

Checkpoints are saved with names like: `checkpoint-epoch{epoch}-step{step}-files{files_processed}`

Examples:
- `checkpoint-epoch0-step500-files0`
- `checkpoint-epoch0-step1000-files2`
- `checkpoint-epoch1-step1500-files5`

## What Gets Restored

When resuming from a checkpoint, the following state is restored:

### Model & Training State
- ✅ Model weights and biases
- ✅ Optimizer state (Adam momentum, etc.)
- ✅ Learning rate scheduler state
- ✅ Current step count
- ✅ Current epoch
- ✅ Best loss achieved

### Training Metrics
- ✅ Loss history
- ✅ Learning rate history
- ✅ Validation metrics
- ✅ Training progress tracking

### Automatic Features
- 🔄 Progress bar starts from correct step
- 🔄 Training loop skips already processed batches
- 🔄 Step counter continues from checkpoint
- 🔄 Background uploads continue as configured

## Expected Behavior

When you run:
```bash
./start_training.sh --config test --resume checkpoint-epoch0-step800-files0
```

You should see:
```
🔄 Attempting to resume training from checkpoint: checkpoint-epoch0-step800-files0
📥 Downloading checkpoint from GCS... (if not local)
✅ Resumed from checkpoint: checkpoint-epoch0-step800-files0
   Starting from Step: 801, Epoch: 0
   Best loss restored: 0.8542
   Loss history entries: 6
   Training will continue from global step 801

📈 Starting optimized training loop...
🔄 Resuming training from step 800, target: 11450
🔄 Resuming epoch 0: skipping 155 batches
```

The training should then continue with loss values consistent with the checkpoint, not starting from ~1.0+ loss.

## Troubleshooting

### Training Starts from Step 0
**Problem**: Training shows "Starting from Step: 1" instead of the expected step.

**Solution**: This was the bug we just fixed. The issue was that `train.py` wasn't calling `checkpoint_manager.load_checkpoint()`. This is now resolved.

### High Loss After Resume
**Problem**: Loss jumps back to ~1.0 after resuming.

**Possible Causes**:
1. Checkpoint wasn't properly loaded
2. Learning rate scheduler wasn't restored
3. Dataloader state mismatch

**Check**: Look for these messages in the logs:
- ✅ "Resumed from checkpoint"
- ✅ "Restored scheduler state"
- ✅ "Loaded training metrics"

### Checkpoint Not Found
**Problem**: "No files found for checkpoint"

**Solutions**:
1. Check checkpoint name spelling
2. Verify GCS credentials
3. Ensure checkpoint was successfully uploaded

## Examples

### Resume from Latest
```bash
# Find the latest checkpoint and resume
./start_training.sh --config test --resume checkpoint-epoch0-step1000-files2
```

### Resume Production Training
```bash
# Resume production training from specific step
./start_training.sh --config production --resume checkpoint-epoch1-step5000-files10
```

### Resume with Different Settings
```bash
# Resume but override max steps
./start_training.sh --config test --resume checkpoint-epoch0-step800-files0 --max-steps 2000
```

## Technical Details

### Checkpoint Loading Process
1. Parse `--resume` argument
2. Initialize CheckpointManager
3. Call `load_checkpoint()` which:
   - Downloads checkpoint from GCS if needed
   - Loads model/optimizer state via `accelerator.load_state()`
   - Restores scheduler state from `scheduler_state.pt`
   - Loads training metrics from `training_metrics.json`
   - Loads metadata from `metadata.json`

### Dataloader Handling
- The training loop calculates which batches to skip in the current epoch
- If resuming mid-epoch, it skips the appropriate number of batches
- This ensures no data is reprocessed when resuming

### Progress Tracking
- Progress bar initializes with `initial=completed_steps`
- Step counter continues from checkpoint value
- Training stops when reaching `config.max_steps` as before 