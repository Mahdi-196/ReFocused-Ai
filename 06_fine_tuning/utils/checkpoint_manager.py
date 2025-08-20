#!/usr/bin/env python3
"""
Checkpoint manager for fine-tuning
"""

import os
import json
import torch
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class FineTuningCheckpointManager:
    """Manages checkpoints during fine-tuning"""
    
    def __init__(
        self,
        output_dir: Path,
        task_type: str,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        keep_best_n: int = 3
    ):
        self.output_dir = Path(output_dir)
        self.task_type = task_type
        self.push_to_hub = push_to_hub
        self.hub_model_id = hub_model_id
        self.keep_best_n = keep_best_n
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.best_model_dir = self.output_dir / "best_model"
        self.final_model_dir = self.output_dir / "final_model"
        
        # Track checkpoints
        self.checkpoint_scores = {}
        
    def save_checkpoint(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        epoch: int,
        best_eval_loss: float,
        accelerator,
        additional_info: Optional[Dict] = None
    ):
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint-step-{step}"
        checkpoint_path = self.checkpoints_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        logger.info(f"💾 Saving checkpoint to {checkpoint_path}")
        
        # Save model state
        if hasattr(model, 'module'):
            model_to_save = model.module
        else:
            model_to_save = model
        
        # Save model
        model_to_save.save_pretrained(checkpoint_path)
        
        # Save optimizer and scheduler states
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': step,
            'epoch': epoch,
            'best_eval_loss': best_eval_loss,
            'task_type': self.task_type,
            'timestamp': datetime.now().isoformat(),
            'additional_info': additional_info or {}
        }, checkpoint_path / "training_state.pt")
        
        # Save accelerator state if available
        if accelerator and hasattr(accelerator, 'save_state'):
            accelerator.save_state(checkpoint_path / "accelerator_state")
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_name}")
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
    
    def save_best_model(
        self,
        model,
        tokenizer: AutoTokenizer,
        eval_loss: float,
        step: int,
        additional_metrics: Optional[Dict] = None
    ):
        """Save the best model based on evaluation loss"""
        logger.info(f"🏆 Saving best model with eval_loss={eval_loss:.4f}")
        
        # Save model
        if hasattr(model, 'module'):
            model_to_save = model.module
        else:
            model_to_save = model
        
        model_to_save.save_pretrained(self.best_model_dir)
        tokenizer.save_pretrained(self.best_model_dir)
        
        # Save metadata
        metadata = {
            'eval_loss': eval_loss,
            'step': step,
            'task_type': self.task_type,
            'timestamp': datetime.now().isoformat(),
            'metrics': additional_metrics or {}
        }
        
        with open(self.best_model_dir / "best_model_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Best model saved to {self.best_model_dir}")
    
    def save_final_model(
        self,
        model,
        tokenizer: AutoTokenizer,
        training_args: Dict,
        final_metrics: Dict
    ):
        """Save the final fine-tuned model"""
        logger.info("📦 Saving final model...")
        
        # Save model
        if hasattr(model, 'module'):
            model_to_save = model.module
        else:
            model_to_save = model
        
        model_to_save.save_pretrained(self.final_model_dir)
        tokenizer.save_pretrained(self.final_model_dir)
        
        # Create model card
        model_card = self._create_model_card(training_args, final_metrics)
        with open(self.final_model_dir / "README.md", 'w') as f:
            f.write(model_card)
        
        # Save training info
        training_info = {
            'task_type': self.task_type,
            'training_args': training_args,
            'final_metrics': final_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.final_model_dir / "training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"✅ Final model saved to {self.final_model_dir}")
        
        # Push to hub if requested
        if self.push_to_hub and self.hub_model_id:
            self._push_to_hub(model_to_save, tokenizer)
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model,
        optimizer,
        scheduler,
        accelerator
    ) -> Optional[Dict]:
        """Load checkpoint for resuming training"""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            # Try to find in checkpoints directory
            checkpoint_path = self.checkpoints_dir / checkpoint_path
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return None
        
        logger.info(f"📂 Loading checkpoint from {checkpoint_path}")
        
        try:
            # Load model
            if (checkpoint_path / "pytorch_model.bin").exists():
                model.load_state_dict(
                    torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
                )
            elif (checkpoint_path / "model.safetensors").exists():
                from safetensors.torch import load_file
                model.load_state_dict(
                    load_file(checkpoint_path / "model.safetensors")
                )
            
            # Load training state
            training_state = torch.load(
                checkpoint_path / "training_state.pt",
                map_location="cpu"
            )
            
            optimizer.load_state_dict(training_state['optimizer_state_dict'])
            scheduler.load_state_dict(training_state['scheduler_state_dict'])
            
            # Load accelerator state if available
            if accelerator and (checkpoint_path / "accelerator_state").exists():
                accelerator.load_state(checkpoint_path / "accelerator_state")
            
            logger.info(f"✅ Checkpoint loaded successfully")
            return training_state
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the best N"""
        checkpoints = list(self.checkpoints_dir.glob("checkpoint-*"))
        
        if len(checkpoints) <= self.keep_best_n:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest checkpoints
        for checkpoint in checkpoints[:-self.keep_best_n]:
            logger.info(f"🗑️ Removing old checkpoint: {checkpoint.name}")
            shutil.rmtree(checkpoint)
    
    def _create_model_card(self, training_args: Dict, final_metrics: Dict) -> str:
        """Create a model card for the fine-tuned model"""
        model_card = f"""# {training_args.get('base_model', 'ReFocused-AI')} Fine-tuned for {self.task_type}

This model is a fine-tuned version of {training_args.get('base_model', 'ReFocused-AI')} for {self.task_type} tasks.

## Model Details

- **Base Model**: {training_args.get('base_model', 'ReFocused-AI')}
- **Task Type**: {self.task_type}
- **Fine-tuning Dataset**: {training_args.get('dataset', 'Custom dataset')}
- **Training Steps**: {final_metrics.get('total_steps', 'N/A')}
- **Final Loss**: {final_metrics.get('final_loss', 'N/A'):.4f}
- **Best Eval Loss**: {final_metrics.get('best_eval_loss', 'N/A'):.4f}

## Training Configuration

- **Learning Rate**: {training_args.get('learning_rate', 'N/A')}
- **Batch Size**: {training_args.get('per_device_train_batch_size', 'N/A')}
- **Epochs**: {training_args.get('num_epochs', 'N/A')}
- **Mixed Precision**: {training_args.get('mixed_precision', 'N/A')}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{self.hub_model_id or 'path/to/model'}")
tokenizer = AutoTokenizer.from_pretrained("{self.hub_model_id or 'path/to/model'}")

# Example usage
input_text = "Your input here"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Training Metrics

- **Training Time**: {final_metrics.get('training_time', 0)/3600:.2f} hours
- **Final Loss**: {final_metrics.get('final_loss', 'N/A'):.4f}

## License

This model inherits the license from the base model.

## Citation

If you use this model, please cite:

```
@misc{{refocused-ai-{self.task_type},
  title={{ReFocused-AI Fine-tuned for {self.task_type}}},
  author={{Your Name}},
  year={{2024}},
  publisher={{HuggingFace}}
}}
```
"""
        return model_card
    
    def _push_to_hub(self, model, tokenizer):
        """Push model to HuggingFace Hub"""
        try:
            logger.info(f"🚀 Pushing model to HuggingFace Hub: {self.hub_model_id}")
            
            model.push_to_hub(
                self.hub_model_id,
                private=True,
                commit_message=f"Fine-tuned for {self.task_type}"
            )
            
            tokenizer.push_to_hub(
                self.hub_model_id,
                private=True,
                commit_message=f"Tokenizer for {self.task_type} fine-tuning"
            )
            
            logger.info(f"✅ Model pushed to: https://huggingface.co/{self.hub_model_id}")
            
        except Exception as e:
            logger.error(f"Error pushing to hub: {e}")
    
    def wait_for_uploads(self):
        """Wait for any background uploads to complete"""
        # This is a placeholder for future async upload functionality
        pass 