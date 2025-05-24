#!/usr/bin/env python3
"""
Training Preparation for ReFocused AI
Sets up model training environment and configurations
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset, DatasetDict, load_dataset
from loguru import logger

class TrainingPreparator:
    """Prepares training environment and configurations"""
    
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.config_dir = Path("configs")
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.processed_dir = self.data_dir / "processed"
        
    def check_data_ready(self) -> bool:
        """Check if processed data is ready for training"""
        logger.info("🔍 Checking if data is ready for training...")
        
        required_files = [
            self.processed_dir / "train.jsonl",
            self.processed_dir / "val.jsonl", 
            self.processed_dir / "test.jsonl",
            self.processed_dir / "metadata.json"
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        
        if missing_files:
            logger.warning(f"❌ Missing files: {[str(f) for f in missing_files]}")
            logger.info("Run 'python data_processor.py' first to prepare data")
            return False
        
        # Check metadata
        with open(self.processed_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        if metadata['train_items'] < 100:
            logger.warning(f"⚠️ Very small training set: {metadata['train_items']} items")
            logger.info("Consider collecting more data for better training results")
        
        logger.success(f"✅ Data ready: {metadata['train_items']} training items")
        return True
    
    def setup_model_config(self, base_model: str = "microsoft/DialoGPT-medium") -> Dict[str, Any]:
        """Set up model configuration for training"""
        logger.info(f"⚙️ Setting up model config for {base_model}...")
        
        config = {
            'model_name': base_model,
            'task_type': 'causal_lm',
            'max_length': 512,
            'training_args': {
                'output_dir': str(self.model_dir / "checkpoints"),
                'overwrite_output_dir': True,
                'num_train_epochs': 3,
                'per_device_train_batch_size': 4,
                'per_device_eval_batch_size': 4,
                'gradient_accumulation_steps': 2,
                'learning_rate': 5e-5,
                'weight_decay': 0.01,
                'logging_steps': 50,
                'eval_steps': 500,
                'save_steps': 1000,
                'evaluation_strategy': 'steps',
                'save_strategy': 'steps',
                'load_best_model_at_end': True,
                'metric_for_best_model': 'eval_loss',
                'greater_is_better': False,
                'warmup_steps': 100,
                'fp16': torch.cuda.is_available(),
                'dataloader_pin_memory': False,
                'remove_unused_columns': False
            },
            'tokenizer_args': {
                'padding': True,
                'truncation': True,
                'max_length': 512,
                'return_tensors': 'pt'
            }
        }
        
        return config
    
    def save_training_config(self, config: Dict[str, Any]):
        """Save training configuration to file"""
        config_path = self.config_dir / "training_config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.success(f"✅ Saved training config to {config_path}")
    
    def prepare_datasets(self) -> Optional[DatasetDict]:
        """Prepare datasets for training"""
        logger.info("📚 Preparing datasets...")
        
        try:
            # Load datasets from JSON Lines files
            train_dataset = load_dataset('json', data_files=str(self.processed_dir / "train.jsonl"))['train']
            val_dataset = load_dataset('json', data_files=str(self.processed_dir / "val.jsonl"))['train']
            test_dataset = load_dataset('json', data_files=str(self.processed_dir / "test.jsonl"))['train']
            
            dataset_dict = DatasetDict({
                'train': train_dataset,
                'validation': val_dataset,
                'test': test_dataset
            })
            
            logger.success(f"✅ Datasets prepared: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
            return dataset_dict
            
        except Exception as e:
            logger.error(f"❌ Error preparing datasets: {e}")
            return None
    
    def setup_tokenizer(self, model_name: str) -> Optional[AutoTokenizer]:
        """Set up tokenizer for the model"""
        logger.info(f"🔤 Setting up tokenizer for {model_name}...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            logger.success("✅ Tokenizer setup complete")
            return tokenizer
            
        except Exception as e:
            logger.error(f"❌ Error setting up tokenizer: {e}")
            return None
    
    def test_model_loading(self, model_name: str) -> bool:
        """Test if the model can be loaded"""
        logger.info(f"🧪 Testing model loading for {model_name}...")
        
        try:
            # Test tokenizer
            tokenizer = self.setup_tokenizer(model_name)
            if tokenizer is None:
                return False
            
            # Test model loading (don't actually load to save memory/time)
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name)
            
            logger.success("✅ Model loading test passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model loading test failed: {e}")
            return False
    
    def create_training_script(self):
        """Create a ready-to-run training script"""
        script_content = '''#!/usr/bin/env python3
"""
ReFocused AI Training Script
Run this script to start model training
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from datasets import load_dataset
import yaml
import json
from pathlib import Path
from loguru import logger

def load_config():
    """Load training configuration"""
    with open('configs/training_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def prepare_data(tokenizer, config):
    """Prepare and tokenize datasets"""
    logger.info("📚 Loading and tokenizing datasets...")
    
    # Load datasets
    train_dataset = load_dataset('json', data_files='data/processed/train.jsonl')['train']
    val_dataset = load_dataset('json', data_files='data/processed/val.jsonl')['train']
    
    def tokenize_function(examples):
        # Use the 'text' field from our processed data
        return tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=config['max_length']
        )
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, val_dataset

def main():
    """Main training function"""
    logger.info("🚀 Starting ReFocused AI training...")
    
    # Load configuration
    config = load_config()
    
    # Setup tokenizer and model
    logger.info(f"🔤 Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"🤖 Loading model: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])
    
    # Prepare datasets
    train_dataset, val_dataset = prepare_data(tokenizer, config)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(**config['training_args'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("🎯 Starting training...")
    trainer.train()
    
    # Save final model
    output_dir = Path(config['training_args']['output_dir'])
    final_model_dir = output_dir.parent / "final_model"
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    logger.success(f"✅ Training complete! Model saved to {final_model_dir}")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path("train_model.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.success(f"✅ Created training script: {script_path}")
    
    def create_inference_script(self):
        """Create a script for model inference/testing"""
        script_content = '''#!/usr/bin/env python3
"""
ReFocused AI Inference Script
Test your trained model with custom prompts
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from loguru import logger

class ReFocusedAI:
    """ReFocused AI model for inference"""
    
    def __init__(self, model_path: str = "models/final_model"):
        self.model_path = Path(model_path)
        self.tokenizer = None
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        logger.info(f"🤖 Loading model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("🚀 Model loaded on GPU")
            else:
                logger.info("💻 Model loaded on CPU")
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.info("Make sure you've trained a model first using train_model.py")
    
    def generate_response(self, prompt: str, max_length: int = 200, temperature: float = 0.7) -> str:
        """Generate a response to the given prompt"""
        if self.model is None or self.tokenizer is None:
            return "Model not loaded. Please train a model first."
        
        # Encode input
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        return response

def main():
    """Interactive inference session"""
    print("🧠 ReFocused AI - Interactive Mode")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    ai = ReFocusedAI()
    
    if ai.model is None:
        print("❌ No trained model found. Run 'python train_model.py' first.")
        return
    
    while True:
        try:
            prompt = input("\\n💭 You: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("🤖 ReFocused AI: ", end="")
            response = ai.generate_response(prompt)
            print(response)
            
        except KeyboardInterrupt:
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
'''
        
        script_path = Path("test_model.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        logger.success(f"✅ Created inference script: {script_path}")
    
    def generate_readme(self):
        """Generate updated README for training phase"""
        readme_content = '''# ReFocused AI - Training Phase

## Overview
This project trains an AI model focused on productivity, self-improvement, and personal development using collected data.

## Quick Start

### 1. Process Your Data
```bash
python data_processor.py
```
This will:
- Load all collected data from `data/` directories
- Clean and filter the content
- Create train/validation/test splits
- Save processed data to `data/processed/`

### 2. Start Training
```bash
python train_model.py
```
This will:
- Load the processed datasets
- Train the model using your configuration
- Save checkpoints and the final model

### 3. Test Your Model
```bash
python test_model.py
```
This starts an interactive session to test your trained model.

## Project Structure

```
ReFocused-AI/
├── data/                     # Your collected data
│   ├── reddit_ultra_fast/    # Reddit data
│   ├── reddit_enhanced/      # Enhanced Reddit data
│   └── processed/            # Processed training data
├── models/                   # Trained models and checkpoints
├── configs/                  # Training configurations
├── data_processor.py         # Data processing pipeline
├── training_prep.py          # Training preparation
├── train_model.py           # Main training script
└── test_model.py            # Model inference/testing
```

## Configuration

Training settings are stored in `configs/training_config.yaml`. Key parameters:
- `model_name`: Base model to fine-tune
- `num_train_epochs`: Number of training epochs
- `learning_rate`: Learning rate for training
- `batch_size`: Training batch size

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Processing

The data processor:
1. Loads data from all collection directories
2. Standardizes format across sources
3. Cleans text content
4. Filters for quality
5. Creates training splits
6. Saves in formats ready for training

## Training

The training process:
1. Loads processed datasets
2. Sets up tokenizer and model
3. Configures training parameters
4. Trains using HuggingFace Transformers
5. Saves checkpoints and final model

## Model Testing

Use the interactive test script to:
- Chat with your trained model
- Test different prompts
- Evaluate model responses
- Experiment with generation parameters

## Tips for Better Results

1. **More Data**: Collect more high-quality training data
2. **Quality Filtering**: Adjust quality filters in data_processor.py
3. **Training Time**: Train for more epochs for better results
4. **Model Size**: Use larger base models if you have computational resources

## Troubleshooting

**Out of Memory**: Reduce batch size in training config
**Poor Quality**: Improve data filtering or collect more data
**Slow Training**: Use GPU if available, reduce model size
'''
        
        readme_path = Path("README_TRAINING.md")
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        logger.success(f"✅ Generated training README: {readme_path}")

def main():
    """Main preparation pipeline"""
    logger.info("🚀 Preparing training environment...")
    
    prep = TrainingPreparator()
    
    # Check if data is ready
    if not prep.check_data_ready():
        logger.info("💡 Run 'python data_processor.py' to prepare your data first")
        return
    
    # Setup model configuration
    config = prep.setup_model_config()
    prep.save_training_config(config)
    
    # Test model loading
    model_ready = prep.test_model_loading(config['model_name'])
    if not model_ready:
        logger.warning("⚠️ Model loading test failed - you may need internet to download the base model")
    
    # Prepare datasets
    datasets = prep.prepare_datasets()
    if datasets is None:
        logger.error("❌ Failed to prepare datasets")
        return
    
    # Create training scripts
    prep.create_training_script()
    prep.create_inference_script()
    prep.generate_readme()
    
    logger.success("✅ Training environment ready!")
    logger.info("📋 Next steps:")
    logger.info("   1. Review configs/training_config.yaml")
    logger.info("   2. Run: python train_model.py")
    logger.info("   3. Test with: python test_model.py")

if __name__ == "__main__":
    main() 