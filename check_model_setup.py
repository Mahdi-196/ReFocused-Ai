#!/usr/bin/env python3
"""
Check Model Setup
Verifies that all necessary files and directories exist for training
"""

import os
import sys
import json
from pathlib import Path
from transformers import GPT2Config, GPT2Tokenizer

# Required paths
REQUIRED_PATHS = {
    "models/gpt_750m/config.json": "Model configuration",
    "models/tokenizer/tokenizer": "Tokenizer directory",
    "05_model_training/config/h100_multi_gpu.yaml": "Multi-GPU training config",
    "05_model_training/config/h100_deepspeed_multi.json": "DeepSpeed configuration",
    "05_model_training/train_pytorch.py": "Training script",
    "05_model_training/download_data.py": "Data download script",
    "05_model_training/quick_test.py": "Testing script"
}

def print_status(message, success=True):
    """Print a status message with an emoji indicator"""
    indicator = "✅" if success else "❌"
    print(f"{indicator} {message}")

def check_file_exists(filepath, description):
    """Check if a file exists and print status"""
    path = Path(filepath)
    if path.exists():
        print_status(f"{description} found at: {path}")
        return True
    else:
        print_status(f"{description} not found at: {path}", success=False)
        return False

def check_model_config():
    """Check if model config is valid"""
    config_path = Path("models/gpt_750m/config.json")
    if not config_path.exists():
        print_status("Model config file missing", success=False)
        return False
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Check key parameters
        if 'hidden_size' not in config_data:
            print_status("Model config missing 'hidden_size'", success=False)
            return False
            
        print_status(f"Valid model config found: {config_data['hidden_size']} hidden size, "
                    f"{config_data.get('num_hidden_layers', 'unknown')} layers")
        
        # Try to load with transformers
        try:
            config = GPT2Config.from_dict(config_data)
            print_status(f"Successfully loaded config with transformers")
            return True
        except Exception as e:
            print_status(f"Error loading config with transformers: {e}", success=False)
            return False
            
    except Exception as e:
        print_status(f"Error reading model config: {e}", success=False)
        return False

def check_tokenizer():
    """Check if tokenizer is valid"""
    tokenizer_path = Path("models/tokenizer/tokenizer")
    if not tokenizer_path.exists():
        print_status("Tokenizer directory missing", success=False)
        return False
    
    # Check for vocab.json and merges.txt
    vocab_file = tokenizer_path / "vocab.json"
    merges_file = tokenizer_path / "merges.txt"
    
    if not vocab_file.exists():
        print_status("Tokenizer missing vocab.json", success=False)
        return False
        
    if not merges_file.exists():
        print_status("Tokenizer missing merges.txt", success=False)
        return False
    
    # Try to load tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_path))
        vocab_size = len(tokenizer)
        print_status(f"Successfully loaded tokenizer with {vocab_size} tokens")
        return True
    except Exception as e:
        print_status(f"Error loading tokenizer: {e}", success=False)
        return False

def check_deepspeed_config():
    """Check DeepSpeed configuration"""
    config_path = Path("05_model_training/config/h100_deepspeed_multi.json")
    if not config_path.exists():
        print_status("DeepSpeed config missing", success=False)
        return False
    
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Check key parameters
        if 'zero_optimization' not in config_data:
            print_status("DeepSpeed config missing 'zero_optimization'", success=False)
            return False
            
        zero_stage = config_data['zero_optimization'].get('stage', 'unknown')
        micro_batch = config_data.get('train_micro_batch_size_per_gpu', 'unknown')
        
        print_status(f"Valid DeepSpeed config found: ZeRO stage {zero_stage}, "
                    f"micro-batch size {micro_batch}")
        return True
            
    except Exception as e:
        print_status(f"Error reading DeepSpeed config: {e}", success=False)
        return False

def main():
    print("\n=== Model Setup Checker ===\n")
    
    # Check all required paths
    all_paths_exist = True
    for path, description in REQUIRED_PATHS.items():
        if not check_file_exists(path, description):
            all_paths_exist = False
    
    # Check model config
    model_config_valid = check_model_config()
    
    # Check tokenizer
    tokenizer_valid = check_tokenizer()
    
    # Check DeepSpeed config
    deepspeed_config_valid = check_deepspeed_config()
    
    # Overall status
    print("\n=== Summary ===")
    if all_paths_exist and model_config_valid and tokenizer_valid and deepspeed_config_valid:
        print_status("All checks passed! Your setup appears to be ready for training.")
    else:
        print_status("Some checks failed. Please fix the issues above before training.", success=False)
    
    # Provide next steps
    print("\nNext steps:")
    print("1. Run test bucket access: python 05_model_training/test_bucket_access.py")
    print("2. Run a quick test: python 05_model_training/quick_test.py")
    print("3. Start training: cd 05_model_training && bash h100_runner.sh")

if __name__ == "__main__":
    main() 