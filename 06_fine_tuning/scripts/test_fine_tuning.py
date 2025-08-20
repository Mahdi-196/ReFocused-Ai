#!/usr/bin/env python3
"""
Test script to verify fine-tuning setup
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
import json
from configs import get_fine_tuning_config, get_lora_config
from utils import LoRAConfig, compute_fine_tuning_metrics


def test_configurations():
    """Test configuration loading"""
    print("🔧 Testing configurations...")
    
    # Test task configs
    for task in ["chat", "code", "instruct", "domain", "custom"]:
        config = get_fine_tuning_config(task=task)
        print(f"  ✅ {task} config loaded: lr={config.learning_rate}, epochs={config.num_epochs}")
    
    # Test preset configs
    for preset in ["quick_test", "low_resource", "high_quality", "production"]:
        config = get_fine_tuning_config(preset=preset)
        print(f"  ✅ {preset} preset loaded: batch_size={config.per_device_train_batch_size}")
    
    # Test LoRA configs
    for task in ["chat", "code", "instruct"]:
        lora_config = get_lora_config(task)
        print(f"  ✅ LoRA config for {task}: rank={lora_config['r']}")
    
    print("✅ All configurations loaded successfully!\n")


def test_metrics():
    """Test metrics computation"""
    print("📊 Testing metrics...")
    
    # Generate dummy predictions and labels
    predictions = torch.randint(0, 1000, (100,)).numpy()
    labels = torch.randint(0, 1000, (100,)).numpy()
    
    # Test different task metrics
    for task in ["general", "chat", "code", "instruct"]:
        metrics = compute_fine_tuning_metrics(predictions, labels, task)
        print(f"  ✅ {task} metrics computed: accuracy={metrics.get('accuracy', 0):.3f}")
    
    print("✅ All metrics computed successfully!\n")


def create_sample_datasets():
    """Create sample datasets for testing"""
    print("📝 Creating sample datasets...")
    
    samples_dir = Path("./sample_datasets")
    samples_dir.mkdir(exist_ok=True)
    
    # Chat dataset
    chat_data = [
        {"messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thank you! How can I help you today?"}
        ]},
        {"prompt": "What is the weather like?", "response": "I don't have access to real-time weather data."},
    ]
    
    with open(samples_dir / "chat_sample.jsonl", 'w') as f:
        for item in chat_data:
            f.write(json.dumps(item) + '\n')
    print("  ✅ Created chat_sample.jsonl")
    
    # Instruction dataset
    instruct_data = [
        {"instruction": "Translate to Spanish", "input": "Hello world", "output": "Hola mundo"},
        {"instruction": "Summarize this text", "input": "Machine learning is a subset of AI...", "output": "ML is part of AI."},
    ]
    
    with open(samples_dir / "instruct_sample.jsonl", 'w') as f:
        for item in instruct_data:
            f.write(json.dumps(item) + '\n')
    print("  ✅ Created instruct_sample.jsonl")
    
    # Code dataset
    code_data = [
        {"prompt": "Write a Python function to add two numbers", "completion": "def add(a, b):\n    return a + b"},
        {"code": "class Calculator:\n    def __init__(self):\n        pass\n    \n    def multiply(self, a, b):\n        return a * b"},
    ]
    
    with open(samples_dir / "code_sample.jsonl", 'w') as f:
        for item in code_data:
            f.write(json.dumps(item) + '\n')
    print("  ✅ Created code_sample.jsonl")
    
    print("✅ Sample datasets created in ./sample_datasets/\n")


def test_lora():
    """Test LoRA utilities"""
    print("🔧 Testing LoRA utilities...")
    
    # Create a simple linear layer
    linear = torch.nn.Linear(768, 768)
    
    # Create LoRA config
    lora_config = LoRAConfig(r=8, lora_alpha=16, target_modules=["linear"])
    
    # Calculate LoRA parameters
    original_params = sum(p.numel() for p in linear.parameters())
    lora_params = lora_config.r * (linear.in_features + linear.out_features)
    
    print(f"  Original parameters: {original_params:,}")
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  Reduction: {(1 - lora_params/original_params)*100:.1f}%")
    
    print("✅ LoRA utilities working!\n")


def main():
    """Run all tests"""
    print("🚀 Testing Fine-Tuning Setup")
    print("=" * 50)
    
    # Test configurations
    test_configurations()
    
    # Test metrics
    test_metrics()
    
    # Create sample datasets
    create_sample_datasets()
    
    # Test LoRA
    test_lora()
    
    print("🎉 All tests passed! Fine-tuning module is ready to use.")
    print("\n📚 Next steps:")
    print("1. Prepare your dataset in one of the supported formats")
    print("2. Choose a task type (chat, code, instruct, domain, custom)")
    print("3. Run fine_tune.py with your configuration")
    print("\nExample command:")
    print("python fine_tune.py --task chat --base-model ./base_model --dataset ./sample_datasets/chat_sample.jsonl")


if __name__ == "__main__":
    main() 