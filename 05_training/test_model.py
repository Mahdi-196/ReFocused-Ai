#!/usr/bin/env python3
"""
Test script to verify GPT model architecture and parameter counts
"""

import sys
import torch
from train import GPTModel, MODEL_CONFIGS


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model_sizes():
    """Test each model configuration"""
    print("Testing GPT model configurations...\n")
    
    for size_name, config in MODEL_CONFIGS.items():
        print(f"Testing {size_name} model:")
        print(f"  Config: {config}")
        
        # Create model
        model = GPTModel(config)
        
        # Count parameters
        total_params = count_parameters(model)
        total_params_m = total_params / 1e6
        
        print(f"  Total parameters: {total_params:,} ({total_params_m:.1f}M)")
        
        # Test forward pass
        batch_size = 2
        seq_len = 128
        dummy_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
        
        try:
            with torch.no_grad():
                output = model(dummy_input)
                assert output["logits"].shape == (batch_size, seq_len, config["vocab_size"])
                print(f"  ✓ Forward pass successful")
        except Exception as e:
            print(f"  ✗ Forward pass failed: {e}")
        
        # Memory estimate
        param_memory_gb = (total_params * 4) / (1024**3)  # FP32
        param_memory_gb_fp16 = (total_params * 2) / (1024**3)  # FP16
        
        print(f"  Memory (params only):")
        print(f"    FP32: {param_memory_gb:.2f} GB")
        print(f"    FP16: {param_memory_gb_fp16:.2f} GB")
        print()


def test_attention_implementations():
    """Test standard vs flash attention"""
    print("Testing attention implementations...\n")
    
    config = MODEL_CONFIGS["125M"]
    model = GPTModel(config)
    
    # Test input
    batch_size = 4
    seq_len = 512
    dummy_input = torch.randint(0, config["vocab_size"], (batch_size, seq_len))
    
    # Test with CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()
        print("Testing on CUDA")
    else:
        print("Testing on CPU (CUDA not available)")
    
    # Run forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"✓ Attention test passed")
    print(f"  Output shape: {output['logits'].shape}")
    
    # Check if flash attention is being used
    from train import FLASH_ATTN_AVAILABLE
    if FLASH_ATTN_AVAILABLE:
        print("✓ Flash Attention is available and will be used")
    else:
        print("ℹ Flash Attention not available, using standard attention")


def main():
    print("=" * 60)
    print("ReFocused-AI GPT Model Test")
    print("=" * 60)
    print()
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    # Run tests
    test_model_sizes()
    print("-" * 60)
    test_attention_implementations()
    
    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main() 