#!/usr/bin/env python3
"""
Test script to validate performance optimizations
"""

import torch
import time
import psutil
import gc
from transformers import GPTNeoXForCausalLM
from accelerate import Accelerator
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_model_config, get_training_config
from utils import create_dataloader

def test_memory_usage():
    """Test GPU memory usage with different batch sizes"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU memory test")
        return
    
    print("🧪 Testing GPU Memory Usage")
    print("=" * 50)
    
    # Test configurations
    test_configs = [
        {"batch_size": 1, "grad_acc": 1, "mixed_precision": "no"},
        {"batch_size": 2, "grad_acc": 2, "mixed_precision": "no"},
        {"batch_size": 2, "grad_acc": 2, "mixed_precision": "bf16"},
        {"batch_size": 4, "grad_acc": 4, "mixed_precision": "bf16"},
    ]
    
    for test_config in test_configs:
        print(f"\nTesting batch_size={test_config['batch_size']}, "
              f"grad_acc={test_config['grad_acc']}, "
              f"mixed_precision={test_config['mixed_precision']}")
        
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            gc.collect()
            
            # Initialize accelerator
            accelerator = Accelerator(
                gradient_accumulation_steps=test_config['grad_acc'],
                mixed_precision=test_config['mixed_precision']
            )
            
            # Create small model for testing
            model_config = get_model_config()
            model_config.hidden_size = 512  # Smaller for testing
            model_config.num_hidden_layers = 4
            model = GPTNeoXForCausalLM(model_config)
            model = accelerator.prepare(model)
            
            # Create dummy batch
            seq_len = 1024
            batch = {
                'input_ids': torch.randint(0, 1000, (test_config['batch_size'], seq_len)),
                'attention_mask': torch.ones(test_config['batch_size'], seq_len),
                'labels': torch.randint(0, 1000, (test_config['batch_size'], seq_len))
            }
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            # Measure memory before
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1024**3  # GB
            
            # Forward pass
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
            
            # Measure memory after
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / 1024**3  # GB
            
            effective_batch = test_config['batch_size'] * test_config['grad_acc']
            print(f"  ✅ Memory used: {mem_after - mem_before:.2f} GB")
            print(f"  📊 Effective batch size: {effective_batch}")
            print(f"  🎯 Memory per effective batch item: {(mem_after - mem_before) / effective_batch:.3f} GB")
            
            # Cleanup
            del model, accelerator, batch, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            torch.cuda.empty_cache()
            gc.collect()


def test_dataloader_performance():
    """Test DataLoader performance with different settings"""
    print("\n🧪 Testing DataLoader Performance")
    print("=" * 50)
    
    config = get_training_config("test")
    
    # Test different DataLoader configurations
    test_configs = [
        {"num_workers": 0, "pin_memory": False, "prefetch_factor": 2},
        {"num_workers": 2, "pin_memory": True, "prefetch_factor": 2},
        {"num_workers": 4, "pin_memory": True, "prefetch_factor": 4},
    ]
    
    for test_config in test_configs:
        print(f"\nTesting num_workers={test_config['num_workers']}, "
              f"pin_memory={test_config['pin_memory']}, "
              f"prefetch_factor={test_config['prefetch_factor']}")
        
        try:
            # Update config
            config.dataloader_num_workers = test_config['num_workers']
            config.pin_memory = test_config['pin_memory']
            config.prefetch_factor = test_config['prefetch_factor']
            config.max_files = 2  # Limit for testing
            
            accelerator = Accelerator(mixed_precision="no")
            
            # Create dataloader
            start_time = time.time()
            dataloader, num_files = create_dataloader(config, accelerator)
            creation_time = time.time() - start_time
            
            print(f"  📁 Files loaded: {num_files}")
            print(f"  ⏱️  DataLoader creation time: {creation_time:.2f}s")
            
            # Test iteration speed
            start_time = time.time()
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if batch_count >= 10:  # Test first 10 batches
                    break
            
            iteration_time = time.time() - start_time
            batches_per_second = batch_count / iteration_time if iteration_time > 0 else 0
            
            print(f"  🚀 Batches per second: {batches_per_second:.2f}")
            print(f"  📊 Total iteration time (10 batches): {iteration_time:.2f}s")
            
            # Cleanup
            del dataloader, accelerator
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")


def test_mixed_precision_performance():
    """Test mixed precision performance impact"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping mixed precision test")
        return
    
    print("\n🧪 Testing Mixed Precision Performance")
    print("=" * 50)
    
    precision_modes = ["no", "fp16", "bf16"]
    
    for precision in precision_modes:
        print(f"\nTesting mixed_precision={precision}")
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            gc.collect()
            
            accelerator = Accelerator(mixed_precision=precision)
            
            # Create model
            model_config = get_model_config()
            model_config.hidden_size = 768
            model_config.num_hidden_layers = 6
            model = GPTNeoXForCausalLM(model_config)
            model = accelerator.prepare(model)
            
            # Create batch
            batch_size = 2
            seq_len = 1024
            batch = {
                'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
                'attention_mask': torch.ones(batch_size, seq_len),
                'labels': torch.randint(0, 1000, (batch_size, seq_len))
            }
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            # Warmup
            for _ in range(3):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
            
            torch.cuda.synchronize()
            
            # Benchmark
            start_time = time.time()
            num_iterations = 10
            
            for _ in range(num_iterations):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations
            memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            print(f"  ⚡ Average forward+backward time: {avg_time:.3f}s")
            print(f"  💾 Peak memory usage: {memory_used:.2f} GB")
            print(f"  🎯 Tokens per second: {(batch_size * seq_len) / avg_time:.0f}")
            
            # Cleanup
            del model, accelerator, batch, outputs, loss
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            torch.cuda.empty_cache()
            gc.collect()


def test_gradient_accumulation():
    """Test gradient accumulation effectiveness"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping gradient accumulation test")
        return
    
    print("\n🧪 Testing Gradient Accumulation")
    print("=" * 50)
    
    # Test configurations
    configs = [
        {"batch_size": 4, "grad_acc": 1},  # Large batch, no accumulation
        {"batch_size": 2, "grad_acc": 2},  # Medium batch, 2x accumulation
        {"batch_size": 1, "grad_acc": 4},  # Small batch, 4x accumulation
    ]
    
    for config in configs:
        print(f"\nTesting batch_size={config['batch_size']}, grad_acc={config['grad_acc']}")
        
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            accelerator = Accelerator(
                gradient_accumulation_steps=config['grad_acc'],
                mixed_precision="bf16"
            )
            
            # Create model
            model_config = get_model_config()
            model_config.hidden_size = 512
            model_config.num_hidden_layers = 4
            model = GPTNeoXForCausalLM(model_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            
            model, optimizer = accelerator.prepare(model, optimizer)
            
            # Create batch
            seq_len = 512
            batch = {
                'input_ids': torch.randint(0, 1000, (config['batch_size'], seq_len)),
                'attention_mask': torch.ones(config['batch_size'], seq_len),
                'labels': torch.randint(0, 1000, (config['batch_size'], seq_len))
            }
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            
            # Measure gradient accumulation
            start_time = time.time()
            total_loss = 0
            
            for step in range(config['grad_acc']):
                with accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item()
                    accelerator.backward(loss)
                    
                    if accelerator.sync_gradients:
                        optimizer.step()
                        optimizer.zero_grad()
            
            end_time = time.time()
            
            effective_batch = config['batch_size'] * config['grad_acc']
            print(f"  🎯 Effective batch size: {effective_batch}")
            print(f"  📊 Average loss: {total_loss / config['grad_acc']:.4f}")
            print(f"  ⏱️  Time per accumulation cycle: {end_time - start_time:.3f}s")
            print(f"  🚀 Effective samples per second: {effective_batch / (end_time - start_time):.1f}")
            
            # Cleanup
            del model, optimizer, accelerator, batch, outputs
            torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            torch.cuda.empty_cache()
            gc.collect()


def main():
    """Run all performance tests"""
    print("🚀 ReFocused-AI Performance Optimization Tests")
    print("=" * 60)
    
    # System info
    print(f"🖥️  System Info:")
    print(f"   Python: {torch.__version__}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   RAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
    
    # Run tests
    test_dataloader_performance()
    test_mixed_precision_performance()
    test_gradient_accumulation()
    test_memory_usage()
    
    print("\n✅ All performance tests completed!")
    print("\n💡 Optimization Summary:")
    print("   - Increased batch sizes for better GPU utilization")
    print("   - Added gradient accumulation for effective larger batches")
    print("   - Enabled mixed precision (bf16/fp16) for memory and speed")
    print("   - Optimized DataLoader with workers and pin_memory")
    print("   - Reduced Python overhead in training loop")
    print("   - Optimized checkpoint frequency")
    print("   - Added cuDNN benchmarking")


if __name__ == "__main__":
    main() 