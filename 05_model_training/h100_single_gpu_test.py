#!/usr/bin/env python3
"""
H100 Single GPU Test Script
Tests model training on a single H100 SXM GPU with 25 files
Optimized for memory efficiency and stability
"""

import os
import sys
import argparse
import yaml
import json
import logging
import time
from pathlib import Path
import torch
import numpy as np

# Set critical environment variables for H100 SXM
os.environ["RDMAV_FORK_SAFE"] = "1"
os.environ["FI_EFA_FORK_SAFE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

# Import training module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_pytorch import ModelTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def create_test_config(output_dir="test_output"):
    """Create optimized configuration for H100 SXM single GPU test"""
    
    # Create base test configuration
    config = {
        "model": {
            "config_path": "../models/gpt_750m/config.json",
            "vocab_size": 50000
        },
        "tokenizer": {
            "path": "../models/tokenizer/tokenizer"
        },
        "data": {
            "use_gcs": False,
            "local_data_dir": "/home/ubuntu/training_data/shards",
            "max_seq_length": 1024,
            "dataloader_num_workers": 2,
            "dataloader_prefetch_factor": 2,
            "npz_key_priority": ["input_ids", "arr_0", "sequences", "text"]
        },
        "training": {
            "output_dir": output_dir,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            "learning_rate": 3e-4,
            "weight_decay": 0.01,
            "warmup_steps": 2,
            "total_steps": 100,
            "save_steps": 50,
            "eval_steps": 50,
            "logging_steps": 1,
            "fp16": False,
            "bf16": True,
            "gradient_checkpointing": True
        },
        "checkpointing": {
            "output_dir": f"{output_dir}/checkpoints",
            "remote_checkpoint_bucket": "refocused-ai",
            "remote_checkpoint_path": "checkpoints/h100_test",
            "save_steps": 50,
            "keep_last_n_checkpoints": 2
        },
        "monitoring": {
            "logging_dir": f"{output_dir}/logs",
            "log_steps": 1,
            "monitor_gpu_memory": False,
            "monitor_system_metrics": False
        },
        "hardware": {
            "num_gpus": 1,
            "gpu_type": "H100_SXM"
        },
        "environment": {
            "seed": 42,
            "cuda_visible_devices": "0",
            "torch_backends_cudnn_benchmark": True
        }
    }
    
    return config

def prepare_test_data(data_dir, num_files=25):
    """Prepare a subset of training files for testing"""
    logger.info(f"Preparing {num_files} test files in {data_dir}")
    
    data_dir = Path(data_dir)
    all_files = list(data_dir.glob("*.npz"))
    
    if len(all_files) < num_files:
        logger.warning(f"Only {len(all_files)} files available, using all of them")
        return all_files
    
    # Select first num_files
    test_files = all_files[:num_files]
    logger.info(f"Selected {len(test_files)} files for testing")
    
    return test_files

def main():
    parser = argparse.ArgumentParser(description="Run H100 SXM single GPU test")
    parser.add_argument("--output_dir", type=str, default="h100_test_output", 
                        help="Directory for test outputs")
    parser.add_argument("--data_dir", type=str, default="/home/ubuntu/training_data/shards",
                        help="Directory containing training data")
    parser.add_argument("--num_files", type=int, default=25,
                        help="Number of files to use for testing")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create test configuration
    logger.info("Creating test configuration")
    config = create_test_config(args.output_dir)
    
    # Update data directory
    config["data"]["local_data_dir"] = args.data_dir
    
    # Prepare test data
    prepare_test_data(args.data_dir, args.num_files)
    
    # Save configuration
    config_path = Path(args.output_dir) / "h100_test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Test configuration saved to {config_path}")
    
    # Start training
    logger.info("Starting H100 SXM single GPU test")
    trainer = ModelTrainer(str(config_path))
    
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    # Report results
    duration = end_time - start_time
    logger.info(f"Test completed in {duration:.2f} seconds")
    
    # Calculate throughput
    steps = trainer.global_step
    tokens_processed = steps * config["training"]["per_device_train_batch_size"] * \
                       config["training"]["gradient_accumulation_steps"] * \
                       config["data"]["max_seq_length"]
    
    throughput = tokens_processed / duration
    logger.info(f"Processed {tokens_processed} tokens at {throughput:.2f} tokens/sec")
    logger.info(f"Completed {steps} steps at {steps/duration:.2f} steps/sec")
    
    # Save benchmark results
    benchmark = {
        "duration_seconds": duration,
        "steps_completed": steps,
        "tokens_processed": tokens_processed,
        "tokens_per_second": throughput,
        "steps_per_second": steps/duration,
        "gpu_type": "H100_SXM",
        "batch_size": config["training"]["per_device_train_batch_size"],
        "gradient_accumulation_steps": config["training"]["gradient_accumulation_steps"],
        "sequence_length": config["data"]["max_seq_length"]
    }
    
    benchmark_path = Path(args.output_dir) / "benchmark_results.json"
    with open(benchmark_path, 'w') as f:
        json.dump(benchmark, f, indent=2)
    
    logger.info(f"Benchmark results saved to {benchmark_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 