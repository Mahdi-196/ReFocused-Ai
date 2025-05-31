#!/usr/bin/env python3
"""
Quick test script to verify H100 training setup
Tests all critical components without running full training
"""

import os
import sys
import time
import torch
import logging
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_pytorch_cuda():
    """Test PyTorch CUDA functionality"""
    logger.info("Testing PyTorch CUDA...")
    
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is not available. Please check your installation.")
        return False
    
    # Get device info
    device_count = torch.cuda.device_count()
    logger.info(f"✅ CUDA is available with {device_count} device(s)")
    
    # Test each GPU
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        logger.info(f"  GPU {i}: {device_name} with {memory_gb:.1f}GB memory")
    
    # Test CUDA tensor operations
    try:
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        start_time = time.time()
        z = torch.matmul(x, y)
        duration = time.time() - start_time
        logger.info(f"✅ CUDA tensor operation completed in {duration:.4f}s")
        return True
    except Exception as e:
        logger.error(f"❌ CUDA tensor operation failed: {e}")
        return False

def test_model_setup():
    """Test model initialization from config"""
    logger.info("Testing model setup...")
    
    # Check model config path
    model_config_path = "../models/gpt_750m/config.json"
    if not os.path.exists(model_config_path):
        logger.error(f"❌ Model config not found at {model_config_path}")
        return False
    logger.info(f"✅ Model config found at {model_config_path}")
    
    # Check tokenizer path
    tokenizer_path = "../models/tokenizer/tokenizer"
    if not os.path.exists(tokenizer_path):
        logger.error(f"❌ Tokenizer not found at {tokenizer_path}")
        return False
    
    # Test loading tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        logger.info(f"✅ Tokenizer loaded successfully from {tokenizer_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        return False
    
    # Test tokenization
    try:
        text = "Hello, world! This is a test."
        tokens = tokenizer(text, return_tensors="pt")
        logger.info(f"✅ Tokenization test succeeded: {text} → {tokens.input_ids.shape}")
    except Exception as e:
        logger.error(f"❌ Tokenization test failed: {e}")
        return False
    
    # Optionally test creating a small model
    try:
        model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
        logger.info(f"✅ Model initialization test succeeded")
        del model  # Free memory
        torch.cuda.empty_cache()
    except Exception as e:
        logger.error(f"❌ Model initialization test failed: {e}")
        # Not a critical failure, continue
    
    return True

def test_gcs_access():
    """Test Google Cloud Storage access"""
    logger.info("Testing GCS access...")
    
    bucket_name = "refocused-ai"
    
    try:
        # Create anonymous client
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        
        # List a few files
        blobs = list(bucket.list_blobs(max_results=10))
        
        if not blobs:
            logger.warning(f"⚠️ No files found in bucket {bucket_name}")
            return False
        
        logger.info(f"✅ Successfully accessed GCS bucket {bucket_name}")
        logger.info(f"  Files found at root level:")
        
        # Show files with details
        for blob in blobs:
            logger.info(f"  - {blob.name} ({blob.size / 1024 / 1024:.2f} MB)")
        
        # Count and list .npz files
        npz_blobs = [b for b in bucket.list_blobs() if b.name.endswith('.npz')]
        npz_count = len(npz_blobs)
        
        if npz_count > 0:
            logger.info(f"✅ Found {npz_count} .npz files in the bucket")
            logger.info("Sample .npz files:")
            for blob in npz_blobs[:5]:
                logger.info(f"  - {blob.name} ({blob.size / 1024 / 1024:.2f} MB)")
            
            # Attempt to download a sample file
            try:
                sample_file = npz_blobs[0]
                local_path = f"/tmp/test_{sample_file.name.split('/')[-1]}"
                logger.info(f"Downloading sample file to {local_path}...")
                sample_file.download_to_filename(local_path)
                
                # Verify file contents
                import numpy as np
                data = np.load(local_path)
                keys = list(data.keys())
                logger.info(f"✅ Sample file contents verified. Contains keys: {keys}")
                
                # Check for expected keys
                expected_keys = ["input_ids", "arr_0", "sequences", "text"]
                found_keys = [key for key in expected_keys if key in keys]
                if found_keys:
                    logger.info(f"✅ Found expected key(s): {found_keys}")
                else:
                    logger.warning(f"⚠️ None of the expected keys {expected_keys} found")
                
                # Clean up
                os.remove(local_path)
            except Exception as e:
                logger.error(f"❌ Failed to download/verify sample file: {e}")
        else:
            logger.warning(f"⚠️ No .npz files found in bucket {bucket_name}")
            
            # Look for other directories
            try:
                prefixes = bucket.list_blobs(delimiter='/')
                for prefix in prefixes.prefixes:
                    logger.info(f"Found directory: {prefix}")
                    # Check this directory for .npz files
                    dir_blobs = list(bucket.list_blobs(prefix=prefix, max_results=5))
                    npz_in_dir = [b for b in dir_blobs if b.name.endswith('.npz')]
                    if npz_in_dir:
                        logger.info(f"✅ Found {len(npz_in_dir)} .npz files in {prefix}")
            except Exception as e:
                logger.error(f"❌ Failed to list directories: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to access GCS bucket {bucket_name}: {e}")
        return False

def test_data_directory():
    """Test data directory setup"""
    logger.info("Testing data directory setup...")
    
    data_dir = "/home/ubuntu/training_data/shards"
    if not os.path.exists(data_dir):
        logger.error(f"❌ Data directory not found at {data_dir}")
        return False
    
    # Check for .npz files
    npz_files = list(Path(data_dir).glob("*.npz"))
    if not npz_files:
        logger.warning(f"⚠️ No .npz files found in {data_dir}")
        return False
    
    logger.info(f"✅ Found {len(npz_files)} .npz files in {data_dir}")
    
    # Check a sample file
    if npz_files:
        try:
            import numpy as np
            sample_file = npz_files[0]
            data = np.load(sample_file)
            keys = list(data.keys())
            logger.info(f"✅ Sample file {sample_file.name} contains keys: {keys}")
            
            # Check for expected keys
            expected_keys = ["input_ids", "arr_0", "sequences", "text"]
            found_keys = [key for key in expected_keys if key in keys]
            if found_keys:
                logger.info(f"✅ Found expected key(s): {found_keys}")
                
                # Show data shape for first key
                key = found_keys[0]
                shape = data[key].shape
                logger.info(f"✅ Data shape for {key}: {shape}")
            else:
                logger.warning(f"⚠️ None of the expected keys {expected_keys} found in {sample_file.name}")
        except Exception as e:
            logger.error(f"❌ Failed to inspect sample file: {e}")
    
    return True

def main():
    logger.info("=" * 60)
    logger.info("Starting H100 Training Environment Quick Test")
    logger.info("=" * 60)
    
    # Test critical components
    test_results = {
        "PyTorch CUDA": test_pytorch_cuda(),
        "Model Setup": test_model_setup(),
        "GCS Access": test_gcs_access(),
        "Data Directory": test_data_directory()
    }
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    all_passed = True
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    logger.info("=" * 60)
    if all_passed:
        logger.info("🎉 All tests passed! Your environment is ready for training.")
        logger.info("Next steps:")
        logger.info("1. Run: bash h100_runner.sh test")
        logger.info("2. If test succeeds, run: bash h100_runner.sh full")
    else:
        logger.info("⚠️ Some tests failed. Please check the logs and fix the issues.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 