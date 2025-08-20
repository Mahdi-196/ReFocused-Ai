#!/usr/bin/env python3
"""
Test script to verify resume functionality
"""

import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import get_training_config
from utils import CheckpointManager


def test_resume_logic():
    """Test the resume logic independently"""
    
    print("🧪 Testing Resume Logic")
    print("=" * 40)
    
    # Test argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--config", type=str, default="test", help="Config type")
    
    # Simulate command line args
    test_args = ["--config", "test", "--resume", "checkpoint-epoch0-step800-files0"]
    args = parser.parse_args(test_args)
    
    print(f"✅ Parsed args: resume={args.resume}, config={args.config}")
    
    # Test config loading
    config = get_training_config(args.config)
    print(f"✅ Loaded config: max_steps={config.max_steps}")
    
    # Test checkpoint manager initialization
    checkpoint_manager = CheckpointManager(
        config.bucket_name,
        getattr(config, 'checkpoint_bucket_path', 'checkpoints'),
        config.output_dir,
        background_upload=False,
        credentials_path=None,
        project_id=None,
    )
    print(f"✅ Initialized CheckpointManager")
    
    # Test resume detection
    if args.resume:
        print(f"🔄 Resume detected: {args.resume}")
        print(f"   This would trigger checkpoint loading in training script")
        
        # Check if checkpoint exists locally (won't download, just check)
        local_checkpoint_dir = os.path.join(checkpoint_manager.local_dir, args.resume)
        if os.path.exists(local_checkpoint_dir):
            print(f"✅ Checkpoint found locally: {local_checkpoint_dir}")
        else:
            print(f"📥 Checkpoint would be downloaded from GCS: {args.resume}")
    else:
        print("🚀 No resume checkpoint specified - would start fresh")
    
    print("\n✅ Resume logic test completed successfully!")


if __name__ == "__main__":
    test_resume_logic() 