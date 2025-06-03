#!/usr/bin/env python3
"""
Upload manager utility for handling checkpoint uploads to GCS
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs import get_training_config


def upload_checkpoint_manual(checkpoint_dir: str, bucket_name: str, bucket_path: str, checkpoint_name: str = None):
    """Manually upload a checkpoint directory to GCS"""
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False
    
    if not checkpoint_name:
        checkpoint_name = os.path.basename(checkpoint_dir)
    
    print(f"📦 Uploading {checkpoint_dir} to GCS...")
    
    # Create tar.gz archive
    tar_path = f"{checkpoint_dir}.tar.gz"
    print(f"Creating archive: {tar_path}")
    
    result = subprocess.run([
        "tar", "czf", tar_path, 
        "-C", os.path.dirname(checkpoint_dir), 
        os.path.basename(checkpoint_dir)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to create tar archive: {result.stderr}")
        return False
    
    # Upload using gsutil
    bucket_uri = f"gs://{bucket_name}/{bucket_path}/{checkpoint_name}.tar.gz"
    print(f"☁️  Uploading to {bucket_uri}")
    
    result = subprocess.run([
        "gsutil", "-m", "cp", tar_path, bucket_uri
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Successfully uploaded {checkpoint_name}")
        # Clean up tar file
        os.remove(tar_path)
        print(f"🗑️  Cleaned up {tar_path}")
        return True
    else:
        print(f"❌ Upload failed: {result.stderr}")
        return False


def list_checkpoints(directory: str = "./checkpoints"):
    """List available local checkpoints"""
    if not os.path.exists(directory):
        print(f"❌ Checkpoint directory not found: {directory}")
        return
    
    checkpoints = [
        d for d in os.listdir(directory) 
        if os.path.isdir(os.path.join(directory, d)) and d.startswith('checkpoint-')
    ]
    
    if not checkpoints:
        print("No checkpoints found")
        return
    
    print(f"📁 Found {len(checkpoints)} checkpoints in {directory}:")
    for i, checkpoint in enumerate(sorted(checkpoints), 1):
        checkpoint_path = os.path.join(directory, checkpoint)
        size = get_directory_size(checkpoint_path)
        print(f"  {i}. {checkpoint} ({size:.1f} MB)")


def get_directory_size(directory: str) -> float:
    """Get directory size in MB"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total_size += os.path.getsize(fp)
    return total_size / (1024 * 1024)


def check_upload_status(bucket_name: str, bucket_path: str):
    """Check what checkpoints are uploaded to GCS"""
    print(f"☁️  Checking uploads in gs://{bucket_name}/{bucket_path}/")
    
    result = subprocess.run([
        "gsutil", "ls", f"gs://{bucket_name}/{bucket_path}/"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to list GCS objects: {result.stderr}")
        return
    
    if not result.stdout.strip():
        print("No checkpoints found in GCS")
        return
    
    files = result.stdout.strip().split('\n')
    tar_files = [f for f in files if f.endswith('.tar.gz')]
    dir_files = [f for f in files if not f.endswith('.tar.gz') and f.endswith('/')]
    
    print(f"📦 Found {len(tar_files)} tar.gz uploads:")
    for tar_file in tar_files:
        name = os.path.basename(tar_file).replace('.tar.gz', '')
        print(f"  ✅ {name}")
    
    if dir_files:
        print(f"📁 Found {len(dir_files)} directory uploads:")
        for dir_file in dir_files:
            name = os.path.basename(dir_file.rstrip('/'))
            print(f"  📂 {name}")


def upload_all_checkpoints(config_name: str = "test", checkpoint_dir: str = "./checkpoints"):
    """Upload all local checkpoints to GCS"""
    config = get_training_config(config_name)
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = [
        d for d in os.listdir(checkpoint_dir) 
        if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('checkpoint-')
    ]
    
    if not checkpoints:
        print("No checkpoints found to upload")
        return
    
    print(f"🚀 Uploading {len(checkpoints)} checkpoints...")
    success_count = 0
    
    for checkpoint in sorted(checkpoints):
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
        if upload_checkpoint_manual(
            checkpoint_path, 
            config.bucket_name, 
            config.checkpoint_bucket_path, 
            checkpoint
        ):
            success_count += 1
    
    print(f"✅ Successfully uploaded {success_count}/{len(checkpoints)} checkpoints")


def main():
    parser = argparse.ArgumentParser(description="Manage checkpoint uploads to GCS")
    parser.add_argument("--config", type=str, choices=["test", "production"], 
                       default="test", help="Training configuration")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Local checkpoint directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List local checkpoints")
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload specific checkpoint")
    upload_parser.add_argument("checkpoint_name", help="Checkpoint directory name")
    
    # Upload all command
    upload_all_parser = subparsers.add_parser("upload-all", help="Upload all local checkpoints")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check upload status in GCS")
    
    args = parser.parse_args()
    
    if args.command == "list":
        list_checkpoints(args.checkpoint_dir)
        
    elif args.command == "upload":
        config = get_training_config(args.config)
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
        upload_checkpoint_manual(
            checkpoint_path, 
            config.bucket_name, 
            config.checkpoint_bucket_path, 
            args.checkpoint_name
        )
        
    elif args.command == "upload-all":
        upload_all_checkpoints(args.config, args.checkpoint_dir)
        
    elif args.command == "status":
        config = get_training_config(args.config)
        check_upload_status(config.bucket_name, config.checkpoint_bucket_path)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 