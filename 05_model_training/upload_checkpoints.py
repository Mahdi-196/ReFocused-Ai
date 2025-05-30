#!/usr/bin/env python3
"""
Checkpoint Upload Script for H100 Training
Uploads checkpoints to Google Cloud Storage with efficient parallel uploads
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def upload_file(bucket, local_file, remote_path):
    """Upload a single file to GCS"""
    try:
        local_file_path = Path(local_file)
        file_size = local_file_path.stat().st_size
        
        # Create blob name with relative path
        blob_name = f"{remote_path}/{local_file_path.name}"
        
        # Create blob and upload
        blob = bucket.blob(blob_name)
        
        # Check if blob already exists with same size
        try:
            blob.reload()
            if blob.size == file_size:
                logger.info(f"Skipping {local_file_path.name} (already exists with same size)")
                return True
        except Exception:
            # Blob doesn't exist or can't be accessed, continue with upload
            pass
        
        logger.info(f"Uploading {local_file_path.name} ({file_size / 1024 / 1024:.1f} MB)")
        blob.upload_from_filename(str(local_file_path))
        return True
    except Exception as e:
        logger.error(f"Failed to upload {local_file}: {e}")
        return False

def upload_checkpoint(checkpoint_dir, bucket_name, remote_path, max_workers=8):
    """Upload checkpoint directory to GCS with parallel uploads"""
    try:
        # Initialize GCS client
        client = storage.Client()
        logger.info(f"Connected to GCS with authenticated client")
        
        bucket = client.bucket(bucket_name)
        
        # Get all files in checkpoint directory
        checkpoint_dir_path = Path(checkpoint_dir)
        if not checkpoint_dir_path.exists():
            logger.error(f"Checkpoint directory {checkpoint_dir} does not exist")
            return False
        
        # Get checkpoint name (directory name)
        checkpoint_name = checkpoint_dir_path.name
        full_remote_path = f"{remote_path}/{checkpoint_name}"
        
        # Find all files recursively
        all_files = []
        for file_path in checkpoint_dir_path.rglob('*'):
            if file_path.is_file():
                all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} files to upload in {checkpoint_dir}")
        
        # Upload files in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda file_path: upload_file(
                    bucket, 
                    file_path, 
                    full_remote_path + "/" + str(file_path.relative_to(checkpoint_dir_path).parent)
                ), 
                all_files
            ))
        
        success_count = sum(results)
        duration = time.time() - start_time
        
        logger.info(f"Uploaded {success_count}/{len(all_files)} files in {duration:.1f}s")
        total_size = sum(f.stat().st_size for f in all_files)
        logger.info(f"Total size: {total_size / 1024 / 1024:.1f} MB")
        logger.info(f"Average speed: {total_size / (1024 * 1024 * duration):.2f} MB/s")
        
        return success_count == len(all_files)
    
    except Exception as e:
        logger.error(f"Error uploading checkpoint: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoints to GCS")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Local checkpoint directory to upload")
    parser.add_argument("--bucket", type=str, default="refocused-ai",
                       help="GCS bucket name")
    parser.add_argument("--remote_path", type=str, default="checkpoints",
                       help="Remote path in GCS bucket")
    parser.add_argument("--workers", type=int, default=8,
                       help="Number of parallel upload workers")
    
    args = parser.parse_args()
    
    logger.info(f"Starting upload from {args.checkpoint_dir} to gs://{args.bucket}/{args.remote_path}")
    
    success = upload_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        bucket_name=args.bucket,
        remote_path=args.remote_path,
        max_workers=args.workers
    )
    
    if success:
        logger.info("Upload completed successfully")
        return 0
    else:
        logger.error("Upload failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 