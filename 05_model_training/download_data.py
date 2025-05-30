#!/usr/bin/env python3
"""
Data Download Script for H100 Training
Downloads training data from Google Cloud Storage with efficient parallel downloads
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

def download_blob(bucket, blob, local_dir):
    """Download a single blob from GCS to local storage"""
    try:
        # Get relative path from remote_path
        local_file_path = Path(local_dir) / blob.name.split("/")[-1]
        
        # Skip if file exists and has same size
        if local_file_path.exists():
            if local_file_path.stat().st_size == blob.size:
                logger.info(f"Skipping {blob.name} (already exists with same size)")
                return True
        
        # Download file
        logger.info(f"Downloading {blob.name} ({blob.size / 1024 / 1024:.1f} MB)")
        blob.download_to_filename(str(local_file_path))
        return True
    except Exception as e:
        logger.error(f"Failed to download {blob.name}: {e}")
        return False

def download_data(bucket_name, remote_path, local_dir, max_workers=8, max_files=None, anonymous=True):
    """Download training data from GCS with parallel downloads"""
    try:
        # Create local directory
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS client
        if anonymous:
            client = storage.Client.create_anonymous_client()
            logger.info(f"Using anonymous GCS client for bucket: {bucket_name}")
        else:
            client = storage.Client()
            logger.info(f"Using authenticated GCS client for bucket: {bucket_name}")
        
        bucket = client.bucket(bucket_name)
        
        # List blobs in the bucket with the given prefix
        logger.info(f"Listing files in gs://{bucket_name}/{remote_path}")
        blobs = list(bucket.list_blobs(prefix=remote_path))
        
        # Filter for .npz files
        npz_blobs = [b for b in blobs if b.name.endswith('.npz')]
        
        if max_files:
            npz_blobs = npz_blobs[:max_files]
            logger.info(f"Limited to {max_files} files")
        
        logger.info(f"Found {len(npz_blobs)} .npz files to download")
        
        # Download files in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda blob: download_blob(bucket, blob, local_dir), 
                npz_blobs
            ))
        
        success_count = sum(results)
        duration = time.time() - start_time
        
        logger.info(f"Downloaded {success_count}/{len(npz_blobs)} files in {duration:.1f}s")
        logger.info(f"Average speed: {sum([b.size for b in npz_blobs]) / (1024 * 1024 * duration):.2f} MB/s")
        
        return success_count == len(npz_blobs)
    
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download training data from GCS")
    parser.add_argument("--bucket", type=str, default="refocused-ai",
                      help="GCS bucket name")
    parser.add_argument("--remote_path", type=str, default="tokenized_data",
                      help="Remote path in GCS bucket")
    parser.add_argument("--local_dir", type=str, default="/home/ubuntu/training_data/shards",
                      help="Local directory to save data")
    parser.add_argument("--workers", type=int, default=8,
                      help="Number of parallel download workers")
    parser.add_argument("--max_files", type=int, default=None,
                      help="Maximum number of files to download (None for all)")
    parser.add_argument("--anonymous", action="store_true", default=True,
                      help="Use anonymous access (for public buckets)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting download from gs://{args.bucket}/{args.remote_path} to {args.local_dir}")
    
    success = download_data(
        bucket_name=args.bucket,
        remote_path=args.remote_path,
        local_dir=args.local_dir,
        max_workers=args.workers,
        max_files=args.max_files,
        anonymous=args.anonymous
    )
    
    if success:
        logger.info("Download completed successfully")
        return 0
    else:
        logger.error("Download failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 