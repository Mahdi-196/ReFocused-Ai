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
        # Extract just the filename without any path
        filename = os.path.basename(blob.name)
        local_file_path = Path(local_dir) / filename
        
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

def list_bucket_prefixes(bucket, delimiter='/'):
    """List all top-level prefixes (folders) in a bucket"""
    prefixes = bucket.list_blobs(delimiter=delimiter)
    return [prefix.prefix for prefix in prefixes.prefixes]

def list_all_blobs(bucket, max_results=100):
    """List all blobs in the bucket without any prefix filtering"""
    all_blobs = list(bucket.list_blobs(max_results=max_results))
    return all_blobs

def list_all_files(bucket, max_files=50):
    """List all files in the bucket with details"""
    all_files = []
    all_npz_files = []
    
    for blob in bucket.list_blobs():
        file_info = {
            'name': blob.name,
            'size_mb': blob.size / (1024 * 1024),
            'updated': blob.updated
        }
        all_files.append(file_info)
        
        if blob.name.endswith('.npz'):
            all_npz_files.append(file_info)
            
        if len(all_files) >= max_files:
            break
    
    return all_files, all_npz_files

def download_data(bucket_name, remote_path, local_dir, max_workers=8, max_files=None, anonymous=True, list_only=False):
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
        
        # First, check if bucket exists by listing all blobs
        logger.info(f"Testing connection to bucket: {bucket_name}")
        try:
            # Just get a few blobs to confirm bucket exists and is accessible
            test_blobs = list(bucket.list_blobs(max_results=5))
            if not test_blobs:
                logger.warning(f"Bucket {bucket_name} appears to be empty or not accessible")
            else:
                logger.info(f"Successfully connected to bucket {bucket_name}")
                logger.info(f"First few files: {[b.name for b in test_blobs]}")
        except Exception as e:
            logger.error(f"Error accessing bucket {bucket_name}: {e}")
            return False

        # List all files (with limit) and NPZ files specifically
        logger.info(f"Scanning for all files and .npz files in the bucket...")
        all_files, all_npz_files = list_all_files(bucket, max_files=500)
        
        logger.info(f"Found {len(all_files)} total files in bucket, {len(all_npz_files)} are .npz files")
        
        if all_npz_files:
            logger.info("Sample .npz files:")
            for file_info in all_npz_files[:5]:
                logger.info(f" - {file_info['name']} ({file_info['size_mb']:.2f} MB)")
            
        # If remote_path is specified, filter by it
        npz_blobs = []
        if remote_path:
            logger.info(f"Filtering by remote path: gs://{bucket_name}/{remote_path}")
            prefix_str = remote_path if remote_path else "(root)"
            blobs = list(bucket.list_blobs(prefix=remote_path))
            logger.info(f"Found {len(blobs)} total files in gs://{bucket_name}/{prefix_str}")
            npz_blobs = [b for b in blobs if b.name.endswith('.npz')]
        else:
            # If no remote_path, use all .npz files
            logger.info(f"No remote path specified, using all .npz files in the bucket")
            npz_blobs = [bucket.blob(f['name']) for f in all_npz_files]
        
        if not npz_blobs:
            logger.warning(f"No .npz files found matching the criteria")
            # Try to get some sample file names to help diagnose the issue
            if all_files:
                logger.info(f"Sample non-npz files found: {[f['name'] for f in all_files[:5]]}")
            
            # Suggest trying without a path prefix
            if remote_path:
                logger.info("Try downloading without a remote_path (root of bucket)")
                
            return False
        
        if max_files and npz_blobs:
            npz_blobs = npz_blobs[:max_files]
            logger.info(f"Limited to {max_files} files")
        
        logger.info(f"Found {len(npz_blobs)} .npz files to download")
        
        if list_only:
            logger.info("List-only mode. Skipping download.")
            return True
            
        if not npz_blobs:
            return False
        
        # Download files in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(
                lambda blob: download_blob(bucket, blob, local_dir), 
                npz_blobs
            ))
        
        success_count = sum(results)
        duration = time.time() - start_time
        
        if success_count > 0:
            logger.info(f"Downloaded {success_count}/{len(npz_blobs)} files in {duration:.1f}s")
            total_size_mb = sum([b.size for b in npz_blobs]) / (1024 * 1024)
            logger.info(f"Total data downloaded: {total_size_mb:.2f} MB")
            logger.info(f"Average speed: {total_size_mb / duration:.2f} MB/s")
            
            # Verify a sample file if available
            try:
                import numpy as np
                sample_files = list(Path(local_dir).glob("*.npz"))
                if sample_files:
                    sample_file = sample_files[0]
                    data = np.load(sample_file)
                    keys = list(data.keys())
                    logger.info(f"Sample file {sample_file.name} contains keys: {keys}")
            except Exception as e:
                logger.warning(f"Could not verify sample file contents: {e}")
        
        return success_count > 0
    
    except Exception as e:
        logger.error(f"Error downloading data: {e}")
        logger.error("Please ensure the bucket exists and is publicly accessible")
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
    parser.add_argument("--list-only", action="store_true",
                      help="Just list files without downloading")
    parser.add_argument("--scan-bucket", action="store_true",
                      help="Scan entire bucket to find where data is stored")
    
    args = parser.parse_args()
    
    logger.info(f"Starting download from gs://{args.bucket}/{args.remote_path} to {args.local_dir}")
    
    success = download_data(
        bucket_name=args.bucket,
        remote_path=args.remote_path,
        local_dir=args.local_dir,
        max_workers=args.workers,
        max_files=args.max_files,
        anonymous=args.anonymous,
        list_only=args.list_only
    )
    
    if success:
        logger.info("Download completed successfully")
        return 0
    else:
        logger.error("Download failed - no files were downloaded")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 