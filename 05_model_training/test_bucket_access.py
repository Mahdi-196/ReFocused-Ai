#!/usr/bin/env python3
"""
Test script for Google Cloud Storage bucket access
Verifies connection to bucket and lists available files
"""

import os
import sys
import argparse
import logging
import tempfile
from pathlib import Path
from google.cloud import storage
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def test_bucket_access(bucket_name, remote_path=None, download_sample=False):
    """Test access to a GCS bucket and list contents"""
    try:
        logger.info(f"Testing access to bucket: {bucket_name}")
        
        # Create anonymous client for public bucket
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        
        # Test basic connection
        try:
            # Just list a few files to confirm bucket exists
            test_blobs = list(bucket.list_blobs(max_results=5))
            if not test_blobs:
                logger.warning(f"⚠️ Bucket {bucket_name} exists but appears empty")
            else:
                logger.info(f"✅ Successfully connected to bucket {bucket_name}")
        except Exception as e:
            logger.error(f"❌ Error accessing bucket {bucket_name}: {e}")
            return False
        
        # List files at root level
        logger.info(f"Files at root level of gs://{bucket_name}/:")
        root_blobs = list(bucket.list_blobs(max_results=20))
        
        if not root_blobs:
            logger.warning("No files found at root level")
        else:
            for blob in root_blobs:
                logger.info(f" - {blob.name} ({blob.size / 1024 / 1024:.2f} MB)")
        
        # List directories (prefixes)
        logger.info("Checking for directories (prefixes):")
        try:
            prefixes = list(bucket.list_blobs(delimiter='/', max_results=20).prefixes)
            if prefixes:
                for prefix in prefixes:
                    logger.info(f" - {prefix}")
            else:
                logger.info("No directories found (all files at root level)")
        except Exception as e:
            logger.error(f"Failed to list directories: {e}")
        
        # Count and list .npz files
        if remote_path:
            logger.info(f"Checking for .npz files in gs://{bucket_name}/{remote_path}")
            npz_blobs = [b for b in bucket.list_blobs(prefix=remote_path) if b.name.endswith('.npz')]
        else:
            logger.info(f"Checking for .npz files in entire bucket")
            npz_blobs = [b for b in bucket.list_blobs() if b.name.endswith('.npz')]
        
        if not npz_blobs:
            logger.warning(f"⚠️ No .npz files found in specified location")
            return False
        
        logger.info(f"✅ Found {len(npz_blobs)} .npz files")
        logger.info("Sample .npz files:")
        for blob in npz_blobs[:5]:
            logger.info(f" - {blob.name} ({blob.size / 1024 / 1024:.2f} MB)")
        
        # Download and inspect a sample file
        if download_sample and npz_blobs:
            sample_blob = npz_blobs[0]
            # Use a temporary file with proper extension
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp_file:
                sample_path = tmp_file.name
            
            logger.info(f"Downloading sample file: {sample_blob.name} to {sample_path}")
            sample_blob.download_to_filename(sample_path)
            
            try:
                data = np.load(sample_path)
                keys = list(data.keys())
                logger.info(f"✅ Sample file loaded successfully. Contains keys: {keys}")
                
                # Check data shape for the first key
                if keys:
                    key = keys[0]
                    shape = data[key].shape
                    logger.info(f"Data shape for '{key}': {shape}")
                
                # Check for expected keys
                expected_keys = ["input_ids", "arr_0", "sequences", "text"]
                found_keys = [key for key in expected_keys if key in keys]
                if found_keys:
                    logger.info(f"✅ Found expected key(s): {found_keys}")
                else:
                    logger.warning(f"⚠️ None of the expected keys {expected_keys} found")
                
                # Clean up
                try:
                    os.unlink(sample_path)  # safer than os.remove on Windows
                    logger.info(f"Removed temporary file: {sample_path}")
                except Exception as e:
                    logger.warning(f"Could not remove temporary file: {e}")
            except Exception as e:
                logger.error(f"❌ Failed to load sample file: {e}")
                return False
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Error testing bucket access: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test GCS bucket access")
    parser.add_argument("--bucket", type=str, default="refocused-ai",
                      help="GCS bucket name")
    parser.add_argument("--remote_path", type=str, default="",
                      help="Remote path within bucket (leave empty for bucket root)")
    parser.add_argument("--download", action="store_true",
                      help="Download and inspect a sample file")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Google Cloud Storage Bucket Access Test")
    logger.info("=" * 60)
    
    success = test_bucket_access(
        bucket_name=args.bucket,
        remote_path=args.remote_path,
        download_sample=args.download
    )
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ Bucket access test PASSED")
        logger.info("=" * 60)
        return 0
    else:
        logger.info("=" * 60)
        logger.info("❌ Bucket access test FAILED")
        logger.info("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 