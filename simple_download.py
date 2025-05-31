#!/usr/bin/env python3
"""
Simple script to download files from GCS
"""

import os
import sys
import traceback
from pathlib import Path
from google.cloud import storage

def main():
    # Configuration
    bucket_name = "refocused-ai"
    output_dir = "data/training/simple"
    max_files = 3
    
    print(f"Downloading {max_files} files from gs://{bucket_name}/")
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        print(f"Using output directory: {os.path.abspath(output_dir)}")
        
        # Create anonymous client
        print("Creating anonymous storage client...")
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        print(f"Connected to bucket: {bucket_name}")
        
        # List files in bucket
        print("Listing files in bucket...")
        try:
            blobs = list(bucket.list_blobs(max_results=100))
            print(f"Found {len(blobs)} total files in the bucket")
            
            if not blobs:
                print("Bucket appears to be empty or inaccessible")
                return 1
                
            # Print first few files
            print("Sample files in bucket:")
            for blob in blobs[:5]:
                print(f"  - {blob.name} ({blob.size / 1024 / 1024:.2f} MB)")
                
        except Exception as e:
            print(f"Error listing files in bucket: {e}")
            traceback.print_exc()
            return 1
        
        # Filter for .npz files
        npz_blobs = [b for b in blobs if b.name.endswith('.npz')]
        
        if not npz_blobs:
            print("No .npz files found in bucket")
            return 1
        
        print(f"Found {len(npz_blobs)} .npz files")
        
        # Take first few files
        download_blobs = npz_blobs[:max_files]
        
        # Download each file
        for blob in download_blobs:
            # Extract filename
            filename = os.path.basename(blob.name)
            local_path = os.path.join(output_dir, filename)
            
            print(f"Downloading {blob.name} to {local_path}...")
            
            try:
                # Download file
                blob.download_to_filename(local_path)
                
                # Verify file was downloaded
                if os.path.exists(local_path):
                    file_size = os.path.getsize(local_path) / (1024 * 1024)
                    print(f"  Downloaded successfully: {file_size:.2f} MB")
                else:
                    print(f"  Failed to download file - file doesn't exist after download")
            except Exception as e:
                print(f"  Error downloading {blob.name}: {e}")
                traceback.print_exc()
        
        # List downloaded files
        print("\nDownloaded files:")
        downloaded_files = [f for f in os.listdir(output_dir) if f.endswith('.npz')]
        
        if not downloaded_files:
            print("No files were downloaded successfully")
            return 1
            
        for file in downloaded_files:
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {file} - {file_size:.2f} MB")
        
        print(f"\nDownload completed successfully: {len(downloaded_files)} files")
        return 0
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 