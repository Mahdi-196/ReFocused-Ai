"""
Utility script to inspect NPZ file structure in the Google Cloud bucket
"""

import os
import sys
import numpy as np
from google.cloud import storage
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def inspect_npz_file(file_path):
    """Inspect the structure of an NPZ file"""
    print(f"Inspecting file: {file_path}")
    
    data = np.load(file_path)
    print(f"Keys in NPZ file: {list(data.keys())}")
    
    for key in data.keys():
        array = data[key]
        print(f"  - {key}: shape={array.shape}, dtype={array.dtype}, ndim={array.ndim}")
        
        # Print a sample of the data
        if array.ndim == 1:
            print(f"    Sample: {array[:10]}...")
        elif array.ndim == 2:
            print(f"    First row: {array[0, :10]}...")
            if array.shape[0] > 1:
                print(f"    Second row: {array[1, :10]}...")
        elif array.ndim == 3:
            print(f"    First element: {array[0, 0, :10]}...")
            if array.shape[0] > 1 and array.shape[1] > 1:
                print(f"    Another element: {array[1, 1, :10]}...")
    
    return data


def download_and_inspect_from_gcs(bucket_name, blob_name, cache_dir="./cache"):
    """Download file from GCS and inspect it"""
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, os.path.basename(blob_name))
    
    # Download file if not exists
    if not os.path.exists(local_path):
        print(f"Downloading {blob_name} from bucket {bucket_name}...")
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
    
    # Inspect file
    return inspect_npz_file(local_path)


def list_npz_files(bucket_name, prefix="", max_files=5):
    """List NPZ files in the bucket"""
    print(f"Listing NPZ files in bucket {bucket_name} with prefix '{prefix}'...")
    
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    
    files = []
    for blob in blobs:
        if blob.name.endswith('.npz') and 'tokenized_' in blob.name:
            files.append(blob.name)
            if len(files) >= max_files:
                break
    
    print(f"Found {len(files)} NPZ files")
    for i, file in enumerate(files):
        print(f"{i+1}. {file}")
    
    return files


def main():
    parser = argparse.ArgumentParser(description="Inspect NPZ files in GCS bucket")
    parser.add_argument("--bucket", type=str, default="refocused-ai",
                        help="GCS bucket name")
    parser.add_argument("--prefix", type=str, default="",
                        help="Prefix for blob listing")
    parser.add_argument("--file", type=str, default=None,
                        help="Specific file to inspect")
    parser.add_argument("--max_files", type=int, default=5,
                        help="Maximum number of files to list")
    args = parser.parse_args()
    
    # If file is specified, inspect it
    if args.file:
        download_and_inspect_from_gcs(args.bucket, args.file)
    else:
        # List files
        files = list_npz_files(args.bucket, args.prefix, args.max_files)
        
        # Ask user to select a file
        if files:
            selection = input("\nEnter file number to inspect (or press Enter to exit): ")
            if selection and selection.isdigit() and 1 <= int(selection) <= len(files):
                file_idx = int(selection) - 1
                download_and_inspect_from_gcs(args.bucket, files[file_idx])


if __name__ == "__main__":
    main() 