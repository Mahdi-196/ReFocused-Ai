import os
from pathlib import Path
import time
from google.cloud import storage
import json

# Set environment variables for Google Cloud authentication
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials/black-dragon-461023-t5-93452a49f86b.json'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'black-dragon-461023-t5'

def download_refocused_data():
    """Download all tokenized training data from refocused-ai bucket"""
    
    bucket_name = "refocused-ai"
    
    # Create local directories
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 DOWNLOADING REFOCUSED-AI TRAINING DATA")
    print("=" * 50)
    print(f"Target directory: {data_dir.absolute()}")
    
    try:
        # Initialize Google Cloud Storage client with credentials
        print("🔐 Authenticating with Google Cloud Storage...")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Get list of all files
        print("📋 Fetching file list...")
        blobs = list(bucket.list_blobs())
        
        if not blobs:
            print("❌ No files found in bucket")
            return False
            
        print(f"✅ Found {len(blobs)} files")
        
        # Filter for .npz files only
        npz_blobs = [blob for blob in blobs if blob.name.endswith('.npz')]
        print(f"🎯 {len(npz_blobs)} tokenized .npz files to download")
        
        # Calculate total size
        total_size = sum(blob.size for blob in npz_blobs)
        total_mb = total_size / (1024 * 1024)
        print(f"📊 Total download size: {total_mb:.1f} MB")
        
        # Download all files
        print("\n🔥 Starting downloads...")
        start_time = time.time()
        downloaded_count = 0
        downloaded_bytes = 0
        
        for i, blob in enumerate(npz_blobs, 1):
            # 1. Build precise local path (preserving any sub-folders)
            relative_path = Path(blob.name)
            local_path = data_dir / relative_path
            file_size = blob.size
            
            # Create any intermediate folders (e.g. data/training/subdir1/)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 2. Skip if already downloaded
            if local_path.exists():
                print(f"[{i:3d}/{len(npz_blobs)}] ⏭️  Skipping (already exists): {relative_path}")
                # Still count as "downloaded" for progress tracking
                downloaded_count += 1
                downloaded_bytes += file_size
                continue
            
            # 3. Otherwise download into that same nested path
            print(f"[{i:3d}/{len(npz_blobs)}] ⬇️  Downloading {relative_path} ({file_size/1024/1024:.1f} MB)")
            
            try:
                # Download file using Google Cloud Storage client
                blob.download_to_filename(str(local_path))
                
                downloaded_count += 1
                downloaded_bytes += file_size
                
                # Progress update every 10 files
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    speed_mbps = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                    print(f"    📈 Progress: {i}/{len(npz_blobs)} files, {speed_mbps:.1f} MB/s")
                
            except Exception as e:
                print(f"    ❌ Failed to download {relative_path}: {e}")
                continue
        
        # Final summary
        elapsed = time.time() - start_time
        avg_speed = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        
        print("\n🎉 DOWNLOAD COMPLETE!")
        print("=" * 50)
        print(f"✅ Downloaded: {downloaded_count}/{len(npz_blobs)} files")
        print(f"📊 Total size: {downloaded_bytes/1024/1024:.1f} MB")
        print(f"⏱️  Time taken: {elapsed:.1f} seconds")
        print(f"🚀 Average speed: {avg_speed:.1f} MB/s")
        print(f"📁 Files saved to: {data_dir.absolute()}")
        
        # Verify downloads (including nested folders)
        print("\n🔍 Verifying downloads...")
        local_files = list(data_dir.glob("**/*.npz"))  # Recursive search
        print(f"✅ {len(local_files)} .npz files found in directory tree")
        
        if len(local_files) == len(npz_blobs):
            print("🎯 All files downloaded successfully!")
            return True
        else:
            print(f"⚠️  Expected {len(npz_blobs)}, got {len(local_files)} files")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print(f"Make sure GOOGLE_APPLICATION_CREDENTIALS is set correctly")
        return False

def create_data_info():
    """Create info file about the downloaded data"""
    data_dir = Path("data/training")
    npz_files = list(data_dir.glob("**/*.npz"))  # Recursive search for nested folders
    
    if not npz_files:
        print("No .npz files found to analyze")
        return
    
    # Analyze file types
    comments_files = [f for f in npz_files if 'comments' in f.name]
    submissions_files = [f for f in npz_files if 'submissions' in f.name]
    
    # Get subreddits
    subreddits = set()
    for f in npz_files:
        parts = f.name.split('_')
        if len(parts) >= 3:
            subreddit = parts[2]  # tokenized_cleaned_SUBREDDIT_...
            subreddits.add(subreddit)
    
    info = {
        "total_files": len(npz_files),
        "comments_files": len(comments_files),
        "submissions_files": len(submissions_files),
        "unique_subreddits": len(subreddits),
        "subreddits": sorted(list(subreddits)),
        "file_format": "npz (tokenized numpy arrays)",
        "download_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ready_for_training": True
    }
    
    # Save info
    info_file = data_dir / "data_info.json"
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n📋 DATA SUMMARY:")
    print(f"   Total files: {info['total_files']}")
    print(f"   Comments: {info['comments_files']}")
    print(f"   Submissions: {info['submissions_files']}")
    print(f"   Subreddits: {info['unique_subreddits']}")
    print(f"   Info saved: {info_file}")
    
    return info

if __name__ == "__main__":
    success = download_refocused_data()
    if success:
        create_data_info()
        print("\n🚀 READY FOR TRAINING!")
        print("Run your training script with: --data-path data/training")
    else:
        print("\n❌ Download failed. Check errors above.") 