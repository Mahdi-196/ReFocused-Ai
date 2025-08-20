import os
from pathlib import Path
import time
from google.cloud import storage
from google.oauth2 import service_account
import json
import numpy as np  # for integrity check

def get_storage_client(credentials_path: str | None = None, project_id: str | None = None) -> storage.Client:
    """Construct a GCS client without relying on environment variables."""
    if credentials_path and os.path.exists(credentials_path):
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        return storage.Client(project=project_id, credentials=creds)
    return storage.Client()

def is_npz_intact(local_path: Path) -> bool:
    """
    Try to load the file with numpy to confirm it's a valid .npz.
    Returns True if it opens without error, False otherwise.
    """
    try:
        with np.load(str(local_path)) as f:
            # just trying to open is enough; close immediately
            _ = f.keys()
        return True
    except Exception:
        return False

def download_refocused_data(credentials_path: str | None = None, project_id: str | None = None):
    """Download all tokenized training data from refocused-ai bucket,
       re-downloading only if the local copy is missing or invalid."""
    
    bucket_name = "refocused-ai"
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 DOWNLOADING REFOCUSED-AI TRAINING DATA")
    print("=" * 50)
    print(f"Target directory: {data_dir.absolute()}")
    
    try:
        print("🔐 Authenticating with Google Cloud Storage...")
        client = get_storage_client(credentials_path, project_id)
        bucket = client.bucket(bucket_name)
        
        print("📋 Fetching file list...")
        blobs = list(bucket.list_blobs())
        if not blobs:
            print("❌ No files found in bucket")
            return False
        
        npz_blobs = [blob for blob in blobs if blob.name.endswith('.npz')]
        print(f"✅ Found {len(npz_blobs)} tokenized .npz files")
        
        total_size = sum(blob.size for blob in npz_blobs)
        total_mb = total_size / (1024 * 1024)
        print(f"📊 Total download size: {total_mb:.1f} MB")
        
        print("\n🔥 Starting downloads...")
        start_time = time.time()
        downloaded_count = 0
        downloaded_bytes = 0
        
        for i, blob in enumerate(npz_blobs, 1):
            relative_path = Path(blob.name)
            local_path = data_dir / relative_path
            file_size = blob.size
            
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            # If file exists, check integrity
            if local_path.exists():
                if is_npz_intact(local_path):
                    print(f"[{i:3d}/{len(npz_blobs)}] ⏭️  Skipping (intact): {relative_path}")
                    downloaded_count += 1
                    downloaded_bytes += file_size
                    continue
                else:
                    # corrupted: delete and re-download
                    print(f"[{i:3d}/{len(npz_blobs)}] ⚠️  Corrupted detected, re-downloading: {relative_path}")
                    local_path.unlink()  # remove the bad file
            
            # Download fresh copy
            print(f"[{i:3d}/{len(npz_blobs)}] ⬇️  Downloading {relative_path} ({file_size/1024/1024:.1f} MB)")
            try:
                blob.download_to_filename(str(local_path))
                downloaded_count += 1
                downloaded_bytes += file_size
            except Exception as e:
                print(f"    ❌ Failed to download {relative_path}: {e}")
                continue
            
            if i % 10 == 0:
                elapsed = time.time() - start_time
                speed_mbps = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                print(f"    📈 Progress: {i}/{len(npz_blobs)} files, {speed_mbps:.1f} MB/s")
        
        elapsed = time.time() - start_time
        avg_speed = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        
        print("\n🎉 DOWNLOAD COMPLETE!")
        print("=" * 50)
        print(f"✅ Downloaded: {downloaded_count}/{len(npz_blobs)} files")
        print(f"📊 Total size: {downloaded_bytes/1024/1024:.1f} MB")
        print(f"⏱️  Time taken: {elapsed:.1f} seconds")
        print(f"🚀 Average speed: {avg_speed:.1f} MB/s")
        print(f"📁 Files saved to: {data_dir.absolute()}")
        
        print("\n🔍 Verifying downloads...")
        local_files = list(data_dir.glob("**/*.npz"))
        print(f"✅ {len(local_files)} .npz files found in directory tree")
        
        if len(local_files) == len(npz_blobs):
            print("🎯 All files downloaded successfully!")
            return True
        else:
            print(f"⚠️  Expected {len(npz_blobs)}, got {len(local_files)} files")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        print("Check your provided GCS credentials path and project ID.")
        return False

def create_data_info():
    """Create info file about the downloaded data."""
    data_dir = Path("data/training")
    npz_files = list(data_dir.glob("**/*.npz"))
    if not npz_files:
        print("No .npz files found to analyze")
        return
    
    comments_files = [f for f in npz_files if 'comments' in f.name]
    submissions_files = [f for f in npz_files if 'submissions' in f.name]
    
    subreddits = set()
    for f in npz_files:
        parts = f.name.split('_')
        if len(parts) >= 3:
            subreddit = parts[2]
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
