import requests
import json
import os
from pathlib import Path
import time

def download_refocused_data():
    """Download all tokenized training data from refocused-ai bucket"""
    
    bucket_name = "refocused-ai"
    base_url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o"
    download_url_base = f"https://storage.googleapis.com/{bucket_name}/"
    
    # Create local directories
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 DOWNLOADING REFOCUSED-AI TRAINING DATA")
    print("=" * 50)
    print(f"Target directory: {data_dir.absolute()}")
    
    # Get list of all files
    print("📋 Fetching file list...")
    try:
        response = requests.get(base_url)
        response.raise_for_status()
        data = response.json()
        
        if 'items' not in data:
            print("❌ No files found in bucket")
            return False
            
        files = data['items']
        print(f"✅ Found {len(files)} files")
        
        # Filter for .npz files only
        npz_files = [f for f in files if f['name'].endswith('.npz')]
        print(f"🎯 {len(npz_files)} tokenized .npz files to download")
        
        # Calculate total size
        total_size = sum(int(f['size']) for f in npz_files)
        total_mb = total_size / (1024 * 1024)
        print(f"📊 Total download size: {total_mb:.1f} MB")
        
        # Download all files
        print("\n🔥 Starting downloads...")
        start_time = time.time()
        downloaded_count = 0
        downloaded_bytes = 0
        
        for i, file_info in enumerate(npz_files, 1):
            filename = file_info['name']
            file_size = int(file_info['size'])
            
            # Create download URL
            download_url = f"{download_url_base}{filename}"
            local_path = data_dir / filename
            
            print(f"[{i:3d}/{len(npz_files)}] {filename} ({file_size/1024/1024:.1f} MB)")
            
            try:
                # Download file
                file_response = requests.get(download_url, stream=True)
                file_response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_count += 1
                downloaded_bytes += file_size
                
                # Progress update every 50 files
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    speed_mbps = (downloaded_bytes / (1024 * 1024)) / elapsed
                    print(f"    📈 Progress: {i}/{len(npz_files)} files, {speed_mbps:.1f} MB/s")
                
            except Exception as e:
                print(f"    ❌ Failed to download {filename}: {e}")
                continue
        
        # Final summary
        elapsed = time.time() - start_time
        avg_speed = (downloaded_bytes / (1024 * 1024)) / elapsed
        
        print("\n🎉 DOWNLOAD COMPLETE!")
        print("=" * 50)
        print(f"✅ Downloaded: {downloaded_count}/{len(npz_files)} files")
        print(f"📊 Total size: {downloaded_bytes/1024/1024:.1f} MB")
        print(f"⏱️  Time taken: {elapsed:.1f} seconds")
        print(f"🚀 Average speed: {avg_speed:.1f} MB/s")
        print(f"📁 Files saved to: {data_dir.absolute()}")
        
        # Verify downloads
        print("\n🔍 Verifying downloads...")
        local_files = list(data_dir.glob("*.npz"))
        print(f"✅ {len(local_files)} .npz files in local directory")
        
        if len(local_files) == len(npz_files):
            print("🎯 All files downloaded successfully!")
            return True
        else:
            print(f"⚠️  Expected {len(npz_files)}, got {len(local_files)} files")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return False

def create_data_info():
    """Create info file about the downloaded data"""
    data_dir = Path("data/training")
    npz_files = list(data_dir.glob("*.npz"))
    
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