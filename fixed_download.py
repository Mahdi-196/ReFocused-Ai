import requests
import json
import os
from pathlib import Path
import time

def download_all_refocused_data():
    """Download ALL tokenized training data with pagination handling"""
    
    bucket_name = "refocused-ai"
    base_url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o"
    download_url_base = f"https://storage.googleapis.com/{bucket_name}/"
    
    # Create local directories
    data_dir = Path("data/training")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 DOWNLOADING ALL REFOCUSED-AI TRAINING DATA")
    print("=" * 60)
    print(f"Target directory: {data_dir.absolute()}")
    
    # Get ALL files with pagination
    print("📋 Fetching complete file list (handling pagination)...")
    all_files = []
    next_page_token = None
    page_count = 0
    
    try:
        while True:
            page_count += 1
            print(f"   Fetching page {page_count}...")
            
            # Build URL with pagination
            url = base_url
            params = {"maxResults": 1000}  # Max per request
            if next_page_token:
                params["pageToken"] = next_page_token
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'items' in data:
                page_files = data['items']
                all_files.extend(page_files)
                print(f"   Found {len(page_files)} files on page {page_count}")
            
            # Check for next page
            if 'nextPageToken' in data:
                next_page_token = data['nextPageToken']
            else:
                break
        
        print(f"✅ Complete file list retrieved: {len(all_files)} total files")
        
        # Filter for .npz files only
        npz_files = [f for f in all_files if f['name'].endswith('.npz')]
        print(f"🎯 {len(npz_files)} tokenized .npz files to download")
        
        # Calculate REAL total size
        total_size = sum(int(f['size']) for f in npz_files)
        total_gb = total_size / (1024 * 1024 * 1024)
        print(f"📊 REAL Total download size: {total_gb:.2f} GB")
        print(f"⚠️  This is {total_gb:.1f}GB - much larger than initially detected!")
        
        # Estimate download time
        estimated_minutes = (total_gb * 1024) / 10  # Assuming 10MB/s
        print(f"⏱️  Estimated download time: {estimated_minutes:.1f} minutes")
        
        # Ask for confirmation for large downloads
        if total_gb > 5:
            print(f"\n⚠️  WARNING: This is a {total_gb:.1f}GB download!")
            print("Continue? (y/n): ", end="")
            # For server use, we'll continue automatically
            print("y (auto-continuing)")
        
        # Download all files with progress tracking
        print(f"\n🔥 Starting download of {len(npz_files)} files...")
        start_time = time.time()
        downloaded_count = 0
        downloaded_bytes = 0
        failed_files = []
        
        for i, file_info in enumerate(npz_files, 1):
            filename = file_info['name']
            file_size = int(file_info['size'])
            file_size_mb = file_size / (1024 * 1024)
            
            # Create download URL
            download_url = f"{download_url_base}{filename}"
            local_path = data_dir / filename
            
            # Skip if file already exists and is correct size
            if local_path.exists() and local_path.stat().st_size == file_size:
                print(f"[{i:3d}/{len(npz_files)}] ✅ SKIP {filename} (already exists)")
                downloaded_count += 1
                downloaded_bytes += file_size
                continue
            
            print(f"[{i:3d}/{len(npz_files)}] 📥 {filename} ({file_size_mb:.1f} MB)")
            
            try:
                # Download with streaming and timeout
                file_response = requests.get(download_url, stream=True, timeout=60)
                file_response.raise_for_status()
                
                # Write file with progress
                with open(local_path, 'wb') as f:
                    downloaded_chunk_bytes = 0
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_chunk_bytes += len(chunk)
                
                # Verify file size
                actual_size = local_path.stat().st_size
                if actual_size != file_size:
                    print(f"    ⚠️  Size mismatch: expected {file_size}, got {actual_size}")
                    failed_files.append(filename)
                else:
                    downloaded_count += 1
                    downloaded_bytes += file_size
                
                # Progress update every 10 files
                if i % 10 == 0:
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        speed_mbps = (downloaded_bytes / (1024 * 1024)) / elapsed
                        remaining_files = len(npz_files) - i
                        eta_minutes = (remaining_files * file_size_mb / 1024) / (speed_mbps / 60) if speed_mbps > 0 else 0
                        print(f"    📈 Progress: {i}/{len(npz_files)} files, {speed_mbps:.1f} MB/s, ETA: {eta_minutes:.1f}min")
                
            except Exception as e:
                print(f"    ❌ Failed to download {filename}: {e}")
                failed_files.append(filename)
                continue
        
        # Final summary
        elapsed = time.time() - start_time
        elapsed_minutes = elapsed / 60
        avg_speed = (downloaded_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
        final_size_gb = downloaded_bytes / (1024 * 1024 * 1024)
        
        print("\n🎉 DOWNLOAD COMPLETE!")
        print("=" * 60)
        print(f"✅ Downloaded: {downloaded_count}/{len(npz_files)} files")
        print(f"📊 Total size: {final_size_gb:.2f} GB")
        print(f"⏱️  Time taken: {elapsed_minutes:.1f} minutes")
        print(f"🚀 Average speed: {avg_speed:.1f} MB/s")
        print(f"📁 Files saved to: {data_dir.absolute()}")
        
        if failed_files:
            print(f"⚠️  Failed files ({len(failed_files)}): {failed_files[:5]}...")
        
        # Verify downloads
        print("\n🔍 Verifying downloads...")
        local_files = list(data_dir.glob("*.npz"))
        print(f"✅ {len(local_files)} .npz files in local directory")
        
        success_rate = downloaded_count / len(npz_files) * 100
        print(f"📊 Success rate: {success_rate:.1f}%")
        
        if success_rate >= 95:
            print("🎯 Download successful! Ready for training!")
            return True
        else:
            print(f"⚠️  Only {success_rate:.1f}% success rate. May need to retry failed files.")
            return False
            
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return False

def analyze_downloaded_data():
    """Analyze the downloaded data"""
    data_dir = Path("data/training")
    npz_files = list(data_dir.glob("*.npz"))
    
    if not npz_files:
        print("No .npz files found")
        return
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in npz_files)
    total_gb = total_size / (1024 * 1024 * 1024)
    
    # Analyze file types
    comments_files = [f for f in npz_files if 'comments' in f.name]
    submissions_files = [f for f in npz_files if 'submissions' in f.name]
    
    # Get subreddits
    subreddits = set()
    for f in npz_files:
        parts = f.name.split('_')
        if len(parts) >= 3:
            subreddit = parts[2]
            subreddits.add(subreddit)
    
    print(f"\n📋 DATASET ANALYSIS:")
    print(f"   Total files: {len(npz_files)}")
    print(f"   Total size: {total_gb:.2f} GB")
    print(f"   Comments files: {len(comments_files)}")
    print(f"   Submissions files: {len(submissions_files)}")
    print(f"   Unique subreddits: {len(subreddits)}")
    print(f"   Average file size: {total_size / len(npz_files) / (1024*1024):.1f} MB")
    
    # Save analysis
    analysis = {
        "total_files": len(npz_files),
        "total_size_gb": total_gb,
        "comments_files": len(comments_files),
        "submissions_files": len(submissions_files),
        "unique_subreddits": len(subreddits),
        "subreddits": sorted(list(subreddits)),
        "average_file_size_mb": total_size / len(npz_files) / (1024*1024),
        "ready_for_training": True
    }
    
    info_file = data_dir / "dataset_analysis.json"
    with open(info_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"   Analysis saved: {info_file}")
    return analysis

if __name__ == "__main__":
    print("🎯 DOWNLOADING COMPLETE REFOCUSED-AI DATASET")
    print("This will download the FULL dataset with proper pagination handling")
    print("=" * 60)
    
    success = download_all_refocused_data()
    if success:
        analyze_downloaded_data()
        print("\n🚀 DATASET READY FOR TRAINING!")
        print("Use with: --data-path data/training")
    else:
        print("\n❌ Download issues occurred. Check errors above.") 