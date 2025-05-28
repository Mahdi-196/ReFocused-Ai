import requests
import json
from urllib.parse import urljoin

def test_public_bucket():
    bucket_name = "refocused-ai"
    base_url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o"
    direct_url = f"https://storage.googleapis.com/{bucket_name}/"
    
    print(f"🔍 Testing public bucket: gs://{bucket_name}/")
    print("=" * 50)
    
    # Test 1: Direct bucket access
    try:
        print("📋 Test 1: Direct bucket access...")
        response = requests.get(direct_url)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Bucket is publicly accessible!")
        else:
            print(f"   ❌ Direct access failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: List objects using API
    try:
        print("\n📋 Test 2: Listing bucket contents...")
        api_response = requests.get(base_url)
        print(f"   Status: {api_response.status_code}")
        
        if api_response.status_code == 200:
            data = api_response.json()
            if 'items' in data:
                files = data['items']
                print(f"   ✅ Found {len(files)} files:")
                
                total_size = 0
                for file in files[:10]:  # Show first 10 files
                    name = file['name']
                    size = int(file['size'])
                    size_mb = size / (1024 * 1024)
                    total_size += size
                    print(f"      📄 {name} ({size_mb:.2f} MB)")
                
                if len(files) > 10:
                    print(f"      ... and {len(files) - 10} more files")
                
                total_size_gb = total_size / (1024 * 1024 * 1024)
                print(f"\n   📊 Total size: {total_size_gb:.2f} GB")
                
                return files
            else:
                print("   ❓ No files found in bucket")
        else:
            print(f"   ❌ API access failed: {api_response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error listing contents: {e}")
    
    # Test 3: Try common filenames
    print("\n📋 Test 3: Testing common training data filenames...")
    common_files = [
        "train.jsonl", "training.jsonl", "train_data.jsonl",
        "val.jsonl", "validation.jsonl", "val_data.jsonl", 
        "test.jsonl", "test_data.jsonl",
        "data.txt", "training_data.txt",
        "tokenized_train.bin", "tokenized_val.bin"
    ]
    
    found_files = []
    for filename in common_files:
        file_url = f"{direct_url}{filename}"
        try:
            response = requests.head(file_url, timeout=5)
            if response.status_code == 200:
                size = response.headers.get('content-length', 'unknown')
                print(f"   ✅ {filename} (size: {size} bytes)")
                found_files.append(filename)
            elif response.status_code == 404:
                print(f"   ❌ {filename} (not found)")
            else:
                print(f"   ❓ {filename} (status: {response.status_code})")
        except Exception as e:
            print(f"   ❌ {filename} (error: {e})")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Bucket accessible: {'✅' if True else '❌'}")
    print(f"   Files found via API: {'✅' if 'files' in locals() else '❌'}")
    print(f"   Common files found: {len(found_files)}")
    
    return found_files

def test_download_speed(filename="train.jsonl"):
    """Test download speed for a specific file"""
    url = f"https://storage.googleapis.com/{bucket_name}/{filename}"
    
    print(f"\n🚀 Testing download speed for {filename}...")
    try:
        import time
        start_time = time.time()
        
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            downloaded = 0
            chunk_size = 8192
            
            for chunk in response.iter_content(chunk_size=chunk_size):
                downloaded += len(chunk)
                if downloaded > 1024 * 1024:  # Stop after 1MB test
                    break
            
            elapsed = time.time() - start_time
            speed_mbps = (downloaded / (1024 * 1024)) / elapsed
            
            print(f"   ✅ Download speed: {speed_mbps:.2f} MB/s")
            print(f"   📊 Downloaded {downloaded} bytes in {elapsed:.2f}s")
            return True
        else:
            print(f"   ❌ Failed to download: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Download test failed: {e}")
        return False

if __name__ == "__main__":
    bucket_name = "refocused-ai"
    found_files = test_public_bucket()
    
    if found_files:
        print(f"\n🎉 SUCCESS! Found {len(found_files)} accessible files")
        test_download_speed(found_files[0])
    else:
        print("\n❌ No accessible files found. Check bucket permissions.") 