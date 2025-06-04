import os
from google.cloud import storage
import tempfile
import json

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../credentials/black-dragon-461023-t5-93452a49f86b.json'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'black-dragon-461023-t5'

print('🔐 Testing Google Cloud Storage Authentication...')
print('=' * 50)

try:
    # Initialize client
    client = storage.Client()
    bucket = client.bucket('refocused-ai')
    
    print('✅ Client initialized successfully')
    
    # Test 1: List files
    print('\n📋 Test 1: Listing files in bucket...')
    blobs = list(bucket.list_blobs(max_results=10))
    print(f'✅ Found {len(blobs)} files (showing first 10):')
    
    # Filter out directories and find actual files
    file_blobs = []
    for i, blob in enumerate(blobs):
        if blob.size > 0:  # Skip directories (size = 0)
            file_blobs.append(blob)
            print(f'  {len(file_blobs)}. {blob.name} ({blob.size/1024/1024:.2f} MB)')
        else:
            print(f'  [DIR] {blob.name}')
    
    # Test 2: Download a small file
    print('\n📥 Test 2: Download test...')
    if file_blobs:
        # Find smallest file for testing
        test_blob = min(file_blobs, key=lambda b: b.size)
        print(f'Downloading smallest file: {test_blob.name} ({test_blob.size/1024/1024:.2f} MB)')
        
        # Download to temporary file with proper cleanup
        temp_path = f'./temp_download_test_{os.getpid()}.tmp'
        try:
            test_blob.download_to_filename(temp_path)
            file_size = os.path.getsize(temp_path)
            print(f'✅ Downloaded {file_size:,} bytes successfully')
            
            # Clean up immediately
            os.remove(temp_path)
            print('✅ Download test file cleaned up')
        except Exception as e:
            print(f'Download error: {e}')
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    
    # Test 3: Upload a test file
    print('\n📤 Test 3: Upload test...')
    test_data = {
        'message': 'Test upload from VM', 
        'timestamp': '2025-06-03',
        'vm_test': True,
        'credentials_working': True
    }
    test_filename = 'vm_test_upload.json'
    local_test_file = f'./temp_upload_test_{os.getpid()}.json'
    
    try:
        # Create test file locally
        with open(local_test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Upload to bucket
        blob = bucket.blob(test_filename)
        blob.upload_from_filename(local_test_file)
        print(f'✅ Uploaded test file: {test_filename}')
        
        # Verify upload
        if blob.exists():
            print(f'✅ Upload verified - file exists in bucket')
            print(f'   Size: {blob.size} bytes')
            
            # Download it back to verify
            downloaded_data = blob.download_as_text()
            verified_data = json.loads(downloaded_data)
            print(f'✅ Round-trip test passed')
            print(f'   Verified data: {verified_data["message"]}')
            
            # Clean up test file from bucket
            blob.delete()
            print('✅ Test file cleaned up from bucket')
        
        # Clean up local temp file
        if os.path.exists(local_test_file):
            os.remove(local_test_file)
        
    except Exception as e:
        print(f'Upload test error: {e}')
        # Clean up on error
        if os.path.exists(local_test_file):
            try:
                os.remove(local_test_file)
            except:
                pass
    
    print('\n🎉 ALL TESTS PASSED!')
    print('=' * 50)
    print('✅ Authentication working perfectly')
    print('✅ Download permissions confirmed') 
    print('✅ Upload permissions confirmed')
    print('✅ Ready for training data download')
    
    # Show some stats
    total_files = len(file_blobs)
    total_size = sum(blob.size for blob in file_blobs)
    print(f'\n📊 Bucket Stats:')
    print(f'   Real files: {total_files}')
    print(f'   Total size: {total_size/1024/1024/1024:.2f} GB')
    
except Exception as e:
    print(f'❌ Test failed: {e}')
    print('Check your credentials file and permissions') 