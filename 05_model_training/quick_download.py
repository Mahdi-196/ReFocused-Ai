import os
from google.cloud import storage
from pathlib import Path
import time

# Set environment variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './credentials/black-dragon-461023-t5-93452a49f86b.json'
os.environ['GOOGLE_CLOUD_PROJECT'] = 'black-dragon-461023-t5'

print('📥 Quick download of first 5 training files...')
data_dir = Path('data/training')
data_dir.mkdir(parents=True, exist_ok=True)

client = storage.Client()
bucket = client.bucket('refocused-ai')
blobs = list(bucket.list_blobs())
npz_blobs = [b for b in blobs if b.name.endswith('.npz')][:5]

print(f'Downloading {len(npz_blobs)} files for testing...')
for i, blob in enumerate(npz_blobs, 1):
    local_path = data_dir / blob.name
    print(f'[{i}/5] {blob.name} ({blob.size/1024/1024:.1f} MB)')
    blob.download_to_filename(str(local_path))

print('✅ Quick download complete!')
print(f'Files in data/training: {len(list(data_dir.glob("*.npz")))}')

# Create basic data info
import json
info = {
    "total_files": len(list(data_dir.glob("*.npz"))),
    "ready_for_training": True,
    "note": "Quick test download"
}
with open(data_dir / "data_info.json", 'w') as f:
    json.dump(info, f)

print('✅ Ready for training test!') 