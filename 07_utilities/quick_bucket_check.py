#!/usr/bin/env python3
"""
Quick Bucket Check - Fast inspection of bucket contents
"""

import os
from google.cloud import storage
import warnings
warnings.filterwarnings("ignore")

def quick_bucket_check(bucket_name="refocused-ai"):
    """Quick check of bucket contents"""
    print(f"🔍 QUICK BUCKET INSPECTION: gs://{bucket_name}")
    print("=" * 50)
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        print("📊 Scanning bucket contents...")
        blobs = list(bucket.list_blobs())
        
        # Categorize files
        file_types = {}
        total_size_gb = 0
        tokenized_files = []
        
        for blob in blobs:
            total_size_gb += blob.size / (1024**3)  # Convert to GB
            
            # Get file extension
            ext = blob.name.split('.')[-1].lower() if '.' in blob.name else 'no_ext'
            file_types[ext] = file_types.get(ext, 0) + 1
            
            # Check for tokenized files
            if blob.name.endswith('.npz'):
                if any(pattern in blob.name.lower() for pattern in ['tokenized', 'cleaned', 'processed']):
                    tokenized_files.append({
                        'name': blob.name,
                        'size_mb': blob.size / (1024**2),
                        'created': blob.time_created
                    })
        
        print(f"\n📈 BUCKET OVERVIEW")
        print(f"   Total files: {len(blobs):,}")
        print(f"   Total size: {total_size_gb:.2f} GB")
        
        print(f"\n📁 FILE TYPES:")
        for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   .{ext}: {count} files")
        
        print(f"\n🎯 TOKENIZED FILES FOR TRAINING:")
        print(f"   Found: {len(tokenized_files)} files")
        
        if tokenized_files:
            total_tokenized_size = sum(f['size_mb'] for f in tokenized_files)
            print(f"   Total size: {total_tokenized_size / 1024:.2f} GB")
            print(f"   Avg size: {total_tokenized_size / len(tokenized_files):.1f} MB")
            
            print(f"\n📋 SAMPLE TOKENIZED FILES:")
            for i, file_info in enumerate(tokenized_files[:10]):
                print(f"   {i+1}. {file_info['name']} ({file_info['size_mb']:.1f} MB)")
            
            if len(tokenized_files) > 10:
                print(f"   ... and {len(tokenized_files) - 10} more files")
        
        else:
            print("   ❌ No tokenized files found!")
            print("   Looking for files with patterns: 'tokenized', 'cleaned', 'processed'")
            
            # Show some .npz files that exist
            npz_files = [b for b in blobs if b.name.endswith('.npz')][:5]
            if npz_files:
                print(f"\n📋 OTHER .NPZ FILES FOUND:")
                for blob in npz_files:
                    print(f"   - {blob.name} ({blob.size / (1024**2):.1f} MB)")
        
        print(f"\n✅ Bucket inspection complete!")
        
        # Quick estimate
        if tokenized_files:
            print(f"\n📊 QUICK TRAINING ESTIMATE:")
            estimated_tokens = len(tokenized_files) * 50000  # Rough estimate
            estimated_steps = estimated_tokens // (4 * 4 * 2 * 1024)  # batch_size * grad_acc * gpus * seq_len
            print(f"   Estimated tokens: ~{estimated_tokens:,}")
            print(f"   Estimated steps (2 GPU): ~{estimated_steps:,}")
            print(f"   Estimated time: ~{estimated_steps / 5.0 / 3600:.1f} hours")
        
        return len(tokenized_files) > 0
        
    except Exception as e:
        print(f"❌ Error accessing bucket: {e}")
        print("\n💡 TROUBLESHOOTING:")
        print("1. Make sure you're authenticated: gcloud auth application-default login")
        print("2. Check bucket name is correct: refocused-ai")
        print("3. Verify you have read access to the bucket")
        return False

if __name__ == "__main__":
    quick_bucket_check() 