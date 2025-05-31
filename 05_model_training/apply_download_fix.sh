#!/bin/bash
#
# Fix for download_data.py NoneType division error
# Run this script on the server to apply the fix
#

set -e

echo "Applying fix for download_data.py NoneType division error..."

# Make backup of original file
if [ -f download_data.py ]; then
    cp download_data.py download_data.py.bak
    echo "Backup created as download_data.py.bak"
else
    echo "ERROR: download_data.py not found in current directory"
    exit 1
fi

# Apply the fixes manually with sed
echo "Applying fixes to download_data.py..."

# Fix 1: Add None check in file skip condition
sed -i 's/if local_file_path.stat().st_size == blob.size:/if blob.size is not None and local_file_path.stat().st_size == blob.size:/' download_data.py

# Fix 2: Add None check for size in download message
sed -i 's/logger.info(f"Downloading {blob.name} ({blob.size \/ 1024 \/ 1024:.1f} MB)")/if blob.size is not None:\n            logger.info(f"Downloading {blob.name} ({blob.size \/ 1024 \/ 1024:.1f} MB)")\n        else:\n            logger.info(f"Downloading {blob.name} (size unknown)")/' download_data.py

# Fix 3: Add safe size calculation in list_all_files
sed -i '/for blob in bucket.list_blobs():/a\        # Safely calculate size\n        size_mb = None\n        if blob.size is not None:\n            size_mb = blob.size / (1024 * 1024)' download_data.py

# Fix 4: Use safe size_mb in file_info
sed -i "s/'size_mb': blob.size \/ (1024 \* 1024),/'size_mb': size_mb,/" download_data.py

# Fix 5: Add None check for size_mb in logging
sed -i 's/logger.info(f" - {file_info\['\''name'\''\]} ({file_info\['\''size_mb'\''\]:.2f} MB)")/if file_info['\''size_mb'\''] is not None:\n                    logger.info(f" - {file_info['\''name'\''] } ({file_info['\''size_mb'\'']:.2f} MB)")\n                else:\n                    logger.info(f" - {file_info['\''name'\''] } (size unknown)")/' download_data.py

# Fix 6: Add safe total size calculation
sed -i 's/total_size_mb = sum(\[b.size for b in npz_blobs\]) \/ (1024 \* 1024)/# Calculate total size safely\n            total_size_mb = 0\n            valid_sizes = 0\n            for b in npz_blobs:\n                if b.size is not None:\n                    total_size_mb += b.size \/ (1024 * 1024)\n                    valid_sizes += 1/' download_data.py

# Fix 7: Add conditional for total size and speed logging
sed -i 's/logger.info(f"Total data downloaded: {total_size_mb:.2f} MB")/if valid_sizes > 0:\n                logger.info(f"Total data downloaded: {total_size_mb:.2f} MB")\n                logger.info(f"Average speed: {total_size_mb \/ duration:.2f} MB\/s")\n            else:\n                logger.info("Total data size unknown")/' download_data.py

# Fix 8: Remove the duplicate average speed line
sed -i '/logger.info(f"Average speed: {total_size_mb \/ duration:.2f} MB\/s")/d' download_data.py

echo "Fix applied successfully!"
echo "To test the fix, run: python3 download_data.py --bucket refocused-ai --local_dir /home/ubuntu/training_data/shards --max_files 25 --workers 8"
echo "If you need to restore the original file, run: cp download_data.py.bak download_data.py" 