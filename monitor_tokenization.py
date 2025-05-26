#!/usr/bin/env python3
"""
Monitor tokenization progress.
"""

import time
import os
from pathlib import Path

def monitor_progress():
    """Monitor the tokenization progress."""
    
    # Find the latest log file
    log_files = list(Path(".").glob("tokenization_*.log"))
    if not log_files:
        print("No tokenization log files found.")
        return
    
    latest_log = max(log_files, key=os.path.getctime)
    print(f"📊 Monitoring: {latest_log}")
    
    last_size = 0
    last_lines = []
    
    try:
        while True:
            current_size = latest_log.stat().st_size
            
            if current_size > last_size:
                # Read new content
                with open(latest_log, 'r') as f:
                    f.seek(last_size)
                    new_lines = f.readlines()
                
                # Show progress lines
                for line in new_lines:
                    if "Processed" in line and "items" in line:
                        print(f"⏳ {line.strip()}")
                    elif "Completed" in line:
                        print(f"✅ {line.strip()}")
                    elif "ERROR" in line:
                        print(f"❌ {line.strip()}")
                
                last_size = current_size
            
            # Check if process is still running
            output_dirs = ["data_tokenized_test", "data_tokenized"]
            for output_dir in output_dirs:
                if Path(output_dir).exists():
                    files = list(Path(output_dir).glob("*.npz"))
                    if files:
                        print(f"📁 Output files created: {len(files)}")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped by user")

if __name__ == "__main__":
    monitor_progress() 