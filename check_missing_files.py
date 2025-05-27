#!/usr/bin/env python3
"""
Check if missing input files actually exist
"""
from pathlib import Path

def check_missing_files():
    # Read the missing files list
    try:
        with open('missing_files_simple.txt', 'r') as f:
            missing_files = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("❌ missing_files_simple.txt not found!")
        return
    
    print(f"🔍 CHECKING {len(missing_files)} MISSING FILES")
    print("=" * 50)
    
    existing_count = 0
    missing_count = 0
    
    for file_path in missing_files[:10]:  # Check first 10
        file_obj = Path(file_path)
        
        if file_obj.exists():
            size_mb = file_obj.stat().st_size / (1024 * 1024)
            print(f"✅ EXISTS: {file_obj.name} ({size_mb:.1f} MB)")
            existing_count += 1
        else:
            print(f"❌ MISSING: {file_path}")
            missing_count += 1
    
    print(f"\n📊 SAMPLE RESULTS:")
    print(f"   Files that exist: {existing_count}")
    print(f"   Files truly missing: {missing_count}")
    
    if existing_count > 0:
        print(f"\n💡 DIAGNOSIS: Input files exist but weren't processed!")
        print(f"   The process_missing_final.py script likely failed silently")
        print(f"   or had errors that prevented file creation.")
    else:
        print(f"\n🤔 DIAGNOSIS: Input files don't exist in expected location")

if __name__ == "__main__":
    check_missing_files() 