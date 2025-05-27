#!/usr/bin/env python3
"""
Cleanup Duplicate Scripts
Removes old/duplicate versions and keeps only working scripts
"""

import os
from pathlib import Path

def cleanup_duplicate_scripts():
    print("🧹 CLEANING UP DUPLICATE SCRIPTS")
    print("=" * 40)
    
    # Define what to keep vs remove
    scripts_to_remove = [
        # Keep only simple_completion_check.py
        "check_completion_status.py",
        "check_completion_fixed.py", 
        "check_completion_final.py",
        
        # Keep only run_full_tokenization_fixed.py
        "run_full_tokenization.py",
        
        # Keep only test_cloud_tokenization_10min_fixed.py
        "test_cloud_tokenization_10min.py",
        
        # Remove old diagnostic scripts (keep diagnose_tokenization_issue.py)
        "debug_filenames.py",
        
        # Remove old test scripts (keep test_tokenizer_small.py)
        "test_small_tokenization.py",
        "test_tokenization.py",
        
        # Remove old logs
        "tokenization_20250525_145359.log",
        "tokenization_diagnosis.log",
    ]
    
    # Additional files to remove (old temporary files)
    temp_files_to_remove = [
        "missing_files.txt",  # Keep missing_files_simple.txt
        "duplicates_to_remove.txt",  # No longer needed
        "analyze_unmapped.py",  # One-time use script
        "find_duplicates.py",  # One-time use script
    ]
    
    all_files_to_remove = scripts_to_remove + temp_files_to_remove
    
    print("📋 FILES TO REMOVE:")
    removed_count = 0
    for filename in all_files_to_remove:
        file_path = Path(filename)
        if file_path.exists():
            print(f"   ❌ {filename}")
            try:
                file_path.unlink()
                removed_count += 1
            except Exception as e:
                print(f"      ⚠️ Error removing: {e}")
        else:
            print(f"   ➖ {filename} (not found)")
    
    print(f"\n✅ KEPT WORKING SCRIPTS:")
    working_scripts = [
        "simple_completion_check.py",
        "process_missing_final.py", 
        "count_final_sequences.py",
        "resume_tokenization.py",
        "run_full_tokenization_fixed.py",
        "test_cloud_tokenization_10min_fixed.py",
        "fix_tokenization_single_threaded.py",
        "diagnose_tokenization_issue.py",
        "test_tokenizer_small.py",
        "monitor_tokenization.py",
    ]
    
    for script in working_scripts:
        if Path(script).exists():
            print(f"   ✅ {script}")
        else:
            print(f"   ⚠️ {script} (missing)")
    
    print(f"\n🎉 CLEANUP COMPLETE!")
    print(f"   Removed: {removed_count} files")
    print(f"   Working scripts preserved: {len([s for s in working_scripts if Path(s).exists()])}")
    
    # Show final essential scripts for reference
    print(f"\n📚 ESSENTIAL SCRIPTS FOR REFERENCE:")
    print(f"   🔍 Check status: python3 simple_completion_check.py")
    print(f"   🚀 Process missing: python3 process_missing_final.py") 
    print(f"   📊 Count sequences: python3 count_final_sequences.py")
    print(f"   🔄 Resume if needed: python3 resume_tokenization.py")

if __name__ == "__main__":
    cleanup_duplicate_scripts() 