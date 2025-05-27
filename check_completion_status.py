#!/usr/bin/env python3
"""
Check Completion Status Script
Verifies which files are completed and which still need processing
"""

import os
from pathlib import Path
import re

def check_completion_status():
    """Check what files have been processed and what remains"""
    
    input_dir = Path("data/cleaned")
    output_dir = Path("data_tokenized_production")
    
    print("🔍 TOKENIZATION COMPLETION STATUS CHECK")
    print("=" * 50)
    
    # Get all input files
    input_files = sorted([f for f in input_dir.glob("*.jsonl")])
    print(f"📁 Total input files: {len(input_files)}")
    
    # Get all output files
    output_files = list(output_dir.glob("*.npz"))
    print(f"💾 Total output files: {len(output_files)}")
    
    # Map output files back to input files
    completed_inputs = set()
    
    for output_file in output_files:
        # Extract original filename from output filename
        # Format: tokenized_cleaned_filename_part001.npz
        name_parts = output_file.stem.split('_')
        if len(name_parts) >= 3:
            # Reconstruct original filename
            original_name = '_'.join(name_parts[2:-1]) + '.jsonl'
            completed_inputs.add(original_name)
    
    print(f"✅ Unique input files completed: {len(completed_inputs)}")
    
    # Find missing files
    input_names = {f.name for f in input_files}
    missing_files = input_names - completed_inputs
    
    print(f"\n📊 DETAILED STATUS:")
    print(f"   Input files: {len(input_files)}")
    print(f"   Completed: {len(completed_inputs)}")
    print(f"   Missing: {len(missing_files)}")
    print(f"   Completion: {len(completed_inputs)/len(input_files)*100:.1f}%")
    
    if missing_files:
        print(f"\n❌ MISSING FILES ({len(missing_files)}):")
        for missing in sorted(missing_files):
            print(f"   - {missing}")
    else:
        print(f"\n🎉 ALL FILES COMPLETED!")
    
    # Check for zero-size output files (failed saves)
    print(f"\n🔍 CHECKING FOR FAILED SAVES:")
    failed_saves = []
    for output_file in output_files:
        if output_file.stat().st_size == 0:
            failed_saves.append(output_file.name)
    
    if failed_saves:
        print(f"⚠️ Found {len(failed_saves)} zero-size files (failed saves):")
        for failed in failed_saves:
            print(f"   - {failed}")
    else:
        print("✅ No zero-size files found")
    
    # Total sequences check
    print(f"\n📈 SEQUENCE COUNT ESTIMATE:")
    total_sequences = 0
    valid_files = 0
    
    for output_file in output_files:
        try:
            if output_file.stat().st_size > 0:
                import numpy as np
                data = np.load(output_file)
                if 'sequences' in data:
                    sequences = data['sequences']
                    total_sequences += len(sequences)
                    valid_files += 1
        except:
            continue
    
    print(f"   Valid output files: {valid_files}")
    print(f"   Total sequences: {total_sequences:,}")
    
    return len(missing_files) == 0 and len(failed_saves) == 0

if __name__ == "__main__":
    is_complete = check_completion_status()
    print(f"\n{'🎉 TOKENIZATION COMPLETE!' if is_complete else '⚠️ TOKENIZATION INCOMPLETE'}") 