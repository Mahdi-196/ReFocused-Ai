#!/usr/bin/env python3
import os
from pathlib import Path

def check_completion_status():
    input_dir = Path("data/cleaned")
    output_dir = Path("data_tokenized_production")
    
    print("🔍 TOKENIZATION COMPLETION STATUS CHECK (FINAL)")
    print("=" * 50)
    
    # Get all input files
    input_files = sorted([f for f in input_dir.glob("*.jsonl")])
    input_names = {f.name for f in input_files}
    print(f"📁 Total input files: {len(input_files)}")
    
    # Get all output files
    output_files = list(output_dir.glob("*.npz"))
    print(f"💾 Total output files: {len(output_files)}")
    
    # Map output files back to input files
    completed_inputs = set()
    
    for output_file in output_files:
        name = output_file.stem
        
        # Remove 'tokenized_' prefix if present
        if name.startswith('tokenized_'):
            name = name[10:]
        
        # Try two approaches for mapping:
        # 1. Direct mapping (for files that originally had _part001)
        original_name_1 = name + '.jsonl'
        
        # 2. Remove _part001 suffix (for files that were split during processing)
        original_name_2 = None
        if name.endswith('_part001'):
            original_name_2 = name[:-8] + '.jsonl'
        
        # Use whichever mapping matches an actual input file
        if original_name_1 in input_names:
            completed_inputs.add(original_name_1)
        elif original_name_2 and original_name_2 in input_names:
            completed_inputs.add(original_name_2)
    
    print(f"✅ Unique input files completed: {len(completed_inputs)}")
    
    # Find missing files
    missing_files = input_names - completed_inputs
    
    print(f"\n📊 DETAILED STATUS:")
    print(f"   Input files: {len(input_files)}")
    print(f"   Completed: {len(completed_inputs)}")
    print(f"   Missing: {len(missing_files)}")
    print(f"   Completion: {len(completed_inputs)/len(input_files)*100:.1f}%")
    
    if missing_files:
        print(f"\n❌ MISSING FILES ({len(missing_files)}):")
        for missing in sorted(list(missing_files)[:30]):  # Show first 30
            print(f"   - {missing}")
        if len(missing_files) > 30:
            print(f"   ... and {len(missing_files) - 30} more")
            
        # Save missing files list for processing
        with open('missing_files.txt', 'w') as f:
            for missing in sorted(missing_files):
                f.write(f"data/cleaned/{missing}\n")
        print(f"\n💾 Saved missing files list to: missing_files.txt")
    else:
        print(f"\n🎉 ALL FILES COMPLETED!")
    
    # Show some examples of completed files for verification
    print(f"\n✅ SAMPLE COMPLETED FILES:")
    for i, completed in enumerate(sorted(completed_inputs)[:10]):
        print(f"   - {completed}")
    if len(completed_inputs) > 10:
        print(f"   ... and {len(completed_inputs) - 10} more")
    
    return len(missing_files) == 0

if __name__ == "__main__":
    is_complete = check_completion_status()
    print(f"\n{'🎉 TOKENIZATION COMPLETE!' if is_complete else '⚠️ TOKENIZATION INCOMPLETE'}") 