#!/usr/bin/env python3
import os
from pathlib import Path

def check_completion_status():
    input_dir = Path("data/cleaned")
    output_dir = Path("data_tokenized_production")
    
    print("🔍 FIXED COMPLETION CHECK")
    print("=" * 40)
    
    # Get all input files
    input_files = sorted([f for f in input_dir.glob("*.jsonl")])
    input_names = {f.name for f in input_files}
    print(f"📁 Total input files: {len(input_files)}")
    
    # Get all output files
    output_files = list(output_dir.glob("*.npz"))
    print(f"💾 Total output files: {len(output_files)}")
    
    # Fixed mapping logic
    completed_inputs = set()
    
    for output_file in output_files:
        # Handle pattern: tokenized_cleaned_X_part001.npz → cleaned_X.jsonl
        name = output_file.stem
        
        if name.startswith('tokenized_'):
            # Remove 'tokenized_' prefix
            without_prefix = name[10:]
            
            # Remove '_part001' suffix if present
            if without_prefix.endswith('_part001'):
                without_suffix = without_prefix[:-8]  # Remove '_part001'
            else:
                without_suffix = without_prefix
                
            # Add .jsonl extension
            input_name = without_suffix + '.jsonl'
            
            # Check if this input file exists
            if input_name in input_names:
                completed_inputs.add(input_name)
    
    print(f"✅ Completed files: {len(completed_inputs)}")
    
    # Find missing files
    missing_files = input_names - completed_inputs
    
    print(f"\n📊 STATUS:")
    print(f"   Input files: {len(input_files)}")
    print(f"   Completed: {len(completed_inputs)}")
    print(f"   Missing: {len(missing_files)}")
    print(f"   Completion: {len(completed_inputs)/len(input_files)*100:.1f}%")
    
    if missing_files:
        print(f"\n❌ MISSING FILES ({len(missing_files)}):")
        missing_list = sorted(list(missing_files))
        for missing in missing_list[:20]:  # Show first 20
            print(f"   - {missing}")
        if len(missing_files) > 20:
            print(f"   ... and {len(missing_files) - 20} more")
    else:
        print(f"\n🎉 ALL FILES COMPLETED!")
        
    # Show some examples of the mapping
    print(f"\n🔍 SAMPLE MAPPINGS:")
    for i, output_file in enumerate(sorted(output_files)[:5]):
        name = output_file.stem
        if name.startswith('tokenized_'):
            without_prefix = name[10:]
            if without_prefix.endswith('_part001'):
                without_suffix = without_prefix[:-8]
            else:
                without_suffix = without_prefix
            expected_input = without_suffix + '.jsonl'
            exists = expected_input in input_names
            print(f"   {output_file.name} → {expected_input} (exists: {exists})")
    
    return len(missing_files) == 0

if __name__ == "__main__":
    is_complete = check_completion_status()
    print(f"\n{'🎉 COMPLETE!' if is_complete else '⚠️ INCOMPLETE'}") 