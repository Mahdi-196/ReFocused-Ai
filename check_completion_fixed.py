#!/usr/bin/env python3
import os
from pathlib import Path

def check_completion_status():
    input_dir = Path("data/cleaned")
    output_dir = Path("data_tokenized_production")
    
    print("🔍 TOKENIZATION COMPLETION STATUS CHECK (FIXED)")
    print("=" * 50)
    
    # Get all input files
    input_files = sorted([f for f in input_dir.glob("*.jsonl")])
    input_names = {f.name for f in input_files}
    print(f"📁 Total input files: {len(input_files)}")
    
    # Get all output files
    output_files = list(output_dir.glob("*.npz"))
    print(f"💾 Total output files: {len(output_files)}")
    
    # Debug: Show some example mappings
    print(f"\n🔍 DEBUG - Sample filename mappings:")
    for i, output_file in enumerate(output_files[:5]):
        print(f"   Output: {output_file.name}")
        # Try to reconstruct input filename
        name = output_file.stem
        if name.startswith('tokenized_'):
            name = name[10:]  # Remove 'tokenized_'
        if name.endswith('_part001'):
            name = name[:-8]  # Remove '_part001'
        reconstructed = name + '.jsonl'
        print(f"   → Input: {reconstructed}")
        print(f"   → Exists: {reconstructed in input_names}")
        print()
    
    # Map output files back to input files
    completed_inputs = set()
    
    for output_file in output_files:
        name = output_file.stem
        
        # Remove 'tokenized_' prefix if present
        if name.startswith('tokenized_'):
            name = name[10:]
        
        # Remove '_part001' suffix if present (handles both _part001 and _part001_part001)
        if '_part001' in name:
            # Find the last occurrence of _part001
            last_part_pos = name.rfind('_part001')
            name = name[:last_part_pos]
        
        # Reconstruct original filename
        original_name = name + '.jsonl'
        
        # Only add if it matches an actual input file
        if original_name in input_names:
            completed_inputs.add(original_name)
    
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
        for missing in sorted(list(missing_files)[:20]):  # Show first 20
            print(f"   - {missing}")
        if len(missing_files) > 20:
            print(f"   ... and {len(missing_files) - 20} more")
            
        # Save missing files list for processing
        with open('missing_files.txt', 'w') as f:
            for missing in sorted(missing_files):
                f.write(f"data/cleaned/{missing}\n")
        print(f"\n💾 Saved missing files list to: missing_files.txt")
    else:
        print(f"\n🎉 ALL FILES COMPLETED!")
    
    return len(missing_files) == 0

if __name__ == "__main__":
    is_complete = check_completion_status()
    print(f"\n{'🎉 TOKENIZATION COMPLETE!' if is_complete else '⚠️ TOKENIZATION INCOMPLETE'}") 