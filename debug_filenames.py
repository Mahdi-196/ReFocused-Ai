#!/usr/bin/env python3
"""
Debug script to examine actual tokenized file naming patterns.
"""

from pathlib import Path

def debug_filenames():
    """Examine actual tokenized file patterns."""
    
    input_dir = Path("data/cleaned")
    output_dir = Path("data_tokenized_production")
    
    print("🔍 DEBUGGING FILENAME PATTERNS")
    print("=" * 50)
    
    # Show sample input files
    print("📁 Sample input files:")
    input_files = list(input_dir.glob("*.jsonl"))[:5]
    for f in input_files:
        print(f"  {f.name}")
    print(f"  ... (total: {len(list(input_dir.glob('*.jsonl')))} files)")
    print()
    
    # Show sample output files
    print("📁 Sample output files:")
    if output_dir.exists():
        output_files = list(output_dir.glob("*.npz"))[:10]
        for f in output_files:
            print(f"  {f.name}")
        print(f"  ... (total: {len(list(output_dir.glob('*.npz')))} files)")
        
        # Analyze patterns
        print("\n🔍 Analyzing filename patterns:")
        for f in output_files[:3]:
            filename = f.name
            print(f"  Original: {filename}")
            
            # Try to extract base name
            if filename.startswith("data_tokenized_"):
                name_part = filename[len("data_tokenized_"):]
                print(f"    After removing 'data_tokenized_': {name_part}")
                
                if "_part" in name_part:
                    base_name = name_part.split("_part")[0]
                    print(f"    Base name: {base_name}")
                    print(f"    Would look for: {base_name}.jsonl")
            print()
            
    else:
        print("  Output directory doesn't exist!")
    
    print("=" * 50)

if __name__ == "__main__":
    debug_filenames() 