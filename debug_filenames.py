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
    print("=" * 60)
    
    # Get all files
    input_files = list(input_dir.glob("*.jsonl"))
    output_files = list(output_dir.glob("*.npz"))
    
    print(f"📁 Input files: {len(input_files)}")
    print(f"📁 Output files: {len(output_files)}")
    print()
    
    # Show sample patterns
    print("📝 Sample input files:")
    for f in input_files[:5]:
        print(f"  {f.name}")
    print()
    
    print("📝 Sample output files:")
    for f in output_files[:5]:
        print(f"  {f.name}")
    print()
    
    # Test the matching logic
    print("🔍 Testing filename matching:")
    completed = set()
    
    for npz_file in output_files[:10]:  # Test first 10
        filename = npz_file.name
        print(f"  Output: {filename}")
        
        if filename.startswith("tokenized_cleaned_"):
            name_part = filename[len("tokenized_cleaned_"):]
            print(f"    After removing prefix: {name_part}")
            
            if "_part" in name_part:
                original_name = name_part.split("_part")[0]
                expected_input = f"cleaned_{original_name}.jsonl"
                print(f"    Looking for input: {expected_input}")
                
                # Check if this input file exists
                input_path = input_dir / expected_input
                if input_path.exists():
                    print(f"    ✅ FOUND: {expected_input}")
                    completed.add(expected_input)
                else:
                    print(f"    ❌ NOT FOUND: {expected_input}")
                    
                    # Try alternative patterns
                    alt_patterns = [
                        f"{original_name}.jsonl",
                        f"cleaned_{original_name}_part001.jsonl",
                        f"{original_name}_part001.jsonl"
                    ]
                    
                    for alt in alt_patterns:
                        alt_path = input_dir / alt
                        if alt_path.exists():
                            print(f"    🔍 Alternative found: {alt}")
                            break
                    else:
                        print(f"    🤔 No alternatives found")
        print()
    
    print(f"📊 Matched so far: {len(completed)} files")
    print("=" * 60)

if __name__ == "__main__":
    debug_filenames() 