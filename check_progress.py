#!/usr/bin/env python3
"""
Check Tokenization Progress
Shows current status of tokenization process.
"""

from pathlib import Path


def check_progress():
    """Check and display tokenization progress."""
    input_dir = Path("data/cleaned")
    output_dir = Path("data_tokenized_production")
    
    print("🔍 TOKENIZATION PROGRESS CHECK")
    print("=" * 50)
    
    # Get all input files
    all_files = list(input_dir.glob("*.jsonl"))
    total_files = len(all_files)
    
    # Get completed files
    completed = set()
    if output_dir.exists():
        for npz_file in output_dir.glob("*.npz"):
            filename = npz_file.name
            if filename.startswith("data_tokenized_tokenized_cleaned_"):
                name_part = filename[len("data_tokenized_tokenized_cleaned_"):]
                if "_part" in name_part:
                    original_name = name_part.split("_part")[0]
                    completed.add(f"{original_name}.jsonl")
    
    completed_count = len(completed)
    remaining_count = total_files - completed_count
    
    print(f"📊 Total files: {total_files}")
    print(f"✅ Completed: {completed_count} ({completed_count/total_files*100:.1f}%)")
    print(f"⏳ Remaining: {remaining_count} ({remaining_count/total_files*100:.1f}%)")
    print()
    
    if remaining_count > 0:
        print(f"🚀 Ready to resume with {remaining_count} files!")
        print("Run: python resume_tokenization.py")
    else:
        print("🎉 All files completed!")
    
    print("=" * 50)


if __name__ == "__main__":
    check_progress() 