#!/usr/bin/env python3
"""
Quick fix script to update all remote_path references to use the root bucket path
instead of "tokenized_data"
"""

import os
import re
import glob
from pathlib import Path

def fix_file(file_path, fix_remote_path=True):
    """Fix remote_path references in a file"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix remote_path references
    if fix_remote_path:
        content = re.sub(r'remote_data_path\s*:\s*[\'"]tokenized_data[\'"]', 'remote_data_path: ""', content)
        content = re.sub(r'remote_path\s*=\s*[\'"]tokenized_data[\'"]', 'remote_path=""', content)
        content = re.sub(r'DATA_REMOTE_PATH\s*=\s*[\'"]tokenized_data[\'"]', 'DATA_REMOTE_PATH=""', content)
        content = re.sub(r'--remote_path\s+tokenized_data', '', content)
        
    # Write the updated content
    with open(file_path, 'w') as f:
        f.write(content)
    
    return True

def find_and_fix_files(directory):
    """Find and fix all relevant files in the directory"""
    fixed_files = []
    
    # Find all Python, YAML, JSON, and shell script files
    patterns = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.sh", "**/*.md"]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(directory, pattern), recursive=True))
    
    # Fix each file
    for file_path in all_files:
        try:
            fixed = fix_file(file_path)
            if fixed:
                fixed_files.append(file_path)
        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
    
    return fixed_files

if __name__ == "__main__":
    # Fix files in the current directory and subdirectories
    directory = os.path.dirname(os.path.abspath(__file__))
    fixed_files = find_and_fix_files(directory)
    
    print(f"Fixed {len(fixed_files)} files")
    for file in fixed_files:
        print(f"- {file}") 