#!/usr/bin/env python3
"""
Emergency Resume Script for Disk Space Recovery
Resumes tokenization after disk space issues, skipping failed files
"""

import json
import logging
import numpy as np
import os
import time
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

class EmergencyResume:
    def __init__(self):
        # Setup logging
        log_filename = f"emergency_resume_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Paths
        self.input_dir = Path("data/cleaned")
        self.output_dir = Path("data_tokenized_production")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load tokenizer
        self.logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
        
        # Files that failed due to disk space
        self.failed_files = {
            "cleaned_transmanlifehacks_comments_part001.jsonl",
            "cleaned_transmanlifehacks_submissions_part001.jsonl"
        }
        
        # Last successfully processed file (based on your logs)
        self.last_successful = "cleaned_trailrunning_submissions_part001.jsonl"
        
    def get_remaining_files(self):
        """Get list of files that still need processing"""
        all_files = sorted([f for f in self.input_dir.glob("*.jsonl")])
        completed_files = set()
        
        # Check which files already have output
        for output_file in self.output_dir.glob("*.npz"):
            # Extract original filename from output filename
            # Format: tokenized_cleaned_filename_part001.npz
            name_parts = output_file.stem.split('_')
            if len(name_parts) >= 3:
                # Reconstruct original filename
                original_name = '_'.join(name_parts[2:-1]) + '.jsonl'
                completed_files.add(original_name)
        
        # Get remaining files
        remaining = []
        found_last_successful = False
        
        for file_path in all_files:
            filename = file_path.name
            
            # Skip if already completed
            if filename in completed_files:
                continue
                
            # If we haven't found our last successful file yet, skip
            if not found_last_successful:
                if filename == self.last_successful:
                    found_last_successful = True
                continue
            
            # Skip the known failed files for now
            if filename in self.failed_files:
                self.logger.warning(f"⚠️ Skipping failed file: {filename}")
                continue
                
            remaining.append(file_path)
        
        return remaining
    
    def check_disk_space(self):
        """Check available disk space"""
        import shutil
        total, used, free = shutil.disk_usage(self.output_dir)
        free_gb = free // (1024**3)
        self.logger.info(f"💽 Available disk space: {free_gb} GB")
        return free_gb > 1  # Require at least 1GB free
    
    def process_file(self, file_path):
        """Process a single file with disk space monitoring"""
        filename = file_path.name
        self.logger.info(f"📂 Processing: {filename}")
        
        # Check disk space before processing
        if not self.check_disk_space():
            self.logger.error("❌ Insufficient disk space!")
            return False
        
        sequences = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        text = data.get('text', '')
                        
                        if text and len(text.strip()) > 10:
                            # Tokenize
                            tokens = self.tokenizer.encode(
                                text,
                                max_length=512,
                                truncation=True,
                                padding=False
                            )
                            
                            if len(tokens) > 5:
                                sequences.append(tokens)
                        
                        # Log progress
                        if line_num % 5000 == 0:
                            self.logger.info(f"  📊 Processed {line_num:,} items, {len(sequences):,} sequences")
                            
                            # Check disk space during processing
                            if not self.check_disk_space():
                                self.logger.error("❌ Disk space exhausted during processing!")
                                return False
                    
                    except Exception as e:
                        continue  # Skip problematic lines
            
            # Save sequences
            if sequences:
                output_name = f"tokenized_{filename.replace('.jsonl', '_part001.npz')}"
                output_path = self.output_dir / output_name
                
                try:
                    np.savez_compressed(output_path, sequences=sequences)
                    self.logger.info(f"💾 Saved {len(sequences)} sequences to {output_path}")
                    return True
                    
                except OSError as e:
                    if "No space left" in str(e):
                        self.logger.error(f"❌ Failed to save {output_name}: No space left on device")
                        return False
                    raise
            else:
                self.logger.warning(f"⚠️ No valid sequences found in {filename}")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error processing {filename}: {e}")
            return False
    
    def resume_processing(self):
        """Resume processing from where we left off"""
        remaining_files = self.get_remaining_files()
        
        if not remaining_files:
            self.logger.info("🎉 All files appear to be processed!")
            return
        
        self.logger.info(f"📋 Found {len(remaining_files)} files to process")
        
        # Process remaining files
        for file_path in tqdm(remaining_files, desc="Processing files"):
            success = self.process_file(file_path)
            
            if not success:
                self.logger.error(f"💥 Stopping due to error processing {file_path.name}")
                break
                
        self.logger.info("✅ Emergency resume completed!")

def main():
    resumer = EmergencyResume()
    resumer.resume_processing()

if __name__ == "__main__":
    main() 