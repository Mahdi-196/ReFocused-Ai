#!/usr/bin/env python3
"""
Resume Tokenization Script
Continues tokenization from where it left off without redoing completed files.
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


class ResumeTokenizer:
    """Resume tokenization from where it left off."""
    
    def __init__(
        self, 
        tokenizer_path: str = "models/tokenizer/transformers_tokenizer",
        input_dir: str = "data/cleaned",
        output_dir: str = "data_tokenized_production",
        max_length: int = 1024,
        stride: int = 512,
        batch_size: int = 500
    ):
        self.tokenizer_path = Path(tokenizer_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        
        # Get special token IDs with fallbacks
        self.start_token_id = self.tokenizer.token_to_id("<|startoftext|>")
        self.end_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        
        # Handle missing special tokens
        if self.pad_token_id is None:
            self.logger.warning("⚠️  <|pad|> token not found, using token ID 0")
            self.pad_token_id = 0
        
        if self.start_token_id is None:
            self.logger.warning("⚠️  <|startoftext|> token not found, using token ID 1")
            self.start_token_id = 1
            
        if self.end_token_id is None:
            self.logger.warning("⚠️  <|endoftext|> token not found, using token ID 2")
            self.end_token_id = 2
        
        self.logger.info(f"🔄 RESUME TOKENIZATION")
        self.logger.info(f"Tokenizer loaded - vocab size: {self.tokenizer.get_vocab_size()}")
        self.logger.info(f"Special tokens - Start: {self.start_token_id}, End: {self.end_token_id}, Pad: {self.pad_token_id}")
    
    def setup_logging(self):
        """Configure logging."""
        log_file = f"resume_tokenization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_tokenizer(self):
        """Load the trained tokenizer."""
        tokenizer_file = self.tokenizer_path / "tokenizer.json"
        if not tokenizer_file.exists():
            # Try alternative paths
            alt_paths = [
                "models/tokenizer/transformers_tokenizer/tokenizer.json",
                "models/tokenizer/tokenizer.json",
                "tokenizer_750M/tokenizer.json"
            ]
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    tokenizer_file = Path(alt_path)
                    self.logger.info(f"Found tokenizer at: {tokenizer_file}")
                    break
            else:
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file}")
        
        return Tokenizer.from_file(str(tokenizer_file))
    
    def get_completed_files(self) -> set:
        """Get list of files that have already been processed."""
        completed = set()
        
        if not self.output_dir.exists():
            self.logger.info("Output directory doesn't exist yet")
            return completed
        
        # Look for .npz files in output directory
        for npz_file in self.output_dir.glob("*.npz"):
            # Extract original filename from tokenized filename
            # Format: data_tokenized_tokenized_cleaned_[ORIGINAL_NAME]_part[N].npz
            filename = npz_file.name
            
            # Remove prefix and suffix to get original name
            if filename.startswith("data_tokenized_tokenized_cleaned_"):
                # Remove prefix
                name_part = filename[len("data_tokenized_tokenized_cleaned_"):]
                # Remove part and extension
                if "_part" in name_part:
                    original_name = name_part.split("_part")[0]
                    completed.add(f"{original_name}.jsonl")
        
        self.logger.info(f"Found {len(completed)} already completed files")
        return completed
    
    def get_remaining_files(self) -> list:
        """Get list of files that still need to be processed."""
        completed = self.get_completed_files()
        
        # Get all .jsonl files in input directory
        all_files = list(self.input_dir.glob("*.jsonl"))
        
        # Filter out completed files
        remaining = []
        for file_path in all_files:
            if file_path.name not in completed:
                remaining.append(file_path)
        
        remaining.sort()  # Process in consistent order
        
        self.logger.info(f"Total files: {len(all_files)}")
        self.logger.info(f"Completed files: {len(completed)}")
        self.logger.info(f"Remaining files: {len(remaining)}")
        
        return remaining
    
    def prepare_text(self, item: dict) -> str:
        """Prepare text from JSON item for tokenization."""
        text_parts = []
        
        # Add title if exists and not empty
        title = item.get('title', '').strip()
        if title and title not in ['', '[deleted]', '[removed]']:
            text_parts.append(title)
        
        # Add main text
        text = item.get('text', '').strip()
        if text and text not in ['', '[deleted]', '[removed]']:
            text_parts.append(text)
        
        # Join with newlines
        full_text = "\n".join(text_parts)
        return full_text
    
    def tokenize_text(self, text: str) -> list:
        """Tokenize text with sliding window for long sequences."""
        try:
            # Tokenize the full text
            encoding = self.tokenizer.encode(text)
            token_ids = encoding.ids
            
            # Handle empty tokenization
            if not token_ids or len(token_ids) == 0:
                return []
            
            # If sequence is shorter than max_length, return as is
            if len(token_ids) <= self.max_length:
                return [token_ids]
            
            # Split into overlapping windows
            sequences = []
            start = 0
            while start < len(token_ids):
                end = min(start + self.max_length, len(token_ids))
                sequence = token_ids[start:end]
                
                # Ensure sequence is not empty
                if len(sequence) > 0:
                    sequences.append(sequence)
                
                # Move start by stride, but ensure we capture the end
                if end == len(token_ids):
                    break
                start += self.stride
            
            return sequences
        except Exception as e:
            self.logger.warning(f"Error tokenizing text: {e}")
            return []
    
    def process_batch(self, batch: list) -> list:
        """Process a batch of items and return tokenized sequences."""
        all_sequences = []
        
        for item in batch:
            try:
                # Prepare text
                text = self.prepare_text(item)
                
                # Skip if text is too short
                if len(text.strip()) < 10:
                    continue
                
                # Tokenize
                sequences = self.tokenize_text(text)
                all_sequences.extend(sequences)
                
            except Exception as e:
                self.logger.warning(f"Error processing item: {e}")
                continue
        
        return all_sequences
    
    def save_tokenized_data(self, sequences: list, output_path: Path):
        """Save tokenized sequences to file."""
        if not sequences:
            self.logger.warning(f"No sequences to save for {output_path}")
            return 0
        
        try:
            # Filter out empty sequences
            valid_sequences = [seq for seq in sequences if seq and len(seq) > 0]
            
            if not valid_sequences:
                self.logger.warning(f"No valid sequences to save for {output_path}")
                return 0
            
            # Pad sequences to same length
            max_len = max(len(seq) for seq in valid_sequences)
            padded_sequences = []
            
            for seq in valid_sequences:
                padded = seq + [self.pad_token_id] * (max_len - len(seq))
                padded_sequences.append(padded)
            
            # Convert to numpy array and save
            token_array = np.array(padded_sequences, dtype=np.int32)
            np.savez_compressed(output_path, data=token_array)
            
            self.logger.info(f"💾 Saved {len(padded_sequences)} sequences to {output_path}")
            return len(padded_sequences)
            
        except Exception as e:
            self.logger.error(f"Error saving {output_path}: {e}")
            return 0
    
    def process_file(self, file_path: Path) -> dict:
        """Process a single file and return results."""
        self.logger.info(f"📂 Processing: {file_path.name}")
        
        stats = {
            'filename': file_path.name,
            'total_items': 0,
            'valid_items': 0,
            'total_sequences': 0,
            'processing_time': 0,
            'success': False
        }
        
        start_time = time.time()
        
        try:
            # Read and process file
            with open(file_path, 'r', encoding='utf-8') as f:
                all_sequences = []
                batch = []
                
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        batch.append(item)
                        stats['total_items'] += 1
                        
                        # Process batch when full
                        if len(batch) >= self.batch_size:
                            sequences = self.process_batch(batch)
                            all_sequences.extend(sequences)
                            stats['valid_items'] += len([item for item in batch if self.prepare_text(item).strip()])
                            batch = []
                            
                            # Log progress every 10 batches
                            if line_num % (self.batch_size * 10) == 0:
                                self.logger.info(f"  📊 Processed {line_num:,} items, {len(all_sequences):,} sequences")
                        
                    except json.JSONDecodeError:
                        continue
                
                # Process remaining batch
                if batch:
                    sequences = self.process_batch(batch)
                    all_sequences.extend(sequences)
                    stats['valid_items'] += len([item for item in batch if self.prepare_text(item).strip()])
                
                # Save results
                if all_sequences:
                    base_name = file_path.stem
                    output_path = self.output_dir / f"data_tokenized_tokenized_cleaned_{base_name}_part001.npz"
                    saved_count = self.save_tokenized_data(all_sequences, output_path)
                    stats['total_sequences'] = saved_count
                    stats['success'] = True
                else:
                    self.logger.warning(f"No sequences generated for {file_path.name}")
            
            stats['processing_time'] = time.time() - start_time
            
            self.logger.info(f"✅ {file_path.name}: {stats['total_items']:,} items → {stats['total_sequences']:,} sequences in {stats['processing_time']:.1f}s")
            
        except Exception as e:
            self.logger.error(f"❌ Error processing {file_path.name}: {e}")
            stats['processing_time'] = time.time() - start_time
        
        return stats
    
    def run_resume(self):
        """Run the resume tokenization process."""
        start_time = time.time()
        
        # Get remaining files to process
        remaining_files = self.get_remaining_files()
        
        if not remaining_files:
            self.logger.info("🎉 All files already completed!")
            return
        
        self.logger.info(f"🚀 Starting to process {len(remaining_files)} remaining files...")
        
        total_stats = {
            'files_processed': 0,
            'total_items': 0,
            'total_sequences': 0,
            'failed_files': 0
        }
        
        # Process each remaining file
        with tqdm(remaining_files, desc="Processing files") as pbar:
            for file_path in pbar:
                try:
                    stats = self.process_file(file_path)
                    
                    # Update totals
                    total_stats['files_processed'] += 1
                    total_stats['total_items'] += stats['total_items']
                    total_stats['total_sequences'] += stats['total_sequences']
                    
                    if not stats['success']:
                        total_stats['failed_files'] += 1
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'Sequences': f"{total_stats['total_sequences']:,}",
                        'Failed': total_stats['failed_files']
                    })
                
                except Exception as e:
                    self.logger.error(f"❌ Failed to process {file_path.name}: {e}")
                    total_stats['failed_files'] += 1
        
        # Final summary
        total_time = time.time() - start_time
        
        self.logger.info("=" * 60)
        self.logger.info("🎯 RESUME TOKENIZATION COMPLETE")
        self.logger.info(f"Files processed: {total_stats['files_processed']:,}")
        self.logger.info(f"Total items: {total_stats['total_items']:,}")
        self.logger.info(f"Total sequences: {total_stats['total_sequences']:,}")
        self.logger.info(f"Failed files: {total_stats['failed_files']:,}")
        self.logger.info(f"Total time: {total_time/3600:.2f} hours")
        if total_stats['total_sequences'] > 0:
            self.logger.info(f"Processing rate: {total_stats['total_sequences']/total_time:.0f} sequences/sec")
        self.logger.info("=" * 60)


def main():
    """Main entry point."""
    try:
        tokenizer = ResumeTokenizer()
        tokenizer.run_resume()
    except KeyboardInterrupt:
        print("\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main() 