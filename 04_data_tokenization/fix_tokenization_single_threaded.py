#!/usr/bin/env python3
"""
Fixed single-threaded tokenization script for cloud environments.
This version avoids multiprocessing issues and provides detailed progress tracking.
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


class SingleThreadedTokenizer:
    """Single-threaded tokenizer that works reliably in any environment."""
    
    def __init__(
        self, 
        tokenizer_path: str = "models/tokenizer/transformers_tokenizer",
        input_dir: str = "data/cleaned",
        output_dir: str = "data_tokenized",
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
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        
        # Get special token IDs
        self.start_token_id = self.tokenizer.token_to_id("<|startoftext|>")
        self.end_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        
        self.logger.info(f"Tokenizer loaded - vocab size: {self.tokenizer.get_vocab_size()}")
        self.logger.info(f"Special tokens - Start: {self.start_token_id}, End: {self.end_token_id}, Pad: {self.pad_token_id}")
    
    def setup_logging(self):
        """Configure logging with plain ASCII characters."""
        log_file = f"tokenization_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    
    def prepare_text(self, item: dict) -> str:
        """Prepare text from JSON item for tokenization."""
        text_parts = []
        
        # Add source marker
        subreddit = item.get('subreddit', 'unknown')
        if 'reddit' in subreddit.lower() or subreddit != 'openwebtext':
            text_parts.append("<|reddit|>")
        else:
            text_parts.append("<|hf|>")
        
        # Add title if exists and not empty
        title = item.get('title', '').strip()
        if title and title not in ['', '[deleted]', '[removed]']:
            text_parts.append(title)
        
        # Add main text
        text = item.get('text', '').strip()
        if text and text not in ['', '[deleted]', '[removed]']:
            text_parts.append(text)
        
        # Join with newlines and add markers
        full_text = "<|startoftext|>" + "\n".join(text_parts) + "<|endoftext|>"
        return full_text
    
    def tokenize_text(self, text: str) -> list:
        """Tokenize text with sliding window for long sequences."""
        try:
            # Tokenize the full text
            encoding = self.tokenizer.encode(text)
            token_ids = encoding.ids
            
            # If sequence is shorter than max_length, return as is
            if len(token_ids) <= self.max_length:
                return [token_ids]
            
            # Split into overlapping windows
            sequences = []
            start = 0
            while start < len(token_ids):
                end = min(start + self.max_length, len(token_ids))
                sequence = token_ids[start:end]
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
                if len(text.strip()) < 20:  # Increased minimum length
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
            # Pad sequences to same length
            max_len = max(len(seq) for seq in sequences)
            
            # Pad sequences
            padded_sequences = []
            attention_masks = []
            
            for seq in sequences:
                padded_seq = seq + [self.pad_token_id] * (max_len - len(seq))
                attention_mask = [1] * len(seq) + [0] * (max_len - len(seq))
                
                padded_sequences.append(padded_seq)
                attention_masks.append(attention_mask)
            
            # Save as numpy arrays
            np.savez_compressed(
                output_path,
                input_ids=np.array(padded_sequences, dtype=np.int32),
                attention_mask=np.array(attention_masks, dtype=np.int32),
                sequence_lengths=np.array([len(seq) for seq in sequences], dtype=np.int32)
            )
            
            self.logger.info(f"Saved {len(sequences)} sequences to {output_path}")
            return len(sequences)
            
        except Exception as e:
            self.logger.error(f"Error saving {output_path}: {e}")
            return 0
    
    def process_file(self, file_path: Path) -> dict:
        """Process a single file with detailed progress tracking."""
        self.logger.info(f"Processing: {file_path.name}")
        start_time = time.time()
        
        # Generate output filename
        output_name = f"tokenized_{file_path.stem}.npz"
        output_path = self.output_dir / output_name
        
        # Skip if already processed
        if output_path.exists():
            self.logger.info(f"Skipping {file_path.name} - already processed")
            # Return info about existing file
            try:
                data = np.load(output_path)
                existing_sequences = data['input_ids'].shape[0]
                return {"sequences": existing_sequences, "items": 0, "skipped": True}
            except:
                self.logger.warning(f"Corrupted existing file {output_path}, reprocessing...")
        
        all_sequences = []
        total_items = 0
        valid_items = 0
        batch = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        total_items += 1
                        
                        # Basic validation
                        text = data.get('text', '').strip()
                        title = data.get('title', '').strip()
                        if text or title:  # Has some content
                            batch.append(data)
                            valid_items += 1
                        
                        # Process batch when full
                        if len(batch) >= self.batch_size:
                            sequences = self.process_batch(batch)
                            all_sequences.extend(sequences)
                            batch = []
                            
                            # Progress update
                            if total_items % (self.batch_size * 5) == 0:
                                self.logger.info(f"  Progress: {total_items} items, {len(all_sequences)} sequences")
                    
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Processing error at line {line_num}: {e}")
                        continue
                
                # Process remaining batch
                if batch:
                    sequences = self.process_batch(batch)
                    all_sequences.extend(sequences)
            
            # Save results
            sequences_saved = self.save_tokenized_data(all_sequences, output_path)
            
            elapsed = time.time() - start_time
            self.logger.info(f"Completed {file_path.name}: {sequences_saved} sequences from {valid_items}/{total_items} items in {elapsed:.1f}s")
            
            return {
                "sequences": sequences_saved,
                "items": total_items,
                "valid_items": valid_items,
                "elapsed": elapsed,
                "skipped": False
            }
            
        except Exception as e:
            self.logger.error(f"Critical error processing {file_path}: {e}")
            return {"sequences": 0, "items": 0, "error": str(e)}
    
    def process_all_files(self):
        """Process all files sequentially with progress tracking."""
        files = sorted(list(self.input_dir.glob("*.jsonl")))
        
        if not files:
            self.logger.error("No JSONL files found in input directory!")
            return
        
        self.logger.info(f"Starting single-threaded tokenization of {len(files)} files")
        start_time = time.time()
        
        total_sequences = 0
        total_items = 0
        total_valid_items = 0
        processed_files = 0
        skipped_files = 0
        
        # Process with progress bar
        for i, file_path in enumerate(tqdm(files, desc="Processing files"), 1):
            try:
                result = self.process_file(file_path)
                
                if result.get("skipped", False):
                    skipped_files += 1
                else:
                    processed_files += 1
                
                total_sequences += result.get("sequences", 0)
                total_items += result.get("items", 0)
                total_valid_items += result.get("valid_items", 0)
                
                # Progress summary every 50 files
                if i % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = total_sequences / elapsed if elapsed > 0 else 0
                    self.logger.info(f"Milestone: {i}/{len(files)} files, {total_sequences:,} sequences, {rate:.0f} seq/sec")
                
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        # Final summary
        elapsed = time.time() - start_time
        self.logger.info(f"TOKENIZATION COMPLETE!")
        self.logger.info(f"Files processed: {processed_files}, skipped: {skipped_files}")
        self.logger.info(f"Total sequences: {total_sequences:,}")
        self.logger.info(f"Total items: {total_items:,} (valid: {total_valid_items:,})")
        self.logger.info(f"Processing time: {elapsed/3600:.2f} hours")
        self.logger.info(f"Average rate: {total_sequences/elapsed:.0f} sequences/sec")
        
        # Save summary
        summary = {
            "total_files": len(files),
            "processed_files": processed_files,
            "skipped_files": skipped_files,
            "total_sequences": total_sequences,
            "total_items": total_items,
            "total_valid_items": total_valid_items,
            "processing_time": elapsed,
            "sequences_per_second": total_sequences / elapsed if elapsed > 0 else 0,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "max_length": self.max_length,
                "stride": self.stride,
                "batch_size": self.batch_size,
                "vocab_size": self.tokenizer.get_vocab_size()
            }
        }
        
        with open(self.output_dir / "tokenization_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Summary saved to {self.output_dir}/tokenization_summary.json")


def main():
    """Main function with user confirmation."""
    print("FIXED SINGLE-THREADED TOKENIZATION")
    print("=" * 50)
    print("This version fixes the multiprocessing issues and will work reliably.")
    print("Processing will be slower but stable.")
    
    # Ask for confirmation
    response = input("\nProceed with single-threaded tokenization? (y/N): ").strip().lower()
    if response != 'y':
        print("Cancelled by user")
        return
    
    # Create tokenizer with production settings
    tokenizer = SingleThreadedTokenizer(
        tokenizer_path="models/tokenizer/transformers_tokenizer",
        input_dir="data/cleaned",
        output_dir="data_tokenized",
        max_length=1024,
        stride=512,
        batch_size=500
    )
    
    print(f"\nConfiguration:")
    print(f"  Input directory: {tokenizer.input_dir}")
    print(f"  Output directory: {tokenizer.output_dir}")
    print(f"  Max sequence length: {tokenizer.max_length}")
    print(f"  Batch size: {tokenizer.batch_size}")
    print(f"  Mode: Single-threaded (stable)")
    
    # Start processing
    print(f"\nStarting tokenization...")
    tokenizer.process_all_files()
    
    print(f"\nDone! Check the logs and output directory for results.")


if __name__ == "__main__":
    main() 