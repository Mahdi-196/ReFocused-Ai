#!/usr/bin/env python3
"""
Data Tokenization Script for GPT Model Training
Tokenizes all cleaned data using the trained BPE tokenizer.
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Iterator, Tuple
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


class DataTokenizer:
    """Handles tokenization of cleaned data for model training."""
    
    def __init__(
        self, 
        tokenizer_path: str = "models/tokenizer/tokenizer",
        input_dir: str = "data/cleaned",
        output_dir: str = "data_tokenized",
        max_length: int = 1024,
        stride: int = 512,
        batch_size: int = 1000
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
        
        self.logger.info(f"Tokenizer loaded with vocab size: {self.tokenizer.get_vocab_size()}")
        self.logger.info(f"Special tokens - Start: {self.start_token_id}, End: {self.end_token_id}, Pad: {self.pad_token_id}")
    
    def setup_logging(self):
        """Configure logging."""
        log_file = f"tokenization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_tokenizer(self) -> Tokenizer:
        """Load the trained tokenizer."""
        tokenizer_file = self.tokenizer_path / "tokenizer.json"
        if not tokenizer_file.exists():
            # Try alternative paths
            alt_paths = [
                "models/tokenizer/transformers_tokenizer/tokenizer.json",
                "tokenizer_750M/tokenizer.json"
            ]
            for alt_path in alt_paths:
                if Path(alt_path).exists():
                    tokenizer_file = Path(alt_path)
                    break
            else:
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_file} or alternative paths")
        
        self.logger.info(f"Loading tokenizer from: {tokenizer_file}")
        return Tokenizer.from_file(str(tokenizer_file))
    
    def get_input_files(self) -> List[Path]:
        """Get all cleaned data files to process."""
        files = list(self.input_dir.glob("*.jsonl"))
        files.sort()
        self.logger.info(f"Found {len(files)} files to process")
        return files
    
    def read_jsonl_batch(self, file_path: Path, batch_size: int = 1000) -> Iterator[List[Dict]]:
        """Read JSONL file in batches."""
        batch = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    batch.append(data)
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Error parsing line in {file_path}: {e}")
                    continue
        
        if batch:  # Yield remaining items
            yield batch
    
    def prepare_text(self, item: Dict) -> str:
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
    
    def tokenize_text(self, text: str) -> List[List[int]]:
        """Tokenize text with sliding window for long sequences."""
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
    
    def process_batch(self, batch: List[Dict]) -> List[List[int]]:
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
    
    def save_tokenized_data(self, sequences: List[List[int]], output_path: Path):
        """Save tokenized sequences to file."""
        # Convert to numpy array for efficient storage
        # Pad sequences to same length
        max_len = max(len(seq) for seq in sequences) if sequences else 0
        
        if max_len == 0:
            self.logger.warning(f"No sequences to save for {output_path}")
            return
        
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
    
    def process_file(self, file_path: Path) -> Dict[str, int]:
        """Process a single file."""
        self.logger.info(f"Processing file: {file_path}")
        start_time = time.time()
        
        # Generate output filename
        output_name = f"tokenized_{file_path.stem}.npz"
        output_path = self.output_dir / output_name
        
        # Skip if already processed
        if output_path.exists():
            self.logger.info(f"Skipping {file_path} - already processed")
            return {"sequences": 0, "skipped": True}
        
        all_sequences = []
        total_items = 0
        
        # Process file in batches
        try:
            for batch in self.read_jsonl_batch(file_path, self.batch_size):
                sequences = self.process_batch(batch)
                all_sequences.extend(sequences)
                total_items += len(batch)
                
                # Log progress every 10 batches
                if total_items % (self.batch_size * 10) == 0:
                    self.logger.info(f"Processed {total_items} items, generated {len(all_sequences)} sequences")
        
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return {"sequences": 0, "error": str(e)}
        
        # Save tokenized data
        if all_sequences:
            self.save_tokenized_data(all_sequences, output_path)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Completed {file_path}: {len(all_sequences)} sequences from {total_items} items in {elapsed:.2f}s")
        
        return {
            "sequences": len(all_sequences),
            "items": total_items,
            "elapsed": elapsed,
            "skipped": False
        }
    
    def process_all_files(self, num_workers: int = None):
        """Process all files with multiprocessing."""
        files = self.get_input_files()
        
        if not files:
            self.logger.warning("No files found to process")
            return
        
        if num_workers is None:
            num_workers = min(mp.cpu_count() - 1, len(files), 4)  # Leave one core free
        
        self.logger.info(f"Starting tokenization with {num_workers} workers")
        start_time = time.time()
        
        total_sequences = 0
        total_items = 0
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {executor.submit(self.process_file, file_path): file_path for file_path in files}
            
            with tqdm(total=len(files), desc="Processing files") as pbar:
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if not result.get("skipped", False):
                            total_sequences += result.get("sequences", 0)
                            total_items += result.get("items", 0)
                        pbar.update(1)
                    except Exception as e:
                        self.logger.error(f"Error processing {file_path}: {e}")
                        pbar.update(1)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Tokenization complete!")
        self.logger.info(f"Total sequences: {total_sequences:,}")
        self.logger.info(f"Total items processed: {total_items:,}")
        self.logger.info(f"Total time: {elapsed:.2f}s")
        
        # Save processing summary
        summary = {
            "total_files": len(files),
            "total_sequences": total_sequences,
            "total_items": total_items,
            "processing_time": elapsed,
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


def main():
    parser = argparse.ArgumentParser(description="Tokenize cleaned data for model training")
    parser.add_argument("--input-dir", default="data/cleaned", help="Input directory with cleaned JSONL files")
    parser.add_argument("--output-dir", default="data_tokenized", help="Output directory for tokenized data")
    parser.add_argument("--tokenizer-path", default="models/tokenizer/tokenizer", help="Path to trained tokenizer")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--stride", type=int, default=512, help="Sliding window stride")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    
    args = parser.parse_args()
    
    # Create tokenizer
    tokenizer = DataTokenizer(
        tokenizer_path=args.tokenizer_path,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_length=args.max_length,
        stride=args.stride,
        batch_size=args.batch_size
    )
    
    # Process all files
    tokenizer.process_all_files(num_workers=args.workers)


if __name__ == "__main__":
    main() 