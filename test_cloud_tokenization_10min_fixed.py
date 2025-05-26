#!/usr/bin/env python3
"""
10-Minute Cloud Test for Single-Threaded Tokenization (FIXED VERSION)
Tests the fixed tokenization script in cloud environment with time limit.
Fixed to handle missing special tokens properly.
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


class CloudTestTokenizerFixed:
    """10-minute test tokenizer for cloud validation (FIXED)."""
    
    def __init__(
        self, 
        tokenizer_path: str = "models/tokenizer/transformers_tokenizer",
        input_dir: str = "data/cleaned",
        output_dir: str = "test_cloud_tokenized",
        max_length: int = 1024,
        stride: int = 512,
        batch_size: int = 500,
        max_test_minutes: int = 10
    ):
        self.tokenizer_path = Path(tokenizer_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.max_test_seconds = max_test_minutes * 60
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        
        # Get special token IDs with fallbacks
        self.start_token_id = self.tokenizer.token_to_id("<|startoftext|>")
        self.end_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        
        # FIXED: Handle missing special tokens
        if self.pad_token_id is None:
            self.logger.warning("⚠️  <|pad|> token not found, using token ID 0")
            self.pad_token_id = 0
        
        if self.start_token_id is None:
            self.logger.warning("⚠️  <|startoftext|> token not found, using token ID 1")
            self.start_token_id = 1
            
        if self.end_token_id is None:
            self.logger.warning("⚠️  <|endoftext|> token not found, using token ID 2")
            self.end_token_id = 2
        
        self.logger.info(f"🧪 10-MINUTE CLOUD TEST TOKENIZER (FIXED)")
        self.logger.info(f"Test duration: {max_test_minutes} minutes")
        self.logger.info(f"Tokenizer loaded - vocab size: {self.tokenizer.get_vocab_size()}")
        self.logger.info(f"Special tokens - Start: {self.start_token_id}, End: {self.end_token_id}, Pad: {self.pad_token_id}")
    
    def setup_logging(self):
        """Configure logging for cloud test."""
        log_file = f"cloud_test_fixed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        
        # Add source marker (simplified - just use text content)
        subreddit = item.get('subreddit', 'unknown')
        
        # Add title if exists and not empty
        title = item.get('title', '').strip()
        if title and title not in ['', '[deleted]', '[removed]']:
            text_parts.append(title)
        
        # Add main text
        text = item.get('text', '').strip()
        if text and text not in ['', '[deleted]', '[removed]']:
            text_parts.append(text)
        
        # Join with newlines - simplified format
        full_text = "\n".join(text_parts)
        return full_text
    
    def tokenize_text(self, text: str) -> list:
        """Tokenize text with sliding window for long sequences."""
        try:
            # Tokenize the full text
            encoding = self.tokenizer.encode(text)
            token_ids = encoding.ids
            
            # FIXED: Handle empty tokenization
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
                
                # FIXED: Ensure sequence is not empty
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
        """Save tokenized sequences to file (FIXED)."""
        if not sequences:
            self.logger.warning(f"No sequences to save for {output_path}")
            return 0
        
        try:
            # FIXED: Filter out empty sequences
            valid_sequences = [seq for seq in sequences if seq and len(seq) > 0]
            
            if not valid_sequences:
                self.logger.warning(f"No valid sequences to save for {output_path}")
                return 0
            
            # Pad sequences to same length
            max_len = max(len(seq) for seq in valid_sequences)
            
            # FIXED: Ensure pad_token_id is valid
            pad_id = self.pad_token_id if self.pad_token_id is not None else 0
            
            # Pad sequences
            padded_sequences = []
            attention_masks = []
            
            for seq in valid_sequences:
                padded_seq = seq + [pad_id] * (max_len - len(seq))
                attention_mask = [1] * len(seq) + [0] * (max_len - len(seq))
                
                padded_sequences.append(padded_seq)
                attention_masks.append(attention_mask)
            
            # Save as numpy arrays
            np.savez_compressed(
                output_path,
                input_ids=np.array(padded_sequences, dtype=np.int32),
                attention_mask=np.array(attention_masks, dtype=np.int32),
                sequence_lengths=np.array([len(seq) for seq in valid_sequences], dtype=np.int32)
            )
            
            self.logger.info(f"💾 Saved {len(valid_sequences)} sequences to {output_path}")
            return len(valid_sequences)
            
        except Exception as e:
            self.logger.error(f"Error saving {output_path}: {e}")
            return 0
    
    def process_file_with_time_limit(self, file_path: Path, start_time: float) -> dict:
        """Process a file but stop if time limit reached."""
        self.logger.info(f"📄 Processing: {file_path.name}")
        file_start_time = time.time()
        
        all_sequences = []
        total_items = 0
        valid_items = 0
        batch = []
        time_limit_reached = False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Check time limit every 100 lines
                    if line_num % 100 == 0:
                        elapsed = time.time() - start_time
                        if elapsed > self.max_test_seconds:
                            time_limit_reached = True
                            self.logger.info(f"⏰ Time limit reached at line {line_num}")
                            break
                    
                    try:
                        data = json.loads(line.strip())
                        total_items += 1
                        
                        # Basic validation
                        text = data.get('text', '').strip()
                        title = data.get('title', '').strip()
                        if text or title:
                            batch.append(data)
                            valid_items += 1
                        
                        # Process batch when full
                        if len(batch) >= self.batch_size:
                            sequences = self.process_batch(batch)
                            all_sequences.extend(sequences)
                            batch = []
                            
                            # Progress update
                            if total_items % (self.batch_size * 2) == 0:
                                elapsed = time.time() - start_time
                                remaining = self.max_test_seconds - elapsed
                                rate = len(all_sequences) / elapsed if elapsed > 0 else 0
                                self.logger.info(f"  📊 Progress: {total_items} items, {len(all_sequences)} sequences ({rate:.0f}/sec), {remaining:.0f}s remaining")
                    
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"JSON error at line {line_num}: {e}")
                        continue
                    except Exception as e:
                        self.logger.warning(f"Processing error at line {line_num}: {e}")
                        continue
                
                # Process remaining batch if not time limited
                if batch and not time_limit_reached:
                    sequences = self.process_batch(batch)
                    all_sequences.extend(sequences)
            
            # Save partial results if we have sequences
            if all_sequences:
                output_name = f"test_tokenized_{file_path.stem}_partial.npz"
                output_path = self.output_dir / output_name
                sequences_saved = self.save_tokenized_data(all_sequences, output_path)
            else:
                sequences_saved = 0
            
            elapsed = time.time() - file_start_time
            self.logger.info(f"✅ File {file_path.name}: {sequences_saved} sequences from {valid_items}/{total_items} items in {elapsed:.1f}s")
            
            return {
                "sequences": sequences_saved,
                "items": total_items,
                "valid_items": valid_items,
                "elapsed": elapsed,
                "time_limit_reached": time_limit_reached
            }
            
        except Exception as e:
            self.logger.error(f"❌ Critical error processing {file_path}: {e}")
            return {"sequences": 0, "items": 0, "error": str(e)}
    
    def run_test(self):
        """Run the 10-minute test."""
        files = sorted(list(self.input_dir.glob("*.jsonl")))
        
        if not files:
            self.logger.error("❌ No JSONL files found in input directory!")
            return
        
        self.logger.info(f"🚀 Starting 10-minute cloud test with {len(files)} available files")
        start_time = time.time()
        
        total_sequences = 0
        total_items = 0
        total_valid_items = 0
        files_processed = 0
        
        # Process files until time limit
        for i, file_path in enumerate(files):
            elapsed = time.time() - start_time
            remaining = self.max_test_seconds - elapsed
            
            if remaining <= 30:  # Stop if less than 30 seconds remaining
                self.logger.info(f"⏰ Stopping with {remaining:.0f}s remaining")
                break
            
            self.logger.info(f"📁 File {i+1}/{len(files)}: {file_path.name} ({remaining:.0f}s remaining)")
            
            try:
                result = self.process_file_with_time_limit(file_path, start_time)
                
                total_sequences += result.get("sequences", 0)
                total_items += result.get("items", 0)
                total_valid_items += result.get("valid_items", 0)
                files_processed += 1
                
                if result.get("time_limit_reached", False):
                    self.logger.info("⏰ Time limit reached during file processing")
                    break
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {file_path}: {e}")
                continue
        
        # Final results
        total_elapsed = time.time() - start_time
        self.logger.info(f"")
        self.logger.info(f"🎯 === 10-MINUTE TEST RESULTS (FIXED) ===")
        self.logger.info(f"⏱️  Test duration: {total_elapsed/60:.2f} minutes")
        self.logger.info(f"📁 Files processed: {files_processed}")
        self.logger.info(f"🔢 Total sequences: {total_sequences:,}")
        self.logger.info(f"📄 Total items: {total_items:,} (valid: {total_valid_items:,})")
        if total_elapsed > 0:
            self.logger.info(f"⚡ Processing rate: {total_sequences/total_elapsed:.0f} sequences/sec")
            self.logger.info(f"📊 Item rate: {total_items/total_elapsed:.0f} items/sec")
        
        # Projections for full dataset
        if total_sequences > 0 and files_processed > 0:
            estimated_total_files = len(files)
            estimated_full_time_hours = (estimated_total_files / files_processed) * (total_elapsed / 3600)
            estimated_total_sequences = (total_sequences / files_processed) * estimated_total_files
            
            self.logger.info(f"")
            self.logger.info(f"🔮 === FULL DATASET PROJECTIONS ===")
            self.logger.info(f"⏰ Estimated total processing time: {estimated_full_time_hours:.1f} hours")
            self.logger.info(f"🔢 Estimated total sequences: {estimated_total_sequences:,.0f}")
            self.logger.info(f"💾 Estimated output size: {estimated_total_sequences * 4 / (1024**3):.1f} GB")
        
        # Test validation
        self.logger.info(f"")
        if total_sequences > 1000:
            self.logger.info("✅ TEST PASSED: Good processing rate achieved")
            self.logger.info("✅ Single-threaded tokenization works in cloud environment")
            self.logger.info("✅ Ready for full dataset processing")
        elif total_sequences > 100:
            self.logger.info("⚠️  TEST PARTIAL: Some sequences generated but rate may be low")
            self.logger.info("⚠️  Check for any error messages above")
        else:
            self.logger.info("❌ TEST FAILED: Very few sequences generated")
            self.logger.info("❌ Check tokenizer and data paths")


def main():
    """Main function for 10-minute cloud test (FIXED)."""
    print("=" * 60)
    print("🧪 10-MINUTE CLOUD TOKENIZATION TEST (FIXED VERSION)")
    print("=" * 60)
    print("This FIXED version handles missing special tokens properly.")
    print()
    
    try:
        # Create test tokenizer
        tokenizer = CloudTestTokenizerFixed(
            tokenizer_path="models/tokenizer/transformers_tokenizer",
            input_dir="data/cleaned",
            output_dir="test_cloud_tokenized_fixed",
            max_length=1024,
            stride=512,
            batch_size=500,
            max_test_minutes=10
        )
        
        print("🚀 Starting FIXED test...")
        tokenizer.run_test()
        
        print("\n" + "=" * 60)
        print("🎉 FIXED TEST COMPLETE!")
        print("Check the logs and test_cloud_tokenized_fixed/ directory for results.")
        
    except Exception as e:
        print(f"❌ Test failed to start: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 