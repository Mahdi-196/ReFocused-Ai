#!/usr/bin/env python3
"""
Production Single-Threaded Tokenization Script
Based on successful 10-minute test results.
Processes all files in the dataset using proven approach.
"""

import os
import json
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


class ProductionTokenizer:
    """Production tokenizer for full dataset processing."""
    
    def __init__(
        self, 
        tokenizer_path: str = "models/tokenizer/transformers_tokenizer",
        input_dir: str = "data/cleaned",
        output_dir: str = "data_tokenized_production",
        max_length: int = 1024,
        stride: int = 512,
        batch_size: int = 500,
        save_frequency: int = 10000  # Save every 10k sequences
    ):
        self.tokenizer_path = Path(tokenizer_path)
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Load tokenizer
        self.tokenizer = self.load_tokenizer()
        
        # Get special token IDs with fallbacks (proven approach)
        self.start_token_id = self.tokenizer.token_to_id("<|startoftext|>")
        self.end_token_id = self.tokenizer.token_to_id("<|endoftext|>")
        self.pad_token_id = self.tokenizer.token_to_id("<|pad|>")
        
        # Handle missing special tokens (from successful test)
        if self.pad_token_id is None:
            self.logger.warning("⚠️  <|pad|> token not found, using token ID 0")
            self.pad_token_id = 0
        
        if self.start_token_id is None:
            self.logger.warning("⚠️  <|startoftext|> token not found, using token ID 1")
            self.start_token_id = 1
            
        if self.end_token_id is None:
            self.logger.warning("⚠️  <|endoftext|> token not found, using token ID 2")
            self.end_token_id = 2
        
        self.logger.info(f"🚀 PRODUCTION TOKENIZER - SINGLE THREADED")
        self.logger.info(f"Tokenizer loaded - vocab size: {self.tokenizer.get_vocab_size()}")
        self.logger.info(f"Special tokens - Start: {self.start_token_id}, End: {self.end_token_id}, Pad: {self.pad_token_id}")
        self.logger.info(f"Expected processing rate: ~3,200 sequences/sec based on test results")
    
    def setup_logging(self):
        """Configure logging for production run."""
        log_file = f"production_tokenization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
        """Prepare text from JSON item for tokenization (proven approach)."""
        text_parts = []
        
        # Add title if exists and not empty
        title = item.get('title', '').strip()
        if title and title not in ['', '[deleted]', '[removed]']:
            text_parts.append(title)
        
        # Add main text
        text = item.get('text', '').strip()
        if text and text not in ['', '[deleted]', '[removed]']:
            text_parts.append(text)
        
        # Join with newlines - simplified format (from successful test)
        full_text = "\n".join(text_parts)
        return full_text
    
    def tokenize_text(self, text: str) -> list:
        """Tokenize text with sliding window for long sequences (proven approach)."""
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
        """Save tokenized sequences to file (proven approach)."""
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
            
            # Ensure pad_token_id is valid
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
    
    def process_file(self, file_path: Path, file_index: int, total_files: int) -> dict:
        """Process a single file completely."""
        self.logger.info(f"📄 Processing file {file_index}/{total_files}: {file_path.name}")
        file_start_time = time.time()
        
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
                        if text or title:
                            batch.append(data)
                            valid_items += 1
                        
                        # Process batch when full
                        if len(batch) >= self.batch_size:
                            sequences = self.process_batch(batch)
                            all_sequences.extend(sequences)
                            batch = []
                            
                            # Progress update every 10k sequences
                            if len(all_sequences) % self.save_frequency == 0:
                                elapsed = time.time() - file_start_time
                                rate = len(all_sequences) / elapsed if elapsed > 0 else 0
                                self.logger.info(f"  📊 File progress: {total_items} items, {len(all_sequences)} sequences ({rate:.0f}/sec)")
                    
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
            
            # Save all sequences for this file
            if all_sequences:
                output_name = f"tokenized_{file_path.stem}.npz"
                output_path = self.output_dir / output_name
                sequences_saved = self.save_tokenized_data(all_sequences, output_path)
            else:
                sequences_saved = 0
            
            elapsed = time.time() - file_start_time
            rate = sequences_saved / elapsed if elapsed > 0 else 0
            self.logger.info(f"✅ File complete: {sequences_saved} sequences from {valid_items}/{total_items} items in {elapsed:.1f}s ({rate:.0f}/sec)")
            
            return {
                "sequences": sequences_saved,
                "items": total_items,
                "valid_items": valid_items,
                "elapsed": elapsed
            }
            
        except Exception as e:
            self.logger.error(f"❌ Critical error processing {file_path}: {e}")
            return {"sequences": 0, "items": 0, "error": str(e)}
    
    def run_production(self):
        """Run the full production tokenization."""
        files = sorted(list(self.input_dir.glob("*.jsonl")))
        
        if not files:
            self.logger.error("❌ No JSONL files found in input directory!")
            return
        
        self.logger.info(f"🚀 Starting PRODUCTION tokenization of {len(files)} files")
        self.logger.info(f"📁 Input: {self.input_dir}")
        self.logger.info(f"📤 Output: {self.output_dir}")
        self.logger.info(f"📊 Expected completion: ~8.7 hours based on test results")
        
        start_time = time.time()
        total_sequences = 0
        total_items = 0
        total_valid_items = 0
        
        # Process each file
        for i, file_path in enumerate(files, 1):
            try:
                result = self.process_file(file_path, i, len(files))
                
                total_sequences += result.get("sequences", 0)
                total_items += result.get("items", 0)
                total_valid_items += result.get("valid_items", 0)
                
                # Overall progress update
                elapsed = time.time() - start_time
                overall_rate = total_sequences / elapsed if elapsed > 0 else 0
                remaining_files = len(files) - i
                estimated_remaining_hours = (remaining_files / i) * (elapsed / 3600) if i > 0 else 0
                
                self.logger.info(f"🔢 Overall progress: {i}/{len(files)} files, {total_sequences:,} total sequences ({overall_rate:.0f}/sec)")
                self.logger.info(f"⏰ Estimated remaining time: {estimated_remaining_hours:.1f} hours")
                self.logger.info(f"")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to process {file_path}: {e}")
                continue
        
        # Final results
        total_elapsed = time.time() - start_time
        self.logger.info(f"")
        self.logger.info(f"🎯 === PRODUCTION TOKENIZATION COMPLETE ===")
        self.logger.info(f"⏱️  Total duration: {total_elapsed/3600:.2f} hours")
        self.logger.info(f"📁 Files processed: {len(files)}")
        self.logger.info(f"🔢 Total sequences: {total_sequences:,}")
        self.logger.info(f"📄 Total items: {total_items:,} (valid: {total_valid_items:,})")
        self.logger.info(f"⚡ Average processing rate: {total_sequences/total_elapsed:.0f} sequences/sec")
        self.logger.info(f"💾 Output directory: {self.output_dir}")
        
        # Create summary file
        summary = {
            "completion_timestamp": datetime.now().isoformat(),
            "total_duration_hours": total_elapsed / 3600,
            "files_processed": len(files),
            "total_sequences": total_sequences,
            "total_items": total_items,
            "total_valid_items": total_valid_items,
            "sequences_per_second": total_sequences / total_elapsed if total_elapsed > 0 else 0,
            "config": {
                "max_length": self.max_length,
                "stride": self.stride,
                "batch_size": self.batch_size,
                "vocab_size": self.tokenizer.get_vocab_size()
            }
        }
        
        with open(self.output_dir / "tokenization_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"📋 Summary saved to {self.output_dir}/tokenization_summary.json")
        self.logger.info(f"")
        self.logger.info(f"🎉 PRODUCTION TOKENIZATION SUCCESSFUL!")
        self.logger.info(f"✅ Ready for model training with {total_sequences:,} sequences")


def main():
    """Main function for production tokenization."""
    print("=" * 60)
    print("🚀 PRODUCTION TOKENIZATION - FULL DATASET")
    print("=" * 60)
    print("Based on successful 10-minute test (3,211 sequences/sec)")
    print("Estimated completion time: ~8.7 hours")
    print()
    print("This will process ALL files in your dataset using the")
    print("proven single-threaded approach that passed testing.")
    print()
    
    # Ask for confirmation
    response = input("Start FULL production tokenization? (y/N): ").strip().lower()
    if response != 'y':
        print("Production run cancelled")
        return
    
    try:
        # Create production tokenizer
        tokenizer = ProductionTokenizer(
            tokenizer_path="models/tokenizer/transformers_tokenizer",
            input_dir="data/cleaned",
            output_dir="data_tokenized_production",
            max_length=1024,
            stride=512,
            batch_size=500
        )
        
        print("🚀 Starting PRODUCTION tokenization...")
        print("💡 This will run for several hours - consider using screen/tmux")
        print()
        
        tokenizer.run_production()
        
        print("\n" + "=" * 60)
        print("🎉 PRODUCTION TOKENIZATION COMPLETE!")
        print("✅ Your dataset is now ready for model training!")
        
    except Exception as e:
        print(f"❌ Production run failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 