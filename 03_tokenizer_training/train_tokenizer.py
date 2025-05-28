#!/usr/bin/env python3
"""
ByteLevel BPE Tokenizer Training Script
======================================

Trains a ByteLevel BPE tokenizer for GPT-style models using Hugging Face's tokenizers library.
Designed to efficiently handle large datasets (64GB+) by processing files in chunks.

Requirements:
- tokenizers
- transformers
- tqdm
- json

Author: AI Assistant
Version: 1.0
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Iterator, List, Optional

import tqdm
from tokenizers import ByteLevelBPETokenizer
from transformers import GPT2TokenizerFast


def setup_logging() -> logging.Logger:
    """Set up logging configuration with both file and console output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tokenizer_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_text_from_jsonl(file_path: Path, text_field: str = "text") -> Iterator[str]:
    """
    Extract text content from JSONL files efficiently.
    
    Args:
        file_path: Path to the JSONL file
        text_field: JSON field containing the text content
        
    Yields:
        Text strings from the JSONL file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    text = data.get(text_field, "")
                    if text and isinstance(text, str) and len(text.strip()) > 0:
                        yield text.strip()
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON decode error in {file_path}:{line_num}: {e}")
                except Exception as e:
                    logging.warning(f"Error processing line {line_num} in {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")


def create_temp_text_files(
    input_dir: Path, 
    temp_dir: Path, 
    max_files: Optional[int] = None,
    chunk_size_mb: int = 100
) -> List[Path]:
    """
    Convert JSONL files to temporary text files for tokenizer training.
    
    Args:
        input_dir: Directory containing JSONL files
        temp_dir: Directory to store temporary text files
        max_files: Maximum number of JSONL files to process (None for all)
        chunk_size_mb: Target size in MB for each temporary text file
        
    Returns:
        List of temporary text file paths
    """
    logger = logging.getLogger(__name__)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    jsonl_files = sorted([f for f in input_dir.glob("*.jsonl")])
    if max_files:
        jsonl_files = jsonl_files[:max_files]
    
    logger.info(f"Found {len(jsonl_files)} JSONL files to process")
    
    temp_files = []
    current_file_idx = 0
    current_size_bytes = 0
    current_file = None
    target_size_bytes = chunk_size_mb * 1024 * 1024
    
    try:
        for jsonl_file in tqdm.tqdm(jsonl_files, desc="Processing JSONL files"):
            logger.info(f"Processing {jsonl_file.name}")
            
            text_count = 0
            for text in extract_text_from_jsonl(jsonl_file):
                # Start new temp file if needed
                if current_file is None or current_size_bytes >= target_size_bytes:
                    if current_file:
                        current_file.close()
                    
                    temp_file_path = temp_dir / f"temp_chunk_{current_file_idx:04d}.txt"
                    temp_files.append(temp_file_path)
                    current_file = open(temp_file_path, 'w', encoding='utf-8')
                    current_file_idx += 1
                    current_size_bytes = 0
                
                # Write text to current temp file
                text_bytes = text.encode('utf-8')
                current_file.write(text + '\n\n')
                current_size_bytes += len(text_bytes) + 2  # +2 for newlines
                text_count += 1
            
            logger.info(f"Extracted {text_count:,} texts from {jsonl_file.name}")
    
    finally:
        if current_file:
            current_file.close()
    
    logger.info(f"Created {len(temp_files)} temporary text files")
    return temp_files


def train_tokenizer(
    text_files: List[Path],
    output_dir: Path,
    vocab_size: int = 50257,
    min_frequency: int = 2
) -> ByteLevelBPETokenizer:
    """
    Train a ByteLevel BPE tokenizer on the provided text files.
    
    Args:
        text_files: List of text files for training
        output_dir: Directory to save tokenizer files
        vocab_size: Target vocabulary size
        min_frequency: Minimum frequency for tokens to be included
        
    Returns:
        Trained tokenizer instance
    """
    logger = logging.getLogger(__name__)
    
    # Initialize the tokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # Define special tokens (GPT-2 style)
    special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    
    logger.info(f"Starting tokenizer training with {len(text_files)} files")
    logger.info(f"Target vocab size: {vocab_size:,}")
    logger.info(f"Special tokens: {special_tokens}")
    
    # Convert Path objects to strings for the tokenizer
    file_paths = [str(f) for f in text_files]
    
    # Train the tokenizer
    start_time = time.time()
    tokenizer.train(
        files=file_paths,
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    training_time = time.time() - start_time
    
    logger.info(f"Tokenizer training completed in {training_time:.2f} seconds")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer files
    vocab_file = output_dir / "vocab.json"
    merges_file = output_dir / "merges.txt"
    tokenizer_file = output_dir / "tokenizer.json"
    
    tokenizer.save(str(tokenizer_file))
    tokenizer.save_model(str(output_dir))
    
    logger.info(f"Tokenizer saved to {output_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - {vocab_file}")
    logger.info(f"  - {merges_file}")
    logger.info(f"  - {tokenizer_file}")
    
    return tokenizer


def analyze_tokenizer(tokenizer: ByteLevelBPETokenizer, sample_texts: List[str]) -> None:
    """
    Analyze the trained tokenizer and log statistics.
    
    Args:
        tokenizer: Trained tokenizer instance
        sample_texts: Sample texts for analysis
    """
    logger = logging.getLogger(__name__)
    
    vocab_size = tokenizer.get_vocab_size()
    logger.info(f"Final vocabulary size: {vocab_size:,}")
    
    # Analyze sample texts
    total_chars = 0
    total_tokens = 0
    
    for text in sample_texts[:10]:  # Analyze first 10 samples
        encoding = tokenizer.encode(text)
        tokens = encoding.tokens
        total_chars += len(text)
        total_tokens += len(tokens)
    
    if total_chars > 0:
        compression_ratio = total_chars / total_tokens
        logger.info(f"Average compression ratio: {compression_ratio:.2f} chars/token")


def test_tokenizer_loading(output_dir: Path) -> None:
    """
    Test loading the tokenizer with transformers and demonstrate usage.
    
    Args:
        output_dir: Directory containing the saved tokenizer
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load tokenizer with transformers
        tokenizer = GPT2TokenizerFast.from_pretrained(str(output_dir))
        
        # Test text
        test_text = "I will build better habits."
        
        # Tokenize the test text
        tokens = tokenizer.tokenize(test_text)
        token_ids = tokenizer.encode(test_text)
        
        logger.info("=" * 50)
        logger.info("TOKENIZER TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"Test text: '{test_text}'")
        logger.info(f"Tokens: {tokens}")
        logger.info(f"Token IDs: {token_ids}")
        logger.info(f"Number of tokens: {len(tokens)}")
        logger.info(f"Vocabulary size: {tokenizer.vocab_size:,}")
        
        # Test decoding
        decoded = tokenizer.decode(token_ids)
        logger.info(f"Decoded text: '{decoded}'")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"Error testing tokenizer: {e}")


def cleanup_temp_files(temp_files: List[Path]) -> None:
    """Clean up temporary files created during processing."""
    logger = logging.getLogger(__name__)
    
    removed_count = 0
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
                removed_count += 1
        except Exception as e:
            logger.warning(f"Could not remove {temp_file}: {e}")
    
    logger.info(f"Cleaned up {removed_count} temporary files")


def main():
    """Main function to orchestrate the tokenizer training process."""
    logger = setup_logging()
    
    # Configuration
    INPUT_DIR = Path("data/cleaned")
    OUTPUT_DIR = Path("tokenizer_750M")
    TEMP_DIR = Path("temp_tokenizer_data")
    VOCAB_SIZE = 50257
    MAX_FILES = None  # Set to a number to limit files for testing, None for all
    
    logger.info("Starting ByteLevel BPE Tokenizer Training")
    logger.info("=" * 60)
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Target vocabulary size: {VOCAB_SIZE:,}")
    
    try:
        # Step 1: Check if input directory exists
        if not INPUT_DIR.exists():
            logger.error(f"Input directory {INPUT_DIR} does not exist!")
            return
        
        # Step 2: Convert JSONL files to temporary text files
        logger.info("Step 1: Converting JSONL files to text format...")
        temp_files = create_temp_text_files(
            input_dir=INPUT_DIR,
            temp_dir=TEMP_DIR,
            max_files=MAX_FILES,
            chunk_size_mb=100
        )
        
        if not temp_files:
            logger.error("No text files were created. Check your input data.")
            return
        
        # Calculate total data size
        total_size_mb = sum(f.stat().st_size for f in temp_files if f.exists()) / (1024 * 1024)
        logger.info(f"Total training data size: {total_size_mb:.1f} MB")
        logger.info(f"Number of training files: {len(temp_files)}")
        
        # Step 3: Train the tokenizer
        logger.info("Step 2: Training ByteLevel BPE tokenizer...")
        tokenizer = train_tokenizer(
            text_files=temp_files,
            output_dir=OUTPUT_DIR,
            vocab_size=VOCAB_SIZE
        )
        
        # Step 4: Analyze the tokenizer
        logger.info("Step 3: Analyzing tokenizer performance...")
        sample_texts = []
        if temp_files:
            try:
                with open(temp_files[0], 'r', encoding='utf-8') as f:
                    sample_texts = [line.strip() for line in f.readlines()[:20] if line.strip()]
            except Exception as e:
                logger.warning(f"Could not read sample texts: {e}")
        
        analyze_tokenizer(tokenizer, sample_texts)
        
        # Step 5: Test loading with transformers
        logger.info("Step 4: Testing tokenizer loading with transformers...")
        test_tokenizer_loading(OUTPUT_DIR)
        
        # Step 6: Cleanup
        logger.info("Step 5: Cleaning up temporary files...")
        cleanup_temp_files(temp_files)
        
        # Clean up temp directory if empty
        try:
            if TEMP_DIR.exists() and not any(TEMP_DIR.iterdir()):
                TEMP_DIR.rmdir()
                logger.info(f"Removed empty temporary directory: {TEMP_DIR}")
        except Exception as e:
            logger.warning(f"Could not remove temp directory: {e}")
        
        logger.info("=" * 60)
        logger.info("TOKENIZER TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Tokenizer saved to: {OUTPUT_DIR}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error during tokenizer training: {e}")
        raise


if __name__ == "__main__":
    main() 