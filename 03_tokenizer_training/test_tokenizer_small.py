#!/usr/bin/env python3
"""
Small Test Version of ByteLevel BPE Tokenizer Training
====================================================

Quick test version to verify the tokenizer training works on a subset of data.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Iterator, List

# Check if required packages are available
try:
    import tqdm
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast
    print("✓ All required packages are available")
except ImportError as e:
    print(f"✗ Missing required package: {e}")
    print("Please install missing packages:")
    print("pip install tokenizers transformers tqdm")
    exit(1)


def setup_logging() -> logging.Logger:
    """Set up simple logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def extract_text_from_jsonl(file_path: Path, max_texts: int = 1000) -> List[str]:
    """Extract limited text content from a JSONL file for testing."""
    texts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if len(texts) >= max_texts:
                    break
                try:
                    data = json.loads(line.strip())
                    text = data.get("text", "")
                    if text and isinstance(text, str) and len(text.strip()) > 0:
                        texts.append(text.strip())
                except json.JSONDecodeError:
                    continue
                except Exception:
                    continue
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
    
    return texts


def main():
    """Test tokenizer training with a small dataset."""
    logger = setup_logging()
    
    # Configuration for small test
    INPUT_DIR = Path("data/cleaned")
    OUTPUT_DIR = Path("tokenizer_test")
    VOCAB_SIZE = 8000  # Smaller vocab for quick testing
    MAX_FILES = 2  # Only process 2 files
    MAX_TEXTS_PER_FILE = 500  # Limit texts per file
    
    logger.info("Starting SMALL TEST of ByteLevel BPE Tokenizer Training")
    logger.info("=" * 60)
    
    try:
        # Check input directory
        if not INPUT_DIR.exists():
            logger.error(f"Input directory {INPUT_DIR} does not exist!")
            return
        
        # Get JSONL files
        jsonl_files = sorted([f for f in INPUT_DIR.glob("*.jsonl")])[:MAX_FILES]
        if not jsonl_files:
            logger.error("No JSONL files found!")
            return
        
        logger.info(f"Processing {len(jsonl_files)} files for testing")
        
        # Extract texts
        all_texts = []
        for jsonl_file in jsonl_files:
            logger.info(f"Processing {jsonl_file.name}")
            texts = extract_text_from_jsonl(jsonl_file, MAX_TEXTS_PER_FILE)
            all_texts.extend(texts)
            logger.info(f"Extracted {len(texts)} texts from {jsonl_file.name}")
        
        logger.info(f"Total texts collected: {len(all_texts)}")
        
        if not all_texts:
            logger.error("No texts extracted!")
            return
        
        # Create temporary text file
        temp_file = Path("temp_test_data.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            for text in all_texts:
                f.write(text + '\n\n')
        
        logger.info(f"Created temporary file: {temp_file}")
        
        # Initialize and train tokenizer
        logger.info("Training tokenizer...")
        tokenizer = ByteLevelBPETokenizer()
        
        special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
        
        start_time = time.time()
        tokenizer.train(
            files=[str(temp_file)],
            vocab_size=VOCAB_SIZE,
            min_frequency=2,
            special_tokens=special_tokens,
            show_progress=True
        )
        training_time = time.time() - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Save tokenizer
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(OUTPUT_DIR / "tokenizer.json"))
        tokenizer.save_model(str(OUTPUT_DIR))
        
        logger.info(f"Tokenizer saved to {OUTPUT_DIR}")
        
        # Test loading with transformers
        logger.info("Testing tokenizer loading...")
        fast_tokenizer = GPT2TokenizerFast.from_pretrained(str(OUTPUT_DIR))
        
        # Test the exact text from requirements
        test_text = "I will build better habits."
        tokens = fast_tokenizer.tokenize(test_text)
        token_ids = fast_tokenizer.encode(test_text)
        
        logger.info("=" * 50)
        logger.info("TOKENIZER TEST RESULTS")
        logger.info("=" * 50)
        logger.info(f"Test text: '{test_text}'")
        logger.info(f"Tokens: {tokens}")
        logger.info(f"Token IDs: {token_ids}")
        logger.info(f"Number of tokens: {len(tokens)}")
        logger.info(f"Vocabulary size: {fast_tokenizer.vocab_size:,}")
        
        decoded = fast_tokenizer.decode(token_ids)
        logger.info(f"Decoded text: '{decoded}'")
        logger.info("=" * 50)
        
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()
            logger.info("Cleaned up temporary file")
        
        logger.info("✓ SMALL TEST COMPLETED SUCCESSFULLY!")
        logger.info(f"✓ Ready to run full training with: python train_tokenizer.py")
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        raise


if __name__ == "__main__":
    main() 