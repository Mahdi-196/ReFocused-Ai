#!/usr/bin/env python3
"""
Diagnostic script to identify tokenization issues.
"""

import os
import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

def setup_logging():
    """Setup logging for diagnosis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('tokenization_diagnosis.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def analyze_cleaned_data():
    """Analyze the cleaned data to identify potential issues."""
    logger = setup_logging()
    logger.info("🔍 Starting tokenization diagnosis...")
    
    cleaned_dir = Path("data/cleaned")
    if not cleaned_dir.exists():
        logger.error(f"❌ Cleaned data directory not found: {cleaned_dir}")
        return
    
    files = list(cleaned_dir.glob("*.jsonl"))
    logger.info(f"📂 Found {len(files)} cleaned files")
    
    # Statistics
    file_stats = []
    total_items = 0
    empty_files = 0
    error_files = 0
    
    # Text length distribution
    text_lengths = []
    subreddit_counts = Counter()
    
    # Sample files for detailed analysis
    sample_files = files[:10]  # Analyze first 10 files
    
    for file_path in sample_files:
        logger.info(f"📄 Analyzing: {file_path.name}")
        
        try:
            file_size = file_path.stat().st_size
            items_in_file = 0
            valid_items = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        items_in_file += 1
                        
                        # Check data quality
                        text = data.get('text', '').strip()
                        title = data.get('title', '').strip()
                        subreddit = data.get('subreddit', 'unknown')
                        
                        # Count subreddits
                        subreddit_counts[subreddit] += 1
                        
                        # Check if text is valid
                        combined_text = f"{title}\n{text}".strip()
                        if combined_text and combined_text not in ['', '[deleted]', '[removed]']:
                            valid_items += 1
                            text_lengths.append(len(combined_text))
                        
                        # Log first few items for inspection
                        if line_num <= 3:
                            logger.info(f"  Sample item {line_num}: subreddit='{subreddit}', text_len={len(text)}, title_len={len(title)}")
                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"  ⚠️  JSON error at line {line_num}: {e}")
                        continue
                    
                    # Limit analysis for large files
                    if line_num >= 1000:
                        break
            
            if items_in_file == 0:
                empty_files += 1
                logger.warning(f"  ⚠️  Empty file: {file_path.name}")
            
            file_stats.append({
                'file': file_path.name,
                'size_mb': file_size / (1024 * 1024),
                'total_items': items_in_file,
                'valid_items': valid_items,
                'validity_rate': valid_items / items_in_file if items_in_file > 0 else 0
            })
            
            total_items += items_in_file
            logger.info(f"  ✅ Items: {items_in_file}, Valid: {valid_items} ({valid_items/items_in_file*100:.1f}%)")
        
        except Exception as e:
            error_files += 1
            logger.error(f"  ❌ Error reading {file_path.name}: {e}")
    
    # Summary statistics
    logger.info(f"\n📊 DIAGNOSIS SUMMARY:")
    logger.info(f"  Total files analyzed: {len(sample_files)}")
    logger.info(f"  Empty files: {empty_files}")
    logger.info(f"  Error files: {error_files}")
    logger.info(f"  Total items sampled: {total_items}")
    
    if text_lengths:
        logger.info(f"  Text length stats:")
        logger.info(f"    Min: {min(text_lengths)} chars")
        logger.info(f"    Max: {max(text_lengths)} chars")
        logger.info(f"    Avg: {sum(text_lengths)/len(text_lengths):.1f} chars")
        logger.info(f"    Median: {sorted(text_lengths)[len(text_lengths)//2]} chars")
    
    logger.info(f"  Top subreddits: {dict(subreddit_counts.most_common(5))}")
    
    # Detailed file analysis
    logger.info(f"\n📋 FILE DETAILS:")
    for stat in file_stats:
        logger.info(f"  {stat['file']}: {stat['size_mb']:.1f}MB, {stat['total_items']} items, {stat['validity_rate']:.1%} valid")

def check_tokenizer():
    """Check if tokenizer is accessible."""
    logger = logging.getLogger(__name__)
    
    try:
        from tokenizers import Tokenizer
        
        # Try different tokenizer paths
        tokenizer_paths = [
            "models/tokenizer/transformers_tokenizer/tokenizer.json",
            "models/tokenizer/tokenizer.json",
            "tokenizer_750M/tokenizer.json"
        ]
        
        for path in tokenizer_paths:
            if Path(path).exists():
                logger.info(f"✅ Found tokenizer at: {path}")
                tokenizer = Tokenizer.from_file(str(path))
                logger.info(f"  Vocab size: {tokenizer.get_vocab_size()}")
                
                # Test tokenization
                test_text = "<|startoftext|>Hello world<|endoftext|>"
                encoding = tokenizer.encode(test_text)
                logger.info(f"  Test encoding: {len(encoding.ids)} tokens")
                return True
        
        logger.error("❌ No tokenizer found!")
        return False
        
    except ImportError as e:
        logger.error(f"❌ Cannot import tokenizer: {e}")
        return False

def check_multiprocessing_issue():
    """Check for potential multiprocessing issues."""
    logger = logging.getLogger(__name__)
    
    try:
        import multiprocessing as mp
        logger.info(f"💻 CPU count: {mp.cpu_count()}")
        
        # Test basic multiprocessing
        def test_worker(x):
            return x * 2
        
        with mp.Pool(2) as pool:
            result = pool.map(test_worker, [1, 2, 3])
            logger.info(f"✅ Multiprocessing test successful: {result}")
            return True
            
    except Exception as e:
        logger.error(f"❌ Multiprocessing error: {e}")
        return False

def main():
    """Run full diagnosis."""
    logger = setup_logging()
    
    logger.info("🚀 TOKENIZATION ISSUE DIAGNOSIS")
    logger.info("=" * 50)
    
    # Check components
    logger.info("\n1. Checking cleaned data...")
    analyze_cleaned_data()
    
    logger.info("\n2. Checking tokenizer...")
    tokenizer_ok = check_tokenizer()
    
    logger.info("\n3. Checking multiprocessing...")
    mp_ok = check_multiprocessing_issue()
    
    # Final recommendations
    logger.info(f"\n🎯 RECOMMENDATIONS:")
    
    if not tokenizer_ok:
        logger.info("  ❌ Fix tokenizer path issues first")
    
    if not mp_ok:
        logger.info("  ❌ Multiprocessing issues detected - try single-threaded processing")
    
    logger.info("  📝 Common issues:")
    logger.info("    - Most text might be filtered out due to length restrictions")
    logger.info("    - JSON parsing errors in many files")
    logger.info("    - Tokenizer path not found during processing")
    logger.info("    - Memory issues with large files")
    logger.info("    - Multiprocessing problems in cloud environments")
    
    logger.info(f"\n📋 Next steps:")
    logger.info("  1. Run this diagnosis to identify the exact issue")
    logger.info("  2. Consider single-threaded processing with error handling")
    logger.info("  3. Add more verbose logging to track progress")
    logger.info("  4. Process files one by one to identify problematic files")

if __name__ == "__main__":
    main() 