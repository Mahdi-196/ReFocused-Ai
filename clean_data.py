#!/usr/bin/env python3
"""
Reddit Data Cleaner
Professional text cleaning and preprocessing for Reddit data
Automatically splits output into chunks for efficient training
"""

import json
import re
import argparse
import os
import logging
from typing import Optional, List, Tuple
from better_profanity import profanity
import pandas as pd


class RedditDataCleaner:
    """Professional text cleaning for Reddit data with file chunking"""
    
    def __init__(self, min_score: int = 5, min_length: int = 10, chunk_size_gb: float = 0.48828125):
        self.min_score = min_score
        self.min_length = min_length
        self.chunk_size_bytes = int(chunk_size_gb * 1024**3)  # Convert GiB to bytes
        self.logger = self._setup_logging()
        profanity.load_censor_words()
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for cleaning operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> Optional[str]:
        """Clean and normalize text content"""
        if not text or text.strip() == "":
            return None
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '', text)  # User mentions
        text = re.sub(r'/r/\w+', '', text)  # Subreddit mentions
        text = re.sub(r'\[deleted\]|\[removed\]', '', text)  # Removed content
        
        # Clean special characters while preserving punctuation
        text = re.sub(r'[^\w\s\.,!?;:\'"()-]', ' ', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Filter profanity
        text = profanity.censor(text)
        
        # Check minimum length
        if len(text) < self.min_length:
            return None
            
        return text
    
    def process_item(self, item: dict) -> Optional[str]:
        """Process a single data item"""
        try:
            # Skip low-scored content
            if item.get('score', 0) < self.min_score:
                return None
            
            # Extract text based on item type
            if item['type'] == 'submission':
                title = item.get('title', '')
                text = item.get('text', '')
                raw_text = f"{title}. {text}".strip() if text else title
            elif item['type'] == 'comment':
                raw_text = item.get('text', '')
            else:
                return None
            
            return self.clean_text(raw_text)
            
        except (KeyError, TypeError):
            return None
    
    def _get_next_filename(self, base_path: str, chunk_index: int) -> str:
        """Generate filename for chunk"""
        base_dir = os.path.dirname(base_path)
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        extension = os.path.splitext(base_path)[1] or '.txt'
        
        if chunk_index == 0:
            return base_path
        else:
            return os.path.join(base_dir, f"{base_name}_chunk_{chunk_index:03d}{extension}")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format byte size for human reading"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.2f} GB"
    
    def process_file(self, input_file: str, output_base: str) -> List[Tuple[str, int, int]]:
        """Process entire data file with automatic chunking"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        os.makedirs(os.path.dirname(output_base), exist_ok=True)
        
        processed_count = 0
        total_kept_count = 0
        chunk_index = 0
        current_chunk_size = 0
        current_chunk_count = 0
        output_files = []
        
        self.logger.info(f"Processing {input_file} with {self._format_size(self.chunk_size_bytes)} chunks")
        
        # Initialize first output file
        current_output_file = self._get_next_filename(output_base, chunk_index)
        outfile = open(current_output_file, 'w', encoding='utf-8')
        
        try:
            with open(input_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    try:
                        item = json.loads(line.strip())
                        processed_count += 1
                        
                        cleaned_text = self.process_item(item)
                        
                        if cleaned_text:
                            text_with_newline = cleaned_text + '\n'
                            text_size = len(text_with_newline.encode('utf-8'))
                            
                            # Check if we need to start a new chunk
                            if current_chunk_size + text_size > self.chunk_size_bytes and current_chunk_count > 0:
                                # Close current file and record its stats
                                outfile.close()
                                output_files.append((current_output_file, current_chunk_count, current_chunk_size))
                                
                                self.logger.info(
                                    f"Completed {os.path.basename(current_output_file)}: "
                                    f"{current_chunk_count} items, {self._format_size(current_chunk_size)}"
                                )
                                
                                # Start new chunk
                                chunk_index += 1
                                current_output_file = self._get_next_filename(output_base, chunk_index)
                                outfile = open(current_output_file, 'w', encoding='utf-8')
                                current_chunk_size = 0
                                current_chunk_count = 0
                            
                            # Write to current file
                            outfile.write(text_with_newline)
                            current_chunk_size += text_size
                            current_chunk_count += 1
                            total_kept_count += 1
                            
                    except json.JSONDecodeError:
                        continue
        
        finally:
            # Close final file and record its stats
            outfile.close()
            output_files.append((current_output_file, current_chunk_count, current_chunk_size))
            
            if current_chunk_count > 0:
                self.logger.info(
                    f"Completed {os.path.basename(current_output_file)}: "
                    f"{current_chunk_count} items, {self._format_size(current_chunk_size)}"
                )
        
        self.logger.info(f"Processing complete: {processed_count} items processed, {total_kept_count} items kept")
        self.logger.info(f"Created {len(output_files)} output file(s)")
        
        return output_files
    
    def get_stats(self, input_file: str) -> dict:
        """Generate statistics about the dataset"""
        if not os.path.exists(input_file):
            return {}
        
        stats = {
            'total_items': 0,
            'submissions': 0,
            'comments': 0,
            'avg_score': 0,
            'score_distribution': {}
        }
        
        scores = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    stats['total_items'] += 1
                    
                    if item['type'] == 'submission':
                        stats['submissions'] += 1
                    elif item['type'] == 'comment':
                        stats['comments'] += 1
                    
                    score = item.get('score', 0)
                    scores.append(score)
                    
                except json.JSONDecodeError:
                    continue
        
        if scores:
            stats['avg_score'] = sum(scores) / len(scores)
            stats['min_score'] = min(scores)
            stats['max_score'] = max(scores)
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='Clean Reddit data with automatic file chunking')
    parser.add_argument('--input', default='data/raw.txt', 
                       help='Input raw data file')
    parser.add_argument('--output', default='data/clean.txt', 
                       help='Output base filename (chunks will be numbered)')
    parser.add_argument('--min-score', type=int, default=5,
                       help='Minimum score threshold (5+ upvotes)')
    parser.add_argument('--min-length', type=int, default=10,
                       help='Minimum text length')
    parser.add_argument('--chunk-size', type=float, default=0.48828125,
                       help='Chunk size in GiB (default: 500MB)')
    parser.add_argument('--stats', action='store_true',
                       help='Show dataset statistics')
    
    args = parser.parse_args()
    
    # Initialize cleaner
    cleaner = RedditDataCleaner(
        min_score=args.min_score,
        min_length=args.min_length,
        chunk_size_gb=args.chunk_size
    )
    
    # Show statistics if requested
    if args.stats:
        stats = cleaner.get_stats(args.input)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
        print()
    
    # Process data
    try:
        output_files = cleaner.process_file(args.input, args.output)
        
        print(f"\nCleaning completed!")
        print(f"Created {len(output_files)} file(s):")
        
        total_items = 0
        total_size = 0
        
        for filename, item_count, file_size in output_files:
            size_str = cleaner._format_size(file_size)
            print(f"  {os.path.basename(filename)}: {item_count} items, {size_str}")
            total_items += item_count
            total_size += file_size
        
        print(f"\nTotal: {total_items} items, {cleaner._format_size(total_size)}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run fetch_data.py first to collect raw data.")


if __name__ == "__main__":
    main() 