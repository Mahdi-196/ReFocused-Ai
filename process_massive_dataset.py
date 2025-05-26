#!/usr/bin/env python3
"""
Process Massive Reddit Dataset
Cleans and formats extracted Reddit data for AI training
"""

import json
import time
import psutil
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional
import multiprocessing as mp
from collections import defaultdict
import hashlib
import re
from datetime import datetime

class MassiveDatasetProcessor:
    """Process massive extracted Reddit dataset"""
    
    def __init__(self):
        self.input_dir = Path("data/unified_raw")
        self.output_dir = Path("data/cleaned")
        self.logs_dir = Path("logs")
        
        # Create directories
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Setup logging
        log_file = self.logs_dir / f"massive_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="100 MB")
        
        # Stats tracking
        self.stats = {
            'files_processed': 0,
            'posts_read': 0,
            'posts_cleaned': 0,
            'duplicates_removed': 0,
            'start_time': time.time()
        }
        
        # Deduplication tracking
        self.seen_hashes = set()
        self.chunk_size = 100_000  # Process in chunks of 100k posts
        
    def clean_text(self, text: str) -> Optional[str]:
        """Clean individual text content"""
        if not text or text in ['[deleted]', '[removed]', '']:
            return None
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove very short content
        if len(text) < 20:
            return None
            
        # Remove URLs (optional)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove Reddit-specific formatting
        text = re.sub(r'/u/\w+', '[USER]', text)
        text = re.sub(r'/r/\w+', '[SUBREDDIT]', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Remove bold
        
        # Final cleanup
        text = text.strip()
        if len(text) < 20:
            return None
            
        return text
    
    def create_text_hash(self, text: str) -> str:
        """Create hash for deduplication"""
        # Normalize text for hashing
        normalized = re.sub(r'\W+', '', text.lower())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def process_single_file(self, file_path: Path) -> Dict:
        """Process a single JSONL file"""
        logger.info(f"📄 Processing: {file_path.name}")
        
        file_stats = {
            'file': file_path.name,
            'posts_read': 0,
            'posts_cleaned': 0,
            'duplicates': 0
        }
        
        cleaned_posts = []
        is_huggingface = 'huggingface' in file_path.name
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                        
                    try:
                        post = json.loads(line)
                        file_stats['posts_read'] += 1
                        
                        # Handle different formats
                        if is_huggingface:
                            # HuggingFace format
                            text_content = post.get('text', '')
                            clean_record = {
                                'subreddit': 'openwebtext',
                                'id': post.get('id', f'hf_{line_num}'),
                                'title': '',  # HF doesn't have titles
                                'text': text_content,
                                'created_utc': 0,  # HF doesn't have timestamps
                                'score': 1,  # Default score for HF data
                                'author': 'openwebtext',
                                'type': 'article'
                            }
                        else:
                            # Reddit format
                            text_content = post.get('text', '')
                            clean_record = {
                                'subreddit': post.get('subreddit', ''),
                                'id': post.get('id', ''),
                                'title': post.get('title', ''),
                                'text': text_content,
                                'created_utc': post.get('created_utc', 0),
                                'score': post.get('score', 0),
                                'author': post.get('author', '[deleted]'),
                                'type': post.get('type', 'unknown')
                            }
                        
                        # Clean the text content
                        cleaned_text = self.clean_text(text_content)
                        if not cleaned_text:
                            continue
                        
                        clean_record['text'] = cleaned_text
                        
                        # Check for duplicates
                        text_hash = self.create_text_hash(cleaned_text)
                        if text_hash in self.seen_hashes:
                            file_stats['duplicates'] += 1
                            continue
                        
                        self.seen_hashes.add(text_hash)
                        
                        # Keep all HuggingFace content (it's already high quality)
                        # For Reddit, only keep posts with good engagement (score > 0)
                        if is_huggingface or clean_record['score'] >= 1:
                            cleaned_posts.append(clean_record)
                            file_stats['posts_cleaned'] += 1
                        
                        # Progress update
                        if line_num % 50000 == 0:
                            logger.info(f"   📊 Processed {line_num:,} lines, {file_stats['posts_cleaned']:,} clean posts")
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        logger.debug(f"Error processing line {line_num}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"❌ Error processing {file_path.name}: {e}")
            return file_stats
        
        logger.success(f"✅ {file_path.name}: {file_stats['posts_read']:,} → {file_stats['posts_cleaned']:,} clean posts")
        
        # Write cleaned data in chunks
        if cleaned_posts:
            self.write_cleaned_chunks(cleaned_posts, file_path.stem)
        
        return file_stats
    
    def write_cleaned_chunks(self, posts: List[Dict], base_name: str):
        """Write cleaned posts in 100MB chunks"""
        chunk_size_bytes = 100 * 1024 * 1024  # 100MB chunks
        current_chunk = []
        current_size = 0
        chunk_num = 1
        
        for post in posts:
            post_json = json.dumps(post, ensure_ascii=False) + '\n'
            post_size = len(post_json.encode('utf-8'))
            
            if current_size + post_size > chunk_size_bytes and current_chunk:
                # Write current chunk
                output_file = self.output_dir / f"cleaned_{base_name}_part{chunk_num:03d}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for clean_post in current_chunk:
                        f.write(json.dumps(clean_post, ensure_ascii=False) + '\n')
                
                logger.info(f"   💾 Wrote chunk {chunk_num}: {len(current_chunk):,} posts to {output_file.name}")
                
                # Reset for next chunk
                current_chunk = []
                current_size = 0
                chunk_num += 1
            
            current_chunk.append(post)
            current_size += post_size
        
        # Write final chunk
        if current_chunk:
            output_file = self.output_dir / f"cleaned_{base_name}_part{chunk_num:03d}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for clean_post in current_chunk:
                    f.write(json.dumps(clean_post, ensure_ascii=False) + '\n')
            
            logger.info(f"   💾 Wrote final chunk: {len(current_chunk):,} posts to {output_file.name}")
    
    def process_all_files(self):
        """Process all extracted files"""
        logger.info("🚀 Starting massive Reddit dataset processing...")
        
        # Find all JSONL files
        jsonl_files = list(self.input_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.error("❌ No JSONL files found in data/unified_raw/")
            return
        
        total_files = len(jsonl_files)
        logger.info(f"📊 Found {total_files} files to process (~27GB)")
        
        # Sort by size (process largest first for better progress indication)
        jsonl_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        
        # Process each file
        for i, file_path in enumerate(jsonl_files, 1):
            logger.info(f"\n📁 Processing file {i}/{total_files}: {file_path.name}")
            file_stats = self.process_single_file(file_path)
            
            # Update global stats
            self.stats['files_processed'] += 1
            self.stats['posts_read'] += file_stats['posts_read']
            self.stats['posts_cleaned'] += file_stats['posts_cleaned']
            self.stats['duplicates_removed'] += file_stats['duplicates']
            
            # Progress report every 10 files
            if i % 10 == 0:
                elapsed = time.time() - self.stats['start_time']
                rate = self.stats['posts_cleaned'] / (elapsed / 3600)  # posts per hour
                
                logger.info(f"\n📈 PROGRESS REPORT ({i}/{total_files} files):")
                logger.info(f"   ⏱️  Time: {elapsed/3600:.1f} hours")
                logger.info(f"   📊 Posts: {self.stats['posts_read']:,} read → {self.stats['posts_cleaned']:,} clean")
                logger.info(f"   ⚡ Rate: {rate:.0f} posts/hour")
                logger.info(f"   💾 Deduplication: {self.stats['duplicates_removed']:,} duplicates removed")
        
        self.generate_final_report()
    
    def generate_final_report(self):
        """Generate final processing report"""
        elapsed = time.time() - self.stats['start_time']
        retention_rate = (self.stats['posts_cleaned'] / max(self.stats['posts_read'], 1)) * 100
        
        # Calculate output size
        output_size = sum(f.stat().st_size for f in self.output_dir.glob("*.jsonl"))
        output_size_gb = output_size / (1024**3)
        
        logger.success("\n🎉 MASSIVE DATASET PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {elapsed/3600:.2f} hours")
        logger.info(f"📁 Files processed: {self.stats['files_processed']}")
        logger.info(f"📊 Posts read: {self.stats['posts_read']:,}")
        logger.info(f"📊 Posts cleaned: {self.stats['posts_cleaned']:,}")
        logger.info(f"🗑️  Duplicates removed: {self.stats['duplicates_removed']:,}")
        logger.info(f"📈 Retention rate: {retention_rate:.1f}%")
        logger.info(f"💾 Output size: {output_size_gb:.2f}GB")
        logger.info(f"⚡ Processing rate: {self.stats['posts_cleaned']/(elapsed/3600):.0f} posts/hour")
        
        # Count output files
        output_files = list(self.output_dir.glob("*.jsonl"))
        logger.info(f"📄 Output files: {len(output_files)} JSONL chunks")
        
        logger.info(f"\n🎯 TRAINING-READY DATA:")
        logger.info(f"   📁 Location: {self.output_dir}/")
        logger.info(f"   📊 Format: JSONL chunks (~100MB each)")
        logger.info(f"   🏷️  Schema: subreddit, id, title, text, created_utc, score, author, type")
        logger.info(f"   ✅ Ready for AI training!")

def main():
    """Main processing function"""
    processor = MassiveDatasetProcessor()
    
    # System check
    memory = psutil.virtual_memory()
    logger.info(f"💻 System: {memory.total/(1024**3):.1f}GB RAM, {psutil.cpu_count()} cores")
    logger.info(f"💾 Available: {memory.available/(1024**3):.1f}GB RAM")
    
    try:
        processor.process_all_files()
    except KeyboardInterrupt:
        logger.warning("⚠️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 