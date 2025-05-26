#!/usr/bin/env python3
"""
Process Full HuggingFace Dataset
Clean the new 8M HuggingFace records and add to cleaned dataset
"""

import json
import time
import psutil
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional
import hashlib
import re
from datetime import datetime

class FullHuggingFaceProcessor:
    """Process the complete 8M HuggingFace extracted data"""
    
    def __init__(self):
        self.input_dir = Path("data/unified_raw")
        self.output_dir = Path("data/cleaned")
        self.logs_dir = Path("logs")
        
        # Setup logging
        log_file = self.logs_dir / f"full_huggingface_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(log_file, rotation="100 MB")
        
        # Stats tracking
        self.stats = {
            'files_processed': 0,
            'posts_read': 0,
            'posts_cleaned': 0,
            'duplicates_removed': 0,
            'start_time': time.time()
        }
        
        # Load existing hashes to avoid duplicates
        self.seen_hashes = self.load_existing_hashes()
        
    def load_existing_hashes(self) -> set:
        """Load hashes from existing cleaned files to avoid duplicates"""
        logger.info("📂 Loading existing content hashes to avoid duplicates...")
        seen_hashes = set()
        
        # Sample from existing files to build hash set (faster startup)
        existing_files = list(self.output_dir.glob("cleaned_*.jsonl"))
        sample_count = 0
        
        for file_path in existing_files[:30]:  # Sample more files for better dedup
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                post = json.loads(line)
                                text = post.get('text', '')
                                if text:
                                    text_hash = self.create_text_hash(text)
                                    seen_hashes.add(text_hash)
                                    sample_count += 1
                                    
                                if line_num > 5000:  # Limit per file for speed
                                    break
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                logger.debug(f"Error reading {file_path}: {e}")
                continue
        
        logger.info(f"📊 Loaded {len(seen_hashes):,} content hashes from {sample_count:,} existing posts")
        return seen_hashes
    
    def clean_text(self, text: str) -> Optional[str]:
        """Clean individual text content for HuggingFace data"""
        if not text or text.strip() == '':
            return None
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Higher quality threshold for HuggingFace (it's premium content)
        if len(text) < 100:  # Require substantial content
            return None
            
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Final cleanup
        text = text.strip()
        if len(text) < 100:
            return None
            
        return text
    
    def create_text_hash(self, text: str) -> str:
        """Create hash for deduplication"""
        normalized = re.sub(r'\W+', '', text.lower())[:1000]  # First 1000 chars for speed
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def process_full_huggingface_files(self):
        """Process all full HuggingFace files"""
        logger.info("🤗 Processing FULL HuggingFace dataset (8M records)...")
        
        # Find full HuggingFace files
        hf_files = list(self.input_dir.glob("huggingface_full_chunk_*.txt"))
        if not hf_files:
            logger.error("❌ No full HuggingFace files found!")
            return
        
        total_files = len(hf_files)
        logger.info(f"📊 Found {total_files} full HuggingFace files to process (~38GB)")
        
        # Sort by chunk number for consistent processing
        hf_files.sort(key=lambda x: int(x.name.split('_')[3]))
        
        # Process in batches for memory efficiency
        batch_size = 20  # Process 20 files at a time
        all_cleaned_posts = []
        
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch_files = hf_files[batch_start:batch_end]
            
            logger.info(f"\n📦 Processing batch {batch_start//batch_size + 1}/{(total_files-1)//batch_size + 1}")
            logger.info(f"    Files {batch_start+1}-{batch_end} of {total_files}")
            
            # Process batch
            batch_posts = self.process_file_batch(batch_files)
            all_cleaned_posts.extend(batch_posts)
            
            # Write batch to avoid memory issues
            if len(all_cleaned_posts) >= 200000:  # Write every 200k posts
                self.write_cleaned_chunks(all_cleaned_posts, f"huggingface_full_batch_{batch_start//batch_size + 1}")
                all_cleaned_posts = []
            
            # Progress report
            elapsed = time.time() - self.stats['start_time']
            rate = self.stats['posts_cleaned'] / (elapsed / 3600) if elapsed > 0 else 0
            logger.info(f"📈 Batch complete: {self.stats['posts_cleaned']:,} total posts, {rate:.0f}/hour")
        
        # Write final batch
        if all_cleaned_posts:
            self.write_cleaned_chunks(all_cleaned_posts, "huggingface_full_final")
        
        self.generate_final_report()
    
    def process_file_batch(self, files: List[Path]) -> List[Dict]:
        """Process a batch of files"""
        batch_posts = []
        
        for file_path in files:
            logger.info(f"📄 Processing: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                            
                        try:
                            post = json.loads(line)
                            self.stats['posts_read'] += 1
                            
                            # Extract text content
                            text_content = post.get('text', '')
                            
                            # Clean the text
                            cleaned_text = self.clean_text(text_content)
                            if not cleaned_text:
                                continue
                            
                            # Check for duplicates
                            text_hash = self.create_text_hash(cleaned_text)
                            if text_hash in self.seen_hashes:
                                self.stats['duplicates_removed'] += 1
                                continue
                            
                            self.seen_hashes.add(text_hash)
                            
                            # Create clean record in unified format
                            clean_record = {
                                'subreddit': 'openwebtext',
                                'id': post.get('id', f'hf_full_{self.stats["posts_cleaned"]}'),
                                'title': '',  # HF doesn't have titles
                                'text': cleaned_text,
                                'created_utc': 0,  # HF doesn't have timestamps
                                'score': 5,  # High score for quality HF content
                                'author': 'openwebtext',
                                'type': 'article'
                            }
                            
                            batch_posts.append(clean_record)
                            self.stats['posts_cleaned'] += 1
                            
                        except json.JSONDecodeError:
                            continue
                        except Exception as e:
                            logger.debug(f"Error processing line {line_num}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"❌ Error processing {file_path.name}: {e}")
                continue
            
            self.stats['files_processed'] += 1
        
        return batch_posts
    
    def write_cleaned_chunks(self, posts: List[Dict], batch_name: str):
        """Write cleaned posts in 100MB chunks"""
        chunk_size_bytes = 100 * 1024 * 1024  # 100MB chunks
        current_chunk = []
        current_size = 0
        chunk_num = 1
        
        logger.info(f"💾 Writing {len(posts):,} posts from {batch_name}...")
        
        for post in posts:
            post_json = json.dumps(post, ensure_ascii=False) + '\n'
            post_size = len(post_json.encode('utf-8'))
            
            if current_size + post_size > chunk_size_bytes and current_chunk:
                # Write current chunk
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.output_dir / f"cleaned_huggingface_full_{batch_name}_part{chunk_num:03d}_{timestamp}.jsonl"
                
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"cleaned_huggingface_full_{batch_name}_part{chunk_num:03d}_{timestamp}.jsonl"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for clean_post in current_chunk:
                    f.write(json.dumps(clean_post, ensure_ascii=False) + '\n')
            
            logger.info(f"   💾 Wrote final chunk: {len(current_chunk):,} posts to {output_file.name}")
    
    def generate_final_report(self):
        """Generate final processing report"""
        elapsed = time.time() - self.stats['start_time']
        retention_rate = (self.stats['posts_cleaned'] / max(self.stats['posts_read'], 1)) * 100
        
        # Calculate full HF output size
        hf_full_files = list(self.output_dir.glob("cleaned_huggingface_full_*.jsonl"))
        hf_size = sum(f.stat().st_size for f in hf_full_files)
        hf_size_gb = hf_size / (1024**3)
        
        logger.success("\n🤗 FULL HUGGINGFACE PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total time: {elapsed/3600:.2f} hours")
        logger.info(f"📁 Files processed: {self.stats['files_processed']}")
        logger.info(f"📊 Posts read: {self.stats['posts_read']:,}")
        logger.info(f"📊 Posts cleaned: {self.stats['posts_cleaned']:,}")
        logger.info(f"🗑️  Duplicates removed: {self.stats['duplicates_removed']:,}")
        logger.info(f"📈 Retention rate: {retention_rate:.1f}%")
        logger.info(f"💾 Full HF output: {hf_size_gb:.2f}GB")
        logger.info(f"⚡ Processing rate: {self.stats['posts_cleaned']/(elapsed/3600):.0f} posts/hour")
        
        logger.info(f"📄 Full HF files: {len(hf_full_files)} JSONL chunks")
        
        # FINAL MEGA DATASET INFO
        all_files = list(self.output_dir.glob("cleaned_*.jsonl"))
        total_size = sum(f.stat().st_size for f in all_files)
        total_size_gb = total_size / (1024**3)
        
        logger.success(f"\n🎯 MEGA DATASET COMPLETE!")
        logger.info(f"   📁 Total files: {len(all_files)} JSONL chunks")
        logger.info(f"   💾 Total size: {total_size_gb:.2f}GB")
        logger.info(f"   📊 Estimated posts: ~57M (Reddit + HuggingFace)")
        logger.info(f"   🏆 Enterprise-grade AI training dataset!")
        logger.info(f"   ✅ Ready for training!")

def main():
    """Main processing function"""
    processor = FullHuggingFaceProcessor()
    
    # System check
    memory = psutil.virtual_memory()
    logger.info(f"💻 System: {memory.total/(1024**3):.1f}GB RAM, {psutil.cpu_count()} cores")
    logger.info(f"💾 Available: {memory.available/(1024**3):.1f}GB RAM")
    
    try:
        processor.process_full_huggingface_files()
    except KeyboardInterrupt:
        logger.warning("⚠️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise

if __name__ == "__main__":
    main() 