#!/usr/bin/env python3
"""
Comprehensive Reddit Data Cleaner for ReFocused AI
Processes and cleans large volumes of Reddit data for training
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import re
from datetime import datetime
import hashlib
from collections import defaultdict, Counter
import multiprocessing as mp
from functools import partial

from loguru import logger
from better_profanity import profanity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

class RedditDataCleaner:
    """Advanced Reddit data cleaner with quality filtering and deduplication"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "data/cleaned", n_workers: int = None):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set number of workers for parallel processing
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        # Initialize text processing
        profanity.load_censor_words()
        self._ensure_nltk_data()
        
        # Quality metrics
        self.stats = {
            'total_processed': 0,
            'kept_posts': 0,
            'removed_duplicates': 0,
            'removed_low_quality': 0,
            'removed_spam': 0,
            'removed_nsfw': 0,
            'removed_deleted': 0,
            'processing_time': 0
        }
        
        # Content hashes for deduplication
        self.content_hashes = set()
        
    def _ensure_nltk_data(self):
        """Download required NLTK data"""
        required_data = ['punkt', 'stopwords', 'vader_lexicon']
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data_name}')
                except LookupError:
                    try:
                        nltk.data.find(f'sentiment/{data_name}')
                    except LookupError:
                        logger.info(f"Downloading NLTK {data_name}...")
                        nltk.download(data_name, quiet=True)
    
    def load_raw_data(self) -> List[Dict]:
        """Load all raw data from collection directories and unified sources"""
        logger.info("📂 Loading raw data from all sources...")
        
        all_data = []
        data_dirs = [
            self.data_dir / "reddit_ultra_fast",
            self.data_dir / "reddit_enhanced", 
            self.data_dir / "reddit_oauth",
            self.data_dir / "reddit_large",
            self.data_dir / "unified_raw"  # New unified format from massive dataset manager
        ]
        
        for data_subdir in data_dirs:
            if data_subdir.exists():
                subdir_data = self._load_from_directory(data_subdir)
                all_data.extend(subdir_data)
                logger.info(f"📊 Loaded {len(subdir_data)} posts from {data_subdir.name}")
        
        logger.success(f"✅ Total raw data loaded: {len(all_data)} posts")
        return all_data
    
    def _load_from_directory(self, directory: Path) -> List[Dict]:
        """Load data from a specific directory"""
        data = []
        
        for file_path in directory.rglob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                item = json.loads(line)
                                # Add source metadata
                                item['source_file'] = str(file_path.relative_to(self.data_dir))
                                item['source_line'] = line_num
                                data.append(item)
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON decode error in {file_path}:{line_num}: {e}")
                                continue
            except Exception as e:
                logger.warning(f"Error reading {file_path}: {e}")
                continue
        
        return data
    
    def normalize_post_format(self, raw_post: Dict) -> Optional[Dict]:
        """Normalize post format and extract key fields - supports Reddit and unified formats"""
        try:
            # Handle unified format from massive dataset manager
            if 'source' in raw_post and raw_post.get('source') in ['reddit', 'huggingface_openwebtext']:
                return self._normalize_unified_format(raw_post)
            
            # Handle different Reddit data formats from various collectors
            if 'data' in raw_post and isinstance(raw_post['data'], dict):
                post_data = raw_post['data']
            else:
                post_data = raw_post
            
            # Extract and normalize fields for Reddit format
            normalized = {
                'id': post_data.get('id', ''),
                'title': post_data.get('title', ''),
                'selftext': post_data.get('selftext', '') or post_data.get('content', ''),
                'score': int(post_data.get('score', 0)),
                'upvote_ratio': float(post_data.get('upvote_ratio', 0.5)),
                'num_comments': int(post_data.get('num_comments', 0)),
                'created_utc': int(post_data.get('created_utc', 0)),
                'subreddit': post_data.get('subreddit', '').lower(),
                'author': post_data.get('author', ''),
                'url': post_data.get('url', ''),
                'is_self': bool(post_data.get('is_self', True)),
                'over_18': bool(post_data.get('over_18', False)),
                'spoiler': bool(post_data.get('spoiler', False)),
                'locked': bool(post_data.get('locked', False)),
                'archived': bool(post_data.get('archived', False)),
                'removed_by_category': post_data.get('removed_by_category'),
                'source_file': raw_post.get('source_file', ''),
                'source_line': raw_post.get('source_line', 0),
                'data_source': 'reddit'
            }
            
            # Calculate derived fields
            normalized['word_count'] = len(normalized['selftext'].split()) if normalized['selftext'] else 0
            normalized['title_word_count'] = len(normalized['title'].split()) if normalized['title'] else 0
            normalized['total_word_count'] = normalized['word_count'] + normalized['title_word_count']
            
            return normalized
            
        except Exception as e:
            logger.debug(f"Error normalizing post: {e}")
            return None
    
    def _normalize_unified_format(self, unified_post: Dict) -> Optional[Dict]:
        """Normalize unified format from massive dataset manager"""
        try:
            source_type = unified_post.get('source', 'unknown')
            
            if source_type == 'reddit':
                # Reddit posts in unified format
                normalized = {
                    'id': unified_post.get('id', ''),
                    'title': '',  # Title already merged into text
                    'selftext': unified_post.get('text', ''),
                    'score': int(unified_post.get('score', 0)),
                    'upvote_ratio': 0.5,  # Default for unified format
                    'num_comments': 0,  # Not available in unified format
                    'created_utc': int(unified_post.get('created_utc', 0)),
                    'subreddit': unified_post.get('subreddit', '').lower(),
                    'author': '',  # Not preserved in unified format for privacy
                    'url': '',
                    'is_self': True,  # Assume self posts in unified format
                    'over_18': False,  # Filtered out during unification
                    'spoiler': False,
                    'locked': False,
                    'archived': False,
                    'removed_by_category': None,
                    'source_file': 'unified',
                    'source_line': 0,
                    'data_source': 'reddit_unified'
                }
            
            elif source_type == 'huggingface_openwebtext':
                # HuggingFace OpenWebText in unified format
                normalized = {
                    'id': unified_post.get('id', ''),
                    'title': '',  # No titles in OpenWebText
                    'selftext': unified_post.get('text', ''),
                    'score': 1,  # Default positive score for quality content
                    'upvote_ratio': 1.0,  # Assume quality content
                    'num_comments': 0,
                    'created_utc': 0,  # No timestamps in OpenWebText
                    'subreddit': 'openwebtext',  # Virtual subreddit for categorization
                    'author': '',
                    'url': '',
                    'is_self': True,
                    'over_18': False,
                    'spoiler': False,
                    'locked': False,
                    'archived': False,
                    'removed_by_category': None,
                    'source_file': 'unified_huggingface',
                    'source_line': 0,
                    'data_source': 'huggingface_openwebtext'
                }
            
            else:
                # Unknown unified format
                normalized = {
                    'id': unified_post.get('id', ''),
                    'title': '',
                    'selftext': unified_post.get('text', ''),
                    'score': 1,
                    'upvote_ratio': 0.5,
                    'num_comments': 0,
                    'created_utc': 0,
                    'subreddit': source_type,
                    'author': '',
                    'url': '',
                    'is_self': True,
                    'over_18': False,
                    'spoiler': False,
                    'locked': False,
                    'archived': False,
                    'removed_by_category': None,
                    'source_file': 'unified_unknown',
                    'source_line': 0,
                    'data_source': source_type
                }
            
            # Calculate derived fields
            text_content = normalized['selftext']
            normalized['word_count'] = len(text_content.split()) if text_content else 0
            normalized['title_word_count'] = 0  # No separate titles in unified format
            normalized['total_word_count'] = normalized['word_count']
            
            return normalized
            
        except Exception as e:
            logger.debug(f"Error normalizing unified post: {e}")
            return None
    
    def clean_text(self, text: str) -> str:
        """Advanced text cleaning for Reddit content"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove common Reddit artifacts
        text = re.sub(r'\[deleted\]', '', text)
        text = re.sub(r'\[removed\]', '', text)
        text = re.sub(r'&gt;!(.*)!&lt;', r'\1', text)  # Spoiler tags
        text = re.sub(r'&gt;(.*)$', r'\1', text, flags=re.MULTILINE)  # Quotes
        
        # Remove URLs but keep context
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
        
        # Clean Reddit-specific formatting
        text = re.sub(r'/u/([a-zA-Z0-9_-]+)', r'[USER]', text)  # Usernames
        text = re.sub(r'/r/([a-zA-Z0-9_-]+)', r'[SUBREDDIT]', text)  # Subreddits
        text = re.sub(r'u/([a-zA-Z0-9_-]+)', r'[USER]', text)  # Usernames without slash
        text = re.sub(r'r/([a-zA-Z0-9_-]+)', r'[SUBREDDIT]', text)  # Subreddits without slash
        
        # Remove Reddit markdown
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'~~(.*?)~~', r'\1', text)  # Strikethrough
        text = re.sub(r'\^(\w+)', r'\1', text)  # Superscript
        
        # Remove edit tags
        text = re.sub(r'EDIT:?', '', text, flags=re.IGNORECASE)
        text = re.sub(r'UPDATE:?', '', text, flags=re.IGNORECASE)
        
        # Clean whitespace and formatting
        text = re.sub(r'\n+', ' ', text)  # Multiple newlines to space
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = text.strip()
        
        return text
    
    def is_high_quality_content(self, post: Dict) -> Tuple[bool, str]:
        """Determine if post is high quality with reason"""
        
        # Check for deleted/removed content
        if not post['selftext'] or post['selftext'].lower() in ['[deleted]', '[removed]', '']:
            if not post['title'] or len(post['title']) < 10:
                return False, "deleted_or_removed"
        
        # Skip NSFW content if needed
        if post['over_18']:
            return False, "nsfw"
        
        # Check minimum content requirements
        if post['total_word_count'] < 5:
            return False, "too_short"
        
        # Skip extremely long posts (likely spam/copypasta)
        if post['total_word_count'] > 10000:
            return False, "too_long"
        
        # Check for spam indicators
        text_content = (post['title'] + ' ' + post['selftext']).lower()
        
        # Common spam patterns
        spam_patterns = [
            r'(upvote|like|share|subscribe|follow).{0,20}(if|to|for)',
            r'check.{0,10}(out|this).{0,10}(link|site|channel)',
            r'(click|visit).{0,10}(here|link|below)',
            r'(buy|purchase|order).{0,10}(now|today|here)',
            r'limited.{0,10}time.{0,10}offer',
            r'(make|earn).{0,10}money.{0,10}(fast|quick|easy)',
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, text_content):
                return False, "spam_pattern"
        
        # Check for repetitive content
        words = text_content.split()
        if len(words) > 10:
            word_counts = Counter(words)
            most_common_word_freq = word_counts.most_common(1)[0][1] if word_counts else 0
            if most_common_word_freq > len(words) * 0.3:  # 30% same word
                return False, "repetitive"
        
        # Check engagement quality (for Reddit posts)
        if post['score'] < -10:  # Heavily downvoted
            return False, "heavily_downvoted"
        
        # Prefer self posts for text content
        if not post['is_self'] and post['word_count'] < 20:
            return False, "link_post_short_text"
        
        return True, "high_quality"
    
    def calculate_content_hash(self, post: Dict) -> str:
        """Calculate hash for duplicate detection"""
        # Use title + content for similarity detection
        content = f"{post['title']} {post['selftext']}"
        content = re.sub(r'\s+', ' ', content.lower().strip())
        return hashlib.md5(content.encode()).hexdigest()
    
    def is_duplicate(self, post: Dict) -> bool:
        """Check if post is a duplicate"""
        content_hash = self.calculate_content_hash(post)
        
        if content_hash in self.content_hashes:
            return True
        
        self.content_hashes.add(content_hash)
        return False
    
    def clean_posts_batch(self, posts_batch: List[Dict]) -> List[Dict]:
        """Clean a batch of posts"""
        cleaned_posts = []
        
        for raw_post in posts_batch:
            self.stats['total_processed'] += 1
            
            # Normalize format
            post = self.normalize_post_format(raw_post)
            if not post:
                continue
            
            # Check for duplicates
            if self.is_duplicate(post):
                self.stats['removed_duplicates'] += 1
                continue
            
            # Clean text content
            post['title'] = self.clean_text(post['title'])
            post['selftext'] = self.clean_text(post['selftext'])
            post['total_word_count'] = len((post['title'] + ' ' + post['selftext']).split())
            
            # Quality check
            is_quality, reason = self.is_high_quality_content(post)
            if not is_quality:
                if reason == "deleted_or_removed":
                    self.stats['removed_deleted'] += 1
                elif reason == "nsfw":
                    self.stats['removed_nsfw'] += 1
                elif reason in ["spam_pattern", "repetitive"]:
                    self.stats['removed_spam'] += 1
                else:
                    self.stats['removed_low_quality'] += 1
                continue
            
            # Add cleaning metadata
            post['cleaned_at'] = datetime.now().isoformat()
            post['quality_reason'] = reason
            
            cleaned_posts.append(post)
            self.stats['kept_posts'] += 1
        
        return cleaned_posts
    
    def save_cleaned_data(self, cleaned_posts: List[Dict], output_file: str = "cleaned_reddit_data.jsonl"):
        """Save cleaned data to file"""
        output_path = self.output_dir / output_file
        
        logger.info(f"💾 Saving {len(cleaned_posts)} cleaned posts to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for post in cleaned_posts:
                json.dump(post, f, ensure_ascii=False)
                f.write('\n')
        
        # Save metadata
        metadata = {
            'cleaning_date': datetime.now().isoformat(),
            'total_posts': len(cleaned_posts),
            'stats': self.stats,
            'subreddits': list(Counter(post['subreddit'] for post in cleaned_posts).most_common(20)),
            'avg_word_count': np.mean([post['total_word_count'] for post in cleaned_posts]),
            'word_count_distribution': {
                'min': min(post['total_word_count'] for post in cleaned_posts),
                'max': max(post['total_word_count'] for post in cleaned_posts),
                'median': np.median([post['total_word_count'] for post in cleaned_posts]),
                'std': np.std([post['total_word_count'] for post in cleaned_posts])
            }
        }
        
        metadata_path = self.output_dir / "cleaning_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.success(f"✅ Saved cleaned data and metadata")
    
    def generate_cleaning_report(self) -> str:
        """Generate comprehensive cleaning report"""
        if self.stats['total_processed'] == 0:
            return "No data processed yet."
        
        kept_percentage = (self.stats['kept_posts'] / self.stats['total_processed']) * 100
        
        report = f"""
🧹 REDDIT DATA CLEANING REPORT
{'='*50}

📊 PROCESSING SUMMARY:
   Total Posts Processed: {self.stats['total_processed']:,}
   Posts Kept: {self.stats['kept_posts']:,} ({kept_percentage:.1f}%)
   Posts Removed: {self.stats['total_processed'] - self.stats['kept_posts']:,}

🗑️ REMOVAL BREAKDOWN:
   Duplicates: {self.stats['removed_duplicates']:,}
   Low Quality: {self.stats['removed_low_quality']:,}
   Spam/Repetitive: {self.stats['removed_spam']:,}
   NSFW Content: {self.stats['removed_nsfw']:,}
   Deleted/Removed: {self.stats['removed_deleted']:,}

⚡ PERFORMANCE:
   Processing Time: {self.stats['processing_time']:.2f} seconds
   Posts/Second: {self.stats['total_processed'] / max(self.stats['processing_time'], 1):.1f}

✅ RESULT:
   Clean, high-quality dataset ready for training!
   Output saved to: {self.output_dir}
"""
        return report
    
    def run_cleaning_pipeline(self, batch_size: int = 1000):
        """Run the complete cleaning pipeline"""
        start_time = datetime.now()
        logger.info("🚀 Starting Reddit data cleaning pipeline...")
        
        # Load raw data
        raw_posts = self.load_raw_data()
        if not raw_posts:
            logger.error("❌ No raw data found to clean")
            return
        
        # Process in batches for memory efficiency
        all_cleaned_posts = []
        total_batches = (len(raw_posts) + batch_size - 1) // batch_size
        
        for i in range(0, len(raw_posts), batch_size):
            batch_num = (i // batch_size) + 1
            batch = raw_posts[i:i + batch_size]
            
            logger.info(f"🔄 Processing batch {batch_num}/{total_batches} ({len(batch)} posts)")
            
            cleaned_batch = self.clean_posts_batch(batch)
            all_cleaned_posts.extend(cleaned_batch)
            
            # Progress update
            if batch_num % 10 == 0 or batch_num == total_batches:
                logger.info(f"📈 Progress: {self.stats['kept_posts']:,} posts cleaned so far")
        
        # Save results
        self.save_cleaned_data(all_cleaned_posts)
        
        # Record timing
        end_time = datetime.now()
        self.stats['processing_time'] = (end_time - start_time).total_seconds()
        
        # Generate and display report
        report = self.generate_cleaning_report()
        logger.info(report)
        
        return all_cleaned_posts

def main():
    """Main cleaning function"""
    logger.info("🧹 Starting Reddit Data Cleaning...")
    
    cleaner = RedditDataCleaner()
    cleaned_posts = cleaner.run_cleaning_pipeline()
    
    if cleaned_posts:
        logger.success(f"✅ Cleaning complete! {len(cleaned_posts)} high-quality posts ready for training.")
        logger.info("📋 Next step: Run 'python data_processor.py' to prepare training splits")
    else:
        logger.error("❌ No posts survived the cleaning process")

if __name__ == "__main__":
    main() 