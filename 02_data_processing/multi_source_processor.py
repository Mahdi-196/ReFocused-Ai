import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
from datetime import datetime
import hashlib
from collections import defaultdict, Counter
import multiprocessing as mp
from functools import partial

from loguru import logger
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


class MultiSourceDataProcessor:
    """
    Advanced multi-source data processor for Wikipedia and Reddit content
    Handles real-time collected data from Wikipedia-Collector.py and Reddit API
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "data/processed", n_workers: int = None):
        self.data_dir = Path(data_dir)
        self.real_data_dir = self.data_dir / "real_multi_source"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set number of workers for parallel processing
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        
        # Initialize text processing
        self._ensure_nltk_data()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'wikipedia_articles': 0,
            'reddit_posts': 0,
            'kept_items': 0,
            'removed_duplicates': 0,
            'removed_low_quality': 0,
            'removed_empty': 0,
            'processing_time': 0
        }
        
        # Content hashes for deduplication
        self.content_hashes = set()
        
    def _ensure_nltk_data(self):
        """Download required NLTK data"""
        required_data = ['punkt', 'stopwords']
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
            except LookupError:
                try:
                    nltk.data.find(f'corpora/{data_name}')
                except LookupError:
                    logger.info(f"Downloading NLTK {data_name}...")
                    nltk.download(data_name, quiet=True)
    
    def load_real_time_data(self) -> List[Dict]:
        """Load real-time collected data from Wikipedia and Reddit"""
        logger.info("🔄 Loading real-time multi-source data...")
        
        all_data = []
        
        # Load Wikipedia data
        wikipedia_data = self._load_wikipedia_data()
        all_data.extend(wikipedia_data)
        logger.info(f"📖 Loaded {len(wikipedia_data)} Wikipedia articles")
        
        # Load Reddit data
        reddit_data = self._load_reddit_data()
        all_data.extend(reddit_data)
        logger.info(f"💬 Loaded {len(reddit_data)} Reddit posts")
        
        # Load legacy data if available
        legacy_data = self._load_legacy_data()
        all_data.extend(legacy_data)
        if legacy_data:
            logger.info(f"📁 Loaded {len(legacy_data)} legacy posts")
        
        logger.success(f"✅ Total data loaded: {len(all_data)} items")
        return all_data
    
    def _load_wikipedia_data(self) -> List[Dict]:
        """Load Wikipedia articles from real-time collection"""
        wikipedia_dir = self.real_data_dir / "wikipedia"
        data = []
        
        if not wikipedia_dir.exists():
            logger.info("📖 No Wikipedia data directory found")
            return data
        
        for jsonl_file in wikipedia_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                article = json.loads(line)
                                article['source_type'] = 'wikipedia'
                                article['source_file'] = str(jsonl_file.name)
                                article['source_line'] = line_num
                                data.append(article)
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON decode error in {jsonl_file}:{line_num}: {e}")
                                continue
            except Exception as e:
                logger.warning(f"Error reading Wikipedia file {jsonl_file}: {e}")
                continue
        
        return data
    
    def _load_reddit_data(self) -> List[Dict]:
        """Load Reddit posts from real-time collection"""
        reddit_dir = self.real_data_dir / "reddit"
        data = []
        
        if not reddit_dir.exists():
            logger.info("💬 No Reddit data directory found")
            return data
        
        for jsonl_file in reddit_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                post = json.loads(line)
                                post['source_type'] = 'reddit'
                                post['source_file'] = str(jsonl_file.name)
                                post['source_line'] = line_num
                                data.append(post)
                            except json.JSONDecodeError as e:
                                logger.debug(f"JSON decode error in {jsonl_file}:{line_num}: {e}")
                                continue
            except Exception as e:
                logger.warning(f"Error reading Reddit file {jsonl_file}: {e}")
                continue
        
        return data
    
    def _load_legacy_data(self) -> List[Dict]:
        """Load legacy data from previous collections if available"""
        data = []
        legacy_dirs = [
            self.data_dir / "cleaned",
            self.data_dir / "unified_raw"
        ]
        
        for legacy_dir in legacy_dirs:
            if legacy_dir.exists():
                for file_path in legacy_dir.rglob("*.jsonl"):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f, 1):
                                line = line.strip()
                                if line:
                                    try:
                                        item = json.loads(line)
                                        item['source_type'] = 'legacy'
                                        item['source_file'] = str(file_path.relative_to(self.data_dir))
                                        item['source_line'] = line_num
                                        data.append(item)
                                    except json.JSONDecodeError:
                                        continue
                    except Exception as e:
                        logger.debug(f"Error reading legacy file {file_path}: {e}")
                        continue
        
        return data
    
    def normalize_content(self, raw_item: Dict) -> Optional[Dict]:
        """Normalize content from different sources into a unified format"""
        try:
            source_type = raw_item.get('source_type', 'unknown')
            
            if source_type == 'wikipedia':
                return self._normalize_wikipedia_article(raw_item)
            elif source_type == 'reddit':
                return self._normalize_reddit_post(raw_item)
            elif source_type == 'legacy':
                return self._normalize_legacy_item(raw_item)
            else:
                logger.debug(f"Unknown source type: {source_type}")
                return None
                
        except Exception as e:
            logger.debug(f"Error normalizing content: {e}")
            return None
    
    def _normalize_wikipedia_article(self, article: Dict) -> Dict:
        """Normalize Wikipedia article format"""
        title = article.get('title', '')
        content = article.get('content', '')
        summary = article.get('summary', '')
        
        # Create comprehensive training text
        full_text = f"Title: {title}\n\n"
        if summary and len(summary.strip()) > 50:
            full_text += f"Summary: {summary.strip()}\n\n"
        if content:
            full_text += f"Content: {content.strip()}"
        
        return {
            'id': article.get('page_id', f"wiki_{hash(title)}")[:16],
            'title': title,
            'content': content,
            'summary': summary,
            'training_text': full_text.strip(),
            'url': article.get('url', ''),
            'category': article.get('category', ''),
            'word_count': len(content.split()) if content else 0,
            'content_length': len(content) if content else 0,
            'source_type': 'wikipedia',
            'source_file': article.get('source_file', ''),
            'source_line': article.get('source_line', 0),
            'collected_at': article.get('collected_at', ''),
            'quality_score': self._calculate_wikipedia_quality(article)
        }
    
    def _normalize_reddit_post(self, post: Dict) -> Dict:
        """Normalize Reddit post format"""
        title = post.get('title', '')
        selftext = post.get('selftext', '') or post.get('content', '')
        
        # Create conversational training text
        if selftext and selftext.strip():
            if '?' in title:
                training_text = f"Question: {title}\n\nAnswer: {selftext}"
            else:
                training_text = f"Topic: {title}\n\nDiscussion: {selftext}"
        else:
            training_text = f"Topic: {title}"
        
        return {
            'id': post.get('id', '')[:16],
            'title': title,
            'content': selftext,
            'training_text': training_text,
            'subreddit': post.get('subreddit', '').lower(),
            'score': int(post.get('score', 0)),
            'num_comments': int(post.get('num_comments', 0)),
            'upvote_ratio': float(post.get('upvote_ratio', 0.5)),
            'author': post.get('author', ''),
            'created_utc': int(post.get('created_utc', 0)),
            'word_count': len(selftext.split()) if selftext else 0,
            'content_length': len(selftext) if selftext else 0,
            'source_type': 'reddit',
            'source_file': post.get('source_file', ''),
            'source_line': post.get('source_line', 0),
            'collected_at': post.get('collected_at', ''),
            'quality_score': self._calculate_reddit_quality(post)
        }
    
    def _normalize_legacy_item(self, item: Dict) -> Dict:
        """Normalize legacy data format"""
        # Try to detect if it's Reddit or other format
        if 'subreddit' in item or 'selftext' in item:
            # Treat as Reddit post
            item['source_type'] = 'reddit'
            return self._normalize_reddit_post(item)
        else:
            # Treat as general text content
            title = item.get('title', '')
            content = item.get('content', '') or item.get('text', '')
            
            return {
                'id': item.get('id', f"legacy_{hash(str(item))}")[:16],
                'title': title,
                'content': content,
                'training_text': f"Topic: {title}\n\nContent: {content}" if title else content,
                'word_count': len(content.split()) if content else 0,
                'content_length': len(content) if content else 0,
                'source_type': 'legacy',
                'source_file': item.get('source_file', ''),
                'source_line': item.get('source_line', 0),
                'quality_score': self._calculate_general_quality(item)
            }
    
    def _calculate_wikipedia_quality(self, article: Dict) -> float:
        """Calculate quality score for Wikipedia articles"""
        score = 0.0
        content = article.get('content', '')
        
        # Content length (0-30 points)
        content_length = len(content) if content else 0
        if content_length > 5000:
            score += 30
        elif content_length > 2000:
            score += 20
        elif content_length > 1000:
            score += 10
        
        # Has summary (10 points)
        if article.get('summary') and len(article.get('summary', '')) > 100:
            score += 10
        
        # Has category (5 points)
        if article.get('category'):
            score += 5
        
        # Content quality indicators (15 points)
        if content:
            # Check for references/citations
            if 'http' in content or 'www.' in content:
                score += 5
            # Check for structured content
            if any(marker in content.lower() for marker in ['==', 'category:', 'infobox', 'references']):
                score += 5
            # Check for good sentence structure
            sentences = content.split('.')
            if len(sentences) > 5:
                score += 5
        
        # Word count bonus (10 points)
        word_count = len(content.split()) if content else 0
        if word_count > 1000:
            score += 10
        elif word_count > 500:
            score += 5
        
        return min(score, 70.0)  # Max 70 for Wikipedia
    
    def _calculate_reddit_quality(self, post: Dict) -> float:
        """Calculate quality score for Reddit posts"""
        score = 0.0
        
        # Score and engagement (20 points)
        reddit_score = int(post.get('score', 0))
        if reddit_score > 100:
            score += 20
        elif reddit_score > 50:
            score += 15
        elif reddit_score > 10:
            score += 10
        elif reddit_score > 0:
            score += 5
        
        # Comment engagement (15 points)
        num_comments = int(post.get('num_comments', 0))
        if num_comments > 50:
            score += 15
        elif num_comments > 20:
            score += 10
        elif num_comments > 5:
            score += 5
        
        # Content quality (25 points)
        content = post.get('selftext', '') or post.get('content', '')
        if content:
            word_count = len(content.split())
            if word_count > 200:
                score += 25
            elif word_count > 100:
                score += 20
            elif word_count > 50:
                score += 15
            elif word_count > 20:
                score += 10
        
        # Title quality (10 points)
        title = post.get('title', '')
        if title and len(title.split()) >= 5:
            score += 10
        elif title and len(title.split()) >= 3:
            score += 5
        
        return min(score, 70.0)  # Max 70 for Reddit
    
    def _calculate_general_quality(self, item: Dict) -> float:
        """Calculate quality score for general content"""
        score = 0.0
        content = item.get('content', '') or item.get('text', '')
        
        if content:
            word_count = len(content.split())
            if word_count > 500:
                score += 40
            elif word_count > 200:
                score += 30
            elif word_count > 100:
                score += 20
            elif word_count > 50:
                score += 10
        
        if item.get('title'):
            score += 10
        
        return min(score, 50.0)  # Max 50 for general content
    
    def filter_quality_content(self, items: List[Dict], min_quality_score: float = 25.0) -> List[Dict]:
        """Filter items based on quality score"""
        logger.info(f"🔍 Filtering content with quality score >= {min_quality_score}")
        
        initial_count = len(items)
        filtered_items = [item for item in items if item.get('quality_score', 0) >= min_quality_score]
        final_count = len(filtered_items)
        
        removed_count = initial_count - final_count
        self.stats['removed_low_quality'] += removed_count
        
        logger.info(f"📊 Quality filter: kept {final_count}/{initial_count} items (removed {removed_count})")
        return filtered_items
    
    def remove_duplicates(self, items: List[Dict]) -> List[Dict]:
        """Remove duplicate content based on content hash"""
        logger.info("🔄 Removing duplicate content...")
        
        unique_items = []
        content_hashes = set()
        
        for item in items:
            # Create content hash from training text or content
            content_for_hash = item.get('training_text', '') or item.get('content', '')
            if not content_for_hash.strip():
                continue
            
            content_hash = hashlib.md5(content_for_hash.lower().strip().encode()).hexdigest()
            
            if content_hash not in content_hashes:
                content_hashes.add(content_hash)
                unique_items.append(item)
            else:
                self.stats['removed_duplicates'] += 1
        
        logger.info(f"📊 Deduplication: kept {len(unique_items)}/{len(items)} items")
        return unique_items
    
    def create_training_splits(self, items: List[Dict], test_size: float = 0.3, val_size: float = 0.5) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create balanced train/validation/test splits"""
        logger.info("📊 Creating training data splits...")
        
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(items)
        
        # Stratify by source type if we have multiple sources
        if len(df['source_type'].unique()) > 1:
            # First split: train vs temp (test+val)
            train_df, temp_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=42,
                stratify=df['source_type']
            )
            
            # Second split: validation vs test
            val_df, test_df = train_test_split(
                temp_df, 
                test_size=val_size, 
                random_state=42,
                stratify=temp_df['source_type']
            )
        else:
            # Regular split if only one source type
            train_df, temp_df = train_test_split(df, test_size=test_size, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=val_size, random_state=42)
        
        # Convert back to lists of dictionaries
        train_items = train_df.to_dict('records')
        val_items = val_df.to_dict('records')
        test_items = test_df.to_dict('records')
        
        logger.success(f"✅ Created splits: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
        
        return train_items, val_items, test_items
    
    def save_processed_data(self, train_items: List[Dict], val_items: List[Dict], test_items: List[Dict]):
        """Save processed data for training"""
        logger.info("💾 Saving processed training data...")
        
        # Save as JSONL for training
        splits = {
            'train': train_items,
            'validation': val_items,
            'test': test_items
        }
        
        for split_name, items in splits.items():
            # Save JSONL format for training
            jsonl_path = self.output_dir / f"{split_name}.jsonl"
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in items:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Save CSV format for inspection
            df = pd.DataFrame(items)
            csv_path = self.output_dir / f"{split_name}.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"📁 Saved {len(items)} items to {split_name} split")
        
        # Save comprehensive metadata
        metadata = self._generate_processing_metadata(train_items, val_items, test_items)
        with open(self.output_dir / "processing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.success(f"✅ All processed data saved to {self.output_dir}")
    
    def _generate_processing_metadata(self, train_items: List[Dict], val_items: List[Dict], test_items: List[Dict]) -> Dict:
        """Generate comprehensive processing metadata"""
        all_items = train_items + val_items + test_items
        
        # Source type distribution
        source_types = Counter(item.get('source_type', 'unknown') for item in all_items)
        
        # Quality score statistics
        quality_scores = [item.get('quality_score', 0) for item in all_items]
        
        # Content statistics
        word_counts = [item.get('word_count', 0) for item in all_items]
        content_lengths = [item.get('content_length', 0) for item in all_items]
        
        return {
            'processing_info': {
                'processed_at': datetime.now().isoformat(),
                'processor_version': '1.0.0',
                'total_items': len(all_items),
                'processing_stats': self.stats
            },
            'data_splits': {
                'train': len(train_items),
                'validation': len(val_items),
                'test': len(test_items),
                'total': len(all_items)
            },
            'source_distribution': dict(source_types),
            'quality_statistics': {
                'mean_score': np.mean(quality_scores),
                'median_score': np.median(quality_scores),
                'min_score': np.min(quality_scores),
                'max_score': np.max(quality_scores),
                'std_score': np.std(quality_scores)
            },
            'content_statistics': {
                'word_count': {
                    'mean': np.mean(word_counts),
                    'median': np.median(word_counts),
                    'min': np.min(word_counts),
                    'max': np.max(word_counts)
                },
                'content_length': {
                    'mean': np.mean(content_lengths),
                    'median': np.median(content_lengths),
                    'min': np.min(content_lengths),
                    'max': np.max(content_lengths)
                }
            },
            'subreddit_distribution': self._get_subreddit_distribution(all_items),
            'wikipedia_category_distribution': self._get_category_distribution(all_items)
        }
    
    def _get_subreddit_distribution(self, items: List[Dict]) -> Dict:
        """Get distribution of Reddit posts by subreddit"""
        reddit_items = [item for item in items if item.get('source_type') == 'reddit']
        subreddits = Counter(item.get('subreddit', 'unknown') for item in reddit_items)
        return dict(subreddits.most_common(20))  # Top 20 subreddits
    
    def _get_category_distribution(self, items: List[Dict]) -> Dict:
        """Get distribution of Wikipedia articles by category"""
        wiki_items = [item for item in items if item.get('source_type') == 'wikipedia']
        categories = Counter(item.get('category', 'unknown') for item in wiki_items)
        return dict(categories.most_common(20))  # Top 20 categories
    
    def run_processing_pipeline(self, min_quality_score: float = 25.0):
        """Run the complete multi-source processing pipeline"""
        start_time = datetime.now()
        logger.info("🚀 Starting multi-source data processing pipeline...")
        
        # Load all data
        raw_items = self.load_real_time_data()
        if not raw_items:
            logger.error("❌ No data found to process")
            return
        
        self.stats['total_processed'] = len(raw_items)
        
        # Normalize all content
        logger.info("🔄 Normalizing content from all sources...")
        normalized_items = []
        for item in raw_items:
            normalized = self.normalize_content(item)
            if normalized:
                normalized_items.append(normalized)
        
        logger.info(f"✅ Normalized {len(normalized_items)}/{len(raw_items)} items")
        
        # Remove empty content
        non_empty_items = [item for item in normalized_items if item.get('training_text', '').strip()]
        self.stats['removed_empty'] = len(normalized_items) - len(non_empty_items)
        
        # Filter quality content
        quality_items = self.filter_quality_content(non_empty_items, min_quality_score)
        
        # Remove duplicates
        unique_items = self.remove_duplicates(quality_items)
        self.stats['kept_items'] = len(unique_items)
        
        # Update source type counts
        for item in unique_items:
            source_type = item.get('source_type', 'unknown')
            if source_type == 'wikipedia':
                self.stats['wikipedia_articles'] += 1
            elif source_type == 'reddit':
                self.stats['reddit_posts'] += 1
        
        # Create training splits
        train_items, val_items, test_items = self.create_training_splits(unique_items)
        
        # Save processed data
        self.save_processed_data(train_items, val_items, test_items)
        
        # Calculate processing time
        self.stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        # Generate and display final report
        self._display_final_report()
        
        logger.success("🎉 Multi-source processing pipeline completed successfully!")
    
    def _display_final_report(self):
        """Display comprehensive processing report"""
        logger.info("\n" + "="*60)
        logger.info("📊 MULTI-SOURCE PROCESSING REPORT")
        logger.info("="*60)
        logger.info(f"⏱️  Processing Time: {self.stats['processing_time']:.2f} seconds")
        logger.info(f"📥 Total Items Processed: {self.stats['total_processed']:,}")
        logger.info(f"📖 Wikipedia Articles: {self.stats['wikipedia_articles']:,}")
        logger.info(f"💬 Reddit Posts: {self.stats['reddit_posts']:,}")
        logger.info(f"✅ Items Kept: {self.stats['kept_items']:,}")
        logger.info(f"🗑️  Removed - Duplicates: {self.stats['removed_duplicates']:,}")
        logger.info(f"🗑️  Removed - Low Quality: {self.stats['removed_low_quality']:,}")
        logger.info(f"🗑️  Removed - Empty: {self.stats['removed_empty']:,}")
        logger.info(f"💾 Output Directory: {self.output_dir}")
        logger.info("="*60)


def main():
    """Main execution function"""
    processor = MultiSourceDataProcessor()
    processor.run_processing_pipeline(min_quality_score=25.0)


if __name__ == "__main__":
    main() 