#!/usr/bin/env python3
"""
Data Processor for ReFocused AI Training
Processes cleaned data and prepares it for model training
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
from datetime import datetime

from loguru import logger
from sklearn.model_selection import train_test_split

class TrainingDataProcessor:
    """Processes cleaned data for training preparation"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cleaned_dir = self.data_dir / "cleaned"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load cleaned data from the cleaner output"""
        logger.info("🔄 Loading cleaned data...")
        
        cleaned_file = self.cleaned_dir / "cleaned_reddit_data.jsonl"
        
        if not cleaned_file.exists():
            logger.error(f"❌ Cleaned data not found at {cleaned_file}")
            logger.info("💡 Run 'python data_cleaner.py' first to clean your data")
            return pd.DataFrame()
        
        # Load cleaned data
        data = []
        with open(cleaned_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        if not data:
            logger.error("❌ No valid data found in cleaned file")
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        logger.success(f"✅ Loaded {len(df)} cleaned posts")
        
        # Load and display cleaning metadata
        metadata_file = self.cleaned_dir / "cleaning_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.info(f"📊 Cleaning stats: {metadata['stats']}")
        
        return df
    
    def prepare_text_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare text content for training"""
        logger.info("📝 Preparing text for training...")
        
        # Create combined text for training
        df['training_text'] = df.apply(self._format_training_text, axis=1)
        
        # Filter out items with insufficient training text
        initial_count = len(df)
        df = df[df['training_text'].str.len() >= 20]
        final_count = len(df)
        
        if final_count < initial_count:
            logger.info(f"🔍 Filtered out {initial_count - final_count} posts with insufficient text")
        
        # Add text statistics
        df['training_text_length'] = df['training_text'].str.len()
        df['training_word_count'] = df['training_text'].apply(lambda x: len(x.split()))
        
        return df
    
    def _format_training_text(self, row: pd.Series) -> str:
        """Format a post for training text"""
        # Create a conversational format suitable for chat training
        title = row['title'].strip()
        content = row['selftext'].strip()
        
        if content:
            # Format as question-answer or topic-discussion
            if '?' in title:
                # Question format
                formatted_text = f"Question: {title}\n\nAnswer: {content}"
            else:
                # Topic discussion format
                formatted_text = f"Topic: {title}\n\nDiscussion: {content}"
        else:
            # Title-only content
            formatted_text = f"Topic: {title}"
        
        return formatted_text
    
    def analyze_subreddit_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze distribution across subreddits"""
        logger.info("📊 Analyzing subreddit distribution...")
        
        subreddit_counts = df['subreddit'].value_counts()
        subreddit_stats = {
            'total_subreddits': len(subreddit_counts),
            'top_10_subreddits': subreddit_counts.head(10).to_dict(),
            'posts_per_subreddit': {
                'mean': subreddit_counts.mean(),
                'median': subreddit_counts.median(),
                'min': subreddit_counts.min(),
                'max': subreddit_counts.max()
            }
        }
        
        logger.info(f"📈 Found content from {subreddit_stats['total_subreddits']} subreddits")
        return subreddit_stats
    
    def create_balanced_splits(self, df: pd.DataFrame, test_size: float = 0.3, val_size: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create balanced train/val/test splits"""
        logger.info("📊 Creating balanced data splits...")
        
        # Stratify by subreddit if we have enough data per subreddit
        subreddit_counts = df['subreddit'].value_counts()
        min_samples_per_subreddit = 10
        
        # Only stratify by subreddits that have enough samples
        stratifiable_subreddits = subreddit_counts[subreddit_counts >= min_samples_per_subreddit].index
        df_stratifiable = df[df['subreddit'].isin(stratifiable_subreddits)]
        df_non_stratifiable = df[~df['subreddit'].isin(stratifiable_subreddits)]
        
        splits = []
        
        if len(df_stratifiable) > 0:
            # Stratified split for subreddits with enough data
            logger.info(f"🎯 Using stratified split for {len(df_stratifiable)} posts from popular subreddits")
            train_strat, temp_strat = train_test_split(
                df_stratifiable, 
                test_size=test_size, 
                random_state=42,
                stratify=df_stratifiable['subreddit']
            )
            val_strat, test_strat = train_test_split(
                temp_strat, 
                test_size=val_size, 
                random_state=42,
                stratify=temp_strat['subreddit']
            )
            splits.append((train_strat, val_strat, test_strat))
        
        if len(df_non_stratifiable) > 0:
            # Regular split for remaining data
            logger.info(f"📋 Using random split for {len(df_non_stratifiable)} posts from smaller subreddits")
            train_reg, temp_reg = train_test_split(
                df_non_stratifiable, 
                test_size=test_size, 
                random_state=42
            )
            val_reg, test_reg = train_test_split(
                temp_reg, 
                test_size=val_size, 
                random_state=42
            )
            splits.append((train_reg, val_reg, test_reg))
        
        # Combine splits
        if len(splits) == 2:
            train_df = pd.concat([splits[0][0], splits[1][0]], ignore_index=True)
            val_df = pd.concat([splits[0][1], splits[1][1]], ignore_index=True)
            test_df = pd.concat([splits[0][2], splits[1][2]], ignore_index=True)
        else:
            train_df, val_df, test_df = splits[0]
        
        # Shuffle the combined datasets
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.success(f"✅ Created splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_training_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save processed data for training"""
        logger.info("💾 Saving training data...")
        
        # Save as CSV for inspection
        train_df.to_csv(self.processed_dir / "train.csv", index=False)
        val_df.to_csv(self.processed_dir / "val.csv", index=False)
        test_df.to_csv(self.processed_dir / "test.csv", index=False)
        
        # Save as JSON Lines for training (optimized format)
        self._save_training_jsonl(train_df, self.processed_dir / "train.jsonl")
        self._save_training_jsonl(val_df, self.processed_dir / "val.jsonl")
        self._save_training_jsonl(test_df, self.processed_dir / "test.jsonl")
        
        # Generate comprehensive metadata
        metadata = self._generate_training_metadata(train_df, val_df, test_df)
        
        with open(self.processed_dir / "training_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.success(f"✅ Saved training data to {self.processed_dir}")
    
    def _save_training_jsonl(self, df: pd.DataFrame, filepath: Path):
        """Save DataFrame as JSON Lines format optimized for training"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for _, row in df.iterrows():
                # Training-optimized format
                item = {
                    'text': row['training_text'],
                    'id': row['id'],
                    'subreddit': row['subreddit'],
                    'score': row['score'],
                    'word_count': row['training_word_count'],
                    'source': 'reddit_cleaned'
                }
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    def _generate_training_metadata(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
        """Generate comprehensive metadata for training"""
        all_data = pd.concat([train_df, val_df, test_df])
        
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'total_items': len(all_data),
            'splits': {
                'train': len(train_df),
                'validation': len(val_df),
                'test': len(test_df)
            },
            'text_statistics': {
                'avg_word_count': float(all_data['training_word_count'].mean()),
                'median_word_count': float(all_data['training_word_count'].median()),
                'min_word_count': int(all_data['training_word_count'].min()),
                'max_word_count': int(all_data['training_word_count'].max()),
                'std_word_count': float(all_data['training_word_count'].std()),
                'avg_text_length': float(all_data['training_text_length'].mean())
            },
            'subreddit_distribution': {
                'total_subreddits': all_data['subreddit'].nunique(),
                'top_subreddits': all_data['subreddit'].value_counts().head(10).to_dict(),
                'posts_per_subreddit_avg': float(all_data['subreddit'].value_counts().mean())
            },
            'score_distribution': {
                'avg_score': float(all_data['score'].mean()),
                'median_score': float(all_data['score'].median()),
                'min_score': int(all_data['score'].min()),
                'max_score': int(all_data['score'].max())
            },
            'quality_indicators': {
                'posts_with_content': int((all_data['training_word_count'] > 20).sum()),
                'high_engagement_posts': int((all_data['score'] > 10).sum()),
                'discussion_posts': int((all_data['num_comments'] > 5).sum())
            }
        }
        
        return metadata
    
    def generate_processing_report(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
        """Generate comprehensive processing report"""
        all_data = pd.concat([train_df, val_df, test_df])
        
        report = f"""
📊 TRAINING DATA PROCESSING REPORT
{'='*60}

📈 DATASET OVERVIEW:
   Total Training Items: {len(all_data):,}
   Train Split: {len(train_df):,} ({len(train_df)/len(all_data)*100:.1f}%)
   Validation Split: {len(val_df):,} ({len(val_df)/len(all_data)*100:.1f}%)
   Test Split: {len(test_df):,} ({len(test_df)/len(all_data)*100:.1f}%)

📝 TEXT STATISTICS:
   Average Words per Item: {all_data['training_word_count'].mean():.1f}
   Median Words per Item: {all_data['training_word_count'].median():.1f}
   Word Count Range: {all_data['training_word_count'].min()}-{all_data['training_word_count'].max()}
   
📚 CONTENT DIVERSITY:
   Total Subreddits: {all_data['subreddit'].nunique()}
   Top 5 Subreddits: {dict(list(all_data['subreddit'].value_counts().head(5).items()))}

⚡ QUALITY METRICS:
   High-Quality Posts (>20 words): {(all_data['training_word_count'] > 20).sum():,}
   Engaged Posts (score >10): {(all_data['score'] > 10).sum():,}
   Discussion Posts (>5 comments): {(all_data['num_comments'] > 5).sum():,}

✅ STATUS: Ready for training!
   Output Directory: {self.processed_dir}
   Training Files: train.jsonl, val.jsonl, test.jsonl
   Metadata: training_metadata.json
"""
        return report
    
    def run_processing_pipeline(self):
        """Run the complete data processing pipeline"""
        logger.info("🚀 Starting training data processing pipeline...")
        
        # Load cleaned data
        df = self.load_cleaned_data()
        if df.empty:
            return
        
        # Prepare text for training
        df = self.prepare_text_for_training(df)
        
        # Analyze data distribution
        subreddit_stats = self.analyze_subreddit_distribution(df)
        
        # Create balanced splits
        train_df, val_df, test_df = self.create_balanced_splits(df)
        
        # Save training data
        self.save_training_data(train_df, val_df, test_df)
        
        # Generate and display report
        report = self.generate_processing_report(train_df, val_df, test_df)
        logger.info(report)
        
        logger.success("✅ Training data processing complete!")
        logger.info("📋 Next step: Run 'python training_prep.py' to set up training environment")

def main():
    """Main processing pipeline"""
    logger.info("🚀 Starting training data processing...")
    
    processor = TrainingDataProcessor()
    processor.run_processing_pipeline()

if __name__ == "__main__":
    main() 