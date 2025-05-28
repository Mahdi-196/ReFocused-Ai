# Comprehensive Data Processing Guide

A complete guide to data processing for AI training, covering multi-source processing, quality filtering, and training data preparation.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Processing Scripts](#processing-scripts)
3. [Multi-Source Processing](#multi-source-processing)
4. [Legacy Data Processing](#legacy-data-processing)
5. [Massive Dataset Processing](#massive-dataset-processing)
6. [Quality Control & Filtering](#quality-control--filtering)
7. [Training Data Preparation](#training-data-preparation)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites

```bash
# Install all dependencies
pip install -r ../requirements.txt

# Core processing packages
pip install pandas numpy scikit-learn nltk better-profanity loguru zstandard
```

### Available Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `multi_source_processor.py` | Process real-time Wikipedia + Reddit data | Handle new multi-source collection format |
| `data_cleaner.py` | Advanced Reddit data cleaning | Clean legacy Reddit collections |
| `data_processor.py` | Training data preparation | Prepare cleaned data for model training |
| `setup_massive_dataset.py` | Large-scale dataset setup | Configure massive public datasets |
| `process_massive_dataset.py` | Process massive datasets | Handle TB-scale dataset processing |

## 📊 Processing Scripts

### 1. multi_source_processor.py
**Primary processor for real-time collected Wikipedia + Reddit data**

**Features:**
- Multi-source data integration (Wikipedia + Reddit + Legacy)
- Quality scoring system (0-70 points)
- Advanced deduplication
- Source-aware data splitting
- Comprehensive metadata generation

**Input Sources:**
- `data/real_multi_source/wikipedia/*.jsonl` - Wikipedia articles
- `data/real_multi_source/reddit/*.jsonl` - Reddit posts  
- `data/cleaned/*.jsonl` - Legacy cleaned data
- `data/unified_raw/*.jsonl` - Unified format data

**Usage:**
```bash
cd 02_data_processing/
python multi_source_processor.py
```

**Quality Scoring:**
- **Wikipedia Articles (max 70):**
  - Content length: 0-30 points
  - Has summary: 10 points
  - Has category: 5 points
  - Quality indicators: 15 points
  - Word count: 10 points

- **Reddit Posts (max 70):**
  - Score/engagement: 20 points
  - Comment count: 15 points
  - Content quality: 25 points
  - Title quality: 10 points

**Output Structure:**
```
data/processed/
├── train.jsonl          # Training split (70%)
├── validation.jsonl     # Validation split (15%)
├── test.jsonl          # Test split (15%)
├── train.csv           # Human-readable training data
├── validation.csv      # Human-readable validation data
├── test.csv           # Human-readable test data
└── processing_metadata.json  # Complete processing stats
```

### 2. data_cleaner.py
**Advanced Reddit data cleaner with quality filtering**

**Features:**
- Multi-format support (Reddit API, .zst files, legacy formats)
- NSFW and spam filtering
- Profanity detection and filtering
- Advanced deduplication
- Parallel processing support

**Input Sources:**
- `data/reddit_*/*.txt` - Reddit collection files
- `subreddits24/*.zst` - Compressed subreddit files
- `data/unified_raw/*.jsonl` - Unified format

**Usage:**
```bash
python data_cleaner.py
```

**Cleaning Pipeline:**
1. **Format Normalization** - Standardize different Reddit formats
2. **Content Filtering** - Remove deleted, NSFW, spam content
3. **Quality Scoring** - Score posts based on engagement and content
4. **Deduplication** - Remove duplicate posts using content hashing
5. **Text Cleaning** - Clean and normalize text content

### 3. data_processor.py
**Training data preparation and final formatting**

**Features:**
- Load cleaned or processed data
- Create conversational training format
- Balanced data splitting with stratification
- Subreddit distribution analysis
- Training metadata generation

**Usage:**
```bash
python data_processor.py
```

**Training Text Formats:**
- **Questions:** `Question: {title}\n\nAnswer: {content}`
- **Topics:** `Topic: {title}\n\nDiscussion: {content}`
- **Wikipedia:** `Title: {title}\n\nSummary: {summary}\n\nContent: {content}`

## 🔄 Multi-Source Processing

### Real-Time Data Integration

The `multi_source_processor.py` handles data from the updated collection system:

**Wikipedia Integration:**
```python
# Processes articles from Wikipedia-Collector.py
{
    "title": "Machine Learning",
    "content": "Full article content...",
    "summary": "Article summary...",
    "category": "Computer_science",
    "url": "https://en.wikipedia.org/wiki/Machine_Learning",
    "collected_at": "2024-05-27T14:30:22"
}
```

**Reddit Integration:**
```python
# Processes posts from Reddit API via Wikipedia-Collector.py
{
    "title": "Best resources for learning ML?",
    "selftext": "I'm looking for good resources...",
    "subreddit": "MachineLearning",
    "score": 156,
    "num_comments": 42,
    "collected_at": "2024-05-27T14:30:22"
}
```

### Processing Workflow

1. **Data Loading**
   ```bash
   📖 Loaded 1,247 Wikipedia articles
   💬 Loaded 3,891 Reddit posts
   📁 Loaded 2,156 legacy posts
   ✅ Total data loaded: 7,294 items
   ```

2. **Normalization**
   ```bash
   🔄 Normalizing content from all sources...
   ✅ Normalized 7,156/7,294 items
   ```

3. **Quality Filtering**
   ```bash
   🔍 Filtering content with quality score >= 25.0
   📊 Quality filter: kept 6,891/7,156 items (removed 265)
   ```

4. **Deduplication**
   ```bash
   🔄 Removing duplicate content...
   📊 Deduplication: kept 6,743/6,891 items
   ```

5. **Data Splitting**
   ```bash
   📊 Creating training data splits...
   ✅ Created splits: Train=4,720, Val=1,012, Test=1,011
   ```

## 🗂️ Legacy Data Processing

### Reddit Data Cleaning

For existing Reddit collections, use `data_cleaner.py`:

**Supported Formats:**
- **PRAW API Output** - Direct Reddit API responses
- **ZST Compressed** - `subreddits24/*.zst` files
- **JSONL Files** - Line-delimited JSON format
- **Unified Format** - Processed dataset format

**Quality Metrics:**
- **Score Threshold** - Minimum upvote score
- **Content Length** - Minimum word count
- **Engagement** - Comment count requirements
- **Spam Detection** - Automated spam filtering

### Processing Large Collections

For massive Reddit collections (subreddits24 format):

```bash
# Process compressed subreddit files
python data_cleaner.py

# Expected processing:
📂 Loading raw data from all sources...
📊 Loaded 45,123 posts from subreddits24
🔄 Processing in parallel (8 workers)...
✅ Cleaned data saved to data/cleaned/
```

## 🏗️ Massive Dataset Processing

### Public Dataset Integration

Use `setup_massive_dataset.py` for large-scale datasets:

**Supported Datasets:**
- **OpenWebText** (40GB) - Web content similar to GPT training data
- **BookCorpus** (4GB) - Books and literature
- **Wikipedia Dumps** (20GB) - Complete Wikipedia content
- **CC-News** (76GB) - News articles from Common Crawl
- **Stories** (31GB) - Creative writing and stories

**Setup Process:**
```bash
python setup_massive_dataset.py

# Interactive dataset selection:
Available datasets:
1. OpenWebText (40GB) - Recommended for general training
2. BookCorpus (4GB) - Literature and books
3. CC-News (76GB) - News articles
4. Stories (31GB) - Creative writing
5. Custom HuggingFace dataset

Enter dataset number: 1
```

### Processing Workflow

```bash
python process_massive_dataset.py

# Processing pipeline:
🔍 Scanning dataset: openwebtext
📊 Found 8,013,769 samples
🔄 Processing in chunks of 10,000...
✅ Processed 1M samples (12.5%)
💾 Saved chunk to data/unified_raw/
```

## 🎯 Quality Control & Filtering

### Quality Scoring System

**Wikipedia Quality (0-70 points):**
- **Content Length** (30 pts)
  - 5000+ chars: 30 pts
  - 2000+ chars: 20 pts  
  - 1000+ chars: 10 pts
- **Has Summary** (10 pts)
- **Has Category** (5 pts)
- **Quality Indicators** (15 pts)
  - References/citations: 5 pts
  - Structured content: 5 pts
  - Good sentences: 5 pts
- **Word Count** (10 pts)
  - 1000+ words: 10 pts
  - 500+ words: 5 pts

**Reddit Quality (0-70 points):**
- **Score/Engagement** (20 pts)
  - 100+ score: 20 pts
  - 50+ score: 15 pts
  - 10+ score: 10 pts
- **Comments** (15 pts)
  - 50+ comments: 15 pts
  - 20+ comments: 10 pts
  - 5+ comments: 5 pts
- **Content Quality** (25 pts)
  - 200+ words: 25 pts
  - 100+ words: 20 pts
  - 50+ words: 15 pts
- **Title Quality** (10 pts)
  - 5+ words: 10 pts
  - 3+ words: 5 pts

### Filtering Options

```python
# Adjust quality thresholds
processor = MultiSourceDataProcessor()
processor.run_processing_pipeline(min_quality_score=30.0)  # Higher quality

# Custom filtering
def custom_filter(item):
    return (
        item.get('word_count', 0) > 100 and
        item.get('quality_score', 0) > 25.0 and
        'nsfw' not in item.get('title', '').lower()
    )
```

## 📚 Training Data Preparation

### Output Formats

**JSONL Training Format:**
```json
{
  "id": "wiki_12345",
  "training_text": "Title: Machine Learning\n\nSummary: ML is...\n\nContent: Detailed content...",
  "source_type": "wikipedia",
  "word_count": 1247,
  "quality_score": 65.0
}
```

**CSV Inspection Format:**
- Human-readable data analysis
- Easy filtering and sorting
- Statistical analysis support

### Training Text Structure

**Wikipedia Articles:**
```
Title: Artificial Intelligence

Summary: AI is the simulation of human intelligence...

Content: Artificial intelligence (AI) is intelligence 
demonstrated by machines, as opposed to natural 
intelligence displayed by animals including humans...
```

**Reddit Posts - Questions:**
```
Question: What's the best way to learn Python?

Answer: I'd recommend starting with the official 
Python tutorial, then moving on to practical projects...
```

**Reddit Posts - Discussions:**
```
Topic: My experience learning machine learning

Discussion: After 6 months of studying ML, here are 
the key insights I've gained...
```

## 📊 Data Splits & Metadata

### Training Splits

- **Training (70%)** - Model training
- **Validation (15%)** - Hyperparameter tuning
- **Test (15%)** - Final evaluation

### Metadata Generation

```json
{
  "processing_info": {
    "processed_at": "2024-05-27T14:30:22",
    "total_items": 6743,
    "processing_stats": {
      "wikipedia_articles": 1205,
      "reddit_posts": 3891,
      "removed_duplicates": 148,
      "removed_low_quality": 265
    }
  },
  "source_distribution": {
    "wikipedia": 1205,
    "reddit": 3891,
    "legacy": 1647
  },
  "quality_statistics": {
    "mean_score": 42.3,
    "median_score": 38.5
  },
  "subreddit_distribution": {
    "explainlikeimfive": 245,
    "todayilearned": 198,
    "science": 167
  }
}
```

## 🔧 Best Practices

### Performance Optimization

1. **Parallel Processing**
   ```python
   # Use multiple workers for large datasets
   processor = MultiSourceDataProcessor(n_workers=8)
   ```

2. **Memory Management**
   ```python
   # Process in batches for memory efficiency
   cleaner.run_cleaning_pipeline(batch_size=1000)
   ```

3. **Quality Thresholds**
   ```python
   # Adjust based on your quality requirements
   min_quality_score = 25.0  # Standard
   min_quality_score = 35.0  # High quality only
   ```

### Data Quality Guidelines

1. **Source Diversity** - Include multiple content types
2. **Quality Filtering** - Remove low-quality content
3. **Deduplication** - Eliminate duplicate content
4. **Balanced Splits** - Maintain source distribution across splits
5. **Metadata Tracking** - Keep comprehensive processing records

### Processing Workflow

```bash
# Recommended processing order:
1. python multi_source_processor.py    # Process real-time data
2. python data_cleaner.py             # Clean legacy data (if needed)
3. python data_processor.py           # Final training preparation
```

## 🛠️ Troubleshooting

### Common Issues

**No Data Found:**
```bash
❌ No data found to process
💡 Run the data collection scripts first
```
- **Solution:** Ensure data collection has run and produced output files

**Low Quality Scores:**
```bash
📊 Quality filter: kept 12/1000 items (removed 988)
```
- **Solution:** Lower quality threshold or improve data collection

**Memory Issues:**
```bash
MemoryError during processing
```
- **Solution:** Reduce batch size or use fewer workers

**Format Errors:**
```bash
JSON decode error in file:line
```
- **Solution:** Check data collection output format or clean corrupted files

### Performance Issues

**Slow Processing:**
- Increase number of workers: `n_workers=mp.cpu_count()`
- Process in smaller batches: `batch_size=500`
- Use SSD storage for better I/O performance

**High Memory Usage:**
- Enable incremental processing
- Reduce batch sizes
- Process files individually for very large datasets

### Quality Control

**Low Quality Data:**
- Adjust quality scoring parameters
- Implement custom quality filters
- Review and update filtering criteria

**Insufficient Data Volume:**
- Lower quality thresholds temporarily
- Include more data sources
- Review collection parameters

## 📁 Output Directory Structure

```
data/
├── processed/                    # Multi-source processed data
│   ├── train.jsonl              # Training split
│   ├── validation.jsonl         # Validation split  
│   ├── test.jsonl              # Test split
│   ├── train.csv               # Human-readable training data
│   ├── validation.csv          # Human-readable validation data
│   ├── test.csv               # Human-readable test data
│   └── processing_metadata.json # Processing statistics
├── cleaned/                     # Cleaned legacy data
│   ├── cleaned_reddit_data.jsonl
│   └── cleaning_metadata.json
└── unified_raw/                 # Massive dataset processing
    ├── openwebtext_chunk_001.jsonl
    └── processing_log.json
```

This comprehensive processing system ensures high-quality training data from multiple sources while maintaining full traceability and metadata for reproducible AI model training. 