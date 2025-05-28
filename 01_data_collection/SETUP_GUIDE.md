# Comprehensive Data Collection Guide

A complete guide to data collection for AI training, covering real-time collection, large-scale datasets, and monitoring tools.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Collection Scripts](#collection-scripts)
3. [Real-Time Data Collection](#real-time-data-collection)
4. [Large-Scale Public Datasets](#large-scale-public-datasets)
5. [Academic & Research Datasets](#academic--research-datasets)
6. [Monitoring & Pipeline Management](#monitoring--pipeline-management)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Prerequisites

```bash
# Install all dependencies
pip install -r ../requirements.txt

# Core data collection packages
pip install praw wikipedia-api aiofiles aiohttp psutil loguru requests beautifulsoup4
```

### Available Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `Wikipedia-Collector.py` | Real-time Wikipedia + Reddit collection | Live data gathering with monitoring |
| `monitor_reddit_pipeline.py` | Pipeline monitoring and statistics | Track processing progress and system health |

## 📡 Collection Scripts

### 1. Wikipedia-Collector.py
**Real-time multi-source data collector with live monitoring**

**Features:**
- Wikipedia article scraping by category
- Reddit API integration with PRAW
- Real-time system monitoring (CPU, RAM, disk)
- Asynchronous processing with rate limiting
- JSON output with full metadata
- Error handling and recovery

**Configuration:**
```python
CONFIG = {
    'data_directory': Path('data/real_multi_source'),
    'cycle_delay_minutes': 15,
    'wikipedia': {
        'categories': ['Science', 'Technology', 'History', ...],
        'articles_per_category': 30,
        'min_content_length': 1000
    },
    'reddit': {
        'subreddits': ['explainlikeimfive', 'todayilearned', ...],
        'posts_per_subreddit': 25
    }
}
```

**Usage:**
```bash
cd 01_data_collection/
python Wikipedia-Collector.py
```

**Output Structure:**
```
data/real_multi_source/
├── wikipedia/
│   └── wikipedia_Science_cycle_1_20240527_143022.jsonl
└── reddit/
    └── reddit_explainlikeimfive_cycle_1_20240527_143122.jsonl
```

### 2. monitor_reddit_pipeline.py
**Real-time pipeline monitoring dashboard**

**Features:**
- System resource monitoring
- Processing status tracking
- Data statistics and progress
- Log file analysis
- Live dashboard updates

**Usage:**
```bash
python monitor_reddit_pipeline.py
```

**Dashboard Output:**
```
🔍 Reddit Data Pipeline Monitor - subreddits24
============================================================
⏰ 2024-05-27 14:30:22
🕐 Runtime: 2:15:30

💻 SYSTEM RESOURCES:
   RAM: 8.2/16.0GB (51.3%)
   CPU: 15.2% (8 cores)
   Disk: 156.8GB used (78.4%)

📱 REDDIT DATA (subreddits24):
   Files: 156 .zst files
   Size: 12.4GB
   Subreddits: 78 unique

⚙️ PROCESSING STATUS:
   ✅ RUNNING (PID: 12345)
      Runtime: 1.5 hours
      Memory: 2.1GB
```

## 🎓 Real-Time Data Collection

### Reddit API Setup

Based on the [Reddit Developer Documentation](https://developers.reddit.com/docs/quickstart) and [integration guides](https://rollout.com/integration-guides/reddit/sdk/step-by-step-guide-to-building-a-reddit-api-integration-in-python):

1. **Create Reddit App**
   - Visit [Reddit App Preferences](https://www.reddit.com/prefs/apps)
   - Click "Create App" → Select "script"
   - Fill required fields:
     - **Name**: `DataCollector`
     - **Description**: `AI training data collection`
     - **Redirect URI**: `http://localhost:8080`

2. **Get Credentials**
   - **Client ID**: 14-character string under app name
   - **Client Secret**: Longer secret string
   - **User Agent**: `DataCollector:v1.0 (by /u/YourUsername)`

3. **Configure Script**
   ```python
   'reddit': {
       'client_id': 'YOUR_14_CHAR_CLIENT_ID',
       'client_secret': 'YOUR_SECRET_KEY',
       'user_agent': 'DataCollector:v1.0 (by /u/YourUsername)'
   }
   ```

### Wikipedia API Integration

Uses the [Wikimedia REST API](https://api.wikimedia.org/wiki/Getting_started_with_Wikimedia_APIs) and [MediaWiki Action API](https://www.mediawiki.org/wiki/API:Tutorial):

**Features:**
- **Rate Limiting**: 0.5 seconds between requests
- **Content Filtering**: Minimum 1000 characters
- **Category-based**: Organized by topic categories
- **Full Content**: Article text, summaries, URLs, metadata

**API Access Methods:**
```python
# Using wikipedia-api package
import wikipediaapi as wikipedia
wiki_wiki = wikipedia.Wikipedia('en')

# Direct REST API calls
headers = {
    'User-Agent': 'DataCollector/1.0 (your@email.com)',
    'Api-User-Agent': 'DataCollector/1.0 (your@email.com)'
}
response = requests.get('https://en.wikipedia.org/api/rest_v1/page/title/Earth', headers=headers)
```

## 🗂️ Large-Scale Public Datasets

### Basic Datasets (10GB - 100GB)

#### Wikipedia Dumps
**Description**: Complete Wikipedia text dumps with 60M+ articles
- **Size**: ~20GB compressed, ~80GB uncompressed
- **License**: Creative Commons
- **Download**: [Wikipedia Database Download](https://dumps.wikimedia.org/enwiki/)
- **Use Case**: General language understanding, factual content
- **Format**: XML, can be processed to text/JSON

#### Project Gutenberg
**Description**: 15,000+ public domain books across all genres
- **Size**: ~10GB
- **License**: Public Domain
- **Download**: [Kaggle - 15000 Gutenberg Books](https://www.kaggle.com/datasets/pgcorpus/gutenberg-books)
- **Use Case**: Literary language patterns, narrative text
- **Content**: Novels, poetry, historical texts

#### WikiText-103
**Description**: Curated subset of verified Wikipedia articles
- **Size**: ~500MB (100M+ tokens)
- **License**: Creative Commons
- **Access**: [Hugging Face - Salesforce/wikitext](https://huggingface.co/datasets/wikitext)
- **Use Case**: Academic benchmarking, smaller-scale training
- **Quality**: Only "Good" and "Featured" Wikipedia articles

#### BookCorpus
**Description**: 11,000+ unpublished novels totaling 1B words
- **Size**: ~4GB
- **License**: Research use
- **Download**: [Papers with Code - BookCorpus](https://paperswithcode.com/dataset/bookcorpus)
- **Use Case**: Conversational and narrative text modeling
- **Content**: Modern fiction, dialogue-heavy content

### Advanced Large-Scale Datasets (100GB - 1TB+)

#### Common Crawl
**Description**: Petabyte-scale web crawl from across the internet
- **Size**: 10+ TB per monthly crawl
- **License**: Various (respect robots.txt)
- **Download**: [Common Crawl Homepage](https://commoncrawl.org/)
- **Use Case**: Massive pretraining corpora
- **Format**: Raw HTML and extracted text
- **Processing**: Requires significant filtering and cleaning

#### OpenWebText
**Description**: Open reproduction of OpenAI's WebText (GPT-2 training data)
- **Size**: ~38GB
- **License**: Various web licenses
- **Access**: [Hugging Face - Skylion007/openwebtext](https://huggingface.co/datasets/openwebtext)
- **Use Case**: GPT-style language modeling
- **Content**: Web pages linked from Reddit posts (≥3 upvotes)

#### The Pile
**Description**: 825GB curated dataset from 22 high-quality sources
- **Size**: ~825GB
- **License**: Mixed (see individual components)
- **Access**: [Hugging Face - EleutherAI/pile](https://huggingface.co/datasets/EleutherAI/pile)
- **Use Case**: State-of-the-art language modeling
- **Components**: Academic papers, books, code, web content, etc.

#### CC100
**Description**: 2TB multilingual corpus covering 100+ languages
- **Size**: ~2TB
- **License**: Common Crawl license
- **Download**: [Metatext CC100](https://huggingface.co/datasets/cc100)
- **Use Case**: Multilingual and cross-lingual models
- **Languages**: 100+ languages with quality filtering

## 🔬 Academic & Research Datasets

### Scientific Literature

#### arXiv Papers
**Description**: 1.7M+ scientific articles across all fields
- **Size**: ~100GB (full text)
- **License**: arXiv license (varies by paper)
- **Access**: [Kaggle - Cornell-University/arxiv](https://www.kaggle.com/Cornell-University/arxiv)
- **Content**: Titles, abstracts, full text, metadata
- **Fields**: Physics, CS, Math, Biology, Economics, etc.

### Academic Torrents

#### Wikipedia + BookCorpus (Shuffled)
**Description**: Pre-processed Wikipedia and BookCorpus combination
- **Size**: ~20GB
- **Download**: [Hugging Face - sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen](https://huggingface.co/datasets/sradc/chunked-shuffled-wikipedia20220301en-bookcorpusopen)
- **Use Case**: Ready-to-use training data
- **Processing**: Pre-chunked and shuffled for efficient training

#### Large-Scale Academic Datasets
**Academic Torrents Platform**: [academictorrents.com](https://academictorrents.com/)

**Featured Dataset**: [Basic Datasets Collection](https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13)
- **Description**: Curated collection of fundamental ML datasets
- **Size**: Various (1GB - 100GB+ depending on selection)
- **License**: Mixed (academic/research friendly)
- **Use Case**: Benchmarking, research, education

### Specialized Datasets

#### Legal Text
- **Case Law**: [Caselaw Access Project](https://case.law/) - 40M+ court decisions
- **Contracts**: [CUAD](https://www.atticusprojectai.org/cuad) - Legal contract analysis
- **Patents**: [Google Patents Public Dataset](https://patents.google.com/api/docs/) - Millions of patents

#### Code & Technical
- **GitHub Code**: [CodeSearchNet](https://github.com/github/CodeSearchNet) - 6M+ functions
- **Stack Overflow**: [Stack Overflow Data Dump](https://archive.org/details/stackexchange) - Q&A pairs
- **Programming Books**: [Free Programming Books](https://github.com/EbookFoundation/free-programming-books)

#### News & Current Events
- **Reuters**: [Reuters Corpus](https://trec.nist.gov/data/reuters/reuters.html) - News articles
- **NYT**: [New York Times API](https://developer.nytimes.com/) - Archive & current news
- **All The News**: [Kaggle All The News](https://www.kaggle.com/snapcrack/all-the-news) - 200k+ articles

#### Conversational Data
- **PersonaChat**: [Facebook Persona Chat](https://huggingface.co/datasets/persona_chat) - Personality-based conversations
- **Reddit Conversations**: [Pushshift Reddit](https://files.pushshift.io/reddit/) - Massive Reddit archive
- **Ubuntu Dialogue**: [Ubuntu Dialogue Corpus](https://www.kaggle.com/rtatman/ubuntu-dialogue-corpus) - Technical support conversations

## 📊 Monitoring & Pipeline Management

### Real-Time Monitoring Features

The monitoring system provides comprehensive oversight:

```bash
# Start monitoring dashboard
python monitor_reddit_pipeline.py

# Monitor specific collection
python Wikipedia-Collector.py  # Includes built-in monitoring
```

**Monitored Metrics:**
- **System Resources**: CPU, RAM, disk usage
- **Collection Stats**: Articles/hour, data size, error rates
- **Process Status**: Running jobs, memory usage, runtime
- **Data Pipeline**: Processing progress, file counts, log activity

### Performance Optimization

**Memory Management:**
- Batch processing to avoid memory overflow
- Configurable chunk sizes
- Automatic garbage collection

**Disk Space:**
- Configurable warnings (default: 10GB remaining)
- Compression for long-term storage
- Automatic cleanup of temporary files

**Rate Limiting:**
- Wikipedia: 0.5s between requests (respects [API guidelines](https://api.wikimedia.org/wiki/Getting_started_with_Wikimedia_APIs))
- Reddit: 2s between subreddits
- Respects API terms of service

## 💡 Best Practices

### Data Quality

1. **Content Filtering**
   - Minimum content length thresholds
   - Language detection and filtering
   - Duplicate detection and removal
   - Quality scoring (for web content)

2. **Metadata Preservation**
   - Source URLs and timestamps
   - Processing metadata
   - Licensing information
   - Quality indicators

### Legal & Ethical Considerations

1. **API Compliance**
   - Respect rate limits and terms of service
   - Use appropriate user agents
   - Monitor for API changes

2. **Content Licensing**
   - Verify licensing compatibility
   - Maintain attribution where required
   - Document data provenance

3. **Privacy & Safety**
   - Filter personal information
   - Exclude sensitive content
   - Follow data protection regulations

### Storage & Processing

1. **File Organization**
   ```
   data/
   ├── raw/           # Original downloaded data
   ├── processed/     # Cleaned and formatted
   ├── chunks/        # Split for training
   └── metadata/      # Processing logs and stats
   ```

2. **Format Standards**
   - JSONL for structured data
   - UTF-8 encoding
   - Consistent field naming
   - Comprehensive metadata

## 🔧 Troubleshooting

### Common Issues

#### Reddit API
- **Error**: `401 Unauthorized`
- **Solution**: Check client_id and client_secret
- **Prevention**: Test credentials with simple request first

#### Wikipedia API
- **Error**: `User-Agent required`
- **Solution**: Set proper User-Agent header as per [Wikimedia guidelines](https://api.wikimedia.org/wiki/Getting_started_with_Wikimedia_APIs)
- **Format**: `'User-Agent': 'YourApp/1.0 (your@email.com)'`

#### Rate Limiting
- **Error**: `429 Too Many Requests`
- **Solution**: Built-in rate limiting should prevent this
- **Recovery**: Automatic backoff and retry

#### Disk Space
- **Warning**: `Low disk space`
- **Solution**: Free space or change data directory
- **Prevention**: Monitor disk usage in dashboard

#### Memory Issues
- **Error**: `MemoryError` or system slowdown
- **Solution**: Reduce batch sizes in CONFIG
- **Prevention**: Monitor RAM usage in dashboard

### Performance Tuning

#### For Large-Scale Collection
```python
CONFIG.update({
    'cycle_delay_minutes': 5,      # Faster cycles
    'articles_per_category': 50,   # More articles
    'posts_per_subreddit': 50,     # More posts
})
```

#### For Resource-Constrained Systems
```python
CONFIG.update({
    'cycle_delay_minutes': 30,     # Slower cycles
    'articles_per_category': 10,   # Fewer articles
    'monitoring': {'enabled': False}  # Disable monitoring
})
```

### Log Analysis

Check logs for issues:
```bash
# View real-time logs
tail -f logs/collection.log

# Search for errors
grep "ERROR" logs/collection.log

# Monitor progress
grep "✅ CYCLE" logs/collection.log
```

## 📥 Dataset Download Scripts

### Quick Download Commands

```bash
# Wikipedia dumps
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

# Common Crawl (example)
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2024-10/warc.paths.gz

# Academic Torrents
# Install academic torrents client first
pip install academictorrents
import academictorrents as at
at.get("ba051999301b109eab37d16f027b3f49ade2de13")  # Basic datasets
```

### Automated Download Script

```python
#!/usr/bin/env python3
"""
Automated dataset downloader
"""
import requests
import os
from pathlib import Path

datasets = {
    'wikitext-103': 'https://huggingface.co/datasets/wikitext',
    'openwebtext': 'https://huggingface.co/datasets/openwebtext',
    'pile': 'https://huggingface.co/datasets/EleutherAI/pile'
}

def download_dataset(name, url):
    """Download dataset using appropriate method"""
    print(f"Downloading {name} from {url}")
    # Implementation would depend on dataset format
    pass
```

## 🎯 Getting Started Checklist

1. ✅ **Install Dependencies**
   ```bash
   pip install -r ../requirements.txt
   ```

2. ✅ **Configure APIs** (if using real-time collection)
   - Reddit API credentials
   - Wikipedia API user agent
   - Test connections

3. ✅ **Choose Collection Method**
   - Real-time: `Wikipedia-Collector.py`
   - Monitoring: `monitor_reddit_pipeline.py`
   - Large datasets: Use download links above

4. ✅ **Start Collection**
   ```bash
   python Wikipedia-Collector.py
   ```

5. ✅ **Monitor Progress**
   ```bash
   python monitor_reddit_pipeline.py
   ```

6. ✅ **Process Data**
   - Move to `02_data_processing/` for cleaning
   - Use `03_tokenizer_training/` for tokenization
   - Proceed to `04_data_tokenization/` for model prep

---

## 📚 Additional Resources

### Official Documentation
- [PRAW Documentation](https://praw.readthedocs.io/) - Reddit API wrapper
- [Wikimedia API Portal](https://api.wikimedia.org/wiki/Getting_started_with_Wikimedia_APIs) - Official Wikipedia API guide
- [MediaWiki API Tutorial](https://www.mediawiki.org/wiki/API:Tutorial) - Comprehensive API guide

### Dataset Repositories
- [Hugging Face Datasets](https://huggingface.co/datasets) - ML dataset repository
- [Academic Torrents](https://academictorrents.com/) - Research data sharing
- [Papers with Code Datasets](https://paperswithcode.com/datasets) - ML research datasets
- [Google Dataset Search](https://datasetsearch.research.google.com/) - Find datasets across the web

### Tools & Libraries
- [Wikipedia API Guide](https://zuplo.com/blog/2024/09/30/wikipedia-api-guide) - Comprehensive API usage guide
- [GCDI Web Scraping Guide](https://gcdi.commons.gc.cuny.edu/2024/11/01/web-scraping-with-python-and-the-reddit-api/) - Academic perspective on Reddit scraping

**Next Steps**: After data collection, proceed to `02_data_processing/` for cleaning and preparation. 