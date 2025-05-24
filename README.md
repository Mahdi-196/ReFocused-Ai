# Reddit Data Collection Pipeline

Professional data collection and cleaning pipeline for Reddit content via Pushshift API with automatic file chunking for large datasets.

## Overview

This pipeline provides efficient collection and preprocessing of Reddit data for downstream analysis and processing. Built with production-ready code standards and comprehensive error handling.

## Features

- **High-performance data collection** from multiple subreddits
- **Professional text cleaning** with configurable parameters
- **Automatic file chunking** splits large datasets into manageable files
- **Robust error handling** and logging
- **Configurable filtering** by score, length, and content type
- **Comprehensive dataset statistics**

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Collection

```bash
python fetch_data.py --api-token YOUR_TOKEN --subreddits science technology --limit 500
```

**Parameters:**
- `--api-token`: Pushshift API Bearer token
- `--subreddits`: Target subreddits (default: high-quality discussion subs)
- `--limit`: Items per subreddit (default: 100)
- `--days-back`: Lookback period in days (default: 30)
- `--no-comments`: Collect submissions only
- `--output`: Output file path

### Data Cleaning

```bash
python clean_data.py --min-score 5 --min-length 20 --chunk-size 0.48828125 --stats
```

**Parameters:**
- `--input`: Raw data file (default: data/raw.txt)
- `--output`: Base filename for cleaned data (chunks will be numbered)
- `--min-score`: Minimum score threshold (default: 1)
- `--min-length`: Minimum text length (default: 10)
- `--chunk-size`: Chunk size in GiB (default: 500MB)
- `--stats`: Show dataset statistics

**File Chunking:**
When cleaned data exceeds the chunk size limit, new files are automatically created:
- `data/clean.txt` (first chunk)
- `data/clean_chunk_001.txt` (second chunk)
- `data/clean_chunk_002.txt` (third chunk)
- etc.

## Pipeline Workflow

```
1. Raw Data Collection → data/raw.txt (JSON lines)
2. Text Cleaning + Chunking → data/clean*.txt (Clean text files)
3. Ready for distributed training
```

## Data Format

**Raw Data (JSON Lines):**
```json
{"type": "submission", "subreddit": "science", "title": "...", "text": "...", "score": 156}
{"type": "comment", "subreddit": "science", "text": "...", "score": 23}
```

**Cleaned Data (Text Lines):**
```
Clean, processed text ready for analysis...
Another cleaned text entry...
```

**Multiple Output Files:**
```
data/clean.txt          # 500 MB
data/clean_chunk_001.txt # 500 MB  
data/clean_chunk_002.txt # 256 MB
```

## Configuration

### File Chunking
- Default chunk size: 500 MB (524,288,000 bytes)
- Optimal for data collection and processing workflows
- Prevents memory issues with large datasets
- Enables parallel processing

### Quality Thresholds
- Score filtering removes low-quality content
- Length filtering removes very short posts
- Profanity filtering for content standards

### Subreddit Selection
Default high-quality subreddits:
- `AskReddit` - General discussion
- `todayilearned` - Educational content  
- `explainlikeimfive` - Clear explanations
- `science` - Scientific discussion

## Error Handling

- Automatic retry on rate limits
- Graceful handling of API errors
- Comprehensive logging
- Input validation
- File size monitoring

## Performance

- Rate limiting respects API constraints
- Efficient JSON streaming for large datasets
- Memory-efficient text processing
- Automatic file splitting for scalability
- Configurable batch sizes

## Training Benefits

**Chunked files enable:**
- **Parallel training** across multiple GPUs/machines
- **Memory efficiency** by loading smaller datasets
- **Incremental processing** for very large corpora
- **Resume capability** if training is interrupted
- **Distributed training** with frameworks like PyTorch DDP

## API Requirements

Requires Pushshift API access token. Visit [pushshift.io](https://pushshift.io) for access.