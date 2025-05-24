# ReFocused-AI: Large-Scale Reddit Data Collection System

Ever wondered what it would look like to collect massive amounts of productivity and self-improvement content from Reddit? This project does exactly that - it's designed to gather around 10GB of high-quality posts and comments from Reddit's most active productivity-focused communities.

**📖 For detailed usage instructions, see [HOW_TO_USE.md](HOW_TO_USE.md)**

## What This Project Does

This system collects comprehensive data from 10 hand-picked subreddits that focus on productivity, self-improvement, and building better habits. We're talking about communities like r/productivity, r/getdisciplined, r/selfimprovement, and others where people share real strategies, success stories, and advice for getting their lives together.

The goal isn't just to grab a few posts - we're aiming for **10 gigabytes** of compressed data, which translates to roughly a million posts plus their comment threads. That's enough content to power serious research, build recommendation systems, or train AI models on what actually helps people become more productive.

## Quick Start

### Test the System (15-minute validation)
```bash
python reddit_enhanced_collector.py --test
```

### Start Full Collection (6-12 hours)
```bash
python reddit_enhanced_collector.py
```

### Monitor Progress
```bash
python monitor_collection.py --detailed
```

## Why This Approach?

### The Reddit API Challenge
Reddit's official API is... let's say "limiting." You can typically only grab about 1,000 posts at a time, and you're restricted to recent content from specific sorting methods. If you want historical data or comprehensive coverage, you're out of luck with basic API calls.

This project solves that by using OAuth authentication (which gives us higher rate limits) combined with intelligent data collection strategies that systematically gather content from multiple time periods and sorting methods.

### Smart Data Collection Strategy
Instead of just grabbing the "hot" posts from last week, our collector:

- **Casts a wide net**: Collects from hot, top, new, and rising posts
- **Goes deep historically**: Pulls top posts from all time, past year, past month, and past week  
- **Includes the conversation**: Grabs up to 15 top comments per post (this is where the real insights often are)
- **Stays efficient**: Uses maximum compression and saves data in manageable batches
- **Respects rate limits**: Built-in delays and retry logic to avoid getting blocked
- **Test mode included**: 15-minute validation run to ensure everything works

## How Everything Works Together

### The Collection Engine (`reddit_enhanced_collector.py`)
This is the heart of the system. It's built around several key principles:

**OAuth Authentication**: Instead of anonymous API calls, we authenticate with Reddit's OAuth system. This gives us higher rate limits and more reliable access. The system automatically handles token refresh and authentication failures.

**Test Mode**: Before running the full 6-12 hour collection, you can run a 15-minute test with `--test` flag. This validates all systems work correctly and collects sample data.

**Parallel Processing**: The collector works through all 10 subreddits systematically, but within each subreddit, it's smart about gathering diverse content. It rotates through different sorting methods and time periods to avoid getting stuck in echo chambers.

**Batch Processing**: Rather than trying to hold a million posts in memory, the system saves data in batches of 2,000 posts. This keeps memory usage low and means you don't lose everything if something goes wrong mid-collection.

**Quality Filtering**: We set a minimum threshold (posts need at least 1 upvote) to filter out spam and deleted content, but it's low enough to capture a wide range of content.

### The Monitoring System (`monitor_collection.py`)
Data collection at this scale takes hours, so you need to know what's happening. The monitor provides:

- Real-time statistics on posts and comments collected
- Progress tracking toward the 10GB goal
- Time estimates for completion
- Per-subreddit breakdowns so you can see which communities are most active
- Support for monitoring multiple collection runs simultaneously

### Testing and Validation (`test_oauth_setup.py`)
Before you start a 10-hour data collection run, you want to know everything works. This script validates your Reddit API credentials, tests connectivity, and makes sure the authentication flow is working properly.

## Project Structure and File Organization

```
ReFocused-Ai/
├── reddit_enhanced_collector.py    # Main collection engine
├── test_oauth_setup.py            # Setup validation and testing
├── monitor_collection.py          # Real-time progress monitoring
├── requirements.txt               # Python dependencies
├── .gitignore                     # Keeps data out of version control
├── HOW_TO_USE.md                  # Detailed usage instructions
└── data/                          # Collection output (ignored by git)
    └── reddit_enhanced/           # Compressed data files organized by subreddit
```

The data directory structure is intentionally simple. Each subreddit gets its own compressed files, named with timestamps so you can track when data was collected. The files use `.txt.gz` format - they're JSON lines compressed with maximum compression to save space.

## Installation and Usage

### Getting Started
```bash
# Clone and set up
git clone <your-repo>
cd ReFocused-Ai
pip install -r requirements.txt

# Test your setup (optional but recommended)
python test_oauth_setup.py

# Run 15-minute validation test
python reddit_enhanced_collector.py --test

# Start collecting data (full collection)
python reddit_enhanced_collector.py
```

### Monitoring Your Collection
```bash
# Simple progress check
python monitor_collection.py --once

# Live monitoring with detailed stats
python monitor_collection.py --detailed

# Monitor specific data directory
python monitor_collection.py --dir data/reddit_enhanced --detailed
```

**📖 For complete step-by-step instructions, troubleshooting, and examples, see [HOW_TO_USE.md](HOW_TO_USE.md)**

## Technical Design Decisions

### Why These Subreddits?
The 10 target subreddits were chosen because they represent different aspects of productivity and self-improvement:

- **r/productivity, r/getdisciplined**: Core productivity advice and systems
- **r/selfimprovement, r/decidingtobebetter**: Personal development and motivation  
- **r/lifehacks, r/getmotivated**: Practical tips and motivational content
- **r/askpsychology**: Science-backed insights into behavior change
- **r/simpleliving, r/zenhabits**: Minimalism and mindful productivity
- **r/fitness**: Physical health as foundation for productivity

### Why Collect Comments?
The real value in Reddit data is often in the comments. Posts might share a productivity tip, but the comments contain:
- Personal experiences with that tip
- Variations and modifications that work for different people
- Discussion of why certain approaches do or don't work
- Follow-up questions and clarifications

By collecting up to 15 top comments per post, we capture these richer conversations.

### Why This Data Volume?
10GB of compressed text data represents roughly:
- 1,000,000 posts across all subreddits
- 15,000,000 comments and replies
- Several years worth of community knowledge and discussion

This volume provides enough data for meaningful analysis while being manageable to process and store.

## Configuration and Customization

The system is designed to be configurable. Key settings in `reddit_enhanced_collector.py`:

```python
# Collection targets
'target_posts_per_subreddit': 100000,  # Aim for 100k posts per subreddit
'min_score': 1,                        # Include posts with 1+ upvotes
'collect_comments': True,              # Grab comment threads
'max_comments_per_post': 15,           # Top 15 comments per post

# Performance settings  
'concurrent_requests': 2,              # Conservative to avoid rate limits
'save_batch_size': 2000,              # Save every 2000 posts
'compression_level': 9,                # Maximum compression
```

You can adjust these based on your needs - want faster collection? Increase concurrent requests. Need more comments? Bump up the comments per post. Want to focus on higher-quality posts? Raise the minimum score threshold.

## Data Format

Collected data is stored as compressed JSON lines files. Each line contains a complete post with its metadata and associated comments:

```json
{
  "id": "abc123",
  "title": "My productivity system that changed everything",
  "selftext": "Here's what I've learned...",
  "score": 156,
  "num_comments": 42,
  "author": "username",
  "subreddit": "productivity",
  "created_utc": 1704067200,
  "comments": [
    {
      "id": "def456", 
      "body": "This is exactly what I needed to hear...",
      "score": 23,
      "author": "commenter1"
    }
  ]
}
```

This format makes it easy to process with standard tools while preserving the complete conversation context.

## Why This Matters

Reddit contains some of the most authentic, real-world advice about productivity and self-improvement available anywhere. Unlike polished blog posts or academic papers, Reddit discussions capture:

- What actually works for real people in practice
- Common obstacles and how people overcome them  
- Diverse perspectives and approaches to the same challenges
- The evolution of ideas through community discussion

By collecting this data systematically and at scale, we create a resource for understanding how people actually build better habits, overcome procrastination, and improve their lives.

## Dependencies and Requirements

The system is built on a minimal set of reliable dependencies:

- **aiohttp**: Async HTTP client for efficient API calls
- **aiofiles**: Async file operations for handling large data volumes  
- **tenacity**: Robust retry logic for network operations
- **loguru**: Clean, informative logging
- **pmaw**: Reddit data collection utilities

Everything is designed to work on Windows, macOS, and Linux with Python 3.8+.

## What You'll End Up With

After a successful collection run (typically 6-12 hours), you'll have:
- Gigabytes of organized, compressed Reddit data
- Comprehensive logs of the collection process
- Data spanning multiple time periods and discussion types
- A dataset suitable for research, analysis, or machine learning projects

The data represents one of the most comprehensive collections of productivity and self-improvement discussions available, capturing both the wisdom and the genuine struggles of people trying to build better lives.

---

*Ready to dive into the collective wisdom of Reddit's productivity communities? Start with the test mode and let it work its magic.* 