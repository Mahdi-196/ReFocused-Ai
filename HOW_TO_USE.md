# 📖 How to Use the ReFocused-AI Reddit Data Collector

This guide walks you through using the Reddit data collection system to gather 10GB of productivity and self-improvement content from Reddit.

## 🚀 Quick Start Commands

### **Test the System (15-minute validation)**
```bash
python reddit_enhanced_collector.py --test
```

### **Start Full Collection (6-12 hours)**
```bash
python reddit_enhanced_collector.py
```

### **Monitor Progress**
```bash
python monitor_collection.py --detailed
```

---

## 📋 Step-by-Step Instructions

### **Step 1: Verify Setup**

First, make sure everything is working:

```bash
# Test OAuth credentials and API connectivity
python test_oauth_setup.py
```

**Expected Output:**
```
🧪 REDDIT OAUTH SETUP TEST
✅ OAuth authentication successful!
✅ API request successful!
✅ All tests passed! You're ready to run the main collector.
```

### **Step 2: Run Test Collection (Recommended)**

Before starting the full 6-12 hour collection, run a 15-minute test:

```bash
python reddit_enhanced_collector.py --test
```

**What this does:**
- Runs for exactly 15 minutes then stops automatically
- Collects sample data from all 10 subreddits
- Limits to ~1,000 posts per subreddit maximum
- Saves files with `test_` prefix to avoid confusion
- Validates that OAuth, rate limiting, and data saving all work

**Expected Output:**
```
🧪 TEST MODE - ENHANCED REDDIT DATA COLLECTOR
🧪 TEST MODE: 15-minute validation run
💡 This will test the system and collect sample data
⏱️ Collection will automatically stop after 15 minutes

Processing subreddit 1/10: r/productivity
Collecting hot posts from r/productivity
Saved batch 1 for r/productivity (TEST): 245 posts + comments (3.2 MB, 4,567 items)

🧪 Test mode: Stopping collection after 15.0 minutes
🧪 TEST COLLECTION COMPLETED!
✅ System is working correctly
🚀 Ready for full collection - run without --test flag
```

### **Step 3: Start Full Collection**

Once the test passes, start the full collection:

```bash
python reddit_enhanced_collector.py
```

**What this does:**
- Targets 100,000 posts per subreddit (1M total)
- Includes up to 15 comments per post
- Runs for 6-12 hours depending on Reddit's rate limits
- Saves data in 2,000-post batches for memory efficiency
- Automatically handles rate limiting and errors

### **Step 4: Monitor Progress**

Open a second terminal window and monitor the collection:

```bash
# Basic monitoring (updates every 30 seconds)
python monitor_collection.py

# Detailed monitoring with full statistics
python monitor_collection.py --detailed

# One-time status check
python monitor_collection.py --once

# Faster updates (every 10 seconds)
python monitor_collection.py --interval 10 --detailed
```

---

## 📊 Understanding the Output

### **Collection Console Output**
```
🚀 FULL COLLECTION - ENHANCED REDDIT DATA COLLECTOR
Target: 10 subreddits, ~100,000 posts each

Processing subreddit 1/10: r/productivity
Collecting hot posts from r/productivity
Collecting top posts from r/productivity
Saved batch 1 for r/productivity: 2,000 posts + comments (15.2 MB, 32,450 items)
Progress: 2,000 total posts, 1,847 posts/hour

Processing subreddit 2/10: r/getdisciplined
...
```

### **Monitor Output**
```
📊 Collection Status: 125 files, 1.2 GB, 125,000 posts, 1,890,000 comments
🎯 Progress: 11.2% towards 10GB goal
⏱️ Time to 10GB: ~8.3 hours
📈 Active Subreddits: 10/10

📋 Subreddit Breakdown:
Subreddit            Files  Size       Posts    Comments  Avg Score Sources
---------------------------------------------------------------------------------------
productivity         12     156.30 MB  12,450   187,650   45.2      reddit_enhanced
getdisciplined       11     142.80 MB  11,890   178,350   38.7      reddit_enhanced
selfimprovement      10     128.90 MB  10,230   153,450   42.1      reddit_enhanced
```

---

## ⚠️ Important Guidelines

### **During Collection**

**✅ DO:**
- Keep both terminals open (collector + monitor)
- Leave your computer on and connected to internet
- Check progress every few hours
- Let it run overnight if needed

**❌ DON'T:**
- Close the collector terminal window
- Put computer to sleep/hibernate
- Run multiple collectors simultaneously
- Modify files in `data/reddit_enhanced/` during collection

### **If Collection Stops**

If something goes wrong, you can:

1. **Check the logs:**
   ```bash
   tail -f data/reddit_enhanced/collection.log
   ```

2. **Test OAuth again:**
   ```bash
   python test_oauth_setup.py
   ```

3. **Restart collection:**
   ```bash
   python reddit_enhanced_collector.py
   ```
   *Note: The system will create new batch files and continue collecting*

### **Pausing and Resuming**

- **To pause:** Press `Ctrl+C` in the collector terminal
- **To resume:** Run `python reddit_enhanced_collector.py` again
- **Data is saved:** All progress is automatically saved in batches

---

## 📁 Understanding the Data Structure

### **File Organization**
```
data/reddit_enhanced/
├── productivity_batch1_20250523_140000.txt.gz      # ~2,000 posts
├── productivity_batch2_20250523_140500.txt.gz      # ~2,000 posts
├── getdisciplined_batch1_20250523_140200.txt.gz    # ~2,000 posts
├── selfimprovement_batch1_20250523_140300.txt.gz   # ~2,000 posts
├── test_productivity_batch1_20250523_120000.txt.gz # Test data (if ran --test)
└── collection.log                                  # Detailed logs
```

### **File Contents**

Each `.txt.gz` file contains compressed JSON data with one post per line:

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
      "body": "This is exactly what I needed...",
      "score": 23,
      "author": "commenter1"
    }
  ]
}
```

### **File Sizes**
- **Per batch file:** 10-50 MB compressed
- **Per subreddit:** 500MB - 2GB total
- **Full collection:** ~10GB total

---

## 🛠️ Customization Options

### **Adjusting Collection Settings**

Edit `reddit_enhanced_collector.py` to customize:

```python
COLLECTION_CONFIG = {
    'min_score': 1,                        # Minimum upvotes (lower = more posts)
    'target_posts_per_subreddit': 100000,  # Posts per subreddit
    'max_comments_per_post': 15,           # Comments to collect per post
    'concurrent_requests': 2,              # Speed (higher = faster, more risky)
    'save_batch_size': 2000,               # Posts per file
}
```

### **Monitoring Options**

```bash
# Monitor specific directory
python monitor_collection.py --dir data/reddit_enhanced

# Monitor multiple directories
python monitor_collection.py --dir data/reddit_oauth --dir data/reddit_enhanced

# Change update frequency
python monitor_collection.py --interval 5  # Update every 5 seconds
```

---

## 🔧 Troubleshooting

### **Common Issues**

**❌ "OAuth authentication failed"**
```bash
# Check your Reddit app credentials in reddit_enhanced_collector.py
python test_oauth_setup.py
```

**❌ "No module named 'aiohttp'"**
```bash
# Install dependencies
pip install -r requirements.txt
```

**❌ "Rate limit exceeded"**
- This is normal - the system will wait and retry automatically
- Don't restart the collector, just let it wait

**❌ "Disk space full"**
- You need ~10GB free space for the full collection
- Delete test files: `rm data/reddit_enhanced/test_*.txt.gz`

### **Performance Issues**

**Collection too slow?**
- Increase `concurrent_requests` from 2 to 3 (risky)
- Check internet connection stability
- Close other bandwidth-heavy programs

**Computer running slow?**
- Reduce `save_batch_size` from 2000 to 1000
- Close other memory-heavy programs
- Monitor RAM usage

---

## 📈 Expected Timeline

### **Test Mode (--test flag)**
- **Duration:** Exactly 15 minutes
- **Data collected:** ~5,000-15,000 posts
- **File size:** ~50-200 MB
- **Purpose:** Validate system works

### **Full Collection**
- **Duration:** 6-12 hours (varies by rate limits)
- **Data collected:** ~1,000,000 posts + 15,000,000 comments
- **File size:** ~10 GB compressed
- **Rate:** ~1,500-3,000 posts/hour per subreddit

### **Progress Milestones**
- **1 hour:** ~100 MB, 15,000 posts
- **3 hours:** ~1 GB, 150,000 posts  
- **6 hours:** ~5 GB, 500,000 posts
- **12 hours:** ~10 GB, 1,000,000 posts

---

## ✅ Success Checklist

Before starting full collection:
- [ ] `python test_oauth_setup.py` passes
- [ ] `python reddit_enhanced_collector.py --test` completes successfully
- [ ] Monitor shows test data correctly
- [ ] At least 10GB free disk space
- [ ] Stable internet connection
- [ ] Computer set to not sleep/hibernate

Ready for full collection:
- [ ] Run `python reddit_enhanced_collector.py` (without --test)
- [ ] Run `python monitor_collection.py --detailed` in second terminal
- [ ] Let it run for 6-12 hours
- [ ] Check progress every few hours

---

## 🎯 Final Tips

1. **Start with test mode** - Always run `--test` first
2. **Monitor regularly** - Check progress every 2-3 hours
3. **Be patient** - Rate limiting means it takes time
4. **Keep computer awake** - Disable sleep/hibernate during collection
5. **Have backup plans** - The system can be restarted if needed

**Remember:** This is collecting years worth of community wisdom from Reddit's productivity communities. The 6-12 hour investment will give you an incredible dataset for analysis, research, or building applications.

---

*Happy collecting! 🚀* 