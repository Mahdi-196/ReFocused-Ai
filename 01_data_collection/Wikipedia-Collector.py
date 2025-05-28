import asyncio
import aiofiles
import json
import time
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
import praw
import wikipediaapi as wikipedia
from loguru import logger
import random
import aiohttp
from collections import defaultdict

# ================================
# CONFIGURATION
# ================================

CONFIG = {
    'data_directory': Path('data/real_multi_source'),
    'run_forever': True,
    'cycle_delay_minutes': 15,  # Wait 15 minutes between cycles
    'max_articles_per_cycle': 500,
    'monitoring': {
        'enabled': True,
        'log_interval_minutes': 5,
        'performance_tracking': True,
        'disk_space_warning_gb': 10
    },
    'reddit': {
        'enabled': True,
        'client_id': 'YOUR_REDDIT_CLIENT_ID',  # Replace with your Reddit app credentials
        'client_secret': 'YOUR_REDDIT_CLIENT_SECRET',
        'user_agent': 'DataCollector:v1.0 (by /u/YourUsername)',
        'subreddits': [
            'explainlikeimfive', 'todayilearned', 'science', 'technology',
            'askscience', 'history', 'philosophy', 'psychology', 'education',
            'books', 'writing', 'selfimprovement', 'productivity', 'lifehacks'
        ],
        'posts_per_subreddit': 25
    },
    'wikipedia': {
        'enabled': True,
        'categories': [
            'Science', 'Technology', 'History', 'Philosophy', 'Psychology',
            'Education', 'Health', 'Mathematics', 'Physics', 'Biology',
            'Chemistry', 'Computer_science', 'Literature', 'Arts'
        ],
        'articles_per_category': 30,
        'min_content_length': 1000
    }
}

CONFIG['data_directory'].mkdir(parents=True, exist_ok=True)

class RealTimeMonitor:
    """Real-time monitoring for the data collection process"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.stats = {
            'cycles_completed': 0,
            'total_articles': 0,
            'total_reddit_posts': 0,
            'total_wikipedia_articles': 0,
            'total_size_bytes': 0,
            'errors': 0,
            'last_update': datetime.now()
        }
        self.monitoring = True
        
    def start_monitoring(self):
        """Start the monitoring thread"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
        logger.info("📊 Real-time monitoring started")
        
    def stop_monitoring(self):
        """Stop the monitoring"""
        self.monitoring = False
        
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._log_system_stats()
                self._log_collection_stats()
                self._check_disk_space()
                time.sleep(CONFIG['monitoring']['log_interval_minutes'] * 60)
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)
                
    def _log_system_stats(self):
        """Log system performance statistics"""
        if not CONFIG['monitoring']['performance_tracking']:
            return
            
        # CPU and Memory usage
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        logger.info(f"💻 System Stats - CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% | Disk: {disk.percent:.1f}%")
        
    def _log_collection_stats(self):
        """Log data collection statistics"""
        runtime = datetime.now() - self.start_time
        hours = runtime.total_seconds() / 3600
        
        # Calculate rates
        articles_per_hour = self.stats['total_articles'] / hours if hours > 0 else 0
        gb_total = self.stats['total_size_bytes'] / (1024**3)
        gb_per_hour = gb_total / hours if hours > 0 else 0
        
        logger.info(f"📈 Collection Stats:")
        logger.info(f"   Runtime: {str(runtime).split('.')[0]}")
        logger.info(f"   Cycles: {self.stats['cycles_completed']}")
        logger.info(f"   Total Articles: {self.stats['total_articles']:,}")
        logger.info(f"   Reddit Posts: {self.stats['total_reddit_posts']:,}")
        logger.info(f"   Wikipedia: {self.stats['total_wikipedia_articles']:,}")
        logger.info(f"   Data Size: {gb_total:.2f} GB")
        logger.info(f"   Rate: {articles_per_hour:.1f} articles/hour, {gb_per_hour:.2f} GB/hour")
        
    def _check_disk_space(self):
        """Check available disk space"""
        disk = psutil.disk_usage('/')
        free_gb = disk.free / (1024**3)
        
        if free_gb < CONFIG['monitoring']['disk_space_warning_gb']:
            logger.warning(f"⚠️ Low disk space: {free_gb:.1f} GB remaining")
            
    def update_stats(self, **kwargs):
        """Update collection statistics"""
        for key, value in kwargs.items():
            if key in self.stats:
                if key.startswith('total_'):
                    self.stats[key] += value
                else:
                    self.stats[key] = value
        self.stats['last_update'] = datetime.now()

class RealDataCollector:
    """Real data collector using actual APIs"""
    
    def __init__(self, monitor: RealTimeMonitor):
        self.monitor = monitor
        self.session = aiohttp.ClientSession()
        self.reddit = None
        self.setup_reddit()
        
    def setup_reddit(self):
        """Initialize Reddit API connection"""
        if not CONFIG['reddit']['enabled']:
            return
            
        try:
            self.reddit = praw.Reddit(
                client_id=CONFIG['reddit']['client_id'],
                client_secret=CONFIG['reddit']['client_secret'],
                user_agent=CONFIG['reddit']['user_agent']
            )
            # Test the connection
            self.reddit.user.me()
            logger.success("✅ Reddit API connected successfully")
        except Exception as e:
            logger.error(f"❌ Reddit API connection failed: {e}")
            logger.warning("🔧 Please update your Reddit API credentials in the config")
            self.reddit = None
            
    async def collect_wikipedia_articles(self, cycle: int) -> int:
        """Collect real Wikipedia articles"""
        if not CONFIG['wikipedia']['enabled']:
            return 0
            
        wiki_dir = CONFIG['data_directory'] / 'wikipedia'
        wiki_dir.mkdir(exist_ok=True)
        
        total_collected = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for category in CONFIG['wikipedia']['categories']:
            try:
                logger.info(f"📖 Scraping Wikipedia: {category}")
                
                # Initialize Wikipedia API
                wiki_wiki = wikipedia.Wikipedia('en')
                
                # Get articles from this category
                articles_data = []
                
                # Try to get articles from the category page
                category_page = wiki_wiki.page(f"Category:{category}")
                if category_page.exists():
                    # Get articles from category members
                    categorymembers = list(category_page.categorymembers.values())[:CONFIG['wikipedia']['articles_per_category']]
                else:
                    # Fall back to search
                    search_url = f"https://en.wikipedia.org/w/api.php"
                    search_params = {
                        'action': 'query',
                        'format': 'json',
                        'list': 'search',
                        'srsearch': category,
                        'srlimit': CONFIG['wikipedia']['articles_per_category']
                    }
                    
                    async with self.session.get(search_url, params=search_params) as response:
                        if response.status == 200:
                            data = await response.json()
                            search_results = data.get('query', {}).get('search', [])
                            categorymembers = [wiki_wiki.page(result['title']) for result in search_results]
                        else:
                            categorymembers = []
                
                for page in categorymembers:
                    try:
                        if not page.exists():
                            continue
                            
                        # Filter by content length
                        if len(page.text) < CONFIG['wikipedia']['min_content_length']:
                            continue
                            
                        article_data = {
                            'title': page.title,
                            'content': page.text,
                            'summary': page.summary,
                            'url': page.fullurl,
                            'category': category,
                            'timestamp': datetime.now().isoformat(),
                            'cycle': cycle,
                            'source': 'wikipedia'
                        }
                        
                        articles_data.append(article_data)
                        logger.debug(f"📄 Collected: {page.title}")
                        
                        # Small delay to be respectful
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        logger.debug(f"⚠️ Skipped article {page.title if hasattr(page, 'title') else 'unknown'}: {e}")
                        continue
                
                # Save category batch
                if articles_data:
                    filename = wiki_dir / f'wikipedia_{category}_cycle_{cycle}_{timestamp}.jsonl'
                    async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
                        for article in articles_data:
                            await f.write(json.dumps(article, ensure_ascii=False) + '\n')
                    
                    total_collected += len(articles_data)
                    self.monitor.update_stats(total_wikipedia_articles=len(articles_data))
                    logger.success(f"📖 Wikipedia {category}: {len(articles_data)} articles")
                
            except Exception as e:
                logger.error(f"❌ Error collecting Wikipedia {category}: {e}")
                self.monitor.update_stats(errors=1)
                continue
                
        return total_collected
        
    async def collect_reddit_posts(self, cycle: int) -> int:
        """Collect real Reddit posts"""
        if not CONFIG['reddit']['enabled'] or not self.reddit:
            return 0
            
        reddit_dir = CONFIG['data_directory'] / 'reddit'
        reddit_dir.mkdir(exist_ok=True)
        
        total_collected = 0
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for subreddit_name in CONFIG['reddit']['subreddits']:
            try:
                logger.info(f"🔴 Scraping Reddit: r/{subreddit_name}")
                
                subreddit = self.reddit.subreddit(subreddit_name)
                posts_data = []
                
                # Get hot posts
                for post in subreddit.hot(limit=CONFIG['reddit']['posts_per_subreddit']):
                    try:
                        # Skip if post is too short
                        content = post.selftext if post.selftext else post.title
                        if len(content) < 50:
                            continue
                            
                        # Load more comments
                        post.comments.replace_more(limit=5)
                        comments = []
                        for comment in post.comments.list()[:10]:  # Top 10 comments
                            if hasattr(comment, 'body') and len(comment.body) > 20:
                                comments.append({
                                    'body': comment.body,
                                    'score': comment.score,
                                    'created_utc': comment.created_utc
                                })
                        
                        post_data = {
                            'title': post.title,
                            'content': post.selftext,
                            'url': post.url,
                            'subreddit': subreddit_name,
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'created_utc': post.created_utc,
                            'author': str(post.author) if post.author else '[deleted]',
                            'comments': comments,
                            'timestamp': datetime.now().isoformat(),
                            'cycle': cycle,
                            'source': 'reddit'
                        }
                        
                        posts_data.append(post_data)
                        logger.debug(f"🔴 Collected: {post.title[:50]}...")
                        
                    except Exception as e:
                        logger.debug(f"⚠️ Skipped post: {e}")
                        continue
                
                # Save subreddit batch
                if posts_data:
                    filename = reddit_dir / f'reddit_{subreddit_name}_cycle_{cycle}_{timestamp}.jsonl'
                    async with aiofiles.open(filename, 'w', encoding='utf-8') as f:
                        for post in posts_data:
                            await f.write(json.dumps(post, ensure_ascii=False) + '\n')
                    
                    total_collected += len(posts_data)
                    self.monitor.update_stats(total_reddit_posts=len(posts_data))
                    logger.success(f"🔴 Reddit r/{subreddit_name}: {len(posts_data)} posts")
                
                # Respectful delay between subreddits
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error collecting Reddit r/{subreddit_name}: {e}")
                self.monitor.update_stats(errors=1)
                continue
                
        return total_collected
        
    async def close(self):
        """Clean up resources"""
        await self.session.close()

async def real_collection_loop():
    """Main collection loop with real APIs"""
    monitor = RealTimeMonitor()
    collector = RealDataCollector(monitor)
    
    logger.info("🚀 STARTING REAL MULTI-SOURCE DATA COLLECTION")
    logger.info("📡 Using actual Wikipedia and Reddit APIs")
    logger.info("📊 Real-time monitoring enabled")
    
    # Start monitoring
    if CONFIG['monitoring']['enabled']:
        monitor.start_monitoring()
    
    cycle = 0
    try:
        while CONFIG['run_forever']:
            cycle += 1
            cycle_start = datetime.now()
            
            logger.info(f"🔄 STARTING COLLECTION CYCLE {cycle}")
            
            # Collect from all sources
            wikipedia_count = 0
            reddit_count = 0
            
            try:
                # Wikipedia collection
                if CONFIG['wikipedia']['enabled']:
                    wikipedia_count = await collector.collect_wikipedia_articles(cycle)
                
                # Reddit collection  
                if CONFIG['reddit']['enabled']:
                    reddit_count = await collector.collect_reddit_posts(cycle)
                
            except Exception as e:
                logger.error(f"❌ Error in cycle {cycle}: {e}")
                monitor.update_stats(errors=1)
            
            # Update cycle stats
            cycle_end = datetime.now()
            cycle_duration = cycle_end - cycle_start
            total_articles = wikipedia_count + reddit_count
            
            monitor.update_stats(
                cycles_completed=1,
                total_articles=total_articles
            )
            
            # Calculate data size
            total_size = 0
            for source_dir in CONFIG['data_directory'].iterdir():
                if source_dir.is_dir():
                    for file in source_dir.rglob('*'):
                        if file.is_file():
                            total_size += file.stat().st_size
            
            monitor.stats['total_size_bytes'] = total_size
            
            logger.success(f"✅ CYCLE {cycle} COMPLETED!")
            logger.info(f"⏱️ Duration: {cycle_duration}")
            logger.info(f"📊 Collected: {total_articles} items ({wikipedia_count} Wikipedia, {reddit_count} Reddit)")
            logger.info(f"💾 Total data: {total_size / (1024**3):.2f} GB")
            
            # Wait before next cycle
            if cycle > 1:  # First cycle runs immediately
                logger.info(f"😴 Waiting {CONFIG['cycle_delay_minutes']} minutes before next cycle...")
                await asyncio.sleep(CONFIG['cycle_delay_minutes'] * 60)
            else:
                await asyncio.sleep(60)  # 1 minute after first cycle
                
    except KeyboardInterrupt:
        logger.info("🛑 Collection stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
    finally:
        monitor.stop_monitoring()
        await collector.close()
        logger.success("🎉 REAL DATA COLLECTION STOPPED")

if __name__ == "__main__":
    print("🚀 REAL MULTI-SOURCE DATA COLLECTOR")
    print("=" * 50)
    print("📡 Wikipedia Articles + Reddit Posts")
    print("📊 Real-time monitoring and statistics")
    print("🔄 Continuous collection with rate limiting")
    print("=" * 50)
    print("\n🔧 SETUP REQUIRED:")
    print("1. Install: pip install praw wikipedia-api aiofiles aiohttp psutil loguru")
    print("2. Get Reddit API credentials: https://www.reddit.com/prefs/apps")
    print("3. Update CONFIG['reddit'] with your credentials")
    print("=" * 50)
    
    try:
        asyncio.run(real_collection_loop())
    except KeyboardInterrupt:
        print("\n🛑 Collection stopped by user")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")