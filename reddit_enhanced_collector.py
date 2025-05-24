#!/usr/bin/env python3
"""
Enhanced Reddit Data Collector with OAuth Authentication
Optimized for collecting ~10GB of productivity/self-improvement data
"""

import asyncio
import aiohttp
import aiofiles
import gzip
import json
import time
import base64
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from loguru import logger

# ================================
# CONFIGURATION SECTION
# ================================

REDDIT_CONFIG = {
    'client_id': 'veDpt5XBEiXhF8D4oR_JkA',
    'client_secret': 'NP4U3DjGdSSL3n40OsibUhPuyEclGQ',
    'redirect_uri': 'http://localhost:8080',
    'user_agent': 'DataCollector/1.0 by PracticalDonkey5910'
}

TARGET_SUBREDDITS = [
    'productivity', 'getdisciplined', 'selfimprovement', 'decidingtobebetter',
    'lifehacks', 'getmotivated', 'askpsychology', 'simpleliving', 'fitness', 'zenhabits'
]

# Optimized settings for 10GB target
COLLECTION_CONFIG = {
    'min_score': 1,  # Low threshold to get more posts
    'posts_per_request': 100,
    'target_posts_per_subreddit': 100000,  # Increased target
    'concurrent_requests': 2,  # Conservative to avoid rate limits
    'rate_limit_per_minute': 50,  # Slightly under Reddit's limit
    'data_directory': Path('data/reddit_enhanced'),
    'compression_level': 9,  # Maximum compression
    'collect_comments': True,
    'max_comments_per_post': 15,  # More comments per post
    'time_periods': ['all', 'year', 'month', 'week'],
    'sort_types': ['hot', 'top', 'new', 'rising'],
    'save_batch_size': 2000  # Save every 2000 posts to manage memory
}

COLLECTION_CONFIG['data_directory'].mkdir(parents=True, exist_ok=True)

# ================================
# OAUTH CLIENT
# ================================

class RedditOAuthClient:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={'User-Agent': self.config['user_agent']},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        await self.authenticate()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def authenticate(self):
        logger.info("Authenticating with Reddit OAuth...")
        
        auth_string = f"{self.config['client_id']}:{self.config['client_secret']}"
        auth_b64 = base64.b64encode(auth_string.encode('ascii')).decode('ascii')
        
        headers = {
            'Authorization': f'Basic {auth_b64}',
            'User-Agent': self.config['user_agent']
        }
        
        data = {'grant_type': 'client_credentials'}
        
        async with self.session.post(
            'https://www.reddit.com/api/v1/access_token',
            headers=headers,
            data=data
        ) as response:
            if response.status == 200:
                token_data = await response.json()
                self.access_token = token_data['access_token']
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                logger.success("OAuth authentication successful")
            else:
                error_text = await response.text()
                logger.error(f"OAuth failed: {response.status} - {error_text}")
                raise Exception(f"OAuth authentication failed: {response.status}")
    
    async def ensure_valid_token(self):
        if not self.access_token or (self.token_expires_at and datetime.now() >= self.token_expires_at):
            await self.authenticate()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def make_request(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        await self.ensure_valid_token()
        
        headers = {
            'Authorization': f'Bearer {self.access_token}',
            'User-Agent': self.config['user_agent']
        }
        
        async with self.session.get(url, headers=headers, params=params) as response:
            if response.status == 429:
                logger.warning("Rate limit hit, waiting...")
                await asyncio.sleep(60)
                raise aiohttp.ClientError("Rate limit exceeded")
            elif response.status == 401:
                await self.authenticate()
                raise aiohttp.ClientError("Token expired")
            elif response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                logger.error(f"API request failed: {response.status} - {error_text}")
                raise aiohttp.ClientError(f"API request failed: {response.status}")

# ================================
# DATA COLLECTOR
# ================================

class EnhancedRedditDataCollector:
    def __init__(self, oauth_client: RedditOAuthClient):
        self.client = oauth_client
        self.rate_limiter = asyncio.Semaphore(COLLECTION_CONFIG['concurrent_requests'])
        self.request_times = []
        self.posts_collected = 0
        
    async def rate_limit_delay(self):
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= COLLECTION_CONFIG['rate_limit_per_minute']:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(now)
    
    async def fetch_subreddit_posts(self, subreddit: str, sort_type: str = 'hot', 
                                  limit: int = 100, after: str = None, time_period: str = None) -> Dict[str, Any]:
        await self.rate_limit_delay()
        
        url = f'https://oauth.reddit.com/r/{subreddit}/{sort_type}'
        params = {'limit': min(limit, 100), 'raw_json': 1}
        
        if after:
            params['after'] = after
        if time_period and sort_type == 'top':
            params['t'] = time_period
            
        async with self.rate_limiter:
            try:
                data = await self.client.make_request(url, params)
                return data
            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit}: {e}")
                return {}
    
    async def fetch_post_comments(self, subreddit: str, post_id: str, limit: int = 15) -> List[Dict[str, Any]]:
        await self.rate_limit_delay()
        
        url = f'https://oauth.reddit.com/r/{subreddit}/comments/{post_id}'
        params = {'limit': limit, 'sort': 'top', 'raw_json': 1}
        
        async with self.rate_limiter:
            try:
                data = await self.client.make_request(url, params)
                if len(data) >= 2:
                    comments_data = data[1].get('data', {}).get('children', [])
                    return [comment['data'] for comment in comments_data if comment.get('kind') == 't1']
                return []
            except Exception as e:
                logger.debug(f"Error fetching comments for {post_id}: {e}")
                return []
    
    def filter_post_data(self, post_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            data = post_data.get('data', {})
            score = data.get('score', 0)
            
            if score < COLLECTION_CONFIG['min_score']:
                return None
            
            return {
                'id': data.get('id'),
                'title': data.get('title'),
                'selftext': data.get('selftext'),
                'score': score,
                'upvote_ratio': data.get('upvote_ratio'),
                'num_comments': data.get('num_comments'),
                'created_utc': data.get('created_utc'),
                'author': data.get('author'),
                'subreddit': data.get('subreddit'),
                'url': data.get('url'),
                'is_self': data.get('is_self'),
                'link_flair_text': data.get('link_flair_text'),
                'domain': data.get('domain'),
                'collection_timestamp': datetime.now().isoformat(),
                'comments': []
            }
        except Exception as e:
            logger.error(f"Error filtering post: {e}")
            return None
    
    def filter_comment_data(self, comment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            score = comment_data.get('score', 0)
            if score < 1:
                return None
            
            return {
                'id': comment_data.get('id'),
                'body': comment_data.get('body'),
                'score': score,
                'created_utc': comment_data.get('created_utc'),
                'author': comment_data.get('author'),
                'parent_id': comment_data.get('parent_id'),
                'is_submitter': comment_data.get('is_submitter')
            }
        except Exception:
            return None
    
    async def collect_subreddit_data(self, subreddit: str) -> int:
        logger.info(f"Starting collection for r/{subreddit}")
        
        all_posts = []
        posts_collected = 0
        target_posts = COLLECTION_CONFIG['target_posts_per_subreddit']
        batch_count = 0
        
        for sort_type in COLLECTION_CONFIG['sort_types']:
            if posts_collected >= target_posts:
                break
                
            logger.info(f"Collecting {sort_type} posts from r/{subreddit}")
            
            time_periods = COLLECTION_CONFIG['time_periods'] if sort_type == 'top' else [None]
            
            for time_period in time_periods:
                if posts_collected >= target_posts:
                    break
                    
                after = None
                pages = 0
                max_pages = 100  # Limit pages per category
                
                while posts_collected < target_posts and pages < max_pages:
                    try:
                        data = await self.fetch_subreddit_posts(
                            subreddit, sort_type, 
                            limit=COLLECTION_CONFIG['posts_per_request'],
                            after=after, time_period=time_period
                        )
                        
                        if not data or 'data' not in data:
                            break
                        
                        posts = data['data'].get('children', [])
                        if not posts:
                            break
                        
                        # Process posts in batch
                        batch_posts = []
                        for post in posts:
                            if posts_collected >= target_posts:
                                break
                                
                            filtered_post = self.filter_post_data(post)
                            if filtered_post:
                                # Collect comments
                                if COLLECTION_CONFIG['collect_comments'] and filtered_post['num_comments'] > 0:
                                    comments = await self.fetch_post_comments(
                                        subreddit, filtered_post['id'], 
                                        COLLECTION_CONFIG['max_comments_per_post']
                                    )
                                    filtered_post['comments'] = [
                                        self.filter_comment_data(comment) 
                                        for comment in comments
                                        if self.filter_comment_data(comment)
                                    ]
                                
                                batch_posts.append(filtered_post)
                                posts_collected += 1
                        
                        all_posts.extend(batch_posts)
                        
                        # Save in batches to manage memory
                        if len(all_posts) >= COLLECTION_CONFIG['save_batch_size']:
                            batch_count += 1
                            await self.save_data(subreddit, all_posts, batch_count)
                            all_posts = []  # Clear memory
                        
                        after = data['data'].get('after')
                        if not after:
                            break
                        
                        pages += 1
                        await asyncio.sleep(1)  # Rate limiting
                        
                    except Exception as e:
                        logger.error(f"Error in collection loop: {e}")
                        break
        
        # Save remaining posts
        if all_posts:
            batch_count += 1
            await self.save_data(subreddit, all_posts, batch_count)
        
        logger.success(f"Completed r/{subreddit}: {posts_collected} posts in {batch_count} batches")
        return posts_collected
    
    async def save_data(self, subreddit: str, posts: List[Dict[str, Any]], batch_num: int):
        if not posts:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = COLLECTION_CONFIG['data_directory'] / f"{subreddit}_batch{batch_num}_{timestamp}.txt.gz"
        
        try:
            json_lines = '\n'.join(json.dumps(post, ensure_ascii=False) for post in posts)
            
            async with aiofiles.open(filename, 'wb') as f:
                compressed_data = gzip.compress(
                    json_lines.encode('utf-8'), 
                    compresslevel=COLLECTION_CONFIG['compression_level']
                )
                await f.write(compressed_data)
            
            file_size = filename.stat().st_size / (1024 * 1024)
            total_items = len(posts) + sum(len(post.get('comments', [])) for post in posts)
            
            logger.success(f"Saved batch {batch_num} for r/{subreddit}: {len(posts)} posts + comments ({file_size:.2f} MB, {total_items} items)")
            
        except Exception as e:
            logger.error(f"Error saving batch for r/{subreddit}: {e}")

# ================================
# MAIN EXECUTION
# ================================

async def collect_reddit_data():
    logger.info("Starting enhanced Reddit data collection")
    logger.info(f"Target: {len(TARGET_SUBREDDITS)} subreddits, ~{COLLECTION_CONFIG['target_posts_per_subreddit']:,} posts each")
    
    start_time = datetime.now()
    
    async with RedditOAuthClient(REDDIT_CONFIG) as oauth_client:
        collector = EnhancedRedditDataCollector(oauth_client)
        
        total_posts = 0
        successful_subreddits = 0
        
        for i, subreddit in enumerate(TARGET_SUBREDDITS, 1):
            try:
                logger.info(f"Processing subreddit {i}/{len(TARGET_SUBREDDITS)}: r/{subreddit}")
                posts = await collector.collect_subreddit_data(subreddit)
                
                if posts > 0:
                    total_posts += posts
                    successful_subreddits += 1
                    
                    # Progress update
                    elapsed = datetime.now() - start_time
                    rate = total_posts / (elapsed.total_seconds() / 3600) if elapsed.total_seconds() > 0 else 0
                    logger.info(f"Progress: {total_posts:,} total posts, {rate:.0f} posts/hour")
                
            except Exception as e:
                logger.error(f"Failed to collect r/{subreddit}: {e}")
                continue
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.success("Collection completed!")
        logger.info(f"Successful subreddits: {successful_subreddits}/{len(TARGET_SUBREDDITS)}")
        logger.info(f"Total posts: {total_posts:,}")
        logger.info(f"Duration: {duration}")
        logger.info(f"Average rate: {total_posts / (duration.total_seconds() / 3600):.0f} posts/hour")

def setup_logging():
    logger.remove()
    
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )
    
    log_file = COLLECTION_CONFIG['data_directory'] / 'collection.log'
    logger.add(log_file, format="{time} | {level} | {message}", level="DEBUG", rotation="50 MB")

def print_configuration():
    print("\n" + "="*60)
    print("🚀 ENHANCED REDDIT DATA COLLECTOR")
    print("="*60)
    print(f"Target Subreddits: {len(TARGET_SUBREDDITS)}")
    for i, sub in enumerate(TARGET_SUBREDDITS, 1):
        print(f"  {i:2d}. r/{sub}")
    
    print(f"\nCollection Settings:")
    print(f"  Minimum Score: {COLLECTION_CONFIG['min_score']} (low for maximum data)")
    print(f"  Target per Subreddit: {COLLECTION_CONFIG['target_posts_per_subreddit']:,} posts")
    print(f"  Comments per Post: {COLLECTION_CONFIG['max_comments_per_post']}")
    print(f"  Sort Types: {', '.join(COLLECTION_CONFIG['sort_types'])}")
    print(f"  Batch Size: {COLLECTION_CONFIG['save_batch_size']} posts")
    print(f"  Estimated Total: ~{COLLECTION_CONFIG['target_posts_per_subreddit'] * len(TARGET_SUBREDDITS):,} posts")
    print("="*60 + "\n")

if __name__ == "__main__":
    setup_logging()
    print_configuration()
    
    print("🎯 Target: 10GB of Reddit data with comments")
    print("⏱️  Estimated time: 6-12 hours")
    print("💾 Data will be saved in compressed batches")
    
    try:
        asyncio.run(collect_reddit_data())
        print("\n✅ Collection completed!")
        print(f"📁 Data saved to: {COLLECTION_CONFIG['data_directory']}")
        
    except KeyboardInterrupt:
        print("\n🛑 Collection interrupted")
    except Exception as e:
        print(f"\n❌ Collection failed: {e}")
        logger.exception("Fatal error") 