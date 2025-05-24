#!/usr/bin/env python3
"""
Reddit Data Collector
Professional data collection from Reddit via Pushshift API
"""

import requests
import json
import argparse
import os
import time
import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


class RedditDataCollector:
    """Professional Reddit data collection using Pushshift API"""
    
    def __init__(self, api_token: Optional[str] = None):
        self.api_token = api_token
        self.base_url = "https://api.pushshift.io/reddit"
        self.session = self._setup_session()
        self.logger = self._setup_logging()
        
    def _setup_session(self) -> requests.Session:
        """Configure HTTP session with proper headers"""
        session = requests.Session()
        headers = {"accept": "application/json"}
        
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        session.headers.update(headers)
        return session
    
    def _setup_logging(self) -> logging.Logger:
        """Configure logging for data collection operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def collect_submissions(self, subreddit: str, limit: int, days_back: int) -> List[Dict[str, Any]]:
        """Collect submissions from a specific subreddit"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        params = {
            "subreddit": subreddit,
            "sort": "score",        # Sort by upvotes (top posts)
            "order": "desc",        # Highest scores first
            "limit": min(limit, 100),
            "after": int(start_time.timestamp()),
            "before": int(end_time.timestamp()),
            "track_total_hits": "false"
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/submission/search",
                params=params,
                timeout=30
            )
            
            if response.status_code == 403:
                self.logger.warning(f"Access denied for r/{subreddit} submissions")
                return []
            elif response.status_code == 429:
                self.logger.warning("Rate limited, waiting...")
                time.sleep(5)
                return self.collect_submissions(subreddit, limit, days_back)
                
            response.raise_for_status()
            data = response.json().get("data", [])
            
            submissions = []
            for item in data:
                submissions.append({
                    'type': 'submission',
                    'subreddit': subreddit,
                    'title': item.get('title', ''),
                    'text': item.get('selftext', ''),
                    'score': item.get('score', 0),
                    'created_utc': item.get('created_utc', 0),
                    'num_comments': item.get('num_comments', 0),
                    'id': item.get('id', '')
                })
            
            self.logger.info(f"Collected {len(submissions)} submissions from r/{subreddit}")
            return submissions
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to collect submissions from r/{subreddit}: {e}")
            return []
    
    def collect_comments(self, subreddit: str, limit: int, days_back: int) -> List[Dict[str, Any]]:
        """Collect comments from a specific subreddit"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        params = {
            "subreddit": subreddit,
            "sort": "score",        # Sort by upvotes (top comments)
            "order": "desc",        # Highest scores first
            "limit": min(limit, 100),
            "after": int(start_time.timestamp()),
            "before": int(end_time.timestamp()),
            "track_total_hits": "false"
        }
        
        try:
            response = self.session.get(
                f"{self.base_url}/comment/search",
                params=params,
                timeout=30
            )
            
            if response.status_code == 403:
                self.logger.warning(f"Access denied for r/{subreddit} comments")
                return []
            elif response.status_code == 429:
                self.logger.warning("Rate limited, waiting...")
                time.sleep(5)
                return self.collect_comments(subreddit, limit, days_back)
                
            response.raise_for_status()
            data = response.json().get("data", [])
            
            comments = []
            for item in data:
                body = item.get('body', '')
                if body not in ['[deleted]', '[removed]', '']:
                    comments.append({
                        'type': 'comment',
                        'subreddit': subreddit,
                        'text': body,
                        'score': item.get('score', 0),
                        'created_utc': item.get('created_utc', 0),
                        'id': item.get('id', '')
                    })
            
            self.logger.info(f"Collected {len(comments)} comments from r/{subreddit}")
            return comments
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to collect comments from r/{subreddit}: {e}")
            return []
    
    def collect_all_data(self, subreddits: List[str], limit: int, 
                        days_back: int, include_comments: bool = True) -> List[Dict[str, Any]]:
        """Collect all data from specified subreddits"""
        all_data = []
        
        for subreddit in subreddits:
            self.logger.info(f"Processing r/{subreddit}")
            
            # Collect submissions
            submissions = self.collect_submissions(subreddit, limit, days_back)
            all_data.extend(submissions)
            time.sleep(1)  # Rate limiting
            
            # Collect comments if requested
            if include_comments:
                comments = self.collect_comments(subreddit, limit, days_back)
                all_data.extend(comments)
                time.sleep(1)  # Rate limiting
        
        return all_data
    
    def collect_balanced_data(self, subreddits: List[str], limit: int, 
                            days_back: int, randomize: bool = True) -> List[Dict[str, Any]]:
        """Collect balanced submissions and comments with optional randomization"""
        all_data = []
        
        for subreddit in subreddits:
            self.logger.info(f"Processing r/{subreddit} with balanced collection")
            
            # Split limit between submissions and comments for balance
            submission_limit = limit // 2
            comment_limit = limit - submission_limit
            
            if randomize:
                # Create multiple random time windows within the period
                self.logger.info("Using randomized time sampling")
                
                # Collect submissions from random time periods
                submissions = self._collect_random_time_samples(
                    subreddit, submission_limit, days_back, 'submission'
                )
                all_data.extend(submissions)
                time.sleep(1)
                
                # Collect comments from random time periods  
                comments = self._collect_random_time_samples(
                    subreddit, comment_limit, days_back, 'comment'
                )
                all_data.extend(comments)
                time.sleep(1)
                
            else:
                # Standard collection
                submissions = self.collect_submissions(subreddit, submission_limit, days_back)
                all_data.extend(submissions)
                time.sleep(1)
                
                comments = self.collect_comments(subreddit, comment_limit, days_back) 
                all_data.extend(comments)
                time.sleep(1)
        
        return all_data
    
    def _collect_random_time_samples(self, subreddit: str, limit: int, 
                                   days_back: int, content_type: str) -> List[Dict[str, Any]]:
        """Collect data from multiple random time windows"""
        all_items = []
        items_per_window = max(20, limit // 5)  # Collect from 5 different time periods
        
        end_time = datetime.now()
        total_seconds = days_back * 24 * 60 * 60
        
        for _ in range(5):  # Sample from 5 random time periods
            if len(all_items) >= limit:
                break
                
            # Create random time window (7 days each)
            window_start_offset = random.randint(7 * 24 * 60 * 60, total_seconds)
            window_end_offset = max(0, window_start_offset - (7 * 24 * 60 * 60))
            
            window_start = end_time - timedelta(seconds=window_start_offset)
            window_end = end_time - timedelta(seconds=window_end_offset)
            
            # Collect from this time window
            if content_type == 'submission':
                items = self._collect_submissions_in_window(
                    subreddit, items_per_window, window_start, window_end
                )
            else:
                items = self._collect_comments_in_window(
                    subreddit, items_per_window, window_start, window_end
                )
            
            all_items.extend(items)
            time.sleep(0.5)  # Short delay between windows
        
        # Randomly shuffle and limit to requested amount
        random.shuffle(all_items)
        return all_items[:limit]
    
    def _collect_submissions_in_window(self, subreddit: str, limit: int, 
                                     start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect submissions from a specific time window"""
        params = {
            "subreddit": subreddit,
            "sort": "score",
            "order": "desc", 
            "limit": min(limit, 100),
            "after": int(start_time.timestamp()),
            "before": int(end_time.timestamp()),
            "track_total_hits": "false"
        }
        
        try:
            response = self.session.get(f"{self.base_url}/submission/search", params=params, timeout=30)
            
            if response.status_code in [403, 429]:
                return []
                
            response.raise_for_status()
            data = response.json().get("data", [])
            
            submissions = []
            for item in data:
                submissions.append({
                    'type': 'submission',
                    'subreddit': subreddit,
                    'title': item.get('title', ''),
                    'text': item.get('selftext', ''),
                    'score': item.get('score', 0),
                    'created_utc': item.get('created_utc', 0),
                    'num_comments': item.get('num_comments', 0),
                    'id': item.get('id', '')
                })
            
            return submissions
            
        except requests.RequestException:
            return []
    
    def _collect_comments_in_window(self, subreddit: str, limit: int,
                                  start_time: datetime, end_time: datetime) -> List[Dict[str, Any]]:
        """Collect comments from a specific time window"""
        params = {
            "subreddit": subreddit,
            "sort": "score",
            "order": "desc",
            "limit": min(limit, 100),
            "after": int(start_time.timestamp()),
            "before": int(end_time.timestamp()),
            "track_total_hits": "false"
        }
        
        try:
            response = self.session.get(f"{self.base_url}/comment/search", params=params, timeout=30)
            
            if response.status_code in [403, 429]:
                return []
                
            response.raise_for_status()
            data = response.json().get("data", [])
            
            comments = []
            for item in data:
                body = item.get('body', '')
                if body not in ['[deleted]', '[removed]', '']:
                    comments.append({
                        'type': 'comment',
                        'subreddit': subreddit,
                        'text': body,
                        'score': item.get('score', 0),
                        'created_utc': item.get('created_utc', 0),
                        'id': item.get('id', '')
                    })
            
            return comments
            
        except requests.RequestException:
            return []
    
    def save_data(self, data: List[Dict[str, Any]], output_file: str):
        """Save collected data to JSON lines format"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        self.logger.info(f"Saved {len(data)} items to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Collect Reddit data via Pushshift API')
    parser.add_argument('--api-token', help='Pushshift API Bearer token')
    parser.add_argument('--subreddits', nargs='+', 
                       default=['Productivity'],
                       help='Subreddits to collect from')
    parser.add_argument('--limit', type=int, default=200, 
                       help='Items per subreddit')
    parser.add_argument('--days-back', type=int, default=1095,
                       help='Days back to search (default: 3 years)')
    parser.add_argument('--no-comments', action='store_true',
                       help='Skip comments collection')
    parser.add_argument('--no-random', action='store_true',
                       help='Disable random time sampling')
    parser.add_argument('--output', default='data/raw.txt', 
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = RedditDataCollector(api_token=args.api_token)
    
    # Collect data with balanced approach
    data = collector.collect_balanced_data(
        subreddits=args.subreddits,
        limit=args.limit,
        days_back=args.days_back,
        randomize=not args.no_random
    )
    
    # Save results
    collector.save_data(data, args.output)
    
    # Summary
    submissions = sum(1 for item in data if item['type'] == 'submission')
    comments = sum(1 for item in data if item['type'] == 'comment')
    
    print(f"\nCollection Summary:")
    print(f"Submissions: {submissions}")
    print(f"Comments: {comments}")
    print(f"Total: {len(data)}")


if __name__ == "__main__":
    main() 