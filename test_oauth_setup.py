#!/usr/bin/env python3
"""
Test script to validate Reddit OAuth setup and API connectivity
Run this before the main data collection to ensure everything works
"""

import asyncio
import aiohttp
import base64
from datetime import datetime

# Reddit OAuth Credentials (same as main script)
REDDIT_CONFIG = {
    'client_id': 'veDpt5XBEiXhF8D4oR_JkA',
    'client_secret': 'NP4U3DjGdSSL3n40OsibUhPuyEclGQ',
    'redirect_uri': 'http://localhost:8080',
    'user_agent': 'DataCollector/1.0 by PracticalDonkey5910'
}

async def test_oauth_authentication():
    """Test OAuth authentication with Reddit"""
    print("🔐 Testing Reddit OAuth authentication...")
    
    # Prepare authentication data
    auth_string = f"{REDDIT_CONFIG['client_id']}:{REDDIT_CONFIG['client_secret']}"
    auth_bytes = auth_string.encode('ascii')
    auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
    
    headers = {
        'Authorization': f'Basic {auth_b64}',
        'User-Agent': REDDIT_CONFIG['user_agent']
    }
    
    data = {
        'grant_type': 'client_credentials'
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                'https://www.reddit.com/api/v1/access_token',
                headers=headers,
                data=data
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    print("✅ OAuth authentication successful!")
                    print(f"   Token type: {token_data.get('token_type', 'N/A')}")
                    print(f"   Expires in: {token_data.get('expires_in', 'N/A')} seconds")
                    print(f"   Scope: {token_data.get('scope', 'N/A')}")
                    return token_data['access_token']
                else:
                    error_text = await response.text()
                    print(f"❌ OAuth authentication failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return None
                    
        except Exception as e:
            print(f"❌ OAuth authentication error: {e}")
            return None

async def test_api_request(access_token):
    """Test a simple API request to Reddit"""
    if not access_token:
        print("⏭️  Skipping API test (no access token)")
        return False
    
    print("\n📡 Testing Reddit API request...")
    
    headers = {
        'Authorization': f'Bearer {access_token}',
        'User-Agent': REDDIT_CONFIG['user_agent']
    }
    
    # Test with a small request to r/productivity
    url = 'https://oauth.reddit.com/r/productivity/hot'
    params = {
        'limit': 5,  # Just get 5 posts for testing
        'raw_json': 1
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get('data', {}).get('children', [])
                    print("✅ API request successful!")
                    print(f"   Retrieved {len(posts)} posts from r/productivity")
                    
                    if posts:
                        first_post = posts[0]['data']
                        print(f"   Sample post: '{first_post.get('title', 'N/A')[:50]}...'")
                        print(f"   Score: {first_post.get('score', 'N/A')}")
                    
                    return True
                else:
                    error_text = await response.text()
                    print(f"❌ API request failed: {response.status}")
                    print(f"   Error: {error_text}")
                    return False
                    
        except Exception as e:
            print(f"❌ API request error: {e}")
            return False

async def test_rate_limits():
    """Test rate limiting behavior"""
    print("\n⏱️  Testing rate limit behavior...")
    print("   Making 3 quick requests to test rate limiting...")
    
    # This will use the basic approach without OAuth to test rate limits
    headers = {'User-Agent': REDDIT_CONFIG['user_agent']}
    
    async with aiohttp.ClientSession() as session:
        start_time = datetime.now()
        
        for i in range(3):
            try:
                async with session.get(
                    'https://www.reddit.com/r/productivity.json',
                    headers=headers,
                    params={'limit': 5}
                ) as response:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    print(f"   Request {i+1}: Status {response.status} ({elapsed:.2f}s)")
                    
                    if response.status == 429:
                        print("   ⚠️  Rate limit detected (this is expected)")
                    
            except Exception as e:
                print(f"   Request {i+1}: Error - {e}")

def test_dependencies():
    """Test if all required dependencies are installed"""
    print("📦 Testing dependencies...")
    
    dependencies = [
        ('aiohttp', 'aiohttp'),
        ('aiofiles', 'aiofiles'),
        ('tenacity', 'tenacity'),
        ('loguru', 'loguru')
    ]
    
    missing_deps = []
    
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"   ✅ {dep_name}")
        except ImportError:
            print(f"   ❌ {dep_name} - NOT INSTALLED")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install " + " ".join(missing_deps))
        return False
    else:
        print("   ✅ All dependencies installed!")
        return True

async def main():
    """Run all tests"""
    print("🧪 REDDIT OAUTH SETUP TEST")
    print("=" * 40)
    
    # Test 1: Dependencies
    deps_ok = test_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies before proceeding")
        return
    
    # Test 2: OAuth Authentication
    access_token = await test_oauth_authentication()
    
    # Test 3: API Request
    api_ok = await test_api_request(access_token)
    
    # Test 4: Rate Limits
    await test_rate_limits()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 TEST SUMMARY")
    print("=" * 40)
    
    if access_token and api_ok:
        print("✅ All tests passed! You're ready to run the main collector.")
        print("🚀 Run: python reddit_oauth_collector.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
        if not access_token:
            print("   - OAuth authentication failed")
            print("   - Check your client_id and client_secret")
        
        if access_token and not api_ok:
            print("   - API requests failed")
            print("   - This might be a temporary issue, try again")
    
    print("\n📚 Useful links:")
    print("   - Reddit API docs: https://www.reddit.com/dev/api/")
    print("   - OAuth guide: https://github.com/reddit-archive/reddit/wiki/OAuth2")

if __name__ == "__main__":
    asyncio.run(main()) 