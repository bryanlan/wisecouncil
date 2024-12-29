import praw
from datetime import datetime, timedelta
from typing import List, Dict, Generator, Optional, Tuple
import pandas as pd
import re
import requests
from PIL import Image
from io import BytesIO
import base64

class RedditSearchWrapper:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        """
        Initialize the Reddit API wrapper.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent string for API requests
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def search_posts(self, 
                    query: str,
                    subreddit: str = "all",
                    time_filter: str = "week",
                    limit: int = 100,
                    sort: str = "relevance",
                    include_comments: bool = True,
                    min_score: int = 0) -> pd.DataFrame:
        """
        Search Reddit posts based on query and filters.
        
        Args:
            query (str): Search query
            subreddit (str): Subreddit to search in (default: "all")
            time_filter (str): One of (hour, day, week, month, year, all)
            limit (int): Maximum number of posts to retrieve
            sort (str): One of (relevance, hot, top, new, comments)
            include_comments (bool): Whether to include top-level comments
            min_score (int): Minimum score (upvotes) threshold
            
        Returns:
            pd.DataFrame: DataFrame containing post data
        """
        subreddit = self.reddit.subreddit(subreddit)
        posts_data = []

        try:
            # Search for posts
            for post in subreddit.search(query, 
                                       sort=sort, 
                                       time_filter=time_filter, 
                                       limit=limit):
                
                if post.score < min_score:
                    continue

                post_data = {
                    'post_id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'url': post.url,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'num_comments': post.num_comments,
                    'subreddit': post.subreddit.display_name,
                    'author': str(post.author),
                    'upvote_ratio': post.upvote_ratio
                }

                # Include top-level comments if requested
                if include_comments:
                    comments = []
                    post.comments.replace_more(limit=0)
                    for comment in post.comments.list():
                        if comment.parent_id.startswith('t3_'):
                            comments.append({
                                'comment_id': comment.id,
                                'text': comment.body,
                                'score': comment.score,
                                'author': str(comment.author),
                                'created_utc': datetime.fromtimestamp(comment.created_utc)
                            })
                    post_data['comments'] = comments

                posts_data.append(post_data)

        except Exception as e:
            print(f"Error searching posts: {str(e)}")
            return pd.DataFrame()

        return pd.DataFrame(posts_data)

    def search_multiple_terms(self, 
                            terms: List[str], 
                            **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Search multiple terms and return results for each term.
        
        Args:
            terms (List[str]): List of search terms
            **kwargs: Additional arguments to pass to search_posts
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping terms to their search results
        """
        results = {}
        for term in terms:
            results[term] = self.search_posts(term, **kwargs)
        return results

    def get_subreddit_sentiment(self,
                              subreddit: str,
                              time_filter: str = "week",
                              limit: int = 100) -> pd.DataFrame:
        """
        Get recent posts from a subreddit for sentiment analysis.
        
        Args:
            subreddit (str): Subreddit name
            time_filter (str): Time filter
            limit (int): Maximum number of posts
            
        Returns:
            pd.DataFrame: DataFrame containing subreddit posts
        """
        subreddit = self.reddit.subreddit(subreddit)
        posts_data = []

        try:
            # Get posts based on time filter
            if time_filter == "hour":
                posts = subreddit.new(limit=limit)
            else:
                posts = subreddit.top(time_filter=time_filter, limit=limit)

            for post in posts:
                post_data = {
                    'post_id': post.id,
                    'title': post.title,
                    'text': post.selftext,
                    'score': post.score,
                    'created_utc': datetime.fromtimestamp(post.created_utc),
                    'num_comments': post.num_comments,
                    'upvote_ratio': post.upvote_ratio
                }
                posts_data.append(post_data)

        except Exception as e:
            print(f"Error getting subreddit sentiment: {str(e)}")
            return pd.DataFrame()

        return pd.DataFrame(posts_data)

    def get_subreddits(self,
                       search_query: str = None,
                       category: str = None,
                       limit: int = 25,
                       nsfw: bool = False) -> pd.DataFrame:
        """
        Get list of subreddits based on search query or category.
        
        Args:
            search_query (str): Search term for subreddits
            category (str): Category to list ('popular', 'new', 'gold', 'default')
            limit (int): Maximum number of subreddits to return
            nsfw (bool): Whether to include NSFW subreddits
            
        Returns:
            pd.DataFrame: DataFrame containing subreddit information
        """
        subreddits_data = []

        try:
            if search_query:
                # Search for subreddits matching the query
                for subreddit in self.reddit.subreddits.search(search_query, limit=limit):
                    if not nsfw and subreddit.over18:
                        continue
                    subreddits_data.append(self._extract_subreddit_info(subreddit))
            
            elif category:
                # Get subreddits by category
                if category == 'popular':
                    subreddits = self.reddit.subreddits.popular(limit=limit)
                elif category == 'new':
                    subreddits = self.reddit.subreddits.new(limit=limit)
                elif category == 'gold':
                    subreddits = self.reddit.subreddits.gold(limit=limit)
                elif category == 'default':
                    subreddits = self.reddit.subreddits.default(limit=limit)
                else:
                    raise ValueError("Invalid category. Use 'popular', 'new', 'gold', or 'default'")

                for subreddit in subreddits:
                    if not nsfw and subreddit.over18:
                        continue
                    subreddits_data.append(self._extract_subreddit_info(subreddit))

        except Exception as e:
            print(f"Error getting subreddits: {str(e)}")
            return pd.DataFrame()

        return pd.DataFrame(subreddits_data)

    def _extract_subreddit_info(self, subreddit) -> dict:
        """Helper method to extract relevant subreddit information."""
        return {
            'name': subreddit.display_name,
            'title': subreddit.title,
            'description': subreddit.public_description,
            'subscribers': subreddit.subscribers,
            'created_utc': datetime.fromtimestamp(subreddit.created_utc),
            'nsfw': subreddit.over18,
            'url': f"https://reddit.com{subreddit.url}"
        }

    def get_trending_subreddits(self) -> List[str]:
        """
        Get currently trending subreddits.
        
        Returns:
            List[str]: List of trending subreddit names
        """
        try:
            return self.reddit.trending_subreddits()
        except Exception as e:
            print(f"Error getting trending subreddits: {str(e)}")
            return []

    def get_related_subreddits(self, subreddit_name: str, limit: int = 25) -> List[str]:
        """
        Get related subreddits by analyzing sidebar and wiki.
        
        Args:
            subreddit_name (str): Name of the subreddit
            limit (int): Maximum number of related subreddits to return
            
        Returns:
            List[str]: List of related subreddit names
        """
        try:
            subreddit = self.reddit.subreddit(subreddit_name)
            sidebar = subreddit.description
            related = set()
            
            pattern = r'(?:^|\s)/?r/([a-zA-Z0-9][a-zA-Z0-9_]{2,20})'
            
            if sidebar:
                matches = re.findall(pattern, sidebar)
                related.update(matches)
            
            return list(related)[:limit]
        except Exception as e:
            print(f"Error getting related subreddits: {str(e)}")
            return []

    def get_subreddit_categories(self) -> List[str]:
        """
        Get a list of common subreddit categories/topics.
        
        Returns:
            List[str]: List of common subreddit categories
        """
        return [
            'Art',
            'Books',
            'Business',
            'Cryptocurrency',
            'Education',
            'Entertainment',
            'Gaming',
            'Health',
            'Hobbies',
            'Humor',
            'Life Advice',
            'Music',
            'News',
            'Politics',
            'Programming',
            'Science',
            'Sports',
            'Technology',
            'Television',
            'World News'
        ]

    def _extract_image_urls(self, post) -> List[str]:
        """Extract image URLs from a Reddit post."""
        image_urls = []
        
        try:
            # Handle direct image posts
            if hasattr(post, 'url'):
                url = post.url
                # Check for direct image URLs
                if any(url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    image_urls.append(url)
                # Handle i.redd.it URLs
                elif 'i.redd.it' in url:
                    image_urls.append(url)
            
            # Handle gallery posts
            if hasattr(post, 'is_gallery') and post.is_gallery:
                if hasattr(post, 'media_metadata'):
                    for item in post.media_metadata.values():
                        if item['e'] == 'Image':
                            if 's' in item and 'u' in item['s']:
                                url = item['s']['u'].replace('&amp;', '&')
                                image_urls.append(url)
            
            # Handle preview images
            if hasattr(post, 'preview') and 'images' in post.preview:
                for image in post.preview['images']:
                    if 'source' in image:
                        url = image['source']['url'].replace('&amp;', '&')
                        image_urls.append(url)
                        
        except Exception as e:
            print(f"Error extracting image URLs: {e}")
            
        return list(set(image_urls))  # Remove duplicates

    def _download_and_encode_image(self, url: str) -> Optional[str]:
        """Download image from URL and encode as base64."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Open and verify the image
            img = Image.open(BytesIO(response.content))
            img.verify()
            
            # Convert to base64
            buffered = BytesIO()
            img = Image.open(BytesIO(response.content))  # Need to reopen after verify
            img.save(buffered, format=img.format)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        except Exception as e:
            print(f"Error processing image from {url}: {e}")
            return None

    def search_posts_with_images(self, 
                            query: str,
                            **kwargs) -> pd.DataFrame:
        """
        Enhanced version of search_posts that includes image data.
        Additional kwargs are passed to search_posts.
        """
        # Get basic post data
        df = self.search_posts(query, **kwargs)
        
        if df.empty:
            return df
            
        # Add image data
        image_data = []
        for _, row in df.iterrows():
            post_images = []
            if 'post_id' in row:
                try:
                    post = self.reddit.submission(id=row['post_id'])
                    image_urls = self._extract_image_urls(post)
                    
                    for url in image_urls:
                        encoded_image = self._download_and_encode_image(url)
                        if encoded_image:
                            post_images.append({
                                'url': url,
                                'base64_data': encoded_image
                            })
                except Exception as e:
                    print(f"Error processing post {row['post_id']}: {e}")
                    
            image_data.append(post_images)
            
        df['images'] = image_data
        return df

    def get_post_images(self, post_id: str) -> List[Dict[str, str]]:
        """
        Get images from a specific post.
        
        Args:
            post_id (str): Reddit post ID
            
        Returns:
            List[Dict[str, str]]: List of dictionaries containing image URLs and base64 data
        """
        try:
            post = self.reddit.submission(id=post_id)
            image_urls = self._extract_image_urls(post)
            
            images = []
            for url in image_urls:
                encoded_image = self._download_and_encode_image(url)
                if encoded_image:
                    images.append({
                        'url': url,
                        'base64_data': encoded_image
                    })
            return images
        except Exception as e:
            print(f"Error getting images for post {post_id}: {e}")
            return []