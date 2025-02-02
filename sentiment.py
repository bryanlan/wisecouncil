# sentiment.py

"""
This file introduces a new 'RedditSentimentTool' that uses redditlib.py to search Reddit
and produce a sentiment/opinion analysis consolidated via GPT. The tool logic is similar
to the approach in the ResearchTool, but leverages the RedditSearchWrapper instead of Google.
"""

from typing import Dict, Any, List, Optional
import json
import datetime
import base64
import os
from pathlib import Path

from tools import Tool
from redditlib import RedditSearchWrapper
from keys import REDDIT_CLIENT_ID, REDDIT_SECRET, REDDIT_USER_AGENT
from utils import clean_response, clean_extracted_text
from config import TOKEN_LIMIT_FOR_SUMMARY
from llms import cheap_llm, HumanMessage, SystemMessage

class RedditSentimentTool(Tool):
    # Add constant at top of class
    MAX_IMAGES_PER_POST = 10  # Maximum number of images to process per post
    
    def __init__(self):
        super().__init__(
            name="do_reddit_sentiment",
            description="Perform sentiment/opinion analysis on Reddit for a given query",
            func=self._sentiment_func
        )
        # Initialize the RedditSearchWrapper client
        self.reddit_client = RedditSearchWrapper(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_SECRET,
            user_agent=REDDIT_USER_AGENT
        )
        # Add after existing initialization code:
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.sentiment_counter = 0

    def _sentiment_func(self, query: str) -> str:
        """
        Main entry point for the sentiment tool. 
        Now includes image analysis capabilities.
        """
        try:
            # Increment the sentiment counter at the start of each new analysis
            self.sentiment_counter += 1
            print(f"Starting reddit sentiment analysis with query: {query}")
            refined_data = self._refine_reddit_query(query)
            
            # Use the new search_posts_with_images method
            df = self.reddit_client.search_posts_with_images(
                query=refined_data.get("search_terms", query),
                subreddit=refined_data.get("subreddit", "all") or "all",
                time_filter=refined_data.get("time_filter", "week") or "week",
                limit=refined_data.get("limit", 20),
                sort="relevance",
                include_comments=refined_data.get("include_comments", True),
                min_score=refined_data.get("min_score", 0)
            )

            if df.empty:
                return f"No relevant Reddit posts found for query '{query}'."

            # Build content summary including image analysis
            top_posts = df.head(10)
            content_data = []
            for _, row in top_posts.iterrows():
                cleaned_title = clean_extracted_text(row["title"] or "")
                cleaned_text = clean_extracted_text(row["text"] or "")
                
                # Process comments
                comment_summaries = ""
                if "comments" in row and isinstance(row["comments"], list):
                    top_comments = row["comments"][:3]
                    comment_bullets = []
                    for c in top_comments:
                        ctext = clean_extracted_text(c.get("text", ""))
                        score = c.get("score", 0)
                        comment_bullets.append(f" - Comment (score={score}): {ctext}")
                    comment_summaries = "\n".join(comment_bullets)
                
                # Process images
                image_analysis = self._process_post_images(row.get('images', []), query)
                
                dt_str = (row["created_utc"].strftime("%Y-%m-%d %H:%M:%S") 
                         if row.get("created_utc") else "Unknown")
                         
                content_data.append(
                    f"Title: {cleaned_title}\n"
                    f"Score: {row.get('score', 0)} | Upvote Ratio: {row.get('upvote_ratio', 0.0)}\n"
                    f"Posted: {dt_str}\n"
                    f"Text: {cleaned_text}\n"
                    f"Image Analysis: {image_analysis}\n"
                    f"Comments:\n{comment_summaries}\n"                   
                    f"---"
                )
            all_content = "\n\n".join(content_data)

            # Analyze sentiment including image content
            final_summary = self._analyze_sentiment(all_content, query)
            return final_summary

        except Exception as e:
            print(f"Error in reddit sentiment tool: {e}")
            return f"Error in RedditSentimentTool: {str(e)}"

    def _refine_reddit_query(self, query: str) -> Dict[str, Any]:
        """
        Use cheap_llm to parse the user query and decide on specific:
          - subreddit
          - search_terms
          - time_filter
          - limit
          - min_score
          - include_comments
        Return them in a dict; fallback defaults if parse fails.
        """
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        prompt = (
            f"You are setting parameters for a Reddit sentiment search.\n"
            f"Today's date is {current_date}.\n"
            f"Original Query: {query}\n\n"
            f"Decide:\n"
            f"1. The most relevant subreddit (or 'all')\n"
            f"2. The precise search_terms to use in searching posts\n"
            f"3. The time_filter (one of 'hour', 'day', 'week', 'month', 'year', 'all'). If information is not extremely time-sensitive, use 'all'.\n"
            f"4. The desired limit of total posts to fetch (max 100)\n"
            f"5. A min_score threshold (0 if none)\n"
            f"6. Whether to include_comments (true or false)\n\n"
            f"Return ONLY in JSON format as:\n"
            "{\n"
            '  "subreddit": "subredditName",\n'
            '  "search_terms": "some refined search string",\n'
            '  "time_filter": "week",\n'
            '  "limit": 20,\n'
            '  "min_score": 10,\n'
            '  "include_comments": true\n'
            "}\n\n"
            f"Ensure valid JSON. If unsure, pick defaults."
        )
        response = cheap_llm.invoke([HumanMessage(content=prompt)])
        raw = clean_response(response.content)
        try:
            data = json.loads(raw)
            return data
        except:
            print("Failed to parse JSON from refining Reddit query. Using defaults.")
            return {
                "subreddit": "all",
                "search_terms": query,
                "time_filter": "week",
                "limit": 20,
                "min_score": 0,
                "include_comments": True
            }

    def _analyze_sentiment(self, all_content: str, query: str) -> str:
        """
        Pass the aggregated Reddit content to GPT to produce a consolidated sentiment analysis.
        Now handles token limits with intermediate summaries that are appended together.
        """
        # Split content into chunks based on the "---" separator
        content_chunks = all_content.split("---")
        summaries = []
        current_chunk = ""
        current_token_count = 0
        chunk_counter = 0
        
        for chunk in content_chunks:
            # Rough token estimate (1 token ≈ 4 chars)
            chunk_tokens = len(chunk) // 4
            
            if current_token_count + chunk_tokens > TOKEN_LIMIT_FOR_SUMMARY:
                # Save the current chunk before summarizing
                if current_chunk:
                    chunk_counter += 1
                    self._save_sentiment_chunk(current_chunk, chunk_counter)
                    summary = self._generate_intermediate_sentiment(current_chunk, query)
                    summaries.append(summary)
                
                # Reset accumulator
                current_chunk = chunk
                current_token_count = chunk_tokens
            else:
                current_chunk += "---" + chunk if current_chunk else chunk
                current_token_count += chunk_tokens

        # Handle remaining content
        if current_chunk:
            chunk_counter += 1
            self._save_sentiment_chunk(current_chunk, chunk_counter)
            summary = self._generate_intermediate_sentiment(current_chunk, query)
            summaries.append(summary)

        # Simply join all summaries with a separator
        return "\n\n=== Next Batch of Reddit Content ===\n\n".join(summaries)

    def _save_sentiment_chunk(self, content: str, chunk_number: int) -> None:
        """Save a chunk of sentiment analysis content to a numbered file."""
        try:
            filename = self.log_dir / f"sentiment{self.sentiment_counter}_{chunk_number}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Error saving sentiment chunk: {e}")

    def _generate_intermediate_sentiment(self, content: str, query: str) -> str:
        prompt = (
            f"Analyze this subset of Reddit content regarding: '{query}'\n\n"
            f"Content:\n{content}\n\n"
            f"Provide an intermediate summary including:\n"
            f"- Key information and themes\n"
            f"- Notable opinions or sentiment patterns\n"
            f"- Relevant quotes or examples\n"
            f"- Engagement metrics if significant\n\n"
            f"Focus on capturing the essential points that will be valuable for the final synthesis."
        )
        
        response = cheap_llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def _analyze_image(self, image_data: str, query: str) -> Optional[str]:
        """
        Analyze an image using cheap_llm's multimodal capabilities.
        Returns a description of the image content relevant to the query context.
        
        Args:
            image_data (str): Base64 encoded image data
            query (str): The search query to provide context
        """
        try:
            # Save image for debugging
            try:
                img_data = base64.b64decode(image_data)
                with open('redditpic.jpg', 'wb') as f:
                    f.write(img_data)
                print("Saved debug image as redditpic.jpg")
            except Exception as e:
                print(f"Failed to save debug image: {e}")

            prompt = (
                f"Analyze this image in the context of the following query: '{query}'\n\n"
                f"Please describe:\n"
                f"1. Any information in the image that's relevant to the query\n"
                f"2. Any visible text that relates to the topic\n"
                f"3. If it's a chart/graph, enumerate all key data points and explain the data especially as it relates to the query\n"
                f"4. If it's a meme, explain its meaning in relation to the topic\n\n"
                f"Do not leave out any key points or data, add as much detail as necessary\n"
                f"Focus only on aspects relevant to '{query}'. Ignore unrelated content. If all content is unrelated, return 'No relevant content found in image'."
            )
            
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ])
            ]
            
            response = cheap_llm.invoke(messages)
            if response.content.strip() == "No relevant content found in image.":
                return ""
            else: 
                return response.content.strip()
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None

    def _process_post_images(self, post_images: List[Dict], query: str) -> str:
        """
        Process all images in a post and return combined analysis.
        Limited to MAX_IMAGES_PER_POST images to prevent excessive processing.
        
        Args:
            post_images (List[Dict]): List of image data dictionaries
            query (str): The search query for context
        """
        if not post_images:
            return ""
            
        analyses = []
        # Add limit to number of images processed
        for img in post_images[:self.MAX_IMAGES_PER_POST]:
            if 'base64_data' in img:
                analysis = self._analyze_image(img['base64_data'], query)
                if analysis:
                    analyses.append(analysis)
                    
        if analyses:
            return "\nImage Analysis:\n" + "\n".join(f"- {a}" for a in analyses)
        return ""

if __name__ == "__main__":
    # Test the RedditSentimentTool with a sample query
    tool = RedditSentimentTool()

    # Sample test queries
    test_queries = [
        "Intercontinental Exchange (ICE) as a potential investment.",

    ]

    for query in test_queries:
        print(f"\nTesting query: '{query}'")
        result = tool._sentiment_func(query)
        print(f"Result:\n{result}\n")