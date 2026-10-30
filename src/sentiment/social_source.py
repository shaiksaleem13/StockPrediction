"""Social Media & Financial Forum Data Fetcher for EAATS.

Fetches real-time sentiment from Reddit (PRAW) and StockTwits public stream.
Graceful degradation ensures zero application crashes when unauthenticated or offline.
"""

import os
from typing import Dict, List
import requests

import config
from src.utils import logger


def fetch_stocktwits_stream(ticker: str, max_items: int = config.MAX_STOCKTWITS_POSTS) -> List[Dict[str, str]]:
    """Fetches public sentiment stream from StockTwits for a given ticker."""
    messages = []
    # Strip market exchange suffixes like .NS
    clean_symbol = ticker.split(".")[0].upper()
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{clean_symbol}.json"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=8)
        if response.status_code == 200:
            data = response.json()
            for msg in data.get("messages", [])[:max_items]:
                body = msg.get("body", "")
                created_at = msg.get("created_at", "")
                msg_id = msg.get("id", "")
                link = f"https://stocktwits.com/message/{msg_id}" if msg_id else "https://stocktwits.com"
                if body:
                    messages.append(
                        {
                            "text": body.strip(),
                            "source": "StockTwits",
                            "timestamp": created_at,
                            "url": link,
                        }
                    )
            logger.info(f"Retrieved {len(messages)} messages from StockTwits for {clean_symbol}.")
        else:
            logger.warning(f"StockTwits API returned status {response.status_code} for {clean_symbol}.")
    except Exception as e:
        logger.warning(f"Failed to fetch StockTwits stream for {ticker}: {e}")

    return messages


def fetch_reddit_posts(ticker: str, max_items: int = config.MAX_REDDIT_POSTS) -> List[Dict[str, str]]:
    """Fetches recent discussions mentioning the ticker from r/stocks, r/wallstreetbets, and r/investing."""
    posts = []
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "eaats_project:v1.0")

    if not client_id or not client_secret:
        logger.debug("Reddit API credentials not provided in .env; skipping live Reddit scrape.")
        return posts

    try:
        import praw

        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )

        subreddits = ["stocks", "wallstreetbets", "investing"]
        query = ticker.split(".")[0]
        per_sub = max(1, max_items // len(subreddits))

        for sub_name in subreddits:
            sub = reddit.subreddit(sub_name)
            for submission in sub.search(query, sort="new", limit=per_sub):
                title = submission.title
                selftext = submission.selftext[:300] if submission.selftext else ""
                combined_text = f"{title} {selftext}".strip()
                if combined_text:
                    posts.append(
                        {
                            "text": combined_text,
                            "source": f"Reddit (r/{sub_name})",
                            "timestamp": str(submission.created_utc),
                            "url": f"https://reddit.com{submission.permalink}",
                        }
                    )
        logger.info(f"Retrieved {len(posts)} Reddit posts for {ticker}.")
    except Exception as e:
        logger.warning(f"Failed to fetch Reddit posts for {ticker}: {e}")

    return posts


def fetch_social_sentiment_data(ticker: str) -> List[Dict[str, str]]:
    """Aggregates all social media and forum feeds."""
    social_items = []
    social_items.extend(fetch_stocktwits_stream(ticker))
    social_items.extend(fetch_reddit_posts(ticker))
    return social_items
