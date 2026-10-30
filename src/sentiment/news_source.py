"""News Source Fetcher for EAATS via Google News RSS."""

from typing import Dict, List
import urllib.parse
import feedparser
from bs4 import BeautifulSoup
import requests

import config
from src.utils import logger


def fetch_google_news(ticker: str, max_items: int = config.MAX_NEWS_HEADLINES) -> List[Dict[str, str]]:
    """Scrapes recent stock news headlines from Google News RSS.

    Returns:
        List of dicts: {"text": str, "source": str, "timestamp": str, "url": str}
    """
    articles = []
    clean_ticker = ticker.replace("^", "").strip()
    encoded_query = urllib.parse.quote(f"{clean_ticker} stock")
    url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-US&gl=US&ceid=US:en"

    try:
        logger.info(f"Fetching Google News RSS for {ticker}...")
        feed = feedparser.parse(url)

        if not feed.entries:
            logger.warning(f"No RSS news entries found for {ticker}.")
            return articles

        for entry in feed.entries[:max_items]:
            title = entry.get("title", "")
            # Clean HTML tags if any in title
            if "<" in title:
                title = BeautifulSoup(title, "html.parser").get_text()

            # Separate publisher if formatted as 'Headline - Publisher'
            link = entry.get("link", "")
            pubdate = entry.get("published", "")

            if title:
                articles.append(
                    {
                        "text": title.strip(),
                        "source": "Google News",
                        "timestamp": pubdate,
                        "url": link,
                    }
                )

        logger.info(f"Retrieved {len(articles)} headlines from Google News RSS for {ticker}.")
    except Exception as e:
        logger.warning(f"Failed to fetch Google News RSS for {ticker}: {e}")

    return articles
