"""Video Source Fetcher for EAATS using YouTube Transcripts.

Discovers relevant financial video commentary and extracts English transcripts,
chunking them into ~500-character segments for sentiment evaluation.
Graceful degradation ensures zero crashes when transcripts are unavailable.
"""

from typing import Dict, List
import urllib.parse
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi

import config
from src.utils import logger


def search_youtube_video_ids(query: str, max_results: int = config.MAX_YOUTUBE_VIDEOS) -> List[str]:
    """Scrapes YouTube search results for video IDs matching the financial query."""
    video_ids = []
    encoded = urllib.parse.quote(query)
    search_url = f"https://www.youtube.com/results?search_query={encoded}"

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(search_url, headers=headers, timeout=8)
        if resp.status_code == 200:
            import re

            # Extract 11-character video IDs from response HTML
            matches = re.findall(r"\"videoId\":\"([a-zA-Z0-9_-]{11})\"", resp.text)
            seen = set()
            for vid in matches:
                if vid not in seen:
                    seen.add(vid)
                    video_ids.append(vid)
                if len(video_ids) >= max_results:
                    break
    except Exception as e:
        logger.warning(f"Could not scrape YouTube search IDs for '{query}': {e}")

    return video_ids


def fetch_youtube_transcripts(ticker: str, max_videos: int = config.MAX_YOUTUBE_VIDEOS) -> List[Dict[str, str]]:
    """Fetches and chunks transcripts from finance videos discussing the ticker."""
    transcript_chunks = []
    query = f"{ticker.split('.')[0]} stock analysis"
    video_ids = search_youtube_video_ids(query, max_videos)

    if not video_ids:
        logger.debug(f"No YouTube videos found for query: {query}")
        return transcript_chunks

    for vid in video_ids:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "en-US"])
            full_text = " ".join([entry["text"] for entry in transcript_list])

            # Chunk into ~500 character segments
            chunk_size = config.YOUTUBE_CHUNK_CHARS
            chunks = [full_text[i : i + chunk_size] for i in range(0, min(len(full_text), 3000), chunk_size)]

            for chunk in chunks:
                if len(chunk.strip()) > 50:
                    transcript_chunks.append(
                        {
                            "text": chunk.strip(),
                            "source": "YouTube Transcript",
                            "timestamp": "Recent",
                            "url": f"https://www.youtube.com/watch?v={vid}",
                        }
                    )
            logger.info(f"Retrieved {len(chunks)} transcript chunks for video {vid}.")
        except Exception as e:
            logger.debug(f"Could not retrieve transcript for video {vid}: {e}")

    return transcript_chunks
