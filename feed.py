"""RSS feed fetching and parsing."""

import feedparser
import requests


def fetch_feed(rss_url):
    """Parse RSS feed and return podcast metadata with episodes."""
    response = requests.get(rss_url, timeout=30)
    response.raise_for_status()
    feed = feedparser.parse(response.content)

    if feed.bozo and not feed.entries:
        raise ValueError(f"Could not parse feed at {rss_url}")

    episodes = []
    for entry in feed.entries:
        audio_url = None
        for enclosure in entry.get("enclosures", []):
            if enclosure.get("type", "").startswith("audio/"):
                audio_url = enclosure.get("href")
                break
        if not audio_url:
            for media in entry.get("media_content", []):
                if media.get("type", "").startswith("audio/"):
                    audio_url = media.get("url")
                    break
        if not audio_url:
            for link in entry.get("links", []):
                href = link.get("href", "")
                if any(href.lower().endswith(ext) for ext in [".mp3", ".m4a", ".wav", ".ogg", ".aac"]):
                    audio_url = href
                    break
        if audio_url:
            episodes.append({
                "title": entry.get("title", "Untitled"),
                "published": entry.get("published", ""),
                "summary": entry.get("summary", "")[:200] if entry.get("summary") else "",
                "audio_url": audio_url,
                "duration": entry.get("itunes_duration", ""),
            })

    description = feed.feed.get("description") or feed.feed.get("subtitle", "")
    image_url = ""
    if feed.feed.get("image", {}).get("href"):
        image_url = feed.feed["image"]["href"]
    elif feed.feed.get("itunes_image", {}).get("href"):
        image_url = feed.feed["itunes_image"]["href"]

    return {
        "title": feed.feed.get("title", "Unknown Podcast"),
        "author": feed.feed.get("author", feed.feed.get("itunes_author", "")),
        "description": description,
        "image_url": image_url,
        "episodes": episodes,
    }
