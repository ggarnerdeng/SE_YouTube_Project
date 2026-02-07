"""
Component 1: Trending Topic Fetcher (past ~24h)

Goal:
- Return a ranked list of trending topics with basic metadata.
- Modular provider interface so we can add more sources later.

Default provider:
- Google Trends Daily Trending Searches RSS (public RSS feed)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Protocol, Dict, Any
import re
import requests
import xml.etree.ElementTree as ET


# -----------------------------
# Data model
# -----------------------------

@dataclass(frozen=True)
class TrendingTopic:
    topic: str
    score: float               # relative confidence / rank score
    source: str                # provider name
    url: Optional[str] = None  # link to related trend page/article
    published_at: Optional[datetime] = None
    extra: Optional[Dict[str, Any]] = None


# -----------------------------
# Provider interface
# -----------------------------

class TrendingProvider(Protocol):
    name: str

    def fetch(self, since: datetime) -> List[TrendingTopic]:
        """Fetch trending topics published since the given timestamp."""
        ...


# -----------------------------
# Utility helpers
# -----------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_rfc822(dt_str: str) -> Optional[datetime]:
    """
    RSS typically uses RFC822-like dates.
    We'll do a lightweight parse that covers common cases.
    """
    # Example: "Fri, 07 Feb 2026 20:00:00 +0000"
    # We avoid external deps to keep it simple.
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(dt_str).astimezone(timezone.utc)
    except Exception:
        return None


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Provider: Google Trends RSS
# -----------------------------

class GoogleTrendsTrendingNowRSS:
    """
    Google Trends 'Trending now' RSS feed.
    Example:
      https://trends.google.com/trending/rss?geo=US
    """
    name = "google_trends_trending_now_rss"

    def __init__(
        self,
        geo: str = "US",
        timeout_s: int = 15,
        user_agent: str = "TrendFetcher/1.0",
    ) -> None:
        self.geo = geo
        self.timeout_s = timeout_s
        self.user_agent = user_agent

    def fetch(self, since: datetime) -> List[TrendingTopic]:
        url = f"https://trends.google.com/trending/rss?geo={self.geo}"
        headers = {"User-Agent": self.user_agent}

        resp = requests.get(url, headers=headers, timeout=self.timeout_s)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        if channel is None:
            return []

        topics: List[TrendingTopic] = []

        for item in channel.findall("item"):
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")

            if title_el is None or not title_el.text:
                continue

            title = _clean_text(title_el.text)
            link = link_el.text.strip() if (link_el is not None and link_el.text) else None
            published_at = _parse_rfc822(pub_el.text) if (pub_el is not None and pub_el.text) else None

            # best-effort filter
            if published_at is not None and published_at < since:
                continue

            topics.append(
                TrendingTopic(
                    topic=title,
                    score=1.0,  # we’ll score by order next
                    source=self.name,
                    url=link,
                    published_at=published_at,
                    extra={"geo": self.geo},
                )
            )

        # Score by rank in feed (top item highest)
        n = len(topics)
        topics = [
            TrendingTopic(
                topic=t.topic,
                score=(n - i) / max(n, 1),
                source=t.source,
                url=t.url,
                published_at=t.published_at,
                extra=t.extra,
            )
            for i, t in enumerate(topics)
        ]

        return topics



# -----------------------------
# Aggregator (supports multiple providers later)
# -----------------------------

class TrendAggregator:
    def __init__(self, providers: List[TrendingProvider]) -> None:
        self.providers = providers

    def get_trending_topics(
        self,
        hours: int = 24,
        limit: int = 10,
    ) -> List[TrendingTopic]:
        since = _now_utc() - timedelta(hours=hours)

        all_topics: List[TrendingTopic] = []
        for p in self.providers:
            try:
                all_topics.extend(p.fetch(since=since))
            except Exception as e:
                # Keep pipeline resilient: skip provider errors
                all_topics.append(
                    TrendingTopic(
                        topic=f"[PROVIDER_ERROR] {p.name}: {e}",
                        score=0.0,
                        source=p.name,
                        extra={"error": str(e)},
                    )
                )

        # Deduplicate by normalized title
        seen = set()
        unique: List[TrendingTopic] = []
        for t in sorted(all_topics, key=lambda x: x.score, reverse=True):
            key = t.topic.lower()
            if key in seen:
                continue
            seen.add(key)
            unique.append(t)

        return unique[:limit]


# -----------------------------
# Quick CLI usage
# -----------------------------

if __name__ == "__main__":
    agg = TrendAggregator(providers=[GoogleTrendsTrendingNowRSS(geo="US")])
    topics = agg.get_trending_topics(hours=24, limit=10)
    for t in topics:
        print(f"{t.score:.2f} | {t.topic} | {t.url or ''}")
