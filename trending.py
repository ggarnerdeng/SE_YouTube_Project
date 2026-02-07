"""
trending.py

Component 1 (Reconfigured):
- Show *why* each topic is trending (traffic + top related news headline/source/url) from Google Trends RSS.
- Focus on US political / breaking news by:
  (A) Filtering Google Trends topics using a politics keyword list
  (B) Backfilling from Google News RSS Search (query-driven, US-scoped)
- Output: a ranked list with explanations.

Dependencies:
  pip install requests
Run:
  python trending.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Protocol
import re
import requests
import xml.etree.ElementTree as ET


# -----------------------------
# Models
# -----------------------------

@dataclass(frozen=True)
class TrendingTopic:
    topic: str
    score: float
    source: str
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    extra: Optional[Dict[str, Any]] = None


class TrendingProvider(Protocol):
    name: str

    def fetch(self, since: datetime) -> List[TrendingTopic]:
        ...


# -----------------------------
# Helpers
# -----------------------------

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_rfc822(dt_str: str) -> Optional[datetime]:
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _get_text(el: ET.Element, path: str, ns: Optional[dict] = None) -> Optional[str]:
    node = el.find(path, ns or {})
    if node is None or node.text is None:
        return None
    return node.text.strip()


def _normalize_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    t = text.lower()
    return any(k.lower() in t for k in keywords)


# -----------------------------
# Provider: Google Trends "Trending Now" RSS (with WHY fields)
# -----------------------------

class GoogleTrendsTrendingNowRSS:
    """
    Google Trends 'Trending now' RSS feed.

    Endpoint:
      https://trends.google.com/trending/rss?geo=US

    This feed includes extra fields in the ht: namespace that explain "why":
      - ht:approx_traffic
      - ht:news_item (title/source/url/snippet)
    """
    name = "google_trends_trending_now_rss"

    # Namespace used by Trends RSS extended fields
    HT_NS = {"ht": "https://trends.google.com/trending/rss"}

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
            title = _get_text(item, "title")
            if not title:
                continue

            pub = _get_text(item, "pubDate")
            published_at = _parse_rfc822(pub) if pub else None

            # Best-effort time filtering (some items may omit pubDate)
            if published_at is not None and published_at < since:
                continue

            # WHY fields
            approx_traffic = _get_text(item, "ht:approx_traffic", self.HT_NS)

            # The feed can include multiple ht:news_item entries; grab the first one
            news_item = item.find("ht:news_item", self.HT_NS)
            top_news_title = None
            top_news_url = None
            top_news_source = None
            top_news_snippet = None

            if news_item is not None:
                top_news_title = _get_text(news_item, "ht:news_item_title", self.HT_NS)
                top_news_url = _get_text(news_item, "ht:news_item_url", self.HT_NS)
                top_news_source = _get_text(news_item, "ht:news_item_source", self.HT_NS)
                top_news_snippet = _get_text(news_item, "ht:news_item_snippet", self.HT_NS)

            # Use the news URL if present (more useful than the feed URL)
            best_url = top_news_url or _get_text(item, "link")

            topics.append(
                TrendingTopic(
                    topic=_clean_text(title),
                    score=1.0,  # we convert feed order to score below
                    source=self.name,
                    url=best_url,
                    published_at=published_at,
                    extra={
                        "geo": self.geo,
                        "approx_traffic": approx_traffic,
                        "top_news_title": top_news_title,
                        "top_news_source": top_news_source,
                        "top_news_snippet": top_news_snippet,
                    },
                )
            )

        # Score by rank/order in feed: top item highest
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
# Provider: Google News RSS Search (politics/breaking news lens)
# -----------------------------

class GoogleNewsSearchRSS:
    """
    Google News query RSS (US-scoped).
    Endpoint:
      https://news.google.com/rss/search?q=<QUERY>&hl=en-US&gl=US&ceid=US:en

    Use this to backfill "breaking/political news" even if it's not trending in search.
    """
    name = "google_news_rss_search"

    def __init__(
        self,
        query: str,
        hl: str = "en-US",
        gl: str = "US",
        ceid: str = "US:en",
        timeout_s: int = 15,
        user_agent: str = "TrendFetcher/1.0",
    ) -> None:
        self.query = query
        self.hl = hl
        self.gl = gl
        self.ceid = ceid
        self.timeout_s = timeout_s
        self.user_agent = user_agent

    def fetch(self, since: datetime) -> List[TrendingTopic]:
        base = "https://news.google.com/rss/search"
        params = {"q": self.query, "hl": self.hl, "gl": self.gl, "ceid": self.ceid}
        headers = {"User-Agent": self.user_agent}

        resp = requests.get(base, params=params, headers=headers, timeout=self.timeout_s)
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
        channel = root.find("channel")
        if channel is None:
            return []

        items: List[TrendingTopic] = []
        now = _now_utc()

        for it in channel.findall("item"):
            title = _get_text(it, "title")
            if not title:
                continue

            link = _get_text(it, "link")

            pub = _get_text(it, "pubDate")
            published_at = _parse_rfc822(pub) if pub else None

            if published_at is not None and published_at < since:
                continue

            # Recency score (0..1 over 24h)
            if published_at is not None:
                age_h = (now - published_at).total_seconds() / 3600.0
                score = max(0.0, 1.0 - (age_h / 24.0))
            else:
                score = 0.25

            items.append(
                TrendingTopic(
                    topic=_clean_text(title),
                    score=score,
                    source=self.name,
                    url=link,
                    published_at=published_at,
                    extra={"query": self.query},
                )
            )

        return sorted(items, key=lambda x: x.score, reverse=True)


# -----------------------------
# Aggregator + Focus (politics filtering + backfill)
# -----------------------------

DEFAULT_US_POLITICS_KEYWORDS = [
    # People / offices
    "trump", "biden", "harris", "white house", "president", "congress", "senate", "house",
    "speaker", "governor",
    # Institutions
    "supreme court", "scotus", "doj", "fbi", "cia", "pentagon", "state department",
    # Elections / policy
    "election", "primary", "ballot", "campaign", "impeachment", "border", "immigration",
    "abortion", "gun", "tax", "tariff", "inflation", "shutdown", "budget",
    # Foreign policy / conflicts often tied to US politics
    "iran", "israel", "gaza", "ukraine", "russia", "china", "taiwan", "nato",
    # General news triggers
    "indictment", "trial", "verdict", "ruling", "bill", "sanctions", "ceasefire", "strike",
]

DEFAULT_NEWS_QUERY = (
    'Trump OR Biden OR Harris OR "White House" OR Congress OR Senate OR DOJ OR FBI OR '
    '"Supreme Court" OR election OR Iran OR Israel OR Gaza OR Ukraine OR Russia OR China'
)


class TrendAggregator:
    def __init__(self, providers: List[TrendingProvider]) -> None:
        self.providers = providers

    def get_topics(self, hours: int = 24) -> List[TrendingTopic]:
        since = _now_utc() - timedelta(hours=hours)
        out: List[TrendingTopic] = []
        for p in self.providers:
            out.extend(p.fetch(since=since))
        return out


def get_us_political_breaking_topics(
    hours: int = 24,
    limit: int = 10,
    min_trends_matches: int = 5,
    keywords: Optional[List[str]] = None,
    news_query: str = DEFAULT_NEWS_QUERY,
) -> List[TrendingTopic]:
    """
    Strategy:
      1) Pull Google Trends (with WHY fields)
      2) Filter to politics/breaking via keywords
      3) If fewer than desired, backfill from Google News RSS search query
      4) Dedupe and rank
    """
    keywords = keywords or DEFAULT_US_POLITICS_KEYWORDS
    since = _now_utc() - timedelta(hours=hours)

    trends_provider = GoogleTrendsTrendingNowRSS(geo="US")
    news_provider = GoogleNewsSearchRSS(query=news_query)

    trends = trends_provider.fetch(since=since)

    # Filter Trends to politics/breaking
    filtered_trends = [
        t for t in trends
        if _contains_any_keyword(t.topic, keywords)
        or _contains_any_keyword((t.extra or {}).get("top_news_title") or "", keywords)
        or _contains_any_keyword((t.extra or {}).get("top_news_snippet") or "", keywords)
    ]

    # If trends filtering yields too few, backfill with news
    backfill_needed = max(0, limit - len(filtered_trends))
    news_items: List[TrendingTopic] = []
    if len(filtered_trends) < max(min_trends_matches, limit):
        news_items = news_provider.fetch(since=since)

    # Combine and dedupe
    combined = filtered_trends + news_items

    seen = set()
    unique: List[TrendingTopic] = []
    for t in sorted(combined, key=lambda x: x.score, reverse=True):
        key = _normalize_key(t.topic)
        if key in seen:
            continue
        seen.add(key)
        unique.append(t)

    return unique[:limit]


# -----------------------------
# Pretty printing
# -----------------------------

def _fmt_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def print_topics(topics: List[TrendingTopic]) -> None:
    for t in topics:
        traffic = (t.extra or {}).get("approx_traffic")
        news_title = (t.extra or {}).get("top_news_title")
        news_source = (t.extra or {}).get("top_news_source")
        when = _fmt_dt(t.published_at)

        why_bits = []
        if traffic:
            why_bits.append(f"traffic={traffic}")
        if news_source or news_title:
            st = f"{news_source}: {news_title}" if news_source else f"{news_title}"
            why_bits.append(st)

        why = " | WHY: " + " | ".join([b for b in why_bits if b]) if why_bits else ""

        print(f"{t.score:.2f} | {t.topic} | {t.source} | {when}{why} | {t.url or ''}")


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    # Focused output: US political / breaking news topics
    topics = get_us_political_breaking_topics(
        hours=24,
        limit=12,
        min_trends_matches=5,
        keywords=DEFAULT_US_POLITICS_KEYWORDS,
        news_query=DEFAULT_NEWS_QUERY,
    )
    print_topics(topics)
