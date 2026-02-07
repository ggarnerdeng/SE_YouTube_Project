"""
trending.py (Config-driven + Debug + Fallback)

Key behavior:
- Fetch Trends (with WHY fields) + optional News backfill
- Filter BOTH sources by keywords.txt (so keywords actually matter)
- If filtering returns 0, apply fallback_mode (config) so you never get a blank screen

Dependencies:
  python -m pip install requests

Run:
  python trending.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Protocol, Tuple
import json
import os
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
    return re.sub(r"\s+", " ", s).strip()


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


def _fmt_dt(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_keywords(path: str) -> List[str]:
    """
    Supports:
      - one keyword per line
      - ignores blank lines
      - ignores lines starting with '#'
      - supports inline comments: "trump # note"
    Returns lowercase keywords.
    """
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            out.append(line.lower())
    return out


def _resolve_path(base_dir: str, maybe_rel_path: str) -> str:
    if os.path.isabs(maybe_rel_path):
        return maybe_rel_path
    return os.path.join(base_dir, maybe_rel_path)


def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    t = (text or "").lower()
    return any(k in t for k in keywords)


def _topic_matches_keywords(t: TrendingTopic, keywords: List[str]) -> bool:
    if _contains_any_keyword(t.topic, keywords):
        return True
    extra = t.extra or {}
    if _contains_any_keyword(extra.get("top_news_title") or "", keywords):
        return True
    if _contains_any_keyword(extra.get("top_news_snippet") or "", keywords):
        return True
    return False


# -----------------------------
# Providers
# -----------------------------

class GoogleTrendsTrendingNowRSS:
    name = "google_trends_trending_now_rss"
    HT_NS = {"ht": "https://trends.google.com/trending/rss"}

    def __init__(self, geo: str = "US", timeout_s: int = 15, user_agent: str = "TrendFetcher/1.0") -> None:
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
            if published_at is not None and published_at < since:
                continue

            approx_traffic = _get_text(item, "ht:approx_traffic", self.HT_NS)

            news_item = item.find("ht:news_item", self.HT_NS)
            top_news_title = top_news_url = top_news_source = top_news_snippet = None
            if news_item is not None:
                top_news_title = _get_text(news_item, "ht:news_item_title", self.HT_NS)
                top_news_url = _get_text(news_item, "ht:news_item_url", self.HT_NS)
                top_news_source = _get_text(news_item, "ht:news_item_source", self.HT_NS)
                top_news_snippet = _get_text(news_item, "ht:news_item_snippet", self.HT_NS)

            best_url = top_news_url or _get_text(item, "link")

            topics.append(
                TrendingTopic(
                    topic=_clean_text(title),
                    score=1.0,
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

        # Score by feed order (top item highest)
        n = len(topics)
        return [
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


class GoogleNewsSearchRSS:
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
# Core logic
# -----------------------------

def _dedupe_sort(topics: List[TrendingTopic]) -> List[TrendingTopic]:
    seen = set()
    out: List[TrendingTopic] = []
    for t in sorted(topics, key=lambda x: x.score, reverse=True):
        key = _normalize_key(t.topic)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def get_focus_topics(
    *,
    hours: int,
    limit: int,
    min_trends_matches: int,
    keywords: List[str],
    enable_trends: bool,
    enable_news_backfill: bool,
    fallback_mode: str,
    fallback_limit: int,
    trends_provider: Optional[GoogleTrendsTrendingNowRSS],
    news_provider: Optional[GoogleNewsSearchRSS],
    debug: bool,
) -> List[TrendingTopic]:
    since = _now_utc() - timedelta(hours=hours)

    raw_trends: List[TrendingTopic] = []
    raw_news: List[TrendingTopic] = []

    if enable_trends and trends_provider is not None:
        raw_trends = trends_provider.fetch(since=since)

    filtered_trends = [t for t in raw_trends if _topic_matches_keywords(t, keywords)]

    if enable_news_backfill and news_provider is not None:
        need_news = len(filtered_trends) < max(min_trends_matches, limit)

    # If user requested fallback-to-news, fetch news regardless so fallback can work.
    if (fallback_mode == "fallback_to_news_unfiltered"):
        need_news = True

    if need_news:
        raw_news = news_provider.fetch(since=since)


    filtered_news = [t for t in raw_news if _topic_matches_keywords(t, keywords)]

    combined_filtered = _dedupe_sort(filtered_trends + filtered_news)[:limit]

    if debug:
        print("[DEBUG] counts:")
        print(f"  raw_trends={len(raw_trends)} filtered_trends={len(filtered_trends)}")
        print(f"  raw_news={len(raw_news)} filtered_news={len(filtered_news)}")
        print(f"  combined_filtered={len(combined_filtered)}")
        print(f"  fallback_mode={fallback_mode}")

    if combined_filtered:
        return combined_filtered

    # ---- Fallbacks when strict filtering yields 0 ----
    fallback_mode = (fallback_mode or "strict").strip()

    if fallback_mode == "strict":
        return []

    if fallback_mode == "fallback_to_news_unfiltered" and raw_news:
        return _dedupe_sort(raw_news)[:fallback_limit]

    if fallback_mode == "fallback_to_trends_unfiltered" and raw_trends:
        return _dedupe_sort(raw_trends)[:fallback_limit]

    # If fallback had nothing to use, return empty
    return []


# -----------------------------
# Output
# -----------------------------

def print_topics(topics: List[TrendingTopic]) -> None:
    if not topics:
        print("[INFO] No topics matched your keywords. (Try broadening keywords.txt or use fallback_mode.)")
        return

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
# CLI entry
# -----------------------------

def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = _resolve_path(base_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"[ERROR] Missing config file: {config_path}")
        return 1

    cfg = _read_json(config_path)

    hours = int(cfg.get("hours", 24))
    limit = int(cfg.get("limit", 10))
    min_trends_matches = int(cfg.get("min_trends_matches", 5))

    geo = str(cfg.get("geo", "US"))
    user_agent = str(cfg.get("user_agent", "TrendFetcher/1.0"))
    timeout_s = int(cfg.get("timeout_s", 15))

    debug = bool(cfg.get("debug", False))
    fallback_mode = str(cfg.get("fallback_mode", "strict"))
    fallback_limit = int(cfg.get("fallback_limit", limit))

    keywords_file = str(cfg.get("keywords_file", "keywords.txt"))
    keywords_path = _resolve_path(base_dir, keywords_file)

    enable_trends = bool(cfg.get("enable_trends", True))
    enable_news_backfill = bool(cfg.get("enable_news_backfill", True))

    if not os.path.exists(keywords_path):
        print(f"[ERROR] Missing keywords file: {keywords_path}")
        return 1

    keywords = _read_keywords(keywords_path)

    trends_provider = GoogleTrendsTrendingNowRSS(
        geo=geo, timeout_s=timeout_s, user_agent=user_agent
    ) if enable_trends else None

    news_cfg = cfg.get("news", {}) if isinstance(cfg.get("news", {}), dict) else {}
    news_query = str(news_cfg.get("query", "")).strip()

    news_provider = None
    if enable_news_backfill and news_query:
        news_provider = GoogleNewsSearchRSS(
            query=news_query,
            hl=str(news_cfg.get("hl", "en-US")),
            gl=str(news_cfg.get("gl", "US")),
            ceid=str(news_cfg.get("ceid", "US:en")),
            timeout_s=timeout_s,
            user_agent=user_agent,
        )

    topics = get_focus_topics(
        hours=hours,
        limit=limit,
        min_trends_matches=min_trends_matches,
        keywords=keywords,
        enable_trends=enable_trends,
        enable_news_backfill=enable_news_backfill,
        fallback_mode=fallback_mode,
        fallback_limit=fallback_limit,
        trends_provider=trends_provider,
        news_provider=news_provider,
        debug=debug,
    )

    print_topics(topics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
print("[DEBUG] using config:", config_path)
print("[DEBUG] enable_news_backfill =", cfg.get("enable_news_backfill"))
print("[DEBUG] news.query =", (cfg.get("news", {}) or {}).get("query"))
