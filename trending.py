"""
trending.py (Config-driven + Debug + Fallback + Simple UI + YouTube links)

Key behavior:
- Fetch Google Trends "Trending Now" RSS (with WHY fields)
- Optional Google News RSS search backfill
- Filter both sources using keywords.txt (so keywords actually matter)
- If filtering yields 0 results, fallback_mode avoids blank output
- UI: lets you run it without editing code
- Output includes YouTube search links per topic (no API key needed)

Dependencies:
  python -m pip install requests

Run:
  python trending.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any, Protocol
import json
import os
import re
import urllib.parse
import webbrowser
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


def _youtube_search_url(topic: str) -> str:
    """
    No API key needed. Returns a YouTube search URL for the topic.
    """
    q = topic.strip()
    # small nudge toward news coverage without being too biased
    if len(q) < 60:
        q = f"{q} news"
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(q)


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
                    score=1.0,  # will be replaced by feed order scoring below
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
# Core logic
# -----------------------------

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

    need_news = False
    if enable_news_backfill and news_provider is not None:
        need_news = len(filtered_trends) < max(min_trends_matches, limit)

        # If user requested fallback-to-news, fetch news regardless so fallback can work.
        if (fallback_mode or "").strip() == "fallback_to_news_unfiltered":
            need_news = True

    if need_news and news_provider is not None:
        raw_news = news_provider.fetch(since=since)

    filtered_news = [t for t in raw_news if _topic_matches_keywords(t, keywords)]

    combined_filtered = _dedupe_sort(filtered_trends + filtered_news)[:limit]

    if debug:
        print("[DEBUG] counts:")
        print(f"  raw_trends={len(raw_trends)} filtered_trends={len(filtered_trends)}")
        print(f"  raw_news={len(raw_news)} filtered_news={len(filtered_news)}")
        print(f"  combined_filtered={len(combined_filtered)}")
        print(f"  fallback_mode={(fallback_mode or '').strip()}")

    if combined_filtered:
        return combined_filtered

    # ---- Fallbacks when strict filtering yields 0 ----
    mode = (fallback_mode or "strict").strip()

    if mode == "strict":
        return []

    if mode == "fallback_to_news_unfiltered" and raw_news:
        return _dedupe_sort(raw_news)[:fallback_limit]

    if mode == "fallback_to_trends_unfiltered" and raw_trends:
        return _dedupe_sort(raw_trends)[:fallback_limit]

    return []


# -----------------------------
# Output formatting
# -----------------------------

def topics_to_rows(topics: List[TrendingTopic]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for t in topics:
        traffic = (t.extra or {}).get("approx_traffic") or ""
        news_title = (t.extra or {}).get("top_news_title") or ""
        news_source = (t.extra or {}).get("top_news_source") or ""
        when = _fmt_dt(t.published_at)
        why = ""

        bits = []
        if traffic:
            bits.append(f"traffic={traffic}")
        if news_source or news_title:
            bits.append(f"{news_source}: {news_title}".strip(": "))

        if bits:
            why = " | ".join(bits)

        yt = _youtube_search_url(t.topic)

        rows.append({
            "score": f"{t.score:.2f}",
            "topic": t.topic,
            "source": t.source,
            "when": when,
            "why": why,
            "news_url": t.url or "",
            "youtube_url": yt,
        })
    return rows


def print_topics_cli(topics: List[TrendingTopic]) -> None:
    if not topics:
        print("[INFO] No topics matched your keywords. (Try broadening keywords.txt or use fallback_mode.)")
        return

    rows = topics_to_rows(topics)
    for r in rows:
        why = f" | WHY: {r['why']}" if r["why"] else ""
        print(
            f"{r['score']} | {r['topic']} | {r['source']} | {r['when']}{why} | news={r['news_url']} | yt={r['youtube_url']}"
        )


# -----------------------------
# Config load
# -----------------------------

def load_from_config(config_path: str) -> Dict[str, Any]:
    cfg = _read_json(config_path)

    base_dir = os.path.dirname(os.path.abspath(config_path))

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
    keywords = _read_keywords(keywords_path) if os.path.exists(keywords_path) else []

    enable_trends = bool(cfg.get("enable_trends", True))
    enable_news_backfill = bool(cfg.get("enable_news_backfill", True))

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

    return {
        "hours": hours,
        "limit": limit,
        "min_trends_matches": min_trends_matches,
        "geo": geo,
        "user_agent": user_agent,
        "timeout_s": timeout_s,
        "debug": debug,
        "fallback_mode": fallback_mode,
        "fallback_limit": fallback_limit,
        "keywords_path": keywords_path,
        "keywords": keywords,
        "enable_trends": enable_trends,
        "enable_news_backfill": enable_news_backfill,
        "trends_provider": trends_provider,
        "news_provider": news_provider,
        "config_dir": base_dir,
    }


# -----------------------------
# UI (Tkinter)
# -----------------------------

def run_ui() -> int:
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except Exception as e:
        print("[ERROR] Tkinter is not available in this Python install.")
        print("Try running in CLI mode instead.")
        print(str(e))
        return 1

    root = tk.Tk()
    root.title("Trending Topics (Trends + News) → with YouTube Links")
    root.geometry("1200x650")

    # State
    state = {
        "config_path": None,
        "keywords_path": None,
        "loaded": None,  # dict from load_from_config
        "rows": [],      # output rows
    }

    # Top controls frame
    top = ttk.Frame(root, padding=10)
    top.pack(fill="x")

    config_var = tk.StringVar(value="(no config loaded)")
    keywords_var = tk.StringVar(value="(no keywords loaded)")

    def set_loaded_labels():
        config_var.set(state["config_path"] or "(no config loaded)")
        keywords_var.set(state["keywords_path"] or "(no keywords loaded)")

    def load_config_clicked():
        path = filedialog.askopenfilename(
            title="Select config.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            loaded = load_from_config(path)
        except Exception as e:
            messagebox.showerror("Config Load Error", str(e))
            return

        state["config_path"] = path
        state["loaded"] = loaded

        # infer keywords path from config
        state["keywords_path"] = loaded.get("keywords_path")
        set_loaded_labels()

        # Populate form fields
        hours_entry.delete(0, tk.END)
        hours_entry.insert(0, str(loaded["hours"]))

        limit_entry.delete(0, tk.END)
        limit_entry.insert(0, str(loaded["limit"]))

        min_matches_entry.delete(0, tk.END)
        min_matches_entry.insert(0, str(loaded["min_trends_matches"]))

        fallback_limit_entry.delete(0, tk.END)
        fallback_limit_entry.insert(0, str(loaded["fallback_limit"]))

        geo_entry.delete(0, tk.END)
        geo_entry.insert(0, str(loaded["geo"]))

        fallback_mode_combo.set(str(loaded["fallback_mode"]))

        enable_trends_var.set(bool(loaded["enable_trends"]))
        enable_news_var.set(bool(loaded["enable_news_backfill"]))
        debug_var.set(bool(loaded["debug"]))

        # Show keywords (preview)
        try:
            kw = _read_keywords(state["keywords_path"]) if state["keywords_path"] else []
            keywords_preview.delete("1.0", tk.END)
            keywords_preview.insert(tk.END, "\n".join(kw))
        except Exception:
            pass

    def load_keywords_clicked():
        path = filedialog.askopenfilename(
            title="Select keywords.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        state["keywords_path"] = path
        set_loaded_labels()
        try:
            kw = _read_keywords(path)
            keywords_preview.delete("1.0", tk.END)
            keywords_preview.insert(tk.END, "\n".join(kw))
        except Exception as e:
            messagebox.showerror("Keywords Load Error", str(e))

    def run_clicked():
        if not state["loaded"]:
            messagebox.showwarning("No Config", "Load config.json first.")
            return

        # Build effective runtime settings from UI inputs
        try:
            hours = int(hours_entry.get().strip())
            limit = int(limit_entry.get().strip())
            min_matches = int(min_matches_entry.get().strip())
            geo = geo_entry.get().strip() or "US"
            fallback_mode = fallback_mode_combo.get().strip() or "strict"
            fallback_limit = int(fallback_limit_entry.get().strip())
            enable_trends = bool(enable_trends_var.get())
            enable_news = bool(enable_news_var.get())
            debug = bool(debug_var.get())
        except Exception as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
            return

        # Reload keywords from selected file
        keywords_path = state["keywords_path"]
        if not keywords_path or not os.path.exists(keywords_path):
            messagebox.showerror("Keywords Missing", "keywords.txt not found. Load or fix path.")
            return

        try:
            keywords = _read_keywords(keywords_path)
        except Exception as e:
            messagebox.showerror("Keywords Error", str(e))
            return

        # Rebuild providers using loaded config defaults for UA/timeout, but with geo override from UI
        loaded = state["loaded"]
        timeout_s = int(loaded["timeout_s"])
        user_agent = str(loaded["user_agent"])

        trends_provider = GoogleTrendsTrendingNowRSS(
            geo=geo, timeout_s=timeout_s, user_agent=user_agent
        ) if enable_trends else None

        # Keep the news provider from loaded config (it contains the query parameters);
        # if enable_news is false, set None.
        news_provider = loaded["news_provider"] if enable_news else None

        try:
            topics = get_focus_topics(
                hours=hours,
                limit=limit,
                min_trends_matches=min_matches,
                keywords=keywords,
                enable_trends=enable_trends,
                enable_news_backfill=enable_news,
                fallback_mode=fallback_mode,
                fallback_limit=fallback_limit,
                trends_provider=trends_provider,
                news_provider=news_provider,
                debug=debug,
            )
        except Exception as e:
            messagebox.showerror("Run Error", str(e))
            return

        rows = topics_to_rows(topics)
        state["rows"] = rows

        # Clear table
        for item in tree.get_children():
            tree.delete(item)

        # Fill table
        for i, r in enumerate(rows):
            tree.insert(
                "",
                "end",
                iid=str(i),
                values=(r["score"], r["topic"], r["source"], r["when"], r["why"], r["news_url"], r["youtube_url"])
            )

        if not rows:
            messagebox.showinfo("No Matches", "No topics matched. Try broadening keywords or change fallback_mode.")

    def open_selected_news():
        sel = tree.selection()
        if not sel:
            return
        r = state["rows"][int(sel[0])]
        url = r.get("news_url")
        if url:
            webbrowser.open(url)

    def open_selected_youtube():
        sel = tree.selection()
        if not sel:
            return
        r = state["rows"][int(sel[0])]
        url = r.get("youtube_url")
        if url:
            webbrowser.open(url)

    # Row 1: config/keywords selectors
    ttk.Button(top, text="Load config.json", command=load_config_clicked).grid(row=0, column=0, padx=5, pady=5, sticky="w")
    ttk.Label(top, textvariable=config_var).grid(row=0, column=1, padx=5, pady=5, sticky="w")

    ttk.Button(top, text="Load keywords.txt", command=load_keywords_clicked).grid(row=1, column=0, padx=5, pady=5, sticky="w")
    ttk.Label(top, textvariable=keywords_var).grid(row=1, column=1, padx=5, pady=5, sticky="w")

    # Row 2+: inputs
    ttk.Label(top, text="Hours").grid(row=0, column=2, padx=5, pady=5, sticky="e")
    hours_entry = ttk.Entry(top, width=8)
    hours_entry.grid(row=0, column=3, padx=5, pady=5, sticky="w")
    hours_entry.insert(0, "24")

    ttk.Label(top, text="Limit").grid(row=0, column=4, padx=5, pady=5, sticky="e")
    limit_entry = ttk.Entry(top, width=8)
    limit_entry.grid(row=0, column=5, padx=5, pady=5, sticky="w")
    limit_entry.insert(0, "12")

    ttk.Label(top, text="Min trends matches").grid(row=0, column=6, padx=5, pady=5, sticky="e")
    min_matches_entry = ttk.Entry(top, width=8)
    min_matches_entry.grid(row=0, column=7, padx=5, pady=5, sticky="w")
    min_matches_entry.insert(0, "5")

    ttk.Label(top, text="Geo").grid(row=1, column=2, padx=5, pady=5, sticky="e")
    geo_entry = ttk.Entry(top, width=8)
    geo_entry.grid(row=1, column=3, padx=5, pady=5, sticky="w")
    geo_entry.insert(0, "US")

    ttk.Label(top, text="Fallback mode").grid(row=1, column=4, padx=5, pady=5, sticky="e")
    fallback_mode_combo = ttk.Combobox(top, width=26, state="readonly")
    fallback_mode_combo["values"] = ("strict", "fallback_to_news_unfiltered", "fallback_to_trends_unfiltered")
    fallback_mode_combo.grid(row=1, column=5, padx=5, pady=5, sticky="w")
    fallback_mode_combo.set("fallback_to_news_unfiltered")

    ttk.Label(top, text="Fallback limit").grid(row=1, column=6, padx=5, pady=5, sticky="e")
    fallback_limit_entry = ttk.Entry(top, width=8)
    fallback_limit_entry.grid(row=1, column=7, padx=5, pady=5, sticky="w")
    fallback_limit_entry.insert(0, "12")

    enable_trends_var = tk.BooleanVar(value=True)
    enable_news_var = tk.BooleanVar(value=True)
    debug_var = tk.BooleanVar(value=False)

    ttk.Checkbutton(top, text="Enable Trends", variable=enable_trends_var).grid(row=2, column=2, padx=5, pady=5, sticky="w")
    ttk.Checkbutton(top, text="Enable News", variable=enable_news_var).grid(row=2, column=3, padx=5, pady=5, sticky="w")
    ttk.Checkbutton(top, text="Debug", variable=debug_var).grid(row=2, column=4, padx=5, pady=5, sticky="w")

    ttk.Button(top, text="Run", command=run_clicked).grid(row=2, column=5, padx=5, pady=5, sticky="w")
    ttk.Button(top, text="Open Selected News", command=open_selected_news).grid(row=2, column=6, padx=5, pady=5, sticky="w")
    ttk.Button(top, text="Open Selected YouTube", command=open_selected_youtube).grid(row=2, column=7, padx=5, pady=5, sticky="w")

    # Main split area: keywords preview + results table
    mid = ttk.Panedwindow(root, orient="horizontal")
    mid.pack(fill="both", expand=True, padx=10, pady=10)

    # Left: keywords preview
    left = ttk.Frame(mid, padding=8)
    mid.add(left, weight=1)

    ttk.Label(left, text="Keywords Preview (from keywords.txt)").pack(anchor="w")
    keywords_preview = __import__("tkinter").Text(left, height=12, wrap="none")
    keywords_preview.pack(fill="both", expand=False, pady=6)

    ttk.Label(left, text="Tip: edit keywords.txt, save, then click Run.").pack(anchor="w")

    # Right: results table
    right = ttk.Frame(mid, padding=8)
    mid.add(right, weight=3)

    cols = ("score", "topic", "source", "when", "why", "news_url", "youtube_url")
    tree = ttk.Treeview(right, columns=cols, show="headings", selectmode="browse")
    tree.heading("score", text="Score")
    tree.heading("topic", text="Topic")
    tree.heading("source", text="Source")
    tree.heading("when", text="Time (UTC)")
    tree.heading("why", text="Why (traffic/headline)")
    tree.heading("news_url", text="News URL")
    tree.heading("youtube_url", text="YouTube URL")

    tree.column("score", width=60, anchor="center")
    tree.column("topic", width=220)
    tree.column("source", width=170)
    tree.column("when", width=130)
    tree.column("why", width=380)
    tree.column("news_url", width=220)
    tree.column("youtube_url", width=220)

    yscroll = ttk.Scrollbar(right, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=yscroll.set)

    tree.pack(side="left", fill="both", expand=True)
    yscroll.pack(side="right", fill="y")

    set_loaded_labels()
    root.mainloop()
    return 0


# -----------------------------
# CLI entry (still works)
# -----------------------------

def run_cli() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = _resolve_path(base_dir, "config.json")

    if not os.path.exists(config_path):
        print(f"[ERROR] Missing config file: {config_path}")
        return 1

    loaded = load_from_config(config_path)

    topics = get_focus_topics(
        hours=loaded["hours"],
        limit=loaded["limit"],
        min_trends_matches=loaded["min_trends_matches"],
        keywords=loaded["keywords"],
        enable_trends=loaded["enable_trends"],
        enable_news_backfill=loaded["enable_news_backfill"],
        fallback_mode=loaded["fallback_mode"],
        fallback_limit=loaded["fallback_limit"],
        trends_provider=loaded["trends_provider"],
        news_provider=loaded["news_provider"],
        debug=loaded["debug"],
    )

    print_topics_cli(topics)
    return 0


if __name__ == "__main__":
    # Default: UI
    # If you want CLI: run `python trending.py --cli`
    import sys
    if "--cli" in sys.argv:
        raise SystemExit(run_cli())
    raise SystemExit(run_ui())
