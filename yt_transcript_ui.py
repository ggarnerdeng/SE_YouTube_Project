#!/usr/bin/env python3
"""
YouTube Transcript Extractor (UI)

Features:
- Paste a YouTube URL (watch/shorts/youtu.be/live/embed) or 11-char video id
- Fetch transcript via youtube-transcript-api (works with v1.x like 1.2.4)
- Adds timestamps to each line [mm:ss]
- Fetches metadata (title, description, channel/author, publish date, duration, view count, tags, etc.)
  via yt-dlp (no API key required)
- Best-effort: fetch top comments via yt-dlp (no API key required; may be brittle)
- Shows everything in the UI and lets you Save As .txt

Install:
  python -m pip install youtube-transcript-api yt-dlp

Run:
  python yt_transcript_ui.py
"""

from __future__ import annotations

import re
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL

VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def extract_video_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("Please paste a YouTube URL (or a video id).")

    # If user pasted just the 11-char ID
    if VIDEO_ID_RE.fullmatch(s):
        return s

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",                 # watch?v=ID
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",         # youtu.be/ID
        r"(?:/shorts/)([A-Za-z0-9_-]{11})",           # /shorts/ID
        r"(?:/live/)([A-Za-z0-9_-]{11})",             # /live/ID
        r"(?:/embed/)([A-Za-z0-9_-]{11})",            # /embed/ID
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)

    raise ValueError("Could not extract a video id from that input.")


def format_mmss(seconds: float | None) -> str:
    if seconds is None:
        return "00:00"
    total = int(round(float(seconds)))
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


def fetch_transcript_segments(video_id: str, lang: str | None = None):
    """
    youtube-transcript-api v1.x:
      api = YouTubeTranscriptApi()
      api.fetch(video_id, languages=[...]) -> list of snippet objects (with .text, .start, .duration)
    """
    api = YouTubeTranscriptApi()
    if lang:
        return api.fetch(video_id, languages=[lang])
    return api.fetch(video_id)


def segments_to_timestamped_text(segments) -> str:
    """Convert snippets to "[mm:ss] text" lines (v1.x snippet objects)."""
    lines: list[str] = []
    for seg in segments:
        text = (getattr(seg, "text", "") or "").replace("\n", " ").strip()
        start = getattr(seg, "start", None)
        if text:
            lines.append(f"[{format_mmss(start)}] {text}")
    return "\n".join(lines)


def fetch_video_metadata(video_id: str, max_comments: int = 30) -> dict:
    """
    Use yt-dlp to fetch metadata + (optionally) top comments without a YouTube API key.
    NOTE: comment scraping can be brittle and may fail depending on yt-dlp version / YouTube changes.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,

        # Enable comment extraction (best effort)
        "getcomments": True,

        # Ask for top comments and cap count (support depends on yt-dlp version)
        "extractor_args": {
            "youtube": {
                "max_comments": [str(max_comments)],
                "comment_sort": ["top"],
            }
        },
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    comments_raw = info.get("comments") or []
    comments = []
    for c in comments_raw:
        if not isinstance(c, dict):
            continue
        text = (c.get("text") or c.get("content") or "").strip()
        if not text:
            continue
        comments.append({
            "author": c.get("author"),
            "text": text,
            "like_count": c.get("like_count"),
        })

    meta = {
        "webpage_url": info.get("webpage_url") or url,
        "title": info.get("title"),
        "description": info.get("description"),
        "channel": info.get("channel") or info.get("uploader"),
        "channel_url": info.get("channel_url") or info.get("uploader_url"),
        "uploader": info.get("uploader"),
        "uploader_url": info.get("uploader_url"),
        "upload_date": info.get("upload_date"),  # YYYYMMDD
        "duration_seconds": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "tags": info.get("tags") or [],
        "categories": info.get("categories") or [],
        "language": info.get("language"),
        "top_comments": comments,
    }
    return meta


def format_metadata_block(meta: dict) -> str:
    """Create a ChatGPT-friendly header block to save with transcript."""

    def fmt_int(x):
        return f"{x:,}" if isinstance(x, int) else (str(x) if x is not None else "")

    def fmt_date(yyyymmdd: str | None) -> str:
        if not yyyymmdd or len(yyyymmdd) != 8:
            return yyyymmdd or ""
        return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

    duration = meta.get("duration_seconds")
    duration_str = format_mmss(duration) if duration is not None else ""

    tags = meta.get("tags") or []
    cats = meta.get("categories") or []

    lines: list[str] = [
        "=== VIDEO METADATA ===",
        f"URL: {meta.get('webpage_url', '')}",
        f"Title: {meta.get('title', '')}",
        f"Channel: {meta.get('channel', '')}",
        f"Channel URL: {meta.get('channel_url', '')}",
        f"Uploader: {meta.get('uploader', '')}",
        f"Uploader URL: {meta.get('uploader_url', '')}",
        f"Upload date: {fmt_date(meta.get('upload_date'))}",
        f"Duration: {duration_str} ({meta.get('duration_seconds') or ''} seconds)",
        f"Views: {fmt_int(meta.get('view_count'))}",
        f"Likes: {fmt_int(meta.get('like_count'))}",
        f"Comments: {fmt_int(meta.get('comment_count'))}",
        f"Categories: {', '.join(cats) if cats else ''}",
        f"Tags: {', '.join(tags) if tags else ''}",
        f"Detected language: {meta.get('language', '')}",
        "",
        "=== VIDEO DESCRIPTION ===",
        (meta.get("description") or "").strip(),
        "",
        "=== TOP COMMENTS ===",
    ]

    top_comments = meta.get("top_comments") or []
    if not top_comments:
        lines.append("(No comments fetched.)")
    else:
        for i, c in enumerate(top_comments, start=1):
            author = c.get("author") or "Unknown"
            likes = c.get("like_count")
            likes_str = f"{likes:,}" if isinstance(likes, int) else ""
            text = (c.get("text") or "").replace("\n", " ").strip()
            # Keep it one-line per comment for easy LLM parsing
            lines.append(f"{i}. {author} ({likes_str} likes): {text}")

    lines += [
        "",
        "=== TRANSCRIPT (TIMESTAMPED) ===",
    ]

    return "\n".join(lines).rstrip() + "\n"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YouTube Transcript Extractor (Metadata + Comments + Timestamps)")
        self.geometry("980x720")
        self.minsize(780, 560)

        self.url_var = tk.StringVar()
        self.lang_var = tk.StringVar(value="en")  # user can clear to auto

        self.status = tk.StringVar(value="Ready.")
        self.current_meta: dict | None = None

        self._build_ui()

    def _build_ui(self):
        pad = 10

        top = ttk.Frame(self)
        top.pack(fill="x", padx=pad, pady=(pad, 0))
        top.columnconfigure(0, weight=1)

        ttk.Label(top, text="YouTube URL or Video ID:").grid(row=0, column=0, sticky="w")
        url_entry = ttk.Entry(top, textvariable=self.url_var)
        url_entry.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(4, 0))
        url_entry.focus()

        ttk.Label(top, text="Language (optional, e.g. en). Clear to auto:").grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(top, textvariable=self.lang_var, width=10).grid(
            row=3, column=0, sticky="w", pady=(4, 0)
        )

        self.fetch_btn = ttk.Button(top, text="Fetch", command=self.on_fetch)
        self.fetch_btn.grid(row=3, column=1, sticky="w", padx=(10, 0))

        self.copy_btn = ttk.Button(top, text="Copy All", command=self.on_copy, state="disabled")
        self.copy_btn.grid(row=3, column=2, sticky="w", padx=(10, 0))

        self.save_btn = ttk.Button(top, text="Save As…", command=self.on_save, state="disabled")
        self.save_btn.grid(row=3, column=3, sticky="w", padx=(10, 0))

        self.clear_btn = ttk.Button(top, text="Clear", command=self.on_clear)
        self.clear_btn.grid(row=3, column=4, sticky="w", padx=(10, 0))

        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=pad, pady=(8, 0))

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=pad, pady=pad)

        self.text = tk.Text(mid, wrap="word")
        self.text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(mid, command=self.text.yview)
        scroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scroll.set)

    def _has_content(self) -> bool:
        return bool(self.text.get("1.0", "end-1c").strip())

    def set_busy(self, busy: bool):
        self.fetch_btn.configure(state=("disabled" if busy else "normal"))
        self.clear_btn.configure(state=("disabled" if busy else "normal"))
        self.copy_btn.configure(state=("disabled" if busy else ("normal" if self._has_content() else "disabled")))
        self.save_btn.configure(state=("disabled" if busy else ("normal" if self._has_content() else "disabled")))

    def on_clear(self):
        self.text.delete("1.0", "end")
        self.current_meta = None
        self.copy_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        self.status.set("Cleared.")

    def on_copy(self):
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self.status.set("Copied to clipboard.")

    def on_fetch(self):
        url_or_id = self.url_var.get().strip()
        lang = self.lang_var.get().strip() or None

        try:
            video_id = extract_video_id(url_or_id)
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.set_busy(True)
        self.status.set("Fetching metadata, comments, and transcript…")

        def worker():
            try:
                meta = fetch_video_metadata(video_id)

                segments = fetch_transcript_segments(video_id, lang=lang)
                transcript_text = segments_to_timestamped_text(segments)

                if not transcript_text.strip():
                    raise RuntimeError("Transcript fetched, but it was empty.")

                full_output = format_metadata_block(meta) + transcript_text + "\n"
                self._ui_success(full_output, meta)
            except Exception as e:
                self._ui_error(e)

        threading.Thread(target=worker, daemon=True).start()

    def _ui_success(self, output: str, meta: dict):
        def run():
            self.current_meta = meta
            self.text.delete("1.0", "end")
            self.text.insert("1.0", output)
            self.status.set("Done.")
            self.set_busy(False)
            self.copy_btn.configure(state="normal")
            self.save_btn.configure(state="normal")

        self.after(0, run)

    def _ui_error(self, err: Exception):
        def run():
            self.status.set("Error.")
            self.set_busy(False)
            messagebox.showerror("Failed", str(err))

        self.after(0, run)

    def on_save(self):
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Nothing to save", "The output box is empty.")
            return

        initial = "transcript.txt"
        if self.current_meta and self.current_meta.get("title"):
            safe = re.sub(r"[\\/:*?\"<>|]+", "", self.current_meta["title"]).strip()
            safe = re.sub(r"\s+", " ", safe)[:80]
            if safe:
                initial = f"{safe}.txt"

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=initial,
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            title="Save output as…",
        )
        if not path:
            return

        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")

        self.status.set(f"Saved: {path}")


if __name__ == "__main__":
    # Nicer DPI handling on Windows (safe no-op elsewhere)
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    App().mainloop()
