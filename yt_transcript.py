#!/usr/bin/env python3
"""
Extract a YouTube transcript from a video URL.

Usage:
  python yt_transcript.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --lang en --out transcript.txt
"""

from __future__ import annotations

import argparse
import re
import sys
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


def extract_video_id(url: str) -> str:
    """
    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
      - https://www.youtube.com/live/VIDEO_ID
      - URLs with extra params (t=, si=, etc.)
    """
    url = url.strip()
    if not url:
        raise ValueError("Empty URL")

    # If user passed just an ID
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url

    u = urlparse(url)

    host = (u.netloc or "").lower()
    path = (u.path or "").strip("/")

    # youtu.be/<id>
    if "youtu.be" in host:
        vid = path.split("/")[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
            return vid

    # youtube.com/watch?v=<id>
    if "youtube.com" in host or "m.youtube.com" in host:
        if path == "watch":
            qs = parse_qs(u.query)
            vid = (qs.get("v") or [""])[0]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
                return vid

        # youtube.com/shorts/<id> or /live/<id> or /embed/<id>
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] in {"shorts", "live", "embed"}:
            vid = parts[1]
            if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid):
                return vid

    raise ValueError(f"Could not extract a video id from: {url}")


from youtube_transcript_api import YouTubeTranscriptApi

def fetch_transcript(video_id: str, preferred_lang: str | None = None):
    try:
        api = YouTubeTranscriptApi()

        if preferred_lang:
            return api.fetch(video_id, languages=[preferred_lang])
        else:
            return api.fetch(video_id)

    except Exception as e:
        raise RuntimeError(f"Failed to fetch transcript: {e}")



def segments_to_text(segments) -> str:
    lines = []
    for s in segments:
        # v1.x returns objects with .text (and usually .start/.duration)
        if hasattr(s, "text"):
            text = (s.text or "").replace("\n", " ").strip()
        else:
            # older versions return dicts
            text = (s.get("text") or "").replace("\n", " ").strip()

        if text:
            lines.append(text)

    return "\n".join(lines)



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="YouTube URL (or 11-char video id)")
    ap.add_argument("--lang", default=None, help="Preferred language code (e.g., en, ru, es)")
    ap.add_argument("--out", default=None, help="Save transcript to this file (txt)")
    args = ap.parse_args()

    try:
        vid = extract_video_id(args.url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    try:
        segments = fetch_transcript(vid, preferred_lang=args.lang)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    text = segments_to_text(segments)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Saved transcript to: {args.out}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
