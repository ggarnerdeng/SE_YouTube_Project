#!/usr/bin/env python3
from __future__ import annotations

import re
import threading
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi

# captions (ASS + word-timed)
from captions import make_ass_from_clip_whisper, burn_in_ass

# NEW: autocrop to 9:16 following largest face
from autocrop import autocrop_to_9x16_face, AutoCropConfig


VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")


def extract_video_id(url_or_id: str) -> str:
    s = (url_or_id or "").strip()
    if not s:
        raise ValueError("Please paste a YouTube URL (or a video id).")

    if VIDEO_ID_RE.fullmatch(s):
        return s

    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:/live/)([A-Za-z0-9_-]{11})",
        r"(?:/embed/)([A-Za-z0-9_-]{11})",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)

    raise ValueError("Could not extract a video id from that input.")


def parse_timestamp(ts: str) -> float:
    s = (ts or "").strip()
    if not s:
        raise ValueError("Timestamp is empty.")

    if s.isdigit():
        return float(int(s))

    parts = s.split(":")
    if any(not p.isdigit() for p in parts):
        raise ValueError(f"Invalid timestamp: {ts}")

    parts = [int(p) for p in parts]
    if len(parts) == 2:
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss

    raise ValueError(f"Invalid timestamp format: {ts}")


def format_mmss(seconds: float | None) -> str:
    if seconds is None:
        return "00:00"
    total = int(round(float(seconds)))
    mm = total // 60
    ss = total % 60
    return f"{mm:02d}:{ss:02d}"


@dataclass
class TranscriptSeg:
    start: float
    duration: float
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration


def fetch_transcript_segments(video_id: str, lang: Optional[str] = None) -> List[TranscriptSeg]:
    api = YouTubeTranscriptApi()
    if lang:
        items = api.fetch(video_id, languages=[lang])
    else:
        items = api.fetch(video_id)

    segs: List[TranscriptSeg] = []
    for it in items:
        text = (getattr(it, "text", "") or "").replace("\n", " ").strip()
        start = float(getattr(it, "start", 0.0) or 0.0)
        duration = float(getattr(it, "duration", 0.0) or 0.0)
        if text:
            segs.append(TranscriptSeg(start=start, duration=duration, text=text))
    return segs


def segments_to_timestamped_text(segs: List[TranscriptSeg]) -> str:
    return "\n".join([f"[{format_mmss(s.start)}] {s.text}" for s in segs])


def fetch_video_metadata(video_id: str, max_comments: int = 30) -> Dict[str, Any]:
    url = f"https://www.youtube.com/watch?v={video_id}"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "getcomments": True,
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
            "text": text.replace("\n", " ").strip(),
            "like_count": c.get("like_count"),
        })

    return {
        "webpage_url": info.get("webpage_url") or url,
        "title": info.get("title"),
        "description": info.get("description"),
        "channel": info.get("channel") or info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "duration_seconds": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "tags": info.get("tags") or [],
        "language": info.get("language"),
        "top_comments": comments,
    }


def format_metadata_block(meta: Dict[str, Any]) -> str:
    def fmt_int(x):
        return f"{x:,}" if isinstance(x, int) else (str(x) if x is not None else "")

    def fmt_date(yyyymmdd: str | None) -> str:
        if not yyyymmdd or len(yyyymmdd) != 8:
            return yyyymmdd or ""
        return f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

    duration = meta.get("duration_seconds")
    duration_str = format_mmss(duration) if duration is not None else ""

    lines = [
        "=== VIDEO METADATA ===",
        f"URL: {meta.get('webpage_url', '')}",
        f"Title: {meta.get('title', '')}",
        f"Channel: {meta.get('channel', '')}",
        f"Upload date: {fmt_date(meta.get('upload_date'))}",
        f"Duration: {duration_str} ({meta.get('duration_seconds') or ''} seconds)",
        f"Views: {fmt_int(meta.get('view_count'))}",
        f"Likes: {fmt_int(meta.get('like_count'))}",
        f"Comments: {fmt_int(meta.get('comment_count'))}",
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
            text = (c.get("text") or "").strip()
            lines.append(f"{i}. {author} ({likes_str} likes): {text}")

    lines += ["", "=== TRANSCRIPT (TIMESTAMPED) ==="]
    return "\n".join(lines).rstrip() + "\n"


def download_youtube(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(title).200s.%(ext)s")

    ydl_opts = {
        "quiet": False,
        "no_warnings": False,
        "outtmpl": out_template,
        "format": "bv*+ba/best",
        "merge_output_format": "mp4",
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)

        # 1) Most reliable: yt-dlp gives you the final filename
        final = info.get("_filename")
        if final and Path(final).exists():
            p = Path(final)
            # sometimes _filename is pre-merge; ensure mp4 exists if merge_output_format used
            if p.suffix.lower() == ".mp4" and p.exists():
                return p

        # 2) Sometimes yt-dlp uses requested_downloads list
        rds = info.get("requested_downloads") or []
        for d in rds:
            fn = d.get("filepath") or d.get("_filename")
            if fn and Path(fn).exists() and Path(fn).suffix.lower() == ".mp4":
                return Path(fn)

        # 3) Fallback: look for the *title-based* mp4 in out_dir (still better than newest)
        title = (info.get("title") or "").strip()
        if title:
            # match by prefix
            candidates = sorted(out_dir.glob(f"{title[:180]}*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0]

    raise FileNotFoundError("Download finished but could not locate the merged .mp4 output.")


def _ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    try:
        return float(p.stdout.strip())
    except Exception:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {p.stdout!r}")


def cut_clip(video_path: Path, start: float, end: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dur = _ffprobe_duration(video_path)

    # clamp & validate
    start = max(0.0, float(start))
    end = float(end)

    if end <= 0:
        raise ValueError("End time must be > 0.")
    if start >= dur:
        raise ValueError(f"Start time ({start:.2f}) is >= video duration ({dur:.2f}).")
    if end > dur:
        end = dur
    if end <= start:
        raise ValueError(f"End time ({end:.2f}) must be > start time ({start:.2f}).")

    clip_len = end - start

    # Reliable: -ss AFTER -i, use -t duration
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", f"{start:.3f}",
        "-t", f"{clip_len:.3f}",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)

    # sanity check
    if not out_path.exists() or out_path.stat().st_size < 50_000:
        raise RuntimeError(
            f"Clip output looks empty/invalid: {out_path}\n"
            f"Requested start={start:.2f}s end={end:.2f}s (len={clip_len:.2f}s), video dur={dur:.2f}s"
        )

    return out_path


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpusLite UI (Transcript → Pick Timestamps → Clip + 9:16 + Captions)")
        self.geometry("1100x820")
        self.minsize(900, 600)

        self.url_var = tk.StringVar()
        self.lang_var = tk.StringVar(value="en")
        self.start_var = tk.StringVar(value="00:30")
        self.end_var = tk.StringVar(value="01:15")

        # NEW: checkbox
        self.autocrop_var = tk.BooleanVar(value=True)

        self.status = tk.StringVar(value="Ready.")
        self.meta: Optional[Dict[str, Any]] = None
        self.segs: Optional[List[TranscriptSeg]] = None
        self.video_id: Optional[str] = None

        self._build_ui()

    def _build_ui(self):
        pad = 10

        top = ttk.Frame(self)
        top.pack(fill="x", padx=pad, pady=(pad, 0))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="YouTube URL or Video ID:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.url_var).grid(row=0, column=1, sticky="ew", padx=(8, 0))

        ttk.Label(top, text="Language (optional, e.g. en). Clear to auto:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.lang_var, width=10).grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(8, 0))

        btns = ttk.Frame(top)
        btns.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.fetch_btn = ttk.Button(btns, text="1) Fetch transcript + comments", command=self.on_fetch)
        self.fetch_btn.pack(side="left")

        self.copy_btn = ttk.Button(btns, text="Copy output for ChatGPT", command=self.on_copy, state="disabled")
        self.copy_btn.pack(side="left", padx=(10, 0))

        self.save_btn = ttk.Button(btns, text="Save As .txt", command=self.on_save, state="disabled")
        self.save_btn.pack(side="left", padx=(10, 0))

        clipbar = ttk.Frame(self)
        clipbar.pack(fill="x", padx=pad, pady=(10, 0))

        ttk.Label(clipbar, text="Clip start (ss / mm:ss / hh:mm:ss):").pack(side="left")
        ttk.Entry(clipbar, textvariable=self.start_var, width=12).pack(side="left", padx=(8, 0))

        ttk.Label(clipbar, text="Clip end:").pack(side="left", padx=(14, 0))
        ttk.Entry(clipbar, textvariable=self.end_var, width=12).pack(side="left", padx=(8, 0))

        # NEW: checkbox for auto crop
        ttk.Checkbutton(clipbar, text="Auto-crop to 9:16 (largest face)", variable=self.autocrop_var).pack(side="left", padx=(16, 0))

        self.process_btn = ttk.Button(clipbar, text="2) Clip + 9:16 + Caption", command=self.on_process, state="disabled")
        self.process_btn.pack(side="left", padx=(14, 0))

        ttk.Label(self, textvariable=self.status, anchor="w").pack(fill="x", padx=pad, pady=(8, 0))

        mid = ttk.Frame(self)
        mid.pack(fill="both", expand=True, padx=pad, pady=pad)

        self.text = tk.Text(mid, wrap="word")
        self.text.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(mid, command=self.text.yview)
        scroll.pack(side="right", fill="y")
        self.text.configure(yscrollcommand=scroll.set)

    def set_busy(self, busy: bool):
        self.fetch_btn.configure(state=("disabled" if busy else "normal"))
        self.copy_btn.configure(state=("disabled" if busy else ("normal" if self._has_content() else "disabled")))
        self.save_btn.configure(state=("disabled" if busy else ("normal" if self._has_content() else "disabled")))
        self.process_btn.configure(state=("disabled" if busy else ("normal" if self.video_id else "disabled")))

    def _has_content(self) -> bool:
        return bool(self.text.get("1.0", "end-1c").strip())

    def on_copy(self):
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        self.status.set("Copied. Paste into ChatGPT and ask for best timestamps.")

    def on_save(self):
        content = self.text.get("1.0", "end-1c")
        if not content.strip():
            messagebox.showinfo("Nothing to save", "Output is empty.")
            return

        initial = "transcript.txt"
        if self.meta and self.meta.get("title"):
            safe = re.sub(r"[\\/:*?\"<>|]+", "", self.meta["title"]).strip()
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

        Path(path).write_text(content + "\n", encoding="utf-8")
        self.status.set(f"Saved: {path}")

    def on_fetch(self):
        url_or_id = self.url_var.get().strip()
        lang = self.lang_var.get().strip() or None

        try:
            vid = extract_video_id(url_or_id)
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self.set_busy(True)
        self.status.set("Fetching metadata, comments, and transcript…")
        self.video_id = None
        self.meta = None
        self.segs = None

        def worker():
            try:
                meta = fetch_video_metadata(vid)
                segs = fetch_transcript_segments(vid, lang=lang)
                out = format_metadata_block(meta) + segments_to_timestamped_text(segs) + "\n"
                self._ui_success_fetch(out, vid, meta, segs)
            except Exception as e:
                self._ui_error(e)

        threading.Thread(target=worker, daemon=True).start()

    def _ui_success_fetch(self, output: str, vid: str, meta: Dict[str, Any], segs: List[TranscriptSeg]):
        def run():
            self.video_id = vid
            self.meta = meta
            self.segs = segs
            self.text.delete("1.0", "end")
            self.text.insert("1.0", output)
            self.status.set("Fetched. Paste into ChatGPT to choose timestamps, then click Clip + 9:16 + Caption.")
            self.set_busy(False)
            self.copy_btn.configure(state="normal")
            self.save_btn.configure(state="normal")
            self.process_btn.configure(state="normal")
        self.after(0, run)

    def _ui_error(self, err: Exception):
        def run():
            self.status.set("Error.")
            self.set_busy(False)
            messagebox.showerror("Failed", str(err))
        self.after(0, run)

    def on_process(self):
        if not self.video_id:
            messagebox.showerror("Missing data", "Fetch transcript/comments first.")
            return

        try:
            clip_start = parse_timestamp(self.start_var.get())
            clip_end = parse_timestamp(self.end_var.get())
            if clip_end <= clip_start:
                raise ValueError("Clip end must be > clip start.")
        except Exception as e:
            messagebox.showerror("Bad timestamps", str(e))
            return

        url = f"https://www.youtube.com/watch?v={self.video_id}"
        do_autocrop = bool(self.autocrop_var.get())

        self.set_busy(True)
        self.status.set("Downloading, clipping, (optional) auto-cropping to 9:16, generating captions, and burning them in…")

        def worker():
            try:
                work = Path("workspace_ui")
                dl_dir = work / "downloads"
                out_dir = work / "outputs"
                tmp_dir = work / "tmp"
                dl_dir.mkdir(parents=True, exist_ok=True)
                out_dir.mkdir(parents=True, exist_ok=True)
                tmp_dir.mkdir(parents=True, exist_ok=True)

                video_path = download_youtube(url, dl_dir)

                raw_clip = out_dir / f"clip_{int(clip_start)}_{int(clip_end)}_raw.mp4"
                cut_clip(video_path, clip_start, clip_end, raw_clip)

                # Decide which video we caption
                caption_base = raw_clip
                vertical_clip = None

                if do_autocrop:
                    vertical_clip = out_dir / f"clip_{int(clip_start)}_{int(clip_end)}_vertical.mp4"
                    cfg = AutoCropConfig(
                        sample_fps=3.0,
                        smoothing=0.85,
                        deadzone_px=40,
                        hold_no_face_sec=2.0,
                    )
                    autocrop_to_9x16_face(raw_clip, vertical_clip, cfg)
                    caption_base = vertical_clip

                # Captions on the final-framed video (raw or vertical)
                ass_path = tmp_dir / "clip.ass"
                make_ass_from_clip_whisper(
                    caption_base,
                    ass_path,
                    model_size="small",
                    max_words_per_line=5,
                    fontname="Impact",
                    fontsize=64,
                )

                final_clip = out_dir / f"clip_{int(clip_start)}_{int(clip_end)}_{'vertical_' if do_autocrop else ''}captioned.mp4"
                burn_in_ass(caption_base, ass_path, final_clip)

                self._ui_success_process(final_clip, vertical_clip)
            except Exception as e:
                self._ui_error(e)

        threading.Thread(target=worker, daemon=True).start()

    def _ui_success_process(self, final_clip: Path, vertical_clip: Optional[Path]):
        def run():
            self.set_busy(False)
            extras = f"\n\nVertical:\n{vertical_clip}" if vertical_clip else ""
            msg = f"Done.\n\nSaved:\n{final_clip}{extras}\n\nCaptions: Whisper word-timestamps (clip-only) → ASS (1 line, max 5 words)"
            self.status.set("Done.")
            messagebox.showinfo("Success", msg)
        self.after(0, run)


if __name__ == "__main__":
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    App().mainloop()