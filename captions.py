from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import whisper


# -----------------------------
# Time + probing helpers
# -----------------------------
def _sec_to_ass_time(t: float) -> str:
    """
    ASS time format: H:MM:SS.cs (centiseconds)
    """
    t = max(0.0, float(t))
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = int(t % 60)
    cs = int(round((t - int(t)) * 100.0))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _ffprobe_wh(video_path: Path) -> Tuple[int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(video_path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    w, h = p.stdout.strip().split("x")
    return int(w), int(h)


def _escape_ass_text(s: str) -> str:
    # Minimal ASS escaping
    s = s.replace("\n", " ").strip()
    s = s.replace("{", r"\{").replace("}", r"\}")
    return s


# -----------------------------
# Style
# -----------------------------
@dataclass
class CaptionStyle:
    fontname: str = "Impact"
    fontsize: int = 64
    outline: int = 6
    shadow: int = 0
    bold: int = 1
    alignment: int = 2  # bottom-center
    margin_l: int = 60
    margin_r: int = 60
    margin_v: int = 420  # raised captions (lower-third-ish)


# -----------------------------
# Event building
# -----------------------------
def _build_events_from_word_items(
    word_items: List[Tuple[float, float, str]],
    max_words_per_line: int,
    max_hold_sec: float,
    min_hold_sec: float,
) -> List[Tuple[float, float, str]]:
    """
    word_items: (start, end, word)
    Returns events: (start, end, "w1 w2 w3 ...") with:
      - one line at a time
      - <= max_words_per_line
      - no frozen captions (cap long holds)
      - no blinking (min duration + fill small gaps)
    """
    events: List[Tuple[float, float, str]] = []

    i = 0
    n = len(word_items)
    while i < n:
        group = word_items[i:i + max_words_per_line]
        start = float(group[0][0])
        end = float(group[-1][1])

        # normalize
        if end <= start:
            end = start + min_hold_sec

        # cap very long holds (prevents freezes)
        if (end - start) > max_hold_sec:
            end = start + max_hold_sec

        # ensure minimum readability (prevents blink)
        if (end - start) < min_hold_sec:
            end = start + min_hold_sec

        text = " ".join([w for _, _, w in group]).strip()
        text = re.sub(r"\s+", " ", text)

        events.append((start, end, text))
        i += max_words_per_line

    # Fix overlaps and fill small gaps so text doesn't disappear too early
    for j in range(len(events) - 1):
        s1, e1, t1 = events[j]
        s2, e2, t2 = events[j + 1]

        # Avoid overlap (one line at a time)
        if e1 >= s2:
            events[j] = (s1, max(s1 + 0.05, s2 - 0.02), t1)
            continue

        # If there's a gap, extend current caption into the gap (up to max_hold_sec)
        gap = s2 - e1
        if gap > 0.15:
            new_end = min(s2 - 0.02, s1 + max_hold_sec)
            if new_end > e1:
                events[j] = (s1, new_end, t1)

    return events


def _looks_bad_word_timing(word_items: List[Tuple[float, float, str]]) -> bool:
    """
    Heuristic: if word timing has big gaps or too few words, it tends to freeze or skip.
    """
    if len(word_items) < 10:
        return True

    gaps = 0
    for i in range(1, len(word_items)):
        prev_end = word_items[i - 1][1]
        cur_start = word_items[i][0]
        if (cur_start - prev_end) > 1.2:
            gaps += 1

    return gaps >= 2


# -----------------------------
# Whisper -> ASS
# -----------------------------
def make_ass_from_clip_whisper(
    video_path: Path,
    ass_path: Path,
    model_size: str = "small",
    max_words_per_line: int = 5,
    fontname: str = "Impact",
    fontsize: int = 64,
    margin_v_ratio: float = 0.38,
    max_hold_sec: float = 1.40,   # longer so captions stay readable
    min_hold_sec: float = 0.35,   # prevents blinking
) -> Path:
    """
    Generates ASS captions:
      - ONE line at a time
      - <= 5 words per line (default)
      - Impact font, outlined
      - positioned around lower third (margin_v_ratio)
      - robust timing: uses Whisper word timestamps, falls back to synthetic timing if needed
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    w, h = _ffprobe_wh(video_path)

    style = CaptionStyle(
        fontname=fontname,
        fontsize=fontsize,
        margin_v=int(h * margin_v_ratio),
    )

    model = whisper.load_model(model_size)

    result = model.transcribe(
        str(video_path),
        fp16=False,
        word_timestamps=True,
        verbose=False,
    )

    segments = result.get("segments", []) or []

    # 1) Try real word timestamps
    word_items: List[Tuple[float, float, str]] = []
    for seg in segments:
        seg_start = float(seg.get("start", 0.0))
        seg_end = float(seg.get("end", seg_start + 1.0))
        for wobj in (seg.get("words") or []):
            text = (wobj.get("word") or "").strip()
            if not text:
                continue
            start = float(wobj.get("start", seg_start))
            end = float(wobj.get("end", seg_end))
            if end <= start:
                end = start + 0.12
            word_items.append((start, end, text))

    # 2) If word timing is bad, fallback to synthetic timing within each segment
    if not word_items or _looks_bad_word_timing(word_items):
        word_items = []
        for seg in segments:
            seg_text = (seg.get("text") or "").strip()
            if not seg_text:
                continue
            seg_start = float(seg.get("start", 0.0))
            seg_end = float(seg.get("end", seg_start + 1.0))
            if seg_end <= seg_start:
                seg_end = seg_start + 1.0

            words = re.findall(r"\S+", seg_text)
            if not words:
                continue

            dur = seg_end - seg_start
            step = dur / max(1, len(words))
            for idx, word in enumerate(words):
                s = seg_start + idx * step
                e = min(seg_end, s + step)
                if e <= s:
                    e = s + 0.12
                word_items.append((s, e, word))

    if not word_items:
        raise RuntimeError("Could not build captions from Whisper output.")

    events = _build_events_from_word_items(
        word_items=word_items,
        max_words_per_line=max_words_per_line,
        max_hold_sec=max_hold_sec,
        min_hold_sec=min_hold_sec,
    )

    ass_path.parent.mkdir(parents=True, exist_ok=True)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{style.fontname},{style.fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,{style.bold},0,0,0,100,100,0,0,1,{style.outline},{style.shadow},{style.alignment},{style.margin_l},{style.margin_r},{style.margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    out_lines = [header]
    for (start, end, text) in events:
        s_ass = _sec_to_ass_time(start)
        e_ass = _sec_to_ass_time(end)
        safe = _escape_ass_text(text)
        out_lines.append(f"Dialogue: 0,{s_ass},{e_ass},Default,,0,0,0,,{safe}\n")

    ass_path.write_text("".join(out_lines), encoding="utf-8")
    return ass_path


# -----------------------------
# Burn ASS into video
# -----------------------------
def _escape_for_ffmpeg(path: Path) -> str:
    """
    FFmpeg filter args treat ':' as a separator; Windows drive 'C:' must be escaped as '\:'.
    """
    p = path.resolve().as_posix()
    p = p.replace(":", "\\:")
    p = p.replace("'", "\\'")
    return p


def burn_in_ass(video_path: Path, ass_path: Path, out_path: Path) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not ass_path.exists():
        raise FileNotFoundError(f"ASS not found: {ass_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ass_escaped = _escape_for_ffmpeg(ass_path)
    vf = f"ass='{ass_escaped}'"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path