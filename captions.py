from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple

import whisper


# -----------------------------
# ASS utilities
# -----------------------------
def _ass_time(t: float) -> str:
    """
    ASS time format: H:MM:SS.cc (centiseconds)
    """
    if t < 0:
        t = 0.0
    cs = int(round(t * 100))  # centiseconds
    h = cs // (3600 * 100)
    cs -= h * 3600 * 100
    m = cs // (60 * 100)
    cs -= m * 60 * 100
    s = cs // 100
    cs -= s * 100
    return f"{h}:{m:02}:{s:02}.{cs:02}"


def _escape_ass_text(text: str) -> str:
    # ASS escapes: \N = newline. We do single-line only.
    # Also escape braces which are control codes in ASS.
    return (
        text.replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\n", " ")
            .strip()
    )


def _ass_header(fontname: str = "Impact", fontsize: int = 64) -> str:
    """
    Impact-like subtitle style with outline.
    (Color not important per your request; we keep a simple white fill + black outline)
    """
    return f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,1,0,0,0,100,100,0,0,1,4,0,2,80,80,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


# -----------------------------
# Word-timed caption generation
# -----------------------------
def _extract_words_from_whisper(result: Dict[str, Any]) -> List[Tuple[float, float, str]]:
    """
    Returns list of (start, end, word) from whisper word_timestamps output.
    """
    words: List[Tuple[float, float, str]] = []
    for seg in result.get("segments", []):
        for w in seg.get("words", []) or []:
            wt = (w.get("word") or "").strip()
            if not wt:
                continue
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))
            words.append((ws, we, wt))
    return words


def make_ass_from_clip_whisper(
    clip_path: Path,
    ass_path: Path,
    model_size: str = "small",
    max_words_per_line: int = 5,
    min_event_dur: float = 0.18,
    gap: float = 0.02,
    fontname: str = "Impact",
    fontsize: int = 64,
) -> None:
    """
    Creates an .ass subtitle file for *the clip only* using Whisper word timestamps.
    - One line at a time
    - Max 5 words per line
    - Non-overlapping events (prevents "old text lingering")
    """

    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")

    model = whisper.load_model(model_size)
    result = model.transcribe(
        str(clip_path),
        fp16=False,
        word_timestamps=True,
    )

    words = _extract_words_from_whisper(result)
    if not words:
        raise RuntimeError("No word timestamps produced by Whisper. Try a larger model (medium) or check audio quality.")

    # Build caption events by grouping words into chunks (<= max_words_per_line)
    events: List[Tuple[float, float, str]] = []

    i = 0
    while i < len(words):
        chunk = words[i:i + max_words_per_line]
        start = chunk[0][0]
        end = chunk[-1][1]

        # Guard against zero/negative durations
        if end <= start:
            end = start + min_event_dur

        # Ensure minimum duration for readability
        if (end - start) < min_event_dur:
            end = start + min_event_dur

        text = " ".join(w[2] for w in chunk)
        text = _escape_ass_text(text)

        events.append((start, end, text))
        i += max_words_per_line

    # Enforce strictly non-overlapping events (fixes the “2-line / lingering” behavior)
    fixed: List[Tuple[float, float, str]] = []
    prev_end = 0.0
    for start, end, text in events:
        if start < prev_end:
            start = prev_end  # push forward if overlap
        if end <= start:
            end = start + min_event_dur
        # Add a tiny gap so ffmpeg/libass doesn't “blend” transitions
        end = max(end, start + min_event_dur)
        fixed.append((start, end, text))
        prev_end = end + gap

    ass_path.parent.mkdir(parents=True, exist_ok=True)

    out = [_ass_header(fontname=fontname, fontsize=fontsize)]
    for start, end, text in fixed:
        out.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}\n")

    ass_path.write_text("".join(out), encoding="utf-8")


# -----------------------------
# FFmpeg burn-in (ASS)
# -----------------------------
def _escape_for_ffmpeg_filter(path: Path) -> str:
    """
    FFmpeg filter args treat ':' specially; Windows drive letters need escaping.
    We use absolute posix path and escape ':' -> '\:'.
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

    ass_escaped = _escape_for_ffmpeg_filter(ass_path)

    # Use ass filter (better for styling than subtitles=)
    vf = f"ass='{ass_escaped}'"

    cmd = [
        "ffmpeg",
        "-y",
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