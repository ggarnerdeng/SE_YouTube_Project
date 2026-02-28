from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import whisper


# ---------------------------------------------------------
# Load + Transcribe
# ---------------------------------------------------------

def transcribe(video_path: Path, model_size: str = "small") -> Dict[str, Any]:
    """
    Transcribe a video file using OpenAI Whisper.

    Returns:
        Whisper result dict containing:
            - text
            - segments (with start/end timestamps)
            - language
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[INFO] Loading Whisper model: {model_size}")
    model = whisper.load_model(model_size)

    print("[INFO] Transcribing (this may take a few minutes on CPU)...")
    result = model.transcribe(
        str(video_path),
        word_timestamps=True,
        fp16=False  # safer on CPU
    )

    print("[OK] Transcription complete.")
    return result


# ---------------------------------------------------------
# Time Formatting
# ---------------------------------------------------------

def _format_srt_time(seconds: float) -> str:
    """
    Convert seconds (float) into SRT timestamp format:
    HH:MM:SS,mmm
    """
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)

    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


# ---------------------------------------------------------
# Full Video SRT
# ---------------------------------------------------------

def save_full_srt(result: Dict[str, Any], srt_path: Path) -> None:
    """
    Generate an SRT file for the full video timeline.
    """
    segments = result.get("segments", [])
    if not segments:
        raise ValueError("No segments found in transcription result.")

    lines = []
    index = 1

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        lines.append(str(index))
        lines.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        lines.append(text)
        lines.append("")
        index += 1

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Saved full SRT: {srt_path}")


# ---------------------------------------------------------
# Clip-Specific SRT (SHIFTED)
# ---------------------------------------------------------

def save_srt_for_clip(
    result: Dict[str, Any],
    clip_start: float,
    clip_end: float,
    srt_path: Path
) -> None:
    """
    Generate an SRT file for a specific clip.
    Timestamps are shifted so clip starts at 00:00:00.
    """

    segments = result.get("segments", [])
    if not segments:
        raise ValueError("No segments found in transcription result.")

    lines = []
    index = 1

    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)

        # Skip segments outside clip
        if seg_end <= clip_start:
            continue
        if seg_start >= clip_end:
            break

        # Clip boundaries
        start = max(seg_start, clip_start)
        end = min(seg_end, clip_end)

        # Shift timestamps relative to clip start
        shifted_start = start - clip_start
        shifted_end = end - clip_start

        text = seg.get("text", "").strip()
        if not text:
            continue

        lines.append(str(index))
        lines.append(
            f"{_format_srt_time(shifted_start)} --> {_format_srt_time(shifted_end)}"
        )
        lines.append(text)
        lines.append("")
        index += 1

    if index == 1:
        print("[WARNING] No subtitles found inside clip range.")

    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Saved clip SRT: {srt_path}")