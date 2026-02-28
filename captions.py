from __future__ import annotations

import subprocess
from pathlib import Path


def _escape_for_ffmpeg_subtitles(path: Path) -> str:
    """
    FFmpeg subtitles filter uses ':' as an option separator, which breaks Windows drive paths (C:).
    We fix this by:
      - converting to absolute POSIX path (forward slashes)
      - escaping ':' as '\:'
      - escaping '\'' if present (rare)
    """
    p = path.resolve().as_posix()

    # Escape drive letter colon: C:/... -> C\:/...
    # Also safe if the path contains other colons (rare but possible)
    p = p.replace(":", r"\:")

    # If you ever had single quotes in a path (rare), escape them
    p = p.replace("'", r"\'")

    return p


def burn_in_srt(video_path: Path, srt_path: Path, out_path: Path) -> Path:
    """
    Burn SRT captions into a video using ffmpeg.

    Notes:
    - Applying a video filter requires re-encoding the video stream.
    - Audio can be copied.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT not found: {srt_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    srt_escaped = _escape_for_ffmpeg_subtitles(srt_path)

    # IMPORTANT: quote the filename inside the filter expression
    vf = f"subtitles=filename='{srt_escaped}'"

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