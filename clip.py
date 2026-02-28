from __future__ import annotations
import subprocess
from pathlib import Path

def cut_clip(video_path: Path, start: float, end: float, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start),
        "-to", str(end),
        "-i", str(video_path),
        "-c", "copy",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path