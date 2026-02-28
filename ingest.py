from __future__ import annotations
import subprocess
from pathlib import Path

def download_youtube(url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(out_dir / "%(title).200s.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f", "bv*+ba/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        url,
    ]
    subprocess.run(cmd, check=True)

    # Find newest mp4 in out_dir
    mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not mp4s:
        raise FileNotFoundError("Download finished but no .mp4 found.")
    return mp4s[0]