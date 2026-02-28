from __future__ import annotations

import sys
from pathlib import Path

from ingest import download_youtube
from transcribe import transcribe, save_srt_for_clip
from segment import pick_segments_by_word_density
from clip import cut_clip
from captions import burn_in_srt


def run(url: str) -> None:
    work = Path("workspace")
    dl_dir = work / "downloads"
    out_dir = work / "outputs"
    tmp_dir = work / "tmp"

    dl_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Downloading video...")
    video_path = download_youtube(url, dl_dir)
    print(f"[OK] Downloaded: {video_path}")

    print("[INFO] Transcribing...")
    result = transcribe(video_path, model_size="small")

    print("[INFO] Picking segments...")
    picks = pick_segments_by_word_density(result, window_s=45, hop_s=5, top_k=3)
    if not picks:
        raise RuntimeError("No segments were selected. Try a different window size or check transcript output.")

    print(f"[OK] Selected {len(picks)} segment(s).")

    for idx, p in enumerate(picks, start=1):
        print(f"\n[INFO] Processing clip {idx}/{len(picks)}  start={p.start:.2f}s end={p.end:.2f}s score={p.score:.3f}")

        raw_clip = out_dir / f"clip_{idx:02}_raw.mp4"
        cut_clip(video_path, p.start, p.end, raw_clip)

        srt_path = tmp_dir / f"clip_{idx:02}.srt"
        save_srt_for_clip(result, p.start, p.end, srt_path)

        final_clip = out_dir / f"clip_{idx:02}_captioned.mp4"
        burn_in_srt(raw_clip, srt_path, final_clip)

        print(f"[OK] Wrote: {final_clip}")
        if p.text:
            preview = p.text.strip().replace("\n", " ")
            print(f"     Text preview: {preview[:200]}{'...' if len(preview) > 200 else ''}")


def main() -> int:
    if len(sys.argv) < 2:
        print('Usage: python main.py "https://www.youtube.com/watch?v=VIDEO_ID"')
        return 2

    url = sys.argv[1].strip()
    run(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())