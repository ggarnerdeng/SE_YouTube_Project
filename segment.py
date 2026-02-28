from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class SegmentPick:
    start: float
    end: float
    score: float
    text: str

def pick_segments_by_word_density(result: dict, window_s: int = 45, hop_s: int = 5, top_k: int = 3) -> List[SegmentPick]:
    # Flatten into (time, text) using whisper segments
    segs = result["segments"]

    # Build a timeline of segment text and times
    # We'll slide a window and score by total words / window length.
    max_t = segs[-1]["end"] if segs else 0.0
    picks: List[SegmentPick] = []

    t = 0.0
    while t + window_s <= max_t:
        w_start, w_end = t, t + window_s
        chunk_text = []
        word_count = 0

        for s in segs:
            if s["end"] <= w_start:
                continue
            if s["start"] >= w_end:
                break
            txt = s["text"].strip()
            if txt:
                chunk_text.append(txt)
                word_count += len(txt.split())

        density = word_count / max(window_s, 1)
        # Mild bonus if the first 5 seconds contains a question mark (cheap "hook" proxy)
        first_five = " ".join(chunk_text)[:200]
        hook_bonus = 0.15 if "?" in first_five else 0.0

        score = density + hook_bonus
        picks.append(SegmentPick(start=w_start, end=w_end, score=score, text=" ".join(chunk_text)))

        t += hop_s

    picks.sort(key=lambda p: p.score, reverse=True)
    return picks[:top_k]