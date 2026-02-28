from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple

import cv2


@dataclass
class AutoCropConfig:
    # detection + smoothing
    sample_fps: float = 6.0
    smoothing: float = 0.92
    deadzone_px: int = 80
    hold_no_face_sec: float = 2.0
    drift_to_center_after_hold: bool = True
    max_pan_speed_px_per_sec: float = 450.0

    # NEW: speaker lock (prevents ping-pong between faces)
    switch_ratio: float = 1.35          # new face must be 35% larger to switch
    switch_confirm_samples: int = 3     # must win for N sampled detections before switching


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _detect_faces(gray, face_cascade) -> List[Tuple[int, int, int, int, float, float]]:
    """
    Returns list of faces as:
      (x, y, w, h, area, cx)
    """
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    out = []
    for (x, y, w, h) in faces:
        area = float(w * h)
        cx = float(x + w / 2.0)
        out.append((x, y, w, h, area, cx))
    return out


def autocrop_to_9x16_face(in_video: Path, out_video: Path, cfg: AutoCropConfig) -> Path:
    if not in_video.exists():
        raise FileNotFoundError(f"Input video not found: {in_video}")

    out_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_silent = out_video.with_name(out_video.stem + "_silent.mp4")

    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Could not read video dimensions.")

    crop_h = h
    crop_w = int(round(h * 9 / 16))
    crop_w = min(crop_w, w)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_silent), fourcc, fps, (crop_w, crop_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open VideoWriter.")

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        cap.release()
        writer.release()
        raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

    sample_every = max(1, int(round(fps / max(0.1, cfg.sample_fps))))
    hold_frames = int(round(cfg.hold_no_face_sec * fps))

    # Smooth crop center
    smooth_cx: Optional[float] = None
    last_good_cx: Optional[float] = None
    no_face_count = 0
    center_cx = w / 2.0

    # Speaker lock state
    locked_cx: Optional[float] = None
    locked_area: Optional[float] = None
    pending_wins = 0
    pending_cx: Optional[float] = None
    pending_area: Optional[float] = None

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detected_cx = None
        detected_area = None

        if (frame_idx % sample_every) == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = _detect_faces(gray, face_cascade)

            if faces:
                # sort by area desc
                faces.sort(key=lambda t: t[4], reverse=True)
                top = faces[0]
                top_area, top_cx = top[4], top[5]

                if locked_cx is None:
                    # first lock
                    locked_cx, locked_area = top_cx, top_area
                    pending_wins = 0
                else:
                    # decide whether to switch lock (hysteresis)
                    assert locked_area is not None
                    if top_area >= locked_area * cfg.switch_ratio:
                        # candidate for switching
                        if pending_cx is None or abs(top_cx - pending_cx) < 1e-6:
                            pending_wins += 1
                            pending_cx, pending_area = top_cx, top_area
                        else:
                            pending_wins = 1
                            pending_cx, pending_area = top_cx, top_area

                        if pending_wins >= cfg.switch_confirm_samples:
                            locked_cx, locked_area = pending_cx, pending_area
                            pending_wins = 0
                            pending_cx, pending_area = None, None
                    else:
                        # keep lock
                        pending_wins = 0
                        pending_cx, pending_area = None, None

                detected_cx = locked_cx
                detected_area = locked_area

        # Use lock output if available; otherwise hold/drift logic
        if detected_cx is not None:
            no_face_count = 0
            target_cx = detected_cx
            last_good_cx = target_cx
        else:
            no_face_count += 1
            if last_good_cx is not None and no_face_count <= hold_frames:
                target_cx = last_good_cx
            else:
                target_cx = center_cx if cfg.drift_to_center_after_hold else (last_good_cx or center_cx)

        if smooth_cx is None:
            smooth_cx = target_cx

        # deadzone reduces jitter
        if abs(target_cx - smooth_cx) < cfg.deadzone_px:
            target_cx = smooth_cx

        prev_smooth_cx = smooth_cx
        smooth_cx = cfg.smoothing * smooth_cx + (1.0 - cfg.smoothing) * target_cx

        # cap movement per frame
        max_step = cfg.max_pan_speed_px_per_sec / fps
        delta = smooth_cx - prev_smooth_cx
        if abs(delta) > max_step:
            smooth_cx = prev_smooth_cx + (max_step if delta > 0 else -max_step)

        x0 = int(round(smooth_cx - crop_w / 2.0))
        x0 = int(_clamp(x0, 0, w - crop_w))

        crop = frame[:, x0:x0 + crop_w]
        if crop.shape[1] != crop_w:
            crop = cv2.resize(crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        writer.write(crop)
        frame_idx += 1

    cap.release()
    writer.release()

    # Reattach audio
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_silent),
        "-i", str(in_video),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "veryfast",
        "-c:a", "aac",
        "-b:a", "192k",
        str(out_video),
    ]
    subprocess.run(cmd, check=True)

    try:
        tmp_silent.unlink(missing_ok=True)
    except Exception:
        pass

    return out_video