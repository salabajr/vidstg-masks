"""Video probing, decode checks and H.264 transcoding.

Frame indices are native decode order (ffmpeg `-vsync 0`, cv2 sequential reads). Nothing
here resamples or indexes by timestamp. A decoded frame count that disagrees with the
annotation is reported (and the clip refused), never rescaled.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

BLACK_MEAN_THRESHOLD = 2.0  # mean intensity below this on every sampled frame = black decode


def have_ffmpeg() -> dict:
    return {tool: shutil.which(tool) for tool in ("ffmpeg", "ffprobe")}


def probe_frame_count(path: Path) -> int:
    """Decoded frame count via `ffprobe -count_frames` (decodes the stream; no timestamps)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "default=noprint_wrappers=1:nokey=1",
         str(path)], capture_output=True, text=True, check=True).stdout.strip()
    return int(out.splitlines()[0])


def decode_check(path: Path, expected_frames: int, expected_size: tuple[int, int] | None = None,
                 n_black_samples: int = 8) -> dict:
    """One sequential cv2 pass: frame count, decoded (W, H), and a black-decode probe over
    n_black_samples evenly spaced frames. Returns a dict with `frame_count`, `size`,
    `black`, `frame_count_ok`, `size_ok`."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 cannot open {path}")
    sample_every = max(1, expected_frames // max(1, n_black_samples))
    n, size, means = 0, None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if size is None:
            size = (int(frame.shape[1]), int(frame.shape[0]))
        if n % sample_every == 0 and len(means) < n_black_samples:
            means.append(float(frame.mean()))
        n += 1
    cap.release()
    black = bool(means) and max(means) < BLACK_MEAN_THRESHOLD
    return {
        "frame_count": n,
        "size": size,
        "black": black,
        "sampled_means": means,
        "frame_count_ok": n == expected_frames,
        "size_ok": expected_size is None or size == tuple(expected_size),
    }


def is_black(path: Path, n_sample: int = 8) -> bool:
    """True when n_sample evenly spaced decoded frames are all (near-)black — the VP6F
    class of VidOR videos that cv2 decodes as black frames."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max(1, n_sample))
    means, n = [], 0
    while len(means) < n_sample:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            means.append(float(np.asarray(frame).mean()))
        n += 1
    cap.release()
    return bool(means) and max(means) < BLACK_MEAN_THRESHOLD


def transcode_h264(src: Path, dst: Path, crf: int = 18) -> Path:
    """Re-encode to H.264/yuv420p without frame drops or duplicates (`-vsync 0`), padding
    odd dimensions by one pixel so yuv420p is legal. Writes to a .part file and renames."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vsync", "0", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-an", "-f", "mp4", str(tmp)], check=True)
    tmp.replace(dst)
    return dst
