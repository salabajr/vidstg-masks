"""QA overlay: one H.264 mp4 per clip with per-tid tinted masks, VidOR boxes (thick =
human keyframe, thin = tracker box), a label at each box (`<tid>:<category>`, with
"(no mask)" on a frame where the object has a box but no mask) and a banner (relations,
per-tid legend). CPU only; frames go to ffmpeg through a pipe. Output stays outside any
release artifact."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .datasets import Roots, build_vidor_index, load_vidor, resolve_video
from .records import read_jsonl, rle_decode

# BGR per-tid colours, cycled.
PALETTE = [(90, 200, 90), (90, 90, 230), (230, 170, 60), (200, 90, 200),
           (60, 220, 220), (140, 100, 230), (120, 220, 140), (250, 160, 100)]
WHITE = (255, 255, 255)


def _banner_lines(vid: str, relations, cats: dict, colors: dict, W: int) -> list[list[tuple]]:
    import cv2
    width = lambda s: cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]

    def wrap(text: str) -> list[str]:
        lines, cur = [], ""
        for w in text.split():
            cand = (cur + " " + w).strip()
            if cur and width(cand) > W - 16:
                lines.append(cur)
                cur = w
            else:
                cur = cand
        return lines + ([cur] if cur else [])

    head = vid + (" | " + " | ".join(f"{s} {p} {o}" for s, p, o in relations) if relations else "")
    out = [[(ln, WHITE)] for ln in wrap(head)]
    out.append([(f"{t}:{str(cats.get(t, '?')).split('/')[0]}", colors[t]) for t in sorted(colors)])
    return out


def _label(frame, text: str, x: int, y: int, color, avoid_top: int) -> None:
    """`text` in `color` on a darkened strip whose bottom-left corner is (x, y), moved below
    the corner when the strip would cover the banner or leave the frame."""
    import cv2
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    H, W = frame.shape[:2]
    x = max(0, min(x, W - tw - 4))
    top = y - th - base - 4
    if top < avoid_top:
        top = min(y + 1, H - th - base - 4)
    top = max(0, top)
    strip = frame[top:top + th + base + 4, x:x + tw + 4]
    strip[:] = (strip * 0.3).astype(np.uint8)
    cv2.putText(frame, text, (x + 2, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def render_overlay(vid: str, records_path: Path, roots: Roots, out_mp4: Path,
                   relations=None, alpha: float = 0.45, crf: int = 20, labels: bool = True,
                   side_by_side: bool = False) -> int:
    """Decode the clip's video sequentially, tint each frame with that frame's masks, draw
    each object's box and label, and encode with libx264 (odd sizes padded by one pixel).
    With `side_by_side` every output frame is the untouched frame on the left and the
    painted one on the right, each captioned, for a review page. Returns frames written."""
    import cv2

    ann = load_vidor(build_vidor_index(roots)[vid])
    video = resolve_video(roots, ann)
    if video is None:
        raise FileNotFoundError(f"no video for {vid} ({ann['video_path']})")
    W, H = ann["width"], ann["height"]
    masks: dict[tuple[int, int], dict] = {}
    tids: set[int] = set()
    for r in read_jsonl(records_path):
        tids.add(r["tid"])
        if r.get("rle"):
            masks[(r["fid"], r["tid"])] = r["rle"]
    tids_sorted = sorted(tids)
    colors = {t: PALETTE[k % len(PALETTE)] for k, t in enumerate(tids_sorted)}
    cats = {o["tid"]: o["category"] for o in ann["subject/objects"]}
    banner = _banner_lines(vid, relations or [], cats, colors, W)
    banner_h = 8 + 18 * len(banner)
    boxes = {t: {} for t in tids_sorted}
    for fid, frame in enumerate(ann["trajectories"]):
        for b in frame:
            if b["tid"] in boxes:
                boxes[b["tid"]][fid] = b

    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mp4.with_suffix(out_mp4.suffix + ".part")
    out_w = 2 * W if side_by_side else W
    ff = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{out_w}x{H}", "-r", f"{ann['fps']:.6f}", "-i", "-", "-vsync", "0",
         "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", str(crf), "-f", "mp4", str(tmp)], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(video))
    fid = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fid += 1
            if frame.shape[:2] != (H, W):
                frame = cv2.resize(frame, (W, H))  # display only; records are never touched
            original = frame.copy() if side_by_side else None
            for t in tids_sorted:
                rle = masks.get((fid, t))
                if rle:
                    m = rle_decode(rle)
                    if m.shape == (H, W):
                        frame[m] = ((1 - alpha) * frame[m] + alpha * np.array(colors[t])).astype(np.uint8)
                box = boxes[t].get(fid)
                if box is not None:
                    b = box["bbox"]
                    cv2.rectangle(frame, (b["xmin"], b["ymin"]), (b["xmax"], b["ymax"]), colors[t],
                                  2 if box.get("generated", 0) == 0 else 1)
            frame[:banner_h] = (frame[:banner_h] * 0.35).astype(np.uint8)
            if labels:
                for t in tids_sorted:
                    box = boxes[t].get(fid)
                    if box is None:
                        continue
                    b = box["bbox"]
                    text = f"{t}:{str(cats.get(t, '?')).split('/')[0]}"
                    if (fid, t) not in masks:
                        text += " (no mask)"
                    _label(frame, text, b["xmin"], b["ymin"], colors[t], banner_h)
            y = 16
            for line in banner:
                x = 8
                for text, color in line:
                    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
                    x += cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] + 12
                y += 18
            if side_by_side:
                _label(original, "original", 0, H - 2, WHITE, 0)
                _label(frame, "SAM 3.1 masks", 0, H - 2, WHITE, 0)
                frame = np.hstack([original, frame])
            ff.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        cap.release()
        ff.stdin.close()
        rc = ff.wait()
    if rc != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg exited {rc} while encoding {out_mp4}")
    tmp.replace(out_mp4)
    return fid + 1
