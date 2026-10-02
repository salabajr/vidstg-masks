"""Synthetic VidSTG + VidOR fixtures (shapes copied from the real files) and a tiny mp4.

Videos:
  A 1000000001  train  tids 0 (adult), 1 (ball), 2 (dog); 2 relations; video on disk, 12 frames
  B 1000000002  val    tids 0, 1; 1 relation; video on disk
  C 1000000003  test   17 relation tids -> too_many_objects; no video
  D 1000000004  val    tids 0, 1; video missing -> video_missing
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

W, H, N_FRAMES, FPS = 64, 48, 12, 10.0
VID_A, VID_B, VID_C, VID_D = "1000000001", "1000000002", "1000000003", "1000000004"


def box(tid, x0, y0, x1, y1, generated, tracker="none"):
    return {"tid": tid, "bbox": {"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1},
            "generated": generated, "tracker": tracker}


def vidor_ann(vid, folder, objects, trajectories, frame_count=N_FRAMES, relations=None):
    return {"version": "VERSION 1.0", "video_id": vid, "video_hash": "0" * 31,
            "video_path": f"{folder}/{vid}.mp4", "frame_count": frame_count, "fps": FPS,
            "width": W, "height": H, "subject/objects": objects,
            "trajectories": trajectories, "relation_instances": relations or []}


def vidstg_record(vid, s, pred, o, b, e, objects, split_frames=N_FRAMES):
    return {"vid": vid, "fps": FPS, "frame_count": split_frames,
            "used_segment": {"begin_fid": b, "end_fid": e}, "width": W, "height": H,
            "subject/objects": objects,
            "used_relation": {"subject_tid": s, "object_tid": o, "predicate": pred, "begin_fid": b, "end_fid": e},
            "temporal_gt": {"begin_fid": b, "end_fid": e},
            "captions": [{"description": f"the {pred}.", "type": "person", "target_id": s}],
            "questions": []}


def traj_a():
    """tid 0 static box all frames, human at 0/4/8; tid 1 frames 2..11 with human at 2/6/10 whose
    box overlaps tid 0 at 2 and 10 but not at 6; tid 2 all tracker boxes (no human keyframe)."""
    frames = []
    for f in range(N_FRAMES):
        fr = [box(0, 10, 10, 30, 30, 0 if f in (0, 4, 8) else 1, "none" if f in (0, 4, 8) else "kcf")]
        if f >= 2:
            if f == 6:
                b1 = (40, 10, 60, 30)
            else:
                b1 = (12, 12, 28, 28)
            fr.append(box(1, *b1, 0 if f in (2, 6, 10) else 1, "none" if f in (2, 6, 10) else "mosse"))
        fr.append(box(2, 0, 30, 20, 47, 1, "linear"))
        frames.append(fr)
    return frames


def traj_two_tids():
    frames = []
    for f in range(N_FRAMES):
        frames.append([box(0, 5, 5, 25, 25, 0 if f % 3 == 0 else 1, "none" if f % 3 == 0 else "kcf"),
                       box(1, 35, 20, 60, 45, 0 if f % 4 == 0 else 1, "none" if f % 4 == 0 else "kcf")])
    return frames


def traj_many(n_tids):
    frames = []
    for f in range(N_FRAMES):
        frames.append([box(t, (t * 3) % 50, (t * 5) % 30, (t * 3) % 50 + 10, (t * 5) % 30 + 10,
                           0 if f % 2 == 0 else 1) for t in range(n_tids)])
    return frames


def write_mp4(path: Path, n_frames=N_FRAMES, black=False):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert w.isOpened()
    rng = np.random.default_rng(0)
    for i in range(n_frames):
        if black:
            frame = np.zeros((H, W, 3), np.uint8)
        else:
            frame = rng.integers(60, 200, size=(H, W, 3), dtype=np.uint8)
            frame[10:30, 10:30] = (255, 255, 255)
        w.write(frame)
    w.release()
    return path


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Build the synthetic dataset, export the roots into the environment, return paths."""
    vidstg = tmp_path / "vidstg" / "annotations"
    vidor_ann_dir = tmp_path / "vidor" / "ann"
    videos = tmp_path / "vidor" / "videos"
    transcoded = tmp_path / "vidor" / "videos_h264"
    for p in (vidstg, vidor_ann_dir, videos, transcoded):
        p.mkdir(parents=True)

    objs_a = [{"tid": 0, "category": "adult"}, {"tid": 1, "category": "ball/sports_ball"},
              {"tid": 2, "category": "dog"}]
    objs_2 = [{"tid": 0, "category": "child"}, {"tid": 1, "category": "toy"}]
    objs_c = [{"tid": t, "category": "adult"} for t in range(17)]

    anns = {
        VID_A: ("training/0001", vidor_ann(VID_A, "0001", objs_a, traj_a())),
        VID_B: ("validation/0002", vidor_ann(VID_B, "0002", objs_2, traj_two_tids())),
        VID_C: ("training/0003", vidor_ann(VID_C, "0003", objs_c, traj_many(17))),
        VID_D: ("validation/0004", vidor_ann(VID_D, "0004", objs_2, traj_two_tids())),
    }
    for vid, (sub, ann) in anns.items():
        p = vidor_ann_dir / sub / f"{vid}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(ann))
    # a stray non-VidSTG VidOR video, as in the real set (7,835 anns vs 6,770 VidSTG videos)
    (vidor_ann_dir / "training/0001/9999999999.json").write_text(
        json.dumps(vidor_ann("9999999999", "0001", objs_2, traj_two_tids())))

    train = [vidstg_record(VID_A, 0, "hold", 1, 0, 11, objs_a),
             vidstg_record(VID_A, 0, "watch", 2, 2, 9, objs_a)]
    val = [vidstg_record(VID_B, 0, "towards", 1, 0, 11, objs_2),
           vidstg_record(VID_D, 0, "hold", 1, 0, 11, objs_2)]
    test = [vidstg_record(VID_C, t, "next_to", t + 1, 0, 11, objs_c) for t in range(0, 16)]
    for name, recs in (("train", train), ("val", val), ("test", test)):
        (vidstg / f"{name}_annotations.json").write_text(json.dumps(recs))

    write_mp4(videos / "0001" / f"{VID_A}.mp4")
    write_mp4(videos / "0002" / f"{VID_B}.mp4")

    # build_worklist asserts the release counts; the fixture is the "release" here
    # (2 / 2 / 16 records, 5 annotation files). test_worker checks a mismatch raises.
    from vidstg_masks import datasets as _ds
    monkeypatch.setattr(_ds, "KNOWN_RECORDS_PER_SPLIT",
                        {"train": len(train), "val": len(val), "test": len(test)})
    monkeypatch.setitem(_ds.KNOWN_FACTS, "vidor_annotations", len(anns) + 1)

    monkeypatch.setenv("VIDSTG_ROOT", str(tmp_path / "vidstg"))
    monkeypatch.setenv("VIDOR_ANN_ROOT", str(vidor_ann_dir))
    monkeypatch.setenv("VIDOR_VIDEO_ROOT", str(videos))
    monkeypatch.setenv("VIDOR_TRANSCODED_ROOT", str(transcoded))
    return {"root": tmp_path, "vidstg": tmp_path / "vidstg", "vidor_ann": vidor_ann_dir,
            "videos": videos, "transcoded": transcoded, "anns": {k: v[1] for k, v in anns.items()}}


@pytest.fixture
def roots(data):
    from vidstg_masks.datasets import Roots
    return Roots.from_env()


@pytest.fixture
def base_prov():
    from vidstg_masks.records import base_provenance
    return base_provenance(code_commit="test0000")


class FakePredictor:
    """Records every request. `outputs` maps fid -> handle_stream_request payload."""

    def __init__(self, outputs=None):
        self.requests = []
        self.outputs = outputs or {}
        self.state = {"feature_cache": {"multigpu_buffer": {}}}

    def handle_request(self, request):
        self.requests.append(request)
        if request["type"] == "start_session":
            return {"session_id": "S"}
        if request["type"] == "add_prompt":
            # simulate the detector buffer leak that purge_prompt_buffers cleans
            self.state["feature_cache"][request["frame_index"]] = "fpn"
            self.state["feature_cache"]["multigpu_buffer"][request["frame_index"]] = "det"
        return {}

    def handle_stream_request(self, request):
        self.requests.append(request)
        for fid in sorted(self.outputs):
            yield {"frame_index": fid, "outputs": self.outputs[fid]}

    def _get_session(self, sid):
        return {"state": self.state}

    def adds(self):
        return [r for r in self.requests if r["type"] == "add_prompt"]


@pytest.fixture
def fake_predictor_cls():
    return FakePredictor
