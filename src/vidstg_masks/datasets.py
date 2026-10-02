"""VidSTG + VidOR loading.

VidSTG records carry relations (used_relation) and segments but no boxes; boxes live only
in the VidOR per-video annotation JSONs, joined on vid + tid. All three VidSTG files
(train, val, test) are loaded; the split name travels with each record as `vidstg_split`.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

VIDSTG_SPLITS = ("train", "val", "test")
VIDSTG_FILES = {s: f"{s}_annotations.json" for s in VIDSTG_SPLITS}

# Computed 2026-07-30 from Guaranteer/VidSTG-Dataset (all three splits) and the
# shangxd/vidor annotation zips; re-verified 2026-09-25 while building this package.
KNOWN_FACTS = {
    "vidstg_records": 44_808,
    "vidstg_videos": 6_770,
    "relation_pairs": 26_016,      # unique (vid, tid) over subject+object of every used_relation
    "vidor_annotations": 7_835,    # per-video JSONs under training/ + validation/
}
# Records per VidSTG annotation file (same source, same date). `build-worklist` asserts
# these for the splits it loads before any GPU work is scheduled.
KNOWN_RECORDS_PER_SPLIT = {"train": 36_202, "val": 3_996, "test": 4_610}
VIDOR_INDEX_PATTERN = "<VIDOR_ANN_ROOT>/**/<digits>.json"


class DataError(RuntimeError):
    """Annotation data disagrees with itself or with what the pipeline requires."""


@dataclass(frozen=True)
class Roots:
    vidstg_root: Path
    vidor_ann_root: Path
    vidor_video_root: Path
    vidor_transcoded_root: Path | None = None

    ENV = ("VIDSTG_ROOT", "VIDOR_ANN_ROOT", "VIDOR_VIDEO_ROOT", "VIDOR_TRANSCODED_ROOT")

    @classmethod
    def from_env(cls, env=None) -> "Roots":
        env = os.environ if env is None else env
        missing = [k for k in cls.ENV[:3] if not env.get(k)]
        if missing:
            raise KeyError(f"missing environment variables: {', '.join(missing)} "
                           f"(see .env.example)")
        t = env.get("VIDOR_TRANSCODED_ROOT") or None
        return cls(Path(env["VIDSTG_ROOT"]), Path(env["VIDOR_ANN_ROOT"]),
                   Path(env["VIDOR_VIDEO_ROOT"]), Path(t) if t else None)

    @classmethod
    def from_args(cls, args, env=None) -> "Roots":
        """CLI flags override environment variables; each root falls back independently."""
        env = os.environ if env is None else env
        vals = {}
        for key in cls.ENV:
            attr = key.lower()
            v = getattr(args, attr, None) or env.get(key) or None
            vals[attr] = Path(v) if v else None
        missing = [k for k in cls.ENV[:3] if vals[k.lower()] is None]
        if missing:
            raise KeyError(f"missing roots: {', '.join(missing)} (flag --{missing[0].lower().replace('_', '-')} "
                           f"or environment variable)")
        return cls(**vals)

    @classmethod
    def from_dict(cls, d: dict) -> "Roots":
        t = d.get("vidor_transcoded_root")
        return cls(Path(d["vidstg_root"]), Path(d["vidor_ann_root"]),
                   Path(d["vidor_video_root"]), Path(t) if t else None)

    def to_dict(self) -> dict:
        return {"vidstg_root": str(self.vidstg_root),
                "vidor_ann_root": str(self.vidor_ann_root),
                "vidor_video_root": str(self.vidor_video_root),
                "vidor_transcoded_root": str(self.vidor_transcoded_root)
                if self.vidor_transcoded_root else None}

    def vidstg_file(self, split: str) -> Path:
        """<root>/annotations/<split>_annotations.json (also accepted directly under <root>)."""
        name = VIDSTG_FILES[split]
        p = self.vidstg_root / "annotations" / name
        return p if p.exists() else self.vidstg_root / name


# ── VidSTG ──────────────────────────────────────────────────────────────────

def load_vidstg_split(roots: Roots, split: str) -> list[dict]:
    path = roots.vidstg_file(split)
    if not path.exists():
        raise FileNotFoundError(f"VidSTG {split} annotations not found at {path}")
    with open(path) as f:
        recs = json.load(f)
    for r in recs:
        r["vidstg_split"] = split
    return recs


def load_vidstg(roots: Roots, splits=VIDSTG_SPLITS) -> list[dict]:
    """All records of the requested VidSTG files, each tagged with `vidstg_split`."""
    out: list[dict] = []
    for s in splits:
        out.extend(load_vidstg_split(roots, s))
    return out


def records_by_vid(records: list[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in records:
        by.setdefault(r["vid"], []).append(r)
    return by


def relation_tids(records: list[dict]) -> list[int]:
    """Sorted union of subject_tid/object_tid over the records' used_relations."""
    return sorted({r["used_relation"][k] for r in records
                   for k in ("subject_tid", "object_tid")})


# ── VidOR ───────────────────────────────────────────────────────────────────

def build_vidor_index(roots: Roots) -> dict[str, Path]:
    """vid -> annotation JSON path: every `<digits>.json` under <ann_root>, at any depth.
    The release layout is {training,validation}/<folder>/<vid>.json; a copy that was
    flattened or nested one level deeper still indexes."""
    index: dict[str, Path] = {}
    for p in sorted(roots.vidor_ann_root.rglob("*.json")):
        if p.stem.isdigit():
            index[p.stem] = p
    return index


def load_vidor(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def boxes_for_tid(vidor_ann: dict, tid: int) -> dict[int, dict]:
    """fid -> box record ({bbox, generated, tracker}) for one trajectory."""
    boxes: dict[int, dict] = {}
    for fid, frame in enumerate(vidor_ann["trajectories"]):
        for box in frame:
            if box["tid"] == tid:
                boxes[fid] = box
                break
    return boxes


def resolve_video(roots: Roots, ann: dict) -> Path | None:
    """Transcoded copy first (VP6F/odd-height cv2 all-black fix), else the raw video."""
    candidates = []
    if roots.vidor_transcoded_root is not None:
        candidates.append(roots.vidor_transcoded_root / ann["video_path"])
    candidates.append(roots.vidor_video_root / ann["video_path"])
    for p in candidates:
        if p.exists():
            return p
    return None


def _under(path: Path, root: Path | None) -> Path | None:
    """`path` relative to `root`, or None when it is not below it."""
    if root is None:
        return None
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def rebase_unit_paths(unit: dict, old: Roots, new: Roots) -> dict:
    """Copy of a worklist unit with `vidor_ann` and `video_path` moved from the roots the
    worklist was built with to `new` (--roots-from-env). The annotation keeps its path
    relative to the annotation root; the video keeps its path relative to whichever video
    root it was under and is re-resolved (transcoded copy first). A unit whose video was
    missing at build time is re-resolved from its annotation."""
    u = dict(unit)
    rel_ann = _under(Path(unit["vidor_ann"]), old.vidor_ann_root)
    if rel_ann is None:
        raise DataError(f"{unit['vid']}: annotation {unit['vidor_ann']} is not under the "
                        f"worklist's VIDOR_ANN_ROOT {old.vidor_ann_root}; rebuild the worklist")
    ann_path = new.vidor_ann_root / rel_ann
    u["vidor_ann"] = str(ann_path)
    rel_video = None
    if unit.get("video_path"):
        vp = Path(unit["video_path"])
        rel_video = _under(vp, old.vidor_transcoded_root) or _under(vp, old.vidor_video_root)
    if rel_video is None and ann_path.is_file():
        rel_video = Path(load_vidor(ann_path)["video_path"])
    video = None
    if rel_video is not None:
        for root in (new.vidor_transcoded_root, new.vidor_video_root):
            if root is not None and (root / rel_video).exists():
                video = root / rel_video
                break
    u["video_path"] = str(video) if video else None
    return u


# ── Known facts ─────────────────────────────────────────────────────────────

def assert_known_counts(records: list[dict], splits, index: dict[str, Path], roots: Roots) -> dict:
    """Raise DataError unless the loaded VidSTG split(s) and the VidOR index carry the
    release counts (KNOWN_RECORDS_PER_SPLIT, KNOWN_FACTS['vidor_annotations']). Meant to
    run before any GPU work is scheduled; returns the counts it checked."""
    per_split = Counter(r["vidstg_split"] for r in records)
    checked = {f"vidstg_{s}_records": per_split.get(s, 0) for s in splits}
    checked["vidor_annotations"] = len(index)
    problems = []
    for s in splits:
        if per_split.get(s, 0) != KNOWN_RECORDS_PER_SPLIT[s]:
            problems.append(f"VidSTG {s}: {per_split.get(s, 0):,} records at {roots.vidstg_file(s)}, "
                            f"expected {KNOWN_RECORDS_PER_SPLIT[s]:,}")
    if len(index) != KNOWN_FACTS["vidor_annotations"]:
        problems.append(f"VidOR annotations: {len(index):,} files matching {VIDOR_INDEX_PATTERN} "
                        f"under {roots.vidor_ann_root}, expected {KNOWN_FACTS['vidor_annotations']:,}")
    if problems:
        raise DataError("dataset counts differ from the release (an incomplete or altered copy; "
                        "pass --no-assert-counts to build anyway): " + "; ".join(problems))
    return checked

def known_facts_check(records: list[dict], index: dict[str, Path]) -> dict:
    """Counts of the loaded annotation set next to KNOWN_FACTS. `ok` is True only when
    every count matches; the caller (doctor) decides whether a mismatch is fatal."""
    vids = {r["vid"] for r in records}
    pairs = {(r["vid"], r["used_relation"][k]) for r in records
             for k in ("subject_tid", "object_tid")}
    splits = {s: 0 for s in VIDSTG_SPLITS}
    for r in records:
        splits[r["vidstg_split"]] = splits.get(r["vidstg_split"], 0) + 1
    counts = {
        "vidstg_records": len(records),
        "vidstg_videos": len(vids),
        "relation_pairs": len(pairs),
        "vidor_annotations": len(index),
    }
    matches = {k: counts[k] == v for k, v in KNOWN_FACTS.items()}
    return {
        "counts": counts,
        "expected": dict(KNOWN_FACTS),
        "matches": matches,
        "records_per_split": splits,
        "videos_with_vidor_annotation": sum(1 for v in vids if v in index),
        "ok": all(matches.values()) and all(v in index for v in vids),
    }
