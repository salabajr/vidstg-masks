import shutil

import pytest

from conftest import N_FRAMES, VID_A, W, H, write_mp4
from vidstg_masks.video import decode_check, is_black, probe_frame_count, transcode_h264


def test_decode_check_counts_frames_and_size(data):
    p = data["videos"] / "0001" / f"{VID_A}.mp4"
    chk = decode_check(p, N_FRAMES, (W, H))
    assert chk["frame_count"] == N_FRAMES and chk["size"] == (W, H)
    assert chk["frame_count_ok"] and chk["size_ok"] and not chk["black"]
    bad = decode_check(p, N_FRAMES + 1, (W, H + 2))
    assert not bad["frame_count_ok"] and not bad["size_ok"]


def test_black_video_is_detected(tmp_path):
    p = write_mp4(tmp_path / "black.mp4", black=True)
    assert is_black(p)
    assert decode_check(p, N_FRAMES)["black"]
    assert not is_black(write_mp4(tmp_path / "bright.mp4"))


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not installed")
def test_probe_frame_count_matches_decode(data):
    p = data["videos"] / "0001" / f"{VID_A}.mp4"
    assert probe_frame_count(p) == N_FRAMES


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_transcode_keeps_frame_count(data, tmp_path):
    src = data["videos"] / "0001" / f"{VID_A}.mp4"
    dst = transcode_h264(src, tmp_path / "out" / "a.mp4")
    assert dst.is_file() and not list((tmp_path / "out").glob("*.part"))
    assert decode_check(dst, N_FRAMES, (W, H))["frame_count_ok"]
