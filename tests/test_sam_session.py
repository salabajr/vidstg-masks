"""The request sequence must stay identical to run_pilot_masks.run_multiplex_session (default
path) in the research repo, so regenerated masks are comparable with existing runs."""

import numpy as np
import pytest

from vidstg_masks import sam_session as ss

BOX = [0.1, 0.2, 0.3, 0.4]
BOX2 = [0.5, 0.5, 0.9, 0.8]


def add_request(obj_id, fid, b):
    return dict(type="add_prompt", session_id="S", frame_index=fid,
                points=[[b[0], b[1]], [b[2], b[3]]], point_labels=[2, 3], obj_id=obj_id)


@pytest.fixture
def no_scores(monkeypatch):
    monkeypatch.setattr(ss, "collect_sam2_scores", lambda predictor, sid: {})


def test_default_path_request_sequence(fake_predictor_cls, no_scores):
    pred = fake_predictor_cls()
    prompts = {7: [(10, BOX), (5, BOX2), (20, BOX)], 3: [(2, BOX2), (5, BOX)]}
    per_frame, wall = ss.run_session(pred, "v.mp4", prompts, 0, 30)
    # phase 1: reference anchors in dict order; phase 2: re-anchors sorted by (fid, obj_id)
    assert pred.adds() == [add_request(7, 10, BOX), add_request(3, 2, BOX2),
                           add_request(3, 5, BOX), add_request(7, 5, BOX2), add_request(7, 20, BOX)]
    types = [r["type"] for r in pred.requests]
    assert types == ["start_session"] + ["add_prompt"] * 5 + ["propagate_in_video", "close_session"]
    assert pred.requests[0] == dict(type="start_session", resource_path="v.mp4", offload_video_to_cpu=True)
    prop = pred.requests[-2]
    assert prop == dict(type="propagate_in_video", session_id="S", propagation_direction="forward",
                        start_frame_index=0, max_frame_num_to_track=31)
    assert pred.requests[-1] == dict(type="close_session", session_id="S")
    assert per_frame == {} and wall >= 0
    # prompt buffers were purged before propagation
    assert [k for k in pred.state["feature_cache"] if isinstance(k, int)] == []
    assert pred.state["feature_cache"]["multigpu_buffer"] == {}


def test_session_closes_when_propagation_raises(fake_predictor_cls, no_scores):
    pred = fake_predictor_cls()

    def boom(request):
        pred.requests.append(request)
        raise RuntimeError("CUDA out of memory")
    pred.handle_stream_request = boom
    with pytest.raises(RuntimeError):
        ss.run_session(pred, "v.mp4", {1: [(0, BOX)]}, 0, 5)
    assert pred.requests[-1]["type"] == "close_session"
    assert ss.is_cuda_oom(RuntimeError("CUDA out of memory"))


def test_outputs_and_scores_are_merged(fake_predictor_cls, monkeypatch):
    m1, m2 = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    m1[0, 0] = True
    outputs = {3: {"out_obj_ids": [1, 2], "out_binary_masks": [m1[None], m2], "out_probs": [0.9, 0.8]}}
    pred = fake_predictor_cls(outputs)
    monkeypatch.setattr(ss, "collect_sam2_scores", lambda p, s: {(3, 1): 0.75})
    per_frame, _ = ss.run_session(pred, "v.mp4", {1: [(0, BOX)], 2: [(1, BOX2)]}, 0, 4)
    assert set(per_frame) == {3}
    mask, conf = per_frame[3][1]
    assert mask.shape == (4, 4) and mask.dtype == bool and mask.sum() == 1 and conf == 0.75
    assert np.isnan(per_frame[3][2][1])  # no tracker score for obj 2 -> nan, never invented


def test_offload_rule():
    assert not ss.needs_state_offload(1500, 1)
    assert ss.needs_state_offload(1501, 1)
    assert ss.needs_state_offload(600, 5)          # 3000 frame-objects
    assert not ss.needs_state_offload(500, 5)      # 2500 is not above the cap


def test_masks_probs_of_ignores_unknown_payloads():
    assert ss.masks_probs_of(None) == {} and ss.masks_probs_of({"foo": 1}) == {}


def test_clicks_travel_with_the_box_corners(fake_predictor_cls, no_scores):
    pred = fake_predictor_cls()
    pts, labels = [[0.2, 0.3], [0.4, 0.5]], [0, 1]
    prompts = {1: [(4, BOX, pts, labels), (9, BOX2)], 2: [(0, BOX2), (4, BOX, [[0.6, 0.6]], [0])]}
    ss.run_session(pred, "v.mp4", prompts, 0, 10)
    adds = pred.adds()
    assert adds[0] == dict(type="add_prompt", session_id="S", frame_index=4, obj_id=1,
                           points=[[BOX[0], BOX[1]], [BOX[2], BOX[3]]] + pts, point_labels=[2, 3] + labels)
    assert adds[1] == add_request(2, 0, BOX2)
    # refinements frame-major by (fid, obj_id), whatever the entry shape
    assert [(r["frame_index"], r["obj_id"]) for r in adds[2:]] == [(4, 2), (9, 1)]
    assert adds[2]["points"] == [[BOX[0], BOX[1]], [BOX[2], BOX[3]], [0.6, 0.6]]
    assert adds[2]["point_labels"] == [2, 3, 0]
    assert adds[3] == add_request(1, 9, BOX2)
    assert pred.requests[-1]["type"] == "close_session"


def test_prompt_encoder_cap_and_click_shape_are_asserted(fake_predictor_cls, no_scores):
    pred = fake_predictor_cls()
    too_many = [[0.5, 0.5]] * (ss.MAX_PROMPT_POINTS - 1)            # 2 corners + 15 clicks = 17
    with pytest.raises(AssertionError, match="prompt-encoder cap"):
        ss.run_session(pred, "v.mp4", {1: [(0, BOX, too_many, [0] * len(too_many))]}, 0, 5)
    assert pred.requests[-1]["type"] == "close_session"             # closed on the way out
    with pytest.raises(AssertionError, match="points vs"):
        ss.run_session(pred, "v.mp4", {1: [(0, BOX, [[0.5, 0.5]], [0, 0])]}, 0, 5)
    with pytest.raises(AssertionError, match="labels must be 0/1"):
        ss.run_session(pred, "v.mp4", {1: [(0, BOX, [[0.5, 0.5]], [2])]}, 0, 5)
    at_cap = [[0.5, 0.5]] * (ss.MAX_PROMPT_POINTS - 2)              # exactly the cap is fine
    ss.run_session(fake_predictor_cls(), "v.mp4", {1: [(0, BOX, at_cap, [1] * len(at_cap))]}, 0, 5)
