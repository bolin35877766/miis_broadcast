from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from miis_broadcast.core.models.livecc_transformers import LiveCCInfer


def _infer(*, interval: int = 1, structured_only: bool = True) -> LiveCCInfer:
    infer = LiveCCInfer.__new__(LiveCCInfer)
    infer.kv_reset_every_segments = interval
    infer.carry_structured_only = structured_only
    infer.carry_recent_k = 2
    return infer


def test_periodic_kv_reset_also_forces_query_to_be_resent() -> None:
    infer = _infer(interval=1)
    state = {"past_ids": object(), "past_key_values": object(), "query": "role prompt"}
    infer._apply_segment_reset_policy(state)
    assert "past_ids" not in state
    assert "past_key_values" not in state
    assert "query" not in state


def test_disabled_kv_reset_preserves_stream_memory() -> None:
    infer = _infer(interval=0)
    state = {"past_ids": object(), "past_key_values": object(), "query": "role prompt"}
    infer._apply_segment_reset_policy(state)
    assert "past_ids" in state
    assert state["query"] == "role prompt"


def test_structured_only_does_not_carry_fragmented_natural_caption() -> None:
    infer = _infer(structured_only=True)
    state = {"recent_texts": []}
    infer._update_recent_texts(state, "A user wearing the VR headset")
    assert state["recent_texts"] == []


def test_role_words_from_prompt_are_not_mistaken_for_prompt_leakage() -> None:
    infer = _infer()
    query = (
        'Answer with one sentence beginning "The player" or "The robot opponent". '
        "The robot opponent has possession when the ball is beside test_bot1."
    )
    assert not infer._is_degenerate(
        "The robot opponent dribbles while the player defends.", query
    )
    assert infer._is_degenerate(
        "The robot opponent has possession when the ball is beside test_bot1.", query
    )


def test_right_half_crop_keeps_authoritative_gameplay_view() -> None:
    infer = _infer()
    infer.input_crop = "right_half"
    clip = np.arange(2 * 3 * 8 * 1).reshape(2, 3, 8, 1)
    cropped = infer._prepare_visual_clip(clip)
    assert cropped.shape == (2, 3, 4, 1)
    assert np.array_equal(cropped, clip[:, :, 4:, :])
