import functools
import time
from typing import Dict, Any, Tuple, Generator, List
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from dataclasses import dataclass
import numpy as np

from livecc_utils import (
    prepare_multiturn_multimodal_inputs_for_generation,
    get_smart_resized_clip,
    get_smart_resized_video_reader,
)
from miis_broadcast.core.models.openai_tts import (
    enqueue_tts_text,
    print_tts_stats,
)

# ==========================================
# 📊 Performance Monitoring: Track LiveCC text generation time only
# ==========================================
perf_stats = {"gen_times": []}


def log_gen_time(value: float) -> None:
    perf_stats["gen_times"].append(value)


def print_final_stats() -> None:
    print("\n" + "=" * 40)
    print("LiveCC Performance")
    print("=" * 40)

    times = perf_stats["gen_times"]
    if times:
        # 1. Show first latency
        print(f"First LiveCC latency (Video->Text): {times[0]:.3f} s")

        # 2. Show average latency for remaining generations
        if len(times) > 1:
            avg_rest = sum(times[1:]) / len(times[1:])
            print(f"Average LiveCC latency (Excluding First): {avg_rest:.3f} s")
        else:
            print("Average LiveCC latency (Excluding First): N/A (only 1 generation)")
    else:
        print("no data for LiveCC latency (Video->Text)")

    print("=" * 40 + "\n")
    perf_stats["gen_times"] = []


# ============================================================
# ✅ Token Budget + Synchronous Slicing of past_ids / past_key_values
# ✅ Align with chat template boundaries to prevent "..." truncation
# ✅ Avoid prefix+tail recombination (prevent KV invalidation → video tokens/features mismatch)
# ============================================================

def _infer_ctx_max(model: Qwen2VLForConditionalGeneration, default: int = 32768) -> int:
    cfg = getattr(model, "config", None)
    for key in ("max_position_embeddings", "max_seq_len", "seq_length"):
        val = getattr(cfg, key, None) if cfg is not None else None
        if isinstance(val, int) and val > 0:
            return val
    return default


def _slice_tensor_on_matching_dim(x: torch.Tensor, past_len: int, keep: int) -> torch.Tensor:
    """
    Find the dimension in the tensor where size == past_len (usually seq_len) 
    and keep the last 'keep' elements.
    Returns the original tensor if no match is found (conservative).
    """
    if not torch.is_tensor(x) or past_len <= 0:
        return x
    for d, s in enumerate(x.shape):
        if s == past_len:
            slc = [slice(None)] * x.dim()
            slc[d] = slice(s - keep, s)
            return x[tuple(slc)]
    return x


def _find_boundary_start(
    ids_1d: List[int],
    *,
    min_start: int,
    boundary_patterns: List[List[int]],
) -> int:
    """
    Look for the nearest boundary start (>= min_start) in ids_1d.
    The closer to min_start, the better (preserves more context).
    boundary_patterns are sequences of token IDs (e.g., "<|im_start|>user").
    Returns min_start as a fallback.
    """
    n = len(ids_1d)
    best = None
    for pat in boundary_patterns:
        m = len(pat)
        if m == 0 or min_start > n - m:
            continue
        for i in range(min_start, n - m + 1):
            if ids_1d[i : i + m] == pat:
                best = i if best is None else min(best, i)
                break
    return best if best is not None else min_start


def truncate_state_by_budget(
    state: Dict[str, Any],
    new_len: int,
    *,
    ctx_max: int,
    max_new_tokens: int,
    headroom: int,
    boundary_patterns: List[List[int]],
) -> None:
    """
    Dynamically truncate state["past_ids"] and state["past_key_values"] 
    so that (past + new + max_new_tokens + headroom) <= ctx_max.

    ✅ Maximize memory retention: allow_past changes with new_len.
    ✅ Synchronous KV slicing (prevents state inconsistency).
    ✅ Boundary-aligned truncation (prevents empty output).
    """
    past_ids = state.get("past_ids", None)
    past_kv = state.get("past_key_values", None)

    if past_ids is None:
        return

    past_len = int(past_ids.shape[1])
    if past_len <= 0:
        return

    allow_past = int(ctx_max - headroom - max_new_tokens - new_len)

    if allow_past <= 0:
        state.pop("past_ids", None)
        state.pop("past_key_values", None)
        return

    if past_len <= allow_past:
        return

    keep = allow_past
    min_start = past_len - keep

    ids_1d = past_ids[0].tolist()
    start_idx = _find_boundary_start(
        ids_1d,
        min_start=min_start,
        boundary_patterns=boundary_patterns,
    )

    keep2 = past_len - start_idx
    if keep2 <= 0:
        state.pop("past_ids", None)
        state.pop("past_key_values", None)
        return

    state["past_ids"] = past_ids[:, -keep2:]

    if past_kv is not None:
        new_pkv = []
        for layer in past_kv:
            if isinstance(layer, (tuple, list)):
                new_layer = []
                for x in layer:
                    if torch.is_tensor(x):
                        new_layer.append(_slice_tensor_on_matching_dim(x, past_len, keep2))
                    else:
                        new_layer.append(x)
                new_pkv.append(tuple(new_layer))
            else:
                new_pkv.append(layer)
        state["past_key_values"] = tuple(new_pkv)


@dataclass
class VideoClip:
    frames: np.ndarray
    fps: float
    t_start: float


class LiveCCInfer:
    fps: float = 4.0
    initial_fps_frames: int = 12            
    streaming_fps_frames: int = 8
    initial_time_interval: float = initial_fps_frames / fps
    streaming_time_interval: float = streaming_fps_frames / fps
    frame_time_interval: float = 1.0 / fps

    def __init__(
        self,
        model_path: str = "chenjoya/LiveCC-7B-Instruct",
        device_id: int = 0,
        mm_window_sec: float = 12.0,  # ✅ Option A: Keep only recent N seconds of multimodal context
        carry_text_max_chars: int = 280,  # ✅ Character limit for context to carry over
        carry_recent_k: int = 3,  # ✅ Max number of recent commentaries to keep in state
    ) -> None:
        print("⏳ Loading LiveCC model, please wait...")

        t_load_start = time.time()
        self.device = f"cuda:{device_id}"
        # Try flash_attention_2 first (requires flash_attn installed);
        # fall back to sdpa which works without any extra package.
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            print("flash_attn not found, falling back to sdpa attention.")
            attn_impl = "sdpa"

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attn_impl,
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=False)

        t_load_end = time.time()
        print(f"⏱️ [Perf] Model weights loaded, time taken: {t_load_end - t_load_start:.4f} seconds")

        self.model.prepare_inputs_for_generation = functools.partial(
            prepare_multiturn_multimodal_inputs_for_generation,
            self.model,
        )

        message = {"role": "user", "content": [{"type": "text", "text": "livecc"}]}
        texts = self.processor.apply_chat_template([message], tokenize=False)
        self.system_prompt_offset = texts.index("<|im_start|>user")

        self._cached_video_readers_with_hw: Dict[str, Any] = {}

        # ✅ Fixed to 48 (Wait, the code says 24 below, let's keep it consistent)
        self.max_new_tokens: int = 24
        self.ctx_max: int = _infer_ctx_max(self.model, default=32768)
        self.headroom: int = 1024

        # ✅ Get boundary patterns using tokenizer for accurate slicing
        tok = self.processor.tokenizer
        self._boundary_patterns: List[List[int]] = [
            tok.encode("<|im_start|>user", add_special_tokens=False),
            tok.encode("<|im_start|>assistant", add_special_tokens=False),
        ]

        # ✅ Option A parameters
        self.mm_window_sec = float(mm_window_sec)
        self.carry_text_max_chars = int(carry_text_max_chars)
        self.carry_recent_k = int(carry_recent_k)

    def init_state(self, video_path: str) -> Dict[str, Any]:
        return {
            "video_path": video_path,
            # ✅ Option A status
            "mm_window_start": None,    # Start time of multimodal memory window
            "carry_text": "",           # Text context carried over after multimodal reset
            "recent_texts": [],         # Recent commentaries (used to update carry_text)
        }

    # ------------------------------
    # ✅ Option A: Multimodal sliding window (keep only recent N seconds)
    # ------------------------------
    def _apply_mm_window_policy(
        self,
        state: Dict[str, Any],
        *,
        start_ts: float,
        stop_ts: float,
    ) -> None:
        """
        If the multimodal context exceeds mm_window_sec, clear past_ids / past_key_values.
        Summarize recent commentaries into carry_text to maintain narrative continuity.
        """
        ws = state.get("mm_window_start", None)
        if ws is None:
            state["mm_window_start"] = float(start_ts)
            return

        ws = float(ws)
        span = float(stop_ts) - ws

        # Still within window: no action
        if span <= self.mm_window_sec:
            return

        # ✅ Window exceeded: prepare carry_text
        recent = state.get("recent_texts", [])
        if isinstance(recent, list) and recent:
            tail = recent[-self.carry_recent_k :]
            carry = " ".join([t.strip() for t in tail if isinstance(t, str) and t.strip()])
            carry = carry.strip()
            if len(carry) > self.carry_text_max_chars:
                carry = carry[-self.carry_text_max_chars :]
            state["carry_text"] = carry
        else:
            state["carry_text"] = state.get("carry_text", "")

        # ✅ Clear multimodal history (prevents degradation / video-token mismatch)
        state.pop("past_ids", None)
        state.pop("past_key_values", None)

        # ✅ Reset window start
        state["mm_window_start"] = float(start_ts)

    def _update_recent_texts(self, state: Dict[str, Any], response: str) -> None:
        if not isinstance(response, str):
            return
        r = response.strip()
        if not r:
            return
        recent = state.get("recent_texts", [])
        if not isinstance(recent, list):
            recent = []
        recent.append(r)
        # Keep only the most recent carry_recent_k sentences
        if len(recent) > self.carry_recent_k:
            recent = recent[-self.carry_recent_k :]
        state["recent_texts"] = recent

    def _build_message_content(
        self,
        *,
        start_ts: float,
        stop_ts: float,
        clip_obj: Any,
        query: str,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Assemble message:
        - Option A: If carry_text exists, prepend it for narrative continuity.
        - Add Time=... + video clip.
        - Append query if updated.
        """
        content = []

        carry = state.get("carry_text", "")
        if isinstance(carry, str) and carry.strip():
            content.append({"type": "text", "text": f"Context so far: {carry.strip()}"})

        content.append({"type": "text", "text": f"Time={start_ts:.1f}-{stop_ts:.1f}s"})
        content.append({"type": "video", "video": clip_obj})

        if query and state.get("query", None) != query:
            content.append({"type": "text", "text": query})
            state["query"] = query

        return {"role": "user", "content": content}

    # ------------------------------
    # LiveCC Main Inference (Generator)
    # ------------------------------
    def live_cc(
        self,
        query: str,
        state: Dict[str, Any],
        max_pixels: int = 384 * 28 * 28,
    ) -> Generator[Tuple[Tuple[float, float], str, Dict[str, Any]], None, None]:

        video_timestamp = state.get("video_timestamp", 0.0)
        last_timestamp = state.get("last_timestamp", -1.0 / self.fps)
        video_path = state["video_path"]

        if video_path not in self._cached_video_readers_with_hw:
            self._cached_video_readers_with_hw[video_path] = get_smart_resized_video_reader(
                video_path,
                max_pixels,
            )
            video_reader = self._cached_video_readers_with_hw[video_path][0]
            video_reader.get_frame_timestamp(0)
            state["video_pts"] = torch.from_numpy(video_reader._frame_pts[:, 1])
            state["last_video_pts_index"] = -1

        video_pts = state["video_pts"]

        if last_timestamp + self.frame_time_interval > video_pts[-1]:
            state["video_end"] = True
            print_final_stats()
            print_tts_stats()
            return

        video_reader, resized_height, resized_width = self._cached_video_readers_with_hw[video_path]
        last_video_pts_index = state["last_video_pts_index"]

        initialized = last_timestamp >= 0
        if not initialized:
            video_timestamp = max(video_timestamp, self.initial_time_interval)

        required_duration = self.streaming_time_interval if initialized else self.initial_time_interval
        if video_timestamp < (last_timestamp + required_duration - 0.01):
            return

        timestamps = torch.arange(
            last_timestamp + self.frame_time_interval,
            video_timestamp,
            self.frame_time_interval,
        )

        clip, clip_timestamps, clip_idxs = get_smart_resized_clip(
            video_reader,
            resized_height,
            resized_width,
            timestamps,
            video_pts,
            video_pts_index_from=last_video_pts_index + 1,
        )
        state["last_video_pts_index"] = clip_idxs[-1]
        state["last_timestamp"] = clip_timestamps[-1]

        interleave_clips, interleave_timestamps = [], []

        if not initialized:
            interleave_clips.append(clip[: self.initial_fps_frames])
            interleave_timestamps.append(clip_timestamps[: self.initial_fps_frames])
            clip = clip[self.initial_fps_frames :]
            clip_timestamps = clip_timestamps[self.initial_fps_frames :]

        if len(clip) > 0:
            interleave_clips.extend(list(clip.split(self.streaming_fps_frames)))
            interleave_timestamps.extend(list(clip_timestamps.split(self.streaming_fps_frames)))

        for clip_part, ts_part in zip(interleave_clips, interleave_timestamps):
            start_timestamp = ts_part[0].item()
            stop_timestamp = ts_part[-1].item() + self.frame_time_interval

            # ✅ Option A: Keep only recent N seconds of multimodal memory (reset if exceeded)
            self._apply_mm_window_policy(state, start_ts=start_timestamp, stop_ts=stop_timestamp)

            message = self._build_message_content(
                start_ts=start_timestamp,
                stop_ts=stop_timestamp,
                clip_obj=clip_part,
                query=query,
                state=state,
            )

            texts = self.processor.apply_chat_template(
                [message],
                tokenize=False,
                add_generation_prompt=True,
            )

            past_ids = state.get("past_ids", None)
            if past_ids is not None:
                texts = "<|im_end|>\n" + texts[self.system_prompt_offset :]

            inputs = self.processor(
                text=texts,
                images=None,
                videos=[clip_part],
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = inputs.to(self.device)

            # dtypes safety check
            if "pixel_values_videos" in inputs:
                pv = inputs["pixel_values_videos"]
                if pv.dtype == torch.float32 and self.model.dtype == torch.bfloat16:
                    inputs["pixel_values_videos"] = pv.to(torch.bfloat16)

            # ✅ token budget truncation (align boundaries, sync KV)
            new_len = int(inputs.input_ids.shape[1])
            truncate_state_by_budget(
                state,
                new_len,
                ctx_max=self.ctx_max,
                max_new_tokens=self.max_new_tokens,
                headroom=self.headroom,
                boundary_patterns=self._boundary_patterns,
            )

            past_ids = state.get("past_ids", None)
            if past_ids is not None:
                # Extend attention_mask to cover the prepended past tokens
                past_mask = torch.ones(
                    (1, past_ids.shape[1]), dtype=torch.long, device=self.device
                )
                inputs["attention_mask"] = torch.cat(
                    [past_mask, inputs["attention_mask"]], dim=1
                )
                inputs["input_ids"] = torch.cat([past_ids, inputs.input_ids], dim=1)

            # [Key] Record inference start time
            t_gen_start = time.time()

            outputs = self.model.generate(
                **inputs,
                past_key_values=state.get("past_key_values", None),
                return_dict_in_generate=True,
                pad_token_id=self.model.config.eos_token_id,
                do_sample=True,
                temperature=0.9,
                top_p=0.9,
                top_k=30,
                repetition_penalty=1.1,
                max_new_tokens=self.max_new_tokens,
            )

            t_gen_end = time.time()
            log_gen_time(t_gen_end - t_gen_start)

            state["past_key_values"] = outputs.past_key_values
            state["past_ids"] = outputs.sequences[:, :-1]

            response = self.processor.decode(
                outputs.sequences[0, inputs.input_ids.size(1) :],
                skip_special_tokens=True,
            )

            # ✅ Option A: Update recent commentaries (for the next reset)
            self._update_recent_texts(state, response)

            # [Key] Pass t_gen_start to TTS queue for latency tracking
            enqueue_tts_text(response, ref_ts=t_gen_start)

            yield (start_timestamp, stop_timestamp), response, state

    # ------------------------------
    # Inference from frames
    # ------------------------------
    def live_cc_from_frames(
        self,
        clip: "VideoClip",
        query: str,
        state: Dict[str, Any],
    ) -> Generator[Tuple[Tuple[float, float], str, Dict[str, Any]], None, None]:

        num_frames = int(clip.frames.shape[0])
        if num_frames == 0:
            return

        duration = num_frames / max(clip.fps, 1e-6)
        start_timestamp = float(clip.t_start)
        stop_timestamp = start_timestamp + duration

        # ✅ Option A: Keep only recent N seconds of multimodal memory (reset if exceeded)
        self._apply_mm_window_policy(state, start_ts=start_timestamp, stop_ts=stop_timestamp)

        message = self._build_message_content(
            start_ts=start_timestamp,
            stop_ts=stop_timestamp,
            clip_obj=clip.frames,
            query=query,
            state=state,
        )

        texts = self.processor.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True,
        )

        past_ids = state.get("past_ids", None)
        if past_ids is not None:
            if not hasattr(self, "system_prompt_offset"):
                temp_msg = {"role": "user", "content": [{"type": "text", "text": "livecc"}]}
                temp_text = self.processor.apply_chat_template([temp_msg], tokenize=False)
                self.system_prompt_offset = temp_text.index("<|im_start|>user")
            texts = "<|im_end|>\n" + texts[self.system_prompt_offset :]

        inputs = self.processor(
            text=texts,
            images=None,
            videos=[clip.frames],
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = inputs.to(self.device)

        if "pixel_values_videos" in inputs:
            pv = inputs["pixel_values_videos"]
            if pv.dtype == torch.float32 and self.model.dtype == torch.bfloat16:
                inputs["pixel_values_videos"] = pv.to(torch.bfloat16)

        # ✅ token budget truncation (align boundaries, sync KV)
        new_len = int(inputs.input_ids.shape[1])
        truncate_state_by_budget(
            state,
            new_len,
            ctx_max=self.ctx_max,
            max_new_tokens=self.max_new_tokens,
            headroom=self.headroom,
            boundary_patterns=self._boundary_patterns,
        )

        past_ids = state.get("past_ids", None)
        if past_ids is not None:
            # Extend attention_mask to cover the prepended past tokens
            past_mask = torch.ones(
                (1, past_ids.shape[1]), dtype=torch.long, device=self.device
            )
            inputs["attention_mask"] = torch.cat(
                [past_mask, inputs["attention_mask"]], dim=1
            )
            inputs["input_ids"] = torch.cat([past_ids, inputs.input_ids], dim=1)

        # [Key] Record inference start time
        t_gen_start = time.time()

        outputs = self.model.generate(
            **inputs,
            past_key_values=state.get("past_key_values", None),
            return_dict_in_generate=True,
            pad_token_id=self.model.config.eos_token_id,
            do_sample=True,
            temperature=1,
            top_p=0.9,
            repetition_penalty=1.1,
            max_new_tokens=self.max_new_tokens,
        )

        t_gen_end = time.time()
        log_gen_time(t_gen_end - t_gen_start)

        state["past_key_values"] = outputs.past_key_values
        state["past_ids"] = outputs.sequences[:, :-1]

        response = self.processor.decode(
            outputs.sequences[0, inputs.input_ids.size(1) :],
            skip_special_tokens=True,
        )

        # ✅ Option A: Update recent commentaries
        self._update_recent_texts(state, response)

        # [Key] Pass t_gen_start to TTS queue for latency tracking
        enqueue_tts_text(response, ref_ts=t_gen_start)

        yield (start_timestamp, stop_timestamp), response, state
