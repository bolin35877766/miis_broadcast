# src/miis_broadcast/core/models/bytetrack_tracker.py
"""
ByteTrack wrapper for miis_broadcast.

Wraps YOLOX detection + BYTETracker association from the ByteTrack_repo.
Internal algorithm logic is NOT modified — all changes from the original
demo_track.py are preserved verbatim:
  - Subject-only mode (ID locking, centre+area re-selection)
  - Subject bounding-box padding
  - FPS / inference timing logs
"""

import sys
import os
import time
import types
import logging
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve ByteTrack repo path
# ---------------------------------------------------------------------------

def _resolve_repo_path(bytetrack_repo: Optional[str] = None) -> Path:
    """
    Resolve the ByteTrack_repo directory.
    Priority: explicit argument → env var BYTETRACK_REPO → sibling folder heuristic.
    """
    if bytetrack_repo:
        return Path(bytetrack_repo).resolve()

    env = os.environ.get("BYTETRACK_REPO")
    if env:
        return Path(env).resolve()

    # Heuristic: look for ByteTrack_repo relative to this file's ancestors
    for parent in Path(__file__).parents:
        candidate = parent / "ByteTrack" / "ByteTrack_repo"
        if candidate.is_dir():
            return candidate
        candidate2 = parent.parent / "ByteTrack" / "ByteTrack_repo"
        if candidate2.is_dir():
            return candidate2

    raise FileNotFoundError(
        "[ByteTrack] Cannot locate ByteTrack_repo. "
        "Set the BYTETRACK_REPO environment variable or pass bytetrack_repo= explicitly."
    )


# ---------------------------------------------------------------------------
# Minimal timer (mirrors yolox/tracking_utils/timer.py behaviour)
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self):
        self._tic: float = 0.0
        self._total: float = 0.0
        self._count: int = 0

    def tic(self):
        self._tic = time.perf_counter()

    def toc(self):
        elapsed = time.perf_counter() - self._tic
        self._total += elapsed
        self._count += 1

    @property
    def average_time(self) -> float:
        return self._total / max(1, self._count)


def _resolve_track_device(device_cfg: str) -> "torch.device":
    """
    Map models.yml ``device`` to ``torch.device``.

    Accepts ``cpu``, ``cuda``, ``cuda:N``, ``gpu``, or a numeric string ``N`` (GPU index).
    """
    import torch

    raw = str(device_cfg or "cpu").strip()
    low = raw.lower()

    if low == "cpu":
        dev = torch.device("cpu")
        print(f"[ByteTrack] 使用裝置: {dev}")
        return dev

    # GPU requested
    if low in ("gpu", "cuda"):
        spec = "cuda:0"
    elif low.startswith("cuda:"):
        spec = raw  # cuda:0, cuda:1, ...
    elif raw.isdigit():
        spec = f"cuda:{int(raw)}"
    else:
        print(f"[ByteTrack] Unknown device {device_cfg!r}; using CPU.")
        dev = torch.device("cpu")
        print(f"[ByteTrack] 使用裝置: {dev}")
        return dev

    if not torch.cuda.is_available():
        print(
            "[ByteTrack] 設定要求 GPU，但 torch.cuda.is_available() 為 False。\n"
            f"  目前 PyTorch {torch.__version__} | torch.version.cuda={torch.version.cuda!r}\n"
            "  Windows 請安裝含 CUDA 的 PyTorch，例如 (依你的 CUDA 版本調整 cu124/cu118)：\n"
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124\n"
            "  已改為使用 CPU。"
        )
        dev = torch.device("cpu")
        print(f"[ByteTrack] 使用裝置: {dev}")
        return dev

    dev = torch.device(spec)
    idx = dev.index if dev.index is not None else 0
    name = torch.cuda.get_device_name(idx)
    print(f"[ByteTrack] 使用裝置: {dev} ({name})")
    return dev


# ---------------------------------------------------------------------------
# ByteTrackWrapper
# ---------------------------------------------------------------------------

class ByteTrackWrapper:
    """
    Drop-in tracker.  Call process(frame_bgr, frame_id) every frame.

    Returns:
        annotated_bgr  — frame with bounding boxes drawn (BGR, same resolution)
        subject_crop_rgb — cropped + padded region of the main subject (RGB)
                           or the full frame if subject_only=False / no detections
    """

    # How many frames of absence before re-selecting the subject
    _RESELECT_PAT = 30

    def __init__(
        self,
        ckpt_path: str,
        exp_file: str,
        bytetrack_repo: Optional[str] = None,
        device: str = "cuda",
        fp16: bool = True,
        fuse: bool = True,
        # BYTETracker args
        track_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        aspect_ratio_thresh: float = 1.6,
        min_box_area: float = 10.0,
        mot20: bool = False,
        # Subject mode
        subject_only: bool = True,
        subject_pad: float = 0.15,
        fps: int = 30,
        # Subject quality gates
        min_subject_area_ratio: float = 0.03,  # ignore tracks smaller than 3% of frame area
        preempt_ratio: float = 4.0,            # switch subject if new track is X times larger
    ) -> None:

        # ── 1. Add ByteTrack_repo to sys.path ──────────────────────────────
        repo = _resolve_repo_path(bytetrack_repo)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        logger.info(f"[ByteTrack] repo path: {repo}")
        print(f"[ByteTrack] 使用 repo: {repo}")

        # Silence PyTorch UserWarning from YOLOX internals (meshgrid indexing).
        warnings.filterwarnings(
            "ignore",
            message=r".*torch\.meshgrid.*",
            category=UserWarning,
        )

        # ── 2. Import yolox after path is set ──────────────────────────────
        import torch
        from yolox.exp import get_exp
        from yolox.utils import fuse_model, get_model_info, postprocess
        from yolox.data.data_augment import preproc
        from yolox.tracker.byte_tracker import BYTETracker
        from yolox.utils.visualize import plot_tracking

        self._torch = torch
        self._preproc = preproc
        self._postprocess = postprocess
        self._plot_tracking = plot_tracking
        self._BYTETracker = BYTETracker

        # ── 3. Build args namespace for BYTETracker (mirrors make_parser defaults) ──
        self._track_args = types.SimpleNamespace(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            aspect_ratio_thresh=aspect_ratio_thresh,
            min_box_area=min_box_area,
            mot20=mot20,
            subject_only=subject_only,
            subject_pad=subject_pad,
        )

        self.subject_only = subject_only
        self.subject_pad = subject_pad
        self.min_subject_area_ratio = min_subject_area_ratio
        self.preempt_ratio = preempt_ratio

        # ── 4. Load YOLOX model ────────────────────────────────────────────
        self.device = _resolve_track_device(device)

        exp = get_exp(exp_file, None)
        model = exp.get_model().to(self.device)
        model.eval()

        print(f"[ByteTrack] 載入權重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[ByteTrack] ✅ 權重載入完成")

        if fuse:
            from yolox.utils import fuse_model as _fuse
            model = _fuse(model)
            print("[ByteTrack] Conv-BN fusion 完成")

        if fp16 and self.device.type == "cuda":
            model = model.half()
            print("[ByteTrack] FP16 模式啟用")

        self.model = model
        self.exp = exp
        self.fp16 = fp16 and self.device.type == "cuda"

        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size

        # ── 5. Init BYTETracker ────────────────────────────────────────────
        self.tracker = BYTETracker(self._track_args, frame_rate=fps)
        self._timer = _Timer()

        # Subject-lock state (mirrors imageflow_demo variables)
        self._subject_tid: Optional[int] = None
        self._subject_lost: int = 0

        # Logging state
        self._total_frames: int = 0
        self._wall_start: float = time.time()

        print("[ByteTrack] 🚀 初始化完成，已進入追蹤待機狀態")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def process(
        self,
        frame_bgr: np.ndarray,
        frame_id: int,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run YOLOX detection + BYTETracker on one BGR frame.

        Args:
            frame_bgr: OpenCV BGR frame (H, W, 3)
            frame_id:  1-based frame counter

        Returns:
            annotated_bgr:     frame with track boxes drawn (BGR)
            subject_crop_rgb:  padded crop of main subject (RGB),
                               or None if no valid subject is detected
        """
        self._total_frames += 1
        h, w = frame_bgr.shape[:2]
        img_info = {"height": h, "width": w, "raw_img": frame_bgr}

        # ── YOLOX inference ───────────────────────────────────────────────
        outputs = self._infer(frame_bgr, img_info)

        online_tlwhs: List = []
        online_ids:   List = []
        online_scores: List = []

        if outputs[0] is not None:
            online_targets = self.tracker.update(
                outputs[0],
                [img_info["height"], img_info["width"]],
                self.test_size,
            )

            for t in online_targets:
                tlwh = t.tlwh
                tid  = t.track_id
                vertical = tlwh[2] / tlwh[3] > self._track_args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > self._track_args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)

        self._timer.toc()

        # ── Subject-only mode (verbatim from imageflow_demo) ──────────────
        vis_tlwhs = online_tlwhs
        vis_ids   = online_ids

        if self.subject_only and online_tlwhs:
            _frame_area = img_info["width"] * img_info["height"]
            _im_cx = img_info["width"]  / 2.0
            _im_cy = img_info["height"] / 2.0

            def _score(i):
                x, y, bw, bh = online_tlwhs[i]
                area = bw * bh
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                dist2 = ((cx - _im_cx) / max(_im_cx, 1)) ** 2 + \
                        ((cy - _im_cy) / max(_im_cy, 1)) ** 2
                return area / (1.0 + dist2)

            # Filter: only consider tracks large enough to be a real subject
            _valid_indices = [
                i for i, (x, y, bw, bh) in enumerate(online_tlwhs)
                if (bw * bh) / max(_frame_area, 1) >= self.min_subject_area_ratio
            ]

            matched_idx = None
            if self._subject_tid is not None:
                for _i, _tid in enumerate(online_ids):
                    if _tid == self._subject_tid:
                        matched_idx = _i
                        break

            if matched_idx is not None:
                # Locked target still visible
                self._subject_lost = 0

                # Preemption check: if a MUCH larger valid track appears, switch to it
                if _valid_indices:
                    _cur_area = online_tlwhs[matched_idx][2] * online_tlwhs[matched_idx][3]
                    best_valid = max(_valid_indices, key=_score)
                    _best_area = online_tlwhs[best_valid][2] * online_tlwhs[best_valid][3]
                    if (online_ids[best_valid] != self._subject_tid and
                            _best_area > self.preempt_ratio * _cur_area):
                        # New dominant subject — preempt
                        print(f"[ByteTrack] 🔄 主體搶佔: ID {self._subject_tid} → {online_ids[best_valid]} "
                              f"(面積 {int(_cur_area)} → {int(_best_area)})")
                        self._subject_tid  = online_ids[best_valid]
                        matched_idx        = best_valid
                        self._subject_lost = 0

                vis_tlwhs = [online_tlwhs[matched_idx]]
                vis_ids   = [online_ids[matched_idx]]
            else:
                self._subject_lost += 1
                if self._subject_tid is None or self._subject_lost >= self._RESELECT_PAT:
                    # Re-select from valid (large enough) tracks only
                    candidates = _valid_indices if _valid_indices else list(range(len(online_tlwhs)))
                    best = max(candidates, key=_score)
                    self._subject_tid  = online_ids[best]
                    self._subject_lost = 0
                    vis_tlwhs = [online_tlwhs[best]]
                    vis_ids   = [online_ids[best]]
                else:
                    # Track temporarily lost — show nothing until it reappears
                    vis_tlwhs = []
                    vis_ids   = []

        # ── Subject padding (verbatim from imageflow_demo) ─────────────────
        if self.subject_only and vis_tlwhs:
            pad  = self.subject_pad
            _iw  = img_info["width"]
            _ih  = img_info["height"]
            _padded = []
            for _bx, _by, _bw, _bh in vis_tlwhs:
                _dx = _bw * pad
                _dy = _bh * pad
                _nx = max(0.0,       _bx - _dx)
                _ny = max(0.0,       _by - _dy)
                _nw = min(_iw - _nx, _bw + 2 * _dx)
                _nh = min(_ih - _ny, _bh + 2 * _dy)
                _padded.append([_nx, _ny, _nw, _nh])
            vis_tlwhs = _padded

        # ── FPS log (every 20 frames, mirrors imageflow_demo) ─────────────
        if frame_id % 20 == 0:
            infer_fps = 1.0 / max(1e-5, self._timer.average_time)
            wall_fps  = self._total_frames / max(1e-5, time.time() - self._wall_start)
            print(
                f"[ByteTrack] Frame {frame_id:5d} | "
                f"Infer FPS: {infer_fps:.1f} | Wall FPS: {wall_fps:.1f} | "
                f"Tracks: {len(online_tlwhs)} | Subject ID: {self._subject_tid}"
            )

        # ── Draw annotated frame ───────────────────────────────────────────
        annotated_bgr = self._plot_tracking(
            img_info["raw_img"], vis_tlwhs, vis_ids,
            frame_id=frame_id,
            fps=1.0 / max(1e-5, self._timer.average_time),
        )

        # ── Extract subject crop (RGB) for LiveCC ─────────────────────────
        subject_crop_rgb = self._extract_subject_crop(frame_bgr, vis_tlwhs)

        return annotated_bgr, subject_crop_rgb

    def reset(self) -> None:
        """Reset tracker state (call when stream restarts)."""
        self.tracker = self._BYTETracker(self._track_args, frame_rate=30)
        self._subject_tid  = None
        self._subject_lost = 0
        self._total_frames = 0
        self._wall_start   = time.time()
        self._timer        = _Timer()
        print("[ByteTrack] 🔄 追蹤狀態已重置")

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _infer(self, frame_bgr: np.ndarray, img_info: dict):
        """Run YOLOX inference on a BGR frame (mirrors Predictor.inference)."""
        img, ratio = self._preproc(frame_bgr, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio

        t = self._torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            t = t.half()

        self._timer.tic()
        with self._torch.no_grad():
            outputs = self.model(t)
            outputs = self._postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
        return outputs

    def _extract_subject_crop(
        self,
        frame_bgr: np.ndarray,
        vis_tlwhs: list,
    ) -> Optional[np.ndarray]:
        """
        Crop the subject region from the frame and convert to RGB.
        Returns None when no valid subject is tracked — callers should
        skip pushing to LiveCC in that case to avoid empty-frame backlog.
        """
        h, w = frame_bgr.shape[:2]
        if vis_tlwhs:
            x, y, bw, bh = vis_tlwhs[0]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(w, int(x + bw))
            y2 = min(h, int(y + bh))
            if x2 > x1 and y2 > y1:
                crop_bgr = frame_bgr[y1:y2, x1:x2]
                return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        # No valid subject — return None to suppress LiveCC push
        return None
