"""Template matching for basketball result banners.

Templates live in ``assets/result_banners/`` (scored.png, out_of_bounds.png,
home.png, away.png). Matching is grayscale ``TM_CCOEFF_NORMED`` over the centre
ROI of the gameplay (right) view.

The search scales each template so its **width becomes a fraction of the ROI
width**, rather than a multiple of the template's own pixel size. This makes
detection independent of both the capture resolution and how large the template
was cropped — the only thing that stays roughly constant across renders is how
much of the screen the banner occupies.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_LOG = logging.getLogger(__name__)

# Project root: …/src/miis_broadcast/core/models/this_file.py → parents[4]
_DEFAULT_DIR = (
    Path(__file__).resolve().parents[4] / "assets" / "result_banners"
)


def default_template_dir() -> Path:
    return _DEFAULT_DIR


class BannerTemplateMatcher:
    """Match ``Scored!`` / ``Out of Bounds!`` / ``Home`` / ``Away`` templates."""

    def __init__(
        self,
        template_dir: Optional[Path | str] = None,
        *,
        score_threshold: float = 0.55,
        side_threshold: float = 0.55,
        work_width: int = 420,
        kind_width_fracs: Optional[np.ndarray] = None,
        side_width_fracs: Optional[np.ndarray] = None,
    ) -> None:
        self.template_dir = Path(template_dir) if template_dir else _DEFAULT_DIR
        self.score_threshold = float(score_threshold)
        self.side_threshold = float(side_threshold)
        # Cap the ROI working width so matchTemplate cost stays bounded even for
        # 1080p/4K frames (this runs on the GUI thread).
        self.work_width = int(work_width)
        # Banner width as a fraction of the kind ROI width. Observed live:
        # "Scored!" ~0.34, "Out of Bounds!" ~0.68 → cover 0.22–0.85 finely.
        # A fine step matters: TM_CCOEFF_NORMED on sharp text drops sharply once
        # the scale is off by more than a few percent.
        self.kind_width_fracs = (
            kind_width_fracs
            if kind_width_fracs is not None
            else np.linspace(0.22, 0.85, 22, dtype=np.float64)
        )
        # Side word as a fraction of the (narrower) side ROI width (~0.6 live).
        self.side_width_fracs = (
            side_width_fracs
            if side_width_fracs is not None
            else np.linspace(0.35, 0.9, 16, dtype=np.float64)
        )
        self._templates: dict[str, np.ndarray] = {}
        self._load()

    @property
    def available(self) -> bool:
        return "scored" in self._templates and "out_of_bounds" in self._templates

    def _load(self) -> None:
        self._templates.clear()
        mapping = {
            "scored": "scored.png",
            "out_of_bounds": "out_of_bounds.png",
            "home": "home.png",
            "away": "away.png",
        }
        for key, filename in mapping.items():
            path = self.template_dir / filename
            if not path.is_file():
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                continue
            self._templates[key] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _LOG.info(
                "[BannerTM] loaded %s %sx%s from %s",
                key, img.shape[1], img.shape[0], path,
            )
        if not self.available:
            _LOG.warning(
                "[BannerTM] templates missing in %s — colour fallback only",
                self.template_dir,
            )

    def _prep_roi(self, roi_bgr: np.ndarray) -> np.ndarray:
        """Grayscale + downscale the ROI to ``work_width`` to bound match cost."""
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        w = gray.shape[1]
        if w <= self.work_width or w == 0:
            return gray
        factor = self.work_width / float(w)
        new_size = (self.work_width, max(1, int(round(gray.shape[0] * factor))))
        return cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)

    def _best_match(
        self,
        roi_bgr: np.ndarray,
        template_gray: np.ndarray,
        width_fracs: np.ndarray,
    ) -> float:
        """Best TM_CCOEFF_NORMED, scaling template width to fractions of ROI.

        Coarse-to-fine: sample every 3rd fraction, then refine around the best
        with its immediate neighbours. Keeps the fine grid's scale accuracy at
        roughly half the matchTemplate calls (this runs on the GUI thread).
        """
        if roi_bgr.size == 0 or template_gray.size == 0:
            return -1.0
        image = self._prep_roi(roi_bgr)
        ih, iw = image.shape[:2]
        th0, tw0 = template_gray.shape[:2]
        if tw0 == 0 or th0 == 0:
            return -1.0
        aspect = th0 / float(tw0)

        def score_at(frac: float) -> float:
            tw = int(round(iw * float(frac)))
            th = int(round(tw * aspect))
            if tw < 8 or th < 8 or th >= ih or tw >= iw:
                return -1.0
            interp = cv2.INTER_AREA if tw < tw0 else cv2.INTER_LINEAR
            resized = cv2.resize(template_gray, (tw, th), interpolation=interp)
            return float(cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED).max())

        n = len(width_fracs)
        coarse_idx = list(range(0, n, 3))
        if coarse_idx[-1] != n - 1:
            coarse_idx.append(n - 1)
        best = -1.0
        best_i = coarse_idx[0]
        for i in coarse_idx:
            s = score_at(float(width_fracs[i]))
            if s > best:
                best, best_i = s, i
        for i in (best_i - 2, best_i - 1, best_i + 1, best_i + 2):
            if 0 <= i < n:
                best = max(best, score_at(float(width_fracs[i])))
        return best

    def match_kind(
        self, gameplay_bgr: np.ndarray
    ) -> tuple[Optional[str], float, float]:
        """Return ``(kind, score_conf, oob_conf)`` for the centre banner ROI.

        kind is ``"score"``, ``"out_of_bounds"``, or None.
        """
        if not self.available:
            return None, -1.0, -1.0
        height, width = gameplay_bgr.shape[:2]
        roi = gameplay_bgr[
            int(height * 0.30) : int(height * 0.66),
            int(width * 0.02) : int(width * 0.98),
        ]
        scored_c = self._best_match(
            roi, self._templates["scored"], self.kind_width_fracs
        )
        oob_c = self._best_match(
            roi, self._templates["out_of_bounds"], self.kind_width_fracs
        )
        kind: Optional[str] = None
        if scored_c >= self.score_threshold or oob_c >= self.score_threshold:
            if scored_c >= oob_c and scored_c >= self.score_threshold:
                kind = "score"
            elif oob_c >= self.score_threshold:
                kind = "out_of_bounds"
        return kind, scored_c, oob_c

    def match_side(self, gameplay_bgr: np.ndarray) -> tuple[Optional[str], float]:
        """Return ``("home"|"away"|None, confidence)`` from side-word templates."""
        height, width = gameplay_bgr.shape[:2]
        roi = gameplay_bgr[
            int(height * 0.55) : int(height * 0.80),
            int(width * 0.12) : int(width * 0.88),
        ]
        home_c = -1.0
        away_c = -1.0
        if "home" in self._templates:
            home_c = self._best_match(
                roi, self._templates["home"], self.side_width_fracs
            )
        if "away" in self._templates:
            away_c = self._best_match(
                roi, self._templates["away"], self.side_width_fracs
            )
        if home_c < self.side_threshold and away_c < self.side_threshold:
            return None, max(home_c, away_c)
        if home_c >= away_c:
            return "home", home_c
        return "away", away_c
