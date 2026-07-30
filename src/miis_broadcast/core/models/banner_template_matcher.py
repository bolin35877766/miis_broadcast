"""Template matching for basketball result banners.

Templates live in ``assets/result_banners/`` (scored.png, out_of_bounds.png,
shot_clock_violation.png, home.png, away.png) and are tight text crops taken
from the LiveCC gameplay view (currently **640×480 / 4:3**). Matching is
grayscale ``TM_CCOEFF_NORMED`` over the banner band of that view.

The search scales each template so its **width becomes a fraction of the ROI
width**, rather than a multiple of the template's own pixel size. This makes
detection independent of absolute pixel size — the only thing that stays
roughly constant across renders is how much of the screen the banner occupies.

Geometry measured on 640×480 gameplay captures (fractions of the view):

    banner text   y 0.42-0.53,  width 0.31 (Scored!) … 0.70 (Shot Clock Violation!)
    Home / Away   y 0.61-0.70,  width 0.22, centred at x 0.47

The templates must keep the render's own width:height ratio. Crops taken from a
different aspect ratio (e.g. 16:9) stretch the glyphs on 4:3 and collapse every
confidence into an indistinguishable 0.3-0.55 band.
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

_KIND_KEYS = ("score", "out_of_bounds", "shot_clock_violation")


def default_template_dir() -> Path:
    return _DEFAULT_DIR


class BannerTemplateMatcher:
    """Match result banners: Scored! / Out of Bounds! / Shot Clock Violation!."""

    def __init__(
        self,
        template_dir: Optional[Path | str] = None,
        *,
        score_threshold: float = 0.70,
        side_threshold: float = 0.75,
        work_width: int = 420,
        kind_width_fracs: Optional[np.ndarray] = None,
        side_width_fracs: Optional[np.ndarray] = None,
    ) -> None:
        self.template_dir = Path(template_dir) if template_dir else _DEFAULT_DIR
        self.score_threshold = float(score_threshold)
        self.side_threshold = float(side_threshold)
        # Cap the ROI working width so matchTemplate cost stays bounded
        # (this runs on the GUI thread). LiveCC frames are 640×480; 420 keeps
        # a little headroom if a higher-res preview is scanned instead.
        self.work_width = int(work_width)
        # Banner width as a fraction of the kind ROI width. Measured on 640×480
        # captures: "Scored!" ~0.34, "Out of Bounds!" ~0.66,
        # "Shot Clock Violation!" ~0.76.
        self.kind_width_fracs = (
            kind_width_fracs
            if kind_width_fracs is not None
            else np.linspace(0.26, 0.90, 22, dtype=np.float64)
        )
        # Home / Away occupies ~0.37 of the side ROI width.
        self.side_width_fracs = (
            side_width_fracs
            if side_width_fracs is not None
            else np.linspace(0.26, 0.58, 12, dtype=np.float64)
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
            "shot_clock_violation": "shot_clock_violation.png",
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
                "[BannerTM] templates missing in %s — banner detection disabled",
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
    ) -> tuple[Optional[str], dict[str, float]]:
        """Return ``(kind, confidences)`` for the centre banner ROI.

        kind is ``"score"``, ``"out_of_bounds"``, ``"shot_clock_violation"``, or None.
        """
        confs = {key: -1.0 for key in _KIND_KEYS}
        if not self.available:
            return None, confs
        height, width = gameplay_bgr.shape[:2]
        roi = gameplay_bgr[
            int(height * 0.36) : int(height * 0.60),
            int(width * 0.04) : int(width * 0.96),
        ]
        template_key = {
            "score": "scored",
            "out_of_bounds": "out_of_bounds",
            "shot_clock_violation": "shot_clock_violation",
        }
        for kind_key, tmpl_key in template_key.items():
            tmpl = self._templates.get(tmpl_key)
            if tmpl is None:
                continue
            confs[kind_key] = self._best_match(roi, tmpl, self.kind_width_fracs)

        # Aligned templates put a real banner at 0.83-0.98 and leave cluttered
        # gameplay below 0.55, so a single threshold separates them. A relaxed
        # "best beats the rest" fallback used to live here and fired the long
        # orange templates on plain scenery, including on the first frame after
        # Start.
        best_kind, best_c = max(confs.items(), key=lambda item: item[1])
        return (best_kind if best_c >= self.score_threshold else None), confs

    def match_side(self, gameplay_bgr: np.ndarray) -> tuple[Optional[str], float]:
        """Return ``("home"|"away"|None, confidence)`` from side-word templates."""
        height, width = gameplay_bgr.shape[:2]
        roi = gameplay_bgr[
            int(height * 0.56) : int(height * 0.80),
            int(width * 0.20) : int(width * 0.80),
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
