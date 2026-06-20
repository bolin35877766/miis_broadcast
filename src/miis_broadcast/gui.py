# src/miis_broadcast/gui.py

from __future__ import annotations

import json
import logging
import os
import time
import re
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
# from .workers.chatterbox_tts import ChatterboxTTSWorker  # [ChatterBox disabled]
from .widgets.text_output import TextOutputWidget
from .workers.livecc import LiveCCWorker, LiveCCCameraWorker
from .workers.gemini import GeminiWorker
from .workers.openai_tts import OpenAITTSWorker
from .workers.obs_input import OBSCameraThread
from .workers.camera_bytetrack import CameraByteTrackThread
from .workers.dual_source import DualSourceCameraThread
from .workers.free_switch import FreeSwitchCameraThread, SOURCE_WEBCAM, SOURCE_VR, SOURCE_DUAL
from .audience.livekit_publisher import AudiencePublisher
from .audience.token_server import AudienceTokenServer
# LiveCCWorker / LiveCCCameraWorker are imported lazily only when not using client-only mode
# so this process never loads the VLM on a thin client.
from .core.prompt.prompt_manager import PromptManager
from .core.match_tracker import match_tracker
from .core.utils.session_logger import SessionLogger
from .core.utils.audio_recorder import AudioRecorder
from .core.utils.gpu_telemetry import (
    cuda_vram_snapshot,
    vram_log_suffix,
    vram_log_suffix_from_wire,
)
from .network.client import SocketClientRunner
from collections import deque

# ============================================================
# High-DPI / Scaling (Must be configured before QApplication for best effect)
# ============================================================

def _configure_qt_highdpi() -> None:
    if QtWidgets.QApplication.instance() is None:
        try:
            QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        except Exception:
            pass
        try:
            QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        except Exception:
            pass

    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")


def _find_project_root(start: Path) -> Path:
    """
    Look up from the current file to find the directory containing configs/ 
    as the project root. This makes the path robust to moves.
    """
    p = start.resolve()
    for parent in [p] + list(p.parents):
        if (parent / "configs").exists():
            return parent
    # Fallback: go back to the parent of src
    for parent in p.parents:
        if parent.name == "src":
            return parent.parent
    return p.parent


_configure_qt_highdpi()


# ============================================================
# Threads
# ============================================================

class CameraThread(QtCore.QThread):
    signal_frame = QtCore.Signal(np.ndarray)
    signal_error = QtCore.Signal(str)

    def __init__(self, camera_index: int = 0, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.camera_index = camera_index
        self._stop_requested = False

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            self.signal_error.emit(f"Failed to open camera (Index: {self.camera_index})")
            return

        while not self._stop_requested:
            ret, frame_bgr = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.signal_frame.emit(frame_rgb)
            time.sleep(0.033)

        cap.release()

    @QtCore.Slot()
    def requestStop(self) -> None:
        self._stop_requested = True


class VideoThread(QtCore.QThread):
    signal_video_loaded = QtCore.Signal(int, float)
    signal_frame = QtCore.Signal(np.ndarray, int, float)
    signal_video_ended = QtCore.Signal()
    signal_invalid_video = QtCore.Signal(str)

    def __init__(self, video_path: str, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self.video_path = video_path
        self._stop_requested = False
        self._seek_requested = False
        self._seek_frame_idx = 0

    def run(self) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.signal_invalid_video.emit(f"Failed to open video: {self.video_path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.signal_video_loaded.emit(frame_count, fps)
        delay_sec = 1.0 / fps
        frame_idx = 0
        last_time = time.time()

        while not self._stop_requested:
            if self._seek_requested:
                cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_frame_idx)
                frame_idx = self._seek_frame_idx
                self._seek_requested = False

            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.signal_frame.emit(frame_rgb, frame_idx, fps)
            frame_idx += 1

            now = time.time()
            elapsed = now - last_time
            sleep_time = delay_sec - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

        cap.release()
        self.signal_video_ended.emit()

    @QtCore.Slot()
    def requestStop(self) -> None:
        self._stop_requested = True

    @QtCore.Slot(int)
    def requestSeek(self, frame_idx: int) -> None:
        self._seek_requested = True
        self._seek_frame_idx = max(0, frame_idx)


# ============================================================
# UI Components
# ============================================================

class VideoPanel(QtWidgets.QWidget):
    seekRequested = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        self.label_video = QtWidgets.QLabel("Waiting for input...")
        self.label_video.setAlignment(QtCore.Qt.AlignCenter)

        # Use Ignored so the label doesn't squeeze the splitter (stabler on small screens)
        self.label_video.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.label_video.setMinimumSize(0, 0)

        self.label_video.setStyleSheet("""
            QLabel {
                color: #9a9a9a;
                background: #1e1e1e;
                border: 2px dashed #444;
                border-radius: 12px;
            }
        """)
        layout.addWidget(self.label_video, stretch=1)

        control_layout = QtWidgets.QHBoxLayout()
        control_layout.setSpacing(12)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.setEnabled(False)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal { height: 10px; background: #333; border-radius: 5px; }
            QSlider::handle:horizontal { background: #3a86ff; width: 26px; margin: -9px 0; border-radius: 13px; }
        """)

        self.lbl_time = QtWidgets.QLabel("00:00 / 00:00")
        self.lbl_time.setMinimumWidth(150)
        self.lbl_time.setStyleSheet("color: #d0d0d0; font-family: monospace; font-weight: 600;")

        control_layout.addWidget(self.slider, stretch=1)
        control_layout.addWidget(self.lbl_time, stretch=0)
        layout.addLayout(control_layout)

        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.fps = 30.0
        self.total_time_str = "00:00"
        self._last_frame_rgb: np.ndarray | None = None
        self._last_frame_is_bgr: bool = False

    @QtCore.Slot(int)
    def on_slider_moved(self, value: int) -> None:
        if self.slider.isEnabled():
            self.seekRequested.emit(value)

    def update_frame(self, frame: np.ndarray, is_bgr: bool = False) -> None:
        self._last_frame_rgb = frame
        self._last_frame_is_bgr = is_bgr

        w_label = max(1, self.label_video.width())
        h_label = max(1, self.label_video.height())

        h, w, c = frame.shape
        fmt = QtGui.QImage.Format.Format_BGR888 if is_bgr else QtGui.QImage.Format.Format_RGB888
        # .copy() ensures the QImage owns its data before the worker overwrites the buffer
        qimg = QtGui.QImage(frame.data, w, h, w * c, fmt).copy()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            w_label, h_label,
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.FastTransformation,
        )
        self.label_video.setPixmap(pix)

    def set_duration(self, frame_count: int, fps: float) -> None:
        self.slider.setEnabled(True)
        self.slider.setRange(0, max(0, frame_count - 1))
        self.fps = fps
        self.total_time_str = MainWindow.fmt_time_ms(frame_count / fps * 1000.0)
        self.lbl_time.setText(f"00:00 / {self.total_time_str}")

    def set_position(self, frame_idx: int, fps: float) -> None:
        if self.slider.isEnabled() and not self.slider.isSliderDown():
            self.slider.setValue(frame_idx)
        cur_time = MainWindow.fmt_time_ms(frame_idx / fps * 1000.0)
        self.lbl_time.setText(f"{cur_time} / {self.total_time_str}")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._last_frame_rgb is not None:
            QtCore.QTimer.singleShot(
                0,
                lambda: self.update_frame(self._last_frame_rgb, self._last_frame_is_bgr),
            )


class ControlPanel(QtWidgets.QWidget):
    requestOpenVideo       = QtCore.Signal()
    requestOpenCamera      = QtCore.Signal()
    requestOpenCameraTrack = QtCore.Signal()
    requestOpenOBS         = QtCore.Signal()
    requestOpenDualSync    = QtCore.Signal()
    requestOpenFreeSwitch  = QtCore.Signal()
    requestSwitchSource    = QtCore.Signal(str)
    freeSwitchAutoCycleToggled = QtCore.Signal(bool)
    requestLoadContext     = QtCore.Signal()
    requestStart           = QtCore.Signal()
    requestRemoteConnect   = QtCore.Signal(str, int)
    requestRemoteDisconnect = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(600)
        self.setup_ui()

    def apply_settings_metrics(self) -> None:
        """Rebuild label/value column widths and combo heights from current app font.

        Stylesheet fixed min-heights clash with enlarged fonts and clip combo text.
        Call after startup and whenever UI font scale changes.
        """
        if not hasattr(self, "cmb_tts"):
            return
        fm = self.fontMetrics()
        combo_h = max(34, fm.height() + 10)
        self._settings_row_min_h = combo_h
        for c in (self.cmb_tts, self.cmb_style, self.cmb_voice):
            c.setMinimumHeight(combo_h)
        titles = (
            "TTS Mode:",
            "Style:",
            "Voice:",
            "Speed:",
            "Exaggeration:",
            "CFG:",
        )
        lw = max(fm.horizontalAdvance(t) for t in titles) + 12
        for lb in (
            self.l_tts,
            self.l_style,
            self.l_voice,
            self.l_speed,
            self.l_exag,
            self.l_cfg,
        ):
            lb.setMinimumWidth(lw)
            lb.setMaximumWidth(lw)
        vw = fm.horizontalAdvance("1.55x") + 18
        for v in (
            self.lbl_speed_val,
            self.lbl_exag_val,
            self.lbl_cfg_val,
        ):
            v.setMinimumWidth(vw)
        for row in getattr(self, "_settings_rows", {}).values():
            row.setMinimumHeight(combo_h)

    def apply_source_metrics(self) -> None:
        """Keep Source button heights in sync with current app font size."""
        if not hasattr(self, "btn_offline"):
            return
        fm = self.fontMetrics()
        main_h = max(42, fm.height() + 18)
        switch_h = max(28, fm.height() + 8)
        for btn in (self.btn_offline, self.btn_online, self.btn_open_remote):
            btn.setMinimumHeight(main_h)
            btn.setMaximumHeight(main_h)
        for btn in (self.btn_sw_webcam, self.btn_sw_vr, self.btn_sw_dual, self.btn_fs_auto_cycle):
            btn.setMinimumHeight(switch_h)
            btn.setMaximumHeight(switch_h)

    def setup_ui(self) -> None:
        # QScrollArea with both scrollbars hidden — no visible draggable bar,
        # but layout always has enough room so widgets never overlap.
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        _scroll = QtWidgets.QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        _scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        outer_layout.addWidget(_scroll)

        _inner = QtWidgets.QWidget()
        _scroll.setWidget(_inner)

        layout = QtWidgets.QVBoxLayout(_inner)
        layout.setSpacing(16)
        layout.setContentsMargins(14, 14, 14, 14)

        # Source
        grp_source = QtWidgets.QGroupBox("影像來源 (Source)")
        self.grp_source = grp_source
        v_src = QtWidgets.QVBoxLayout(grp_source)
        v_src.setSpacing(10)
        v_src.setContentsMargins(12, 16, 12, 10)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(12)

        btn_style = """
            QPushButton {
                background-color: #505050;
                border-radius: 9px;
                padding: 3px 14px;
                font-weight: 650;
                text-align: center;
            }
            QPushButton:hover { background-color: #606060; }
            QPushButton::menu-indicator { image: none; }
        """

        menu_style = """
            QMenu {
                background-color: #3a3a3a;
                color: #f0f0f0;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 4px;
            }
            QMenu::item {
                padding: 8px 24px 8px 16px;
                border-radius: 6px;
            }
            QMenu::item:selected {
                background-color: #3a86ff;
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background: #555;
                margin: 4px 8px;
            }
            QMenu::item:disabled {
                color: #777;
            }
        """

        # ── Button 1: Offline video input ────────────────────────────────
        self.btn_offline = QtWidgets.QPushButton("Offline")
        self.btn_offline.setStyleSheet(btn_style)
        self.btn_offline.clicked.connect(lambda: self.requestOpenVideo.emit())

        # ── Button 2: Online live input (dropdown with 3 sub-modes) ──────
        self.btn_online = QtWidgets.QPushButton("Online  ▾")
        self.btn_online.setStyleSheet(btn_style)

        menu_online = QtWidgets.QMenu(self.btn_online)
        menu_online.setStyleSheet(menu_style)

        # Mode 1: Webcam without tracking
        menu_online.addAction("📷  Webcam",             lambda: self.requestOpenCamera.emit())
        # Mode 2: Webcam with ByteTrack subject tracking
        menu_online.addAction("🎯  Webcam + Tracking",  lambda: self.requestOpenCameraTrack.emit())

        menu_online.addSeparator()

        # Mode 3: VR via OBS Virtual Camera (no tracking needed)
        menu_online.addAction("🥽  VR (OBS Virtual Camera)", lambda: self.requestOpenOBS.emit())

        # Mode 4: Dual source sync — Webcam (idx 0) + VR/OBS (idx 5), side-by-side
        menu_online.addAction("🎮  VR & Webcam (Sync)",      lambda: self.requestOpenDualSync.emit())

        menu_online.addSeparator()

        # Mode 5: Free Switch — both cameras always running, switch without reconnect
        menu_online.addAction("🔀  Free Switch",             lambda: self.requestOpenFreeSwitch.emit())

        self.btn_online.setMenu(menu_online)

        btn_row.addWidget(self.btn_offline)
        btn_row.addWidget(self.btn_online)

        v_src.addLayout(btn_row)

        # Remote: entry button in Source; host/port/enable live in a dialog
        self.btn_open_remote = QtWidgets.QPushButton("連線遠端伺服器")
        self.btn_open_remote.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                font-weight: 700;
                border-radius: 9px;
                padding: 3px 14px;
            }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #555; color: #999; }
        """)
        v_src.addWidget(self.btn_open_remote)

        self.lbl_source_sub = QtWidgets.QLabel("Status: —")
        self.lbl_source_sub.setStyleSheet("color: #b5b5b5; font-size: 11px;")
        self.lbl_source_sub.setWordWrap(True)
        v_src.addWidget(self.lbl_source_sub)

        self.lbl_remote_badge = QtWidgets.QLabel("● 未連線")
        self.lbl_remote_badge.setStyleSheet("color: #888; font-size: 11px;")
        v_src.addWidget(self.lbl_remote_badge)

        # ── Free Switch: single row — equal-width buttons (no extra label row → no scroll)
        self.free_switch_bar = QtWidgets.QWidget()
        _bar_row = QtWidgets.QHBoxLayout(self.free_switch_bar)
        _bar_row.setContentsMargins(0, 2, 0, 0)
        _bar_row.setSpacing(5)

        _sw_style = """
            QPushButton {
                border-radius: 7px;
                padding: 5px 6px;
                font-weight: 600;
                font-size: 11px;
                background-color: #434343;
                color: #eaeaea;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:checked {
                background-color: #2962ff;
                color: white;
            }
            QPushButton:disabled {
                background-color: #2d2d2d;
                color: #666;
            }
            QPushButton:checked:disabled {
                background-color: #1a3f9e;
                color: #aac0ff;
            }
        """
        _exp = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.btn_sw_webcam = QtWidgets.QPushButton("鏡頭")
        self.btn_sw_vr     = QtWidgets.QPushButton("VR")
        self.btn_sw_dual   = QtWidgets.QPushButton("拼接")
        for _btn, _tip in (
            (self.btn_sw_webcam, "實體鏡頭 (Webcam)，與伺服器送出之畫面一致"),
            (self.btn_sw_vr, "OBS 虛擬鏡頭 (VR／遊戲畫面)"),
            (self.btn_sw_dual, "左右並列：Webcam + VR（1280×480）"),
        ):
            _btn.setCheckable(True)
            _btn.setToolTip(_tip)
            _btn.setStyleSheet(_sw_style)
            _btn.setSizePolicy(_exp)
            _btn.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

        self.btn_sw_webcam.clicked.connect(lambda: self.requestSwitchSource.emit("webcam"))
        self.btn_sw_vr.clicked.connect(    lambda: self.requestSwitchSource.emit("vr"))
        self.btn_sw_dual.clicked.connect(  lambda: self.requestSwitchSource.emit("dual"))

        self.btn_fs_auto_cycle = QtWidgets.QPushButton("10s輪播")
        self.btn_fs_auto_cycle.setCheckable(True)
        self.btn_fs_auto_cycle.setToolTip(
            "每 10 秒自動依序切換：鏡頭 → VR → 拼接。啟動後立刻切到下一個；再按一次關閉輪播。"
        )
        self.btn_fs_auto_cycle.setStyleSheet(_sw_style + """
            QPushButton:checked {
                background-color: #e65100;
                color: white;
            }
        """)
        self.btn_fs_auto_cycle.setSizePolicy(_exp)
        self.btn_fs_auto_cycle.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.btn_fs_auto_cycle.toggled.connect(self._forward_free_switch_auto_cycle_toggled)

        _bar_row.addWidget(self.btn_sw_webcam, 1)
        _bar_row.addWidget(self.btn_sw_vr, 1)
        _bar_row.addWidget(self.btn_sw_dual, 1)
        _bar_row.addWidget(self.btn_fs_auto_cycle, 1)

        self.free_switch_bar.setVisible(False)
        v_src.addWidget(self.free_switch_bar)

        ctx_row = QtWidgets.QHBoxLayout()
        ctx_row.setSpacing(8)

        self.btn_context = QtWidgets.QPushButton("載入比賽資訊")
        self.btn_context.setStyleSheet(btn_style)

        self.lbl_context = QtWidgets.QLabel("未載入")
        self.lbl_context.setStyleSheet("color: #888888;")
        self.lbl_context.setWordWrap(True)

        ctx_row.addWidget(self.btn_context)
        ctx_row.addWidget(self.lbl_context, stretch=1)
        v_src.addLayout(ctx_row)


        layout.addWidget(grp_source)
        self.apply_source_metrics()

        # Settings — one row widget per logical row (QHBoxLayout inside QVBoxLayout).
        # QGridLayout + addWidget(..., alignment=...) can mis-bind on some bindings and pile widgets up.
        grp_settings = QtWidgets.QGroupBox("推論設定 (Settings)")
        v_settings = QtWidgets.QVBoxLayout(grp_settings)
        v_settings.setContentsMargins(14, 22, 14, 12)
        v_settings.setSpacing(8)

        self._settings_rows: dict[str, QtWidgets.QWidget] = {}

        def _settings_row_slider(
            row_key: str,
            lbl_w: QtWidgets.QWidget,
            slider_w: QtWidgets.QWidget,
            val_lbl: QtWidgets.QWidget,
        ) -> None:
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(12)
            lbl_w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            h.addWidget(lbl_w, stretch=0)
            h.addWidget(slider_w, stretch=1)
            h.addWidget(val_lbl, stretch=0)
            _rh = max(
                getattr(self, "_settings_row_min_h", 44),
                int(slider_w.sizeHint().height()),
            )
            row.setMinimumHeight(_rh)
            self._settings_rows[row_key] = row
            v_settings.addWidget(row)

        def _settings_row_combo(
            row_key: str,
            lbl_w: QtWidgets.QWidget,
            combo_w: QtWidgets.QWidget,
        ) -> None:
            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(12)
            lbl_w.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            combo_w.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Fixed,
            )
            h.addWidget(lbl_w, stretch=0)
            h.addWidget(combo_w, stretch=1)
            row.setMinimumHeight(getattr(self, "_settings_row_min_h", 44))
            self._settings_rows[row_key] = row
            v_settings.addWidget(row)

        lbl_style = "QLabel { color: #dedede; }"
        lbl_val_style = """
            QLabel { color: #c8c8c8; padding-left: 8px; }
        """

        def _make_lbl(text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet(lbl_style)
            return lbl

        def _apply_val_label(lbl: QtWidgets.QLabel) -> None:
            lbl.setStyleSheet(lbl_val_style)
            lbl.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        combo_style = """
            QComboBox {
                padding: 3px 10px;
                border-radius: 6px;
                background-color: #303030;
                border: 1px solid #484848;
            }
            QComboBox:hover { border-color: #5a5a5a; }
            QComboBox::drop-down {
                width: 20px;
                border: 0px;
            }
            QComboBox QAbstractItemView { 
                background-color: #333; 
                color: #fff;
                selection-background-color: #3a86ff;
                selection-color: white;
                outline: 0px;
            }
        """

        # [FIX] WSL 下拉選單修復 helper
        def _fix_combo_behavior(combo: QtWidgets.QComboBox):
            combo.setItemDelegate(QtWidgets.QStyledItemDelegate())
            # 強制在「按下」時就選取並關閉，避開 WSL 吃掉 MouseRelease 事件的問題
            combo.view().pressed.connect(lambda idx: (
                combo.setCurrentIndex(idx.row()),
                combo.hidePopup()
            ))

        # --- TTS Mode ---
        self.l_tts = _make_lbl("TTS Mode:")
        self.cmb_tts = QtWidgets.QComboBox()
        _fix_combo_behavior(self.cmb_tts)
        self.cmb_tts.addItem("不啟用 (Mute)", userData="none")
        self.cmb_tts.addItem("OpenAI TTS", userData="openai")
        self.cmb_tts.addItem("Local TTS", userData="local")
        self.cmb_tts.setCurrentIndex(1)
        self.cmb_tts.setStyleSheet(combo_style)

        # --- LiveCC style ---
        self.l_style = _make_lbl("Style:")
        self.cmb_style = QtWidgets.QComboBox()
        _fix_combo_behavior(self.cmb_style)
        self.cmb_style.setStyleSheet(combo_style)

        # --- OpenAI: Voice ---
        self.l_voice = _make_lbl("Voice:")
        self.cmb_voice = QtWidgets.QComboBox()
        _fix_combo_behavior(self.cmb_voice)
        self.cmb_voice.setStyleSheet(combo_style)
        for v in ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]:
            self.cmb_voice.addItem(v, userData=v)
        self.cmb_voice.setCurrentText("coral")

        # --- OpenAI: Language (English / Traditional Chinese) ---
        self.l_tts_lang = _make_lbl("Language:")
        self.cmb_tts_lang = QtWidgets.QComboBox()
        _fix_combo_behavior(self.cmb_tts_lang)
        self.cmb_tts_lang.addItem("English", userData="en")
        self.cmb_tts_lang.addItem("繁體中文", userData="zh")
        self.cmb_tts_lang.setCurrentIndex(0)
        self.cmb_tts_lang.setStyleSheet(combo_style)

        # --- OpenAI: Speed slider ---
        self.l_speed = _make_lbl("Speed:")
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_speed.setRange(25, 150)  # 0.25x ~ 1.5x
        self.slider_speed.setValue(100)
        self.lbl_speed_val = QtWidgets.QLabel("1.0x")
        _apply_val_label(self.lbl_speed_val)

        # --- Local: Exaggeration slider (0.2~1.2) ---
        self.l_exag = _make_lbl("Exaggeration:")
        self.slider_exag = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_exag.setRange(20, 120)
        self.slider_exag.setValue(80)
        self.lbl_exag_val = QtWidgets.QLabel("0.8")
        _apply_val_label(self.lbl_exag_val)

        # --- Local: CFG slider (0.2~1.2) ---
        self.l_cfg = _make_lbl("CFG:")
        self.slider_cfg = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cfg.setRange(20, 120)
        self.slider_cfg.setValue(70)
        self.lbl_cfg_val = QtWidgets.QLabel("0.7")
        _apply_val_label(self.lbl_cfg_val)

        _sp_exp = QtWidgets.QSizePolicy.Policy.Expanding
        _sp_fix = QtWidgets.QSizePolicy.Policy.Fixed
        for _s in (
            self.slider_speed,
            self.slider_exag,
            self.slider_cfg,
        ):
            _s.setSizePolicy(_sp_exp, _sp_fix)

        _settings_row_combo("tts", self.l_tts, self.cmb_tts)
        _settings_row_combo("style", self.l_style, self.cmb_style)
        _settings_row_combo("voice", self.l_voice, self.cmb_voice)
        _settings_row_combo("tts_lang", self.l_tts_lang, self.cmb_tts_lang)
        _settings_row_slider("speed", self.l_speed, self.slider_speed, self.lbl_speed_val)
        _settings_row_slider("exag", self.l_exag, self.slider_exag, self.lbl_exag_val)
        _settings_row_slider("cfg", self.l_cfg, self.slider_cfg, self.lbl_cfg_val)

        layout.addWidget(grp_settings)

        self.apply_settings_metrics()

        # Action
        grp_action = QtWidgets.QGroupBox("操作 (Action)")
        v_act = QtWidgets.QVBoxLayout(grp_action)
        v_act.setContentsMargins(14, 18, 14, 12)

        self.btn_start = QtWidgets.QPushButton("開始播報 (Start)")
        self.btn_start.setEnabled(False)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #3a86ff;
                color: white;
                font-weight: 750;
                border-radius: 12px;
                padding: 14px 10px;
            }
            QPushButton:hover { background-color: #2667cc; }
            QPushButton:disabled { background-color: #555; color: #999; }
            QPushButton[active="true"] { background-color: #ef233c; border: 2px solid #ff9999; }
        """)
        v_act.addWidget(self.btn_start)

        self.chk_record = QtWidgets.QCheckBox("錄音 (Record Audio)")
        self.chk_record.setChecked(False)
        self.chk_record.setStyleSheet("color: #c0c0c0; padding-top: 4px;")
        v_act.addWidget(self.chk_record)

        layout.addWidget(grp_action)

        self._init_remote_server_dialog()
        layout.addStretch(1)
        # End of inner scroll container

        # Signals
        self.btn_context.clicked.connect(self.requestLoadContext.emit)
        self.btn_start.clicked.connect(self.requestStart.emit)

        self.slider_speed.valueChanged.connect(lambda v: self.lbl_speed_val.setText(f"{v/100:.1f}x"))
        self.slider_exag.valueChanged.connect(lambda v: self.lbl_exag_val.setText(f"{v/100:.1f}"))
        self.slider_cfg.valueChanged.connect(lambda v: self.lbl_cfg_val.setText(f"{v/100:.1f}"))

        # 模式切換顯示/隱藏
        self.cmb_tts.currentIndexChanged.connect(self._refresh_tts_controls_visibility)
        self._refresh_tts_controls_visibility()

        self.btn_open_remote.clicked.connect(self._show_remote_dialog)

    def _init_remote_server_dialog(self) -> None:
        # Host/port/enable/connection controls (no spin arrows on port)
        self._remote_dialog = QtWidgets.QDialog(self)
        self._remote_dialog.setWindowTitle("遠端伺服器 (Remote server)")
        self._remote_dialog.setWindowModality(QtCore.Qt.NonModal)
        # Minimize/close in title bar so the panel can be tucked away without blocking the main view
        _flags = (
            QtCore.Qt.Window
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        _flags &= ~QtCore.Qt.WindowContextHelpButtonHint
        self._remote_dialog.setWindowFlags(_flags)
        self._remote_dialog.setMinimumWidth(380)

        dlay = QtWidgets.QVBoxLayout(self._remote_dialog)
        dlay.setContentsMargins(16, 16, 16, 16)
        dlay.setSpacing(10)

        self.chk_remote = QtWidgets.QCheckBox("啟用遠端推論 (Use Remote Inference)")
        self.chk_remote.setStyleSheet("color: #dedede;")
        dlay.addWidget(self.chk_remote)

        self._remote_settings_widget = QtWidgets.QWidget()
        remote_form = QtWidgets.QFormLayout(self._remote_settings_widget)
        remote_form.setSpacing(8)
        remote_form.setContentsMargins(0, 0, 0, 0)
        remote_form.setLabelAlignment(QtCore.Qt.AlignRight)

        lbl_host_style = "QLabel { color: #dedede; }"

        self.edit_remote_host = QtWidgets.QLineEdit("127.0.0.1")
        self.edit_remote_host.setStyleSheet(
            "QLineEdit { background:#333; border-radius:6px; padding:4px 8px; }"
        )
        lbl_host = QtWidgets.QLabel("Host:")
        lbl_host.setStyleSheet(lbl_host_style)
        remote_form.addRow(lbl_host, self.edit_remote_host)

        self.edit_remote_port = QtWidgets.QLineEdit("9000")
        self.edit_remote_port.setValidator(QtGui.QIntValidator(1, 65535, self))
        self.edit_remote_port.setStyleSheet(
            "QLineEdit { background:#333; border-radius:6px; padding:4px 8px; }"
        )
        lbl_port = QtWidgets.QLabel("Port:")
        lbl_port.setStyleSheet(lbl_host_style)
        remote_form.addRow(lbl_port, self.edit_remote_port)

        dlay.addWidget(self._remote_settings_widget)

        self.btn_remote_connect = QtWidgets.QPushButton("連線 (Connect)")
        self.btn_remote_connect.setStyleSheet("""
            QPushButton {
                background-color: #2e7d32;
                color: white;
                font-weight: 700;
                border-radius: 8px;
                padding: 8px 10px;
            }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #555; color: #999; }
            QPushButton[connected="true"] {
                background-color: #b71c1c;
            }
            QPushButton[connected="true"]:hover { background-color: #c62828; }
        """)
        dlay.addWidget(self.btn_remote_connect)

        self.lbl_remote_status = QtWidgets.QLabel("● 未連線")
        self.lbl_remote_status.setStyleSheet("color: #888; font-size: 11px;")
        dlay.addWidget(self.lbl_remote_status)

        self._remote_settings_widget.setVisible(False)
        self.btn_remote_connect.setVisible(False)
        self.lbl_remote_status.setVisible(False)

        self.chk_remote.toggled.connect(self._on_remote_toggle)
        self.btn_remote_connect.clicked.connect(self._on_remote_connect_clicked)

    @QtCore.Slot()
    def _show_remote_dialog(self) -> None:
        if self._remote_dialog is not None:
            self._remote_dialog.show()
            self._remote_dialog.raise_()
            self._remote_dialog.activateWindow()

    # ---------------- ControlPanel Helpers ----------------

    def _on_remote_toggle(self, checked: bool) -> None:
        self._remote_settings_widget.setVisible(checked)
        self.btn_remote_connect.setVisible(checked)
        self.lbl_remote_status.setVisible(checked)
        if not checked:
            self.requestRemoteDisconnect.emit()

    def _on_remote_connect_clicked(self) -> None:
        connected = self.btn_remote_connect.property("connected") == True
        if connected:
            self.requestRemoteDisconnect.emit()
        else:
            host = self.edit_remote_host.text().strip() or "127.0.0.1"
            port = self.get_remote_port()
            self.requestRemoteConnect.emit(host, port)

    def set_remote_connected(self, connected: bool, status_text: str = "") -> None:
        if connected:
            self.btn_remote_connect.setText("中斷連線 (Disconnect)")
            self.btn_remote_connect.setProperty("connected", True)
            self.lbl_remote_status.setText(f"● {status_text or '已連線'}")
            self.lbl_remote_status.setStyleSheet("color: #66bb6a; font-size: 11px;")
            st = f"{self.get_remote_host()}:{self.get_remote_port()}"
            self.lbl_remote_badge.setText(f"● 已連線  {st}")
            self.lbl_remote_badge.setStyleSheet("color: #66bb6a; font-size: 12px;")
        else:
            self.btn_remote_connect.setText("連線 (Connect)")
            self.btn_remote_connect.setProperty("connected", False)
            self.lbl_remote_status.setText(f"● {status_text or '未連線'}")
            self.lbl_remote_status.setStyleSheet("color: #888; font-size: 11px;")
            t = status_text or "未連線"
            self.lbl_remote_badge.setText(f"● {t}")
            self.lbl_remote_badge.setStyleSheet("color: #888; font-size: 12px;")
        self.btn_remote_connect.style().unpolish(self.btn_remote_connect)
        self.btn_remote_connect.style().polish(self.btn_remote_connect)

    def set_remote_connecting(self) -> None:
        self.lbl_remote_status.setText("● 連線中…")
        self.lbl_remote_status.setStyleSheet("color: #ffa726; font-size: 11px;")
        self.lbl_remote_badge.setText("● 連線中…")
        self.lbl_remote_badge.setStyleSheet("color: #ffa726; font-size: 12px;")

    def is_remote_mode(self) -> bool:
        return self.chk_remote.isChecked()

    def get_remote_host(self) -> str:
        return self.edit_remote_host.text().strip() or "127.0.0.1"

    def get_remote_port(self) -> int:
        try:
            t = (self.edit_remote_port.text() or "9000").strip()
            v = int(t)
        except ValueError:
            v = 9000
        return max(1, min(65535, v))

    def _refresh_tts_controls_visibility(self) -> None:
        mode = self.get_tts_mode()

        show_openai = (mode == "openai")
        show_local = (mode == "local")
        rows = getattr(self, "_settings_rows", {})
        if rows:
            rows["voice"].setVisible(show_openai)
            rows["tts_lang"].setVisible(show_openai)
            rows["speed"].setVisible(show_openai)
            rows["exag"].setVisible(show_local)
            rows["cfg"].setVisible(show_local)
            return

        for w in (self.l_voice, self.cmb_voice,
                  self.l_tts_lang, self.cmb_tts_lang,
                  self.l_speed, self.slider_speed, self.lbl_speed_val):
            w.setVisible(show_openai)
        for w in (
            self.l_exag,
            self.slider_exag,
            self.lbl_exag_val,
            self.l_cfg,
            self.slider_cfg,
            self.lbl_cfg_val,
        ):
            w.setVisible(show_local)

    def set_tts_controls_enabled(self, enabled: bool) -> None:
        # Lock during inference to prevent state corruption
        self.cmb_tts.setEnabled(enabled)
        self.cmb_style.setEnabled(enabled)

        self.cmb_voice.setEnabled(enabled)
        self.cmb_tts_lang.setEnabled(enabled)
        self.slider_speed.setEnabled(enabled)

        self.slider_exag.setEnabled(enabled)
        self.slider_cfg.setEnabled(enabled)

    def get_tts_mode(self) -> str:
        return self.cmb_tts.currentData()

    def get_selected_style_key(self) -> str:
        return self.cmb_style.currentData()

    def get_selected_style_label(self) -> str:
        return self.cmb_style.currentText()

    def get_openai_voice(self) -> str:
        v = self.cmb_voice.currentData()
        return str(v) if v is not None else "coral"

    def get_tts_language(self) -> str:
        """Return 'en' or 'zh' based on the language combo selection."""
        v = self.cmb_tts_lang.currentData()
        return str(v) if v in ("en", "zh") else "en"

    def get_openai_speed(self) -> float:
        return float(self.slider_speed.value()) / 100.0

    def get_local_exaggeration(self) -> float:
        return float(self.slider_exag.value()) / 100.0

    def get_local_cfg(self) -> float:
        return float(self.slider_cfg.value()) / 100.0

    def set_status(self, text: str) -> None:
        self.lbl_source_sub.setText(f"Status: {text}")

    def set_free_switch_bar_visible(self, visible: bool, active_source: str = "webcam") -> None:
        """Show or hide the Free Switch source bar, and highlight the active button."""
        self.free_switch_bar.setVisible(visible)
        self.apply_source_metrics()
        if visible:
            self.highlight_switch_source(active_source)

    def highlight_switch_source(self, source: str) -> None:
        """Update which switch button appears active (checked/highlighted)."""
        self.btn_sw_webcam.setChecked(source == "webcam")
        self.btn_sw_vr.setChecked(    source == "vr")
        self.btn_sw_dual.setChecked(  source == "dual")

    @QtCore.Slot(bool)
    def _forward_free_switch_auto_cycle_toggled(self, checked: bool) -> None:
        # QPushButton.toggled -> Signal Forward: wire directly to another Signal() often fails
        # to invoke MainWindow slots in PySide6; emit explicitly.
        self.freeSwitchAutoCycleToggled.emit(checked)

    def set_start_button_state(self, running: bool) -> None:
        if running:
            self.btn_start.setText("Stop Broadcasting")
            self.btn_start.setProperty("active", True)
        else:
            self.btn_start.setText("Start Broadcasting")
            self.btn_start.setProperty("active", False)
        self.btn_start.style().unpolish(self.btn_start)
        self.btn_start.style().polish(self.btn_start)
        # Disable source buttons during inference to prevent switching mid-session.
        # free_switch_bar buttons remain enabled so the user can switch sources live.
        self.btn_offline.setEnabled(not running)
        self.btn_online.setEnabled(not running)
        self.btn_open_remote.setEnabled(not running)


# ============================================================
# Main Window
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    signal_start_livecc = QtCore.Signal(str, str, int)
    signal_start_camera_livecc = QtCore.Signal(str)

    # ✅ 用 signal 把設定丟到 tts thread，避免你直接 call slot 其實跑在主執行緒
    signal_tts_apply_settings = QtCore.Signal(str, float)
    signal_tts_warmup = QtCore.Signal()
    signal_tts_stop = QtCore.Signal()

    signal_tts_speak = QtCore.Signal(str, int, float, float, float, object)  # text, pri, ref_ts, start_t, stop_t, log_meta
    signal_tts_interrupt = QtCore.Signal()
    signal_local_tts_apply_settings = QtCore.Signal(float, float) # exag, cfg
    signal_local_tts_speak = QtCore.Signal(str)
    signal_local_tts_interrupt = QtCore.Signal()
    signal_local_tts_stop = QtCore.Signal()
    _signal_to_gemini = QtCore.Signal(float, float, object)
    # Fires when Gemini confirms a P1 event — used to reset LiveCC KV cache
    signal_p1_confirmed = QtCore.Signal()
    # P3 LiveCC description → GeminiBackgroundWorker.update_context (QueuedConnection)
    signal_livecc_context = QtCore.Signal(str)

    # Fast-path keyword sets
    _P1_KEYWORDS = frozenset({
        "scores", "scored",
        "makes the shot", "makes a shot", "makes it", "makes the basket",
        "made the shot", "made a shot",
        "swish", "swishes",
        "through the net", "drops through", "drops in", "goes in",
        "slam", "dunk", "dunks",
        "it's good", "it's in",
        "進球", "得分",
    })
    _P2_KEYWORDS = frozenset({
        "misses", "missed",
        "bounces off", "off the rim",
        "rebound", "rebounds",
        "out of bounds",
        "未進", "彈框", "籃板", "界外",
    })

    def __init__(self, configs: dict, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.configs = configs
        self.parseConfigs()

        self.session_logger = SessionLogger()
        self._audio_recorder = AudioRecorder()
        self.current_video_path: Optional[str] = None
        self.model_ready: bool = False
        self.mode = "file"
        self.is_inference_running = False
        self._livecc_run_id: int = 0

        self.video_thread: Optional[VideoThread] = None
        self.camera_thread: Optional[CameraThread] = None
        self.obs_thread: Optional[OBSCameraThread] = None
        self.obs_bytetrack_thread: Optional[CameraByteTrackThread] = None
        self.dual_sync_thread: Optional[DualSourceCameraThread] = None
        self.free_switch_thread: Optional[FreeSwitchCameraThread] = None
        self.video_fps: float = 30.0
        self.tts_mode: str = "none"
        self._use_gemini: bool = False
        self._last_tts_raw_text: str = ""      # raw text of last TTS emit (dedup)
        self._last_tts_emit_ts: float = 0.0    # wall-clock time of last TTS emit (dedup)
        self._tts_protect_until: float = 0.0   # wall-clock deadline: block lower-priority below this time
        self._tts_protected_priority: int = 5  # priority being protected until _tts_protect_until
        self._p1_filler_armed: bool = False    # True while zh-TW filler is playing (safe to hard-cut)
        self._post_p1_pending: bool = False     # True while waiting for 1.0s post-P1 silence
        self._pending_livecc_fragment: Optional[tuple] = None  # truncated "..." fragment awaiting stitching
        self._livecc_start_wall: float = 0.0   # wall-clock anchor for file-mode "frame appeared" latency (== LiveCC inference start)

        # Audience second-screen services (Free Switch mode only)
        self._audience_publisher: Optional[AudiencePublisher] = None
        self._audience_token_server: Optional[AudienceTokenServer] = None

        self.font_family = "Sans Serif"
        self.font_size = 14

        self.livecc_model = None
        self.prompt_manager: Optional[PromptManager] = None
        self._bytetrack_wrapper = None          # pre-loaded ByteTrackWrapper (set by background thread)
        self._bytetrack_preload_thread = None   # QThread that loads it

        # Remote inference state
        self._socket_runner: Optional[SocketClientRunner] = None
        self._remote_client_ram_timer: Optional[QtCore.QTimer] = None
        self._local_client_diag_samples: list = []
        self._busy_stopping_inference: bool = False

        remote_cfg = configs.get("remote", {})
        self._client_only = bool(remote_cfg.get("client_only", False))
        if self._client_only:
            print("[Main] client_only: local VLM not loaded; connect to remote server.")
        else:
            self._load_livecc_model()

        self._init_fonts()
        self._initUI()
        self._initTTSWorker()
        self._initGeminiWorker()

        # Pre-fill remote panel host/port from config
        if hasattr(self.control_panel, "edit_remote_host"):
            host = str(remote_cfg.get("host", "127.0.0.1"))
            port = int(remote_cfg.get("port", 9000))
            self.control_panel.edit_remote_host.setText(host)
            self.control_panel.edit_remote_port.setText(str(port))

        if self.configs.get("bytetrack"):
            QtCore.QTimer.singleShot(500, self._preload_bytetrack_model)

        self._playback_sec: float = 0.0
        self._broadcast_playback_sec: float = 0.0  # timeline sec since admin Start
        self._broadcast_start_wall: float = 0.0    # wall-clock anchor at admin Start (all modes)
        self._pending_segments = deque()  # items: (start_t, stop_t, text)

        self._subtitle_timer = QtCore.QTimer(self)
        self._subtitle_timer.setInterval(50)  # 20 FPS 更新足夠
        self._subtitle_timer.timeout.connect(self._tick_subtitle_scheduler)
        self._subtitle_timer.start()

        # Free Switch: cycle webcam → VR → dual on a fixed interval (UI toggle)
        self._fs_auto_cycle_interval_ms = 10_000
        self._fs_auto_cycle_timer = QtCore.QTimer(self)
        self._fs_auto_cycle_timer.setInterval(self._fs_auto_cycle_interval_ms)
        self._fs_auto_cycle_timer.timeout.connect(self._on_free_switch_auto_cycle_tick)
        self._fs_cycle_order = ("webcam", "vr", "dual")
        self._fs_cycle_idx = 0

        QtCore.QTimer.singleShot(0, self._apply_initial_geometry)

    # ---------------- Fonts / Styles ----------------

    def _init_fonts(self) -> None:
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        if os.path.exists(font_path):
            font_id = QtGui.QFontDatabase.addApplicationFont(font_path)
            if font_id != -1:
                families = QtGui.QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    self.font_family = families[0]

        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        dpi = float(screen.logicalDotsPerInch()) if screen else 96.0
        scale = max(0.85, min(1.6, dpi / 96.0))
        self.font_size = int(round(14 * scale))
        self.font_size = max(11, min(20, self.font_size))

        self._apply_styles(self.font_size)

    def _apply_styles(self, size_pt: int) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return

        app.setStyle("Fusion")

        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor(220, 220, 220))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor(30, 30, 30))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(45, 45, 45))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(220, 220, 220))
        palette.setColor(QtGui.QPalette.Button, QtGui.QColor(60, 60, 60))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(220, 220, 220))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(255, 255, 255))
        app.setPalette(palette)

        font = QtGui.QFont(self.font_family)
        font.setPointSize(int(size_pt))
        app.setFont(font)

        grp_margin = max(10, int(size_pt * 0.9))
        grp_padding = max(8, int(size_pt * 0.7))

        app.setStyleSheet(f"""
            QGroupBox {{
                font-weight: 750;
                border: 2px solid #555;
                border-radius: 10px;
                margin-top: {grp_margin}px;
                padding-top: {grp_padding}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }}

            QTextEdit {{
                background-color: #252525;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 10px;
                line-height: 150%;
            }}

            QMainWindow {{
                background: #2d2d2d;
            }}
        """)

    # ---------------- Model ----------------

    def _load_livecc_model(self) -> None:
        try:
            from miis_broadcast.core.models.livecc_transformers import LiveCCInfer
            print("[Main] 正在主執行緒載入 LiveCC 模型...")
            classifier_cfg = self.configs.get("model", {}).get("classifier", {})
            self.livecc_model = LiveCCInfer(classifier_cfg=classifier_cfg)
            print("[Main] LiveCC 模型載入完成")
            self.model_ready = True
        except ImportError as e:
            print(f"[Simulate] LiveCC module load failed (ImportError): {e}")
            print("[Simulate] Using simulation mode (GUI testing only).")
            self.model_ready = True
        except Exception as e:
            print(f"[Main] Model load failed: {e}")
            import traceback; traceback.print_exc()
            self.model_ready = False

    def _preload_bytetrack_model(self) -> None:
        """Start a background QThread to pre-load ByteTrackWrapper so mode
        switching to OBS+Track is instant."""
        bt_cfg = self.configs.get("bytetrack", {})
        repo_path = bt_cfg.get("bytetrack_repo") or None
        exp_file  = bt_cfg.get("exp_file",  "exps/example/mot/yolox_s_mix_det.py")
        ckpt_path = bt_cfg.get("ckpt_path", "pretrained/bytetrack_s_mot17.pth.tar")

        import os
        if repo_path and not os.path.isabs(exp_file):
            exp_file  = os.path.join(repo_path, exp_file)
        if repo_path and not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(repo_path, ckpt_path)

        # Capture values for closure
        _repo = repo_path
        _exp  = exp_file
        _ckpt = ckpt_path
        _cfg  = bt_cfg
        _self = self

        class _PreloadThread(QtCore.QThread):
            done = QtCore.Signal(object)   # emits ByteTrackWrapper or None

            def run(self):
                try:
                    from miis_broadcast.core.models.bytetrack_tracker import ByteTrackWrapper
                    print("[Preload] ByteTrack model 載入中...")
                    wrapper = ByteTrackWrapper(
                        ckpt_path              = _ckpt,
                        exp_file               = _exp,
                        bytetrack_repo         = _repo,
                        device                 = _cfg.get("device", "cuda"),
                        fp16                   = bool(_cfg.get("fp16", True)),
                        fuse                   = bool(_cfg.get("fuse", True)),
                        track_thresh           = float(_cfg.get("track_thresh", 0.5)),
                        match_thresh           = float(_cfg.get("match_thresh", 0.8)),
                        track_buffer           = int(_cfg.get("track_buffer", 30)),
                        aspect_ratio_thresh    = float(_cfg.get("aspect_ratio_thresh", 1.6)),
                        min_box_area           = float(_cfg.get("min_box_area", 10)),
                        subject_only           = bool(_cfg.get("subject_only", True)),
                        subject_pad            = float(_cfg.get("subject_pad", 0.15)),
                        min_subject_area_ratio = float(_cfg.get("min_subject_area_ratio", 0.03)),
                        preempt_ratio          = float(_cfg.get("preempt_ratio", 4.0)),
                    )
                    print("[Preload] ✅ ByteTrack model 預載完成")
                    self.done.emit(wrapper)
                except Exception as e:
                    print(f"[Preload] ⚠️ ByteTrack 預載失敗：{e}")
                    self.done.emit(None)

        t = _PreloadThread(self)
        t.done.connect(lambda w: setattr(_self, '_bytetrack_wrapper', w))
        t.done.connect(lambda _: setattr(_self, '_bytetrack_preload_thread', None))
        self._bytetrack_preload_thread = t
        t.start()

    def parseConfigs(self) -> None:
        gui_cfg = self.configs.get("gui_window", {})
        self.window_width_min = gui_cfg.get("min_width", 900)
        self.window_height_min = gui_cfg.get("min_height", 480)
        self.window_title = gui_cfg.get("title", "LiveCC Studio")

        infer_cfg = self.configs.get("inference", {})
        self._DEDUP_WINDOW_S = float(infer_cfg.get("dedup_window_s", 3.0))
        self._DEDUP_THRESHOLD = float(infer_cfg.get("dedup_threshold", 0.75))
        self._MAX_PENDING = int(infer_cfg.get("max_pending", 400))

    # ---------------- UI ----------------

    def _initUI(self) -> None:
        self.setMinimumSize(self.window_width_min, self.window_height_min)
        self.setWindowTitle(self.window_title)

        central = QtWidgets.QWidget()
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(16)

        self.top_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.top_splitter.setChildrenCollapsible(False)

        self.video_panel = VideoPanel()
        self.control_panel = ControlPanel()

        self.top_splitter.addWidget(self.video_panel)
        self.top_splitter.addWidget(self.control_panel)

        self.top_splitter.setStretchFactor(0, 10)
        self.top_splitter.setStretchFactor(1, 1)
        self.top_splitter.splitterMoved.connect(self._on_splitter_moved)

        bottom_group = QtWidgets.QGroupBox("即時解說字幕 (Live Commentary Log)")
        bottom_layout = QtWidgets.QVBoxLayout(bottom_group)
        bottom_layout.setContentsMargins(14, 18, 14, 12)

        self.text_output = TextOutputWidget()
        bottom_layout.addWidget(self.text_output)

        # Click segment line -> seek to corresponding time in video
        self._install_text_output_click_handler()

        # 垂直 splitter — 讓使用者可拖動上(影像)/下(字幕)邊界
        self.main_vsplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.main_vsplitter.setChildrenCollapsible(False)
        self.main_vsplitter.addWidget(self.top_splitter)
        self.main_vsplitter.addWidget(bottom_group)
        self.main_vsplitter.setStretchFactor(0, 4)
        self.main_vsplitter.setStretchFactor(1, 2)
        main_layout.addWidget(self.main_vsplitter)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Initializing system...")

        # Settings rows use font metrics; refresh after panel is under MainWindow (correct font chain)
        self.control_panel.apply_source_metrics()
        self.control_panel.apply_settings_metrics()

        # Signals
        self.control_panel.requestOpenVideo.connect(self.on_open_video_clicked)
        self.control_panel.requestOpenCamera.connect(self.on_open_camera_clicked)
        self.control_panel.requestLoadContext.connect(self.on_load_context_clicked)
        self.control_panel.requestOpenCameraTrack.connect(self.on_open_camera_track_clicked)
        self.control_panel.requestOpenOBS.connect(self.on_open_obs_clicked)
        self.control_panel.requestOpenDualSync.connect(self.on_open_dual_sync_clicked)
        self.control_panel.requestOpenFreeSwitch.connect(self.on_open_free_switch_clicked)
        self.control_panel.requestSwitchSource.connect(self.on_switch_source)
        self.control_panel.freeSwitchAutoCycleToggled.connect(
            self._on_free_switch_auto_cycle_toggled
        )
        self.control_panel.requestStart.connect(self.on_start_clicked)
        self.video_panel.seekRequested.connect(self.on_seek_requested)

        # 載入 prompts.yml 並填入下拉式選單
        self._init_prompt_manager_and_fill_styles()

        if self.livecc_model is not None:
            self._initLiveCCWorker()
            self._initCameraWorker()
            self.livecc_worker.signal_model_loaded.emit()
        else:
            self.statusBar().showMessage("請連線遠端伺服器後開始播報", 0)
            self.control_panel.set_status("請點「連線遠端伺服器」按鈕設定 Host/Port 並連線")

        # Remote control panel signals
        self.control_panel.requestRemoteConnect.connect(self.on_remote_connect_clicked)
        self.control_panel.requestRemoteDisconnect.connect(self.on_remote_disconnect_clicked)

        # Audience viewer page (HTTP) — start early so http://localhost:8080/audience works
        # before entering Free Switch. LiveKit publisher still starts with Free Switch only.
        self._ensure_audience_token_server()

    def _init_prompt_manager_and_fill_styles(self) -> None:
        try:
            project_root = _find_project_root(Path(__file__))
            cfg_path = project_root / "configs" / "livecc_prompts.yml"
            self.prompt_manager = PromptManager(cfg_path)

            self.control_panel.cmb_style.blockSignals(True)
            self.control_panel.cmb_style.clear()
            for item in self.prompt_manager.list_styles():
                self.control_panel.cmb_style.addItem(item.label, userData=item.key)

            default_key = self.prompt_manager.default_style_key()
            idx = self.control_panel.cmb_style.findData(default_key)
            if idx >= 0:
                self.control_panel.cmb_style.setCurrentIndex(idx)
            self.control_panel.cmb_style.blockSignals(False)

        except Exception as e:
            self.append_text(f"Failed to load broadcast style settings: {e}")
            self.control_panel.cmb_style.clear()
            self.control_panel.cmb_style.addItem("Default (Fallback)", userData="fallback")
            self.prompt_manager = None

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._apply_initial_geometry)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if not hasattr(self, "top_splitter") or not hasattr(self, "_splitter_ratio"):
            return
        m = self.centralWidget().layout().contentsMargins()
        w = max(1, self.width() - m.left() - m.right())
        ctrl_min = self.control_panel.minimumWidth()
        ctrl_w = max(ctrl_min, int(w * (1.0 - self._splitter_ratio)))
        video_w = max(1, w - ctrl_w)
        self.top_splitter.setSizes([video_w, ctrl_w])

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        sizes = self.top_splitter.sizes()
        total = sum(sizes)
        if total > 0:
            self._splitter_ratio = sizes[0] / total

    def _apply_initial_geometry(self) -> None:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()

        target_w = max(self.minimumWidth(), int(geo.width() * 0.90))
        target_h = max(self.minimumHeight(), int(geo.height() * 0.90))

        if self.width() < self.minimumWidth() or self.height() < self.minimumHeight():
            self.resize(target_w, target_h)

        self._splitter_ratio = 0.72
        w = max(1, self.width())
        h = max(1, self.height())
        self.top_splitter.setSizes([int(w * 0.72), int(w * 0.28)])
        if hasattr(self, "main_vsplitter"):
            self.main_vsplitter.setSizes([int(h * 0.60), int(h * 0.40)])

    # ---------------- Workers ----------------

    def _initLiveCCWorker(self) -> None:
        from .workers.livecc import LiveCCWorker
        self.livecc_thread = QtCore.QThread(self)
        self.livecc_worker = LiveCCWorker()
        self.livecc_worker.livecc = self.livecc_model
        self.livecc_worker.moveToThread(self.livecc_thread)
        self.livecc_worker.signal_model_loaded.connect(self.on_model_loaded)
        self.livecc_worker.signal_segment.connect(self._route_segment)
        self.livecc_worker.signal_finished.connect(self.on_livecc_finished)
        self.livecc_worker.signal_error.connect(self.on_error)
        self.signal_start_livecc.connect(self.livecc_worker.runInference)

        self.livecc_thread.start()

    def _initCameraWorker(self) -> None:
        from .workers.livecc import LiveCCCameraWorker
        self.cam_worker_thread = QtCore.QThread(self)
        camera_cfg = self.configs.get("model", {}).get("classifier", {}).get("camera", {})
        self.cam_worker = LiveCCCameraWorker(
            device_id=int(self.configs.get("model", {}).get("classifier", {}).get("device_id", 0)),
            window_sec=float(camera_cfg.get("window_sec", 2.0)),
            target_fps=float(camera_cfg.get("target_fps", 2.0)),
            infer_interval=float(camera_cfg.get("infer_interval", 2.0)),
            memory_reset_every=int(camera_cfg.get("memory_reset_every", 5)),
        )
        self.cam_worker.livecc = self.livecc_model
        self.cam_worker.moveToThread(self.cam_worker_thread)
        self.cam_worker.signal_model_loaded.connect(self.on_model_loaded)
        self.cam_worker.signal_segment.connect(self._route_segment)
        self.cam_worker.signal_error.connect(self.on_error)
        self.signal_start_camera_livecc.connect(self.cam_worker.runCameraInference)
        self.signal_p1_confirmed.connect(
            self.cam_worker.requestMemoryReset,
            QtCore.Qt.QueuedConnection,
        )
        self.cam_worker_thread.start()

    def _initGeminiWorker(self) -> None:

        # --- Existing GeminiWorker (kept for file-mode / legacy) ---
        self.gemini_thread = QtCore.QThread(self)
        self.gemini_worker = GeminiWorker()
        self.gemini_worker.moveToThread(self.gemini_thread)
        self._signal_to_gemini.connect(self.gemini_worker.process_segment)
        self.gemini_worker.signal_priority.connect(self._on_gemini_priority)
        self.gemini_worker.signal_broadcast.connect(self.on_segment)
        self.gemini_worker.signal_error.connect(self.on_gemini_error)
        self.gemini_thread.started.connect(self.gemini_worker.initialize)
        self.gemini_thread.start()

        # --- GeminiBackgroundWorker (slow blade continuous background) ---
        from .workers.gemini import GeminiBackgroundWorker
        self.gemini_bg_thread = QtCore.QThread(self)
        self.gemini_bg_worker = GeminiBackgroundWorker(get_remaining_sec_fn=self._get_active_tts_remaining_sec)
        self.gemini_bg_worker.moveToThread(self.gemini_bg_thread)
        self.gemini_bg_worker.signal_broadcast.connect(self.on_segment)
        self.gemini_bg_worker.signal_error.connect(self.on_gemini_error)

        self.signal_livecc_context.connect(
            self.gemini_bg_worker.update_context,
            QtCore.Qt.QueuedConnection,
        )

        self.gemini_bg_thread.started.connect(self.gemini_bg_worker.initialize)
        self.gemini_bg_thread.start()

        # signal_tts_done → P1 post-interrupt silence handler
        self.tts_worker.signal_tts_done.connect(self._on_tts_done)
        self.tts_worker.signal_playback_start.connect(self._on_tts_playback_log)

    # Time-aware Gemini priority: Gemini-originated content never interrupts
    # TTS playback directly (only LiveCC's fast-blade P1 in _route_segment may
    # do that). Instead, Gemini's priority decides queue placement, adjusted by
    # how stale the content has become and whether a more urgent item is still
    # protecting its queue position.
    _PRIORITY_DECAY_INTERVAL_SEC = 2.0          # every N sec of staleness, priority worsens by 1
    _TTS_PROTECT_WINDOW_SEC = {1: 6.0, 2: 3.0}  # after sending P1/P2, shield queue position this long
    _FAST_BLADE_DEDUP_WINDOW_S = 5.0            # suppress repeated P1/P2 triggers for the same event
    _P1_FILLER_EN = "Hold on!"          # English interrupt cue
    _P1_FILLER_ZH = "等等！"            # zh-TW interrupt cue
    _P1_FILLER = _P1_FILLER_EN          # active filler, updated by _apply_tts_settings_before_start

    def _is_p1_filler_text(self, text: str) -> bool:
        t = (text or "").strip()
        return t in (self._P1_FILLER_EN, self._P1_FILLER_ZH, self._P1_FILLER)
    _P3_BACKPRESSURE_WATERMARK_SEC = 2.0        # only feed GeminiWorker while TTS backlog is below this

    def _format_segment_ui_line(self, start_t: float, stop_t: float, tag: str, body: str) -> str:
        """Build one timestamped UI line with an explicit source tag."""
        return f"[{self._fmt_time(start_t)}-{self._fmt_time(stop_t)}] {tag} {body}"

    @staticmethod
    def _preview_text(text: str, limit: int = 72) -> str:
        text = (text or "").strip()
        if len(text) <= limit:
            return text
        return text[:limit] + "…"

    def _gemini_ui_display(self, data: dict) -> str:
        """Human-readable Gemini line — distinct from LiveCC / interrupt filler."""
        priority = data.get("priority", "?")
        broadcast = (data.get("broadcast_text") or "").strip()
        if data.get("_background"):
            return f"[Gemini·BG] [P{priority}] {broadcast}"
        return f"[Gemini→TTS] [P{priority}] {broadcast}"

    def _segment_ui_body(self, data: object, display_text: str, *, priority_jump: bool = False) -> str:
        """Map segment payload to a clearly labelled UI string."""
        if isinstance(data, dict) and data.get("broadcast_text"):
            return self._gemini_ui_display(data)
        prefix = "[⚡ PRIORITY] " if priority_jump else ""
        return f"{prefix}{display_text}"

    @QtCore.Slot(float, float, int, bool)
    def _on_gemini_priority(self, start_t: float, stop_t: float, priority: int, should_speak: bool) -> None:
        """
        Fires as soon as Gemini returns the PRIORITY line (before SPEAK text arrives).
        Gemini priority is queue-ordering information only — it never interrupts
        current TTS playback. A P1 confirmation still resets LiveCC's KV cache.
        """
        if priority == 1:
            # Gemini confirmed P1 — reset LiveCC KV cache via QueuedConnection
            self.signal_p1_confirmed.emit()

    def _effective_gemini_priority(self, base_priority: int, enqueue_ts: float) -> int:
        """Time-aware priority for Gemini-originated content (queue placement only).

        Staleness decay: content that waited longer before reaching TTS describes
        an increasingly stale moment, so its priority worsens over time.
        Protection clamp: if a more urgent item was sent to TTS recently, this
        item can't claim a better queue position until that window expires.
        """
        priority = base_priority + int((time.time() - enqueue_ts) // self._PRIORITY_DECAY_INTERVAL_SEC)
        if time.time() < self._tts_protect_until:
            priority = max(priority, self._tts_protected_priority + 1)
        return min(priority, 5)

    def _register_tts_priority(self, priority: int) -> None:
        """Arm the protection window when P1/P2 content is sent to TTS, so
        Gemini's subsequent (time-decayed) priority can't immediately bump it."""
        window = self._TTS_PROTECT_WINDOW_SEC.get(priority)
        if window:
            self._tts_protected_priority = priority
            extra = self._get_active_tts_remaining_sec() if priority in (1, 2) else 0.0
            self._tts_protect_until = time.time() + window + extra

    def _is_p1_audio_active(self) -> bool:
        """True while P1 filler or broadcast is playing — block another P1 interrupt."""
        if self._p1_filler_armed:
            return True
        if time.time() < self._tts_protect_until and self._tts_protected_priority == 1:
            return True
        if self._tts_protected_priority == 1 and self._get_active_tts_remaining_sec() > 0.3:
            return True
        return False

    def _is_p2_audio_active(self) -> bool:
        """True while P2 broadcast is playing — block duplicate P2 preempt."""
        if self._is_p1_audio_active():
            return True
        if time.time() < self._tts_protect_until and self._tts_protected_priority == 2:
            return True
        if self._tts_protected_priority == 2 and self._get_active_tts_remaining_sec() > 0.3:
            return True
        return False

    def _should_hard_cut_for_p1(self) -> bool:
        """Only hard-cut for P1 when replacing filler or non-P1 audio — never cut active P1."""
        if self._p1_filler_armed:
            return True
        return not self._is_p1_audio_active()

    def _get_active_tts_remaining_sec(self) -> float:
        """Remaining queued playback time of whichever TTS engine is active.

        tts_mode can change at runtime via the dropdown, so callers (e.g.
        GeminiBackgroundWorker's backpressure watermark) must resolve the
        engine dynamically rather than holding a fixed worker reference —
        otherwise they pace against an idle engine's always-empty queue.
        """
        if self.tts_mode == "openai" and hasattr(self, "tts_worker"):
            return self.tts_worker.get_queue_remaining_sec()
        return 0.0

    @staticmethod
    def _scan_priority(text: str) -> int:
        """Return 1 (P1), 2 (P2), or 3 (P3) based on keyword presence in text."""
        lower = text.lower()
        for kw in MainWindow._P1_KEYWORDS:
            if kw in lower:
                return 1
        for kw in MainWindow._P2_KEYWORDS:
            if kw in lower:
                return 2
        return 3

    @staticmethod
    def _livecc_event_dict(raw: str) -> dict:
        """Normalize a LiveCC caption into the dict shape GeminiBroadcaster expects."""
        return {"metadata": {"raw": raw}, "event": "raw_description"}

    def _broadcast_tts_allowed(self, data: object) -> bool:
        """Only speak Gemini broadcast_text (zh-TW), never raw LiveCC — all TTS engines."""
        if not isinstance(data, dict):
            return False
        if not data.get("broadcast_text"):
            return False
        return bool(data.get("should_speak", True)) or bool(data.get("_background"))

    def _p1_hard_interrupt_with_filler(self, start_t: float, already_p1: bool) -> None:
        """Hard-cut current audio and play the interrupt filler via OpenAI TTS."""
        if already_p1:
            return
        self._p1_filler_armed = True
        self._register_tts_priority(1)
        if self.tts_mode == "openai":
            self.signal_tts_interrupt.emit()
            self.signal_tts_speak.emit(self._P1_FILLER, 1, time.time(), start_t, start_t, {})
            # Filler bypasses on_segment — arm post-P1 tracking here so background
            # Gemini resumes after the interrupt sequence finishes.
            self._post_p1_pending = True
            self._arm_p1_fallback_timer()
        elif self.tts_mode == "local":
            self.signal_local_tts_interrupt.emit()

    def _fast_blade_enqueue_gemini(
        self,
        start_t: float,
        stop_t: float,
        raw: str,
        data: object,
        *,
        already_p1: bool,
        already_p2: bool = False,
        flush: bool,
        p2: bool = False,
    ) -> None:
        """Send LiveCC text to GeminiBroadcaster; UI shows LiveCC preview (all TTS modes).

        When already_p1 is True AND the filler is still armed, it means Gemini has
        not yet returned a translation for the first P1 event — skip enqueue_front so
        we don't stack up duplicate Gemini requests that would each produce an utterance.
        When already_p2 is True, skip duplicate P2 Gemini requests while P2 is playing.
        """
        gem_event = self._fast_blade_gemini_event(raw, data)
        if flush and not already_p1 and hasattr(self, "gemini_worker"):
            self.gemini_worker.flush_and_abort()
        if hasattr(self, "gemini_worker"):
            # Don't add another Gemini request while filler is still playing (first
            # P1 translation not yet returned). This prevents cascaded duplicate P1s.
            if already_p1 and self._p1_filler_armed:
                logging.info("[FastBlade] P1 already queued to Gemini (filler armed), skipping enqueue_front")
            elif already_p2 and p2:
                logging.info("[FastBlade] P2 already active, skipping enqueue_front")
            else:
                self.gemini_worker.enqueue_front(start_t, stop_t, gem_event)
        livecc_preview = self._preview_text(raw)
        if p2:
            tag = "[LiveCC·排隊]" if already_p2 else "[LiveCC→P2]"
            hint = "" if already_p2 else "（等 Gemini 中文稿）"
        elif already_p1:
            tag, hint = "[LiveCC·排隊]", ""
        else:
            tag = "[LiveCC→INT]"
            hint = f"（口播「{self._P1_FILLER}」→ 等 Gemini 中文稿）"
        body = f"{livecc_preview}  {hint}".strip() if hint else livecc_preview
        self._append_ui(self._format_segment_ui_line(start_t, stop_t, tag, body))

    def _emit_openai_tts_speak(
        self,
        text: str,
        priority: int,
        ref_ts: float,
        start_t: float,
        stop_t: float,
        *,
        cut_current: bool = False,
        log_meta: Optional[dict] = None,
    ) -> None:
        if cut_current and priority <= 1:
            if self._should_hard_cut_for_p1():
                self.signal_tts_interrupt.emit()
            self._p1_filler_armed = False
        elif not self._is_p1_filler_text(text):
            # Gemini may return P2+ for a P1 LiveCC trigger — disarm filler so
            # subsequent P1 events are not stuck in [LiveCC·排隊] forever.
            self._p1_filler_armed = False
        self.signal_tts_speak.emit(
            text, priority, ref_ts, start_t, stop_t, log_meta or {}
        )

    def _fast_blade_gemini_event(self, raw: str, data: object) -> dict:
        if isinstance(data, dict):
            return data
        return self._livecc_event_dict(raw)

    @QtCore.Slot(float, float, object)
    def _route_segment(self, start_t: float, stop_t: float, data: object) -> None:
        """Fast-slow blade routing: P1/P2 direct-to-TTS, P3 → context pool."""
        self._ensure_log_dir()

        if not self._use_gemini:
            self.on_segment(start_t, stop_t, data)
            return

        raw = ""
        if isinstance(data, dict):
            raw = data.get("metadata", {}).get("raw", "") or data.get("event", "")
        elif isinstance(data, str):
            raw = data

        # Stitch truncated fragments: LiveCC frequently cuts text off mid-thought
        # ("..."), and a P1/P2 trigger keyword can straddle that boundary and be
        # missed if scanned in isolation. Hold one such non-triggering fragment
        # and re-scan it merged with the very next segment ("..." to next "...")
        # before deciding how to route — bounded to a single stitch so latency
        # for genuine immediate triggers is unaffected.
        pending = self._pending_livecc_fragment
        self._pending_livecc_fragment = None
        scan_raw = raw
        if pending is not None:
            pending_start, _pending_stop, pending_raw = pending
            scan_raw = f"{pending_raw.rstrip()} {raw.strip()}".strip()
            start_t = pending_start
            logging.info("[FastBlade] stitched truncated fragment %r + %r -> %r",
                         pending_raw[:60], raw[:60], scan_raw[:120])

        fast_priority = self._scan_priority(scan_raw)

        if pending is None and fast_priority == 3 and raw.rstrip().endswith("..."):
            self._pending_livecc_fragment = (start_t, stop_t, raw.strip())
            logging.info("[FastBlade] buffering truncated fragment for stitching: %r", raw[:80])
            return

        raw = scan_raw

        # [延遲][LiveCC] Stage 1: time from "frame appeared" (stop_t on the shared
        # wall-clock anchor) to LiveCC emitting this segment. frame_wall_ts also
        # anchors [延遲][語音][中斷] below (dimension 2: frame -> sound for P1/P2).
        frame_wall_ts = 0.0
        if self.mode == "file" and self._livecc_start_wall > 0:
            frame_wall_ts = self._livecc_start_wall + stop_t
            latency = time.time() - frame_wall_ts
            if not hasattr(self, "_livecc_latencies"):
                self._livecc_latencies = []
            self._livecc_latencies.append(latency)
            avg = sum(self._livecc_latencies) / len(self._livecc_latencies)
            logging.info(
                "[延遲][LiveCC] 段落 %.1f-%.1fs → LiveCC 輸出 latency=%.2fs (平均=%.2fs, n=%d)",
                start_t, stop_t, latency, avg, len(self._livecc_latencies),
            )
        tts_ref_ts = frame_wall_ts if frame_wall_ts > 0 else time.time()

        if fast_priority == 1:
            logging.info("[FastBlade] P1 hit: %r", raw[:80])
            tts_text = raw.strip()
            if tts_text and self._is_duplicate_tts(tts_text, window=self._FAST_BLADE_DEDUP_WINDOW_S):
                logging.info("[FastBlade] P1 duplicate suppressed: %r", tts_text[:80])
            else:
                if hasattr(self, "gemini_bg_worker"):
                    self.gemini_bg_worker.pause()
                already_p1 = self._is_p1_audio_active()
                self._p1_hard_interrupt_with_filler(start_t, already_p1)
                if tts_text:
                    self._fast_blade_enqueue_gemini(
                        start_t, stop_t, raw, data,
                        already_p1=already_p1, flush=True,
                    )
            self.signal_p1_confirmed.emit()

        elif fast_priority == 2:
            logging.info("[FastBlade] P2 hit: %r", raw[:80])
            tts_text = raw.strip()
            if tts_text and self._is_duplicate_tts(tts_text, window=self._FAST_BLADE_DEDUP_WINDOW_S):
                logging.info("[FastBlade] P2 duplicate suppressed: %r", tts_text[:80])
            elif tts_text:
                already_p2 = self._is_p2_audio_active()
                self._fast_blade_enqueue_gemini(
                    start_t, stop_t, raw, data,
                    already_p1=False, already_p2=already_p2, flush=False, p2=True,
                )

        else:
            # P3: feed the slow-blade context pool only. GeminiBackgroundWorker is
            # the single continuous-commentary source — it polls the pool and fires
            # under its own backpressure/interval. Previously P3 ALSO went to the
            # foreground GeminiWorker (_signal_to_gemini), so the same description
            # was voiced twice (foreground + background), doubling utterances,
            # fighting over the TTS queue, and stalling the opening seconds.
            description = raw.strip()
            if description:
                self.signal_livecc_context.emit(description)

    # ---------------- Remote Socket ----------------

    def _stop_remote_client_ram_monitor(self) -> None:
        if self._remote_client_ram_timer is not None:
            self._remote_client_ram_timer.stop()
            self._remote_client_ram_timer.deleteLater()
            self._remote_client_ram_timer = None

    def _start_remote_client_ram_monitor(self) -> None:
        """Every ~2s: send CLIENT_DIAG so the server prints [Client] (no duplicate GUI stdout)."""
        self._stop_remote_client_ram_monitor()
        timer = QtCore.QTimer(self)
        timer.setInterval(2000)
        timer.timeout.connect(self._log_remote_client_ram_tick)
        self._remote_client_ram_timer = timer
        timer.start()
        self._log_remote_client_ram_tick()

    @QtCore.Slot()
    def _log_remote_client_ram_tick(self) -> None:
        if not self.is_inference_running or self._socket_runner is None:
            self._stop_remote_client_ram_monitor()
            return
        try:
            import psutil

            rss_mb = psutil.Process().memory_info().rss / (1024.0**2)
            sys_pct = psutil.virtual_memory().percent
            q_used, q_max = self._socket_runner.get_frame_send_queue_levels()
            gpu_snap = cuda_vram_snapshot(0)
            gpu_suffix = vram_log_suffix(gpu_snap)
            msg = (
                f"[Client] RSS={rss_mb:.1f} MiB | JPEG send_queue={q_used}/{q_max} | "
                f"system_RAM_used={sys_pct:.0f}%{gpu_suffix}"
            )
            if hasattr(self, "session_logger") and self.session_logger.current_log_file:
                self.session_logger.log_system("Memory", "INFO", msg)
            gu = gt = None
            if gpu_snap is not None:
                gu = gpu_snap.used_mib
                gt = gpu_snap.total_mib
                g_torch = gpu_snap.torch_alloc_mib
            else:
                g_torch = None
            self._local_client_diag_samples.append(
                {
                    "rss_mib": float(rss_mb),
                    "jpeg_q_used": int(q_used),
                    "jpeg_q_max": int(q_max),
                    "sys_ram_pct": float(sys_pct),
                    "gpu_vram_used_mib": gu,
                    "gpu_vram_total_mib": gt,
                    "gpu_torch_alloc_mib": g_torch,
                }
            )
            self._socket_runner.send_client_diagnostic(
                rss_mb,
                q_used,
                q_max,
                sys_ram_pct=sys_pct,
                gpu_vram_used_mib=gu,
                gpu_vram_total_mib=gt,
                gpu_torch_alloc_mib=g_torch,
            )
        except Exception as e:
            print(f"[Remote] telemetry send failed: {e}")

    def _join_worker_thread_smooth(
        self,
        worker: Optional[QtCore.QThread],
        *,
        timeout_ms: int = 6000,
        terminate_grace_ms: int = 1200,
        slice_ms: int = 75,
    ) -> None:
        """Wait for worker to finish while pumping Qt events (keeps UI repainting).

        Camera workers often block inside OpenCV capture or GPU inference until the next loop
        tick observes requestStop(); a single blocking wait(...) would freeze MainWindow."""
        if worker is None or not worker.isRunning():
            return
        app = QtWidgets.QApplication.instance()
        elapsed = QtCore.QElapsedTimer()
        elapsed.start()
        while worker.isRunning() and int(elapsed.elapsed()) < timeout_ms:
            remaining = timeout_ms - int(elapsed.elapsed())
            chunk = max(1, min(slice_ms, remaining))
            worker.wait(chunk)
            if app is not None:
                app.processEvents(
                    QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
                )
        if worker.isRunning():
            worker.terminate()
            worker.wait(terminate_grace_ms)

    def _clear_local_inference_telemetry_for_new_session(self) -> None:
        """Reset per-broadcast samples (CLIENT_DIAG copies + audience stats)."""
        self._local_client_diag_samples.clear()
        pub = self._audience_publisher
        if pub is not None:
            pub.reset_session_telemetry()

    def _flush_local_inference_telemetry_summary(
        self,
        obs_tracker_opt: Optional[Any] = None,
    ) -> None:
        """Print mean telemetry for this broadcast on this machine (thin-client + extras)."""
        def _avg(nums: list) -> Optional[float]:
            return sum(nums) / len(nums) if nums else None

        xs = self._local_client_diag_samples
        if xs:
            rss_m = _avg([float(r["rss_mib"]) for r in xs])
            q_u_m = _avg([float(r["jpeg_q_used"]) for r in xs])
            q_max = int(xs[-1].get("jpeg_q_max", 0))
            sysp_m = _avg([float(r["sys_ram_pct"]) for r in xs])
            gpu_used_nums = [
                float(r["gpu_vram_used_mib"])
                for r in xs
                if r.get("gpu_vram_used_mib") is not None
            ]
            gu_m = _avg(gpu_used_nums)
            gt_ref: Optional[float] = None
            for row in reversed(xs):
                t = row.get("gpu_vram_total_mib")
                if t is not None:
                    gt_ref = float(t)
                    break
            torch_nums = [
                float(r["gpu_torch_alloc_mib"])
                for r in xs
                if r.get("gpu_torch_alloc_mib") is not None
            ]
            ga_m = _avg(torch_nums) if torch_nums else None
            sfx = vram_log_suffix_from_wire(gu_m, gt_ref, ga_m)
            print(
                f"[SESSION AVG] Client-local (n={len(xs)})  RSS={rss_m:.1f} MiB | "
                f"JPEG send_queue={q_u_m:.2f}/{q_max} | "
                f"system_RAM_used={sysp_m:.0f}%{sfx}"
            )

        pub = self._audience_publisher
        if pub is not None:
            for ln in pub.consume_session_telemetry_average_lines():
                print(ln)

        if obs_tracker_opt is not None:
            ln = obs_tracker_opt.consume_session_perf_average_line()
            if ln:
                print(ln)

        self._local_client_diag_samples.clear()

    @QtCore.Slot(str, int)
    def on_remote_connect_clicked(self, host: str, port: int) -> None:
        self._stop_remote_client_ram_monitor()
        if self._socket_runner is not None:
            self._socket_runner.disconnect_and_quit()
            self._socket_runner = None

        self.append_text(f"[Remote] 正在連線至 {host}:{port}…")
        self.control_panel.set_remote_connecting()

        runner = SocketClientRunner(host, port, parent=self)
        runner.signal_connected.connect(self.on_remote_connected)
        runner.signal_disconnected.connect(self.on_remote_disconnected)
        runner.signal_connect_error.connect(self.on_remote_connect_error)
        runner.signal_segment.connect(self.on_remote_segment)
        runner.signal_status.connect(self.on_remote_status)
        runner.signal_error.connect(self.on_remote_server_error)
        self._socket_runner = runner
        runner.start()

    @QtCore.Slot()
    def on_remote_disconnect_clicked(self) -> None:
        self._stop_remote_client_ram_monitor()
        if self._socket_runner is not None:
            self.append_text("[Remote] 中斷連線")
            self._socket_runner.disconnect_and_quit()
            self._socket_runner = None
        self.model_ready = (self.livecc_model is not None)
        self.control_panel.set_remote_connected(False, "未連線")
        self._update_start_button_state()

    @QtCore.Slot()
    def on_remote_connected(self) -> None:
        self.append_text(f"[Remote] 已連線至 {self.control_panel.get_remote_host()}:{self.control_panel.get_remote_port()}")
        self.control_panel.set_remote_connected(True, "已連線")
        self.model_ready = True
        self._update_start_button_state()

    @QtCore.Slot(str)
    def on_remote_disconnected(self, reason: str) -> None:
        self.append_text(f"[Remote] 連線中斷: {reason}")
        self.control_panel.set_remote_connected(False, "連線中斷")
        self._socket_runner = None
        self.model_ready = (self.livecc_model is not None)
        if self.is_inference_running:
            self.stop_inference()
        self._update_start_button_state()

    @QtCore.Slot(str)
    def on_remote_connect_error(self, msg: str) -> None:
        self.append_text(f"[Remote] 連線失敗: {msg}")
        self.control_panel.set_remote_connected(False, "連線失敗")
        self._socket_runner = None
        self.model_ready = (self.livecc_model is not None)
        self._update_start_button_state()

    @QtCore.Slot(str)
    def on_remote_status(self, msg: str) -> None:
        self.statusBar().showMessage(f"[Remote] {msg}", 3000)

    @QtCore.Slot(str)
    def on_remote_server_error(self, msg: str) -> None:
        self.append_text(f"[Remote Error] {msg}")
        if self.is_inference_running:
            self.stop_inference()

    def _initTTSWorker(self) -> None:
            """Initialize all TTS Workers (OpenAI + Local Chatterbox)"""
            
            # ==========================================
            # 1. OpenAI TTS Worker (Cloud)
            # ==========================================
            self.tts_thread = QtCore.QThread(self)
            self.tts_worker = OpenAITTSWorker()
            self.tts_worker.moveToThread(self.tts_thread)

            # Auto-call worker.start() to initialize connection when thread starts
            self.tts_thread.started.connect(self.tts_worker.start)

            # Connect OpenAI dedicated signals
            self.signal_tts_apply_settings.connect(self.tts_worker.apply_settings, QtCore.Qt.QueuedConnection)
            self.signal_tts_warmup.connect(self.tts_worker.warmup_connect, QtCore.Qt.QueuedConnection)
            self.signal_tts_stop.connect(self.tts_worker.stop, QtCore.Qt.QueuedConnection)
            self.signal_tts_speak.connect(self.tts_worker.speak, QtCore.Qt.QueuedConnection)
            self.signal_tts_interrupt.connect(self.tts_worker.interrupt, QtCore.Qt.QueuedConnection)

            self.tts_thread.start()

            # ==========================================
            # 2. Local TTS Worker (本地 Chatterbox) [ChatterBox disabled]
            # ==========================================
            from .workers.chatterbox_tts import ChatterboxTTSWorker
            self.local_tts_thread = QtCore.QThread(self)
            self.local_tts_worker = ChatterboxTTSWorker()
            self.local_tts_worker.moveToThread(self.local_tts_thread)
            
            # Thread 啟動時，自動呼叫 worker.start() 載入模型 (需時較久)
            self.local_tts_thread.started.connect(self.local_tts_worker.start)

            # 連接 Local TTS 專用信號
            self.signal_local_tts_apply_settings.connect(self.local_tts_worker.apply_settings, QtCore.Qt.QueuedConnection)
            self.signal_local_tts_stop.connect(self.local_tts_worker.stop, QtCore.Qt.QueuedConnection)
            self.signal_local_tts_speak.connect(self.local_tts_worker.speak, QtCore.Qt.QueuedConnection)
            self.signal_local_tts_interrupt.connect(self.local_tts_worker.interrupt, QtCore.Qt.QueuedConnection)

            # 🔥 [修改點 1] 註解掉或刪除原本的直接啟動，改為 Lazy Load
            # self.local_tts_thread.start() 

            # 🔥 監聽下拉選單變化（需在所有 worker 建立完後再 connect）
            self.control_panel.cmb_tts.currentIndexChanged.connect(self._on_tts_mode_changed)

            # 初始化時根據預設模式啟動對應 worker
            self._on_tts_mode_changed()

    # ---------------- Slots ----------------

    def _tick_subtitle_scheduler(self) -> None:
        """
        File mode: Use playback time to decide when to show subtitles 
        (Inference can be ahead, but display must be synchronized).
        """
        if self.mode != "file":
            return
        if not self.is_inference_running:
            return
        if not hasattr(self, "_pending_segments"):
            return
        if not self._pending_segments:
            return

        cur = float(getattr(self, "_playback_sec", 0.0))

        # 把「已經到時間」的段落全部取出（避免只顯示最後一段造成跳秒/漏段）
        ready: list[tuple] = []
        while self._pending_segments and float(self._pending_segments[0][0]) <= cur:
            st, ed, data = self._pending_segments.popleft()
            ready.append((float(st), float(ed), data))

        if not ready:
            return

        for start_t, stop_t, data in ready:
            display_text, _ = self._extract_segment_texts(data)
            if not display_text.strip():
                continue

            ui_body = self._segment_ui_body(
                data, display_text,
                priority_jump=isinstance(data, dict) and bool(data.get("_priority_jump")),
            )
            line = f"[{self._fmt_time(start_t)}-{self._fmt_time(stop_t)}] {ui_body}"
            self._append_ui(line)

        while len(self._pending_segments) > self._MAX_PENDING:
            self._pending_segments.popleft()

    @QtCore.Slot()
    def _on_tts_mode_changed(self) -> None:
        mode = self.control_panel.get_tts_mode() if hasattr(self, "control_panel") else None
        self.tts_mode = mode or getattr(self, "tts_mode", "none")
        if self._audience_publisher is not None:
            self._clear_all_pcm_sinks()
            self._register_audience_pcm_sink()

    @QtCore.Slot()
    def on_open_video_clicked(self) -> None:
        dlg = QtWidgets.QFileDialog(self, "Select Video File")
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilter("Video Files (*.mp4 *.mov *.avi *.mkv);;All Files (*)")

        dlg_font = dlg.font()
        dlg_font.setPointSize(self.font_size)
        dlg.setFont(dlg_font)
        dlg.resize(1000, 700)

        if dlg.exec():
            paths = dlg.selectedFiles()
            if paths:
                self.on_video_selected(paths[0])

    @QtCore.Slot(str)
    def on_video_selected(self, path: str) -> None:
        self.stop_inference()
        self.mode = "file"
        if hasattr(self, "_pending_segments"):
            self._pending_segments.clear()
        self._playback_sec = 0.0
        self._broadcast_playback_sec = 0.0

        self.current_video_path = path
        self.control_panel.set_status(f"File: {os.path.basename(path)}")
        self.append_text(f"已載入影片：{os.path.basename(path)}")

        self._load_video_preview(path)
        self.video_panel.slider.setEnabled(True)
        self._update_start_button_state()

    def _stop_all_source_threads(self) -> None:
        """Stop and clean up all live-source threads before switching sources."""
        if self.video_thread:
            self.video_thread.requestStop()
            try:
                self.video_thread.signal_frame.disconnect()
            except RuntimeError:
                pass
            self.video_thread.wait()
            self.video_thread = None
        if self.camera_thread:
            self.camera_thread.requestStop()
            try:
                self.camera_thread.signal_frame.disconnect()
            except RuntimeError:
                pass
            self.camera_thread.wait()
            self.camera_thread = None
        if self.obs_thread:
            self.obs_thread.requestStop()
            try:
                self.obs_thread.signal_frame.disconnect()
            except RuntimeError:
                pass
            self._join_worker_thread_smooth(self.obs_thread)
            self.obs_thread = None
        if self.obs_bytetrack_thread:
            self.obs_bytetrack_thread.requestStop()
            try:
                self.obs_bytetrack_thread.signal_frame.disconnect()
                self.obs_bytetrack_thread.signal_subject_frame.disconnect()
            except RuntimeError:
                pass
            self._join_worker_thread_smooth(self.obs_bytetrack_thread)
            self.obs_bytetrack_thread = None
        if getattr(self, "dual_sync_thread", None):
            self.dual_sync_thread.requestStop()
            try:
                self.dual_sync_thread.signal_frame.disconnect()
            except RuntimeError:
                pass
            self._join_worker_thread_smooth(self.dual_sync_thread)
            self.dual_sync_thread = None
        if getattr(self, "free_switch_thread", None):
            self._stop_audience_publisher_only()
            self.free_switch_thread.requestStop()
            try:
                self.free_switch_thread.signal_frame.disconnect()
                self.free_switch_thread.signal_source_changed.disconnect()
                self.free_switch_thread.signal_vr_frame.disconnect(
                    self._deliver_audience_vr_frame
                )
            except RuntimeError:
                pass
            self._join_worker_thread_smooth(self.free_switch_thread)
            self.free_switch_thread = None
        if hasattr(self, "_stop_free_switch_auto_cycle"):
            self._stop_free_switch_auto_cycle()
        if hasattr(self.control_panel, "set_free_switch_bar_visible"):
            self.control_panel.set_free_switch_bar_visible(False)


    @QtCore.Slot()
    def on_open_camera_clicked(self) -> None:
        self.stop_inference()
        self.mode = "camera"
        self.current_video_path = "Live Camera"
        self.control_panel.set_status("模式: 即時鏡頭")
        self.append_text("已切換至鏡頭模式")
        self._stop_all_source_threads()

        # Auto-detect physical camera index (skip OBS Virtual Camera if present)
        from .core.io.obs_input import find_physical_camera_index
        cam_idx = find_physical_camera_index()
        print(f"[Camera] Using physical camera index {cam_idx}")

        self.camera_thread = CameraThread(camera_index=cam_idx)
        self.camera_thread.signal_frame.connect(self.on_camera_frame)
        self.camera_thread.signal_error.connect(self.on_error)
        self.camera_thread.start()

        self.video_panel.slider.setEnabled(False)
        if hasattr(self.control_panel, "set_free_switch_bar_visible"):
            self.control_panel.set_free_switch_bar_visible(False)
        self._update_start_button_state()

    @QtCore.Slot()
    def on_load_context_clicked(self) -> None:
        dlg = QtWidgets.QFileDialog(self, "載入比賽資訊")
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilter("Text Files (*.txt *.md *.yaml *.yml);;All Files (*)")

        if hasattr(self, "_last_context_dir"):
            if os.path.isdir(self._last_context_dir):
                start_dir = self._last_context_dir
            else:
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                start_dir = project_root
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            start_dir = project_root
        dlg.setDirectory(start_dir)

        dlg_font = dlg.font()
        dlg_font.setPointSize(self.font_size)
        dlg.setFont(dlg_font)
        dlg.resize(1000, 700)

        if dlg.exec():
            paths = dlg.selectedFiles()
            if paths:
                path = paths[0]
                self._last_context_dir = os.path.dirname(path)
                try:
                    from .core.models.gemini_broadcaster import load_game_context_file
                    text = load_game_context_file(path)
                    fname = os.path.basename(path)
                    self.control_panel.lbl_context.setText(fname)
                    self.control_panel.lbl_context.setStyleSheet("color: #69db7c;")
                    self.append_text(f"已載入比賽資訊：{fname} ({len(text)} 字元)")
                except Exception as e:
                    self.append_text(f"載入比賽資訊失敗：{e}")

    @QtCore.Slot()
    def on_open_obs_clicked(self) -> None:
        """Switch to OBS Virtual Camera mode."""
        self.stop_inference()
        self.mode = "obs"
        self.current_video_path = "OBS Virtual Camera"
        self.control_panel.set_status("Mode: OBS Virtual Camera")
        self.append_text("Switched to OBS stream mode - ensure OBS Virtual Camera is active")
        self._stop_all_source_threads()
        self.obs_thread = OBSCameraThread()
        self.obs_thread.signal_frame.connect(self.on_camera_frame)
        self.obs_thread.signal_error.connect(self.on_error)
        self.obs_thread.start()

        self.video_panel.slider.setEnabled(False)
        self._update_start_button_state()

    @QtCore.Slot()
    def on_open_camera_track_clicked(self) -> None:
        """Webcam + tracking: ByteTrack always runs locally; only the subject crop
        is forwarded to the remote server for LiveCC inference."""
        self.stop_inference()
        self.mode = "obs_track"
        self.current_video_path = "Camera + ByteTrack (local)"
        self.control_panel.set_status("Mode: Webcam + Tracking (local ByteTrack)")
        self.append_text("Switched to Webcam + Tracking (ByteTrack runs locally)")
        self.append_text("已切換至「鏡頭 + 追蹤」；ByteTrack 在本機執行，僅 subject crop 送至遠端")
        self._stop_all_source_threads()

        bt_cfg = self.configs.get("bytetrack", {})
        repo_path = bt_cfg.get("bytetrack_repo") or None
        exp_file  = bt_cfg.get("exp_file",  "exps/example/mot/yolox_s_mix_det.py")
        ckpt_path = bt_cfg.get("ckpt_path", "pretrained/bytetrack_s_mot17.pth.tar")

        import os
        if repo_path and not os.path.isabs(exp_file):
            exp_file  = os.path.join(repo_path, exp_file)
        if repo_path and not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(repo_path, ckpt_path)

        # Auto-detect physical camera index (skip OBS Virtual Camera)
        from .core.io.obs_input import find_physical_camera_index
        cam_idx = find_physical_camera_index()
        print(f"[CameraTrack] Using physical camera index {cam_idx}")
        print(f"[CameraTrack] 使用實體攝影機 index {cam_idx}")

        self.obs_bytetrack_thread = CameraByteTrackThread(
            ckpt_path              = ckpt_path,
            exp_file               = exp_file,
            bytetrack_repo         = repo_path,
            device                 = bt_cfg.get("device", "cuda"),
            fp16                   = bool(bt_cfg.get("fp16", True)),
            fuse                   = bool(bt_cfg.get("fuse", True)),
            track_thresh           = float(bt_cfg.get("track_thresh", 0.5)),
            match_thresh           = float(bt_cfg.get("match_thresh", 0.8)),
            track_buffer           = int(bt_cfg.get("track_buffer", 30)),
            aspect_ratio_thresh    = float(bt_cfg.get("aspect_ratio_thresh", 1.6)),
            min_box_area           = float(bt_cfg.get("min_box_area", 10)),
            subject_only           = bool(bt_cfg.get("subject_only", True)),
            subject_pad            = float(bt_cfg.get("subject_pad", 0.15)),
            min_subject_area_ratio = float(bt_cfg.get("min_subject_area_ratio", 0.03)),
            preempt_ratio          = float(bt_cfg.get("preempt_ratio", 4.0)),
            camera_index           = cam_idx,
            preloaded_tracker      = self._bytetrack_wrapper,
        )
        self.obs_bytetrack_thread.signal_frame.connect(self.on_obs_track_frame)
        self.obs_bytetrack_thread.signal_subject_frame.connect(self.on_obs_track_subject_frame)
        self.obs_bytetrack_thread.signal_error.connect(self.on_error)
        self.obs_bytetrack_thread.start()

        self.video_panel.slider.setEnabled(False)
        self._update_start_button_state()

    @QtCore.Slot()
    def on_open_dual_sync_clicked(self) -> None:
        """Switch to synchronized dual-source mode: Webcam (idx 0) + VR/OBS (idx 5)."""
        self.stop_inference()
        self.mode = "dual_sync"
        self.current_video_path = "Dual Source: Webcam + VR"
        self.control_panel.set_status("Mode: VR & Webcam (Sync)")
        self.append_text("Switched to Dual Source Sync mode (Webcam + VR side-by-side)")
        self.append_text("已切換至雙路同步模式 (Webcam + VR 左右拼接)")
        self._stop_all_source_threads()

        self.dual_sync_thread = DualSourceCameraThread(cam_idx=0, vr_idx=5)
        self.dual_sync_thread.signal_frame.connect(self.on_camera_frame)
        self.dual_sync_thread.signal_error.connect(self.on_error)
        self.dual_sync_thread.start()

        self.video_panel.slider.setEnabled(False)
        self._update_start_button_state()

    @QtCore.Slot()
    def on_open_free_switch_clicked(self) -> None:
        """Show source-selection dialog, then start FreeSwitchCameraThread."""
        # ── Initial source dialog ─────────────────────────────────────────────
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Free Switch — 選擇初始輸入源")
        dlg.setModal(True)
        dlg.setMinimumWidth(360)
        _dlg_layout = QtWidgets.QVBoxLayout(dlg)
        _dlg_layout.setSpacing(14)
        _dlg_layout.setContentsMargins(20, 20, 20, 20)

        _lbl = QtWidgets.QLabel("請問您初始的輸入源是？")
        _lbl.setStyleSheet("font-size: 14px; font-weight: 600;")
        _dlg_layout.addWidget(_lbl)

        _sub = QtWidgets.QLabel("啟動後可隨時點擊切換按鈕，攝影機不須重新連線。")
        _sub.setStyleSheet("color: #aaa; font-size: 12px;")
        _sub.setWordWrap(True)
        _dlg_layout.addWidget(_sub)

        _btn_row = QtWidgets.QHBoxLayout()
        _btn_row.setSpacing(10)
        chosen = [None]

        _dlg_btn_style = """
            QPushButton {
                background-color: #505050;
                border-radius: 10px;
                padding: 10px 14px;
                font-weight: 650;
            }
            QPushButton:hover { background-color: #3a86ff; color: white; }
        """
        for _label, _key in [
            ("📷  Webcam",      SOURCE_WEBCAM),
            ("🥽  VR",          SOURCE_VR),
            ("🔀  Webcam + VR", SOURCE_DUAL),
        ]:
            _b = QtWidgets.QPushButton(_label)
            _b.setStyleSheet(_dlg_btn_style)
            _b.clicked.connect(lambda _, k=_key: (chosen.__setitem__(0, k), dlg.accept()))
            _btn_row.addWidget(_b)

        _dlg_layout.addLayout(_btn_row)
        dlg.exec()

        if chosen[0] is None:
            return  # User closed dialog without choosing

        initial_source: str = chosen[0]

        # ── Start Free Switch mode ─────────────────────────────────────────────
        self.stop_inference()
        self.mode = "free_switch"
        self.current_video_path = f"FreeSwitch:{initial_source}"
        self.control_panel.set_status(f"Mode: Free Switch  ({initial_source})")
        self.append_text(f"[FreeSwitch] 初始來源：{initial_source}  (兩組攝影機同時開啟)")
        self._stop_all_source_threads()

        self.free_switch_thread = FreeSwitchCameraThread(
            initial_source=initial_source,
            cam_idx=0,
            vr_idx=5,
        )
        self.free_switch_thread.signal_frame.connect(self.on_camera_frame)
        self.free_switch_thread.signal_error.connect(self.on_error)
        self.free_switch_thread.signal_source_changed.connect(self._on_free_switch_source_changed)
        self.free_switch_thread.start()

        self.control_panel.set_free_switch_bar_visible(True, initial_source)
        self.video_panel.slider.setEnabled(False)
        self._update_start_button_state()

        # Start audience second-screen services
        self._start_audience_services()

    def _ensure_audience_token_server(self) -> None:
        """HTTP server for /audience + /api/audience/join — runs for app lifetime when enabled."""
        audience_cfg = self.configs.get("audience", {})
        if not audience_cfg.get("enabled", False):
            return
        if self._audience_token_server is not None:
            return
        lk_url = audience_cfg.get("livekit_url", "ws://localhost:7880")
        api_key = audience_cfg.get("api_key", "devkey")
        api_secret = audience_cfg.get("api_secret", "")
        room_name = audience_cfg.get("room", "broadcast-room")
        port = int(audience_cfg.get("port", 8080))
        self._audience_token_server = AudienceTokenServer(
            lk_url, api_key, api_secret, room_name, port=port
        )
        self._audience_token_server.start()

    def _stop_audience_token_server(self) -> None:
        if self._audience_token_server is not None:
            self._audience_token_server.stop()
            self._audience_token_server = None

    @QtCore.Slot(np.ndarray)
    def _deliver_audience_vr_frame(self, frame_rgb: np.ndarray) -> None:
        """Forward VR frames to LiveKit publisher.

        Use this stable MainWindow slot instead of connecting worker signals directly to
        AudiencePublisher.push_video_frame — bound publisher methods + QueuedConnection
        can invoke the slot with a broken ``self`` (method-wrapper), raising AttributeError.
        """
        pub = self._audience_publisher
        if pub is not None:
            pub.push_video_frame(frame_rgb)

    def _clear_all_pcm_sinks(self) -> None:
        """Clear audience PCM sinks on every TTS backend that supports them."""
        from .core.models import openai_tts as _oai_tts_mod
        _oai_tts_mod.register_pcm_sink(None, mute_local=False)

    def _audience_pcm_tts_module(self):
        """Core TTS module for the active dropdown mode (OpenAI only)."""
        mode = getattr(self, "tts_mode", "none")
        if mode == "openai":
            from .core.models import openai_tts as mod
            return mod
        return None

    def _register_audience_pcm_sink(self) -> None:
        """Route TTS PCM to LiveKit for the currently selected TTS engine."""
        mod = self._audience_pcm_tts_module()
        if mod is None or self._audience_publisher is None:
            return
        mod.register_pcm_sink(
            self._audience_publisher.push_audio_chunk,
            mute_local=True,
            flush_callback=self._audience_publisher.flush_pending_audio,
        )

    def _stop_audience_publisher_only(self) -> None:
        """Tear down LiveKit publisher + TTS sink; keep HTTP token server running."""
        self._clear_all_pcm_sinks()
        if self._audience_publisher is not None:
            try:
                if self.free_switch_thread is not None:
                    self.free_switch_thread.signal_vr_frame.disconnect(
                        self._deliver_audience_vr_frame
                    )
            except (RuntimeError, AttributeError, TypeError):
                pass
            self._audience_publisher.stop()
            self._audience_publisher = None

    def _start_audience_services(self) -> None:
        """Free Switch: LiveKit publisher + TTS PCM sink (HTTP server already up)."""
        audience_cfg = self.configs.get("audience", {})
        if not audience_cfg.get("enabled", False):
            return

        self._ensure_audience_token_server()
        self._stop_audience_publisher_only()

        lk_url = audience_cfg.get("livekit_url", "ws://localhost:7880")
        api_key = audience_cfg.get("api_key", "devkey")
        api_secret = audience_cfg.get("api_secret", "devsecret")
        room_name = audience_cfg.get("room", "broadcast-room")

        self._audience_publisher = AudiencePublisher(
            lk_url, api_key, api_secret, room_name
        )
        self._audience_publisher.reset_session_telemetry()
        self._audience_publisher.start()

        if self.free_switch_thread is not None:
            self.free_switch_thread.signal_vr_frame.connect(
                self._deliver_audience_vr_frame,
                QtCore.Qt.QueuedConnection,
            )

        self._register_audience_pcm_sink()

    def _stop_audience_services(self) -> None:
        """Tear down publisher only (HTTP /audience stays up for the app lifetime)."""
        self._stop_audience_publisher_only()

    @QtCore.Slot(str)
    def on_switch_source(self, source: str) -> None:
        """Instantly switch the active camera source inside FreeSwitchCameraThread."""
        if self.free_switch_thread is not None:
            self.free_switch_thread.set_active_source(source)
        # Highlight button immediately (don't wait for signal_source_changed round-trip)
        self.control_panel.highlight_switch_source(source)
        if self._fs_auto_cycle_timer.isActive():
            try:
                self._fs_cycle_idx = self._fs_cycle_order.index(source)
            except ValueError:
                pass

    def _stop_free_switch_auto_cycle(self) -> None:
        """Stop the 10s source rotation and clear the toggle (no signal loop)."""
        self._fs_auto_cycle_timer.stop()
        self._set_free_switch_manual_buttons_enabled(True)
        b = self.control_panel.btn_fs_auto_cycle
        if b.isChecked():
            b.blockSignals(True)
            b.setChecked(False)
            b.blockSignals(False)

    def _set_free_switch_manual_buttons_enabled(self, enabled: bool) -> None:
        """While 10s auto-rotate runs, disable webcam/vr/dual to avoid fighting the timer."""
        p = self.control_panel
        p.btn_sw_webcam.setEnabled(enabled)
        p.btn_sw_vr.setEnabled(enabled)
        p.btn_sw_dual.setEnabled(enabled)

    @QtCore.Slot(bool)
    def _on_free_switch_auto_cycle_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._fs_auto_cycle_timer.stop()
            self._set_free_switch_manual_buttons_enabled(True)
            return
        if self.mode != "free_switch" or self.free_switch_thread is None:
            self.append_text("[FreeSwitch] 10s 輪播需在 Free Switch 模式且攝影機已啟動時使用")
            self._stop_free_switch_auto_cycle()
            return
        cur = self.free_switch_thread.active_source  # @property, not a method call
        try:
            self._fs_cycle_idx = self._fs_cycle_order.index(cur)
        except ValueError:
            self._fs_cycle_idx = 0
        self._set_free_switch_manual_buttons_enabled(False)
        # Advance to next source immediately, then let the timer fire every 10s.
        self._on_free_switch_auto_cycle_tick()
        self._fs_auto_cycle_timer.start()

    @QtCore.Slot()
    def _on_free_switch_auto_cycle_tick(self) -> None:
        if self.mode != "free_switch" or self.free_switch_thread is None:
            self._stop_free_switch_auto_cycle()
            return
        order = self._fs_cycle_order
        self._fs_cycle_idx = (self._fs_cycle_idx + 1) % len(order)
        self.on_switch_source(order[self._fs_cycle_idx])

    @QtCore.Slot(str)
    def _on_free_switch_source_changed(self, source: str) -> None:
        """Called when FreeSwitchCameraThread confirms the new source."""
        self.control_panel.highlight_switch_source(source)
        self.append_text(f"[FreeSwitch] 已切換至：{source}")

    def _apply_tts_settings_before_start(self) -> None:
        """根據目前模式套用對應設定。"""
        self.tts_mode = self.control_panel.get_tts_mode()

        if self.tts_mode == "openai":
            voice = self.control_panel.get_openai_voice()
            speed = self.control_panel.get_openai_speed()
            lang = self.control_panel.get_tts_language()

            # Apply language to both Gemini broadcaster and OpenAI TTS
            from .core.models.gemini_broadcaster import set_language as _gb_set_lang
            from .core.models.openai_tts import set_tts_language as _tts_set_lang
            _gb_set_lang(lang)
            _tts_set_lang(lang)

            # Sync P1 interrupt filler with selected language
            self._P1_FILLER = self._P1_FILLER_ZH if lang == "zh" else self._P1_FILLER_EN

            self.signal_tts_apply_settings.emit(voice, float(speed))

        elif self.tts_mode == "local":
            exag = self.control_panel.get_local_exaggeration()
            cfg = self.control_panel.get_local_cfg()
            self.signal_local_tts_apply_settings.emit(float(exag), float(cfg))

    @QtCore.Slot()
    def on_start_clicked(self) -> None:
        if self.is_inference_running:
            self.stop_inference()
            return

        if not self.model_ready:
            self.append_text("請先連線遠端伺服器，或等待本機模型載入完成")
            return

        # Session file: same layout for every input mode; header records where LiveCC runs
        if self._socket_runner is not None:
            inference_backend = "remote"
        elif getattr(self, "livecc_model", None) is not None:
            inference_backend = "local"
        else:
            inference_backend = "unknown"
        log_path = self.session_logger.start_new_session(
            self.mode, inference_backend=inference_backend
        )
        print(f"[Main] Session log started: {log_path}")

        self._clear_local_inference_telemetry_for_new_session()

        style_key = self.control_panel.get_selected_style_key()
        style_label = self.control_panel.get_selected_style_label()

        if self.prompt_manager is not None:
            prompt = self.prompt_manager.livecc_query()
            from .core.models.gemini_broadcaster import set_style
            set_style(style_key)
        else:
            prompt = "Describe only what you see on screen right now in one objective sentence."

        if self.prompt_manager is not None:
            stem = self.prompt_manager.livecc_response_prefix()
            response_prefix = (stem + " ") if stem else ""
        else:
            response_prefix = ""
        if hasattr(self, "livecc_worker") and self.livecc_worker is not None:
            self.livecc_worker.response_prefix = response_prefix
        if hasattr(self, "cam_worker") and self.cam_worker is not None:
            self.cam_worker.response_prefix = response_prefix
        self._use_gemini = True

        # Apply TTS settings before start, then pre-connect OpenAI Realtime (warmup).
        self._apply_tts_settings_before_start()
        if self.tts_mode == "openai":
            self.signal_tts_warmup.emit()

        self.is_inference_running = True
        self._reset_broadcast_timeline()

        # Start audio recording only when user has opted in via checkbox
        if hasattr(self.control_panel, "chk_record") and self.control_panel.chk_record.isChecked():
            rec_path = self._audio_recorder.start(self.mode)
            print(f"[Main] Audio recording started: {rec_path}")
            if self.tts_mode == "openai":
                from .core.models import openai_tts as _oai_tts_mod
                _oai_tts_mod.register_recording_sink(self._audio_recorder.write_chunk)
            elif self.tts_mode == "local":
                from .core.models import chatterbox_tts as _local_tts_mod
                _local_tts_mod.register_recording_sink(self._audio_recorder.write_chunk)

        self.control_panel.set_start_button_state(True)
        self.control_panel.set_tts_controls_enabled(False)  # Lock during inference

        # Start background Gemini loop (restart if stopped from previous session).
        if not getattr(self, "_gemini_bg_loop_started", False):
            self._gemini_bg_loop_started = True
            self.gemini_bg_worker._stop_requested = False
            self.gemini_bg_worker._paused = False
            QtCore.QMetaObject.invokeMethod(
                self.gemini_bg_worker, "run_background_loop",
                QtCore.Qt.QueuedConnection,
            )
        self.text_output.setText("")

        self.append_text(f"Starting inference (Style: {style_label}, TTS: {self.tts_mode})")
        self.append_text(f"開始推論 (Style: {style_label}, TTS: {self.tts_mode})")
        if self.tts_mode == "none":
            self.append_text(
                "提示：TTS 為「不啟用 (Mute)」— 不會播出語音。需要朗讀請選 OpenAI TTS。"
            )
            self.statusBar().showMessage("TTS: 靜音（不播語音）", 5000)

        if self.mode == "file":
            if hasattr(self, "_pending_segments"):
                self._pending_segments.clear()
            self._playback_sec = 0.0
            # _livecc_start_wall aligns LiveCC segment latency stats with the same anchor.
            self._livecc_start_wall = self._broadcast_start_wall

        if self.mode == "file":
            if self.video_thread:
                self.video_thread.requestStop()
                self.video_thread.wait()

            self.video_thread = VideoThread(self.current_video_path)
            self.video_thread.signal_video_loaded.connect(self.video_panel.set_duration)
            self.video_thread.signal_frame.connect(self.on_video_frame)
            self.video_thread.signal_video_ended.connect(self.on_finished)
            self.video_thread.signal_invalid_video.connect(self.on_error)
            self.video_thread.start()

            if self._socket_runner is not None:
                # Remote: frames are streamed via on_video_frame → send_frame
                self._socket_runner.start_inference(self.mode, prompt)
                self._start_remote_client_ram_monitor()
            elif self.livecc_model is not None:
                # Local: pass file path directly to local LiveCC worker
                self._livecc_run_id += 1
                self.signal_start_livecc.emit(self.current_video_path, prompt, self._livecc_run_id)
            else:
                self.append_text(
                    "未連線遠端且本機無 LiveCC 模型，無法開始。請先連線遠端或安裝本機模型。"
                )
                self.is_inference_running = False
                self.control_panel.set_start_button_state(False)
                self.control_panel.set_tts_controls_enabled(True)
                return

        elif self.mode in ("camera", "obs", "obs_track", "dual_sync", "free_switch"):
            if self._socket_runner is not None:
                # ByteTrack runs only on the client; the server decodes JPEG + LiveCC. Report real
                # mode so server logs match the GUI (MSG_START "mode" is informational only).
                self._socket_runner.start_inference(self.mode, prompt)
                self._start_remote_client_ram_monitor()
            elif self.livecc_model is not None:
                self.signal_start_camera_livecc.emit(prompt)
            else:
                self.append_text("未連線遠端，無法開始。請先按「連線遠端伺服器」。")
                self.is_inference_running = False
                self.control_panel.set_start_button_state(False)
                self.control_panel.set_tts_controls_enabled(True)
                return

    def stop_inference(self) -> None:
        if not self.is_inference_running:
            return
        if self._busy_stopping_inference:
            return
        obs_tracker_for_avg = None
        self._busy_stopping_inference = True
        try:
            self.append_text("停止推論")
            self.is_inference_running = False
            self._stop_remote_client_ram_monitor()
            self._last_tts_raw_text = ""
            self._last_tts_emit_ts = 0.0
            self._tts_protect_until = 0.0
            self._tts_protected_priority = 5
            if hasattr(self, "_pending_segments"):
                self._pending_segments.clear()
            self._pending_livecc_fragment = None
            self._livecc_start_wall = 0.0
            self._broadcast_start_wall = 0.0
            self._broadcast_playback_sec = 0.0
            if self.tts_mode == "openai":
                try: self.signal_tts_interrupt.emit()
                except Exception: pass
            elif self.tts_mode == "local":
                try: self.signal_local_tts_interrupt.emit()
                except Exception: pass
            if hasattr(self, "gemini_worker"):
                self.gemini_worker.flush_and_abort()
            if hasattr(self, "gemini_bg_worker"):
                self.gemini_bg_worker.requestStop()
            self._gemini_bg_loop_started = False  # allow loop restart on next session
            self._post_p1_pending = False

            # Remote: tell server to stop
            if self._socket_runner is not None:
                try:
                    self._socket_runner.stop_inference()
                except Exception:
                    pass

            if hasattr(self, "livecc_worker") and self.livecc_worker is not None:
                self.livecc_worker.requestStop()
            if hasattr(self, "cam_worker") and self.cam_worker is not None:
                self.cam_worker.requestStop()

            if self.mode == "file" and self.video_thread:
                self.video_thread.requestStop()
                self.video_thread.wait()
                self.video_thread = None

            if self.mode == "obs" and self.obs_thread is not None:
                self.obs_thread.requestStop()
                self._join_worker_thread_smooth(self.obs_thread)
                self.obs_thread = None

            # Webcam+ByteTrack stays alive here: inference ended above; preview keeps updating.
            if self.mode == "obs_track" and self.obs_bytetrack_thread is not None:
                obs_tracker_for_avg = getattr(
                    self.obs_bytetrack_thread, "_active_tracker", None
                )

            if self.mode == "dual_sync" and self.dual_sync_thread is not None:
                self.dual_sync_thread.requestStop()
                self._join_worker_thread_smooth(self.dual_sync_thread)
                self.dual_sync_thread = None

            self._flush_local_inference_telemetry_summary(obs_tracker_for_avg)

            # Session latency summary (same as worker.stop() on window close)
            if self.tts_mode == "openai":
                from .core.models.openai_tts import print_tts_stats as _print_tts_stats
                _print_tts_stats()

            # Stop audio recording and clear recording sinks
            from .core.models import openai_tts as _oai_tts_mod
            from .core.models import chatterbox_tts as _local_tts_mod
            _oai_tts_mod.clear_recording_sink()
            _local_tts_mod.clear_recording_sink()
            if self._audio_recorder.is_recording:
                rec_path = self._audio_recorder.stop()
                if rec_path:
                    self.append_text(f"錄音已儲存: {rec_path}")

            self._last_track_preview_mono = 0.0
            self.control_panel.set_start_button_state(False)
            self.control_panel.set_tts_controls_enabled(True)  # Unlock after stop
        finally:
            self._busy_stopping_inference = False

    # ---------------- Frame handlers ----------------

    @QtCore.Slot(np.ndarray, int, float)
    def on_video_frame(self, frame_rgb: np.ndarray, frame_idx: int, fps: float) -> None:
        self.video_panel.update_frame(frame_rgb)
        self.video_panel.set_position(frame_idx, fps)

        if self.mode == "file" and fps and fps > 0:
            sec = float(frame_idx) / float(fps)
            self._playback_sec = sec
            if self.is_inference_running:
                self._broadcast_playback_sec = sec

        # Stream video frames to remote server for file-mode inference
        if self.mode == "file" and self.is_inference_running and self._socket_runner is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._socket_runner.send_frame(frame_bgr, self._playback_sec)

    @QtCore.Slot(np.ndarray)
    def on_camera_frame(self, frame_rgb: np.ndarray) -> None:
        # obs_track no longer uses this slot — CameraByteTrackThread emits to
        # on_obs_track_frame (preview) and on_obs_track_subject_frame (remote send).
        self.video_panel.update_frame(frame_rgb)
        if not self.is_inference_running:
            return
        if self.mode not in ("camera", "obs", "dual_sync", "free_switch"):
            return

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        t_relative = self._broadcast_timeline_sec()

        if self._socket_runner is not None:
            self._socket_runner.send_frame(frame_bgr, t_relative)
        elif hasattr(self, "cam_worker") and self.cam_worker is not None:
            self.cam_worker.push_frame(frame_bgr, t_relative)

    @QtCore.Slot(np.ndarray)
    def on_obs_track_frame(self, annotated_bgr: np.ndarray) -> None:
        """Display ByteTrack annotated frame (BGR) in the video panel."""
        self.video_panel.update_frame(annotated_bgr, is_bgr=True)

    @QtCore.Slot(np.ndarray)
    def on_obs_track_subject_frame(self, subject_crop_rgb: np.ndarray) -> None:
        """Forward the padded subject crop (RGB) to LiveCC inference (local or remote)."""
        if self.mode != "obs_track":
            return
        # Before Start: thread still emits subject crops — expected; do not warn.
        if not self.is_inference_running:
            return

        fixed = cv2.resize(subject_crop_rgb, (640, 480))
        subject_bgr = cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR)
        t_relative = self._broadcast_timeline_sec()

        if self._socket_runner is not None:
            self._socket_runner.send_frame(subject_bgr, t_relative)
        elif hasattr(self, "cam_worker") and self.cam_worker is not None:
            self.cam_worker.push_frame(subject_bgr, t_relative)

    @QtCore.Slot(str, str, str)
    def on_backend_log(self, source: str, level: str, msg: str) -> None:
        """Receive logs from background workers and save to session log."""
        if hasattr(self, 'session_logger'):
            self.session_logger.log_system(source, level, msg)

    # ---------------- Model callbacks ----------------

    @staticmethod
    def _extract_segment_texts(data: object) -> tuple[str, str]:
        """Return (display_text, tts_text) from a segment payload (dict or str)."""
        if isinstance(data, dict):
            # Gemini broadcast format
            if "broadcast_text" in data:
                priority = data.get("priority", 1)
                label = data.get("action_label", "")
                broadcast = data.get("broadcast_text", "")
                display_text = f"[P{priority}] {label}: {broadcast}" if label else f"[P{priority}] {broadcast}"
                tts_text = broadcast if data.get("should_speak") else ""
                return display_text, tts_text
            # LiveCC objective JSON format
            event = data.get("event", "")
            if event and event != "raw_description":
                display_text = json.dumps(data, ensure_ascii=False)
                tts_text = event.replace("_", " ")
            else:
                raw = data.get("metadata", {}).get("raw", "")
                display_text = raw
                tts_text = raw
        else:
            display_text = str(data)
            tts_text = display_text
        return display_text, tts_text

    @QtCore.Slot()
    def on_model_loaded(self) -> None:
        self.model_ready = True
        self.statusBar().showMessage("Model Ready")
        self.control_panel.set_status("Model ready, please select source")
        self._update_start_button_state()

    def _is_duplicate_tts(self, raw_text: str, window: Optional[float] = None) -> bool:
        """True when raw_text is near-identical to the last TTS emit within the dedup window."""
        if not self._last_tts_raw_text:
            return False
        win = self._DEDUP_WINDOW_S if window is None else window
        if time.time() - self._last_tts_emit_ts > win:
            return False
        new_words = set(raw_text.lower().split())
        last_words = set(self._last_tts_raw_text.lower().split())
        if not new_words or not last_words:
            return False
        overlap = len(new_words & last_words) / min(len(new_words), len(last_words))
        return overlap >= self._DEDUP_THRESHOLD

    @QtCore.Slot(float, float, str)
    def on_remote_segment(self, start_t: float, stop_t: float, text: str) -> None:
        """Dedicated slot for SocketClientRunner.signal_segment (Signal(float, float, str)).
        PySide6 QueuedConnection requires an exact type match between Signal and @Slot;
        routing str through @Slot(object) silently drops the call across threads."""
        if hasattr(self, "session_logger"):
            self.session_logger.log_commentary(text)
        try:
            # Same Fast-Slow Blade path as local LiveCC — never send raw English
            # captions directly to TTS (only Gemini broadcast_text is spoken).
            self._route_segment(start_t, stop_t, self._livecc_event_dict(text))
        except Exception:
            logging.exception("[on_remote_segment] unhandled exception")

    @QtCore.Slot(float, float, object)
    def on_segment(self, start_t: float, stop_t: float, data: object) -> None:
        try:
            self._on_segment_impl(start_t, stop_t, data)
        except Exception:
            logging.exception("[on_segment] unhandled exception")

    def _on_segment_impl(self, start_t: float, stop_t: float, data: object) -> None:
        logging.info("[on_segment] called start_t=%.2f use_gemini=%s", start_t, self._use_gemini)
        now = time.time()
        # Use per-segment enqueue timestamp (embedded by GeminiWorker) for accurate E2E latency.
        # _gemini_ref_ts is a shared scalar that gets overwritten by newer segments, so it only
        # works correctly when there is no queue backlog.
        enqueue_ts = None
        if self._use_gemini and isinstance(data, dict):
            enqueue_ts = data.get("_enqueue_ts")
        if enqueue_ts is not None:
            e2e_latency = now - enqueue_ts
            if not hasattr(self, "_gemini_latencies"):
                self._gemini_latencies = []
            self._gemini_latencies.append(e2e_latency)
            avg = sum(self._gemini_latencies) / len(self._gemini_latencies)
            logging.info(
                "[延遲] 段落 %.1f-%.1fs → TTS 排隊 E2E=%.2fs  (平均=%.2fs, n=%d)",
                start_t, stop_t, e2e_latency, avg, len(self._gemini_latencies),
            )
            ref_ts = enqueue_ts
        elif self._use_gemini and hasattr(self, "_gemini_ref_ts"):
            ref_ts = self._gemini_ref_ts  # fallback: non-Gemini or old path
        else:
            ref_ts = now

        # [延遲][Gemini] Stage 2: time from "frame appeared" to Gemini's broadcast
        # output reaching here. Also re-anchor ref_ts to the frame's wall-clock
        # timestamp so Stage 3 (voice latency) is measured from the same origin.
        is_background = isinstance(data, dict) and data.get("_background")
        if self.mode == "file" and self._livecc_start_wall > 0 and not is_background:
            frame_wall_ts = self._livecc_start_wall + stop_t
            latency = now - frame_wall_ts
            if not hasattr(self, "_frame_gemini_latencies"):
                self._frame_gemini_latencies = []
            self._frame_gemini_latencies.append(latency)
            avg = sum(self._frame_gemini_latencies) / len(self._frame_gemini_latencies)
            logging.info(
                "[延遲][Gemini] 段落 %.1f-%.1fs → Gemini 輸出 latency=%.2fs (平均=%.2fs, n=%d)",
                start_t, stop_t, latency, avg, len(self._frame_gemini_latencies),
            )
            ref_ts = frame_wall_ts

        display_text, tts_text = self._extract_segment_texts(data)

        # Resolve priority for this segment (Gemini dict has it; fallback to 5).
        # Gemini-originated content (carries _enqueue_ts) is time-decayed and
        # clamped against the active protection window — this only affects its
        # own queue placement (_register_tts_priority), never an interrupt.
        seg_priority = 5
        gemini_priority_jump = False
        if isinstance(data, dict):
            seg_priority = int(data.get("priority", 5))
            enqueue_ts = data.get("_enqueue_ts")
            base_priority = seg_priority
            if enqueue_ts is not None:
                seg_priority = self._effective_gemini_priority(seg_priority, enqueue_ts)
                gemini_priority_jump = seg_priority <= 2
                self._register_tts_priority(base_priority)

        # MatchTracker scoring — only when dual-team match mode is enabled
        if isinstance(data, dict) and "action_label" in data:
            action_label = data["action_label"] or ""
            if action_label.startswith("score_"):
                from .core.models.gemini_broadcaster import match_tracking_enabled
                if match_tracking_enabled():
                    team = action_label[len("score_"):]  # "red" or "blue"
                    match_tracker.add_score(team)
                    match_tracker.set_last_event(action_label)
                    red, blue = match_tracker.get_scores()
                    logging.info("[MatchTracker] %s scored → Red %d : Blue %d", team, red, blue)

        # Camera mode：沒有播放器時間軸可排程，所以直接顯示/唸
        if self.mode != "file":
            if not self.is_inference_running:
                return
            ui_body = self._segment_ui_body(data, display_text, priority_jump=gemini_priority_jump)
            # Background Gemini uses wall-clock epoch for start_t — show stream timeline instead.
            if isinstance(data, dict) and data.get("_background"):
                cur = self._video_playback_sec_for_log()
                line = f"[{self._fmt_time(cur)}-{self._fmt_time(cur)}] {ui_body}"
            else:
                line = f"[{self._fmt_time(start_t)}-{self._fmt_time(stop_t)}] {ui_body}"
            self._append_ui(line)

            if not tts_text.strip():
                return
            if tts_text.strip().lower() == "silence":
                return  # model silence sentinel — skip TTS
            if self.tts_mode == "openai" and self._use_gemini and not self._broadcast_tts_allowed(data):
                return
            # P1 scoring plays must always be voiced; only dedup routine commentary.
            if seg_priority > 1 and self._is_duplicate_tts(tts_text):
                return
            now = time.time()
            self._last_tts_raw_text = tts_text
            self._last_tts_emit_ts = now
            log_meta = self._build_tts_log_meta(data, tts_text, seg_priority, start_t, stop_t)
            if self.tts_mode == "openai":
                if seg_priority <= 1:
                    self._post_p1_pending = True
                    self._arm_p1_fallback_timer()
                self._emit_openai_tts_speak(
                    tts_text, seg_priority, ref_ts, start_t, stop_t,
                    cut_current=(seg_priority <= 1), log_meta=log_meta,
                )
            elif self.tts_mode == "local":
                self.signal_local_tts_speak.emit(tts_text)
            return

        # File mode: Enter queue, wait until video playback reaches corresponding time (stay in sync)
        if not hasattr(self, "_pending_segments"):
            self._pending_segments = deque()

        if isinstance(data, dict) and data.get("_background"):
            # epoch start_t would wedge at the head of _pending_segments forever
            cur = float(getattr(self, "_playback_sec", 0.0))
            ui_body = self._segment_ui_body(data, display_text, priority_jump=gemini_priority_jump)
            line = f"[{self._fmt_time(cur)}-{self._fmt_time(cur)}] {ui_body}"
            self._append_ui(line)
        else:
            # Tag segment so the playback-time consumer can show the priority-jump marker
            seg_data = dict(data, _priority_jump=True) if (gemini_priority_jump and isinstance(data, dict)) else data
            self._pending_segments.append((float(start_t), float(stop_t), seg_data))

        if not tts_text.strip():
            return
        if tts_text.strip().lower() == "silence":
            return  # model silence sentinel — skip TTS
        if self.tts_mode == "openai" and self._use_gemini and not self._broadcast_tts_allowed(data):
            return
        # P1 scoring plays must always be voiced; only dedup routine commentary.
        if seg_priority > 1 and self._is_duplicate_tts(tts_text):
            return
        now = time.time()
        self._last_tts_raw_text = tts_text
        self._last_tts_emit_ts = now
        log_meta = self._build_tts_log_meta(data, tts_text, seg_priority, start_t, stop_t)
        if self.tts_mode == "openai":
            if seg_priority <= 1:
                self._post_p1_pending = True
                self._arm_p1_fallback_timer()
            self._emit_openai_tts_speak(
                tts_text, seg_priority, ref_ts, start_t, stop_t,
                cut_current=(seg_priority <= 1), log_meta=log_meta,
            )
        elif self.tts_mode == "local":
            self.signal_local_tts_speak.emit(tts_text)

    @QtCore.Slot(int)
    def on_livecc_finished(self, run_id: int) -> None:
        if run_id != self._livecc_run_id:
            logging.info("[on_livecc_finished] stale signal (run_id=%d, current=%d), ignored", run_id, self._livecc_run_id)
            return
        self.on_finished()

    @QtCore.Slot()
    def _on_tts_done(self) -> None:
        if self._post_p1_pending:
            self._post_p1_pending = False
            self._p1_filler_armed = False
            if hasattr(self, "_p1_fallback_timer") and self._p1_fallback_timer is not None:
                self._p1_fallback_timer.stop()
                self._p1_fallback_timer = None
            QtCore.QTimer.singleShot(1000, self._resume_gemini_background)
            logging.info("[P1 Silence] TTS done naturally, scheduling 1.0s before Gemini resumes")

    def _arm_p1_fallback_timer(self) -> None:
        """Start a 12s safety timer that force-resumes background if signal_tts_done never fires
        (e.g. P1 was interrupted before natural completion)."""
        if hasattr(self, "_p1_fallback_timer") and self._p1_fallback_timer is not None:
            self._p1_fallback_timer.stop()
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(12000)
        timer.timeout.connect(self._p1_fallback_resume)
        timer.start()
        self._p1_fallback_timer = timer

    @QtCore.Slot()
    def _p1_fallback_resume(self) -> None:
        self._p1_fallback_timer = None
        if self._post_p1_pending:
            self._post_p1_pending = False
            self._p1_filler_armed = False
            logging.warning("[P1 Fallback] signal_tts_done never fired after 12s — force-resuming background")
            self._resume_gemini_background()

    def _resume_gemini_background(self) -> None:
        if hasattr(self, "gemini_bg_worker") and self.is_inference_running:
            QtCore.QMetaObject.invokeMethod(
                self.gemini_bg_worker, "resume",
                QtCore.Qt.QueuedConnection,
            )
            logging.info("[P1 Silence] 1.0s elapsed, Gemini background resumed")

    @QtCore.Slot()
    def on_finished(self) -> None:
        self.append_text("Playback/Inference finished")
        self.stop_inference()

    @QtCore.Slot(str)
    def on_error(self, msg: str) -> None:
        self.append_text(f"Error: {msg}")
        self.append_text(f"錯誤：{msg}")
        self.stop_inference()

    @QtCore.Slot(str)
    def on_gemini_error(self, msg: str) -> None:
        """Gemini errors are non-fatal: log to file and UI but keep inference running."""
        self._ensure_log_dir()
        self._write_log(self.gemini_log_file, f"[Gemini ERROR] {msg}")
        self.append_text(f"[Gemini] 錯誤（推論繼續）：{msg}")

    # ---------------- Helpers ----------------

    def _load_video_preview(self, path: str) -> None:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                self.video_panel.update_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()

    def _update_start_button_state(self) -> None:
        # Allow start when: remote connected OR local model ready, and a source is selected
        ready = self._socket_runner is not None or self.model_ready
        can_start = ready and (self.current_video_path is not None)
        self.control_panel.btn_start.setEnabled(bool(can_start))

    def _ensure_log_dir(self) -> None:
        if not hasattr(self, "log_dir"):
            self.log_dir = _find_project_root(Path(__file__)) / "log"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.livecc_log_file = self.log_dir / "livecc_output.log"
            self.gemini_log_file = self.log_dir / "gemini_output.log"
            self.combination_log_file = self.log_dir / "combination_output.log"

    @staticmethod
    def _estimate_playback_duration(text: str) -> float:
        """CJK-aware spoken duration estimate for log end timestamps."""
        cjk_count = sum(
            1 for c in text
            if '一' <= c <= '鿿' or '぀' <= c <= 'ヿ'
        )
        ascii_only = ''.join(' ' if '一' <= c <= '鿿' else c for c in text)
        other_words = len(ascii_only.split())
        return max(cjk_count / 4.0 + other_words / 2.5, 0.5)

    def _build_tts_log_meta(
        self,
        data: object,
        tts_text: str,
        seg_priority: int,
        start_t: float,
        stop_t: float,
    ) -> Optional[dict]:
        """Build log payload written when TTS audio actually starts playing."""
        text = (tts_text or "").strip()
        if not text or text.lower() == "silence" or self._is_p1_filler_text(text):
            return None

        is_bg = isinstance(data, dict) and bool(data.get("_background"))
        livecc_raw = ""
        if isinstance(data, dict):
            livecc_raw = data.get("metadata", {}).get("raw", "") or ""
            event = data.get("event", "")
            if not livecc_raw and event and event != "raw_description":
                livecc_raw = event.replace("_", " ")
        elif isinstance(data, str):
            livecc_raw = data

        gemini_text = ""
        base_p = seg_priority
        if isinstance(data, dict):
            gemini_text = (data.get("broadcast_text") or "").strip()
            base_p = int(data.get("priority", seg_priority))

        combination_text = gemini_text if (self._use_gemini and gemini_text) else text

        return {
            "log": True,
            "log_livecc": bool(livecc_raw.strip()) and not is_bg,
            "log_gemini": bool(self._use_gemini and gemini_text),
            "log_combination": bool(combination_text.strip()) and not is_bg,
            "livecc_text": livecc_raw.strip(),
            "gemini_priority": base_p,
            "gemini_text": gemini_text,
            "combination_text": combination_text.strip(),
            "seg_start_t": float(start_t),
            "seg_stop_t": float(stop_t),
            "is_background": is_bg,
        }

    def _reset_broadcast_timeline(self) -> None:
        """Reset timeline anchor to admin Start — used by all modes."""
        self._broadcast_start_wall = time.time()
        self._broadcast_playback_sec = 0.0

    def _broadcast_timeline_sec(self) -> float:
        """Elapsed seconds on the broadcast timeline since admin Start."""
        if not self.is_inference_running:
            return 0.0
        if self.mode == "file":
            return float(getattr(self, "_broadcast_playback_sec", 0.0))
        wall = float(getattr(self, "_broadcast_start_wall", 0.0))
        if wall > 0:
            return max(0.0, time.time() - wall)
        return 0.0

    def _video_playback_sec_for_log(self) -> float:
        """Seconds since admin Start (file: frame index; live modes: wall clock)."""
        return self._broadcast_timeline_sec()

    @QtCore.Slot(object)
    def _on_tts_playback_log(self, meta: object) -> None:
        """Write LiveCC / Gemini / combination logs at TTS playback start.
        Also re-arms the P1 protection window the moment audio actually starts,
        so the window covers the full spoken duration regardless of generation latency."""
        if not isinstance(meta, dict) or not meta.get("log"):
            return
        if not self.is_inference_running:
            return
        self._ensure_log_dir()

        play_t = self._video_playback_sec_for_log()
        spoken = (meta.get("text") or meta.get("combination_text") or "").strip()
        est = self._estimate_playback_duration(spoken)
        log_start, log_end = play_t, play_t + est
        ts = f"{self._fmt_time(log_start)}-{self._fmt_time(log_end)}"
        logging.info(
            "[Log][TTS playback] video_t=%.2fs text=%r",
            play_t, spoken[:48],
        )

        # Re-arm P1/P2 protection window when audio actually starts to avoid it
        # expiring mid-sentence (Gemini generation latency is not accounted for
        # in the initial window set at interrupt time).
        gemini_pri = int(meta.get("gemini_priority", 5))
        if gemini_pri <= 2:
            self._tts_protected_priority = gemini_pri
            self._tts_protect_until = time.time() + est + 1.5
            if gemini_pri <= 1:
                self._p1_filler_armed = False
            logging.info("[P%d Guard] re-armed protect window: %.1fs + 1.5s buffer", gemini_pri, est)

        if meta.get("log_livecc") and meta.get("livecc_text"):
            self._write_log(
                self.livecc_log_file,
                f"[LiveCC] [{ts}] {meta['livecc_text']}",
            )
        if meta.get("log_gemini") and meta.get("gemini_text"):
            priority = meta.get("gemini_priority", "?")
            self._write_log(
                self.gemini_log_file,
                f"[Gemini] [{ts}] [P{priority}] {meta['gemini_text']}",
            )
        if meta.get("log_combination") and meta.get("combination_text"):
            self._write_log(
                self.combination_log_file,
                f"[{ts}] {meta['combination_text']}",
            )

    def _write_log(self, filepath: "Path", msg: str) -> None:
        try:
            cleaned_msg = msg.replace('\n', ' ').replace('\r', ' ')
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {cleaned_msg}\n")
        except Exception as e:
            print(f"[Logger] 無法寫入日誌: {e}")

    def append_text(self, msg: str) -> None:
        """Append a system/status message to the UI and write it to the log."""
        self.text_output.appendText(msg)
        self._ensure_log_dir()
        self._write_log(self.livecc_log_file, msg)

    def _append_ui(self, msg: str) -> None:
        """Append a segment line to the UI only — file logs are written at TTS playback."""
        self.text_output.appendText(msg)


    def _install_text_output_click_handler(self) -> None:
        '''Allow users to click a segment line in the log to seek to the corresponding time in the video.
        Try to find the internal QTextEdit/QPlainTextEdit/QTextBrowser without modifying TextOutputWidget.
        '''
        self._text_click_widget = None
        self._text_click_viewport = None

        # Look for child components (TextOutputWidget might wrap it)
        for cls in (QtWidgets.QTextBrowser, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit):
            w = self.text_output.findChild(cls)
            if w is not None:
                self._text_click_widget = w
                break

        # If the widget itself is a text widget
        if self._text_click_widget is None and isinstance(
            self.text_output, (QtWidgets.QTextBrowser, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)
        ):
            self._text_click_widget = self.text_output

        if self._text_click_widget is None:
            self.append_text("[System] Could not mount click-to-seek: Text output component not found.")
            return

        # Mouse events usually occur on the viewport
        # mouse event 多半在 viewport 上
        self._text_click_viewport = getattr(self._text_click_widget, "viewport", lambda: None)()
        if self._text_click_viewport is None:
            self._text_click_viewport = self._text_click_widget

        self._text_click_viewport.installEventFilter(self)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if getattr(self, "_text_click_viewport", None) is not None and obj is self._text_click_viewport:
            if event.type() == QtCore.QEvent.Type.MouseButtonRelease:
                try:
                    if event.button() == QtCore.Qt.MouseButton.LeftButton:
                        w = getattr(self, "_text_click_widget", None)
                        if w is None:
                            return False
                        pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                        if hasattr(w, "cursorForPosition"):
                            cursor = w.cursorForPosition(pos)
                            line = cursor.block().text()
                            t_sec = self._parse_seek_time_from_line(line)
                            if t_sec is not None:
                                self.seek_to_seconds(t_sec)
                                return True
                except Exception:
                    return False
        return super().eventFilter(obj, event)

    @staticmethod
    def _parse_seek_time_from_line(line: str) -> float | None:
        '''Support formats:
          [00:07.25-00:09.25] ...
          [00:07.25] ...
        Returns the seconds to seek (defaults to start time).
        '''
        s = line.strip()

        m = re.search(r"\[(\d{2}):(\d{2}(?:\.\d+)?)\s*-\s*(\d{2}):(\d{2}(?:\.\d+)?)\]", s)
        if m:
            mm = int(m.group(1)); ss = float(m.group(2))
            return mm * 60.0 + ss

        m = re.search(r"\[(\d{2}):(\d{2}(?:\.\d+)?)\]", s)
        if m:
            mm = int(m.group(1)); ss = float(m.group(2))
            return mm * 60.0 + ss

        return None

    def seek_to_seconds(self, t_sec: float) -> None:
        '''File mode: seek to specified seconds in video (converts to frame index).'''
        if self.mode != "file":
            return
        if not self.video_thread:
            return

        fps = float(getattr(self.video_panel, "fps", 30.0) or 30.0)
        frame_idx = int(round(max(0.0, float(t_sec)) * fps))

        # clamp
        try:
            frame_idx = max(0, min(frame_idx, int(self.video_panel.slider.maximum())))
        except Exception:
            frame_idx = max(0, frame_idx)

        # Update slider and emit seek request to video thread
        try:
            self.video_panel.slider.setValue(frame_idx)
        except Exception:
            pass

        self.video_thread.requestSeek(frame_idx)
        self._playback_sec = float(frame_idx) / fps
        self.control_panel.set_status(f"Seek to {self._fmt_time(self._playback_sec)}")
        self.control_panel.set_status(f"跳轉到 {self._fmt_time(self._playback_sec)}")

    @QtCore.Slot(int)
    def on_seek_requested(self, frame_idx: int) -> None:
        if self.mode == "file" and self.video_thread:
            self.video_thread.requestSeek(frame_idx)

    @staticmethod
    def _fmt_time(t: float) -> str:
        '''Format seconds to mm:ss.xx (keep 2 decimals to avoid repeated timestamps).'''
        t = max(0.0, float(t))
        m = int(t // 60)
        s = t - (m * 60)
        return f"{m:02d}:{s:05.2f}"

    @staticmethod
    def fmt_time_ms(ms: float) -> str:
        m, s = divmod(int(max(0, ms) / 1000), 60)
        return f"{m:02d}:{s:02d}"

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Stop inference before closing"""
        # Stop remote socket first
        if self._socket_runner is not None:
            try:
                self._socket_runner.stop_inference()
            except Exception:
                pass
            self._socket_runner.disconnect_and_quit()
            self._socket_runner = None

        # 先停推論
        self.stop_inference()

        # ✅ 停 camera thread
        if self.camera_thread:
            self.camera_thread.requestStop()
            self.camera_thread.wait(1000)
            self.camera_thread = None

        # ✅ 停 TTS（避免 QThread: Destroyed while thread is still running）
        try:
            self.signal_tts_stop.emit()
        except Exception:
            pass
        if hasattr(self, "tts_thread") and self.tts_thread:
            self.tts_thread.quit()
            self.tts_thread.wait(2000)

        # Stop local LiveCC worker threads (only exist when model was loaded locally)
        if hasattr(self, "cam_worker_thread") and self.cam_worker_thread:
            self.cam_worker_thread.quit()
            self.cam_worker_thread.wait(2000)
        if hasattr(self, "livecc_thread") and self.livecc_thread:
            self.livecc_thread.quit()
            self.livecc_thread.wait(2000)

        if hasattr(self, "gemini_thread") and self.gemini_thread:
            self.gemini_thread.quit()
            self.gemini_thread.wait(2000)
        if hasattr(self, "gemini_bg_thread") and self.gemini_bg_thread:
            self.gemini_bg_thread.quit()
            self.gemini_bg_thread.wait(2000)
        try:
            self.signal_local_tts_stop.emit()
        except: pass
        if hasattr(self, "local_tts_thread") and self.local_tts_thread:
            self.local_tts_thread.quit()
            self.local_tts_thread.wait(2000)

        if hasattr(self, "_stop_audience_publisher_only"):
            self._stop_audience_publisher_only()
        if hasattr(self, "_stop_audience_token_server"):
            self._stop_audience_token_server()

        super().closeEvent(event)