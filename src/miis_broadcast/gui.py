# src/miis_broadcast/gui.py

from __future__ import annotations

import os
import time
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
# from .workers.chatterbox_tts import ChatterboxTTSWorker  # [ChatterBox disabled]
from .widgets.text_output import TextOutputWidget
from .workers.openai_tts import OpenAITTSWorker
from .workers.obs_input import OBSCameraThread
from .workers.camera_bytetrack import CameraByteTrackThread
from .workers.dual_source import DualSourceCameraThread
from .workers.free_switch import FreeSwitchCameraThread, SOURCE_WEBCAM, SOURCE_VR, SOURCE_DUAL
# LiveCCWorker / LiveCCCameraWorker are imported lazily only when not using client-only mode
# so this process never loads the VLM on a thin client.
from .core.prompt.prompt_manager import PromptManager
from .core.utils.session_logger import SessionLogger
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

        self.signal_video_loaded.emit(frame_count, fps)
        delay_sec = 1.0 / fps
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
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
    requestOpenCameraTrack = QtCore.Signal()   # webcam + ByteTrack
    requestOpenOBS         = QtCore.Signal()
    requestOpenDualSync    = QtCore.Signal()   # Webcam + VR side-by-side
    requestOpenFreeSwitch  = QtCore.Signal()   # Free Switch (both cams always running)
    requestSwitchSource    = QtCore.Signal(str)  # "webcam" | "vr" | "dual"
    requestStart           = QtCore.Signal()
    requestFontScale       = QtCore.Signal(int)
    requestRemoteConnect   = QtCore.Signal(str, int)  # host, port
    requestRemoteDisconnect = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(380)
        self.setMaximumWidth(520)  # Keep the preview large and prevents the panel from being too squashed
        self.setup_ui()

    def apply_settings_metrics(self) -> None:
        """Rebuild label/value column widths and combo heights from current app font.

        Stylesheet fixed min-heights clash with enlarged fonts and clip combo text.
        Call after startup and whenever UI font scale changes.
        """
        if not hasattr(self, "cmb_tts"):
            return
        fm = self.fontMetrics()
        combo_h = max(36, fm.height() + 14)
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
            "UI Scaling:",
        )
        lw = max(fm.horizontalAdvance(t) for t in titles) + 12
        for lb in (
            self.l_tts,
            self.l_style,
            self.l_voice,
            self.l_speed,
            self.l_exag,
            self.l_cfg,
            self.l_ui,
        ):
            lb.setMinimumWidth(lw)
            lb.setMaximumWidth(lw)
        vw = max(
            fm.horizontalAdvance("999pt"),
            fm.horizontalAdvance("1.55x"),
        ) + 18
        for v in (
            self.lbl_speed_val,
            self.lbl_exag_val,
            self.lbl_cfg_val,
            self.lbl_ui_scale_val,
        ):
            v.setMinimumWidth(vw)
        for row in getattr(self, "_settings_rows", {}).values():
            row.setMinimumHeight(combo_h)

    def setup_ui(self) -> None:
        # Single column layout (no scroll area) — sidebar never shows a vertical scrollbar.
        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        _inner = QtWidgets.QWidget()
        outer_layout.addWidget(_inner)

        layout = QtWidgets.QVBoxLayout(_inner)
        layout.setSpacing(16)
        layout.setContentsMargins(14, 14, 14, 14)

        # Source
        grp_source = QtWidgets.QGroupBox("影像來源 (Source)")
        v_src = QtWidgets.QVBoxLayout(grp_source)
        v_src.setSpacing(10)
        v_src.setContentsMargins(14, 18, 14, 12)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(10)

        btn_style = """
            QPushButton {
                background-color: #505050;
                border-radius: 10px;
                padding: 10px 20px;
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
        self.btn_offline = QtWidgets.QPushButton("📁  Offline")
        self.btn_offline.setStyleSheet(btn_style)
        self.btn_offline.clicked.connect(lambda: self.requestOpenVideo.emit())

        # ── Button 2: Online live input (dropdown with 3 sub-modes) ──────
        self.btn_online = QtWidgets.QPushButton("🌐  Online  ▾")
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
                border-radius: 10px;
                padding: 12px 14px;
            }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:disabled { background-color: #555; color: #999; }
        """)
        v_src.addWidget(self.btn_open_remote)

        self.lbl_source_sub = QtWidgets.QLabel("Status: —")
        self.lbl_source_sub.setStyleSheet("color: #b5b5b5; font-size: 12px;")
        self.lbl_source_sub.setWordWrap(True)
        v_src.addWidget(self.lbl_source_sub)

        self.lbl_remote_badge = QtWidgets.QLabel("● 未連線")
        self.lbl_remote_badge.setStyleSheet("color: #888; font-size: 12px;")
        v_src.addWidget(self.lbl_remote_badge)

        # ── Free Switch: single row — equal-width buttons (no extra label row → no scroll)
        self.free_switch_bar = QtWidgets.QWidget()
        _bar_row = QtWidgets.QHBoxLayout(self.free_switch_bar)
        _bar_row.setContentsMargins(0, 4, 0, 0)
        _bar_row.setSpacing(5)

        _sw_style = """
            QPushButton {
                border-radius: 7px;
                padding: 5px 6px;
                font-weight: 600;
                font-size: 11px;
                background-color: #434343;
                color: #eaeaea;
                min-height: 26px;
                max-height: 28px;
            }
            QPushButton:hover { background-color: #555; }
            QPushButton:checked {
                background-color: #2962ff;
                color: white;
            }
        """
        _exp = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self.btn_sw_webcam = QtWidgets.QPushButton("鏡頭")
        self.btn_sw_vr     = QtWidgets.QPushButton("VR")
        self.btn_sw_dual   = QtWidgets.QPushButton("雙拼")
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

        _bar_row.addWidget(self.btn_sw_webcam, 1)
        _bar_row.addWidget(self.btn_sw_vr, 1)
        _bar_row.addWidget(self.btn_sw_dual, 1)

        self.free_switch_bar.setVisible(False)
        v_src.addWidget(self.free_switch_bar)

        layout.addWidget(grp_source)

        # Settings — one row widget per logical row (QHBoxLayout inside QVBoxLayout).
        # QGridLayout + addWidget(..., alignment=...) can mis-bind on some bindings and pile widgets up.
        grp_settings = QtWidgets.QGroupBox("推論設定 (Settings)")
        v_settings = QtWidgets.QVBoxLayout(grp_settings)
        v_settings.setContentsMargins(14, 18, 14, 12)
        v_settings.setSpacing(10)

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
        # self.cmb_tts.addItem("Local TTS", userData="local")  # [ChatterBox disabled]
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

        # --- UI scale slider ---
        self.l_ui = _make_lbl("UI Scaling:")
        self.slider_ui_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_ui_scale.setRange(10, 26)
        self.slider_ui_scale.setValue(14)
        self.lbl_ui_scale_val = QtWidgets.QLabel("14pt")
        _apply_val_label(self.lbl_ui_scale_val)

        _sp_exp = QtWidgets.QSizePolicy.Policy.Expanding
        _sp_fix = QtWidgets.QSizePolicy.Policy.Fixed
        for _s in (
            self.slider_speed,
            self.slider_exag,
            self.slider_cfg,
            self.slider_ui_scale,
        ):
            _s.setSizePolicy(_sp_exp, _sp_fix)

        self.apply_settings_metrics()

        _settings_row_combo("tts", self.l_tts, self.cmb_tts)
        _settings_row_combo("style", self.l_style, self.cmb_style)
        _settings_row_combo("voice", self.l_voice, self.cmb_voice)
        _settings_row_slider("speed", self.l_speed, self.slider_speed, self.lbl_speed_val)
        _settings_row_slider("exag", self.l_exag, self.slider_exag, self.lbl_exag_val)
        _settings_row_slider("cfg", self.l_cfg, self.slider_cfg, self.lbl_cfg_val)
        _settings_row_slider("ui", self.l_ui, self.slider_ui_scale, self.lbl_ui_scale_val)

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
        layout.addWidget(grp_action)

        self._init_remote_server_dialog()
        layout.addStretch(1)
        # End of inner scroll container

        # Signals (source buttons emit directly via menu lambdas)
        self.btn_start.clicked.connect(self.requestStart.emit)

        self.slider_speed.valueChanged.connect(lambda v: self.lbl_speed_val.setText(f"{v/100:.1f}x"))
        self.slider_exag.valueChanged.connect(lambda v: self.lbl_exag_val.setText(f"{v/100:.1f}"))
        self.slider_cfg.valueChanged.connect(lambda v: self.lbl_cfg_val.setText(f"{v/100:.1f}"))
        self.slider_ui_scale.valueChanged.connect(self.on_font_scale_changed)

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
            rows["speed"].setVisible(show_openai)
            rows["exag"].setVisible(show_local)
            rows["cfg"].setVisible(show_local)
            return

        # Fallback for partially initialized panels.
        for w in (self.l_voice, self.cmb_voice, self.l_speed, self.slider_speed, self.lbl_speed_val):
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
        self.slider_speed.setEnabled(enabled)

        self.slider_exag.setEnabled(enabled)
        self.slider_cfg.setEnabled(enabled)

    def on_font_scale_changed(self, value: int) -> None:
        self.lbl_ui_scale_val.setText(f"{value}pt")
        self.requestFontScale.emit(value)

    def get_tts_mode(self) -> str:
        return self.cmb_tts.currentData()

    def get_selected_style_key(self) -> str:
        return self.cmb_style.currentData()

    def get_selected_style_label(self) -> str:
        return self.cmb_style.currentText()

    def get_openai_voice(self) -> str:
        v = self.cmb_voice.currentData()
        return str(v) if v is not None else "coral"

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
        if visible:
            self.highlight_switch_source(active_source)

    def highlight_switch_source(self, source: str) -> None:
        """Update which switch button appears active (checked/highlighted)."""
        self.btn_sw_webcam.setChecked(source == "webcam")
        self.btn_sw_vr.setChecked(    source == "vr")
        self.btn_sw_dual.setChecked(  source == "dual")

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
    signal_start_livecc = QtCore.Signal(str, str)
    signal_start_camera_livecc = QtCore.Signal(str)

    # Remote obs_track: suppress raw-camera overlay this long after each PREVIEW (seconds).
    # Keeps the last annotated frame visible between PREVIEW deliveries (~15 Hz).
    # Falls back to raw camera if no PREVIEW arrives for this long (e.g. disconnect).
    _OBS_TRACK_PREVIEW_HOLD_SEC = 2.50

    # signal to apply settings to tts thread to avoid calling slots directly on the main thread
    signal_tts_apply_settings = QtCore.Signal(str, float)
    signal_tts_stop = QtCore.Signal()

    signal_tts_speak = QtCore.Signal(str)
    signal_tts_interrupt = QtCore.Signal()
    # [ChatterBox disabled]
    # signal_local_tts_apply_settings = QtCore.Signal(float, float)
    # signal_local_tts_speak = QtCore.Signal(str)
    # signal_local_tts_interrupt = QtCore.Signal()
    # signal_local_tts_stop = QtCore.Signal()

    def __init__(self, configs: dict, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.configs = configs
        self.parseConfigs()

        self.session_logger = SessionLogger()
        self.current_video_path: Optional[str] = None
        self.model_ready: bool = False
        self.mode = "file"
        self.is_inference_running = False

        self.video_thread: Optional[VideoThread] = None
        self.camera_thread: Optional[CameraThread] = None
        self.obs_thread: Optional[OBSCameraThread] = None
        self.obs_bytetrack_thread: Optional[CameraByteTrackThread] = None
        self.dual_sync_thread: Optional[DualSourceCameraThread] = None
        self.free_switch_thread: Optional[FreeSwitchCameraThread] = None
        self.video_fps: float = 30.0
        self.tts_mode: str = "none"

        self.font_family = "Sans Serif"
        self.font_size = 14

        self.livecc_model = None
        self.prompt_manager: Optional[PromptManager] = None
        self._bytetrack_wrapper = None          # pre-loaded ByteTrackWrapper (set by background thread)
        self._bytetrack_preload_thread = None   # QThread that loads it

        # Remote inference state — use _socket_runner is not None to check active connection
        self._socket_runner: Optional[SocketClientRunner] = None
        # Periodic RAM/JPEG-queue telemetry during remote obs_track (see _start_obs_track_ram_monitor)
        self._obs_track_ram_timer: Optional[QtCore.QTimer] = None
        # When remote sends PREVIEW (boxes), avoid raw camera overwriting it for ~one frame period
        self._last_track_preview_mono: float = 0.0

        # client_only: only controls whether local LiveCC/ByteTrack are loaded at startup.
        # All GUI behavior is identical once connected; default = True (don't load 7B locally).
        remote_cfg = configs.get("remote", {})
        self._client_only = bool(remote_cfg.get("client_only", True))
        if self._client_only:
            print("[Main] client_only: local VLM not loaded; connect to remote server.")
        else:
            self._load_livecc_model()

        self._init_fonts()
        self._initUI()
        self._initTTSWorker()

        # Pre-fill remote panel host/port from config
        host = str(remote_cfg.get("host", "127.0.0.1"))
        port = int(remote_cfg.get("port", 9000))
        self.control_panel.chk_remote.setChecked(True)
        self.control_panel.edit_remote_host.setText(host)
        self.control_panel.edit_remote_port.setText(str(port))
        if self._client_only:
            self.control_panel.chk_remote.setEnabled(False)

        # Pre-load local ByteTrack only when running locally (non-client_only)
        if not self._client_only:
            QtCore.QTimer.singleShot(500, self._preload_bytetrack_model)

        self._playback_sec: float = 0.0
        self._pending_segments = deque()  # items: (start_t, stop_t, text)

        self._subtitle_timer = QtCore.QTimer(self)
        self._subtitle_timer.setInterval(50)  # 20 FPS 更新足夠
        self._subtitle_timer.timeout.connect(self._tick_subtitle_scheduler)
        self._subtitle_timer.start()

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
            print("[Main] Loading LiveCC model in main thread...")
            self.livecc_model = LiveCCInfer(device_id=0)
            print("[Main] LiveCC model load complete")
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
        exp_file  = bt_cfg.get("exp_file",  "exps/example/mot/yolox_x_mix_det.py")
        ckpt_path = bt_cfg.get("ckpt_path", "pretrained/bytetrack_x_mot17.pth.tar")

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
        self.window_width_min = gui_cfg.get("min_width", 1400)
        self.window_height_min = gui_cfg.get("min_height", 900)
        self.window_title = gui_cfg.get("title", "LiveCC Studio")

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

        # Make preview take more space, but prevent the control panel from being too squashed
        self.top_splitter.setStretchFactor(0, 10)
        self.top_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(self.top_splitter, stretch=4)

        bottom_group = QtWidgets.QGroupBox("Live Commentary Log")
        bottom_layout = QtWidgets.QVBoxLayout(bottom_group)
        bottom_layout.setContentsMargins(14, 18, 14, 12)

        self.text_output = TextOutputWidget()
        bottom_layout.addWidget(self.text_output)

        # Click segment line -> seek to corresponding time in video
        self._install_text_output_click_handler()

        main_layout.addWidget(bottom_group, stretch=2)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Initializing system...")

        # Settings rows use font metrics; refresh after panel is under MainWindow (correct font chain)
        self.control_panel.apply_settings_metrics()

        # Signals
        self.control_panel.requestOpenVideo.connect(self.on_open_video_clicked)
        self.control_panel.requestOpenCamera.connect(self.on_open_camera_clicked)
        self.control_panel.requestOpenCameraTrack.connect(self.on_open_camera_track_clicked)
        self.control_panel.requestOpenOBS.connect(self.on_open_obs_clicked)
        self.control_panel.requestOpenDualSync.connect(self.on_open_dual_sync_clicked)
        self.control_panel.requestOpenFreeSwitch.connect(self.on_open_free_switch_clicked)
        self.control_panel.requestSwitchSource.connect(self.on_switch_source)
        self.control_panel.requestStart.connect(self.on_start_clicked)
        self.control_panel.requestFontScale.connect(self.on_font_scale_request)
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

    def _apply_initial_geometry(self) -> None:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()

        target_w = max(self.minimumWidth(), int(geo.width() * 0.90))
        target_h = max(self.minimumHeight(), int(geo.height() * 0.90))

        if self.width() < self.minimumWidth() or self.height() < self.minimumHeight():
            self.resize(target_w, target_h)

        w = max(1, self.width())
        self.top_splitter.setSizes([int(w * 0.72), int(w * 0.28)])

    # ---------------- Workers ----------------

    def _initLiveCCWorker(self) -> None:
        from .workers.livecc import LiveCCWorker
        self.livecc_thread = QtCore.QThread(self)
        self.livecc_worker = LiveCCWorker()
        self.livecc_worker.livecc = self.livecc_model
        self.livecc_worker.moveToThread(self.livecc_thread)
        self.livecc_worker.signal_model_loaded.connect(self.on_model_loaded)
        self.livecc_worker.signal_segment.connect(self.on_segment)
        self.livecc_worker.signal_finished.connect(self.on_finished)
        self.livecc_worker.signal_error.connect(self.on_error)
        self.signal_start_livecc.connect(self.livecc_worker.runInference)

        self.livecc_thread.start()

    def _initCameraWorker(self) -> None:
        from .workers.livecc import LiveCCCameraWorker
        self.cam_worker_thread = QtCore.QThread(self)
        self.cam_worker = LiveCCCameraWorker(device_id=0)
        self.cam_worker.livecc = self.livecc_model
        self.cam_worker.moveToThread(self.cam_worker_thread)
        self.cam_worker.signal_model_loaded.connect(self.on_model_loaded)
        self.cam_worker.signal_segment.connect(self.on_segment)
        self.cam_worker.signal_error.connect(self.on_error)
        self.signal_start_camera_livecc.connect(self.cam_worker.runCameraInference)
        self.cam_worker_thread.start()

    # ---------------- Remote Socket ----------------

    def _stop_obs_track_ram_monitor(self) -> None:
        if self._obs_track_ram_timer is not None:
            self._obs_track_ram_timer.stop()
            self._obs_track_ram_timer.deleteLater()
            self._obs_track_ram_timer = None

    def _start_obs_track_ram_monitor(self) -> None:
        """Log sender-PC RAM + JPEG queue (optional); server-side RAM prints on inference host."""
        self._stop_obs_track_ram_monitor()
        timer = QtCore.QTimer(self)
        timer.setInterval(2000)
        timer.timeout.connect(self._log_obs_track_ram_tick)
        self._obs_track_ram_timer = timer
        timer.start()
        self._log_obs_track_ram_tick()

    @QtCore.Slot()
    def _log_obs_track_ram_tick(self) -> None:
        if (
            self.mode != "obs_track"
            or not self.is_inference_running
            or self._socket_runner is None
        ):
            self._stop_obs_track_ram_monitor()
            return
        try:
            import psutil

            rss_mb = psutil.Process().memory_info().rss / (1024.0**2)
            q_used, q_max = self._socket_runner.get_frame_send_queue_levels()
            sys_pct = psutil.virtual_memory().percent
            msg = (
                f"sender_PC RSS={rss_mb:.1f} MiB | JPEG send_queue={q_used}/{q_max} | "
                f"sender system_RAM_used={sys_pct:.0f}%"
            )
            # Remote: forward to server so it prints on the same stdout as ByteTrack FPS
            # (see server session _handle_client_diag); keep session file on this machine.
            self._socket_runner.send_obs_track_diagnostic(
                rss_mb, q_used, q_max, sys_pct
            )
            if hasattr(self, "session_logger") and self.session_logger.current_log_file:
                self.session_logger.log_system("Memory", "INFO", msg)
        except Exception as e:
            print(f"[Memory][obs_track] telemetry failed: {e}")

    @QtCore.Slot(str, int)
    def on_remote_connect_clicked(self, host: str, port: int) -> None:
        self._stop_obs_track_ram_monitor()
        if self._socket_runner is not None:
            self._socket_runner.disconnect_and_quit()
            self._socket_runner = None

        self.append_text(f"[Remote] 正在連線至 {host}:{port}…")
        self.control_panel.set_remote_connecting()

        runner = SocketClientRunner(host, port, parent=self)
        runner.signal_connected.connect(self.on_remote_connected)
        runner.signal_disconnected.connect(self.on_remote_disconnected)
        runner.signal_connect_error.connect(self.on_remote_connect_error)
        runner.signal_segment.connect(self.on_segment)
        runner.signal_status.connect(self.on_remote_status)
        runner.signal_error.connect(self.on_remote_server_error)
        runner.signal_preview.connect(
            self.on_remote_track_preview, QtCore.Qt.QueuedConnection
        )
        self._socket_runner = runner
        runner.start()

    @QtCore.Slot()
    def on_remote_disconnect_clicked(self) -> None:
        self._stop_obs_track_ram_monitor()
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

    @QtCore.Slot(object)
    def on_remote_track_preview(self, frame_bgr: object) -> None:
        # Server sends annotated BGR with ByteTrack boxes (throttled) during obs_track
        if not isinstance(frame_bgr, np.ndarray) or self.mode != "obs_track":
            return
        if not self._socket_runner:
            return
        if not self.is_inference_running:
            return
        # First PREVIEW arrival — confirm ByteTrack is active on server
        if self._last_track_preview_mono == 0.0:
            self.statusBar().showMessage("[Remote] ByteTrack tracking active ✓", 3000)
            print("[Remote] First tracking PREVIEW received — ByteTrack is running on server")
        self._last_track_preview_mono = time.monotonic()
        self.video_panel.update_frame(frame_bgr, is_bgr=True)

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
            self.signal_tts_stop.connect(self.tts_worker.stop, QtCore.Qt.QueuedConnection)
            self.signal_tts_speak.connect(self.tts_worker.speak, QtCore.Qt.QueuedConnection)
            self.signal_tts_interrupt.connect(self.tts_worker.interrupt, QtCore.Qt.QueuedConnection)

            self.tts_thread.start()

            # ==========================================
            # 2. Local TTS Worker (本地 Chatterbox) [ChatterBox disabled]
            # ==========================================
            # self.local_tts_thread = QtCore.QThread(self)
            # self.local_tts_worker = ChatterboxTTSWorker()
            # self.local_tts_worker.moveToThread(self.local_tts_thread)
            # self.local_tts_thread.started.connect(self.local_tts_worker.start)
            # self.signal_local_tts_apply_settings.connect(self.local_tts_worker.apply_settings, QtCore.Qt.QueuedConnection)
            # self.signal_local_tts_stop.connect(self.local_tts_worker.stop, QtCore.Qt.QueuedConnection)
            # self.signal_local_tts_speak.connect(self.local_tts_worker.speak, QtCore.Qt.QueuedConnection)
            # self.signal_local_tts_interrupt.connect(self.local_tts_worker.interrupt, QtCore.Qt.QueuedConnection)

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

        # Take all segments that have reached their start time
        ready: list[tuple[float, float, str]] = []
        while self._pending_segments and float(self._pending_segments[0][0]) <= cur:
            st, ed, tx = self._pending_segments.popleft()
            ready.append((float(st), float(ed), str(tx)))

        if not ready:
            return

        for start_t, stop_t, text in ready:
            text = str(text)
            if not text.strip():
                continue

            line = f"[{self._fmt_time(start_t)}-{self._fmt_time(stop_t)}] {text}"
            self.text_output.appendText(line)

            # TTS: Only speak when "displaying" to stay in sync with the video
            if self.tts_mode == "openai":
                self.signal_tts_speak.emit(text)
            # elif self.tts_mode == "local":  # [ChatterBox disabled]
            #     self.signal_local_tts_speak.emit(text)

        # Prevent inference from running too far ahead and blowing up the queue
        MAX_PENDING = 400
        while len(self._pending_segments) > MAX_PENDING:
            self._pending_segments.popleft()

    @QtCore.Slot()
    def _on_tts_mode_changed(self) -> None:
        pass  # [ChatterBox disabled]

    @QtCore.Slot(int)
    def on_font_scale_request(self, size_pt: int) -> None:
        self.font_size = int(size_pt)
        self._apply_styles(self.font_size)
        self.control_panel.apply_settings_metrics()
        self.statusBar().showMessage(f"Font size adjusted to: {self.font_size}pt", 2000)
        QtCore.QTimer.singleShot(0, self._apply_initial_geometry)

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

        self._stop_all_source_threads()

        self.current_video_path = path
        self.control_panel.set_status(f"File: {os.path.basename(path)}")
        self.append_text(f"已載入影片：{os.path.basename(path)}")

        self._load_video_preview(path)
        self.video_panel.slider.setEnabled(True)
        self._update_start_button_state()

    def _stop_all_source_threads(self) -> None:
        """Stop and clean up all live-source threads before switching sources.
        Explicitly disconnect signals to prevent residual frame emissions after stop."""
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
            if not self.obs_thread.wait(3000):
                self.obs_thread.terminate()
                self.obs_thread.wait(1000)
            self.obs_thread = None
        if self.obs_bytetrack_thread:
            self.obs_bytetrack_thread.requestStop()  # also releases DirectShow capture
            try:
                self.obs_bytetrack_thread.signal_frame.disconnect()
                self.obs_bytetrack_thread.signal_subject_frame.disconnect()
            except RuntimeError:
                pass
            # Use a timeout so the GUI main thread never freezes if the worker
            # thread is still blocked (e.g. DirectShow did not release in time).
            if not self.obs_bytetrack_thread.wait(3000):
                self.obs_bytetrack_thread.terminate()
                self.obs_bytetrack_thread.wait(1000)
            self.obs_bytetrack_thread = None
        if self.dual_sync_thread:
            self.dual_sync_thread.requestStop()
            try:
                self.dual_sync_thread.signal_frame.disconnect()
            except RuntimeError:
                pass
            if not self.dual_sync_thread.wait(3000):
                self.dual_sync_thread.terminate()
                self.dual_sync_thread.wait(1000)
            self.dual_sync_thread = None
        if self.free_switch_thread:
            self.free_switch_thread.requestStop()
            try:
                self.free_switch_thread.signal_frame.disconnect()
                self.free_switch_thread.signal_source_changed.disconnect()
            except RuntimeError:
                pass
            if not self.free_switch_thread.wait(3000):
                self.free_switch_thread.terminate()
                self.free_switch_thread.wait(1000)
            self.free_switch_thread = None
        self.control_panel.set_free_switch_bar_visible(False)

    @QtCore.Slot()
    def on_open_camera_clicked(self) -> None:
        self.stop_inference()
        self.mode = "camera"
        self.current_video_path = "LiveMode: Live Camera"
        self.append_text("Switched to camera mode")
        self._stop_all_source_threads()
        self.camera_start_time = time.time()
        from .core.io.obs_input import find_physical_camera_index
        cam_idx = find_physical_camera_index()
        print(f"[Camera] Using physical camera index {cam_idx}")
        self.camera_thread = CameraThread(camera_index=cam_idx)
        self.camera_thread.signal_frame.connect(self.on_camera_frame)
        self.camera_thread.signal_error.connect(self.on_error)
        self.camera_thread.start()

        self.video_panel.slider.setEnabled(False)
        self._update_start_button_state()

    @QtCore.Slot()
    def on_open_obs_clicked(self) -> None:
        """Switch to OBS Virtual Camera mode."""
        self.stop_inference()
        self.mode = "obs"
        self.current_video_path = "OBS Virtual Camera"
        self.control_panel.set_status("Mode: OBS Virtual Camera")
        self.append_text("Switched to OBS stream mode - ensure OBS Virtual Camera is active")
        self._stop_all_source_threads()
        self.camera_start_time = time.time()
        self.obs_thread = OBSCameraThread()
        self.obs_thread.signal_frame.connect(self.on_camera_frame)
        self.obs_thread.signal_error.connect(self.on_error)
        self.obs_thread.start()

        self.video_panel.slider.setEnabled(False)
        self._update_start_button_state()

    @QtCore.Slot()
    def on_open_camera_track_clicked(self) -> None:
        """Webcam + tracking: ByteTrack runs on remote server; local may show full-frame preview only (thin client)."""
        self.stop_inference()
        self.mode = "obs_track"
        self.current_video_path = "Camera + ByteTrack (remote)"
        self.control_panel.set_status("Mode: Webcam + Tracking (remote server)")
        self.append_text("Switched to Webcam + Tracking (ByteTrack on remote; sending frames)")
        self.append_text("已切換至「鏡頭 + 追蹤」；追蹤在遠端執行，本機只送畫面")
        self._stop_all_source_threads()

        use_remote_track = self._socket_runner is not None or self._client_only
        if use_remote_track:
            # Remote path: plain camera; server receives raw frames and runs ByteTrack there.
            from .core.io.obs_input import find_physical_camera_index
            self.camera_start_time = time.time()
            cam_idx = find_physical_camera_index()
            self.camera_thread = CameraThread(camera_index=cam_idx)
            self.camera_thread.signal_frame.connect(self.on_camera_frame)
            self.camera_thread.signal_error.connect(self.on_error)
            self.camera_thread.start()
            self.video_panel.slider.setEnabled(False)
            self._update_start_button_state()
            return

        bt_cfg = self.configs.get("bytetrack", {})
        repo_path = bt_cfg.get("bytetrack_repo") or None
        exp_file  = bt_cfg.get("exp_file",  "exps/example/mot/yolox_x_mix_det.py")
        ckpt_path = bt_cfg.get("ckpt_path", "pretrained/bytetrack_x_mot17.pth.tar")

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

        self.camera_start_time = time.time()
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

        self.camera_start_time = time.time()
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

        self.camera_start_time = time.time()
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

    @QtCore.Slot(str)
    def on_switch_source(self, source: str) -> None:
        """Instantly switch the active camera source inside FreeSwitchCameraThread."""
        if self.free_switch_thread is not None:
            self.free_switch_thread.set_active_source(source)
        # Highlight button immediately (don't wait for signal_source_changed round-trip)
        self.control_panel.highlight_switch_source(source)

    @QtCore.Slot(str)
    def _on_free_switch_source_changed(self, source: str) -> None:
        """Called when FreeSwitchCameraThread confirms the new source."""
        self.control_panel.highlight_switch_source(source)
        self.append_text(f"[FreeSwitch] 已切換至：{source}")

    def _apply_tts_settings_before_start(self) -> None:
        """Apply TTS settings according to current mode"""
        self.tts_mode = self.control_panel.get_tts_mode()

        if self.tts_mode == "openai":
            voice = self.control_panel.get_openai_voice()
            speed = self.control_panel.get_openai_speed()
            # Ensure reset of Local (optionaltrol_panel.get_openai_speed()
            # 確保重置 Local (可選)
            self.signal_tts_apply_settings.emit(voice, float(speed))

        # elif self.tts_mode == "local":  # [ChatterBox disabled]
        #     exag = self.control_panel.get_local_exaggeration()
        #     cfg = self.control_panel.get_local_cfg()
        #     self.signal_local_tts_apply_settings.emit(float(exag), float(cfg))

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

        style_key = self.control_panel.get_selected_style_key()
        style_label = self.control_panel.get_selected_style_label()

        if self.prompt_manager is not None:
            prompt = self.prompt_manager.build_query(style_key)
        else:
            prompt = "Please broadcast the screen in Traditional Chinese in real-time. Do not use emojis, and do not speculate."

        # Apply TTS settings before start
        self._apply_tts_settings_before_start()

        self.is_inference_running = True
        self.control_panel.set_start_button_state(True)
        self.control_panel.set_tts_controls_enabled(False)  # Lock during inference
        self.text_output.setText("")
        self._obs_drop_logged = False

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
            elif self.livecc_model is not None:
                # Local: pass file path directly to local LiveCC worker
                self.signal_start_livecc.emit(self.current_video_path, prompt)
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
                self._socket_runner.start_inference(self.mode, prompt)
                if self.mode == "obs_track":
                    self._start_obs_track_ram_monitor()
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
        self._stop_obs_track_ram_monitor()
        self.append_text("Stopping inference")
        if hasattr(self, "_pending_segments"):
            self._pending_segments.clear()
        if self.tts_mode == "openai":
            try: self.signal_tts_interrupt.emit()
            except: pass

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
            if not self.obs_thread.wait(3000):
                self.obs_thread.terminate()
                self.obs_thread.wait(1000)
            self.obs_thread = None

        if self.mode == "obs_track" and self.obs_bytetrack_thread is not None:
            self.obs_bytetrack_thread.requestStop()
            if not self.obs_bytetrack_thread.wait(3000):
                self.obs_bytetrack_thread.terminate()
                self.obs_bytetrack_thread.wait(1000)
            self.obs_bytetrack_thread = None

        if self.mode == "dual_sync" and self.dual_sync_thread is not None:
            self.dual_sync_thread.requestStop()
            if not self.dual_sync_thread.wait(3000):
                self.dual_sync_thread.terminate()
                self.dual_sync_thread.wait(1000)
            self.dual_sync_thread = None

        self.is_inference_running = False
        self._last_track_preview_mono = 0.0
        self.control_panel.set_start_button_state(False)
        self.control_panel.set_tts_controls_enabled(True)  # Unlock after stop

    # ---------------- Frame handlers ----------------

    @QtCore.Slot(np.ndarray, int, float)
    def on_video_frame(self, frame_rgb: np.ndarray, frame_idx: int, fps: float) -> None:
        self.video_panel.update_frame(frame_rgb)
        self.video_panel.set_position(frame_idx, fps)

        if self.mode == "file" and fps and fps > 0:
            self._playback_sec = float(frame_idx) / float(fps)

        # Stream video frames to remote server for file-mode inference
        if self.mode == "file" and self.is_inference_running and self._socket_runner is not None:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self._socket_runner.send_frame(frame_bgr, self._playback_sec)

    @QtCore.Slot(np.ndarray)
    def on_camera_frame(self, frame_rgb: np.ndarray) -> None:
        # obs_track + remote + infer: raw 30fps camera must not overwrite server PREVIEW (boxes).
        # Hold window (_OBS_TRACK_PREVIEW_HOLD_SEC): if too tight, annotated/raw alternate visibly;
        # if too wide, the panel keeps stale PREVIEW longer than desired when PREVIEW stalls.
        if (
            self.mode == "obs_track"
            and self._socket_runner is not None
            and self.is_inference_running
        ):
            if (
                time.monotonic() - self._last_track_preview_mono
                < self._OBS_TRACK_PREVIEW_HOLD_SEC
            ):
                pass  # keep last PREVIEW visible
            else:
                self.video_panel.update_frame(frame_rgb)
        else:
            self.video_panel.update_frame(frame_rgb)
        if not self.is_inference_running:
            return
        # obs_track on thin client: frames come from plain CameraThread and
        # must be forwarded to the remote server so it can run ByteTrack there.
        # Include obs_track here for remote-only (client_only) path.
        if self.mode not in ("camera", "obs", "dual_sync", "obs_track", "free_switch"):
            return

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        t_relative = time.time() - self.camera_start_time

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
        if not self.is_inference_running or self.mode != "obs_track":
            if not hasattr(self, '_obs_drop_logged'):
                self._obs_drop_logged = True
                print(f"[GUI] ⚠️  on_obs_track_subject_frame dropped: "
                      f"is_inference_running={self.is_inference_running}, mode='{self.mode}'")
            return

        fixed = cv2.resize(subject_crop_rgb, (640, 480))
        subject_bgr = cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR)
        t_relative = time.time() - self.camera_start_time

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

    @QtCore.Slot()
    def on_model_loaded(self) -> None:
        self.model_ready = True
        self.statusBar().showMessage("Model Ready")
        self.control_panel.set_status("Model ready, please select source")
        self._update_start_button_state()

    @QtCore.Slot(float, float, str)
    def on_segment(self, start_t: float, stop_t: float, text: str) -> None:
        # Log to session file
        if hasattr(self, 'session_logger'):
            self.session_logger.log_commentary(text)

        # Camera mode: No player timeline to schedule, so display and speak immediately
        if self.mode != "file":
            line = f"[{self._fmt_time(start_t)}] {text}"
            self.text_output.appendText(line)

            if not str(text).strip():
                return
            if self.tts_mode == "openai":
                self.signal_tts_speak.emit(text)
            # elif self.tts_mode == "local":  # [ChatterBox disabled]
            #     self.signal_local_tts_speak.emit(text)
            return

        # File mode: Enter queue, wait until video playback reaches corresponding time (stay in sync)
        if not hasattr(self, "_pending_segments"):
            self._pending_segments = deque()

        self._pending_segments.append((float(start_t), float(stop_t), str(text)))

        if not text.strip():
            return

        # TTS: only when _tick_subtitle_scheduler displays the line (stays in sync with video time).
        # Do not speak here, or OpenAI TTS will fire early / replace queue and desync from on-screen text.

    @QtCore.Slot()
    def on_finished(self) -> None:
        self.append_text("Playback/Inference finished")
        self.stop_inference()

    @QtCore.Slot(str)
    def on_error(self, msg: str) -> None:
        self.append_text(f"Error: {msg}")
        self.append_text(f"錯誤：{msg}")
        self.stop_inference()

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

    def append_text(self, msg: str) -> None:
        self.text_output.appendText(msg)
        if hasattr(self, 'session_logger'):
            self.session_logger.log_system("GUI", "INFO", msg)


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
        try:
            self.signal_local_tts_stop.emit()
        except: pass
        
        if hasattr(self, "local_tts_thread") and self.local_tts_thread:
            self.local_tts_thread.quit()
            self.local_tts_thread.wait(2000)

        super().closeEvent(event)