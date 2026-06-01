# src/miis_broadcast/gui.py

from __future__ import annotations

import json
import logging
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
from .workers.livecc import LiveCCWorker, LiveCCCameraWorker
from .workers.gemini import GeminiWorker
from .workers.openai_tts import OpenAITTSWorker
from .workers.obs_input import OBSCameraThread
from .workers.obs_bytetrack import OBSByteTrackThread
from .core.prompt.prompt_manager import PromptManager
from .core.match_tracker import match_tracker
from .core.utils.session_logger import SessionLogger
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
    requestOpenVideo = QtCore.Signal()
    requestOpenCamera = QtCore.Signal()
    requestLoadContext = QtCore.Signal()
    requestStart = QtCore.Signal()
    requestFontScale = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(600)
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
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

        # ── 電腦 button ──────────────────────────────────────────────────
        self.btn_computer = QtWidgets.QPushButton("💻  電腦  ▾")
        self.btn_computer.setStyleSheet(btn_style)

        menu_computer = QtWidgets.QMenu(self.btn_computer)
        menu_computer.setStyleSheet(menu_style)
        menu_computer.addAction("🎬  影片上傳",    lambda: self.requestOpenVideo.emit())

        submenu_webcam = menu_computer.addMenu("📷  Webcam")
        submenu_webcam.setStyleSheet(menu_style)
        submenu_webcam.addAction("⬜  Plain Stream",       lambda: self.requestOpenCamera.emit())
        submenu_webcam.addAction("🎯  Stream + Track",  lambda: self.requestOpenCameraTrack.emit())

        self.btn_computer.setMenu(menu_computer)

        # ── OBS button ───────────────────────────────────────────────────
        self.btn_obs_main = QtWidgets.QPushButton("🎙  OBS  ▾")
        self.btn_obs_main.setStyleSheet(btn_style)

        menu_obs = QtWidgets.QMenu(self.btn_obs_main)
        menu_obs.setStyleSheet(menu_style)

        # OBS camera stream (direct physical camera connection, bypasses OBS Virtual Camera)
        submenu_camstream = menu_obs.addMenu("📡  Direct Camera")
        submenu_camstream.setStyleSheet(menu_style)
        submenu_camstream.addAction("⬜  Plain Stream",       lambda: self.requestOpenCamera.emit())
        submenu_camstream.addAction("🎯  Stream + Track",  lambda: self.requestOpenCameraTrack.emit())

        menu_obs.addSeparator()

        # OBS Virtual Camera
        submenu_virtual = menu_obs.addMenu("🖥  Virtual Camera")
        submenu_virtual.setStyleSheet(menu_style)
        submenu_virtual.addAction("⬜  Plain OBS",       lambda: self.requestOpenOBS.emit())
        submenu_virtual.addAction("🎯  OBS + Track",   lambda: self.requestOpenOBSTrack.emit())

        menu_obs.addSeparator()

        # VR (Meta Quest via OBS Virtual Camera)
        submenu_vr = menu_obs.addMenu("🥽  VR")
        submenu_vr.setStyleSheet(menu_style)
        submenu_vr.addAction("⬜  Plain VR",       lambda: self.requestOpenOBS.emit())
        submenu_vr.addAction("🎯  VR + Track",     lambda: self.requestOpenOBSTrack.emit())

        self.btn_obs_main.setMenu(menu_obs)

        btn_row.addWidget(self.btn_computer)
        btn_row.addWidget(self.btn_obs_main)

        self.lbl_status = QtWidgets.QLabel("Status: Not Loaded")
        self.lbl_status.setStyleSheet("color: #b5b5b5;")
        self.lbl_status.setWordWrap(True)

        v_src.addLayout(btn_row)
        v_src.addWidget(self.lbl_status)

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

        # Settings
        grp_settings = QtWidgets.QGroupBox("推論設定 (Settings)")
        form = QtWidgets.QFormLayout(grp_settings)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setFormAlignment(QtCore.Qt.AlignTop)
        form.setSpacing(12)
        form.setContentsMargins(14, 18, 14, 12)

        # [CSS]
        combo_style = """
            QComboBox {
                padding: 6px 10px;
                border-radius: 8px;
                background-color: #333;
                min-height: 30px;
            }
            QComboBox::drop-down { border: 0px; }
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
        self.cmb_tts = QtWidgets.QComboBox()
        _fix_combo_behavior(self.cmb_tts) # [Apply Fix]
        
        self.cmb_tts.addItem("不啟用 (Mute)", userData="none")
        self.cmb_tts.addItem("OpenAI TTS", userData="openai")
        self.cmb_tts.addItem("Gemini TTS", userData="gemini")
        self.cmb_tts.addItem("Local TTS", userData="local")
        self.cmb_tts.setCurrentIndex(1)
        self.cmb_tts.setStyleSheet(combo_style)

        # --- LiveCC style ---
        self.cmb_style = QtWidgets.QComboBox()
        _fix_combo_behavior(self.cmb_style) # [Apply Fix]
        self.cmb_style.setStyleSheet(combo_style)

        # --- OpenAI: Voice ---
        self.cmb_voice = QtWidgets.QComboBox()
        _fix_combo_behavior(self.cmb_voice) # [Apply Fix]
        self.cmb_voice.setStyleSheet(combo_style)
        for v in ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]:
            self.cmb_voice.addItem(v, userData=v)
        self.cmb_voice.setCurrentText("coral")

        # --- OpenAI: Speed slider ---
        self.slider_speed = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_speed.setRange(25, 150)  # 0.25x ~ 1.5x
        self.slider_speed.setValue(100)      # 預設 1.0x

        self.lbl_speed_val = QtWidgets.QLabel("1.0x")
        self.lbl_speed_val.setMinimumWidth(55)
        self.lbl_speed_val.setAlignment(QtCore.Qt.AlignCenter)

        speed_row = QtWidgets.QHBoxLayout()
        speed_row.setSpacing(10)
        speed_row.addWidget(self.slider_speed, stretch=1)
        speed_row.addWidget(self.lbl_speed_val, stretch=0)

        self._speed_row_widget = QtWidgets.QWidget()
        self._speed_row_widget.setLayout(speed_row)

        # --- Local: Exaggeration slider (0.2~1.2) ---
        self.slider_exag = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_exag.setRange(20, 120)
        self.slider_exag.setValue(80)

        self.lbl_exag_val = QtWidgets.QLabel("0.8")
        self.lbl_exag_val.setMinimumWidth(55)
        self.lbl_exag_val.setAlignment(QtCore.Qt.AlignCenter)

        exag_row = QtWidgets.QHBoxLayout()
        exag_row.setSpacing(10)
        exag_row.addWidget(self.slider_exag, stretch=1)
        exag_row.addWidget(self.lbl_exag_val, stretch=0)

        self._exag_row_widget = QtWidgets.QWidget()
        self._exag_row_widget.setLayout(exag_row)

        # --- Local: CFG slider (0.2~1.2) ---
        self.slider_cfg = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_cfg.setRange(20, 120)
        self.slider_cfg.setValue(70)

        self.lbl_cfg_val = QtWidgets.QLabel("0.7")
        self.lbl_cfg_val.setMinimumWidth(55)
        self.lbl_cfg_val.setAlignment(QtCore.Qt.AlignCenter)

        cfg_row = QtWidgets.QHBoxLayout()
        cfg_row.setSpacing(10)
        cfg_row.addWidget(self.slider_cfg, stretch=1)
        cfg_row.addWidget(self.lbl_cfg_val, stretch=0)

        self._cfg_row_widget = QtWidgets.QWidget()
        self._cfg_row_widget.setLayout(cfg_row)

        # --- UI scale slider ---
        self.slider_ui_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_ui_scale.setRange(10, 26)
        self.slider_ui_scale.setValue(14)

        self.lbl_ui_scale_val = QtWidgets.QLabel("14pt")
        self.lbl_ui_scale_val.setMinimumWidth(55)
        self.lbl_ui_scale_val.setAlignment(QtCore.Qt.AlignCenter)

        font_row = QtWidgets.QHBoxLayout()
        font_row.setSpacing(10)
        font_row.addWidget(self.slider_ui_scale, stretch=1)
        font_row.addWidget(self.lbl_ui_scale_val, stretch=0)

        self._font_row_widget = QtWidgets.QWidget()
        self._font_row_widget.setLayout(font_row)

        lbl_style = "QLabel { color: #dedede; }"
        self.l_tts = QtWidgets.QLabel("TTS Mode:")
        self.l_style = QtWidgets.QLabel("Style:")
        self.l_voice = QtWidgets.QLabel("Voice:")
        self.l_speed = QtWidgets.QLabel("Speed:")
        self.l_exag = QtWidgets.QLabel("Exaggeration:")
        self.l_cfg = QtWidgets.QLabel("CFG:")
        self.l_ui = QtWidgets.QLabel("UI Scaling:")

        for x in (self.l_tts, self.l_style, self.l_voice, self.l_speed, self.l_exag, self.l_cfg, self.l_ui):
            x.setStyleSheet(lbl_style)

        form.addRow(self.l_tts, self.cmb_tts)
        form.addRow(self.l_style, self.cmb_style)
        form.addRow(self.l_voice, self.cmb_voice)
        form.addRow(self.l_speed, self._speed_row_widget)
        form.addRow(self.l_exag, self._exag_row_widget)
        form.addRow(self.l_cfg, self._cfg_row_widget)
        form.addRow(self.l_ui, self._font_row_widget)

        layout.addWidget(grp_settings)

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

        layout.addStretch(1)

        # Signals
        self.btn_context.clicked.connect(self.requestLoadContext.emit)
        self.btn_start.clicked.connect(self.requestStart.emit)

        self.slider_speed.valueChanged.connect(lambda v: self.lbl_speed_val.setText(f"{v/100:.1f}x"))
        self.slider_exag.valueChanged.connect(lambda v: self.lbl_exag_val.setText(f"{v/100:.1f}"))
        self.slider_cfg.valueChanged.connect(lambda v: self.lbl_cfg_val.setText(f"{v/100:.1f}"))
        self.slider_ui_scale.valueChanged.connect(self.on_font_scale_changed)

        # 模式切換顯示/隱藏
        self.cmb_tts.currentIndexChanged.connect(self._refresh_tts_controls_visibility)
        self._refresh_tts_controls_visibility()

    # ---------------- ControlPanel Helpers ----------------

    def _refresh_tts_controls_visibility(self) -> None:
        mode = self.get_tts_mode()

        show_openai = (mode == "openai")
        self.l_voice.setVisible(show_openai)
        self.cmb_voice.setVisible(show_openai)
        self.l_speed.setVisible(show_openai)
        self._speed_row_widget.setVisible(show_openai)

        show_local = (mode == "local")
        self.l_exag.setVisible(show_local)
        self._exag_row_widget.setVisible(show_local)
        self.l_cfg.setVisible(show_local)
        self._cfg_row_widget.setVisible(show_local)
        # Gemini TTS: no extra UI controls needed (voice set in app.yml)

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
        self.lbl_status.setText(f"Status: {text}")

    def set_start_button_state(self, running: bool) -> None:
        if running:
            self.btn_start.setText("Stop Broadcasting")
            self.btn_start.setProperty("active", True)
        else:
            self.btn_start.setText("Start Broadcasting")
            self.btn_start.setProperty("active", False)
        self.btn_start.style().unpolish(self.btn_start)
        self.btn_start.style().polish(self.btn_start)
        # Disable source buttons during inference to prevent switching mid-session
        self.btn_computer.setEnabled(not running)
        self.btn_obs_main.setEnabled(not running)


# ============================================================
# Main Window
# ============================================================

class MainWindow(QtWidgets.QMainWindow):
    signal_start_livecc = QtCore.Signal(str, str, int)
    signal_start_camera_livecc = QtCore.Signal(str)

    # ✅ 用 signal 把設定丟到 tts thread，避免你直接 call slot 其實跑在主執行緒
    signal_tts_apply_settings = QtCore.Signal(str, float)
    signal_tts_stop = QtCore.Signal()

    signal_tts_speak = QtCore.Signal(str, int, float, float)  # (text, priority, ref_ts, start_t)
    signal_tts_interrupt = QtCore.Signal()
    signal_local_tts_apply_settings = QtCore.Signal(float, float) # exag, cfg
    signal_local_tts_speak = QtCore.Signal(str)
    signal_local_tts_interrupt = QtCore.Signal()
    signal_local_tts_stop = QtCore.Signal()
    signal_gemini_tts_speak = QtCore.Signal(str, int, float, float)
    signal_gemini_tts_interrupt = QtCore.Signal()
    signal_gemini_tts_stop = QtCore.Signal()
    _signal_to_gemini = QtCore.Signal(float, float, object)
    # Fires when Gemini confirms a P1 event — used to reset LiveCC KV cache
    signal_p1_confirmed = QtCore.Signal()
    # P3 LiveCC description → GeminiBackgroundWorker.update_context (QueuedConnection)
    signal_livecc_context = QtCore.Signal(str)

    # Fast-path keyword sets
    _P1_KEYWORDS = frozenset({
        "score", "goal", "foul", "shot", "basket",
        "進球", "得分", "犯規", "投籃", "dunk", "slam",
    })
    _P2_KEYWORDS = frozenset({
        "pass", "timeout", "intercept", "block",
        "傳球", "暫停", "抄截", "封蓋",
    })

    def __init__(self, configs: dict, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.configs = configs
        self.parseConfigs()

        self.session_logger = SessionLogger()
        self.current_video_path: Optional[str] = None
        self.model_ready: bool = False
        self.mode = "file"
        self.is_inference_running = False
        self._livecc_run_id: int = 0

        self.video_thread: Optional[VideoThread] = None
        self.camera_thread: Optional[CameraThread] = None
        self.obs_thread: Optional[OBSCameraThread] = None
        self.obs_bytetrack_thread: Optional[OBSByteTrackThread] = None
        self.video_fps: float = 30.0
        self.tts_mode: str = "none"
        self._use_gemini: bool = False
        self._tts_last_priority: int = 5       # priority of last content sent to TTS
        self._pending_tts_transition: str = "" # transition phrase to prepend on next speak
        self._last_tts_raw_text: str = ""      # raw text of last TTS emit (dedup)
        self._last_tts_emit_ts: float = 0.0    # wall-clock time of last TTS emit (dedup)
        self._tts_protect_until: float = 0.0   # wall-clock deadline: block lower-priority below this time
        self._post_p1_pending: bool = False     # True while waiting for 1.0s post-P1 silence

        self.font_family = "Sans Serif"
        self.font_size = 14

        self.livecc_model = None
        self.prompt_manager: Optional[PromptManager] = None
        self._bytetrack_wrapper = None          # pre-loaded ByteTrackWrapper (set by background thread)
        self._bytetrack_preload_thread = None   # QThread that loads it


        self._load_livecc_model()

        self._init_fonts()
        self._initUI()
        self._initTTSWorker()
        self._initGeminiWorker()

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

        # Signals
        self.control_panel.requestOpenVideo.connect(self.on_open_video_clicked)
        self.control_panel.requestOpenCamera.connect(self.on_open_camera_clicked)
        self.control_panel.requestLoadContext.connect(self.on_load_context_clicked)
        self.control_panel.requestStart.connect(self.on_start_clicked)
        self.control_panel.requestFontScale.connect(self.on_font_scale_request)
        self.video_panel.seekRequested.connect(self.on_seek_requested)

        # 載入 prompts.yml 並填入下拉式選單
        self._init_prompt_manager_and_fill_styles()

        self._initLiveCCWorker()
        self._initCameraWorker()

        if self.livecc_model is not None:
            self.livecc_worker.signal_model_loaded.emit()

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
        self.gemini_bg_worker = GeminiBackgroundWorker(tts_worker_ref=self.tts_worker)
        self.gemini_bg_worker.moveToThread(self.gemini_bg_thread)
        self.gemini_bg_worker.signal_broadcast.connect(self.on_segment)
        self.gemini_bg_worker.signal_error.connect(self.on_gemini_error)

        # P3 context pool: signal_livecc_context emitted in GUI thread,
        # update_context() slot runs in gemini_bg_thread via QueuedConnection.
        self.signal_livecc_context.connect(
            self.gemini_bg_worker.update_context,
            QtCore.Qt.QueuedConnection,
        )

        self.gemini_bg_thread.started.connect(self.gemini_bg_worker.initialize)
        # run_background_loop is NOT started here — it starts when inference
        # begins (see on_start_clicked), preventing HTTP calls at startup that
        # race with PyTorch CUDA background threads and cause heap corruption.
        self.gemini_bg_thread.start()

        # signal_tts_done → P1 post-interrupt silence handler (all TTS workers)
        self.tts_worker.signal_tts_done.connect(self._on_tts_done)
        self.gemini_tts_worker.signal_tts_done.connect(self._on_tts_done)

    # Interrupt threshold: new_priority must be strictly better than current by this margin
    _INTERRUPT_MATRIX = {
        1: 2,   # P1 interrupts anything currently >= P2 (i.e. anything but another P1 that's just starting)
        2: 3,   # P2 interrupts if current >= P3
        3: 4,   # P3 interrupts if current >= P4
    }
    _TRANSITION_PHRASES = {
        1: "Oh!—",
        2: "And—",
        3: "",   # P3: seamless, no explicit word
    }

    @QtCore.Slot(float, float, int, bool)
    def _on_gemini_priority(self, start_t: float, stop_t: float, priority: int, should_speak: bool) -> None:
        """
        Fires as soon as Gemini returns the PRIORITY line (before SPEAK text arrives).
        Decides whether to interrupt current TTS and what transition phrase to use.
        Also triggers KV cache reset for P1 events.
        """
        if priority == 1:
            # Gemini confirmed P1 — reset LiveCC KV cache via QueuedConnection
            self.signal_p1_confirmed.emit()

        if not should_speak:
            return

        interrupt_threshold = self._INTERRUPT_MATRIX.get(priority)
        if interrupt_threshold is None:
            return  # P4/P5 never interrupt

        if self._tts_last_priority >= interrupt_threshold:
            phrase = self._TRANSITION_PHRASES.get(priority, "")
            self._pending_tts_transition = phrase
            # Clear audio immediately; new text will arrive shortly via signal_broadcast
            if self.tts_mode == "openai":
                self.signal_tts_interrupt.emit()
            elif self.tts_mode == "gemini":
                self.signal_gemini_tts_interrupt.emit()
            elif self.tts_mode == "local":
                self.signal_local_tts_interrupt.emit()
            logging.info(
                "[Priority] P%d interrupting P%d — transition: %r → signal_tts_interrupt emitted",
                priority, self._tts_last_priority, phrase,
            )

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

    @QtCore.Slot(float, float, object)
    def _route_segment(self, start_t: float, stop_t: float, data: object) -> None:
        """Fast-slow blade routing: P1/P2 direct-to-TTS, P3 → context pool."""
        self._ensure_log_dir()
        display, _ = self._extract_segment_texts(data)
        self._write_log(self.livecc_log_file, f"[LiveCC] [{self._fmt_time(start_t)}-{self._fmt_time(stop_t)}] {display}")

        if not self._use_gemini:
            self.on_segment(start_t, stop_t, data)
            return

        raw = ""
        if isinstance(data, dict):
            raw = data.get("metadata", {}).get("raw", "") or data.get("event", "")
        elif isinstance(data, str):
            raw = data

        fast_priority = self._scan_priority(raw)

        if fast_priority == 1:
            logging.info("[FastBlade] P1 hit: %r", raw[:80])
            if hasattr(self, "gemini_bg_worker"):
                self.gemini_bg_worker.pause()
            if self.tts_mode == "openai":
                self.signal_tts_interrupt.emit()
            elif self.tts_mode == "gemini":
                self.signal_gemini_tts_interrupt.emit()
            elif self.tts_mode == "local":
                self.signal_local_tts_interrupt.emit()
            tts_text = raw.strip()
            if tts_text:
                self._post_p1_pending = True
                if self.tts_mode == "gemini":
                    self.signal_gemini_tts_speak.emit(tts_text, 1, time.time(), start_t)
                else:
                    self.signal_tts_speak.emit(tts_text, 1, time.time(), start_t)
            self.signal_p1_confirmed.emit()

        elif fast_priority == 2:
            logging.info("[FastBlade] P2 hit: %r", raw[:80])
            tts_text = raw.strip()
            if tts_text:
                if self.tts_mode == "gemini":
                    self.signal_gemini_tts_speak.emit(tts_text, 2, time.time(), start_t)
                else:
                    self.signal_tts_speak.emit(tts_text, 2, time.time(), start_t)

        else:
            # P3: feed description into context pool via typed Signal (QueuedConnection).
            # GeminiBackgroundWorker picks it up and uses it as context for slow-blade commentary.
            description = raw.strip()
            if description:
                self.signal_livecc_context.emit(description)

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

            # 🔥 [修改點 2] 監聽下拉選單變化
            self.control_panel.cmb_tts.currentIndexChanged.connect(self._on_tts_mode_changed)

            # 如果預設選項剛好就是 Local (雖然通常預設是 OpenAI)，初始化時檢查一次
            self._on_tts_mode_changed()

            # ==========================================
            # 3. Gemini TTS Worker  [lazy-started like Chatterbox]
            # ==========================================
            # Thread is set up here but NOT started at launch.
            # It starts the first time the user selects "gemini" TTS mode
            # (see _on_tts_mode_changed), avoiding a startup race between
            # httpx's DNS/SSL threads and PyTorch's CUDA runtime threads.
            from .workers.gemini_tts import GeminiTTSWorker
            self.gemini_tts_thread = QtCore.QThread(self)
            self.gemini_tts_worker = GeminiTTSWorker()
            self.gemini_tts_worker.moveToThread(self.gemini_tts_thread)
            self.gemini_tts_thread.started.connect(self.gemini_tts_worker.start)
            self.signal_gemini_tts_speak.connect(self.gemini_tts_worker.speak, QtCore.Qt.QueuedConnection)
            self.signal_gemini_tts_interrupt.connect(self.gemini_tts_worker.interrupt, QtCore.Qt.QueuedConnection)
            self.signal_gemini_tts_stop.connect(self.gemini_tts_worker.stop, QtCore.Qt.QueuedConnection)
            # gemini_tts_thread.start() is called lazily in _on_tts_mode_changed()

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

            line = f"[{self._fmt_time(start_t)}-{self._fmt_time(stop_t)}] {display_text}"
            self._append_ui(line)

        while len(self._pending_segments) > self._MAX_PENDING:
            self._pending_segments.popleft()

    @QtCore.Slot()
    def _on_tts_mode_changed(self) -> None:
        mode = self.control_panel.get_tts_mode() if hasattr(self, "control_panel") else None
        if mode == "gemini" and not self.gemini_tts_thread.isRunning():
            self.gemini_tts_thread.start()

    @QtCore.Slot(int)
    def on_font_scale_request(self, size_pt: int) -> None:
        self.font_size = int(size_pt)
        self._apply_styles(self.font_size)
        self.statusBar().showMessage(f"Font size adjusted to: {self.font_size}pt", 2000)
        QtCore.QTimer.singleShot(0, self._apply_initial_geometry)

    def _stop_all_source_threads(self) -> None:
        """Stop every input-source thread unconditionally (used when switching modes)."""
        if self.obs_thread is not None:
            self.obs_thread.requestStop()
            if not self.obs_thread.wait(3000):
                self.obs_thread.terminate()
                self.obs_thread.wait(1000)
            self.obs_thread = None

        if self.obs_bytetrack_thread is not None:
            self.obs_bytetrack_thread.requestStop()
            if not self.obs_bytetrack_thread.wait(3000):
                self.obs_bytetrack_thread.terminate()
                self.obs_bytetrack_thread.wait(1000)
            self.obs_bytetrack_thread = None

        if self.camera_thread is not None:
            self.camera_thread.requestStop()
            self.camera_thread.wait(1000)
            self.camera_thread = None

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

    @QtCore.Slot()
    def on_open_camera_clicked(self) -> None:
        self.stop_inference()
        self.mode = "camera"
        self.current_video_path = "Live Camera"
        self.control_panel.set_status("模式: 即時鏡頭")
        self.append_text("已切換至鏡頭模式")

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
        self.camera_start_time = time.time()
        self.obs_thread = OBSCameraThread()
        self.obs_thread.signal_frame.connect(self.on_camera_frame)
        self.obs_thread.signal_error.connect(self.on_error)
        self.obs_thread.start()

        self.video_panel.slider.setEnabled(False)
        self._update_start_button_state()

    @QtCore.Slot()
    def on_open_camera_track_clicked(self) -> None:
        """Switch to physical webcam + ByteTrack subject-tracking mode."""
        self.stop_inference()
        self.mode = "obs_track"
        self.current_video_path = "Camera + ByteTrack"
        self.control_panel.set_status("Mode: Camera + ByteTrack")
        self.append_text("Switched to Camera + ByteTrack tracking modeyteTrack 追蹤")
        self.append_text("已切換至鏡頭 + ByteTrack 追蹤模式")
        self._stop_all_source_threads()

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
        self.obs_bytetrack_thread = OBSByteTrackThread(
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

    def on_open_obs_track_clicked(self) -> None:
        """Switch to OBS Virtual Camera + ByteTrack subject-tracking mode."""
        self.stop_inference()
        self.mode = "obs_track"
        self.current_video_path = "OBS Mode: OBS + ByteTrack"
        self.append_text("Switched to OBS + ByteTrack tracking mode - ensure OBS Virtual Camera is active")
        self.append_text("已切換至 OBS + ByteTrack 追蹤模式 — 請確認 OBS 已啟動虛擬攝影機")
        self._stop_all_source_threads()

        # Read bytetrack config from configs dict
        bt_cfg = self.configs.get("bytetrack", {})

        # Resolve repo path: config value takes priority, env var is fallback
        # (ByteTrackWrapper itself also does the same resolution internally)
        repo_path = bt_cfg.get("bytetrack_repo") or None
        exp_file  = bt_cfg.get("exp_file",  "exps/example/mot/yolox_x_mix_det.py")
        ckpt_path = bt_cfg.get("ckpt_path", "pretrained/bytetrack_x_mot17.pth.tar")

        # If paths are relative, resolve them against the ByteTrack_repo dir
        import os
        if repo_path and not os.path.isabs(exp_file):
            exp_file  = os.path.join(repo_path, exp_file)
        if repo_path and not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(repo_path, ckpt_path)

        self.camera_start_time = time.time()
        self.obs_bytetrack_thread = OBSByteTrackThread(
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
            preloaded_tracker      = self._bytetrack_wrapper,
        )
        # Annotated BGR preview → GUI video panel
        self.obs_bytetrack_thread.signal_frame.connect(self.on_obs_track_frame)
        # Subject crop RGB → LiveCC inference
        self.obs_bytetrack_thread.signal_subject_frame.connect(self.on_obs_track_subject_frame)
        self.obs_bytetrack_thread.signal_error.connect(self.on_error)
        self.obs_bytetrack_thread.start()

        self.video_panel.slider.setEnabled(False)
        self._update_start_button_state()

    def _apply_tts_settings_before_start(self) -> None:
        """根據目前模式套用對應設定，並更新 GeminiBackgroundWorker 的 TTS 參考。"""
        self.tts_mode = self.control_panel.get_tts_mode()

        if self.tts_mode == "openai":
            voice = self.control_panel.get_openai_voice()
            speed = self.control_panel.get_openai_speed()
            self.signal_tts_apply_settings.emit(voice, float(speed))

        elif self.tts_mode == "gemini":
            pass  # voice set via app.yml; no UI controls needed

        elif self.tts_mode == "local":
            exag = self.control_panel.get_local_exaggeration()
            cfg = self.control_panel.get_local_cfg()
            self.signal_local_tts_apply_settings.emit(float(exag), float(cfg))

        # Point GeminiBackgroundWorker at the active TTS worker for backpressure
        if hasattr(self, "gemini_bg_worker"):
            if self.tts_mode == "gemini" and hasattr(self, "gemini_tts_worker"):
                self.gemini_bg_worker._tts_worker = self.gemini_tts_worker
            else:
                self.gemini_bg_worker._tts_worker = self.tts_worker

    @QtCore.Slot()
    def on_start_clicked(self) -> None:
        if self.is_inference_running:
            self.stop_inference()
            return

        if not self.model_ready:
            self.append_text("Model not ready yet")
            return

        # Start new log session before inference
        log_path = self.session_logger.start_new_session(self.mode)
        print(f"[Main] Session log started: {log_path}")

        style_key = self.control_panel.get_selected_style_key()
        style_label = self.control_panel.get_selected_style_label()

        if self.prompt_manager is not None:
            prompt = self.prompt_manager.livecc_query()
            from .core.models.gemini_broadcaster import set_style
            set_style(style_key)
        else:
            prompt = "Describe only what you see on screen right now in one objective sentence."

        self.livecc_worker.response_prefix = ""
        self.cam_worker.response_prefix = ""
        self._use_gemini = True

        # Apply TTS settings before start
        self._apply_tts_settings_before_start()

        self.is_inference_running = True
        self.control_panel.set_start_button_state(True)
        self.control_panel.set_tts_controls_enabled(False)  # Lock during inference

        # Start background Gemini loop on first inference run only.
        if not getattr(self, "_gemini_bg_loop_started", False):
            self._gemini_bg_loop_started = True
            QtCore.QMetaObject.invokeMethod(
                self.gemini_bg_worker, "run_background_loop",
                QtCore.Qt.QueuedConnection,
            )
        self.text_output.setText("")
        self._obs_drop_logged = False  # reset drop-log flag so it fires again if needed

        self.append_text(f"Starting inference (Style: {style_label}, TTS: {self.tts_mode})")
        self.append_text(f"開始推論 (Style: {style_label}, TTS: {self.tts_mode})")

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
            self._livecc_run_id += 1
            self.signal_start_livecc.emit(self.current_video_path, prompt, self._livecc_run_id)

        elif self.mode in ("camera", "obs", "obs_track"):
            self.signal_start_camera_livecc.emit(prompt)

    def stop_inference(self) -> None:
        if not self.is_inference_running:
            return

        self.append_text("停止推論")
        self._tts_last_priority = 5
        self._pending_tts_transition = ""
        self._last_tts_raw_text = ""
        self._last_tts_emit_ts = 0.0
        self._tts_protect_until = 0.0
        if hasattr(self, "_pending_segments"):
            self._pending_segments.clear()
        if self.tts_mode == "openai":
            try: self.signal_tts_interrupt.emit()
            except: pass
        elif self.tts_mode == "gemini":
            try: self.signal_gemini_tts_interrupt.emit()
            except: pass
        elif self.tts_mode == "local":
            try: self.signal_local_tts_interrupt.emit()
            except: pass
        if hasattr(self, "gemini_worker"):
            self.gemini_worker.flush_and_abort()
        if hasattr(self, "gemini_bg_worker"):
            self.gemini_bg_worker.requestStop()
        self._post_p1_pending = False
        if hasattr(self, "livecc_worker"):
            self.livecc_worker.requestStop()
        if hasattr(self, "cam_worker"):
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

        self.is_inference_running = False
        self.control_panel.set_start_button_state(False)
        self.control_panel.set_tts_controls_enabled(True)  # Unlock after stop
        self.control_panel.set_tts_controls_enabled(True)  # ✅ 解鎖：停止後可改

    # ---------------- Frame handlers ----------------

    @QtCore.Slot(np.ndarray, int, float)
    def on_video_frame(self, frame_rgb: np.ndarray, frame_idx: int, fps: float) -> None:
        self.video_panel.update_frame(frame_rgb)
        self.video_panel.set_position(frame_idx, fps)

        if self.mode == "file" and fps and fps > 0:
            self._playback_sec = float(frame_idx) / float(fps)

    @QtCore.Slot(np.ndarray)
    def on_camera_frame(self, frame_rgb: np.ndarray) -> None:
        self.video_panel.update_frame(frame_rgb)
        if self.is_inference_running and self.mode in ("camera", "obs"):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            t_relative = time.time() - self.camera_start_time
            self.cam_worker.push_frame(frame_bgr, t_relative)

    @QtCore.Slot(np.ndarray)
    def on_obs_track_frame(self, annotated_bgr: np.ndarray) -> None:
        """Display ByteTrack annotated frame (BGR) in the video panel."""
        self.video_panel.update_frame(annotated_bgr, is_bgr=True)

    @QtCore.Slot(np.ndarray)
    def on_obs_track_subject_frame(self, subject_crop_rgb: np.ndarray) -> None:
        """Forward the padded subject crop (RGB) from ByteTrack to LiveCC cam_worker."""
        if self.is_inference_running and self.mode == "obs_track":
            # Resize to fixed size so np.stack in build_clip_from_buffer never fails
            # with variable-sized crops from ByteTrack.
            fixed = cv2.resize(subject_crop_rgb, (640, 480))
            # cam_worker.push_frame expects BGR
            subject_bgr = cv2.cvtColor(fixed, cv2.COLOR_RGB2BGR)
            t_relative = time.time() - self.camera_start_time
            self.cam_worker.push_frame(subject_bgr, t_relative)
        else:
            # Diagnostic: print why frames are being dropped
            if not hasattr(self, '_obs_drop_logged'):
                self._obs_drop_logged = True
                print(f"[GUI] ⚠️  on_obs_track_subject_frame dropped: "
                      f"is_inference_running={self.is_inference_running}, mode='{self.mode}'")

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

    def _is_duplicate_tts(self, raw_text: str) -> bool:
        """True when raw_text is near-identical to the last TTS emit within the dedup window."""
        if not self._last_tts_raw_text:
            return False
        if time.time() - self._last_tts_emit_ts > self._DEDUP_WINDOW_S:
            return False
        new_words = set(raw_text.lower().split())
        last_words = set(self._last_tts_raw_text.lower().split())
        if not new_words or not last_words:
            return False
        overlap = len(new_words & last_words) / min(len(new_words), len(last_words))
        return overlap >= self._DEDUP_THRESHOLD

    def _apply_tts_transition(self, tts_text: str, priority: int) -> str:
        """Prepend any pending transition phrase and update _tts_last_priority."""
        phrase = self._pending_tts_transition
        self._pending_tts_transition = ""
        self._tts_last_priority = priority
        if phrase:
            return f"{phrase} {tts_text}"
        return tts_text

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
        display_text, tts_text = self._extract_segment_texts(data)

        # Log Gemini output separately when using Gemini
        if self._use_gemini and isinstance(data, dict):
            self._ensure_log_dir()
            priority = data.get("priority", "?")
            broadcast_text = data.get("broadcast_text", "")
            self._write_log(
                self.gemini_log_file,
                f"[Gemini] [{self._fmt_time(start_t)}-{self._fmt_time(stop_t)}] [P{priority}] {broadcast_text}",
            )

        # Resolve priority for this segment (Gemini dict has it; fallback to 5)
        seg_priority = 5
        if isinstance(data, dict):
            seg_priority = int(data.get("priority", 5))

        # MatchTracker scoring — triggered by Gemini action_label
        if isinstance(data, dict) and "action_label" in data:
            action_label = data["action_label"] or ""
            if action_label.startswith("score_"):
                team = action_label[len("score_"):]  # "red" or "blue"
                match_tracker.add_score(team)
                match_tracker.set_last_event(action_label)
                red, blue = match_tracker.get_scores()
                logging.info("[MatchTracker] %s scored → Red %d : Blue %d", team, red, blue)

        # Camera mode：沒有播放器時間軸可排程，所以直接顯示/唸
        if self.mode != "file":
            line = f"[{self._fmt_time(start_t)}] {display_text}"
            self._append_ui(line)

            if not tts_text.strip():
                return
            if self._is_duplicate_tts(tts_text):
                return
            now = time.time()
            if seg_priority > self._tts_last_priority and now < self._tts_protect_until:
                return  # priority guard: don't let lower-priority interrupt active playback
            self._last_tts_raw_text = tts_text
            self._last_tts_emit_ts = now
            self._tts_protect_until = now + max(2.0, len(tts_text.split()) * 0.3)
            tts_text = self._apply_tts_transition(tts_text, seg_priority)
            if self.tts_mode == "openai":
                self.signal_tts_speak.emit(tts_text, seg_priority, ref_ts, start_t)
            elif self.tts_mode == "gemini":
                self.signal_gemini_tts_speak.emit(tts_text, seg_priority, ref_ts, start_t)
            elif self.tts_mode == "local":
                self.signal_local_tts_speak.emit(tts_text)
            return

        # File mode: Enter queue, wait until video playback reaches corresponding time (stay in sync)
        if not hasattr(self, "_pending_segments"):
            self._pending_segments = deque()

        self._pending_segments.append((float(start_t), float(stop_t), data))

        if not tts_text.strip():
            return
        if self._is_duplicate_tts(tts_text):
            return
        now = time.time()
        if seg_priority > self._tts_last_priority and now < self._tts_protect_until:
            return  # priority guard: don't let lower-priority interrupt active playback
        self._last_tts_raw_text = tts_text
        self._last_tts_emit_ts = now
        self._tts_protect_until = now + max(2.0, len(tts_text.split()) * 0.3)
        tts_text = self._apply_tts_transition(tts_text, seg_priority)
        # 根據模式分流
        if self.tts_mode == "openai":
            self.signal_tts_speak.emit(tts_text, seg_priority, ref_ts, start_t)
        elif self.tts_mode == "gemini":
            self.signal_gemini_tts_speak.emit(tts_text, seg_priority, ref_ts, start_t)
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
        """Called when TTS naturally finishes one segment (never on forced interrupt).
        Triggers the mandatory 1.0s post-P1 silence before Gemini background resumes."""
        if self._post_p1_pending:
            self._post_p1_pending = False
            QtCore.QTimer.singleShot(1000, self._resume_gemini_background)
            logging.info("[P1 Silence] TTS done naturally, scheduling 1.0s before Gemini resumes")

    def _resume_gemini_background(self) -> None:
        """Called 1.0s after P1 TTS finishes. Resumes Gemini slow blade."""
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
        can_start = self.model_ready and (self.current_video_path is not None)
        self.control_panel.btn_start.setEnabled(bool(can_start))

    def _ensure_log_dir(self) -> None:
        if not hasattr(self, "log_dir"):
            self.log_dir = _find_project_root(Path(__file__)) / "log"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.livecc_log_file = self.log_dir / "livecc_output.log"
            self.gemini_log_file = self.log_dir / "gemini_output.log"

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
        """Append a segment line to the UI only — do not re-log (already logged at source)."""
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
        # ✅ 先停推論
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

        # ✅ 停 LiveCC worker threads
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

        try:
            self.signal_gemini_tts_stop.emit()
        except: pass
        if hasattr(self, "gemini_tts_thread") and self.gemini_tts_thread:
            self.gemini_tts_thread.quit()
            self.gemini_tts_thread.wait(2000)

        super().closeEvent(event)