# Input Source Workers

This directory contains all `QThread` worker classes responsible for ingesting video frames and forwarding them to the LiveCC inference pipeline.

---

## Available Input Sources

| Mode | Worker Class | File |
|---|---|---|
| Video file playback | `VideoThread` | `input.py` |
| Physical webcam (plain stream) | `CameraThread` | `input.py` |
| Physical webcam + ByteTrack tracking | `CameraByteTrackThread` | `camera_bytetrack.py` |
| OBS Virtual Camera / VR headset (plain stream) | `OBSCameraThread` | `obs_input.py` |
| **VR + Webcam synchronized dual-source** | `DualSourceCameraThread` | `dual_source.py` |

---

## Signal → Slot → Thread Mapping

`ControlPanel` signals are wired to `MainWindow` slots which start the corresponding thread:

| `ControlPanel` Signal | `MainWindow` Slot | QThread started | Mode string |
|---|---|---|---|
| `requestOpenVideo` | `on_open_video_clicked()` | `VideoThread` | `"file"` |
| `requestOpenCamera` | `on_open_camera_clicked()` | `CameraThread` | `"camera"` |
| `requestOpenCameraTrack` | `on_open_camera_track_clicked()` | `CameraByteTrackThread` | `"obs_track"` |
| `requestOpenOBS` | `on_open_obs_clicked()` | `OBSCameraThread` | `"obs"` |
| `requestOpenDualSync` | `on_open_dual_sync_clicked()` | `DualSourceCameraThread` | `"dual_sync"` |

---

## Frame Emission Signals

Each source thread emits one or two frame signals that `MainWindow` connects to:

| Thread | Signal | Payload | Connected to |
|---|---|---|---|
| `VideoThread` | `signal_frame` | `(frame_rgb: np.ndarray, frame_idx: int, fps: float)` | `on_video_frame()` |
| `CameraThread` | `signal_frame` | `frame_rgb: np.ndarray` | `on_camera_frame()` |
| `OBSCameraThread` | `signal_frame` | `frame_rgb: np.ndarray` | `on_camera_frame()` |
| `CameraByteTrackThread` | `signal_frame` | `annotated_bgr: np.ndarray` | `on_obs_track_frame()` |
| `CameraByteTrackThread` | `signal_subject_frame` | `subject_crop_rgb: np.ndarray` | `on_obs_track_subject_frame()` |
| `DualSourceCameraThread` | `signal_frame` | `combined_rgb: np.ndarray` | `on_camera_frame()` |

---

## Dual-Source Sync Worker (`dual_source.py`)

`DualSourceCameraThread` captures a physical webcam and an OBS Virtual Camera simultaneously and stitches them into a single side-by-side `1280×480` frame.

### Synchronization Strategy

Frames from two independent USB devices are aligned using OpenCV's two-stage grab/retrieve pattern:

```
cap_cam.grab()   ─┐  (hardware latch — both within ~0.004 ms)
cap_vr.grab()    ─┘
        │
cap_cam.retrieve()  ─── CPU decode
cap_vr.retrieve()   ─── CPU decode
        │
np.hstack([frame_cam, frame_vr])   ─── 1280×480 composite
        │
signal_frame.emit(combined_rgb)
```

- **`grab()`** locks the hardware timestamp of both cameras near-simultaneously (observed Δt ≈ 0.004 ms).
- **`retrieve()`** decodes each frame independently; typical combined decode time is 7–11 ms.
- **`np.hstack`** + `cv2.cvtColor` composite step typically takes 1–2 ms.

### Performance Characteristics

Under normal conditions the worker sustains **~29 FPS** with total per-frame processing well under the 33 ms budget:

```
Dec (decode):  7–11 ms
Proc (hstack): 1–2 ms
Gap (Δt):      ~0.004 ms
```

A live performance log is written to `logs/dual_sync_live.log` in the format:

```
[HH:MM:SS] TotalFPS:29.10 | Cam:29.10 | Vr:29.10 | Dec:8.0ms | Proc:1.2ms | Gap:0.004ms | (Normal)
```

### Camera Index Configuration

By default:
- `cam_idx = 0` — physical webcam
- `vr_idx = 5` — OBS Virtual Camera

Both values are configurable at construction time. If the VR index fails to open with MSMF, the worker automatically retries with DSHOW.

---

## Subject Tracking Behavior (`camera_bytetrack.py`)

- YOLOX detects all people in the frame each tick; BYTETracker assigns persistent IDs.
- The subject with the largest weighted score (area / centre distance) is locked as the primary subject.
- The bounding box is padded by 15% and cropped, then resized to `640×480` before being forwarded to LiveCC.
- If a new person enters the frame with more than **4×** the current subject's area, tracking switches automatically.
- When no valid subject is detected, frames are **not** forwarded to LiveCC (prevents empty-scene descriptions).
- Color space handling ensures correct BGR/RGB channel display during high-speed tracking.

---

## How to Add a New Input Source

1. **Define a new `QThread` subclass** in this directory that emits `signal_frame` (and optionally `signal_subject_frame`) following the payload convention in the table above.
2. **Add a `QtCore.Signal()`** to `ControlPanel` in `gui.py` (e.g. `requestOpenNewSource = QtCore.Signal()`).
3. **Wire up the menu action** in `ControlPanel.setup_ui()` to emit the new signal.
4. **Add a slot** `on_open_new_source_clicked()` in `MainWindow` that calls `_stop_all_source_threads()`, instantiates the new thread, connects its signals, and starts it.
5. **Register the thread** in `_stop_all_source_threads()` using the same `wait(3000) + terminate()` pattern to ensure clean shutdown on mode switch.
6. **Connect the new signal** in `MainWindow._initUI()` alongside the existing signal connections.

> **Important:** Always call `_stop_all_source_threads()` before starting a new source thread. This disconnects residual frame signals and prevents ghost frames after switching modes.
