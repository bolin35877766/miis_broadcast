# Input Source Workers

This directory contains all `QThread` worker classes responsible for ingesting video frames and forwarding them to the LiveCC inference pipeline (local or remote).

---

## Available Input Sources

| Mode string | Worker Class | File |
|---|---|---|
| `"file"` | `VideoThread` | `input.py` |
| `"camera"` | `CameraThread` | `input.py` |
| `"obs_track"` | `CameraByteTrackThread` | `camera_bytetrack.py` |
| `"obs"` | `OBSCameraThread` | `obs_input.py` |
| `"dual_sync"` | `DualSourceCameraThread` | `dual_source.py` |
| `"free_switch"` | `FreeSwitchCameraThread` | `free_switch.py` |

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
| `requestOpenFreeSwitch` | `on_open_free_switch_clicked()` | `FreeSwitchCameraThread` | `"free_switch"` |

**Free Switch extras:** `ControlPanel.requestSwitchSource(str)` is connected to `MainWindow.on_switch_source()` and forwarded to `FreeSwitchCameraThread.set_active_source()` (`"webcam"` \| `"vr"` \| `"dual"`).

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
| `FreeSwitchCameraThread` | `signal_frame` | `frame_rgb: np.ndarray` (`640×480` **or** stitched `1280×480`) | `on_camera_frame()` |
| `FreeSwitchCameraThread` | `signal_source_changed` | `str` (`webcam` / `vr` / `dual`) | `_on_free_switch_source_changed()`, also updates switch-bar highlight |

## Inference Backend: Local vs Remote

When the GUI is connected to a remote server (`SocketClientRunner` active), every input
mode routes frames to the server instead of the local LiveCC worker:

```
on_camera_frame / on_video_frame
        │
        ├── remote connected? ──► SocketClientRunner.send_frame()
        │                               └── _frame_sender_loop (background thread)
        │                                       └── TCP → server LiveCC + ByteTrack
        └── local model?      ──► cam_worker.push_frame() / signal_start_livecc
                                        └── local LiveCC GPU inference
```

`obs_track` has two paths depending on whether a remote server is connected:

| Condition | Tracking runs on | Preview frames |
|-----------|-----------------|----------------|
| Remote connected | Server (ByteTrack inside `ClientSession`; decoded frames queued, GPU work serialized with LiveCC) | Server sends **`MSG_PREVIEW` JPEG once per tracked frame** (rate ≈ ByteTrack Wall FPS, typically well below raw 30 fps upload) |
| Local only | `CameraByteTrackThread` on client GPU | `signal_frame` emits annotated BGR directly |

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

### Camera Index Configuration

By default:
- `cam_idx = 0` — physical webcam
- `vr_idx = 5` — OBS Virtual Camera

Both values are configurable at construction time. If the VR index fails to open with MSMF, the worker automatically retries with DSHOW.

### Terminal output

The worker prints a session banner and one stats line per second to stdout:

```
============================================================
DualSource Live Session  |  HH:MM:SS
CAM idx=0  |  VR idx=5  |  Target: 30 FPS
============================================================
[Timestamp]  TotalFPS | CamFPS | VrFPS | Dec | Proc | Gap | (reason)

[HH:MM:SS] TotalFPS:29.10 | Cam:29.10 | Vr:29.10 | Dec: 8.0ms | Proc: 1.2ms | Gap:0.004ms | (Normal)
```

| Field | Meaning |
|-------|---------|
| `TotalFPS` | Composite frames emitted per second |
| `Cam` / `Vr` | Successful `grab()` count/s for each source |
| `Dec` | `retrieve()` decode time (both sources combined) |
| `Proc` | `hstack` + BGR→RGB conversion time |
| `Gap` | Δt between the two `grab()` calls (sync quality indicator) |
| `(reason)` | `Normal` when FPS ≥ 80 % of target; otherwise `Heavy-Decode`, `Heavy-Proc`, `Bus-Congestion`, or `System-Lag` |

## Free Switch Worker (`free_switch.py`)

`FreeSwitchCameraThread` opens **both** the physical webcam and the OBS Virtual Camera once at startup and keeps them alive for the session. The user selects which view is **emitted**:

| Selector | Resolution | Behaviour |
|---------|------------|-----------|
| `webcam` (`SOURCE_WEBCAM`) | `640×480` RGB | `retrieve()` webcam only |
| `vr` (`SOURCE_VR`) | `640×480` RGB | `retrieve()` OBS/VR only |
| `dual` (`SOURCE_DUAL`) | `1280×480` RGB | Same `hstack` layout as **`DualSourceCameraThread`** |

### Why switching feels instant

Each loop iteration:

1. `cap_cam.grab()` **and** `cap_vr.grab()` — both hardware buffers advance every tick (avoids stale frames when switching back to an idle source).
2. Read `_active_source` under a mutex (set via `set_active_source()` / GUI buttons).
3. `retrieve()` only the frame(s) needed for that composition.
4. `signal_frame.emit()` → `MainWindow.on_camera_frame()` → `video_panel` + (when inferring) **`SocketClientRunner.send_frame`**.

No `VideoCapture` reopen; switching is a Python variable flip + next-frame retrieve path.

### GUI wiring

- **Online ▾ → Free Switch** → modal dialog chooses **initial** source.
- **`free_switch_bar`** (Webcam / VR / W+VR) stays visible while `mode == "free_switch"`; buttons stay enabled **during broadcasting** so the operator can swap sources mid-session.

### Inference

Same path as **`camera`** / **`dual_sync`**: `start_inference(..., mode="free_switch")` routes JPEGs to the server; server treats it like plain camera ingestion (ByteTrack **not** loaded unless `obs_track`). See root [README.md](../../../README.md) **Free Switch** subsection.

---

## Subject Tracking Behavior (`camera_bytetrack.py`)

- YOLOX detects all people in the frame each tick; BYTETracker assigns persistent IDs.
- The subject with the largest weighted score (area / centre distance) is locked as the primary subject.
- The bounding box is padded by 15% and cropped, then resized to `640×480` before being forwarded to LiveCC.
- If a new person enters the frame with more than **4×** the current subject's area, tracking switches automatically.
- When no valid subject is detected, frames are **not** forwarded to LiveCC (prevents empty-scene descriptions).
- Color space handling ensures correct BGR/RGB channel display during high-speed tracking.

Every 20 frames, a stats line is printed to stdout:

```
[ByteTrack] Frame   160 | Infer FPS: 61.9 | Wall FPS: 10.7 | Tracks: 1 | Subject ID: 1
```

| Field | Meaning |
|-------|---------|
| `Frame` | Frame counter since tracker was created |
| `Infer FPS` | `1 / average_single_frame_time` — pure YOLOX+ByteTracker throughput |
| `Wall FPS` | `total_frames / elapsed_wall_time` — actual end-to-end stream rate |
| `Tracks` | Number of active bounding boxes drawn this frame |
| `Subject ID` | BYTETracker ID currently locked as the primary subject (`None` if no subject) |

`Infer FPS` is typically much higher than `Wall FPS` because the camera capture loop,
Qt signal overhead, and inter-thread latency dominate the wall time.

---

## Server-side Log Reference

When using remote inference, the server (`python -m miis_broadcast.server`) emits
`logging.INFO` lines to stderr.  Format:

```
[HH:MM:SS] INFO miis_broadcast.server.session — message
```

### Startup (once per server process)

| Log line | Meaning |
|----------|---------|
| `Loaded model config from …` | `configs/models.yml` parsed successfully |
| `Loading LiveCC model on device=0 …` | Model loading started |
| `LiveCC model loaded ✓` | Ready to accept clients |
| `Listening on 0.0.0.0:9000 — waiting for clients…` | TCP server socket open |

### Per client connection

| Log line | Meaning |
|----------|---------|
| `New client connected: ('ip', port)` | TCP accept |
| `[Session …] HELLO ok, mode=camera` | Handshake complete (mode from HELLO, defaults to `camera`) |
| `[Session …] Disconnected` | Client closed connection or error |

### Per broadcast session (`MSG_START` → `MSG_STOP`)

| Log line | Meaning |
|----------|---------|
| `MSG_START mode=obs query_len=N` | Client clicked Start; `mode` is the input source |
| `Inference START mode=obs` | Inference loop thread started |
| `ByteTrack loaded for mode=obs_track` | ByteTrack model loaded successfully (**obs_track only**) |
| `obs_track but ByteTrack not loaded — …` | ByteTrack load failed; preview will be raw frames (**obs_track only**) |
| `First FRAME decoded shape=… t=…` | First frame successfully decoded from client |
| `FRAME stats rx=120 tx_previews=0 buffer_len=119 mode=obs` | Periodic stats every 120 received frames |
| `LiveCC run #N buffer_size=M clip_ok` | Background inference cycle started (≈ every 2 s) |
| `SEGMENT out #N t=[s,e] text…` | Commentary sent to client (logged on #1 and every 10th; others at DEBUG) |
| `MSG_STOP rx_frames=N tx_previews=M tx_segments=K` | Client clicked Stop |
| `Inference STOP rx_frames=N tx_previews=M tx_segments=K infer_cycles=J` | Session totals |

**`FRAME stats` field meanings**

| Field | Meaning |
|-------|---------|
| `rx` | Total frames decoded from client since last reset |
| `tx_previews` | `MSG_PREVIEW` frames sent back (tracking overlay); `0` for non-tracking modes |
| `buffer_len` | Frames currently in the LiveCC clip buffer (max 180) |
| `mode` | Input mode from `MSG_START` — confirms which source the client is using (`free_switch` is plain inference path, same as camera/dual for ByteTrack) |

**`tx_previews = 0` is normal** for `camera`, `obs`, `file`, `dual_sync`, and `free_switch` because
`MSG_PREVIEW` is only sent in `obs_track` mode.

Remote inference also uses **`MSG_CLIENT_DIAG`** (optional): the GUI sends compact JSON
every ~2 s with sender-PC RSS, outbound JPEG queue depth, and sender system RAM.
The session prints one line per message on **stdout** — see below (not mixed into the
`logging` stderr table).

### Server stdout (`print`, same terminal as ByteTrack FPS)

The Python **`logging`** lines above go to **stderr**.  Separately, **stdout** carries
`print()` lines from **`ByteTrackWrapper`** (YOLOX + tracker timing) and from
**`ClientSession`** (`server/session.py`) so operators can correlate FPS and RAM in one stream:

| Example prefix | Origin | Meaning |
|---|---|---|
| `[ByteTrack] Frame … \| Infer FPS … \| Wall FPS …` | `bytetrack_tracker.py` | Every **20** frames when ByteTrack runs on **this host** |
| `[Server RSS] full python process (LiveCC+ByteTrack+decode): …` | `session.py` | **Entire server Python process** RSS (includes LiveCC + ByteTrack + OpenCV decode; labels next to FPS lines only for readability) (`obs_track`, every **20** BT frames) |
| `[ByteTrack] Thin-client (sender PC) RAM: …` | `session.py` (payload from GUI via **`MSG_CLIENT_DIAG`**) | **Laptop / GUI** RSS, JPEG **send** queue, sender system RAM — upstream encode/TCP health |

**Remote vs local:** with **thin client + `obs_track`**, tracking runs on the server, so
these lines appear on the **remote machine’s** terminal (SSH/tmux). Local-only
`obs_track` (`CameraByteTrackThread` on your PC) prints the first line from the same
tracker class on **your** stdout; **`[Server RSS]`** is **whole-process** RAM (LiveCC + ByteTrack +
decode), not ByteTrack-only, on whatever machine runs `miis_broadcast.server`.

**Server (`obs_track`):** incoming `FRAME` JPEGs are decoded on a dedicated thread.
`bt.process()` and `live_cc_from_frames` run sequentially on that thread; all network I/O
runs freely outside it. The session uses a **single-slot `_frame_queue`** (`maxsize=1`,
latest FRAME overwrites). **PREVIEW** messages are **subsampled** to ≈**15 Hz** (see
`_PREVIEW_SAMPLE_OUT_FPS` in `session.py`). On the **client**,
`MainWindow._OBS_TRACK_PREVIEW_HOLD_SEC` (default **2.5 s**) suppresses raw camera between
PREVIEW updates to avoid flickering.

For a full narrative (30→15 phase sampling, PREVIEW caps, protocol field list), see the root
**Memory telemetry (remote Webcam + Tracking)** section in [README.md](../../../README.md).

---

## Session Log Files (client-side)

Every broadcast writes a file to `logs/sessions/{mode}_{YYYYMMDD_HHMMSS}.log`:

```
==================================================
Session Started: YYYY-MM-DD HH:MM:SS
Input Mode: obs_track
Inference: remote
==================================================

[HH:MM:SS] [GUI] [INFO] Starting inference (Style: …, TTS: …)
[HH:MM:SS] [COMMENTARY] AI-generated commentary text…
[HH:MM:SS] [GUI] [INFO] [Remote] 連線中斷: …
[HH:MM:SS] [GUI] [INFO] Stopping inference
[HH:MM:SS] [Memory] [INFO] sender_PC RSS=… MiB | JPEG send_queue=… | sender system_RAM_used=…%
```

**`[Memory] [INFO]`** — only when **remote `obs_track`** and a session file is active:
sender-PC RSS / outbound queue (see **`MSG_CLIENT_DIAG`** above). Does **not** include
server-side RSS; watch the inference host **stdout** lines for remote process RAM.

**All five input modes produce this same structure** (except optional `[Memory]` lines).
The `Input Mode:` header and `[COMMENTARY]` content differ; everything else is identical.

| Header field | Values |
|---|---|
| `Input Mode` | `camera` / `obs` / `obs_track` / `file` / `dual_sync` |
| `Inference` | `remote` (server TCP) / `local` (on-device LiveCC) / `unknown` |

---

## How to Add a New Input Source

1. **Define a new `QThread` subclass** in this directory that emits `signal_frame` (and optionally `signal_subject_frame`) following the payload convention in the table above.
2. **Add a `QtCore.Signal()`** to `ControlPanel` in `gui.py` (e.g. `requestOpenNewSource = QtCore.Signal()`).
3. **Wire up the menu action** in `ControlPanel.setup_ui()` to emit the new signal.
4. **Add a slot** `on_open_new_source_clicked()` in `MainWindow` that calls `_stop_all_source_threads()`, instantiates the new thread, connects its signals, and starts it.
5. **Register the thread** in `_stop_all_source_threads()` using the same `wait(3000) + terminate()` pattern to ensure clean shutdown on mode switch.
6. **Connect the new signal** in `MainWindow._initUI()` alongside the existing signal connections.

> **Important:** Always call `_stop_all_source_threads()` before starting a new source thread. This disconnects residual frame signals and prevents ghost frames after switching modes.
