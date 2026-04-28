# MIIS Broadcast

A real-time AI sports broadcasting commentary system with a desktop GUI. It ingests video from a file or live camera, generates commentary using the **LiveCC-7B** streaming video captioner, and reads it aloud via a TTS engine — all with sub-second end-to-end latency.

---

## Features

- **Five input modes** (matches the Source panel: **Offline** + **Online ▾** menu):
  - **Video file** — local file playback with seek bar
  - **Webcam** — physical camera only; auto-detect skips the OBS Virtual Camera device
  - **Webcam + Tracking** — ByteTrack subject lock and box overlay; with **remote inference** the server runs ByteTrack on the received frames; with **local inference** tracking runs on this machine via `CameraByteTrackThread`
  - **VR (OBS Virtual Camera)** — any source you route into OBS (e.g. Quest Link / game capture) and expose as **OBS Virtual Camera**; same “plain” full-frame stream as Webcam, different device index
  - **VR & Webcam (Sync)** — synchronized dual capture: physical webcam + OBS Virtual Camera stitched side-by-side (`1280×480`) using back-to-back `grab()` / `retrieve()`
- **Session Logging**: All terminal logs and AI-generated commentary (TTS output) are automatically saved to a unified log file in `logs/sessions/` for each broadcast session.
- **Thin-client telemetry (remote `obs_track`)**: The **inference server** prints **process RSS on the GPU host** (decode + ByteTrack + LiveCC) and, optionally, **sender-PC** stats (JPEG queue + client RSS) on the **same stdout** as ByteTrack **Infer FPS / Wall FPS**, so tuning **15 fps send / PREVIEW throttle** vs RAM is observable in one terminal. Sender stats use a tiny `CLIENT_DIAG` control message (~hundreds of bytes, no meaningful overhead).
- **Optimized Performance**: High-FPS video rendering with reduced jitter and correct color channel handling (BGR/RGB auto-switching).
- **Clean Source Switching**: Automated thread management ensuring smooth transitions between different video inputs. On Windows, a safe `wait(timeout) + terminate()` fallback prevents GUI freezes caused by DirectShow blocking `cap.read()` during mode switches.
- **Background Model Preloading**: The ByteTrack (YOLOX) model is loaded in a background thread 0.5 s after startup. Switching to any tracking mode is instant instead of freezing the UI for several seconds.
- **Multiple commentary styles** switchable at runtime:
  - 嘴砲型實況主 (Trash-talk / Roast)
  - 熱血沸騰型主播 (High-energy Hype)
  - 冷靜分析型 (Calm & Analytical)
- **Text-to-Speech** with two backend options:
  - OpenAI Realtime API (low-latency streaming, cloud)
  - ChatterBox TTS (local, voice-cloning)
- **Latency monitoring** — tracks LiveCC inference time, TTS latency, and end-to-end (vision → audio) latency
- **PySide6 GUI** with dark theme, video seek bar, and live transcript panel

---

## Architecture

Two deployment modes share the same GUI; the inference backend is selected at connection time.

### Local inference (offline)

```
Video File / Camera / OBS / DualSync
        │  RGB frames (30 fps)
        ▼
VideoThread / CameraThread / OBSCameraThread / DualSourceCameraThread
        │
  (obs_track only)
CameraByteTrackThread  ──  YOLOX + BYTETracker
  • annotated BGR  → video panel
  • subject crop (640×480 RGB)
        │
        ▼
LiveCCWorker / LiveCCCameraWorker
  (LiveCC-7B-Instruct, GPU, local)
        │ commentary text
        ▼
TTS Engine (OpenAI Realtime WebSocket or ChatterBox local)
        │ PCM 24 kHz
        ▼
sounddevice or ffplay (audio output)
```

### Remote inference (online / thin-client)

```
Camera / OBS / DualSync / File
        │  RGB frames (GUI thread, 30 fps raw)
        ▼
on_camera_frame / on_video_frame
        │  BGR JPEG (75%, async encode)
        ▼  non-blocking enqueue (max 30 frames)
SocketClientRunner._frame_queue
        │
_frame_sender_loop (background thread)
        │  TCP sendall  ─────────────────────────────────────────────►  Remote server
        │                                                               │
        │  ◄── MSG_SEGMENT (text) ◄── LiveCC inference (GPU)  ◄────────┤
        │  ◄── MSG_PREVIEW (JPEG) ◄── ByteTrack overlay ───────────────┘
        │                               (obs_track mode only, up to ~30 fps)
        ▼
on_segment → text panel + OpenAI TTS (local audio)
on_remote_track_preview → video panel (annotated frames with tracking boxes)
```

#### Why the frame sender is in a dedicated thread

`sock.sendall()` blocks the caller until the TCP send buffer is drained.  At 30 fps the
client produces ~90 KB/s of JPEG data; if the server GPU is busy the kernel buffer fills
and `sendall` stalls for tens of milliseconds — long enough to freeze the Qt event loop
and make the GUI unresponsive.

The fix: `send_frame()` JPEG-encodes the frame (on the calling thread) and drops it
into a `queue.Queue(maxsize=30)`.  A second background thread (`_frame_sender_loop`)
drains the queue and calls `sendall`.  The GUI thread is never blocked by TCP I/O.
If the queue is full the newest frame is silently dropped (`put_nowait`), keeping
memory bounded and backpressure natural.

#### Client send rate and PREVIEW throttle (currently 30 fps)

Both `_FRAME_SEND_FPS_MAX` (client → server) and the server-side PREVIEW throttle
(`1.0 / 30.0`) are set to **30 fps** as the testing baseline, matching the camera source.

Measured on server with ByteTrack + LiveCC sharing one GPU (obs_track):
- **ByteTrack Infer FPS** exceeds 30 fps — the GPU can keep up at 30 fps input
- **Server RSS** stabilises at ~2 500 MiB and is driven by LiveCC KV cache, not frame rate
- **Host system RAM** remains at ~28 % with 30 fps — headroom is comfortable
- **Client JPEG send queue** stays at 0 / 30 — no backpressure at 30 fps

LiveCC itself only uses a 2 s / 2 fps clip per inference cycle regardless of send rate;
the higher send rate benefits ByteTrack tracking smoothness and PREVIEW display quality.

Reduce `_FRAME_SEND_FPS_MAX` and the PREVIEW throttle together if bandwidth or GPU
becomes a constraint (e.g. 15 fps on a weaker GPU or a slow network link).

#### Why the PREVIEW display threshold is 1500 ms

The camera thread emits raw frames at 30 fps (~33 ms).  Server PREVIEW frames nominally
arrive at ~33 ms intervals (30 fps), but ByteTrack (YOLOX) running alongside LiveCC
inference can push actual intervals above that under GPU load.

A 1500 ms threshold means: after the last server PREVIEW arrives, raw frames are suppressed
for 1.5 s.  This keeps annotated frames visible even when the server is momentarily busy,
without permanently freezing if the connection drops (after 1.5 s of silence the client
falls back to raw camera).  A narrower threshold (e.g. 400 ms) caused the annotated and
raw frames to alternate visibly — tracking boxes appeared to jump.

Key modules:

| Path | Role |
|---|---|
| [src/miis_broadcast/network/protocol.py](src/miis_broadcast/network/protocol.py) | TCP wire format (`pack_message`, `read_message`), message constants including `CLIENT_DIAG` |
| [src/miis_broadcast/network/client.py](src/miis_broadcast/network/client.py) | `SocketClientRunner` — non-blocking JPEG send queue, optional thin-client diagnostics |
| [src/miis_broadcast/server/session.py](src/miis_broadcast/server/session.py) | Per-connection handler: FRAME decode, ByteTrack, PREVIEW throttle, stdout RAM telemetry |
| [src/miis_broadcast/gui.py](src/miis_broadcast/gui.py) | Main window, video panel, control widgets |
| [src/miis_broadcast/workers/livecc.py](src/miis_broadcast/workers/livecc.py) | QThread workers for LiveCC inference (file & camera) |
| [src/miis_broadcast/workers/camera_bytetrack.py](src/miis_broadcast/workers/camera_bytetrack.py) | Physical webcam + YOLOX/BYTETracker subject tracking worker (`CameraByteTrackThread`) |
| [src/miis_broadcast/core/models/bytetrack_tracker.py](src/miis_broadcast/core/models/bytetrack_tracker.py) | ByteTrackWrapper — YOLOX inference, BYTETracker association, subject crop extraction |
| [src/miis_broadcast/core/models/livecc_transformers.py](src/miis_broadcast/core/models/livecc_transformers.py) | LiveCCInfer — model loading, streaming inference, KV-cache management |
| [src/miis_broadcast/core/models/openai_tts.py](src/miis_broadcast/core/models/openai_tts.py) | OpenAI Realtime WebSocket TTS engine |
| [src/miis_broadcast/core/models/chatterbox_tts.py](src/miis_broadcast/core/models/chatterbox_tts.py) | Local ChatterBox TTS engine |
| [src/miis_broadcast/core/utils/session_logger.py](src/miis_broadcast/core/utils/session_logger.py) | SessionLogger — handles unified logging of system events and commentary |
| [src/miis_broadcast/core/prompt/prompt_manager.py](src/miis_broadcast/core/prompt/prompt_manager.py) | Loads and builds commentary style prompts from YAML |
| [configs/livecc_prompts.yml](configs/livecc_prompts.yml) | Commentary style definitions |
| [configs/models.yml](configs/models.yml) | Model registry — LiveCC and ByteTrack configs |
| [configs/app.yml](configs/app.yml) | GUI and default model settings |

---

## Requirements

- Python 3.10
- CUDA-capable GPU (tested with CUDA 12.x)
- For **OpenAI TTS** playback, either **`sounddevice`** (default; uses PortAudio) or **`ffplay`** (FFmpeg) on your **PATH** as a fallback
- OpenAI API key (if using the OpenAI TTS backend)
- [ByteTrack_repo](https://github.com/ifzhang/ByteTrack) — required for OBS + tracking mode (set path in `configs/models.yml` or via `BYTETRACK_REPO` env var)

**No TTS sound?**  Check these in order:

1. TTS combobox is **OpenAI TTS**, not **不啟用 (Mute)**.
2. `pip install sounddevice` (or ensure `sounddevice` from `requirements.txt` is installed). If neither **sounddevice** nor **ffplay** works, the console prints an error and there will be no audio even if commentary text appears.
3. `OPENAI_API_KEY` is valid; invalid keys stop the TTS WebSocket and also yield no sound.

**Remote inference:** OpenAI TTS and audio playback run on the **client machine** (where you run the GUI) — not on the headless `miis_broadcast.server` host. The server only needs GPU for LiveCC + ByteTrack.

**Similar lines repeating in remote commentary?** The server builds **overlapping ~2s video clips** every inference tick; the VLM can echo the same phrasing. The server also **skips near-duplicate** segments (vs. the previous line) and periodically resets model state to mitigate loops; for variety, adjust the **commentary style** / `query` in `configs/livecc_prompts.yml` (e.g. ask for 繁體中文).

### Python dependencies

Install via pip:

```bash
pip install -r requirements.txt
```

Or use the provided conda environment (includes PyAudio and system audio libs):

```bash
conda env create -f environment.yml
conda activate livecc_chatterboxtts
pip install -r requirements.txt
```

> **Note:** `torch`, `torchaudio`, `torchvision`, and NVIDIA CUDA packages are commented out in `requirements.txt`. Install the versions that match your CUDA setup from [pytorch.org](https://pytorch.org/get-started/locally/) before running pip install.

```bash
# Example for CUDA 12.9
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

---

## Configuration

### Environment variables

Create a `.env` file at the project root:

```env
OPENAI_API_KEY=sk-...
```

This is required for the OpenAI Realtime TTS backend.

### App config ([configs/app.yml](configs/app.yml))

```yaml
gui_window:
  title: MISLAB
  min_width: 1200
  min_height: 500
  default_open_dir: ./
model:
  classifier_name: livecc_7b   # must match a key in configs/models.yml
```

### Commentary styles ([configs/livecc_prompts.yml](configs/livecc_prompts.yml))

Add or edit styles under the `styles` key. Each style needs:
- `label` — display name shown in the GUI
- `description` — short description
- `query` — the system prompt sent to LiveCC

### ByteTrack config ([configs/models.yml](configs/models.yml))

Required when using OBS + tracking mode:

```yaml
bytetrack:
  bytetrack_repo: "/path/to/ByteTrack_repo"   # or set BYTETRACK_REPO env var
  exp_file: "/path/to/yolox_x_mix_det.py"
  ckpt_path: "/path/to/bytetrack_x_mot17.pth.tar"
  device: "cuda"
  fp16: true
  # Subject quality gates
  min_subject_area_ratio: 0.03   # ignore tracks smaller than 3% of frame area
  preempt_ratio: 4.0             # switch subject when a larger person appears
```

The `bytetrack_repo` path can also be set via the `BYTETRACK_REPO` environment variable.

---

## Installation

```bash
git clone <repo-url>
cd miis_broadcast
pip install -e .
```

The editable install registers the `livecc-run` entry point.

---

## Usage

```bash
livecc-run
```

Or run as a module from the project root:

```bash
python -m miis_broadcast
```

### In the GUI

1. **Select input** — two buttons in the Source panel:
   - **📁 Offline** — open a local video file
   - **🌐 Online ▾** — four live sources:
     - **📷 Webcam** — physical webcam, plain stream (auto-skips OBS Virtual Camera)
     - **🎯 Webcam + Tracking** — ByteTrack; **remote** = tracking on server, **local** = `CameraByteTrackThread` on this PC
     - **🥽 VR (OBS Virtual Camera)** — e.g. Quest Link / capture into OBS, then use OBS Virtual Camera as the device
     - **🎮 VR & Webcam (Sync)** — `1280×480` side-by-side: webcam + OBS Virtual Camera
2. **Choose a commentary style** from the dropdown
3. **Select TTS backend** — OpenAI Realtime or ChatterBox Local (Note: ChatterBox may be disabled in some environments)
4. **Click Start Broadcasting** — the model loads on first run (LiveCC-7B takes ~30–60 s on first local use). ByteTrack (YOLOX) preloads in the background so **Webcam + Tracking** is ready without a long stall (local path only).
5. Commentary text appears in the transcript panel and is read aloud in real time
6. **Click Stop Broadcasting** to end inference; latency statistics are printed to the console

#### Subject Tracking behavior

See [src/miis_broadcast/workers/README.md](src/miis_broadcast/workers/README.md) for full details.

---

## Input Source Interface (For Backend Engineers)

Full documentation — including signal/slot mapping, frame emission contracts, dual-source sync architecture, subject tracking behaviour, and a step-by-step guide for adding new sources — is in:

👉 [src/miis_broadcast/workers/README.md](src/miis_broadcast/workers/README.md)

---

## Remote Inference Server

The server is a standalone headless Python process that loads LiveCC once and serves
multiple successive client connections.

### Starting the server

```bash
# From project root on the remote machine:
python -m miis_broadcast.server                        # default: 0.0.0.0:9000, CUDA:0
python -m miis_broadcast.server --host 0.0.0.0 --port 9000 --device 0
```

### Connecting from a local machine over SSH

```bash
# On the local machine — forward port 9000 through SSH tunnel:
ssh -p 2225 -L 9000:127.0.0.1:9000 miislab-server3@10.50.0.103

# Then in the GUI, set Remote Host = 127.0.0.1, Port = 9000 and click Connect.
```

### TCP protocol (wire format)

Every message on the socket is framed as:

```
[4B total_len (BE uint32)] [4B json_len (BE uint32)] [json_bytes] [binary_payload]
```

`total_len = json_len + len(binary)`.  If there is no binary, `total_len == json_len`.

| Message | Direction | JSON fields | Binary |
|---------|-----------|-------------|--------|
| `HELLO` | C → S | `protocol_version` | — |
| `ACK` | S → C | `protocol_version` | — |
| `START` | C → S | `mode`, `query` | — |
| `FRAME` | C → S | `frame_id`, `t` | JPEG bytes |
| `PREVIEW` | S → C | `t` | JPEG bytes (annotated BGR) |
| `SEGMENT` | S → C | `start_t`, `stop_t`, `text` | — |
| `STATUS` | S → C | `msg` | — |
| `ERROR` | S → C | `msg` | — |
| `STOP` | C → S | — | — |
| `PING` / `PONG` | bidirectional | — | — |
| `CLIENT_DIAG` | C → S | optional `rss_mib`, `jpeg_q_used`, `jpeg_q_max`, `sys_ram_pct` | — |

### Memory telemetry (remote Webcam + Tracking)

When inference runs on a **remote** server (`obs_track` + TCP), **RSS** readings refer to **different machines** unless stated otherwise:

| Printed line (server **stdout**, same stream as ByteTrack FPS `print`s) | Meaning |
|---|---|
| `[ByteTrack] Frame … \| Infer FPS … \| Wall FPS …` | Tracking throughput on the **inference host** (existing `ByteTrackWrapper` log, every 20 frames). |
| `[Server RSS] full python process (LiveCC+ByteTrack+decode): …` | **Whole** `miis_broadcast.server` process RSS (LiveCC / Qwen **and** ByteTrack / YOLO **and** JPEG decode — not split per model). Emitted every **20** ByteTrack frames (same cadence as FPS lines). |
| `[ByteTrack] Thin-client (sender PC) RAM: …` | **Laptop / GUI machine** that encodes JPEGs and sends `FRAME`s: client RSS, outbound JPEG queue depth, and sender system RAM. The client forwards a small `CLIENT_DIAG` message so these lines appear in the **server terminal** next to FPS, not only in the GUI console. |

**Tuning 15 fps send / PREVIEW:** prioritise the **`[Server RSS]`** line when asking whether the remote box is memory-bound. Sender-PC lines help if you suspect encode or TCP backlog on the client.

The diagnostic payload is a short JSON message (order of **hundreds of bytes** every ~2 s from the GUI timer). It does not meaningfully block other OS processes; control sends hold the client socket lock only for that small `sendall`.

### Server logging

The server uses Python `logging` at `INFO` level, written to `stderr` with forced line
buffering so output appears immediately in SSH / tmux sessions.  Format:

```
[HH:MM:SS] LEVEL miis_broadcast.server.session — message
```

See **Server-side log reference** in `src/miis_broadcast/workers/README.md` for a
full table of every log line and what it means.

---

## Session Log Files

Every time you click **Start Broadcasting**, a new file is created at:

```
logs/sessions/{mode}_{YYYYMMDD_HHMMSS}.log
```

where `{mode}` is the active input source (`camera`, `obs`, `obs_track`, `file`,
`dual_sync`).

### File format

```
==================================================
Session Started: YYYY-MM-DD HH:MM:SS
Input Mode: obs_track
Inference: remote
==================================================

[HH:MM:SS] [GUI] [INFO] Starting inference (Style: …, TTS: …)
[HH:MM:SS] [COMMENTARY] AI-generated commentary text…
[HH:MM:SS] [GUI] [INFO] Stopping inference
[HH:MM:SS] [Memory] [INFO] sender_PC RSS=… MiB | JPEG send_queue=… | …
```

| Line type | Meaning |
|-----------|---------|
| `[GUI] [INFO]` | System events from the GUI (start, stop, remote connection changes, errors) |
| `[Memory] [INFO]` | (Remote `obs_track` only) Sender-PC RSS and outbound JPEG queue depth written on the **client** while a session file is open; does **not** duplicate the server’s own RSS (see **Memory telemetry** above for server stdout). |
| `[COMMENTARY]` | Every segment of AI commentary as it arrives from LiveCC (remote or local) |
| `Inference: remote` | LiveCC ran on the remote server (thin-client mode) |
| `Inference: local` | LiveCC ran on the local GPU |

All five input modes produce the same file structure.  The only differences per mode are
the `Input Mode:` header line and the content of the `[COMMENTARY]` entries.

---

## Latency

The system measures and reports three metrics at the end of each session:

| Metric | Description |
|---|---|
| First LiveCC latency | Video → first commentary text (includes model warm-up) |
| Average LiveCC latency | Average video → text time after the first inference |
| Average TTS latency | Text → audio starts playing (OpenAI API response) |
| Average E2E latency | Vision inference start → audio playback start |

---

## Project Structure

```
miis_broadcast/
├── configs/
│   ├── app.yml               # GUI / model settings
│   ├── models.yml            # Model registry (LiveCC + ByteTrack paths)
│   └── livecc_prompts.yml    # Commentary style prompts
├── logs/
│   ├── app_error.log
│   └── sessions/             # Per-broadcast session logs ({mode}_{timestamp}.log)
├── src/miis_broadcast/
│   ├── app.py                # Entry point
│   ├── gui.py                # Main window
│   ├── network/
│   │   ├── protocol.py       # TCP wire format (pack/read message)
│   │   └── client.py         # SocketClientRunner — non-blocking frame sender
│   ├── server/
│   │   ├── __main__.py       # Headless server entry point
│   │   └── session.py        # ClientSession — per-connection handler
│   ├── core/
│   │   ├── io/               # Input helpers (camera, OBS virtual camera)
│   │   ├── models/           # LiveCC, OpenAI TTS, ChatterBox TTS, ByteTrackWrapper
│   │   ├── prompt/           # Prompt management
│   │   └── utils/            # Config, session_logger, latency monitor
│   ├── widgets/              # Custom Qt widgets
│   └── workers/              # QThread workers (LiveCC, TTS, input, OBS+ByteTrack, DualSync)
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

---

## License

Internal research project — MIISLab.
