# MIIS Broadcast

A real-time AI sports broadcasting commentary system with a desktop GUI. It ingests video from a file or live camera, generates short **English** scene descriptions with the **LiveCC-7B** streaming video captioner, enriches them into **Traditional Chinese** broadcast lines via **Gemini**, and reads them aloud through **OpenAI Realtime TTS** — with a fast–slow blade for priority events (scores, misses) and sub-second latency on the hot path.

---

## Features

- **Six input modes** (matches the Source panel: **Offline** + **Online ▾** menu):
  - **Video file** — local file playback with seek bar
  - **Webcam** — physical camera only; auto-detect skips the OBS Virtual Camera device
  - **Webcam + Tracking** — ByteTrack subject lock and box overlay; **always runs locally** on the client machine via `CameraByteTrackThread` (YOLOX + BYTETracker on local GPU/CPU). Only the extracted **subject crop** is forwarded to the remote server for LiveCC inference — raw frames are never sent to the server in this mode.
  - **VR (OBS Virtual Camera)** — any source you route into OBS (e.g. Quest Link / game capture) and expose as **OBS Virtual Camera**; same “plain” full-frame stream as Webcam, different device index
  - **VR & Webcam (Sync)** — synchronized dual capture: physical webcam + OBS Virtual Camera stitched side-by-side (`1280×480`) using back-to-back `grab()` / `retrieve()`
  - **Free Switch** — both Webcam and OBS Virtual Camera are opened at startup; only the **active** source (Webcam, VR, or stitched dual) is emitted to the video panel and forwarded to LiveCC (local/remote). Switching is a **software selector** only — **no camera reconnection**, sub-frame latency typical.
- **Session Logging**: All terminal logs and AI-generated commentary (TTS output) are automatically saved to a unified log file in `logs/sessions/` for each broadcast session.
- **Audience second screen (optional)**: In **Free Switch** mode, a browser viewer can subscribe via **LiveKit** to **`broadcast_video`** (full-screen VR at **native resolution**, typically **1920×1080**) plus **`narration`** (TTS) while the operator’s GUI continues to preview the active source. **`broadcast_video` is VR-only** (no mascot burned in by Python): an optional mascot/anchor clip is **composited in the browser** (`static/index.html` + `/assets/avatar/…` served from the token server). Setup, flow diagrams, and **`[AUDIENCE]` / `[MEDIA]` / `[AUDIO]`** log reference: [src/miis_broadcast/audience/README.md](src/miis_broadcast/audience/README.md).
- **Thin-client telemetry**: During **any** remote inference, the **inference server** stdout shows **`[Client]`** and **`[Server]`** lines: host **RAM** (RSS, system %) plus **CUDA VRAM** on **device 0** where available (global used/total, `torch_alloc` for this process). Lines are on a shared ~2 s cadence via `CLIENT_DIAG` and decode-thread sampling. The GUI does **not** print duplicate `[Client]` lines to its own console; optional **session log** may still record the same payload under `[Memory]`.
- **Session averages on Stop**: When **Stop Broadcasting** finishes, the server prints **`[SESSION AVG] Server`** and **`[SESSION AVG] Client`** (means of the periodic samples) on **server stdout**. The GUI prints **`[SESSION AVG] Client-local`**, optional audience **`[SESSION AVG] [MEDIA]`** / **`[AUDIO]`** (Free Switch + audience enabled), and **`[SESSION AVG] ByteTrack`** when **Webcam + Tracking** was used — all on the **client** machine.
- **Optimized Performance**: High-FPS video rendering with reduced jitter and correct color channel handling (BGR/RGB auto-switching).
- **Clean source switching vs Stop Broadcasting**: **Changing the input** from the Source menu (`_stop_all_source_threads`) fully stops the old worker with **`QThread.wait(~6s)` + `terminate()`** so a stuck DirectShow `cap.read()` cannot leak threads — the window **may hitch briefly** while joining. **Stop Broadcasting** only ends the **inference session** (`MSG_STOP`, `is_inference_running = false`); in **Webcam + Tracking**, **`CameraByteTrackThread` stays alive** and the **annotated preview keeps updating** (subject crops to the server stop as soon as Stop is pressed).
- **Background Model Preloading**: The ByteTrack (YOLOX) model is loaded in a background thread 0.5 s after startup. Switching to any tracking mode is instant instead of freezing the UI for several seconds.
- **Multiple commentary styles** switchable at runtime (Gemini broadcast tone; LiveCC vision prompt is shared):
  - 標準播報型 (Objective / professional)
  - 嘴砲型實況主 (Trash-talk / Roast)
  - 熱血沸騰型主播 (High-energy Hype)
  - 冷靜分析型 (Calm & Analytical)
- **Fast–slow blade**: LiveCC keyword hits (P1/P2) hard-cut lower-priority TTS; P3 descriptions feed **Gemini** (per-segment + background worker) for richer broadcast copy before TTS.
- **Text-to-Speech** (GUI dropdown):
  - **不啟用 (Mute)** — subtitles only
  - **OpenAI TTS** (default) — Realtime WebSocket, cloud, used for audience `narration` when enabled
  - **Local TTS** — Chatterbox voice-cloning (lazy-loaded; may be unavailable in some environments)
- **Latency monitoring** — tracks LiveCC inference time, TTS latency, and end-to-end (vision → audio) latency
- **OpenAI TTS live queue**: Incoming commentary lines enqueue **FIFO** in `core/models/openai_tts.py` (bounded by **`_MAX_PENDING_UTTERANCES`**; exceeding drops **oldest** backlog). **`drop_outdated=True`** (optional API) still clears pending text explicitly.
- **PySide6 GUI** with dark theme, video seek bar, and live transcript panel

---

## Architecture

Two deployment modes share the same GUI; the inference backend is selected at connection time.

### Local inference (offline)

```
Video File / Camera / OBS / DualSync / Free Switch
        │  RGB frames (30 fps)
        ▼
VideoThread / CameraThread / OBSCameraThread / DualSourceCameraThread / FreeSwitchCameraThread
        │
  (obs_track only)
CameraByteTrackThread  ──  YOLOX + BYTETracker
  • annotated BGR  → video panel
  • subject crop (640×480 RGB)
        │
        ▼
LiveCCWorker / LiveCCCameraWorker
  (LiveCC-7B-Instruct, GPU, local)
        │ English scene text
        ▼
GeminiWorker + GeminiBackgroundWorker  (gemini_broadcaster.py)
        │ zh-TW broadcast_text + priority
        ▼
TTS (OpenAI Realtime WebSocket, or Chatterbox local, or mute)
        │ PCM 24 kHz
        ▼
sounddevice (ffplay fallback if sounddevice unavailable)
```

### Remote inference (online / thin-client)

```
Camera / OBS / DualSync / File / Free Switch
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
        │  ◄── MSG_SEGMENT (English text) ◄── LiveCC inference (GPU)  ◄────────┘
        ▼
on_remote_segment → _route_segment → Gemini (client) → OpenAI TTS (local audio)


Webcam + Tracking (obs_track) — always local ByteTrack:

Webcam
  │  raw BGR frame (30 fps)
  ▼
CameraByteTrackThread (local)
  ├── YOLOX detection + BYTETracker  ──► annotated BGR → video panel (local)
  └── subject crop (640×480 BGR)
            │  JPEG (async, matches callback rate)
            ▼
      SocketClientRunner → TCP → Remote server
                                      │
                          LiveCC inference (GPU, no ByteTrack on server)
                                      │
                          MSG_SEGMENT ─────────────────────────────────► _route_segment → Gemini → TTS
```

#### Why the frame sender is in a dedicated thread

`sock.sendall()` blocks the caller until the TCP send buffer is drained.  At typical camera
rates the client produces a much higher JPEG bitrate than before when every frame is sent; if the
server or network is slow the kernel buffer fills
and `sendall` stalls for tens of milliseconds — long enough to freeze the Qt event loop
and make the GUI unresponsive.

The fix: `send_frame()` JPEG-encodes the frame (on the calling thread) and drops it
into a `queue.Queue(maxsize=30)`.  A second background thread (`_frame_sender_loop`)
drains the queue and calls `sendall`.  The GUI thread is never blocked by TCP I/O.
If the queue is full the newest frame is silently dropped (`put_nowait`), keeping
memory bounded and backpressure natural.

#### Client send rate

**Thin client:** each `send_frame()` call queues **one** `FRAME` JPEG.  The **wired FPS** therefore tracks
how often the GUI / worker invokes `send_frame` (commonly **~30 Hz** for live camera sources; file
playback follows the video thread).  There is **no** client-side downsampling.

**Ingress on server** uses a single-slot JPEG buffer (`maxsize=1`): each new FRAME **overwrites** pending decode work («always latest»).

LiveCC runs on a **2 s interval** (`infer_interval = 2.0` in `session.py`), independent of wire FPS.

Tune **`_FRAME_QUEUE_MAX`** (`client.py`, default 30) or JPEG quality if you need different backpressure.

#### Free Switch (`free_switch` mode)

**Online ▾ → Free Switch** opens a dialog (“初始輸入源”) with **Webcam**, **VR**, or **Webcam+VR**. The worker (`FreeSwitchCameraThread` in `workers/free_switch.py`) then keeps **both** capture devices running: every loop it `grab()`s both cameras but `retrieve()`s only the frames needed for the currently selected view. **Webcam** is captured at **`640×480`** for the main pipeline. **OBS/VR** is requested at **`1920×1080`** (when the driver honours it) and is **resized to `640×480` only for `signal_frame`** (GUI + LiveCC / remote inference). **Dual** is **`1280×480`** (webcam | VR-downscaled), same layout as **VR & Webcam (Sync)**. A separate **`signal_vr_frame`** always carries the **native-resolution VR** feed for the optional audience LiveKit path (see below). Changing the source updates an in-memory selector only; **TCP send_frame** and `on_camera_frame` keep running, so remote LiveCC receives a continuous JPEG stream whose content switches instantly. The Source panel shows a small **切換輸入源** bar (鏡頭 / VR / 拼接; buttons remain usable during broadcasting). **10s輪播** (toggle) runs a **10-second** `QTimer` that cycles **Webcam → VR → dual** in order; toggle again to stop. Leaving Free Switch clears the timer and the toggle.

**Audience (second display):** when `audience.enabled` is true in `configs/app.yml`, starting **Free Switch** also starts a **LiveKit publisher** that publishes **`broadcast_video` at `1920×1080`** (**VR only**; resize + RGBA pack in `livekit_publisher`) plus **`narration`** (TTS PCM). The viewer page composites an optional mascot on a **canvas** client-side (`/assets/*`); the publisher **does not** chroma-key or overlay the mascot in Python. Viewers use a local HTTP page on port **8080** (default). Wait for **`[MEDIA] connected`** in the GUI terminal. Higher resolution uses more uplink and encoder load; see **[src/miis_broadcast/audience/README.md](src/miis_broadcast/audience/README.md)** for Docker, firewall, mascot overlay tuning, flowcharts, and logs.

Key modules:

| Path | Role |
|---|---|
| [src/miis_broadcast/network/protocol.py](src/miis_broadcast/network/protocol.py) | TCP wire format (`pack_message`, `read_message`), message constants including `CLIENT_DIAG` |
| [src/miis_broadcast/network/client.py](src/miis_broadcast/network/client.py) | `SocketClientRunner` — non-blocking JPEG send queue, `send_client_diagnostic` / `CLIENT_DIAG` during remote inference |
| [src/miis_broadcast/server/session.py](src/miis_broadcast/server/session.py) | Per-connection handler: FRAME → JPEG decode → LiveCC buffer (**no** ByteTrack or `PREVIEW` on the server; tracking overlay stays on the client) |
| [src/miis_broadcast/gui.py](src/miis_broadcast/gui.py) | Main window, video panel — **Free Switch** source bar (`切換輸入源`); `obs_track` shows local ByteTrack annotated preview |
| [src/miis_broadcast/workers/free_switch.py](src/miis_broadcast/workers/free_switch.py) | `FreeSwitchCameraThread` — dual always-on captures; VR at **1080p** for audience `signal_vr_frame`; **640×480** / **1280×480** on `signal_frame` for LiveCC |
| [src/miis_broadcast/workers/gemini.py](src/miis_broadcast/workers/gemini.py) | `GeminiWorker` + `GeminiBackgroundWorker` — zh-TW broadcast from LiveCC text |
| [src/miis_broadcast/core/models/gemini_broadcaster.py](src/miis_broadcast/core/models/gemini_broadcaster.py) | Gemini API, RAG, stream parsing, style prompts |
| [src/miis_broadcast/workers/livecc.py](src/miis_broadcast/workers/livecc.py) | QThread workers for LiveCC inference (file & camera) |
| [src/miis_broadcast/workers/camera_bytetrack.py](src/miis_broadcast/workers/camera_bytetrack.py) | Physical webcam + YOLOX/BYTETracker subject tracking worker (`CameraByteTrackThread`) |
| [src/miis_broadcast/core/models/bytetrack_tracker.py](src/miis_broadcast/core/models/bytetrack_tracker.py) | ByteTrackWrapper — YOLOX inference, BYTETracker association, subject crop extraction |
| [src/miis_broadcast/core/models/livecc_transformers.py](src/miis_broadcast/core/models/livecc_transformers.py) | LiveCCInfer — model loading, streaming inference, KV-cache management |
| [src/miis_broadcast/core/models/openai_tts.py](src/miis_broadcast/core/models/openai_tts.py) | OpenAI Realtime WebSocket TTS engine |
| [src/miis_broadcast/core/models/chatterbox_tts.py](src/miis_broadcast/core/models/chatterbox_tts.py) | Local ChatterBox TTS engine |
| [src/miis_broadcast/core/utils/session_logger.py](src/miis_broadcast/core/utils/session_logger.py) | SessionLogger — handles unified logging of system events and commentary |
| [src/miis_broadcast/core/utils/gpu_telemetry.py](src/miis_broadcast/core/utils/gpu_telemetry.py) | CUDA VRAM snapshot (`torch.cuda.mem_get_info`, `memory_allocated`) for `[Client]` / `[Server]` lines |
| [src/miis_broadcast/core/prompt/prompt_manager.py](src/miis_broadcast/core/prompt/prompt_manager.py) | Loads and builds commentary style prompts from YAML |
| [configs/livecc_prompts.yml](configs/livecc_prompts.yml) | Commentary style definitions |
| [configs/models.yml](configs/models.yml) | Model registry — LiveCC and ByteTrack configs |
| [configs/app.yml](configs/app.yml) | GUI, default model, optional **audience** (LiveKit + HTTP viewer) |
| [src/miis_broadcast/audience/](src/miis_broadcast/audience/) | Second-screen package: token server, LiveKit publisher, static viewer — [README](src/miis_broadcast/audience/README.md) |

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

**Remote inference:** OpenAI TTS and audio playback run on the **client machine** (where you run the GUI) — not on the headless `miis_broadcast.server` host. The server only needs GPU for LiveCC. ByteTrack (YOLOX) always runs on the **client machine** (local GPU or CPU).

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
GEMINI_API_KEY=...    # required for Gemini zh-TW broadcast (always used on Start)
```

`OPENAI_API_KEY` is required when TTS is **OpenAI TTS**. `GEMINI_API_KEY` is required for the Gemini translation layer (runs on the GUI machine even in remote LiveCC mode).

### App config ([configs/app.yml](configs/app.yml))

```yaml
gui_window:
  title: MIISLAB
  min_width: 900
  min_height: 480
  default_open_dir: ./
model:
  classifier_name: livecc_7b   # must match a key in configs/models.yml

# Optional: audience browser + LiveKit (see src/miis_broadcast/audience/README.md)
audience:
  enabled: true
  livekit_url: "ws://127.0.0.1:7880"   # use LAN IP for phones on same Wi-Fi
  api_key: "devkey"
  api_secret: "your_secret_at_least_32_chars"
  room: "broadcast-room"
  port: 8080
```

### Commentary styles ([configs/livecc_prompts.yml](configs/livecc_prompts.yml))

Add or edit styles under the `styles` key. Each style needs:
- `label` — display name shown in the GUI
- `description` — short description
- `query` — the system prompt sent to LiveCC

### ByteTrack config ([configs/models.yml](configs/models.yml))

Required when using **Webcam + Tracking** (`obs_track`):

```yaml
bytetrack:
  bytetrack_repo: "/path/to/ByteTrack_repo"   # or set BYTETRACK_REPO env var
  exp_file: "/path/to/yolox_s_mix_det.py"
  ckpt_path: "/path/to/bytetrack_s_mot17.pth.tar"
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
   - **🌐 Online ▾** — five live sources:
     - **📷 Webcam** — physical webcam, plain stream (auto-skips OBS Virtual Camera)
     - **🎯 Webcam + Tracking** — ByteTrack always runs locally (`CameraByteTrackThread`); only the subject crop is forwarded to the server for LiveCC
     - **🥽 VR (OBS Virtual Camera)** — e.g. Quest Link / capture into OBS, then use OBS Virtual Camera as the device
     - **🎮 VR & Webcam (Sync)** — `1280×480` side-by-side: webcam + OBS Virtual Camera
     - **🔀 Free Switch** — both cameras always-on; switch Webcam / VR / dual without reconnecting (audience second-screen when enabled)
2. **Choose a commentary style** from the dropdown (affects Gemini broadcast tone)
3. **Select TTS** — **OpenAI TTS** (default), **Local TTS** (Chatterbox), or **不啟用 (Mute)**. Audience `narration` requires **OpenAI TTS**.
4. **Click Start Broadcasting** — the model loads on first run (LiveCC-7B takes ~30–60 s on first local use). ByteTrack (YOLOX) preloads in the background so **Webcam + Tracking** is ready without a long stall (local path only).
5. Commentary text appears in the transcript panel and is read aloud in real time (when TTS is not muted)
6. **Click Stop Broadcasting** to end inference (server + client **`[SESSION AVG]`** lines as above; latency stats may still print per your build). In **Webcam + Tracking**, the **camera + ByteTrack overlay keep running** until you pick another input source.

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
| `PREVIEW` | S → C *(optional / legacy)* | `t` | JPEG bytes; **not** sent by the current headless server (preview/overlay is client-side) |
| `SEGMENT` | S → C | `start_t`, `stop_t`, `text` | — |
| `STATUS` | S → C | `msg` | — |
| `ERROR` | S → C | `msg` | — |
| `STOP` | C → S | — | — |
| `PING` / `PONG` | bidirectional | — | — |
| `CLIENT_DIAG` | C → S | `rss_mib`, `jpeg_q_used`, `jpeg_q_max`, `sys_ram_pct`; optional `gpu_vram_used_mib`, `gpu_vram_total_mib`, `gpu_torch_alloc_mib` (CUDA device **0**, when available) | — |

### Memory telemetry (remote inference)

When inference runs on a **remote** server (file, webcam, OBS, dual sync, free switch, or `obs_track` with TCP), **`[Client]`** and **`[Server]`** lines refer to **two different machines**. They are printed on the **server** terminal (the process that runs `miis_broadcast.server`).

| Printed line (server **stdout**) | Meaning |
|---|---|
| `[ByteTrack] Frame … \| Infer FPS … \| Wall FPS …` | Only when **ByteTrack runs in this process** (GUI / local workers). The headless remote server does **not** load ByteTrack. |
| `[Server] RSS=… \| livecc_buffer=… \| system_RAM_used=…% \| GPU_VRAM=…` | Same as before, plus **CUDA device 0** VRAM: global used/total MiB (from `torch.cuda.mem_get_info`), **%**, and `torch_alloc` = PyTorch allocator bytes for **this process** on that device. `GPU_VRAM=n/a` if CUDA is unavailable. |
| `[Client] RSS=… \| JPEG send_queue=… \| system_RAM_used=…% \| GPU_VRAM=…` | Same for the **sender** machine; GPU fields come from the GUI’s snapshot (also device **0**). Printed on **server** stdout from `CLIENT_DIAG`; session log file may duplicate under `[Memory]`. |

**Tuning:** use **`[Server]`** alongside **`[Client]`** on the **same** terminal to see whether the bottleneck is decode/LiveCC on the host or encode/TCP on the sender.

The `CLIENT_DIAG` payload is a short JSON message (order of **hundreds of bytes** every ~2 s). Control sends hold the client socket lock only for that small `sendall`.

### Session averages `[SESSION AVG]` (on Stop Broadcasting)

Printed **once** when **`MSG_STOP`** completes (means of snapshots collected during that broadcast):

| Printed line | Where |
|---|---|
| `[SESSION AVG] Server (n=…) …` | **Inference server** stdout (`session.py`) |
| `[SESSION AVG] Client (n=…) …` | **Same server terminal** — mean of **`[Client]`** rows received from `CLIENT_DIAG` |
| `[SESSION AVG] Client-local (n=…) …` | **Client / GUI** stdout — local CLIENT_DIAG aggregates (`gui.py`) |
| `[SESSION AVG] [MEDIA]` / `[SESSION AVG] [AUDIO]` … | **Client** stdout when the audience publisher ran (**Free Switch** + `audience.enabled`) (`livekit_publisher.py`) |
| `[SESSION AVG] ByteTrack (n=…) …` | **Client** stdout in **Webcam + Tracking** (`bytetrack_tracker.py`) |

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
`dual_sync`, `free_switch`).

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
[HH:MM:SS] [Memory] [INFO] [Client] RSS=… MiB | JPEG send_queue=…/… | system_RAM_used=…% | GPU_VRAM=… or n/a …
```

| Line type | Meaning |
|-----------|---------|
| `[GUI] [INFO]` | System events from the GUI (start, stop, remote connection changes, errors) |
| `[Memory] [INFO]` | **Remote inference only:** sender PC metrics (same as server stdout **`[Client]`**), including **`GPU_VRAM`** when CUDA is available on the sender — written to the **session file** only, not the GUI console. Server RSS and server GPU stay on the host **`[Server]`** lines. |
| `[COMMENTARY]` | Every segment of AI commentary as it arrives (English from remote/local LiveCC, or zh-TW after Gemini on the client) |
| `Inference: remote` | LiveCC ran on the remote server (thin-client mode) |
| `Inference: local` | LiveCC ran on the local GPU |

All input modes share the same file layout. The `Input Mode:` header and `[COMMENTARY]`
content differ per session.

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
│   │   ├── models/           # LiveCC, Gemini broadcaster, OpenAI TTS, Chatterbox, ByteTrack
│   │   ├── prompt/           # Prompt management
│   │   └── utils/            # Config, session_logger, latency monitor, gpu_telemetry
│   ├── workers/              # LiveCC, Gemini, TTS, camera/OBS/dual/free_switch, ByteTrack
│   ├── widgets/              # Custom Qt widgets
│   └── audience/             # LiveKit publisher, token server, browser viewer
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) (this file) | Install, usage, remote server, configuration |
| [src/miis_broadcast/workers/README.md](src/miis_broadcast/workers/README.md) | Input sources, frame signals, server log reference |
| [src/miis_broadcast/audience/README.md](src/miis_broadcast/audience/README.md) | LiveKit second-screen setup and troubleshooting |

---

## License

Internal research project — MIISLab.
