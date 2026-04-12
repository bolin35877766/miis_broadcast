# MIIS Broadcast

A real-time AI sports broadcasting commentary system with a desktop GUI. It ingests video from a file or live camera, generates commentary using the **LiveCC-7B** streaming video captioner, and reads it aloud via a TTS engine — all with sub-second end-to-end latency.

---

## Features

- **Three input modes**:
  - Video file playback
  - Live camera feed
  - OBS Virtual Camera + ByteTrack subject tracking
- **Session Logging**: All terminal logs and AI-generated commentary (TTS output) are automatically saved to a unified log file in `logs/sessions/` for each broadcast session.
- **Real-time commentary generation** via [LiveCC-7B-Instruct](https://huggingface.co/chenjoya/LiveCC-7B-Instruct) (Qwen2VL-based)
- **Subject-aware tracking mode** — YOLOX + BYTETracker locks onto the dominant person, crops and forwards their region to LiveCC; switches automatically when a larger subject enters the frame
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

```
Video File  ──────────────────────────► VideoThread
                                              │
Live Camera ─────────────────────────► CameraThread
                                              │
OBS Virtual Camera ─► OBSByteTrackThread      │
                       (YOLOX + BYTETracker)   │
                       subject crop (640×480) ─┘
                                              │
                                              ▼
                             LiveCCWorker / LiveCCCameraWorker
                             (LiveCC-7B-Instruct, GPU inference)
                                              │ commentary text
                                              ▼
                             TTS Engine (OpenAI Realtime or ChatterBox)
                                              │ PCM audio
                                              ▼
                                         ffplay (audio output)
```

Key modules:

| Path | Role |
|---|---|
| [src/miis_broadcast/gui.py](src/miis_broadcast/gui.py) | Main window, video panel, control widgets |
| [src/miis_broadcast/workers/livecc.py](src/miis_broadcast/workers/livecc.py) | QThread workers for LiveCC inference (file & camera) |
| [src/miis_broadcast/workers/obs_bytetrack.py](src/miis_broadcast/workers/obs_bytetrack.py) | OBS Virtual Camera + YOLOX/BYTETracker subject tracking worker |
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
- `ffplay` (from FFmpeg) — required for audio output
- OpenAI API key (if using the OpenAI TTS backend)
- [ByteTrack_repo](https://github.com/ifzhang/ByteTrack) — required for OBS + tracking mode (set path in `configs/models.yml` or via `BYTETRACK_REPO` env var)

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

1. **Select input** — choose one of three modes:
   - **Video file** — open a local video file
   - **Camera** — use a connected webcam
   - **OBS + 追蹤** — use OBS Virtual Camera with YOLOX + BYTETracker subject tracking
2. **Choose a commentary style** from the dropdown
3. **Select TTS backend** — OpenAI Realtime or ChatterBox Local
4. **Click Start** — the model loads on first run (LiveCC-7B takes ~30–60 s to load)
5. Commentary text appears in the transcript panel and is read aloud in real time
6. **Click Stop** to end inference; latency statistics are printed to the console

#### OBS + 追蹤 mode behaviour

- YOLOX detects all people in the frame each tick; BYTETracker assigns persistent IDs
- The subject with the largest weighted score (area / centre distance) is locked as the primary subject
- The subject's bounding box is padded by 15% and cropped, then resized to 640×480 before being forwarded to LiveCC
- If a new person enters the frame with more than 4× the current subject's area, tracking switches automatically
- When no valid subject is detected, frames are not forwarded to LiveCC (prevents empty-scene descriptions)

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
│   ├── models.yml            # Model registry
│   └── livecc_prompts.yml    # Commentary style prompts
├── logs/
│   └── app_error.log
├── src/miis_broadcast/
│   ├── app.py                # Entry point
│   ├── gui.py                # Main window
│   ├── core/
│   │   ├── io/               # Input helpers (camera, OBS virtual camera)
│   │   ├── models/           # LiveCC, OpenAI TTS, ChatterBox TTS, ByteTrackWrapper
│   │   ├── prompt/           # Prompt management
│   │   └── utils/            # Config, formatting, latency monitor
│   ├── widgets/              # Custom Qt widgets
│   └── workers/              # QThread workers (LiveCC, TTS, input, OBS+ByteTrack)
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

---

## License

Internal research project — MIISLab.
