# MIIS Broadcast — Architecture Reference

> **Purpose of this file**: AI-first architecture map. Read this before touching any source file.  
> Last updated: 2026-06-13

---

## 1. What This System Does

Real-time sports broadcast commentary generator. It:

1. Reads video (file or live camera — 5 camera source modes)
2. Runs a vision-language model (LiveCC / Qwen2VL-7B) to convert video clips → text descriptions — either **locally** or via a **remote inference server** over TCP
3. Routes those descriptions through a fast-slow blade:
   - P1/P2 keyword hits → direct TTS interrupt
   - P3 → `GeminiWorker` (per-segment enrichment) and/or `GeminiBackgroundWorker` (continuous commentary)
4. Speaks the broadcast text via TTS (OpenAI Realtime WebSocket, Gemini TTS via ffplay, or local Chatterbox)
5. Displays live subtitles in a PySide6 GUI
6. Optionally streams VR video + narration audio to a browser-based audience second-screen via LiveKit WebRTC
7. Optionally records TTS audio output to a timestamped WAV file

---

## 2. Directory Map

```
miis_broadcast/
├── src/miis_broadcast/          # installable Python package (source of truth)
│   ├── app.py                   # entry point: pre-init Gemini, QApplication, MainWindow
│   ├── gui.py                   # MainWindow + all Qt UI, signal wiring, all worker init (3325 lines)
│   │
│   ├── workers/                 # Qt QObject workers (each runs in its own QThread)
│   │   ├── livecc.py            # LiveCCWorker (file mode), LiveCCCameraWorker (camera mode)
│   │   ├── gemini.py            # GeminiWorker + GeminiBackgroundWorker
│   │   ├── openai_tts.py        # OpenAITTSWorker — WebSocket to gpt-realtime
│   │   ├── gemini_tts.py        # GeminiTTSWorker — Gemini AUDIO modality TTS via ffplay
│   │   ├── chatterbox_tts.py    # ChatterboxTTSWorker — local Chatterbox TTS (lazy-loaded)
│   │   ├── obs_input.py         # OBSCameraThread — OBS Virtual Camera source
│   │   ├── camera_bytetrack.py  # CameraByteTrackThread — webcam + YOLOX+ByteTrack tracking
│   │   ├── dual_source.py       # DualSourceCameraThread — webcam + OBS side-by-side
│   │   ├── free_switch.py       # FreeSwitchCameraThread — both cameras always-on, click-switch
│   │   └── input.py             # Legacy BaseWorkerThread / VideoWorkerThread
│   │
│   ├── core/
│   │   ├── models/
│   │   │   ├── livecc_transformers.py   # LiveCCInfer: Qwen2VL inference, KV cache, hallucination filter
│   │   │   ├── gemini_broadcaster.py    # Gemini API calls, RAG retriever, stream parsing
│   │   │   ├── openai_tts.py            # TTS queue, enqueue_tts_text(), recording sink
│   │   │   ├── gemini_tts.py            # Gemini TTS singleton: enqueue, interrupt, ffplay playback
│   │   │   ├── chatterbox_tts.py        # Chatterbox model wrapper
│   │   │   └── bytetrack_tracker.py     # ByteTrackWrapper (YOLOX+ByteTrack person tracking)
│   │   ├── io/
│   │   │   ├── input.py                 # Generic video input abstraction
│   │   │   └── obs_input.py             # OBS-specific input
│   │   ├── prompt/
│   │   │   └── prompt_manager.py        # Loads livecc_prompts.yml, style list + query
│   │   ├── match_tracker.py             # Thread-safe singleton: score, period, last_event
│   │   └── utils/
│   │       ├── config.py                # parse_configs(), load YAML configs
│   │       ├── audio_recorder.py        # AudioRecorder — WAV sink for TTS recording feature
│   │       ├── gpu_telemetry.py         # cuda_vram_snapshot() via PyTorch (no nvidia-smi)
│   │       ├── session_logger.py
│   │       ├── latency_monitor.py
│   │       ├── vram_monitor.py          # Legacy VRAM monitor (superseded by gpu_telemetry.py)
│   │       ├── format_output.py
│   │       ├── format_string.py
│   │       └── counter.py
│   │
│   ├── network/
│   │   ├── client.py            # SocketClientRunner (QThread TCP client for remote inference)
│   │   └── protocol.py          # Binary wire protocol: pack_message / read_message
│   │
│   ├── server/
│   │   ├── __main__.py          # Server entry point: load LiveCC once, accept TCP clients
│   │   └── session.py           # ClientSession: per-client frame decode + inference loop
│   │
│   ├── audience/
│   │   ├── livekit_publisher.py # AudiencePublisher: LiveKit WebRTC video+audio publisher
│   │   └── token_server.py      # AudienceTokenServer: FastAPI HTTP /audience + /api/audience/join
│   │
│   └── widgets/
│       ├── text_output.py       # TextOutputWidget: scrolling subtitle display
│       └── file_explorer.py
│
├── configs/
│   ├── app.yml                  # GUI settings, TTS params, Gemini params, remote, audience
│   ├── models.yml               # LiveCC classifier config (fps, tokens, generation params)
│   ├── livecc_prompts.yml       # Broadcast styles (labels for UI) + livecc_query string (v3)
│   └── system_prompts.yml       # Gemini system prompts per style (4 styles)
├── build/                       # pip build artifacts — NOT source of truth, ignore for dev
├── log/                         # runtime logs: livecc_output.log, gemini_output.log
└── logs/                        # app_error.log, vram_tracker.log, recordings/
```

---

## 3. Full Data Pipeline

### 3A. File Mode

```
VideoThread (QThread)
  └─ cv2.VideoCapture → signal_frame(frame_rgb, frame_idx, fps)
       └─ MainWindow.on_video_frame()
            ├─ updates VideoPanel display + _playback_sec
            └─ if remote connected: SocketClientRunner.send_frame(frame_bgr, _playback_sec)

LiveCCWorker (QThread)                       [runs ahead of playback; local model only]
  └─ LiveCCInfer.live_cc(query, state)
       ├─ get_smart_resized_video_reader()   livecc_utils
       ├─ get_smart_resized_clip()           livecc_utils — 4 frames/clip at 3.0 fps
       ├─ Qwen2VL.generate()                 local GPU (cuda:0), bfloat16
       │    └─ KV cache (past_key_values, past_ids) — maintained across clips
       ├─ _is_degenerate()                   hallucination filter (hard-coded phrase list)
       └─ signal_segment(start_t, stop_t, parsed_dict)

SocketClientRunner (QThread)                 [remote path: replaces LiveCCWorker]
  └─ recv MSG_SEGMENT(start_t, stop_t, text) from server
       └─ signal_segment → MainWindow.on_remote_segment() → _on_segment_impl()

MainWindow._route_segment()                  [fast-slow blade; local path only]
  ├─ P1 keyword → flush_and_abort() + enqueue_front() + TTS interrupt
  ├─ P2 keyword → enqueue_front()
  └─ P3 (normal)
       ├─ signal_livecc_context → GeminiBackgroundWorker.update_context()
       └─ if tts_remaining ≤ 2.0s: _signal_to_gemini.emit() → GeminiWorker.process_segment()

GeminiWorker (QThread)
  └─ stream_gemini(event_data)
       ├─ RAG, MatchTracker, Gemini API (streaming)
       └─ parses "P<1-5>: text\nLABEL: xxx"
            ├─ signal_priority(start_t, stop_t, priority, should_speak)
            └─ signal_broadcast(start_t, stop_t, result_dict)

GeminiBackgroundWorker (QThread)             [independent continuous loop]
  └─ polls tts_remaining every 200ms
       └─ if remaining < 1.0s AND ≥4s since last fire:
            stream_gemini(context_pool[-3:])
            → signal_broadcast(t_now, t_now, result) with result["_background"]=True

MainWindow.on_segment() / _on_segment_impl()
  ├─ FILE MODE: push to _pending_segments deque; _subtitle_timer drains when _playback_sec >= start_t
  ├─ dedup check (word overlap ≥ 0.85 within 15s window)
  ├─ priority guard (_tts_protect_until blocks lower priority for 6s / 3s after P1 / P2)
  └─ signal_gemini_tts_speak / signal_tts_speak / signal_local_tts_speak

GeminiTTSWorker / OpenAITTSWorker / ChatterboxTTSWorker (QThread)
  └─ audio output → speakers + optional recording sink → logs/recordings/*.wav
```

### 3B. Camera Modes

Five camera source modes, all feeding the same `on_camera_frame()` → remote-or-local pipeline:

| Mode key | Camera Thread | Description |
|----------|--------------|-------------|
| `"camera"` | `CameraThread` (webcam) | Single webcam, no tracking |
| `"obs"` | `OBSCameraThread` | OBS Virtual Camera (VR capture card or screen) |
| `"obs_track"` | `CameraByteTrackThread` | Webcam + YOLOX+ByteTrack → annotated frame + subject crop |
| `"dual_sync"` | `DualSourceCameraThread` | Webcam (idx 0) + OBS (idx 5) synchronized via `cap.grab()` → 1280×480 side-by-side |
| `"free_switch"` | `FreeSwitchCameraThread` | Both cameras always open; click-switch webcam / VR / dual; VR at native 1920×1080 for audience |

```
Camera Thread (any mode above)
  └─ signal_frame(frame_rgb)
       └─ MainWindow.on_camera_frame()
            ├─ updates VideoPanel
            ├─ if remote connected: SocketClientRunner.send_frame(frame_bgr, t_relative)
            └─ elif local: LiveCCCameraWorker.push_frame(frame_bgr, t_relative)

[obs_track only] CameraByteTrackThread
  └─ signal_subject_frame(crop_rgb)
       └─ MainWindow.on_obs_track_subject_frame()
            ├─ if remote: SocketClientRunner.send_frame(subject_bgr, t)
            └─ elif local: cam_worker.push_frame(subject_bgr, t)

[free_switch only] FreeSwitchCameraThread
  └─ signal_vr_frame(frame_rgb native 1920×1080)
       └─ MainWindow._deliver_audience_vr_frame()
            └─ AudiencePublisher.push_video_frame()

LiveCCCameraWorker (QThread)               [local model only]
  └─ buffer: deque[FrameItem(t, frame)] maxlen=180
     build_clip_from_buffer(window_sec=1.5, target_fps=2.0) every 1.0s
     LiveCCInfer.live_cc_from_frames() → signal_segment → _route_segment (same as file mode)
```

### 3C. Remote Inference Path

```
                   GUI (client)                              Server
                   ─────────────────────────────────────────────────────────────
Start click   →   SocketClientRunner.start_inference(mode, query)
                     ↓ MSG_START
Video frames  →   SocketClientRunner.send_frame(jpeg, t)
                     ↓ MSG_FRAME (binary JPEG + timestamp JSON)
                                                  ClientSession._frame_processor_loop
                                                    JPEG decode → deque buffer
                                                  ClientSession._inference_loop (every 2s)
                                                    build_clip_from_buffer()
                                                    LiveCCInfer.live_cc_from_frames()
                                                    duplicate filter (SequenceMatcher ≥0.86)
                                                    ↓ MSG_SEGMENT (start_t, stop_t, text)
                  signal_segment(start_t, stop_t, text)
                  → MainWindow.on_remote_segment()
                  → _on_segment_impl()         [no _route_segment; segments sent directly]
                     → TTS / subtitles / dedup / priority guard (same as local path)
```

Remote segments skip `_route_segment()` (no Gemini enrichment, no fast-slow blade) — the server returns pre-formed text that goes straight to TTS.

---

## 4. Threading Model

| Thread | Worker | Notes |
|--------|--------|-------|
| Main (GUI) | MainWindow | Qt event loop; all signal wiring |
| `livecc_thread` | `LiveCCWorker` | Blocks on GPU; file mode, local model only |
| `cam_worker_thread` | `LiveCCCameraWorker` | Blocks on GPU; camera modes, local model only |
| `gemini_thread` | `GeminiWorker` | Blocks on HTTP; P3 per-segment enrichment |
| `gemini_bg_thread` | `GeminiBackgroundWorker` | Continuous background commentary; polls TTS backlog |
| `tts_thread` | `OpenAITTSWorker` | WebSocket to gpt-realtime; eager-started |
| `gemini_tts_thread` | `GeminiTTSWorker` | Gemini AUDIO API + ffplay; lazy-started |
| `local_tts_thread` | `ChatterboxTTSWorker` | Local model; lazy-loaded on first "local" selection |
| `video_thread` | `VideoThread` | cv2 decode loop; file mode |
| `obs_thread` | `OBSCameraThread` | OBS Virtual Camera; camera "obs" mode |
| `obs_bytetrack_thread` | `CameraByteTrackThread` | Webcam + YOLOX+ByteTrack; "obs_track" mode |
| `dual_sync_thread` | `DualSourceCameraThread` | Dual webcam+OBS sync; "dual_sync" mode |
| `free_switch_thread` | `FreeSwitchCameraThread` | Both cameras always-on; "free_switch" mode |
| `SocketClientRunner` | — | QThread; TCP receive loop; spawns daemon `_frame_sender_loop` |
| `AudiencePublisher` | — | QThread daemon + asyncio; VR video+audio → LiveKit |
| `AudienceTokenServer` | — | Daemon thread + uvicorn; HTTP /audience; app lifetime |
| `_bytetrack_preload_thread` | — | Background thread; preloads ByteTrackWrapper 500ms after startup |

**Signal safety rule**: Workers only receive data via Qt Signals (QueuedConnection). P1/P2 fast-path calls (`flush_and_abort`, `enqueue_front`) are CPython-GIL-safe deque ops — no explicit lock needed.

**Gemini TTS core threads** (inside `core/models/gemini_tts.py`):
- `GeminiTTSWorker` thread: drain `_text_queue`, call `generate_content`, prefetch next utterance via `ThreadPoolExecutor(max_workers=2)`, play via `ffplay`
- `_frame_sender_loop` inside `SocketClientRunner`: daemon thread draining JPEG queue → `sendall()`

---

## 5. Key Classes

### `LiveCCInfer` (`core/models/livecc_transformers.py`)

- **Model**: `Qwen2VLForConditionalGeneration`, HuggingFace, loaded once (~20–40s)
- **Inference state dict**: `video_path`, `past_ids`, `past_key_values`, `video_pts`, `last_timestamp`, `mm_window_start`, `carry_text`, `recent_texts`, `video_end`
- **KV cache management**: `truncate_state_by_budget()` trims tokens when approaching `ctx_max=32768`. `_apply_mm_window_policy` clears KV cache every `mm_window_sec` (6.0s default) and injects `carry_text` (last K sentences at 120 chars max) into next clip.
- **Hallucination filter** (`_is_degenerate`): rejects empty, <2-word, >80-word, prompt-leaking, and hallucination-phrase outputs. Silent `continue` in worker loop — no `signal_segment` emitted, causing apparent timestamp "gaps" (next visible segment jumps 2× `streaming_time_interval`). This is intentional noise filtering.
- **Output**: raw text → `_parse_visual_json()` → `{"event": str, "urgency": int, "metadata": {"raw": str}}`

### `GeminiWorker` (`workers/gemini.py`)

- **Queue design**: `_task_deque` with P1/P2 fast-path ops from GUI thread:
  - P1: `flush_and_abort()` (clears deque + sets `_abort_current`) then `enqueue_front()`
  - P2: `enqueue_front()`
  - P3: `process_segment()` Slot via QueuedConnection (tail-append)
- **Signals**: `signal_priority(start_t, stop_t, priority, should_speak)` fires first; `signal_broadcast(start_t, stop_t, result_dict)` fires immediately after
- **Output format** (Gemini instructed): `P<1-5>: <text ≤12 words>\nLABEL: <snake_case>`

### `GeminiBackgroundWorker` (`workers/gemini.py`)

- **Purpose**: Continuous background commentary independent of per-segment Gemini calls. Fires when TTS backlog is nearly empty, ensuring a steady narration stream during low-action periods.
- **Context pool**: `deque(maxlen=3)` — rolling window of last 3 LiveCC P3 descriptions, updated via `update_context(description)` slot
- **Fire condition** (every poll at 200ms): `tts_remaining < WATERMARK_SEC (1.0)` AND `now - last_fire > MIN_FIRE_INTERVAL_SEC (4.0)`
- **Inter-sentence gap**: 500ms sleep after each successful fire
- **Result tag**: `result["_background"] = True` — tells `_on_segment_impl` to skip `_pending_segments` scheduling (no video-relative timestamp anchor; `start_t = stop_t = t_now`)
- **Pause/resume**: P1 interrupt calls `pause()` (sets `_paused=True`, aborts in-flight stream); `_on_tts_done()` after P1's 1.0s silence calls `resume()`

### `MatchTracker` (`core/match_tracker.py`)

- Module-level singleton. Tracks `red_score`, `blue_score`, `period`, `last_event`
- `get_state_string()` injected into every Gemini prompt
- Updated by `on_segment()` when `action_label.startswith("score_")`

### `SocketClientRunner` (`network/client.py`)

- **Transport**: Raw TCP over IPv4, custom binary protocol (4B total_len + 4B json_len + JSON + binary payload)
- **Run loop**: QThread blocking on `read_message()`; dispatches MSG_SEGMENT → `signal_segment`
- **Frame sending**: non-blocking enqueue to `_frame_queue(maxsize=30)`; dedicated `_frame_sender_loop` daemon drains to socket
- **Key message types**: MSG_START, MSG_FRAME (JPEG + timestamp), MSG_SEGMENT (start_t, stop_t, text), MSG_STOP, MSG_CLIENT_DIAG (RAM/GPU telemetry), MSG_PING/PONG

### `AudiencePublisher` (`audience/livekit_publisher.py`)

- Runs asyncio event loop in a QThread daemon
- Two async tasks: `_video_pump_vr()` (30 fps, 1920×1080) and `_audio_pump_direct()` (24 kHz mono PCM)
- Frame queues: `_video_q(maxsize=3)`, `_audio_q(maxsize=48)` — drops oldest when full
- VR resize handled in `ThreadPoolExecutor(max_workers=2, name="audience-vr-*")`
- Activated only in `free_switch` mode; audio PCM injected via `openai_tts.register_pcm_sink()`

---

## 6. Configuration Files

### `configs/app.yml`

```yaml
gui_window:
  title: MIISLAB
  min_width: 900
  min_height: 480
  default_open_dir: ./

model:
  classifier_name: livecc_7b    # key in models.yml

openai_tts:
  model_url: wss://api.openai.com/v1/realtime?model=gpt-realtime
  default_voice: coral
  default_speed: 1.5
  temperature: 0.7

gemini_tts:
  model_name: models/gemini-3.1-flash-tts-preview
  voice: Kore

chatterbox_tts:
  audio_prompt_path: ...        # reference WAV for voice cloning
  temperature: 0.7
  cfg_weight: 1.0
  chunk_size: 25

gemini:
  model_name: gemini-3.1-flash-lite-preview
  rag_threshold: 600            # chars; below this, full context used verbatim
  temperature: 0.7
  max_output_tokens: 60

inference:
  dedup_window_s: 15.0          # suppress near-identical TTS within this window
  dedup_threshold: 0.85         # word overlap ratio for dedup
  max_pending: 400              # max pending_segments (file mode)

remote:
  enabled: true
  client_only: false            # true → skip local Qwen2VL load; use remote server only
  host: 127.0.0.1
  port: 9000

audience:
  enabled: true
  livekit_url: ws://10.75.212.31:7880
  api_key: devkey
  api_secret: ...               # LiveKit API secret (≥32 chars)
  room: broadcast-room
  port: 8080                    # HTTP port for /audience viewer + token server
```

### `configs/models.yml`

```yaml
classifiers:
  livecc_7b:
    model_path: chenjoya/LiveCC-7B-Instruct
    fps: 3.0
    initial_fps_frames: 4
    streaming_fps_frames: 4     # 4 frames/clip → streaming_time_interval ≈ 1.33s
    max_new_tokens: 26
    mm_window_sec: 6.0          # KV cache reset interval
    carry_text_max_chars: 120
    carry_recent_k: 2
    generation:
      temperature: 0.5
      top_p: 0.9
      top_k: 30
      repetition_penalty: 1.15
      no_repeat_ngram_size: 3
    camera:
      window_sec: 1.5           # seconds of frames per clip
      target_fps: 2.0
      infer_interval: 1.0       # inference cadence (seconds)
      memory_reset_every: 5     # hard KV reset every N inferences
```

### `configs/livecc_prompts.yml`

Version 3 — primary query assumes full-frame capture (person + VR game view visible together):

- `livecc_query`: present-tense single sentence about ball/player action (3rd person)
- `livecc_query_splitscreen`: fallback for uncropped left-right split frames (LEFT=person wearing headset, RIGHT=VR game — describe RIGHT half)
- `default_style`: `objective`
- `styles`: `objective`, `hype`, `calm`, `trash_talk` — display labels + descriptions for GUI dropdown

### `configs/system_prompts.yml`

Contains per-style Gemini system prompts:

```yaml
gemini_broadcaster:
  objective: "..."    # Professional, concise
  hype: "..."         # High energy
  calm: "..."         # Technical analysis
  trash_talk: "..."   # Fast-paced commentary
```

Each style instructs Gemini to output exactly 2 lines per response:
```
P<1-5>: <broadcast text, max 12 words, English only>
LABEL: <snake_case_action_label>
```

Priority 1=critical event, 2=high, 3=medium, 4=low, 5=no active play.

---

## 7. Priority & Interrupt System

### Priority levels

| Priority | Meaning | TTS behavior |
|----------|---------|--------------|
| P1 | Critical event (score, goal) | Immediately interrupts current TTS + pauses GeminiBackgroundWorker |
| P2 | Important event (timeout, steal) | Queues at front of TTS after current sentence |
| P3 | Normal commentary | Gated by TTS backpressure watermark (2.0s) |
| P4 | Low urgency | No interrupt |
| P5 | No active play | `should_speak=False` — Gemini result discarded |

### Fast-slow blade routing constants (`gui.py`)

```python
_P3_BACKPRESSURE_WATERMARK_SEC = 2.0     # P3 → Gemini only when backlog ≤ 2.0s
_PRIORITY_DECAY_INTERVAL_SEC   = 2.0     # effective priority decays after 2s inactivity
_TTS_PROTECT_WINDOW_SEC        = {1: 6.0, 2: 3.0}   # protection window after P1/P2
_FAST_BLADE_DEDUP_WINDOW_S     = 5.0     # suppress duplicate P1/P2 texts within 5s
```

### `_route_segment()` logic (fast-slow blade)

1. **Keyword scan** (`_scan_priority`):
   - P1 keywords (`score, goal, dunk, 進球, 得分, ...`): `flush_and_abort()` (clear Gemini queue + abort in-flight) + `enqueue_front()` + TTS interrupt + prefix "Oh wait!—" if TTS was active + `GeminiBackgroundWorker.pause()`
   - P2 keywords (`pass, block, 傳球, 封蓋, ...`): `enqueue_front()` only
2. **P3 normal path**:
   - Always: `signal_livecc_context.emit(description)` → `GeminiBackgroundWorker.update_context()` (context pool update, maxlen=3)
   - Conditionally (if `_get_active_tts_remaining_sec() ≤ 2.0s`): `_signal_to_gemini.emit()` → `GeminiWorker.process_segment()` for per-segment enrichment
3. **Fragment stitching**: Segments ending in `"..."` are buffered and re-scanned on the next arrival to avoid sending incomplete thoughts.

### Interrupt matrix

```python
_INTERRUPT_MATRIX = {1: 2, 2: 3, 3: 4}
# new_priority can interrupt if _tts_last_priority >= _INTERRUPT_MATRIX[new_priority]
# i.e.: P1 interrupts if current ≥ P2; P2 interrupts if current ≥ P3; P3 if current ≥ P4
```

Transition phrases prepended when interrupting: `"Oh wait!—"` (P1), `"And—"` (P2).

### P1 post-interrupt silence

`_on_tts_done()` after a P1 interrupt: `QTimer.singleShot(1000ms)` → silence → `GeminiBackgroundWorker.resume()`.

### P1 confirmed → KV cache reset

`signal_priority` → `_on_gemini_priority()`: if P1, `signal_p1_confirmed` → `LiveCCCameraWorker.requestMemoryReset()`. Clears inference state dict in camera mode.

---

## 8. State Management

### LiveCC inference state (per run)

| Key | Type | Purpose |
|-----|------|---------|
| `past_ids` | Tensor `[1, seq]` | KV history token ids |
| `past_key_values` | tuple[tuple[Tensor]] | Transformer KV cache |
| `video_pts` | Tensor | all frame presentation timestamps |
| `last_timestamp` | float | last processed video second |
| `mm_window_start` | float | start of current KV cache window |
| `carry_text` | str | injected after KV reset (max 120 chars) |
| `recent_texts` | list[str] | last K LiveCC responses (source for carry_text) |
| `video_end` | bool | sentinel to break inference loop |

### Game context + RAG

- Set via GUI "載入比賽資訊" → `load_game_context_file(path)` → `set_game_context(text)`
- Short context (≤600 chars): injected verbatim into every Gemini prompt
- Long context (>600 chars): `_ContextRetriever` (TF-IDF, top-3 chunks) retrieves relevant portion per visual query

### Match score

`MatchTracker` singleton updated when Gemini `action_label` starts with `"score_"`. State string injected into every Gemini prompt.

---

## 9. TTS Backends

### Shared interface

All three TTS workers expose identical Qt signals/slots:
- `speak(text, priority, ref_ts, start_t)` — enqueue utterance
- `interrupt()` — cancel current + clear queue
- `start()` / `stop()`
- `apply_settings(voice, speed)`
- `signal_tts_done` — emitted only on **natural** completion (never on interrupt)
- `get_queue_remaining_sec()` — backpressure estimate (see below)

### TTS queue architecture (`core/models/gemini_tts.py`, `openai_tts.py`)

- **`_text_queue`**: `queue.Queue`, bounded FIFO with `_MAX_QUEUE_DEPTH = 2`
- **`enqueue_tts_text(text, drop_outdated, priority, ref_ts, start_t)`**:
  - `drop_outdated=True` (P1/P2): `clear_text_queue()` — wipes all pending
  - `drop_outdated=False` (P3-P5): `_trim_text_queue(_MAX_QUEUE_DEPTH)` — drops **oldest** until room exists, then enqueues. This is a depth-bounded FIFO — continuous broadcast keeps flowing rather than being silently discarded by the next arrival.

### Backpressure estimation: virtual fluid level (`workers/gemini_tts.py`, `workers/openai_tts.py`)

`get_queue_remaining_sec()` returns a **cumulative** estimate of all backlog audio (queued + playing), not just the most-recently-enqueued item:

```python
def _decay_locked(self):
    elapsed = time.time() - self._last_decay_wall
    self._queue_remaining_sec = max(0.0, self._queue_remaining_sec - elapsed)
    self._last_decay_wall = time.time()

def speak(self, text, priority, ...):
    est = _estimate_tts_duration(text)      # CJK-aware: CJK/4 + words/2.5 seconds
    with lock:
        _decay_locked()
        if priority <= 2:                   # P1/P2: replace (matches clear_text_queue)
            _queue_remaining_sec = est
        else:                               # P3-P5: accumulate (matches bounded FIFO)
            _queue_remaining_sec += est
```

`interrupt()` resets `_queue_remaining_sec = 0.0` and `_last_decay_wall = 0.0`.

Both `GeminiBackgroundWorker` (watermark 1.0s) and `_route_segment` P3 gate (watermark 2.0s) read this value via `_get_active_tts_remaining_sec()`, which dispatches to the currently-active TTS backend based on `tts_mode`.

### OpenAI Realtime TTS (`workers/openai_tts.py`)

- Protocol: WebSocket (`gpt-realtime`, GA model)
- Sends text → receives base64 audio delta chunks → plays via sounddevice
- **Warm-up**: `warmup_connection()` opens WebSocket before first segment (called on Start click) to eliminate first-utterance latency
- `register_pcm_sink(callback, mute_local, flush_callback)`: optional PCM tap for audience second-screen + recording

### Gemini TTS (`workers/gemini_tts.py`, `core/models/gemini_tts.py`)

- Protocol: `google.genai` `generate_content` with `response_modalities=["AUDIO"]`
- Playback: raw PCM16 streamed to `ffplay` subprocess (1:1 wall clock, interruptible via `_soft_stop_proc`)
- **Prefetch**: `_tts_executor (ThreadPoolExecutor, max_workers=2)` generates next utterance's audio while current plays, eliminating the generate→play gap
- **Fade**: 15ms linear fade-in/out on every utterance via NumPy (prevents click/pop at ffplay process boundaries)
- `register_recording_sink(callback)`: PCM bytes forwarded to `AudioRecorder.write_chunk()`

### Chatterbox (local) TTS (`workers/chatterbox_tts.py`)

- Model loaded lazily on first "Local TTS" selection
- Reference voice WAV for voice cloning (`audio_prompt_path`)
- `speak(text)` / `interrupt()`
- Also supports `register_recording_sink()`

### Recording feature

`AudioRecorder` (`core/utils/audio_recorder.py`):
- Enabled by "錄音 (Record Audio)" checkbox in ControlPanel (default: unchecked)
- On Start: `_audio_recorder.start(mode)` creates `logs/recordings/{mode}_{timestamp}.wav` (24 kHz, mono, int16)
- All TTS backends forward PCM to `write_chunk()` while active; silence-pads gaps to match wall-clock elapsed time
- On Stop: `stop()` pads trailing silence, closes file, prints path

---

## 10. GUI Components

### `MainWindow` (`gui.py`)

- Subclasses `QMainWindow`
- Creates and owns all workers and threads
- Central routing: `_route_segment()` (LiveCC output → fast-path → Gemini), `_on_segment_impl()` (TTS dispatch)
- `_tick_subtitle_scheduler()`: 50ms QTimer syncs subtitle display to video playback position (file mode)

### `VideoPanel` (`gui.py`)

- `label_video`: QLabel frame display with aspect-ratio-preserving scaling
- `slider`: QSlider seek (disabled until video loaded)
- `lbl_time`: time display
- Click-to-seek via eventFilter on text output widget (parses `[MM:SS.xx-MM:SS.xx]` timestamps from subtitle log)

### `ControlPanel` (`gui.py`)

**Source group:**
- `btn_offline`: Open video file dialog
- `btn_online`: Dropdown with 5 camera modes (Webcam / Webcam+Tracking / VR(OBS) / VR&Webcam Sync / Free Switch)
- `btn_open_remote`: Opens remote inference dialog
- `lbl_remote_badge`: Live connection status badge (●未連線 / ●已連線)
- `free_switch_bar` (hidden by default): Source toggle buttons (`btn_sw_webcam`, `btn_sw_vr`, `btn_sw_dual`) + `btn_fs_auto_cycle` (10s auto-cycle)

**Settings group:**
- `cmb_tts`: TTS mode selector — options: 不啟用(Mute) / OpenAI TTS / Gemini TTS / Local TTS; **default: Gemini TTS (index 2)**
- `cmb_style`: Broadcast style (populated from `livecc_prompts.yml`; default: objective)
- `cmb_voice`: OpenAI voice selector (8 options; default: coral) — visible only in openai mode
- Speed slider (OpenAI only), Exaggeration + CFG sliders (Local TTS only)

**Action group:**
- `btn_start`: Start / Stop broadcast
- `chk_record`: "錄音 (Record Audio)" checkbox (default: unchecked)

**Remote inference dialog** (non-modal, opened by `btn_open_remote`):
- "啟用遠端推論" toggle, host/port inputs, Connect/Disconnect button, status label
- Pre-filled from `configs/app.yml remote.host` / `remote.port`

**Note**: UI Scaling was **removed** (commit `9ae81ed`). DPI scaling is now automatic only (`dpi / 96.0`, clamped to 0.85–1.6× font scale).

### `TextOutputWidget` (`widgets/text_output.py`)

- `QPlainTextEdit`, read-only, dark theme
- `appendText()` / `clearText()` / `getText()`

---

## 11. Entry Point & Startup Sequence

```
python -m miis_broadcast
  → app.py:main()
    1. Pre-init both Gemini clients (_get_client() for broadcaster + TTS)
       [BEFORE QApplication to avoid httpx/torch SSL-context thread race]
    2. QApplication()
    3. parse_configs(app.yml) + parse_configs(models.yml)
    4. MainWindow(configs)
         a. parse gui_window / inference settings
         b. init SessionLogger + AudioRecorder
         c. if NOT client_only: _load_livecc_model()  ← BLOCKS ~20-40s (loads Qwen2VL-7B on GPU)
         d. _init_fonts()          ← DPI scaling, NotoSansCJK, dynamic font size
         e. _initUI()              ← widgets, splitters, prompt manager, style dropdown
                                      └─ if local model: _initLiveCCWorker() + _initCameraWorker()
                                      └─ _ensure_audience_token_server()  ← HTTP /audience starts here
         f. _initTTSWorker()       ← OpenAI (eager-started) + Gemini (lazy) + Local (lazy)
         g. _initGeminiWorker()    ← GeminiWorker + GeminiBackgroundWorker threads
         h. pre-fill remote dialog host/port from config
         i. if bytetrack config: QTimer.singleShot(500ms, _preload_bytetrack_model)
         j. subtitle timer start (50ms)
         k. QTimer.singleShot(0, _apply_initial_geometry)
    5. mw.show()
    6. app.exec_()
```

**Server** (separate process):
```
python -m miis_broadcast.server [--host 0.0.0.0] [--port 9000] [--device 0]
  → loads LiveCCInfer once (shared across all sessions)
  → TCP listen, accept → ClientSession(sock, model) in daemon thread
```

`TTS_DRY_RUN=1` disables audio output (log-only mode for testing interrupt logic).

---

## 12. Remote Inference Architecture

### Protocol (`network/protocol.py`)

Custom binary framing over TCP:
```
[4B total_len BE] [4B json_len BE] [JSON payload] [optional binary]
```
where `total_len = json_len + len(binary)`.

Message types: MSG_HELLO / MSG_ACK / MSG_START / MSG_FRAME / MSG_SEGMENT / MSG_STATUS / MSG_ERROR / MSG_STOP / MSG_PING / MSG_PONG / MSG_CLIENT_DIAG.

### Client (`network/client.py` — `SocketClientRunner`)

- QThread: blocking `read_message()` receive loop
- Daemon `_frame_sender_loop`: drains `_frame_queue(maxsize=30)` → TCP `sendall()`
- JPEG encoding (quality 75) happens on GUI thread; enqueue is non-blocking (drops oldest if full)
- Sends `MSG_CLIENT_DIAG` every 2s during inference: client RSS, JPEG queue levels, system RAM%, GPU VRAM (via `cuda_vram_snapshot()`)

### Server (`server/session.py` — `ClientSession`)

Per-connection threads:
1. **`_frame_processor_loop`**: JPEG → `cv2.imdecode` → `deque(maxlen=180)` frame buffer
2. **`_inference_loop`**: every 2s → `build_clip_from_buffer(window_sec=2.0)` → `live_cc_from_frames()` → `MSG_SEGMENT`; resets KV cache every 3 inferences; handles CUDA OOM + cuBLAS recoverable errors (clear KV + retry)

Server-side duplicate filter: `difflib.SequenceMatcher ≥ 0.86` suppresses overlapping 2s clips.

**Key design**: single-slot server frame queue (maxsize=1) — server always processes newest frame, never accumulates lag.

---

## 13. Audience Second-Screen

Activated only in **Free Switch** mode. Requires a running local LiveKit SFU at `audience.livekit_url`.

### Video path

```
FreeSwitchCameraThread.signal_vr_frame (native 1920×1080 RGB)
  → Qt QueuedConnection
  → MainWindow._deliver_audience_vr_frame()
  → AudiencePublisher.push_video_frame()   [enqueue to _video_q maxsize=3]
  → _video_pump_vr() asyncio task
  → cv2.resize + RGBA pack (ThreadPoolExecutor audience-vr-*)
  → rtc.VideoSource.capture_frame()
  → LiveKit room: track "broadcast_video" @ 30 fps
```

### Audio path

```
OpenAI TTS audio_player_worker thread
  → register_pcm_sink callback → AudiencePublisher.push_audio_chunk()
  → _audio_q maxsize=48
  → _audio_pump_direct() asyncio task
  → rtc.AudioSource.capture_frame()
  → LiveKit room: track "narration" @ 24 kHz mono
```

When audience is active, `mute_local=True` suppresses local speaker output on the control machine.

### Token server (`audience/token_server.py`)

Starts at app startup (if `audience.enabled: true`); runs for app lifetime.

- `GET /audience` → serve audience HTML viewer page
- `POST /api/audience/join` → issue subscribe-only LiveKit JWT
- `GET /assets/*` → serve avatar MP4 + PNG for browser-side chroma-keying

---

## 14. Known Design Decisions & Invariants

- **LiveCC is always called**; hallucination filtering happens inside LiveCC (`_is_degenerate`). Gemini receives all non-degenerate output and assigns P5 to suppress speaking. Silent filter → apparent timestamp gaps in log (1× vs 2× `streaming_time_interval` ≈ 1.33s step) — this is intentional, not a bug.
- **Gemini never sees video frames** — text-in/text-out broadcaster only. All visual understanding lives in LiveCC.
- **Remote segments skip `_route_segment()`** — no keyword scan, no Gemini enrichment. The server returns pre-formed commentary text that routes directly to TTS via `_on_segment_impl()`.
- **TTS backpressure is cumulative, not point-in-time**: `get_queue_remaining_sec()` tracks the sum of all queued+playing audio and drains by real wall-clock elapsed time (virtual fluid level). P1/P2 resets the sum (matches `clear_text_queue`); P3-P5 accumulates (matches bounded FIFO). Without this, long sentences cause systematic backpressure underestimation → queue fills to `_MAX_QUEUE_DEPTH=2` → unplayed content silently dropped.
- **GeminiBackgroundWorker fires independently** of per-segment Gemini calls. The two paths coexist: `GeminiWorker` enriches specific LiveCC events; `GeminiBackgroundWorker` maintains continuous narration during quiet periods. Both read TTS backlog via the same `get_queue_remaining_sec()`.
- **KV cache is per-video-run** (file mode) or per-session with periodic resets (camera mode: every `memory_reset_every=5` inferences; server: every 3 inferences). Not persistent across videos.
- **File mode subtitles are time-synced**: LiveCC runs ahead of playback; `_pending_segments` deque holds them until `_playback_sec >= start_t`. Background worker segments (`_background=True`) bypass this and display immediately regardless of playback position.
- **Camera mode subtitles are immediate**: no playback timeline.
- **Free Switch always opens both cameras at startup** and grabs both every frame — even the inactive source — to keep its buffer current and enable instantaneous switching without reconnect latency.
- **DualSource synchronization**: uses `cap.grab()` (hardware latch, ~0.1–0.3ms) followed by `cap.retrieve()` (decode) for both cameras, achieving sub-millisecond inter-camera Δt.
- **Gemini client pre-init in app.py** (before QApplication, before torch loads) avoids a race condition between httpx's SSL context setup and CUDA initialization when both happen on different threads.
- **`build/` directory** is a stale pip build artifact. Never edit files there.
- **`livecc_utils`** is an external package providing `prepare_multiturn_multimodal_inputs_for_generation`, `get_smart_resized_clip`, `get_smart_resized_video_reader`.
