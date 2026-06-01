# MIIS Broadcast — Architecture Reference

> **Purpose of this file**: AI-first architecture map. Read this before touching any source file.  
> Last updated: 2026-06-01

---

## 1. What This System Does

Real-time sports broadcast commentary generator. It:
1. Reads video (file or live camera)
2. Runs a local vision-language model (LiveCC / Qwen2VL-7B) to convert video clips → text descriptions
3. Sends those descriptions to the Gemini API, which produces broadcast-quality text with priority scoring (P1–P5)
4. Speaks the broadcast text via TTS (OpenAI Realtime WebSocket or local Chatterbox model)
5. Displays live subtitles in a PySide6 GUI

---

## 2. Directory Map

```
miis_broadcast/
├── src/miis_broadcast/          # installable Python package (source of truth)
│   ├── app.py                   # entry point: reads configs, creates QApplication + MainWindow
│   ├── gui.py                   # MainWindow + all Qt UI, signal wiring, VideoThread, CameraThread
│   ├── workers/                 # Qt QObject workers (each runs in its own QThread)
│   │   ├── livecc.py            # LiveCCWorker (file mode), LiveCCCameraWorker (camera mode)
│   │   ├── gemini.py            # GeminiWorker — calls Gemini API, manages priority queue
│   │   ├── openai_tts.py        # OpenAITTSWorker — WebSocket to GPT-4o Realtime TTS
│   │   ├── gemini_tts.py        # GeminiTTSWorker — Gemini AUDIO modality TTS via ffplay
│   │   ├── chatterbox_tts.py    # ChatterboxTTSWorker — local Chatterbox TTS model
│   │   ├── input.py             # Legacy BaseWorkerThread / VideoWorkerThread (not main pipeline)
│   │   ├── obs_input.py         # OBS input (currently inactive in main pipeline)
│   │   └── obs_bytetrack.py     # ByteTrack object tracker for OBS (inactive in main pipeline)
│   ├── core/
│   │   ├── models/
│   │   │   ├── livecc_transformers.py   # LiveCCInfer: Qwen2VL inference, KV cache, hallucination filter
│   │   │   ├── gemini_broadcaster.py    # Gemini API calls, RAG retriever, stream parsing
│   │   │   ├── openai_tts.py            # TTS queue, enqueue_tts_text(), print_tts_stats()
│   │   │   ├── gemini_tts.py            # Gemini TTS singleton: enqueue, interrupt, ffplay playback
│   │   │   ├── chatterbox_tts.py        # Chatterbox model wrapper
│   │   │   └── bytetrack_tracker.py     # ByteTrack (inactive)
│   │   ├── io/
│   │   │   ├── input.py                 # Generic video input abstraction
│   │   │   └── obs_input.py             # OBS-specific input
│   │   ├── prompt/
│   │   │   └── prompt_manager.py        # Loads livecc_prompts.yml, exposes style list + query
│   │   ├── match_tracker.py             # Thread-safe singleton: score, period, last_event
│   │   └── utils/
│   │       ├── config.py                # parse_configs(), load_app_config(), load_system_prompts()
│   │       ├── format_output.py
│   │       ├── format_string.py
│   │       ├── latency_monitor.py
│   │       ├── vram_monitor.py
│   │       ├── session_logger.py
│   │       └── counter.py
│   └── widgets/
│       ├── text_output.py       # TextOutputWidget: scrolling subtitle display
│       └── file_explorer.py
├── configs/
│   ├── app.yml                  # GUI settings, TTS params, Gemini params, dedup params
│   ├── models.yml               # LiveCC classifier config (fps, tokens, generation params)
│   ├── livecc_prompts.yml       # Broadcast styles (labels for UI) + livecc_query string
│   └── system_prompts.yml       # Gemini system prompts per style (gemini_broadcaster dict)
├── build/                       # pip build artifacts — NOT source of truth, ignore for dev
├── log/                         # runtime logs: livecc_output.log, gemini_output.log
└── logs/                        # app_error.log (Python logging), vram_tracker.log
```

---

## 3. Full Data Pipeline

### 3A. File Mode

```
VideoThread (QThread)
  └─ cv2.VideoCapture → signal_frame(frame_rgb, frame_idx, fps)
       └─ MainWindow.on_video_frame()  → updates VideoPanel display + _playback_sec

LiveCCWorker (QThread)                       [runs ahead of playback]
  └─ LiveCCInfer.live_cc(query, state)
       ├─ get_smart_resized_video_reader()   livecc_utils — reads video file
       ├─ get_smart_resized_clip()           livecc_utils — extracts N frames per clip
       ├─ Qwen2VL.generate()                 local GPU (cuda:0), bfloat16, flash_attention_2
       │    └─ KV cache (past_key_values, past_ids) — maintained across clips
       ├─ _is_degenerate()                   hallucination filter (hard-coded phrase list)
       ├─ _parse_visual_json()               tries JSON, falls back to raw_description dict
       └─ signal_segment(start_t, stop_t, parsed_dict)

MainWindow._route_segment()                  [fast keyword scan BEFORE Gemini]
  ├─ P1 keyword hit → flush_and_abort() + enqueue_front() + signal_tts_interrupt
  ├─ P2 keyword hit → enqueue_front()
  └─ P3 (normal)   → _signal_to_gemini (QueuedConnection)

GeminiWorker (QThread)
  └─ stream_gemini(event_data)
       ├─ _get_context_for_query()   RAG: TF-IDF retriever over loaded game context file
       ├─ _get_match_state()         reads MatchTracker singleton
       ├─ client.models.generate_content_stream()   Gemini API (streaming)
       └─ parses "P<1-5>: text\nLABEL: xxx" format
            ├─ signal_priority(start_t, stop_t, priority, should_speak)   fires first
            └─ signal_broadcast(start_t, stop_t, result_dict)             fires after

MainWindow._on_gemini_priority()
  └─ if P1 → signal_p1_confirmed → LiveCCCameraWorker.requestMemoryReset()
           → interrupt TTS

MainWindow.on_segment()
  ├─ FILE MODE: pushes (start_t, stop_t, data) to _pending_segments deque
  │             _subtitle_timer (50ms) drains deque when _playback_sec >= start_t
  ├─ CAMERA MODE: displays immediately
  ├─ dedup check (_is_duplicate_tts, word overlap ≥ 0.75 within 3s window)
  ├─ priority guard (_tts_protect_until: blocks lower-priority for estimated speak time)
  └─ signal_tts_speak(text, priority, ref_ts, start_t)  OR  signal_local_tts_speak(text)

OpenAITTSWorker (QThread) — WebSocket to gpt-4o-realtime-preview
  └─ audio output → sounddevice / pyaudio
```

### 3B. Camera Mode

```
CameraThread (QThread)
  └─ cv2.VideoCapture(camera_index) → signal_frame(frame_rgb)
       └─ MainWindow.on_camera_frame()
            ├─ updates VideoPanel
            └─ cam_worker.push_frame(frame_bgr, t_relative)

LiveCCCameraWorker (QThread)
  └─ buffer: deque[FrameItem(t, frame)]  maxlen=180
     runCameraInference loop (every infer_interval seconds):
       build_clip_from_buffer() → VideoClip(frames, fps, t_start)
       LiveCCInfer.live_cc_from_frames(clip, query, state)
         └─ same Qwen2VL inference as file mode
       every memory_reset_every inferences: state = {}  (KV cache cleared)
  └─ signal_segment → same _route_segment → Gemini → TTS pipeline as file mode
```

---

## 4. Threading Model

| Thread | Worker | Notes |
|--------|--------|-------|
| Main (GUI) | MainWindow | Qt event loop; all signal connections here |
| livecc_thread | LiveCCWorker | Blocks on GPU; file mode only |
| cam_worker_thread | LiveCCCameraWorker | Blocks on GPU; camera mode only |
| gemini_thread | GeminiWorker | Blocks on HTTP; priority deque |
| tts_thread | OpenAITTSWorker | WebSocket; lazy connect |
| gemini_tts_thread | GeminiTTSWorker | Gemini AUDIO API + ffplay; lazy-started on first "gemini" selection |
| local_tts_thread | ChatterboxTTSWorker | Local model; lazy-loaded on first "local" selection |
| VideoThread | VideoThread | cv2 decode loop; file mode |
| CameraThread | CameraThread | cv2 capture loop; camera mode |

**Signal safety rule**: Workers only receive data via Qt Signals (QueuedConnection). P1/P2 fast-path calls (`flush_and_abort`, `enqueue_front`) are CPython-GIL-safe deque ops — no explicit lock needed.

---

## 5. Key Classes

### `LiveCCInfer` (`core/models/livecc_transformers.py`)

The local VLM inference engine.

- **Model**: `Qwen2VLForConditionalGeneration` from HuggingFace, loaded at startup (~20–40s)
- **State dict per inference run**: `video_path`, `past_ids` (Tensor), `past_key_values` (KV cache), `video_pts`, `last_timestamp`, `mm_window_start`, `carry_text`, `recent_texts`
- **KV cache management**: `truncate_state_by_budget()` trims past_ids + past_key_values when context would overflow ctx_max (32768 default). Aligns cuts to chat template boundaries (`<|im_start|>user` / `assistant`) to avoid corrupt output.
- **Multimodal window policy** (`_apply_mm_window_policy`): clears KV cache every `mm_window_sec` seconds to prevent infinite accumulation. `carry_text` (last K sentences) is injected into next clip's prompt for continuity.
- **Hallucination filter** (`_is_degenerate`): rejects empty, <3-word, >80-word, first-person-heavy, and YouTube-commentary-phrase outputs. All non-degenerate outputs are forwarded to Gemini.
- **Output**: raw text → `_parse_visual_json()` → `{"event": str, "urgency": int, "metadata": {"raw": str}}` or richer structured dict

### `GeminiWorker` + `gemini_broadcaster.py`

- **Model**: `gemini-3.1-flash-lite-preview` (configurable via `configs/app.yml`)
- **Input**: event dict from LiveCC; extracts `metadata.raw` or `event` as the visual description
- **Prompt construction**: `[Game context: ...]` + `[Match state: ...]` + visual description
- **RAG**: `_ContextRetriever` — TF-IDF over game context file; used when context > 600 chars. Retrieves top-3 relevant chunks.
- **Output format** (Gemini is instructed to produce exactly 2 lines):
  ```
  P<1-5>: <broadcast text>
  LABEL: <action_label>
  ```
  Parsed by `stream_gemini()` as chunks arrive. `signal_priority` fires as soon as P-line is seen; `signal_broadcast` fires after LABEL line.
- **Priority queue**: deque with `_abort_current` flag. P1 calls `flush_and_abort()` then `enqueue_front()`. P2 calls `enqueue_front()`. P3 goes through QueuedConnection (normal tail append).

### `MatchTracker` (`core/match_tracker.py`)

- Module-level singleton: `from miis_broadcast.core.match_tracker import match_tracker`
- Tracks `red_score`, `blue_score`, `period`, `last_event`
- `get_state_string()` injected into every Gemini prompt
- Updated by `MainWindow.on_segment()` when `action_label.startswith("score_")`

### `PromptManager` (`core/prompt/prompt_manager.py`)

- Loads `configs/livecc_prompts.yml`
- `livecc_query()` → fixed instruction string sent to LiveCC as the generation prompt
- `list_styles()` → list of `StyleItem(key, label, description)` displayed in GUI dropdown
- Style keys map to Gemini system prompts in `configs/system_prompts.yml` (`gemini_broadcaster` dict)

---

## 6. Configuration Files

### `configs/app.yml`

```yaml
model:
  classifier_name: livecc_7b      # must match a key in models.yml classifiers

openai_tts:
  model_url: wss://...            # gpt-4o-realtime WebSocket URL
  default_voice: coral
  default_speed: 1.5
  temperature: 0.7

gemini_tts:
  model_name: models/gemini-3.1-flash-tts-preview
  voice: Kore                     # prebuilt voice name

chatterbox_tts:
  audio_prompt_path: ...          # reference WAV for voice cloning

gemini:
  model_name: gemini-3.1-flash-lite-preview
  rag_threshold: 600              # chars; below this, full context used verbatim
  temperature: 0.4
  max_output_tokens: 50

inference:
  dedup_window_s: 3.0             # suppress near-identical TTS within this window
  dedup_threshold: 0.75           # word overlap ratio for dedup
  max_pending: 400                # max queued pending_segments (file mode)
```

### `configs/models.yml`

```yaml
classifiers:
  livecc_7b:
    fps: 4.0
    initial_fps_frames: 6         # frames in first clip
    streaming_fps_frames: 3       # frames per subsequent clip
    max_new_tokens: 12
    mm_window_sec: 8.0            # KV cache reset interval
    carry_text_max_chars: 0       # 0 = no carry text on reset
    carry_recent_k: 0
    generation:
      temperature: 0.3
      repetition_penalty: 1.5
    camera:
      window_sec: 1.5             # seconds of frames to include per clip
      target_fps: 2.0
      infer_interval: 1.0         # inference cadence
      memory_reset_every: 5       # hard KV reset every N inferences
```

### `configs/livecc_prompts.yml`

Defines:
- `livecc_query`: the fixed system instruction sent to Qwen2VL (e.g. "Describe only what you see...")
- `default_style`: key of default Gemini broadcast style
- `styles`: dict of `{key: {label, description}}` — populates the GUI dropdown

### `configs/system_prompts.yml`

```yaml
gemini_broadcaster:
  objective: "..."    # Gemini system prompt for objective style
  exciting: "..."     # ...other styles
```

Each value is the full system instruction telling Gemini what output format to produce (P1–P5 + LABEL lines).

---

## 7. Priority & Interrupt System

### Priority levels

| Priority | Meaning | TTS behavior |
|----------|---------|--------------|
| P1 | Critical event (goal, score, foul) | Immediately interrupts any current TTS |
| P2 | Important event (pass, timeout) | Interrupts if current ≥ P3 |
| P3 | Normal commentary | Interrupts if current ≥ P4 |
| P4 | Background/low urgency | Does not interrupt |
| P5 | No active play | Does not speak (`should_speak=False`) |

### Fast-path keyword scan (before Gemini)

`MainWindow._scan_priority()` runs on the raw LiveCC text before Gemini:
- P1 keywords: `score, goal, foul, shot, basket, 進球, 得分, 犯規, 投籃, dunk, slam`
- P2 keywords: `pass, timeout, intercept, block, 傳球, 暫停, 抄截, 封蓋`

P1 hit → `flush_and_abort()` (clears Gemini queue + sets abort flag) + TTS interrupt immediately, then Gemini is still called with the segment at front of queue for proper broadcast text.

### Interrupt matrix

```python
_INTERRUPT_MATRIX = {1: 2, 2: 3, 3: 4}
# new_priority can interrupt if _tts_last_priority >= interrupt_matrix[new_priority]
```

Transition phrases prepended when interrupting: `"Oh!—"` (P1), `"And—"` (P2), `""` (P3).

### P1 confirmed → KV cache reset

When Gemini confirms P1 (`signal_priority` → `_on_gemini_priority`), `signal_p1_confirmed` is emitted → `LiveCCCameraWorker.requestMemoryReset()` (camera mode) clears `self._state = {}`. In file mode, this is a no-op since the LiveCC state is already managed by the mm_window_sec policy.

---

## 8. State Management

### LiveCC inference state (per run)

Stored in a plain dict passed through generators:

| Key | Type | Purpose |
|-----|------|---------|
| `video_path` | str | identifies video reader cache |
| `past_ids` | Tensor `[1, seq]` | KV history token ids |
| `past_key_values` | tuple[tuple[Tensor]] | Transformer KV cache |
| `video_pts` | Tensor | all frame presentation timestamps |
| `last_video_pts_index` | int | last consumed frame index |
| `last_timestamp` | float | last processed video second |
| `video_timestamp` | float | wall-clock elapsed (file mode: set by LiveCCWorker loop) |
| `mm_window_start` | float | start of current KV cache window |
| `carry_text` | str | summary injected after KV reset |
| `recent_texts` | list[str] | last K LiveCC responses (source for carry_text) |
| `video_end` | bool | sentinel to break inference loop |

### Game context + RAG

- Set via GUI "載入比賽資訊" button → `load_game_context_file(path)` → `set_game_context(text)`
- Short context (≤600 chars): injected verbatim into every Gemini prompt
- Long context (>600 chars): `_ContextRetriever` (TF-IDF, top-3 chunks) retrieves relevant portion per visual query

### Match score

`MatchTracker` singleton updated by Gemini `action_label` values starting with `"score_"` (e.g. `"score_red"`, `"score_blue"`). State string injected into every Gemini prompt.

---

## 9. TTS Backends

### OpenAI Realtime TTS (`workers/openai_tts.py`)

- Protocol: WebSocket (`gpt-4o-realtime-preview-2025-06-03`)
- Streaming: sends text, receives base64 audio chunks → plays via sounddevice
- `speak(text, priority, ref_ts, start_t)` — queues utterance
- `interrupt()` — sends cancellation message, clears buffer

### Gemini TTS (`workers/gemini_tts.py`, `core/models/gemini_tts.py`)

- Protocol: `google.genai` `generate_content` with `response_modalities=["AUDIO"]`
- Playback: raw PCM16 audio streamed to `ffplay` subprocess (interruptible)
- Module-level singleton threads + queues (same pattern as `openai_tts.py`)
- `enqueue_tts_text(text, drop_outdated=True)` — clears queue before enqueuing when `drop_outdated` (default)
- `interrupt_tts()` — terminates the current `ffplay` process + clears queue + sets `_interrupt_event`
- Config: `gemini_tts.model_name` and `gemini_tts.voice` in `app.yml`
- `GeminiTTSWorker` exposes the same Qt interface as `OpenAITTSWorker`: `speak`, `interrupt`, `start`, `stop`, `apply_settings`, `signal_tts_done`

### Chatterbox (local) TTS (`workers/chatterbox_tts.py`)

- Model loaded lazily on first "Local TTS" selection
- Reference voice WAV for voice cloning (`audio_prompt_path`)
- Params: `temperature`, `cfg_weight`, `chunk_size`
- `speak(text)` / `interrupt()`

### TTS dedup & priority guard (`MainWindow.on_segment`)

- Dedup: word-overlap ratio ≥ 0.75 within 3s window → skip
- Priority guard: `_tts_protect_until = now + max(2s, word_count × 0.3s)` — lower-priority segments blocked while current speech is still playing

---

## 10. GUI Components

### `MainWindow` (`gui.py`)

- Subclasses `QMainWindow`
- Creates and owns all workers and threads
- Contains all signal wiring
- `_route_segment()`: the central routing function (LiveCC output → fast-path → Gemini)
- `_tick_subtitle_scheduler()`: 50ms timer syncs subtitle display to video playback position

### `VideoPanel`

- Shows video preview (`QLabel` with scaled pixmap)
- Seek slider + time display
- Click-to-seek: installs eventFilter on text output widget; parses `[MM:SS.xx-MM:SS.xx]` from subtitle lines

### `ControlPanel`

- Source selection (file / camera)
- Game context file loader
- TTS mode selector (none / openai / local)
- Broadcast style selector (populated from `livecc_prompts.yml`)
- Voice/speed/exaggeration/CFG controls (conditionally visible per TTS mode)
- Start/Stop button

### `TextOutputWidget` (`widgets/text_output.py`)

- Scrolling subtitle log
- Appends lines with timestamps

---

## 11. Entry Point & Startup Sequence

```
python -m miis_broadcast
  → app.py:main()
    → parse_configs(app.yml) + parse_configs(models.yml)
    → merge: configs['model']['classifier'] = classifier_configs[classifier_name]
    → QApplication()
    → MainWindow(configs)
         → _load_livecc_model()      ← BLOCKS: loads Qwen2VL-7B on GPU (~30s)
         → _initUI()                 ← creates VideoPanel, ControlPanel
         → _initTTSWorker()          ← starts OpenAI TTS thread (Chatterbox lazy)
         → _initGeminiWorker()       ← starts Gemini thread + initializes client
         → _initLiveCCWorker()       ← starts LiveCC thread (model already loaded)
         → _initCameraWorker()       ← starts camera worker thread
    → mw.show()
    → app.exec_()
```

`TTS_DRY_RUN=1` environment variable disables actual audio output (log-only mode for testing interrupt logic).

---

## 12. Known Design Decisions & Invariants

- **LiveCC is always called**; hallucination filtering happens *inside* LiveCC (degenerate check), not as a pre-filter. Gemini receives all non-degenerate LiveCC output and assigns P5 to skip speaking.
- **Gemini never sees video frames** — it only receives the text description from LiveCC plus game context + match state. Gemini is purely a text-in / text-out broadcaster.
- **KV cache is per-video-run**, not persistent across videos. Clearing cached video readers on stop/restart is explicit (`livecc._cached_video_readers_with_hw.clear()`).
- **File mode subtitles are time-synced**: LiveCC runs ahead of video playback; segments are queued and only displayed/spoken when `_playback_sec >= start_t`.
- **Camera mode subtitles are immediate**: no playback timeline, segments displayed as they arrive.
- **Priority guard** (`_tts_protect_until`) prevents a fast burst of lower-priority segments from stepping on each other after an interrupt.
- **`build/` directory** is a stale pip build artifact. Never edit files there — always edit under `src/miis_broadcast/`.
- **`livecc_utils`** is an external package (not in this repo) providing `prepare_multiturn_multimodal_inputs_for_generation`, `get_smart_resized_clip`, `get_smart_resized_video_reader`.
