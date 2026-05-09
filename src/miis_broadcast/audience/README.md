# Audience second screen (LiveKit)

This package implements a **second display** for viewers: they open a browser page and receive **video** (one LiveKit track: `broadcast_video`) plus **TTS narration** (`narration`) over WebRTC (LiveKit). **Same track name in both modes; only the pixel content changes** (full-screen VR vs VR + LiveAvatar PiP). The **control GUI (first screen)** continues to show whichever source the operator selects in **Free Switch**; **no narration is meant to play on the first screen** when the audience pipeline is active (`mute_local=True`).

Two operating modes are supported (toggled by `configs/app.yml`):

| Mode | Video source | Audio path |
|------|-------------|------------|
| **VR mode** (default, `liveavatar.enabled: false`) | VR camera frames from `FreeSwitchCameraThread` | TTS PCM → local room directly |
| **LiveAvatar mode** (`liveavatar.enabled: true`) | VR full screen + LiveAvatar avatar as **picture-in-picture** (bottom-right, optional) | TTS PCM → LiveAvatar **WebSocket** (`agent.speak`) **and** local room (**delayed** by `audio_delay_ms`, default ~520 ms, for A/V sync) |

Full integration points live in [gui.py](../gui.py) (`_ensure_audience_token_server`, `_start_audience_services`, `_deliver_audience_vr_frame`). For input workers and frame contracts, see [workers/README.md](../workers/README.md).

---

## When it runs

| Condition | Behavior |
|-----------|----------|
| `audience.enabled: true` in [configs/app.yml](../../../configs/app.yml) | HTTP token server may start with the GUI (`GET /audience`, `POST /api/audience/join`). |
| Operator chooses **Online ▾ → Free Switch** | **LiveKit publisher** starts: publishes `broadcast_video` + `narration` (pixels: VR-only vs VR+LiveAvatar PiP per `liveavatar.enabled`). |
| Operator leaves Free Switch / switches mode | Publisher stops; HTTP server keeps running until the app exits (if enabled). |

---

## End-to-end flow

**At a glance — how the two modes differ**

| Topic | **VR mode** (`liveavatar.enabled: false`) | **LiveAvatar mode** (`liveavatar.enabled: true`) |
|------|--------------------------------------|------------------------------------------|
| **Prerequisite** | **Free Switch** must be on | Same |
| **What viewers see** | Full-screen VR only | Full-screen VR + **bottom-right avatar (PiP)** |
| **Narration audio** | Straight to **local** LiveKit | **WebSocket** to LiveAvatar (`agent.speak`); **delayed** path (`audio_delay_ms`) to **local** LiveKit (A/V alignment) |
| **How viewers join** | Same for both: `http://…:8080/audience` → JWT → WebRTC to **local** LiveKit | Same |

The LiveAvatar diagram below splits **control-plane** (REST, once per run) from **media**: **WebSocket** carries narration PCM (`agent.speak`); **LiveAvatar LiveKit** carries the rendered avatar video back. Local Docker LiveKit is unchanged for the browser.

> **VR is fully independent of the avatar cloud stream.**
> `_video_pump_vr_pip` runs at a fixed **30 fps** from the moment Free Switch starts, regardless of whether the avatar track has arrived. The avatar PiP is overlaid only when a cached PiP tile is available (`_liveavatar_pip_tile`); if the cloud stalls the VR output continues uninterrupted.

> **CPU / threading (matches `livekit_publisher.py`):**
> - **OpenCV (`cv2`)** work never runs on the asyncio event loop: it is submitted via `loop.run_in_executor(...)`.
> - Two **dedicated** `ThreadPoolExecutor`s, each **`max_workers=1`**: **`_vr_executor`** (resize / composite / RGBA for `_video_pump_vr` and `_video_pump_vr_pip` only) and **`_avatar_executor`** (RGB decode + PiP prescale for `_avatar_frame_reader_task` only). This avoids TTS-heavy avatar decode starving VR compositing when both contend for the default shared pool.

---

### VR mode (narration + full-screen VR)

```mermaid
flowchart LR
  subgraph SRC["① Sources (Free Switch on)"]
    direction TB
    VR["VR video<br/>FreeSwitch → MainWindow"]
    PCM["Narration PCM<br/>OpenAI TTS → push_audio_chunk"]
  end

  PUB["② AudiencePublisher"]

  subgraph LOC["③ Local SFU"]
    LK["LiveKit (Docker)<br/>room · signaling :7880"]
  end

  subgraph AUD["④ Audience"]
    direction TB
    HTTP["FastAPI :8080<br/>/audience · /api/audience/join"]
    BR["Browser<br/>livekit-client"]
  end

  VR --> PUB
  PCM --> PUB
  PUB -->|"publish broadcast_video (full-screen VR) + narration"| LK
  HTTP -->|"HTML + JWT"| BR
  BR <-->|"subscribe"| LK
```

In **VR-only** mode, `_video_pump_vr` uses the same **`_vr_executor`** (single worker) for `cv2` resize + RGBA before `capture_frame`.

### LiveAvatar mode (VR background + avatar PiP + narration)

**Bootstrap (once when publisher starts):** `POST https://api.liveavatar.com/v1/sessions/token` → `POST …/v1/sessions/start` (Bearer) → receive `livekit_url`, `livekit_client_token`, `ws_url`; connect WebSocket and wait `session.state_updated: connected`.

```mermaid
flowchart LR
  subgraph SRC["① Sources (Free Switch on)"]
    direction TB
    VR["VR video<br/>FreeSwitch → MainWindow"]
    PCM["Narration PCM<br/>OpenAI TTS → push_audio_chunk"]
  end

  subgraph PUB["② AudiencePublisher — asyncio + dedicated cv2 threads"]
    direction TB
    TVRPIP["_video_pump_vr_pip<br/>async: pacing + capture_frame"]
    TAVR["_avatar_frame_reader_task<br/>async: VideoStream loop"]
    TAPU["_audio_pump_liveavatar"]
    TREL["_audio_delay_relay"]
    subgraph EXEC["cv2 via run_in_executor (isolated)"]
      direction LR
      VRX["_vr_executor<br/>1 worker · composite"]
      AVX["_avatar_executor<br/>1 worker · decode+prescale"]
    end
    TVRPIP -.->|offload| VRX
    TAVR -.->|offload| AVX
  end

  subgraph CLD["③ LiveAvatar LITE cloud"]
    direction TB
    WS["Events WebSocket<br/>api route via ws_url<br/>↑ agent.speak PCM Base64"]
    CLK["LiveAvatar LiveKit<br/>subscribe only · avatar video → publisher"]
  end

  subgraph LOC["④ Local SFU (browsers here)"]
    LK["LiveKit Docker<br/>room · :7880"]
  end

  subgraph AUD["⑤ Audience"]
    direction TB
    HTTP["FastAPI :8080<br/>/audience · join"]
    BR["Browser livekit-client"]
  end

  VR --> TVRPIP
  PCM --> TAPU
  CLK -->|"avatar frames"| TAVR
  TAVR -.->|"PiP tile cache"| TVRPIP
  TAPU -->|"agent.speak"| WS
  TVRPIP -->|"broadcast_video VR+PiP"| LK
  TREL -->|"narration (delayed)"| LK
  HTTP -->|"HTML + JWT"| BR
  BR <-->|"subscribe"| LK
```

**LiveAvatar mode — tasks (match ② above)**

| Task | Starts | Depends on cloud? | What it does |
|--|--|--|--|
| `_video_pump_vr_pip` | Immediately | **No** | Async pacing at 30 fps; **cv2** composite on **`_vr_executor`** only |
| `_avatar_frame_reader_task` | Immediately | Yes (up to 30 s wait) | Async `VideoStream`; **cv2** decode + PiP tile on **`_avatar_executor`** → `_liveavatar_pip_tile` |
| `_audio_pump_liveavatar` | Immediately | No | Drains TTS PCM → WebSocket `agent.speak` + delay queue |
| `_audio_delay_relay` | Immediately | No | Forwards delayed PCM → local `narration` AudioSource |

**LiveAvatar mode — steps (match numbered bands above)**

| Step | Who | What happens |
|:--:|--|--|
| ① | GUI + TTS | Free Switch feeds VR; TTS feeds PCM into `AudiencePublisher` |
| ② | `AudiencePublisher` | Four tasks start in parallel; VR publishing never waits for avatar; PiP appears once first cloud frame is cached |
| ③ | LiveAvatar | **WebSocket** ingests PCM for lip sync; **cloud LiveKit** delivers avatar video to `_avatar_frame_reader_task` |
| ④ | Local LiveKit | Receives `broadcast_video` (VR or VR+PiP) + delayed `narration` |
| ⑤ | Audience | Same as VR: :8080 → JWT → subscribe **local** room |

### TTS path (why it matches first-screen timing)

PCM still flows through a **single** `_audio_output_queue`. The player thread forwards each chunk to `push_audio_chunk` **and**, when `mute_local=True`, feeds **silence** to the local audio device so **hardware playback timing** stays aligned with unmuted mode. That keeps the LiveKit audio stream paced like normal local playback (see [openai_tts.py](../core/models/openai_tts.py)).

```mermaid
sequenceDiagram
  participant API as OpenAI Realtime
  participant Q as _audio_output_queue
  participant P as Audio player thread
  participant SD as sounddevice
  participant LK as AudiencePublisher _audio_q

  API->>Q: response.audio.delta chunks
  Q->>P: dequeue
  P->>LK: push_audio_chunk (real PCM)
  alt mute_local
    P->>SD: write zero PCM (silent, same duration)
  else not muted
    P->>SD: write real PCM
  end
```

**LiveAvatar mode:** the same `push_audio_chunk` traffic is also consumed inside `AudiencePublisher` → WebSocket `agent.speak` (see flowchart above). Local `narration` is intentionally **delayed** by `liveavatar.audio_delay_ms` (default **520**) so it lines up with lip motion in the PiP. The delay **deadline is set when each PCM chunk is dequeued** from the TTS path (not after WebSocket `send_pcm_chunk` returns), so variable encode/network time does not jitter playout timing. **Override** without editing YAML: set **`LIVEAVATAR_AUDIO_DELAY_MS`** in project-root `.env`. **Tune:** if the **mouth visibly lags** the sound you hear in the browser, **increase** the delay; if sound is clearly **after** the mouth, **decrease** it (try steps of ~50 ms).

```mermaid
sequenceDiagram
  participant TTS as OpenAI TTS / player
  participant Q as AudiencePublisher _audio_q
  participant PUB as asyncio loop tasks
  participant VRX as _vr_executor
  participant AVX as _avatar_executor
  participant WS as LiveAvatar WebSocket
  participant CLK as LiveAvatar LiveKit
  participant LKloc as Local LiveKit

  TTS->>Q: push_audio_chunk (PCM int16 24kHz)
  Q->>PUB: drain chunk
  PUB->>WS: agent.speak (Base64 PCM)
  CLK-->>PUB: VideoStream frames
  PUB->>AVX: decode + PiP prescale (cv2)
  AVX-->>PUB: updates _liveavatar_pip_tile
  PUB->>VRX: composite VR + tile to RGBA (cv2)
  VRX-->>PUB: buffer
  PUB->>LKloc: capture_frame broadcast_video
  Note over PUB,LKloc: narration after audio_delay_ms via _audio_delay_relay
```

---

## Components

| Path | Role |
|------|------|
| [token_server.py](token_server.py) | FastAPI + uvicorn on `0.0.0.0`; serves viewer HTML and short-lived subscribe-only JWTs. |
| [static/index.html](static/index.html) | LiveKit JS viewer: subscribes to published video + audio tracks (`broadcast_video`, `narration`). |
| [livekit_publisher.py](livekit_publisher.py) | Background asyncio thread: VR or LiveAvatar dual-room + WebSocket PCM; **sleep-until-deadline** 30 fps pacing; **cv2** on **`_vr_executor`** / **`_avatar_executor`** (single worker each); PiP tile built in avatar reader, composited on VR path. |
| [liveavatar_session.py](liveavatar_session.py) | LiveAvatar LITE: token/start REST, WebSocket `agent.speak` / `agent.interrupt`, session stop. |
| [gui.py](../gui.py) | Reads `liveavatar` config block, builds dict, passes `liveavatar_cfg` to `AudiencePublisher`. |
| [openai_tts.py](../core/models/openai_tts.py) | `register_pcm_sink`, `clear_audio_queue` + optional `flush_callback` for LiveKit backlog. |
| [workers/free_switch.py](../workers/free_switch.py) | Emits **`signal_vr_frame`** (always VR) alongside **`signal_frame`** (active source). |
| [docker-compose.yml](../../../docker-compose.yml) | Local LiveKit container; map signaling + UDP ports. |
| [livekit.yaml](../../../livekit.yaml) | LiveKit server config (keys, RTC port range). |
| [scripts/open-audience-firewall.ps1](../../../scripts/open-audience-firewall.ps1) | Windows inbound rules for LAN viewers (run elevated). |

---

## Setup (local)

1. **LiveKit** (from repo root):

   ```bash
   docker compose up -d
   ```

2. **Align URLs for your LAN** (same host in all three places):

   - `configs/app.yml` → `audience.livekit_url` (e.g. `ws://<PC_LAN_IP>:7880`)
   - `docker-compose.yml` → `--node-ip <PC_LAN_IP>` (ICE must advertise a reachable IP)
   - Optional: pass `lan_hint_host=` into `AudienceTokenServer` from config if your `gui.py` is wired for it (prints a same-Wi-Fi URL in the console).

3. **Firewall** (other devices on Wi‑Fi): run `scripts/open-audience-firewall.ps1` **as Administrator** once, or manually allow TCP **8080, 7880, 7881** and UDP **50000–50020**.

4. **GUI**:

   ```bash
   python -m miis_broadcast
   ```

5. **Operator**: **Free Switch** → wait for **`[MEDIA] connected`** (VR-only mode) **or** **`[MEDIA] local_room connected`** (LiveAvatar mode) in the terminal → viewers open `http://<PC_LAN_IP>:8080/audience` (or `localhost` on the same machine).

---

## Configuration reference ([configs/app.yml](../../../configs/app.yml))

### `audience` section

| Key | Meaning |
|-----|---------|
| `audience.enabled` | Master switch for the feature. |
| `audience.livekit_url` | WebSocket URL passed to the browser (`ws://…:7880`). |
| `audience.api_key` / `api_secret` | Must match `livekit.yaml` `keys` (secret ≥ 32 chars on recent LiveKit). |
| `audience.room` | LiveKit room name (publisher + viewers join the same room). |
| `audience.port` | HTTP port for `/audience` and `/api/audience/join` (default **8080**). |

### LiveAvatar secrets (`.env`)

Keep **`LIVEAVATAR_API_KEY`** (and optionally **`LIVEAVATAR_AVATAR_ID`**, **`LIVEAVATAR_VOICE_ID`**) in the project root **`.env`** (gitignored). With `liveavatar.enabled: true` in `app.yml`, the GUI reads the key from the environment first, then falls back to `liveavatar.api_key` in YAML if set.

### `liveavatar` section

| Key | Default | Meaning |
|-----|---------|---------|
| `liveavatar.enabled` | `false` | Set `true` to activate LiveAvatar avatar PiP mode. |
| `liveavatar.api_key` | `""` | Optional if **`LIVEAVATAR_API_KEY`** is set in `.env`. Otherwise use API key from [app.liveavatar.com/developers](https://app.liveavatar.com/developers). |
| `liveavatar.avatar_id` | `""` | Avatar UUID from LiveAvatar, or **`LIVEAVATAR_AVATAR_ID`** in `.env`. |
| `liveavatar.voice_id` | `""` | Optional; **`LIVEAVATAR_VOICE_ID`** in `.env` (reserved for future use; LITE uses avatar default voice). |
| `liveavatar.quality` | `"low"` (see `app.yml`) | Video quality: `"low"` / `"medium"` / `"high"`. |
| `liveavatar.audio_delay_ms` | `520` | Delay (ms) before local-audience **narration** track plays, so it matches lip timing in the PiP (network-dependent; tune ±50 ms or set **`LIVEAVATAR_AUDIO_DELAY_MS`** in `.env`). |
| `liveavatar.sandbox` | `false` | When `true`, token requests use `is_sandbox` (see LiveAvatar docs). |

> **Network requirements (LiveAvatar mode)**
> - Docker host must reach **HTTPS 443** (`api.liveavatar.com`, WebSocket) and **UDP high ports** (LiveAvatar cloud LiveKit).
> - Install `httpx` and `websockets`: `pip install httpx websockets`.
> - Windows Firewall: run `scripts/open-audience-firewall.ps1` as Administrator.

---

## Client terminal log reference (prefixes)

These lines appear on **`python -m miis_broadcast`** stdout (not the browser). They are **separate** from session files under `logs/sessions/` and from server `[Server]` / `[Client]` telemetry in remote inference.

**`[MEDIA]`** always means the **local audience** LiveKit path (composite video → `broadcast_video`, and session connect/disconnect). **`[AUDIO]`** means the **`narration`** track on that same local room **and** the TTS → publisher PCM pipeline stats — the prefixes did **not** change meaning.

### `[AUDIENCE]` — HTTP token server

| Example | Meaning |
|---------|---------|
| `[AUDIENCE] token server started → http://localhost:8080/audience` | Uvicorn bound; local viewer URL. |
| `[AUDIENCE] same-WiFi URL → http://…:8080/audience` | Printed when a LAN hint host is configured. |
| `[AUDIENCE] if other devices cannot open…` | Reminder about Windows firewall. |
| `[AUDIENCE] join id=audience-… room=broadcast-room` | A viewer called `POST /api/audience/join`; JWT issued. |
| `[AUDIENCE] token server stopped` | App shutdown or server torn down. |
| `[ERR] [AUDIENCE] token server bind failed …` | Port in use or permission issue. |

### `[MEDIA]` — LiveKit publisher (video + session)

| Example | Meaning |
|---------|---------|
| `[MEDIA] publisher starting \| room=… mode=vr` | `AudiencePublisher.start()` in VR mode. |
| `[MEDIA] publisher starting \| room=… mode=liveavatar` | `AudiencePublisher.start()` in LiveAvatar mode. |
| `[MEDIA] connected \| room=…` | Local room connected. |
| `[MEDIA] local_room connected \| room=…` | LiveAvatar mode: local room connected. |
| `[MEDIA] publish_start track=broadcast_video (vr mode)` | VR-only pixels on the shared video track name. |
| `[MEDIA] publish_start track=broadcast_video (liveavatar mode)` | VR + PiP composite on the same track name. |
| `[MEDIA] fps=29.0 drop=0 (vr)` | VR pump stats; `drop` = `_video_q` depth (not dropped frames). |
| `[MEDIA] fps=29.0 (pip) vr_q=…` | LiveAvatar mode; PiP overlay active; `vr_q` = `_video_q` depth. |
| `[MEDIA] fps=29.0 (vr-only) vr_q=…` | LiveAvatar mode; avatar cache empty (cloud not ready / stalled); VR published without PiP. |
| `[MEDIA] disconnected from LiveKit` | Clean disconnect. |
| `[MEDIA] publisher stopped` | Thread joined after `stop()`. |
| `[WARN] [MEDIA] publisher thread hung; forcing event loop stop` | Graceful shutdown timed out. |
| `[ERR] publisher session (vr/liveavatar): …` | Python-side failure. |

### `[LIVEAVATAR]` — LiveAvatar (cloud REST + WebSocket + cloud LiveKit subscribe)

Routine **success** paths are kept short; **warnings/errors** still print in full.

| Example | Meaning |
|---------|---------|
| `[LIVEAVATAR] on \| avatar=… q=… delay=…ms` | From `gui.py` when audience starts with LiveAvatar config (not every API step). |
| `[LIVEAVATAR] token OK \| avatar=…` | POST `/v1/sessions/token` succeeded. |
| `[LIVEAVATAR] session ready \| id=…` | POST `/v1/sessions/start` + events WebSocket connected. |
| `[LIVEAVATAR] cloud room OK` | Subscribed to LiveAvatar **cloud** LiveKit (for avatar video). |
| `[LIVEAVATAR] PiP video \| participant=…` | Avatar video track received; PiP decode loop running. |
| `[LIVEAVATAR] tile/s=12.0 (cloud PiP refresh rate)` | About every **2 s**: how many times the PiP tile cache was updated from cloud video (healthy stream often **~10–30**/s depending on LiveAvatar encode). |
| `[LIVEAVATAR] stopped \| id=…` | POST `/v1/sessions/stop` succeeded on shutdown. |
| `[WARN] [LIVEAVATAR] avatar video track not received in 30s; PiP disabled` | No cloud video track within 30 s; VR continues without PiP. |
| `[WARN] …` / `[ERR] …` | e.g. WebSocket send failure, REST errors, decode issues (see message). |
| `[ERR] [LIVEAVATAR] failed to start session: …` | Bootstrap failed; check `LIVEAVATAR_API_KEY` / `avatar_id`. |

*(TTS flush sends `agent.interrupt` + short silence over the WebSocket but does **not** spam the console on success.)*

### `[AUDIO]` — publisher narration track + PCM hook

| Example | Meaning |
|---------|---------|
| `[AUDIO] publish_start track=narration` | LiveKit audio track published. |
| `[AUDIO] chunks/s=4.0 sample_rate≈24000 aq=8` | Periodic audio pump stats; **`aq`** = `_audio_q` depth (chunks waiting to send); not called `drop`. Rate should stay near **24000** samples/s when speech is active. |
| `[AUDIO] PCM sink registered \| mute_local=True` | From [openai_tts.py](../core/models/openai_tts.py) when Free Switch registers the audience sink. |
| `[AUDIO] PCM sink cleared \| mute_local=False` | Publisher stopped / sink removed. |

### LiveKit **server** (Docker logs)

Rust lines such as `failed to negotiate the publisher` may appear in **`docker compose logs`** during bad ICE / reconnect races; they are **not** the Python `[MEDIA]` prefixes above. Correlate with publisher start/stop on the client.

---

## Troubleshooting (short)

| Symptom | Check |
|---------|--------|
| `ERR_CONNECTION_REFUSED` on `:8080` | GUI not running or `audience.enabled: false`. |
| `ERR_CONNECTION_REFUSED` on `:7880` | `docker compose up -d` and firewall. |
| Phone cannot connect | Same Wi‑Fi, correct LAN IP in `livekit_url` + `--node-ip`, firewall script. |
| Overlapping audio in browser | Ensure a single `narration` element (see `index.html` dedupe by track name); avoid duplicate tabs both unmuted in the same room. |
| `[MEDIA] fps=…` not ~30 in LiveAvatar/VR mode | Current build uses **deadline-based** pacing; sustained **~60+** may indicate an old build or clock skew. |
| Playback stutters when TTS / LiveAvatar is active | **Same asyncio loop** runs VR pacing and WebSocket audio. Heavy **Base64 / `json.dumps`** for `agent.speak` used to block that loop (`liveavatar_session.py` now uses **`asyncio.to_thread`**). Also check client **`JPEG send_queue` full** or **high RAM** (whole GUI starves). VR/avatar **cv2** uses `_vr_executor` / `_avatar_executor`. |
| Stutter with remote inference; server stdout shows `JPEG send_queue=30/30` | TCP/client cannot drain frames as fast as produced; often **high client RAM** or network. Root README → *Client send rate* / `_FRAME_QUEUE_MAX`. |
| PiP lip sync off | Tune `liveavatar.audio_delay_ms` or **`LIVEAVATAR_AUDIO_DELAY_MS`** (mouth **lags** sound → **increase**; sound **lags** mouth → **decrease**). Delay is anchored at TTS chunk dequeue so WebSocket timing does not smear rhythm. |
| `QThread: Destroyed while thread '' is still running` on exit | A background thread (e.g. publisher) may still be stopping; ensure Free Switch / audience teardown completes before closing the app window, or wait for `[MEDIA] publisher stopped`. |

---

## See also

- Root project overview: [README.md](../../../README.md)
- Input workers and `signal_vr_frame`: [workers/README.md](../workers/README.md)
