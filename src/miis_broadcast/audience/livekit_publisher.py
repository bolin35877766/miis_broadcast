# src/miis_broadcast/audience/livekit_publisher.py
"""
AudiencePublisher
=================
Publishes audio/video to the local LiveKit room for audience viewing.

Two operating modes
-------------------
1. **VR mode** (default, liveavatar_cfg=None):
   - Video source: VR frames from FreeSwitchCameraThread.signal_vr_frame
   - Audio source: OpenAI TTS PCM sink (push_audio_chunk)
   - Track names: broadcast_video (video), narration (audio)

2. **LiveAvatar LITE mode** (when ``liveavatar_cfg`` is set):
   - Connects to TWO LiveKit rooms simultaneously:
       local_room   → local audience SFU (from configs)
       avatar_room  → LiveAvatar cloud LiveKit (video subscribe only)
   - TTS PCM is sent to LiveAvatar via **WebSocket** ``agent.speak`` (not a LiveKit mic track)
   - The same TTS PCM is also delayed and published as ``narration`` on the local room for lip-sync
   - Local audio gets a configurable delay (``audio_delay_ms``, default ~450 ms) so browser
     narration matches lip motion in the PiP; tune per network / machine.
   - VR video is published at a fixed 30 fps **independently** of the avatar cloud stream.
     When an avatar frame is available it is composited as a bottom-right PiP tile.
     If the cloud stream stalls or has not yet delivered a frame, the VR-only output is
     published without interruption.
   - Track names: broadcast_video (video), narration (audio)

Thread / task model (LiveAvatar mode)
--------------------------------------
  Qt main thread  → push_video_frame() / push_audio_chunk()   (non-blocking enqueue)
  asyncio thread  ← four fully independent concurrent tasks:
    _video_pump_vr_pip        – fixed 30 fps VR + optional avatar PiP (never waits for cloud)
    _avatar_frame_reader_task – cloud frames → pre-scaled PiP tile cache (decodes off hot path)
    _audio_pump_liveavatar    – TTS PCM → WebSocket agent.speak + delay queue
    _audio_delay_relay        – delayed PCM → local narration AudioSource

  Thread executors (two DEDICATED single-worker pools, never share threads):
    _vr_executor     – used exclusively by _video_pump_vr / _video_pump_vr_pip for composite
    _avatar_executor – used exclusively by _avatar_frame_reader_task for decode + prescale
    Isolation guarantee: active TTS (avatar rendering at full fps) never delays VR compositing.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np


def _ts() -> str:
    return time.strftime("%H:%M:%S")


async def _async_sleep_until_deadline(deadline: float) -> None:
    """Block until time.perf_counter() >= deadline.

    Loops asyncio.sleep(remaining) because a single sleep may wake early on Windows,
    which previously pushed video pumps to ~60+ fps instead of the target rate.
    """
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        await asyncio.sleep(remaining)


_PIP_MAX_FRAC = 0.30
_PIP_MARGIN_PX = 10


class AudiencePublisher:
    """Connects to a local LiveKit room and publishes broadcast_video + narration tracks.

    When liveavatar_cfg is supplied the publisher manages a **LiveAvatar LITE** session (REST +
    WebSocket audio), subscribes to the cloud avatar video, and composites PiP onto the VR frame
    for the local audience room.  VR publishing is fully independent of avatar availability.
    """

    VIDEO_W = 640
    VIDEO_H = 480
    AUDIO_SAMPLE_RATE = 24_000
    AUDIO_CHANNELS = 1
    STATS_INTERVAL_S = 2.0

    def __init__(
        self,
        livekit_url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        liveavatar_cfg: Optional[dict] = None,
    ) -> None:
        """
        Args:
            livekit_url: ws(s)://host:port for the local LiveKit server.
            api_key: local LiveKit API key.
            api_secret: local LiveKit API secret.
            room_name: local LiveKit room name.
            liveavatar_cfg: dict with keys:
                api_key   (str)  – LiveAvatar API key (app.liveavatar.com/developers)
                avatar_id (str)  – LiveAvatar avatar UUID (required)
                quality   (str)  – video quality hint ("low" | "medium" | "high")
                sandbox   (bool) – optional; maps to token ``is_sandbox``
            Pass None to fall back to VR-frame video mode.
        """
        self._url = livekit_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._liveavatar_cfg: Optional[dict] = liveavatar_cfg

        # Local audience audio delay in LiveAvatar mode (seconds); from configs/app.yml audio_delay_ms
        self._local_audio_delay_s = 0.45
        if liveavatar_cfg is not None:
            self._local_audio_delay_s = max(
                0.0, float(liveavatar_cfg.get("audio_delay_ms", 450)) / 1000.0
            )

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # VR frame queue (both modes)
        self._video_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)

        # TTS PCM queue – LiveAvatar WebSocket + delayed local narration relay
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=48)

        # asyncio queue for the delayed local-audio relay (populated from _audio_pump)
        self._local_audio_delay_q: Optional[asyncio.Queue] = None

        self._liveavatar_session = None  # LiveAvatarSession | None

        # Latest VR RGB frame cached for compositing (updated from _video_q)
        self._liveavatar_last_vr_rgb: Optional[np.ndarray] = None

        # Pre-scaled PiP tile (BGR) + rect from cloud avatar (updated by _avatar_frame_reader_task)
        self._liveavatar_pip_tile: Optional[
            tuple[np.ndarray, int, int, int, int]
        ] = None

        self._connected = False

        # Dedicated single-worker thread executors (see module docstring).
        self._vr_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="audience-vr"
        )
        self._avatar_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="audience-avatar"
        )

        # Prevents duplicate agent.interrupt calls when flush_pending_audio is
        # called rapidly from multiple Qt signals within the same ~500 ms window.
        self._interrupt_in_flight = False
        self._interrupt_lock = threading.Lock()

    # ── Public API (Qt-thread safe) ───────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        if self._liveavatar_cfg:
            self._liveavatar_last_vr_rgb = None
            self._liveavatar_pip_tile = None
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AudiencePublisher"
        )
        self._thread.start()
        mode = "liveavatar" if self._liveavatar_cfg else "vr"
        print(
            f"{_ts()} | [MEDIA] publisher starting | "
            f"room={self._room_name} mode={mode}"
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=15)
        if self._thread and self._thread.is_alive():
            print(f"{_ts()} | [WARN] [MEDIA] publisher thread hung; forcing event loop stop")
            if self._loop and not self._loop.is_closed():
                try:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                except RuntimeError:
                    pass
            self._thread.join(timeout=5)
        self._connected = False
        self._vr_executor.shutdown(wait=False, cancel_futures=True)
        self._avatar_executor.shutdown(wait=False, cancel_futures=True)
        print(f"{_ts()} | [MEDIA] publisher stopped")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def push_video_frame(self, frame_rgb: np.ndarray) -> None:
        """Called from Qt thread via signal_vr_frame."""
        _enqueue_drop_oldest(self._video_q, frame_rgb)

    def _drain_video_q_latest(self) -> Optional[np.ndarray]:
        """Pop all pending VR frames and return the newest (rest dropped)."""
        last: Optional[np.ndarray] = None
        try:
            while True:
                last = self._video_q.get_nowait()
        except queue.Empty:
            pass
        return last

    def push_audio_chunk(self, pcm_int16: np.ndarray) -> None:
        """Called from TTS PCM sink callback (any thread)."""
        _enqueue_drop_oldest(self._audio_q, pcm_int16)

    def flush_pending_audio(self) -> None:
        """Drop buffered PCM not yet sent to LiveKit (call when TTS is interrupted/preempted).

        In LiveAvatar mode this also:
        - clears the local-audio delay buffer
        - sends a short silence chunk over the events WebSocket (optional mouth settle)
        - sends `agent.interrupt` on that WebSocket (best-effort)
        """
        _drain_queue(self._audio_q)

        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._flush_delay_queue_sync)

        if self._liveavatar_cfg and self._loop and not self._loop.is_closed():
            with self._interrupt_lock:
                if not self._interrupt_in_flight:
                    self._interrupt_in_flight = True
                    self._loop.call_soon_threadsafe(
                        lambda: asyncio.ensure_future(self._liveavatar_interrupt_async())
                    )

    # ── Internal ──────────────────────────────────────────────────────────

    def _flush_delay_queue_sync(self) -> None:
        """Clear the asyncio delay queue from the asyncio thread."""
        q = self._local_audio_delay_q
        if q is None:
            return
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _liveavatar_interrupt_async(self) -> None:
        """Send a short silence chunk + `agent.interrupt` on LiveAvatar WebSocket."""
        session = self._liveavatar_session
        if session is None:
            with self._interrupt_lock:
                self._interrupt_in_flight = False
            return
        try:
            await session.send_silence()
        except Exception as exc:
            print(f"{_ts()} | [WARN] [LIVEAVATAR] silence flush: {exc}")
        try:
            await session.interrupt()
        finally:
            with self._interrupt_lock:
                self._interrupt_in_flight = False

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:
            print(f"{_ts()} | [ERR] AudiencePublisher loop: {exc}")
        finally:
            self._loop.close()

    async def _async_main(self) -> None:
        if self._liveavatar_cfg:
            await self._async_main_liveavatar()
        else:
            await self._async_main_vr()

    # ── VR mode ───────────────────────────────────────────────────────────

    async def _async_main_vr(self) -> None:
        """Original VR-frame mode: publish VR frames + TTS audio to local room."""
        from livekit import rtc
        from livekit.api import AccessToken, VideoGrants

        token = _make_publisher_token(
            self._api_key, self._api_secret, self._room_name
        )

        room = rtc.Room()
        try:
            await room.connect(self._url, token)
            self._connected = True
            print(f"{_ts()} | [MEDIA] connected | room={self._room_name}")

            video_source = rtc.VideoSource(self.VIDEO_W, self.VIDEO_H)
            video_track = rtc.LocalVideoTrack.create_video_track(
                "broadcast_video", video_source
            )
            await room.local_participant.publish_track(
                video_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
            )
            print(f"{_ts()} | [MEDIA] publish_start track=broadcast_video (vr mode)")

            audio_source = rtc.AudioSource(self.AUDIO_SAMPLE_RATE, self.AUDIO_CHANNELS)
            audio_track = rtc.LocalAudioTrack.create_audio_track(
                "narration", audio_source
            )
            await room.local_participant.publish_track(
                audio_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            print(f"{_ts()} | [AUDIO] publish_start track=narration (vr mode)")

            await asyncio.gather(
                self._video_pump_vr(video_source),
                self._audio_pump_direct(audio_source),
            )

        except Exception as exc:
            print(f"{_ts()} | [ERR] publisher session (vr): {exc}")
        finally:
            self._connected = False
            await _safe_disconnect(room)

    # ── LiveAvatar LITE mode ───────────────────────────────────────────────

    async def _async_main_liveavatar(self) -> None:
        """LiveAvatar LITE: four fully independent tasks running in parallel.

        Task layout
        -----------
        _video_pump_vr_pip        VR at fixed 30 fps; overlays avatar PiP if cache is populated.
                                  Starts immediately – never blocked by cloud avatar availability.
        _avatar_frame_reader_task Subscribes to cloud avatar LiveKit room and continuously
                                  updates self._liveavatar_pip_tile.  Failure or stall
                                  has zero impact on VR publishing.
        _audio_pump_liveavatar    Drains TTS PCM queue → WebSocket agent.speak + delay queue.
        _audio_delay_relay        Forwards delayed PCM to local narration AudioSource.
        """
        from livekit import rtc
        from .liveavatar_session import LiveAvatarSession

        cfg = self._liveavatar_cfg or {}
        session = LiveAvatarSession(
            api_key=cfg.get("api_key", ""),
            avatar_id=cfg.get("avatar_id", ""),
            voice_id=cfg.get("voice_id", ""),
            quality=cfg.get("quality", "medium"),
            sandbox=bool(cfg.get("sandbox", False)),
        )
        self._liveavatar_session = session

        try:
            await session.create()
            await session.start()
        except Exception as exc:
            print(f"{_ts()} | [ERR] [LIVEAVATAR] failed to start session: {exc}")
            return

        avatar_room_url = session.room_url
        avatar_room_token = session.access_token

        local_token = _make_publisher_token(
            self._api_key, self._api_secret, self._room_name
        )
        local_room = rtc.Room()
        avatar_room = rtc.Room()

        self._local_audio_delay_q = asyncio.Queue()

        tasks: list[asyncio.Task] = []
        try:
            await local_room.connect(self._url, local_token)
            print(f"{_ts()} | [MEDIA] local_room connected | room={self._room_name}")

            local_video_source = rtc.VideoSource(self.VIDEO_W, self.VIDEO_H)
            local_video_track = rtc.LocalVideoTrack.create_video_track(
                "broadcast_video", local_video_source
            )
            await local_room.local_participant.publish_track(
                local_video_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
            )
            print(f"{_ts()} | [MEDIA] publish_start track=broadcast_video (liveavatar mode)")

            local_audio_source = rtc.AudioSource(
                self.AUDIO_SAMPLE_RATE, self.AUDIO_CHANNELS
            )
            local_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "narration", local_audio_source
            )
            await local_room.local_participant.publish_track(
                local_audio_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            print(f"{_ts()} | [AUDIO] publish_start track=narration (liveavatar mode)")

            await avatar_room.connect(avatar_room_url, avatar_room_token)
            print(f"{_ts()} | [LIVEAVATAR] avatar_room connected | url={avatar_room_url}")

            self._connected = True

            # All four tasks are independent; cancellation is handled in the finally block.
            tasks = [
                asyncio.create_task(
                    self._video_pump_vr_pip(local_video_source),
                    name="vr_pip",
                ),
                asyncio.create_task(
                    self._avatar_frame_reader_task(avatar_room),
                    name="avatar_reader",
                ),
                asyncio.create_task(
                    self._audio_pump_liveavatar(session),
                    name="audio_pump",
                ),
                asyncio.create_task(
                    self._audio_delay_relay(local_audio_source),
                    name="audio_relay",
                ),
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as exc:
            print(f"{_ts()} | [ERR] publisher session (liveavatar): {exc}")
        finally:
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            self._connected = False
            await _safe_disconnect(local_room)
            await _safe_disconnect(avatar_room)
            await session.stop()
            self._liveavatar_session = None

    # ── Pump coroutines ───────────────────────────────────────────────────

    async def _video_pump_vr(self, source) -> None:
        """VR mode: drain _video_q → local VideoSource at a fixed 30 fps.

        cv2 resize + RGBA convert run in a thread-pool executor so the asyncio
        event loop is never blocked by CPU-bound image operations.
        """
        from livekit import rtc

        TARGET_FPS = 30
        frame_interval = 1.0 / TARGET_FPS
        loop = asyncio.get_running_loop()

        fps_count = 0
        skip_count = 0
        fps_ts = time.perf_counter()
        last_frame_rgb: Optional[np.ndarray] = None
        last_lk_frame = None
        next_deadline = time.perf_counter()

        while not self._stop_event.is_set():
            latest = self._drain_video_q_latest()
            if latest is not None:
                last_frame_rgb = latest

            if last_frame_rgb is None:
                next_deadline += frame_interval
                now = time.perf_counter()
                if now > next_deadline:
                    next_deadline = now + frame_interval
                await _async_sleep_until_deadline(next_deadline)
                continue

            behind = time.perf_counter() - next_deadline
            if behind > frame_interval and last_lk_frame is not None:
                source.capture_frame(last_lk_frame)
                skip_count += 1
            else:
                # Dedicated VR executor – never shares threads with avatar decode.
                buf = await loop.run_in_executor(
                    self._vr_executor, _build_vr_only_frame_sync,
                    last_frame_rgb, self.VIDEO_W, self.VIDEO_H,
                )
                last_lk_frame = rtc.VideoFrame(
                    width=self.VIDEO_W,
                    height=self.VIDEO_H,
                    type=rtc.VideoBufferType.RGBA,
                    data=buf,
                )
                source.capture_frame(last_lk_frame)

            fps_count += 1
            now = time.perf_counter()
            if now - fps_ts >= self.STATS_INTERVAL_S:
                fps = fps_count / (now - fps_ts)
                drop = self._video_q.qsize()
                print(
                    f"{_ts()} | [MEDIA] fps={fps:.1f} drop={drop} (vr)"
                    + (f" skip={skip_count}" if skip_count else "")
                )
                fps_count = 0
                skip_count = 0
                fps_ts = now

            next_deadline += frame_interval
            now = time.perf_counter()
            if now > next_deadline:
                next_deadline = now + frame_interval
            await _async_sleep_until_deadline(next_deadline)

    async def _video_pump_vr_pip(self, local_video_source) -> None:
        """LiveAvatar mode: VR at fixed 30 fps with optional avatar PiP overlay.

        cv2 composite + RGBA convert run in a thread-pool executor so the asyncio
        event loop is never stalled by CPU-bound image operations.  VR is published
        regardless of avatar cloud stream availability.
        """
        from livekit import rtc

        TARGET_FPS = 30
        frame_interval = 1.0 / TARGET_FPS
        loop = asyncio.get_running_loop()

        fps_count = 0
        skip_count = 0
        fps_ts = time.perf_counter()
        next_deadline = time.perf_counter()

        # Cache the last rendered RGBA buffer so we can re-publish it without
        # going through the executor when the machine is behind schedule.
        last_buf: bytearray | None = None
        last_lk_frame: "rtc.VideoFrame | None" = None

        while not self._stop_event.is_set():
            latest_vr = self._drain_video_q_latest()
            if latest_vr is not None:
                self._liveavatar_last_vr_rgb = latest_vr

            vr_rgb = self._liveavatar_last_vr_rgb
            if vr_rgb is None:
                next_deadline += frame_interval
                now = time.perf_counter()
                if now > next_deadline:
                    next_deadline = now + frame_interval
                await _async_sleep_until_deadline(next_deadline)
                continue

            # Snapshot pip tile before await (another task may update it during await).
            pip_snapshot = self._liveavatar_pip_tile
            pip_state = "pip" if pip_snapshot is not None else "vr-only"

            # If we are already more than one frame behind, skip the expensive
            # executor composite and re-publish the last buffer instead.
            # This prevents the executor queue from backing up when the OS is
            # under memory/CPU pressure (e.g. JPEG send_queue full, RAM ~86%).
            behind = time.perf_counter() - next_deadline
            if behind > frame_interval and last_lk_frame is not None:
                local_video_source.capture_frame(last_lk_frame)
                skip_count += 1
            else:
                # Dedicated VR executor – never shares threads with avatar decode.
                buf = await loop.run_in_executor(
                    self._vr_executor, _build_vr_pip_frame_sync,
                    vr_rgb, pip_snapshot, self.VIDEO_W, self.VIDEO_H,
                )
                last_buf = buf
                last_lk_frame = rtc.VideoFrame(
                    width=self.VIDEO_W,
                    height=self.VIDEO_H,
                    type=rtc.VideoBufferType.RGBA,
                    data=buf,
                )
                local_video_source.capture_frame(last_lk_frame)

            fps_count += 1
            now = time.perf_counter()
            if now - fps_ts >= self.STATS_INTERVAL_S:
                fps = fps_count / (now - fps_ts)
                vr_drop = self._video_q.qsize()
                print(
                    f"{_ts()} | [MEDIA] fps={fps:.1f} ({pip_state}) vr_q={vr_drop}"
                    + (f" skip={skip_count}" if skip_count else "")
                )
                fps_count = 0
                skip_count = 0
                fps_ts = now

            next_deadline += frame_interval
            now = time.perf_counter()
            if now > next_deadline:
                next_deadline = now + frame_interval
            await _async_sleep_until_deadline(next_deadline)

    async def _avatar_frame_reader_task(self, avatar_room) -> None:
        """Subscribe to cloud avatar LiveKit video and update self._liveavatar_pip_tile.

        This task runs independently of VR publishing.  If the cloud stream stalls,
        delivers corrupt frames, or disconnects, the exception is caught and logged;
        the VR compositor continues to run using the last cached tile (or no PiP).
        """
        from livekit import rtc

        video_stream = None
        track_found = asyncio.Event()

        def _on_track_subscribed(track, pub, participant):
            nonlocal video_stream
            if track.kind == rtc.TrackKind.KIND_VIDEO and video_stream is None:
                print(
                    f"{_ts()} | [LIVEAVATAR] avatar video track subscribed | "
                    f"participant={participant.identity}"
                )
                video_stream = rtc.VideoStream(
                    track, format=rtc.VideoBufferType.RGB24
                )
                track_found.set()

        avatar_room.on("track_subscribed", _on_track_subscribed)

        try:
            await asyncio.wait_for(track_found.wait(), timeout=30)
        except asyncio.TimeoutError:
            print(f"{_ts()} | [WARN] [LIVEAVATAR] avatar video track not received in 30s; PiP disabled")
            return

        print(f"{_ts()} | [LIVEAVATAR] avatar frame reader started")

        loop = asyncio.get_running_loop()
        try:
            async for frame_event in video_stream:
                if self._stop_event.is_set():
                    break
                raw_frame = frame_event.frame
                try:
                    if raw_frame.type == rtc.VideoBufferType.RGB24:
                        frame_src = raw_frame
                    else:
                        frame_src = raw_frame.convert(rtc.VideoBufferType.RGB24)
                    h, w = frame_src.height, frame_src.width
                    # Copy data to bytes BEFORE awaiting so the livekit frame
                    # object can be released safely during the executor call.
                    raw_bytes = bytes(frame_src.data)
                    # Dedicated avatar executor – never shares threads with VR composite.
                    tile = await loop.run_in_executor(
                        self._avatar_executor, _decode_avatar_frame_sync,
                        raw_bytes, h, w,
                        self.VIDEO_W, self.VIDEO_H, _PIP_MAX_FRAC, _PIP_MARGIN_PX,
                    )
                    if tile is not None:
                        self._liveavatar_pip_tile = tile
                except Exception as exc:
                    print(f"{_ts()} | [WARN] [LIVEAVATAR] avatar frame decode: {exc}")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            print(f"{_ts()} | [WARN] [LIVEAVATAR] avatar frame reader stopped: {exc}")

    async def _audio_pump_direct(self, source) -> None:
        """VR mode: drain _audio_q → local AudioSource (no delay)."""
        from livekit import rtc

        chunk_count = 0
        samples_out = 0
        stats_ts = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                pcm = self._audio_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            samples = len(pcm)
            lk_frame = rtc.AudioFrame(
                data=bytearray(pcm.tobytes()),
                sample_rate=self.AUDIO_SAMPLE_RATE,
                num_channels=self.AUDIO_CHANNELS,
                samples_per_channel=samples,
            )
            await source.capture_frame(lk_frame)

            chunk_count += 1
            samples_out += samples
            now = time.perf_counter()
            if now - stats_ts >= self.STATS_INTERVAL_S:
                chps = chunk_count / (now - stats_ts)
                rate = samples_out / (now - stats_ts)
                drop = self._audio_q.qsize()
                print(
                    f"{_ts()} | [AUDIO] chunks/s={chps:.1f} "
                    f"sample_rate≈{rate:.0f} aq={drop}"
                )
                chunk_count = 0
                samples_out = 0
                stats_ts = now

    async def _audio_pump_liveavatar(self, avatar_session) -> None:
        """LiveAvatar mode: TTS PCM → WebSocket `agent.speak` + _local_audio_delay_q."""
        chunk_count = 0
        samples_out = 0
        stats_ts = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                pcm = self._audio_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue

            try:
                await avatar_session.send_pcm_chunk(pcm)
            except Exception as exc:
                print(f"{_ts()} | [WARN] [LIVEAVATAR] audio send: {exc}")

            deadline = time.perf_counter() + self._local_audio_delay_s
            if self._local_audio_delay_q is not None:
                try:
                    self._local_audio_delay_q.put_nowait((deadline, pcm))
                except asyncio.QueueFull:
                    pass

            chunk_count += 1
            samples_out += len(pcm)
            now = time.perf_counter()
            if now - stats_ts >= self.STATS_INTERVAL_S:
                chps = chunk_count / (now - stats_ts)
                rate = samples_out / (now - stats_ts)
                drop = self._audio_q.qsize()
                print(
                    f"{_ts()} | [AUDIO] chunks/s={chps:.1f} "
                    f"sample_rate≈{rate:.0f} aq={drop} (liveavatar)"
                )
                chunk_count = 0
                samples_out = 0
                stats_ts = now

    async def _audio_delay_relay(self, local_audio_source) -> None:
        """Drain _local_audio_delay_q respecting the per-frame deadline timestamp.

        Each item is (deadline: float, pcm: np.ndarray).
        We sleep until deadline, then forward to the local audience AudioSource.
        This creates the configurable delay (``audio_delay_ms``) that aligns local narration
        with lip motion in the PiP stream.
        """
        from livekit import rtc

        while not self._stop_event.is_set():
            if self._local_audio_delay_q is None or self._local_audio_delay_q.empty():
                await asyncio.sleep(0.005)
                continue

            deadline, pcm = await self._local_audio_delay_q.get()
            await _async_sleep_until_deadline(deadline)

            if self._stop_event.is_set():
                break

            samples = len(pcm)
            lk_frame = rtc.AudioFrame(
                data=bytearray(pcm.tobytes()),
                sample_rate=self.AUDIO_SAMPLE_RATE,
                num_channels=self.AUDIO_CHANNELS,
                samples_per_channel=samples,
            )
            try:
                await local_audio_source.capture_frame(lk_frame)
            except Exception as exc:
                print(f"{_ts()} | [WARN] [MEDIA] local audio relay: {exc}")


# ── Module-level helpers ───────────────────────────────────────────────────────


def _decode_avatar_frame_sync(
    raw_bytes: bytes,
    h: int,
    w: int,
    out_w: int,
    out_h: int,
    max_frac: float,
    margin: int,
):
    """Decode raw RGB24 bytes and pre-scale to PiP tile.  Runs in thread pool."""
    img_data = np.frombuffer(raw_bytes, dtype=np.uint8)
    if img_data.size != h * w * 3:
        return None
    img_rgb = img_data.reshape(h, w, 3)
    avatar_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return _make_pip_tile(avatar_bgr, out_w, out_h, max_frac=max_frac, margin=margin)


def _build_vr_pip_frame_sync(
    vr_rgb: np.ndarray,
    pip,
    out_w: int,
    out_h: int,
) -> bytearray:
    """Composite VR + optional pre-scaled PiP → RGBA bytearray.  Runs in thread pool."""
    if pip is not None:
        pip_bgr, x0, y0, nw, nh = pip
        out_rgb = _composite_vr_with_pip_tile(
            vr_rgb, pip_bgr, x0, y0, nw, nh, out_w, out_h
        )
    else:
        out_rgb = cv2.resize(vr_rgb, (out_w, out_h))
    frame_rgba = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2RGBA)
    return bytearray(frame_rgba.tobytes())


def _build_vr_only_frame_sync(vr_rgb: np.ndarray, out_w: int, out_h: int) -> bytearray:
    """Resize + RGB→RGBA convert for VR-only mode.  Runs in thread pool."""
    h, w = vr_rgb.shape[:2]
    frame_rgb = cv2.resize(vr_rgb, (out_w, out_h)) if (w != out_w or h != out_h) else vr_rgb
    frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)
    return bytearray(frame_rgba.tobytes())


def _make_pip_tile(
    avatar_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    max_frac: float,
    margin: int,
) -> tuple[np.ndarray, int, int, int, int]:
    """Scale avatar to PiP size; return (pip_bgr, x0, y0, new_w, new_h)."""
    ah, aw = avatar_bgr.shape[:2]
    max_pw = max(1, int(out_w * max_frac))
    max_ph = max(1, int(out_h * max_frac))
    scale = min(max_pw / aw, max_ph / ah)
    new_w = max(1, int(aw * scale))
    new_h = max(1, int(ah * scale))
    pip = cv2.resize(avatar_bgr, (new_w, new_h))
    x0 = max(0, out_w - new_w - margin)
    y0 = max(0, out_h - new_h - margin)
    return pip, x0, y0, new_w, new_h


def _composite_vr_with_pip_tile(
    vr_rgb: np.ndarray,
    pip_bgr: np.ndarray,
    x0: int,
    y0: int,
    new_w: int,
    new_h: int,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    """Full-frame VR with a pre-scaled PiP (RGB uint8)."""
    bg = cv2.resize(vr_rgb, (out_w, out_h))
    canvas_bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    canvas_bgr[y0 : y0 + new_h, x0 : x0 + new_w] = pip_bgr
    cv2.rectangle(
        canvas_bgr,
        (x0 - 1, y0 - 1),
        (x0 + new_w, y0 + new_h),
        (40, 40, 40),
        2,
    )
    return cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)


def _make_publisher_token(api_key: str, api_secret: str, room_name: str) -> str:
    from livekit.api import AccessToken, VideoGrants

    return (
        AccessToken(api_key, api_secret)
        .with_identity("broadcast-publisher")
        .with_name("MIIS Broadcast Publisher")
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=False,
            )
        )
        .to_jwt()
    )


async def _safe_disconnect(room) -> None:
    try:
        await room.disconnect()
    except Exception as exc:
        print(f"{_ts()} | [MEDIA] disconnect skipped: {exc}")
    else:
        print(f"{_ts()} | [MEDIA] disconnected from LiveKit")


def _enqueue_drop_oldest(q: queue.Queue, item) -> None:
    """Put item; if full, drop the oldest entry first."""
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass


def _drain_queue(q: queue.Queue) -> None:
    try:
        while True:
            q.get_nowait()
    except queue.Empty:
        pass
