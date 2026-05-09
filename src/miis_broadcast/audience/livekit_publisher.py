# src/miis_broadcast/audience/livekit_publisher.py
"""
AudiencePublisher
=================
Publishes audio/video to the local LiveKit room for audience viewing.

Two operating modes
-------------------
1. **VR mode** (default, heygen_cfg=None):
   - Video source: VR frames from FreeSwitchCameraThread.signal_vr_frame
   - Audio source: OpenAI TTS PCM sink (push_audio_chunk)
   - Track names: broadcast_video (video), narration (audio)

2. **HeyGen avatar mode** (heygen_cfg provided):
   - Connects to TWO LiveKit rooms simultaneously:
       local_room  → ws://192.168.50.150:7880  (audience-facing)
       heygen_room → HeyGen cloud LiveKit URL
   - TTS PCM is forwarded to BOTH rooms (drives avatar mouth + local audience hears speech)
   - Local audio gets a ~300 ms delay buffer to align lip sync with cloud video round-trip
   - Each outgoing video frame composites the **VR program** (full frame from ``_video_q``)
     with the **HeyGen avatar** in a **bottom-right picture-in-picture** tile, then publishes
     to local_room as ``broadcast_video``.
   - Track names: broadcast_video (video), narration (audio)

Thread model
------------
  Qt main thread  → push_video_frame() / push_audio_chunk()  (non-blocking enqueue)
  asyncio thread  ← drains queues, manages rooms, pumps A/V
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from typing import Optional

import cv2
import numpy as np


def _ts() -> str:
    return time.strftime("%H:%M:%S")


_PIP_MAX_FRAC = 0.30
_PIP_MARGIN_PX = 10


class AudiencePublisher:
    """Connects to a local LiveKit room and publishes broadcast_video + narration tracks.

    When heygen_cfg is supplied the publisher also manages a HeyGen cloud session,
    routing TTS audio to the cloud avatar and relaying the rendered avatar (picture-in-picture
    on the VR frame) to the local room for the audience.
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
        heygen_cfg: Optional[dict] = None,
    ) -> None:
        """
        Args:
            livekit_url: ws(s)://host:port for the local LiveKit server.
            api_key: local LiveKit API key.
            api_secret: local LiveKit API secret.
            room_name: local LiveKit room name.
            heygen_cfg: dict with keys:
                api_key   (str)  – HeyGen API key
                avatar_id (str)  – HeyGen avatar ID (empty = use default)
                quality   (str)  – "medium" | "high"
            Pass None to fall back to VR-frame video mode.
        """
        self._url = livekit_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._heygen_cfg: Optional[dict] = heygen_cfg

        # Local audience audio delay in HeyGen mode only (seconds); from configs/app.yml audio_delay_ms
        self._local_audio_delay_s = 0.30
        if heygen_cfg is not None:
            self._local_audio_delay_s = max(
                0.0, float(heygen_cfg.get("audio_delay_ms", 300)) / 1000.0
            )

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # VR frame queue (only used in VR mode)
        self._video_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)

        # TTS PCM queue – fed by push_audio_chunk(), forwarded to HeyGen + local rooms
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=48)

        # asyncio queue for the delayed local-audio relay (populated from _audio_pump)
        self._local_audio_delay_q: Optional[asyncio.Queue] = None

        # Reference kept so flush_pending_audio() can also send silence to HeyGen
        self._heygen_audio_source = None   # livekit.rtc.AudioSource | None
        self._heygen_session = None        # HeyGenSession | None

        # Latest VR RGB frame for HeyGen PiP compositing (updated when _video_q is drained)
        self._heygen_last_vr_rgb: Optional[np.ndarray] = None

        self._connected = False

    # ── Public API (Qt-thread safe) ───────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        if self._heygen_cfg:
            self._heygen_last_vr_rgb = None
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AudiencePublisher"
        )
        self._thread.start()
        mode = "heygen" if self._heygen_cfg else "vr"
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

        In HeyGen mode this also:
        - clears the local-audio delay buffer
        - sends a silence frame to HeyGen so the avatar's mouth closes immediately
        - fires the HeyGen interrupt API (best-effort)
        """
        # Clear the main TTS queue
        _drain_queue(self._audio_q)

        # Clear the asyncio-side delay queue (thread-safe via put_nowait from event loop)
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._flush_delay_queue_sync)

        # Send silence + API interrupt to HeyGen (schedule on the asyncio thread)
        if self._heygen_cfg and self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self._heygen_interrupt_async())
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

    async def _heygen_interrupt_async(self) -> None:
        """Send silence to HeyGen audio source + fire interrupt API."""
        src = self._heygen_audio_source
        session = self._heygen_session
        if session is not None and src is not None:
            try:
                await session.send_silence(src)
            except Exception as exc:
                print(f"{_ts()} | [WARN] [HEYGEN] silence flush: {exc}")
        if session is not None:
            await session.interrupt()

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
        if self._heygen_cfg:
            await self._async_main_heygen()
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

    # ── HeyGen mode ───────────────────────────────────────────────────────

    async def _async_main_heygen(self) -> None:
        """HeyGen avatar mode: dual room + audio delay relay + avatar video relay."""
        from livekit import rtc
        from .heygen_session import HeyGenSession

        cfg = self._heygen_cfg or {}
        session = HeyGenSession(
            api_key=cfg.get("api_key", ""),
            avatar_id=cfg.get("avatar_id", ""),
            voice_id=cfg.get("voice_id", ""),
            quality=cfg.get("quality", "medium"),
        )
        self._heygen_session = session

        # 1. Create HeyGen session to obtain cloud LiveKit URL + token
        try:
            session_info = await session.create()
            await session.start()
        except Exception as exc:
            print(f"{_ts()} | [ERR] [HEYGEN] failed to start session: {exc}")
            return

        heygen_room_url   = session.room_url
        heygen_room_token = session.access_token

        # 2. Connect local room
        local_token = _make_publisher_token(
            self._api_key, self._api_secret, self._room_name
        )
        local_room = rtc.Room()
        heygen_room = rtc.Room()

        # asyncio queue for the delayed local audio relay
        self._local_audio_delay_q = asyncio.Queue()

        try:
            await local_room.connect(self._url, local_token)
            print(f"{_ts()} | [MEDIA] local_room connected | room={self._room_name}")

            # 3. Publish broadcast_video + narration tracks to local room
            local_video_source = rtc.VideoSource(self.VIDEO_W, self.VIDEO_H)
            local_video_track = rtc.LocalVideoTrack.create_video_track(
                "broadcast_video", local_video_source
            )
            await local_room.local_participant.publish_track(
                local_video_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
            )
            print(f"{_ts()} | [MEDIA] publish_start track=broadcast_video (heygen mode)")

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
            print(f"{_ts()} | [AUDIO] publish_start track=narration (heygen mode)")

            # 4. Connect to HeyGen room (subscribe only; we publish audio there)
            await heygen_room.connect(heygen_room_url, heygen_room_token)
            print(f"{_ts()} | [HEYGEN] heygen_room connected | url={heygen_room_url}")

            # Publish audio track to HeyGen room (drives avatar mouth)
            heygen_audio_source = rtc.AudioSource(
                self.AUDIO_SAMPLE_RATE, self.AUDIO_CHANNELS
            )
            heygen_audio_track = rtc.LocalAudioTrack.create_audio_track(
                "tts_audio", heygen_audio_source
            )
            await heygen_room.local_participant.publish_track(
                heygen_audio_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            self._heygen_audio_source = heygen_audio_source
            self._connected = True

            print(f"{_ts()} | [HEYGEN] audio track published to heygen_room")

            # 5. Run all concurrent tasks
            await asyncio.gather(
                # Forward TTS PCM → HeyGen room (immediate) + local delay queue
                self._audio_pump_heygen(heygen_audio_source),
                # Drain delay queue → local room after delay
                self._audio_delay_relay(local_audio_source),
                # Subscribe HeyGen avatar video → republish to local room
                self._video_pump_heygen(heygen_room, local_video_source),
            )

        except Exception as exc:
            print(f"{_ts()} | [ERR] publisher session (heygen): {exc}")
        finally:
            self._connected = False
            self._heygen_audio_source = None
            await _safe_disconnect(local_room)
            await _safe_disconnect(heygen_room)
            await session.stop()
            self._heygen_session = None

    # ── Pump coroutines ───────────────────────────────────────────────────

    async def _video_pump_vr(self, source) -> None:
        """VR mode: drain _video_q → local VideoSource."""
        from livekit import rtc

        fps_count = 0
        fps_ts = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                frame_rgb = self._video_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.01)
                continue

            h, w = frame_rgb.shape[:2]
            if h != self.VIDEO_H or w != self.VIDEO_W:
                frame_rgb = cv2.resize(frame_rgb, (self.VIDEO_W, self.VIDEO_H))

            frame_rgba = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2RGBA)
            lk_frame = rtc.VideoFrame(
                width=self.VIDEO_W,
                height=self.VIDEO_H,
                type=rtc.VideoBufferType.RGBA,
                data=bytearray(frame_rgba.tobytes()),
            )
            source.capture_frame(lk_frame)

            fps_count += 1
            now = time.perf_counter()
            if now - fps_ts >= self.STATS_INTERVAL_S:
                fps = fps_count / (now - fps_ts)
                drop = self._video_q.qsize()
                print(f"{_ts()} | [MEDIA] fps={fps:.1f} drop={drop} (vr)")
                fps_count = 0
                fps_ts = now

    async def _video_pump_heygen(
        self, heygen_room, local_video_source
    ) -> None:
        """HeyGen mode: avatar from cloud + VR from _video_q → composite (PiP) → local room."""
        from livekit import rtc

        fps_count = 0
        fps_ts = time.perf_counter()
        video_stream = None
        track_found = asyncio.Event()

        def _on_track_subscribed(track, pub, participant):
            nonlocal video_stream
            if track.kind == rtc.TrackKind.KIND_VIDEO and video_stream is None:
                print(
                    f"{_ts()} | [HEYGEN] avatar video track subscribed | "
                    f"participant={participant.identity}"
                )
                video_stream = rtc.VideoStream(track)
                track_found.set()

        heygen_room.on(rtc.RoomEvent.TrackSubscribed, _on_track_subscribed)

        # Wait for HeyGen to start sending video (up to 30s)
        try:
            await asyncio.wait_for(track_found.wait(), timeout=30)
        except asyncio.TimeoutError:
            print(f"{_ts()} | [WARN] [HEYGEN] avatar video track not received in 30s")
            return

        print(f"{_ts()} | [HEYGEN] starting avatar video relay → local_room (VR + PiP)")

        async for frame_event in video_stream:
            if self._stop_event.is_set():
                break

            raw_frame = frame_event.frame

            try:
                img_data = np.frombuffer(raw_frame.data, dtype=np.uint8)
                src_h = raw_frame.height
                src_w = raw_frame.width
                buf_type = raw_frame.type

                if buf_type == rtc.VideoBufferType.RGBA:
                    img = img_data.reshape(src_h, src_w, 4)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif buf_type == rtc.VideoBufferType.RGB24:
                    img = img_data.reshape(src_h, src_w, 3)
                    img_bgr = img
                else:
                    img = img_data.reshape(src_h, src_w, 4)
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

                latest_vr = self._drain_video_q_latest()
                if latest_vr is not None:
                    self._heygen_last_vr_rgb = latest_vr
                vr_rgb = self._heygen_last_vr_rgb
                if vr_rgb is None:
                    vr_rgb = np.zeros(
                        (self.VIDEO_H, self.VIDEO_W, 3), dtype=np.uint8
                    )

                out_rgb = _composite_vr_avatar_pip(
                    vr_rgb,
                    img_bgr,
                    self.VIDEO_W,
                    self.VIDEO_H,
                    max_frac=_PIP_MAX_FRAC,
                    margin=_PIP_MARGIN_PX,
                )
                frame_rgba = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2RGBA)
                out_frame = rtc.VideoFrame(
                    width=self.VIDEO_W,
                    height=self.VIDEO_H,
                    type=rtc.VideoBufferType.RGBA,
                    data=bytearray(frame_rgba.tobytes()),
                )
                local_video_source.capture_frame(out_frame)

            except Exception as exc:
                print(f"{_ts()} | [WARN] [HEYGEN] video frame conversion: {exc}")
                continue

            fps_count += 1
            now = time.perf_counter()
            if now - fps_ts >= self.STATS_INTERVAL_S:
                fps = fps_count / (now - fps_ts)
                vr_drop = self._video_q.qsize()
                print(
                    f"{_ts()} | [MEDIA] fps={fps:.1f} (heygen+vr pip) vr_q={vr_drop}"
                )
                fps_count = 0
                fps_ts = now

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
                    f"sample_rate≈{rate:.0f} drop={drop}"
                )
                chunk_count = 0
                samples_out = 0
                stats_ts = now

    async def _audio_pump_heygen(self, heygen_audio_source) -> None:
        """HeyGen mode: TTS PCM → HeyGen AudioSource (immediate) + _local_audio_delay_q.

        The delay queue is consumed by _audio_delay_relay after self._local_audio_delay_s seconds,
        keeping local audience audio in sync with the cloud avatar video.
        """
        from livekit import rtc

        chunk_count = 0
        samples_out = 0
        stats_ts = time.perf_counter()

        while not self._stop_event.is_set():
            try:
                pcm = self._audio_q.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.005)
                continue

            samples = len(pcm)
            lk_frame = rtc.AudioFrame(
                data=bytearray(pcm.tobytes()),
                sample_rate=self.AUDIO_SAMPLE_RATE,
                num_channels=self.AUDIO_CHANNELS,
                samples_per_channel=samples,
            )

            # Send to HeyGen immediately to drive avatar lip sync
            try:
                await heygen_audio_source.capture_frame(lk_frame)
            except Exception as exc:
                print(f"{_ts()} | [WARN] [HEYGEN] audio send: {exc}")

            # Enqueue for delayed local relay (with wall-clock timestamp)
            deadline = time.perf_counter() + self._local_audio_delay_s
            if self._local_audio_delay_q is not None:
                try:
                    self._local_audio_delay_q.put_nowait((deadline, pcm))
                except asyncio.QueueFull:
                    pass  # drop if overflow (shouldn't happen with unlimited queue)

            chunk_count += 1
            samples_out += samples
            now = time.perf_counter()
            if now - stats_ts >= self.STATS_INTERVAL_S:
                chps = chunk_count / (now - stats_ts)
                rate = samples_out / (now - stats_ts)
                drop = self._audio_q.qsize()
                print(
                    f"{_ts()} | [AUDIO] chunks/s={chps:.1f} "
                    f"sample_rate≈{rate:.0f} drop={drop} (heygen)"
                )
                chunk_count = 0
                samples_out = 0
                stats_ts = now

    async def _audio_delay_relay(self, local_audio_source) -> None:
        """Drain _local_audio_delay_q respecting the per-frame deadline timestamp.

        Each item is (deadline: float, pcm: np.ndarray).
        We sleep until deadline, then forward to the local audience AudioSource.
        This creates the ~300 ms delay that aligns lips with the relayed avatar video.
        """
        from livekit import rtc

        while not self._stop_event.is_set():
            if self._local_audio_delay_q is None or self._local_audio_delay_q.empty():
                await asyncio.sleep(0.005)
                continue

            deadline, pcm = await self._local_audio_delay_q.get()
            wait = deadline - time.perf_counter()
            if wait > 0:
                await asyncio.sleep(wait)

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


def _composite_vr_avatar_pip(
    vr_rgb: np.ndarray,
    avatar_bgr: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    max_frac: float,
    margin: int,
) -> np.ndarray:
    """Resize VR to out_w×out_h, place avatar in bottom-right (RGB uint8)."""
    bg = cv2.resize(vr_rgb, (out_w, out_h))
    canvas_bgr = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)

    ah, aw = avatar_bgr.shape[:2]
    max_pw = max(1, int(out_w * max_frac))
    max_ph = max(1, int(out_h * max_frac))
    scale = min(max_pw / aw, max_ph / ah)
    new_w = max(1, int(aw * scale))
    new_h = max(1, int(ah * scale))
    pip = cv2.resize(avatar_bgr, (new_w, new_h))

    x0 = out_w - new_w - margin
    y0 = out_h - new_h - margin
    x0 = max(0, x0)
    y0 = max(0, y0)

    canvas_bgr[y0 : y0 + new_h, x0 : x0 + new_w] = pip
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
