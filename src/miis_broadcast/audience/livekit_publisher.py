# src/miis_broadcast/audience/livekit_publisher.py
"""
AudiencePublisher
=================
Publishes audio/video to the local LiveKit room for audience viewing.

VR mode (only mode)
-------------------
- Video source: VR frames from FreeSwitchCameraThread.signal_vr_frame
- Audio source: OpenAI TTS PCM sink (push_audio_chunk)
- Track names: broadcast_video (video), narration (audio)
- Avatar: two-state (open/closed) 2D cat avatar composited as PiP in
  the bottom-right corner of each broadcast frame.  Mouth state is derived
  from PCM RMS with a short hangover so the mouth does not flicker.

Heavy cv2 work runs on a dedicated single-worker ThreadPoolExecutor (_vr_executor)
so the asyncio event loop is never blocked by resize / RGBA conversion.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import queue
import threading
import time
from pathlib import Path
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


# ── Avatar constants ────────────────────────────────────────────────────────
_AVATAR_HEIGHT_FRAC = 0.20   # avatar occupies this fraction of frame height
_AVATAR_MARGIN_PX   = 20     # gap from right/bottom edge (pixels)
_RMS_OPEN_THRESHOLD = 300    # int16 RMS above this value → mouth open
_MOUTH_HANGOVER_S   = 0.15   # keep mouth open N seconds after last active chunk


def _normalize_avatar_to_bgra(img: np.ndarray, path: Path) -> np.ndarray:
    """Normalize loaded PNG to BGRA so open/closed use the same alpha compositing."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.ndim == 3:
        channels = img.shape[2]
        if channels == 3:
            h, w = img.shape[:2]
            alpha = np.full((h, w, 1), 255, dtype=img.dtype)
            img = np.concatenate([img, alpha], axis=2)
            print(
                f"{_ts()} | [AVATAR] WARN: no alpha in {path.name}; compositing as opaque "
                "(re-export PNG with transparency for keyed assets)"
            )
        elif channels == 4:
            pass
        else:
            raise ValueError(f"unexpected channel count for avatar: {path}")
    return img


class AudiencePublisher:
    """Connects to a local LiveKit room and publishes broadcast_video + narration tracks.

    A 2D avatar image (two PNG frames: mouth closed / open) is composited into
    the bottom-right corner of every video frame. The open/closed state follows
    the TTS PCM volume via RMS with a short hangover.
    """

    VIDEO_W = 1920
    VIDEO_H = 1080
    AUDIO_SAMPLE_RATE = 24_000
    AUDIO_CHANNELS = 1
    STATS_INTERVAL_S = 2.0

    def __init__(
        self,
        livekit_url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
    ) -> None:
        """
        Args:
            livekit_url: ws(s)://host:port for the local LiveKit server.
            api_key: local LiveKit API key.
            api_secret: local LiveKit API secret.
            room_name: local LiveKit room name.
        """
        self._url = livekit_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._video_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=48)

        self._connected = False

        self._vr_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="audience-vr"
        )

        # Avatar state -- written by push_audio_chunk (any thread), read by
        # _video_pump_vr (asyncio thread).  float assignment is atomic in CPython.
        self._mouth_open_until: float = 0.0

        # Pre-loaded avatar images as BGRA (same format for keyed PNGs).
        self._avatar_closed: Optional[np.ndarray] = None
        self._avatar_open:   Optional[np.ndarray] = None
        self._load_avatar_images()

    # ── Public API (Qt-thread safe) ───────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AudiencePublisher"
        )
        self._thread.start()
        print(
            f"{_ts()} | [MEDIA] publisher starting | room={self._room_name} mode=vr"
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
        """Called from TTS PCM sink callback (any thread).

        Also updates mouth-open state: if the chunk is loud enough (RMS above
        threshold) the mouth stays open until _MOUTH_HANGOVER_S after the last
        active chunk.
        """
        _enqueue_drop_oldest(self._audio_q, pcm_int16)

        if pcm_int16.size > 0:
            rms = float(np.sqrt(np.mean(pcm_int16.astype(np.float32) ** 2)))
            if rms > _RMS_OPEN_THRESHOLD:
                # Extend the deadline; float store is atomic in CPython.
                self._mouth_open_until = time.perf_counter() + _MOUTH_HANGOVER_S

    def flush_pending_audio(self) -> None:
        """Drop buffered PCM not yet sent to LiveKit (call when TTS is interrupted/preempted)."""
        _drain_queue(self._audio_q)
        # Close the mouth immediately on interrupt.
        self._mouth_open_until = 0.0

    # ── Internal ──────────────────────────────────────────────────────────

    def _load_avatar_images(self) -> None:
        """Load and pre-scale the two avatar PNG files once at start-up."""
        asset_dir = Path(__file__).resolve().parents[3] / "assets" / "avatar"
        pairs = [
            (asset_dir / "cat_mouth_shut.png",   "_avatar_closed"),
            (asset_dir / "cat_mouth_opened.png", "_avatar_open"),
        ]
        target_h = max(1, int(self.VIDEO_H * _AVATAR_HEIGHT_FRAC))

        for path, attr in pairs:
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                print(f"{_ts()} | [AVATAR] WARNING: cannot load {path}")
                continue
            img = _normalize_avatar_to_bgra(raw, path)
            h, w = img.shape[:2]
            scale  = target_h / max(h, 1)
            new_w  = max(1, int(w * scale))
            img    = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
            setattr(self, attr, img)

        if self._avatar_closed is not None:
            sz = (self._avatar_closed.shape[1], self._avatar_closed.shape[0])
            print(f"{_ts()} | [AVATAR] loaded | pip_size={sz[0]}x{sz[1]} px")
        else:
            print(f"{_ts()} | [AVATAR] WARNING: avatar images missing; no PiP overlay")

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
        await self._async_main_vr()

    async def _async_main_vr(self) -> None:
        """Publish VR frames + TTS audio to local room."""
        from livekit import rtc

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

    async def _video_pump_vr(self, source) -> None:
        """VR mode: drain _video_q → local VideoSource at a fixed 30 fps.

        Each frame is composited with the avatar PiP in the bottom-right corner.
        The avatar image (open/closed) is selected based on current TTS audio
        activity (mouth_open_until deadline) and passed as a snapshot to the
        thread-pool compositor so the asyncio loop is never blocked by cv2.
        """
        from livekit import rtc

        TARGET_FPS = 30
        frame_interval = 1.0 / TARGET_FPS
        loop = asyncio.get_running_loop()

        fps_count = 0
        fps_ts = time.perf_counter()
        last_frame_rgb: Optional[np.ndarray] = None
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

            # Snapshot mouth state before entering the thread pool.
            is_open    = time.perf_counter() < self._mouth_open_until
            avatar_img = self._avatar_open if is_open else self._avatar_closed

            buf = await loop.run_in_executor(
                self._vr_executor,
                _build_vr_avatar_frame_sync,
                last_frame_rgb,
                avatar_img,
                self.VIDEO_W,
                self.VIDEO_H,
            )
            lk_frame = rtc.VideoFrame(
                width=self.VIDEO_W,
                height=self.VIDEO_H,
                type=rtc.VideoBufferType.RGBA,
                data=buf,
            )
            source.capture_frame(lk_frame)

            fps_count += 1
            now = time.perf_counter()
            if now - fps_ts >= self.STATS_INTERVAL_S:
                fps  = fps_count / (now - fps_ts)
                drop = self._video_q.qsize()
                mouth_state = "open" if is_open else "closed"
                print(f"{_ts()} | [MEDIA] fps={fps:.1f} drop={drop} mouth={mouth_state}")
                fps_count = 0
                fps_ts    = now

            next_deadline += frame_interval
            now = time.perf_counter()
            if now > next_deadline:
                next_deadline = now + frame_interval
            await _async_sleep_until_deadline(next_deadline)

    async def _audio_pump_direct(self, source) -> None:
        """Drain _audio_q → local AudioSource (no delay)."""
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


# ── Module-level frame compositors (run inside _vr_executor) ──────────────


def _build_vr_avatar_frame_sync(
    vr_rgb: np.ndarray,
    avatar_img: Optional[np.ndarray],
    out_w: int,
    out_h: int,
) -> bytearray:
    """Resize VR to output size, overlay avatar PiP in bottom-right corner.

    Avatars are always BGRA after load (keyed PNGs): blend using the alpha channel.

    Runs in a thread-pool so the asyncio loop is never stalled by cv2.
    """
    h, w = vr_rgb.shape[:2]
    canvas: np.ndarray = (
        cv2.resize(vr_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        if (w != out_w or h != out_h)
        else vr_rgb.copy()
    )

    if avatar_img is not None:
        ah, aw = avatar_img.shape[:2]
        x0 = max(0, out_w - aw - _AVATAR_MARGIN_PX)
        y0 = max(0, out_h - ah - _AVATAR_MARGIN_PX)
        x1 = min(out_w, x0 + aw)
        y1 = min(out_h, y0 + ah)
        aw_clip = x1 - x0
        ah_clip = y1 - y0

        avatar_crop = avatar_img[:ah_clip, :aw_clip]

        alpha_raw = avatar_crop[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha_raw[:, :, np.newaxis]
        avatar_bgr = avatar_crop[:, :, :3]

        # Convert avatar BGR → RGB to match canvas colour order.
        avatar_rgb = cv2.cvtColor(avatar_bgr, cv2.COLOR_BGR2RGB)

        roi     = canvas[y0:y1, x0:x1].astype(np.float32)
        blended = roi * (1.0 - alpha) + avatar_rgb.astype(np.float32) * alpha
        canvas[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)

    frame_rgba = cv2.cvtColor(canvas, cv2.COLOR_RGB2RGBA)
    return bytearray(frame_rgba.tobytes())


# ── Token / LiveKit helpers ────────────────────────────────────────────────


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
