# src/miis_broadcast/audience/livekit_publisher.py
"""
AudiencePublisher
=================
Publishes two LiveKit tracks from a background asyncio thread:
  - vr_program  : video (VR frames from FreeSwitchCameraThread.signal_vr_frame)
  - narration   : audio (PCM int16 24 kHz from TTS PCM sink)

Thread model
------------
  Qt main thread  → push_video_frame() / push_audio_chunk()  (non-blocking enqueue)
  asyncio thread  ← drains queues and forwards to LiveKit
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


class AudiencePublisher:
    """Connects to a LiveKit room and publishes vr_program + narration tracks."""

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
    ) -> None:
        self._url = livekit_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Drop-oldest queues so Qt thread never blocks
        self._video_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=3)
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=48)

        self._connected = False

    # ── Public API (Qt-thread safe) ───────────────────────────────────────

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="AudiencePublisher"
        )
        self._thread.start()
        print(f"{_ts()} | [MEDIA] publisher starting | room={self._room_name}")

    def stop(self) -> None:
        # Rely on _stop_event so _async_main can run room.disconnect() while the loop is still
        # valid.
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

    def push_audio_chunk(self, pcm_int16: np.ndarray) -> None:
        """Called from TTS PCM sink callback (any thread)."""
        _enqueue_drop_oldest(self._audio_q, pcm_int16)

    def flush_pending_audio(self) -> None:
        """Drop buffered PCM not yet sent to LiveKit (call when TTS is interrupted / preempted)."""
        try:
            while True:
                self._audio_q.get_nowait()
        except queue.Empty:
            pass

    # ── Internal ──────────────────────────────────────────────────────────

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
        from livekit import rtc
        from livekit.api import AccessToken, VideoGrants

        token = (
            AccessToken(self._api_key, self._api_secret)
            .with_identity("broadcast-publisher")
            .with_name("MIIS Broadcast Publisher")
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=self._room_name,
                    can_publish=True,
                    can_subscribe=False,
                )
            )
            .to_jwt()
        )

        room = rtc.Room()
        try:
            await room.connect(self._url, token)
            self._connected = True
            print(f"{_ts()} | [MEDIA] connected | room={self._room_name}")

            # Video track
            video_source = rtc.VideoSource(self.VIDEO_W, self.VIDEO_H)
            video_track = rtc.LocalVideoTrack.create_video_track("vr_program", video_source)
            await room.local_participant.publish_track(
                video_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_CAMERA),
            )
            print(f"{_ts()} | [MEDIA] publish_start track=vr_program")

            # Audio track
            audio_source = rtc.AudioSource(self.AUDIO_SAMPLE_RATE, self.AUDIO_CHANNELS)
            audio_track = rtc.LocalAudioTrack.create_audio_track("narration", audio_source)
            await room.local_participant.publish_track(
                audio_track,
                rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
            )
            print(f"{_ts()} | [AUDIO] publish_start track=narration")

            await asyncio.gather(
                self._video_pump(video_source),
                self._audio_pump(audio_source),
            )

        except Exception as exc:
            print(f"{_ts()} | [ERR] publisher session: {exc}")
        finally:
            self._connected = False
            try:
                await room.disconnect()
            except Exception as exc:
                print(f"{_ts()} | [MEDIA] publish_stop | disconnect skipped: {exc}")
            else:
                print(f"{_ts()} | [MEDIA] publish_stop | disconnected from LiveKit")

    async def _video_pump(self, source) -> None:
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
                print(f"{_ts()} | [MEDIA] fps={fps:.1f} drop={drop}")
                fps_count = 0
                fps_ts = now

    async def _audio_pump(self, source) -> None:
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
                # Effective sample rate fed to LiveKit this interval
                rate = samples_out / (now - stats_ts)
                drop = self._audio_q.qsize()
                print(
                    f"{_ts()} | [AUDIO] chunks/s={chps:.1f} "
                    f"sample_rate≈{rate:.0f} drop={drop}"
                )
                chunk_count = 0
                samples_out = 0
                stats_ts = now


# ── Helpers ───────────────────────────────────────────────────────────────────

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
