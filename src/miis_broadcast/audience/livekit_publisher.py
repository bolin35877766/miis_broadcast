# src/miis_broadcast/audience/livekit_publisher.py
"""
AudiencePublisher
=================
Publishes audio/video to the local LiveKit room for audience viewing.

VR mode (only mode)
-------------------
- Video source: VR frames from FreeSwitchCameraThread.signal_vr_frame
- Audio source: OpenAI TTS PCM (啟用播報) or desktop/game capture (維持原聲)
- Track names: broadcast_video (video), narration (audio)

Heavy cv2 work runs on a ThreadPoolExecutor (_vr_executor) so the asyncio event
loop is never blocked by resize / RGBA conversion.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import queue
import threading
import time
from typing import Any, List, Optional, Tuple

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


class AudiencePublisher:
    """Connects to a local LiveKit room and publishes broadcast_video + narration tracks."""

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
        self._room: Optional[Any] = None  # livekit.rtc.Room while connected
        self._narration_enabled = True

        # Pipeline: compositing one frame while the previous may still encode.
        self._vr_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="audience-vr"
        )

        # Session totals: periodic [MEDIA]/[AUDIO] stats for end-of-broadcast averages.
        self._telemetry_lock = threading.Lock()
        self._telemetry_video_samples: List[Tuple[float, float]] = []
        self._telemetry_audio_samples: List[Tuple[float, float, float]] = []

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

    def reset_session_telemetry(self) -> None:
        """Clear accumulated [MEDIA]/[AUDIO] samples (call when a new broadcast starts)."""
        with self._telemetry_lock:
            self._telemetry_video_samples.clear()
            self._telemetry_audio_samples.clear()

    def consume_session_telemetry_average_lines(self) -> List[str]:
        """
        Return lines for averages over periodic stats this session; clear stored samples.
        """
        with self._telemetry_lock:
            vs = list(self._telemetry_video_samples)
            audi = list(self._telemetry_audio_samples)
            self._telemetry_video_samples.clear()
            self._telemetry_audio_samples.clear()
        lines: List[str] = []
        if vs:
            fps_m = sum(t[0] for t in vs) / len(vs)
            dr_m = sum(t[1] for t in vs) / len(vs)
            lines.append(
                f"[SESSION AVG] [MEDIA] fps={fps_m:.1f} drop={dr_m:.2f} (n={len(vs)})"
            )
        if audi:
            c_m = sum(t[0] for t in audi) / len(audi)
            r_m = sum(t[1] for t in audi) / len(audi)
            aq_m = sum(t[2] for t in audi) / len(audi)
            lines.append(
                f"[SESSION AVG] [AUDIO] chunks/s={c_m:.1f} "
                f"sample_rate≈{r_m:.0f} aq={aq_m:.2f} (n={len(audi)})"
            )
        return lines

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
        """TTS PCM sink callback (any thread). Only while AI narration is on."""
        if not self._narration_enabled:
            return
        _enqueue_drop_oldest(self._audio_q, pcm_int16)

    def push_original_audio_chunk(self, pcm_int16: np.ndarray) -> None:
        """Desktop/game PCM for 維持原聲 (any thread). Ignored while AI narration is on."""
        if self._narration_enabled:
            return
        _enqueue_drop_oldest(self._audio_q, pcm_int16)

    def flush_pending_audio(self) -> None:
        """Drop buffered PCM not yet sent to LiveKit (call when TTS is interrupted/preempted)."""
        _drain_queue(self._audio_q)

    def set_narration_enabled(self, enabled: bool) -> None:
        """Notify audience clients whether the cat avatar / AI narration mode is on.

        Does not change the published video track. Switching modes flushes the audio
        queue so TTS and original desktop audio do not cross-fade into each other;
        clients hide the mascot when narration is off but keep playing the audio track.
        """
        self._narration_enabled = bool(enabled)
        self.flush_pending_audio()
        loop = self._loop
        if loop is None or loop.is_closed() or not self._connected:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self._publish_audience_mode(), loop
            )
        except RuntimeError:
            pass

    async def _publish_audience_mode(self) -> None:
        room = self._room
        if room is None:
            return
        payload = json.dumps(
            {
                "type": "audience_mode",
                "narration_enabled": self._narration_enabled,
            }
        ).encode("utf-8")
        try:
            await room.local_participant.publish_data(
                payload,
                reliable=True,
                topic="audience_mode",
            )
            print(
                f"{_ts()} | [AUDIENCE] mode notify narration_enabled="
                f"{self._narration_enabled}"
            )
        except Exception as exc:
            print(f"{_ts()} | [WARN] [AUDIENCE] mode notify failed: {exc}")

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
        await self._async_main_vr()

    async def _async_main_vr(self) -> None:
        """Publish VR frames + TTS audio to local room."""
        from livekit import rtc

        token = _make_publisher_token(
            self._api_key, self._api_secret, self._room_name
        )

        room = rtc.Room()
        self._room = room
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

            # Sync current operator mode to any viewers already in the room.
            await self._publish_audience_mode()

            await asyncio.gather(
                self._video_pump_vr(video_source),
                self._audio_pump_direct(audio_source),
            )

        except Exception as exc:
            print(f"{_ts()} | [ERR] publisher session (vr): {exc}")
        finally:
            self._connected = False
            self._room = None
            await _safe_disconnect(room)

    async def _video_pump_vr(self, source) -> None:
        """VR mode: drain _video_q → local VideoSource at a fixed 30 fps."""
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

            buf = await loop.run_in_executor(
                self._vr_executor,
                _build_vr_frame_rgba_sync,
                last_frame_rgb,
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
                fps = fps_count / (now - fps_ts)
                drop = self._video_q.qsize()
                print(f"{_ts()} | [MEDIA] fps={fps:.1f} drop={drop}")
                with self._telemetry_lock:
                    self._telemetry_video_samples.append((float(fps), float(drop)))
                fps_count = 0
                fps_ts = now

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
                with self._telemetry_lock:
                    self._telemetry_audio_samples.append(
                        (float(chps), float(rate), float(drop))
                    )
                chunk_count = 0
                samples_out = 0
                stats_ts = now


# ── Module-level frame helper (runs inside _vr_executor) ──────────────────


def _build_vr_frame_rgba_sync(vr_rgb: np.ndarray, out_w: int, out_h: int) -> bytearray:
    """Resize VR RGB frame to output size and pack as RGBA for LiveKit VideoFrame."""
    h, w = vr_rgb.shape[:2]
    if w != out_w or h != out_h:
        canvas = cv2.resize(vr_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    else:
        canvas = vr_rgb

    frame_rgba = np.empty((out_h, out_w, 4), dtype=np.uint8)
    frame_rgba[:, :, :3] = canvas
    frame_rgba[:, :, 3] = 255
    return bytearray(frame_rgba.data)


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
