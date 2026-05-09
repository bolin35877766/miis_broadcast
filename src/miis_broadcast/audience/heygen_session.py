# src/miis_broadcast/audience/heygen_session.py
"""
HeyGenSession
=============
Manages a HeyGen Streaming Avatar session lifecycle via the HeyGen REST API.

Responsibilities:
- Create a new session (POST /v1/streaming.new) → get session_id, access_token, url
- Start the session (POST /v1/streaming.start)
- Stop the session  (POST /v1/streaming.stop)
- Interrupt avatar  (POST /v1/streaming.interrupt) + send silence frames

HeyGen API docs: https://docs.heygen.com/reference/streaming-avatar

Usage:
    session = HeyGenSession(api_key="...", avatar_id="...")
    info = await session.create()      # returns {"url": ..., "access_token": ..., "session_id": ...}
    await session.start()
    ...
    await session.send_silence(audio_source)   # stop avatar mouth instantly
    await session.stop()
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


_HEYGEN_BASE = "https://api.heygen.com"

_SILENCE_DURATION_S = 0.1   # seconds of silence per interrupt flush frame
_SILENCE_SAMPLE_RATE = 24_000
_SILENCE_CHANNELS = 1


def _ts() -> str:
    return time.strftime("%H:%M:%S")


class HeyGenSession:
    """Wraps HeyGen Streaming Avatar session API calls (async)."""

    def __init__(
        self,
        api_key: str = "",
        avatar_id: str = "",
        voice_id: str = "",
        quality: str = "medium",
    ) -> None:
        self._api_key = api_key
        self._avatar_id = avatar_id
        self._voice_id = voice_id
        self._quality = quality

        self._session_id: Optional[str] = None
        self._access_token: Optional[str] = None
        self._room_url: Optional[str] = None

    # ── Public props ──────────────────────────────────────────────────────

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def access_token(self) -> Optional[str]:
        return self._access_token

    @property
    def room_url(self) -> Optional[str]:
        return self._room_url

    @property
    def is_ready(self) -> bool:
        return bool(self._session_id and self._access_token and self._room_url)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def create(self) -> dict:
        """POST /v1/streaming.new — create session, return dict with url/token/session_id.

        Returns the raw JSON payload from HeyGen so callers can inspect all fields.
        """
        import httpx

        # HeyGen streaming.new: avatar_id + voice_id are required for many accounts (see HeyGen streaming docs).
        payload: dict = {"quality": self._quality}
        if self._avatar_id:
            payload["avatar_id"] = self._avatar_id
        if self._voice_id:
            payload["voice_id"] = self._voice_id

        print(f"{_ts()} | [HEYGEN] creating session | avatar={self._avatar_id or 'default'}")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{_HEYGEN_BASE}/v1/streaming.new",
                headers=self._headers(),
                json=payload,
            )
            try:
                data = resp.json() if resp.content else {}
            except Exception:
                data = {}

        err = data.get("error")
        if err:
            msg = err if isinstance(err, str) else err.get("message", str(err))
            raise RuntimeError(f"[HEYGEN] streaming.new error: {msg} (HTTP {resp.status_code})")

        if not resp.is_success:
            raise RuntimeError(
                f"[HEYGEN] streaming.new HTTP {resp.status_code}: {data or resp.text[:500]}"
            )

        # HeyGen wraps the result under {"data": {...}, "error": null}
        info = data.get("data") or data
        if not isinstance(info, dict):
            raise RuntimeError(f"[HEYGEN] create: unexpected data shape: {data}")

        self._session_id   = info.get("session_id") or info.get("sessionId")
        self._access_token = (
            info.get("access_token")
            or info.get("accessToken")
            or info.get("token")
        )
        self._room_url = (
            info.get("url")
            or info.get("livekit_url")
            or info.get("livekitUrl")
        )

        if not self.is_ready:
            raise RuntimeError(f"[HEYGEN] create: missing session_id/token/url in response: {data}")

        print(
            f"{_ts()} | [HEYGEN] session created | "
            f"session_id={self._session_id} url={self._room_url}"
        )
        return info

    async def start(self) -> None:
        """POST /v1/streaming.start — tell HeyGen to start rendering the avatar."""
        import httpx

        if not self._session_id:
            raise RuntimeError("[HEYGEN] call create() before start()")

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{_HEYGEN_BASE}/v1/streaming.start",
                headers=self._headers(),
                json={"session_id": self._session_id},
            )
            resp.raise_for_status()
        print(f"{_ts()} | [HEYGEN] session started | session_id={self._session_id}")

    async def stop(self) -> None:
        """POST /v1/streaming.stop — cleanly close the HeyGen session."""
        import httpx

        if not self._session_id:
            return

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    f"{_HEYGEN_BASE}/v1/streaming.stop",
                    headers=self._headers(),
                    json={"session_id": self._session_id},
                )
                resp.raise_for_status()
            print(f"{_ts()} | [HEYGEN] session stopped | session_id={self._session_id}")
        except Exception as exc:
            print(f"{_ts()} | [WARN] [HEYGEN] stop error (ignored): {exc}")
        finally:
            self._session_id = None
            self._access_token = None
            self._room_url = None

    async def interrupt(self) -> None:
        """POST /v1/streaming.interrupt — tell HeyGen to stop current speech immediately."""
        import httpx

        if not self._session_id:
            return
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.post(
                    f"{_HEYGEN_BASE}/v1/streaming.interrupt",
                    headers=self._headers(),
                    json={"session_id": self._session_id},
                )
                resp.raise_for_status()
            print(f"{_ts()} | [HEYGEN] interrupt sent")
        except Exception as exc:
            print(f"{_ts()} | [WARN] [HEYGEN] interrupt error (ignored): {exc}")

    # ── Audio helpers ─────────────────────────────────────────────────────

    async def send_silence(self, audio_source) -> None:
        """Push a short silence frame to the HeyGen audio source to close avatar mouth.

        Call this immediately after flush_pending_audio() when TTS is interrupted.

        Args:
            audio_source: livekit.rtc.AudioSource connected to the HeyGen room.
        """
        from livekit import rtc  # import lazily; only needed when avatar is active

        n_samples = int(_SILENCE_SAMPLE_RATE * _SILENCE_DURATION_S)
        silence = np.zeros(n_samples, dtype=np.int16)
        frame = rtc.AudioFrame(
            data=bytearray(silence.tobytes()),
            sample_rate=_SILENCE_SAMPLE_RATE,
            num_channels=_SILENCE_CHANNELS,
            samples_per_channel=n_samples,
        )
        await audio_source.capture_frame(frame)
        print(f"{_ts()} | [HEYGEN] silence frame sent ({_SILENCE_DURATION_S*1000:.0f} ms)")

    # ── Internal ──────────────────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "X-Api-Key": self._api_key,
            "Content-Type": "application/json",
        }
