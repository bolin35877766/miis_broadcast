# src/miis_broadcast/audience/liveavatar_session.py
"""
LiveAvatarSession — LITE mode REST + WebSocket
==============================================
- Create session token (POST /v1/sessions/token, mode LITE)
- Start session (POST /v1/sessions/start, Bearer session_token) → LiveKit URL + client token + ws_url
- WebSocket LITE events: TTS as `agent.speak` (PCM int16 24 kHz, Base64)
- Interrupt: `agent.interrupt`
- Stop: POST /v1/sessions/stop

API key: https://app.liveavatar.com/developers

Docs: https://docs.liveavatar.com/docs/lite-mode/lifecycle.md
Events: https://docs.liveavatar.com/docs/lite-mode/events.md

Heavy **Base64** encoding and **JSON serialization** for outbound `agent.speak` run via
``asyncio.to_thread`` so they do not block the GUI's shared asyncio event loop (video pacing).
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Optional

import numpy as np

_LIVEAVATAR_BASE = "https://api.liveavatar.com"

_SILENCE_DURATION_S = 0.1
_SILENCE_SAMPLE_RATE = 24_000
_SILENCE_CHANNELS = 1


def _build_agent_speak_payload(pcm_int16: np.ndarray) -> dict:
    """Build agent.speak JSON-serializable payload. CPU-heavy; run via asyncio.to_thread."""
    if pcm_int16.dtype != np.int16:
        pcm_int16 = pcm_int16.astype(np.int16, copy=False)
    b64 = base64.b64encode(pcm_int16.tobytes()).decode("ascii")
    return {"type": "agent.speak", "audio": b64}


def _ts() -> str:
    return time.strftime("%H:%M:%S")


class LiveAvatarSession:
    """LiveAvatar LITE: REST token/start + WebSocket `agent.speak`."""

    def __init__(
        self,
        api_key: str = "",
        avatar_id: str = "",
        voice_id: str = "",
        quality: str = "medium",
        sandbox: bool = False,
    ) -> None:
        # voice_id reserved for future use; LITE token uses avatar default voice.
        self._api_key = api_key
        self._avatar_id = avatar_id
        self._voice_id = voice_id
        self._quality = quality
        self._sandbox = sandbox

        self._session_token: Optional[str] = None
        self._session_id: Optional[str] = None
        self._access_token: Optional[str] = None
        self._room_url: Optional[str] = None
        self._ws_url: Optional[str] = None

        self._ws: Optional[Any] = None
        self._ws_reader_task: Optional[asyncio.Task] = None
        self._ws_connected = asyncio.Event()
        self._ws_send_lock = asyncio.Lock()
        self._closed = False

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

    async def create(self) -> dict:
        """POST /v1/sessions/token — obtain session_token for /sessions/start."""
        import httpx

        aid = (self._avatar_id or "").strip()
        if not aid:
            raise RuntimeError(
                "[LIVEAVATAR] avatar_id is required (UUID from app.liveavatar.com → Avatars)."
            )

        payload: dict[str, Any] = {
            "mode": "LITE",
            "avatar_id": aid,
            "video_settings": {
                "quality": self._video_quality(),
                "encoding": "H264",
            },
        }
        if self._sandbox:
            payload["is_sandbox"] = True

        print(f"{_ts()} | [LIVEAVATAR] creating token | avatar={aid} sandbox={self._sandbox}")
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f"{_LIVEAVATAR_BASE}/v1/sessions/token",
                headers=self._api_headers(),
                json=payload,
            )
            try:
                body = resp.json() if resp.content else {}
            except Exception:
                body = {}

        if not resp.is_success:
            raise RuntimeError(
                f"[LIVEAVATAR] sessions/token HTTP {resp.status_code}: {body or resp.text[:500]}"
            )

        code = body.get("code")
        if code is not None and code not in (100, 1000):
            raise RuntimeError(f"[LIVEAVATAR] sessions/token error code {code}: {body}")

        inner = body.get("data")
        if not isinstance(inner, dict):
            raise RuntimeError(f"[LIVEAVATAR] token: unexpected response: {body}")

        self._session_token = inner.get("session_token")
        if not self._session_token:
            raise RuntimeError(f"[LIVEAVATAR] token: missing session_token: {body}")

        print(f"{_ts()} | [LIVEAVATAR] session token created")
        return inner

    async def start(self) -> None:
        """POST /v1/sessions/start, connect WebSocket, wait until `connected`."""
        import httpx

        if not self._session_token:
            raise RuntimeError("[LIVEAVATAR] call create() before start()")

        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{_LIVEAVATAR_BASE}/v1/sessions/start",
                headers={
                    "Authorization": f"Bearer {self._session_token}",
                    "Content-Type": "application/json",
                },
            )
            try:
                body = resp.json() if resp.content else {}
            except Exception:
                body = {}

        if not resp.is_success:
            raise RuntimeError(
                f"[LIVEAVATAR] sessions/start HTTP {resp.status_code}: {body or resp.text[:500]}"
            )

        code = body.get("code")
        if code is not None and code not in (100, 1000):
            raise RuntimeError(f"[LIVEAVATAR] sessions/start error code {code}: {body}")

        inner = body.get("data")
        if not isinstance(inner, dict):
            raise RuntimeError(f"[LIVEAVATAR] start: unexpected response: {body}")

        self._session_id = inner.get("session_id")
        self._room_url = inner.get("livekit_url")
        self._access_token = inner.get("livekit_client_token")
        self._ws_url = inner.get("ws_url")

        if not self._session_id or not self._room_url or not self._access_token:
            raise RuntimeError(f"[LIVEAVATAR] start: missing LiveKit fields: {body}")

        if not self._ws_url:
            raise RuntimeError(
                "[LIVEAVATAR] start: response missing ws_url — LITE audio requires the events socket."
            )

        await self._connect_events_ws()
        print(
            f"{_ts()} | [LIVEAVATAR] session started | session_id={self._session_id} "
            f"livekit={self._room_url}"
        )

    async def stop(self) -> None:
        """Close WebSocket and POST /v1/sessions/stop."""
        import httpx

        sid = self._session_id
        await self._disconnect_ws()
        self._session_token = None
        self._session_id = None
        self._access_token = None
        self._room_url = None
        self._ws_url = None

        if not sid:
            return

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{_LIVEAVATAR_BASE}/v1/sessions/stop",
                    headers={
                        **self._api_headers(),
                        "Content-Type": "application/json",
                    },
                    json={"session_id": sid, "reason": "USER_CLOSED"},
                )
                if resp.is_success:
                    print(f"{_ts()} | [LIVEAVATAR] session stopped | session_id={sid}")
                else:
                    print(
                        f"{_ts()} | [WARN] [LIVEAVATAR] stop HTTP {resp.status_code}: "
                        f"{(resp.text or '')[:200]}"
                    )
        except Exception as exc:
            print(f"{_ts()} | [WARN] [LIVEAVATAR] stop error (ignored): {exc}")

    async def interrupt(self) -> None:
        """Send `agent.interrupt` on the events WebSocket."""
        if self._ws is None or not self._ws_connected.is_set():
            return
        await self._ws_send_json({"type": "agent.interrupt"})
        print(f"{_ts()} | [LIVEAVATAR] interrupt sent")

    async def send_pcm_chunk(self, pcm_int16: np.ndarray) -> None:
        """Stream one TTS chunk as `agent.speak` (PCM 16-bit LE mono 24 kHz, Base64)."""
        # Base64 + large JSON building must not run on the asyncio loop: the GUI publisher
        # shares the same loop with VR compositing; blocking here causes visible video stutter.
        payload = await asyncio.to_thread(_build_agent_speak_payload, pcm_int16)
        await self._ws_send_json(payload)

    async def send_silence(self) -> None:
        """Short silence via WebSocket after a hard stop (optional mouth settle)."""
        n_samples = int(_SILENCE_SAMPLE_RATE * _SILENCE_DURATION_S)
        silence = np.zeros(n_samples, dtype=np.int16)
        try:
            await self.send_pcm_chunk(silence)
            print(f"{_ts()} | [LIVEAVATAR] silence chunk sent ({_SILENCE_DURATION_S*1000:.0f} ms)")
        except Exception as exc:
            print(f"{_ts()} | [WARN] [LIVEAVATAR] silence send: {exc}")

    def _api_headers(self) -> dict[str, str]:
        return {
            "X-API-KEY": (self._api_key or "").strip(),
            "Content-Type": "application/json",
        }

    def _video_quality(self) -> str:
        q = (self._quality or "medium").lower().strip()
        if q in ("low", "medium", "high", "very_high"):
            return q
        if q == "very high":
            return "very_high"
        return "medium"

    async def _connect_events_ws(self) -> None:
        import websockets

        self._ws_connected.clear()
        self._closed = False

        try:
            self._ws = await websockets.connect(
                self._ws_url,
                max_size=2**23,
            )
        except Exception as exc:
            raise RuntimeError(f"[LIVEAVATAR] WebSocket connect failed: {exc}") from exc

        self._ws_reader_task = asyncio.create_task(self._ws_reader_loop())
        try:
            await asyncio.wait_for(self._ws_connected.wait(), timeout=45.0)
        except asyncio.TimeoutError as exc:
            await self._disconnect_ws()
            raise RuntimeError(
                "[LIVEAVATAR] WebSocket did not report connected state in time"
            ) from exc

    async def _ws_reader_loop(self) -> None:
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                st = _extract_session_state(msg)
                if st == "connected":
                    self._ws_connected.set()
                if msg.get("type") == "error":
                    print(f"{_ts()} | [WARN] [LIVEAVATAR] ws error event: {msg}")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            if not self._closed:
                print(f"{_ts()} | [WARN] [LIVEAVATAR] WebSocket reader ended: {exc}")

    async def _ws_send_json(self, payload: dict) -> None:
        if self._ws is None:
            return
        if not self._ws_connected.is_set():
            try:
                await asyncio.wait_for(self._ws_connected.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                print(f"{_ts()} | [WARN] [LIVEAVATAR] ws not connected; drop send")
                return
        # json.dumps on large agent.speak strings blocks the event loop; offload to thread.
        text = await asyncio.to_thread(json.dumps, payload)
        async with self._ws_send_lock:
            await self._ws.send(text)

    async def _disconnect_ws(self) -> None:
        self._closed = True
        t = self._ws_reader_task
        self._ws_reader_task = None
        if t is not None:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

        ws = self._ws
        self._ws = None
        self._ws_connected.clear()
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass


def _extract_session_state(msg: dict) -> Optional[str]:
    if msg.get("type") != "session.state_updated":
        return None
    st = msg.get("state")
    if isinstance(st, str):
        return st.lower()
    payload = msg.get("payload")
    if isinstance(payload, dict):
        inner = payload.get("state")
        if isinstance(inner, str):
            return inner.lower()
    return None
