# src/miis_broadcast/audience/token_server.py
"""
AudienceTokenServer
===================
Lightweight FastAPI server (background thread) that:
  GET  /audience            — serves the audience HTML viewer page
  POST /api/audience/join   — issues a subscribe-only LiveKit JWT

Usage:
    server = AudienceTokenServer(livekit_url, api_key, api_secret, room, port=8080)
    server.start()   # non-blocking
    ...
    server.stop()
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional


def _ts() -> str:
    return time.strftime("%H:%M:%S")


class AudienceTokenServer:
    def __init__(
        self,
        livekit_url: str,
        api_key: str,
        api_secret: str,
        room_name: str,
        host: str = "0.0.0.0",
        port: int = 8080,
        lan_hint_host: Optional[str] = None,
    ) -> None:
        self._livekit_url = livekit_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._room_name = room_name
        self._host = host
        self._port = port
        self._lan_hint_host = (lan_hint_host or "").strip() or None
        self._server: Optional[Any] = None  # uvicorn.Server, imported lazily
        self._thread: Optional[threading.Thread] = None
        # When False ("維持原聲"), the audience page plays original audio instead
        # of AI narration. The cat avatar is hidden in that mode regardless.
        self._narration_enabled = True
        # Independent operator switch for the cat mascot itself.
        self._mascot_enabled = True
        self._state_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────

    def set_narration_enabled(self, enabled: bool) -> None:
        """Operator toggle: AI narration on → audience hears TTS; off → original audio."""
        with self._state_lock:
            self._narration_enabled = bool(enabled)

    def get_narration_enabled(self) -> bool:
        with self._state_lock:
            return self._narration_enabled

    def set_mascot_enabled(self, enabled: bool) -> None:
        """Operator toggle for the cat mascot, independent of the audio mode."""
        with self._state_lock:
            self._mascot_enabled = bool(enabled)

    def get_mascot_enabled(self) -> bool:
        with self._state_lock:
            return self._mascot_enabled

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AudienceTokenServer"
        )
        self._thread.start()
        print(
            f"{_ts()} | [AUDIENCE] token server started"
            f" → http://localhost:{self._port}/audience"
        )
        if self._lan_hint_host:
            print(
                f"{_ts()} | [AUDIENCE] same-WiFi URL → "
                f"http://{self._lan_hint_host}:{self._port}/audience"
            )
            print(
                f"{_ts()} | [AUDIENCE] if other devices cannot open the page or WebRTC fails, "
                f"allow ports in Windows Firewall (run scripts/open-audience-firewall.ps1 as Administrator)"
            )

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
        print(f"{_ts()} | [AUDIENCE] token server stopped")

    # ── Internal ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from livekit.api import AccessToken, VideoGrants

        app = FastAPI(docs_url=None, redoc_url=None)
        static_dir = Path(__file__).parent / "static"

        # Serve project-level assets (avatar PNGs, etc.) under /assets
        assets_dir = Path(__file__).resolve().parents[3] / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        # Check avatar assets once at startup; inject result into HTML so the
        # browser never makes a speculative request for files that don't exist.
        _avatar_video = assets_dir / "avatar" / "cat_anchor.mp4"
        _avatar_img   = assets_dir / "avatar" / "cat_mouth_opened.png"
        _avatar_enabled = _avatar_video.exists()
        _avatar_img_enabled = _avatar_img.exists()
        _avatar_inject = (
            f"<script>window._AVATAR_ENABLED={str(_avatar_enabled).lower()};"
            f"window._AVATAR_IMG_ENABLED={str(_avatar_img_enabled).lower()};</script>"
        )

        @app.get("/audience", response_class=HTMLResponse)
        async def audience_page():
            html_path = static_dir / "index.html"
            content = html_path.read_text(encoding="utf-8")
            content = content.replace("</head>", _avatar_inject + "\n</head>", 1)
            return HTMLResponse(content=content)

        @app.post("/api/audience/join")
        async def join():
            identity = f"audience-{int(time.time() * 1000) % 100000}"
            token = (
                AccessToken(self._api_key, self._api_secret)
                .with_identity(identity)
                .with_name("Audience Viewer")
                .with_grants(
                    VideoGrants(
                        room_join=True,
                        room=self._room_name,
                        can_subscribe=True,
                        can_publish=False,
                        can_publish_data=False,
                    )
                )
                .to_jwt()
            )
            print(f"{_ts()} | [AUDIENCE] join id={identity} room={self._room_name}")
            return JSONResponse(
                {
                    "url": self._livekit_url,
                    "token": token,
                    "narration_enabled": self.get_narration_enabled(),
                    "mascot_enabled": self.get_mascot_enabled(),
                }
            )

        @app.get("/api/audience/status")
        async def status():
            """Pollable mode flags so late joiners / missed data packets stay in sync."""
            return JSONResponse(
                {
                    "narration_enabled": self.get_narration_enabled(),
                    "mascot_enabled": self.get_mascot_enabled(),
                }
            )

        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        try:
            self._server.run()
        except OSError as exc:
            print(
                f"{_ts()} | [ERR] [AUDIENCE] token server bind failed "
                f"host={self._host} port={self._port} — {exc}"
            )
        except Exception as exc:
            print(f"{_ts()} | [ERR] [AUDIENCE] token server: {exc}")
