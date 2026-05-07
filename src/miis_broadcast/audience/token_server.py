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
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse


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
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None

    # ── Public API ────────────────────────────────────────────────────────

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
        from livekit.api import AccessToken, VideoGrants

        app = FastAPI(docs_url=None, redoc_url=None)
        static_dir = Path(__file__).parent / "static"

        @app.get("/audience", response_class=HTMLResponse)
        async def audience_page():
            html_path = static_dir / "index.html"
            return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

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
            return JSONResponse({"url": self._livekit_url, "token": token})

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
