"""
Remote inference server — headless entry point.

Usage:
    # From project root:
    python -m miis_broadcast.server
    python -m miis_broadcast.server --host 0.0.0.0 --port 9000 --device 0

    # Or with SSH port-forward on the client side:
    ssh -p 2225 -L 9000:127.0.0.1:9000 miislab-server3@10.50.0.103
    # then on local machine connect to 127.0.0.1:9000
"""
from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading

# Force unbuffered output so logs appear immediately in SSH/tmux sessions
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# force=True ensures this takes effect even if a previously imported package
# (e.g. transformers, torch) already attached a handler to the root logger.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
    force=True,
)
log = logging.getLogger("miis_broadcast.server")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="miis_broadcast remote inference server"
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind host (default: 0.0.0.0 — all interfaces)"
    )
    parser.add_argument(
        "--port", type=int, default=9000,
        help="Bind port (default: 9000)"
    )
    parser.add_argument(
        "--device", type=int, default=0,
        help="CUDA device ID for LiveCC (default: 0)"
    )
    parser.add_argument(
        "--config", default="./configs/models.yml",
        help="Path to models.yml (default: ./configs/models.yml)"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Validate / load model config (paths, YAML). Server does not load ByteTrack on host.
    # ------------------------------------------------------------------ #
    try:
        from miis_broadcast.core.utils.config import parse_configs

        parse_configs(args.config)
        log.info("Loaded model config from %s", args.config)
    except Exception as e:
        log.warning("Could not load model config %s: %s", args.config, e)

    # ------------------------------------------------------------------ #
    # Load LiveCC model (once, shared across all sessions)
    # ------------------------------------------------------------------ #
    log.info("Loading LiveCC model on device=%d …", args.device)
    livecc_model = None
    try:
        from miis_broadcast.core.models.livecc_transformers import LiveCCInfer
        livecc_model = LiveCCInfer(device_id=args.device)
        log.info("LiveCC model loaded ✓")
    except Exception as e:
        log.error("Failed to load LiveCC model: %s", e)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # TCP server
    # ------------------------------------------------------------------ #
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((args.host, args.port))
    server_sock.listen(5)
    log.info("Listening on %s:%d — waiting for clients…", args.host, args.port)

    try:
        while True:
            client_sock, addr = server_sock.accept()
            log.info("New client connected: %s", addr)

            from miis_broadcast.server.session import ClientSession

            session = ClientSession(
                sock=client_sock,
                addr=addr,
                livecc_model=livecc_model,
            )
            t = threading.Thread(target=session.run, daemon=True, name=f"session-{addr}")
            t.start()

    except KeyboardInterrupt:
        log.info("Server interrupted, shutting down…")
    finally:
        server_sock.close()


if __name__ == "__main__":
    main()
