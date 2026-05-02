"""
Shared network protocol for client-server communication.

Wire format per message:
    [4B total_len | 4B json_len | JSON bytes | binary payload]

- total_len  = json_len + len(binary)
- If no binary: total_len == json_len, binary = b""
- All integers are big-endian unsigned 32-bit.
"""
from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, Optional, Tuple

PROTOCOL_VERSION = 1

# Message type constants
MSG_HELLO   = "HELLO"
MSG_ACK     = "ACK"
MSG_START   = "START"
MSG_FRAME   = "FRAME"
MSG_SEGMENT = "SEGMENT"
MSG_STATUS  = "STATUS"
MSG_ERROR   = "ERROR"
MSG_STOP    = "STOP"
MSG_PING    = "PING"
MSG_PONG    = "PONG"
MSG_PREVIEW = "PREVIEW"  # JPEG frame with tracking overlay for thin-client GUI
# CLIENT_DIAG: thin-client stats (RAM + optional CUDA VRAM on device 0) for server stdout
MSG_CLIENT_DIAG = "CLIENT_DIAG"

_HEADER_FMT  = ">II"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 8 bytes


def pack_message(msg: Dict[str, Any], binary: bytes = b"") -> bytes:
    """Serialize a message dict and optional binary payload into wire bytes."""
    json_bytes = json.dumps(msg, ensure_ascii=False).encode("utf-8")
    json_len   = len(json_bytes)
    total_len  = json_len + len(binary)
    header     = struct.pack(_HEADER_FMT, total_len, json_len)
    return header + json_bytes + binary


def read_message(sock: socket.socket) -> Optional[Tuple[Dict[str, Any], bytes]]:
    """
    Read exactly one message from a blocking socket.
    Returns (msg_dict, binary_payload) or None if the connection was closed cleanly.
    Raises socket.error on other errors.
    """
    raw_header = _recv_exact(sock, _HEADER_SIZE)
    if raw_header is None:
        return None

    total_len, json_len = struct.unpack(_HEADER_FMT, raw_header)

    json_bytes = _recv_exact(sock, json_len)
    if json_bytes is None:
        return None

    binary_len = total_len - json_len
    binary = b""
    if binary_len > 0:
        binary = _recv_exact(sock, binary_len)
        if binary is None:
            return None

    msg = json.loads(json_bytes.decode("utf-8"))
    return msg, binary


def _recv_exact(sock: socket.socket, n: int) -> Optional[bytes]:
    """Receive exactly n bytes. Returns None if the connection closed before n bytes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)
