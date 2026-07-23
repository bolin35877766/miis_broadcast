# src/miis_broadcast/audience/original_audio_capture.py
"""Capture desktop / game audio for Audience 「維持原聲」 mode.

OBS Virtual Camera is video-only. We capture Windows WASAPI *loopback* (or a
configured input such as VB-Cable Output) via PyAudioWPatch, resample to
24 kHz mono, and push PCM to LiveKit.

Stock ``sounddevice`` cannot open WASAPI loopback endpoints.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

TARGET_SR = 24_000
_SILENCE_RMS = 1e-4


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _resample_mono_f32(mono: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or mono.size == 0:
        return mono
    n_out = max(1, int(round(mono.size * float(dst_sr) / float(src_sr))))
    x = np.arange(mono.size, dtype=np.float64)
    xp = np.linspace(0.0, float(mono.size - 1), num=n_out, dtype=np.float64)
    return np.interp(xp, x, mono.astype(np.float64, copy=False)).astype(np.float32)


def _to_pcm16_mono(indata: np.ndarray, src_sr: int) -> np.ndarray:
    if indata.ndim == 1:
        mono = indata.astype(np.float32, copy=False)
    else:
        mono = indata.mean(axis=1).astype(np.float32, copy=False)
    if indata.dtype == np.int16:
        mono = mono / 32768.0
    mono = _resample_mono_f32(mono, src_sr, TARGET_SR)
    return (np.clip(mono, -1.0, 1.0) * 32767.0).astype(np.int16)


def _list_loopbacks(pa: Any) -> List[dict]:
    out: List[dict] = []
    try:
        for d in pa.get_loopback_device_info_generator():
            info = dict(d)
            info["index"] = int(d["index"])
            info["_kind"] = "loopback"
            out.append(info)
    except Exception:
        for i in range(pa.get_device_count()):
            d = pa.get_device_info_by_index(i)
            if d.get("isLoopbackDevice"):
                info = dict(d)
                info["index"] = i
                info["_kind"] = "loopback"
                out.append(info)
    return out


def _list_cable_inputs(pa: Any) -> List[dict]:
    """VB-Cable / similar: capture the *Output* side (normal input device)."""
    out: List[dict] = []
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if int(d.get("maxInputChannels") or 0) < 1:
            continue
        if d.get("isLoopbackDevice"):
            continue
        name = str(d.get("name", "")).lower()
        # CABLE Output is the recording end of VB-Audio Virtual Cable.
        if "cable output" in name or "vb-audio" in name and "output" in name:
            info = dict(d)
            info["index"] = i
            info["_kind"] = "cable_input"
            out.append(info)
    return out


def _prefer_score(name: str, kind: str = "loopback") -> int:
    n = name.lower()
    if kind == "cable_input":
        return 85  # OBS monitoring → CABLE is a strong path when configured
    # Real playback devices first; HDMI monitors last (often silent).
    if "speaker" in n or "realtek" in n:
        return 100
    if "oculus" in n or "headset" in n or "耳機" in name:
        return 95
    if "cable" in n or "vb-audio" in n:
        return 40
    if "nvidia" in n or "benq" in n or "hdmi" in n or "display" in n:
        return 5
    return 50


def _match_device(pa: Any, device: Union[int, str]) -> dict:
    if isinstance(device, (int, float)) or (
        isinstance(device, str) and device.strip().isdigit()
    ):
        idx = int(device)
        info = dict(pa.get_device_info_by_index(idx))
        info["index"] = idx
        info["_kind"] = "loopback" if info.get("isLoopbackDevice") else "input"
        return info

    needle = str(device).strip().lower()
    if not needle or needle in ("null", "none"):
        raise RuntimeError("empty original_audio_device")

    # Prefer loopback name match, then any input.
    for d in _list_loopbacks(pa):
        if needle in str(d.get("name", "")).lower():
            return d
    for d in _list_cable_inputs(pa):
        if needle in str(d.get("name", "")).lower():
            return d
    for i in range(pa.get_device_count()):
        d = pa.get_device_info_by_index(i)
        if int(d.get("maxInputChannels") or 0) < 1:
            continue
        if needle in str(d.get("name", "")).lower():
            info = dict(d)
            info["index"] = i
            info["_kind"] = "input"
            return info
    names = [str(d.get("name", "")) for d in _list_loopbacks(pa)]
    raise RuntimeError(
        f"找不到音訊裝置符合 {device!r}。可用 loopback：{names}"
    )


def _probe_rms(pa: Any, pyaudio_mod: Any, info: dict, seconds: float = 0.3) -> Tuple[float, int]:
    idx = int(info["index"])
    sr = int(info.get("defaultSampleRate") or 48000)
    ch = max(1, min(2, int(info.get("maxInputChannels") or 2)))
    chunks: List[np.ndarray] = []
    n_cb = 0

    def _cb(in_data, frame_count, time_info, status_flags):  # noqa: ARG001
        nonlocal n_cb
        n_cb += 1
        chunks.append(np.frombuffer(in_data, dtype=np.float32).copy())
        return (None, pyaudio_mod.paContinue)

    stream = None
    try:
        stream = pa.open(
            format=pyaudio_mod.paFloat32,
            channels=ch,
            rate=sr,
            input=True,
            input_device_index=idx,
            frames_per_buffer=960,
            stream_callback=_cb,
        )
        time.sleep(seconds)
    except Exception:
        return 0.0, 0
    finally:
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass

    if not chunks:
        return 0.0, n_cb
    arr = np.concatenate(chunks)
    rms = float(np.sqrt(np.mean(np.square(arr), dtype=np.float64)))
    return rms, n_cb


def _default_output_loopback(pa: Any) -> Optional[dict]:
    """WASAPI loopback of the current Windows default playback device."""
    try:
        info = dict(pa.get_default_wasapi_loopback())
        info["index"] = int(info["index"])
        info["_kind"] = "loopback"
        return info
    except Exception:
        return None


def _auto_pick_device(pa: Any, pyaudio_mod: Any) -> dict:
    """Prefer devices with real signal; otherwise follow Windows default output."""
    candidates = _list_loopbacks(pa) + _list_cable_inputs(pa)
    if not candidates:
        raise RuntimeError(
            "找不到可用音訊擷取裝置；請安裝 pyaudiowpatch，或設定 "
            "audience.original_audio_device"
        )

    default_lb = _default_output_loopback(pa)
    if default_lb is not None:
        print(
            f"{_ts()} | [ORIGINAL-AUDIO] Windows 預設輸出 loopback = "
            f"#{default_lb['index']} {default_lb.get('name')}"
        )

    print(f"{_ts()} | [ORIGINAL-AUDIO] probing capture devices…")
    ranked: List[Tuple[float, int, int, dict]] = []
    for d in candidates:
        rms, n_cb = _probe_rms(pa, pyaudio_mod, d)
        score = _prefer_score(str(d.get("name", "")), str(d.get("_kind", "")))
        # Boost the current Windows default so silent-probe still prefers it.
        if default_lb is not None and int(d["index"]) == int(default_lb["index"]):
            score = max(score, 120)
        print(
            f"{_ts()} | [ORIGINAL-AUDIO]   "
            f"#{d['index']} rms={rms:.4f} cb={n_cb} prefer={score} "
            f"| {d.get('name')}"
        )
        if n_cb <= 0:
            continue
        ranked.append((rms, score, n_cb, d))

    if not ranked:
        if default_lb is not None:
            print(
                f"{_ts()} | [ORIGINAL-AUDIO] pick #{default_lb['index']} "
                f"{default_lb.get('name')} (Windows 預設輸出)"
            )
            return default_lb
        ranked_fallback = sorted(
            candidates,
            key=lambda d: _prefer_score(str(d.get("name", "")), str(d.get("_kind", ""))),
            reverse=True,
        )
        picked = ranked_fallback[0]
        print(
            f"{_ts()} | [ORIGINAL-AUDIO] pick #{picked['index']} "
            f"{picked.get('name')} (fallback, no active callbacks)"
        )
        return picked

    # 1) Any device with real signal wins (by rms, then prefer).
    loud = [t for t in ranked if t[0] >= _SILENCE_RMS]
    if loud:
        loud.sort(key=lambda t: (t[0], t[1]), reverse=True)
        picked = loud[0][3]
        print(
            f"{_ts()} | [ORIGINAL-AUDIO] pick #{picked['index']} "
            f"{picked.get('name')} (loudest rms={loud[0][0]:.4f})"
        )
        return picked

    # 2) All silent: follow Windows default output (e.g. BenQ if that's selected).
    if default_lb is not None:
        for rms, score, n_cb, d in ranked:
            if int(d["index"]) == int(default_lb["index"]):
                print(
                    f"{_ts()} | [ORIGINAL-AUDIO] pick #{d['index']} "
                    f"{d.get('name')} (all silent → Windows 預設輸出)"
                )
                print(
                    f"{_ts()} | [ORIGINAL-AUDIO] WARN: 探測時全是靜音。"
                    f"請確認該輸出裝置真的有在播（或 OBS「監控並輸出」到此裝置）。"
                )
                return d
        print(
            f"{_ts()} | [ORIGINAL-AUDIO] pick #{default_lb['index']} "
            f"{default_lb.get('name')} (Windows 預設輸出, probe 無 cb)"
        )
        return default_lb

    ranked.sort(key=lambda t: t[1], reverse=True)
    picked = ranked[0][3]
    print(
        f"{_ts()} | [ORIGINAL-AUDIO] pick #{picked['index']} "
        f"{picked.get('name')} (all silent → highest prefer={ranked[0][1]})"
    )
    return picked


class OriginalAudioCapture:
    """WASAPI-loopback / cable capture → ``on_pcm(int16 mono @ 24 kHz)``."""

    def __init__(
        self,
        on_pcm: Callable[[np.ndarray], None],
        device: Optional[Union[int, str]] = None,
        blocksize: int = 960,
    ) -> None:
        self._on_pcm = on_pcm
        self._device_cfg = device
        self._blocksize = blocksize
        self._pa: Any = None
        self._stream: Any = None
        self._pyaudio_mod: Any = None
        self._sr = 48000
        self._channels = 2
        self._lock = threading.Lock()
        self._running = False
        self._live_rms = 0.0
        self._rms_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._candidate_infos: List[dict] = []
        self._current_info: Optional[dict] = None

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        try:
            import pyaudiowpatch as pyaudio
        except ImportError as exc:
            raise RuntimeError(
                "需要 pyaudiowpatch（pip install pyaudiowpatch）才能擷取系統播放音"
            ) from exc

        self._pyaudio_mod = pyaudio
        pa = pyaudio.PyAudio()
        try:
            self._candidate_infos = _list_loopbacks(pa) + _list_cable_inputs(pa)
            if self._device_cfg is not None and str(self._device_cfg).lower() not in (
                "",
                "null",
                "none",
            ):
                info = _match_device(pa, self._device_cfg)
            else:
                info = _auto_pick_device(pa, pyaudio)
        except Exception:
            pa.terminate()
            raise

        self._pa = pa
        self._open_stream(info)
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="OriginalAudioMonitor"
        )
        self._monitor_thread.start()

    def _open_stream(self, info: dict) -> None:
        pa = self._pa
        pyaudio = self._pyaudio_mod
        assert pa is not None and pyaudio is not None

        # Close previous stream if switching devices.
        old = self._stream
        self._stream = None
        if old is not None:
            try:
                old.stop_stream()
            except Exception:
                pass
            try:
                old.close()
            except Exception:
                pass

        idx = int(info["index"])
        sr = int(info.get("defaultSampleRate") or 48000)
        channels = max(1, min(2, int(info.get("maxInputChannels") or 2)))
        self._sr = sr
        self._channels = channels
        self._current_info = info

        def _callback(in_data, frame_count, time_info, status_flags):  # noqa: ARG001
            try:
                arr = np.frombuffer(in_data, dtype=np.float32)
                if channels > 1:
                    arr = arr.reshape(-1, channels)
                # Live RMS on float input (pre-resample).
                flat = arr.reshape(-1).astype(np.float64, copy=False)
                rms = float(np.sqrt(np.mean(np.square(flat)))) if flat.size else 0.0
                with self._rms_lock:
                    # EMA so brief spikes don't hide sustained silence.
                    self._live_rms = 0.85 * self._live_rms + 0.15 * rms
                pcm = _to_pcm16_mono(arr, self._sr)
                if pcm.size:
                    self._on_pcm(pcm)
            except Exception as exc:
                print(f"{_ts()} | [WARN] [ORIGINAL-AUDIO] callback: {exc}")
            return (None, pyaudio.paContinue)

        stream = pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=sr,
            input=True,
            input_device_index=idx,
            frames_per_buffer=self._blocksize,
            stream_callback=_callback,
        )
        with self._lock:
            self._stream = stream
            self._running = True
        print(
            f"{_ts()} | [ORIGINAL-AUDIO] started #{idx} {info.get('name')} "
            f"sr={sr} ch={channels} → {TARGET_SR}Hz mono"
        )

    def _monitor_loop(self) -> None:
        """If capture stays silent, rotate to next preferred device once."""
        silent_ticks = 0
        tried: set[int] = set()
        if self._current_info is not None:
            tried.add(int(self._current_info["index"]))
        log_every = 0
        while not self._monitor_stop.wait(1.0):
            with self._rms_lock:
                rms = self._live_rms
            log_every += 1
            if log_every % 2 == 0:
                print(f"{_ts()} | [ORIGINAL-AUDIO] live_rms={rms:.4f}")

            if rms >= _SILENCE_RMS:
                silent_ticks = 0
                continue
            silent_ticks += 1
            if silent_ticks < 3:
                continue  # ~3s silence before switching

            # Build rotation order: prefer score desc, skip tried.
            alts = sorted(
                self._candidate_infos,
                key=lambda d: _prefer_score(
                    str(d.get("name", "")), str(d.get("_kind", ""))
                ),
                reverse=True,
            )
            nxt = next((d for d in alts if int(d["index"]) not in tried), None)
            if nxt is None:
                if silent_ticks == 3:
                    print(
                        f"{_ts()} | [ORIGINAL-AUDIO] WARN: 持續靜音。"
                        f"OBS 音量條有動 ≠ Windows 有播出。"
                        f"請將 OBS 音訊設「監控並輸出」到 Speakers／耳機／CABLE。"
                    )
                continue

            tried.add(int(nxt["index"]))
            silent_ticks = 0
            print(
                f"{_ts()} | [ORIGINAL-AUDIO] silence → switching to "
                f"#{nxt['index']} {nxt.get('name')}"
            )
            try:
                self._open_stream(nxt)
            except Exception as exc:
                print(f"{_ts()} | [WARN] [ORIGINAL-AUDIO] switch failed: {exc}")

    def stop(self) -> None:
        self._monitor_stop.set()
        t = self._monitor_thread
        self._monitor_thread = None
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        with self._lock:
            stream = self._stream
            pa = self._pa
            self._stream = None
            self._pa = None
            self._running = False
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception:
                pass
            try:
                stream.close()
            except Exception:
                pass
        if pa is not None:
            try:
                pa.terminate()
            except Exception:
                pass
        print(f"{_ts()} | [ORIGINAL-AUDIO] stopped")
