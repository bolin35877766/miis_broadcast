#!/usr/bin/env python3
"""
eval/bench_latency.py — 無頭延遲基準測試

測試 LiveCC + Gemini + OpenAI TTS 全鏈路延遲。
對 VR_Frank_basketball.mp4 進行 NUM_RUNS 次 × MAX_DURATION_SEC 秒的推論：

  測項 1 — 各系統延遲
    - LiveCC drift   : frame_wall_ts → signal_segment 到達（正值=模型比實時慢，負值=快）
    - Gemini API     : segment 進 queue → Gemini 回傳
    - TTS            : text 送入 WebSocket → 第一個音訊 chunk 開始播放
    - Pipeline E2E   : LiveCC signal 到達 → TTS 音訊開始（signal→Gemini→TTS）

  測項 2 — Interrupt 延遲
    - P1 detection → interrupt TTS 音訊開始播放
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from PySide6 import QtCore, QtWidgets

from miis_broadcast.workers.livecc import LiveCCWorker
from miis_broadcast.workers.gemini import GeminiWorker
import miis_broadcast.core.models.openai_tts as _tts_mod
from miis_broadcast.core.models.openai_tts import (
    enqueue_tts_text,
    interrupt_tts,
    set_playback_start_callback,
    start_tts_system,
    stop_tts_system,
    warmup_tts_connection,
)

# ──────────────────────────────────────────────────────────────────────────────
# CLI 參數（main() 解析後填入）
# ──────────────────────────────────────────────────────────────────────────────

VIDEO_PATH: str = str(_PROJECT_ROOT / "examples/vr_test/VR_Frank_basketball.mp4")
NUM_RUNS: int = 10
MAX_DURATION_SEC: int = 60

LIVECC_QUERY = (
    "A real person wearing a VR headset is shown alongside the first-person view of the "
    "basketball game they are playing — they are the player practicing shots at a hoop. "
    "In ONE short present-tense sentence, describe only what the ball and the player are "
    "doing right now: the shot going up, the ball dropping through the net, bouncing off "
    "the rim, or being dribbled and reset. "
    'Use third person ("the player", "the ball"). Do NOT use "I" or "you", do NOT talk '
    "about the video, the camera or the viewer, and do NOT read scoreboard text. Plain words only."
)

FILLER_TEXT = "等等！"

_P1_KEYWORDS = frozenset({
    "scores", "scored",
    "makes the shot", "makes a shot", "makes it", "makes the basket",
    "made the shot", "made a shot",
    "swish", "swishes",
    "through the net", "drops through", "drops in", "goes in",
    "slam", "dunk", "dunks",
    "it's good", "it's in",
    "進球", "得分",
})
_P2_KEYWORDS = frozenset({
    "misses", "missed",
    "bounces off", "off the rim",
    "rebound", "rebounds",
    "out of bounds",
    "未進", "彈框", "籃板", "界外",
})


def _scan_priority(text: str) -> int:
    lower = text.lower()
    for kw in _P1_KEYWORDS:
        if kw in lower:
            return 1
    for kw in _P2_KEYWORDS:
        if kw in lower:
            return 2
    return 3


# ──────────────────────────────────────────────────────────────────────────────
# 統計工具
# ──────────────────────────────────────────────────────────────────────────────

def _stat(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None, "std": None}
    return {
        "n": len(vals),
        "mean": round(mean(vals), 3),
        "median": round(median(vals), 3),
        "min": round(min(vals), 3),
        "max": round(max(vals), 3),
        "std": round(stdev(vals), 3) if len(vals) > 1 else 0.0,
    }


def _fmt(s: dict) -> str:
    if not s or s["n"] == 0:
        return "N/A"
    return (
        f"mean={s['mean']:.3f}s  median={s['median']:.3f}s  "
        f"min={s['min']:.3f}s  max={s['max']:.3f}s  "
        f"std={s['std']:.3f}s  (n={s['n']})"
    )


def _pool(runs: list[dict], key: str) -> list[float]:
    return [v for r in runs for v in r.get(key, [])]


# ──────────────────────────────────────────────────────────────────────────────
# Signal Bridge
# ──────────────────────────────────────────────────────────────────────────────

class _SignalBridge(QtCore.QObject):
    signal_start_livecc = QtCore.Signal(str, str, int)


# ──────────────────────────────────────────────────────────────────────────────
# BenchmarkRouter
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkRouter(QtCore.QObject):
    """
    路由探針：LiveCC → Gemini → TTS，量測各階段延遲。

    延遲定義：
    - livecc_infer    : segment 間隔時間（第一個 segment 從 start_run() 起算，後續為連續間隔）
    - gemini_api      : enqueue_ts → Gemini broadcast（純 API round-trip）
    - tts             : text 送入 WebSocket → 第一個音訊 chunk（即時 callback 收集）
    - pipeline_e2e    : LiveCC signal 到達 → TTS 音訊開始（= Gemini wait + API + TTS）
    - interrupt_e2e   : P1 偵測瞬間 → interrupt TTS 音訊開始
    """

    _sig_p3_to_gemini = QtCore.Signal(float, float, object)

    def __init__(self, gemini_worker: GeminiWorker, parent=None) -> None:
        super().__init__(parent)
        self._gemini_worker = gemini_worker
        self._run_id: int = 0
        self._collecting: bool = False

        # LiveCC 推論延遲：用連續 segment 到達時間差估計每批次推論耗時
        self._livecc_infer: list[float] = []
        self._livecc_prev_seg_ts: float = 0.0  # 上一個 segment 到達時刻
        self._run_trigger_ts: float = 0.0       # start_run() 時刻，供第一個 segment 使用

        self._gemini_api: list[float] = []
        self._segment_arrivals: dict[str, float] = {}
        self._cur_tts: list[float] = []
        self._cur_pipeline_e2e: list[float] = []
        self._cur_interrupt_e2e: list[float] = []
        self._p1_count = 0
        self._p2_count = 0
        self._p3_count = 0

        self._sig_p3_to_gemini.connect(
            gemini_worker.process_segment, QtCore.Qt.QueuedConnection
        )

        # 即時收集 TTS 延遲，不依賴 _perf_stats 在 run 結束後是否被清除
        set_playback_start_callback(self._on_tts_playback)

    # ── TTS 音訊開始 callback（從 audio player thread 呼叫）─────────────────

    def _on_tts_playback(self, payload: dict) -> None:
        try:
            if not self._collecting:
                return
            now = time.time()
            sent_ts = _tts_mod._perf_stats.get("last_text_sent_ts", 0.0)
            ref_ts  = _tts_mod._perf_stats.get("current_ref_ts", 0.0)
            priority = _tts_mod._perf_stats.get("current_priority", 5)

            if sent_ts > 0:
                tts_lat = now - sent_ts
                self._cur_tts.append(tts_lat)
                logging.info("[Bench][TTS] run=%d text→audio=%.3fs", self._run_id, tts_lat)

            if ref_ts > 0:
                e2e = now - ref_ts
                if priority <= 2:
                    self._cur_interrupt_e2e.append(e2e)
                    logging.info("[Bench][Interrupt] run=%d P1→audio=%.3fs", self._run_id, e2e)
                else:
                    self._cur_pipeline_e2e.append(e2e)
                    logging.info("[Bench][Pipeline] run=%d signal→audio=%.3fs", self._run_id, e2e)
        except Exception as _exc:
            logging.warning("[Bench][TTS] callback error (ignored): %s", _exc)

    # ── Run 控制 ─────────────────────────────────────────────────────────────

    def start_run(self, run_id: int) -> None:
        self._run_id = run_id
        self._livecc_infer = []
        self._livecc_prev_seg_ts = 0.0
        self._run_trigger_ts = time.time()
        self._gemini_api = []
        self._segment_arrivals = {}
        self._cur_tts = []
        self._cur_pipeline_e2e = []
        self._cur_interrupt_e2e = []
        self._p1_count = 0
        self._p2_count = 0
        self._p3_count = 0
        self._collecting = True

    def collect_run_stats(self) -> dict:
        self._collecting = False
        return {
            "run_id": self._run_id,
            "livecc_infer": list(self._livecc_infer),
            "gemini_api": list(self._gemini_api),
            "tts": list(self._cur_tts),
            "pipeline_e2e": list(self._cur_pipeline_e2e),
            "interrupt_e2e": list(self._cur_interrupt_e2e),
            "p1_count": self._p1_count,
            "p2_count": self._p2_count,
            "p3_count": self._p3_count,
        }

    # ── Stage 1: LiveCC segment 到達 ─────────────────────────────────────────

    @QtCore.Slot(float, float, object)
    def on_livecc_segment(self, start_t: float, stop_t: float, data: object) -> None:
        arrival_ts = time.time()

        # LiveCC 推論延遲：第一個 segment 從 start_run() 算起，後續從上一個 segment 算起
        ref = self._livecc_prev_seg_ts if self._livecc_prev_seg_ts > 0 else self._run_trigger_ts
        infer_lat = arrival_ts - ref
        self._livecc_infer.append(infer_lat)
        self._livecc_prev_seg_ts = arrival_ts

        # 記錄本 segment 的到達時刻，供 Gemini broadcast 端當 ref_ts
        seg_key = f"{start_t:.2f}-{stop_t:.2f}"
        self._segment_arrivals[seg_key] = arrival_ts

        raw = ""
        if isinstance(data, dict):
            raw = data.get("metadata", {}).get("raw", "") or data.get("event", "")
        elif isinstance(data, str):
            raw = data

        priority = _scan_priority(raw)
        gem_data = {"metadata": {"raw": raw}, "event": "raw_description"}

        if priority == 1:
            self._p1_count += 1
            logging.info(
                "[Bench][LiveCC] P1 run=%d seg=%.1f-%.1fs infer=%.3fs: %r",
                self._run_id, start_t, stop_t, infer_lat, raw[:70],
            )
            interrupt_tts()
            self._gemini_worker.flush_and_abort()
            enqueue_tts_text(
                FILLER_TEXT,
                ref_ts=arrival_ts,
                drop_outdated=True,
                priority=1,
                start_t=start_t,
                stop_t=stop_t,
                log_meta={"log": True},
            )
            self._gemini_worker.enqueue_front(start_t, stop_t, gem_data)

        elif priority == 2:
            self._p2_count += 1
            logging.info(
                "[Bench][LiveCC] P2 run=%d seg=%.1f-%.1fs infer=%.3fs: %r",
                self._run_id, start_t, stop_t, infer_lat, raw[:70],
            )
            self._gemini_worker.enqueue_front(start_t, stop_t, gem_data)

        else:
            self._p3_count += 1
            logging.info(
                "[Bench][LiveCC] P3 run=%d seg=%.1f-%.1fs infer=%.3fs: %r",
                self._run_id, start_t, stop_t, infer_lat, raw[:70],
            )
            self._sig_p3_to_gemini.emit(start_t, stop_t, gem_data)

    # ── Stage 2: Gemini broadcast 到達 ───────────────────────────────────────

    @QtCore.Slot(float, float, object)
    def on_gemini_broadcast(self, start_t: float, stop_t: float, data: object) -> None:
        now = time.time()
        if not isinstance(data, dict):
            return

        enqueue_ts = data.get("_enqueue_ts", 0.0)
        is_background = bool(data.get("_background", False))

        # Gemini API latency
        if enqueue_ts > 0:
            api_lat = now - enqueue_ts
            self._gemini_api.append(api_lat)
            logging.info(
                "[Bench][Gemini] run=%d seg=%.1f-%.1fs api=%.3fs",
                self._run_id, start_t, stop_t, api_lat,
            )

        text = data.get("broadcast_text", "")
        if not text or not data.get("should_speak", True):
            return

        priority = int(data.get("priority", 5))

        # ref_ts = LiveCC signal 到達時刻（讓 TTS callback 量測 pipeline E2E）
        seg_key = f"{start_t:.2f}-{stop_t:.2f}"
        seg_arrival = self._segment_arrivals.get(seg_key, 0.0)
        ref_ts = seg_arrival if seg_arrival > 0 else (enqueue_ts if enqueue_ts > 0 else now)

        enqueue_tts_text(
            text,
            ref_ts=ref_ts,
            drop_outdated=(priority <= 2),
            priority=priority,
            start_t=start_t,
            stop_t=stop_t,
            log_meta={"log": True},
        )


# ──────────────────────────────────────────────────────────────────────────────
# BenchmarkRunner
# ──────────────────────────────────────────────────────────────────────────────

class BenchmarkRunner(QtCore.QObject):
    signal_all_done = QtCore.Signal()

    def __init__(
        self,
        bridge: _SignalBridge,
        livecc_worker: LiveCCWorker,
        router: BenchmarkRouter,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._livecc_worker = livecc_worker
        self._router = router
        self._run_id = 0
        self._all_stats: list[dict] = []

        self._run_timer = QtCore.QTimer(self)
        self._run_timer.setSingleShot(True)
        self._run_timer.timeout.connect(self._on_timeout)

        livecc_worker.signal_finished.connect(self._on_livecc_finished)
        livecc_worker.signal_error.connect(self._on_livecc_error)

    @QtCore.Slot()
    def start(self) -> None:
        self._start_next_run()

    def _start_next_run(self) -> None:
        self._run_id += 1
        if self._run_id > NUM_RUNS:
            self._finish_all()
            return

        print(f"\n{'='*60}")
        print(f"[Bench] Run {self._run_id}/{NUM_RUNS} 開始")
        print(f"{'='*60}")

        self._router.start_run(self._run_id)
        self._run_timer.start(int(MAX_DURATION_SEC * 1000))
        self._bridge.signal_start_livecc.emit(VIDEO_PATH, LIVECC_QUERY, self._run_id)

    @QtCore.Slot()
    def _on_timeout(self) -> None:
        logging.info("[Bench] Run %d: %ds timeout → 停止 LiveCC", self._run_id, MAX_DURATION_SEC)
        self._livecc_worker.requestStop()

    @QtCore.Slot(str)
    def _on_livecc_error(self, msg: str) -> None:
        logging.error("[Bench] Run %d LiveCC error: %s", self._run_id, msg)
        self._run_timer.stop()
        QtCore.QTimer.singleShot(0, self._collect_and_next)

    @QtCore.Slot(int)
    def _on_livecc_finished(self, run_id: int) -> None:
        if run_id != self._run_id:
            return
        self._run_timer.stop()
        # 等 500ms 讓最後的 TTS 音訊開始（callback 收資料），再收統計
        QtCore.QTimer.singleShot(500, self._collect_and_next)

    @QtCore.Slot()
    def _collect_and_next(self) -> None:
        interrupt_tts()
        self._router._gemini_worker.flush_and_abort()

        stats = self._router.collect_run_stats()
        self._all_stats.append(stats)

        print(f"\n[Bench] Run {self._run_id} 結果：")
        print(f"  LiveCC infer  : {_fmt(_stat(stats['livecc_infer']))}  ← live_cc()→signal")
        print(f"  Gemini API    : {_fmt(_stat(stats['gemini_api']))}")
        print(f"  TTS           : {_fmt(_stat(stats['tts']))}")
        print(f"  Pipeline E2E  : {_fmt(_stat(stats['pipeline_e2e']))}  ← signal→audio")
        print(f"  Interrupt E2E : {_fmt(_stat(stats['interrupt_e2e']))}  ← P1→audio")
        print(f"  Events        : P1={stats['p1_count']}  P2={stats['p2_count']}  P3={stats['p3_count']}")

        # run 間等待讓 TTS WebSocket 恢復
        QtCore.QTimer.singleShot(5000, self._start_next_run)

    def _finish_all(self) -> None:
        agg = {
            "num_runs": len(self._all_stats),
            "duration_sec_per_run": MAX_DURATION_SEC,
            "livecc_infer":  _stat(_pool(self._all_stats, "livecc_infer")),
            "gemini_api":    _stat(_pool(self._all_stats, "gemini_api")),
            "tts":           _stat(_pool(self._all_stats, "tts")),
            "pipeline_e2e":  _stat(_pool(self._all_stats, "pipeline_e2e")),
            "interrupt_e2e": _stat(_pool(self._all_stats, "interrupt_e2e")),
            "per_run": [
                {
                    "run_id":        r["run_id"],
                    "livecc_infer":  _stat(r["livecc_infer"]),
                    "gemini_api":    _stat(r["gemini_api"]),
                    "tts":           _stat(r["tts"]),
                    "pipeline_e2e":  _stat(r["pipeline_e2e"]),
                    "interrupt_e2e": _stat(r["interrupt_e2e"]),
                    "p1": r["p1_count"],
                    "p2": r["p2_count"],
                    "p3": r["p3_count"],
                }
                for r in self._all_stats
            ],
        }

        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = Path(VIDEO_PATH).stem
        out_path = _PROJECT_ROOT / "eval" / f"bench_{stem}_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"video": VIDEO_PATH, "aggregate": agg, "runs": self._all_stats}, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE — 彙整結果")
        print("=" * 70)
        print(f"\n【測項 1】各系統延遲（{NUM_RUNS} runs × {MAX_DURATION_SEC}s）")
        print(f"  LiveCC 推論         : {_fmt(agg['livecc_infer'])}  ← live_cc()→signal")
        print(f"    （VLM 視覺推論耗時：呼叫 live_cc() → segment signal 到達主執行緒）")
        print(f"  Gemini API          : {_fmt(agg['gemini_api'])}")
        print(f"  TTS（text→audio）   : {_fmt(agg['tts'])}")
        print(f"  Pipeline E2E        : {_fmt(agg['pipeline_e2e'])}")
        print(f"    （LiveCC signal 到達 → TTS 音訊開始，= Gemini queue + API + TTS）")
        print(f"\n【測項 2】Interrupt 延遲")
        print(f"  P1 偵測 → 音訊開始  : {_fmt(agg['interrupt_e2e'])}")
        print(f"\n詳細結果已存至: {out_path}")
        print("=" * 70)

        stop_tts_system()
        self.signal_all_done.emit()


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LiveCC → Gemini → OpenAI TTS 端到端延遲基準測試",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--video", "-v",
        default=str(_PROJECT_ROOT / "examples/vr_test/VR_Frank_basketball.mp4"),
        help="輸入影片路徑",
    )
    p.add_argument(
        "--runs", "-n",
        type=int,
        default=10,
        help="重複跑幾次 (每次結束後自動重啟推論)",
    )
    p.add_argument(
        "--duration", "-d",
        type=int,
        default=60,
        help="每次推論最長秒數 (超時後自動停止)",
    )
    p.add_argument(
        "--device",
        type=int,
        default=0,
        help="LiveCC GPU device ID",
    )
    p.add_argument(
        "--log",
        default=str(_PROJECT_ROOT / "eval" / "bench_latency.log"),
        help="日誌輸出路徑",
    )
    return p.parse_args()


def main() -> None:
    global VIDEO_PATH, NUM_RUNS, MAX_DURATION_SEC

    args = _parse_args()
    VIDEO_PATH = args.video
    NUM_RUNS = args.runs
    MAX_DURATION_SEC = args.duration

    if not Path(VIDEO_PATH).exists():
        print(f"[Bench] ERROR: 影片不存在: {VIDEO_PATH}")
        sys.exit(1)

    log_path = args.log
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
        force=True,
    )
    logging.info("[Bench] 影片: %s", VIDEO_PATH)
    logging.info("[Bench] runs=%d  duration=%ds  device=%d", NUM_RUNS, MAX_DURATION_SEC, args.device)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    livecc_thread = QtCore.QThread()
    livecc_worker = LiveCCWorker(device_id=args.device)
    livecc_worker.moveToThread(livecc_thread)
    livecc_thread.start()

    gemini_thread = QtCore.QThread()
    gemini_worker = GeminiWorker()
    gemini_worker.moveToThread(gemini_thread)
    gemini_thread.started.connect(gemini_worker.initialize)
    gemini_thread.start()

    start_tts_system()
    warmup_tts_connection()

    bridge = _SignalBridge()
    bridge.signal_start_livecc.connect(livecc_worker.runInference, QtCore.Qt.QueuedConnection)

    router = BenchmarkRouter(gemini_worker)

    livecc_worker.signal_segment.connect(router.on_livecc_segment, QtCore.Qt.QueuedConnection)
    gemini_worker.signal_broadcast.connect(router.on_gemini_broadcast, QtCore.Qt.QueuedConnection)

    model_loaded = threading.Event()

    @QtCore.Slot()
    def _on_model_loaded():
        model_loaded.set()

    livecc_worker.signal_model_loaded.connect(_on_model_loaded, QtCore.Qt.QueuedConnection)

    print("[Bench] 正在載入 LiveCC 模型，請稍候...")
    QtCore.QMetaObject.invokeMethod(livecc_worker, "loadModel", QtCore.Qt.QueuedConnection)

    deadline = time.time() + 300
    while not model_loaded.is_set() and time.time() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 200)

    if not model_loaded.is_set():
        print("[Bench] ERROR: LiveCC 模型載入逾時，退出。")
        sys.exit(1)

    print("[Bench] 模型載入完成，準備開始基準測試。")

    runner = BenchmarkRunner(bridge, livecc_worker, router)

    all_done = threading.Event()

    @QtCore.Slot()
    def _on_all_done():
        all_done.set()

    runner.signal_all_done.connect(_on_all_done)

    QtCore.QTimer.singleShot(1500, runner.start)

    deadline = time.time() + NUM_RUNS * (MAX_DURATION_SEC + 60) + 120
    while not all_done.is_set() and time.time() < deadline:
        app.processEvents(QtCore.QEventLoop.AllEvents, 200)

    if not all_done.is_set():
        print("[Bench] WARNING: 基準測試逾時未完成，強制退出。")

    livecc_worker.requestStop()
    livecc_thread.quit()
    livecc_thread.wait(5000)
    gemini_thread.quit()
    gemini_thread.wait(5000)
    stop_tts_system()

    print("[Bench] 完畢。")
    sys.exit(0)


if __name__ == "__main__":
    main()
