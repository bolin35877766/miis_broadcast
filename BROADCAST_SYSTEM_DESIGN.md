# MIIS VR 廣播系統設計藍圖 / MIIS VR Broadcast System Design Blueprint

> 本文件為雙語版本：繁體中文（供人閱讀與決策）與 English（供 AI 開發者實作參考）。  
> This document is dual-language: Traditional Chinese (human-readable) and English (AI implementation blueprint).  
> **最後更新 / Last Updated**: 2026-05-29

---

# ═══════════════════════════════════════
# 第一部分：繁體中文版（人類閱讀）
# ═══════════════════════════════════════

## 一、系統定位與目標

**系統名稱**：MIIS VR 即時播報系統  
**核心目標**：針對 VR 籃球直播，實現端對端延遲 **< 1.5 秒**（事件發生 → 語音輸出）的即時語音播報。  
**核心比喻**：快慢刀——兩個 AI 播報員分工合作，一快一慢，快者在關鍵時刻打斷慢者。

---

## 二、快慢刀架構（核心設計）

### 2.1 角色定義

| 角色 | 模型 | 輸出長度 | 優先級 |
|---|---|---|---|
| **快刀** | LiveCC（本地 GPU, Qwen2VL-7B） | 12 tokens（極短） | P1 / P2 |
| **慢刀** | Gemini API（雲端, text-only） | 50 tokens（約 2–3 秒） | P3 / P4 |
| **發聲器** | OpenAI Realtime TTS（WebSocket） | 語音輸出 | — |

### 2.2 核心設計原則

1. **快刀輸出可以短，但不能沉默**：LiveCC 無明確關鍵動作時，輸出曖昧過渡句（如「雙方還在尋找機會」），而非靜音或 "No active play visible."。
2. **慢刀不看畫面**：Gemini 完全不接收圖片。視覺資訊透過 LiveCC 的文字描述（context pool）間接傳遞。
3. **慢刀受 Backpressure 節制**：Gemini 不靠固定 timer，而是監聽 TTS 佇列水位，只在剩餘播報時間低於閾值（1.0s）時才發下一個 API 請求，防止佇列溢位。
4. **P1 打斷走最短路徑**：LiveCC 偵測到 P1 關鍵字 → 直接送 TTS，完全繞過 Gemini。
5. **打斷後有留白**：P1 事件播完後，強制靜默 1.0 秒，才讓 Gemini 背景播報恢復，符合真實播報節奏。
6. **慢刀句間有微停頓**：每句 Gemini 播報之間強制 0.3 秒靜默，維持流暢感並減輕聽覺疲勞。

---

## 三、第一項改動：快刀參數與曖昧過渡詞

### 3.1 `configs/models.yml` 參數修改

| 參數 | 舊值 | 新值 | 原因 |
|---|---|---|---|
| `max_new_tokens` | 48 | **12** | 極壓延遲，快刀只需短句即可觸發打斷 |
| `streaming_fps_frames` | 6 | **3** | 減少視覺 token，加速推論 |
| `camera.infer_interval` | 1.5 | **1.0** | 縮短推論週期，提高事件偵測時效性 |

### 3.2 `configs/livecc_prompts.yml` Prompt 修改

**兩個改動點**：
1. 加入分割畫面空間語義映射（左側 VR 第一人稱、右側第三人稱全局）
2. 廢除 "No active play visible."，改為要求輸出曖昧過渡句

**新增 Prompt 樣板（英文指令，模型響應更好）**：
```
The screen uses a LEFT-RIGHT split-screen format:
- LEFT half: VR player's first-person perspective.
- RIGHT half: Third-person full-court view.
Naming: left → "the attacker"/"the defender"; right → "Player <N>".
Output 1 short present-tense sentence. If a key event (score/dunk/steal/block/foul)
is happening, name it. If NOT, output a neutral transition phrase — never "No active
play visible." Vary from: "Both teams still looking for an opening.",
"Players continue off-ball movement.", "Pace has slowed, both sides reset.",
"Setting up for the next offensive push.", "Defense holding, waiting for the attack."
Rules: plain sentences ONLY.
```

---

## 四、第二項改動：雙視角原圖水平拼接

### 4.1 設計決策

- **不做任何縮放或 Letterboxing**：追求速度，直接 `np.concatenate` 左右原圖，允許高度不同時裁齊較矮的那邊。
- **時間戳對齊**：以兩個 buffer 各自最早的取樣幀時間戳取最小值作為 `t_start`。
- **解析度**：各路 640×480 → 拼接後 1280×480，符合 Qwen2VL 上限（~1344×896 等效面積）。

### 4.2 新增函式 `build_composite_clip()`（`workers/livecc.py`）

位於現有 `build_clip_from_buffer()` 之後。邏輯：
1. 從兩個 buffer 各取最近 `window_sec` 的幀
2. 取幀數較少的那邊，step 降採樣至 `target_fps`
3. 對每對幀：裁齊高度 → `np.concatenate(..., axis=1)` → BGR 轉 RGB
4. `np.stack` 所有複合幀 → 組成 `VideoClip`

### 4.3 新增類別 `LiveCCSplitCameraWorker`（`workers/livecc.py`）

與 `LiveCCCameraWorker` 平行存在，保留完全相同的 Signal 介面（`signal_segment`, `signal_finished`, `signal_error`）。

主要差異：
- 持有 `_fp_buffer`（第一人稱）和 `_tp_buffer`（第三人稱）兩個獨立 buffer
- 暴露兩個 Slot：`push_fp_frame(frame, t)` 和 `push_tp_frame(frame, t)`
- `runCameraInference()` 呼叫 `build_composite_clip()` 替代原有函式

### 4.4 `gui.py` 配合改動

- `_initCameraWorker()` 改為建立 `LiveCCSplitCameraWorker`
- 建立兩個 `CameraThread`（`index=0` 第一人稱、`index=1` 第三人稱）
- 各自的 `signal_frame` 分別連接到 `push_fp_frame` 和 `push_tp_frame`

> **執行緒說明**：Signal 連接到 lambda（非 QObject slot）時，PySide6 使用 DirectConnection，lambda 在 **CameraThread**（emitter）執行，而非 GUI thread。色彩轉換和 copy 不阻塞介面渲染。

---

## 五、第三項改動：慢刀背景輪詢與 Backpressure

### 5.1 `GeminiBackgroundWorker` 運作邏輯

```
run_background_loop() 迴圈：
  if _paused: sleep 200ms; continue
  remaining = tts_worker.get_queue_remaining_sec()
  if remaining > WATERMARK (1.0s): sleep 200ms; continue   ← Backpressure 閘
  context = build_context(context_pool + match_state + game_context)
  stream_gemini(context, text_only=True)
  → signal_broadcast(P3/P4) → on_segment() → TTS 佇列
  sleep 300ms   ← 句間微停頓
  回到迴圈開頭
```

### 5.2 TTS 水位查詢（`OpenAITTSWorker`）

新增方法 `get_queue_remaining_sec() -> float`：
- 用 `threading.Lock` 保護一個浮點計數器
- `speak()` 被呼叫時：以 **CJK 感知公式**加入預估時長（見下）
- TTS **自然播完**一個段落後：減去實際已播秒數，emit `signal_tts_done`

**[修正] CJK 感知時長估算**：中文沒有空格分詞，`len(text.split())` 在純中文時永遠回傳 1，導致水位嚴重低估，反壓完全失效。應改用字元計數：
```
CJK 字符（一-鿿）：約 4 字/秒（1.0x 速度）
英文單字：約 2.5 words/秒（1.0x 速度）
估算 = (cjk_count / 4.0 + other_words / 2.5) / speed
```

**[修正] 中斷時絕不 emit signal_tts_done**：`interrupt()` 被呼叫時設 `_interrupted = True`，segment 自然播完的回調判斷 `if not _interrupted` 才 emit。這是防止 P1 連環 Race Condition 的關鍵（見第十一節）。

### 5.3 context_pool

- 類型：`deque(maxlen=3)`，存最近 3 句 LiveCC P3 輸出
- 由 `gui.py _route_segment()` 的 P3 分支透過 **`signal_livecc_context`** Signal 寫入（見第六節修正）
- `GeminiBackgroundWorker` 的 `update_context(desc)` Slot 在 worker thread 接收並 append

---

## 六、第四項改動：TTS 節奏與打斷留白

### 6.1 `_route_segment()` 新路由邏輯

| 快刀輸出 | 動作 |
|---|---|
| **P1** | `gemini_bg_worker.pause()` → `signal_tts_interrupt` → LiveCC 文字直送 TTS → `_post_p1_pending = True` → `signal_p1_confirmed`（KV cache 重置） |
| **P2** | LiveCC 文字直送 TTS（P2 優先），Gemini 繼續不打斷 |
| **P3** | `signal_livecc_context.emit(description)`（Qt Signal，QueuedConnection）→ `update_context()` slot |

> **[修正] P3 改用 Signal**：原設計使用 `QMetaObject.invokeMethod + Q_ARG(str, ...)` 雖然有效，但 slot 名稱為魔術字串，重構時不易維護。改用 `signal_livecc_context = QtCore.Signal(str)` 更安全且符合 Qt 慣例。

### 6.2 P1 打斷後的 1.0 秒留白流程

```
LiveCC P1 文字 → TTS 播放
     ↓ TTS 自然播完（不被再次中斷）
signal_tts_done → MainWindow._on_tts_done()
     ↓ if _post_p1_pending
QTimer.singleShot(1000, _resume_gemini_background)
     ↓ 1.0 秒後
gemini_bg_worker.resume()   ← Gemini 重新開始背景輪詢
```

### 6.3 慢刀句間 0.3 秒停頓

在 `GeminiBackgroundWorker.run_background_loop()` 每次 `signal_broadcast` emit 後執行 `QtCore.QThread.msleep(300)`。運行在 worker thread，不阻塞 GUI。

---

## 七、完整資料流圖

```
【輸入層】
CameraThread(index=0) → signal_frame → lambda(DirectConn, CameraThread) → push_fp_frame() ─┐
CameraThread(index=1) → signal_frame → lambda(DirectConn, CameraThread) → push_tp_frame() ─┘
                                                                                             ↓
                                     build_composite_clip() → np.concatenate(axis=1) → 1280×480
                                                                                             ↓
【快刀：LiveCCSplitCameraWorker】
  Qwen2VL-7B | max_tokens=12 | fps_frames=3 | interval=1.0s
                                                                                             ↓
                                     signal_segment(start_t, stop_t, parsed_dict)
                                                                                             ↓
【路由：_route_segment()】
          ├─ P1 → pause Gemini | TTS interrupt | LiveCC→TTS直送 | _post_p1_pending=True
          ├─ P2 → LiveCC→TTS直送（P2優先）
          └─ P3 → signal_livecc_context.emit() → update_context() [QueuedConnection]

【慢刀：GeminiBackgroundWorker】
  Backpressure 輪詢 → stream_gemini(text_only) → signal_broadcast(P3/P4)
  → on_segment() → TTS 佇列 → msleep(300) → 下一輪

【留白：signal_tts_done → _on_tts_done】
  條件：自然播完（_interrupted == False）才觸發
  _post_p1_pending → QTimer(1000ms) → gemini_bg_worker.resume()

【輸出：OpenAI Realtime TTS WebSocket】
  coral voice | 1.5x speed | P1/P2 插隊 | P3/P4 佇列
  自然播完 → signal_tts_done（中斷時不發）
```

---

## 八、新增介面清單

### 新 Signal

| Signal | 發出者 | 接收者 | 用途 |
|---|---|---|---|
| `signal_tts_done` | `OpenAITTSWorker` | `MainWindow` | 觸發 P1 留白計時（**自然播完才發**） |
| `signal_broadcast` | `GeminiBackgroundWorker` | `MainWindow.on_segment()` | 背景播報文字輸出 |
| `signal_livecc_context` | `MainWindow` | `GeminiBackgroundWorker.update_context()` | P3 描述寫入 context_pool（替代 QMetaObject） |

### 新 Slot / Method

| 方法 | 所在類別 | 用途 |
|---|---|---|
| `push_fp_frame(frame, t)` | `LiveCCSplitCameraWorker` | 接收第一人稱幀 |
| `push_tp_frame(frame, t)` | `LiveCCSplitCameraWorker` | 接收第三人稱幀 |
| `pause()` | `GeminiBackgroundWorker` | P1 觸發時暫停背景輪詢 |
| `resume()` | `GeminiBackgroundWorker` | 留白結束後恢復 |
| `update_context(desc)` | `GeminiBackgroundWorker` | 寫入 context_pool |
| `run_background_loop()` | `GeminiBackgroundWorker` | 主輪詢迴圈 |
| `get_queue_remaining_sec()` | `OpenAITTSWorker` | 執行緒安全的水位查詢（CJK 感知） |

---

## 九、不需改動的現有系統

- `GeminiWorker`（檔案模式/舊架構，保留並行）
- `LiveCCWorker`（檔案模式推論）
- `LiveCCCameraWorker`（單鏡頭模式，保留並行新類別）
- Priority 關鍵字集合 `_P1_KEYWORDS`, `_P2_KEYWORDS`
- `MatchTracker` singleton
- RAG context retriever
- 所有 Gemini system prompts（objective / hype / calm / trash_talk）
- `ChatterboxTTSWorker`

---

## 十、確認的設計隱患與處置

| 隱患 | 判定 | 嚴重程度 | 處置 |
|---|---|---|---|
| Lambda 在 GUI thread 執行導致卡頓 | **幻覺**（DirectConn → CameraThread） | 無 | 加說明注釋，不改程式碼 |
| 中文播報 Backpressure 失效 | **真實** | 高 | ✅ 改用 CJK 感知估算公式 |
| NumPy GC Stop-The-World | **幻覺**（NumPy 用 C-level malloc） | 無 | 無需處理 |
| QMetaObject 字串 slot 不易維護 | **真實（低風險）** | 低 | ✅ 改用 `signal_livecc_context` Signal |
| P1 連環時 signal_tts_done 競爭 | **真實 Race Condition** | 極高 | ✅ 中斷時設 `_interrupted` 旗標，禁止發出 done |

---

## 十一、建議實作順序

1. 改 `configs/models.yml`（三個參數，零風險）
2. 改 `configs/livecc_prompts.yml`（更新 query，測試輸出品質）
3. 實作 `build_composite_clip()`（用假 buffer 單元測試）
4. 實作 `LiveCCSplitCameraWorker`（連接兩路 CameraThread 測試）
5. 在 `OpenAITTSWorker` 加 `signal_tts_done`（帶 `_interrupted` 防護）、`get_queue_remaining_sec()`（CJK 感知）
6. 實作 `GeminiBackgroundWorker` 最小版（無 Backpressure，先驗證 text-only 正確）
7. 加入 Backpressure 水位控制
8. 改 `_route_segment()` 路由（P1直送TTS，P3 → `signal_livecc_context`）
9. 接 `signal_tts_done` → P1 留白計時
10. 加句間 0.3s msleep，測試整體節奏

---
---

# ═══════════════════════════════════════
# Part Two: English Blueprint (AI Reference)
# ═══════════════════════════════════════

## 1. System Overview

**System Name**: MIIS VR Real-Time Broadcast System  
**Core Goal**: End-to-end latency **< 1.5 seconds** (event → first audio) for VR basketball live streams.  
**Architecture**: Fast-Slow Blade (快慢刀) — LiveCC interrupts Gemini at key moments; Gemini provides uninterrupted background commentary with backpressure control.

---

## 2. Role Definitions

| Role | Model | Output | Priority |
|---|---|---|---|
| **Fast Blade** | LiveCC (local GPU, Qwen2VL-7B) | 12 tokens (terse, immediate) | P1 / P2 |
| **Slow Blade** | Gemini API (cloud, text-only) | 50 tokens (~2–3s of speech) | P3 / P4 |
| **Voice** | OpenAI Realtime TTS (WebSocket) | Audio output only | — |

---

## 3. Change 1: Fast Blade Parameters & Ambiguous Transition Phrases

### 3.1 `configs/models.yml` — Complete Updated File

```yaml
classifiers:
  livecc_7b:
    desc: LiveCC (streaming video captioner)
    model_path: chenjoya/LiveCC-7B-Instruct
    device_id: 0
    fps: 4.0
    initial_fps_frames: 6
    streaming_fps_frames: 3        # CHANGED from 6
    max_new_tokens: 12             # CHANGED from 48
    headroom: 1024
    mm_window_sec: 8.0
    carry_text_max_chars: 0
    carry_recent_k: 0
    generation:
      temperature: 0.3
      top_p: 0.75
      top_k: 15
      repetition_penalty: 1.5
      no_repeat_ngram_size: 4
    camera:
      window_sec: 1.5
      target_fps: 2.0
      infer_interval: 1.0          # CHANGED from 1.5
      memory_reset_every: 5
```

### 3.2 `configs/livecc_prompts.yml` — Updated `livecc_query`

Replace the entire `livecc_query` value with:

```yaml
livecc_query: |-
  The screen uses a LEFT-RIGHT split-screen format:
  - LEFT half: VR player's first-person perspective — observe this player's movement
    and on-ball actions.
  - RIGHT half: Third-person full-court view — shows overall game flow, teammate
    positions, and court layout.

  Naming conventions:
  - First-person view (left): "the attacker" (ball-handler) or "the defender".
  - Third-person view (right): "Player <N>" by jersey number.

  Integrate both perspectives. Output exactly 1 short, objective, present-tense sentence
  describing the current game moment.

  If a clear key event is happening (score, dunk, steal, block, foul), name it explicitly.
  If NO key event is happening, output a neutral transition phrase — do NOT output
  "No active play visible." Vary freely from these examples:
  "Both teams are still looking for an opening."
  "Players continue their off-ball movement."
  "The pace has slowed as both sides reset."
  "Setting up for the next offensive push."
  "Defense holding position, waiting for the attack."
  "Ball movement keeping the defense honest."
  "Both sides maintaining their spacing."

  Rules: plain sentences ONLY — no JSON, no markdown, no special symbols.
```

---

## 4. Change 2: Dual-View Horizontal Composite (`workers/livecc.py`)

### 4.1 Design Decisions

- **No aspect-ratio preservation, no letterboxing, no scaling**: Raw `np.concatenate(axis=1)` for maximum speed. If heights differ, crop the taller frame to match the shorter.
- **Time-sync**: Both buffers downsample to `target_fps` independently. Use `min(fp_sampled[0].t, tp_sampled[0].t)` as `t_start`.
- **Resolution**: Each camera at 640×480 → composite 1280×480. Within Qwen2VL's ViT budget (~1344×896 equivalent).
- **Thread note**: Lambdas connected to signals use DirectConnection in PySide6. The lambda runs in `CameraThread` (emitter), NOT the GUI thread. Color conversion and `.copy()` do not block GUI rendering.

### 4.2 New Function: `build_composite_clip()`

**File**: `src/miis_broadcast/workers/livecc.py`  
**Location**: After existing `build_clip_from_buffer()`.

```python
def build_composite_clip(
    fp_buffer: Deque[FrameItem],
    tp_buffer: Deque[FrameItem],
    window_sec: float,
    target_fps: float,
) -> Optional["VideoClip"]:
    """
    Build a VideoClip by horizontally concatenating frames from two camera buffers.
    Left = first-person VR (fp_buffer), Right = third-person court (tp_buffer).
    No scaling or aspect ratio correction — raw concatenation for maximum speed.
    Heights are aligned by cropping the taller frame to match the shorter.
    NumPy memory is C-level malloc, not tracked by Python's cyclic GC — no GC
    pause concern at the 1.0s inference cadence used here.
    """
    from ..core.models.livecc_transformers import VideoClip

    if not fp_buffer or not tp_buffer:
        return None

    fp_now = fp_buffer[-1].t
    tp_now = tp_buffer[-1].t
    fp_frames = [fi for fi in fp_buffer if fi.t >= fp_now - window_sec]
    tp_frames = [fi for fi in tp_buffer if fi.t >= tp_now - window_sec]

    if len(fp_frames) < 2 or len(tp_frames) < 2:
        return None

    n = min(len(fp_frames), len(tp_frames))
    duration = max(
        fp_frames[-1].t - fp_frames[0].t,
        tp_frames[-1].t - tp_frames[0].t,
        1e-6,
    )
    step = max(1, int(round(n / (target_fps * duration))))
    fp_sampled = fp_frames[::step]
    tp_sampled = tp_frames[::step]
    count = min(len(fp_sampled), len(tp_sampled))

    if count < 1:
        return None

    composites = []
    for i in range(count):
        try:
            fp_rgb = fp_sampled[i].frame[..., ::-1]  # BGR → RGB
            tp_rgb = tp_sampled[i].frame[..., ::-1]  # BGR → RGB
            h = min(fp_rgb.shape[0], tp_rgb.shape[0])
            composite = np.concatenate([fp_rgb[:h], tp_rgb[:h]], axis=1)
            composites.append(composite)
        except Exception as e:
            logging.exception("[build_composite_clip] frame processing failed: %s", e)
            return None

    try:
        frames_np = np.stack(composites).astype(np.uint8)
    except Exception as e:
        logging.exception("[build_composite_clip] np.stack failed: %s", e)
        return None

    t_start = min(fp_sampled[0].t, tp_sampled[0].t)
    return VideoClip(frames=frames_np, fps=float(target_fps), t_start=t_start)
```

### 4.3 New Class: `LiveCCSplitCameraWorker`

**File**: `src/miis_broadcast/workers/livecc.py`  
**Location**: After `LiveCCCameraWorker`.  
**Signal interface**: Identical to `LiveCCCameraWorker` — `signal_segment`, `signal_finished`, `signal_error`.

```python
class LiveCCSplitCameraWorker(QtCore.QObject):
    """
    Dual-camera variant of LiveCCCameraWorker.
    Maintains two separate frame buffers (first-person VR + third-person court),
    composites them side-by-side via build_composite_clip(), and runs LiveCC inference
    on the composite. Signal interface is identical to LiveCCCameraWorker.
    """
    signal_model_loaded = QtCore.Signal()
    signal_segment = QtCore.Signal(float, float, object)
    signal_finished = QtCore.Signal()
    signal_error = QtCore.Signal(str)

    def __init__(
        self,
        device_id: int = 0,
        window_sec: float = 1.5,
        target_fps: float = 2.0,
        infer_interval: float = 1.0,
        memory_reset_every: int = 5,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.device_id = device_id
        self.window_sec = window_sec
        self.target_fps = target_fps
        self.infer_interval = infer_interval
        self.memory_reset_every = memory_reset_every
        self.livecc: Optional[LiveCCInfer] = None
        self._stop_requested = False
        self.response_prefix: str = ""
        self._fp_buffer: Deque[FrameItem] = deque(maxlen=180)
        self._tp_buffer: Deque[FrameItem] = deque(maxlen=180)
        self._state: Dict[str, Any] = {}
        self._query: str = ""

    @QtCore.Slot()
    def loadModel(self) -> None:
        try:
            self.livecc = LiveCCInfer(device_id=self.device_id)
            self.signal_model_loaded.emit()
        except Exception as e:
            logging.exception("[LiveCCSplitCameraWorker] model load failed")
            self.signal_error.emit(str(e))

    @QtCore.Slot(np.ndarray, float)
    def push_fp_frame(self, frame: np.ndarray, t: float) -> None:
        """Receive first-person VR frame (BGR format)."""
        if not self._stop_requested:
            self._fp_buffer.append(FrameItem(t=t, frame=frame.copy()))

    @QtCore.Slot(np.ndarray, float)
    def push_tp_frame(self, frame: np.ndarray, t: float) -> None:
        """Receive third-person court frame (BGR format)."""
        if not self._stop_requested:
            self._tp_buffer.append(FrameItem(t=t, frame=frame.copy()))

    @QtCore.Slot(str)
    def runCameraInference(self, query: str) -> None:
        if self.livecc is None:
            self.signal_error.emit("LiveCCSplitCameraWorker: model not loaded")
            return

        self._query = query or ""
        self._stop_requested = False
        self._state = {}
        self._fp_buffer.clear()
        self._tp_buffer.clear()

        inference_count = 0
        last_infer_t = time.time()

        logging.info("[LiveCCSplitCameraWorker] Start split-camera inference loop")
        try:
            while not self._stop_requested:
                now = time.time()
                if now - last_infer_t < self.infer_interval:
                    QtCore.QThread.msleep(100)
                    continue

                clip = build_composite_clip(
                    self._fp_buffer, self._tp_buffer,
                    self.window_sec, self.target_fps,
                )
                if clip is None:
                    QtCore.QThread.msleep(100)
                    continue

                last_infer_t = now
                inference_count += 1
                if inference_count % self.memory_reset_every == 0:
                    self._state = {}
                    logging.info("[LiveCCSplitCameraWorker] Memory reset (count=%d)", inference_count)

                try:
                    for (start_ts, stop_ts), text, self._state in self.livecc.live_cc_from_frames(
                        clip=clip,
                        query=self._query,
                        state=self._state,
                        response_prefix=self.response_prefix,
                    ):
                        parsed = self.livecc._parse_visual_json(text)
                        self.signal_segment.emit(float(start_ts), float(stop_ts), parsed)
                except Exception as e:
                    logging.exception("[LiveCCSplitCameraWorker] inference error")
                    self.signal_error.emit(str(e))
                    break
        finally:
            logging.info("[LiveCCSplitCameraWorker] Inference loop finished")
            self.signal_finished.emit()

    @QtCore.Slot()
    def requestStop(self) -> None:
        self._stop_requested = True

    @QtCore.Slot()
    def requestMemoryReset(self) -> None:
        """P1 event triggered — reset KV cache. Must be called via QueuedConnection."""
        self._state = {}
        logging.info("[LiveCCSplitCameraWorker] Memory reset triggered by P1 event")
```

### 4.4 `gui.py` — Updated `_initCameraWorker()`

```python
def _initCameraWorker(self) -> None:
    from .workers.livecc import LiveCCSplitCameraWorker

    self.cam_worker_thread = QtCore.QThread(self)
    camera_cfg = self.configs.get("model", {}).get("classifier", {}).get("camera", {})
    device_id = int(self.configs.get("model", {}).get("classifier", {}).get("device_id", 0))

    self.cam_worker = LiveCCSplitCameraWorker(
        device_id=device_id,
        window_sec=float(camera_cfg.get("window_sec", 1.5)),
        target_fps=float(camera_cfg.get("target_fps", 2.0)),
        infer_interval=float(camera_cfg.get("infer_interval", 1.0)),
        memory_reset_every=int(camera_cfg.get("memory_reset_every", 5)),
    )
    self.cam_worker.livecc = self.livecc_model
    self.cam_worker.moveToThread(self.cam_worker_thread)
    self.cam_worker.signal_segment.connect(self._route_segment)
    self.cam_worker.signal_error.connect(self.on_error)
    self.signal_start_camera_livecc.connect(self.cam_worker.runCameraInference)
    self.signal_p1_confirmed.connect(
        self.cam_worker.requestMemoryReset, QtCore.Qt.QueuedConnection
    )

    # Two independent CameraThreads.
    # Lambdas use DirectConnection → execute in CameraThread, NOT the GUI thread.
    # Color conversion and .copy() do not block GUI rendering.
    self._camera_fp_thread = CameraThread(camera_index=0)   # VR first-person
    self._camera_tp_thread = CameraThread(camera_index=1)   # third-person court

    self._camera_fp_thread.signal_frame.connect(
        lambda frame_rgb: self.cam_worker.push_fp_frame(
            frame_rgb[..., ::-1].copy(),
            time.time() - getattr(self, "camera_start_time", time.time()),
        )
    )
    self._camera_tp_thread.signal_frame.connect(
        lambda frame_rgb: self.cam_worker.push_tp_frame(
            frame_rgb[..., ::-1].copy(),
            time.time() - getattr(self, "camera_start_time", time.time()),
        )
    )

    self.cam_worker_thread.start()
```

---

## 5. Change 3: Slow Blade with Backpressure (`workers/gemini.py`)

### 5.1 `GeminiBackgroundWorker` — New Class

**File**: `src/miis_broadcast/workers/gemini.py`  
**Location**: After existing `GeminiWorker` class.

```python
class GeminiBackgroundWorker(QtCore.QObject):
    """
    Continuously generates background broadcast commentary (slow blade).

    Backpressure control: fires a new Gemini call only when TTS queue remaining
    time drops below WATERMARK_SEC (1.0s). Prevents unbounded queue growth when
    Gemini API (~300-500ms/call) outpaces TTS playback (~3-4s/sentence).

    Pause/resume: P1 events call pause() to halt the loop; after the mandatory
    1.0s post-interrupt silence, resume() re-enables it.

    Thread safety: _paused and _abort_current are bool flags (CPython GIL-safe
    for single assignment). context_pool is a deque; snapshot with list() before use.
    context_pool writes arrive via update_context() Slot through signal_livecc_context
    (QueuedConnection) — no lock needed.
    """

    signal_broadcast = QtCore.Signal(float, float, object)
    signal_error = QtCore.Signal(str)

    WATERMARK_SEC = 1.0        # trigger next call when TTS remaining < this
    POLL_INTERVAL_MS = 200     # polling interval while watermark not reached
    INTER_SENTENCE_MS = 300    # silence injected between consecutive sentences

    def __init__(self, tts_worker_ref, parent=None) -> None:
        super().__init__(parent)
        self._tts_worker = tts_worker_ref
        self._stop_requested: bool = False
        self._abort_current: bool = False
        self._paused: bool = False
        self._initialized: bool = False
        self._context_pool: deque = deque(maxlen=3)

    @QtCore.Slot()
    def initialize(self) -> None:
        try:
            from ..core.models.gemini_broadcaster import _get_client
            _get_client()
            self._initialized = True
            logging.info("[GeminiBackgroundWorker] Ready")
        except Exception as e:
            logging.exception("[GeminiBackgroundWorker] Initialization failed")
            self.signal_error.emit(str(e))

    @QtCore.Slot()
    def run_background_loop(self) -> None:
        """Main continuous loop. Start via QueuedConnection after initialize()."""
        logging.info("[GeminiBackgroundWorker] Background loop started")
        while not self._stop_requested:
            if self._paused:
                QtCore.QThread.msleep(self.POLL_INTERVAL_MS)
                continue

            remaining = self._tts_worker.get_queue_remaining_sec()
            if remaining > self.WATERMARK_SEC:
                QtCore.QThread.msleep(self.POLL_INTERVAL_MS)
                continue

            self._abort_current = False
            context = self._build_context()
            t_now = time.time()

            try:
                from ..core.models.gemini_broadcaster import stream_gemini
                for ev in stream_gemini(context):
                    if self._abort_current or self._stop_requested:
                        logging.info("[GeminiBackgroundWorker] Stream aborted mid-way")
                        break
                    if ev.broadcast_text:
                        result = ev.to_dict()
                        result["_enqueue_ts"] = t_now
                        self.signal_broadcast.emit(t_now, t_now, result)
            except Exception as e:
                logging.exception("[GeminiBackgroundWorker] stream_gemini error")
                self.signal_error.emit(str(e))

            if not self._abort_current and not self._stop_requested:
                QtCore.QThread.msleep(self.INTER_SENTENCE_MS)

        logging.info("[GeminiBackgroundWorker] Background loop stopped")

    def _build_context(self) -> dict:
        from ..core.match_tracker import match_tracker
        recent_descriptions = list(self._context_pool)
        event_text = " ".join(recent_descriptions) if recent_descriptions else "Game in progress."
        return {
            "event": event_text,
            "match_state": match_tracker.get_state_string(),
        }

    @QtCore.Slot()
    def pause(self) -> None:
        self._abort_current = True
        self._paused = True
        logging.info("[GeminiBackgroundWorker] Paused by P1 interrupt")

    @QtCore.Slot()
    def resume(self) -> None:
        self._paused = False
        logging.info("[GeminiBackgroundWorker] Resumed after P1 silence")

    @QtCore.Slot(str)
    def update_context(self, description: str) -> None:
        """Receive LiveCC P3 description. Called via signal_livecc_context (QueuedConnection)."""
        self._context_pool.append(description)

    def flush_and_abort(self) -> None:
        """Direct call from GUI thread — GIL-safe bool assignment."""
        self._abort_current = True
        self._paused = True

    @QtCore.Slot()
    def requestStop(self) -> None:
        self._abort_current = True
        self._stop_requested = True
```

### 5.2 `OpenAITTSWorker` — Required Additions

**File**: `src/miis_broadcast/workers/openai_tts.py`

#### [FIX A] CJK-aware duration estimation

Chinese text has no word-boundary spaces. `len(text.split())` returns `1` for an entire Chinese sentence, causing the watermark to never trigger — Gemini fires uncontrolled API calls. Use character count instead:

```python
def _estimate_tts_duration(text: str, speed: float) -> float:
    """
    CJK-aware TTS duration estimate for backpressure watermark.
    Chinese/Japanese characters: ~4 chars/sec at 1.0x speed.
    English words: ~2.5 words/sec at 1.0x speed.
    """
    cjk_count = sum(
        1 for c in text
        if '一' <= c <= '鿿'   # CJK Unified Ideographs
        or '぀' <= c <= 'ヿ'   # Hiragana / Katakana
    )
    # Replace CJK chars with spaces, then split for English words
    ascii_only = ''.join(' ' if '一' <= c <= '鿿' else c for c in text)
    other_words = len(ascii_only.split())
    estimated = (cjk_count / 4.0 + other_words / 2.5) / max(speed, 0.1)
    return max(estimated, 0.5)  # floor at 0.5s to avoid instant re-fire
```

#### [FIX B] Interrupt guard — never emit `signal_tts_done` on forced interrupt

This is the critical fix for the P1 cascade race condition (see Section 11).

```python
# Add to __init__():
import threading
self._queue_remaining_lock = threading.Lock()
self._queue_remaining_sec: float = 0.0
self._interrupted: bool = False          # guards against spurious signal_tts_done

# New signal:
signal_tts_done = QtCore.Signal()        # emitted ONLY on natural completion

# New methods:
def get_queue_remaining_sec(self) -> float:
    """Thread-safe query for GeminiBackgroundWorker backpressure."""
    with self._queue_remaining_lock:
        return max(0.0, self._queue_remaining_sec)

def _add_queue_time(self, seconds: float) -> None:
    with self._queue_remaining_lock:
        self._queue_remaining_sec += seconds

def _subtract_queue_time(self, seconds: float) -> None:
    with self._queue_remaining_lock:
        self._queue_remaining_sec = max(0.0, self._queue_remaining_sec - seconds)

# In speak() slot:
# estimated = self._estimate_tts_duration(text, self._speed)
# self._add_queue_time(estimated)
# self._interrupted = False    # new speak always clears the flag

# In interrupt() slot:
# self._interrupted = True     # mark as interrupted: do NOT emit done
# ... (stop audio playback) ...

# In the internal segment-completion callback:
# self._subtract_queue_time(actual_elapsed_sec)
# if not self._interrupted:
#     self.signal_tts_done.emit()   # only on natural completion
# self._interrupted = False         # reset for next segment
```

---

## 6. Change 4: TTS Rhythm & Interrupt Silence (`gui.py`)

### 6.1 Updated `_route_segment()` — Fast-Slow Blade Routing

**[FIX] P3 path uses `signal_livecc_context` Signal** instead of `QMetaObject.invokeMethod`. This avoids magic string slot names and is more refactor-safe.

```python
# Declare in MainWindow class body (alongside other signals):
signal_livecc_context = QtCore.Signal(str)

@QtCore.Slot(float, float, object)
def _route_segment(self, start_t: float, stop_t: float, data: object) -> None:
    """Route LiveCC segments: fast blade direct-to-TTS or P3 to context pool."""
    self._ensure_log_dir()
    display, _ = self._extract_segment_texts(data)
    self._write_log(
        self.livecc_log_file,
        f"[LiveCC] [{self._fmt_time(start_t)}-{self._fmt_time(stop_t)}] {display}",
    )

    raw = ""
    if isinstance(data, dict):
        raw = data.get("metadata", {}).get("raw", "") or data.get("event", "")
    elif isinstance(data, str):
        raw = data

    fast_priority = self._scan_priority(raw)

    if fast_priority == 1:
        logging.info("[FastBlade] P1 hit: %r", raw[:80])
        if hasattr(self, "gemini_bg_worker"):
            self.gemini_bg_worker.pause()
        if self.tts_mode == "openai":
            self.signal_tts_interrupt.emit()
        elif self.tts_mode == "local":
            self.signal_local_tts_interrupt.emit()
        tts_text = raw.strip()
        if tts_text:
            self._post_p1_pending = True
            self.signal_tts_speak.emit(tts_text, 1, time.time(), start_t)
        self.signal_p1_confirmed.emit()

    elif fast_priority == 2:
        logging.info("[FastBlade] P2 hit: %r", raw[:80])
        tts_text = raw.strip()
        if tts_text:
            self.signal_tts_speak.emit(tts_text, 2, time.time(), start_t)

    else:
        # P3: update context pool via typed Qt Signal (QueuedConnection)
        # Avoids QMetaObject magic-string slot lookup; safe across refactors.
        description = raw.strip()
        if description:
            self.signal_livecc_context.emit(description)
```

### 6.2 Post-P1 Silence (1.0s) — `gui.py` Additions

```python
# In MainWindow.__init__():
self._post_p1_pending: bool = False

@QtCore.Slot()
def _on_tts_done(self) -> None:
    """
    Called when TTS naturally finishes one segment.
    NOT called when TTS is forcibly interrupted (OpenAITTSWorker._interrupted guard).
    Triggers P1 post-interrupt silence.
    """
    if self._post_p1_pending:
        self._post_p1_pending = False
        QtCore.QTimer.singleShot(1000, self._resume_gemini_background)
        logging.info("[P1 Silence] TTS done naturally, scheduling 1.0s before Gemini resumes")

def _resume_gemini_background(self) -> None:
    """Called 1.0s after P1 TTS finishes. Resumes Gemini slow blade."""
    if hasattr(self, "gemini_bg_worker") and self.is_inference_running:
        # Use QueuedConnection to safely call resume() in gemini_bg_thread
        QtCore.QMetaObject.invokeMethod(
            self.gemini_bg_worker, "resume",
            QtCore.Qt.QueuedConnection,
        )
        logging.info("[P1 Silence] 1.0s elapsed, Gemini background resumed")
```

### 6.3 Updated `_initGeminiWorker()` in `gui.py`

```python
def _initGeminiWorker(self) -> None:
    # --- Existing GeminiWorker (kept for file-mode / legacy) ---
    self.gemini_thread = QtCore.QThread(self)
    self.gemini_worker = GeminiWorker()
    self.gemini_worker.moveToThread(self.gemini_thread)
    self._signal_to_gemini.connect(self.gemini_worker.process_segment)
    self.gemini_worker.signal_priority.connect(self._on_gemini_priority)
    self.gemini_worker.signal_broadcast.connect(self.on_segment)
    self.gemini_worker.signal_error.connect(self.on_gemini_error)
    self.gemini_thread.started.connect(self.gemini_worker.initialize)
    self.gemini_thread.start()

    # --- New GeminiBackgroundWorker (fast-slow blade continuous background) ---
    from .workers.gemini import GeminiBackgroundWorker
    self.gemini_bg_thread = QtCore.QThread(self)
    self.gemini_bg_worker = GeminiBackgroundWorker(tts_worker_ref=self.tts_worker)
    self.gemini_bg_worker.moveToThread(self.gemini_bg_thread)
    self.gemini_bg_worker.signal_broadcast.connect(self.on_segment)
    self.gemini_bg_worker.signal_error.connect(self.on_gemini_error)

    # Connect signal_livecc_context (P3 path) → update_context slot
    # QueuedConnection: emission from GUI thread, slot runs in gemini_bg_thread
    self.signal_livecc_context.connect(
        self.gemini_bg_worker.update_context,
        QtCore.Qt.QueuedConnection,
    )

    self.gemini_bg_thread.started.connect(self.gemini_bg_worker.initialize)
    self.gemini_bg_thread.started.connect(
        lambda: QtCore.QMetaObject.invokeMethod(
            self.gemini_bg_worker, "run_background_loop",
            QtCore.Qt.QueuedConnection,
        )
    )
    self.gemini_bg_thread.start()

    # Connect TTS done signal (tts_worker must exist before this call)
    self.tts_worker.signal_tts_done.connect(self._on_tts_done)
```

### 6.4 `stop_inference()` — Add Gemini Background Stop

```python
# In existing stop_inference(), add alongside gemini_worker.flush_and_abort():
if hasattr(self, "gemini_bg_worker"):
    self.gemini_bg_worker.requestStop()
self._post_p1_pending = False
```

---

## 7. Complete Data Flow

```
[INPUT]
CameraThread(index=0) → signal_frame → lambda [DirectConn, CameraThread] → push_fp_frame(bgr, t) ─┐
CameraThread(index=1) → signal_frame → lambda [DirectConn, CameraThread] → push_tp_frame(bgr, t) ─┘
                                                                                                    ↓
                                      build_composite_clip()
                                      np.concatenate(axis=1) → ~1280×480, t_start=min(fp,tp)
                                                                                                    ↓
[FAST BLADE: LiveCCSplitCameraWorker — QThread]
  Qwen2VL-7B | cuda:0 | bfloat16 | flash_attn2
  window=1.5s | fps=2.0 | interval=1.0s | max_tokens=12 | fps_frames=3
                                                                                                    ↓
                                    signal_segment(start_t, stop_t, parsed_dict)
                                                                                                    ↓
[ROUTING: MainWindow._route_segment() — GUI thread]
  keyword scan → P1 / P2 / P3
       │
       ├─ P1: gemini_bg_worker.pause()           ← CPython GIL-safe bool flag
       │       signal_tts_interrupt              ← QueuedConnection → TTS thread
       │       signal_tts_speak(text, P1, ...)   ← QueuedConnection → TTS thread
       │       _post_p1_pending = True
       │       signal_p1_confirmed               ← QueuedConnection → KV cache reset
       │
       ├─ P2: signal_tts_speak(text, P2, ...)
       │       (Gemini background unaffected)
       │
       └─ P3: signal_livecc_context.emit(description)
               → update_context() [QueuedConnection → gemini_bg_thread]
               → context_pool.append(description)

[SLOW BLADE: GeminiBackgroundWorker — QThread, continuous loop]
  while not stopped:
    if _paused: sleep 200ms; continue
    remaining = tts_worker.get_queue_remaining_sec()   ← threading.Lock
    if remaining > 1.0s: sleep 200ms; continue         ← Backpressure gate
    context = {context_pool[-3:] + match_state}
    stream_gemini(context, text_only=True)              ← NO image upload
    → signal_broadcast(P3/P4) → on_segment() → signal_tts_speak
    msleep(300)                                         ← inter-sentence pause

[TTS COMPLETION — OpenAITTSWorker thread]
  Natural end: _subtract_queue_time(); if not _interrupted: signal_tts_done.emit()
  Interrupt:   _interrupted = True; do NOT emit signal_tts_done

[POST-P1 SILENCE — GUI thread]
  signal_tts_done → _on_tts_done()
  if _post_p1_pending:
    _post_p1_pending = False
    QTimer.singleShot(1000, _resume_gemini_background)
    → gemini_bg_worker.resume()   [QueuedConnection]

[OUTPUT: OpenAI Realtime TTS WebSocket]
  gpt-4o-realtime-preview | voice=coral | speed=1.5x
  P1: interrupt queue; P2: high-priority; P3/P4: backpressure-gated
```

---

## 8. Thread Safety Reference

| State | Owner | Access Pattern | Safety Mechanism |
|---|---|---|---|
| `_paused`, `_abort_current` | `GeminiBackgroundWorker` | Written from GUI thread (CPython GIL atomic bool); read in worker thread | CPython GIL |
| `_context_pool` | `GeminiBackgroundWorker` | Written via `update_context()` Slot through `signal_livecc_context` (QueuedConnection); read in worker thread via `list()` snapshot | QueuedConnection serializes writes; GIL-safe deque read |
| `_queue_remaining_sec` | `OpenAITTSWorker` | Written in TTS worker thread; read from Gemini background thread | `threading.Lock` |
| `_interrupted` | `OpenAITTSWorker` | Written in TTS thread (interrupt/speak slots); read in TTS thread (completion callback) | Single-thread access only — no lock needed |
| `_post_p1_pending` | `MainWindow` | Read and written only in GUI thread (Qt slots) | GUI thread only — no lock needed |

---

## 9. Known Issues, Analysis & Resolutions

| Issue | Verdict | Severity | Resolution |
|---|---|---|---|
| Lambda executes in GUI thread blocking rendering | **Hallucination** — DirectConnection runs lambda in emitter's (CameraThread) thread | None | Documented in Section 4.1 & 4.4; no code change |
| Chinese text causes backpressure failure (`split()`) | **Real** | High — entire mechanism broken for CJK | ✅ CJK-aware `_estimate_tts_duration()` in Section 5.2 |
| NumPy GC stop-the-world pauses inference | **Hallucination** — NumPy uses C-level `malloc`, not tracked by Python's cyclic GC | None | Documented in Section 4.2 comment; no code change |
| `QMetaObject.invokeMethod` string-based slot fragility | **Real (low risk)** — valid but not refactor-safe | Low | ✅ Replaced with `signal_livecc_context = QtCore.Signal(str)` in Section 6.1 & 6.3 |
| `signal_tts_done` fires on interrupt → P1 cascade race | **Real Race Condition** | Critical — Gemini resumes mid-P1-B speech | ✅ `_interrupted` flag in `OpenAITTSWorker`; `signal_tts_done` only on natural completion (Section 5.2) |

### Race Condition Detail — Consecutive P1 Events

Without the fix, this sequence causes Gemini to resume while P1-B is still speaking:

```
t=0.0  P1-A fires  → _post_p1_pending=True, TTS plays P1-A
t=0.8  P1-B fires  → signal_tts_interrupt → TTS cuts P1-A
         [BUG] TTS emits signal_tts_done on interrupt
         → _on_tts_done(): _post_p1_pending=True → QTimer(1000ms) starts
         → _post_p1_pending = False
t=0.8  P1-B text → TTS → _post_p1_pending = True
t=1.8  First QTimer fires → gemini_bg_worker.resume()  ← P1-B still playing!
t=2.5  P1-B ends naturally → signal_tts_done → another QTimer(1000ms) → double resume
```

With the `_interrupted` flag fix:
```
t=0.8  signal_tts_interrupt → _interrupted = True
         TTS stops audio; does NOT emit signal_tts_done  ← fix
t=0.8  P1-B text → new speak() → _interrupted = False, _post_p1_pending = True
t=2.5  P1-B ends naturally → not _interrupted → signal_tts_done.emit()
         → _on_tts_done() → QTimer(1000ms) → resume  ← correct timing
```

---

## 10. Files Changed Summary

| File | Type | Key Changes |
|---|---|---|
| `configs/models.yml` | Modify | `max_new_tokens: 12`, `streaming_fps_frames: 3`, `infer_interval: 1.0` |
| `configs/livecc_prompts.yml` | Modify | New `livecc_query` — split-screen spatial mapping + neutral transition phrases |
| `workers/livecc.py` | Add | `build_composite_clip()`, `LiveCCSplitCameraWorker` |
| `workers/gemini.py` | Add | `GeminiBackgroundWorker` (backpressure + pause/resume) |
| `workers/openai_tts.py` | Modify | `signal_tts_done` (with `_interrupted` guard), `get_queue_remaining_sec()` (CJK-aware), `_estimate_tts_duration()` |
| `gui.py` | Modify | `signal_livecc_context`, `_route_segment()`, `_initGeminiWorker()`, `_initCameraWorker()`, `_on_tts_done()`, `_resume_gemini_background()`, `_post_p1_pending` |

## 11. Unchanged Components

- `GeminiWorker` (kept for file-mode; `GeminiBackgroundWorker` is additive)
- `LiveCCWorker` (file-mode inference)
- `LiveCCCameraWorker` (single-camera mode; `LiveCCSplitCameraWorker` is additive)
- `_P1_KEYWORDS`, `_P2_KEYWORDS` sets
- `MatchTracker` singleton
- RAG context retriever (`rag_threshold=600`)
- All Gemini system prompts (objective / hype / calm / trash_talk)
- `ChatterboxTTSWorker` (local TTS)
- `signal_p1_confirmed` → KV cache reset chain
