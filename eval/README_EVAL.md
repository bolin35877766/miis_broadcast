# 播報 Pipeline 評估（BC-Align & GVEval-Align）

本目錄提供**兩套獨立**的評估腳本，評估整條播報 pipeline（LiveCC → Gemini →
TTS 播出，記錄於 `log/combination_output.log`）是否與人工標註 Ground Truth
（`eval/groundTruth.json`）一致。

| 腳本 | 主指標 | 評分協議 |
|------|--------|----------|
| `run_bc_align.py` | **BC-Align Score**（1–5） | LiveCC 風格語意適配 judge |
| `run_gveval_align.py` | **GVEval-Align Score**（0–100） | [G-VEval](https://github.com/ztangaj/gveval)（AAAI 2025）ACCR judge |

兩者**共用 Stage 1（時間對齊）**，**Stage 2（打分）完全不同**——對同一份 log、
同一次 run，`pred_text` 與 coverage 會相同，但分數與解讀維度不同。

---

## 0. 共用架構：Stage 1 時間對齊

```
combination_output.log                groundTruth.json
        │                                     │
        ▼                                     │
  1. 解析成 [start, end, text] 片段              │
        │                                     │
        ▼                                     │
  2. 依時間軸偵測、切分成獨立的「一次完整播放」（run）  │
        │                                     │
        ▼                                     ▼
  3. 用時間區間重疊，把每個片段歸入重疊最大的 GT event
        │
        ▼
  4. 同一 event 內的片段依時間排序、串接成 Pred_e
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
  run_bc_align.py    run_gveval_align.py   （Stage 2 分歧）
  1–5 judge          G-VEval ACCR judge
```

`run_gveval_align.py` 直接 `import` `run_bc_align.py` 的對齊函式，確保兩邊
**永遠不會在對齊邏輯上分岔**。

### Run 偵測

`log/combination_output.log` 會累積所有歷史播報。Ground Truth 的 `[begin, end]`
只在**單次連續播放**內有意義。

判斷「新的一次播放」：看**影片時間軸是否倒退**——若某片段的開始時間比目前
這次播放已出現過的最大開始時間**倒退超過 `--reset-gap` 秒**（預設 2 秒），
就判定為新的一次播放（**不是**看 wall-clock 間隔）。

```bash
python eval/run_bc_align.py --list-runs
# 或
python eval/run_gveval_align.py --list-runs
```

### Overlap 指派（不是 t-IoU 門檻）

片段 `[t_s, t_e]` 與 event `[begin, end]` 的重疊長度：

```
overlap = max(0, min(t_e, end) - max(t_s, begin))
```

- 候選條件：`overlap > 0`
- 指派規則：**overlap 最大者勝**；同分取較小 `event_id`
- 與任何 event 都無重疊的片段會被捨棄

為何不用 t-IoU ≥ 0.5 當門檻：本專案播報句平均約 4 秒、GT event 平均約 16 秒，
正確配對時 t-IoU 常只有 0.13–0.4；用文獻常見 0.5 門檻會把幾乎所有句子判成
unassigned。實測上 argmax-overlap 與 argmax-t-IoU 在目前 33 句 log 上指派
結果完全相同。

### 同 event 聚合

同一 `event_id` 下所有片段依開始時間排序後串接成 `Pred_e`。若完全沒有片段被
指派，`Pred_e` 為空字串。

---

## 1. BC-Align（`run_bc_align.py`）

### 1.1 方法定位

改編自 LiveCC（Chen et al., CVPR 2025）
[`evaluation/livesports3kcc/llm_judge.py`](https://github.com/showlab/livecc/blob/main/evaluation/livesports3kcc/llm_judge.py)
的 LLM-as-judge 精神：

| | LiveCC 原版 | BC-Align |
|--|-------------|----------|
| 評估任務 | Model vs. GPT-4o baseline **成對比較** | **單一 pipeline** vs. Ground Truth |
| 時間對齊 | 推論前依 `[begin, end]` 裁切 | 推論後 overlap 對齊（見 §0） |
| Judge 準則 | Semantic Alignment + Stylistic Consistency | **動作適配**（主客隊不扣；結果矛盾仍扣） |
| 指標 | Win Rate | **BC-Align Score**（1–5 平均） |

### 1.2 Stage 2：Judge 協議

- 模型：OpenAI **GPT-4o**，`temperature=0`、`seed=42`
- 輸出：JSON `{"score": 1-5, "reason": "一句話"}`
- 評分焦點：**籃球動作與結果**是否對上 GT；不因風格、語氣扣分
- **主客隊放寬**：動作對上即可，不因我方/對手歸屬錯誤扣分
- **結果矛盾仍扣**：例如 GT 沒進、播報說進球
- **空 Pred**：不呼叫 API，直接給 **1 分**（1–5 尺度下限）

### 1.3 主指標

```
BC-Align Score = mean(score_e for e in all 22 events)
```

未覆蓋 event 計 1 分，因此單一分數已同時反映涵蓋率與正確性。
`summary.coverage_rate` 僅供診斷。

### 1.4 使用方式

```bash
python eval/run_bc_align.py --list-runs
python eval/run_bc_align.py --dry-run
python eval/run_bc_align.py
python eval/run_bc_align.py --run-index 0
```

| 參數 | 說明 | 預設 |
|------|------|------|
| `--log` | `combination_output.log` 路徑 | `log/combination_output.log` |
| `--groundtruth` | `groundTruth.json` 路徑 | `eval/groundTruth.json` |
| `--run-index` | 選第幾次播放（`auto` 或整數） | `auto` |
| `--reset-gap` | 判定新播放的時間倒退門檻（秒） | `2.0` |
| `--model` | Judge 模型 | `gpt-4o` |
| `--temperature` | Judge 溫度 | `0.0` |
| `--output` | 結果 JSON | `eval/results/bc_align_<timestamp>.json` |

### 1.5 輸出格式（摘要）

```json
{
  "judge_model": "gpt-4o",
  "summary": {
    "bc_align_score": 2.0,
    "num_events": 22,
    "num_covered": 21,
    "coverage_rate": 0.955,
    "num_judge_errors": 0
  },
  "events": [
    {
      "event_id": 1,
      "covered": false,
      "score": 1,
      "reason": "No commentary overlapped this event window.",
      "pred_text": "",
      "gt_asr_text": "..."
    }
  ]
}
```

---

## 2. GVEval-Align（`run_gveval_align.py`）

### 2.1 方法定位

在共用 Stage 1 之後，改用
[G-VEval](https://github.com/ztangaj/gveval)（Tang et al., AAAI 2025）
的 **ACCR 四維度 judge** 取代 BC-Align 的自訂 1–5 rubric。

| | BC-Align | GVEval-Align |
|--|----------|--------------|
| 方法來源 | LiveCC semantic alignment | G-VEval ACCR（MSVD-Eval 驗證模式） |
| 分數尺度 | 1–5 單一值 | 0–100，四維再平均 |
| 評分維度 | 動作/結果適配 | Accuracy + Completeness + Conciseness + Relevance |
| 推理方式 | 短 prompt → JSON | **CoT**（Evaluation Steps → 長篇 reason → 分數） |
| 取分機制 | 直接 parse JSON | **logprob 期望值**（top-5 機率加權） |
| temperature | `0` + `seed=42` | `1`（G-VEval 期望值技巧需要） |
| 空 Pred | 1/5 | 0/100 |

### 2.2 Stage 2：G-VEval Judge 協議

忠實重現官方 `evaluation/gveval/scorer.py` 與
`prompts/vid/accr/ref-only.txt` 的核心機制：

1. **ACCR 四維度**：每維 0–100，`final_score = (Acc + Comp + Conc + Rel) / 4`
2. **CoT**：prompt 內 `Evaluation Steps` 引導模型先寫詳細 reason，結尾用
   希臘字母包分數（α/β/ψ/δ）
3. **期望值打分**：`logprobs=True, top_logprobs=5`，對標記後的分數 token
   做機率加權平均（不是直接讀模型吐出的整數）
4. **取樣參數**：`temperature=1, top_p=1`（與原版一致）
5. **無影片幀**：本專案未 per-event 抽幀，走原版無影像時的 text-only 路徑
   （`system` message，無 `image_url`）

#### CoT 是什麼？

G-VEval 的 CoT **不是** o1 那種內建推理模式，而是 **prompt 驅動的逐步評分**：
模型依 Evaluation Steps 先分析、再輸出分數。reason 存於 JSON 的 `reason`
欄位供人閱讀；**數學分數來自 logprob 期望值**，不 parse reason 文字。

#### 領域適配（相對原版的刻意偏離）

- **主客隊放寬**：與 BC-Align 相同，不因我方/對手歸屬錯誤扣分（1v1 練球場景）
- **純 reference 評分**：prompt 對照 `gt_asr_text`，未餵影片幀（原版 MSVD 常搭配
  影片幀或 combined 模式）
- **模型預設**：`gpt-4o`（可用 `--model gpt-4o-2024-05-13` 還原論文 snapshot）

#### 四維度定義（本專案版本）

| 維度 | 評什麼 |
|------|--------|
| **Accuracy** | 動作與結果是否正確、無幻覺、無結果矛盾（主客隊歸屬不扣） |
| **Completeness** | 是否涵蓋 reference 中重要籃球動作（dribble, steal, miss…） |
| **Conciseness** | 是否簡潔、不冗贅 |
| **Relevance** | 是否與此籃球片段相關、無離題內容 |

**空 Pred**：不呼叫 API，四維與 `final_score` 皆為 **0**。

### 2.3 主指標

```
GVEval-Align Score = mean(final_score_e for e in all 22 events)
```

其中 `final_score_e = (Acc + Comp + Conc + Rel) / 4`（0–100）。
`summary` 另報四維 macro-average 供診斷。

### 2.4 使用方式

```bash
python eval/run_gveval_align.py --list-runs
python eval/run_gveval_align.py --dry-run
python eval/run_gveval_align.py
python eval/run_gveval_align.py --run-index 0
```

建議用專案 venv 執行（需 `openai`、`python-dotenv`）：

```bash
.venv\Scripts\python.exe eval/run_gveval_align.py
```

| 參數 | 說明 | 預設 |
|------|------|------|
| `--log` | `combination_output.log` 路徑 | `log/combination_output.log` |
| `--groundtruth` | `groundTruth.json` 路徑 | `eval/groundTruth.json` |
| `--run-index` | 選第幾次播放（`auto` 或整數） | `auto` |
| `--reset-gap` | 判定新播放的時間倒退門檻（秒） | `2.0` |
| `--model` | Judge 模型 | `gpt-4o` |
| `--temperature` | 取樣溫度（G-VEval 預設 1） | `1.0` |
| `--top-p` | 取樣 top_p | `1.0` |
| `--top-logprobs` | logprob 候選數 | `5` |
| `--output` | 結果 JSON | `eval/results/gveval_align_<timestamp>.json` |

### 2.5 輸出格式（摘要）

```json
{
  "judge_model": "gpt-4o",
  "summary": {
    "gveval_align_score": 19.57,
    "avg_accuracy": 10.96,
    "avg_completeness": 10.36,
    "avg_conciseness": 41.13,
    "avg_relevance": 15.82,
    "num_events": 22,
    "num_covered": 21,
    "coverage_rate": 0.955,
    "num_judge_errors": 0
  },
  "events": [
    {
      "event_id": 2,
      "covered": true,
      "final_score": 23.82,
      "accuracy": 15.42,
      "completeness": 10.58,
      "conciseness": 39.71,
      "relevance": 29.56,
      "extraction_mode": "expected_value",
      "reason": "...(CoT 長篇分析)... The Accuracy score is α20α. ...",
      "pred_text": "...",
      "gt_asr_text": "..."
    }
  ]
}
```

`extraction_mode` 為 `expected_value`（正常）或 `regex_fallback`（logprob
提取失敗時的備援）。

### 2.6 與 BC-Align 的實測差異（同一 log）

以 `combination_output.log` run #0 為例（對齊結果相同：21/22 covered）：

| 指標 | BC-Align | GVEval-Align |
|------|----------|--------------|
| 總分 | 2.0（1–5） | 19.57（0–100） |
| 解讀 | 動作/結果多數對不上 | Acc/Comp 很低；**Conc 偏高**（播報簡潔但內容常錯） |

兩者回答不同問題：

- **BC-Align**：「播報的籃球動作與結果有沒有對上 GT？」
- **GVEval-Align**：「播報作為 caption，在 ACCR 四維上多像 GT？」

---

## 3. 安裝與環境

腳本沿用專案既有依賴（`requirements.txt` 已包含）：

```
openai>=2.0,<3
python-dotenv>=1.0
```

在專案根目錄 `.env` 設定：

```
OPENAI_API_KEY=sk-...
```

`--dry-run` 不需 API key，只驗證 Stage 1 對齊。

---

## 4. 已知限制

### 共用（Stage 1）

1. **時間戳非影格級精準**：`combination_output.log` 為 TTS 播放時刻加估算語音
   長度，event 邊界附近可能有數秒誤差。
2. **Event 級比對**：目前只用 `gt_asr_text`，未用子句級 `gt_asr` 做更細對齊。
3. **Overlap tie-break**：兩 event 重疊完全相等時一律歸較小 `event_id`。

### BC-Align 專屬

4. **Judge 非完全確定性**：即使 `temperature=0`，輸出仍可能有微小波動。

### GVEval-Align 專屬

5. **無影片幀**：未實作 G-VEval combined 模式（reference + 影片幀），Accuracy
   完全依 reference 文字，無法驗證「文字對但看錯畫面」。
6. **期望值依賴模型/tokenizer**：`temperature=1` + logprob 取分，分數為連續值
   （如 15.42），跨模型/版本比較時應固定 `--model`。
7. **ACCR rubric 有領域適配**：相對原版 G-VEval 加了主客隊放寬、拿掉影片相關
   步驟；寫論文時應明確說明為 reference-only sports commentary 變體。

---

## 5. 檔案

| 檔案 | 用途 |
|------|------|
| `run_bc_align.py` | BC-Align 評估（Stage 1 + 1–5 judge） |
| `run_gveval_align.py` | GVEval-Align 評估（Stage 1 共用 + G-VEval ACCR judge） |
| `groundTruth.json` | 22 個 event 的人工標註 Ground Truth |
| `results/bc_align_*.json` | BC-Align 執行結果 |
| `results/gveval_align_*.json` | GVEval-Align 執行結果 |
| `README_EVAL.md` | 本文件 |
