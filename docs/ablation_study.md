# Ablation Study：LiveCC Prompt、時間視窗、KV Cache 與 Temperature

> **更新日期：2026-07-20。** 第 1–7 節保留早期 Phase A（60–180 秒、
> LiveCC→Gemini）的實驗與 F08 決策；第 8–13 節加入 Phase B 的 CLI-only
> 控制實驗。新證據顯示：若目標是 LiveCC 視覺描述的完整性與長時間穩定性，
> F06 優於 F08；F08 仍可作為降低 TTS 輸出密度的系統層候選。兩者衡量目標不同，
> 不應將早期 F08 結論直接解讀成視覺辨識品質最佳。

## 1. Phase A 研究問題

本實驗檢驗 LiveCC 每次推論使用的 frame 數，如何影響雙視角 VR 籃球播報的：

1. LiveCC 與 Gemini 的處理延遲；
2. LiveCC 原始描述與 Gemini 最終播報的事件對齊；
3. 句子完整度、重複程度與人物幻覺；
4. 適合即時 TTS 播放的播報密度。

本節是**時間視窗（temporal window）消融**。所有條件均使用相同的第一人稱／第三人稱拼接畫面，因此結果不能用來宣稱雙視角優於單一視角；該主張仍需另外執行 FPV-only、third-person-only 與 dual-view 對照實驗。

## 2. 實驗設計

### 2.1 測試資料

- 影片：`test_merged.mp4`
- 評估區間：影片第 60–180 秒，共 120 秒
- 畫面：1280×480 的雙視角拼接影片
- Ground Truth：`eval/groundTruth.json` 中與該區間重疊的 8 個事件
- 裁切 GT：`eval/results/frame_gt_validation/groundtruth_60_180.json`

使用同一段影片能控制事件難度與內容差異。GT 並非逐幀且不保證完全正確，因此只作為弱監督參考，不作為唯一決策依據。

### 2.2 控制變因

四組實驗使用相同的：

- LiveCC-7B-Instruct 模型；
- 雙視角籃球 prompt；
- Gemini prompt 與模型設定；
- 4 FPS 取樣率；
- `max_new_tokens: 42`；
- `temperature: 0.2`、`top_p: 0.8`、`top_k: 20`；
- `repetition_penalty: 1.12`、`no_repeat_ngram_size: 3`；
- 每個離線 segment 後清除 KV continuation，避免上一個未完成句延續到下一段。

唯一主要自變因是每次推論使用的 frame 數；推論週期隨視窗長度調整：

| 條件 | Frames | 時間視窗 | 推論週期 |
|---|---:|---:|---:|
| F06 | 6 | 1.5 秒 | 1.0 秒 |
| F08 | 8 | 2.0 秒 | 1.0 秒 |
| F12 | 12 | 3.0 秒 | 1.5 秒 |
| F16 | 16 | 4.0 秒 | 2.0 秒 |

### 2.3 評估方法

每組先執行 LiveCC，再將其結果送入 Gemini，避免 Gemini 網路等待改變影片取樣時序。評估包含：

- LiveCC 與 Gemini 推論時間；
- 從事件進入視窗到 Gemini 完成的估算反應時間；
- BC-Align 事件語意分數；
- 句子片段、啟發式離題與隊友幻覺；
- `should_speak` 數量、唯一輸出數與最高重複次數。

估算反應時間定義為：

```text
推論週期的一半 + LiveCC 平均時間 + Gemini 平均時間
```

此數值不包含 TTS 首聲、音訊排隊、GUI 排程及播放中斷時間，不能當作完整端到端延遲。

## 3. 結果

### 3.1 延遲與輸出數量

| 指標 | F06 | F08 | F12 | F16 |
|---|---:|---:|---:|---:|
| LiveCC segments | 80 | 60 | 41 | 29 |
| 被過濾 segments | 0 | 1 | 0 | 1 |
| LiveCC mean | 0.628 s | 0.687 s | 0.924 s | 1.115 s |
| LiveCC median | 0.522 s | 0.617 s | 0.830 s | 1.046 s |
| LiveCC P95 | 1.198 s | 1.151 s | 1.573 s | 1.725 s |
| LiveCC maximum | 3.128 s | 3.141 s | 2.679 s | 2.890 s |
| Gemini mean | 0.530 s | 0.533 s | 0.552 s | 0.542 s |
| Gemini P95 | 0.629 s | 0.657 s | 0.716 s | 0.721 s |
| LiveCC + Gemini mean | 1.158 s | 1.221 s | 1.476 s | 1.656 s |
| 估算反應時間 | 1.66 s | 1.72 s | 2.23 s | 2.66 s |

F06 與 F08 的模型鏈延遲只差 0.063 秒，估算反應時間只差約 0.06 秒；F08 的輸出數卻比 F06 少 25%，較不容易形成 TTS 積壓。F12 與 F16 提供較長上下文，但反應時間分別增加到 2.23 與 2.66 秒。

### 3.2 呈現品質代理指標

| 指標 | F06 | F08 | F12 | F16 |
|---|---:|---:|---:|---:|
| 句子片段 | 1 | 4 | 1 | 2 |
| 啟發式離題 | 9 | 9 | 1 | 2 |
| 隊友幻覺 | 0 | 0 | 3 | 1 |
| Gemini `should_speak` | 25 | 22 | 14 | 12 |
| Gemini 唯一輸出 | 40 | 36 | 25 | 21 |
| 單一輸出最高重複次數 | 23 | 16 | 8 | 8 |

F06 的即時性最好，但 segment 密度與最高重複次數最大，較接近逐動作描述而非連貫播報。F08 保留快速反應，同時降低輸出量與重複上限。F12 與 F16 的離題較少，但較長上下文沒有消除幻覺，反而出現將場上另一人物描述成隊友的錯誤。

### 3.3 Ground Truth／BC-Align

四組條件的 LiveCC 與 Gemini 都覆蓋 8/8 個 GT 時間區間。GPT-4o judge 使用固定 temperature 0，共完成 64 次事件評估。

| 條件 | LiveCC | Gemini | Gemini − LiveCC |
|---|---:|---:|---:|
| F06 | 2.250 | 2.250 | 0.000 |
| F08 | 1.875 | 2.125 | **+0.250** |
| F12 | 2.250 | 1.875 | **−0.375** |
| F16 | 2.500 | 2.250 | **−0.250** |

逐事件分數如下：

```text
F06 LiveCC  [4, 2, 2, 2, 2, 2, 2, 2]
F06 Gemini  [4, 2, 2, 2, 2, 2, 2, 2]

F08 LiveCC  [4, 2, 2, 2, 1, 2, 1, 1]
F08 Gemini  [3, 2, 2, 2, 2, 2, 2, 2]

F12 LiveCC  [4, 2, 2, 2, 2, 2, 2, 2]
F12 Gemini  [2, 2, 2, 2, 2, 2, 2, 1]

F16 LiveCC  [4, 2, 4, 2, 2, 2, 2, 2]
F16 Gemini  [5, 2, 2, 2, 2, 2, 2, 1]
```

F16 的 LiveCC 分數最高，但 Gemini 改寫後下降 0.25。F12 也下降 0.375。這顯示增加視窗長度雖可能改善 LiveCC 的局部事件理解，卻也讓 Gemini 更容易把多個模糊動作組合成過度確定的敘事。F08 是唯一在本次 GT 參考下由 Gemini 產生正向變化的條件。

## 4. Phase A 決策（已由 Phase B 補充）

Phase A 當時的正式播報候選為 **F08：8 frames、4 FPS、2 秒視窗、每 1 秒推論一次**。

決策不是依單一最高分，而是基於以下聯合證據：

1. F08 與最低延遲的 F06 只差約 0.06 秒；
2. F08 比 F06 少 25% LiveCC segments，可降低 TTS 排隊與過時播報風險；
3. F08 沒有觀察到隊友幻覺；
4. F08 是唯一 Gemini 相對 LiveCC 得分提高的條件；
5. F12 與 F16 的較長視窗增加延遲，也增加 Gemini 過度擴寫的風險；
6. 兩秒內容較能涵蓋「持球—移動—出手」的短事件，又不易混合多個連續事件。

正式設定位於 `configs/models.yml`：

```yaml
fps: 4.0
initial_fps_frames: 8
streaming_fps_frames: 8
camera:
  window_sec: 2.0
  target_fps: 4.0
  infer_interval: 1.0
  memory_reset_every: 1
```

F06 保留為低延遲模式；F16 可用於非即時離線分析。F12 不建議作為目前正式播報設定。

此決策包含 Gemini 與輸出密度的系統層考量。後續 Phase B 的 LiveCC-only 長測顯示，
F06 在句子完整度、人物幻覺與長時間穩定性較佳，因此第 13 節給出更新後的分層決策。

## 5. 有效性限制

### 5.1 Ground Truth 限制

目前 GT 不是逐幀標註，可能包含漏事件、時間邊界誤差或事件描述不完整。BC-Align 偏向檢查播報是否提到相同事件，無法完整衡量：

- 第一人稱與第三人稱的視角歸因是否正確；
- 投籃嘗試、命中與未進之間的差異；
- Gemini 是否把不確定描述改為肯定結果；
- 播報密度、重複程度與實際聽感；
- 錯誤的球權、人物或隊伍歸因。

因此 GT 分數是弱證據，必須與錯誤分析及延遲共同解讀。

### 5.2 評估範圍

- 主要消融只使用一支影片中的兩分鐘片段，不能代表所有場景。
- 品質欄位包含啟發式統計，尚未以多人盲測驗證。
- Gemini 是文字改寫器，沒有直接觀看畫面，不能獨立驗證 LiveCC 的視覺判斷。
- 本次 CLI 跳過 TTS，因此尚未測得觀眾實際聽到第一個音訊的端到端延遲。
- 完整 360.6 秒影片曾以 F12 執行，但尚未以正式 F08 設定重新完成 LiveCC → Gemini → TTS 全鏈測試。

## 6. Phase A 當時規劃的後續實驗

為補足目前結論，後續至少需要：

1. 以 F08 重跑完整影片並量測 LiveCC → Gemini → TTS 首聲時間（Phase B 已先以
   F06 完成 LiveCC-only 長測，完整含 TTS 的 F08 系統測試仍未執行）；
2. 記錄 TTS 佇列、取消、硬中斷與實際播放完成時間；
3. 執行 FPV-only、third-person-only、dual-view 三組視角消融；
4. 加入人工標註的視角、球權、出手與結果欄位；
5. 由多位評估者對準確度、自然度、即時性與重複度進行盲測；
6. 將 Gemini 新增事實率與 LiveCC → Gemini contradiction rate 列為獨立指標。

## 7. 可重現產物

兩分鐘 pipeline 輸出：

- `eval/results/frames_06_2min_livecc_gemini.jsonl`
- `eval/results/frames_08_2min_livecc_gemini.jsonl`
- `eval/results/frames_12_2min_livecc_gemini.jsonl`
- `eval/results/frames_16_2min_livecc_gemini.jsonl`

完整 BC-Align judge 結果：

- `eval/results/frame_gt_validation/frames_06_{livecc,gemini}_bc.json`
- `eval/results/frame_gt_validation/frames_08_{livecc,gemini}_bc.json`
- `eval/results/frame_gt_validation/frames_12_{livecc,gemini}_bc.json`
- `eval/results/frame_gt_validation/frames_16_{livecc,gemini}_bc.json`

對應實驗設定：

- `eval/tuning/models_frames_06.yml`
- `eval/tuning/models_frames_08.yml`
- `eval/tuning/models_frames_12.yml`
- `eval/tuning/models_frames_16.yml`

## 8. Phase B 實驗範圍

Phase B 使用 `eval/run_commentary_pipeline.py` 的 headless CLI，關閉 Gemini 與 TTS，
將評估焦點收斂到 LiveCC 視覺層。實驗環境如下：

- GPU：NVIDIA RTX 4090 24 GiB；
- PyTorch：2.8.0+cu129；Transformers：4.57.1；
- 模型：`chenjoya/LiveCC-7B-Instruct`；
- 影片：`examples/test_merged.mp4`，1280×480、30 FPS、360.6 秒；
- 畫面：外部人物與第一人稱遊戲的左右雙視角；
- 固定生成參數：`top_p=0.8`、`top_k=20`、`repetition_penalty=1.12`、
  `no_repeat_ngram_size=3`、`max_new_tokens=42`；
- 除 KV 消融外，每一 segment 後清除 KV continuation。

Phase B 回答四個問題：

1. Natural sentence 或 tagged event contract 哪一種較穩定？
2. 6／8／12／16 frames 哪一種較不易跨動作拼接？
3. 是否應保留跨 segment KV cache？
4. Temperature 如何影響事件召回、重複與幻覺？

## 9. Prompt 與輸出契約消融

固定 F12，比較 natural prompt、tagged prompt，以及 tagged prompt 加 `[` response
prefix。這一階段是生成契約 smoke test，不作為事件準確度的正式主表。

| 條件 | 有效／總輸出 | 有效率 | 平均延遲 | 主要現象 |
|---|---:|---:|---:|---|
| Natural prompt | 3/4 | 75.0% | 2.428 s | 可產生完整自然句 |
| Tagged prompt | 1/21 | 4.8% | 0.876 s | 19 次裸 `made`、1 次裸 `dunk` |
| Tagged + `[` prefix | 0/21 | 0% | 0.895 s | 21 次塌縮成 `[made]` |

結論是 LiveCC-7B 不適合被強迫輸出 event tag；response prefix 會進一步放大塌縮。
後續一律使用 natural prompt 與空 prefix，事件 label 交由 deterministic parser 產生。

## 10. Frame 數與六分鐘長測

### 10.1 Frame 數 smoke test

| Frames | 有效率 | 平均延遲 | 觀察 |
|---:|---:|---:|---|
| 6 | 100% | 2.380 s | 三句完整，未出現明顯幻覺 |
| 8 | 100% | 2.475 s | 出現 pass 與截斷句 |
| 12 | 75% | 2.428 s | 出現 teammate 幻覺 |
| 16 | 100% | 2.374 s | 虛構 teammate、pass、layup |

較長窗口會把相鄰動作合併成不存在的 play；F16 的 teammate 與 layup 尤其違反
一對一比賽設定。這一小樣本支持 F06，但仍需以下方長測確認。

### 10.2 F06 六分鐘長測

Natural prompt、F06、temperature 0.2、KV reset=1 跑完整 360.6 秒影片：

| 指標 | 結果 |
|---|---:|
| 有效輸出 | 237 |
| 覆蓋時間 | 360.5 s |
| 平均／中位延遲 | 0.560／0.538 s |
| P95／最大延遲 | 0.615／3.061 s |
| 不同句子 | 59/237（24.9%） |
| 重複實例／連續完全重複 | 178／29 |
| 設備或畫面描述 | 15 |
| teammate／pass 宣稱 | 2／3 |
| 第一或第二人稱 | 0 |

F06 可穩定處理完整影片，且延遲沒有隨時間明顯惡化；主要問題從生成崩潰轉為
高重複率與少量 off-topic 描述。現有 degenerate filter 尚未攔截 headset、
controller、screen、camera 與一對一場景中的 teammate/pass。

## 11. KV-cache 消融

固定 natural prompt、F06、temperature 0.2，比較共同前 40 秒：

| 指標 | Reset=1 | Reset=0 |
|---|---:|---:|
| 有效輸出率 | 26/26（100%） | 22/26（84.6%） |
| 被拒絕 | 0 | 4 |
| 截斷／省略號 | 0 | 13 |
| 句子碎片 | 0 | 8 |
| Team/pass 幻覺 | 0 | 3 |
| 多句輸出 | 0 | 7 |
| 不同句子比例 | 46.2% | 90.9% |
| 平均延遲 | 0.857 s | 0.810 s |

Reset=0 的高多樣性主要來自跨窗口續寫，例如 `Player 2 passes to ...`、
`He catches the ball at the top ...`、`of his key. He ...`，不是品質改善。
因此正式設定應保留 `kv_reset_every_segments: 1`。

## 12. Temperature 消融與 GT 驗證

### 12.1 重要窗口選擇

使用 ground truth 搜尋事件密集的連續兩分鐘，選中原影片 80–200 秒。該窗口包含
22 個 GT 子片段：3 次 score、6 次 steal、5 次 out-of-bounds、6 次 shot、
3 次 rebound，並有 11 個包含 dribble 的子片段。三組均使用 F06、natural prompt、
KV reset=1，唯一變因是 temperature 0.1／0.2／0.5。

### 12.2 品質、重複與延遲

| 指標 | T=0.1 | T=0.2 | T=0.5 |
|---|---:|---:|---:|
| 有效輸出 | 81 | 80 | 81 |
| 不同句子比例 | 33.3% | 43.8% | 80.2% |
| 重複實例 | 54 | 45 | 16 |
| 連續重複 | 9 | 10 | 1 |
| 平均延遲 | 0.618 s | 0.634 s | 0.629 s |
| P95 延遲 | 1.236 s | 1.213 s | 1.339 s |
| 多句輸出 | 0 | 1 | 6 |
| 截斷句 | 0 | 0 | 8 |
| 設備描述 | 4 | 5 | 2 |
| Team/pass 幻覺 | 4 | 2 | 8 |

### 12.3 關鍵字與時間重疊事件 Recall

| 事件（GT 數） | T=0.1 | T=0.2 | T=0.5 |
|---|---:|---:|---:|
| Dribble（11） | 100% | 100% | 90.9% |
| Shot（6） | 100% | 83.3% | 83.3% |
| Score（3） | 0% | 0% | 66.7% |
| Steal（6） | 0% | 0% | 0% |
| Out-of-bounds（5） | 20% | 20% | 40% |
| Rebound（3） | 0% | 0% | 0% |

T=0.5 捕捉到兩個與 GT 重疊的 score，但同時出現 `passes to an unseen teammate`、
`home team is down by one point` 與多個未完成句。T=0.1 最保守且完整，但高度
重複並漏掉終局事件。T=0.2 是目前穩定性與多樣性的折衷；temperature 對三組延遲
幾乎沒有影響。

此處 precision/recall 是以事件關鍵字與時間區間重疊自動計算，屬弱監督代理指標，
不是人工逐幀裁決。尤其 `shot`、`score` 與 `out-of-bounds` 仍需人工複核視覺證據。

## 13. 更新後決策與限制

### 13.1 推薦設定

目前 LiveCC 視覺層的穩定 baseline 為：

```yaml
fps: 4.0
initial_fps_frames: 6
streaming_fps_frames: 6
max_new_tokens: 42
kv_reset_every_segments: 1
generation:
  temperature: 0.2
  top_p: 0.8
  top_k: 20
  repetition_penalty: 1.12
  no_repeat_ngram_size: 3
```

並搭配 natural dual-view prompt 與空 response prefix。若系統層更重視降低 TTS
密度，可另以 F08 作候選，但不能直接視為 LiveCC 準確度更高。

### 13.2 尚未解決

1. Steal 與 rebound 在 temperature 三組的召回皆為 0；
2. T=0.2 對 score 的召回為 0，T=0.5 雖改善但增加幻覺；
3. F06 長測重複率偏高，需要 semantic dedup，而非保留 KV；
4. off-topic 與 teammate/pass 規則尚未完整進入 filter；
5. 尚未完成 FPV-only、third-person-only、dual-view 的核心視角消融；
6. 尚未以多人盲測或逐幀人工標註驗證自動事件分數；
7. Phase A 與 Phase B 的任務層不同（含 Gemini vs. LiveCC-only），不能把數值直接
   合併成單一排名。

### 13.3 Phase B 可重現產物

- `eval/tuning/dual_view_natural_query.txt`
- `eval/tuning/models_frames_{06,08,12,16}.yml`
- `eval/tuning/models_frames_06_no_kv_reset.yml`
- `eval/tuning/models_frames_06_temp_{01,05}.yml`
- `eval/results/long_test_merged_frames06_natural.jsonl`
- `eval/results/long_test_merged_frames06_natural_no_kv_reset.jsonl`
- `eval/results/ablation_temp{01,02,05}_window080_200.jsonl`

以上 JSONL 與裁切影片屬本機 evaluation data，預設不應提交 Git；文件只記錄聚合
結果與可重現的 CLI 條件。
