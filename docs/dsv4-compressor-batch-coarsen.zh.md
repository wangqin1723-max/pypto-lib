# DSv4 Decode Compressor 逐 batch 循环 coarsening 调优

> 目标 kernel：`models/deepseek/v4/decode_indexer_compressor.py`（`indexer_compressor`，被 `decode_indexer.py` 调用）
> 平台：Ascend a2a3（910B/C 真机）。配置 B=64, S=2, D=4096, HEAD_DIM=128, OUT_DIM=256, POST_CHUNK=16。
> 结果：full-indexer 泳道图「中间碎任务」区 **807→3515us（~2700us）塌缩到 825→1042us（~217us）**，
> AICore span **3783→1303us（−65.5%）**，Total Test Time **1312us**，精度全 PASS（idx_kv_cache / score / topk_idxs）。

---

## 1. 诊断：泳道图「中间一大片碎任务」

跑完整 `decode_indexer.py --enable-l2-swimlane`，时间轴中段（compressor 区）是一大片
1–4us 的细任务，task 数高达 64/128/256，`Exec%` 极低——典型的**调度开销受限**
（每个 task 的 Head/Tail OH ~5–10us 淹没了实际算力）。

这一段对应 `indexer_compressor` 的逐 batch 循环：

```python
for c_idx in pl.parallel(0, B, BATCH_CHUNK_1):   # BATCH_CHUNK_1 = 1
    ...  # state_scatter / softmax_pool / state_shift / rmsnorm_rope_slice
         # / rope_fused / kv_hadamard / kv_and_cache_write —— 全是逐 batch 的小 scope
```

`BATCH_CHUNK_1=1` ⇒ B=64 个 batch 各自一迭代，每迭代里再切出一堆小 scope。

### 改动前泳道图（原始粘贴）

```
（在此粘贴改动前的泳道图 / merged_swimlane 摘要）
```

---

## 2. 关键洞察：matmul 已被 box 到 POST_CHUNK 行，padding 在空算

后处理的三个 matmul scope（`rmsnorm_rope_slice` / `rope_fused` / `kv_hadamard`）的 tile
本来就被 padding 到 **`POST_CHUNK=16` 行**（`pooled_kv` / `normed_kv` / `kv_final` / `kv_rope`
都是 `[POST_CHUNK, …]`）。`BATCH_CHUNK_1=1` 时只有 1 行是真 batch，**剩下 15/16 行是 padding 在空算**。

> 所以「把任务做大」的正确杠杆是：**把 padding 填上真 batch**。
> 分组到 `BATCH_CHUNK_1 = POST_CHUNK` 后，这些 matmul scope 单 task 耗时几乎不变（本来就在算 16 行），
> 但**有用功 ×16、task 数 ÷16**——这是近乎免费的赢点。scatter/shift 这类向量 scope 则按行数亚线性增长。

净效果：**任务变少（不是单个变大）**，调度开销蒸发。单 task 仍是 2–8us。
要单 task 真到 ~50us 得把 `BATCH_CHUNK_1`+`POST_CHUNK` 一起提到 64（1 组跑完 64 batch），
但那样外层没并行度、还丢 hetero 测试的分支覆盖，而 compressor 已只占总时长 ~16%，ROI 低（见 §5）。

---

## 3. 改动（3 处，`decode_indexer_compressor.py`）

### 3.1 `BATCH_CHUNK_1 = POST_CHUNK`（1 → 16）

```python
BATCH_CHUNK_1 = POST_CHUNK
assert B % BATCH_CHUNK_1 == 0
assert BATCH_CHUNK_1 <= POST_CHUNK     # 不得超过 matmul 的 box 行数
```

循环 64 → 4 迭代；`kv_state_flat[c_idx:c_idx+16]` / `cos[c_idx:c_idx+16]` / `pooled_kv[16 行]`
全部自然吃满 16 行（这些读写本就按 `BATCH_CHUNK_1` 参数化、按行独立）。

### 3.2 两个 batch 跨步写改成逐 batch 内层散写

`kv_flat`（行步长 S）与 `kv_cache_flat`（行步长 IDX_KV_LEN）每个 batch 只写一行，且行间是
**batch 跨步**，拼不成一块连续 tile：

```python
with pl.at(level=pl.Level.CORE_GROUP, name_hint="kv_and_cache_write"):
    for r in pl.range(0, BATCH_CHUNK_1):
        kv_row_fp32 = kv_final[r : r + 1, 0 : HEAD_DIM]
        kv_flat = pl.assemble(kv_flat, kv_row_fp32, [(c_idx + r) * S, 0])
        cache_row = (c_idx + r) * IDX_KV_LEN + cache_col
        kv_cache_flat = pl.assemble(kv_cache_flat, pl.cast(kv_row_fp32, pl.BF16, "rint"), [cache_row, 0])
```

仍是一个 CORE_GROUP task 覆盖整组；`cache_col` 组内共享（同组 start_pos 同）。

### 3.3 rope 的 cos/sin：`col_expand_mul` → 逐行 `pl.mul`

`col_expand_mul(even_acc, cos_b)` 要求 cos_b 行=1（沿行广播同一条 cos）。分组后每个 batch
有自己的 cos（测试 `cos=rand(B,…)` 随机、golden 用 `cos[b]`），必须逐行：

```python
rope_even = pl.cast(pl.sub(pl.mul(even_acc, cos_b), pl.mul(odd_acc, sin_b)), pl.BF16, "rint")
rope_odd  = pl.cast(pl.add(pl.mul(even_acc, sin_b), pl.mul(odd_acc, cos_b)), pl.BF16, "rint")
```

`even_acc`/`odd_acc` 是 `[POST_CHUNK,32]`、`cos_b`/`sin_b` 是 `[BATCH_CHUNK_1,32]` →
**所以 `BATCH_CHUNK_1` 必须等于 `POST_CHUNK`**，两者 shape 才对齐成逐行 elementwise。
（`ape` 的 `col_expand` 广播**不改**——同组 start_pos 相同 ⇒ token_ape_row 相同 ⇒ ape 组内共享。）

### homogeneity 约束

组内所有 batch 必须 **start_pos 相同**（标量控制流 / `ape_row` / `cache_col` 都从组首 batch 读）。
hetero 测试按 `b // BATCH_CHUNK_1` 分配 start_pos，故 chunk=16 保留 4 个不同 start_pos 组的覆盖；
chunk=64（单组）会丢掉 exact/crossing 分支覆盖。

---

## 4. 前后对比

### 4.1 单 scope 数据（从两份 swimlane 提取）

| scope | 改前 n / avg(us) | 改后 n / avg(us) |
|---|---|---|
| state_scatter_exact_boundary | 256 / 1.44 | 16 / 2.17 |
| softmax_pool | 256 / 2.74 | 16 / 8.01 |
| state_shift | 256 / 1.82 | 16 / 2.79 |
| rmsnorm_rope_slice | 64 / 3.79 | 4 / 3.92 |
| rope_fused_#（aiv） | 128 / 3.04 | 8 / 3.13 |
| rope_fused_#（aic） | 64 / 3.23 | 4 / 3.37 |
| kv_hadamard | 64 / 1.88 | 4 / 2.02 |
| kv_and_cache_write | 64 / 0.90 | 4 / 5.08 |

注意 matmul scope（rmsnorm/rope/hadamard）**avg 几乎不变、数量 ÷16**——印证 §2（本来就在算 16 行）；
向量 scope（scatter/shift）数量 ÷16、avg 仅小幅上升（亚线性）。

### 4.2 区间与总量

| 指标 | 改前 | 改后 |
|---|---|---|
| compressor「中间区」窗口 | 807 → 3515us（~2700us） | 825 → 1042us（~217us） |
| AICore span | 3783us | 1303us（**−65.5%**） |
| Total Test Time | — | **1312us** |
| 精度 | — | idx_kv_cache / score / topk_idxs 全 PASS |

### 改动后泳道图（原始粘贴）

```
（在此粘贴改动后的泳道图 / merged_swimlane 摘要）
```

---

## 5. 为什么没继续（spmd / 更大 chunk 的判断）

改后调度 profile：**Idle 63% / Complete-poll 22% / Dispatch 仅 15%**，Tail OH 均值 8.1us。
即**依赖停顿型**，不是 dispatch 型。

- **不开 spmd**：spmd 只在「AICPU dispatch 是瓶颈」时划算，且会抬高每 task Tail OH 5–10us。
  本例 dispatch 才 15%、Tail OH 已 8us（法则要求 ≤2us + 有 dispatch 拖尾）→ 两个前提都不满足，开了大概率回退。
- **不提到 chunk=64**：单 task 也就 ~8–25us（每行算力本就小），到不了 50us；还丢外层并行度与 hetero 覆盖。
- compressor 现已只占 Total ~16%，瓶颈转到 `decode_indexer.py` 本体的
  `qr_proj`(0–159us) / `qr_hadamard`(314–600us) / `score_fused`(1044–1183us，32×110us)——已接近调优地板。

---

## 6. 可复用的经验法则

1. **泳道图「一大片低 Exec% 碎任务」= 调度开销受限**，第一杠杆是**减 task 数**（coarsening），不是优化单算子。
2. **若 matmul tile 被 box 到固定行数（如 POST_CHUNK）而实际只用 1 行 → padding 在空算**；
   把 `pl.parallel` 的并行维分组到 box 行数，免费吃满 padding（matmul 单 task 不变、数量 ÷box）。
3. coarsening 的副作用要逐个修：**batch 跨步写**拼不成连续 tile → 内层 `pl.range` 逐行散写；
   **广播算子**（`col_expand_mul` 行=1）在每行需独立数据时 → 改逐行 `pl.mul`，并让分组数 = box 行数对齐 shape。
4. **同组必须共享标量控制流**（start_pos / 分支 / 广播的 ape）；测试要按分组粒度构造 start_pos 才不丢覆盖。
5. **「任务变大」常常体现为数量变少而非单个变大**——别盯着 avg us，盯 task 数 × 区间宽度。
6. **coarsening 后再看是否值得 spmd**：profile 是 dispatch 型且 Tail OH ≤2us 才开；停顿型 / Tail OH 高就别开。
