# HCA decode sparse_attn qk_pv 调优日志（2026-07-13）

> 目标：缩小 HCA decode attention 里 qk_pv 阶段相对 CSA 的性能差距。
> 结论：**静态拆分拿到 −23% qk_pv（已提交 `0e9adbc`）**；其余杠杆（head-batch 拆 / block-load / CSA 游标）均证伪。
> 下一杠杆：KV INT8（W8A8），是另一条硬线，未启动。

## 0. 起点：CSA vs HCA 泳道对比

对比两份 orchestrator 泳道（`decode_attention_csa.py` vs `decode_attention_hca.py`，均 a2a3）：

| stage | CSA | HCA | 备注 |
|---|---|---|---|
| **qk_pv (AIV)** | 44us / 48 核 | **64.4us / 16 核** | HCA 慢 45%，只占 1/3 vector 核 |
| qk_pv (AIC) | 44us / 24 核 | 65.9us / 8 核 | HCA 只占 1/3 cube 核 |
| merge_norm / proj_a / proj_b | ~持平 | ~持平 | 不是瓶颈 |

根因（源码级）：CSA `qk_pv` 用 `pl.spmd(NUM_QK_CORES=24)` 定宽扇出 + qk_plan 负载均衡游标；HCA 用 `pl.spmd(T)`（一 token 一 lane，T=8）→ 只开 8 cube / 16 vector lane，2/3 核空转。

## 1. 已提交收益：qk_pv 静态拆分（−23%）

**改动**（`models/deepseek/v4/decode_sparse_attn_hca.py`）：
- `pl.spmd(T)` → `pl.spmd(T * SPARSE_BLOCKS)`
- `qk_item = pl.tile.get_block_idx(); qk_t = qk_item // SPARSE_BLOCKS; qk_sb = qk_item - qk_t * SPARSE_BLOCKS`
- 删 `for qk_sb in pl.unroll(SPARSE_BLOCKS)`，body 减 4 缩进
- 写偏移 `qk_row = qk_blk_base + qk_sb * H_TILE` 不变 → bit-identical

**为什么用静态拆、不搬 CSA 游标**：HCA 的 item 是 uniform 的（无 valid_block_mask 跳块），`pl.spmd(N)` 本来就让 runtime 自动负载均衡 uniform item；CSA 游标是为 non-uniform item 负载均衡，对 HCA 是 no-op（QK_ITEMS=16 < 24，8 lane 空转 + 多一个 qk_plan 预处理开销）。

**实测**（3 跑，原始基线 64.4us）：
- qk_pv_aiv **64.4 → 49.8us（−23%）**，σ≈0.3us
- TOTAL 395 → ~378us（−4%）
- 核占用翻倍：AIV 16→32、AIC 8→16
- orchestrator golden **PASS**（bit-exact，rtol=1/128）

**提交**：分支 `perf/dsv4-hca-qkpv-spmd-widen`，commit `0e9adbc`。已 push？**否**（待定）。

## 2. 核内 profile：qk_pv 是访存/load-bound，不是算力-bound

对静态拆版 qk_pv 跑 in-core op-sim（单核 `<<<1>>>` = token0/block0）。产物：
`build_output/incore_qk_pv_hca_staticsplit_20260713_024023/`（`qk_pv.clean.json` + `instr_metrics.json` + `summary.txt`）。

cubecore0（221k cyc，长杆）per-pipe：
| pipe | cyc | 占比 | 含义 |
|---|---|---|---|
| **MTE2** | 107k | **48%** | GM→L1 的 KV gather（瓶颈） |
| MTE1 | 48k | 22% | L1→L0 cube 操作数 staging |
| **CUBE** | 29k | **13%** | QK/PV matmul 真算力 |
| FIXP/SCALAR/FLOWCTRL | ~33k | 15% | |

**结论**：load 合计（MTE2+MTE1）= 70%，CUBE 算力只 13%。瓶颈是 gather 访存，不是 matmul。

> ⚠️ 方法论：核内 profile 给「机制上限」，单核隔离 stall 放大 ~4×，绝对值不可信、占比/吞吐比可信。不能直接外推成「墙 −X%」—— 墙变化必须泳道实测。

## 3. 证伪的杠杆（都回退了）

### 3.1 CSA 游标 → no-op
见 §1，uniform item 下静态拆已是最优，游标纯增开销。**不搬。**

### 3.2 head-batch 拆 → +82% 回退
把 `qk_hb`（head-batch 维，`H//QK_M_TILE=4`）从内层 `pl.pipeline` 提到 spmd 索引，item = token×block×head-batch = 64。
- golden **PASS**（bit-exact）
- qk_pv_aiv **49.8 → 90.9us（+82%）**
- 原因：每个 head-batch task 必须各自重 gather 128 行 KV → gather ×4，MTE2（48% 长杆）被放大 → 碾压并行收益。
- **回退**。这与此前 qk_pv 拆分实验的结论一致。

### 3.3 window block-load → 死路（最值得记的一条）

**动机**：核内 profile 里 window gather 的 128 次标量 `gather_row`（`MOV_OUT_TO_L1_MULTI_ND2NZ`）跑 **1.3 B/cyc**，而同 kernel 的连续 ND2NZ load 跑 **~30 B/cyc**（差 ~25×）。WIN=BLOCK_SIZE=128 → window 看似占一个连续块 → 一次连续 bulk load 替代 128 次间接 gather。

**改动**：block 0（window）用单次多行 `gather_row(qk_kv, ori_kv_flat, [0,0], [qk_win_base,0], [128, HEAD_DIM])`，block id = `window_swa_indices[t,0]//BLOCK_SIZE`；block 1（compressed）保持逐行。利用 attention 对 K/V lane 置换不变（softmax + PV 求和 commute），按物理序加载、忽略 ring 旋转。

**结果**：
- standalone golden（`decode_sparse_attn_hca.py`）**PASS**
- swimlane timing：qk_pv_aiv **44.9 → 36.9us（−18%）**（静态拆 vs block-load，同 rebuilt env）
- **但 orchestrator golden（`decode_attention_hca.py`）FAIL**：element rdiff 0.6–3% > rtol=1/128

**根因（死路）**：standalone 的 `init_window_swa_indices` fixture 把 window 塞进**一个物理块**（线性）→ block-load 按同序加载 → bit-exact → **假阳性 PASS**。真 path 的 window 在 `decode_metadata.py: history_window_swa_indices_and_lens` 里是**分页跨 2 个物理块**：
```python
for pos in range(start, abs_pos + 1):     # 128 个 window 位置
    logical_blk = pos // block_size
    blk = block_table[b, logical_blk]      # 分页块表 → 物理 block id
    indices[t, out_k] = blk * block_size + intra
```
128-token window 结束在块中间时跨 2 个 logical block → 2 个物理 block。block-load 只取 `window_swa_indices[t,0]//BLOCK_SIZE` 那一块 → **漏第二块** → ~一半 lane 取错行 → orchestrator golden FAIL。和 CSA 此前遇到的问题相同：ring/paged window 必须逐行 gather，per-block gather 已接近最优。**回退，死路。**

> **教训**：gather/index 改动必须先过 orchestrator golden；standalone 的线性 window fixture 会系统性掩盖分页 bug。exit≠0 是红旗，查清再报性能数。下「数据布局连续」结论前先读真 path 的布局构造函数。

## 4. 环境侧支线：simpler 子模块版本回归

调到一半，所有上板跑突然崩 `AttributeError: 'CallConfig' object has no attribute 'enable_dump_args'`（`device_runner.py:664` 无条件 `cfg.enable_dump_args=...`）。

**根因**：pypto HEAD `fe8ebbfe` 的 gitlink 指向 simpler **438d5cb1**（加 `enable_dump_args` 的 nanobind setter，#1270/#1236），但 simpler 实际 checkout 在 **c94aa9f3**（旧，无该 setter）。重编 pypto 没用 —— `CallConfig` 实际住在 simpler 的 `_task_interface.so`（不是 `pypto_core.so`），pypto 的 `pip install -e .` 不编它。

**修复**：
1. `git -C runtime stash`（挡 checkout 的本地改动 `profiling_config.h`）
2. `git -C runtime checkout 438d5cb1` —— **严格包含 c94aa9f3**（c94 是祖先，438 是后代 + 43 commit，含 early-dispatch 修复 + enable_dump_args）。**不影响 early-dispatch，是纯升级。**
3. 重编 simpler：`cd runtime && pip install -e . --no-build-isolation` → `_task_interface.so` 04:55 重生，`enable_dump_args` 计数 0→3。
4. pypto 也重编过（`pypto_core.so` 04:49）。

> 注：438d5cb1 的 PR 号 #1236 比 c94aa9f3 的 #1288 小，但**实际更晚**（438d5cb1 = 07-13，c94aa9f3 = 07-08）。PR 号不代表合并顺序，以 git ancestry 为准。

## 5. profiling_config.h stash 恢复

`git -C runtime stash pop` 冲突：438d5cb1 把宏 `PTO2_ORCH/SCHED_PROFILING` 改名为 `SIMPLER_*`（默认 0），stash 用旧名、值=1。**按「新名 + 值=1」合并**（保 WIP 意图、落到新名）：
```c
#define SIMPLER_ORCH_PROFILING  1   // 0 -> 1
#define SIMPLER_SCHED_PROFILING 1   // 0 -> 1
```
净 diff 2 行，stash 已 drop。
⚠️ 头文件改动，**要重编 simpler 才生效**；`=1` 会开 orch/sched 插桩、**污染计时**，干净 perf 跑前改回 0。

## 6. 当前状态（明天接着干的起点）

- **pypto-lib**：分支 `perf/dsv4-hca-qkpv-spmd-widen`，HEAD `0e9adbc`（静态拆，−23%，已验证）。工作树干净（block-load 已回退）。未 push、未开 PR。
- **pypto/simpler**：simpler @ 438d5cb1，pypto_core.so / _task_interface.so 均已重编，env 可上板。
- **simpler 本地**：`profiling_config.h`（SIMPLER_*_PROFILING=1，未提交，未重编生效）；`profiling_config.h.orig`（旧备份，未跟踪）。

## 7. 下一杠杆：KV INT8（W8A8）—— 未启动

核内 profile 说 qk_pv 是 gather(MTE2)-bound（48%），算力只 13%。继续加并行已无路（§3 全证伪），**唯一剩下的杠杆是降 gather 字节数 = KV INT8**。

- 仓里 INT8 KV **只用在 indexer**（`idx_kv_cache`，quant-on-write + per-position scale + dequant-on-read 融进 score matmul），**attention 的 `ori_kv`/`cmp_kv` 全仓 BF16，没人动过**。
- 但给 cmp_kv 上 INT8 **已试过撞墙**：写侧 32B 对齐 + 读侧 L1-dequant 都难；判定可行路径是 **W8A8（indexer 风格，q 也 quant）**，不是单纯 KV-INT8。
- W8A8 对 qk_pv 的复杂点：indexer 只有一步 score（QK）matmul，W8A8 干净；但 qk_pv 是 **QK + softmax + PV 三件套，共用 qk_kv**，QK 可 INT8、但 **PV 的 exp 是 FP 概率、不能 INT8**，PV 仍需 FP KV → dequant 墙只是从 QK 挪到 PV。且跨 kernel（compressor 写 INT8 cmp_kv + qk_pv 读）+ 精度重验。
- 已有实验估计的天花板：~20% qk_pv Exec，standalone wall 可能被 ~3% 噪声淹没。**性价比偏低**，是另一条硬线，开干前要先 in-core profile 拿到 gather vs cube 占比确认。

## 附：关键文件 / 产物路径

- 改动文件：`models/deepseek/v4/decode_sparse_attn_hca.py`（qk_pv scope，~line 234）
- 提交：`0e9adbc`（`perf/dsv4-hca-qkpv-spmd-widen`）
- 核内 profile：`build_output/incore_qk_pv_hca_staticsplit_20260713_024023/`
- 基线泳道：`build_output/_jit_attention_hca_test_20260712_193319/`（原始 64.4us）
- 本文已记录静态拆分结果和三条已证伪路径。
