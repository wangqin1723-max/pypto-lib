# DSv4 Compressor 性能调试日志

> 记录 DeepSeek-V4 compressor（`models/deepseek/v4/decode_compressor_ratio4.py`
> 及 ratio128 / prefill 同族）的性能调试**过程**：每个 scope 的瓶颈画像、试过的
> 杠杆（含负结果）、以及还没动的方向。新发现往后追加，**负结果也写**——省得重复踩。
>
> 相关专题文档（已有，互补不重复）：
> - `dsv4-compressor-batch-coarsen.zh.md` — 逐 batch 循环 coarsening
> - `dsv4-compressor-kv-hadamard-cache-write-mix.zh.md` — kv_hadamard + cache_write 融合
> - `swimlane-tuning-log.zh.md` — L2 泳道逐条改动

---

## 0. 背景 / 现状

- **基线**：`decode_compressor_ratio4` ≈ **312us**（真机 a2a3，`--enable-l2-swimlane`）。
- **临界 scope**：`kv_score_proj` 占 ~77% wall（32 任务 × ~90us）；`state_scatter_paged` ~15%。
- 多数 tile 级杠杆**已试穿**（见 §2 表）；当前判断为**结构性下限**，进一步收益要么来自
  算法层（提高压缩率 / 摊薄权重读），要么来自 L2 任务间排布。

### kernel 清单与角色

| kernel | 角色 | 瓶颈类型 |
|---|---|---|
| `kv_score_proj` | `x @ W_score`（两路输出 64×64，K≈16384） | **MTE2 load-bound**（搬权重） |
| `state_scatter_paged` | 把 KV state 按 paged block-table 散写 | **MTE2+MTE3 搬运 bound**（access-count） |
| `rmsnorm_rope` | RMSNorm + RoPE | VECTOR-bound（均衡） |
| `kv_and_cache_write` | KV 写 cache（数据相关分支） | 数据相关，需真值才能测 |
| `softmax_pool` | softmax + pool（数据相关分支） | 数据相关，需真值才能测 |

---

## 1. 核内泳道画像（in-core，msprof op-sim，2026-06-26）

> 工具：`.claude/skills/incore-profiling`（task-submit 跑采集 + `clean_sim_trace` 清洗）。
> 源 build：`build_output/_jit_compressor_test_20260626_101323`。
> **单核 `<<<1>>>` 隔离视角**：绝对 stall% 偏高不可引，但"哪个 pipe 主导、各 pipe 相对忙闲"可信。
> 清洗后泳道在 `build_output/incore_<kernel>_compressor_20260626_101930/<kernel>.clean.json`（Perfetto 打开）。

| kernel | 主导 pipe（busy% of wall） | 判读 |
|---|---|---|
| **kv_score_proj** | MTE2 ~74% / CUBE ~19% / MTE1 ~17% / MMAD 仅 11 条 | **MTE2 load-bound**。低算术强度：读一大块 W(K≈16384) 只产 64×64×4=16KB 输出，cube 算力闲 ~80%。 |
| **state_scatter_paged** | MTE2 ~86% **且** MTE3 ~85% 并发 / 无计算引擎 | **搬运/access-count bound**（gather+scatter）。 |
| **rmsnorm_rope** | VECTOR ~34% / MTE2 ~28% / MTE3 ~16% | 偏 **vector-bound**，较均衡。 |
| kv_and_cache_write | ~0.2ns 退化 | auto golden 把数据相关控制输入清零 → 0 迭代。需灌真值重测。 |
| softmax_pool | ~0.4ns 退化 | 同上。 |

### 两个易错点（已查清，留作 reference）

1. **kv_score_proj 泳道里没有 MTE3，不是漏了**。它是纯 matmul：结果累在 **L0C**，`TSTORE` 一个
   L0C 常驻 tile 时硬件走 **FIXPIPE(L0C→GM)**，仿真器归到 FIX 管线（cubecore 上 9 条 FIX），
   不占 MTE3。MTE3 只搬 UB→GM。源码两条 `TSTORE` 前有 `set_flag(PIPE_M, PIPE_FIX)` 为证。
   对照 `rmsnorm_rope`（向量写回）有 12 个 `PIPE_MTE3`，泳道里 MTE3 就在。
2. **后两个退化 = 数据相关 kernel + 合成零输入**。`softmax_pool` 里 `v26=v1[..]; if (v26%4>=2)`、
   `kv_and_cache_write` 里 `v26=v3[..]; if (v26%4>=2)` 这种**从 GM 读整数门控整段计算**的，被零输入
   判假跳过 → 近空轨迹。要真测：改 `golden.py` 把这些**控制张量**（block-table / valid-length）
   灌成合法非零值（让分支成立、work-table 稠密），重建 `*_sim` 再采集；bulk 数据可继续随机/零。

---

## 2. 已试杠杆（含负结果）

> 来源：ratio4 真机实验（kv_score_proj 占 77% wall）。今天的 in-core 画像与此自洽，未翻案。

### kv_score_proj（MTE2 load-bound）

| 尝试 | 结果 | 备注 |
|---|---|---|
| INT8 权重（少读一半字节） | **+2% 更慢** ❌ | 关键证据：它是 **latency/transaction-bound 不是 raw-byte-bound**；减字节不减事务数无用 |
| K 方向 overlap | **+6% 更慢** ❌ | |
| 加宽 row / N tile（N 64→128） | **L2 中性** ⚪ | x 重读被 L2 吸收，非 HBM-bound |
| task 内 split-K | **被堵** 🚫 | ptoas tmov 不支持 |

→ 核内/tile 层判**结构性下限**。

### state_scatter_paged（~15%，access-count bound）

| 尝试 | 结果 | 备注 |
|---|---|---|
| parallelize（散到 64 任务） | **+2.6% 更慢** ❌ | >核数多波次 + 摊薄 core 反而推高 kv_score_proj |
| coarsen（合并任务） | 真机 507018 🚫 | 但**触发机制存疑**，见下方调查 |

→ 维持串行。

#### coarsen 507018 复现调查（2026-06-26，结论：self-RAW 归因被证伪）

之前把 coarsen 的 507018 归因为"单 task 在 `compress_state_flat` 上 self-RAW"。
写了两个最小 repro 直接在设备上验证（`models/deepseek/v4/_tmp_self_raw_repro.py`、
`_tmp_coarsen_softmax_repro.py`，6 次 device run）：

| repro | 结构 | 结果 |
|---|---|---|
| #1 单 task 读+写同一 GM scratch（动态行） | create_tensor scratch | **PASS** |
| #1 内层 `pl.range` 就地 RMW scratch | loop-carried | **PASS** |
| #2 paged-Out reshape + 跨 scope scatter→pool + 在线 softmax，内层 `pl.range` coarsen | B=8/D=64 | **PASS** |
| #2 同上，**真实规模 B=64 / D=512 / 512 paged 行** | range/unroll/serial 三 mode | **全 PASS** |

**默认池下也没一个复现 507018。** 于是反向做决定性实验：**把 ring 池调小**
（`PTO2_RING_TASK_WINDOW=DEP_POOL` 设 8 / 32 / 128），同一个真实规模 repro：

| 池大小 | serial | range |
|---|---|---|
| 8 | **507018** | **507018** |
| 32 | **507018** | **507018** |
| 128 | 507018\* | PASS |
| 131072 | PASS | PASS |

\* sz=128 serial 错误码不同（`sched=100` 非 `orch=3`），疑似小 kernel ~50% intermittent 噪声。

**已证实：507018 由 ring 池大小直接决定。** 缩到 8/32 必挂,放大到 131072 必 PASS。因此：
- **"self-RAW 必死锁" / "coarsen 结构性死锁" 作为机制被证伪**——pypto 处理动态行 self-RAW、
  跨 scope paged 读写、`pl.range` coarsen 都正常。
- 真机"coarsen 507018" = **全 5-scope 任务图把默认池子撑爆的资源限制**，不是逻辑死锁、不是 bug。
  **解法 = 调大 `PTO2_RING_TASK_WINDOW` / `PTO2_RING_DEP_POOL` / `PTO2_RING_HEAP`**（run.sh 那三行），
  然后随便 coarsen。
- caveat：repro 用 `rtol=1e9`，"range loop-carries softmax → **NaN**"那条正确性问题未验（与挂死两码事）；
  serial/range 谁更耗池被 intermittent 噪声盖住，读不出。

→ 任何大/coarsen kernel 撞 507018，**先抬 `PTO2_RING_*` 再怀疑逻辑死锁**；不当调度器 bug 提。

---

## 3. 还没动的方向（headroom）

都在**更高层**，不在单个 kernel 的 tile：

1. **摊薄权重读**：同一份 `W` 喂更多行。decode 单 token 行数少 → `W` 读摊不开（decode 本质瓶颈）；
   能批 token（prefill / 多 token decode）才摊得动。
2. **higher compression ratio（ratio128）**：算法层少算少读，直接砍工作量。← 之前记的 next step。
3. **L2 任务间排布**：让 32 个并行任务的 MTE2 叠得更满；单看 kernel 测不出，要整网 schedule。

---

## 变更历史

- **2026-06-26**：建档。补 in-core 泳道画像（§1）+ 收敛已试杠杆（§2）。
