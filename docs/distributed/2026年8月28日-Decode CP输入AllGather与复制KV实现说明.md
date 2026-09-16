# 2026年8月28日 Decode CP 输入 AllGather 与复制 KV 实现说明

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/2026年8月28日-Decode CP输入AllGather与复制KV实现说明.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

- 日期：2026-08-28
- 背景任务：[Issue #905 P3：Decode parallelism and communication](https://github.com/hw-native-sys/pypto-lib/issues/905)
- 实现 PR：[PR #1071：Add: context-parallel DeepSeek V4 decode inputs](https://github.com/hw-native-sys/pypto-lib/pull/1071)
- 基线：`upstream/main@6c292d3`，其中已包含 [PR #1034](https://github.com/hw-native-sys/pypto-lib/pull/1034) 和 [PR #1070](https://github.com/hw-native-sys/pypto-lib/pull/1070)
- 后续独立优化：[PR #1059](https://github.com/hw-native-sys/pypto-lib/pull/1059)，未包含在本 PR 中
- 结构参考：本地任务模板 `工作流程.md`
- 文档目的：解释本次实现补齐的 DSA CP 输入语义、三类 attention 的计算边界、同步协议、cache/metadata 所有权和验证证据

## 0. 如何阅读本文

本文使用三种标签区分结论来源：

- **代码事实**：可从 PR #1071 当前代码直接确认。
- **验证结果**：本次开发中已经实际执行并记录的编译、模拟器或真机结果。
- **尚未验证**：当前没有足够证据，不应写成已经完成或已有收益。

先记住一句话：

> 每个 die 只保留本地 query token；RMSNorm 后将 hidden rows AllGather 成完整 TP/CP group，每个 die 再用完整 token 流独立计算并写入相同的 raw/compressed/indexer KV 副本，最后仍只对本地 query 做 attention。

TP4 满载时就是：

```text
本地 query：128 token × 全部 64 heads
完整 KV：   4 × 128 = 512 token，每个 die 都有一份
```

## 1. 要解决的问题

### 1.1 原来的流程

PR #1034 已补齐 attention 输出侧的 TP 通信：

```text
本地 token × 全部 heads 的 attention output
  -> AllToAll，切换到完整 group token × 本地 O groups
  -> O-A
  -> O-B
  -> ReduceScatter，回到本地 token
```

但输入侧仍缺少 DSA CP 的关键语义。每个 die 只有自己的 `local_t` 行 hidden，`wkv` 和 compressor 也只消费这些本地行，因此每个 die 只能更新本地 token 对应的 KV；这与目标部署的“query 分片、KV 复制”不一致。

### 1.2 具体问题

Issue #905 P3 定义的目标部署是 DP4 × TP4 ⇒ EP16。一个 DP rank 由 4 个 die 组成，一步 decode 有 `64 requests × 8 tokens = 512 tokens`：

- 每个 die 持有 128 个本地 query token。
- 每个 die 计算这 128 个 query 的全部 64 个 heads。
- `wkv`、主 compressor 和 CSA indexer-cache compressor 必须看到完整 512-token hidden stream。
- raw KV、compressed KV、indexer KV 和对应 compressor state 在 4 个 die 内复制。
- query-side metadata 只覆盖本地 128 token；KV 写入侧 metadata 覆盖完整 512 token。

因此，问题不是“再切一次 head”，也不是“交换 KV cache”，而是在本地 RMSNorm 后补一次 hidden AllGather，使每个 die 能独立构造相同的完整 KV 副本。

### 1.3 本次目标

- 在 SWA、HCA、CSA 三条 decode attention 路径中加入 CP 输入 AllGather。
- 保持 HC-pre、RMSNorm、Q projection、indexer query/scoring 和 sparse attention 只处理本地 token。
- 让 raw WKV、HCA/CSA 主 compressor、CSA indexer compressor 处理完整 group token。
- 将 KV/cache/state 写入所需的 RoPE、position、slot mapping 和 state block metadata 改为完整 group 语义。
- 沿用 main 上 #1034 的输出通信语义和 #1070 的 O-A/O-B 实现，不在本任务中重新设计输出路径。
- 保持 TP1 原路径可编译，并支持 TP2、TP4 的动态 active token 数。

### 1.4 不在本次范围内

- 不实现 KV cache sharding、ring attention、zigzag 或跨 die KV page exchange。
- 不修改 #1034 的输出侧协议。
- 不合入 #1059 的 HCA receive → O-A 流水优化。
- 不完成 Issue #905 P2 剩余的 1M context 集成，也不处理 splitfuse、P4/P5/P7。
- 不把“语义补齐”写成“性能提升”；本次新增通信和复制计算的端到端成本尚未实测。

## 2. 数据流

### 2.1 并行概念

代码没有单独的 CP process group。物理 rank 位于 EP world 中，并按下面的公式划分 4-die TP/CP group：

```text
tp_rank    = rank % TP_SIZE
group_base = rank - tp_rank
```

同一个 group 在不同阶段承担不同角色：

| 阶段 | 并行含义 | 每个 die 拥有什么 |
|---|---|---|
| HC-pre / RMS / Q / attention | SP/CP token 切分 | 本地 token，全部 64 heads |
| KV 与 compressor | 复制计算 | 完整 group token，复制 cache/state |
| O-A / O-B | TP output-group/K 切分 | 完整 group token，本地 O groups/K shard |
| ReduceScatter 后 | SP/CP token 切分 | 回到本地 token |

这里的“DP rank”是由一个 TP/CP group 组成的逻辑模型副本，不等于单个物理 rank。

### 2.2 关键静态参数

当前 DeepSeek-V4-Flash DSpark decode 配置：

| 符号 | 含义 | 值 |
|---|---|---:|
| `DECODE_BATCH` | 一个 DP rank 的 request 数 | `64` |
| `DECODE_SEQ` | 每个 request 的 decode token 数 | `8` |
| `DECODE_TOKENS` | 一个 TP/CP group 的 token capacity | `512` |
| `D` | hidden size | `4096` |
| `H` | attention heads | `64` |
| `HEAD_DIM` | 每个 value head 的宽度 | `512` |
| `HC_MULT` | hyper-connection 分支数 | `4` |

满 capacity 时的派生 shape：

| 项目 | TP1 | TP2 | TP4 |
|---|---:|---:|---:|
| `local_t = 512 / TP` | `512` | `256` | `128` |
| `group_t = TP × local_t` | `512` | `512` | `512` |
| 本地 `x_hc` | `[512, 4, 4096]` | `[256, 4, 4096]` | `[128, 4, 4096]` |
| 本地 normalized hidden | `[512, 4096]` | `[256, 4096]` | `[128, 4096]` |
| 完整 group hidden | 本地直通 | `[512, 4096]` | `[512, 4096]` |
| 本地 Q | `[512, 64, 512]` | `[256, 64, 512]` | `[128, 64, 512]` |
| 每个 die 的 raw KV update | `512` 行 | `512` 行 | `512` 行 |

`local_t` 可以小于 capacity；运行时只处理 active 行。AllGather 的有效输出行数始终是：

```text
group_t = TP_SIZE * local_t
```

本文后面的 `local_t=127` 只用于验证 AllGather retained-window 协议能够处理非满窗口，不等价于 attention/model 接口支持任意 token 数。模型侧 active token 仍来自 `active_batch × S`，并继续受当前 8-row tiling/alignment contract 约束。

### 2.3 端到端流程

```text
本地 x_hc [local_t, HC_MULT, D]
  │
  ├─ HC-pre                                      local
  ├─ RMSNorm                                     local
  │    x_normed_local [local_t, D]
  │
  ├──────────── Query 分支 ─────────────────────────────────────┐
  │  Q projection + Q RoPE                       local_t × 64 h │
  │  CSA indexer query / scoring / TopK           local         │
  │                                                            │
  └─ hidden AllGather，rank-major                              │
       x_normed_group [TP * local_t, D]                        │
         │                                                     │
         ├─ raw WKV + KV RoPE                   full group      │
         ├─ raw KV cache write                  replicated      │
         ├─ HCA/CSA main compressor             full group      │
         └─ CSA indexer-cache compressor         full group      │
              compressed/indexer cache + state replicated       │
                                                               │
       local sparse attention ◄─────────────────────────────────┘
         │
         ├─ #1034 attention output AllToAll
         ├─ sharded O-A
         ├─ O-B + ReduceScatter
         ├─ HC-post                               local
         └─ MoE / 下一层                          local
```

### 2.4 AllGather 的行布局

每个 source rank `r` 将自己的全部本地行写入所有 peer 的相同区间：

```text
target_row = r * local_t
rank r rows = [r * local_t, (r + 1) * local_t)
```

所以每个 rank 的本地 `gather_window` 都形成同一份 rank-major 布局：

```text
[rank 0 local rows]
[rank 1 local rows]
...
[rank TP-1 local rows]
```

TP4、`local_t = 128` 时：

```text
rows   0..127  <- rank 0
rows 128..255  <- rank 1
rows 256..383  <- rank 2
rows 384..511  <- rank 3
```

这与 host 侧生成的 global RoPE、position IDs 和 slot mappings 的 rank-major 顺序一致。

### 2.5 三类 attention 的 local/full-group 边界

| 路径 | 只处理 local token | 处理 full-group token，并在各 rank 复制结果 |
|---|---|---|
| SWA | HC-pre、RMS、Q projection、sparse attention | raw WKV、KV RoPE、raw KV cache write |
| HCA | HC-pre、RMS、Q projection、HCA sparse attention | raw WKV、raw KV cache、ratio-128 compressor、compressed KV、compressor state |
| CSA | HC-pre、RMS、Q projection、indexer query/scoring/TopK、CSA sparse attention | raw WKV、ratio-4 main compressor、indexer compressor、raw/cmp/indexer cache、两套 compressor state |

CSA 的 indexer 必须拆开理解：

- `indexer_compressor` 消费完整 group hidden，更新每个 rank 上复制的 indexer KV/scale cache。
- `indexer_query` 只消费本地 hidden/Q，针对本地 query 做 scoring 和 TopK。

## 3. 方案选择

### 候选方案 A：RMSNorm 后 AllGather normalized hidden

处理方式：

- HC-pre 和 RMSNorm 在本地执行。
- 将 `[local_t, D]` BF16 normalized hidden AllGather 为 `[group_t, D]`。
- Q 分支继续读取本地 normalized hidden。
- KV/compressor 分支读取完整 group normalized hidden，并在每个 rank 冗余计算相同结果。

优点：

- 与 DSA CP 的“local query + replicated KV”契约一致。
- 只通信 `[local_t, D]` BF16，不通信 `[local_t, HC_MULT, D]` FP32 的 HC stream。
- HC-pre、RMS 和 Q 不做无意义的重复计算。
- cache 不需要新的 shard layout 或跨 rank page exchange。

缺点：

- 每层新增一次 hidden AllGather。
- `wkv` 和 compressor 在 TP group 内重复计算；TP4 是 4 份相同 KV update。
- retained window 跨 43 层复用，需要严格的 readback-completion 协议。

### 候选方案 B：KV cache 分片，再在 attention 中交换 KV

优点：

- 可以减少重复 WKV/compressor 计算和每 rank cache 容量。

缺点：

- 这不是 Issue #905 引用的 `dsa_cp.py` 契约。
- 需要重新设计 cache page ownership、block table、跨 rank KV 访问和 attention 调度。
- 会把当前只在输入边界发生的通信扩散到 sparse attention 内部。

### 候选方案 C：在 HC-pre 或 RMSNorm 前 AllGather

优点：

- 下游可以统一使用一份 full-group hidden。

缺点：

- 会通信更宽的 HC 状态，或让 HC-pre/RMS/Q 在每个 rank 重复计算完整 group。
- 破坏了本地 query 路径的清晰边界，没有额外正确性收益。

### 我的选择

选择方案 A。它直接对应目标部署的语义，同时把 full-group 范围限制在确实需要复制 KV 的算子上；query 和 layer resident layout 始终保持本地 token。

## 4. 关键实现

### 4.1 代码入口

| 函数/模块 | 输入输出 | 职责 | 为什么需要 |
|---|---|---|---|
| `decode_cp_token_allgather_step` | local normalized hidden → rank-major group hidden | push、两阶段 wait、readback 和 signal retire | 为每个 rank 提供相同的 full-group WKV 输入 |
| `decode_swa` | local query + group KV metadata → local layer output | local Q、group raw KV、local SWA attention | 补齐 dense/SWA 层的复制 KV |
| `decode_hca` | local query + group raw/compressor metadata → local layer output | local Q、group raw KV 和 ratio-128 compressor | 复制 HCA raw/compressed KV 与 state |
| `decode_csa` | local query + group raw/main/indexer metadata → local layer output | local Q/indexer query，group 三类 KV update | 复制 CSA raw/main/indexer cache |
| `decode_compressor_ratio4.py` / `decode_compressor_ratio128.py` | full-group hidden + group metadata → compressed KV/state | 让主 compressor 接受 CP group token 和动态 batch | 保证 HCA/CSA compressed cache 在各 rank 一致 |
| `decode_indexer_compressor.py` | full-group hidden + group metadata → index KV/scale/state | 只保留 index cache 的生产侧 | 与本地 query/scoring 清晰分工 |
| `indexer_query` | local hidden/Q + replicated index cache → local TopK | 从原组合 indexer 中拆出 query/scoring 部分 | 防止 indexer query 被误扩到 full group |
| `rope_interleave.py` | dynamic token RoPE tensors → interleaved Q/K | 传播 local/group 的动态 token extent | 避免 RoPE helper 把 group token 固化为本地 shape |
| `decode_layer.py` | attention + MoE layer | 将 gather window/signal 和 group metadata 传入三类 attention | 验证单层真实调用链 |
| `decode_fwd.py` | 43-layer full forward | 分配一次 retained window，跨层复用并传递 CP ABI | 验证完整模型拓扑和生命周期 |
| `build_distributed_tensor_specs` | host fixture → local/group/replicated tensors | 构造与部署所有权一致的 golden 输入 | 防止测试仍按“每 rank 只有本地 KV”运行 |

### 4.2 AllGather 双阶段 signal 协议

`gather_window` 是每个 rank 上 `[DECODE_TOKENS, D]` 的 HCCL symmetric window；`gather_signal` 是 `[TP_SIZE, 1]` INT32 counter window。窗口在完整 forward 开始时只分配一次，然后跨层复用。

协议如下：

| 阶段 | 当前 rank 的操作 | 本地 remote-source slot 状态 |
|---|---|---:|
| 1. Push | 将本地 hidden 写入所有 peer 的 rank-major 区间；payload 完成后通知 peer | `0 → 1` |
| 2. Payload wait | 等待所有远端 source 的 payload ready | 等待 `>= 1` |
| 3. Readback | 从本地 window 复制 active `group_t` 行到普通 GM tensor；完成后通知所有 peer | `1 → 2` |
| 4. Readback wait | 等待所有 rank 都完成各自 window 的 readback | 等待 `>= 2` |
| 5. Retire | 对本地所有远端 source slot 原子加 `-2` | `2 → 0` |

self rank 不等待自己的 signal：本地 `put` 由 `push_tid` 依赖保证完成。远端 source 才使用 signal row。

第二阶段不能删除。第一阶段只证明“本 rank 可以读自己的 window”；只有所有 rank 都完成 readback，下一层才可以安全覆盖所有 peer 的 retained window。

完整 forward 的生命周期是：

```text
host：分配一份 gather_window + gather_signal
  -> layer 0：0 -> 1 -> 2 -> 0
  -> layer 1：0 -> 1 -> 2 -> 0
  -> ...
  -> layer 42：0 -> 1 -> 2 -> 0
```

独立 fixture 连续执行两轮，专门验证 signal retire 后同一 window 可以再次使用。

### 4.3 SWA

SWA 的切分最简单：

```text
x_hc local
  -> HC-pre + RMS local
  -> ┬─ Q local
     └─ AllGather -> WKV full group -> raw KV cache replicated
  -> sparse attention local
  -> #1034 output path
```

`swa_slot_mapping` 和 KV RoPE 表是 full-group、每个 rank 复制；`swa_indices`、`swa_lens`、`position_ids` 仍是 local query shard。

### 4.4 HCA

HCA 在 raw KV 之外还要复制 ratio-128 compressor：

```text
x_normed_group
  ├─ raw WKV -> raw KV cache
  └─ compressor_ratio128
       ├─ cmp WKV / Wgate
       ├─ compressor state update
       └─ compressed KV cache write
```

因此 `group_ori_slot_mapping`、`group_cmp_slot_mapping`、`group_state_slot_mapping` 和 `group_position_ids` 都覆盖 `group_t`；`compress_state_block_table` 覆盖完整 group batch。

HCA sparse attention 的 query、window indices、sequence lengths 和 compressed block-table view 仍按本地 request/token 组织。

### 4.5 CSA

CSA 同时维护三种 KV 数据：

```text
x_normed_group
  ├─ raw WKV -> raw KV cache
  ├─ compressor_ratio4 -> main compressed KV + main state
  └─ indexer_compressor -> INT8 index KV/scale cache + inner state

x_normed_local
  └─ indexer_query -> local scoring / TopK
```

`ori_slot_mapping`、`cmp_slot_mapping`、`idx_slot_mapping`、`state_slot_mapping`、`inner_state_slot_mapping` 和 `group_position_ids` 都是 full-group metadata。

`cmp_block_table`、`idx_block_table` 和 `kv_seq_lens` 仍按本地 query request 分片，因为它们描述的是本 rank query 要读取哪些历史 cache pages，而不是本轮要写入哪些 global slots。

### 4.6 Metadata 与 cache 所有权

| 数据 | token/batch 范围 | TP/CP group 内布局 |
|---|---|---|
| `x_hc`、Q RoPE、query position IDs | local | 分片 |
| Q、indexer query/scoring/TopK | local | 每 rank 不同 |
| `wq_a`、完整 head 的 `wq_b`、`attn_sink` | local query 的复制参数 | 每 rank 复制 |
| `wkv`、主 compressor 和 indexer-compressor weights | full-group KV 生产参数 | 每 rank 复制 |
| window/SWA indices 和 lens | local query | 分片 |
| sparse-attention compressed/index block table | local query request | 分片 |
| KV/compressor RoPE | full group | 每 rank 复制 |
| raw/compressor/indexer slot mapping | full group | 每 rank 复制 |
| compressor state block table | full group batch | 每 rank 复制 |
| raw KV cache | full history + 本轮 full-group update | 每 rank 复制 |
| HCA/CSA compressed KV 和 compressor state | full history + 本轮 full-group update | 每 rank 复制 |
| CSA indexer KV/scale cache 和 inner state | full history + 本轮 full-group update | 每 rank 复制 |
| O-A/O-B weights | output-group/K shard | TP 分片，沿用 #1034 |

“cache 复制”不是把 cache 本身做 AllGather。每个 rank 接收相同的 normalized hidden，使用相同的复制权重和 global slot mapping，独立算出并写入相同的 cache 内容。

### 4.7 Dynamic shape 与 TP1 适配

新增两个动态 token 符号：

- `CP_LOCAL_T_DYN`：本地 query token 数。
- `CP_GROUP_T_DYN`：`TP_SIZE × local_t` 的完整 group token 数。

TP1 的 compressor 使用 local symbol；TP2/TP4 的 compressor 使用 group symbol。不能把两者强行共用一个 specialization，否则 PyPTO 会在嵌套 inline 调用中混淆 tensor metadata。

`decode_fwd.py` 和 `decode_layer.py` 在 import 时按 TP1/TP>1 选择显式 ABI adapter：

- TP1 调用原有本地 attention，不经过分布式 AllGather。
- TP2/TP4 调用 CP attention，并传入 gather window/signal 与 full-group metadata。

另外，`hc_pre_decode_attention` 和 `hc_post_decode_attention` 不是新数学算法。它们保留一份独立的完整 JIT function identity，用来隔离动态 attention token shape 与固定-capacity MoE 对 HC-pre/HC-post 的不同 specialization。

dynamic `group_t` 会把 tail 行数传到 output projection。rebase 后采用 #1070 的 owner-private O-A 路径：quant 阶段从静态 `[LOCAL_T_PAD, LOCAL_O_WIDTH]` 中间 buffer 读取完整 `[QUANT_T_TILE, O_LORA]` 的 `qz_tile`，不把动态 `qz_rows` 附到 narrowing cast 的输入上；`qz_rows` 只在 scale 和 INT8 结果写回时用于有效区。这样 FP16 → INT8 cast 的 scratch extent 始终是静态的，也不需要额外 `fillpad`。同时，CP adapters 中的 `o_window` 类型统一为 #1070 当前 ABI 的 BF16，避免 wrapper 注解与真实 window 不一致。

### 4.8 与 #1034、#1059 的关系

```text
#1071：local normalized hidden
          -> 输入 AllGather
          -> replicated KV/cache
          -> local query attention
                         │
                         ▼
#1034：attention output AllToAll
          -> sharded O-A
          -> O-B + ReduceScatter
                         │
                         ▼
#1070：按 ReduceScatter owner 拆分 O-A/O-B 中间 buffer 并流水
                         │
                         ▼
#1059：只优化 HCA receive -> O-A 的流水，不改变上述语义
```

当前 #1071 已 rebase 到 #1070，沿用它更新后的 `o_group_a2a`、BF16 reduce window 和 owner-private `o_proj_reduce_scatter`，只把新增 CP wrappers 的 window ABI 与之对齐，不再单独修改 O-A 算法。以后 #1059 rebase 到 main 时，主要冲突面仍在 HCA 输出段；输入 AllGather、full-group WKV 和 replicated cache 语义应保持不变。

## 5. 同步与生命周期

- 谁生产数据：每个 source rank 的 RMSNorm 生成 `x_normed_local`；AllGather push 将它写入所有 peer 的 rank-major window 区间。
- 谁消费数据：本 rank readback task 将完整 active window 复制为 `x_normed_group`；raw WKV 和 compressor 随后消费它。
- 如何通知 ready：payload `put` 之后，对目标 rank 的 `gather_signal[source_rank]` 执行 `AtomicAdd(+1)`。
- 如何确认消费完成：每个 rank 完成本地 readback 后，再对所有 peer 的同一 source row 执行第二次 `AtomicAdd(+1)`。
- buffer 什么时候可以复用：本地所有远端 signal row 都达到 2，并完成 `-2` retire 回 0 后，下一层才能覆盖 window。
- cache 什么时候可以读：raw/compressor/indexer cache write task 通过显式 dependency 接入 sparse attention 或 indexer query。

必须保持的 invariant：

1. 每个 source rank 只写 `[tp_rank * local_t, (tp_rank + 1) * local_t)`，不同 source 区间不得重叠。
2. 所有 rank 的 `local_t` 必须一致，否则 rank-major offset 和 `group_t` 不一致。
3. payload notify 必须晚于同一 push scope 内的 `put`。
4. WKV/compressor 只能消费 active `group_t` 行，不能把 retained window 的 stale padding 带入 cache。
5. local query metadata 必须与当前 rank 的 token slice 对齐；global KV-write metadata 必须与 rank-major group rows 对齐。
6. 下一层覆盖 window 前，所有 rank 必须已经完成上一层 readback。
7. 每轮 retire 必须把远端 signal rows 精确恢复为 0；漏减会让后续层提前通过，重复减会永久等待。

典型故障表现：

| 违反项 | 可能表现 |
|---|---|
| 行 offset 错误 | rank 顺序混乱，KV 写到错误 request/token slot |
| payload notify 过早 | 偶发读取未完成的 hidden rows，产生不稳定精度错误 |
| 缺少 readback completion | 快 rank 在下一层覆盖慢 rank 仍在读取的 window |
| local/global metadata 混用 | cache 内容数值看似合理，但位置或 page 所有权错误 |
| indexer compressor/query 未拆分 | 要么 index cache 不完整，要么 query 工作被重复到 full group |
| TP1/TP>1 specialization 混用 | 编译期 tensor metadata 或 dynamic shape 传播失败 |

## 6. 正确性与性能验证

### 6.1 已完成的本地验证

| 验证项 | 配置 | 结果 |
|---|---|---|
| AllGather retained-window fixture | TP2，`local_t=127`，连续 2 轮 | BF16 逐元素精确通过，`rtol=0`、`atol=0` |
| AllGather retained-window fixture | TP4，`local_t=128`，连续 2 轮 | BF16 逐元素精确通过，`rtol=0`、`atol=0` |
| SWA distributed leaf | TP2，包含非零 `start_pos` 与 cache 输出检查 | 通过 |
| HCA distributed leaf | TP2，包含 raw/cmp cache 与 compressor state 检查 | 通过 |
| CSA distributed leaf | TP2，包含 raw/cmp/index cache、scale 与两套 state 检查 | 通过 |
| SWA/HCA/CSA + MoE layer | TP1、TP2 | 编译通过 |
| `prefill_csa.py` 回归 | TP2，512 tokens，compile-only | 通过 |
| #1070 rebase 与 O-A dynamic-tail 回归 | CI 同版 PyPTO `2b9b9da`、PTOAS 0.60；SWA/HCA/CSA、`decode_layer`，A2/A3 TP2 compile-only | 全部通过，原 `InitMemRef` narrowing-cast 错误消失 |
| 43-layer `decode_fwd.py` | TP1/EP2 | 编译通过，29.41 s |
| 43-layer `decode_fwd.py` | TP2/EP2 | 编译通过；rebase 后复测 34.38 s |
| 43-layer `decode_fwd.py` | TP4/EP4 | 编译通过，33.39 s |
| 43-layer `decode_fwd.py` CI-toolchain 回归 | A2/A3，TP2/EP2，PyPTO `2b9b9da`、PTOAS 0.60 | 编译通过，35.25 s |
| golden harness unit tests | `python -m pytest tests/golden -q` | `289 passed` |
| repository pre-commit/lint | headers、English-only、Ruff、diff check | 通过 |

这些编译时间只是开发环境的 compile wall time，不是设备执行性能，也不能用于比较 TP1/TP2/TP4 的运行速度。

### 6.2 CI 状态

PR #1071 首轮 CI 中：

- docs build、pre-commit、unit tests 和 CodeRabbit 已通过。
- 新增的 `decode_cp_token_allgather.py` 在 `a2a3`、`a2a3sim`、`a5sim` 均通过。
- 首轮 real-device `a2a3` 在旧基线的 output projection 上暴露了动态 `group_t` 的静态 narrowing-cast 约束；rebase 到已解决该约束的 #1070 后，本地已使用该轮 CI 的精确 PyPTO/PTOAS pin 完成五条失败入口的 compile-only 回归，线上复测状态以 PR 最新 check 为准。
- simulator jobs 同时包含本分支新暴露的 dynamic valid-shape 编译问题和仓库既有的 A2/A3/A5 simulator backend/capacity 失败，不能把整组红灯全部归因于本实现，也不能全部当作无关基线忽略。

文档合入前应以 PR 最新 checks 更新本节，不用首轮 CI 代替最终结论。

### 6.3 性能

当前没有 #1071 的真实设备 before/after benchmark，因此结论只能是：

- 本实现补齐了部署正确性和 parallel contract。
- 每层新增一次 `[local_t, D]` BF16 AllGather。
- `wkv`、HCA/CSA compressor 和 CSA indexer compressor 在 TP group 内重复计算。
- CP 的预期收益来自每个 die 只处理本地 query 子集；每个 query 仍从本 rank 的复制 cache 读取它需要的完整历史，而不是这些复制算子本身变快。
- #1034 或 #1059 的性能数字不能归到 #1071。

后续若做性能验收，应在相同 input/golden、相同 TP/EP、相同 active batch 和相同真实设备上，对比：

- decode step 的 per-rank effective wall time；
- hidden AllGather 的通信时间；
- replicated WKV/compressor 的 core busy time；
- sparse attention 因 local query/context 读取减少获得的收益；
- rank 间 start skew 和 load balance。

## 7. 我的判断与疑问

### 我的判断

- “第一个 WKV 前没有 128→512 AllGather”是原实现的真实缺口；#1071 已在 RMSNorm 后补上。
- DSA CP 的准确形态是 local query + replicated KV，不是把 KV cache 在 4 个 die 间分片。
- SWA、HCA、CSA 都必须修改；只改 HCA 会让其他 layer 继续违反同一部署契约。
- CSA 的 indexer compressor 必须 full-group，而 indexer query/scoring 必须 local；这是三条路径中最容易写错的边界。
- #1034 是输出侧正确性/通信基础，#1059 是输出侧 HCA 性能后续；二者都不能替代本次输入侧改动。

### 目前的担忧

- AllGather + replicated compressor 的成本可能抵消部分 local-query attention 收益，必须以真机 wall time 判断。
- dynamic `local_t` 会把过去被静态 capacity 掩盖的 tile/valid-shape 约束暴露到下游算子，必须以完整 CI 而不是单个 fixture 为准。
- simulator 对部分动态 shape/ISA 组合仍有既有支持缺口，sim 红灯需要逐 case 区分模型回归与工具链限制。
- 当前 fixture 使用合成权重和 metadata；真实 checkpoint、serving scheduler 和长上下文组合仍需后续集成验证。

### 尚未完成

- #1071 的最终全 CI 绿灯与 reviewer 认可。
- TP4 真实设备上的 43-layer 数值运行和端到端性能数据。
- 1M context 集成和 splitfuse。
- #1059 在 #1071 合入后的 rebase、冲突处理和 HCA 性能复测。

### 如果重新设计，我可能会

- 将 local-query metadata 和 full-group KV-write metadata 在 public ABI 中做更明确的类型/命名前缀，减少误传。
- 为 retained-window 协议增加独立的多层/多 epoch contract test，而不仅是两轮 fixture。
- 在编译矩阵中固定加入 dynamic tail token 数，避免只用满 capacity 掩盖 valid-shape 问题。
- 在真实设备 trace 中把 AllGather、复制 WKV/compressor 和 local sparse attention 分别标注，避免只看总时间无法解释收益来源。

## 8. 建议阅读顺序

1. `decode_cp_token_allgather.py`：先理解 rank-major 行映射和 `0 → 1 → 2 → 0` 协议。
2. `decode_swa.py`：看最简单的 local Q / full-group KV 分叉。
3. `decode_hca.py`：看 ratio-128 compressor 如何扩到完整 group。
4. `decode_compressor_ratio4.py` / `decode_compressor_ratio128.py`：看主 compressor 的 full-group token/batch 适配。
5. `decode_csa.py` + `decode_indexer.py` + `decode_indexer_compressor.py`：看 indexer cache 生产与 query/scoring 的拆分。
6. `rope_interleave.py`：看 local/group 动态 token shape 如何进入 RoPE helper。
7. `decode_layer.py`：看 CP ABI 如何接到 attention + MoE 单层。
8. `decode_fwd.py`：看 full-group metadata、本地 Q slice、packed cache pools 和 retained window 如何跨 43 层复用。
9. `decode_o_proj.py`：确认 #1034 的输出 AllToAll 语义和 #1070 的 owner-private O-A/O-B 流水仍保持独立。
