# 2026年8月29日 Decode SWA CP 实现讲解

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/2026年8月29日-Decode SWA CP实现讲解.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

- 日期：2026-08-29
- 对应实现：[PR #1071](https://github.com/hw-native-sys/pypto-lib/pull/1071)
- 代码快照：`4090f4d`（PR #1071 合入 `main` 后的提交）
- 主要文件：`models/deepseek_v4_flash_dspark/decode_swa.py`
- 文档范围：只讲 SWA 的 CP 输入改动，不讲 HCA compressor、CSA indexer，也不讲后续性能优化
- 保存方式：仅作为本地记录，不提交、不推送到远程

## 1. 先记住结论

SWA 的 CP 改动可以概括成一句话：

> 每张卡只计算自己的 Q 和 attention 输出，但每张卡都要用整个 TP/CP group 的 hidden 计算完整 KV，并把完整 KV 写进本卡的复制 KV cache。

TP4 满载时：

```text
总请求数：64
每个请求：8 个 decode token
整个 group：64 × 8 = 512 token
每张卡：512 / 4 = 128 token
```

因此每张卡上的计算范围是：

```text
Q：         本地 128 token
KV：        完整 512 token
Attention：本地 128 个 query
输出：      本地 128 token
```

这里的“完整 512”来自当前部署配置，不是 SWA 的窗口 128。SWA 的窗口 128 表示每个 query 最多读取最近 128 个历史位置。

## 2. 为什么原来的代码不够

原来的 decode SWA 每张卡只有自己的 128 行 hidden：

```text
rank0：只看到 token   0～127
rank1：只看到 token 128～255
rank2：只看到 token 256～383
rank3：只看到 token 384～511
```

如果 `wkv` 直接使用本地 hidden，那么每张卡只能生成自己的 128 行 KV：

```text
rank0 cache：只更新 rank0 的 token
rank1 cache：只更新 rank1 的 token
...
```

但当前部署契约中，`kv_cache` 在这 4 张卡上是完整复制的。每张卡都应该保存相同的 512-token 更新，因此必须先把四张卡的 normalized hidden 聚齐。

## 3. 改完后的完整数据流

```text
本卡 x_hc [128, 4, 4096]
  │
  ├─ hc_pre（本地）
  │    把 4 路 HC 状态整理成 attention 使用的一路 hidden
  │
  ├─ RMSNorm（本地）
  │    x_normed_local [128, 4096]
  │
  ├────────────── 本地 Query 分支 ──────────────┐
  │  Q projection + Q RoPE                     │
  │  q [128, 64, 512]                          │
  │                                            │
  └─ AllGather                                 │
       rank0/1/2/3 的 128 行拼成 512 行         │
       x_normed_group [512, 4096]              │
           │                                   │
           ├─ KV projection + KV RoPE          │
           │    kv [512, 512]                  │
           │                                   │
           └─ 写入本卡完整复制的 kv_cache      │
                                               │
       本地 q + 本卡完整 kv_cache ◄────────────┘
           │
           ├─ SWA sparse attention，只算本地 128 个 query
           ├─ attention 输出 AllToAll
           ├─ 分片 O-A / O-B + ReduceScatter
           ├─ hc_post（本地）
           └─ 本卡输出 [128, 4, 4096]
```

最简单的记法是：

> 写 KV 要全，读 KV 和算 query 只处理本卡请求。

## 4. 按代码一步一步看

### 4.1 区分本地 token 和完整 group token

文件：`models/deepseek_v4_flash_dspark/decode_swa.py`

```python
T_DYN = CP_LOCAL_T_DYN
CP_T_DYN = CP_GROUP_T_DYN
```

- `T_DYN`：本卡 token 数，TP4 满载时是 128。
- `CP_T_DYN`：整个 group 的 token 数，TP4 满载时是 512。

这两个动态维不能混用。看到 `T_DYN`，通常说明数据只属于本地 query；看到 `CP_T_DYN`，通常说明数据要覆盖完整 KV 写入路径。

### 4.2 接口中哪些参数是本地，哪些是完整组

`decode_swa` 的接口把两类数据明确分开：

| 参数 | TP4 shape | 范围 | 用途 |
|---|---:|---|---|
| `x_hc` | `[128, 4, 4096]` | 本地 | 本卡输入 |
| `freqs_cos/sin` | `[128, 64]` | 本地 | 本地 Q RoPE |
| `position_ids` | `[128]` | 本地 | 本地 query 位置 |
| `swa_indices` | `[128, 128]` | 本地 | 本地 query 去 cache 读哪些行 |
| `swa_lens` | `[128]` | 本地 | 本地 query 的有效窗口长度 |
| `kv_freqs_cos/sin` | `[512, 64]` | 完整组 | 完整 KV RoPE |
| `swa_slot_mapping` | `[512]` | 完整组 | 512 行 KV 分别写到 cache 哪一行 |
| `kv_cache` | 完整 cache | 每卡复制 | 本地 attention 读取 |

`swa_slot_mapping` 可以理解为一张“写入地址表”：第 `i` 个新 KV 应该写到 cache 的哪一行。

### 4.3 hc_pre 和 RMSNorm 保持本地

在 `decode_swa` 中先执行：

```python
hc_pre(..., x_mixed, post_t, comb_t)
rms_norm(x_mixed, attn_norm_w, x_normed_t)
```

`hc_pre` 和 RMSNorm 都是逐 token 计算，不需要其他 rank 的 token。

`hc_pre` 还会生成 `post_t`、`comb_t`。这两份数据只在最后的本地 `hc_post` 使用，不应该被 AllGather。

### 4.4 在 RMSNorm 后做 AllGather

代码：`models/deepseek_v4_flash_dspark/decode_swa.py` 中的 `decode_swa`，调用公共函数：

```python
x_normed_group, gather_signal = decode_cp_token_allgather_step(
    x_normed_t,
    x_normed_group,
    gather_window,
    gather_signal,
    group_base,
    tp_rank,
)
```

TP4 时，四张卡最终都得到相同布局：

```text
rows   0～127：rank0 的 hidden
rows 128～255：rank1 的 hidden
rows 256～383：rank2 的 hidden
rows 384～511：rank3 的 hidden
```

为什么在 RMSNorm 后通信，而不是直接通信 `x_hc`：

```text
x_hc：     [128, 4, 4096] FP32，约 8 MiB/卡
x_normed： [128,    4096] BF16，约 1 MiB/卡
```

后者通信量约小 8 倍，而且 Q、hc_pre、hc_post 仍然可以保持本地。

### 4.5 Q 使用本地 hidden

```python
q_proj_rope(
    x_normed_t,
    ...,
    freqs_cos,
    freqs_sin,
    ...,
)
```

输入是本地 `[128, 4096]`，所以每张卡只生成本地 Q：

```text
q [128, 64, 512]
```

没有必要生成完整 512-token Q，因为本卡最终只负责自己的 128 个 query。完整计算 Q 会把工作重复 4 次。

### 4.6 KV 使用 AllGather 后的完整 hidden

```python
kv_proj_rope(
    x_normed_group,
    wkv,
    gamma_ckv,
    kv_cos_il,
    kv_sin_signed,
    kv_swap_idx,
    kv,
    late_dep,
)
```

输入是完整 `[512, 4096]`，所以每张卡都生成：

```text
kv [512, 512]
```

这也解释了为什么接口要有两套 RoPE：

- `freqs_cos/sin` 对应本地 128 个 Q。
- `kv_freqs_cos/sin` 对应完整 512 个 KV。

### 4.7 用完整 slot mapping 写复制 KV cache

```python
for write_t in pl.range(group_t_dim):
    write_row = swa_slot_mapping[write_t]
    kv_cache_flat[write_row] = kv[write_t]
```

循环范围是 `group_t_dim`，TP4 满载时为 512。因此每张卡都会用相同的 KV、相同的 mapping 更新自己的 cache 副本，更新后四张卡上的 `kv_cache` 保持一致。

这里没有再 AllGather KV cache，因为 hidden 已经聚齐；每张卡使用相同权重独立计算，就能得到相同 KV。

### 4.8 Attention 又回到本地范围

```python
sparse_attn_swa(
    q,
    kv_cache,
    swa_indices,
    sparse_bias,
    freqs_cos,
    freqs_sin,
)
```

传入的是：

- 本地 `q`；
- 本地 `swa_indices/swa_lens`；
- 每卡完整复制的 `kv_cache`。

所以每张卡只计算本地 128 个 query，但这些 query 可以从完整 cache 中读取属于自己的历史窗口。

### 4.9 输出仍然回到本地 token

SWA attention 产生本地 token × 全部 heads 的结果。后面的输出路径沿用已有实现：

```text
本地 token × 全部 heads
  -> AllToAll
完整 group token × 本地 O groups
  -> 分片 O-A / O-B
  -> ReduceScatter
本地 token × hidden
  -> hc_post
```

因此 CP 输入 AllGather 不会改变层与层之间的 token 所有权。下一层收到的仍然是本卡 128 个 token。

## 5. AllGather 公共函数做了什么

文件：`models/deepseek_v4_flash_dspark/decode_cp_token_allgather.py`

`decode_cp_token_allgather_step` 可以分成五步：

1. 每个 rank 把自己的 hidden 写到所有 peer 的 `gather_window`。
2. 通知 peer：“我的数据已经写好。”
3. 等待其他 rank 的数据全部到达。
4. 把本地 window 中的完整 group hidden 读回普通 tensor。
5. 确认所有 rank 都读完，再把 signal 恢复到 0，供下一层复用。

第五步很重要。完整 forward 会在多层之间复用同一块 window；如果某张卡还没有读完，下一层就覆盖 window，会造成偶发错误。

可以把 signal 的一轮生命周期理解为：

```text
0：还没到
1：payload 已经到，可以读
2：所有人已经读完，可以复用
再减 2：回到 0，开始下一轮
```

## 6. TP1 为什么有单独路径

对上层来说统一调用：

```python
decode_swa_attention(...)
```

但在模块加载时会根据 `TP_SIZE` 选择：

```text
TP_SIZE == 1 -> decode_swa_tp1
TP_SIZE > 1  -> decode_swa
```

TP1 时 local token 就是完整 group token，不需要 AllGather：

```text
local_t = group_t = 512
```

这个 adapter 不是运行时在设备上判断，而是 Python import 时只保留一个调用图。这样既保持上层 ABI 一致，也避免 PyPTO 把本地动态维和 group 动态维错误地 specialization 到同一个 inline 函数中。

`hc_pre` 和 `hc_post` 没有复制实现；TP1、TP2、TP4 都复用公共函数。差别只在 attention wrapper 选择有通信还是无通信路径。

## 7. 各文件的职责

| 文件 | 职责 |
|---|---|
| `decode_swa.py` | SWA 主流程：本地 Q、完整 KV、本地 attention 和输出路径 |
| `decode_cp_token_allgather.py` | 把各 rank 的 normalized hidden 拼成完整 group hidden |
| `decode_layer.py` | 单层接口、动态 shape 检查、通信 window/signal 传递 |
| `decode_fwd.py` | 在完整模型 forward 中加入 SWA group metadata，并跨层复用 gather window |
| `decode_sparse_attn_swa.py` | 使用本地 Q 和复制 KV cache 计算本地滑窗 attention；这次没有改成计算 512 个 query |
| `hc_pre.py` / `hc_post.py` | 公共 HC 前处理和后处理；没有为 CP 复制代码 |

## 8. 常见问题

### 8.1 SWA attention 是不是算完整 512 个 query？

不是。每张卡只算自己的 128 个 query。

### 8.2 为什么 KV 一定要算完整 512？

不是 SWA 数学上天然要求，而是当前 `kv_cache` 采用 TP group 内复制契约。每张卡都要把自己的 cache 副本更新完整，因此当前实现让每张卡计算完整 group KV。

如果未来改成 cache 按 request owner 分片，可以只算本地 KV，但那是另一种缓存架构。

### 8.3 这是 TP 还是 CP？

同一组物理 rank 承担了两种角色：

- 输入和 query token 按 rank 分片，这是 CP/SP 式语义。
- 输出 O projection 的权重/输出组按 rank 分片，这是 TP 语义。

代码没有单独创建一个 CP process group，而是借用 `TP_SIZE` 对应的 rank group 做 hidden AllGather。

### 8.4 SWA 有没有 compressor 或 indexer？

没有。SWA 在配置中是 ratio-0，只维护原始 sliding-window KV cache。因此它是三类 attention 中最简单的 CP 输入改动。

### 8.5 为什么不直接 AllGather KV？

对 SWA 单独而言，“每张卡先算本地 KV，再 AllGather KV”也是可行方案，而且通信张量更窄。当前实现选择 AllGather normalized hidden，是为了严格对齐 Issue #905 引用的 DSA CP 参考语义，并为还需要 hidden 的 compressor 路径保持统一的输入边界。

因此，本文描述的是当前实现契约，不表示“先 AllGather hidden”是 SWA 数学上唯一可行的方案。

## 9. 看代码时的推荐顺序

1. 在 `decode_swa.py` 找 `T_DYN` 和 `CP_T_DYN`，先区分本地/完整组维度。
2. 看 `decode_swa` 接口，区分 Q metadata 和 KV metadata。
3. 顺着 `hc_pre -> rms_norm -> decode_cp_token_allgather_step` 看输入通信。
4. 对比 `q_proj_rope(x_normed_t)` 和 `kv_proj_rope(x_normed_group)`。
5. 看 `swa_slot_mapping` 如何写完整 cache。
6. 看 `sparse_attn_swa` 为什么仍然只消费本地 query metadata。
7. 最后看 `decode_swa_attention` 的 TP1/TP>1 adapter。

## 10. 最终总结

这次 SWA CP 输入改动没有重写 attention 数学，也没有把所有计算都扩成 512 行。它只在 RMSNorm 后建立了一个清晰的分界：

```text
本地路径：hc_pre、RMSNorm、Q、sparse attention、hc_post
完整路径：normalized hidden AllGather、WKV、KV RoPE、KV cache write
```

所以它的核心不是“所有东西都做 CP”，而是：

> 保留 local query，补齐 replicated KV。
