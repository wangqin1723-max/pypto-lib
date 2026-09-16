# DSpark Decode HCA 泳道分析与优化方向

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/2026年9月7日-DSpark HCA泳道分析与优化方向.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

日期：2026 年 9 月 7 日。

本文记录现有泳道的离线分析结果，供下一轮优化评审。分析阶段没有修改模型代码、编译产物或原始泳道，也没有重新运行 NPU。下文的优化收益均未实测；时间区间表示当前执行中的工作或等待，不能直接当成可节省时间。

建议先检查 **raw 分支的额外依赖、raw KV gather 的标量开销，以及 O projection 的跨组依赖**，再优化 merge/publish 和短上下文 compressed attention。Qwen 和 MTP 的参考价值主要在分页搬运、任务依赖及流水组织。

**1. 分析对象与测量口径**

| 项目 | 本次配置 |
| --- | --- |
| 捕获程序 | `decode_hca_test` |
| 捕获代码版本 | `fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77` |
| 平台 | A2A3 真机 |
| TP / 物理设备 | TP4；设备 4、6、8、10 |
| Batch / sequence | 每 rank B=16，全组 B=64；S=8 |
| 本地 token 数 | T=128 |
| Attention shape | H=64，head dimension=512，sliding window=128 |
| KV cache 页大小 | 32 行 |
| Position IDs | 所有 request 均为 256..263 |
| KV sequence length | 264 |
| 每 token 有效 compressed KV | 2 行 |
| 采样方式 | 一次 chip swimlane perf level 4 捕获；不是多轮 benchmark |
| 正确性 | 原始运行记录显示精度验证通过 |

已经检查四个 rank 的 program identity、level 4 标记、依赖与名字映射，以及 AICore、AICPU 和合并后记录的完整性。每个 rank 均有 **1044 条物理执行记录**，与依赖中的 block 数及有效 kernel slot 数匹配。任务分析使用 `swimlane_converter.read_perf_data()` 合并时钟域，并在临时副本上重新运行 canonical critical-path analyzer。

捕获版本已经包含 `3fe1bbe` 的“本地投影后再 all-gather”改动，以及后续 HC pre-mix normalization、HCA merge/cache writeback 优化。本次核对的 HCA、sparse HCA、compressor 和 O projection 保存源码均与 `fa5bf7e` 完全一致。分析时主工作区位于 `3dc5edb`，因此下文的 HCA 行号以捕获源码为准。

| Rank | 物理设备 | dispatch → finish（µs） | AICore 总跨度（µs） |
| --- | ---: | ---: | ---: |
| rank0 | 4 | 14674.02 | 14669.72 |
| rank1 | 6 | 6392.08 | 6388.12 |
| rank2 | 8 | 12757.96 | 12752.70 |
| **rank3** | **10** | **871.12** | **866.16** |

按照最短 dispatch → finish 选择 rank3。该口径用于减少分布式启动时差对调优分析的影响，**不是整个 TP step 的端到端延迟**，也不能据此认定所有 rank 差异都来自启动时差。

| Rank3 关键路径指标 | 时间（µs） |
| --- | ---: |
| AICore makespan | **866.16** |
| Observed 路径计算贡献 | 679.96 |
| Observed 路径空档 | 186.20 |
| Static CPM | 591.32 |
| Observed 路径任务数 | 33 |

679.96 + 186.20 = 866.16。Static CPM 采用无限核的依赖路径模型，仅作交叉检查；591.32 µs 和“去掉所有 gap 后的 679.96 µs”都不是承诺能够达到的性能。

**除明确标为持续时间的数字外，本文时间点统一以 rank3 第一个 AICore 任务开始为 0。** 原始自动报告中的部分时间点使用不同原点，比较时需先对齐。

**2. 热点与优先级**

| 优先级 | 优化对象 | 当前证据 | 建议方向 |
| --- | --- | --- | --- |
| 第一批 | raw 分支 readiness | raw cache 比 compressed cache 早结束 34.68 µs | 分别表达两种 cache 的真实读写依赖 |
| 第一批 | `hca_gather_kv` | 59.74 µs；逐 slot 检查连续性 | 借鉴 MTP 的分页连续区间搬运，保留 S=8 复用 |
| 第一批 | O projection 跨组依赖 | A→最后 dequant 跨度 123.10 µs；存在额外 group 依赖 | 按 owner/group 隔离 buffer 或精确声明依赖 |
| 第二批 | merge/normalization/publish | 83.70 µs；AIV 执行时间分布不均 | 归一化实现、工作分配和加载/写回/发送流水 |
| 第二批 | compressed attention | 2 个有效 KV 仍采用 128 行 tile；任务跨度 95.54 µs | 小 K 路径、减少重复初始化、缩短 mixed 工作块 |
| 后续 | 投影通信与 early dispatch | pack→unpack 跨度 119.08 µs | 减少中间复制与任务边界，逐项验证调度提示 |
| 长期 | attention 内部流水融合 | raw/comp 中间 O 各 16 MiB | 参考 Qwen mixed kernel，减少完整中间流的 GM 往返 |

这些阶段相互重叠，表中的数字不能相加作为预计收益。优先级兼顾证据清晰度、改动范围和验证成本。

**3. raw 分支：先减少额外等待，再优化 gather**

当前 [decode_hca.py](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_dspark/decode_hca.py) 的第 317 行将 raw/compressed cache 写入合并成一个 `cache_ready_dep`；[decode_sparse_attn_hca.py](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py) 第 161 行让 raw gather 也等待它。

```text
raw cache write ──────────┐
                         ├─ cache_ready_dep → raw gather / compressed gather
compressed cache write ──┘
```

| 任务 | AICore 结束（µs） | AICPU FIN（µs） |
| --- | ---: | ---: |
| `hca_cache_writeback` | 299.48 | 307.84 |
| `rmsnorm_rope_cache_write` | 334.16 | 334.68 |

raw gather 在 345.32 µs 才开始。它读取原始 KV cache 和窗口元数据，本身不需要 compressed cache 内容。因此值得分别传递 raw-cache-ready 和 compressed-cache-ready，并在最终 merge 汇合两条计算分支。

34.68 µs 是两个 cache producer 的结束时间差。解开依赖后，raw gather 可能与 compressor 后处理重叠；但 AIV/AIC 资源竞争、其他真实依赖和调度开销都会影响实际收益，不能直接宣称节省 34.68 µs。

raw gather 自身持续 **59.74 µs**，16 个 AIV 的单条记录耗时为 45.54..59.60 µs。当前实现每搬运一个 16 行 band，都读取并检查该 band 的全部 16 个 slot；128 行窗口需要检查全部位置，随后另行追加 7 个推测 token 的 KV。

[MTP CSA](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py) 第 221..247 行根据页内偏移计算连续区间，并用 `valid_shape` 完成多行 `gather_row`。HCA 可以借鉴这种地址计算方式，减少逐 slot 的标量读取与分支。

本例每个 request 的共享窗口是 `128 + 8 - 1 = 135` 行，按当前起始位置覆盖 5 个 32 行逻辑页。但物理页不一定相邻，不能把本例中连续的物理地址当成通用假设。实现需要保留页映射、环形边界、无效 slot 和尾部处理。

**保留现有的 request 级 gather 和 S=8 复用。** MTP 的逐 token gather 主循环不能完整照搬，否则会重新引入 HCA 已经消除的重复搬运。

**4. O projection：跨 owner 已隔离，跨 group 仍有额外依赖**

从第一批 `tp_o_a` 开始，到最后一批 `tp_o_b_dequant` 结束，时间为 **633.62..756.72 µs，跨度 123.10 µs**。之后还有 35.76 µs 的 `tp_o_b_publish`。

当前 [decode_o_proj.py](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_dspark/decode_o_proj.py) 第 493 行起已经按 owner 分配独立中间 buffer；但是同一个 owner 的两个 local group 仍共享这些张量。

`deps.json` 中有如下具体证据：

| 任务 ID | 任务 | AICore 时间（µs） | 依赖现象 |
| --- | --- | --- | --- |
| `12884901911` | `tp_o_a` | 634.92..652.82 | 一个 group 的 A 投影已经结束 |
| `12884901908` | `tp_o_a` | 665.08..687.22 | 另一个 group 的 A 投影更晚结束 |
| `12884901912` | `tp_o_a_quant` | 701.82..707.38 | 同时依赖上述两个 A 任务 |

对应量化需要等待比自身 group 更晚完成的 sibling。`tp_o_b` 也存在类似跨 group 前驱。

[MTP CSA](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py) 第 389..463 行采用每组显式 `A → quant → B` 链。可以在保留 HCA owner 并行的基础上进一步使用 group 独立 buffer，或经过读写审查的显式依赖，减少张量级依赖带来的额外串行化。

两个 A 任务相差的 34.40 µs 只是候选重叠区间。调整后 B 可能与其他 A 竞争 Cube，因此应以整个 HCA wall time 判断效果。每组 amax、量化 scale、分组反量化和最终求和语义必须保留。

**5. merge/publish：继续优化内部工作，保留已完成的连续写回**

`hca_stream_merge_pack_publish` 位于 **511.92..595.62 µs**，跨度 **83.70 µs**。48 个 AIV 的单条记录耗时为 41.68..83.48 µs，中位数为 68.38 µs。

本例共有 `(128 / 4) × (64 / 16) = 128` 个工作块，分给 48 个 cyclic workers。每个 worker 执行 2 或 3 块，即 8 或 12 个 token/head-tile 单元，存在工作量差异。joined records 没有逻辑 block ID，因此不能仅凭物理 core 编号进一步断言哪个 worker 或发送目标导致了某条长尾。

三个具体候选如下：

- **归一化实现。** 捕获源码第 457 行仍使用 `row_expand_div`，生成代码明确包含 `TROWEXPANDDIV`。[Qwen PyPTO paged attention](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/qwen3_14b/paged_attention_pypto.py) 最后输出使用 `row_expand_mul(o, recip(l_sum))`。可作为范围较小的实验，但浮点舍入可能变化，需要验证精度。
- **工作分配。** 优化 token/head 的任务分配，减少 2 块与 3 块带来的负载差异。缩小 token tile 可以改善均衡，同时会增加 put 次数，不能只看 AIV 利用率。
- **加载、计算、写回和发送的重叠。** 当前两条 FP32 O 中间流各为 16 MiB，merge 读取两条流并产生 8 MiB BF16 输出。生成的 TPUT 路径仍从 GM staging 重新加载，且带有 `PIPE_ALL` 和缓存发布同步；融合成一个 DSL 任务没有消除所有中间搬运。

这版已经包含按 group 连续 store 和 4 行 put，不能再次把“合并零碎 store”列为尚未实施的新优化。进一步融合需要检查 UB 容量和活跃 buffer；chip 泳道无法分解这 83.70 µs 中除法、访存、同步和发送各自占比。

逆 RoPE 的 gather 仍需按完整父 tile 的物理 stride 计算索引，不能把有 stride 的列切片直接当成连续 `[16,64]` 张量做 flat gather。

**6. compressed attention：小 K 与 mixed 核资源分配**

保存的输入确认：每个 token 只有 **2 行有效 compressed KV**；`cmp_block_table` 为 `[4,16,1]`。但当前 `hca_cmp_qk_pv` 仍使用物理 `ATTN_K_TILE=128`，QK 后才应用有效范围，并且先初始化完整 partial O，再覆盖有效输出。

compressed mixed 任务位于 **357.66..453.20 µs**，跨度 **95.54 µs**，包含 16 个 AIC 和 32 个 AIV 物理执行记录。它虽然没有作为一个独立节点出现在本次 Observed 路径中，仍影响 raw 分支的启动与完成。

| raw AIC 分组 | 开始时间（µs） | 观察 |
| --- | --- | --- |
| 前 8 个 AIC | 413.92..414.12 | 对应 core 的先前计算早已结束 |
| 后 12 个 AIC | 444.48..453.36 | 在同核 compressed-attention 任务结束后开始 |

例如 core23 的 compressed AIC 在 452.78 µs 结束，raw AIC 在 453.36 µs 开始、504.64 µs 结束。421.8..440 µs 期间 24 个 AIC 均在执行任务。raw 任务总跨度为 **90.72 µs**，单条 AIC 执行时间只有 **49.58..62.28 µs**，最早与最晚 AIC 开始相差 **39.44 µs**。

这些记录支持“后 12 个 raw 物理任务存在同核启动竞争”。它们不支持“compressed 任务阻止了整个 raw task 的最早 dispatch/start”，两种说法应区分。

值得研究的候选包括：

- 对短 compressed context 选择更小且满足硬件对齐的物理 K tile，例如把 32 行作为待验证候选。只调整 `valid_shape` 不能证明 QK/PV 的实际计算量缩小。
- 有效路径直接写计算结果，避免完整 partial O 先清零再覆盖；无效路径仍需要合法的中性 softmax 状态。
- 缩短工作块并平衡 raw/comp mixed 核占用。MTP 的 24-worker 工作序列与 head-loop pipeline 可以参考，但必须保留 HCA 已有的跨 S=8 compressed KV 复用。

本例还有一个与压缩边界相关的机会。所有位置是 256..263，没有跨越 128-token 边界，`cmp_slot_mapping` 全为 `-1`。因此 `scatter_softmax_pool` 的 39.06 µs 在本例主要对应 state 更新，而不是边界上的 128 行 softmax pooling；之后的 `rmsnorm_rope_cache_write` 仍计算了 **13.22 µs**。

可以研究“本批没有任何压缩边界”时跳过 normalization/cache emission 的路径，**但每个 token 的 compressor state 更新仍必须完成**。分支必须由真实 request 元数据决定，不能硬编码 context256；混合 batch 和跨越 127、255 等边界的情况仍需完整处理。

**7. Qwen 的长期参考方向与通信优先级**

当前仓库没有 Qwen CSA 文件，本次同时参考了 [Qwen PyPTO paged attention](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/qwen3_14b/paged_attention_pypto.py) 和 [Qwen CCE paged attention](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/qwen3_14b/paged_attention_cce.py)。

PyPTO 版本使用持久 24-worker mixed kernel、分页加载、显式 AIC/AIV 事件、3 个 transfer slots、提前 2 个 stack 的流水，以及 online-softmax 状态。CCE 版本也有 QK/PV、online-softmax 和 rescale 阶段。这些结构可以用于研究将 HCA raw/comp 累积、merge 和最终归一化放得更近，减少完整 O 中间流的 GM 往返。

Qwen 的 head dimension 是 128，且使用 grouped-query attention；本例 HCA 的 dimension 是 512，还有 shared MLA KV、S8 verification、attention sink、逆 RoPE 和 TP 输出重分布。不能直接套用 Qwen 的 tile、buffer 大小或提前执行深度。context256 的 K 工作块很少，固定提前两个 stack 可能增加流水填充/排空开销；更适合考察 token/head 维度上的重叠。融合时也要审查 raw/comp BF16 probability 的舍入和 merge 顺序。

输入投影的 pack→all-gather→unpack 位于 **149.64..268.72 µs，跨度 119.08 µs**，其中 pack 为 7.86 µs、unpack 为 21.20 µs。按多行 band 处理、融合 readback 与 unpack 都是候选，但必须保持 retained-window epoch 的完成与复用协议。119.08 µs 包含本地计算、复制、同步和 gap，且与 query projection 重叠，不应全算作 fabric 通信时间。

early-dispatch 适合放在具体关键依赖明确之后逐项验证。`hc_pre_linear_reduce`、`mix_x_rms_norm`、`kv_score_proj` 的部分物理任务，以及 payload wait 已被实际提前派发。仍然存在 gap 的事实说明，不能把全部 186.20 µs 空档都当成“缺一个 early-resolve 标记”。

**8. raw gather 前 gap 的归因修正**

canonical Observed 表将 raw gather 前的 30.76 µs 标为 `core-wait`；自动 dispatch 报告给出了 36.82 µs 的 FIN→dispatch 区间。这里需要进一步检查 dummy 前驱。

`hca_gather_kv` 的直接前驱 `4294967338` 是没有 kernel 执行记录的 dummy，它递归依赖两种 cache write。只统计直接且有时间戳的前驱，会看到 raw cache 在 299.48 µs 数据就绪、307.84 µs FIN，却漏掉 dummy 后面的 compressed cache。

| 事件 | 相对时间（µs） |
| --- | ---: |
| raw cache FIN | 307.84 |
| dummy 的最后一个有计时祖先 FIN，即 compressed cache FIN | 334.68 |
| raw gather dispatch | 344.66 |
| raw gather start | 345.32 |

最后一个有计时祖先 FIN 到 dispatch 为 **9.98 µs**。这个区间仍包含没有时间记录的 dummy 处理和调度，不能全部归为调度器空转。本文保留 canonical 路径与 gap 标记以便复核，但该行的原因解释以上述递归依赖分析为准。

其他 gap 也需区分 producer end→FIN、FIN→dispatch 和 dispatch→start。对于已经提前 dispatch 的任务，后续等待不再是“前驱阻止 dispatch”。对于资源竞争，只有兼容引擎的 descriptor 容量在目标窗口内确实饱和，才能命名为 dispatch resource blocker；原始报告没有证明这种 task-global blocker。

**9. 完整 Observed 关键路径**

“任务跨度”可与其他任务重叠；只有“路径计算贡献 + 前置 gap”能相加复原 866.16 µs。🐌 表示 gap 严格大于 1 µs；⭐ 表示依赖结构允许且时间戳证明实际提前 dispatch，不是仅有 `allow_early_resolve` 标记。首行 gap 不计。

| # | 任务 | Task ID | 任务跨度（µs） | 路径计算贡献（µs） | 前置 gap（µs） | gap 类型 | 标记 |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 0 | hc_pre_linear | `4294967304` | 23.74 | 23.74 | — | — | |
| 1 | hc_pre_linear_reduce | `4294967306` | 2.40 | 2.40 | 9.52 | data-wait | 🐌 ⭐ |
| 2 | split_pre_post | `4294967307` | 5.38 | 5.38 | 8.58 | data-wait | 🐌 |
| 3 | mix_x_rms_norm | `4294967310` | 18.62 | 18.62 | 5.00 | data-wait | 🐌 ⭐ |
| 4 | kv_score_proj | `4294967324` | 44.62 | 44.62 | 5.76 | data-wait | 🐌 ⭐ 24/32 |
| 5 | kv_proj_matmul | `12884901897` | 25.64 | 5.90 | 0.00 | — | |
| 6 | kv_rms_norm_rope | `12884901898` | 7.94 | 7.94 | 5.50 | data-wait | 🐌 |
| 7 | hca_projection_pack | `4294967326` | 7.86 | 7.86 | 6.68 | data-wait | 🐌 |
| 8 | cp_hca_projection_allgather_push | `4294967328` | 28.02 | 28.02 | 7.86 | data-wait | 🐌 |
| 9 | cp_hca_projection_allgather_payload_wait | `4294967329` | 1.22 | 1.22 | 3.72 | data-wait | 🐌 ⭐ |
| 10 | cp_hca_projection_allgather_readback | `4294967330` | 16.62 | 16.62 | 9.36 | data-wait | 🐌 |
| 11 | cp_hca_projection_allgather_readback_wait | `4294967331` | 1.60 | 1.60 | 8.54 | data-wait | 🐌 |
| 12 | cp_hca_projection_allgather_retire | `4294967332` | 1.90 | 1.90 | 1.76 | data-wait | 🐌 |
| 13 | hca_projection_unpack | `4294967333` | 21.20 | 21.20 | 9.42 | data-wait | 🐌 |
| 14 | scatter_softmax_pool | `4294967336` | 39.06 | 39.06 | 6.78 | data-wait | 🐌 |
| 15 | hca_gather_kv | `12884901900` | 59.74 | 59.74 | 30.76 | core-wait，见第 8 部分修正 | 🐌；early 不可验证 |
| 16 | hca_raw_attn_aic | `12884901903` | 90.72 | 90.72 | 8.86 | data-wait | 🐌 |
| 17 | hca_stream_merge_pack_publish | `8589934599` | 83.70 | 83.70 | 7.28 | data-wait | 🐌 |
| 18 | o_group_a2a_wait | `8589934600` | 1.38 | 1.38 | 2.56 | data-wait | 🐌 |
| 19 | o_group_a2a_gather | `8589934601` | 15.20 | 15.20 | 5.08 | data-wait | 🐌 |
| 20 | o_group_a2a_complete | `8589934602` | 5.40 | 5.40 | 4.84 | data-wait | 🐌 |
| 21 | tp_o_a | `12884901927` | 18.52 | 18.52 | 4.34 | data-wait | 🐌 |
| 22 | tp_o_a | `12884901919` | 17.76 | 14.94 | 0.00 | — | |
| 23 | tp_o_a | `12884901908` | 22.14 | 19.34 | 0.00 | — | |
| 24 | tp_o_b | `12884901934` | 15.48 | 15.48 | 3.42 | core-wait | 🐌 |
| 25 | tp_o_b | `12884901921` | 21.14 | 21.14 | 0.20 | core-wait | |
| 26 | tp_o_b | `12884901910` | 16.68 | 11.10 | 0.00 | — | |
| 27 | tp_o_b_dequant | `12884901914` | 12.26 | 12.26 | 5.90 | data-wait | 🐌 |
| 28 | tp_o_b_publish | `8589934603` | 35.76 | 35.76 | 5.54 | data-wait | 🐌 |
| 29 | tp_o_rs_wait | `8589934604` | 4.64 | 4.64 | 3.26 | data-wait | 🐌 |
| 30 | tp_o_rs_reduce | `8589934605` | 11.18 | 11.18 | 5.22 | data-wait | 🐌 |
| 31 | tp_o_rs_complete | `8589934606` | 11.64 | 11.64 | 5.14 | data-wait | 🐌 |
| 32 | hc_post | `8589934607` | 21.74 | 21.74 | 5.32 | data-wait | 🐌 |

**10. 后续实验的验证口径**

每次只改变一个可以解释的方向，先使用相同 B16/S8/TP4/context256 fixture 比较，再覆盖混合起始位置、压缩边界与更长 compressed context。验证需要包含输出、raw/compressed cache 和 compressor state，不能只看 attention 输出。

wall-time 结论采用关闭 profiling 后、相同 rounds/warmup、冻结 golden 的最快 rank mean。新的泳道用于解释变化，不替代 benchmark。只减少 core busy time 而没有减少完整 HCA wall time，不能认定为性能收益。

本次只有一个 profiled capture。AICore makespan 不包含 host/orchestrator 前段和 AICPU/host 尾段；level-4 instrumentation 本身也有观测开销。对不同优化的收益只能在相同基线上逐项测量，不能累加本文列出的重叠等待区间。

**11. 本地证据入口**

- 原始捕获配置与各 rank 汇总（本地证据：`build_output/hca_main_ctx256_cli_20260907_2032/summary.json`；未随文归档）
- 原始 rank3 chip swimlane（本地证据：`build_output/hca_main_ctx256_cli_20260907_2032/dfx_outputs/rank3/d0/chip_swimlane_records.json`；未随文归档）
- rank3 merged swimlane（本地证据：`build_output/hca_main_ctx256_cli_20260907_2032/dfx_outputs/rank3/d0/merged_swimlane_20260907_203454.json`；未随文归档）
- rank3 任务依赖（本地证据：`build_output/hca_main_ctx256_cli_20260907_2032/dfx_outputs/rank3/d0/deps.json`；未随文归档）
- 原始关键路径、全部 gap 拆解与前驱阻塞表（本地证据：`build_output/hca_main_ctx256_cli_20260907_2032/critical_path_summary.md`；未随文归档）
- canonical rank3 critical-path 报告（本地证据：`build_output/hca_main_ctx256_cli_20260907_2032/dfx_outputs/rank3/d0/critical_path_report.md`；未随文归档）
- [捕获版本的 HCA 调用与 merge/publish](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_dspark/decode_hca.py)
- [捕获版本的 raw/compressed attention](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)
- [捕获版本的 ratio128 compressor](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_dspark/decode_compressor_ratio128.py)
- [捕获版本的 O projection](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_dspark/decode_o_proj.py)
- [MTP CSA 参考实现](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/deepseek_v4_flash_mtp/decode_sparse_attn_csa.py)
- [Qwen PyPTO paged attention 参考实现](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/qwen3_14b/paged_attention_pypto.py)
- [Qwen CCE paged attention 入口](https://github.com/hw-native-sys/pypto-lib/blob/fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77/models/qwen3_14b/paged_attention_cce.py)

引用的 `build_output` 内容属于本地生成产物；清理相应目录后，这些证据链接将失效。本文已保留核心配置、关键时间点、任务 ID、完整路径和归因修正，便于独立阅读。


**12. O projection 调优实测补记（2026-09-07）**

**实测结论：尚未获得可复现的完整 HCA 性能提升，实验模型改动已全部撤回。** 第 4 部分关于跨 group 额外依赖的观察仍然成立，但消除这些依赖、扩大 tile 或减少 block 数，在本次配置下都没有形成可接受的提速结果。以下补记更新前文“待尝试”的状态，不把单次泳道缩短或某一组较低均值当作优化收益。

**12.1 工作区、配置与验证范围**

实验在独立工作区 `pypto-lib-o-proj-tuning-20260907`、分支 `perf/dspark-hca-o-proj-group-deps` 中进行，基于原泳道对应的 `fa5bf7eb89e407c2cb1fcf61fe3cf02644db5b77`。实验没有修改主工作区的模型实现。

- 平台为真实 A2A3，TP4，物理设备 3、5、7、9；每 rank 为 B16/S8，即 128 个 local tokens，所有 request 的 start position 为 256。
- 复用原始通过精度验证的 `_jit_l3_decode_hca_20260907_203428/data` 冻结输入及 golden，每个版本均确认输入、输出 cache hit。
- 工具链保持 PyPTO `f1bb0860`、simpler `15f5cbd9`、PTOAS `0.57`、PTO ISA `96ba706c`、CANN `9.0.0`，核对了 pin 关系。
- 性能计时关闭 chip profiling；每个窗口计时 100 轮，另丢弃 5 轮 warmup。每个窗口开始前恢复冻结的 InOut 状态，首个 warmup 校验 `compress_state`、`kv_cache`、`cmp_kv` 和 `x_out`。
- 性能指标为完整 HCA 的**最快 rank 的 `effective_us` 算术均值**，不是所有 TP rank 的端到端 step latency，也不是 host wall time。泳道只用于解释变化。

依赖版本另通过了 TP4、B3/S8、start positions 为 `127,256,7` 的新 golden 校验，覆盖不足 128 行的 cube tile、量化补零、混合上下文及压缩边界。该用例因输入形状改变，单独生成并保存参考结果。本轮没有覆盖 TP1、TP2 或所有其他形状。

**12.2 第一阶段：依赖与 tile 单轴实验**

| 候选 | 实际改动 | 验证与结论 |
| --- | --- | --- |
| `group_deps` | `own_a_fp32`、`own_a_i8` 设置 `manual_dep=True`；quant 显式依赖本组 A，B 显式依赖本组 quant | 精度通过；泳道确认跨组多余前驱已消除，但未确认稳定提速 |
| `b_k512` | `O_B_K_TILE` 从 256 增至 512 | 精度通过；每个 N fragment 的 DSL K 分片从 4 次减为 2 次，未确认稳定提速 |
| `a_n256` | `O_A_N_TILE` 从 128 增至 256 | 精度通过；A 的物理 block 从 64 减至 32，单 block 变慢，未确认稳定提速 |
| `a_k512` | `O_A_K_TILE` 从 256 增至 512 | 精度通过；生成 PTO 的 Mat 地址范围达到 512 KiB，未确认稳定提速 |
| `b_d1024` | `O_B_D_TILE` 从 512 增至 1024，每个 block 处理更多输出列 | 精度通过；B 的物理 block 从 64 减至 32，单 block 耗时接近翻倍，未确认稳定提速 |

依赖修改没有取消真正需要的 group 汇总：dequant 仍自动等待该 owner 的两个 B group 输出，量化 scale 的就绪由 quant→B→dequant 链保证。原版一半 quant 任务额外等待 sibling A，一半 B 任务额外等待 sibling quant；修改后每个 quant/B 都只保留本组的一个计算前驱，另外保留必要的 allocation 依赖。

每个版本编译一次，再独立测量三组 100 轮。下面每个数均为该窗口最快 rank 的均值，单位 µs：

| 版本 | 窗口 1 | 窗口 2 | 窗口 3 |
| --- | ---: | ---: | ---: |
| 原版 | 1033.57 | 912.18 | 1516.78 |
| `group_deps` | 947.12 | 916.63 | 889.09 |
| `b_k512` | 917.34 | 1613.98 | 961.71 |
| `a_n256` | 933.25 | 1249.72 | 977.78 |
| `a_k512` | 901.89 | 1177.53 | 1342.57 |
| `b_d1024` | 912.80 | 922.88 | 1030.17 |

原版自身波动达到 912.18–1516.78 µs，不能用表中较低的一组或三组均值的中位数之比直接宣称提速。后续因此改用同一组常驻 worker 做交替对照。

独立 level-4 捕获提供了“减少 block 不保证提速”的具体证据。以下是各捕获按完整程序 elapsed 选出的最快 rank 上的单次记录；不同捕获所选 rank 可以不同，因此仅用于解释，不能替代反复测量的 wall time：

| 版本 | 所选 rank | A/B 物理 block 数 | A 单 block 中位数（µs） | B 单 block 中位数（µs） | A 开始至最后 dequant 结束（µs） |
| --- | --- | --- | ---: | ---: | ---: |
| 原版 | rank3 | 64 / 64 | 14.69 | 11.19 | 123.78 |
| `a_n256` | rank3 | 32 / 64 | 37.84 | 11.41 | 121.48 |
| `b_d1024` | rank2 | 64 / 32 | 14.86 | 21.42 | 134.34 |

A 扩 N 后，其自身跨度从 57.38 增至 70.80 µs；B 扩每 block 输出范围后，B 自身跨度虽从 56.64 降至 48.36 µs，O 段整体并未缩短。只看 B 的局部跨度会得到错误的收益判断。

这些构建未输出 `memory_after_AllocateMemoryAddr.txt`。资源记录采用生成 PTO 中显式地址加 tile 大小的最大范围：原版 A 为 Mat 256 KiB / Acc 64 KiB，A N256 为 384 / 128 KiB，A K512 为 512 / 64 KiB；原版 B、B K512、B D1024 均为 Mat 192 KiB / Acc 128 KiB。B K512 的实际代码复用初始与剩余 K 分片的 Mat 地址，没有保留原版相同的重复 L1 流水，不能仅按“双缓冲容量预估”推断加速。

**12.3 第二阶段：常驻 worker、交替顺序与 IO 对照**

为减少重建 worker 的差异，将原版和依赖版注册到同一个 persistent multi-program worker，共用设备 buffer，并检查整个实验的 worker PID 不变。第一轮按“原版、依赖、依赖、原版”重复三次；第二轮加入提前调度候选，按“原版、依赖、提前调度、提前调度、依赖、原版”重复三次。

每个候选在每种 IO 口径下有 6 个计时窗口，共 **600 个有效样本/rank**；每窗口仍另有 5 轮 warmup。下面先合并某版本同一 rank 的 600 个样本求算术均值，再选最快 rank。没有使用“每一轮选最快卡再求平均”、trimmed mean 或样本中位数替代规定指标。

两种 IO 口径必须分开：

- `original_io`：保留原模型 harness 的 resident weight/state 与 host IO 放置方式。
- `all_resident`：进一步把每轮的**输入和输出都驻留设备**，把对应传输移出计时循环，用于观察完整 HCA kernel 的差异。这不是修改正常模型入口后得到的端到端收益，不能跨 IO 口径计算加速比。

| IO 口径 | 候选 | 第一轮均值（µs） | 第二轮均值（µs） |
| --- | --- | ---: | ---: |
| `original_io` | 原版 | 1192.36 | — |
| `original_io` | 仅收紧跨组依赖 | 1734.46 | — |
| `all_resident` | 原版 | 877.24 | 876.24 |
| `all_resident` | 仅收紧跨组依赖 | 871.75 | 892.47 |
| `all_resident` | 收紧依赖并加提前调度标记 | — | 898.86 |

`original_io` 在固定 worker 后仍有明显长尾；按要求的算术均值，该次依赖版本反而更慢。`all_resident` 的原版两轮结果相近，但依赖版第一轮约 **0.63%** 的表面收益没有复现。第一轮三个对称顺序组的依赖版相对变化分别为约 **提速 1.93%、变慢 1.30%、提速 1.22%**，方向也不一致。因此不能接受“跨组依赖修正已提速”的结论。

**12.4 提前调度实验、失败记录与结论边界**

在依赖版本之上，分别给 `tp_o_a` 和 `tp_o_a_quant` 的 producer 提交位置增加字面量 `allow_early_resolve=True`，目标是让 quant 和 B 提前进入调度队列。修改前对 rank3 的两类任务各 8 个 owner/group occurrence 单独核对：均未实际提前派发，各自只有本组的一个未标记计算前驱及 allocation 依赖；quant 的 producer-end→start gap 为 5.12–12.76 µs，B 为 6.86–15.48 µs。生成代码中确认新增了两个预期的 producer 标记。

三个版本均通过普通完整 HCA golden 校验，随后 18 个不带 chip profiling 的计时窗口全部完成，各窗口首个 warmup 的四项输出均通过。之后依赖版本的 level-4 捕获也通过；但提前调度版本在 **run 1/2 的 dependency-generation pass** 中失败，尚未进入 clean timing pass。日志报告 `finalize_native_run failed with code -100`，随后出现 mailbox 清理超时和 `release_domain` 失败，队列最终记录 exit 139。

该失败捕获只有 dispatch marker，没有完整泳道，故**无法验证提前调度版是否实际提前派发，也无法比较其 after gap**。producer 标记本身不构成实际 early-dispatch 证据。失败前已完成的计时记录保留供复核，但不能把整个提前调度实验写为成功。

该运行时问题已记录到主工作区 KNOWN_PYPTO_ISSUES.md（本地证据：`KNOWN_PYPTO_ISSUES.md`；未随文归档），标题为 `HCA early-resolve candidate fails during dep-gen on a reused multi-program worker`。尚未得到最小复现，也未确定 early resolve、dep-gen 与多程序常驻状态之间的具体故障关系，没有修改 runtime 来绕过它。

另一次设备 0、2、4、6 上的尝试，在未改动原版的普通校验阶段遇到 device6 HDC disconnect，未进入成对计时。这轮作为设备/基础设施无效尝试剔除，不计作候选的数值失败或性能结果。

**12.5 最终处理与证据入口**

截至本次补记，独立工作区的模型源码已恢复到 `fa5bf7e` 基线，依赖与提前调度改动均已撤回；工作区、实验源码、补丁、原始样本及日志保留，没有创建 commit 或 PR。前文 O projection 的方向应更新为“已尝试，当前 B16/S8/TP4/context256 配置未取得可接受收益”，不能继续描述为已验证有效的优化，也不能外推为所有形状都没有优化空间。

- 首轮依赖与 tile 实验记录（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_sweep_20260907/REPORT.md`；未随文归档）
- 首轮原始计时样本（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_sweep_20260907/results.json`；未随文归档）、泳道统计（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_sweep_20260907/swimlane_metrics.json`；未随文归档）
- A K512 / B D1024 原始计时样本（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_round2_20260907/results.json`；未随文归档）
- 常驻 worker 复测报告（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_paired_20260907/REPORT.md`；未随文归档）、原始样本（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_paired_20260907/results.json`；未随文归档）、合并统计（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_paired_20260907/summary.json`；未随文归档）
- 提前调度对照原始样本（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_early_20260907/results.json`；未随文归档）、阶段完成与失败状态（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_early_20260907/completion.json`；未随文归档）、失败日志（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_early_20260907/paired.log`；未随文归档）
- 依赖实验补丁（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_sweep_20260907/group_deps.patch`；未随文归档）、含提前调度标记的完整实验补丁（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_early_20260907/early.patch`；未随文归档）
- 混合 start position / 小 batch 校验日志（本地证据：`pypto-lib-o-proj-tuning-20260907/build_output/o_proj_sweep_20260907/validate_tail.log`；未随文归档）

相关队列任务：首轮 `task_20260907_212638_37849673390`，补充 tile 实验 `task_20260907_213301_401800711391`，常驻 worker 复测 `task_20260907_220335_102065519451`，提前调度前的 level-4 捕获 `task_20260907_220728_117511029017`，提前调度对照与最后失败捕获 `task_20260907_221011_128382322243`。链接均指向本地生成产物，清理工作区或 build_output 后会失效；核心配置、数值和结论已在本节保留。


**13. raw / compressed cache 就绪依赖拆分实测（2026-09-08）**

**结论：依赖拆分正确且已生效，但没有获得可复现的完整 HCA 性能提升，本轮不提性能 PR。** 本轮只验证第 3 节的就绪依赖拆分，没有调整 tile、计算逻辑、gather 搬运方式或 early-resolve 标记。

实验工作区为 `pypto-lib-hca-cache-ready-20260908`，分支 `perf/dspark-hca-split-cache-ready`，基于当时 upstream main `7ceff7d9`。该版本相对原泳道基线 `fa5bf7e` 只新增 MoE 改动，HCA 文件和 harness 相同。独立工作区的两个模型文件已恢复基线，补丁、候选源码、完整泳道和原始样本均保留。

**13.1 改动与正确性**

- 删除分布式 HCA 中合并两种 cache 写入的 dummy，将 `ori_cache_write_tid`、`cmp_cache_write_tid` 分别传给 sparse attention。
- raw gather 只等待原始 KV writer；compressed gather 等待 compressed KV writer；最终 merge 保留两条 attention 分支的完成依赖。
- TP1 仍把原有合并依赖传给两个参数，保持原来的执行约束。
- 46 个生成 PTO kernel 在去掉源码位置及 inline 展开编号后完全一致，变化限于 orchestration 的依赖。
- 两版普通完整 HCA 校验、24 个计时窗口首个 warmup、4 次 level-4 捕获均通过四项输出校验。
- TP4、B3/S8、start positions `127,256,7` 复用已有冻结 golden 校验通过，覆盖压缩边界、短窗口和不足完整 tile 的 token 数。
- TP1 调用兼容性编译通过；鉴于已有 TP1 runtime 问题，本轮没有声称 TP1 真机精度或性能通过。没有遇到新的编译器或 runtime 问题。

**13.2 交替计时**

真实 A2A3，分配设备 `1,3,5,7`，TP4，每 rank B16/S8、localT128，总 B64，所有请求 start position 为 256。工具链继续使用已核对 pin 的 PyPTO `f1bb0860`、simpler `15f5cbd9`、PTOAS `0.57`、PTO ISA `96ba706c`、CANN `9.0.0`。

两版各编译一次，复用原 context256 冻结输入和 golden，确认 39 个输入、4 个输出及两类 cache hit。两版注册到同一组常驻 worker，共用设备 buffer，PID 保持不变。每种 IO 口径按 A B B A 重复三组，每窗口 100 个有效样本、5 个 warmup；每版每 rank 每种 IO 口径合计 **600 个样本**。每个窗口前恢复 InOut 状态，首个 warmup 校验四项输出；没有剔除任何样本或窗口。

下表先合并同一 rank 的 600 个样本求算术均值，再取最快 rank；计时关闭 chip profiling、PMU 和 dep-gen。它是完整 HCA 的调优指标，不是 TP step latency 或 host wall time。正值表示变慢。

| IO 口径 | 原版（µs） | 拆分依赖（µs） | 变化 |
| --- | ---: | ---: | ---: |
| 正常模型 resident weight/state 与 host IO | 1381.09 | 1369.57 | -0.83% |
| 输入、输出全部驻留设备，仅作诊断 | 859.24 | 869.57 | +1.20% |

正常 IO 的三个平衡顺序组分别变化 **-7.54%、-18.01%、+25.15%**，方向不一致且波动大，不能把合并后的 -0.83% 当成已确认收益。全部驻留设备时，三个组分别变化 **+1.62%、+2.72%、-0.71%**，合并后更慢。两种 IO 口径不能相互计算加速比。

**13.3 泳道确认与结论边界**

计时结束后复用 live compiled programs，以非持久 worker 按 A B B A 顺序做 4 次 level-4 捕获；每次 dep-gen 与 clean timing 分开。16 份 rank 捕获的 AICore、AICPU、joined records 均为 1044 行，每个 task 的物理行数也符合 `block_num × active kernel_ids`。

8 份原版 rank 依赖图均显示 raw gather 的祖先包含 compressed cache writer；8 份候选图均移除了该祖先，同时保留正确的 raw writer。compressed gather 的 writer 依赖及 merge 的两分支依赖也全部保留。

以下每次捕获独立选 dispatch→finish 最短的 rank；时间以首个实际 AICore task 为零点，只用于解释调度，不替代无 profiling 计时：

| 捕获 | rank | raw cache 结束（µs） | compressed cache 结束（µs） | raw gather 开始（µs） | 完整 AICore 跨度（µs） |
| --- | --- | ---: | ---: | ---: | ---: |
| 原版 A1 | rank3 | 308.76 | 338.38 | 345.06 | 888.04 |
| 拆分 B1 | rank2 | 320.10 | 331.22 | 327.86 | 866.14 |
| 拆分 B2 | rank3 | 306.90 | 324.00 | 312.58 | 915.64 |
| 原版 A2 | rank2 | 319.06 | 329.40 | 339.98 | 918.64 |

两次所选候选捕获中，raw gather 都在 compressed cache 完成前开始，说明提前启动的机会确实释放了。但这不等于完整 HCA 获得稳定提速，不能将原文的 34.68 µs producer 时间差当成可兑现收益。本项应更新为“已尝试，当前配置不接受为性能优化”。

证据入口：完整报告（本地证据：`pypto-lib-hca-cache-ready-20260908/build_output/cache_ready/REPORT.md`；未随文归档）、原始计时（本地证据：`pypto-lib-hca-cache-ready-20260908/build_output/cache_ready/results.json`；未随文归档）、均值统计（本地证据：`pypto-lib-hca-cache-ready-20260908/build_output/cache_ready/timing_summary.json`；未随文归档）、全部 rank 的依赖与泳道核验（本地证据：`pypto-lib-hca-cache-ready-20260908/build_output/cache_ready/swimlane_metrics.json`；未随文归档）、实验补丁（本地证据：`pypto-lib-hca-cache-ready-20260908/build_output/cache_ready/change.patch`；未随文归档）、混合边界校验（本地证据：`pypto-lib-hca-cache-ready-20260908/build_output/cache_ready/boundary.log`；未随文归档）。完整 Observed 路径及各 gap 的报告由完整报告链接进入。

任务 `task_20260908_000851_6968155408` 完成交替计时及四次泳道；`task_20260908_001221_1270477992` 完成混合边界校验和 TP1 编译，均 exit 0。


**13.4 后续处理：按依赖图简洁目标保留（2026-09-08）**

用户随后明确希望依赖图更简单。因此将同一份已验证补丁恢复到独立工作区 `pypto-lib-hca-cache-ready-20260908`，作为依赖清理保留；第 13 节的性能数据及“没有确认可复现提速”结论保持不变。

本项删除分布式路径中合并 raw / compressed cache writer 的 dummy，去掉 raw gather 对 compressed writer 的无关等待，以及该 join 引入的反方向等待。每个 gather 的 cache writer 依赖、query / metadata 等真实数据依赖和最终 merge 的两分支依赖仍保留。TP1 保持原有合并依赖。

恢复后的两个源码文件与实测候选逐字节一致，复用已通过的精度、边界和依赖图验证；没有因恢复相同补丁而重复跑性能。改动留在独立工作区，尚未创建 commit 或 PR。


## 14. 删除 HCA 多余依赖并交付泳道与 PR（2026-09-08）

根据后续“去掉所有多余依赖，直接跑泳道并提 PR”的要求，本次按依赖图简洁目标完成，不将其表述为已确认的性能提升。

- 工作区：`pypto-lib-hca-cache-ready-20260908`。
- 分支：`refactor/simplify-dspark-hca-dependencies`。
- 基线已同步到 `7733dff2`，包括主分支最新的共享 QKV 修改；以下数据都来自同步后的原版与候选，不能与第 13 节旧基线直接计算加速比。
- 推送前重放到 `3e380c8c`；新增上游提交只涉及 MTP 与编码文档，全部 DSpark 模型源码和 golden harness 与已验证版本一致，无需重复实卡计时。
- 提交：`99894adbbffbdf9b1d0e3c49c91c5f12f1cabdf8`。
- PR：[#1172](https://github.com/hw-native-sys/pypto-lib/pull/1172)。

**14.1 实际删除与保留的依赖**

参考 `docs/models/deepseek_v4_flash_mtp/decode_optimization.md:471` 的做法，审计排除所有触及 allocation task 的边，并对分配保留依赖另作逐边比较。

| TP4、B16/S8、ctx256，每个 rank | 原版 | 最终版 |
| --- | ---: | ---: |
| 计算及 dummy task | 72 | 71 |
| 去重后的非分配依赖边 | 103 | 84 |
| 可通过传递路径消除的重复边 | 9 | 0 |
| allocation 保留依赖 | 保留 | 逐边一致 |

19 条净删除包含：cache-ready join 的 4 条边；O 投影跨 local group 的 8 条错误关联；通信与 `hc_post` 的 7 条传递重复边。每个 gather 保留自己的 cache writer，每个 owner/group 保留 `A -> quant -> B`，dequant 仍等待其全部 B 组。去掉上述有意取消的无关约束后，原版与最终版的完整可达关系一致。

通信链完整保留：`push -> payload_wait -> readback -> readback_wait -> retire -> unpack`；attention exchange 和 reduce-scatter 也保留 `publish -> wait -> gather/reduce -> complete`。put、notify、计数器复位与跨轮次完成协议未改变。

当前 PyPTO 将嵌套 SPMD 的 `no_dep_args` 标记丢失，已用不需要设备的最小用例复现并记入主工作区 `KNOWN_PYPTO_ISSUES.md`。最终采用三个独立 InCore helper 的显式 `pl.spmd_submit(..., pl.no_dep(...))` 提交，生成图确认删边实际生效；未修改工具链。

**14.2 最终版本验证**

TP4、B16/S8、ctx256：原版及最终版首次完整校验、8 个交替计时窗口的首次 warmup 校验、4 次 level-4 捕获均通过 `compress_state`、`kv_cache`、`cmp_kv`、`x_out`。4 次捕获顺序为 A B B A，16 份 rank 数据全部完整，每份 1040 行；AICore、AICPU、joined 行数与每个 task 的 block/subslot 数一致。

TP4、B3/S8、starts=127,256,7 全部输出通过，4 份泳道每份 694 行，非分配边 84 条，重复边 0；TP2 相同 starts 全部输出通过，2 份泳道每份 658 行，非分配边 82 条，重复边 0。额外通过 HCA all-gather `local-t=7 --raw-bits` 的两轮次位精确检查及 O-group exchange 的短行检查。TP1 仅做编译覆盖。完整模型 forward 未在本项中验证。

最终源码与实测 variant 逐字节相同，`pre-commit run --all-files` 四项全部通过。最终设备任务 `task_20260908_004718_1273753399` 完成，exit 0。

**14.3 性能仍不能作为收益申报**

同一 persistent worker、相同 buffers、正常模型 resident weight/state 与 host IO；A B B A 重复两组，每窗口 100 次计时加 5 次 warmup，每版每 rank 共 400 样本。计时关闭 profiling、PMU 和 dep-gen；每窗口恢复冻结 InOut，未丢弃样本。下表先合并同一 rank 的样本求均值，再选最快 rank；它不是 TP step latency 或 host wall time。正值为变慢。

| 平衡顺序组 | 原版 µs | 最终版 µs | 变化 |
| --- | ---: | ---: | ---: |
| 第 1 组 | 1244.36 | 1037.45 | -16.63% |
| 第 2 组 | 1301.22 | 1654.00 | +27.11% |
| 全部 400 样本/rank | 1272.79 | 1345.72 | +5.73% |

两组方向相反，合并结果更慢。本次没有确认稳定提速，PR 类型为 `Refactor`。最终两次候选泳道均按 dispatch→finish 选择 rank3，其完整 AICore 跨度为 957.06 / 943.42 µs；原版为 967.50 / 873.00 µs，不能用挑选单次泳道得出加速结论。

**14.4 交付文件**

最终第二次候选四卡完整泳道已复制到主工作区 `build_output/hca_dependency_cleanup_20260908/`：

- rank3 泳道，可在 Perfetto 打开（本地证据：`build_output/hca_dependency_cleanup_20260908/dfx_outputs/rank3/d0/merged_swimlane_20260908_004839.json`；未随文归档）
- rank3 chip_swimlane_records.json（本地证据：`build_output/hca_dependency_cleanup_20260908/dfx_outputs/rank3/d0/chip_swimlane_records.json`；未随文归档）
- rank3 deps.json（本地证据：`build_output/hca_dependency_cleanup_20260908/dfx_outputs/rank3/d0/deps.json`；未随文归档）
- 交付说明、精度与性能报告（本地证据：`build_output/hca_dependency_cleanup_20260908/README.md`；未随文归档）
- 所有 rank 的删边、分配保留及可达性审计（本地证据：`build_output/hca_dependency_cleanup_20260908/audit.json`；未随文归档）
- 边界用例泳道审计（本地证据：`build_output/hca_dependency_cleanup_20260908/boundary_audit.json`；未随文归档）
- 合并计时统计（本地证据：`build_output/hca_dependency_cleanup_20260908/timing_summary.json`；未随文归档）
- 完整实验、原版及两次候选证据（本地证据：`pypto-lib-hca-cache-ready-20260908/build_output/dependency_cleanup/rebased/REPORT.md`；未随文归档）

交付时 CI 状态：新增页面的导航登记遗漏已修复，GitHub 文档构建、pre-commit、unit-tests、A5 simulator 和 CodeRabbit 均通过；A2/A3 simulator 与设备 CI 仍运行中，尚不能声明所有 CI 完成。PR 为 open / mergeable，无未解决的 review thread。
