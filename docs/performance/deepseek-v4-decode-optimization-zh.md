# DeepSeek V4 Decode 优化

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/deepseek-v4-decode-optimization-zh.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

> 原文：[DeepSeek V4 Decode Optimization](https://www.pypto.ai/pypto-lib/debug-and-tune/deepseek-v4-decode-optimization/)

本文是一篇案例研究，而不是参考手册。它沿着
[`models/deepseek_v4_flash_mtp/`](https://github.com/hw-native-sys/pypto-lib/tree/d6f2920046df7a19c28a5079c096a53b7e600219/models/deepseek_v4_flash_mtp/) 的演进过程展开：
这是一个包含 43 层、采用 MTP 投机解码和 W8A8 量化、具有三条注意力路径和
256 个专家 MoE 的 DeepSeek V4-Flash 实现。本文从它的第一批 kernel 一直追踪到
当前状态，记录了哪些手段真正改善了性能数字、哪些没有，以及每种手段付出了什么代价。

各项机制本身记录在其他文档中：如何测量和采集数据，参见
[性能调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/performance-tuning.md)；如何选择 tile，参见
[Cube Tile 调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/cube-tile-tuning.md)；任务图和调度器，参见
[依赖与调度](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/dependency-and-scheduling.md)；阈值和舍入，参见
[精度调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/precision-tuning.md)。这些文档说明“怎么做”；本文说明
“按什么顺序做，以及可以期待什么结果”。

括号中的数字是 pypto-lib 的 Pull Request 编号，保留它们是为了让每项结论都能追溯到
相应改动及其测量结果。文中引用的每个测量值，都是该改动作者当时报告的数字。

## 工作的整体形态

整个工作分成五组，它们不能互相替代。只有前一组工作已经扎实完成，后一组才会产生收益。

| # | 它回答的问题 | 单次改动的典型收益 |
|---|---|---|
| 0. 契约、golden、逐算子精度 | 后续任何结论是否可信？ | 无——这是入场券 |
| 1. 通用手段：切分、并行、tiling、融合、混合 kernel | 每个 kernel 是否在最多的核上完成了最少的工作？ | 10–60% |
| 2. 算子专项改写 | 算法本身的形态是否不适合这款硬件？ | 对该算子提升 10–40% |
| 3. 调度 | 是否以正确的顺序下发了正确的任务？ | 2–15% |
| 4. 服务集成与 lowering | 时间是否完全花在了 kernel 之外？ | 2–8%，外加 host 侧的毫秒级时间 |

越往后，收益越小，但后面的工作无法提前开展：如果一个 kernel 仍在对权重进行 8 倍过量读取，
第 3 节的问题就看不出来；如果 golden 还无法验证一次改写，第 4 节的工作也没有意义。

---

## 0. 契约、golden 与逐算子精度

这项工作的开局阶段完全没有带来性能变化。这是正确的。下面几乎每项优化都会改变数值运算顺序，
因此如果没有可信的参考实现，它们就都不能被接受。

### 从参考模型推导算子拆分

kernel 边界不是凭空设计出来的。它们来自 HuggingFace 官方的
**DeepSeek-V4-Flash** torch 实现——checkpoint 随附的 modeling 代码就是规格；
每个 kernel 入口都对应其中一段适合作为一个整体来调度的代码。
[config.py](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/models/deepseek_v4_flash_mtp/config.py) 中的 `FLASH` 预设逐字段对应
checkpoint 的 `config.json`，因此形状或超参数方面的问题应由参考实现回答，而不是靠猜测。

沿着参考实现自身的结构进行切分，会带来三个结果：

- **每个 golden 都成为现有 torch 函数的一小段直译**，而不是重新推导。即使旁边的 kernel
  被改写五次，这也能让 golden 始终可信。
- **参考实现限定了 kernel 可以融合的范围。** 如果 torch 代码里存在某个边界，是因为两侧
  的形状或数据类型不同，那么它是真实边界；如果只是为了可读性而存在，它就是潜在的融合点。
- **checkpoint 同时也是测试数据的来源。** 独立 harness 生成的合成 tensor，会依据该
  checkpoint 真实权重的统计特征来校准——参见 compressor、indexer 和 expert 模块中的
  fixture 注释——从而确保 kernel 在实际会遇到的数据分布下接受测试。另见
  [精度调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/precision-tuning.md#8-test-with-real-weights-and-matched-data-distribution)。

### 首先冻结部署契约

`config.py` 是目录级的单例，每个 kernel 都把它作为同目录模块直接导入：batch、投机 token
数量、page size、逐层注意力调度、量化布局。它是编译期常量的来源，因此不是运行时选项——当同类
构建需要不同的投机 token 预算时，应新建一个目录，而不是增加一个开关。

编写 kernel 之前要确定：

- **Token 形状。** `DECODE_BATCH × DECODE_SEQ`——这里是 4 个请求 × 2 行
  （MTP 验证一个 draft token）= 每步 8 行 token。这个数字决定了第 1 节中的每个并行策略。
- **量化布局。** W8A8：INT8 权重配合逐输出通道的 FP32 反量化 scale；在 INT8 matmul
  处，激活值采用**逐 token 动态量化**，amax 的下限由 `INT8_AMAX_EPS` 限定，然后缩放至
  `INT8_SCALE_MAX`。不使用校准数据，也不使用静态激活 scale。
- **哪些数据保持宽类型。** 层间 hyper-connection 隐状态、compressor 状态以及所有
  反量化 scale 均保持 FP32。

量化选择会在后续一些意想不到的地方产生收益：由于对称的逐 token INT8 量化对正标量缩放
具有不变性，gate router 可以把 `inv_rms` 一直延后到量化步骤之后再处理（第 2.5 节）。
看起来像记账细节的契约，实际上决定了哪些融合是合法的。

### 保持 golden 与 kernel 的计算顺序完全一致

这是一条长期要求，不是按个案判断：

> golden 必须以**相同的方式**计算相同的内容——每一步的运算顺序和数据类型都必须一致。

浮点运算不满足结合律，并且每次窄化都会丢失信息，因此代数等价的重排仍会改变最低有效位。
如果 golden 遵循 torch 的自然顺序，而 kernel 按 tile 顺序累加，那么报告出来的不是容差，
而是噪声；一旦真正的误差出现，这些噪声就会掩盖它。具体方法见
[精度调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/precision-tuning.md#2-make-the-kernel-and-golden-implementations-identical)。

这对优化工作的直接要求，也正是本节要确立的规则：

> 当 kernel 为了性能而重排计算时，golden 也必须改写成相同顺序——而且必须先证明改变的只是
> **顺序**，不是**语义**。

最清晰的案例是无 gather 的注意力改写（第 2.1 节）。由于 BF16 PV 累加不满足结合律，
参考实现必须改写为 kernel 的物理块顺序。只有在 FP32 下先证明新旧顺序等价——在 identity、
partial、overlay 和 rotated fixture 上误差约为 1e-7——这一改动才被接受（#629）。
compressor 的加宽 pooling tile（第 2.4 节）则是在低成本方向上应用同一原则：重排预先被证明
逐 bit 完全一致，因此 golden 根本不需要改动。

---

## 1. 通用手段

本节中的所有方法都可以迁移到其他 kernel，并按实际收益从高到低排列。

### 1.1 消除串行向量归约

解码路径上的首要成本通常不是算术，而是某个归约只在一条向量 lane 上运行，其他部分全部等待。

| 改动 | 效果 |
|---|---|
| 串行 `attn_norm_rms` 归约 → 两路 partial sum 加一次最终归约 | 消除了原始瓶颈（#339） |
| 将 `qr_quant_amax` 融入 `qr_norm_apply`——每个任务同时写出一份 partial amax | 剩余 scope 从 30 µs 降至 1.7 µs（#339） |
| 按 K slice 切分 `hc_head` 的平方和，每个 slice 各写一行；消费者完成求和并内联应用 rsqrt | `inv_rms` buffer 完全消失（#822） |

两个值得直接借鉴的细节：

- **使用 `PARTIALS=2`，而不是 4 或更多。** 两路方式能保持 FP32 加法的确定性，进而确保下游
  tensor 在不同设备上都能通过验证。继续增加并行度收益很小，却会牺牲可复现性。
- **把 amax 融入已经读取数据的那一趟处理中。** partial amax 就在归一化后的同一个 tile 上
  计算；原来的 scope 本来也会从 GM 再读一次该 tile，因此结果逐 bit 一致，量化 scale 也不变。

这些手段是一系列优化的核心；该系列还扩大了 K tile 并加深了流水线（第 1.2 和 1.6 节），
最终让 projection 家族从 1868 µs 降至 545 µs（−70.8%）。

### 1.2 Tiling：在调优其他内容之前，先与 cache line 对齐

整个代码树中最大的结构性收益，来自一个窄到无法填满 cache line 的 tile。

`qr_proj` / `kv_proj` 使用的 N-tile 是 32。因此，每个权重行的读取只填充了 **512 B L2
cache line 中的 64 B，形成 8 倍过量读取**。改用 split-K（先写零，再 atomic-add）后，
N-tile 可以扩大到 256。结合 RMSNorm+RoPE 融合，projection 的 decode case 端到端从
**936 µs 降至 407 µs（−56%）**——数据来自 a2a3 swimlane 的 5 次运行中位数（#578）。

后续改动把 `QPROJ_MM_N_TILE` 提高到 1024——完整使用 cache line，不再过量读取权重——
同时把 K tile 从 512 **减小**到 128，从而获得 8 个 slice，形成非退化的 `stage=2`
流水线，并将 L0C 占用率从 25% 提高到 50%。`qproj_matmul` 从 56.3 µs 降至 36.0 µs，
**任务数量完全没有变化**（#718）。

下面是反复遇到的几面“墙”（参见
[Cube Tile 调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/cube-tile-tuning.md#model-the-three-practical-constraints)）：

| 限制 | 数值 | 在这里造成的约束 |
|---|---|---|
| L0C accumulator | `TM*TN*4 ≤ 128 KB` | `TN=512` 时需要 `TM=64`；端到端实测并未更快 |
| Vector UB | 192 KB | `hc_pre` 的融合 scope 必须把 `D_TILE` 从 512 降至 256 |
| `alloc_tile` 最小值 | 32 B | `[1,1]` 的 FP32 归约结果只有 4 B，会被**拒绝** |
| CANN template | — | `Q_PROJ_OUT_CHUNK=256` 会触发 `ACL_ERROR_RT_AICORE_TIMEOUT` |

32 B 下限值得牢记，因为每当归约收缩成标量时都会遇到它。解决方法总是相同：分配一个能够超过
下限的物理 tile，并只把实际使用的行标记为有效——例如，逐行 norm 可使用 `[8, FFN_D_TILE]`
tile，并将有效形状设为 `[1, FFN_D_TILE]`（#784）；对整个词表进行 max 扫描时，则可使用
`[8, 808]` 的网格视图（#985）。

### 1.3 提高并行度：寻找第二条轴

当只有 8 行 token、设备有 48 个向量核时，`pl.spmd(T)` 只使用了机器的六分之一。
这棵代码树里的每次并行化收益都来自同一个动作——**寻找第二条相互独立的轴，然后在两者的
乘积上展开并行**。

| 算子 | 原网格 | 新网格 | 结果 |
|---|---|---|---|
| `merge_norm` | `spmd(T=8)`，内部串行执行 4 次 head-tile 循环 | `spmd(T × head-tile) = 32` | 填满一轮 wave（#651） |
| `hc_head_reduce` | 一个任务——一个核流式处理 512 KB，另有 23 个核空闲 | token-tile × D-slice | 58.8 → 42.6 µs（#822） |
| 逆 RoPE | 每个 token 重建一次 | head 并行 + `ROPE_OUT_TILES` | AIV 占用率 67% → 89%（#525） |
| MoE dispatch / combine | 一个 `pl.at(CORE_GROUP)`——约 72 个核中只使用 2 个 | `pl.spmd(N_RANKS × N_LOCAL)` | （#705、#736） |
| `ffn_norm` | 整个 tensor | 每核一个 token | 12.6 → 6.2 µs（#784） |
| dispatch + combine scatter | 单核 | 并行 | span 1514 → 1028 µs（#473） |

**block 越多并不总是越好。** 两个经过校准的停止点：

- `ROPE_OUT_TILES=4`（256 个任务）会过度 tiling——每个任务的工作量接近下发开销，
  RoPE span 膨胀到 107 µs。默认值是 2（#525）。
- `merge_norm` 使用 32 个 block，而不是 64 个：48 个核上，64 个 block 需要两轮 wave，
  却没有增加任何在途工作（#651）。

### 1.4 合并算子：scope 边界本身就有成本

每个 scope 边界都意味着一次 GM 写、一次 GM 读和一次 dispatch。消除边界往往比调优边界内部
更有价值。

- `hc_pre`：**将五个 `pl.spmd` scope 融合为一个**，删除了四个仅用于连接这些 scope 的
  GM 中间结果（#533）。
- indexer 的 RoPE 做到**四合一**——slice matmul → cos/sin 应用 → assemble matmul →
  写出，全部位于一个 `CORE_GROUP` scope 中：约 1357 → 1094 µs（−19%，#401）。
- 在整个 decode pipeline 中删除 **18 个沿 M 轴 pad/unpad 的 scope**（#653）。cube 运算
  在 M 轴上逐行独立，因此 `valid_shape` slice 可以直接作为 A operand；封装出来的 pad 行
  只会落入从未被读取的输出 pad 行。
- `hc_head`：从五个 scope 降到三个（#822）。

### 1.5 混合 kernel：什么该融合，什么该拆分

把 matmul 与其向量 epilogue 融合，可以让 accumulator 保持在 scope 内，并去掉一次握手。
但它并非总是可行；这里总结出的规则很明确：

> **将输出 FP32 的 matmul 与其向量 epilogue 融合。不要融合 INT8（INT32 累加）的 scope。**

INT8 scope——`qr_proj+write`、`qr_hadamard+quant`、`score_accum+store`——
即使使用 `UP_DOWN`，也会触发 ptoas 的 `pto.subview valid_shape` 错误，因为 column slice
加 `row_sum` 与 row split 冲突。应保持拆分。当融合 scope 的 buffer 溢出时，增加一层
`pl.range(ROW_CHUNK)`；**不要**把 group size 减半（#371）。

反方向的操作同样重要。routed-expert 的 cube 与 vector 阶段被**解耦**：纯 cube scope
把 INT32 写到 GM，独立的 vector scope 再执行反量化、SwiGLU 和 routing-weight 乘法（#594）。
原因很明确：解耦后，cube 和 vector 可以分别针对**各自的瓶颈**（L1/Mat 与 UB）来确定 N
fragment 的大小，而不必共用一个折中值。随后将 gate 和 up matmul 拆成独立 cube 任务，
使每个任务只持有一个权重 L1 tile，为大小为 256 的 N fragment 腾出了 Mat 空间。

> 如果两个阶段共享同一个瓶颈，就融合；如果瓶颈不同，就拆分。

### 1.6 流水线与负载均衡

- `pl.pipeline(stage=2→4)` 需要足够的迭代深度：32 个 K block 可以支撑四级 ping-pong；
  而一次 retiling 后循环只剩 2 个 block，就无法做到（#339）。
- **在做流水线之前先看 PMU。** gate/up 的 K 循环受 MTE2 限制（忙碌度约 80%），
  能从流水线中获益；w2 的 K 循环受 scalar 限制（忙碌度约 97%），因此有意保持串行（#473）。
  参见[性能调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/performance-tuning.md#4-read-pmu-utilization)。
- `pl.split(UP_DOWN)` 可以消除混合区域中的 vector 长尾。在 `proj_b` 中，INT8 GEMM
  很早结束，整个区域的 wall time 取决于两条 vector lane 中哪一条分到了更多反量化 epilogue；
  拆分每个任务后，不均衡程度减半，性能提升 16%（#547）。

---

## 2. 算子专项改写

这些改写本身不能直接迁移，但其中的推理方式可以。

### 2.1 注意力：让 gather 消失

decode 收益最集中的部分，是通过四个步骤实现的。每一步都从不同角度回答“gather 的实际成本
到底是什么？”

**第 1 步——认清这是数据访问形态问题，而不是计算问题。** `gather_kv` 是纯粹的
`GM→UB→GM` 复制：受 MTE 限制，vector 和 scalar 都处于空闲状态。逐行 `GM→GM`
复制会让 **MTE2 和 MTE3 完全串行**——无论 `pl.pipeline` 的 stage 是多少，codegen
都会把 `GM→GM` ping-pong 限制在两级；实测 `MTE2_busy + MTE3_busy ≈ wall` 也验证了
这一点（#539）。

**第 2 步——先放入 UB staging 以合并访问。** 每次 scattered load 向 UB tile 填充一行
（这样 load 可以在 MTE2 上流式进行，不会产生 buffer 复用的 WAR），随后通过一次宽 store
刷出整个 block。使用原本空闲的 vector 单元把 tile 清零，从而让 `-1` 和 padding slot
继续保持为零（#539、#571）。

**第 3 步——把 gather 融入其消费者。** 首先是 bulk-zero 加 batched `rope_pack`
（−14%，#509），随后把 KV gather 通过 `gather_row` 直接融入 `qk_pv`（#615）。
与此同时，`qk_pv` 被批处理为 **M=32**：共享的 sparse-KV tile 每两个 head tile 只需从
L1→L0 提取一次，而不是每个 head tile 提取一次，实现 2 倍复用——**cube core-time
降低 52.9%，vector 降低 53.5%**（#535）。M=64 被否决，因为其 softmax tile 与同时驻留的
QK+PV L0C accumulator 会超出容量预算。

**第 4 步——彻底不做 gather。** sliding-window 路径把 ring page 和 overlay tensor
作为**直接的 GM slice**参与注意力计算，并使用预先计算好的物理顺序 bias 进行 mask——没有
gather、没有 scatter，也没有 `gather_row`。hybrid 路径对 window block 做零 gather，
只在真正离散的 block（由 indexer 选出的压缩行）上保留 `gather_row`。**SWA 提升 14.2%，
HCA 提升 18.6%**（#629）。

剪枝也沿用了同一认识——在无效 slot 上做的工作仍然是工作：

- 丢弃全部由 `-1` padding 构成的 block，将 SWA/HCA 的 sparse-K block 从 5 个减到 2 个
  （#516）。
- SWA 进一步特化为**单个** block：`PADDED_TOPK` 减半，并彻底删除跨 block 的在线
  softmax merge 循环（#630）。
- CSA 在构造 sparse index 时记录 block 级有效性，让 `qk_pv` 跳过完全无效的 block——
  但 **`merge_norm` 保持原有 merge 顺序**；曾尝试跳过 merge block，但造成了数值漂移
  （#641）。

需要了解的代价是：这些路径很难在模拟器上准确建模，应在真实设备上验证。

### 2.2 Hyper-connection：两个小算子，两种失败模式

hyper-connection stack 有 4 条 stream，它的两个算子以不同方式出现问题。`hc_pre` 每个
sublayer 混合一次——每层两次、每次 forward 共 86 次——因此它携带的每份固定开销都会反复
支付。`hc_head` 只在尾部运行一次，但在只有 8 行 token 时并行度严重不足。

- **`hc_pre`——融合。** 从五个 scope 融为一个（#533），之后又融合成单个 syncall 任务，
  并吸收 gate 阶段（#684）。
- **`hc_head`——让 matmul 成为纯 cube kernel。** 一个专用 cast scope 把 x 一次性流式
  转为 FP32，使 head projection 成为干净的 cube kernel，而不是 cube+cast 混合 kernel；
  使用先写零、再 atomic-add FP32 partial 的 split-K 来填满空闲 cube。独立测试约从
  199 µs 降至 66 µs，约 3 倍提升（#606）。
- **`hc_head`——然后再次展开并行**（#822）：reduce 在 token-tile × D-slice 上展开，
  `LINEAR_OK` 从 8 提到 16；还有一个值得借鉴的细节——在 AICPU 上通过
  `pl.create_tensor(init_value=0)` 清零 accumulator，而不是使用专门的 seed kernel。
  后者曾占用约 5 µs 的 cube 关键路径，只为清零 1 KB。延迟从 58.8 降至 42.6 µs，
  运行间波动也从 45–65 µs 收窄至 39–43 µs。
- **两者共同——确定整条 stream 的数据类型。** residual stream 以 BF16 存放在 GM 中，
  因此每个 kernel 边界都要付出 cast 成本：写出时 FP32→BF16，每条 stream 读取 residual
  时 BF16→FP32，另外 `hc_pre` 中还有专用 cast scope。让这条 stream **端到端保持 FP32**，
  删除了约 78 个 cast 和 dispatch 任务（1621 → 1543）以及约 1 MB staging，最终让
  **decode_layer 提升 7.4%**（#732）。

最后一点可以推广：**如果按 kernel 选择数据类型，每个边界都会因此纳税。** 应该为整条 stream
选择数据类型，而不是为单个算子选择。

### 2.3 MoE：两个相互独立的问题

**Expert GEMM。** profiling 显示它受 vector 限制（AIV 忙碌度约 89%），同时有 1184 个
任务给 dispatcher 造成压力——swimlane 显示 ready queue 经常为空，约有 21% 时间在空转。
解决方案是用更大的 receive tile 将任务数减半（同时缩小 inner tile 和 quant tile，使更大的
M 下 vector 工作集仍能放入 UB），并融合两个逐行 w2 scale——逐行反量化 scale 与 routing
weight——让每个 block 只计算一次组合后的 row scale，而不是每个输出 tile 都多做一次
broadcast-multiply（`exp_w2_aiv` 降低 35%）。整体提升约 13%（#445）。

**EP collective。** 这里有几处陷阱。

- 直接以 INT8 推送 dispatch payload，不要先加宽为 FP16 再窄化——receive window 和
  buffer 都能减半（#499）。
- **每次 publish 和 barrier notify 都必须使用 AtomicAdd。** `Set` 会与重排后的 notify
  发生竞态，在 world size 大于 2 时导致 wait 死锁。data phase 应使用独立的 completion
  window，不能复用 count phase 的 window（#499）。
- **一次只并行化一个任务。** 某次改动同时对 dispatch 和 combine 做 SPMD 并行，导致 8-rank
  prefill 卡住并触发 AICPU stream-sync timeout，最后不得不整体撤回。后来这项改动分五步重新
  落地；每一步都同时通过 2-rank golden case（捕获数值错误）和 8-card 真实权重 prefill
  （捕获 hang）进行验证，并在改动中列出逐步结果（#743）。**这张表是整个演进历史中最值得
  复用的产物。**
- 将握手拆出：wait task 与 push grid 分离，并用显式 `deps` 形成 fence，使 push 保持为纯粹的
  one-sided scatter。

### 2.4 Indexer 与 compressor：写入时量化，扩大 pooling 宽度

- **写 indexer cache 时完成量化。** 每个新的 compressed KV row 以 INT8 存储，并附带逐
  position 的 FP32 反量化 scale。这样 score 路径可以直接读取分页的 INT8 cache，不必每步
  都重新量化整个 KV 历史，从而删除一次每步 O(seq) 的遍历。逐行量化与 position 无关，
  因此其数值结果与旧的 score-time 量化**完全一致**（#725）。
- **加宽 pooling tile。** compressor 的 `softmax_pool` head-chunk 循环从 8 个 64 列
  tile 合并成单个 512 列 tile，使 vector op 数量和 GM load transaction 均降低 8 倍。
  这在数值上**逐 bit 一致**，其证明值得完整说明：在线 softmax 是**逐列 elementwise** 的——
  每列各自维护 running max、sum 和 output，不存在跨列交互；而四个 state region 都是连续的
  全宽 slab，因此一个宽 slice 读取的内容，恰好等于八个窄 slice 读取内容的拼接（#624）。
- 如果消费者需要转置 load 路径，就**以转置形式存储权重**，使权重通过 `b_trans` 到达
  （wall time 降低 7%，#628、#653）。

后两项体现了同一个模式：**只要能够证明一次纯 tiling 或 layout 改动逐 bit 一致，它就是成本
最低的优化**——不需要新的 golden，也没有数值风险。

### 2.5 Router：榨干一个 40 µs 的算子

gate 很小，因此需要多个改动同时作用才会有收益。下面四项改动使 in-pipeline window 的延迟
从约 40 µs 降至约 33 µs（#784）：

1. **延后逐 token 的 `inv_rms`。** 存储 `x*gamma`，延后应用 reciprocal norm——
   一次是作为 logits 的 row scale；另一次则利用对称逐 token INT8 量化对正标量的不变性，
   只在 quantization scale 上应用一次。平方和与缩放后的激活值通过**同一趟**输入扫描得到。
2. **原生保持 FP32。** matmul 前不做 BF16 往返转换。
3. **norm 每核处理一个 token**：12.6 → 6.2 µs，并使用第 1.2 节的物理 tile 技巧绕过
   32 B 下限。
4. **向量化 score gather**：把 48 次逐 expert scalar read 改为一次批量 `pl.gather`，
   配合 `set_validshape` / `fillpad`，从 9.9 降至 5.2 µs。

为匹配 A2/A3 参考实现，matmul 有意保持 FP32×FP32；只有其他 platform target 才在那里
使用 BF16。

同一种“替换算法，而不是调整 tiling”方法还有另一个案例：greedy sampling 过去会对全部
505 个 vocab chunk 排序，却只是为了获取每个 chunk 的最大值，然后再通过两次串行 scalar
scan 找到胜者。现在，它通过一个 `[8, 808]` 网格视图逐行 fold，只流式扫描**一次**，同时
携带 running maximum 以及产生该最大值的 block；block 更新时使用严格的大于比较，以保持
`torch.argmax` 的首次出现 tie 顺序。两次串行 scan 均被删除（#985）。

---

## 3. 调度

在算术方面的收益被挖掘完之后，dispatch 数量与计算图形态就成为首要因素。相关机制记录在
[依赖与调度](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/dependency-and-scheduling.md)中；下面说明它们在这里带来了
什么收益。

### 3.1 删除任务与 barrier

output projection 的 amax 从**逐行**缩小到**逐行且逐 `O_LORA` group**，每个 group 的
INT32 partial 使用自己的 group scale 进行反量化。这消除了两个 projection 阶段之间的
整行归约 **barrier**：每个 group 的量化结果一落地，它自己的第二个 matmul 就可以启动。
vector 侧工作量下降约 86%，单次调用还减少了 **160 次 AICPU dispatch（1002 → 842 个任务）**
（#620）。

这是本节中唯一一项**数值不保持中立**的改动。更窄范围的 amax 会形成更细的量化网格，因此
INT8 scale 和 projection output 都与逐行版本不同——这是以数值变化为代价换来的调度收益，
需要在全部三种 attention variant 上通过 `attn_out` golden 才能接受，不能只看任务数。
第 0 节的规则在这里同样适用：golden 随 kernel 一同改动。

按 K 而不是按 N 拆分 projection，使延迟从 26 µs 降至 10.7 µs（#749）——在哪条轴上拆分，
不仅是 tiling 决策，也是调度决策。

### 3.2 提前下发

提前下发沿整条链应用，因为一个消费者的**每个**直接生产者都必须设置相应标记。这里明确得到一条
注意事项：如果一个**共享** kernel 的其他调用方尚未 benchmark，就有意不为它设置提前 resolve
标记——改变它的 early-resolve 状态会悄悄改变尚未测量的其他路径（#915）。

### 3.3 使用 dummy edge 为同级消费者排序

这是整个代码树里最犀利的技巧。当一次 normalization 结束时，五个消费者会**同时**进入 ready
状态并竞争核资源——但只有一个位于关键路径上。将一个 `pl.system.task_dummy` 挂在生产者的
TaskId 上，再让其余四个消费者经由该 dummy，可以保留自动追踪的边不变，同时让这四个消费者
严格稍后 ready，使关键消费者最先 dispatch（#749）。

它会付出一次 dispatch hop 的代价，也需要修改源代码：生产者必须切换到
`with pl.spmd(...) as tid` 捕获形式，因为 `for ... in` 形式不会生成 TaskId。没有生产者的
独立入口使用空的 `task_dummy(deps=[])`。两种写法都记录在
[有意延迟任务](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/dependency-and-scheduling.md#deliberately-delaying-a-task)中。

### 3.4 删除冗余边，并正确锚定 wait

一次改动删除了 83 条冗余边——8 条跨 stage、50 条 dispatch-to-expert、25 条 expert-local——
同时排除了所有触及 allocation task 的边，并保留必要的传递顺序（#803）。

**wait scope 锚定的位置决定了它是在帮忙还是添乱。** 完全没有依赖的 wait，会在调度器走到它时
立即 dispatch，随后**占住一个 core group 自旋**——反过来饿死它正在等待的 scatter。
将它锚定到某个上游 read，可让它与本地 push **并行自旋**，而不是排在 push 后面（#820、#840）。
锚点必须谨慎选择：一个候选 view 无法通过编译，因为它是 cube output，其推导出的 layout 与
outlined scalar-read kernel 声明的 layout 不同。

把 notify 融入正在 push 的 block，可以从跨 rank 关键路径中再删除一次 launch——只有当这些
put 是单次操作并且会在 notify 发出前自行 drain 时，这样做才合法（#820）。

### 3.5 一项未能成立的调度改动

曾有一项融合，把每个已激活行从 expert GEMM 内部直接推送到 fabric。它确实改善了 core-time
和 drain——但**端到端 wall time 毫无变化**。更糟的是，其 row-count 握手不包含 epoch 维度：
两个 counter 会一直累加，直到一次每个 program 只执行一次、而不是每个 wave 执行一次的 clear；
因此，在多层 forward 中，较快 rank 的下一 epoch 数据可能满足较慢 rank 当前 epoch 的预期。
最终该改动被完整撤回（#975、#978）。

由此得到两条通用结论：**未转化为 wall-time 收益的 core-time 收益就不算收益**（参见
[性能调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/performance-tuning.md)）；**任何跨 wave 复用的计数器握手都需要
epoch 维度。**

---

## 4. 服务集成与 lowering

最后一组改动不再只看 kernel，而是观察完整的一次服务步骤。

### 4.1 让权重与状态常驻

静态权重 shard 只上传到对应卡一次，并跳过每次 dispatch 的 host↔device 传输。常驻集合的
构造方式是：*堆叠后的逐层权重减去 cache pool，再加上 RoPE table 和 head norm*；同时要
**排除**所有逐步变化的数据：KV 与 state cache、slot mapping、block table、position id、
sparse index、activation 和 output（#687）。

在投机路径上，同样的处理也覆盖 output 和 recurrent state：常驻 output 跳过每轮 copy-back；
已初始化的 pool 只上传一次，并跨 round 保留 device handle；recurrent tail/draft/position/length
字段放在稳定的逐请求 device slot 中，已接受 token 在 generation guard 下提交回这些位置
（#894、#895、#917）。

一个值得了解的副作用是：**常驻也消除了各 rank 的启动偏移。** 每轮重新 staging 权重，会让各
rank 在不同时间启动。

### 4.2 将 host 工作下沉到设备

一次服务 profiling 显示，在 host 上构建与 position 相关的 metadata，主 decode 路径每步
耗时 **12.3 ms**，投机路径每步耗时 **4.9 ms**。一个共享的设备侧 metadata builder 在
**现有的** rank-local decode graph 内生成相同的 slot 和 index metadata——不增加单独的
dispatch——同时 argmax sampling 也被融合进 grouped LM-head graph（#862）。

### 4.3 合并 L2 submission

一次投机步骤最初要提交三个 L2 program：main decode、verification 和 draft decode。
将 verification 融入 main decode 后减少为两个，关键 rank 的 host decode 延迟从
**45.159 ms 降至 42.980 ms**（#884）。再将 draft layer 内联到相同的逐 rank callable 后，
减少为一个（#901）——而这项改动如实地**只**宣称减少了 dispatch 数量，因为 8-card profile
虽然显示 bind 开销下降，却没有表现出实质性的稳态收益。**报告实际测量到的结果，而不是预期结果。**

### 4.4 改写 collective

两个 LM-head collective 都从各一个 `CORE_GROUP` 任务，改成由以下部分组成：SPMD push
并将 notify 融入其中、一个只负责 wait 的 scope，以及 parallel gather：

- **Combine** 在多个 block 上展开 vocab all-to-all。每次 put——包括目标为本卡的 put——
  都使用会 drain 的 put primitive；不 drain 的 remote-store 不会在融合后的 notify 发出前
  排空，因此 peer 可能 gather 仍在传输途中的 tile。
- **Dispatch** 从 pull 翻转为 push。window 加宽到每个 logit row 一个 row slot，让每张卡
  发布自己的 slot，而不是每个 K tile 都阻塞等待 `TP_SIZE - 1` 次 remote load。

**快卡延迟从 528 降至 301 µs**：combine 234 → 62 µs，dispatch 70 → 30 µs（#840）。

### 4.5 使用 L2 对抗层间驱逐

这是整个代码树中最偏系统层面的改动，也带来了最具启发性的约束。

每个 decode attention layer 在一次 forward 中都会从 HBM 流式读取它的整套权重，而且在完整
forward 中，这些流量始终是**冷访问**：两层之间的 MoE 会把 427.8 MB 数据推过 L2，因此
上一层 attention 读取的任何内容都无法留存到下一层。现在，每层使用一次 SDMA CMO warm，
覆盖该层会读取的全部权重，并按消费者 deadline 排序；它被**锚定在层入口**，此时 core 仍在忙于
上一阶段，因此 warm 可以与其重叠——实测表明，锚定得更晚反而更差。

**快 rank 的 p50 从 40132.0 降至 39287.9 µs（−2.10%）**；每个 attention block 都恢复到
独立运行时的速度，MoE 则不受影响，符合预期（#963）。

有三项硬约束，每一项都有各自的负面实验结果作为依据：

| 约束 | 证据 |
|---|---|
| **一个 scope，一个 context。** | 把 warm 拆到两个 `pl.at` scope 会让两条 SDMA stream 同时在途，使聚合吞吐量减半（285 → 153 GB/s），把 1.1% 的收益变成 0.7% 的损失。 |
| **要么 warm 全部权重，要么一个也不 warm。** | 只 warm 一个 projection 比完全不 warm **更差**：warm 在该 segment 中花费近似固定的约 20 µs，而收益随覆盖范围扩大。 |
| **warm 集合必须放得进 L2。** | 157.9 MB 和 146.9 MB 的集合能放入 192 MiB 并产生收益；268.4 MB 的集合（L2 的 1.33 倍）会驱逐自身，造成 3% 损失。 |

warm 是一种没有 destination 的 cache hint——删除这个 scope 不会改变任何值。正因如此，
它可以安全地进行激进调优。

---

## 另请参阅

- [性能调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/performance-tuning.md)——测量、采集以及 L2 / L1 / L0
  调优规则
- [Cube Tile 调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/cube-tile-tuning.md)——依据编译器的内存报告选择
  row、N 和 K tile
- [依赖与调度](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/dependency-and-scheduling.md)——边如何形成、调度器何时下发、
  提前下发以及 dummy-task 写法
- [精度调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/precision-tuning.md)——舍入模式、数据类型对齐和阈值选择
- [DeepSeek V4-Flash（MTP）](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/models/deepseek_v4_flash_mtp/index.md)——本文从头到尾追踪的模型
