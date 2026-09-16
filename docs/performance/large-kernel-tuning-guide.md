# 遇到“大 kernel”的排查与调优方法：HCA #1187 案例复盘

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/large-kernel-tuning-guide.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

记录日期：2026-09-09。

这里的“大 kernel”，指泳道图中执行时间长的 kernel，而不是源码行数多。本文记录本次 HCA 调优得到的方法、实测证据和需要避免的误判，供后续本地调优参考。

**核心教训：泳道图能定位“哪里慢”，但不能仅凭一根长条判断“为什么慢”。**

建议按这个顺序排查：

```text
固定配置和统计口径
    ↓
确认关键路径，区分启动前的空隙和 kernel 内部耗时
    ↓
检查每个 AIC/AIV 实际承担的工作
    ↓
沿中间结果追踪搬运和最终写出
    ↓
寻找可以重叠的独立工作
    ↓
检查任务划分，再调整 tile
    ↓
拆项实验解释原因，不开 profiling 的 benchmark 判断收益
```

## 一、遇到“大 kernel”时怎么处理

### 1. 先固定输入配置和性能口径

至少记录：

- 前后源码版本，以及 PyPTO、simpler、PTOAS、PTO ISA、CANN 版本。
- 平台、实际分配的卡组，以及前后是否在同一次设备预约中运行。
- tensor shape、dtype、每个 rank 和全局的请求数。
- 每次调用的新 query token 数、每个请求的上下文长度。
- profiling 是否开启，benchmark 的 warmup、rounds 和统计口径。

特别注意 attention 中的几个量：

| 配置 | 含义 | 容易混淆的地方 |
| --- | --- | --- |
| batch / request count | 请求数量 | 多卡入口中的局部数量不一定等于全局数量 |
| S / query tokens | 本次调用处理的新 query token 数 | 不等于请求已有的上下文长度 |
| start-pos / context length | 当前 query 对应的上下文位置或长度 | `256` 不代表本次新增 256 个 token |
| KV block 数 | 实际处理的 KV 分块数量 | 需要结合压缩比例、有效行数和 tile 大小计算 |

输入和 golden 尽量冻结一次、后续复用，避免前后样本不同。具体方法见[保存与回放 golden](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/run-and-validate/save-and-replay.md)。

本次 HCA 工作采用的最终 benchmark 口径是：**关闭 profiling，取各 rank 的 `effective_us` 中位数，再取其中最小值**。这个指标用于本次调优比较，不能当作等待所有 rank 完成的全局 step 延迟，也不能和 CI 使用的均值直接混比。

### 2. 区分“核外等”和“核内慢”，确认是否影响关键路径

先在 chip 泳道图中看清楚：

1. 上游 producer 什么时候完成，目标任务什么时候 ready。
2. 任务何时 dispatch、何时被核拾取、何时开始和结束。
3. 一共有多少 block，各 block 的执行时间分布如何。
4. 整个 kernel 组从最早开始到最后完成用了多久。
5. 它是否延迟下游，是否真正影响最终完成时间。

需要区分的现象包括：

| 现象 | 优先调查的方向 |
| --- | --- |
| kernel 开始前有很长空隙 | producer 依赖、ready 后的调度延迟、dispatch 后的拾取延迟 |
| 单个 block 执行很长 | block 内工作量、搬运、计算和内部同步 |
| 单个 block 不长，但整个 kernel 组拖得很长 | block 数量、分多波执行、尾部负载不均 |
| kernel 很长，但与其他关键工作重叠 | 先判断缩短它是否会改善模型完成时间 |

**尚未开始执行的任务等上游，与已经运行的 Cube 等 Vector 产出概率，是两种问题。**

调度时间戳和依赖分析方法见[依赖与调度](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/dependency-and-scheduling.md)。

两个 kernel 的执行窗口如果互相重叠，不能把它们各自缩短的时间相加，当成端到端收益。

### 3. 分别检查 AIC、AIV0、AIV1，不能只看总时长

对于 Cube/Vector 混合 kernel，要确认：

- 每个核处理哪些 head、query 或 KV block。
- tile 的有效 shape 是多少，循环实际执行多少次。
- 两个 AIV 是否都在处理有效元素。
- 是否有一个 AIV 主要执行空 tile 和同步协议。

泳道图里两条 bar 一样长，并不说明两个核的计算量相同。本次 HCA 就出现了一个 AIV 承担主要计算，另一个 AIV 长时间维持同步的情况。

使用 PMU 时，不能做下面这些推导：

- Cube busy 只有 16%，所以剩下 84% 都在等待。
- 把 Cube、Vector、Scalar、MTE busy 周期相加，作为总耗时。
- MTE busy 周期减少 47%，所以搬运字节数也减少 47%。

各流水线可能重叠执行，busy 周期也不是字节计数器。PMU 能缩小原因范围，但不能自动给出精确等待占比。

如果需要确认具体哪个阶段、哪条同步指令造成停顿，再使用[核内模拟器 profiling](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/incore-simulator-profiling.md)或[设备上的 CCE 分阶段时间戳](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/cce-incore-profiling.md)。必须先验证实际工作量，不能把默认控制值导致的空执行当成“非常快”。

### 4. 沿着中间结果追踪搬运，而不是只数前端算子

对于大的中间结果，逐个回答：

```text
在哪里产生？
    ↓
下一个消费者需要它处于什么位置或布局？
    ↓
是否跨核、经过 GM，或发生额外复制？
    ↓
由哪个核写入最终输出？
```

重点检查：

- `concat`、布局转换或切片是否迫使 Cube 的结果先进入 Vector，再写到最终地址。
- 同一块概率或 KV 是否被不同消费者重复传输。
- 某次初始化是否紧接着就被有效输出覆盖。
- 不同 head tile 是否重复把同一个操作数从 L1 搬入 L0。
- 某些 GM 暂存是否用于换取更好的计算重叠。

查看生成的 PTO/C++，关注 `TPUSH`、`TPOP`、`TLOAD`、`TSTORE`、`TCONCAT` 及其外层循环、有效 shape。

**一个前端操作可能改变整条数据通路。** 例如删掉 `concat`，影响的可能不只是 Vector 上少算一次拼接，还包括 PV 结果不必再跨核传输。

同样，不能只数生成代码里出现了几次指令。需要结合循环次数、有效分支和每次处理的数据量，判断动态执行工作。

减少搬运是值得验证的方向，但不是最终验收标准。有时增加 GM 活动可以换来更短的串行执行链，最终仍然更快。

### 5. 想加流水时，先明确沿哪个维度展开

先写出依赖链，例如：

```text
QK → softmax 相关计算 → PV
```

然后确认等待期间能执行什么独立工作：另一个 query、另一组 head，还是同一个 query 的下一块 KV。

需要具体计算：

- 每个 worker 实际有多少个 query。
- 每个 query 有多少块有效 KV。
- 有多少迭代只是尾块、无效块或流水排空。
- 原代码是否已经在另一个维度上使用了流水。

**有 `pl.pipeline`，或者分配了多个 slot，都不能直接证明有效工作发生了重叠。** 如果每个 query 只有一块有效 KV，跨 KV block 的流水就没有第二块有效工作可以提前算。

验证流水收益时，可以保留 worker 数、tile、输出布局和缓冲区数量，只关闭提前计算。这样才能尽量区分“执行重叠变化”和“缓冲区大小变化”。

### 6. 先看每个 block 分到多少工作，再考虑调大 tile

worker 增加后，单个任务可能变短，只是因为每个 worker 分到的 query 更少。

所以比较时必须同时看：

- block / worker 数量。
- 每个 worker 的有效 query 或 KV block 数。
- 单任务时长分布和整个组的完成时间。
- 总 busy 工作以及模型端到端时间。

总资源消耗增加，与最长执行链缩短，可以同时发生。

之后再参考[Cube tile 调优](https://github.com/hw-native-sys/pypto-lib/blob/d6f2920046df7a19c28a5079c096a53b7e600219/docs/debug-and-tune/cube-tile-tuning.md)，调整 M、N、K。检查编译器给出的 Mat、Acc、Vec 占用，包含系统为 pipe 预留的 buffer。

不要假设“新版可以放下 64-head tile，旧版直接改成 64 就一定也能放下”。不同流水组织的缓冲区需求可能完全不同。编译时资源超限的方案没有硬件性能结果。

### 7. 用拆项实验解释原因，用不开 profiling 的 benchmark 判断收益

拆项实验要说明：改动了哪个前端因素，其他哪些量保持固定，以及生成代码到底发生了什么变化。

即使前端只改一处，也可能同时改变编译器的自动流水、buffer 大小和结果写出位置，不能声称这是“只改了一条硬件指令”。

实验要求：

- 前后使用相同卡组，尽量放在同一次设备预约里。
- 保留未改动的 kernel 作为对照，保留所有 rank 和所有已完成采样。
- 如果未改动的对照 kernel 也发生相近幅度的波动，不能把目标的小幅变化全部归因于代码修改。
- 检查局部算子结果和完整模型结果，并记录现有校验的覆盖范围。
- 有收益的独立改动分开记录，组合后重新判断端到端表现。

最终验收时关闭 chip profiling、PMU 和诊断插桩。每个版本运行一次 benchmark，在同一进程中完成 5 次 warmup 和 100 次计时，保留 `PYPTO_BENCH_RAW=1` 的样本，按照事先约定的中位数口径比较。

“运行一次 benchmark”不等于“只测一次 dispatch”。它包含多轮计时，但不需要为了凑样本重复提交同一个版本。

## 二、HCA #1187 案例

### 1. 配置与证据范围

目标 kernel：`hca_raw_attn` 和 `hca_cmp_qk_pv`。

前后源码：

- [优化前 216b497e](https://github.com/hw-native-sys/pypto-lib/blob/216b497e4b7d8bb7b47e304e79fd59e01de2a2e9/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)。
- [合入 #1187 后 cec12680](https://github.com/hw-native-sys/pypto-lib/blob/cec12680d61fba51199d0c82166cb641b3b194c1/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)。

基线已经包含此前的 gather / publication 优化。

| 项目 | 本次配置 |
| --- | --- |
| 模型入口 | `models/deepseek_v4_flash_dspark/decode_hca.py` |
| 平台 / 并行 | a2a3 / TP4 |
| 请求数 | 每个 rank 16 个，全局 64 个 |
| 上下文参数 | `--start-pos` 传入 16 个 `256` |
| 本次新 query token | S=8；每个 rank 128 行 query，位置为 256..263 |
| 工具链 | PyPTO f1bb086、simpler 15f5cbd、PTOAS 0.57、PTO ISA 96ba706、CANN 9.0.0 |
| 诊断方式 | `PYPTO_BENCH=0`、chip level 0、PMU event group 2、dependency generation 关闭 |
| 校验 | 固定输入和 golden；所有已完成版本的四项现有完整模型校验通过 |

这里的“16 × 256”对应每个 rank 的 16 个请求，各请求从位置 256 开始；不能把这个入口直接理解为全局 batch=16、每次只做一个 decode token。

主要前后 PMU 对照在卡组 `4,6,8,10` 的同一次预约中完成。拆项使用固定卡组 `6,8,10,14`：raw 提前计算开关、CMP worker 数实验与各自的新版对照在同一次预约；后面的 CMP 搬运实验与旧版基线来自同卡组的不同预约。

下文两个统计量分别是：

1. **任务周期中位数**：每个 rank 内先取该 kernel 的 AIC 任务周期中位数，再取四个 rank 的中位数，单位 cycle。
2. **busy 工作总量**：每个 rank 内汇总该 kernel 的指定 PMU 计数，再取四个 rank 的中位数。

这与最终 benchmark 的“各 rank 的 `effective_us` 中位数取最小值”不同。**本轮是 PMU 原因分析，不是新增的一组不开 profiling 的端到端性能结果。**

在本次固定的 runtime 版本中，开启 PMU 会关闭 pending 提前派发和 speculative early dispatch，以避免计数窗口被污染。因此 PMU 采样不能直接复现正常调度的全部重叠。

完整模型校验通过，也没有消除先前单独发现的旧版压缩分支 m/l 元数据跨步写出精度问题，不能将本次诊断当成对该问题的修复证明。

### 2. 发现一：旧版两个 AIV 的分工不均

旧版生成代码让 AIV0 处理有效的 32-head tile，每个 query 做两轮；AIV1 走有效 shape 为空的分支，同时维持 pipe 协议。两个核的任务可以持续差不多长，但有效计算量完全不同。

主要前后采样中，rank 3 上一组处理六个 query 的 raw 任务，Vector busy 计数为：

| Vector 实际忙碌周期 | 优化前 | 优化后 |
| --- | ---: | ---: |
| AIV0 | 19,212 | 9,350 |
| AIV1 | 746 | 9,368 |

新版显式使用 `pl.split_aiv(2, mode=pl.SplitMode.NONE)`，两个 AIV 各处理 32 个 head。生成代码和硬件计数都支持这个分工变化。

**教训：两个 AIV 的泳道一样长，不代表两个 AIV 都在持续计算。**

但这项分工变化没有被单独测成整个 PR 的收益百分比。新版 CMP 还增加了在线累积工作，其 Vector 总 busy 工作反而增加，不能简单说“两个 AIV 平分后，所有向量工作就减半”。

### 3. 发现二：64-head matmul 对应更少的 MTE1 工作

旧版分两轮处理 32 个 head；新版 Cube 一次处理 64 个 head，并使用完整输出宽度的 PV。

这改变了操作数的重复准备和概率 tile 的消费方式，但 attention 的数学乘加工作量并没有减半。

| AIC MTE1 busy 周期，按每 rank 总量取中位数 | 优化前 | 优化后 | 变化 |
| --- | ---: | ---: | ---: |
| `hca_raw_attn` | 654,544.5 | 347,343.0 | -46.93% |
| `hca_cmp_qk_pv` | 664,559.0 | 347,070.0 | -47.77% |

同时观察到：

- Cube busy 总量减少约 14%～15%。
- AIC MTE2 busy 总量增加。
- 旧版 Cube busy 占任务窗口的比例约为 raw 16.0%、CMP 13.7%。

这些证据不支持“旧版主要是 Cube 算满了”的解释，也不能推出“优化来自所有层级的搬运都减少”。MTE1 的变化也不能全部单独归因于 head tile，因为完整改写还改变了 PV 和流水组织。

### 4. 发现三：raw 的跨 query 流水有明确作用

每个 raw worker 需要处理多个 query。新版可以在 Vector 处理前一个 query 的 softmax 时，让 Cube 提前计算后续 query 的 QK。

下面表示 Cube 侧的工作顺序，`P(q)` 为 query q 的概率/权重 tile：

```text
关闭提前计算：QK(q0) → 等 P(q0) → PV(q0) → QK(q1) → ……
开启提前计算：QK(q0) → QK(q1) → QK(q2) → PV(q0) → QK(q3) → ……
```

拆项时只把 raw 的提前量从 2 改为 0，保留：

- 三个 transfer slot。
- 64-head matmul。
- 两个有效工作的 AIV。
- 20 个 worker。
- 相同输出布局。

结果：任务周期中位数从 **61,269.5 → 84,382.5，增加 37.72%**。四个 rank 都变慢，MTE1 工作基本不变。

**能证明的是：跨 query 的执行重叠有作用。不能声称精确测出了某条 wait 指令占用了多少周期。**

### 5. 发现四：ctx256 的 CMP 不能主要归功于跨 KV block 流水

CMP 是压缩 attention 分支，压缩比例为 128。位置 256..263 只有两个有效压缩行，放在一个 128-row attention block 中。

新版两个分支的流水维度不同：

| 分支 | 流水展开的维度 | 本次配置有没有后续有效工作 |
| --- | --- | --- |
| raw | 不同 query | 每个 worker 有多个 query，可以重叠 |
| CMP | 同一个 query 的不同 KV block | 只有一个有效 block，没有第二块可以重叠 |

而且旧版 CMP 的 head 循环已经有 `pl.pipeline(..., stage=2)`。因此不能只看新版“加了流水”的写法，就把收益都归到流水上。

CMP 的任务划分变化则有明确作用：128 个 query 原来分给 16 个 worker，每个处理 8 个；新版分给 24 个 worker，每个处理 5～6 个。

保留新版其他逻辑，只恢复为 16 个 worker：

| 配置 | AIC 任务周期中位数 |
| --- | ---: |
| 新版 24 workers | 74,062.0 |
| 新版逻辑，改回 16 workers | 107,130.5 |

四个 rank 的任务都变长，MTE1 总工作基本不变。这里主要反映每个任务分到的工作量和并发组织变化。

该次采样中，16 workers 的 AIC 任务周期求和反而更低。这说明“总资源消耗少”和“最长工作链短”不是一回事。

另外，把 CMP 提前量改为 0 得到 75,981.75 cycle，对照为 74,062.0 cycle，各 rank 变化方向不一致，且来自同卡组的不同预约。这不足以证明小幅收益或回退，也不支持将本配置下的 CMP 提升归因于多 KV block 重叠。

### 6. 发现五：旧 CMP 的 concat 引入了额外跨核搬运

旧代码先分别计算 PV 的左右两个 256 列，再 `concat` 后写出。生成代码显示：同一份概率被消费两次，两半 FP32 PV 结果还要传给 Vector。

```text
Cube 产生左右两半 PV
    ↓
跨核传输
    ↓
AIV 在 UB 中 concat
    ↓
写入最终 GM 输出
```

对旧版保留 32-head tile、16 workers，做了两个拆项：

| 方案 | AIC 任务周期中位数 | 更可靠的证据 |
| --- | ---: | --- |
| 原始两次 PV + concat | 154,328.0 | 生成代码中，AIV 接收并拼接两半 PV |
| 保留两次 PV，只改成两半分别直接写出 | 143,138.5 | AIC 直接写出 PV；AIV MTE2 busy 工作减少 73.86% |
| 改成一次 512 列 PV，去掉 concat | 112,720.5 | AIC 直接写出，去掉重复概率消费；Cube busy 工作基本不变 |

这里有两个必须保留的限制：

1. 这些旧版拆项与基线来自相同卡组的不同预约。只去掉 concat 的任务周期下降了 7.25%，但未改动的 raw 对照也下降了约 6.8%。因此，不能仅凭这点任务时长变化，认定有稳定的独立性能收益。**生成代码确认搬运路径消失，以及对应 busy 计数降低，是更强的证据。**
2. 一次 512 列 PV 同时改变了概率消费次数、自动流水和 pipe buffer 大小。不能把该方案的全部缩短都归因于删掉 concat。

#1187 去掉了原来的 concat 和重复两半组织，但新版 CMP 仍然需要 Vector 消费 PV 结果来做在线累积。**诊断中的直接写出方案不等于完整 PR；不能说 #1187 完全消除了 CMP 的 Cube→Vector 通信。**

### 7. 没有收益或没有有效结果的尝试，也要记下来

- 只把旧 raw 的 matmul 加宽，保留 32-head tile 和旧调度：任务周期 107,148.0 → 108,874.25，没有观察到改善。
- 将旧 raw 的 head tile 直接改成 64：在原来的自动 pipe 组织下，Vec 需要 229,376 字节，超过 188,416 字节可用预算，没有硬件性能结果。
- 本次没有采到可用于结论的核内模拟器 trace。生成器对计算得到的动态维度、默认 scalar 和 buffer 分配还需要额外适配；没有把未正确配置的空执行当成性能证据。
- 没有得到精确同步等待比例，也没有测出各项改动对整个 PR 的可相加贡献。

**这次不能用一个“都在等”或“matmul 变大了”的解释包办两个 kernel。raw 和 CMP 的有效工作维度、结果消费方式和任务划分都不同。**

## 三、以后怎么判断证据是否足够

| 想下的结论 | 至少需要什么证据 |
| --- | --- |
| 某条搬运路径被去掉了 | 前后生成代码、tile 布局和消费者位置；用对应 PMU 工作计数交叉验证 |
| 第二个 AIV 被有效利用了 | 分别查看两个 AIV 的有效 tile、循环和 busy 工作 |
| 流水隐藏了等待 | 确认可重叠的有效迭代；保留其他条件，关闭提前量做拆项 |
| kernel 组完成得更早 | 相同口径的整组起止时间、block 数和关键路径，不能只看一个 bar |
| 模型端到端变快了 | 相同配置、关闭 profiling 的 benchmark，按约定中位数比较 |
| 某种等待占了总耗时的百分之几 | 能对应到该阶段的核内 trace 或时间戳，不能用 `1 - Cube busy` 估算 |

## 四、后续实验记录模板

```text
目标 kernel：
前后源码版本及完整工具链：
输入配置：每 rank / 全局请求数、S、start-pos、dtype：
卡组、前后是否同一次预约：
最终 benchmark 口径、rounds / warmup、profiling 开关：
目标任务/组是否位于关键路径：
block 数、每个 block 的有效工作量：
AIC / AIV0 / AIV1 实际分工：
怀疑的搬运路径或串行依赖：
本次只改变的前端因素：
生成代码实际发生的变化：
正确性校验及覆盖范围：
任务周期、busy 计数、单位和汇总口径：
未改动的对照 kernel 是否也发生波动：
不开 profiling 的端到端中位数及验收结论：
无收益尝试、失败采样和未解决问题：
泳道图、PMU、生成代码、patch、日志路径：
```

一次记录中，应分别写清楚“搬运工作减少”“任务周期变短”“端到端变快”，不要把三个层面的结论混写。

## 五、本次原始资料

本地实验目录：`build_output/hca_1187_cause_20260909_091306/`。

- 完整英文实验报告（本地证据：`build_output/hca_1187_cause_20260909_091306/REPORT.md`；未随文归档）。
- 全部计数与每个 rank 的原始文件路径（本地证据：`build_output/hca_1187_cause_20260909_091306/all_counters.json`；未随文归档）。
- 计数汇总表（本地证据：`build_output/hca_1187_cause_20260909_091306/counter_summary.csv`；未随文归档）。
- 生成代码中的搬运证据（本地证据：`build_output/hca_1187_cause_20260909_091306/lowering_evidence.json`；未随文归档）。
- 汇总脚本（本地证据：`build_output/hca_1187_cause_20260909_091306/analyze_counters.py`；未随文归档）。
- [HCA 历史调优日志](../case-studies/dspark-hca-decode-tuning-log.md)。

各拆项的 `source.patch`、运行命令、任务日志，以及每个 rank 的 `pmu.csv` 均保留在实验目录中。完整证据包含 12 次已完成采样、48 份 rank CSV，也保留了采样完成后驱动异常退出的结果。这些是 PMU 诊断采样，不是 12 次独立的 100 轮 benchmark，没有丢弃已经完成的采样。
