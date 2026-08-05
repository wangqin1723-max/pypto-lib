> [!NOTE]
> 🟡 **临时个人参考**：本文是当前英文 `README.md` 的中文伴随译本；英文版是规范版本，本文后续可以直接删除。

# DeepSeek V4 Pro W8A8 128K Bring-up 计划

![里程碑 3 HCA](https://img.shields.io/badge/M3_HCA-已完成-2ea44f)
![里程碑 4 CSA](https://img.shields.io/badge/M4_CSA-已完成-2ea44f)
![里程碑 5 计算](https://img.shields.io/badge/M5_Compute-已完成-2ea44f)
![里程碑 6 Forward](https://img.shields.io/badge/M6_Forward-已完成-2ea44f)
![COMM EP2/4/8](https://img.shields.io/badge/COMM_EP2%2F4%2F8-仅合同-f0883e)
![Target](https://img.shields.io/badge/Target-a2a3-0969da)
![Context](https://img.shields.io/badge/Context-128K-8250df)
![Deployment EP](https://img.shields.io/badge/Deployment_EP-128-f0883e)
![Document](https://img.shields.io/badge/Document-TEMPORARY-d4a72c)

> **颜色标记**：🟢 已完成　·　🟡 部分完成　·　🟠 仅合同/未实测　·
> 🔴 禁止或高风险　·　🟣 设备实测证据　·　🔵 固定合同

## 🧭 状态与来源

本目录是 DeepSeek V4 Pro W8A8 解码基准的实现脚手架，该基准由 GitHub
issue 873 跟踪。

- 种子目录：models/deepseek/v4-flash
- 种子提交：2388850d82f40df3596b78e882a9903d255ae275
- 目标后端：当前 Ascend 910 服务器上的 a2a3
- 初始状态：Python 文件以受版本控制文件的精确副本作为种子

Phase 1 至 Phase 4 现已覆盖不可变目标合同、叶子/SWA 路径、128K HCA，以及
完整历史 CSA/indexer 路径。Phase 5 已启用单 die compute-only MoE 代理；其中
物理 COMM_EP2/4/8 仍只是主机侧布局合同，尚无分布式设备实测，因此里程碑 5
只能标记为部分完成。里程碑 6 已在明确限定的单 die 计算代理边界内完成：
最深四层的有界 golden 通过，静态阶梯的每个深度均完成 100 轮设备重复，
完整 61 层驻留图可装入并运行于一个目标 die。物理 EP128 通信、SWA、MTP、
embedding 和最终 model head 仍在该结果边界之外。现有 models/deepseek/v4-flash 和
models/deepseek/v4-pro 文件保持不变。

### 🟢 Phase 1 已完成

- B=2、S=4、起始位置 131072、最大序列长度 131584
- 在保留规范 1M-position PRO 的同时，冻结一个 Pro-W8A8 preset
- 纯 Python 的 main/MTP 层归属验证
- 将 deployment-EP128 与物理 COMM_EP2/4/8 形状合同分离
- 混合逻辑与物理缓存容量常量
- 将 CPU-only 配置合同测试作为 Phase 1 唯一的 CI allowlist 条目
- 在广泛的 PR、模拟器和 A2/A3 设备 sweep 中按目录排除本目录

### 🟢 Phase 2 已完成

- 非别名可搜索历史和请求隔离环形缓冲区的主机侧合同
- 在 384 个专家上进行 Gate 分数路由，采用宽度 512 的排序和精确尾部 ID
- 单 die 的 deployment-EP128 计算形状，包含三个本地专家和生产接收容量 1024
- 均衡、偏斜和尾部路由工作负载，分别为 [16, 16, 16]、[48, 0, 0] 和
  [17, 16, 15]
- 位置 131072 和 131071 的 ratio-0 SWA 路径，使用包含六个物理页的
  original-KV 环形缓冲区，且不包含 compressor 或 indexer 工作
- 每个 Phase 2 可执行程序都提供冻结数据回放控制和 level-4 L2 swimlane 控制

Phase 3 和 Phase 4 已在真实设备上完成元数据、HCA、compressor、indexer 与
组合 CSA golden，覆盖所需的 16K、128K、边界和最大尾部位置。Phase 5
compute-only 已通过均衡与偏斜路由工作负载；其 level-4 trace 可独立识别 Gate、
共享专家与路由专家工作。独立通信入口仍明确标记为未实测，请求执行时会主动
fail-fast。

> [!WARNING]
> 🔴 **Q projection 布局约束**：目标服务器上的 W8 query projection 必须使用
> head-major 输出通道权重 `[H, HEAD_DIM, Q_LORA]`，并使用转置的 cube 操作数。
> 种子实现的扁平 `[Q_LORA, H * HEAD_DIM]` GM-to-Mat 访问无法生成正确的 Pro
> 尺寸结果。已经验证的 HCA 和 CSA 路径现已使用 head-major 布局；后续组合模型
> 调用方必须继续保持该布局。

Phase 2 实测基线先丢弃五轮 warmup，再运行 100 轮冻结数据。L4 是单独采集的
插桩图耗时；下表所有数值的单位均为微秒。测量来自基于 pypto-lib
3c3db6c07a97cc67382109dfefb2fc32e722e112 的未提交 Phase 2 工作树。

| 程序 | 用例 | L4 | Min | **Median** | Mean | Max |
|---|---|---:|---:|---:|---:|---:|
| 🔵 Gate | 分数路由，尾部专家 | 74.98 | 50.6 | **67.0** | 65.5 | 72.3 |
| 🟢 共享专家 | 8 行 | 111.82 | 118.2 | **132.1** | 131.4 | 142.5 |
| 🟣 路由专家 | [16, 16, 16]，容量 1024 | 243.96 | 215.3 | **223.6** | 225.3 | 250.0 |
| 🟣 SWA | B=2，S=4，位置 131072 | 605.48 | 540.5 | **597.6** | 590.4 | 665.6 |

| 程序 | a2a3sim | a2a3 golden 用例 |
|---|---|---|
| Metadata | ✅ 主机侧合同通过 | 环形缓冲区、完整历史、边界、非法池 |
| Gate | ✅ 通过 | random 和 tail-expert |
| 共享专家 | ✅ 通过 | 8 行 |
| 路由专家 | ✅ 通过 | [16,16,16]、[48,0,0] 和 [17,16,15] |
| SWA | 🟡 编译通过 | 位置 131072 和 131071 |

Gate、共享专家和路由专家均通过完整的 a2a3sim golden。组合 SWA 图可在
a2a3sim 上编译，但其完整模拟器运行停滞在 qk_pv；独立 sparse-attention
组件和两个必需的 a2a3 SWA 位置均通过。测量环境固定到 PyPTO
7d743e8d35bfd45df2b09a08b6a79308fada1342、simpler
dccb8379080b43173744b9981de2542b3d025e19，以及 PTO ISA
83d01313d9bfc247c4b7c8bcf969d1019f0d106f。

规范项目指南仍位于 docs/。尤其应遵循：

- docs/pypto-coding/pypto-coding-style.md
- docs/run-and-validate/compile-runtime-workflow.md
- docs/run-and-validate/golden-harness.md
- docs/run-and-validate/save-and-replay.md
- docs/debug-and-tune/performance-tuning.md

不要提交生成的 build_output 产物或保存的机器特定运行时数据。

## 🎯 目标

测量 DeepSeek V4 Pro 主模型在 128K 上下文执行一次验证步骤时的单 die 计算成本，
然后使用独立 attention 和 MoE 组件测量结果解释该墙钟时间。

固定目标点为：

| 维度 | 目标 |
|---|---:|
| 平台 | a2a3 |
| 解码 batch | 2 个请求 |
| 每个请求的序列行数 | 4 |
| 总行数 | 8 |
| 起始位置 | 131072 |
| 最大序列长度 | 131584 |
| 主模型隐藏层数 | 61 |
| 主模型 attention 构成 | 31 HCA、30 CSA、0 SWA |
| 学习得到的 MTP 层数 | 1 个独立 ratio-0/SWA 层 |
| 路由专家数 | 全局 384 |
| 有效 deployment EP | 128 |
| 本地路由专家数 | 3 |
| 路由专家计算 dtype | W8A8 |

S=4 中的第一行加三行 draft 描述一个目标模型验证 batch。它本身不会生成三个
自回归依赖的 draft。

## ⏱️ 测量边界

### 主模型验证

主要结果是直接测得的 decode_fwd 在 B=2、S=4 时的墙钟时间。其解释性重构为：

~~~text
L_main_verify =
    31 * L_HCA
  + 30 * L_CSA
  + 61 * L_MoE_compute
  + R_main
~~~

> [!IMPORTANT]
> 🔵 `R_main` 包含编排间隙、最终 head 与 normalization 工作，以及独立运行与
> 组合运行之间的调度差异。**SWA 在主模型中的系数为零。**

### MTP draft 生成

独立 MTP 层包含 projection、SWA attention、MoE、head 和 normalization 工作；
当所选 runner 包含这些操作时，还包括服务侧 packing、sampling 和缓存管理。

未来 draft-depth-three 服务结果必须按以下方式测量：

~~~text
L_cycle_D3 =
    L_main_verify(B=2, S=4)
  + L_verify_commit_handoff
  + sum(L_mtp_iteration[j], j=1..3)
  + L_control_gaps
~~~

> [!CAUTION]
> 🔴 在真正的三轮迭代 runner 出现之前，单次 MTP 延迟的三倍只是**外推**，
> 不能标记为端到端实测。

### EP128 范围

部署分片与物理通信 world size 相互独立：

~~~text
GLOBAL_EXPERTS = 384
DEPLOYMENT_EP = 128
LOCAL_EXPERTS = GLOBAL_EXPERTS / DEPLOYMENT_EP = 3
COMM_EP = one of 2, 4, or 8 when communication is measured
~~~

> [!IMPORTANT]
> 🔵 EP128 交付物是**单 die 计算代理**。它不得分配带有 128-rank 前导维度的
> tensor，也不得提交分布式 peer 操作。实测 EP2/4/8 通信必须单独报告，
> 绝不能标记为实测 EP128 通信。

对于 B=2、S=4 和 top-k=6，均衡路由负载为：

~~~text
global routes                 = 128 * 8 * 6 = 6144
mean rows per global expert   = 6144 / 384 = 16
mean rows per local shard     = 3 * 16 = 48
worst recv capacity per expert = 128 * 8 = 1024
~~~

生产布局基准保持 recv capacity 1024，并将 recv_expert_count 设置为
[16, 16, 16]。在该均衡用例中，路由 kernel 必须只为每个专家 dispatch 一个
16 行计算 tile。紧凑的 capacity-16 运行可用作对照，但不能作为唯一官方结果。

## 💾 缓存分配合同

> [!IMPORTANT]
> 🔵 **逻辑可寻址范围与物理驻留相互独立**：可搜索完整历史的缓存需要唯一物理
> 存储；有限视野消费者使用请求隔离的环形缓冲区，并且只能复用已经过期的逻辑页。

| 池 | 每个请求的逻辑宽度 | B=2 物理目标 | 策略 |
|---|---:|---:|---|
| Original KV | 1028 页 | 覆盖所有对齐情况共需 6 页 | 🔄 环形 |
| Ratio-4 compressed KV | 257 页 | 514 页 | 🔒 完整、无别名 |
| Indexer KV | 257 页 | 514 页 | 🔒 完整、无别名 |
| Ratio-128 compressed KV | 9 页 | 18 页 | 🔒 完整、无别名 |
| HCA compressor state | 16448 个 state 页 | 32 页 | 🔄 环形 |
| CSA compressor state | 32896 个 state 页 | 4 页 | 🔄 环形 |
| CSA inner compressor state | 32896 个 state 页 | 4 页 | 🔄 环形 |

精确的起始位置步骤每个请求会触及两个 original-KV 页，全局共四个。全局六页可覆盖
128 行窗口加四行在途验证数据的所有对齐情况。

当前统一 forward 的 compressed-pool 形状也可以对 HCA 层使用 ratio-4 容量。
拆分 HCA 与 CSA 的物理池形状是一项可选内存优化，并非正确性的前置条件。

block-table 验证必须证明以下所有条件：

1. 不同请求绝不共享同时存活的物理页。
2. 环形缓冲区绝不会让两个同时存活的逻辑行发生别名。
3. 每个 ratio-4 compressed 页和 indexer 逻辑页在完整可见历史上都有唯一物理存储。
4. Golden 验证不能只依赖与设备 kernel 相同的别名映射。
5. 在 128K 性能运行中，物理缓存内容必须具有足够的地址多样性，避免人为制造 L2 复用。

## 🟢 里程碑 1：配置与 CI 隔离（已完成）

### 工作内容

- 添加不可变的 Pro-W8A8 kernel preset，同时不修改架构 PRO preset。
- 设置 B=2、S=4、起始位置 131072 和最大序列长度 131584。
- 让 384 个全局专家与所选通信 EP 保持独立。
- 将路由专家存储和计算设置为 INT8 W8A8 语义。
- 从前 61 个 compression ratio 推导主模型层构成。
- 让末尾的 ratio-0 条目归属于 MTP 层。
- 仅从 models/deepseek/v4-pro 移植每个 kernel 所需的 Pro 维度和 tiling 修复；
  不要整体替换持续维护的种子实现。
- 在本目录尚未完成时，将其从广泛模型发现机制中排除。
- 随着 kernel 得到验证，添加明确且精简的 CI allowlist。

### 退出标准

- 导入 config 不会导入或初始化分布式运行时状态。
- 断言报告 31 个 HCA、30 个 CSA 和零个主模型 SWA 层。
- 对每个受支持的 COMM_EP 值，Gate 路由空间始终恰好为 384。
- 不修改任何现有模型目录。

## 🟢 里程碑 2：叶子计算与 SWA bring-up（已完成）

在 HCA、CSA、分布式 MoE 或完整 forward 之前，先 bring up 最小的单 die 程序。

### 顺序

1. 主机元数据与 block-table 测试
2. gate.py
3. expert_shared.py
4. expert_routed.py
5. decode_attention_swa.py

### 必需用例

Gate：

- 8 行输入
- 384 个全局专家
- 分数 padding 宽度 512
- top-k 6
- 每个输出专家索引均位于 [0, 384)

路由专家：

- 3 个本地专家
- recv capacity 1024
- 均衡计数 [16, 16, 16]
- 偏斜计数 [48, 0, 0]
- 零行覆盖和尾部计数 [17, 16, 15]

SWA：

- B=2 和 S=4 的因果元数据
- 目标起始位置 131072
- 一个页边界正确性用例
- 请求隔离的 original-KV 环形缓冲区
- Pro hidden、head 和 projection 维度

### 验证顺序

对于每个可执行程序：

1. 运行导入和 tensor 形状断言。
2. 在支持时于 a2a3sim 上编译并验证。
3. 通过 task-submit 在一个 a2a3 die 上运行 golden 验证。
4. 保存一次已验证的输入和 golden 数据。
5. 回放冻结数据以进行重复计时。
6. 采集 level-4 L2 swimlane。

### 退出标准

- 每个叶子程序均可编译并通过其 golden 比较。
- 均衡路由 trace 对每个本地专家执行一个 16 行 tile，而不是 64 行。
- SWA 不包含 compressed-cache 或 indexer 工作。
- 重复设备计时足够稳定，可以建立基线 median。

## 🟢 里程碑 3：128K HCA（已完成）

先 bring up HCA，再处理 CSA，因为 HCA 没有学习得到的 indexer。

### 工作内容

- 使用由安全环形缓冲区支持的 1028 列 original-KV 逻辑 block table。
- 独立 HCA 使用 18 个无别名 ratio-128 compressed 页。
- 使用由 32 个环形页支持的 16448 列 HCA-state table。
- 验证在主机和设备路径上生成的元数据。
- 保持固定 top-k 合同，同时标记无效的 compressed 尾部行。

### 必需位置

- 131072：官方性能点，其中四行 chunk 不跨越 ratio-128 compression 边界。
- 131071：用于覆盖 ratio-128 边界处的 state pooling 和 compressed-cache
  writeback 正确性。

### 🟣 完成证据

- Pro B=2/S=4 的目标与最大尾部 fixture 均在真实 a2a3 硬件上通过设备侧
  元数据验证。
- 完整 HCA 在官方位置 131072 和 ratio-128 边界位置 131071 均通过 golden。
- 边界回放及其元数据/输出 golden 共同证明 state pooling，以及每个受影响请求
  恰好一次的 compressed-cache writeback；original-KV 环、18 个唯一 compressed
  页和 32 个 HCA-state 环形页均保持请求隔离。
- 边界 level-4 L2 插桩图耗时为 **834.26 us**。该数值是插桩 trace 耗时，
  不是重复运行的延迟 median。

### 退出标准

- 两个位置都通过 a2a3 golden 验证。
- 主机/设备元数据表和通过 golden 验证的缓存输出表明不存在跨请求
  的存活别名。
- 边界 trace、元数据和 golden 共同覆盖预期的窗口与 compressed-tail 读取。
- 边界元数据和输出验证证明每个受影响请求恰好发生一次 ratio-128 compression
  事件；对应 trace 记录组合后的 compressor 任务。

## 🟢 里程碑 4：128K CSA 与 indexer（已完成）

CSA 是风险最高的独立 operator，在其完整历史路径正确之前保持隔离。

### 工作内容

- 在 16K context 下使用完整 32896-value score buffer 验证 B=2、S=4。
- 将固定的 4096-value 双半区 top-k merge 替换为支持分数长度 32896 的层次结构。
- 分配 514 个唯一 ratio-4 compressed 页和 514 个唯一 indexer 页。
- 使用相互独立的四页环形缓冲区支持 main 和 inner compressor state table。
- 确保目标步骤写入逻辑 compressed page 256 时不会将其别名到 page zero。
- 使 indexer 分数生成量与完整可见 ratio-4 历史成正比，同时让 sparse attention
  保持受 top-k 1024 限制。

### 🟣 完成证据

- 真实设备上的 B=2/S=4 16K 回归用例已在完整 32896-score 实现下通过。
- 官方 128K CSA 用例在位置 131072 使用冻结数据通过 a2a3 golden。
- 独立 indexer 在最大尾部位置 131580 通过，覆盖最后一个 128-value 分数组；
  score 和 1024 个选中 ID 均与参考实现一致。
- main 与 inner ratio-4 compressor 叶子用例均通过边界 cache writeback。
  ratio-4 compressed 历史和 indexer 历史分别使用 514 个唯一物理页；逻辑页 256
  映射到物理页 512、513，不会别名到 page zero。
- 目标 level-4 L2 插桩图耗时为 **2374.68 us**。

> [!IMPORTANT]
> 🔵 Trace 中的任务数量不等于扫描页数：每个 token 级 `score_mat` 任务会在
> 内部循环扫描其全部可见页。位置 131072 的四行 token 对每个请求分别可见
> 256、256、256、257 页。因此，完整历史覆盖由可见长度合同、514 个唯一页
> block table、真实设备 score/top-k golden、compressor writeback 和组合 CSA
> trace 共同证明，不能把 trace 中的 task count 直接解释成 257 页。

### 退出标准

- 16K B=2/S=4 回归用例在完整分数路径启用时通过。
- 32896-score top-k 与 torch 参考一致。
- 128K 编排使用冻结数据通过 golden 验证。
- L2 trace 与显示最大 257 页扫描范围的元数据完成对账。
- 不使用重复物理页来代表多个可见历史页。

不接受将 16K 到 128K 的 CSA 外推作为最终 128K 数值。

## 🟡 里程碑 5：EP128 计算代理与通信合同（部分完成）

> [!WARNING]
> 🟠 单 die 计算代理完成，并不代表本里程碑的通信交付物已经完成。

### 当前状态

- Compute-only：**已完成**。单设备入口显式输出 Gate 量化交接结果，采用
  gate `[384, D]`、路由权重 `[3, ...]` 和 recv `[3, 1024, D]`，并在真实设备
  上通过均衡 `[16, 16, 16]` 与偏斜 `[48, 0, 0]` golden。
- 计算代理使用稳定的单位幅值输入行，同时保留所选 Gate 路由 fixture；随机输入
  Gate 行为继续由独立 Gate golden 覆盖。
- 均衡用例的 level-4 L2 插桩图耗时为 **276.76 us**；trace 可分别识别 Gate、
  共享专家和路由专家任务族。
- 通信：**仅合同、未实测**。主机侧合同保持 384 个全局专家，并定义
  COMM_EP2、COMM_EP4 和 COMM_EP8 形状，但尚未实测任何物理 peer dispatch
  或 combine kernel。
- 请求执行通信入口时会主动 fail-fast。在真实分布式路径分别以对应 world size
  运行之前，不得发布 dispatch/combine 延迟表，也不得将其标记为实测 EP128 通信。

### Compute-only 入口

单 die 计算入口为 `moe_compute.py`，其中包含：

- 面向八个本地行的完整 384-expert gate
- 面向八个本地行的共享专家
- 三个处理合成接收行的路由专家
- 生产 recv stride 与 capacity
- 仅在成本可单独识别时包含本地 packing 或 reduction
- 不包含 pld window、peer put、notify、wait 或 DistributedConfig

计算代理可以复现单 die 形状和调度，但不能声称与 EP128 分布式输出数值等价。
真实本地专家分片处理的大多数行来自其他 rank，而该 rank 自己 token 的大多数路由
会终止在远端。

### 通信入口

当前通信入口定义了物理 COMM_EP 值 2、4 和 8 的主机侧布局合同。后续独立
的分布式 MoE 路径必须使用实际 world size 报告 dispatch 和 combine 测量
结果，且不得为了保持恒定的本地专家数而修改全局专家数。

### 退出标准

- Compute-only 仅接受一个设备。
- 形状为 gate [384, D]、路由权重 [3, ...]、recv [3, 1024, D]。
- 均衡和偏斜路由工作负载通过独立 golden。
- L2 trace 分别识别 gate、共享专家和路由专家计算。
- 通信表明确标识 COMM_EP。

## 🟢 里程碑 6：缩减深度与完整主模型 forward（计算代理已完成）

> [!IMPORTANT]
> 🟣 **完整 61 层、31 HCA/30 CSA 单 die 计算代理已完成目标设备实测。**
> 所有声明为驻留的 compute-ABI spec 均放置于一个 a2a3 die，并完成 100 个实测轮次。
> 这证明了该计算边界的驻留适配与重复 dispatch 活性。深度用例没有 Torch
> golden，因此不声称 depth16/depth31/depth61 的深层数值正确性。

### 🧩 可执行边界与已交付图

- `main_compute_manifest.py` 以不实例化 tensor 的方式定义 `hca1`、`csa1`、
  `depth2`、`depth4`、`depth16`、`depth31` 和 `depth61` 完整静态阶梯。
- `decode_layer_compute.py` 将已验证的 HCA 或 CSA 路径与 Gate、共享专家和
  三个本地路由专家的计算代理组合。
- `decode_fwd_compute.py` 在单 worker 驻留 L3 程序中堆叠每层独立的权重和
  cache。每层都拥有自己的物理 cache 字节区间和本层局部 block ID，不存在跨层环形别名。
- ABI 覆盖 `x_hc` 到 `x_next`、所有选定 HCA/CSA 层、本地 deployment-EP128 形状的
  MoE 计算、逐层 cache 以及 RoPE。
- ABI **不包含**物理 EP128 dispatch/combine 通信、SWA、MTP、embedding lookup、
  最终 normalization 和 model/LM head。该结果既不是 EP128 端到端，也不是完整 serving 周期耗时。

> [!NOTE]
> 🔵 Original KV 和有限视野 compressor state 保留已验证的请求隔离环形策略。
> 可搜索 compressed-KV 与 indexer cache 按完整声明的生产容量分配，
> **不使用环形历史别名**。

### ✅ 正确性、适配与重复证据

所有用例均使用起始位置 131072、B=2、S=4、本地专家均衡负载和固定
PTOAS 0.54。有界 `commit` golden 比较 forward 的全部 11 个输出；更深用例明确仅表示
驻留适配与活性。

| 用例 | 层组成 | `commit` 正确性或驻留适配证据 | 100 轮 `overwrite` |
|---|---:|---|---|
| `hca1` | 1 HCA | ✅ Golden 通过 | ✅ 通过 |
| `csa1` | 1 CSA | ✅ Golden 通过 | ✅ 通过 |
| `depth2` | 2 HCA | ✅ Golden 通过 | ✅ 通过 |
| `depth4` | 3 HCA + 1 CSA | ✅ Golden 通过 | ✅ 通过 |
| `depth16` | 9 HCA + 7 CSA | 🟣 驻留适配通过，无 golden | ✅ 通过 |
| `depth31` | 16 HCA + 15 CSA | 🟣 驻留适配通过，无 golden | ✅ 通过 |
| `depth61` | 31 HCA + 30 CSA | 🟣 驻留适配通过，无 golden | ✅ 通过 |

`depth4` golden 会先在每个 top-k 行内按专家 ID/权重对做规范化，因此能接受无害的
成对顺序变化，同时不会让权重与所选专家脱钩。

静态阶梯包含共享 embedding、最终 normalization/head、LM-head 和 RoPE 资产。
RoPE 在可执行 ABI 内；embedding 和最终 model/LM-head 资产在其外。共享分配约为
3.484 GiB，metadata 会随所选 attention 类型变化。

| 阶梯用例 | HCA | CSA | 静态核算字节数 | 静态核算 GiB |
|---|---:|---:|---:|---:|
| `hca1` | 1 | 0 | 4,426,231,636 | 4.122 |
| `csa1` | 0 | 1 | 4,534,786,588 | 4.223 |
| `depth2` | 2 | 0 | 5,111,448,620 | 4.760 |
| `depth4` | 3 | 1 | 6,587,466,476 | 6.135 |
| `depth16` | 9 | 7 | 15,421,774,604 | 14.363 |
| `depth31` | 16 | 15 | 26,518,737,844 | 24.697 |
| `depth61` | 31 | 30 | **48,604,508,164** | **45.266** |

> [!IMPORTANT]
> 🔵 `depth61` 的 **45.266 GiB** 是完整静态核算值，包含共享 embedding、
> LM-head 和 RoPE 资产。静态核算本身不能证明运行时适配；上表中成功的
> 驻留设备任务才是直接适配证据。

`depth61` 默认为活跃 ring 0 配置 14 GiB heap，其余三个非活跃 ring 各配置
256 MiB。单次 scope-stat 运行观测到 ring 0 heap 高水位为 12,593,982,464 字节，
task 条目高水位为 6,362，dependency 条目高水位为 7,126。16,384 的 task window 和
65,536 的 dependency pool 均高于已观测需求。

| `depth61` 规划边界 | 字节 | GiB |
|---|---:|---:|
| 完整静态 manifest | 48,604,508,164 | 45.266 |
| 实际 compute-ABI 驻留 specs | 44,896,648,568 | 41.813 |
| 已配置 ring heaps | 15,837,691,904 | 14.750 |
| 运行时 TMR 共享内存 | 325,583,744 | 0.303 |
| 固定运行时 private-arena 上界 | 28,285,824 | 0.026 |
| 保留的非驻留 ABI staging | 24,591,360 | 0.023 |
| Compute ABI + 已知运行时 + staging | 61,112,801,400 | 56.916 |
| 静态 manifest + 已知运行时 + staging | 64,820,660,996 | 60.369 |
| 距标称 64-GiB 上限剩余 | 3,898,815,740 | 3.631 |

> [!WARNING]
> 🟠 Compiler code、device context、allocator 碎片和其他未暴露的运行时开销不在表内。
> **60.369 GiB 是跨边界保守规划值，不是实测峰值。**实际建立适配结论的是
> 成功的 `depth61` 设备运行，而不是任何一行单独的核算值。

### ⏱️ 重复 Effective 耗时

每个计时任务先丢弃 5 轮 warmup，再强制要求恰好 100 个实测轮次、每轮 1 个 rank
和 1 次 dispatch。Harness 会拒绝缺失统计、轮次 flatten、dispatch 槽位变化、不完整网格和
非正样本。下表是运行时 Effective 设备窗口，单位为毫秒。

| 用例 | Min | **Median** | Mean | Max |
|---|---:|---:|---:|---:|
| `hca1` | 0.9630 | **1.0045** | 1.0281 | 1.1236 |
| `csa1` | 1.9113 | **2.5259** | 2.3695 | 2.6178 |
| `depth2` | 1.9918 | **2.0565** | 2.1052 | 2.2449 |
| `depth4` | 5.0205 | **5.8124** | 5.6978 | 5.9203 |
| `depth16` | 23.4219 | **23.7290** | 25.5988 | 28.3371 |
| `depth31` | 47.5022 | **56.8154** | 53.0156 | 57.4078 |
| `depth61` | 93.4279 | **112.7978** | 105.9753 | 113.8827 |

原始样本反复出现两个耗时带，因此表中保留完整 min/median/mean/max 上下文。
任务在同一台目标服务器上使用自动分配的 die；下面的重构仅用于解释，且将运行和
卡间差异及组合效应都保留在 residual 中。

HCA/CSA 单层项已经包含 attention 和同一个 compute-only MoE 代理，因此不能再加
`61 * L_MoE`：

~~~text
L_layer_reconstruction = 31 * 1.0045 ms + 30 * 2.5259 ms
                       = 106.9165 ms
R_proxy                = 112.7978 ms - 106.9165 ms
                       = +5.8813 ms  (+5.214% of direct median)
~~~

> [!IMPORTANT]
> 🔵 Residual 包含静态层堆叠的调度/重叠、跨层依赖以及运行与卡间差异。
> 不能将它重新标记为 SWA、MTP、物理 EP128 通信、embedding 或最终 head 耗时，
> 因为这些操作都没有在本代理中执行。

### 🔄 计时 cache 策略

`commit` 用于单次正确性/smoke。重复计时必须使用 `overwrite`：外部输入、位置和
写映射保持固定；每轮都向同一槽位执行完整 cache 写入，驻留 cache/state 不在轮次之间复位。
这保留了 store 和调度工作量，但从第二轮开始属于合成的、会修改状态的热重复。

> [!CAUTION]
> 🔴 `overwrite` **不代表** serving 序列推进、commit/rollback，也不是经过 golden 校验的回放。

已完成的验证阶梯为：

~~~text
one HCA layer
one CSA layer
first two Pro layers
four layers including both attention kinds
16 layers
31 layers
61 layers
~~~

七个深度均使用一个驻留 worker 完成请求的重复网格；完整图可装入一个目标 die；
直接耗时与重构保留显式 residual；所有结果均标记为主模型计算代理，而不是 EP128 端到端。
这些结果在明确声明的边界内满足里程碑 6。独立 MTP-D3 serving 周期属于里程碑 7。

## 🟠 里程碑 7：MTP-D3 服务周期与报告（待实施）

只有在主模型 B=2/S=4 验证路径稳定之后，才开始 MTP 工作。

### 工作内容

- 验证一次完整 MTP 迭代，包括所选测量边界内的每个操作。
- 定义 draft 接受和拒绝时的缓存 commit 与 rollback 行为。
- 使用一个学习得到的 MTP 层实现三次串行 draft 迭代。
- 将产生的四行请求布局送入主模型验证。
- 测量集成墙钟时间，并将其与三次独立测量 MTP 迭代加主模型验证之和比较。
- 同时按周期和 accepted token 数量归一化服务结果。

### 最终报告

至少发布以下表格：

1. 16K 和 128K 下的独立 operator 延迟
2. 加权主模型重构与实测 decode_fwd 墙钟时间
3. 单次 MTP 与三轮迭代 draft 延迟
4. 完整服务周期延迟与 accepted-token 归一化
5. EP128 计算代理与实测 EP2/4/8 通信
6. 按权重和缓存家族划分的单 die HBM 分配

每一行都必须标明平台、commit、B、S、上下文、active token 数、deployment EP、
物理 communication EP、缓存策略，以及该数值是实测还是外推。

### CI 提升

- 将完整 128K 和 61 层性能运行排除在广泛的逐提交 sweep 之外。
- 为配置、元数据、叶子 operator 和代表性 HCA/CSA 边界添加小型编译与 golden 用例。
- 根据新目录路径触发专用设备基准。
- 只为内存和指令集受支持的用例保留模拟器覆盖。

## 📊 基准流程

> [!TIP]
> 🟣 官方性能数字统一采用“生成一次 golden → 冻结数据回放 → 重复计时 → L4
> trace 对账”的流程，避免把编译和输入生成计入 kernel 延迟。

对每个官方设备数值使用相同流程：

1. 记录仓库和依赖 pin。
2. 只生成一次输入和 torch golden。
3. 对 warmup 和测量迭代回放相同冻结数据。
4. 从 kernel 延迟中排除编译、输入生成、host-to-device 初始化和输出读回。
5. 报告 median、尾部离散范围和重复次数。
6. 采集对应的 level-4 L2 swimlane。
7. 将独立组件与组合墙钟时间进行核对。

代表性命令形式如下：

~~~bash
python models/deepseek/v4-pro-w8a8/gate.py -p a2a3sim --fixture tail-expert

task-submit --device auto --run \
  'python models/deepseek/v4-pro-w8a8/expert_routed.py -p a2a3 \
     -d $TASK_DEVICE --workload balanced --save-data --enable-l2-swimlane 4'

task-submit --device auto --run \
  'PYPTO_BENCH=1 python models/deepseek/v4-pro-w8a8/decode_attention_swa.py \
     -p a2a3 -d $TASK_DEVICE --start-pos 131072 \
     --golden-data build_output/SAVED_RUN/data'
~~~

使用命令前请检查每个脚本的 help；CLI 支持会随着各里程碑逐步添加并验证。

## 🔴 停止条件

- 🔴 不要将复制来的 Flash 行为报告为 Pro W8A8 行为。
- 🔴 不要将单 die 计算代理称为 EP128 端到端。
- 🔴 官方延迟不得使用发生别名的 ratio-4 或 indexer 历史。
- 🔴 不要在主模型和 MTP 总耗时中重复计入独立 SWA。
- 🔴 不要将单次 MTP 调用耗时的三倍报告为实测 draft-depth-three。
- 🔴 OOM 后不要静默缩小 batch、上下文或层数。
- 🔴 不要仅仅因为生成了性能 trace，就在 golden 失败的情况下继续推进。

## 🗂️ 建议的变更切片

按以下切片保持实现可审查：

1. 脚手架、配置合同、元数据测试与 CI 隔离
2. Gate、共享专家、路由专家与 SWA bring-up
3. HCA 128K 缓存与边界支持
4. CSA 128K indexer 与完整历史缓存支持
5. EP128 计算代理与独立通信测量
6. 缩减深度与 61 层主模型 forward
7. MTP-D3 runner、基准报告与 CI 提升

每个切片都必须让其新启用的入口点通过后，才能开始下一个切片。
