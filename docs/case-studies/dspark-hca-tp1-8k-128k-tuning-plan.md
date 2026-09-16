# DSpark HCA TP1：8K / 128K 性能调优计划

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/dspark-hca-tp1-8k-128k-tuning-plan.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

> 2026-09-15 后续实验完成：以 `1a9e487c` 在偶数卡 12 测试 CMP 每 block 处理 2 / 3 个连续 query。128K 无 profiling 中位数为 747.64 → 743.39 / 745.49 µs，仅降低 0.57% / 0.29%；8K 为 545.10 → 543.77 / 544.94 µs。六组 benchmark 及两候选的 attention 边界精度通过。2 query 的 CMP AIC block 达到 59.54–75.66 µs，但尚无明确的稳定中位数收益，保留实验，不提交 PR。详见 [实验记录](dspark-hca-tp1-cmp-split-results-20260915.md)，产物位于 `build_output/hca_tp1_cmp_split_20260915_203011/`。

> 2026-09-15 合入状态：#1226（`f200ae1`）和 #1228（`1a9e487c`）均已合入 main。后续 TP1 128K 分析以 `1a9e487c` 为基线；本次仅确认合入并核对源码，没有新跑性能。此前 benchmark 数字仍对应原测量版本。按用户要求，下一步先分析 CMP 任务切分与调度，不修改 kernel、不启动实验。

> 2026-09-15 更新：#1226 已以 `f200ae1` 合入 main；#1228 已 rebase 到该 main 并改为直接向 main 提交，仍保留 CMP tile 和 UB merge/pack 两个独立 commit。8K/128K 编译检查及全量 pre-commit 通过，本轮没有重新上板或重测性能。main 还包含 #1212，下面的性能数字继续对应原先记录的测量基线。

> PR 分组已按用户要求更新：原 #1226 + #1227 合并到 [#1226](https://github.com/hw-native-sys/pypto-lib/pull/1226)，共 3 个独立 commit（编译前置修复 + 两项优化）；原 #1228 + #1229 合并到 [#1228](https://github.com/hw-native-sys/pypto-lib/pull/1228)，共 2 个独立优化 commit。#1227/#1229 已关闭，没有 squash。#1228 依赖新的 #1226 分支。下文逐项编号、原始 CI 状态和测量记录保留首次发布时的历史；当前审阅入口以上述两个 PR 为准。代码与原测量版本逐文件一致，本次只重组提交，没有重跑性能。

日期：2026-09-14
状态：本轮 P0–P4 已完成。四项保留优化分别提交 PR #1226、#1227、#1228、#1229；归一化单独候选未提交。详见第 7 节和[完整结果](dspark-hca-tp1-8k-128k-tuning-results-20260914.md)。

## 1. 目标与范围

降低 HCA 在 TP1、8K 和 128K 两种上下文下的端到端耗时，保留原有数值校验标准和输入语义。

- 主要验收指标：**关闭全部 profiling 后，benchmark 的中位数**。
- 辅助证据：开启 chip swimlane perf level 4 后的 AICore 端到端时间、关键路径、block 数和执行窗口。
- 两种上下文分别报告结果，不把某一个配置的收益推广到另一个配置。
- 本阶段不设未经对齐的绝对耗时目标，不将此前 TP4、短上下文的目标直接套到 TP1。
- 当前入口为 `models/deepseek_v4_flash_dspark/decode_hca.py --tp 1`。
- 首先处理明确的无效工作和过细的工作划分，再考虑更复杂的融合与调度。

第 4 节保留实验前的计划；是否实现、实测收益和最终取舍以第 7 节及完整结果为准。

## 2. 固定测试口径

| 项目 | 固定值或规则 |
|---|---|
| 平台 | `a2a3` |
| TP | 1 |
| batch | 总共 16 个请求 |
| 每请求本轮 token 数 | 8，因此本轮共有 128 个查询 token |
| 8K 上下文 | `--start-pos` 传 16 个 `8192` |
| 128K 上下文 | `--start-pos` 传 16 个 `131072` |
| 设备 | 通过 `task-submit` 分配偶数卡，优先沿用卡 6；同一组前后必须使用同一张卡 |
| benchmark | `PYPTO_BENCH=1`、`PYPTO_BENCH_RAW=1`，5 次 warmup、100 次有效采样 |
| profiling | 正式 benchmark 时 chip swimlane、dep-gen、PMU、参数 dump、scope stats 全部关闭 |
| 泳道图 | 单独采集，`PYPTO_BENCH=0`、`--enable-chip-swimlane 4` |
| 精度 | 原有 golden 和比较器，不放宽阈值 |

这里测的是已有历史上下文后的 decode，不是一次 prefill 8192 或 131072 个 token。`--start-pos` 列表长度控制请求数，不能改成单个标量，否则会变成 batch=1。

每个版本、每个上下文只启动一次 benchmark 进程，取其进程内 100 个采样的中位数。不要通过反复启动脚本挑最好的一次，不改用最小值或平均值，不自行丢弃不利样本。发生运行失败、配置变化或有明确的环境干扰证据时，记录原因后再决定是否补测。

TP1 只有一个 rank，直接报告该 rank 的中位数。若换卡或换工具链，应在新的条件下建立配套基线，不能将不同卡上的数字拼成前后收益。

## 3. 基线来源与现有结果

### 3.1 代码与工具链

- pypto-lib：main `617e165fe45dc1b595bfdd0aa998fd500ceda473`。
- 基线额外包含一项必要的 TP1 编译修复：`proj_a_mm` 用显式 FP32 的首次 `pl.matmul` 初始化累加器，再按原顺序累加其余 K 块。
- 原版 main 在所选工具链上触发 `AccCompactValid`。因此当前数字应标注为 **main + 编译修复**，不能称为完全未修改的 main。
- 补丁保存在两个基线目录的 `tp1_projection_compile_fix.patch` 中；后续实验共同保留该补丁，不把它计为性能收益。
- PyPTO：`f1bb0860885247ecdcaf2ccddcc45c9f1eb14382`。
- simpler：`15f5cbd922494c62444e75fa1585e39f86a64a78`。
- PTOAS：0.57；PTO ISA：`96ba706ce1697dd5febe107ee41a72b26e687b42`；CANN：9.0.0。
- Python：3.10.19；torch：2.6.0+cpu；golden / OMP / MKL / OpenBLAS 线程数均为 8。

执行时复用基线保存的环境配置并核对版本。基线 `env.sh` 会设置 `PYPTO_BENCH=0`，因此 benchmark 开关必须在加载环境后再显式设置。

### 3.2 现有泳道图结果

两组均使用偶数卡 6，HCA 原有精度校验均通过。

| 指标 | 8K | 128K |
|---|---:|---:|
| AICore 端到端时间 | 722.14 µs | 1268.54 µs |
| dispatch → finish | 726.78 µs | 1273.38 µs |
| Static CPM | 577.92 µs | 1146.22 µs |
| 关闭 profiling 的 benchmark 中位数 | 待测 | 待测 |

以上已有耗时来自各一次开启 profiling 的捕获，不是关闭 profiling 后的中位数。

| Kernel / 阶段 | 8K 执行窗口 | 128K 执行窗口 | 8K / 128K block 数 |
|---|---:|---:|---:|
| `hca_cmp_work_gather` | 9.22 µs | 219.10 µs | 32 / 512 |
| `hca_cmp_qk_pv` | 54.70 µs | 470.42 µs | 24 / 24 |
| `proj_b_mm`，全部 8 组 | 154.64 µs | 160.94 µs | 256 / 256 |
| `hca_stream_merge_pack` | 69.26 µs | 67.88 µs | 48 / 48 |
| `hca_raw_attn` | 38.62 µs | 40.56 µs | 20 / 20 |
| `hca_gather_kv` | 9.90 µs | 10.40 µs | 16 / 16 |

表内 block 数为逻辑 SPMD block 数。MIX kernel 的窗口覆盖其 AIC 和 AIV 记录；`proj_b_mm` 覆盖 8 个分组任务。阶段可能重叠，不能将窗口直接求和作为端到端耗时。

### 3.3 文件入口

- 8K HCA 泳道图（本地证据：`build_output/dspark_tp1_b16_ctx8k_617e165f_20260914_184939/chip_swimlanes/hca/chip_swimlane_records.json`；未随文归档）
- 128K HCA 泳道图（本地证据：`build_output/dspark_tp1_b16_ctx128k_617e165f_20260914_192415/chip_swimlanes/hca/chip_swimlane_records.json`；未随文归档）
- 8K 完整关键路径、等待与阻塞证据（本地证据：`build_output/dspark_tp1_b16_ctx8k_617e165f_20260914_184939/chip_swimlanes/hca/critical_path_summary.md`；未随文归档）
- 128K 完整关键路径、等待与阻塞证据（本地证据：`build_output/dspark_tp1_b16_ctx128k_617e165f_20260914_192415/chip_swimlanes/hca/critical_path_summary.md`；未随文归档）
- 两种上下文的逐 kernel 对比数据（本地证据：`build_output/dspark_tp1_b16_ctx128k_617e165f_20260914_192415/hca_context_comparison.json`；未随文归档）
- 8K 已保存的输入与 golden（本地证据：`build_output/dspark_tp1_b16_ctx8k_617e165f_20260914_184939/source/build_output/_jit_decode_hca_tp1_test_20260914_185849/data`；未随文归档）
- 128K 已保存的输入与 golden（本地证据：`build_output/dspark_tp1_b16_ctx128k_617e165f_20260914_192415/source/build_output/_jit_decode_hca_tp1_test_20260914_192535/data`；未随文归档）

两个 HCA 泳道图目录均已有 `CPM_static.json` 和 `CPM_observed.json`；128K 的文件也已同步到原始 `dfx_outputs` 目录。

## 4. 实验顺序

### P0：补齐关闭 profiling 的基线

1. 建立独立实验工作区，固定上述代码、编译修复及工具链，保留现有基线目录。
2. 分别复用 8K、128K 对应的输入和 golden，不能交叉使用缓存。
3. 每个上下文执行一次 5 / 100 benchmark，记录全部原始采样及中位数。
4. 核对实际配置中的全部 DFX 开关，而不是仅凭命令行没有写泳道图参数判断。
5. 记录设备、版本、输入来源和命令，作为后续各项实验的可复现基线。

源码或相关形状变化时重新编译。只有编译产物与当前源码、规格兼容时才使用 `runtime_dir`；输入和 golden 在计算语义不变时继续复用。

### P1：`proj_b_mm` 只处理有效 token 所在的行块

**优先级最高：明确减少工作量，同时覆盖 8K 和 128K。**

位置：`models/deepseek_v4_flash_dspark/decode_o_proj.py` 的 `decode_o_proj_tp1`。

现状：运行时 `t_dim=128`，但 `T_PAD=512`。`proj_b_mm` 仍执行 4 个 128 行块；后 384 行来自补零。

拟改动：

- 将行块数量限制为 `ceil(t_dim / PROJ_B_MM_T_TILE)`。
- `quant` 只清零最后一个有效行块的必要尾部，不再清零其后的所有容量行。
- 首次实验保留 tensor 容量、布局、matmul tile 形状和现有依赖关系，只改变实际工作范围。
- 检查所有下游消费者，保证缩减清零后不会读取未初始化的无效行。

本配置下预计 block 数由 `8组 × 4行块 × 8列块 = 256` 降为 `8组 × 1行块 × 8列块 = 64`。这表示该 matmul 的行计算量减少 75%，**不表示端到端性能提升 75%**。

验证：

- 用输出投影的直接用例覆盖有效行数 8、120、128、136、512，检查不足一个 tile、整 tile、跨 tile 和满容量。
- HCA 的 8K、128K 完整精度与 benchmark 均需通过。
- 该函数是共享 TP1 代码，提交前对受影响的 SWA、CSA TP1 做必要的精度回归；不扩展为全模型性能调优。
- 泳道图核对 `proj_b_mm` block 数、`quant` 工作范围及输出投影尾段是否缩短。

### P2：合并 `hca_cmp_work_gather` 的小 block

**主要面向 128K，保持压缩注意力计算粒度不变，以单独验证 gather 收益。**

位置：`models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py` 的压缩 KV gather。

当前 HCA 压缩比例为 128，页面大小为 32 行：

| 配置 | 每请求压缩 KV 行数 | 每请求页数 | 16 请求的 gather block 数 |
|---|---:|---:|---:|
| 8K | 64 | 2 | 32 |
| 128K | 1024 | 32 | 512 |

128K 的 gather 单 block 中位数只有 2.28 µs，最大 6.92 µs，但整体窗口是 219.10 µs。优先假设是小 block 数量和分批执行成本过高；现有图不能证明纯粹是 GM 带宽不足。

拟实验：

- 先保持 `CMP_ATTN_K_TILE=32`，测试每 block 连续处理 4 页、8 页。
- 128K 的逻辑 block 数分别尝试降至 128、64。
- 保持现有输出布局与页映射语义；检查尾部不足一组时的边界。
- 8K 可保留原有小规模路径，避免仅剩少量 block 导致并行度下降。
- 暂不同时修改零填充、QK/PV tile 或调度依赖，便于归因。

验证覆盖：连续页、非连续页、无效页、页数不足分组长度和混合请求长度。不能因为当前性能样例的页号连续，就把通用路径改为无条件连续搬运。

### P3：增大长上下文的 CMP K tile

**主要压缩 128K 的 QK / softmax / PV 循环及其内部数据交换。**

位置：`decode_sparse_attn_hca.py` 的 `CMP_ATTN_K_TILE` 及相关 gather、buffer、循环和 mask。

| K tile | 128K 每 query 的 KV 分块数 |
|---|---:|
| 当前 32 | 32 |
| 候选 64 | 16 |
| 候选 128 | 8 |

每块当前都涉及 QK scores、probabilities、PV 输出的 GM 交换和 AIC/AIV 同步，并更新 online-softmax 状态。较大的 tile 可能摊薄这些开销，但会增加片上 buffer 压力，也会改变数值归约分组。

执行约束：

- 先检查编译器的 UB / L1 / L0C 占用、自动切分及是否出现 spill，再上卡。
- 保持 raw attention 的 tile 不变。
- 检查最后一块有效行数、mask、无效页以及同步次数是否配对。
- K tile 同时影响 gather 工作划分，必须记录与 P2 的交互，不能把全部变化归因于 QK/PV。
- 分别验证 8K 和 128K。若最终采用短、长上下文不同策略，需要补临界长度和混合请求长度测试。
- 如果收益仍受内部等待限制，再做必要的 kernel 内部 profiling；不要直接将整个执行窗口称为纯 Cube 计算。

### P4：重新评估 TP1 merge/pack 与调度

仅在前述实验完成、重新确认关键路径后推进。

- 当前 TP1 是 `hca_stream_merge_pack`，与 TP4 的 `hca_stream_merge_pack_publish` 是不同实现。
- 候选包括：系数提前归一化、UB 内打包、减少实际存在的 GM 往返；每次只验证一项。
- 先核对生成代码，确认源码中的 store/reload 是否仍产生真实搬运及是否具有必要的副作用。
- 当前窗口约 68–69 µs，且没有随上下文明显增长，因此排在前面三项之后。
- `hca_raw_attn` 与 `hca_gather_kv` 当前分别约 40 µs、10 µs，也先不作为主攻方向。
- 只有关键路径显示新的显著启动间隙时，再研究 early dispatch 或依赖调整。

## 5. 收益判定与提交规则

每个候选版本完成以下闭环：

1. 记录它相对哪个基线修改了什么。
2. 编译检查及有针对性的边界精度检查通过。
3. 在同一组测试条件下取得 8K、128K 的无 profiling benchmark 中位数。
4. 对保留候选补前后泳道图，解释它改变了哪个 kernel、block 数或关键路径。
5. 按下表分别给出结论，再决定是否进入最终组合。

| 结果 | 处理 |
|---|---|
| 精度通过，两种上下文中位数均下降 | 保留候选，并记录两种配置各自收益 |
| 仅长上下文有收益，短上下文退化 | 尝试有明确边界的长上下文策略，再验证分支与混合长度；不能直接作为通用优化保留 |
| 仅泳道图或 busy time 变好，中位数没有收益 | 记录为实验结果，不按正式性能收益提交 |
| 中位数下降，但泳道图变差 | 同时写明两种结果；按无 profiling 指标评估，不能称两个指标都改善 |
| 改变量很小或样本波动明显 | 如实标注证据有限，不宣称已经证明稳定收益 |
| 精度失败或中位数回退 | 保存补丁与结果，退出最终组合 |

降幅统一按 `(基线中位数 - 候选中位数) / 基线中位数` 计算。

- 不同优化使用不同 commit；同一优化内必要的配套修改可归为一个 commit。
- 编译前置修复与性能优化分开标注，前置修复不能伪装为有实测收益的优化 commit。
- 每个性能 commit 应有明确基线、配置、精度结果及中位数数据。
- 叠加时比较“已保留组合”与“该组合 + 新改动”，独立测试的收益不能直接相加。
- PR 描述保留逐 commit 的增量数据及最终组合数据；生成产物和本地实验记录不提交到仓库。

## 6. 避坑清单

开始实现前先查阅 [历史调优记录](dspark-hca-decode-tuning-log.md)，尤其注意其结论对应的版本和配置。

1. 历史记录主要来自 TP4、context=256；当前是 TP1、8K/128K，不能直接照搬收益与瓶颈排序。
2. 历史记录开头关于“小于 16K 时 cmp_work_count=1”的说法对应旧实现。当前 K tile 是 32，8K 实际有 2 个 work，128K 有 32 个。
3. 历史 ABBA 多次启动、挑选或排除部分测试的做法不作为本计划默认流程；遵循当前“一次 benchmark 进程，比较中位数”的口径。
4. 区分整个 kernel 窗口和单 block 时间。512 个短 block 的长窗口，不等于某一次搬运本身耗时 219 µs。
5. 区分启动前等待和 kernel 内部等待。128K CMP 的前驱结束到开始约 11.38 µs，kernel 窗口约 470.42 µs；后者仍可能包含搬运与 AIC/AIV 同步，不能全算为矩阵乘时间。
6. 不能凭时间重叠就称某任务是阻塞者；没有依赖、同核占用或容量证据时保留“不确定”。
7. 历史直接删除 merge 的 GM 往返、对行切片 reshape，以及某些显式 gather 改动曾出现编译、精度或性能问题。只有新的生成代码证据支持时才重新尝试。
8. 历史 CMP 直接存储方案在后续版本曾失去收益；不要将旧 commit 的成绩作为当前基线的增量收益。
9. 编译修复已明确要求首次 matmul 输出 FP32。不要在本轮性能实验中顺便改动该前置条件。
10. 不为取得性能数字放宽精度。若发现疑似工具链问题，按仓库流程记录；不悄悄修改工具链或比较器绕过。

## 7. 产物与进度

所有实验放到新的 `build_output/` 子目录，保留独立源码、补丁、版本、命令、配置、设备信息、输入来源、精度日志和 benchmark 原始采样。

每个用于展示的 chip 泳道图目录应同时保存 records、deps、name map、merged trace、`CPM_static.json` 和 `CPM_observed.json`。比较前校验 level 4、原始 AICore/AICPU 与合并记录数，以及依赖声明的 block 数一致。

| 阶段 | 偶数卡 | 8K 中位数前 → 后 | 128K 中位数前 → 后 | 精度 | 结论 |
|---|---|---|---|---|---|
| P0：关闭 profiling 的基线 | 6 | 701.46 µs | 1197.17 µs | 两组原有校验通过 | 完成 |
| P1：有效行范围 | 6 | 701.46 → 603.96 µs | 1197.17 → 1104.44 µs | HCA、SWA、CSA 通过；5 个行数边界前后逐位一致 | [PR #1226](https://github.com/hw-native-sys/pypto-lib/pull/1226) |
| P2：gather 每 block 8 页 | 6 | 603.96 → 601.94 µs | 1104.44 → 1038.99 µs | HCA 两组及页边界通过 | [PR #1227](https://github.com/hw-native-sys/pypto-lib/pull/1227)，8K 收益不确定 |
| P3：TP1 CMP K tile=128 | 12 | 613.96 → 581.11 µs | 1030.54 → 779.92 µs | HCA 两组及页边界通过 | [PR #1228](https://github.com/hw-native-sys/pypto-lib/pull/1228) |
| P4：TP1 UB merge/pack | 12 | 581.11 → 548.21 µs | 779.92 → 753.38 µs | HCA 两组及页边界通过 | [PR #1229](https://github.com/hw-native-sys/pypto-lib/pull/1229) |
| P4：单独提前归一化 | 12 | 581.11 → 577.12 µs | 779.92 → 774.12 µs | HCA 两组及页边界通过 | 变化接近样本波动，保存但不提交；没有叠加到 UB 版本 |

卡 12 的完整组合对比：基线含共同编译修复，8K **705.25 → 548.21 µs（22.27%）**，128K **1197.98 → 753.38 µs（37.11%）**。均为关闭全部 profiling 的 benchmark 中位数，不与卡 6 拼接。

执行顺序：**P0 → P1 → P2 → P3 → 重新读关键路径 → 决定是否进入 P4**。

## 执行记录（2026-09-14）

实验目录：`build_output/hca_tp1_tuning_20260914_195536/`（本地证据：`build_output/hca_tp1_tuning_20260914_195536/`；未随文归档）。

P0/P1：同一任务分配的偶数卡 6，TP1、batch16、每请求 8 token；每配置单进程 5 warmup / 100 samples，实际 runtime 配置确认全部 DFX 关闭，复用对应 8K/128K golden。原始样本见各版本的 `8192_bench/benchmark.json` 和 `131072_bench/benchmark.json`。

P1 的新建直接输出投影用例在 8 行时与 torch 的比较未通过（1.1322% 超差点，阈值 0.8%）。已用相同输入运行基线，结论见下段；未放宽 HCA 原有精度阈值。

P1：直接输出投影的新随机用例在五个行数上，基线与候选均不满足从完整 HCA 借来的严格 torch 阈值；但五组前后输出全部逐位一致，未引入新的数值差异。完整 HCA、SWA、CSA 用例仍使用原比较器并通过。原始失败与精确对比均保存在实验目录，不能将直接用例称为 torch 精度通过。

P2：4 页候选的 8K/128K 中位数为 609.82/1052.22 µs；保留 8 页候选。8K 的约 2 µs 变化在样本波动内，不宣称稳定收益。128K 的 100 个样本有 37 个超过 1200 µs，分布呈双峰，所有样本均保留；PR 明确仅报告中位数收益。

P1/P2 新泳道图位于实验目录 `chip_swimlanes/{baseline,p1_active_rows,p2_gather8}/{8192,131072}/`，包含 CPM_static/observed。P1 的 proj_b_mm 逻辑 block 数为 256 → 64；P2 的 128K gather 为 512 → 64，窗口为 225.78 → 34.68 µs。

P3：因卡 6 被其他任务占用，切换到偶数卡 12；先重建卡 12 的 baseline 与 P1+P2 基线，后续结果不得与卡 6 拼接。64/128 tile 已完成编译和内存检查：Mat 为 139264/212992 B，Vec 为 140160/148608 B，Acc 均为 131072 B，均在编译器容量限制内。较大 tile 配套逐页有效性 mask，避免将首个无效页之后的有效页跳过。

P3：保留仅 TP1 使用 K=128 的最终版本 `p3_tp1_k128`；TP4 保留 K=32，并完成编译检查，没有 TP4 性能结论。K=64 候选为 568.06/826.44 µs，8K 更快但 128K 较慢，未提交该替代候选。

P4：UB 版本只改变 `hca_stream_merge_pack.cpp`；去除注释和自动生成 inline 编号后，其余 31 个生成 kernel 均一致。生成代码的 TSTORE 调用位置由 17 处降到 2 处，TLOAD 由 10 处降到 9 处。对应每轮去掉一次 16×512 FP32 的 GM 往返，并将 16 次单 head 写合并为 2 次完整分组写。归一化算法保持原样。

最终泳道图在实验目录 `chip_swimlanes/p4_ub_merge/{8192,131072}/`，同卡完整基线在 `chip_swimlanes/baseline_d12/{8192,131072}/`；四个目录均有 records、CPM_static 和 CPM_observed。单次 AICore 端到端时间为 706.90 → 574.34 µs、1268.60 → 769.48 µs；这组不是 benchmark 中位数。

PR 建议顺序为 #1226 → #1227 → #1228 → #1229。#1228 的 base 是 #1227 的上游同名分支，#1229 的 base 是 #1228 的上游同名分支，以保证每个 PR 只展示自身优化；前置 PR 合入后需要 retarget 到 main。没有自动合入。

CI 状态：#1226/#1227 的 a2a3sim 遇到已知 FFTS intrinsic 不可用问题，见 PTOAS #1513；a2a3 真卡 CI 已通过，serving-dspark 尚在执行。#1228/#1229 当前以功能分支为 base，仅有文档构建等检查，尚无模型测试 CI，retarget 后仍需正常 CI。不得将本地真卡通过描述为 CI 全绿。
