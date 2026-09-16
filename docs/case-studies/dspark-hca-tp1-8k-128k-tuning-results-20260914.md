# DSpark HCA TP1：8K / 128K 调优结果

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/dspark-hca-tp1-8k-128k-tuning-results-20260914.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

> 2026-09-15 合入状态：#1226（`f200ae1`）和 #1228（`1a9e487c`）均已合入 main。后续 TP1 128K 分析以 `1a9e487c` 为基线；本次仅确认合入并核对源码，没有新跑性能。此前 benchmark 数字仍对应原测量版本。按用户要求，下一步先分析 CMP 任务切分与调度，不修改 kernel、不启动实验。

> 2026-09-15 更新：#1226 已以 `f200ae1` 合入 main；#1228 已 rebase 到该 main 并改为直接向 main 提交，仍保留 CMP tile 和 UB merge/pack 两个独立 commit。8K/128K 编译检查及全量 pre-commit 通过，本轮没有重新上板或重测性能。main 还包含 #1212，下面的性能数字继续对应原先记录的测量基线。

> PR 分组已按用户要求更新：原 #1226 + #1227 合并到 [#1226](https://github.com/hw-native-sys/pypto-lib/pull/1226)，共 3 个独立 commit（编译前置修复 + 两项优化）；原 #1228 + #1229 合并到 [#1228](https://github.com/hw-native-sys/pypto-lib/pull/1228)，共 2 个独立优化 commit。#1227/#1229 已关闭，没有 squash。#1228 依赖新的 #1226 分支。下文逐项编号、原始 CI 状态和测量记录保留首次发布时的历史；当前审阅入口以上述两个 PR 为准。代码与原测量版本逐文件一致，本次只重组提交，没有重跑性能。

日期：2026-09-14。本轮执行 P0–P4 完成，四项优化分别已提交 PR，尚未合入。

## 测试口径

正式性能均为 **关闭全部 profiling 的 benchmark 中位数**。平台 a2a3，TP1，batch16，每请求本轮 8 个 decode token；16 个 start-pos 全部为 8192 或全部为 131072。每版本、每上下文单次进程，5 次 warmup，100 次有效采样，保留全部原始样本。所有真卡任务通过 task-submit 排队。

基线为 main `617e165f` 加共同的 TP1 首次 matmul FP32 累加器编译修复。原版在选定工具链上不能编译，因此编译修复不是性能收益。发布 PR 以 `c0ec3d1f` 为 main 基础；两版本之间 DSpark 和 golden 路径无变化。

工具链固定为 PyPTO `f1bb086`、simpler `15f5cbd`、PTOAS 0.57、PTO ISA `96ba706`、CANN 9.0.0。环境、完整版本、实际 runtime 配置、命令和源码补丁均在实验目录（本地证据：`build_output/hca_tp1_tuning_20260914_195536/`；未随文归档）中。HCA 复用原有 8K / 128K frozen golden，未放宽比较器。

卡 6 被占用后切换卡 12，并先补测卡 12 的原始基线及 P1+P2 基线。下面每组前后来自同一张卡，不把卡 6 与卡 12 的数字拼接。

## 已提交优化的增量

| PR | 修改及主要 kernel | 偶数卡 | 8K 中位数 / µs | 128K 中位数 / µs |
|---|---|---:|---:|---:|
| [#1226](https://github.com/hw-native-sys/pypto-lib/pull/1226) | 输出投影只处理有效 token 所在行块；`proj_b_mm`、`quant` | 6 | 701.46 → 603.96，13.90% | 1197.17 → 1104.44，7.75% |
| [#1227](https://github.com/hw-native-sys/pypto-lib/pull/1227) | 大规模压缩 KV gather 每 block 连续处理 8 页；`hca_cmp_work_gather` | 6 | 603.96 → 601.94，变化很小 | 1104.44 → 1038.99，5.93% |
| [#1228](https://github.com/hw-native-sys/pypto-lib/pull/1228) | TP1 CMP K tile 32 → 128，配套逐页 mask 与 gather 分组；`hca_cmp_qk_pv`、`hca_cmp_work_gather` | 12 | 613.96 → 581.11，5.35% | 1030.54 → 779.92，24.32% |
| [#1229](https://github.com/hw-native-sys/pypto-lib/pull/1229) | TP1 合并输出在 UB 完成 RoPE、BF16 与打包；`hca_stream_merge_pack` | 12 | 581.11 → 548.21，5.66% | 779.92 → 753.38，3.40% |

各行收益是相对已保留组合的增量，百分比不可直接相加。P2 在 8K 的约 2 µs 不足以证明稳定收益；128K 的分布双峰，卡 6 有 37/100 样本超过 1200 µs，全部保留。后续 128K 样本也有明显波动，本轮结论是用户指定的单次进程中位数改善，不代表长期尾延迟已经稳定。

## 完整组合，同卡 12

| 指标 | 8K | 128K |
|---|---:|---:|
| 基线，profiling 关闭，中位数 | 705.25 µs | 1197.98 µs |
| P1+P2+P3+P4 UB，中位数 | 548.21 µs | 753.38 µs |
| 降幅 | 22.27% | 37.11% |
| 基线，单次 chip AICore 端到端 | 706.90 µs | 1268.60 µs |
| 最终，单次 chip AICore 端到端 | 574.34 µs | 769.48 µs |

后两行开启 chip level 4，是单次捕获，不能称为 benchmark 中位数。P3 最终 128K 泳道图为 917.24 µs，而未限定 TP1 的 K128 原型为 800.16 µs；同时最终版本的关闭 profiling 中位数更好。两份记录均保留，没有重跑挑最快图。

## 每项实际减少了什么

P1：本例有效 token 为 128，容量仍为 512。输出投影的行 block 从 4 个缩为 1 个；全部 8 组 `proj_b_mm` 逻辑 block 从 256 减少为 64。只清零最后一个有效行 tile 的尾部。生成 matmul 内核本身不变，改变的是编排的工作数量。

P2：原 128K 有 512 个很短的 gather block。每 block 顺序搬运 8 页后降到 64 个，保留逐页地址映射和无效页行为，8K 仍采用原来 32 个 block。卡 6 的 gather 执行窗口由 225.78 缩短到 34.68 µs；同期 CMP 窗口没有缩短，不能把本项收益归因于 CMP 计算。

P3：128K 每 query 的 K 循环由 32 次减少为 8 次，减少每块 QK/PV 交换、softmax 更新及 AIC/AIV 同步次数。每块跨 4 个 cache 页，因此增加逐页有效性 mask；无效首个页不能导致后面有效页被整块跳过。gather 分组保持每 block 8 个 cache 页，避免并行度随 K tile 无意变化。只对 TP1 使用 K=128，TP4 保持 K=32。

P3 编译器分配：K64 / K128 的 Mat 分别为 139264 / 212992 B，Vec 为 140160 / 148608 B，Acc 均为 131072 B。均通过编译器容量检查；这不是用 DSL shape 推测的占用。内存图保存在对应原型目录。

P4：原 merge 每轮把 16×512 FP32 结果写到 GM，又读回来做 RoPE；之后逐 head 写 16 次 BF16 输出。新版本在 UB 保留合并结果，用扁平 gather 索引旋转 RoPE 维度，将 16 个 head reshape 为两个完整输出投影分组，一组写一次。每轮减少 64 KiB 的 FP32 GM 往返，最终 BF16 输出字节数不变。生成代码 TSTORE 调用位置从 17 个减至 2 个，TLOAD 从 10 个减至 9 个。清除注释及自动 inline 编号后，仅 merge 的 C++ 改变，另外 31 个 kernel 一致。

P4 的 merge block 数保持 48：8K 执行窗口 73.10 → 35.14 µs，128K 67.72 → 42.40 µs。最终实现没有叠加系数提前归一化。

## 未采用的候选及错误记录

| 候选 | 8K / 128K 中位数，µs | 取舍 |
|---|---|---|
| P2 每 block 4 页 | 609.82 / 1052.22 | 相比 8 页候选较差，保留补丁，不提交 |
| P3 K64 | 568.06 / 826.44 | 8K 优于最终 K128，但 128K 较慢；本轮保留两配置均改善且长上下文更好的 K128 |
| P3 K128 未限定 TP1 原型 | 592.53 / 788.18 | 最终限定到 TP1 后另行编译测量；原型不发布 |
| P4 单独提前归一化 | 577.12 / 774.12 | 相对 581.11 / 779.92 变化小于 1%，接近样本波动，暂不提交 |

P4 的首次 UB 改写遇到两个已知 DSL 使用限制：Tile 切片不能使用动态下界；Tensor 下标赋值不能接收 Tile。已改成两个常量分组切片和显式 pl.store，属于本次迁移写法错误，不是新的编译器问题。原始失败日志保留在 `p4_ub_merge/attention_boundary_*_error.log`。

## 精度与检查边界

- 所有保留组合的 8K、128K HCA 都通过原有精度校验。
- 压缩 attention 直接用例覆盖连续页、碎片页、负页号、越界页、零历史、部分页、不同长度、无效首个页后仍有效的后续页，以及 gather 分组尾部。
- P1 是共享输出投影改动，SWA / CSA TP1 也通过原有精度。CSA 需要已有的 512 MiB ring2 heap 容量配置，不属于性能优化。
- P1 直接输出投影的 8/120/128/136/512 行用例，前后输出逐位一致。但新随机用例的基线与候选都不能满足从完整 HCA 借来的 torch 阈值，不能称它们通过 torch 精度。完整 attention 的精度标准未改变。
- P3 完成 TP4 编译检查；没有 TP4 真卡性能结论，也不宣称生成代码逐字相同。
- 四个发布工作区全量 pre-commit 通过，tracked 状态干净；没有提交 build_output 或本地记录。

## PR 依赖及 CI

建议依次处理 #1226 → #1227 → #1228 → #1229。#1226 含独立的编译前置修复 commit 和有效行优化 commit，编译修复不单独声称性能收益。其他 PR 各一项性能 commit。

#1228 以 #1227 的上游同名分支为 base，#1229 以 #1228 的上游同名分支为 base，保持各 PR diff 只含自身优化；前置 PR 合入后需 retarget 到 main。已推送 commit，没有自动合入。

截至本轮结束，#1226/#1227 的 a2a3sim 因已知 FFTS intrinsic 不可用而失败，a2a3 真卡 CI 已通过，serving-dspark 仍在执行；[PTOAS #1513](https://github.com/hw-native-sys/PTOAS/issues/1513) 已有记录，本地 KNOWN_PYPTO_ISSUES 追加复现证据。未禁用 simulator 测试。功能分支 base 的 #1228/#1229 仅有文档构建等检查，尚无模型测试 CI，不能标注完整验证通过，retarget 后需要完整检查。

## 泳道图和原始数据

| 配置 | 基线，卡 12 | 最终，卡 12 |
|---|---|---|
| 8K | chip records（本地证据：`build_output/hca_tp1_tuning_20260914_195536/chip_swimlanes/baseline_d12/8192/chip_swimlane_records.json`；未随文归档） | chip records（本地证据：`build_output/hca_tp1_tuning_20260914_195536/chip_swimlanes/p4_ub_merge/8192/chip_swimlane_records.json`；未随文归档） |
| 128K | chip records（本地证据：`build_output/hca_tp1_tuning_20260914_195536/chip_swimlanes/baseline_d12/131072/chip_swimlane_records.json`；未随文归档） | chip records（本地证据：`build_output/hca_tp1_tuning_20260914_195536/chip_swimlanes/p4_ub_merge/131072/chip_swimlane_records.json`；未随文归档） |

四个目录都包含 CPM_static.json、CPM_observed.json、critical_path_summary.md，以及完整 chip 数据和依赖文件。中间各项也保留在同级 `chip_swimlanes/<variant>/<context>/`，可查看单项前后。

22 组 benchmark 汇总（本地证据：`build_output/hca_tp1_tuning_20260914_195536/benchmark_summary.json`；未随文归档）包含每组设备、中位数、四分位数和原始样本路径。`<variant>/<context>_bench/benchmark.json` 保留 100 个 full-precision 样本；`result.json` / `runtime_config.json` 保存真实配置与精度状态。`run_case.py`、`run_batch.py`、`run_batch.sh`、`env.sh` 保留实际运行方式。重新实验请新建结果目录，避免覆盖原始证据。
