# `attn_norm_apply` 长短 Kernel 调查记录

## 问题背景

在 `qkv_proj_rope` 的 L2 swimlane 中，`attn_norm_apply` 的不同实例时长差异很大，表现为有些 kernel 明显更长，有些更短。

本次主要分析的泳道图为：
- `build_output/_jit_qkv_proj_rope_test_20260520_091254/dfx_outputs/merged_swimlane_20260520_091300.json`

## 基线现象

从源码看，`attn_norm_apply` 的静态分工是均匀的：
- 文件：`models/deepseek/v4/qkv_proj_rope.py`
- 启动方式：`for db in pl.parallel(0, D_BLOCKS, 1)`
- 每个任务处理一个 `D_CHUNK`
- 当前配置下，`D_BLOCKS = 32`，`D_CHUNK = 128`

因此从源码层面的工作量划分看，32 个 `attn_norm_apply` 任务应当是同形、同量的。

但在基线泳道图中，`attn_norm_apply` 并不是小范围随机抖动，而是呈现稳定的“长/短分层”现象。

基线 trace 中，`pid == 2` 的代表性时长如下：

- `r2t2` / `CoreId:34` = `119.48us`
- `r2t3` / `CoreId:35` = `70.82us`
- `r2t4` / `CoreId:40` = `115.04us`
- `r2t5` / `CoreId:41` = `84.22us`
- `r2t6` / `CoreId:46` = `130.42us`
- `r2t7` / `CoreId:47` = `84.24us`
- `r2t8` / `CoreId:52` = `116.80us`
- `r2t9` / `CoreId:53` = `85.84us`
- `r2t10` / `CoreId:58` = `118.94us`
- `r2t11` / `CoreId:59` = `88.96us`
- `r2t12` / `CoreId:64` = `121.98us`
- `r2t13` / `CoreId:65` = `91.72us`

这说明需要区分两种可能：
- 时长主要跟着 `db` / 数据地址映射走
- 时长主要跟着物理 core / lane 走

## 代码库已有背景

仓库中原有的调优记录已经提到过这个 scope 的双峰分布：
- `docs/dsv4-qkv-proj-rope-perf-tuning.md`

那份记录主要把早期的双峰现象归因于与相邻 kernel 的 HBM 竞争。

但对 `20260520_091300` 这份 trace 来说，`qr_proj_matmul` 的启动时间晚于绝大多数 `attn_norm_apply` 的结束时间，因此这一次的长短差，不能简单用“与 `qr_proj_matmul` 直接重叠”来解释。

## 实验目标

核心思路是扰动“逻辑块编号 `db`”和“任务发射顺序”之间的对应关系。

判断原则：
- 如果长短跟着 `db` 走，更像是数据地址 / HBM 映射问题
- 如果长短固定绑在某些 core 上，更像是物理 lane 问题

## 实验 1：奇偶拆分，使用 `pl.parallel(start=1, step=2)`

尝试的源码形态：
- 第一段处理偶数 `db`
- 第二段处理奇数 `db`

实际观察到的 lowering 结果：
- 前半段保持为并行的 `attn_norm_apply(r2t0..15)`
- 后半段被降级成单核串行链 `attn_norm_apply_0(r2t16..31)`

对应 trace：
- `build_output/_jit_qkv_proj_rope_test_20260520_113642/dfx_outputs/merged_swimlane_20260520_113647.json`

前半段最重要的结果如下：
- `CoreId:34` 和 `CoreId:35` 变成 `35.82us` / `35.54us`
- `CoreId:40` 和 `CoreId:41` 变成 `37.18us` / `37.12us`
- `CoreId:46` 和 `CoreId:47` 变成 `40.48us` / `40.56us`
- `CoreId:52` 和 `CoreId:53` 变成 `40.76us` / `40.72us`

这些 core 在基线里原本是明显的长/短对，但当它们都只处理偶数块后，时长几乎完全拉平。

### 解释

这是本次调查中最有力的一组证据。

长短属性并没有固定绑在物理 core 上：
- 原本偏“短”的 core，在逻辑块分配改变后，变得和相邻原本偏“长”的 core 基本一致

这更支持以下判断：
- 时长主要受 `db` / 地址映射影响
- 而不是某些物理 lane 天生更慢或更快

### 局限性

这个实验的后半段被 lowering 成串行执行，因此它不是一个完整的 32 路严格对照实验。

## 实验 2：两段标准并行域内做仿射映射

尝试的源码形态如下：

```python
for i in pl.parallel(0, D_BLOCKS // 2, 1):
    db = 2 * i
    ...

for i in pl.parallel(0, D_BLOCKS // 2, 1):
    db = 2 * i + 1
    ...
```

设计动机：
- 让两段循环都保持规范的 `start=0, step=1` 形式
- 避开 `pl.parallel(1, D_BLOCKS, 2)` 可能带来的 lowering 问题

结果：
- 编译期直接失败
- 错误落在 `optimize_orch_tensors_pass` 与 broadcast 相关 pass 上

对应 build 目录：
- `build_output/_jit_qkv_proj_rope_test_20260520_115102`

这说明：
- 当前编译器栈对 `attn_norm_apply` 这类模式的支持并不稳健
- 更具体地说，把同一个输出 tensor 的 `assemble` 拆成两段并行区，并在每段里做仿射重映射，容易触发编译器限制或 bug

## 失败的 reverse-order 尝试

还尝试过两种 reverse-order 方案。

### 方案一：模块级 tuple 查表

思路：
- 定义一个 `db_order` tuple
- 在循环中做 `db = db_order[idx]`

结果：
- JIT 报模块级映射对象未定义或不支持捕获

### 方案二：函数内局部 tuple 查表

思路：
- 在函数内部定义 tuple
- 在循环中做 `db = db_order[db_idx]`

结果：
- 编译器只支持常量整数索引 tuple
- 不支持用并行循环变量做动态 tuple 索引

## 当前结论

截至目前，最合理的结论是：
- `attn_norm_apply` 的长短现象，更符合“逻辑块 / 地址映射效应”，而不是“固定物理 lane 效应”

依据：
- 基线里相邻 core 的长/短对，在“只保留偶数块”的实验前半段中消失了
- 长短属性没有继续固定附着在同一组 core 上

最可能的机制是：
- `attn_norm_apply` 对内存比较敏感
- 不同 `db` 对应不同的 `d0 = db * D_CHUNK` 地址区间
- 这些地址区间很可能映射到不同的 HBM channel / bank 或相关内存拓扑
- 最终表现为不同执行带宽或竞争程度，从而形成稳定的长/短执行带

当前还没有被完全证明的部分：
- 精确的硬件机制是什么
- 到底是 channel 映射、bank 映射、cache 行为、调度放置，还是它们的组合

## 编译器 / DSL 层面的发现

这次调查也暴露了当前 PyPTO 工具链的一些限制：

- `pl.parallel(1, D_BLOCKS, 2)` 可能会被 lowering 成单核串行链，而不是保留并行
- 在这个上下文里，不支持基于 tuple 的动态重映射
- 把同一个输出 tensor 的 `assemble` 拆成两段并行区，并使用仿射重映射索引，可能触发编译失败

这些限制对后续实验设计很关键，因为它们会影响“只改块顺序、不改数学工作量”的实验能否成立。

## 建议的下一步

后续实验应尽量避开上述 lowering / 编译边界。

更稳妥的方向包括：

1. 保持 `for db in pl.parallel(0, D_BLOCKS, 1)` 不变，只调整 `D_CHUNK`。
   - 这样可以改变地址步长和块数，但不引入非规范循环结构。

2. 保持 kernel 结构不变，只扰动 tensor 布局或地址对齐。
   - 这样更容易直接验证长短是否跟物理地址映射走。

3. 检查 `attn_norm_apply` 的 lowered IR / pass dump。
   - 目标是看清楚为什么 `pl.parallel(1, D_BLOCKS, 2)` 会被串行化，以及为什么两段仿射映射版本会编译失败。

4. 如果条件允许，在不同设备 / 不同卡上做对比。
   - 这样可以区分是设备特定的内存拓扑效应，还是源码层面的普遍现象。

## 本次调查产生的产物

相关 trace / 日志如下：
- baseline: `build_output/_jit_qkv_proj_rope_test_20260520_091254/dfx_outputs/merged_swimlane_20260520_091300.json`
- odd/even split experiment: `build_output/_jit_qkv_proj_rope_test_20260520_113642/dfx_outputs/merged_swimlane_20260520_113647.json`
- odd/even run log: `qkv_proj_rope_l2_swimlane_odd_even.log`
- affine remap compile-fail log: `qkv_proj_rope_l2_swimlane_affine_odd_even.log`

## 工作区状态

实验期间对源码做过的改动已经全部回退。
当前 `models/deepseek/v4/qkv_proj_rope.py` 已恢复原始 `attn_norm_apply` 结构，并重新通过 `python -m py_compile` 检查。


# 5.29 — 其他 kernel 能否放大到 ~50us

判据：「per-task 平均时长 ~50us 才算健康」。以下为 full indexer swimlane 中每个 scope 的具体情况。

## A. 已经 ≥50us——不用动

| scope | per-task | 为什么本来就大 |
|---|---|---|
| `kv_score_proj` | ~109–118us | HBM-bound 的大 matmul（读整个 x、按完整 D 累加），本身就是 compressor 最重的访存 scope |
| `weights_proj` | ~60us | `WEIGHTS_ROW_CHUNK` 行分块后每 task 已经够大 |

## B. 有空间放大的候选（都是一行常量改动）

**`rope_fused`：~37us，32 核，`HEAD_GROUP_ROPE=8`**
- 关键：它**不是 buffer-bound**。内层 `pl.range(ROPE_ROW_CHUNK=IDX_N_HEADS)` 把每个 chunk 的 Vec/L0C 固定在 IDX_N_HEADS 行，跟外层 group 多大无关 → task 大小随 group 线性增长。
- 8→16：~74us，task 16→8，核 32→16。约束满足（HEAD_ROWS_ROPE=64×16=1024，8192%1024=0）。
- 隐患：注释明说 GRP=8 是刻意为了和并行的 `kv_score_proj` 做 cube overlap 而选的；调大可能破坏这个 overlap。

**`qr_hadamard`：~32us，32 核，`HEAD_GROUP_HAD=8`**
- 关键：它在 scope 内部按 `HEAD_ROWS=128` 行切 matmul，L0C 累加器永远只有 `[128,128]` FP32（64KB），和 group 大小解耦 → 所以能折得比累加器单独允许的更大。
- 8→16：~63us，task 16→8，核 32→16。约束满足（1024%128=0、8192%1024=0）。
- 同样隐患：核数减半。

**`topk`：~44us，8 核，`TOPK_GROUP=16`**
- 每行 sort 独立、bit-identical，纯靠内层 pl.range 折行。16→32：~87us，task 8→4。
- 但它不在关键路径上（笔记里记过「topk leaf fold 是中性的」）→ 放大安全但纯属让 kernel 数字好看，对 Total 没收益。

**`window_gather`(~34us) / `softmax_pool`(~11us)：compressor，16 核，`POOL_GROUP=4`**
- 这俩是可分组的（动态读 + 仿射写，已验证过 POOL_GROUP=4 能用）。4→8：gather ~60us、softmax ~21us，task 16→8。
- 隐患：它们是 dispatch-bound，且在 full indexer 上 gate 下游的 `score_fused`；核 16→8 可能反而推迟 `score_fused` 启动。

## C. 卡死、放不大

| scope | per-task | 为什么动不了 |
|---|---|---|
| `qr_proj` | ~40us | 列 tile `Q_OUT_CHUCK=256` 已到 buffer 天花板：512 → INT32 acc `[128,512]`=256KB > 192KB Vec，且 UP_DOWN 行切不缩 create_tensor |
| `score_fused` | ~27us | `SCORE_B_GROUP=4` 是刻意压的：注释明说 =8 会让 `score_fused` 变成最大的单 task、成为关键路径下限；=4 才好和别的重叠 |
| `state_scatter_paged` / `kv_and_cache_write` | ~11 / ~9us | 数据依赖地址的 paged scatter，任何分组都死锁 507018（这次亲手验证过 group 8 和 2 都挂） |

> 另：底层的 `HEAD_GROUP=2` 是 quant 那条路的 Vec 上限（GRP=4 → 199KB 溢出、GRP=8 → L0C 256KB 溢出）；但 `qr_hadamard` 自己走的是解耦后的 `HEAD_GROUP_HAD=8`，不受这条限制。

## D. 天生就小的 init/epilogue（不值得放大）

`score_init`(~4us)、`pooled_load`(~2us)、`kv_final_store`(~5us)、`kv_hadamard`(~4us)、`rope_fused_0`(~6us)、`rmsnorm_rope_slice`(~14us)——都是初始化/搬运/收尾，要么活儿本来就少、要么已经按 `POST_CHUNK` 顶到最大组了。

## 为什么当时选了「不进行」

真正能动的只有 B 组那几个，但它们都是**拿核数换 kernel 大小**：`rope_fused`/`qr_hadamard` 放大 → 核 32→16，而它们正好和关键路径（`qr_proj`/`kv_score_proj`/`score_fused`）并行重叠，「大而少」很可能让 Total 回退，即使单个 kernel 看着更健康；`topk` 安全但纯中性；`POOL_GROUP` 可能拖慢 `score_fused`。

也就是说——这里**「串行→并行」和「kernel 要大」这两条 perf 原则是打架的**，而现有的 GRP 值恰恰是之前为了平衡这对矛盾调出来的。所以没有一个是「稳赢」，全靠真机赌，性价比低，所以先不动。
