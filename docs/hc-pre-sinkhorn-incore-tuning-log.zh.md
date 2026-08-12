# hc_pre 核内调优记录（2026-07-06）

> 目标：对 DeepSeek V4 `hc_pre`（MoE gate 前置）做核内（in-core）profiling，
> 定位瓶颈并尝试优化。**结论：6 个方向全部回退，`hc_pre.py` 保持 baseline 不动，
> 转 lib/pypto 各提一个性能 issue。**
>
> 关联 kernel：`models/deepseek/v4/hc_pre.py`（`hc_pre_fused`）。
> 关联 issue：pypto-lib #700、pypto #1958。

---

## 1. 背景：hc_pre 在算什么，sinkhorn 为什么是瓶颈

`hc_pre` 是 DeepSeek V4 MoE 的 gate 前置模块，一个 token 出来后，先在这里算出：

- `post = sigmoid(mix[:,:hc] * inv * scale0 + base)` — 每 token 的 pre gate（4 个标量）
- `comb[T, hc, hc]` — 一个 `4×4` 的专家组合权重矩阵，由 **Sinkhorn 归一化** 得到

核内分成 AIC（cube/matmul）和 AIV（vector）两条 lane：

- **AIC**：linear 投影（`mixes_raw`），cube matmul，K-tile 链。
- **AIV**：RMS、cast、Phase C 的 gate、**Phase D 的 sinkhorn + mix_x + write_post**（三段 fuse 在同一个 flattened pool 里，靠 grid-stride 落到不同 lane）。

> 见 `hc_pre.py` 顶部 docstring：sinkhorn 的 20 轮串行归一化被刻意排在 pool 最前面，
> 让 busy lane 先吃掉这段 latency floor，其余 lane 同时跑 mix_x/write_post。

Sinkhorn 是一个 **20 轮的行/列交替归一化递推**，每轮都依赖上一轮结果 ——
天然串行，无法并行展开。这就是这轮调优要啃的硬骨头。

---

## 2. 核内 profile 采集（环境坑）

流程参考 `run.sh`：

```bash
python models/deepseek/v4/hc_pre.py -p a2a3 --enable-l2-swimlane
```

再按核内 profiling 流程，对 build 目录里的 `hc_pre_fused` 做 op-simulator trace。

**这轮踩的环境坑（写在这里供复现）：**

1. **simulator 卡型**：默认 `Ascend910B1` 直接 `status 139` 崩溃，换 `Ascend910B2` 才跑通。
   exporter 第一次还报「退化 trace：无 MMAD、VECTOR ~0 cycle」——
   原因是 simulator 自动输入让动态控制/循环边界变成 0，需要喂真实 golden。
2. **设备访问**：直接跑报 `halMemCtl rc=13 (EACCES)` —— 锁的是一张卡，runtime 却访问 device 0。
   修复：在 `task-submit` 里显式 `-d $TASK_DEVICE`。
3. **profiling 解析器修正**：新 codegen 会 emit `AICORE void <name>`，旧解析 regex 没认出来；
   本轮使用的解析器已扩展函数签名识别，并修正聚合逻辑，否则会错误地产生退化 trace。

---

## 3. 瓶颈定位：vector-bound，不是 cube-bound

同口径 `Ascend910B2` 真实核内 trace：

| 核 | 占用 |
|---|---|
| cubecore | `CUBE = 4 271 cycles` |
| veccore0 | `VECTOR = 526 615 cycles` |
| veccore1 | `VECTOR = 523 449 cycles` |

两个 vector lane 差距 <1%（**没有 lane 不均衡**，别去想「把 veccore1 的活挪到 veccore0」）。
热点全部压在 Phase D sinkhorn 的小矩阵 `div / row_normalize / col_normalize` 链路上。

> 注意：先前观察到的「prefill 不均衡」是 **L2 任务级** AIC(16) vs AIV(32) 的 tail 差异，
> 不是核内 lane 不均衡。两个粒度别混淆。

---

## 4. 优化尝试与前后代码（全部回退）

下面每一段都是「baseline 代码 → 实验代码 → 数据 → 裁决」。
baseline 即当前 `hc_pre.py`（`COMB_T_TILE = 8`）。

### 4.1 sinkhorn 热路径：`div` → `recip + mul`

按 `docs/pypto-coding-style.md` 的热路径建议，把除法换成「倒数 + 乘」。

**baseline（`hc_pre.py:296-299, 312-315`）：**

```python
col_sum = pl.add(col_sum, HC_EPS)
row0_cur = pl.div(row0_eff, col_sum)
row1_cur = pl.div(row1_eff, col_sum)
row2_cur = pl.div(row2_eff, col_sum)
row3_cur = pl.div(row3_eff, col_sum)
```

```python
# 迭代体内同样是 div / row_expand_div
row0_norm = pl.row_expand_div(row0_cur, row0_rowsum)
...
row0_cur = pl.div(row0_norm, col_sum)
```

**实验：**

```python
col_sum_rcp = pl.recip(pl.add(col_sum, HC_EPS))
row0_cur = pl.mul(row0_eff, col_sum_rcp)
row1_cur = pl.mul(row1_eff, col_sum_rcp)
row2_cur = pl.mul(row2_eff, col_sum_rcp)
row3_cur = pl.mul(row3_eff, col_sum_rcp)
```

**数据：**

| | decode Total | prefill Total |
|---|---:|---:|
| baseline | 56.22 us | 84.72 us |
| 优化后 | 55.72 / 65.20 us | 83.42 us |

AIV exec 从 25.69us 降到 22.68us（~7%），**但** L2 `Total Test Time` decode 反而抖到 65.20us。
看核内 `MOV_SRC_TO_DST_ALIGN` 之间的 gap：原本最大 gap ~10.25us（里面是 `VDIV 4.21 / MOVEMASK 4.19 / VCADD 4.09`），
换 recip+mul 后 gap 反而变 13.34us，`MOVEMASK/MOVEV/VMUL` 增多，vector runtime 17.6→19.8us。

**裁决：负优化，回退。** 单条算子替换省下的 cycle 被 codegen 多出来的搬运指令吃掉了。

---

### 4.2 删 `post_pad_store` 同核 scratch

`post_pad_store` 是 same-core scratch（`assemble` 写、本核 `load` 回读，无 barrier），
目的是给下游 store 一个 `HC_PAD` 宽（32B 对齐）的 tile。想直接消掉这次搬运。

**实验 A — `set_validshape + store`：**

```python
post_eff = pl.set_validshape(post_soft, T_TILE, HC_MULT)
pl.store(post_eff, [t0, 0], post)   # 直接存窄 tile
```

→ **编译失败**：`store` 需要 `TileType`，窄 valid-shape tile 不满足。

**实验 B — `assemble(post, post_pad)`：**

```python
post_pad = pl.assemble(post_pad_store, post_soft, [t0, 0])
```

→ 编译过，**运行 `AICore 507018`**（资源/同步问题）。

**裁决：回退。** 当前 DSL 对「窄宽度 tile 直接 store」支持不够，必须走 padded scratch。

---

### 4.3 `COMB_T_TILE 8 → 16`（仅 sinkhorn 分支）

这是这轮最像「有收益」的一次。把 sinkhorn 的 token row-tile 从 8 行扩到 16 行，
**但 pre/mix/post 仍保持 `T_TILE = 8`**（否则 decode `T=8` 会越界）。

**baseline（`hc_pre.py:91`）：**

```python
COMB_T_TILE = 8  # sinkhorn row-tile
```

**实验：**

```python
COMB_T_TILE = 16  # sinkhorn row-tile ( widened to feed more vector sub-channels )
# 注意：mix/post 分支仍用 T_TILE = 8，不受影响
```

因为 `COMB_T_TILE` 原来被 post gate 复用，扩之前要先扫一遍残留用法，确保只有 sinkhorn 分支用 16。

**数据（初看很漂亮）：**

| | decode Total | prefill Total |
|---|---:|---:|
| baseline `=8` | 68.06 us | 94.36 us |
| `=16` | 61.36 / 58.16 us | 84.22 us |

核内也证实了先前观察到的「VECTOR#7/#8 空」确实被喂起来了：

| 子通道 | tile=8 | tile=16 |
|---|---|---|
| `VECTOR#7` | 0.022 us / 2 events | 2.637 us / 37 events |
| `VECTOR#8` | 0.018 us / 2 events | 0.233 us / 7 events |

**但是** —— 多跑几轮（decode/prefill 各 3 次同流程对比）：

| 模式 | tile=8 均值 | tile=16 均值 | 结论 |
|---|---:|---:|---|
| decode 3 次均值 | 65.43 us | 65.33 us | 持平 |
| decode 3 次中位数 | 64.70 us | 68.72 us | tile16 反而差 |
| prefill 3 次均值 | 95.03 us | 93.27 us | 仅 ~1.8us |

**裁决：回退到 8。** 核内 lane 利用率确实改善了（**泳道更好看**），但端到端 wall 受调度 / head-tail overhead 波动影响很大，decode 没有稳定收益。这条教训很重要：**核内子通道利用率 ≠ 端到端 wall，必须以 L2 Total Test Time 为准。**

---

### 4.4 `COMB_T_TILE = 32`

顺着 4.3 继续拉宽，想把 `#9..#15` 也喂起来。需要把 `mixes_raw/sq_sum_acc/comb_logits` 的 scratch 行数 pad 到 32。

**数据：** decode PASS，但 Total `69.64 us`，比 tile16 的好结果还差。

**裁决：回退到 16（随后又整体回退到 8）。** decode 只有 8 个有效 token，padding 太重，单条 sinkhorn op 变重不划算。

---

### 4.5 Sinkhorn layout rewrite（结构性改写）

这是唯一动算法结构的尝试。思路：不再拆成 `row0..row3` 四个窄 `[T,4]` tile 分别归一化，
而是把 4 行堆成一个 `[4T, 8]` 的大 tile，一次做完 `row_max/row_sum/row_expand_div`，再切回 4 行做 column normalize。

**baseline 结构（`hc_pre.py:256-299`，节选）：**

```python
# 4 个独立的 [COMB_T_TILE, HC_PAD] 窄 tile，valid 列只有 HC_MULT=4
row0 = pl.load(comb_logits, [t0, 0*HC_MULT], [COMB_T_TILE, HC_PAD], valid_shapes=[COMB_T_TILE, HC_MULT], ...)
row1 = pl.load(comb_logits, [t0, 1*HC_MULT], [COMB_T_TILE, HC_PAD], ...)
row2 = pl.load(comb_logits, [t0, 2*HC_MULT], [COMB_T_TILE, HC_PAD], ...)
row3 = pl.load(comb_logits, [t0, 3*HC_MULT], [COMB_T_TILE, HC_PAD], ...)
# 每个 row 各自做 max/exp/sum/div，4 路并行但每路都很窄
row0_max = pl.row_max(row0_p, row_max_tmp)
...
row0_soft = pl.add(pl.row_expand_div(row0_exp, row0_sum), HC_EPS)
...
```

**实验结构：**

```python
# stacked scratch: [T*4, HC_PAD]，把 row0..row3 堆成 [4T, 8]
stacked = pl.create_tensor([t_linear*4, HC_PAD], dtype=pl.FP32)  # GM scratch
# 一次 row_max / row_sum / row_expand_div 覆盖所有 4 行
...
# 再切回 4 个 row 做 column normalize
```

**问题：**

1. `pl.assemble` 不能直接把 `Tile` 写进 `Tensor`，只能换成 `pl.store` + `pl.load` 走 GM scratch 往返（多一次 MTE round-trip，把省的 cycle 又吃回去了）。
2. 跑通后 `comb` 精度 **FAIL 122/128**，性能 87.86us（比 baseline ~65us 还慢）。

**裁决：回退。** 方向理论上对，但当前 PyPTO 表达能力下，靠 same-core GM scratch 模拟 stack/reload 会引入额外搬运和语义风险。要真正做好，需要 DSL/codegen 层支持「tile 内部 stack / valid-shape 保持」。

---

### 4.6 减少 `HC_SINKHORN_ITER`（20 → 10）

`HC_SINKHORN_ITER = config.FLASH.hc_sinkhorn_iters`（源码注释写 20 次）。

**裁决：未做。** 改迭代数会改变算法效果，需要模型精度验证，风险高，本轮没碰。列为「可能有效但需模型侧确认」的方向。

---

## 5. 根因：`VECTOR#7/#8` 到底是什么

初看核内 `VECTOR (compute) #7`、`#8` 大段空白，容易误判为「调度没给这两个 lane 派活」。
**这是误解。**

关键事实：

- `VECTOR#0..#15` **不是 `pl.spmd` 层面的 AIV lane**，而是 **simulator 把一条 vector 指令内部展开后的 compute 子通道**。
- Python 层没法直接「给 `#7` 派活」，只能通过 **tile 形状** 间接影响 codegen 对这条 op 的拆分。

落到这个 kernel：

- 每个 sinkhorn tile 名义形状 `[COMB_T_TILE, HC_PAD] = [8, 8]`
- 但**有效列只有 `HC_MULT = 4`**（剩下 4 列是 `fillpad` 出来的）
- 还**拆成了 4 个独立 row tile**：`row0/row1/row2/row3`
- 每个 row tile 真正参与计算的是 `[8, 4]` = **32 个 FP32 元素**
- 后续是 `row_sum / row_expand_div / div / add` 这类**短向量 reduction/broadcast**，不是大块连续 elementwise

所以 codegen 只需要前几个子通道就能覆盖这点工作量，`#7/#8/#9...` 自然没数据可分。

**一句话：8 行的时候不是「后面两个不愿意算」，而是这个 tile 的有效工作量太小，编译器只用前几个子通道就盖住了。**
`16` 让单条 op 从 `[8,4]` 变 `[16,4]`，元素翻倍，`#7` 才被喂起来；但 `#9..#15` 仍空，因为列方向还是只有 4。

> 推论：**真要利用后面的子通道，不是把 `[8,4]` 切得更碎，而是往更宽、更连续的 tile 合并**
>（比如让一次 op 处理 `[T, 16]`）。但这需要重写 row/col normalization 的表达，并确认 codegen 能对这种 layout 生成更好的指令 —— 即 issue #1958 要解决的。

---

## 6. 为什么 tile 只能 `8 → 16`，不能 `8 → 10`

一个自然问题是：既然 16 能喂起 `#7`，为什么不试 10？

三个原因：

1. **32B 对齐**：FP32 是 4B，`8` 个 FP32 正好 32B，`16` 是 64B，都天然对齐；`10` 是 40B，大量 `[1, COMB_T_TILE]` 的 row tile / rsqrt / row_sum 临时 tile 会变成非 32B 友好形状。
2. **形状整除**：decode `T=8`、prefill `T=128`，都天然适配 8/16；`10` 会让 prefill 产生大量非整除尾块，每块都得靠 valid-shape / tail mask，调度和 codegen 更差。
3. **`COMB_T_TILE` 不是普通 Python loop chunk**：它会影响 `pl.load` tile shape、`row_sum/row_expand_div`、临时 Vec tile 分配、store valid shape。Ascend/PyPTO 这类参数只能按硬件粒度 `8 / 16 / 32` 试，不能是任意整数。

---

## 7. DSL/codegen 能力缺口：为什么窄 tile 只能靠模拟

第 4、5 节的失败背后是同一件事：**sinkhorn 的自然工作单元是「每个 token 一个 4×4 小矩阵」，
但 PyPTO 的 tile / reduce / store 原语是围绕「一块连续的 2D slab、整轴 reduce、32B 对齐」设计的。**
这两者形状对不上，于是 kernel 作者只能用 padding + 拆窄 tile + GM scratch 去「模拟」，
而每次模拟都要交税（MTE 往返 / vector 子通道喂不饱）。

### 7.1 数据形状 vs. tile 形状

`comb` 输出 `[T, HC_MULT*HC_MULT] = [T, 16]`（`hc_pre.py:140`）。逻辑上每个 token 是一个 **4×4 矩阵**，
sinkhorn 在上面交替做行归一化（4 元素 softmax）和列归一化，重复 20 轮、强串行。

当前 tile 化（`hc_pre.py:256-259`）：

```python
# 4×4 矩阵的 4 行 → 拆成 4 个独立 [COMB_T_TILE, HC_PAD] tile，valid 列只有 HC_MULT=4
row0 = pl.load(comb_logits, [t0, 0*HC_MULT], [COMB_T_TILE, HC_PAD],
               valid_shapes=[COMB_T_TILE, HC_MULT], ...)  # [8,8] padded, valid [8,4]
row1 = pl.load(comb_logits, [t0, 1*HC_MULT], [COMB_T_TILE, HC_PAD], ...)  # 第 2 行
row2 = pl.load(comb_logits, [t0, 2*HC_MULT], [COMB_T_TILE, HC_PAD], ...)  # 第 3 行
row3 = pl.load(comb_logits, [t0, 3*HC_MULT], [COMB_T_TILE, HC_PAD], ...)  # 第 4 行
```

即 **4×4 的 4 行被拆成 4 个 `[8,8]` tile，每个有效数据只有 `[8,4]`（32 个 FP32），另 4 列是 `fillpad`**。
然后每行各自 max/exp/sum/div，4 路并行但每路都很窄。这就是「窄 tile」的字面含义：
**有效 32 元素，padding 占一半，reduce 只在 4 列上发生。**

### 7.2 kernel 作者想怎么做，但 DSL 不让

自由发挥的话最自然是下面三种写法之一，每种都撞一道 DSL/codegen 的墙：

#### Wish A：把 4×4 打包成宽 tile 一次算

```python
# 想要：一个 token-tile = [8 tokens, 16]（16 = 4×4 摊平）
comb_tile = pl.load(comb_logits, [t0, 0], [8, 16], ...)
# 行 max：每 4 个元素一组取 max → [8, 4]
row_max = pl.row_max_grouped(comb_tile, group=4)   # ← 原语不存在
```

**为什么不行：**

- `pl.row_sum / pl.row_max` 是**整轴 reduce**（一行的 16 个全加/全 max），没有「每 4 个一组」的 grouped/strided reduction。
- 想分组就得 reshape 成 `[8, 4, 4]` 对最内轴 reduce —— 但当前 **UB tile 的物理表示仅支持 2D**，3D tile 不支持，`transpose` 还会撞 `pypto#1651`（N-D transpose 在 `FlattenTileNdTo2D` 里 abort）。
- 只能退回「拆 4 个 `[8,4]` 窄 tile 分别 reduce」，回到现状。

> 这也正是 `VECTOR#7/#8` 空闲的根因：一条 vector 指令的有效工作量只有 `[8,4]=32` 元素，
> codegen 用前几个子通道就盖住了，后面没数据可分。

#### Wish B：直接把窄 valid-shape tile 存回去

```python
# 想要：post / comb 的 [T,4] 切片直接存，不走 padded scratch
post_eff = pl.set_validshape(post_soft, T_TILE, HC_MULT)
pl.store(post_eff, [t0, 0], post)
```

**为什么不行（4.2 实验 A 编译失败）：** `pl.store` 需要 `TileType`，窄 valid-shape tile 不满足。
于是被迫走 padded scratch（`post_pad_store`，`[t_linear, HC_PAD]`）：先 `assemble` 写进宽 scratch，
本核再 `load` 回来。**这就是「same-core GM scratch 往返税」** —— 一次 MTE load/store 没干活，只为凑 32B 对齐。

> 直接原因：Ascend UB 的 vector load/store 要 **32B 对齐的 tile base**。`[8,4]` FP32 行 stride 是 16B，不对齐；
> 要么 pad 到 `[8,8]`（stride 32B），要么 store 非对齐（慢 / 不支持）。`valid_shapes=[8,4]` 只告诉 codegen
> 「计算时 mask 掉后 4 列」，**tile 占的物理面积还是 `[8,8]`**。

#### Wish C：在 UB 内把 4 个 tile 堆成 1 个宽 tile

```python
# 想要：row0..row3 在片内堆成 [4T, 8]，一次 row_max/row_sum/row_expand_div 搞定
stacked = stack_ub(row0, row1, row2, row3)   # ← 原语不存在
```

**为什么不行（4.5 layout rewrite 实测）：**

- `pl.assemble` 只能 **Tensor←Tensor slice** 写，不能把一个 `Tile` 直接写进 `Tensor` 的某个切片。
- 只能换成 `pl.store` 写到 GM scratch，再 `pl.load` 回来 —— **多一次 GM 往返，把省下的 cycle 又吃回去**，
  还跑出 `comb` 精度 FAIL（122/128）。

**没有「UB-local stack / reshape 若干 tile 成一个宽 tile」的原语。** 想在片内重组布局，当前只能借 GM 中转。

### 7.3 根因与 issue #1958 的诉求

三条墙是同一件事的三个面：

> **PyPTO 的 tile 原语假设「工作单元 = 一块连续、32B 对齐、整轴 reduce 的 2D slab」。
> 而 sinkhorn 的工作单元是「每 token 一个 4×4 小矩阵」—— grouped（4 个一组）、strided、narrow、还涉及片内 reshape。
> 这类形状在当前 DSL 里没有一等公民表达，只能用 padding + 多窄 tile + GM scratch 模拟，
> 每次模拟都交税：要么 MTE 往返（Wish B/C），要么 vector 子通道喂不饱（Wish A）。**

issue **pypto #1958** 想要的能力，合起来叫 **valid-shape-preserving packed tile**：

| 能力 | 解决哪条墙 |
|---|---|
| tile 带逻辑形状 + packed layout（`[8, 4×4]` 连续打包，不 pad） | Wish A：一次 op 处理 128 元素，喂满子通道 |
| grouped / strided reduction（`row_sum(group=4)`） | Wish A：4×4 上做行/列归一化不用拆 4 路 |
| 窄 valid-shape tile 可直接 store（codegen emit masked/gathered store） | Wish B：消掉 padded scratch 往返 |
| UB-local tile stack / reshape | Wish C：片内重组布局不走 GM |

有了这些，4.5 那种 layout rewrite 才能真正落地，而不是被 GM 中转吃光收益。
这就是为什么本轮 6 个方向全回退、最后把球踢给 pypto codegen 的原因。

---

## 8. pto-isa 现状与缺口（实测 audit）

第 7 节讲的是「DSL/codegen 想要但缺的能力」。这一节是**去 pto-isa tile 库实测**，
把「缺」钉到具体原语和文件行号。结论先说：笼统表述为 **"pto-isa 没有 layout 原语"并不准确**
——layout 原语其实有一批，sinkhorn 要的几种特定形态才恰好缺。

> audit 基线：`$PTO_ISA_ROOT`（运行前设为本机 pto-isa checkout）@ `f72c24cd`。

### 8.1 pto-isa **有**的 layout 原语（别以为啥都没有）

| 原语 | 位置 | 关键约束 |
|---|---|---|
| `TReshape` | `include/pto/npu/a2a3/TReshape.hpp` | **UB 内扁平 reshape**；`:41` `Loc==NewLoc` 强制源/目标同 buffer（UB→UB 不强制走 GM）；`:43` 纯字节守恒（`sizeof*DNumel==…`）|
| `TConcat` | `include/pto/npu/a5/TConcat.hpp` | UB 级 tile 拼接，**带 valid row/col 校验**（"output valid col = sum of input cols"，`:101`）|
| `TFILLPAD` | `docs/PTOISA.md:122` | 一等 op："Copy+pad a tile outside the valid region" |
| 其他 | — | `ND2NZ / DN2ZN / NZ2ND / Swizzle / Reorder / Transpose / Gather / Scatter`；**valid region 是一等概念** |

也就是说：UB-local reshape / concat 是存在的，valid region 也被广泛支持。
**问题不在"有没有 layout 原语"，在"有没有小矩阵要的那种"。**

### 8.2 pto-isa **缺**的（sinkhorn 卡的正是这几条）

**① 窄 / 部分-valid tile 的 reduce 没有优化路径** ← "喂不饱 vector 通道"的 ISA 根

`include/pto/npu/a2a3/TRowReduceOps.hpp:225-256` 的 `TryOptimizeFP32Reduce` 只认 **4 种全 valid 的固定 shape**：

```cpp
ShapeOf64x128 : Rows==64 && ValidRow==64 && Cols==128 && ValidCol==128
ShapeOf32x256 :              ... Cols==256 && ValidCol==256
ShapeOf16x512 :              ... Cols==512 && ValidCol==512
ShapeOf8x1024:               ... Cols==1024 && ValidCol==1024   // 全部要求 ValidCol == Cols（满 valid）
```

本 kernel 的 sinkhorn tile 是 `[8,8]`-valid-`[8,4]`（`Cols=8, ValidCol=4`）—— **4 种一个都不匹配**，
`TryOptimizeFP32Reduce` 直接 `return false`，掉进通用 `TRowReduceInstr(dst, src, tmp, validCol, validRow)`
（`:259`，带 runtime validCol/validRow 的慢路径）。→ 4 个有效元素走通用 reduce，向量通道大片空转。
**这就是 `VECTOR#7/#8` 空闲在 ISA 层的落点。**

**② 没有 grouped / strided reduce**

`[8,16]` 想"每 4 个一组 reduce"——没有这条原语。最接近的是 `ReduceThenGroupValIdx`，但那是 topk 专用，不是通用 grouped reduce。

**③ 有 pad、没有 unpad** ← `post_pad_store` GM 中转的根

`TFILLPAD` 负责"copy + 填 pad"，但**没有反向 op** 把 padding 抠掉、直接以窄 packed 形状存回 GM。
所以 `[8,4]` 有效数据存不回，必须垫宽落 GM 再 load（`hc_pre.py:375-376`）。

**④ TReshape 是 UB 内扁平 2D，3D / grouped 表达不了**

`TReshape.hpp:43` 的纯字节守恒只支持扁平 view 改形；再叠上 UB 物理只有 2D + `pypto#1651`（N-D transpose
在 `FlattenTileNdTo2D` abort）→ `[8,16] → [8,4,4]` 这种 grouped reshape 走不通。

### 8.3 结论：领导方向对，措辞要更准

| 说法 | 准不准 |
|---|---|
| "pto-isa 没有 layout 变换" | ❌ 太满 —— reshape/concat/pad/swizzle 都有 |
| "pto-isa 缺**小矩阵要的那种** layout 变换" | ✅ —— 窄-tile 优化 reduce、grouped reduce、unpad、3D grouped reshape 这 4 条确实没有 |

**真正卡 sinkhorn 的是 ①（reduce 只优化 4 种全-valid 宽 shape）+ ③（有 pad 没 unpad）**，
这两条直接对应"喂不饱"和"GM 中转"；②④次要。

### 8.4 对 issue #1958 的修订建议

把 issue body 从泛泛的"support packed tile"改成 4 条带证据的具体诉求，每条点一个 pto-isa 落点，
上游更容易评估和接：

1. **窄 / 部分-valid tile 的 reduce 优化路径**（扩 `TryOptimizeFP32Reduce` 的 shape 集，或给 `[., ≤16]`-valid 加一条优化 reduce）— 引 `TRowReduceOps.hpp:225`。
2. **grouped / strided reduce 原语**（`row_sum(group=K)`）。
3. **unpad / 窄 packed store**（`TFILLPAD` 的逆，让 `[.,4]` 直接存 GM）— 引 `docs/PTOISA.md:122`。
4. **3D grouped reshape**（配合 UB 物理 2D 的解法，或跨 `pypto#1651`）。

---

## 9. 提的两个关联 issue

因为瓶颈本质是「DSL/codegen 对小矩阵 sinkhorn 的窄 tile 支持不够」，所以转去提了 issue：

- **pypto-lib #700** — `[Performance] DeepSeek V4 hc_pre Sinkhorn layout underutilizes vector sublanes`
  （lib 侧：场景与性能问题描述）
- **pypto #1958** — `[Performance] Support valid-shape-preserving packed tile layouts for small Sinkhorn reductions`
  （pypto 侧：需要 codegen 支持 valid-shape 保持的 packed tile layout）

两个仓当前都没有 `performance` 标签，暂用 `enhancement`；lib issue 下已评论互链到 pypto issue。

---

## 10. 结论与遗留

- **`hc_pre.py` 最终保持 baseline `COMB_T_TILE = 8`，所有实验 diff 已回退，未保留。**
- 核内 profile 流程跑通（含卡型/设备坑），定位到 sinkhorn vector-bound 本质。
- 「tile 加宽改善核内 lane 利用率但端到端不稳」是这轮最重要的反直觉结论 —— **别拿核内子通道利用率当 wall 收益。**
- 真正推进要等 issue #1958：DSL/codegen 支持 valid-shape-preserving packed tile，才能做 layout rewrite（4.5）那种结构性优化，而不是在 kernel 里绕 GM scratch。
- profiling 辅助解析器已适配 `AICORE void <name>` 函数签名，避免生成退化 trace。

### 可复现的关键产物

- decode L2：`build_output/_jit_hc_pre_test_20260706_*/dfx_outputs/merged_swimlane_*.json`
- 核内 trace：`build_output/incore_hc_pre_fused_hc_pre_*/`（clean 后的 Perfetto JSON）

---

## 11. Ascend C `[R,4,8]` 实测对照：pypto 为什么表达不了

> 起因：确认「Ascend C 列归一化用 `ReduceSumARAPerf`（4 矩阵行带步长加法），不是在 pad 轴上做列求和」之后，去翻了 Ascend C 参考实现，把「pypto 表达不了」钉到具体源码行。结论：**当前框架不支持，但卡点比「reduce」更靠上游——是「布局变换」。**

### 11.1 Ascend C 实际怎么做（源码核对）

参考实现：`cann-recipes-infer/ops/ascendc/src/hc_pre_sinkhorn/op_kernel/hc_pre_sinkhorn_base.h`（`hc_pre/op_kernel/hc_pre_base.h` 里有同款 `ReduceSumARAPerf`，约在 :468-480）。sinkhorn 专用版是同一份。

**列归一化 — `ReduceSumARAPerf`（sinkhorn_base.h:402-440）**，在**单个 `[dim0=R, dim1=4, dim2Align=8]` UB LocalTensor** 上：

```cpp
// 拷贝第 0 行（DataCopy 带 srcStride 跨过 dim1）
copyParams.blockCount = dim0;
copyParams.srcStride  = (dim1 - 1) * (dim2Align / elemInOneBlock);  // 跳过 dim1-1 行
DataCopy(output, input, copyParams);
// 把 j=1..dim1-1 行带步长加进去
for (i in 0..dim0) for (j in 1..dim1-1)
    Add(output[i*dim2Align], output[i*dim2Align],
        input[i*dim1*dim2Align + j*dim2Align], ...);   // ← 裸偏移地址算术
```

关键：`input[i*dim1*dim2Align + j*dim2Align]` —— **直接对 UB tile 做任意字节偏移寻址**，把同一块里第 j 个矩阵行加到累加器上。整块从第 1 轮到第 20 轮**从不拆成 4 个 tile**。

**行归一化 — `SoftmaxFP32Perf`（:510-525）**，用 `WholeReduceMax` / `WholeReduceSum`（`LastDimReduceMaxPerf/SumPerf` :494-507）——**硬件 reduce 指令，任意 shape 都走快路径**，对 `[R*4, 8]` 每行 reduce 8 列。

**列归一化的除法 — `DivABABrcInline`（:528+）**，把列和沿 dim1 broadcast 回 4 行做 Div。

布局要点：cols pad 到 8（`hcMultAlign = RoundUp(4,8) = 8`），矩阵行**保持 4 不 pad**；reduce 在「4」轴上做。因为每行 8 元素 = 32B，**每次带步长访问天然 32B 对齐**——这是 `[R,4,8]` 布局的精髓：用列 pad 换行对齐，靠 stride 在行间跳。

### 11.2 pypto 卡在哪：三层，逐层往上

| 层 | 现象 | 本质 |
|---|---|---|
| **L1**（最浅）| comb_logits 同核往返 + 32B 墙 | `pto.alloc_tile` 拒 `[R,4]`=16B 行；load-back 靠 `valid_shape` 把物理 tile 撑到 `[R,8]`(32B)。**已实测无法消除**（见 §4.2 与 §7.2 的对齐约束） |
| **L2** | pypto 必须把 4 个矩阵行**拆成 4 个独立 tile** 才能 col-reduce | `pl.add(tile, tile)` 只吃**整块** tile；没有「子 tile 裸地址算术」，表达不了 `input[offset + j*stride]`。这是 Ascend C `LocalTensor[offset]`（裸指针模型）vs pypto Tile（不透明整块模型）的编程模型差异 |
| **L3**（根因）| 数据是 `[R,16]` 交错布局，不是 `[R,4,8]` 行堆叠 | matmul 输出的 `mixes_raw` 本身就是 `[R,16]`（4×4 摊平）；de-interleave 回 `[R,4,8]` 是 grouped reshape，被 **UB 物理 2D + pypto#1651**（N-D transpose 在 `FlattenTileNdTo2D` abort）挡死 |

**链条**：没有布局变换（L3）→ 拆 4 tile（L2）→ 产生 `[R,4]`=16B 子 tile → 32B 墙（L1）→ comb_logits 往返。前文 kill 的是 L1；`ReduceSumARAPerf` 对照的是 L2，但**真正的根在 L3**。

> 即便 pypto 能用「1 次 load + 行切片 + 4 tile-add」近似 Ascend C（行切片是支持的，见 `hc_pre.py:343` `pre_tile_t[0:1, 0:T_TILE]`），也得先让数据进 `[R*4,8]` 布局——而这一步的 de-interleave 正是 L3 卡的地方。

### 11.3 对应的 issue

`ReduceSumARAPerf` 那套 = **#1958 的 (b) grouped/strided reduce + (d) 3D/grouped reshape**，外加一条 pypto tile 抽象**按设计就不暴露**的「子 tile 裸地址算术」。前两条可以提 issue 推 codegen/pto-isa；第三条是编程模型差异（裸指针 vs 不透明整块 tile），不一定靠 issue 能拿到，可能永远不会有。

### 11.4 量级上的诚实话：这条 gap 不是本 kernel 的 wall 杠杆

- 往返是**一次性物化**（Ascend C 1 次连续 load vs pypto 4 次重叠 `[8,8]` load）；20 轮循环 pypto 也在 UB-resident 的 4 tile 上跑（不碰 GM），Ascend C 在 1 块上跑——**每轮 op 数差不多**，差距不在循环里。
- 真正吃 wall 的 ISA gap 是 **行归一化的 `row_max/row_sum` 落到 generic 慢路径**（§8.2①：`[8,8]`-valid-`[8,4]` 不匹配 `TryOptimizeFP32Reduce` 那 4 个全-valid 优化 shape → 通用 `TRowReduceInstr` 慢路径 → vector 子通道空转）；而 Ascend C 的 `WholeReduceSum` 是一条硬件指令、不分 shape。

所以「Ascend C 更快」主要来自 **行归一化的硬件 reduce + 整块 `[R,4,8]` 布局**，列归一化的 strided add 只是锦上添花。要追 wall，优先级仍是 §8.2① 的窄-tile reduce 优化路径，其次才是这里的布局变换（L3）。换句话说：**`[R,4,8]` 对照把「pypto 为什么表达不了」讲清楚了，但它不是接下来该啃的 perf 方向。**
