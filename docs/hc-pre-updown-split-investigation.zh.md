# hc_pre UP_DOWN 切分调查记录（收尾）

> 目标：在 `models/deepseek/v4/hc_pre.py`（DeepSeek-V4 的 RMS+linear+pre/post/comb/mix_x 融合算子）上，
> 调查 `pl.split(SplitMode.UP_DOWN)` 能否把向量 epilogue 铺到第二个 AIV 子块来提速。
>
> **一句话结论**：已拿到的确定收益是 **K-tile 的 −16%（110.5→92.6us）**；切分这条线在所有实测路径上要么 ≤ 基线、要么死锁，**收益仍未证明**；调查中挖出并上报了一个真 pypto bug **#1854**。

---

## 1. 确定交付（已实测、已落地）

| 项 | 结果 |
|---|---|
| **#1789 死锁源码级解决** | colslice 根因 = 非 32B 对齐的 row-major `cols=1` tile base（`pre_eps[:, h:h+1]` 在 FP32 下是 4B/8B/12B 偏移）。改写法：逐 head GM 列读 → **row-major 做 sigmoid** → `row_sum` 取出 **col-major `[T,1]`**（无需 transpose）。真机 decode+prefill 全 PASS，无需等 PTOAS。生产版 `hc_pre.py` 用的是等价的 transpose 形式，同样 OK。 |
| **性能杠杆** | `LINEAR_K_TILE 128→256`：缩短串行 matmul 的 K 链（K=16384）。**110.5→92.6us（−16%）**，精度不变。512 会 L0B OOM，256 是上限。 |
| **最快可用版** | `hc_pre.py`（融合单 spmd + K=256）≈ **92.6us**，不用改。 |

精度三条硬规则（#1789 改写法的踩坑）：
1. 新 ptoas 拒绝 **row-major `cols=1`** FP32 tile 的 `alloc_tile`（4B 行 < 32B）；**col-major `cols=1`** 合法（`row_sum`/`reshape([1,T]→[T,1])` 产出）。这条律和设备 507018 同源，现在编译期就抓。
2. 超越函数（exp/recip）要在 **row-major** 上做、再 `row_sum` 取列；在 col-major 单列 tile 上做会把 row 0 广播到所有行。
3. `row_sum` 前**不能 fillpad**：fillpad 把 pad 列标 valid，`row_sum` 会把相邻真实数据一起加进来。

---

## 2. 切分调查：全部路径与结论

### 2.1 实测矩阵

| 切法 | L2 (prefill) | 结论 |
|---|---|---|
| 融合不切 K=256（基线） | **92.6us** ✅ | 当前最优 |
| route-2：mix_x 拆独立 scope + UP_DOWN | 127us | 拆 scope 亏 ~16us（GM 往返）；且 CCE 显示切分**没生效**（见 2.3） |
| multiscope baseline（5 scope） | 122.8us | GM 往返先亏 30us，本就比融合慢 |
| multiscope 切 linear(matmul) scope | 121.6us + **跑错** | cube+split 的 cube→vec 分发出错；linear 是 cube-bound，UP_DOWN 切 AIV 不切 AIC |
| route-1：完整 epilogue 切（assemble draft） | **507018 死锁** | → bug #1854 |

### 2.2 in-core 泳道分析（关键，但有反转）

对融合 hc_pre 的核内泳道（`build_output/incore_hc_pre_1spmd_latest_143533/`）：
- span 67.9us：**veccore0 满载 66us（97%，关键路径）**，cube 在 34us 就做完（非瓶颈），**veccore1 末事件停在 37.5us、之后全闲**。
- 第二半段（34–68us）veccore0 上 **VECTOR union-busy 28.6us（84%）**，MTE2 5.7 / MTE3 4.2 —— **是 VECTOR-bound，不是 HBM**（一度误判为 HBM，已更正）。

→ 单任务看：确有一条 ~28us 的 VECTOR 尾压在单子块、另一子块全闲，**理论上可切**。
→ 但 L2 实测切了不赚 / 死锁。差距来自：(a) 切要拆 scope，拆 scope 的 GM 成本 > 切省的；(b) 8 个 token-tile 的 inter-task 重叠可能已把单任务尾巴藏掉。

### 2.3 CCE 证据：UP_DOWN 在「纯向量 standalone scope」上不咬合

子块 id 的 codegen 由 `pypto/python/pypto/backend/pto_backend.py:592` 决定，三种模式：
- `fixed_subblock_id`（切分特化，固定 id）/ `runtime_bridge`（双核动态，`get_sub_block_id(args)→pypto_runtime_subblock_id`）/ **无**。

实测对比：
- **融合 hc_pre scope（cube+vec，即便 `SplitMode.NONE`）**：CCE 有 `get_subblockid` 分支——`if(subblock==0){…整段 mix_x，含全部 x_mixed GM 写…} else {…77 行 UB 辅助…}`，**两核严重不对等（~1132 行 vs 77 行）**。这解释了泳道里 veccore0 66us / veccore1 5us。**泳道是对的，不对等是「没切分」的必然结果。**
- **route-2 的 mix_x 纯向量 scope（挂了 UP_DOWN）**：CCE **0 个 `get_subblockid`/`get_sub_block_id`**，只有 `get_block_idx` → **UP_DOWN 没生成任务内子块切分**。

→ 重要结论：**UP_DOWN 只在 cube+vec 融合 scope 上「咬合」**（hc_head 的 linear、sparse_attn 的 proj_b 都是这种，且 sparse_attn 实测有用）。在纯向量 standalone scope 上大概率是 no-op。所以 route-2 的「0 收益」其实是「切分没生效」，那个负面值无效。

> 注意 `get_subblockid()` 是 body 宏（`#define`→`pypto_runtime_subblock_id`），`get_sub_block_id(args)` 是 wrapper 取逻辑 id 的运行时函数。**约束（同事确认）：取子块 id 必须走这个宏，别直接调底层原生 intrinsic**——原生返回物理子块号，在 A2A3 mixed-task 运行时下 ≠ 逻辑 lane，会错。

### 2.4 为什么 sparse_attn 的 UP_DOWN 有用、hc_pre 没用

`decode_sparse_attn.py:447` 的 `proj_b` scope 三条全满足，hc_pre 一条都不满足：

| | proj_b（切了有用） | hc_pre（切了没用） |
|---|---|---|
| 切分加法 | 独立 cube+vec scope，原地免费加 UP_DOWN | 必须先拆 scope → 亏 GM |
| 向量活是否在 per-task 关键路径 | 是（INT8 dequant） | 被 inter-task 重叠藏掉 |
| 输出 | 直接写最终 attn_out | 切完整版要写 scratch→unsplit → 撞 #1854 |
| UP_DOWN 是否咬合 | 是（含 matmul） | route-2 纯向量 scope 不咬合 |

---

## 3. route-1（完整 epilogue 切）的进展与剩余墙

structure：A=matmul+RMS（**不切**，避开 matmul+row_sum-under-split 跑错）；B=pre+mix_x+comb（`pl.spmd(UP_DOWN)`，用 `pl.store` 不是 `pl.assemble`——assemble 只收 Tensor 不收 Vec tile）；C=post + comb 压缩（**不切**）。

**已绕过 ✅**（`hc_pre_assemble_draft.py` 整个编译通过）：

| 问题 | 绕法 |
|---|---|
| post 的 set_validshape | 挪到不切 scope C |
| comb 的 set_validshape | split scope 写 8 宽/组 scratch `[T,32]`（无 set_validshape），内部归一化的 set_validshape 改 `fillpad`；C 里压缩 `[T,32]→[T,16]` |
| matmul+RMS-under-split 正确性 | 避开——matmul 留在不切 scope A |

**仍挡着 ❌**：comb 走 scratch → 被后续 **unsplit** scope C 消费 = **#1854 死锁**（运行期 507018，编译干净）。

`D_TILE=256` 死锁 / `D_TILE=128` 跑通但 comb 损坏（同一 UB-OOB 信号的轻重两态，但 IR UB 高水位仅 44KB，**非 UB 预算溢出**）。

---

## 4. pypto#1854（调查副产品，真 bug）

**`[Bug] pl.split(UP_DOWN) scope producing a GM tensor for a following unsplit scope 507018-deadlocks`**
https://github.com/hw-native-sys/pypto/issues/1854

触发条件（bisect 隔离、对照全 PASS）：**一个重型 `pl.spmd(UP_DOWN)` scope 写 GM scratch，被后续一个 unsplit `pl.spmd` scope 消费** → 运行期 507018。编译干净、UB IR 仅 ~44KB、与 `pl.pipeline`/`pl.range` 无关。

对照（全 PASS，只组合死锁）：同 scope 写最终输出 / 裸 UP_DOWN→scratch→unsplit + trivial op / **反方向**（unsplit→GM→UP_DOWN，= route-2）/ mix_x 单独 / comb 单独 / 整核不切。

最小复现：`models/deepseek/v4/_tmp_updown_combo.py`（已内联进 issue）。

**尝试过的绕法（不成立）**：把消费 scope C 也改 UP_DOWN（split→split）——但 C 的窄写靠 `set_validshape`，**`set_validshape` 与 UP_DOWN 互斥**；要让 C 能 split 就得去掉窄写（改整行），而那本身就去掉了 #1854 的触发器。所以 **#1854 和 comb 的窄写是缠在一起的**，不是翻个开关能绕的——要么等 pypto 修 #1854，要么等 pypto 支持 set_validshape-under-split。

---

## 5. 仍开放的问题

1. **hc_pre 的向量 epilogue 真咬合双子块时，L2 到底提不提速**——未证明。唯一能定论的是对一个「确认咬合 + 正确运行」的版本做核内泳道看 veccore1 占用，但这种版本被 #1854/set_validshape 挡着没产出。
2. 想在融合 scope（`hc_pre.py:98`，把 `NONE→UP_DOWN`）上切——UP_DOWN **会咬合**（cube+vec），但 set_validshape 当场挡；把 route-1 的绕法搬进来 comb 又走 scratch→unsplit → 回到 #1854。**两条路最后都汇到同一堵窄写墙。**

---

## 6. 结论与建议

- **性能**：确定收益 = K-tile 的 −16%（92.6us）。切分（mix_x / linear / 完整 epilogue）实测**全部 ≤ 基线或死锁**，是死胡同。核内泳道的「可能性」在 L2 没兑现。
- **若还想榨 matmul 侧**（和切分无关）：可试 `pl.pipeline` stage=2→4（K 有 64 iter）或 T_TILE 16→32/64（放大 matmul M）。
- **切分线建议 park**：等 pypto#1854 修复 + set_validshape-under-split 支持后再回头，否则结构性绕不过去。

### 文件状态（清理后）
- 保留：`hc_pre.py`（生产）、`hc_pre_multiscope.py`（可跑变体）、`_tmp_updown_combo.py`（#1854 复现）、`run.sh`、本文档。
- 已删：`hc_pre_{colslice,updown,assemble,multiscope}_draft.py`、`_tmp_updown_{rowsum,comb_min,b2c_min}.py`。

---

## 7. 补充（2026-06-25）：dummy-matmul 让切分「真生效」+ 对 §5/§6 结论的更正

> 前面 §1–§6 写于 dummy-matmul 实验之前，其「切分收益未证明 / route-2 切分没生效」的措辞需按本节更正。**前文不改，以本节为准。**

### 7.1 思路：给纯向量 scope 注入哑 matmul，把 UP_DOWN 从 no-op 翻成真咬合

§2.3 已确认「UP_DOWN 只在含 cube(matmul) 的 scope 上咬合，纯向量 standalone scope 上是 no-op」。由此：**在 mix_x scope 里塞一个微型哑 matmul**，让 UP_DOWN 生效，把 mix_x 的行劈到两个 AIV 子块。

### 7.2 验证链（三层，全部坐实「机制成功」）

1. **编译期 CCE 对照**（`_tmp_updown_dummymm.py`）：
   - mix_x only + UP_DOWN → `get_subblockid = 0`（no-op，印证 §2.3）。
   - **+ 哑 matmul（T_TILE=32, N=16, K=128）→ `get_subblockid = 3` + `get_sub_block_id = 1`（真咬合）**。
2. **in-core 泳道**（生效版，op-sim）：**veccore0 = veccore1 = 22us（各 567 条指令、~87% util），哑 matmul 仅 1.3us**。对比原融合（不咬合）veccore0 66us / veccore1 5us —— **mix_x 50/50 劈到两子块、向量尾 44→22us 真 halved、veccore1 满血**。「veccore0/veccore1 不均」被这招彻底解决。
3. **两个硬约束**：① **T_TILE ≥ 32**（劈半后 cube 要 ≥16 行 fractal 下限，否则 `boxed tile rows multiple of innerRows(16) got 8`）；② 哑 matmul 维度 cube-legal（N 为 16 的倍数）。

### 7.3 L2 实测：3-scope **de-fused** 版 = 109us（全 PASS）

`hc_pre_dummymm_draft.py`，结构 A(不切 matmul→mixes_gm) / B(哑mm+mix_x，UP_DOWN，T_TILE=32，写最终输出 x_mixed) / C(不切 post+comb)。`mixes_gm` 走 A(unsplit)→B(split) **反方向**，不踩 #1854；B 写最终输出、无 scratch→unsplit。

- **L2 = 109.12us，x_mixed/post/comb 全 PASS。** vs 融合基线 **92.6us → +16.5us**。

### 7.4 ⚠️ 结论更正：109us 证明的是「**拆开切**亏」，**不是**「单 spmd 切没收益」

这是本次最重要的更正（之前把两者混为一谈）：

| 版本 | de-fusion 成本 | 测到没有 | 结果 |
|---|---|---|---|
| **3-scope dummy-mm（§7.3）** | **有**（拆 3 scope、GM 往返） | ✅ 测到 | 109us（亏） |
| **原地切融合单 spmd** | **无**（1 scope） | ❌ **从没跑成** | **未知** |

- **已证明**：de-fused 切 = 109 > 92.6，de-fusion 成本（≈16us，2 个 GM 桥）压倒 per-task 向量尾 halving 的收益。
- **未证明**：**原地切融合单 spmd 到底提不提速** —— 它是唯一能避开 de-fusion 成本、真有机会赢 92.6 的形态，但**编译就挂、跑不出来**（见 §7.5），**既没证实也没证伪**。
- **弱估算**（非实测）：109 − 16 ≈ **93us ≈ 基线**，暗示 per-task 向量尾 halving 在 L2 被 8 个 token-tile 的 **inter-task 重叠**吃掉、净 ~0。但这是减出来的，不是原地版实测。

→ 正确说法：**凡能跑起来的切分形态（de-fused）都亏；唯一可能不亏的「原地单 spmd 切」被 pypto 编译器墙挡死、未能测。**

### 7.5 原地切融合 scope 的「四堵墙」（实测，逐层）

把 `hc_pre.py:98` 的 `NONE→UP_DOWN` 直接编译：

| # | 墙 | 报错 | 能否过 |
|---|---|---|---|
| 1 | **transpose**（生产版 pre 取列） | `pl.split(UP_DOWN) but contains a tile.transpose that swaps the split axis` | ✅ 换 #1789 的 row_sum 形式（去 transpose） |
| 2 | **cube 16 行下限**（T_TILE=16 劈成 8） | `boxed tile rows must be multiple of innerRows(16), got 8` | ✅ T_TILE=32 |
| 3 | **AllocateMemoryAddr 校验失败**（T_TILE=32，降 D_TILE=64 也报） | `Verification failed after 'AllocateMemoryAddr'` | ❌ **过不了**——pypto 内部内存分配 limitation（updown_draft 早先注释的 "memory_reuse OOB at compile"） |
| 4 | **set_validshape**（post/comb 窄写） | `set_validshape(literal T_TILE) > 劈后 rows` | ❌ 结构性死墙（还没走到，卡在 3） |

→ 前两堵可清，**第三堵（AllocateMemoryAddr）是卡住「原地单 spmd 切」的真正 blocker**，纯 pypto 编译器内部，源码绕不过；即便绕过，第四堵 set_validshape 仍在后面。

### 7.6 #1854 仍未解决

3-scope 版**没解决** #1854，只是靠「反方向 + 写最终输出」**结构性绕开**了。#1854（split→scratch→unsplit 死锁）本身仍 OPEN。

### 7.7 本节新增/相关文件
- `models/deepseek/v4/_tmp_updown_dummymm.py`：哑 matmul 咬合对照（CCE `get_subblockid` 0 vs 3）。
- `models/deepseek/v4/hc_pre_dummymm_draft.py`：3-scope 生效切分版（L2=109us，全 PASS）。

### 7.8 下一步（若要把「原地单 spmd 切」彻底定论）
1. **isolate §7.5 的第三堵墙 `AllocateMemoryAddr`@T_TILE=32**（pypto 内部内存分配），最小复现 → 大概率值得开 issue；这才是真正卡住原地切的东西。
2. 或等 pypto 支持 **set_validshape-under-split** + 修 **#1854**，再回头测原地切能否赢 92.6。
- 在这之前，性能上的确定结论仍是：**融合单 spmd + K=256 = 92.6us 最优**；切分**未被证伪、但也无任何可跑形态能赢**。

---

## 8. 决定性更新（2026-06-25 续）：原地切 mix_x **实测 −25%**，真墙只剩 `set_validshape`

> 本节**纠正 §7.4/§7.5/§7.8** 两处:(a) §7.5 把第三堵墙说成「pypto 内部 limitation」是**错的**;(b) §7.4 的「切分未被证伪、无可跑形态能赢」已被**正面推翻**——隔离掉 de-fusion 后,原地切 mix_x 实测 **−25%**。

### 8.1 单 spmd 隔离实测：原地切 mix_x = **−25%**（决定性）

`_tmp_updown_dummymm.py` 两个函数**都是单 spmd**,唯一差别 = 有没有哑 matmul（= UP_DOWN 咬不咬合）。**没有 de-fusion 干扰**,直接比 L2:

| 单 spmd 版本 | L2 |
|---|---|
| **不切**（mix_x 跑单子块，UP_DOWN no-op） | **56.76us** |
| **切**（哑 mm 咬合，mix_x 50/50 双子块） | **42.42us** |

→ **−25%（56.76→42.42）。原地切 mix_x 实打实快 1/4。** 之前 §2/§7 的「切分不提速」是被 **de-fusion 成本污染**的结论（3-scope=109us 里，mix_x 那 −14us 被拆 scope 的 +30us GM 往返赔光）。**隔离掉就是真赚。**

### 8.2 ⚠️ 更正 §7.5 第三堵墙：是 **UB 溢出（可过）**，不是「内部 limitation」

逐层 dump 完整报错（不再 grep 过滤）后,原地切完整 hc_pre 的墙是:

| # | 墙 | 真相（更正） | 能否过 |
|---|---|---|---|
| 1 | transpose | UP_DOWN 不能切 transpose 轴 | ✅ 换 row_sum |
| 2 | cube 16 行下限 | T_TILE=16 劈成 8 | ✅ T_TILE=32 |
| 3 | ~~AllocateMemoryAddr 内部限制~~ | **❌ 错判!实测是普通 UB 溢出**：`Vec buffer 193344 > 188416`，只超 ~4KB（T_TILE=32 把 tile 翻倍）。 | ✅ **可过**：mix_x 改顺序累加 + `LINEAR_K_TILE 256→128`（缩 x_lin）就清了 |
| 4 | **`set_validshape`** | `row operand(32) > 劈后 shape dim(16)`；post/comb 窄列写 | ❌ **唯一真·结构性墙**，源码无解 |

→ 前三堵全是工程问题、都能过。**卡死「原地切完整 hc_pre」的只剩第四堵 `set_validshape`。**

### 8.3 `set_validshape × UP_DOWN` = 唯一真墙（含最小复现 + 机制）

- **机制**：FP32 UB tile 行须 ≥32B（≥8 列），所以 `[T,4]` 这类窄输出只能用 8 宽 tile + `pl.set_validshape(tile, T_TILE, 4)` 写出 4 个有效列（=「窄列写」）。UP_DOWN 把行劈给两子块、每块物理只剩 16 行,但 `set_validshape` 的 **row 参数写死 T_TILE=32**、split pass 没改写它 → `32 > 16` 报 `set_validshape op expects row operand <= shape dim (16)`。
- **最小复现（自包含 ~60 行，给同事/开 issue 用）**：`models/deepseek/v4/_tmp_setvalidshape_split.py`
  - `--no-split`（NONE）→ **compile PASS**；默认（UP_DOWN）→ **`set_validshape` 报错**。差别仅一个 `optimizations=[pl.split(UP_DOWN)]`。
- **修复方向（二选一）**：① split pass 把 `set_validshape` 的 row 操作数自动改写成劈后行数（T_TILE→T_TILE/2）；② 让 UP_DOWN 行劈支持「窄列写」这类 valid-shape 操作。

### 8.4 桌上的真金白银

- **原地切 mix_x = −25%（实测）**；前三堵墙都能过；**唯一拦路 = `set_validshape × UP_DOWN`**。
- 修了这堵墙,完整 hc_pre 大概率 **92.6 → ~78us（−15%）**（mix_x 的 −25% 是实测、其余墙可过）。
- 绕它的两条路都已知亏：de-fuse post/comb → 要么 #1854、要么 3-scope 109us。

### 8.5 结论再更正（替代 §6/§7.8 的「切分死路」措辞）

- **切分不是死路,是被一堵定义清晰的 pypto 墙挡住的 −15% 机会。** 该 push pypto 的就是 **`set_validshape` 兼容 UP_DOWN 行劈**（最小复现见 §8.3）。
- **当前可交付**仍是融合单 spmd + K=256 = **92.6us**；但「切分无收益」的旧结论**作废**——原地切 −25% 已实测。

### 8.6 本节相关文件
- `models/deepseek/v4/_tmp_setvalidshape_split.py`：`set_validshape × UP_DOWN` 最小复现（NONE PASS / UP_DOWN 报错）。
- `models/deepseek/v4/_tmp_updown_dummymm.py`：单 spmd 切 vs 不切的 −25% 隔离（加了 `--enable-l2-swimlane`）。
