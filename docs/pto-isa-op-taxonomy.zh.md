# PTO-ISA 算子分类与数据通路（教学/速查）

> 目标：从「算子做什么 / 在哪条硬件流水线跑 / 数据走哪条内存边」三个维度，
> 建立对 pto-isa tile 指令的整体认知。排查 fence / 同步 / 精度类问题时，
> 先用这张图判断「这是哪条边、谁该插栅栏」。
>
> 出处：`pto-isa/docs/PTOISA.md`（指令索引）、`pto-isa/docs/machine/abstract-machine.md`
> （抽象机器模型）、`pto-isa/docs/isa/programming-model/tiles-and-valid-regions.md`、
> `pto-isa/docs/isa/state-and-types/location-intent-and-legality.md`。

---

## 0. 一句话总览

一个 AICore = **一个标量控制单元（SU）+ 若干条异步执行流水线（Vector / Cube / MTE）**。
所有 tile 指令本质上只在**三套「路」**上跑：

1. **Vector 路（UB→UB）** —— 片上向量自产自销，绝大多数算子在这。
2. **Memory 路（GM↔UB，走 MTE）** —— 唯一进出全局内存的桥。
3. **Cube 路（GM→L1→L0→Cube→L0C→出口）** —— 只有矩阵乘走的专用路径。

跨流水线的每条边都要 **event 栅栏**（`set_flag`/`wait_flag`），由 pypto 编译器
（dep-gen / codegen）自动插入。**常见边处理成熟，冷门边容易漏插**——这是大量同步类
bug 的根。

---

## 1. 路网（内存层级 + 流水线）

```
                    ┌───────────────────────  GM (HBM/DDR, 全局, 大)  ───────────────────────┐
                    │                                                                          │
            MTE2 ↓ (load)                                                              MTE3 ↑ (store)
                    │                                                                          │
   ┌────────────────┴──────────────────  UB (Unified Buffer, ~192KB)  ────────────────────────┘
   │                          ↑↓  Vector pipe (PIPE_V)：在 UB 内自产自销
   │   ┌───────────────────────────────────────────────────────────┐
   │   │  Elementwise / Tile-Scalar / Reduce-Expand / Cast / Gather  │  ← 全是 UB→UB
   │   └───────────────────────────────────────────────────────────┘
   │
   └── Cube 专用路：
        GM →(MTE2)→ L1 →(MTE1)→ L0A / L0B → Cube(MAC) → L0C →(TMOV / fixpipe)→ UB 或 GM
        └──────────────────── 只有 TMATMUL / TGEMV 走这条 ────────────────────┘

Scalar Unit (SU)：驱动控制流 / 循环 / 地址计算；也能直接「标量写 UB」（TCI 就是这样写的）
```

### 硬件单元 / 流水线

| 单元 / pipe | 缩写 | 跑哪些算子 | tile loc |
|---|---|---|---|
| Scalar Unit | SU | 控制流、循环、地址计算、标量运算（含 TCI 的标量逐元素写） | — |
| Vector pipeline | PIPE_V | 绝大多数 tile 计算：elementwise、reduce/expand、cast、gather | `Vec` |
| Cube / Matrix pipeline | PIPE_M | `TMATMUL`、`TGEMV` | `Mat` |
| Memory pipelines (MTE) | MTE1/2/3 | GM↔片上搬运 + 布局：MTE2=GM→UB/L1(load)、MTE3=UB→GM(store)、MTE1=L1→L0(喂 cube) | — |

> 关键性质：这些 pipe **彼此异步执行**；ISA 给每条指令分配一个 pipe class，
> 用 event 表达跨 pipe 顺序，而不需要整核 barrier
> （`pto-isa/docs/isa/conventions.md` §Events and synchronization）。

### 内存层级

| 层 | 角色 |
|---|---|
| GM (HBM/DDR) | 全局内存，最大最慢 |
| UB (Unified Buffer, ~192KB) | 向量工作区；`loc=Vec` 的 tile 住这 |
| L1 / L0A / L0B / L0C | Cube 专用路：L1 暂存、L0A/L0B 喂 MAC、L0C 存累加结果 |

---

## 2. 指令分类 × 走的边（核心速查表）

来自 `PTOISA.md` 的 Instruction Index，按类别列出**典型指令**与**数据通路**：

| 类别 | 典型指令 | 走的边（源→目的 / pipe） |
|---|---|---|
| **Memory & Data Movement** | `TLOAD`、`TSTORE`、`MGATHER`、`MSCATTER`、`TPREFETCH` | **GM↔UB 桥**。load: GM→UB(MTE2)；store: UB→GM(MTE3)；mgather/mscatter: 带逐元素 GM 地址的 load/store；prefetch: GM→L2(提示) |
| **Elementwise (Tile-Tile)** | `TADD/TSUB/TMUL/TDIV`、`TCVT`(=cast)、`TEXP/TLOG`、`TSEL`、`TMAX/TMIN`、`TCMP` | **UB→UB，Vector pipe**。操作数与结果全在 UB，不碰 GM/L0 |
| **Tile-Scalar / Immediate** | `TADDS/TMULS/TSUBS`、`TEXPANDS`(标量广播)、`TMAXS` | 同上，**UB→UB，Vector pipe**；一个操作数是来自 SU 的标量 |
| **Axis Reduce / Expand** | `TROWSUM/TCOLSUM`、`TROWMAX`、`TROWARGMAX`、`TROWEXPANDMUL`、`TCOLEXPANDMUL`(=col_expand_mul) | **UB→UB，Vector pipe**；访问模式特殊：row 系沿「行内列」连续读；**col 系沿「列跨行」读 = 带 stride**（更慢、对齐更敏感） |
| **Matrix / Matrix-Vector** | `TMATMUL`(+acc/bias)、`TGEMV` | **Cube 专用路**：GM→L1(MTE2)→L0A/L0B(MTE1)→Cube(MAC)→**L0C**；结果在 L0C，需 `TMOV`/fixpipe 搬到 UB/GM 才能被向量用。loc=Mat |
| **Layout & Rearrangement** | `TMOV`、`TRESHAPE`、`TTRANS`(转置)、`TEXTRACT/TINSERT`、`TCONCAT`、`subview`、`TFILLPAD` | 多数 **UB→UB**（片上重排、不改数值）。特例：`TMOV` 兼任 **L0C→UB** 的 matmul 收尾桥；`TRESHAPE`/`subview` 常是零成本（只换 view） |
| **Irregular & Complex** | `TGATHER`、`TCI`(=arange)、`TTRI`、`TRANDOM`、`TSORT32`/`TMRGSORT`、`TQUANT`、`TPARTADD/...` | **UB 内，Vector/Scalar 混合**。`TCI`=SU 标量逐元素写 UB；`TGATHER`=UB 内按索引取；`TQUANT`=向量 + 额外产出 scale/exp tile。**「非标准边」，跨 pipe 同步最易出微妙问题** |
| **Synchronization** | `TSYNC`(pipe barrier)、`SYNCALL`(跨核) | 不搬数据，**插栅栏**：异步 pipe 之间 / 跨核之间强制先后 |
| **Manual / Resource Binding** | `TASSIGN`(手动绑地址)、`setfmatrix`、`set_img2col_*` | 手动放置 / 配置寄存器 |
| **Comm（多 NPU）** | `TPUT/TGET`、`TBROADCAST`、`TREDUCE`、`TNOTIFY/TWAIT` | 跨卡：本地 GM →DMA→ 远端 GM（常经 GM→UB→GM） |

---

## 3. 三套路的「一句话记法」

1. **Vector 路（UB→UB）**：Elementwise / Tile-Scalar / Reduce-Expand / Cast / Gather / TCI
   —— 进了 UB 就在 UB 里反复算，不下 GM。**最多数算子在这。**
2. **Memory 路（GM↔UB，MTE）**：`TLOAD`/`TSTORE`/`mgather`/`mscatter`
   —— 唯一进出 GM 的桥。**load-then-compute、compute-then-store 是每个 kernel 的骨架。**
3. **Cube 路（GM→L1→L0→Cube→L0C→出口）**：`TMATMUL`/`TGEMV`
   —— 自带一条专用片上路径，结果落 L0C，必须 `TMOV` 出来才能接向量后处理。
   **这就是 matmul+向量融合要专门处理 L0C→UB 那一跳的原因。**

---

## 4. 同步：哪条边谁来插栅栏（排 bug 直接相关）

跨 pipe 的边都需要 event 栅栏（`set_flag`/`wait_flag`），由 pypto dep-gen 自动插入。
经验上「成熟度」差异很大：

| 边 | 频率 | 栅栏可靠性 |
|---|---|---|
| **MTE2 → Vector**（load 完再算） | 每个 kernel 都有，一等公民 | **必然插**（数据搬运依赖，整块原子就绪） |
| **Vector → MTE3**（算完再 store） | 极常见 | 必然插 |
| **Cube → Vector**（matmul 后处理，L0C→UB→V） | 常见 | 一般可靠（mix 融合偶有坑） |
| **Irregular/Scalar → Vector**（如 TCI→cast） | 罕见 | 薄弱 / 可能漏插 |
| **跨核 / 跨任务 GM 可见性**（一个核 store，另一个核/host load） | 取决于 runtime 契约 | 历史上踩过坑（simpler#982 类） |

**实战提示**
- 现象「某 tile 尾部/部分列**间歇性**错、加任意等量流量就好、单核更难复现」→ 多半是
  **冷门边漏插栅栏 / 跨核 GM 可见性竞争**，不是算子算错。
- 判别「是哪条边」：先用第 2 节表把每个 op 归到 Vector/Memory/Cube 路，再看相邻两个
  op 是否跨 pipe；跨 pipe 且属于上表「薄弱」行的，就是嫌疑边。
- 验证手段：本地 `skip_ptoas` 编译看 `build_output/.../passes_dump/32_after_AllocateMemoryAddr.py`
  （UB 地址、各 op 的 producer/consumer），设备 ptoas 出 `.pto` 看 `set_flag`/`wait_flag`
  在嫌疑边前是否缺失（对比一个「正常边」版本做 diff）。
  ⚠️ 注意「加流量就好」可能是**延迟掩盖（latency masking）**而非真修复——做对照时要
  用「等量但不改变被测边」的控制组排除这个混淆。

---

## 5. 一个实战注脚：rope 索引构建的竞争

`models/deepseek/v4/_tmp_arange_tci.py` / `_tmp_col_only.py` 复现：在 spmd 多核下，
`col = col_expand_mul(ones, cast(arange))` 产出的 tile **尾部若干列间歇性 stale**，
经广播后「所有 token 同列全错」（`wrong_cols` 固定几列、`tokens=128/128`）。

排查链（含一次自我纠正）：
1. 不是 gather（去掉 gather 的 `_tmp_col_only.py` 仍坏）。
2. 不是 `col_expand_mul`/`cast`/`arange` 的**算子逻辑**（单核 10/10 全对）。
3. 不是 pypto 静态 UB 分配别名（IR 实测各 tile 区间不重叠）。
4. 一度怀疑是 `pl.arange`→`pto.tci` 单行尾写无 fence（tci vs GM-load：7/3 vs 10/0）；
   但 **`--ctrl` 延迟掩盖对照推翻了它**：给 tci 版加等量 GM-load 流量后也 10/10——
   说明「加流量就好」是延迟掩盖，tci-vs-load 的差异**不能**单独坐实 tci。
5. 当前最优解释：这是一个**时序敏感的跨核/跨任务 GM store 可见性竞争**（simpler#982 类），
   tci 只是「流量更小、暴露窗口更大」的变体；与 pypto#1648（vec→matmul-RHS）很可能同源。

**工作区规避**：把 arange / 已建好的索引作为 GM 入参传入（走 MTE2 一等公民边），
经验上 10/10 PASS——但理解为「时序规避」，可能对负载敏感，非内容修复。

> 教训：**「确定性 vs 间歇」「加流量是否变好」必须用等量对照排除延迟掩盖**，否则容易把
> 「暴露面」误判成「根因」。
