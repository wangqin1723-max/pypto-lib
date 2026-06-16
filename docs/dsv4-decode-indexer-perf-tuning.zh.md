# DeepSeek-V4 Decode Indexer 性能调优总结

> 目标 kernel：`models/deepseek/v4/decode_indexer.py`（DSv4 decode 分支的 Indexer）
> 平台：Ascend a2a3（910B/C 真机）。配置 B=64, S=2, T=128, IDX_N_HEADS=64, IDX_HEAD_DIM=128。
> 最终结果：**Total Test Time 4583us → ~1357us，累计 −70.4%**（多次跑确认）。

---

## 1. 背景与方法论

`decode_indexer` 由一串细粒度的 cube/vector scope 组成（rope、hadamard、score、topk 等），
很多 scope 的 `Count` 高达 64/128，且 `Exec%` 很低（20%~60%）——典型的**调度开销受限**
（overhead-bound）：每个 task 的启动/换手开销（Head + Tail OH，约 5~10us）淹没了实际算力。

调优围绕一条主线展开：**减少 task 数、增大单 task 粒度**，让算力摊薄启动开销。

> 期间也尝试了「mixed cube+vector 融合」（把 matmul 和其 vector epilogue 塞进同一个 `pl.at`），
> 结论是**对本 kernel 行不通**，详见第 2 节。真正奏效的是第 3 节的 GROUP-chunking。

### 工作流要点

- **a2a3sim 跑不起本 kernel**：`score_store` 的 ReLU 在一个带步长子视图上做 `pl.maximum`，
  会触发模拟器 CPU 端 `TMAX_IMPL` 的严格模板检查（baseline 也挂）。但 sim 仍会先跑完
  **IR 验证 + ptoas codegen**，所以可以用它**本地预检 buffer 溢出 / 布局错误**，再上真机。
- **真机验证**：`task-submit --device auto --max-time 1800 --run "<conda+exports> && python models/deepseek/v4/decode_indexer.py -p a2a3 -d $TASK_DEVICE --enable-l2-swimlane > LOG 2>&1"`。
  分配到的卡号通过 `$TASK_DEVICE` 传入（不是 `ASCEND_RT_VISIBLE_DEVICES`）；env 不继承，payload 要自带 conda activate + exports。
- **噪声**：wall-clock 有 ~7% 的跑间噪声，每个改动跑 2~3 次再下结论。
- **精度**：`score` 与 `idx_kv_cache` 全程 PASS；`topk_idxs` 在当前 pypto HEAD 上有一个**底层精度 FAIL**，与本次优化无关，已按要求忽略。

---

## 2. 失败的方向：mixed cube+vector 融合（`pl.split(UP_DOWN)`）

文档（`docs/pypto-coding-style.md`、`docs/performance-tuning.md`）描述了「同一个 `pl.at` 内放
cube matmul + vector epilogue」的融合模式，靠 `optimizations=[pl.split(pl.SplitMode.UP_DOWN)]`
处理 L0C(NZ) → GM(ND) 的布局转换。我们逐个验证后发现**对本 kernel 是死路**：

| 尝试 | 结果 |
|---|---|
| `weights_proj` + `weights_write` 融合 | 编过、精度过，但 **+1.4% 回退**。baseline 已是单 task、99.3% Exec、tail OH 0.9us——没有 hand-off 可省，反而被切 tile 拖累。 |
| `rope_slice` + `rope_apply` 融合（旧 pypto2） | **编不过**：`col_expand_mul` 要求 src0 row-major，但融合 scope 内 matmul 输出还是 NZ。 |
| 同上（更新 pypto2 后，含 #1431） | 编过了，但 **+35% 回退** 且 topk 精度崩。`pl.split(UP_DOWN)` 会生成**分离的 `_aic` + `_aiv` 两组协同 task**，甚至翻倍 vector task 数，在这些小 scope 上 tail OH 暴涨。 |

**关键结论**：
- `pl.split(UP_DOWN)` 不会生成「一个高效融合 kernel」，它永远拆成协同调度的 `_aic`/`_aiv`，
  在已经很小的 scope 上得不偿失。
- W8A8 反量化 epilogue 都建立在 `col_expand_mul`/`row_expand_mul` 之上，这类算子**要求 row-major src0**，
  而融合 scope 内 matmul 输出是 NZ —— 这正是旧 pypto2 编不过的原因。
- **顺带发现的依赖项问题**：`col_expand_mul` 编不过其实是**本地 pypto2 仓太旧**（落后 origin/main 39 个提交）。
  提交 `#1431 (fix Acc→Vec tile.move layout)` 修了这条 cube→vector 链路。更新+重建 native 扩展后能编过——
  但即便如此，性能仍回退（见上）。pypto 是 editable 安装，改 C++ pass 必须**重建 `build/.../pypto_core*.so`**，`git pull` 不够。

---

## 3. 奏效的方向：GROUP-chunking（`pl.parallel(0,N,GRP)` 折叠）

核心做法：把原本 `for x in pl.parallel(0, N, step)`（N 个 task）改成每个 task 处理 GRP 份工作，
task 数降为 N/GRP。两种实现：

- **Option B（加高 tile）**：当各行/各 token 完全独立、且数据在内存里连续时，直接把 tile 行数 ×GRP。
  无需 GM 中间张量，**数值与原来逐 token 完全一致（bit-identical）**。
- **Option A（内层 `pl.range`）**：当 scope 之间有耦合中间量、或并行维数据不连续时，
  用内层 `pl.range(GRP)`，跨 scope 的中间量通过**全局索引的 GM 张量**传递。

各 scope 的 GRP 上限由 **buffer** 决定（Vec UB 192KB / L0C），所以拆成多个循环、各用各的 GRP。

### 3.1 rope/hadamard 循环（per-head，128 task/scope）

- 每个 op 都是逐行独立（matmul 按行、`col_expand_mul` 把 cos/sin 沿行广播、`row_max`/`row_expand_mul` 按行），
  且 cos/sin 对所有 token 相同 → 用 **Option B 加高 tile**，`HEAD_ROWS = GRP*IDX_N_HEADS`。
- 单循环 GRP 被 `qr_hadamard_quant` 的 Vec buffer 卡在 **2**（GRP=4 → 257KB，且 INT8 store 需 ≥32 列对齐无法再缩；GRP=8 → L0C 溢出）。
- **拆循环**：4 个 rope scope（rope_slice/apply/assemble/write）无 `[.,128]` 常驻 tile，独立循环跑 **GRP=4**；
  `qr_hadamard`（纯 cube matmul，输出 [256,128] FP32 = 128KB L0C）也拆出独立循环跑 **GRP=4**，
  通过 `qr_hadamard_acc_g` GM 张量喂给留在 **GRP=2** 的 `qr_hadamard_quant`。

### 3.2 score 循环（per-(batch, cache-block)，128 task/scope）

- 这是 overhead 最重的一组（`score_accum` Exec% 仅 20%）。并行维是 `b`（64 个 batch）+ 嵌套 `cb`，
  跨 b 的数据**不连续** → 用 **Option A 内层 `pl.range(SCORE_B_GROUP)`**，`SCORE_B_GROUP=8`。
- 三个 scope（quant→accum→store）耦合，中间量（量化后的 kv tile、score matmul 输出）通过
  **全局 `score_row0 = (b*MAX_CACHE_BLOCKS+cb)*CACHE_TILE` 索引的 GM 张量**（`kv_tile_i8_g`、`score_acc_g`）传递，
  每个 (b,cb) 写不相交切片 → **无竞争**（与 `score_kv_scale` 已有的写法同构）。
- 踩坑：把列拼装的 kv tile 整块 move 到 GM 会报 `pto.tmov ... matching src/dst shapes`，
  改成在 h1 循环里**直接写 GM 切片**即可。

### 3.3 `score_store` 去掉 `chunked_loop_optimizer`

- `score_store` 是 vector scope（含多次 cast）。`chunked_loop_optimizer` 的 cast-split 子任务在**全局调度层面**
  和其它 scope 抢核——去掉它后 Total 砍了约 500us，**而 score_store 自己的 scope 数字几乎没变**（说明是调度/重叠效应，不是单 scope 效应）。
- 教训：`chunked_loop_optimizer` 只在 buffer 溢出时才用；vector scope 上默认关，即使编得过。

### 3.4 关键路径 vs leaf：哪些 scope 值得 chunk

到 ~1759us 时 swimlane 显示 **Exec/Sched ≈ 90%**（AICPU 调度与 AICore 算力已平衡）。此时：

- 折叠 `topk`（128→32 task）**对 wall-clock 是 neutral 的**——topk 是 leaf，完全和别的工作重叠，
  砍它的 task 既不解放关键路径、也不解放饱和的调度线程。**已回退**（保持改动集「每改必有收益」）。
- 折叠 `rope`（在关键路径 qr_proj→rope→hadamard→score 上）**直接降 wall-clock −20.9%**。
- **结论：调度平衡后，只有砍关键路径上的 scope 才动 wall-clock。**

### 3.5 rope 写 fresh tensor（非 in-place）+ qr_hadamard K-split

GROUP-chunking 之后，rope 仍把结果**写回 `qr_proj_flat` 的 ROPE 列**（in-place）。这给 scheduler
留了一个无法在 slice 粒度消歧的 read+write（RAW/WAR）冲突：`pl.parallel` 的 32 个迭代被串到
**2 个核**上，rope 窗口占满 863~1357us，是新的瓶颈。

修法（`decode_indexer.py:131-138, 188-191`）：

- **rope 写独立张量 `qr_rope_out`**，保持 `qr_proj_flat` 在 rope 循环里**只读** → scheduler 能证明各 `o0`
  迭代独立，把 rope 铺到 18 核，窗口 863-1357us → ~128us。
- 下游 `qr_hadamard` 据此**按 K 拆 matmul**：NOPE 半（`qr_proj_flat[:,0:NOPE]`）+ ROPE 半（`qr_rope_out`），
  两段 `matmul_acc` 拼回。
- 收益 **−31%**（多核展开），数值中立。**诊断顺序：先看 core-spread，再考虑 mix-fusion**（第2节）。

> 旁注：`score_store` 还把 dequant + per-token 加权 reduce 融进**一个** scope（省掉 `score_logits` GM 往返、
> 跨 task windowed-write→full-read 依赖）；`QUANT_CHUNK`/`HEAD_GROUP`/`SCORE_B_GROUP` 都有 `T/B>=N` 兜底分支保证小配置可编。

---

## 4. 结果

| 步骤 | Total Test Time | 累计 |
|---|---|---|
| baseline（更新后的 pypto2） | 4583 us | — |
| rope/hadamard GRP=2 | 2850 us | −37.8% |
| + score 循环 SCORE_B_GROUP=8 | 2285 us | −50.1% |
| + score_store 去 chunked_loop_optimizer | 1737 us | −62.1% |
| + rope 拆循环 GRP=4 | 1378 us | −69.9% |
| + qr_hadamard 拆循环 GRP=4 | ~1357 us | −70.4% |
| + rope 写 fresh tensor（非 in-place）多核展开 | **再 −31%** | — |

最终保留的 6 处改动（均在 `decode_indexer.py`，每处都是实测 wall-clock 收益）：

1. rope 4 个 scope 独立循环、GRP=4（`HEAD_ROWS_ROPE`）
2. `qr_hadamard`（cube matmul）拆出独立循环 GRP=4，经 `qr_hadamard_acc_g` 喂给 GRP=2 的 quant
3. score 循环 `SCORE_B_GROUP=8`（全局 `score_row0` 索引的 GM 中间张量）
4. `score_store` 移除 `chunked_loop_optimizer`
5. rope/hadamard 基础 GRP=2 折叠（加高 tile，bit-identical）
6. rope 写 `qr_rope_out`（非 in-place）+ qr_hadamard K-split，rope 铺到 18 核（窗口 1357→128us，见 3.5）

回退的尝试：`weights_proj` 融合、rope/`rope_apply` UP_DOWN 融合、`topk` 折叠（均 neutral 或 regress）。

---

## 5. 可复用的经验法则

1. **overhead-bound（低 Exec%、高 Count）的 scope 优先 GROUP-chunking**——减 task 数是第一杠杆。
2. **逐行独立 + 数据连续 → Option B（加高 tile）**：最简单、零中间张量、数值 bit-identical。
3. **scope 间耦合 / 并行维不连续 → Option A（内层 `pl.range` + 全局索引 GM 中间张量）**，注意写不相交切片避免竞争。
4. **GRP 上限由 buffer 决定**（Vec UB 192KB、L0C、INT8 ≥32 列对齐），不同 scope 拆成不同循环各取所需。
5. **`chunked_loop_optimizer` 是双刃剑**：能解 buffer 溢出，但在 vector(含 cast) scope 上会引入全局调度争用，能不用就不用。
6. **`pl.split(UP_DOWN)` 融合在细粒度 scope 上通常得不偿失**，且 `*_expand_mul` 需要 row-major、不能直接吃 matmul 的 NZ 输出。
7. **调度平衡（Exec/Sched≈90%）后，只 chunk 关键路径上的 scope**；leaf scope 折叠是 neutral。
8. **先用 a2a3sim 预检 buffer/布局，再上真机**；wall-clock 跑 2~3 次去噪。
9. **in-place read+write 会把 `pl.parallel` 串到 2 核**——scheduler 在 slice 粒度无法消歧 RAW/WAR。写 fresh tensor 让输入只读、下游按 K-split 吃两半，rope 才能铺到 18 核。**先诊断 core-spread，再试 mix-fusion。**

## 6. 剩余杠杆

- `qr_proj`（关键路径头，64 task、最大单簇）。但 `qr_proj_write` 受 Vec buffer 限制（[T,128] FP32 tile），
  GRP=2 就要靠 `chunked_loop_optimizer`（vector scope 有害），预期边际收益小、风险高。
- 已在 −70.4% 处收益拍平（最后一步仅 ~2.5%，落入噪声带），建议固化。

---

## 7. mix 融合实验 + pypto2 调度回退（2026-05-25）

### 7.1 mix 融合：内层 `pl.range` 切片是关键
对 rope 四个 scope 直接整块 mix（matmul+epilogue 同一 `pl.at` + `pl.split(UP_DOWN)`）在 GRP=4 必爆 Vec：`rope_slice` 344KB、`rope_assemble` 286KB、`weights_proj` 288KB > 192KB 上限。
两条压 buffer 的路：
- **砍 GRP=2**：buffer 够，但 64→128 task，1690 vs 1357 = **+24%**，得不偿失。
- **`pl.auto_chunk`**：无效——它只切 `pl.parallel(...,chunk=N)`，rope 步长即整组，没空间。
- **内层 `pl.range(ROPE_ROW_CHUNK=IDX_N_HEADS)` 手切**：GRP 保 4，融合后每片 ~16KB。`rope_slice+apply` 融合 3 次实测 1047 均值 / 905 min，vs baseline 1218 均值 / 1133 min，全 PASS。收益来自删 `rope_apply` scope 握手（dispatch 153→97、apply busy 364→10）+ 细任务摊匀核心，**不是省计算**。

### 7.2 不要融合 INT8（INT32-accumulate）
`qr_proj`（INT8→INT32）内切后过 buffer，但 score/topk 精度挂（4→1~2 PASS）且更慢（1193 均值）。`score_accum` 同理。**mix 只对 FP32-out scope 安全。**

### 7.3 baseline 1133→1367 的 25% 抖动是 pypto2 回退，非代码
同源码 21f11ecbb(898) vs f56a254(1123)，task graph 一致，单核堆到 366（基线峰 294）。元凶 **#1468 guard unsafe window externalization**：为修 #1444 正确性，对「窗口写 `score_flat`/`qr_rope_out` + 全量读」保守不外化 → 失核扩散。有意权衡，不可 revert。lib 侧解法：消费侧也按窗口读，避开全量读同 buffer。

> 提醒：以上内核精度/buffer 数为单/三测，方差大，固化前各再跑 2 次。


mix的尝试

┌───────────────────────────────┬──────────────────────────┐
│             改动              │           结果           │
├───────────────────────────────┼──────────────────────────┤
│ rope_slice+apply 内切融合     │ ✅ −14%，唯一可用赢点    │
├───────────────────────────────┼──────────────────────────┤
│ 整块融合 GRP=4                │ ❌ buffer 344KB 爆       │
├───────────────────────────────┼──────────────────────────┤
│ GRP=2 砍半                    │ ❌ +24%（任务翻倍）      │
├───────────────────────────────┼──────────────────────────┤
│ auto_chunk                    │ ❌ 无效（无 chunk 可切） │
├───────────────────────────────┼──────────────────────────┤
│ qr_proj/score_accum INT8 融合 │ ❌ topk 稳挂 + 更慢      │
├───────────────────────────────┼──────────────────────────┤
│ rope_assemble、weights_proj   │ 中性/无收益              │
└───────────────────────────────┴──────────────────────────┘

法则

- 只融 FP32-out；INT8(INT32-acc) 不碰
- buffer 爆就内层 pl.range 切片保 GRP，别砍 GRP 别靠 auto_chunk
- 收益来自删 scope 握手，非省算

已记录

docs §7、memory、issue #368（topk 非确定，已转交 pypto）。
