# pypto 陷阱集：`pl.parallel` 的调度、依赖与展开

> 适用框架：pypto2（compiler）+ simpler（runtime）。例子取自 DSv4 decode indexer / compressor，
> 但结论对任意 `pl.parallel` 写法通用。
>
> 目录：
> - 陷阱 1：in-place 读写同一张量 → 串行化（性能）
> - 陷阱 2：粗化批次组用 `pl.range` loop-carry 累加器 → 间歇 NaN/死锁（**正确性**，改用 `pl.unroll`）
> - 陷阱 3：各任务写"跨整张量的 strided 行区间" → 串行化（性能，slot-major vs batch-major）

---

# 陷阱 1：`pl.parallel` 循环里 in-place 读写同一张量会被串行化

## 一句话

`pl.parallel` 循环里如果**读 + 写同一张量**，即使每个迭代只动不相交切片，调度器也会把全部
迭代判成相互依赖、串到 ~2 个核上。写到一个**独立 fresh tensor**、保持输入只读，循环才能铺满核。

## 现象（最小复现）

```python
# 【慢】32 迭代写不相交切片，但读+写同一张量 qr  →  串行，2 核
qr = create_tensor([8192, 128])
for o0 in pl.parallel(0, 8192, 256):
    rope = matmul(qr[o0:o0+256, 64:128])   # 读 qr
    qr[o0:o0+256, 64:128] = rope           # 写回 qr（同张量）
hadamard = matmul(qr)

# 【快】写独立张量，qr 全程只读  →  18 核，bit-identical
rope_out = create_tensor([8192, 64])
for o0 in pl.parallel(0, 8192, 256):
    rope = matmul(qr[o0:o0+256, 64:128])   # 只读 qr
    rope_out[o0:o0+256, :] = rope          # 写 fresh
hadamard = matmul_acc(matmul(qr[:,0:64]), rope_out)   # K-split: NOPE 半 + ROPE 半
```

实测（indexer rope 循环）：核占用 2→18，rope 窗口 863-1357us → ~128us，总 −31%，数值不变。

## 根因（pypto2 源码确认）

依赖图建在 `src/ir/transforms/utils/stmt_dependency_analysis.cpp:BuildStmtDependencyGraph`，
按 `Var*` 指针相等建边：

```cpp
for (const Var* v : collector.var_uses)   // 读 qr → 依赖最近一次 def
    if (last_def.find(v)) predecessors.insert(qr_def_stmt);
for (const Var* v : collector.var_defs)    // 写 qr
    last_def[v] = stmt;
```

只比张量身份，**不解析 `[o0:o0+256]` 区间是否相交**；`var_collectors.h` 收 Var 时丢下标；
静态/动态索引一视同仁。所以 32 个写不相交切片的迭代被全判成 WAW/RAW 链 → 串行。
属 RFC #1026 "InOut-use discipline" 的有意保守：sound 但不精。

## 怎么写

1. **输入只读，输出写 fresh**：让 `pl.parallel` 各迭代写不相交切片到一张新张量，scheduler 即可证独立、铺多核。
2. **下游按 K-split 吃两半**：若结果原本要拼回原张量（如 NOPE+ROPE 合成 [.,128]），下游 matmul 拆成
   NOPE 来自原张量、ROPE 来自 fresh，两段 `matmul_acc` 拼回，数学等价 bit-identical。
3. 代价仅一份额外内存，近乎免费。

## 不影响的场景

并行维数 < 核数、或循环本就 leaf 与他人重叠时，2 核串行未必上关键路径，收益为零。改之前确认是瓶颈。
（通用诊断顺序见文末。）

---

# 陷阱 2：粗化批次组的内层用 `pl.range` loop-carry 累加器 → 间歇 NaN/死锁

> **这是正确性陷阱，不是性能问题。** 例子：DSv4 compressor 把 `softmax_pool` 的 64 个 per-batch
> 任务合成 16 个粗任务（每任务 4 个 batch）以摊薄 AICPU dispatch。

## 一句话

把 `pl.parallel` 的批次维"分组粗化"时，如果内层批次循环用 **`pl.range`** 且循环体里有
**loop-carried 累加器**（如 online-softmax 的 `mi`/`li`/`oi`，每个 slot 做 `mi = mi_next`），
`pl.range` 会把这些名字跨 batch 串成一条隐式 SSA 链 → **间歇性算错（NaN）或 AICPU 死锁（507018）**。
把内层换成 **`pl.unroll`**（trace 时展开，每个 batch 是独立 AST 值）即解。

## 现象（最小复现）

```python
POOL_GROUP = 4

# 【错】内层 pl.range：mi/li/oi 被跨 batch loop-carry → ~44% 概率 NaN 或 507018 死锁
for cg in pl.parallel(0, B, POOL_GROUP):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool"):
        for ci in pl.range(0, POOL_GROUP):          # ← 运行时循环，loop-carry 名字
            base = (cg + ci) * STATE_LEN
            mi = window_score[base:base+1, :]
            li = pl.exp(pl.sub(mi, mi))
            oi = window_kv[base:base+1, :]
            for slot in pl.range(1, STATE_LEN):
                ...
                mi = mi_next                         # ← 累加器，被内层 ci 跨 batch 串链
            pooled_all = pl.assemble(pooled_all, pl.div(oi, li), [cg+ci, 0])

# 【对】内层 pl.unroll：每个 batch 独立展开，无跨 batch loop-carry → 稳定
for cg in pl.parallel(0, B, POOL_GROUP):
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="softmax_pool"):
        for ci in pl.unroll(POOL_GROUP):            # ← trace 时展开
            ... 同上 ...
```

实测（全 decode_indexer.py）：
- `pl.range` 版：9 次跑 = 5 PASS / **3 次 `score` 算成 NaN** / **1 次 507018 死锁**（~44% 坏，不确定）。
- `pl.unroll` 版：**10/10 uniform + hetero PASS**，0 NaN、0 死锁；性能不变（~764us，−42% vs baseline）。

## 为什么难定位

- **`idx_kv_cache`（压缩器输出）一直是干净的、精确的**，NaN 出在没改过的 `score`/query 路径 →
  第一直觉会以为"全局调度器崩了"或是别的 scope 的 bug。其实是 **trace 层的 loop-carry 串名**把
  语义搞乱，不是数据依赖（RAW）问题，也不是被改的那个 scope 算错。
- **间歇性**：同一份二进制，时对、时 NaN、时死锁，取决于调度时序。**一次 PASS 证明不了 race 已消除**
  ——要多跑（≥6 次），且要检查**所有输出**，不只看有没有死锁。（曾经"跑一次没死锁"的那次，其实
  `score` 已经 FAIL 成 NaN，只是没注意。）

## 怎么写

1. **分组粗化的内层批次循环用 `pl.unroll(N)`，不要用 `pl.range`** ——尤其当循环体里有 loop-carried
   累加器时。`pl.unroll` 是 trace 时展开，每个 batch 生成独立 AST 值，不会跨 batch 串名。
2. 纯 Python `range()` 在 kernel 里**不允许**（报错 "For loop must use pl.range/parallel/unroll/..."）；
   要 trace 时展开就用 `pl.unroll(N)`。
3. 参见 [[feedback-pypto-python-local-name-collision]]：同一 Python 名字在 `pl.range`/`pl.pipeline`
   体里被反复赋值会织出隐式 SSA 链，是同一类坑。

---

# 陷阱 3：`pl.parallel` 各任务写"跨整张量的 strided 行区间" → 串行化

> 例子：DSv4 compressor 的 `window_gather` —— 把分散在 paged `compress_state` 里的窗口行
> gather 到一块连续 scratch，供后续 softmax 读。

## 一句话

`pl.parallel` 各任务即使写**不相交**的行，如果每个任务的写**行区间跨度很大/横扫整张量**
（如 slot-major：行 = `slot*B + c_idx`，一个任务摸到 `c_idx, B+c_idx, 2B+c_idx, ...`），
调度器判成互相重叠 → 串到 1 核。改成 **batch-major**（行 = `c_idx*STATE_LEN + slot`，
每个任务写**一段连续 block**）→ 各任务区间不重叠 → 铺开。

## 现象

```python
# 【慢】slot-major：每个 c_idx 写的行散布全张量 → gather 串到 1 核（847us）
window[(slot)*B + c_idx, :] = ...          # 任务 c_idx 区间 ≈ [c_idx, 7B+c_idx]，几乎全覆盖

# 【快】batch-major：每个 c_idx 写一段连续 [c_idx*SL : c_idx*SL+SL] → 铺到 ~20 核（61us）
window[c_idx*STATE_LEN + slot, :] = ...    # 任务 c_idx 区间 = [c_idx*SL, c_idx*SL+SL)，不重叠
```

附带：**不要用单独的"整张量 init pass"**给这块 scratch 清零/填 NEG_INF——它会建一个覆盖全张量的
def，让后续 gather 任务都挂在上面（WAW）再次串行。改成在每个 gather 任务里"先填自己那段、再覆盖"
（init-all-then-overwrite in-task）。

实测：`window_gather` 1 核（847us）→ 20 核（61us）；`softmax_pool` 690us/1核 → 122us/10核；
全 indexer −20%（这一步之后再叠陷阱 2 的粗化到 −42%）。

> 注：上面是同时改了两件事（layout slot→batch-major + 去掉整张量 init pass），二者都指向"让每个
> 任务的访问区间连续、可证独立"。这与[陷阱 1]同源——都是 pypto 依赖分析对共享张量访问保守。机制细节
> （是按区间重叠判、还是 init pass 的 WAW 主导）未在源码层完全隔离，但**行动结论稳**：让每个
> `pl.parallel` 任务写一段连续 block，别横扫整张量，也别加整张量 init pass。

## 诊断顺序（适用陷阱 1 & 3）

低 Exec% 的 scope 先看 **core-spread（占核数）**：核占太少（1~4）十之八九是 in-place 读写、
loop-carried 张量、或写区间横扫整张量——让访问连续/写 fresh 即解锁。**优先于** mix-fusion / GRP 调参。
