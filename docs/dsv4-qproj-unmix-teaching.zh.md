# Mix → Un-mix 教学案例：qproj 拆融合让 vec 躲出关键窗

> 案例文件：`models/deepseek/v4/decode_qkv_proj_rope.py` 的 `qproj`。
> 这是 [dsv4-indexer-mix-fusion.zh.md](dsv4-indexer-mix-fusion.zh.md)（讲「怎么融」）的**反面教材**：
> 有时候**拆开（un-mix）反而更快**。本文把这个反直觉点讲透，并给出可复用的改写模板 + 判据。

---

## 1. 背景：什么是 "mix"

一个 scope 里同时有 **cube（matmul）** 和 **vec（收尾，cast/scale/norm…）** 两段，编译器把它拆成
`<name>_aic`（cube）+ `<name>_aiv`（vec）两个 function，但二者**绑在同一个 task 的时间窗里**
——cube 一出结果，vec 立刻在同窗接手。这就是 mix（融合）。

融合的好处：省一次 GM 往返（cube 结果不落 GM，直接喂 vec）、省 scope handshake。
**所以默认我们倾向融合**（见 mix-fusion 那篇）。

**但融合有个隐藏代价**：它把 vec 段**钉死**在 cube 的时间窗。如果那个时间点的 AIV 核
正被**关键路径上的别的 vec scope**要着，两者就抢核 —— 而被融的这段 vec 可能根本不急。

---

## 2. 案例：qproj 算什么

q 投影 = `q = dequant(qr_i8 @ wq_b)`：
- **matmul**：INT8 × INT8 → INT32 累加（cube / AIC）
- **dequant**：INT32 → FP32 × `qr_scale`(逐 token/行) × `wq_b_scale`(逐通道/列)（vec / AIV）

天生横跨 cube + vec。

**关键事实（决定该不该拆）**：qproj 的产物 `q` 有**巨大下游 slack** ——
`q` 在 ~660µs 就备好，但要到 `qk_pv`（~1240µs）才被消费，中间 ~580µs 空闲。
而它的 dequant(vec) 段，在融合时却和**关键路径**的 `qr_proj_aiv`（indexer 的 q 投影收尾）
在 [~300-420µs] 同窗抢 AIV 核。

---

## 3. 代码对照

### MIX（改动前）——一个 scope，matmul 后紧跟 dequant

```python
q_proj_fp32 = pl.create_tensor([T, H*HEAD_DIM], dtype=pl.FP32)
for hg_idx in pl.spmd(..., name_hint="qproj"):              # 一个 scope
    col_acc = pl.create_tensor([T, Q_PROJ_OUT_TILE], dtype=pl.INT32)
    for h_inner in pl.pipeline(16, stage=2):
        for qb in pl.pipeline(...):                          # cube：K 累加
            col_acc = pl.matmul(...) / pl.matmul_acc(...)
        for tc in pl.pipeline(0, T, QPROJ_T_TILE):           # vec：紧贴着反量化
            col_fp32 = pl.cast(col_acc[...], FP32)            # col_acc 当场喂 vec
            col_dequant = col_expand_mul(row_expand_mul(col_fp32, qr_scale), w_scale)
            q_proj_fp32[...] = col_dequant
```

→ 编译出 `qproj_aic` + `qproj_aiv`，绑在同窗。`qproj_aiv` 在 [~311,418] 占 32 个 AIV 核。

### UN-MIX（改动后）——拆成两个 scope，中间用 GM 张量搭桥

```python
q_proj_fp32 = pl.create_tensor([T, H*HEAD_DIM], dtype=pl.FP32)
q_proj_i32  = pl.create_tensor([T, H*HEAD_DIM], dtype=pl.INT32)   # ← 新增：搭桥 GM 张量

# scope A：纯 matmul（只 cube）
for hg_idx in pl.spmd(..., name_hint="qproj_matmul"):
    col_acc = pl.create_tensor([T, Q_PROJ_OUT_TILE], dtype=pl.INT32)
    for h_inner in pl.pipeline(16, stage=2):
        for qb in pl.pipeline(...):
            col_acc = pl.matmul(...) / pl.matmul_acc(...)
        q_proj_i32[:, ...] = col_acc          # ← cube 结果写回 GM，不当场反量化

# scope B：纯 dequant（只 vec）
for hg_idx in pl.spmd(..., name_hint="qproj_dequant"):
    for h_inner in pl.pipeline(16, stage=2):
        w_scale = pl.reshape(wq_b_scale[w_col0 : w_col0+Q_PROJ_OUT_TILE], [1, Q_PROJ_OUT_TILE])
        for tc in pl.pipeline(0, T, QPROJ_T_TILE):
            col_acc_t = q_proj_i32[tc:..., w_col0:...]         # ← 从 GM 读回 INT32
            col_fp32  = pl.cast(col_acc_t, FP32)
            col_dequant = col_expand_mul(row_expand_mul(col_fp32, qr_scale[tc:...]), w_scale)
            q_proj_fp32[...] = col_dequant
```

→ `qproj_dequant`(vec) 不再绑在 matmul 窗；调度器把它**推迟到 AIV 空闲的 [~485,566]**。

---

## 4. 改写的「三步机械动作」（套路）

把一个 mix 拆成 un-mix，本质就三步：

1. **加一个 GM 搭桥张量**：`q_proj_i32 = create_tensor(..., INT32)`（承接 cube 输出、喂给 vec）。
2. **第一个 scope 只留 cube**：matmul 算完 `col_acc` 后不当场 dequant，`q_proj_i32[...] = col_acc`（cube → GM）。
3. **第二个 scope 只留 vec**：新开同 grain 的循环，从 `q_proj_i32` 读回，做 cast + scale，写 `q_proj_fp32`（GM → vec → GM）。

数学一字没改 → **bit-identical**。这就是「**cube → GM → vec 三段式**」。

---

## 5. 为什么拆开反而快

| | MIX | UN-MIX |
|---|---|---|
| scope 数 | 1（cube+vec 绑一起） | 2（cube / vec 各一个） |
| dequant(vec) 跑在哪 | **钉死**在 matmul 窗 [~300-420µs] | 调度器**自由挪**到 AIV 空闲的 [~485-566µs] |
| 对关键路径 | 抢 `qr_proj_aiv` 的 AIV 核 | 让出核，`qr_proj_aiv` 拿满 48 核 |
| `qr_proj_aiv` 完成 | 442µs | **369µs（−73µs）** |
| `gather_kv` 起跑 | 792µs | **760µs（−32µs）** |
| 代价 | 无额外 GM | 多一个 `q_proj_i32` GM 一来一回 |

**核心 insight**：mix 把 vec 绑在 cube 后面 → vec 被迫在那个时间点跑。一旦那个时刻的 AIV 核
正被关键路径要着，就互相抢。拆开后 vec 成独立 scope，调度器看到它的产物有大把 slack，
就推迟到没人抢核的空窗 —— 关键路径因此提前。

**性能数据**（`decode_attention_csa.py`，真机 a2a3，x_out PASS；同 session 2×2 对照，区间不重叠）：
`un-mix 2139.8 / 2167.0` vs `mix 2193.4 / 2251.5` → **~−50µs**。

---

## 6. 什么时候该 un-mix（判据）

**两条同时满足**才值得把一个 mix 拆开：

1. **被融的 vec 段，和关键路径上的另一个 vec scope 抢同一批 AIV 核**（时间窗重叠）；
2. **本 scope 的产物离它的消费点很远（有大把 slack）** —— 这样拆出来的 vec 才挪得走。

qproj 两条都满足：`qproj_aiv` 与关键 `qr_proj_aiv` 在 [300-420] 重叠；`q` 备好@660、用@1240。

> 反过来：如果 (1) vec 段不抢关键路径的核，或 (2) 产物马上要用（无 slack），就**别拆**——
> 拆了只多一次 GM 往返、没收益。默认仍应**融合**（省 GM）。

---

## 7. 代价与一个失败的"二次优化"（实测过）

un-mix 引入了 `q_proj_fp32` 的 16MB GM 一来一回。直觉上想「把这个 dequant 再融进后面的
vec scope（`q_head_rms_nope` / `q_head_rope_fused`）省掉 round-trip」。**实测：neutral，不值得。**

- **UB 爆**：dequant 折进 `q_head_rms_nope` 后，多出的 cast/expand 临时 tile 把 Vec 顶到
  199168 B > 196608 B（192KB），只能把 RMS/NOPE 流水从 stage=2 降到 stage=1 才编得过。
- **2× dequant**：RMS 那遍 + NOPE/rope 那遍各算一次（整个 head [128,512]FP32=256KB 装不下 UB，没法缓存复用）。
- **Total 持平**：2165.88µs，落在 un-mix 区间高端 —— 因为被删的 GM 往返**本来就花在关键路径外的空闲时间**，删掉它省 HBM 但不动 wall-clock。

→ 结论：**un-mix 的 GM 往返是「用 slack 付的账」，不花 wall-clock，所以没必要再消除它。**

---

## 8. 一句话总结

> **默认融合（省 GM）；但当被融的 vec 收尾 ① 抢关键路径的 AIV 核、② 自己又有大把 slack 时，
> 用「cube → GM → vec 三段式」把它拆成独立 scope，让调度器把它挪去空窗 —— 拆开反而快。**

判据、泳道路径见 `docs/swimlane-tuning-log.zh.md` §14。
