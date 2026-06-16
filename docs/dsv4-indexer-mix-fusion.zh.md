# DSv4 Indexer Mix 融合记录

记录在 `models/deepseek/v4/decode_indexer.py` 与 `decode_indexer_compressor.py` 上的
cube+vector mix 融合尝试。目标:把 matmul(cube)+ 收尾(vec)折进一个
`pl.at(level=CORE_GROUP)` scope,省 scope handshake / GM 往返。

## 已合并(保留)

| scope | split | 说明 |
|---|---|---|
| qr_proj、qr_hadamard | UP_DOWN | cube + vec 单累加(原有) |
| **rope_fused**(slice matmul + cos/sin apply + assemble matmul + write 四合一) | **UP_DOWN** | 见下方"rope 四合一"。真机 PASS,Total 1094us vs ~1357 基线(≈−19%) |
| weights_proj + weights_write | UP_DOWN | scale 内层 row-chunk,使 Vec 收尾 ≤192KB(整 T=288KB 超) |
| **score_fused**(KV INT8 quant + score matmul + dequant/ReLU/加权 reduce 三合一) | **NONE** | 见下方"score 三合一"。真机 PASS,Total 持平(~1094us,score 不在关键路径);去掉 kv_tile_i8_g / score_kv_scale 两个 GM 往返 |

## score 三合一(quant + accum,已解决)

把 score_quant(KV INT8 量化)折进 score 的 matmul+dequant+reduce scope,量化后的 KV 不再经
`kv_tile_i8_g` / `score_kv_scale` GM 往返。关键:

- **只能 NONE**(不能 UP_DOWN):末尾 `row_sum([32,64])→[32,1]` reshape 成 `[1,32]`,32 个 cache
  位置塌进一个输出行,行切会跨子块耦合。
- **不能拼整块 INT8 KV 也不能整块量化**:`kv_i8_tile[:, h:h+32] = ...` 的列子视图写在 NONE idle
  子块(valid=[0,0])触发 ptoas `'pto.subview' op valid_row must be positive when constant`;
  整块 `[32,128]` 量化则 FP32 临时量把 Vec 顶到 210816 > 196608。
- **解法 = K-tiling matmul + 即时按 `[32,HEAD_DIM_CHUCK]` 量化**:每个 K-chunk 现量化现喂进
  matmul 的 INT32 累加,不拼整块、临时量只 `[32,32]`。per-row scale(全 128 维 amax)逐 chunk 应用 +
  INT32 K 累加 → 位级等价。代价:量化在 s 循环内重算 S 次(S=2),score_fused 单 task exec 偏高,
  但 score 非关键路径,Total 持平。

## rope 四合一(已解决,曾以为"待底层修")

把 rope_slice / rope_apply / rope_assemble / rope_write 折进一个 `pl.at(CORE_GROUP)` scope
(内层 `pl.range(0, HEAD_ROWS_ROPE, IDX_N_HEADS)`),关键是**两处**:

1. **`pl.split(NONE)` → `pl.split(UP_DOWN)`**。NONE ≠ "不拆 AIV 子块":SplitVectorKernel
   照样发 `if subblock_idx==0 / else` 分流,NONE 让 work 全压 subblock 0、subblock 1 成
   `valid_shape=[0,0]` 空壳克隆,分配器又不做分支互斥感知,把两支 Vec buffer **叠加** →
   c2v pipe 64KB + if 64KB + else 副本 64KB = **197120 > 196608(超 512B)**。
   UP_DOWN 把逐行独立的 rope work **行切到两个 AIV 子块**,每块 tile 半高 `[32,…]`,Vec 塞下
   (per-row 独立 → 位级等价、精度安全)。

2. **assemble 用单次 `pl.matmul`(整块 K=ROPE_HEAD_DIM//2=32),不再 K-tiling**。
   UP_DOWN 行切后 rope_even/rope_odd 物理是 32 行,但 `rope_even[:, k0:k0+16]` 列子视图仍按
   切前静态行数 64 → ptoas `'pto.subview' op expects result valid_shape[0] to match valid_row`
   (64 vs 32)。整块喂进去就没有这个子视图;K=32 一把乘足够。

## 附:subview "align" 是什么(图解)

`pl.slice` / `t[r0:r0+R, c0:c0+C]` 取出的是父 tile 的一个**子窗口**,不拷贝数据,只记
`offset`(起点)/ `sizes`(静态标称形状)/ `valid_shape`(真正有效的行列数)三个字段。
硬件指令拿这份描述去寻址。"align" = 这份描述必须同时满足两类约束,否则 ptoas 直接 assert。

```
父 tile rope_even  标称 sizes=[64,32]
┌───────────────┬───────────────┐
│               │███████████████│  t[:, 16:32] 子视图:
│   不属于它      │██ 框住的区域 ██│   offset=[0,16] sizes=[64,16]
│               │███████████████│   valid_shape=[64,16]
└───────────────┴───────────────┘
```

### 约束① 地址对齐(32B block / 16×16 分形)

local buffer 按 **32B 一个 block** 寻址(bf16: 1 block = 16 元素),矩阵排成 **16×16** 分形块。
切片起点和宽度必须落在格子边界上:

```
│0────15│16───31│32───47│ ...     ← block 边界
✅ t[:, 16:32]  起点16 宽16  = blk1        对齐
✅ t[:, 0:32]   起点0  宽32  = blk0+blk1   对齐
❌ t[:, 5:21]   起点5  宽16  横跨 blk0/blk1 错位   ← 仅为"错位长啥样"的泛化示例
```

> ⚠️ 上面的 `5:21` 只是演示"非 16 倍数会错位",**不是 rope 的真实切法**。
> rope 切的是行(64/32/48),32 本身就是 16 的倍数、行 offset 也都是 16 倍数,
> 所以 "32 行踩 align" **不属于这种 offset 错位**。其真实根因当时只记了结论、
> 未留 ptoas 日志,更可能是约束②那一类(待查证,勿当定论)。

### 约束② valid_shape 必须 == 实际 valid_row(rope 真正踩的坑)

`pl.split(UP_DOWN)` 把 tile **按行劈成两半**分到两个 AIV,每块实际只剩 32 有效行;
但在劈分后的 tile 上再取**列切片**,子视图仍沿用劈分前的静态行数 64:

```
split 前 64 行整块          UP_DOWN 后物理劈两半,各 32 有效行
┌──────────────┐          ┌────────┐  ┌────────┐
│ valid_row=64 │   ──►     │上32 行 │  │下32 行 │  valid_row=32
└──────────────┘          └────────┘  └────────┘
```

```python
# ❌ 行劈分后还按列 K-tiling
for k0 in pl.range(0, 32, 16):
    sub = rope_even[:, k0:k0+16]      # valid_shape[0]=64 (切前静态)
    rope_acc = pl.matmul_acc(rope_acc, sub, even_select[k0:k0+16], b_trans=True)
# ptoas: valid_shape[0]=64 != valid_row=32  → ASSERT ✗

# ✅ 整块喂, K=32 不 tile → 不产生列子视图, 自然没有 valid_shape!=valid_row
rope_acc = pl.matmul(rope_even,  even_select, out_dtype=pl.FP32, b_trans=True)
rope_acc = pl.matmul_acc(rope_acc, rope_odd,  odd_select,        b_trans=True)
```

### 收口

```
              subview "align" 两类约束
        ┌──────────────┴──────────────┐
   ① 地址对齐                    ② valid_shape == valid_row
   落在 32B block / 16×16        切片静态行数 == 该 AIV 子块实际有效行
   分形边界                      踩它: UP_DOWN 行劈分后 + 列切片
                                绕法: 整块喂 matmul, K=32 不列 tile (已修✅)
```

`build_output/.../passes_dump/32_after_AllocateMemoryAddr.py` 看逐张量 Vec 偏移(report/*.txt 只有 high-water 汇总)。
a2a3sim 上这个双 AIV mix **runtime 会 STALL 死锁**(`TIMEOUT_EXIT`,各 cluster 卡在 `aic:2 aiv0:3 aiv1:3` RUNNING)——
是 sim 对双 AIV mix 的已知不可靠;codegen 仍能本地验证,精度/性能以真机 `-p a2a3` 为准。

## 合不了 — 根因 + issue

| scope 组合 | 失败阶段 | 根因 | issue |
|---|---|---|---|
| ~~rope_slice + rope_assemble~~ | **已解决** | UP_DOWN + 整块 assemble,见上方"rope 四合一";曾以为差 512B 待底层修 chunk subview,实为 NONE 的 idle-AIV 副本被叠加,换 UP_DOWN 即解 | transpose #198 / mem_acc #1523 / #1525 已修 |
| ~~quant + accuπm~~ | **已解决** | 真因不是 set_validshape,而是拼 INT8 KV 的列子视图写在 idle 子块 valid_row=0;改 K-tiling+即时量化即解,见上方"score 三合一" | — |
| kv_hadamard + kv_and_cache_write | runtime | matmul+per-row scatter:SplitVectorKernel 第二路 AIV valid_shape=[0,0] → 单写全零、双写 507018 死锁 | [#1507](https://github.com/hw-native-sys/pypto/issues/1507) 评论 |
| kv_score_proj + state_scatter | IR | 跨 [B,S] 重排需 3D reshape,tile.slice 不支持 >2D | —(设计限制) |

## 根因三类
1. **同参数双向加载** — transpose pass 硬限制。
2. **cube→vec→cube 串联** — mem_acc 漏映射 / valid_shape=[0,0]:#1523、#1525、#1507、#1352。
3. **cube→vec 双 AIV 拆分** — 第二路 valid_shape 清零,scatter 行丢/死锁。

## 关联 issue
- #1523 — matmul_acc 链 mem_acc crash(已提）
- #1525 — 单 matmul 链全零(已提）
- #1507 — cube→vec 桥丢 valid_shape;+2 评论(UP_DOWN reduce 漂、scatter 双 AIV)
- #1352 — acc MemRef 基址不统一(既有）
- #362(pypto-lib）— UP_DOWN 合并 matmul+dequant ~1-ULP bf16 漂

## 复现脚本(仓根）
- `repro_transpose_param.py` — 同参数双向 → pass#198
- `repro_two_accum.py` — matmul_acc 链 → mem_acc crash
- `repro_split_param.py` — 单 matmul 链 → device 全零
- `repro_matmul_scatter_deadlock.py` — matmul+per-row scatter → 全零/死锁

## 不可融的根本结论
能融:cube+vec 单累加(qr_proj/hadamard/weights）、cube+vec 单累加+per-s(score）、
**cube→vec→cube→vec 四合一(rope,UP_DOWN + 整块 assemble，逐行独立才行)**。
不可融:① matmul+scatter ② INT8 动态子视图 ③ 3D scatter。
accum+store 是"能融但只能 NONE"。
**经验:逐行独立的 mix scope 若 Vec 略超,先试 UP_DOWN(行切到双 AIV、buffer 自动减半），
别去切中间量(切 UP_DOWN tile 的列子视图会撞 valid_row 墙)。**
原以为的"同参数转置冲突"在 #198 修后已不再是 rope 的障碍。

## 5.28 增量(#395 把 compressor rope 拆回后重新融 + 四个新尝试)

背景:#395/#400 做 per-row start_pos 重构时把 compressor 的 rope mix 拆回了
rope_slice/rope_apply/rope_assemble/rope_write 四个独立 scope。本轮重新融回,并顺手试了
另外几处。结果:**2 个成、1 个物理不可行、2 个撞底层墙**。

### 成了

| scope | 文件 | split | 说明 |
|---|---|---|---|
| **rope_fused**(slice matmul + rotate + assemble matmul + write) | decode_indexer_compressor.py | UP_DOWN | 把 #395 拆开的四段重新合一,recipe 同 decode_indexer 的 rope_fused(整块 assemble K=32) |
| **weights_proj + weights_write** | decode_indexer.py | 无(行 chunk) | 见下"weights 行 chunk 配方" |

### weights 行 chunk 配方(终于落地,doc 老版"已合并"其实从没进过 main)

cube(matmul)+vec(mul) 融合,坑在 **fuse 后 weights_acc 全 T 常驻**,撞两堵墙,逐一解:

1. **Vec 超(294912 > 196608)**:fuse 让 `weights_acc[T,IDX_N_HEADS]` 经 c2v 全常驻 + mul 输出
   → 实测 **~IDX_N_HEADS*36 B/row**(matmul-out staging + mul in/out + UP_DOWN NZ padding,
   约 4.5x 朴素 2-FP32-tile 估算,**必须实测不能估**)。
2. **Mat 超(1056768 > 524288)**:把整块 `x_flat[t0:t0+CHUNK, :]`(全 D 列)pre-slice 成左操作数
   → 整个 [CHUNK,D] 被 pin 进 L1(1MB)。
3. **UP_DOWN 在这里反作用**:让 matmul 的 L1 输入翻倍(Mat 又超),且没能把 weights_acc 行切。

**解法 = 整段 row-chunk(matmul+mul 一起按 WEIGHTS_ROW_CHUNK 切)+ 按 K-tile 直接索引 x_flat
(不 pre-slice 整 [CHUNK,D])+ 去掉 UP_DOWN**。每 chunk:Mat ~45KB / L0C 73KB / Vec 147KB 都进限。
`WEIGHTS_ROW_CHUNK = T; while CHUNK*IDX_N_HEADS*36 > 196608: CHUNK//=2`(T=128,heads=64 → 64)。

### 物理不可行:qr_hadamard 不能 double

想把 qr_hadamard 任务做大(GRP 4→8),报 **Acc buffer 262144 > 131072**:
**L0C 硬上限 128KB**,GRP=4 的 `[HEAD_ROWS_ROPE, IDX_HEAD_DIM]` FP32 已正好占满,GRP=8=256KB。
这个 kernel"看着小"就是因为它已经顶在 L0C 天花板上,不是没调。

### 撞底层墙:rmsnorm 折不进 UP_DOWN rope_fused(新规律)

想把 rmsnorm_rope_slice(vec)折进 rope_fused 前缀(POST_CHUNK=16 行极小,buffer 完全够),
报 codegen **`Tensor view not found for parameter: kv_rope_inline...`**。根因比 doc 老结论更精确:

> **在 UP_DOWN scope 内,`pl.assemble` 进预建张量、再喂给 matmul 的中间量,行切后无可解析 tile view → codegen 挂。
> 对比:`rope_even = pl.cast(...)` 这种干净 SSA 中间量喂 matmul 是 OK 的(所以 rope_fused 本身能成)。**

- 失败:`kv_rope = pl.create_tensor(...)` 后 `kv_rope = pl.assemble(kv_rope, normed_bf16, [0,off])` → matmul。
- 成功:rope_fused 里 kv_rope 是**外部 scope 输入**(rmsnorm_rope_slice NONE scope 产出),view 干净。

**推论(不用试也知道挂)**:`kv_hadamard` 想并进 rope_fused(cube+cube)也会挂——hadamard 左操作数
`normed_kv` 是 assemble 拼出来的,同一堵墙。

### compressor 融合上限(本轮结论)

decode_indexer_compressor.py 已融到当前 pypto2 codegen 上限:
- 能融的已融(rope_fused 四合一、kv_score_proj 双 matmul)。
- 剩余全是死路:① kv_hadamard+kv_and_cache_write = #1507 matmul+scatter 死锁(sim 测不出)
  ② 任何"assemble 中间量→matmul under UP_DOWN" = view-not-found ③ scatter/3D 设计限制。
都不是改写法能绕的,是 lowering 限制。
