# DSv4 Compressor:kv_hadamard + kv_and_cache_write 融合尝试(未合,507018)

记录在 `models/deepseek/v4/decode_indexer_compressor.py` 上,把 `kv_hadamard`(cube matmul)
与 `kv_and_cache_write`(逐行 block-table scatter 写 `kv_flat` + paged `idx_kv_cache`)
融成一个 mix scope 的尝试。

**一句话结论:字面 mix 合不了 —— 编译干净但运行时 507018 死锁;触发点经实测隔离 = mix scope
里 vec 侧的「动态地址 GM 写」。最终保留原 2-pass 设计,并提了 pypto#1592(只含已证实的现象)。**

## 背景 / 动机

- 现状是两段:Pass 2 的 `kv_hadamard`(cube,POST_CHUNK 组,产 `kv_final` → GM `kv_final_all`)
  + Pass 3 的 `kv_and_cache_write`(per-batch、block-table 绑定,从 `kv_final_all` 读、scatter 到
  `kv_flat` + paged `idx_kv_cache`)。
- `kv_final_all` 是**故意的解耦**(源码注释):让 Pass 2 纯计算、在 64 个 batch 上铺开多核;
  Pass 3 是 block-table 绑定的 scatter,被串行化但很小。参见
  [[pypto-inplace-rw-parallel-pitfall]] / `feedback_pypto_inplace_rw_serializes_parallel_loop`。
- 之前认为合不了的原因记在 #1564(SplitVectorKernel 退化 subblock)。本轮前提:**#1564 已修**,
  故重试。
- 注意:两个 scope **本来就用对了输出绑定**(都是 `pl.assemble`),所以这跟 bare-reassign 的
  "静默全零"(见 [[feedback_pypto_mix_kernel_assemble_or_silent_zero]] / pypto#1525)**无关**。

## 尝试与结果(a2a3 真机)

| 方案 | 结果 |
|---|---|
| **字面 mix**:hadamard(cube)+ 逐行 gated 动态地址 scatter(vec)塞进一个 scope,`NONE` | 编译干净,运行时 **507018** |
| 同上,`UP_DOWN` | 同样 **507018** |
| **dataflow 融合**:hadamard 一个 cube scope + scatter 另一个独立 scope,都在 Pass 2 的 c0 循环里(去掉 `kv_final_all` + Pass 3) | **PASS**(default + hetero start_pos),但**重耦合**了原本特意解耦的 block-table scatter,有掉性能风险 → **放弃** |

> 字面 mix **编译干净** = #1564 的 codegen 退化修复确实生效;问题纯在运行时。

## 隔离实验(定位触发点,均 a2a3 真机)

| mix scope 里 vec 侧的操作 | 结果 |
|---|---|
| 动态地址**写** `idx_kv_cache_flat[cache_row] = ...`(完整版) | **507018 死锁** |
| 只连续地址**写** `kv_flat[c_idx*S] = ...`(V2,去掉 block-table 读 + idx_kv_cache 写) | **跑通** |
| 动态地址**读** `idx_kv_cache_flat[cache_row]` 折进值 + 连续写(V3) | **跑通** |
| 现有 `score_fused`(NONE mix,block-table **读** + 连续 `score_flat` 写) | 一直 **OK** |

**→ 触发点 = mix scope 里 vec 侧的「动态地址(数据依赖)GM 写」。block-table 读、连续地址写都没事。**

补充:default `START_POS = COMPRESS_RATIO-1 = 3` 是均匀的,gate `pos_b+S>=COMPRESS_RATIO`(`3+2>=4`)
对所有行恒真 → 死锁那次**没有逐行 gate 发散**,所以**不是** cube/vec 数据依赖发散导致的。

## 确定 vs 不确定(重要)

- **确定(实测现象)**:cube+vec mix scope 里 vec 侧的**动态地址 GM 写 → 507018**;动态读 / 连续写不挂。
- **不确定(机制 + 层)**:为什么挂、卡在哪条 lane、归 pypto codegen / runtime / pto-isa FFTS ——
  **没有定论**。本轮试过两个机制解释,**都不成立或没证实**,在此明确记下避免后人重蹈:
  1. ❌ "AIV1 漏了 store 的 MTE3 同步 → 卡在循环后的 `wait_flag`":**证伪**。生成的
     `kv_hadamard_cache_write_aiv.cpp` 里,两条 lane 在 if/else **之前** L148-150 都
     `set_flag(PIPE_V/MTE3, ...)` 三个,**之后** L209-211 都 `wait_flag` 三个,AIV1 是 3 set ↔ 3 wait
     **自洽**,卡不住。
  2. ⚠️ "inactive lane(`else`)没忠实复刻 store 路径(指令 + 同步),违反 NONE 下 AIV1 应跑全指令流
     的契约":这是个**能从 cpp 看到的观察**(AIV1 的 else 分支只有 `(0,0) TCVT` + `pipe_barrier(PIPE_V)`,
     没有 TLOAD/TSTORE 和那套 MTE 同步),但**没证明它就是 507018 的原因**——只是观察,不是定因。

  教训(同 pto-isa#149):**别抓一个看着合理的机制就当结论**;尤其别把没证实的机制塞进共享仓的 issue。

## 决策

**保留原 2-pass 设计**(`kv_hadamard` → `kv_final_all` → Pass 3 `kv_and_cache_write`)。它正确、且为多核铺开
特意解耦。kernel 文件已还原(`git diff` 空,无残留实验改动)。

## 产出

- **pypto#1592**:`[Bug] Dynamic-address GM scatter on the vec side of a cube+vec mix CORE_GROUP scope
  deadlocks (507018); dynamic reads + contiguous writes are fine`。**只含已证实的实测现象 + 隔离矩阵 +
  最小 repro**;明确声明根因/层未定,lane 非对称只作 observation,交给懂 FFTS 的人 root-cause。

## 复现文件(本轮新建,仓根目录)

- `repro_1525_none.py`、`repro_updown_assemble_min.py`、`repro_none_assemble_min.py`
  (bare-reassign / 正确绑定 idiom 的对照,#1525 相关;非本融合直接 repro,但同期产物)
- 本融合的 repro 是直接改 `decode_indexer_compressor.py` 跑 `compile_only` / 真机得到的,改动均已还原。

## 环境

pypto `7e1bbd0a` / runtime `324df3d6` / ptoas `0.43`(CI pin `v0.41`)/ pto-isa `0f171f73` /
pypto-lib `25ac6cf`(分支 `perf/dsv4-decode-indexer-rope-core-spread`)/ CANN 9.0.0 / a2a3 aarch64。
