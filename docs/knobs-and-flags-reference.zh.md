# 开关 / 代码作用速查

记录一些具体开关 / DSL 旋钮 / 编译路径**到底干了什么**。每条先给一句话结论，细节按需展开。

---

## `pl.split(mode, slot_num=N)` —— 跨核 ring 深度

**一句话：** 设定 cube↔vec 之间 `tpush`/`tpop` 环里有几个 slot，也就是 cube→vector 交接的流水深度。用 buffer（UB/L0C）换 cube-vec 重叠度。

- **Ring 大小** = `slot_size × slot_num`。`slot_size` 由 tile 类型推断，`slot_num` 就是深度。
- **默认（不设旋钮）：** 写死 单向 `8` / 双向 `4`（`GetSlotNumForDirMask`，`cross_core_pipe.cpp`）。
- **`slot_num` 大** → 生产者领先更多，cube/vec 重叠好，但占 UB 多；**小** → 腾出 UB（可放大 kernel tile），但流水变浅。
- **什么时候用：** 融合 matmul+vec 的 scope，自动插入的 ring 撑爆 192KB Vec 预算时（如 INT8 w2 matmul+dequant：`32768 × 8 = 256KB`）。缩 `slot_num` 比缩 kernel tile 更直接。
- **来源：** pypto PR #1774（解决 issue #1472 reopen 部分），commit `4a532dd3`。在此之前深度只能在手动 `pl.aic/aiv_initialize_pipe` 路径上调（PR #1584）。

**坑 1 —— 只对 `pl.split(UP_DOWN / LEFT_RIGHT)` 生效：** 解析器拒绝在 `SplitMode.NONE` 或没写 `pl.split` 的 scope 上传 `slot_num`（`ast_parser.py:3232`）。这是**前端 API 校验**，不是底层做不到——`BuildAutomaticPipeSetup` 对任何 mixed scope 都通用地从 func attr 读 `slot_num`。NONE-mode 的 mixed scope（如 `hc_pre`）照样有 ring，只是永远是默认 8/4。

**坑 2 —— scope 本身得吃得下这个 split：** 给手工调过的 scope 加 `UP_DOWN`/`LEFT_RIGHT` 可能直接编不过（跟 `slot_num` 无关）：
- `UP_DOWN`（行折半）→ `memory_reuse_pass` 无法 rebase 折半后尺寸不一致的 reshape 共享组。
- `LEFT_RIGHT`（列折半）→ `SplitVectorKernel` 拒绝在 split 轴上的 `row_sum`/规约（"partial reduction not supported"）。

`models/deepseek/v4/hc_pre.py` 的融合 scope `hc_pre_1spmd` 两个坑都踩 → 这个旋钮在那儿用不上。`exp_w2` / `proj_a` / `proj_b` 目前用的是老办法（缩 kernel tile / 把 vec 侧按 T 切小），没用这个旋钮。

**坑 3 —— 深度 > 默认值会静默把输出炸成 NaN（实测）：** 在 `models/qwen3/14b/decode_layer.py` 的 `fa_fused`（UP_DOWN 双向、默认 slot_num=4）上实测：

| slot_num | Total Test Time（3 次） | 精度 |
|----------|------------------------|------|
| baseline（不指定=4） | ~944.6 us | ✅ PASS 3/3 |
| 2（更浅） | ~970 us | ✅ PASS（更慢） |
| 4（显式，=默认） | ~923 us | ✅ PASS 3/3 |
| 8（更深） | ~904 us | ❌ **FAIL 3/3（NaN）** |

- 趋势单调：ring 越深越快（cube/vec 流水重叠更多），但**>默认值会确定性地算错**。
- `slot_num=4`（显式）≈ baseline 且 PASS → override 配线本身没问题，bug 只出在 depth > 默认。
- `slot_num=8` 的 FAIL 不是「偏差大一点」，而是**几乎全 NaN**：`illegal values in actual: NaN=76800 Inf=0`，输出 (16,5120) 里 76800=15×5120，即 16 个 batch 行有 15 行全 NaN。疑似更深的 C2V/V2C ring 读到未初始化/陈旧 slot，再被 `online_softmax` 归约扩散。
- **结论：默认 4 是这里的正确性上限，拿不到可用提速。** `slot_num` 本应只改缓冲深度、绝不该让结果变 NaN → 疑似 pypto bug（PR #1774 只验证了 reserved-buffer 大小，没验真实 attention scope 的输出正确性）。**调 `slot_num` 时必带精度校验**，别只看 Total Test Time。

**想让 NONE / 自动 mix 的 scope 也能调 `slot_num`：** 要同时（1）放开解析校验 `ast_parser.py:3232`，（2）改打印器 `python_printer.cpp` 里 `split_ != None` 那几处发射判断，否则 print→reparse 丢掉 slot_num、`structural_equal` 会挂。更干净的设计是把 `slot_num=` 做成 `pl.at` / `pl.spmd` 的独立 kwarg，而不是塞进 `pl.split(NONE, ...)`。
