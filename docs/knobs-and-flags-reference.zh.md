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

**坑 1（已修复）—— `SplitMode.NONE` 现在也能传 `slot_num` 了：** 早期解析器拒绝在 `SplitMode.NONE` 的 scope 上传 `slot_num`，需要写成 `UP_DOWN`/`LEFT_RIGHT`。**当前 pypto HEAD 已放开**（`parser/ast_parser.py:3175-3179`，注释 "slot_num is valid with any split mode, including SplitMode.NONE"）。所以 NONE-mode 的 mixed scope（如 `hc_pre`）现在可以写 `pl.split(pl.SplitMode.NONE, slot_num=N)` 显式调环深，2026-06-24 实测能编能跑（见下方 hc_pre 扫描）。注意：**不写任何 `pl.split` 的裸 `pl.spmd` 仍走自动默认 8/4**，要调就得显式带 `pl.split(NONE, slot_num=N)`。

**坑 2 —— scope 本身得吃得下这个 split：** 给手工调过的 scope 加 `UP_DOWN`/`LEFT_RIGHT` 可能直接编不过（跟 `slot_num` 无关）：
- `UP_DOWN`（行折半）→ `memory_reuse_pass` 无法 rebase 折半后尺寸不一致的 reshape 共享组。
- `LEFT_RIGHT`（列折半）→ `SplitVectorKernel` 拒绝在 split 轴上的 `row_sum`/规约（"partial reduction not supported"）。

`models/deepseek/v4/hc_pre.py` 的融合 scope `hc_pre_1spmd` 两个 split 模式都踩坑 → `UP_DOWN`/`LEFT_RIGHT` 用不上。但走 `NONE` 仍可调 `slot_num`（坑 1 已修复），只是**调了拿不到性能**（见下方实测）。`exp_w2` / `proj_a` / `proj_b` 目前用的是老办法（缩 kernel tile / 把 vec 侧按 T 切小），没用这个旋钮。

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

**实测 —— hc_pre（NONE-mode mix）扫 `slot_num`，拿不到性能（2026-06-24）：** `models/deepseek/v4/hc_pre.py` decode 形态（B=64, S=2, T=128），用 `pl.split(pl.SplitMode.NONE, slot_num=N)`，L2 swimlane Total Test Time（每档 1 个干净样本）：

| slot_num | Total Test Time | 结果 |
|----------|-----------------|------|
| 1 | 115.96 us | ✅ PASS，无双缓冲、push/pop 串行，**慢 ~27%** |
| 2 | 91.26 us | ✅ PASS |
| 4（≈自动默认） | 90.34 us | ✅ PASS，与 2 差 ~1%（3-5% 噪声内，等价） |
| 8 | —— | ❌ **编译 FAIL**：`AllocateMemoryAddr` 校验失败（ring buffer 超分） |

- 唯一真实收益在 `1→2`（双缓冲打开）；`2→4` 平、`4→8` 直接编译挂。
- hc_pre 失败方式和 `fa_fused` 不同：这里是**编译期 buffer 超分**（不是运行期 NaN），因为 NONE-mode 不折半、ring 占满 UB 预算，深度一加就编不过。
- **结论：slot_num 不是 hc_pre 的杠杆。** 墙钟卡在串行 `matmul_acc` 的 K-chain（K=16384），加深 ring 移不动它。hc_pre 的真实杠杆是 `LINEAR_K_TILE 128→256`（已验证 −19.7%）。

**残留待办（让 NONE / 自动 mix 完整支持 `slot_num`）：** 解析校验已放开（坑 1）。若 print→reparse 仍丢 `slot_num`，需检查打印器 `python_printer.cpp` 里 `split_ != None` 那几处发射判断（否则 `structural_equal` 会挂）。更干净的设计仍是把 `slot_num=` 做成 `pl.at` / `pl.spmd` 的独立 kwarg，而不是塞进 `pl.split(NONE, ...)`。
