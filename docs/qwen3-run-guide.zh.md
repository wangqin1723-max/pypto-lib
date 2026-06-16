# 在 pypto-lib 仓里运行 Qwen3 的实操指南（中文）

本文整理 `models/qwen3/` 下各 kernel 的**运行入口、命令行、fixture 生成、性能测量**，以及实测踩过的坑。
所有命令默认通过 `task-submit` 下发到真机（conda 环境 `wq3`），CWD 为仓根
`/data/<user>/newpto/pypto-lib`。

---

## 0. 模型与脚本总览

### Qwen3-14B（`models/qwen3/14b/`）

| 脚本 | 可直接运行 | 作用 |
|------|:---:|------|
| [decode_layer.py](../models/qwen3/14b/decode_layer.py) | ✅ | decode 单层单测 + **融合 N 层 decode_fwd**（含 on-device LM head） |
| [prefill_fwd.py](../models/qwen3/14b/prefill_fwd.py) | ✅ | prefill 前向（默认 2 层，可 `--num-layers` 调） |
| [qwen3_14b_l3_generate.py](../models/qwen3/14b/qwen3_14b_l3_generate.py) | ❌ 库模块 | L3 整网生成的 L2 定义，无 `__main__`，被上层 example 调用 |
| [rms_lm_head.py](../models/qwen3/14b/rms_lm_head.py) | — | RMSNorm + LM head kernel |
| [config.py](../models/qwen3/14b/config.py) | — | 维度常量（见 §5） |

> ⚠️ 历史上 run.sh 里 `#测试40层kernel的性能` 用的 `qwen3_14b_decode_full.py`
> **已被删除**（commit `9fd2a02 Replace decode_layer with the manual-scope e2e kernel #449`）。
> 现在 40 层 decode 的入口是 **`decode_layer.py --validate-fwd --fwd-layers 40`**。

### Qwen3-32B（`models/qwen3/32b/`）

| 脚本 | 可直接运行 | 作用 |
|------|:---:|------|
| [qwen3_32b_decode.py](../models/qwen3/32b/qwen3_32b_decode.py) | ✅ | 32B decode |
| [qwen3_32b_decode_4d.py](../models/qwen3/32b/qwen3_32b_decode_4d.py) | ✅ | 32B decode（4D-blocked 布局） |
| `qwen3_32b_prefill_draft.py` | ✅(draft) | prefill 草稿，未入 CI |

---

## 1. 环境前置（每条 task-submit 都隐含）

`task-submit` 下发的 payload **不继承登录 shell 的 env**，需自带激活与导出。最稳的写法是把
`cd <仓根> && <python ...>` 整条塞进 `--run`。`wq3` 环境已配好 ptoas / pto-isa 路径，
正常直接 `python kernel.py -p a2a3` 即可；若手动 bash 跑则参考 [run.sh](../run.sh) 顶部
三段 `export`（`PTOAS_ROOT` / `PTO_ISA_ROOT` 等）。

通用形式：

```bash
task-submit --device auto --max-time 0 --run "cd /data/<user>/newpto/pypto-lib && \
python <脚本> -p a2a3 <参数> 2>&1 | tee <日志>"
```

- `--device auto`：自动抢锁占用一张空闲卡（**不要写死 device 0**，那张常被占/坏）。
- `--max-time 0`：不限执行时长（40 层 + LM head 较慢）。
- `-p`：`a2a3`（真机）/ `a2a3sim`（仿真，只跑 IR-verify + codegen，**不上设备**）/ `a5` / `a5sim`。

---

## 2. 通用命令行参数（`decode_layer.py`）

| 参数 | 含义 |
|------|------|
| `-p {a2a3,a2a3sim,a5,a5sim}` | 平台，默认 `a2a3` |
| `-d <id>` | 设备号（task-submit 下用 `auto` 抢锁更稳） |
| `--enable-l2-swimlane` | **性能测量开关**：产出 swimlane + 逐 func 统计表 |
| `--max-seq` | 把每条 seq 都设成 `MAX_SEQ=4096`（满 KV，稳定最大负载）；默认采样随机变长 |
| `--seed N` | 随机 fixture 的种子（可复现） |
| `--save-data` | 把生成的输入 + golden 落盘以便复用（默认关，大 fixture 才需要） |
| `--validate-fwd` | 跑**融合 N 层 decode_fwd + LM head**，对照 host 链式参考 |
| `--fwd-layers N` | `--validate-fwd` 的层数，默认 4；**40 = 完整 Qwen3-14B 深度** |
| `--data-dir DIR` | `--validate-fwd` 读取预存输入的目录（默认 `build_output/data`） |
| `--smoke` | 仅编译（不上设备）；`*sim` 平台隐含此行为 |
| `--no-dep-gen` | 关闭 dep_gen（避开大图的 register 溢出 / buffer 泄漏，见 §4 坑 3）。**注意 `--validate-fwd` 路径已自动关闭 dep_gen，无需手动加** |

不带 `--validate-fwd` 时，默认跑的是**单层 decode golden 单测**（随机输入 + 即时 torch golden + 真机对比）。

---

## 3. Qwen3-14B decode 全流程

### 3.1 单层 decode 单测（最快的功能/精度自检）

```bash
task-submit --device auto --max-time 0 --run "cd /data/<user>/newpto/pypto-lib && \
python models/qwen3/14b/decode_layer.py -p a2a3 --max-seq"
```

- 通过判据：`[RUN] 'out' PASS`（bf16 容忍 2% 离群，`ratio_allclose atol=rtol=3e-3`）。

### 3.2 生成 40 层所需的 fixture（一次性，关键步骤）

`--validate-fwd` **不自己造输入**，而是从磁盘 `load_inputs` 读那 19 个 `.pt`
（[decode_layer.py:1141](../models/qwen3/14b/decode_layer.py#L1141) `INPUT_NAMES`）。
这些文件由单层路径加 `--save-data` 写出，**但落点是带时间戳的 JIT 子目录**
`build_output/_jit_qwen3_decode_mpmd_<ts>/data/in/`（[golden/runner.py:258](../golden/runner.py#L258)），
**不是** `--validate-fwd` 默认读的 `build_output/data/in/`。而且 JIT 每次编译都会
**清掉旧时间戳目录**，所以不能直接把 `--data-dir` 指向它。

✅ **正确做法：生成后立刻把 fixture 拷到 build_output 之外的稳定目录**：

```bash
# 1) 单层 + --save-data 生成（务必带 --max-seq，让 seq=4096 满 KV）
task-submit --device auto --max-time 0 --run "cd /data/<user>/newpto/pypto-lib && \
python models/qwen3/14b/decode_layer.py -p a2a3 --max-seq --save-data"

# 2) 把刚生成的 in/ 拷到稳定目录（在本地 host 执行即可，文件系统共享）
cd /data/<user>/newpto/pypto-lib
LATEST=$(ls -dt build_output/_jit_qwen3_decode_mpmd_*/data | head -1)
mkdir -p models/qwen3/14b/fixture_40L
cp -r "$LATEST/in" models/qwen3/14b/fixture_40L/
ls models/qwen3/14b/fixture_40L/in/ | wc -l   # 应为 19
```

`fixture_40L/` 在 `build_output` 之外，不会被 JIT 清理，**生成一次后长期复用**
（除非改了 batch / seq / seed）。

### 3.3 跑 40 层融合 decode

`--validate-fwd` 把单层权重沿 dim0 复制 ×40（[decode_layer.py:1514](../models/qwen3/14b/decode_layer.py#L1514)），
跑融合 `decode_fwd`（40 层 + on-device LM head → logits），再对照 host 链式参考算 argmax。

性能和精度**不能在同一次运行里同时拿到**（一个进程不能背靠背承载两个程序的采集器，见 §4 坑 1），
分两次跑即可。两条路径现在**都能干净 exit 0**。

**A. 只要性能数**（开 swimlane）：

```bash
task-submit --device auto --max-time 0 --run "cd /data/<user>/newpto/pypto-lib && \
python models/qwen3/14b/decode_layer.py \
  -p a2a3 --validate-fwd --fwd-layers 40 \
  --enable-l2-swimlane \
  --data-dir models/qwen3/14b/fixture_40L 2>&1 | tee run_decode_40L_swimlane.log"
```

出完性能表后打印 `swimlane perf run complete ...` 并干净退出（不做 argmax 校验）。

**B. 只要正确性**（argmax 对不对；**不开 `--enable-l2-swimlane`**）：

```bash
task-submit --device auto --max-time 0 --run "cd /data/<user>/newpto/pypto-lib && \
python models/qwen3/14b/decode_layer.py \
  -p a2a3 --validate-fwd --fwd-layers 40 \
  --data-dir models/qwen3/14b/fixture_40L 2>&1 | tee run_decode_40L_validate.log"
```

通过判据：`[stacked-fwd 40L+LMhead] argmax match 16/16`。

> 两条命令都**不用再手动加 `--no-dep-gen`**：validate-fwd 路径已在 `run_cfg` 里强制
> `enable_dep_gen=False`（dep_gen 在这个大图上既会崩又无产出，见 §4 坑 1 / 坑 3）。

---

## 4. 已知坑（实测踩过）

### 坑 1：`--validate-fwd` 一个进程内连开两个 device 程序 → 采集器重复注册崩溃（已在脚本里规避）

`--validate-fwd` 是两段式：先跑 `decode_fwd`（一个 device 程序），再在 host 参考循环里
`for _ in range(N): qwen3_decode_mpmd(...)` 又起**第二个** device 程序。**L2 swimlane 和
dep_gen 两个采集器都**不支持一个进程内连开两个 device 程序——第一个程序退出时 pinned host
buffer 没注销干净（`halHostUnregister ... 8`），第二个程序采集器 init 时拿不到 buffer：

```
# 开 --enable-l2-swimlane 时：
[l2_swimlane_collector.cpp:141] Memory registration failed: 8  ->  init_l2_swimlane failed
# dep_gen（曾经默认开）时：
[dep_gen_collector.cpp:66] DepGenCollector: halHostRegister for dep_gen SHM failed: 8  ->  init_dep_gen failed: 8
```

两者都表现为 `RuntimeError: run_prepared failed with code 8`。与设备 / 精度 / fixture 都无关。

**脚本已规避（无需用户额外处理）**：
- **性能路径**（`--enable-l2-swimlane`）：`decode_fwd` 出完 swimlane 表后直接打印
  `swimlane perf run complete ...` 并 `exit 0`，**跳过 host 参考循环**。
- **精度路径**：validate-fwd 的 `run_cfg` 强制 `enable_dep_gen=False`，参考循环不带任何采集器，
  argmax 校验能跑完。
- 因此**两条命令都不必再手动加 `--no-dep-gen`**。性能与精度仍是**两次运行各取其一**
  （同一进程容不下两个程序的采集器），但都能干净退出。

### 坑 2：fixture 找不到（`FileNotFoundError: .../data/in/hidden_states.pt`）

`--save-data` 落点 ≠ `--validate-fwd` 默认读点，且 JIT 目录会被轮换清除。
按 §3.2 拷到 `fixture_40L/` 并用 `--data-dir` 指它即可。**别把 `--data-dir` 指向
`_jit_<ts>` 时间戳目录**——下一次编译就把它删了。

### 坑 3：大图 dep_gen 记录丢失 / register 溢出

40 层图很大，dep_gen 常 `reconcile: NNNN records dropped` + `register failed: 8`，
`deps.json` 不完整（Perfetto 没有依赖箭头）。所以 validate-fwd 路径已**强制关掉 dep_gen**
（见坑 1），不影响逐 func 性能统计。其他路径（如单层单测）若需要可加 `--no-dep-gen` 手动关闭。

### 坑 4：`a2a3sim` 不上设备

`*sim` 平台只跑 IR-verify + ptoas codegen（隐含 `--smoke`），用于本地快速抓
buffer 溢出 / 布局错误，**不产出真机性能数**。性能/精度必须 `-p a2a3` 真机。

---

## 5. Qwen3-14B 关键维度（[config.py](../models/qwen3/14b/config.py)）

| 常量 | 值 | 说明 |
|------|----|------|
| `BATCH` | 16 | 默认 batch（batch=1 在融合 kernel 会 NaN，须 ≥16） |
| `MAX_SEQ` | 4096 | 满 KV cache 序列长度 |
| `NUM_LAYERS` | 40 | 完整深度 |
| `NUM_HEADS` / `NUM_KV_HEADS` | 40 / 8 | GQA，`q_per_kv = 5` |
| `HEAD_DIM` | 128 | |
| `HIDDEN` | 5120 | `= NUM_HEADS * HEAD_DIM` |
| `INTERMEDIATE` | 17408 | FFN 中间维 |
| `VOCAB` | 152064 | LM head 输出维 |

---

## 6. 读性能表（swimlane 输出）

`--enable-l2-swimlane` 会打印 `Task Statistics by Function` 表，关键列：

- **Exec(us)**：AICore 上 kernel 净时间；**Latency**：dispatch→finish（含头尾 OH）。
- **Exec%**：`Exec/Latency`，越高越说明这个 scope 是计算瓶颈、调度开销小。
- **Total Test Time**：最早 dispatch → 最晚 finish 的墙钟，**用这个做版本间对比**
  （`[RUN] PASS` 时间含 30s 的 pto-isa `git fetch`，不可用于对比）。

40 层满 KV（seq=4096）实测墙钟 ≈ **274 ms**，主要热点：

| Func | 量级 | 备注 |
|------|------|------|
| `lm_head` | ~1243 us × 24，≈29 ms | 99.8% Exec，cube-bound |
| `fa_fused_aic/aiv` | ~296 us | 98% Exec，attention 主体（满 4096 KV） |
| `gate/up/down_proj` | ~29 us，94-95% Exec | FFN matmul，效率已很高 |
| `silu__windowed` | Exec 仅 15% | Tail OH ~30us 占大头，潜在优化点 |

> 性能数有 3-5% 单次噪声，**重要结论请重复 3+ 次**；并行作业会抢同一 build 目录
> 互相污染 Total，**性能对比务必单跑**。

---

## 7. Qwen3-14B prefill

[prefill_fwd.py](../models/qwen3/14b/prefill_fwd.py)，默认 2 层，参数：

| 参数 | 含义 |
|------|------|
| `-b/--batch N` | batch（默认 `BATCH=16`） |
| `--num-layers N` | 层数（默认 2） |
| `--max-seq N` | 合成序列长度（≤ 模型 `MAX_SEQ`） |
| `--use-max-seq` | 把所有 seq 设成 `--max-seq` |
| `--chunk-start` / `--chunk-size` | chunked-prefill 合成参数（0 = 整 prompt） |
| `--enable-l2-swimlane` / `--save-data` | 同 decode |

示例（2 层 prefill 自检）：

```bash
task-submit --device auto --max-time 0 --run "cd /data/<user>/newpto/pypto-lib && \
python models/qwen3/14b/prefill_fwd.py -p a2a3 --num-layers 2 --use-max-seq --max-seq 256"
```

精度容忍 `rtol=atol=5e-3`。

---

## 8. Qwen3-32B

```bash
# 32B decode
task-submit --device auto --max-time 0 --run "cd /data/<user>/newpto/pypto-lib && \
python models/qwen3/32b/qwen3_32b_decode.py -p a2a3 --max-seq --enable-l2-swimlane"

# 32B decode（4D-blocked 布局变体）
task-submit --device auto --max-time 0 --run "cd /data/<user>/newpto/pypto-lib && \
python models/qwen3/32b/qwen3_32b_decode_4d.py -p a2a3 --max-seq --enable-l2-swimlane"
```

参数与 14b decode 一致（`-p` / `-d` / `--enable-l2-swimlane` / `--max-seq`）。

---

## 9. 速查

| 目的 | 命令骨架 |
|------|----------|
| 单层 decode 自检 | `decode_layer.py -p a2a3 --max-seq` |
| 生成 40 层 fixture | `decode_layer.py -p a2a3 --max-seq --save-data` → 拷到 `fixture_40L/` |
| 40 层**性能** | `decode_layer.py -p a2a3 --validate-fwd --fwd-layers 40 --enable-l2-swimlane --data-dir models/qwen3/14b/fixture_40L` |
| 40 层**精度** | 同上去掉 `--enable-l2-swimlane` |
| 2 层 prefill 自检 | `prefill_fwd.py -p a2a3 --num-layers 2 --use-max-seq --max-seq 256` |
| 本地仅编译检查 | 任意脚本 `-p a2a3sim`（或 `--smoke`） |
