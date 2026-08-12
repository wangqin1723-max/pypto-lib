# Qwen3-14B 40 层 Decode × Runtime PR #1137 — A/B 性能测试计划

> 目的：量化 simpler runtime PR #1137（*Replace wiring with polling-based task readiness test*，作者声称 ~17% median device speedup）对 Qwen3-14B 40 层 decode 的真实影响，给出"按上下文长度衰减"的 win 曲线，作为是否采纳的依据。

> 路径约定：以下命令运行前，按本机布局初始化一次；激活 `wq3` 环境后，Python
> 环境根目录统一使用 conda 自动设置的 `$CONDA_PREFIX`。

```bash
export WORKSPACE=/path/to/workspace
export PYPTO_ROOT="$WORKSPACE/pypto"
export PTO_ISA_ROOT="$WORKSPACE/pto-isa"
export PTOAS_ROOT=/path/to/ptoas-bin
```

---

## 1. 背景与变量

### 1.1 被测改动（自变量，唯一）
PR #1137 是 runtime 调度层改动：把 task readiness 的 **wiring（事件/信号）** 换成 **polling（轮询内存）**。本质是降低 per-task 的 dispatch/readiness 开销。

| 侧 | runtime commit | pypto 侧 |
|---|---|---|
| **before** | `a756969c`（当前 HEAD，= PR #1137 的精确 base） | `7759b264`（锁 submodule 于 `a756969c`） |
| **after** | `385ad7d8`（本地 ref `pr-1137`，分支 `polling-pr-minimal`） | 同 `7759b264`（pypto 不动） |

> `merge-base(385ad7d8, a756969c) = a756969c`，PR 的 5 个 commit 干净叠在 base 上，**没有其他 commit 串味**。pypto/compiler 两侧完全一致，只有 simpler runtime 这一个变量。这是干净 A/B 的前提。

### 1.2 固定变量（两侧必须严格一致）

| 项 | 值 | 备注 |
|---|---|---|
| batch | 16 | `-b` 默认；也是 fused-decode 的安全下限（`<16` 会 NaN） |
| num_layers | 40 | `--num-layers 40` |
| seed | 0 | `--seed 0`，`seq_lens` 逐 run 确定性 → before/after 输入逐位一致 |
| ptoas | 0.48 | `--ptoas 0.48`，两侧同版本 |
| die | 锁定**单张偶数 die** | even=primary（PCIe-attached，~313us 基准）；odd=secondary 慢 20-25% |
| 环境变量 | 见 §6 模板 | `PTO2_RING_*` 两侧同值 |

### 1.3 扫描变量
`--max-seq` ∈ { **128, 1024, 4096** } —— 调度 bound → 带宽 bound 的两端 + 中点。
- 128：投影 matmul + 调度/dispatch 主导 → **PR #1137 最可能在这里显现**。
- 4096：attention KV-load（KV cache ~2.5GiB）主导带宽 → PR #1137 **预期接近 neutral**（这本身是有效结论）。
- 1024：过渡区。

> `seq_lens` 是 `[1, max_seq]` 的随机值、均值 ≈ max_seq/2；seed 固定下两侧逐位相同。

---

## 2. 实验矩阵

每个 `max_seq`：在**同一张偶数 die** 上做 **interleaved A/B**，B/A 严格交替 × **5 轮**：

```
seq=128:  B A B A B A B A B A   (B=before.so, A=after.so)
seq=1024: B A B A B A B A B A
seq=4096: B A B A B A B A B A
```

- 每条 = 1 次 task-submit（内部已跑 100 轮取 median，单次调用自带稳健统计）。
- 共 3 seq × 2 side × 5 round = **30 次 task-submit**。
- 每张表项报告：5 个 median 的 **median + min + IQR**。
- **铁律：before 与 after 必须同 die**。"跑前找最空闲卡"只在每个 A/B 对的**开头选一次**，选定后这对 B/A 全程用同一张卡；不要让 idle-scan 在 B/A 之间把你换到别的 die。

> 为什么必须 interleaved：共享设备 ~25% 噪声，曾因非交错对比把 gather_kv 的真实 win 误判成"失败"。same-session + interleaved 是底线。

---

## 3. 第 0 步：钉死 runtime 实际加载的 .so（必做，否则全废）

runtime C++ 静态编进 `pypto_core.so`，而当前环境 editable-install 与 site-packages 残留**并存**，加载点有歧义。先确认：

```bash
conda activate wq3
"$CONDA_PREFIX/bin/python" - <<'PY'
import pypto, pypto.pypto_core as c, os
print("pypto pkg   :", os.path.dirname(pypto.__file__))
print("pypto_core  :", getattr(c, "__file__", "?"))
# simpler（runtime）从哪加载：
import simpler
print("simpler pkg :", os.path.dirname(simpler.__file__))
PY
```

**判定**：`pypto_core.__file__` 指向的那颗 .so，就是 swap 的目标。若报 site-packages，swap 那颗；若报源码树 `python/pypto/`（symlink → `build/...`），swap build 产物。把这条输出的绝对路径记为 **`P`**（例如 `export P=/path/from-output/pypto_core.so`），后续所有 swap / cp 都写到 `P`。

---

## 4. 构建 before / after 两颗 .so

编包在 `$PYPTO_ROOT` 下，流程 = 该目录 `run.sh`：**先 `cd runtime && pip install -e .`（编 simpler），再回根 `pip install -e .`（编 pypto，把新 runtime 链进 `pypto_core.so`）**。两次都是 editable，带 tsinghua 镜像 + proxy。目标：每侧各编一次，存成文件，后续靠 swap **秒切**（避免每轮 rebuild 329MB .so）。

> ⚠️ **不要执行 `git submodule update --init --recursive`**（`pypto/run.sh` 第 5 行有这条，编包时手动跳过）：
> - before 侧 runtime 已在 `a756969c` = pypto 的 pin，跑它无害但多余；
> - **after 侧** runtime 已 checkout 到 `pr-1137`(`385ad7d8`)，跑 `submodule update` 会把 runtime **重置回 pin `a756969c`，悄悄覆盖 after checkout** → 编出来的是 before，整组 after 静默作废。
>
> ⚠️ **editable rebuild 不会自动覆盖 site-packages 残留的 .so**：新编的 .so 落在 `build/python/bindings/`，经源码树 `python/pypto/pypto_core.so` 软链暴露；但 §3 若查到加载点是 site-packages 物理残留，rebuild 不会动它。所以每侧编完后**必须把新 .so 拷到 `P`**，再 §5 验证。

```bash
cd "$PYPTO_ROOT"
conda activate wq3
# 如环境需要代理，请在 shell 中预先设置 http_proxy / https_proxy。

PYBIN="$CONDA_PREFIX/bin/pip"
IDX=-i\ https://pypi.tuna.tsinghua.edu.cn/simple
SRC_SO="$PYPTO_ROOT/python/pypto/pypto_core.cpython-310-aarch64-linux-gnu.so"

# --- BEFORE ---
git -C runtime stash                 # 收起未提交的 profiling_config.h (+ .orig)
git -C runtime checkout a756969c
( cd runtime && $PYBIN install -e . $IDX )     # 编 simpler（build_runtimes.py 在此触发）
$PYBIN install -e . $IDX                       # 编 pypto → pypto_core.so
cp "$SRC_SO" /tmp/pypto_core.before.so
cp /tmp/pypto_core.before.so "$P"              # 若 P≠源码树那颗，覆盖 site-packages 残留

# --- AFTER ---
git -C runtime checkout pr-1137      # !! 之后绝不再跑 git submodule update
( cd runtime && $PYBIN install -e . $IDX )
$PYBIN install -e . $IDX
cp "$SRC_SO" /tmp/pypto_core.after.so
cp /tmp/pypto_core.after.so "$P"
```

> 顺序铁律：**切 runtime commit → 编 simpler → 编 pypto**，缺一不可；编 simpler 前的 commit 才是真正生效的那个。

---

## 5. 验证 runtime 确实切换（每个变体都要过）

只看 git checkout 不够，必须确认加载进进程的 .so 真的是新编的：

```bash
# 5a. 文件指纹：swap 后，加载点的 .so 与存档一致
sha256sum "$P" /tmp/pypto_core.before.so   # before 轮：必须一致
sha256sum "$P" /tmp/pypto_core.after.so    # after 轮：必须一致

# 5b. 正确性门：golden 必须过（pass-rate≥0.98）。after 侧若 FAIL = 精度回归，立即停。
#     （日志里找 "pass-rate" / golden 校验行）

# 5c. runtime commit 落地确认
git -C "$PYPTO_ROOT/runtime" rev-parse --short HEAD   # before→a756969c / after→385ad7d8
```

**任一项不符 → 该侧作废，重建后再进。** golden FAIL 尤其要警惕（参考 fcc33bcb 那次 arena 回归量级）。

---

## 6. 运行流程（interleaved A/B）

环境前置（每次 task-submit 都要）：

```bash
conda activate wq3
export PATH="$PTOAS_ROOT:$PATH"
export TZ=Asia/Shanghai
```

> `PTO2_RING_*`：放在 `--run` 的命令串里（和 run.sh 一致），**两侧、各 seq 都用同一组值**。4096 用 131072/131072/4G（任务图大）；128/1024 可同用一组，保证可比。

选定一张最空闲的偶数 die（例如 4），B/A 全程用它。逐 seq 跑：

```bash
DEV=4                       # ← 跑前扫一次选定的偶数 die，本对全程固定
PY="$CONDA_PREFIX/bin/python"
SO="$P"
RUN=models/qwen3/14b/qwen3_14b_decode_tq_draft.py

for SEQ in 128 1024 4096; do
  for R in 1 2 3 4 5; do
    for SIDE in before after; do
      # swap .so
      cp /tmp/pypto_core.${SIDE}.so $SO
      # 5a 自检
      sha256sum $SO /tmp/pypto_core.${SIDE}.so | awk '!seen[$1]++{c++} END{exit (c!=1)}' \
        || { echo "SWAP MISMATCH $SIDE"; exit 1; }

      task-submit --device $DEV --ptoas 0.48 --max-time 1800 --run " \
        PATH=$PTOAS_ROOT:\$PATH \
        PTOAS_ROOT=$PTOAS_ROOT \
        PTO_ISA_ROOT=$PTO_ISA_ROOT \
        PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 PTO2_RING_HEAP=4294967296 \
        $PY $RUN -p a2a3 --num-layers 40 --max-seq $SEQ --enable-l2-swimlane \
        > qwen3_40L_seq${SEQ}_${SIDE}_r${R}.log 2>&1" 2>&1 | tail -3
    done
  done
done
```

> 上面的嵌套循环是"每 seq 内 B/A 交替 ×5"。把 SEQ 拆成三段分别起也行，但**不要**把所有 before 堆一起、所有 after 堆一起（会丢 same-session 交错性）。

---

## 7. 指标采集与读取

每个 `.log` 抓这些行：

```bash
grep -nE "Total Test Time|effective_us|device_wall|Exec/Latency|median|pass-rate|Swimlane JSON" \
  qwen3_40L_seq${SEQ}_${SIDE}_r${R}.log
```

| 指标 | 含义 | 用途 |
|---|---|---|
| **`Total Test Time: XXX us`** | 单次 40 层 decode 的 wall（最早 dispatch→最晚 finish） | **主指标** |
| **`effective_us ... median=XXX`** | 100 轮 benchmark 的 median | **主指标**（统计稳健） |
| `Exec/Latency = XX%` | 计算/调度-带宽占比 | 判 regime；低 = 调度空泡大 = PR #1137 有空间 |
| `pass-rate` | golden 正确率 | **硬门**，<0.98 即回归 |
| `Swimlane JSON → merged_swimlane_*.json` | per-pipe Perfetto | 看 dispatch/tail 空泡（PR #1137 该咬的地方） |

**注意**：`[RUN]` 总时间含 ~30s git fetch，**不能当 perf**。只看 `Total Test Time` / `effective_us`。

> 备用重建：亦可从 `l2_perf_records.json` 算 `Total = max(finish) − min(dispatch)`（same-session 基线重建法）。

---

## 8. 分析与判定

### 8.1 汇总表（每 seq 一行）
| max_seq | before median (us) | after median (us) | Δ% | Exec/Latency% (B→A) | golden |
|---|---|---|---|---|---|
| 128 |  |  |  |  |  |
| 1024 |  |  |  |  |  |
| 4096 |  |  |  |  |  |

`Δ% = (after − before) / before × 100%`，负 = 提速。

### 8.2 判读规则
- **短 seq（128）显著为负**（如 ≤ −10%）且 Exec/Latency% 明显上升 → 印证 ~17% 主张，PR #1137 在调度 bound 区有效。
- **长 seq（4096）接近 0** → 符合预期：满上下文 decode 带宽 bound，调度优化无处下口。**不是"没用"，是"在该 regime 下不该有用"**。
- **所有 seq 都 ≈ 0** → 在 Qwen3 decode 这个负载上 PR #1137 无收益（即便作者在别的负载上有 17%）。
- **任意 seq golden FAIL** → 立即停，按精度回归处理（PR #1137 是 4312+/10826− 大改，回归风险与 fcc33bcb 同级）。

### 8.3 采纳决策（建议阈值）
- 128 处 Δ ≤ −10% 且无回归 → 倾向采纳（调度收益真实）。
- 仅长 seq 有微小波动、短 seq 无感 → 不采纳（对本负载无价值）。
- 收益 < 单卡噪声（~3-5%）且无 regime 解释 → 视为 neutral，不采纳。

---

## 9. 风险与回滚

| 风险 | 处置 |
|---|---|
| **runtime 回归**（arena/OOM/精度，类 fcc33bcb） | golden 门拦截；FAIL 即停；回滚 `git -C runtime checkout a756969c` + rebuild |
| **swap 错 .so**（加载点歧义） | §3 第 0 步先钉死路径；每轮 §5a sha256 自检 |
| **editable / site-packages 残留冲突** | 若 swap 后加载的仍是旧 .so，临时改 `_editable_skbc_*.pth` 或直接覆写 site-packages 那颗，再重验 §5 |
| **die 漂移** | 每个 A/B 对开头选一次偶数 die，B/A 全程固定 |
| **session 漂移** | 严格 interleaved；同一 seq 的 B/A 必须同 session 连续跑完 |

**回滚到干净态**（恢复 before 基线）：
```bash
cd "$PYPTO_ROOT"
conda activate wq3
# 如环境需要代理，请在 shell 中预先设置 http_proxy / https_proxy。
git -C runtime checkout a756969c          # 回到 PR base
git -C runtime stash pop                   # 恢复 profiling_config.h（若需要）
( cd runtime && "$CONDA_PREFIX/bin/pip" install -e . \
      -i https://pypi.tuna.tsinghua.edu.cn/simple )
"$CONDA_PREFIX/bin/pip" install -e . \
      -i https://pypi.tuna.tsinghua.edu.cn/simple
# 用 /tmp/pypto_core.before.so 覆盖 P，§5 复验
```

---

## 10. 时间与资源估算

| 项 | 估算 |
|---|---|
| 两颗 .so 构建 | ~20-40 min（runtime rebuild × 2 + pypto_core.so × 2） |
| 30 次 task-submit | ~1-1.5 h（40L decode，每次内部 100 轮） |
| 单 seq 编译缓存 | `--max-seq` 走动态维，预计一套编译覆盖三 seq；首跑触发编译 |
| 合计 | ~2 h 量级（不含排错） |

---

## 附录 A：命令速查

```bash
# 选最空闲偶数 die
task-submit --devices list              # 看白名单/占用
# 选定 DEV=<偶数>

# 单发 before（手工验证用）。P = §3 查到的实际加载路径
cp /tmp/pypto_core.before.so "$P"
task-submit --device $DEV --ptoas 0.48 --max-time 1800 --run " \
  PATH=$PTOAS_ROOT:\$PATH PTOAS_ROOT=$PTOAS_ROOT \
  PTO_ISA_ROOT=$PTO_ISA_ROOT \
  PTO2_RING_TASK_WINDOW=131072 PTO2_RING_DEP_POOL=131072 PTO2_RING_HEAP=4294967296 \
  $CONDA_PREFIX/bin/python \
  models/qwen3/14b/qwen3_14b_decode_tq_draft.py \
  -p a2a3 --num-layers 40 --max-seq 4096 --enable-l2-swimlane \
  > qwen3_40L_seq4096_before_r1.log 2>&1" 2>&1 | tail -3
```

## 附录 B：commit 速查

```
before : runtime a756969c  (pypto 7759b264 锁于此 = PR #1137 base)
after  : runtime 385ad7d8  (本地 ref pr-1137, 分支 polling-pr-minimal, PR #1137 head, OPEN)
PR     : hw-native-sys/simpler #1137  (~17% median device speedup, 作者 SergioMartin86)
```

---

## 附录 C：实测结果与结论（2026-07-17 ~ 07-19 跑完）

### C.1 计划执行中的关键修正（覆写 §3/§4 的早期假设）
1. **swap 目标不是 `pypto_core.so`**：那是 pypto **编译器**，before/after 逐位相同。PR #1137 的 C++ 在 simpler 的 `runtime/build/lib/<plat>/<mode>/libhost_runtime.so`（+ libaicpu_kernel.so 等），simpler 用 `ctypes` dlopen 它们。真正 swap = **in-place 覆盖 `runtime/build/lib/`**（存档 `/tmp/ab_libs/{before,after}/lib`）。校验：before/after `libhost_runtime.so` sha256 不同（`261ceb03` vs `046f6356`）。
2. **指标不是 "Total Test Time"/"median"**（那是 DeepSeek 脚本格式）：本 draft 脚本的 perf 在 STRACE span `simpler_run.runner_run.device_wall{,.graph_build,.orch} ts=N dur=D clk=dev`。Δ% 用同单位 clk，单位自洽。
3. **实际 A/B 顺序采用分组**：每 seq 走 A(after)×N → B(before)×N，非交错。
4. **task-submit 在 die 占用时异步排队**（立即返回、稍后执行）：runner 不能用"空文件即失败"判断，改为**轮询 `$OUT` 等 `[RUN] PASS/FAIL`**。

### C.2 权威结果：seq=128 clean（**无 `--enable-l2-swimlane`**，单次运行，N=5，die 0）
| 指标 | before (wiring) | after (polling) | Δ |
|---|---|---|---|
| **orch**（编排/就绪轮询） | 77.09M clk | 64.85M clk | **−15.9%** |
| **device_wall**（端到端） | 125.58M | 120.86M | **−3.8%** |
| graph_build（建图/dep-gen） | 125.26M | 120.60M | −3.7% |
| `[RUN] runtime done`（host 聚合，噪声大） | 8.51s | 8.34s | −2.0% |

before/after 分布完全不重叠（device_wall 124–126M vs 120–121M；orch 76–80M vs 64–65M），spread 0.8–4.5%。**信号坐实。** golden 全 0.98 PASS，无精度回归。

### C.3 泳道模式（**被抬高的上限**，仅看趋势）
`--enable-l2-swimlane` 开时测得（N=5/3/3，die 0/2）：
| seq | orch Δ | device_wall Δ | cold（首次执行）Δ |
|---|---|---|---|
| 128 | −21.5% | −6.3% | −85% |
| 1024 | −29.6% | −5.3% | −93% |
| 4096 | −30.0% | −17.7% | −94% |

### C.4 关键教训：泳道 instrumentation 混淆（为什么 C.3 是上限）
`--enable-l2-swimlane` 让 measured run 收集 per-task timing：**before 的 runtime 写 AICore perf 记录（有开销、生成泳道 JSON），after（pr-1137）不写**（日志 `export_swimlane_json: No performance data to export`——**PR #1137 弄坏了 AICore perf emission**，这是它附带的 profiling 回归）。于是 before 被额外罚 ~3%（seq=128：device_wall 125.6M clean vs 129.0M swim），after 不变（120.86 vs 120.83）→ **win 被放大约 30%（相对）**。且生产 decode 永不开 `--enable-l2-swimlane`。

> **教训：测一个会改变 profiling 行为的 runtime，profiling 必须关掉（OFF）。** 否则被测对象和测量工具耦合，before/after instrumentation 不对称。clean 的 orch −15.9% 才 ≈ 作者 ~17% 主张；泳道 −21.5% 是 inflated。

### C.5 结论与建议
- **PR #1137 收益真实**：clean 测得编排层 **−15.9%**（≈ ~17% claim），端到端 device_wall **−3.8%**（seq=128，被 graph_build 稀释）。趋势上（泳道 upper bound）长上下文编排-bound、win 不衰减、首 token 延迟 −85~−94%。
- **端到端 wall 温和**：device_wall 大头是 graph_build（建图/dep-gen），orch 虽 −16% 却是 graph_build 区间内子段。端到端要更大赢面需优化 graph_build 或长上下文（4096 时 orch 占满 device_wall，泳道测端到端 −18%，但为上限）。
- **附带回归**：PR #1137 弄坏了 AICore perf emission（泳道图出不来）——若采纳，需让作者修复该 profiling 路径。
- **建议：采纳**（以 clean 测量为准），并附注 profiling 回归需修。
