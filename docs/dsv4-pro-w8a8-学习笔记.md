# DeepSeek V4 Pro W8A8 — 学习笔记

> 仓库根：`$PYPTO_LIB_ROOT`（运行前设为本机 pypto-lib checkout）；模型目录：`models/deepseek/v4-pro-w8a8/`。
> 配套交付文档:`models/deepseek/v4-pro-w8a8/issue-873-benchmark.md`。
> 所有性能数均为**偶数 die(primary)**、`PYPTO_BENCH=1`、5 warmup + 100 轮中位数。

---

## 0. 一句话全局

> **128k decode 一步 ~95ms,其中 ~62% 花在 CSA"翻长历史"(显存带宽 bound),不在算力。**
> **W8A8 已把算力侧搞定;要再快就得减少 CSA 扫描的历史页数,不是堆专家/matmul。**

---

## 1. 这个性能测试是干嘛的(作用)

它不是造功能,是**测量 → 给决策提供事实**:
1. **可部署性判断**:128k 下一步 95ms → 关系能否上服务(吞吐/延迟 SLA)。
2. **找瓶颈 = 找优化杠杆**:知道 95ms 花在哪,才知道优化谁。
3. **建立基线**:以后任何优化(改 kernel/量化/调度)拿这个数对照。

它把"模型结构"翻译成"硬件瓶颈 + 优化方向"。

---

## 2. 关键参数(配置)逐个讲

| 参数 | 是什么 | 对成本的影响 |
|---|---|---|
| **61 层** | transformer 串 61 个层,每层变换隐状态 | 层数 × 每层成本 ≈ 总成本 |
| **HCA / CSA / SWA** | 每层里"看历史 token"的模块,三种压缩粒度 | **成本分化根源**(见 §3) |
| **B=2 / S=4 / T=8** | 2 请求,每请求每步 4 行(1 主 + 3 猜),共 8 行 | 行数决定每步算力/访存 |
| **128k 上下文** | 模型记得前面 13 万 token | **8× 于 16k**,上下文越长翻历史越贵 |
| **MoE / EP=128** | 384 个专家小网络,每 token 选 top-6 个;EP=128 = 分散 128 卡,每卡 3 本地专家 | 远端专家要走通信(EP 通信是未测缺口) |
| **W8A8 (INT8)** | 权重+激活从 BF16 压到 INT8 | **带宽减半 + INT8 cube 吞吐更高**,算力/访存都受益 |
| **MTP=3** | 投机解码,每步多猜 3 个 token 并行验证 | 猜中就一步出 4 token,吞吐翻倍(见 §5) |

---

## 3. 注意力与 KV cache(核心概念)

### 3.1 什么是 attention / KV cache
生成一个 token 时,模型要**回头看之前所有 token**(这堆历史叫 **KV cache**)。
- 上下文越长,KV cache 越大,**要扫描的历史越多 → 越慢**(这就是"序列缩放")。
- 用硬件说:**扫 KV cache = 大量 GM 读 → 显存带宽 bound**。MoE 算 matmul = 算力 bound。

### 3.2 为什么压缩
128k = 13 万历史 token,每步逐个精看 → 读不动。所以**把老历史合并成"摘要"**,只看摘要。

### 3.3 三种 attention 的全称与压缩比

| 缩写 | 全称 | 压缩比 | 用在哪 | 有无 indexer |
|---|---|---|---|---|
| **HCA** | Hierarchical Compressed Attention(分层压缩) | ratio-128 | 主模型 31 层 | 无 |
| **CSA** | Compressed Sparse Attention(压缩稀疏) | ratio-4 | 主模型 30 层 | **有**(sparse topk) |
| **SWA** | Sliding Window Attention(滑窗) | ratio-0 | MTP 层(1 层) | — |

### 3.4 "压缩 128→1 / 4→1" 是什么意思
原始历史按 **128 token 一组**存成 block,13 万 token ≈ **1028 个原始块**。压缩比 = 多少原始块合并成 1 个摘要块(一个学出来的小矩阵乘做的池化):

| | 合并多少 | 1028 块压成 | 每摘要代表 |
|---|---|---|---|
| HCA ratio-128 | 128 原始块→1 摘要 | **~9 个摘要** | ~1.6 万 token |
| CSA ratio-4 | 4 原始块→1 摘要 | **~257 个摘要** | ~512 token |
| SWA ratio-0 | 不合并 | 只看最近 ~128 token 滑窗 | — |

**类比(聊天记录找信息):**
- HCA = 每 1.6 万条浓缩成 1 句极简摘要 → 只剩 9 句要读 → 飞快,细节丢得多。
- CSA = 每 512 条浓缩成 1 句 → 257 句摘要,细节全但太多,还得**检索哪几句相关** → 慢。
- SWA = 老的不看,只翻最近 128 条。

**权衡:压得越狠 → 读得越少 → 越快,但历史越模糊。** 模型混用:HCA 便宜够用(31 层);CSA 要细历史(30 层),配 indexer 从 257 摘要里挑 top-1024 再精看。

### 3.5 为什么 CSA 贵(62%)、HCA 便宜(33%)
- HCA 只 9 个摘要 → 直接全看,**无 indexer** → 便宜。
- CSA 有 257 个摘要 → 看不完 → 要 **sparse indexer** 打分挑 top-1024 → **挑选本身(`topk_score_pad` ~586µs + TopK merge 链)就是 CSA 大头**。即"决定看哪些历史"比"看"本身还贵。

---

## 4. CSA 压缩的代码实现(4→1 在哪)

**文件:**
- 压缩 kernel 本体:`models/deepseek/v4-pro-w8a8/decode_compressor_ratio4.py`(`compressor_ratio4` 函数,line 73)
- CSA 里调用它:`models/deepseek/v4-pro-w8a8/decode_attention_csa.py`(line 230,HC-pre 阶段)

**数据流:** 原始 KV → `compressor_ratio4` → `cmp_kv_cache`(257 摘要)→ 后面 sparse attention + indexer 读它做 topk。

**kernel 内部三步:**

**① 投影(两个 cube matmul,line ~100)** — per-token,还没压:
```python
kv_acc    = pl.matmul(x_tile, wkv_tile,  b_trans=True)   # x · wkv  → kv_proj(压缩 KV)
score_acc = pl.matmul(x_tile, wgate_tile, b_trans=True)  # x · wgate → score(重要性)
```

**② scatter 进滚动 state(line ~127, `scatter_softmax_pool`)** — 把 (kv_proj, score+ape) 攒进 8 槽滚动缓冲,4 个一组凑齐:
```python
compress_state_flat[..., 0:OUT_DIM]    = kv_tile
compress_state_flat[..., OUT_DIM:2*OUT] = score_tile + ape
```

**③ softmax + pool —— 真正 4→1(line ~155)** — 在线 softmax(mi/li/oi)对 4 个 score 做归一化 → 权重 → 加权合并 4 个 kv → 1 个摘要,加位置编码 + RMSNorm,写进 `cmp_kv_cache`,state 后移。

> **实质:一个学出来的、用 softmax 加权的 4 选 1 池化**(不是简单平均,按重要性 score 加权);`wkv`/`wgate` 学"怎么压最有用"。

---

## 5. MoE(专家网络)与 W8A8

- 不用一个大 FFN,而是 **384 个专家**,每 token 选 **top-6**。`gate.py` 做路由(RMSNorm+gate+topk)。
- **EP=128**:专家分散 128 卡,每卡 **3 个本地专家**;远端专家要走 all-to-all 通信(物理 EP 通信 = 未测缺口)。
- 本 benchmark 用**单 die 计算 proxy**(`moe_compute.py`,3 本地专家)。
- **W8A8** 让 MoE 很便宜:gate 52.2µs / shared 120.9µs / routed 232.3µs。模型的**算力侧已被 W8A8 解决**,不是矛盾点。

---

## 6. MTP(投机解码,Multi-Token Prediction)

### 解决什么
普通生成一次出 1 token(跑一遍 61 层 forward)。要出 4 个得跑 4 遍 → 慢。

### 怎么做(猜 + 批量验证)
- 一个**很小的"草稿"模块**先**猜**接下来的几个 token(便宜)。
- **主模型一次性验证全部**(验证 N 个 ≈ 验证 1 个的成本,并行)。
- 猜对白赚,猜错回退。

### V4 Pro 里的形态
- **1 个额外的"MTP 层"**(就是那个 ratio-0 / SWA 层)当草稿头,**不在 61 层主模型内**。
- **draft depth = 3**(S = 1 主 + 3 猜):主模型先出 1 真 token,MTP 层串行猜 3 个 draft,下一步主模型顺便验证。

### 测的数对应什么
| 数 | 对应 |
|---|---|
| SWA 565µs | MTP 层的 attention(ratio-0 滑窗,固定短窗口 → 便宜、不随上下文涨) |
| MTP 迭代 1.687ms | MTP 层跑一遍(projection+SWA+MoE+head+norm)= 猜 1 个 draft |
| D3 5.07ms | 3 次串行 MTP 迭代 = draft depth 3 |

### 为什么划算
- MTP 层 ratio-0 SWA,窗口固定 128 → **16k≈128k(序列平坦),die-parity 不敏感(算力 bound)**。
- 成本小:主 forward 95ms,D3 才 5ms(~5%)。猜中即吞吐近 4×。纯解码期吞吐优化,不改质量(错会回退)。

---

## 7. 性能结果(偶数 die)

### 7.1 主表:占一个 decode step 的份额(128k)
锚点 `depth61` = **94.9003 ms**。

| 算子(按层配比加权) | 单位成本 | 加权 | 份额 |
|---|---:|---:|---:|
| CSA 层 ×30 | 1.9732 ms | 59.20 ms | **62.4 %** |
| HCA 层 ×31 | 0.9976 ms | 30.93 ms | **32.6 %** |
| Σ 重建 | — | 90.12 ms | 95.0 % |
| R 残差 | — | 4.78 ms | 5.0 % |

加权和重建直接结果 **94.96%** → §6 重建校验通过(模型无隐藏成本)。

### 7.2 16k ↔ 128k 序列缩放
| 算子 | 16k | 128k | Δ |
|---|---:|---:|---:|
| CSA 层 | 1.7161 ms | 1.9732 ms | **+15.0 %** |
| HCA 层 | 0.9867 ms | 0.9976 ms | +1.1 % |
| depth61 forward | 85.71 ms | 94.90 ms | +10.7 % |

16k 时 CSA 已是 HCA 的 1.74× → **crossover 发生在 16k 之前**。

### 7.3 MTP / D3 / 叶子(偶数 die)
| 算子 | 16k | 128k |
|---|---:|---:|
| gate / shared / routed | — (per-token,上下文无关) | 52.2 / 120.9 / 232.3 µs |
| SWA | 565.5 µs | 564.7 µs |
| MTP 迭代 | 1.686 ms | 1.687 ms |
| D3 | 5.079 ms | 5.072 ms |

---

## 8. die-parity(为什么只用偶数 die)

- npu-smi topo:dies 成对 (0,1)(2,3)...(14,15) 经 SIO。
- **偶数 = primary(PCIe 直连,快)**;**奇数 = secondary(经 SIO,+20-25%)**。
- **只惩罚带宽-bound 算子**:CSA 奇数 die 慢 ~20%;MTP/MoE 这些算力 bound 的 <1% 变化。
- 所以**所有发布值只用偶数 die**。旧奇数 die depth61 = 112.80ms vs 偶数 94.90ms(~18% 虚高)。
- 跑测要用 `--device 0,2,4,6,8,10,12,14`(显式偶数),别用 `--device auto`(曾因白名单是奇数而污染)。

---

## 9. 测量基底(M / T / P)与残差 R

- **M** = 100 轮 Effective 中位数(权威)。`overwrite` = 固定输入/写槽、不复位状态的热重复。
- **T** = 单发 level-4 L2 插桩 trace 跨度(**不是**中位数,只作内部分拆,不能当墙钟份额)。
- **P** = 相减/代理派生。
- **R 残差 ~5%** = 重建和 vs 实测整体的差 = 调度/dispatch/层间交接开销。
- M 与 T **不交叉相减**生成发布延迟。

---

## 10. 关键文件索引(仓库相对路径)

**主 forward / 层循环:**
- `decode_fwd_compute.py` — 61 层计算 proxy(depth 阶梯 hca1/csa1/depth61),**主网格基准入口**
- `decode_layer_compute.py` — 单层(attention + MoE compute)组合
- `main_compute_manifest.py` — 静态阶梯定义

**注意力(三种):**
- `decode_attention_hca.py` — HCA(ratio-128)编排
- `decode_attention_csa.py` — CSA(ratio-4)编排,**line 230 调 compressor**
- `decode_attention_swa.py` — SWA(ratio-0,MTP 层 attention)
- `decode_sparse_attn.py` / `decode_sparse_attn_hca.py` / `decode_sparse_attn_swa.py` — sparse attention 叶子

**CSA 压缩 / indexer(成本大头在这):**
- `decode_compressor_ratio4.py` — **ratio-4 压缩 kernel**(4→1 softmax+pool,§4 详解)
- `decode_compressor_ratio128.py` — HCA 的 ratio-128 压缩
- `decode_indexer.py` / `decode_indexer_compressor.py` — CSA indexer(打分挑 top-1024,`topk_score_pad` 在这)

**MoE:**
- `moe_compute.py` — 3 本地专家计算 proxy
- `gate.py` / `expert_shared.py` / `expert_routed.py` — MoE 叶子
- `moe_communication.py` — EP2/4/8 通信合同(未实测,fail-fast)

**MTP:**
- `decode_mtp_compute.py` — 单次 MTP 迭代
- `decode_mtp_d3_compute.py` — D3(3 次串行)
- `mtp_projection.py` — MTP projection 叶子

**配置 / 其它:**
- `config.py` — Pro-W8A8 preset(B=2/S=4/131072/EP128 等)
- `hc_pre.py` / `hc_post.py` — HC pre/post(HC-pre 里含压缩)
- `lm_head.py` / `rmsnorm.py` / `lookup_embedding.py` — head/ norm/ embedding 叶子

**文档:**
- `issue-873-benchmark.md` — 性能交付文档(主)
- `README.md` / `README.zh-CN.md` — bring-up 计划与里程碑日志

---

## 11. 缺口(尚未测量)

1. **真正的 `decode_fwd.py` 端到端** — 文件仍是 43 层 Flash 种子;`moe.py` 上限 `--ep≤8`,跑不出 Pro EP=128/3-local,故用计算 proxy 替身。
2. **物理 EP2/4/8 dispatch/combine 通信** — 仅合同,设备 fail-fast;按 issue §5 单独测。
3. **精确 crossover 位置** — 已证 CSA/HCA crossover 在 16k 前;需 sub-16k 同卡 sweep 定位。

---

## 12. 速查:一句话结论

- **瓶颈**:128k decode 慢在 CSA 翻历史(显存带宽),不在算力。
- **W8A8**:已把算力侧(MoE)搞定,叶子都几十~几百 µs。
- **优化方向**:减少 CSA 扫描的历史页数(更狠压缩 / 更早 sparse 过滤 / 用计算掩盖访存),不是堆专家或 matmul。
- **crossover**:CSA 成本超 HCA 已发生在 16k 之前。
- **die-parity**:只对带宽-bound 算子(CSA)有 ~20% 影响;测必须偶数 die。
- **MTP**:主模型之上挂的便宜草稿头,~5ms 换吞吐翻倍,纯解码期优化。
