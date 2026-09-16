# 从 MTP 到 DSpark：DeepSeek 推测解码原理解析

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/20260817-main-cleanup/docs/models/deepseek_v4_flash_dspark_tp_training_script.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

> 形式：30 分钟内部培训讲稿 + 代码导读。
>
> 时间：Part 1（5 分钟）+ Part 2（8 分钟）+ Part 3（10 分钟）+ Part 4（5 分钟）+ 总结（2 分钟）。
>
> 代码链接：归档副本使用历史提交链接；实现状态按原文时间阅读。
>
> 场景边界：本文跟随 [issue #905](https://github.com/hw-native-sys/pypto-lib/issues/905)，讲的是 43 层 DeepSeek-V4-Flash、每轮最多 7 枚草稿，以及主模型验证侧 TP4。

> 配套展示文件：[从 MTP 到 DSpark：DeepSeek 推测解码原理解析](deepseek_v4_flash_mtp_to_dspark_tp4.md)。

标题中的“从 MTP 到 DSpark”表示推测解码方案的演进，不表示运行时先执行
MTP-1、再执行 DSpark。两者是不同的草稿提议方式。

## 开场先统一一个准确口径

下面两句话不能用于培训：

- “MTP 负责猜，DSpark 负责验。”
- “DSpark 的三个 Prediction Layer 每层生成一个 token。”

正确口径是：

> **MTP-1 一次产生一枚草稿；DSpark 一次产生一块草稿；下一轮主模型和接受逻辑负责验证并提交结果。**

DSpark 的三个预测层会依次处理同一个七位置候选块，层与层之间不采样 token。

---

## Part 1：为什么需要 MTP 和 DSpark（5 分钟）

### 1.1 页面展示：普通 Decode 的瓶颈

```text
时间 ───────────────────────────────────────────────►

轮次 1    主模型 Forward ───────────────► Token 1
轮次 2                                 主模型 Forward ───────────────► Token 2
轮次 3                                                                      主模型 Forward ───────────────► Token 3
```

普通自回归 Decode 每次主模型 Forward 通常只提交一个新 token。主模型很大，
因此生成多个 token 时，要反复调用这条昂贵的计算链。

### 1.2 向听众提问

> 能不能先提出多个候选 token，再让主模型用一次 Forward 一起检查？

这就是推测解码的共同思路：

```text
轻量草稿器：先提出多个候选
                 │
                 ▼
主模型：一次检查一组位置
                 │
                 ▼
接受连续正确的前缀
```

这里的目标不是让一次 Forward 变成零成本，而是当草稿命中时，用更少的主模型
Forward 提交同样数量的 token。

### 1.3 口头讲稿

“普通 Decode 的问题很直观：每得到一个 token，都要再走一次完整主模型。推测
解码增加了一个更便宜的草稿阶段，先给主模型准备多个候选。主模型下一次不是只看
一个位置，而是一起验证一组位置。如果前几个候选都正确，我们就能用一次主模型
Forward 提交多个 token。接下来先看 DeepSeek 的 MTP-1 如何提供草稿能力。”

---

## Part 2：MTP-1 到底是什么（8 分钟）

### 2.1 页面展示：从普通模型到 MTP-1

普通模型：

```text
主模型 Transformer
        │
        ▼
      LM Head
        │
        ▼
      Token+1
```

当前仓库的 MTP-1：

```text
主模型 Transformer
        ├────────────► 主模型 LM Head + Sampling ──► Token+1
        │                                                │
        │                                      token embedding
        │                                                │
        └────────────► target hidden ────────────────────┤
                                                ▼
                                      MTP Prediction Layer
                                      projection + Attention + MoE
                                                │
                                                ▼
                                              LM Head
                                                │
                                                ▼
                                         Token+2 草稿
```

这一页只讲一个 Prediction Layer，因为当前模型配置是 `MTP-1`。不要在这里画
“Layer 1 生成 Token+2、Layer 2 生成 Token+3、Layer 3 生成 Token+4”；那会和
DSpark 的三层块草稿骨干混在一起。

### 2.2 MTP 是模型能力，还是 Runtime？

最准确的说法分两层：

- **MTP Prediction Layer 是模型能力。**它有训练出来的权重，输入 token embedding
  和主模型 hidden，继续模拟 hidden state 的演化。
- **完整 MTP-1 推测解码仍然需要 Runtime。**Runtime 要调用主模型、验证上一枚草稿、
  运行下一枚草稿并提交跨轮状态。

因此可以说“Prediction Layer 不是 Runtime 临时拼出的规则”，但不要说“MTP 整体
完全不是 Runtime”。

### 2.3 为什么不能直接接多个 Linear？

一句话版本：

> **后续 token 依赖已经演化的 hidden state；只接多个互不相关的 Linear，无法充分模拟 Attention、上下文和 MoE 带来的状态变化。**

当前 MTP 预测层本身就包含 embedding、输入投影、滑窗 Attention、MoE、最终
归一化和 LM Head，而不是一个简单的词表 Linear。

### 2.4 代码跳转

按下面顺序打开代码即可，不需要逐行展开参数列表：

1. [模型配置：`num_nextn_predict_layers = 1`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_mtp/config.py#L145-L189)
2. [MTP 层入口：`mtp_decode_layer_inline`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_mtp/decode_mtp.py#L87-L167)
3. [MTP 层内部：embedding → projection → Attention → MoE → head](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_mtp/decode_mtp.py#L168-L243)
4. [MTP-1 运行闭环：主模型 → 验证 → 下一草稿 → 状态提交](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_mtp/decode_fwd_mtp.py#L203-L307)

### 2.5 口头讲稿

“MTP-1 的重点不是在 LM Head 后面多接一个 Linear，而是增加一层真正能继续演化
hidden state 的预测模型。当前代码只配置一个预测层，所以它每轮提供一枚草稿。
模型提供‘能猜’的能力；运行时仍要让主模型验证这枚草稿，并把接受结果和下一枚
草稿保存给下一轮。”

### 2.6 转场句

> MTP-1 一次猜一个。DSpark 的目标是把草稿器改成一次猜一块。

---

## Part 3：DSpark 怎样演进为块草稿器（10 分钟）

### 3.1 页面展示：DSpark 的一轮

```text
前一轮主模型
  输出 anchor token + 第 40/41/42 层 hidden
                 │
                 ▼
┌────────────────────── DSpark 草稿器 ──────────────────────┐
│ Stream A：第 40/41/42 层 hidden 拼接                       │
│           → main_proj → 分别写入 context KV 0/1/2          │
│                                                              │
│ Stream B：anchor → [anchor, noise × 6] → Token Embedding    │
│                                                              │
│ draft query ──► mtp.0 ──► mtp.1 ──► mtp.2                  │
│                   ▲          ▲          ▲                    │
│              context KV0 context KV1 context KV2             │
│ 三层都处理整个查询块；两条流在 Attention 汇合；层间不采样   │
│        ↓                                                    │
│ HC Head → RMSNorm → 共享 LM Head → 七组 base logits        │
│        ↓                                                    │
│ Markov Head 顺序加入 token 条件并采样七枚草稿              │
└──────────────────────────┬─────────────────────────────────┘
                           │ 7 枚候选
                           ▼
下一轮主模型一次验证 1 + 7 = 8 个位置
                           │
                           ▼
接受逻辑提交连续正确的前缀
```

DSpark 有两条输入流：target hidden 用来更新每个草稿层自己的上下文 KV；anchor
和 noise token 经过 embedding 形成七位置查询块。两条流在每个草稿层的 Attention
中汇合。

DSpark 使用 checkpoint 中的 `mtp.0`、`mtp.1`、`mtp.2` 三层权重作为草稿骨干。
这是模型权重和结构上的继承，不是运行时先执行一次 MTP-1。

### 3.2 页面展示：谁猜，谁验？

```text
DSpark 提议： A  B  C  D  E  F  G
              │  │  │  │  │  │  │
主模型验证：  ✓  ✓  ✓  ✗  ·  ·  ·
              │
              └────────────► 本轮接受 A B C
```

首个不匹配位置出现后，后续候选不能继续提交。DSpark 负责产生候选；真正给出
权威分布的是主模型，接受逻辑负责提交最长连续前缀。

最容易记住的表达是：

> **MTP-1 和 DSpark 都负责猜，主模型负责验；区别在于怎么猜、一次猜几个。**

### 3.3 MTP-1 与 DSpark 对比

| 比较项 | MTP-1 | DSpark |
|---|---|---|
| 一轮草稿数 | 1 | 最多 7 |
| 草稿骨干 | 1 个额外预测层 | `mtp.0 → mtp.1 → mtp.2` 三层块骨干 |
| 层间是否采样 | 一层结束后得到一枚草稿 | 三层之间不采样 |
| token 产生位置 | 预测层和 LM Head 之后 | 三层全部结束后，由 base logits + Markov Head 顺序产生 |
| 主模型验证宽度 | `[tail, draft]`，即 2 个位置 | `1 + 7 = 8` 个位置 |

### 3.4 代码跳转

本地当前已有 DSpark 叶子算子，但还没有把它们组合成完整草稿器入口：

1. [真实 Decode 形状：`B=64`、草稿数 7、验证宽度 8](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/config.py#L241-L249)
2. [三层 target hidden 投影：`dspark_proj`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/dspark_proj.py#L25-L65)
3. [每个草稿层的上下文 KV：`dspark_context_kv`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/dspark_context_kv.py#L49-L85)
4. [七位置非因果块 Attention：`dspark_attention`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/dspark_attention.py#L37-L93)
5. [rank-256 Markov bias：`markov_head`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/markov_head.py#L20-L62)
6. [待实现的完整组合契约：issue #935](https://github.com/hw-native-sys/pypto-lib/issues/935)
7. [上游三层组合参考：`DeepseekV4DSparkModel`](https://github.com/vllm-project/vllm-ascend/blob/ac19e1e647785be51d22a87f336ba03c02357e18/vllm_ascend/models/deepseek_v4_dspark.py#L98-L157)

### 3.5 口头讲稿

“DSpark 不是验证器，它仍然是草稿器。它复用了 checkpoint 中三个 MTP 命名的
预测层，但运行方式已经变成块草稿：三个层处理同一个七位置查询块，第三层之后
一次得到七组基础 logits，再用轻量 Markov Head 顺序产生七枚候选。下一轮完整
主模型才负责验证这些候选。因此，MTP-1 是单草稿路径，DSpark 是块草稿路径。”

---

## Part 4：PyPTO 中主模型 TP4 为什么这样设计（5 分钟）

### 4.1 先用四个数字锁定形状

| 数字 | 含义 |
|---:|---|
| 64 | 一个主模型 Decode 步的请求数 |
| 8 | 每个请求验证的位置数：1 个 anchor + 7 个草稿 |
| 512 | 四卡合计 token 行数：`64 × 8` |
| 128 | 每卡归属 token 行数：`512 ÷ 4` |

模型还有 64 个 Attention heads，每个 head 的输出宽度是 512。输出投影把 64 个
heads 重新融合回 4096 维 hidden。

[代码：模型维度与 `o_groups=8`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/config.py#L145-L166) ·
[代码：`TP=4`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/config.py#L270-L273)

### 4.2 页面展示：一层 Attention 输出的完整 TP4 数据流

```text
每卡持有 128 行 hidden
          │
          ├── 收集四卡 hidden ──► 每卡得到完整 512 行，用于构建完整 KV
          │
          ▼
每卡计算本卡 128 行 Query × 全部 64 heads
          │
          ▼
Attention 输出 [128 token, 64 heads, 512]
          │
          ▼
按模型的 8 个 O Groups 重排
[8 groups, 128 token, 8 heads × 512]
          │
          ▼
AllToAll：从“本卡 token”改成“本卡权重组”
每卡得到 [2 groups, 512 token, 4096]
          │
          ▼
本卡分片 O Projection
每卡只持有并计算 2 个 groups 的权重
          │
          ▼
每卡产生 [512 token, 4096] 部分和
          │
          ▼
ReduceScatter：四卡求和，并按 token 归还
          │
          ▼
每卡得到最终 [128 token, 4096] hidden
```

### 4.3 三个问题的简短回答

#### ① 为什么要 Group？

不是为了通信临时创造 Group。模型的 O Projection 本来就把 64 个 heads 分成
8 组，每组 8 个 heads；TP4 正好把 8 组权重分成每卡 2 组。Group 因而成为自然的
权重分片单位。

[代码：`HEADS_PER_GROUP`、`O_GROUP_IN`、`LOCAL_O_GROUPS`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L24-L38)

#### ② 为什么要 AllToAll？

AllToAll 之前，每卡拥有“128 个 token 的全部 8 组 Attention 结果”；但每卡只拥有
“2 组 O Projection 权重”。因此需要交换成“全部 512 个 token 的本卡 2 组结果”，
让数据归属与权重归属一致。

[代码：`attention_token_head_all_to_all_step`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L291-L325)

#### ③ 为什么要 ReduceScatter？

每卡只计算 2 个 groups，所以每张卡得到的 `[512, 4096]` 只是最终 hidden 的一部分。
ReduceScatter 同时完成两件事：把四张卡的部分和相加，并把最终 512 行重新按 token
归还给四张卡，每卡得到 128 行。

[代码：`o_projection_reduce_scatter_step`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L328-L361)

### 4.4 O Projection 是什么？

一句话解释：

> **O Projection 把 Attention 的多 head 输出融合回模型的 4096 维 hidden，供残差和后续 MoE 使用。**

当前 TP4 路径先让每卡只计算本卡 2 个 groups 对最终 hidden 的贡献，再跨卡求和。

[代码：分片 `decode_sharded_o_projection`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L508-L619)

### 4.5 不要漏掉第一种通信：完整 KV 输入收集

用户容易只记住 AllToAll 和 ReduceScatter，但当前设计每层一共有三次布局转换：

1. `128 → 512`：收集 hidden，让每卡构建完整 token 流的 KV。
2. `token 归属 → O Group 归属`：交换 Attention 输出。
3. `部分和 → token 归属`：求和并把最终 hidden 归还各卡。

[代码：`kv_token_allgather_step`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L261-L288) ·
[代码：三种通信的组合 fixture](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L364-L425)

### 4.6 当前实现边界

`decode_o_proj.py` 已包含三种通信、分片 O Projection、独立 fixture 和 Golden；但
当前高层主模型 Decode 还没有把三种 Attention 路径完整接入这条 TP4 输出路径。
因此可以演示和测量组件，不能把组件结果称为“端到端 DSpark Decode TP4 性能”。

[代码：单卡完整投影 `decode_o_proj_tp1`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L97-L220) ·
[代码：当前 SWA 兼容入口仍走本地投影](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_sparse_attn_swa.py#L435-L477)

### 4.7 口头讲稿

“DSpark 让主模型一次验证 512 行，TP4 决定这 512 行怎样在四张卡上计算。每卡先
保留自己的 128 行 Query 和全部 64 个 heads。Attention 结束后，数据按 8 个输出组
重排，再交换到持有对应权重的卡上。每卡只算 2 组权重，因此只得到部分 hidden；
最后通过求和和分发行恢复每卡 128 行。这就是 Group、AllToAll、O Projection 和
ReduceScatter 连在一起的原因。”

---

## 最后两分钟：一张图总结

```text
普通 Decode 瓶颈：每个 token 都要调用一次昂贵主模型
                            │
                            ▼
                  选择一种草稿提议路径
                  ┌─────────┴─────────┐
                  ▼                   ▼
       MTP-1：提出一枚草稿     DSpark：提出七枚草稿
                  │                   │
                  ▼                   ▼
       主模型验证两个位置       主模型验证八个位置
                                      │
                              当前 #905 在主模型内部
                              使用 TP4 做四卡协同计算
                  │                   │
                  └─────────┬─────────┘
                            ▼
                接受逻辑提交连续正确前缀
```

| 层次 | 解决什么问题？ |
|---|---|
| MTP-1 | 如何用额外预测层产生一枚较便宜的草稿？ |
| DSpark | 如何把三层预测骨干和 Markov Head 组织成七枚块草稿？ |
| 主模型与接受逻辑 | 哪些候选可以成为连续、正确的输出前缀？ |
| TP4 | 主模型的 512 行验证计算怎样高效分到四张 NPU？ |

最后一句：

> **MTP-1 提供单步草稿，DSpark 提供块草稿，主模型与接受逻辑保证结果正确，TP4 让验证计算在四卡协同执行。**

---

## 附录 A：现场代码演示顺序

如果只有 5 分钟打开代码，按下面顺序即可：

1. 打开 [`config.py`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/config.py#L241-L273)，确认 `64 × 8 = 512` 和 `TP = 4`。
2. 打开 [`decode_mtp.py`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_mtp/decode_mtp.py#L168-L243)，证明 MTP Prediction Layer 不是一个简单 Linear。
3. 打开 [`dspark_proj.py`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/dspark_proj.py#L25-L65)、[`dspark_attention.py`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/dspark_attention.py#L37-L93) 和 [`markov_head.py`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/markov_head.py#L20-L62)，说明本地已有组件但尚无完整组合入口。
4. 打开 [`decode_o_proj.py` 的形状定义](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L24-L50)，先讲每卡 128 行、8 组变 2 组。
5. 顺序跳转 [`AllToAll`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L291-L325)、[`分片 O Projection`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L508-L619) 和 [`ReduceScatter`](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/decode_o_proj.py#L328-L361)。

## 附录 B：容易讲错的五句话

| 不建议 | 建议表达 |
|---|---|
| DSpark 负责验证 | DSpark 负责提议草稿，主模型和接受逻辑负责验证、提交 |
| DSpark 三层各生成一个 token | 三层处理同一个七位置块，第三层结束后才进入 token 采样 |
| MTP 完全不是 Runtime | MTP Prediction Layer 是模型能力；完整 MTP-1 解码仍需要运行时闭环 |
| DSpark 必须使用 TP | 普通 Decode、MTP-1 和 DSpark 都可以使用 TP；当前优先服务 DSpark 部署点 |
| TP4 已端到端打通 | 三种通信和分片 O Projection 已有组件验证，高层 target Decode 尚待接入 |

## 附录 C：43 层和 61 层不要混用

本文代码和 issue #905 对应 V4-Flash：

- 主模型 43 层；DSpark 读取第 40、41、42 层 hidden。
- 64 heads，8 个 O Groups，hidden size 4096。

61 层是 V4-Pro 场景，它读取第 58、59、60 层 hidden，并且模型维度、head 数量和
O Group 数量不同。讲 V4-Pro 时可以复用“DSpark 提议、主模型验证、TP4 分工”这套
思路，但不能直接复用本文所有数值。

[代码：Flash 与 Pro preset 对比](https://github.com/hw-native-sys/pypto-lib/blob/c9ec1d9836ae647a6de36fc4a4500ed2507abb7c/models/deepseek_v4_flash_dspark/config.py#L145-L236)

## 附录 D：常见问题

### TP=1 是否等于不开 TP？

数学上是退化成不分片；但当前 `decode_o_proj.py` 的 TP4 形状由 `config.TP = 4`
静态推导，不能在运行时传 `--tp 1` 让整条路径自动切换。`decode_o_proj_tp1` 是单卡
完整投影函数，不代表 TP4 程序已经支持动态 TP=1。

### 现在能否测端到端 DSpark Decode TP4 性能？

还不能。可以测独立 DSpark 算子、三种通信和分片 O Projection；完整性能结论仍需要
DSpark 草稿组合入口、主模型 TP4 高层接线、接受与跨轮状态，以及真实权重和启动方式。

### 为什么不在这次培训里展开 Prefill？

Prefill 的 token 规模和输出投影策略不同。本文只讲 issue #905 的 Decode 主线，避免
把两套数据流混在一起。

## 参考资料

- [Issue #905：DeepSeek-V4-Flash DSpark A3 单节点部署对齐](https://github.com/hw-native-sys/pypto-lib/issues/905)
- [Issue #935：实现 DeepSeek-V4-Flash DSpark 草稿器](https://github.com/hw-native-sys/pypto-lib/issues/935)
- [PR #945：将 Decode TP 通信与 O Projection 收敛到 `decode_o_proj.py`](https://github.com/hw-native-sys/pypto-lib/pull/945)
- [PR #947：增加 DSpark 草稿器叶子算子](https://github.com/hw-native-sys/pypto-lib/pull/947)
- [vLLM Ascend：DeepSeek-V4-Flash 部署说明](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/DeepSeek-V4-Flash.html)
- [vLLM Ascend：DSpark 模型实现](https://github.com/vllm-project/vllm-ascend/blob/ac19e1e647785be51d22a87f336ba03c02357e18/vllm_ascend/models/deepseek_v4_dspark.py)
