# DeepSeek V4-Flash DSpark 多卡 Attention 基础

> 归档日期：2026-09-15。来源：`pypto-lib/local_archive/20260817-main-cleanup/docs/models/deepseek_v4_flash_dspark.md`。
> 原文的实现、测量与计划保留其历史语境；阅读说明见[学习资料索引](../README.md)。

本文记录 [PR #931](https://github.com/hw-native-sys/pypto-lib/pull/931) 的设计及
合入时状态。该 PR 以提交 `a92a983` 合入，是
[issue #905](https://github.com/hw-native-sys/pypto-lib/issues/905) 的一部分。

PR #931 定义了四卡数据布局、通信原语、分片的 Attention 输出投影，以及三种
稀疏 Attention 变体中的接入边界。它为多卡 decode 路径提供基础，但尚未实现
完整的端到端多卡 decode 路径。

源码统一区分 TP 组、token 归属和权重分片，不再用多套并行缩写表示
同一组四张卡。`DSA-CP` 仅作为官方 Attention 布局名称保留。本文也统一
用 token 归属、四卡通信和权重分片等直观说法来描述。

> 源码命名对照：原文的 `decode_dsa_cp_collectives.py`、`decode_sharded_o_projection.py`
> 对应 PR #931 合入版本中的 `decode_attention_cp.py`、`decode_o_projection_cp.py`；
> 本归档的相关链接指向合入版本，原文命名保留作为设计记录。

## 概要

目标布局由四张卡协作处理同一组 512 个 decode token 行，每张卡最初拥有 128
行。每张卡对自己的本地行计算全部 64 个 Attention head，随后重新分配 head
结果，使每张卡拥有八个输出组中的两个，并覆盖全部 512 行。接下来，每张卡用
自己对应的输出投影权重分片进行计算，得到一个 FP32 部分结果；最后四卡求和，
并向每个 token 所有者返回 128 行 BF16 结果。

PR #931 完成了可复用的通信和投影基础模块，并在所需的 tensor 边界拆分了稀疏
Attention kernel。原有 wrapper 及其数值行为保持不变。该 PR 合入时，还没有
任何实际 decode 入口将这些新模块组合成完整的一层。

## 部署与布局约定

配置定义在
[`config.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/config.py)，由配置推导出的
DSA-CP Attention 布局定义在
[`decode_dsa_cp_collectives.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/decode_attention_cp.py)。

| 属性 | 数值 | 含义 |
| --- | ---: | --- |
| 协作卡数（`TP`） | 4 | 四张卡共同承担 Attention 输出计算 |
| token 所有者数量（`TOKEN_OWNER_COUNT`） | 4 | 同一个 TP 组内每张卡对应一个所有者 |
| 每个四卡组的请求数 | 64 | `DECODE_BATCH` |
| 每张卡的本地行所代表的请求数 | 16 | `64 / 4` |
| 每个请求验证的 token 数 | 8 | 一个目标 token 加七个草稿 token |
| 四卡组 token 行数 | 512 | `64 * 8` |
| 本卡 token 行数 | 128 | `512 / 4` |
| hidden 宽度 | 4,096 | 模型 hidden state 宽度 |
| Attention head 数 | 64 | 每张卡为本地行计算全部 head |
| head 宽度 | 512 | 每个 Attention head 的 value 宽度 |
| 输出组数 | 8 | 每个输出组包含八个 head |
| 每张卡的输出组数 | 2 | `8 / 4` |
| 展平后的单组宽度 | 4,096 | `8 heads * 512` |
| 每个输出组的 O-A 宽度 | 1,024 | `o_lora_rank` |
| 每张卡的 O-B 输入宽度 | 2,048 | 两个本地 O-A 组 |

配置只显式声明已经在该 PR 中实现的 token 归属和输出投影权重分片：

- `TOKEN_OWNER_COUNT` 声明同一个 TP 组内的 token 所有者数量。
- O-A 按输出组切分，每张卡拥有两个组。
- O-B 沿输入维度切分，每张卡拥有 2,048 列。

`config.py` 中的 `O_A_WEIGHT_SHARDS` 和 `O_B_WEIGHT_SHARDS` 分别对应上述
O-A/O-B 分片。Q-B、Attention sink、shared expert 和词表的行为不由这份
配置声明，应以它们各自的实际实现为准。仅修改配置常量，也不会让现有高层
decode 路径自动加载分片权重或调用新的通信函数。

## 预期数据流

```text
本卡 hidden 行 [128, 4096]
    -> 跨四卡收集
四卡组 hidden 行 [512, 4096]
    -> 各卡完整持有的 KV 相关投影，以及面向本地行的 Attention

本卡 Attention head 结果
    逻辑布局 [8 个输出组, 128 行, 8 个 head, 512]
    展平布局 [8 个输出组, 128 行, 4096]
    -> 跨四卡重新分配行和输出组
本卡投影输入 [2 个输出组, 512 行, 4096]
    -> 两个 O-A 投影
    -> 按输出组、按 token 行分别进行 INT8 激活量化
    -> 一个宽度为 2048 的 O-B 输入分片
本卡部分结果 [512, 4096] FP32
    -> 四卡求和，并将行返回给对应所有者
本卡 Attention 输出 [128, 4096] BF16
```

在完整 decode 层中，hidden 行收集与 Attention 输出路径在逻辑上前后关联；但
PR #931 的通信测试程序为二者提供了相互独立的合成输入。该测试程序验证的是
每一个边界，并不能证明上图所示完整链路已经打通。

## 输出投影的作用

Attention 会为每个 head 产生一个结果。这些 head 结果不能直接送入残差路径，
而要先重新融合成模型宽度为 4,096 的 hidden state。最后这次线性变换就是
Attention 输出投影，通常记作 `W_O`。

DSpark 实现将其分解为两个阶段：

```text
Attention head 结果
    -> O-A：将每个宽度为 4096 的输出组变为 1024 个值
    -> 针对每个输出组、每个 token 行独立进行 A8 量化
    -> O-B：将组合后的 O-A 值变为宽度为 4096 的 hidden state 部分结果
```

在四卡权重分片下，每张卡只计算两个 O-A 组及对应的 2,048 宽 O-B 输入分片。
四张卡的 O-B 结果都是部分和，因此在四卡归约完成前一直保持 FP32；只有归约
完成后、属于本卡的最终结果才转换为 BF16。

每个激活 scale 只覆盖某一个输出组中的一行、宽度为 1,024 的 O-A 结果。计算
scale 前不会先把本卡的两个输出组拼接起来并共用同一个 scale。

## 按文件说明实现

### 四卡通信

[`decode_dsa_cp_collectives.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/decode_attention_cp.py)
包含共用布局常量、信号同步协议、三个通信步骤，以及一个四卡精确 golden 对账
测试程序。

`kv_token_allgather_step` 收集每张卡的有效 hidden 行：

```text
四个输入 [local_t, 4096] -> 四个完整副本输出 [4 * local_t, 4096]
```

卡 `rank` 从 `rank * local_t` 位置开始写入，因此即使 `local_t` 小于 128 的
静态容量，运行时有效行仍然紧凑排列。输入和输出 buffer 的静态容量始终分别为
`[128, 4096]` 和 `[512, 4096]`，只有前 `local_t` 行和前 `4 * local_t` 行
有效。

`attention_token_head_all_to_all_step` 将数据归属从 token 转换为输出组：

```text
转换前：每张卡拥有本卡 local_t 行对应的全部 8 个输出组
转换后：每张卡拥有覆盖全部 4 * local_t 行的 2 个输出组
```

目标卡 0 接收输出组 0-1，卡 1 接收输出组 2-3，卡 2 接收输出组 4-5，卡 3
接收输出组 6-7。在每个接收到的输出组内部，各行按来源卡顺序排列。

`o_projection_reduce_scatter_step` 从每张卡的 FP32 部分结果中，发送属于各 token
所有者的行；目标卡将四份贡献相加，把归约结果转换为 BF16，并将该所有者的
`local_t` 个有效行写入静态 `[128, 4096]` 输出 buffer 的前缀。

三个步骤都使用两阶段信号同步协议：

1. 每张卡发布自己的行、通知其他卡，并等待所有写入完成。
2. 每张卡复制或归约自己的本地通信窗口，再次通知其他卡并等待。
3. 在通信窗口可被复用前，将信号单元重置为零。

测试程序使用一个连续的四卡组，并固定 `group_base = 0`。如果一个进程 world
中包含多个四卡组，实际 host wrapper 必须计算当前四卡组的起始位置，以及本卡
在组内的 rank。

### 分片输出投影计算

[`decode_sharded_o_projection.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/decode_o_projection_cp.py)
包含可复用的 `inline` 计算函数 `decode_sharded_o_projection`，以及一个单卡 golden
对账测试程序。

其主要 tensor 如下：

| Tensor | 形状与数据类型 | 作用 |
| --- | --- | --- |
| `attention_local_groups` | `[2, 512, 4096]` BF16 | 覆盖四卡组全部行的两个输出组 |
| `wo_a` | `[2, 1024, 4096]` BF16 | 两组本地 O-A 权重 |
| `wo_b` | `[4096, 2048]` INT8 | 本地 O-B 输入分片 |
| `wo_b_scale` | `[4096]` FP32 | 每输出通道的权重 scale |
| `o_partial` | `[512, 4096]` FP32 | 四卡求和前，本卡贡献的部分结果 |

运行时只有前 `4 * local_t` 行有效，静态容量的尾部不属于数学结果。测试会用
NaN 污染接收输入的 padding，以发现意外读取；`o_partial` 输出尾部初始化为零，
并检查是否发生意外写入。

### 稀疏 Attention 接入边界

PR #931 重构了三个稀疏 Attention 文件：

- [`decode_sparse_attn_swa.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/decode_sparse_attn_swa.py)
- [`decode_sparse_attn_hca.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/decode_sparse_attn_hca.py)
- [`decode_sparse_attn_csa.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/decode_sparse_attn_csa.py)

原先的每个单体实现现在都分成三层：

```text
*_heads
    写入由调用方提供的 packed head 结果，并返回覆盖这些写入的完成 task

*_local_o_proj
    保留原先完整的本地 O-A/O-B 实现

原有公开 wrapper
    依次调用 *_heads 和 *_local_o_proj
```

该边界位于 Attention 合并、sink 归一化及 inverse RoPE 之后，O-A 之前。未来
的多卡路径会在这里插入 Attention 输出重新分配。原有 wrapper 保持旧 API，
并继续执行完整的本地输出投影，因此既有调用方保持不变。

SWA 将每个 head 暴露为单独一行，物理形状为
`[output_group * padded_token * heads_per_group, head_width]`。HCA 和 CSA
则把 head 拼接在列方向，形状为
`[output_group * padded_token, heads_per_group * head_width]`。二者都保持
相同的逻辑顺序：
`[output_group, padded_token, head_in_group, head_width]`。

## PR #931 已完成的工作

- 定义四卡组件布局及由此推导出的 tensor 容量。
- 实现能够感知运行时有效行数的 hidden 行收集、Attention 输出重新分配和
  FP32 结果归约。
- 实现接收侧分片 O-A/A8/O-B 计算路径。
- 保持 tensor 静态容量，同时从 `rank * local_t` 开始紧凑排列运行时有效行。
- 在通信测试程序中保证无效容量尾部不被改写。
- 在 SWA、HCA 和 CSA 中暴露共用的 packed head tensor 及完成 task 边界。
- 保留之前所有稀疏 Attention 公开 wrapper 及本地执行行为。
- 增加相互独立的精确通信 golden 对账和数值投影 golden 对账。

## PR #931 尚未完成的工作

- hidden 行收集尚未连接到实际的 WKV、compressor 或 indexer-cache 投影。
- SWA、HCA 和 CSA 的实际执行路径仍调用兼容 wrapper。这些 wrapper 将
  `*_heads` 输出直接传给 `*_local_o_proj`，没有经过新的数据重新分配、分片
  输出投影和结果归约。
- 高层 decode 文件尚未导入新的通信或投影模块。
- 尚未实现分片权重加载和 host 侧分布式启动接线。
- 尚未集成通信窗口分配、跨层信号复用，以及真实 Attention 生产者和通信
  消费者之间的依赖。
- 没有任何完整 decoder 层展示最终通信次数或测量端到端性能。
- prefill 行为有意保持不变。

因此，该 PR 建立并独立验证了各组件接口和基础模块，但没有宣称四卡 decode
已经完成了功能打通或性能闭环。

## 合入时验证记录

以下结果描述的是合入时的代码版本，不应当作持续有效的当前 CI 状态：

- pre-commit、单元测试、构建检查和 A3 真机 CI 通过。
- 四卡通信测试程序在满容量（`local_t = 128`）和带污染尾部的非满容量
  （`local_t = 127`）下通过精确 golden 对账。
- 输出投影测试程序在四卡组有效行数分别为 512 和 508 时，通过严格的有效
  前缀检查；输入 padding 被污染，输出尾部也经过检查。
- 分阶段的 SWA、HCA 和 CSA 程序完成了 A3 真机任务。
- 两个平台的模拟器都先通过了两个新增的独立测试程序，之后复现了目录中既有
  的全局精度不匹配，以及未修改的 `lm_head.py` 卡住问题。这些目录级状态被
  视为基线或运行时遗留问题，而不是本功能引入的失败；两个任务最终都在工作流
  30 分钟时限到达后被取消。
- 没有可用的 A5 真机结果。

唯一一个行内 review thread 指出了测试程序可能出现空的 `local_t = 0` 情况；
增加范围检查后问题得到修复，该 thread 也已解决。review 加固还包括：显式返回
带污染尾部的 Attention tensor，并为三个 packed head 边界补全固定 BF16 布局
标注。GitHub 记录中，执行合入的账号没有留下文字 review comment 或 approval，
其操作只有 merge/close 事件。把剩余逐行通信窗口复制改为更大 tile 的建议，
被保留为后续性能工作；它不是正确性缺口，并且 FP32 归约需要单独设计兼顾
buffer 容量的 tiling，而不能机械地修改行步长。

## 为什么两个基础模块保持分离

通信代码和投影代码即使不放在同一个 Python 文件中，也可以导入同一个编译程序。
`decode_sharded_o_projection` 是 `inline` 函数，因此源码文件分离不会引入运行时边界
或性能开销。

保持文件分离也保留了有价值的验证边界：

- 通信测试程序是使用分布式通信窗口和信号的四卡精确对账程序。
- 投影测试程序是速度更快的单卡数值测试，包含量化容差、满容量和非满容量用例。
- 两者的命令行设备参数、比较规则和 CI 资源需求不同。
- 可以单独调优 O-A/O-B tiling，而不会把计算修改混入通信协议。

因此，预期架构是保留这两个基础模块，再由 SWA、HCA 和 CSA 执行路径进行组合。
如果投影模块单向导入布局常量带来不便，可以将这些共用常量移入一个小型、不可
独立运行的布局模块；没有必要把通信、投影、测试程序和两个命令行入口合并成
一个大文件。

## 推荐阅读顺序

1. 阅读 [`config.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/config.py) 中的并行
   配置常量。
2. 阅读
   [`decode_dsa_cp_collectives.py`](https://github.com/hw-native-sys/pypto-lib/blob/a92a9830b9b311cc46017492aef5f68587814206/models/deepseek_v4_flash_dspark/decode_attention_cp.py)
   中推导出的形状和 `decode_dsa_cp_collectives_fixture`。
3. 分别追踪 `kv_token_allgather_step`、
   `attention_token_head_all_to_all_step` 和
   `o_projection_reduce_scatter_step`。
4. 从函数签名开始阅读 `decode_sharded_o_projection`，依次查看 O-A、量化、O-B 和
   FP32 反量化。
5. 阅读 SWA 的 `*_heads`、`*_local_o_proj` 和兼容 wrapper；HCA 与 CSA 的
   拆分方式相同。
6. 最后再阅读测试程序、tensor spec、golden 和命令行入口。

阅读代码时，不要把 `decode_dsa_cp_collectives_fixture` 误认为真实的端到端链路：
它的收集、重新分配和归约分别使用相互独立的测试输入。同样，测试程序中的
`kv_local` 表示用于 KV 相关投影、宽度等于 hidden state 的行，而不是已经
生成的 KV cache。

## 给相关负责人的简短说明

> PR #931 是 DeepSeek V4-Flash DSpark decode 的多卡 Attention 基础改造。它将
> Attention head 计算与后续输出融合层拆开，增加数据收集、重新分配和结果求和
> 操作，并实现单卡的输出投影分片以及独立的四卡结果合并原语。现有执行路径保持
> 兼容。该 PR 建立并验证了各组件边界；后续集成工作还需要把这些组件连接成完整
> decode 层。
