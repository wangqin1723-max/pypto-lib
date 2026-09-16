# 性能分析与工程方法

[返回学习资料索引](../README.md)

学习如何读生成代码、找关键路径、删除冗余依赖并组织可追溯实验。

| 资料 | 阅读重点 |
| --- | --- |
| [HCA 生成 Kernel 与任务结构分析](2026%E5%B9%B49%E6%9C%882%E6%97%A5-HCA%E7%94%9F%E6%88%90Kernel%E4%B8%8E%E4%BB%BB%E5%8A%A1%E7%BB%93%E6%9E%84%E5%88%86%E6%9E%90.md) | 从 deps.json 和 C++ 还原 block 到数据的映射。 |
| [关键路径分析 · Critical Path](%E5%85%B3%E9%94%AE%E8%B7%AF%E5%BE%84%E5%88%86%E6%9E%90.html) | 关键路径分析的可交互说明；下载 HTML 后用浏览器打开。 |
| [多余依赖是什么、如何影响性能、怎样删除](2026%E5%B9%B49%E6%9C%889%E6%97%A5-DSpark%20HCA%E5%A4%9A%E4%BD%99%E4%BE%9D%E8%B5%96%E5%88%A0%E9%99%A4%E6%8C%87%E5%8D%97-PR1172.md) | 冗余边、无关等待、no_dep 与依赖验证。 |
| [遇到“大 kernel”的排查与调优方法：HCA #1187 案例复盘](large-kernel-tuning-guide.md) | 大 kernel 的计算、搬运、流水和拆项实验方法。 |
| [DeepSeek V4 Decode 优化](deepseek-v4-decode-optimization-zh.md) | Decode 优化的综合方法与算子专项案例。 |
| [优化任务记录模板](%E4%BC%98%E5%8C%96%E4%BB%BB%E5%8A%A1%E8%AE%B0%E5%BD%95%E6%A8%A1%E6%9D%BF.md) | 记录问题、数据流、方案、同步和验证的可复用模板。 |
| [性能看护（perf-tracking）原理](%E6%80%A7%E8%83%BD%E7%9C%8B%E6%8A%A4%E5%8E%9F%E7%90%86.md) | 逐 PR 重建、跑分、归因与幂等归档的历史设计。 |

原文版本、性能数字和未完成计划均按各篇记录的日期阅读。HTML 文件可下载后在浏览器中打开。
