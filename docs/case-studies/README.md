# 调优案例与实验记录

[返回学习资料索引](../README.md)

按实验版本阅读 HCA、MoE、W8A8 和 TP1 调优记录。

| 资料 | 阅读重点 |
| --- | --- |
| [2026年8月26日 MoE 统计驱动专家分配任务](2026%E5%B9%B48%E6%9C%8826%E6%97%A5-MoE%E7%BB%9F%E8%AE%A1%E9%A9%B1%E5%8A%A8%E4%B8%93%E5%AE%B6%E5%88%86%E9%85%8D%E4%BB%BB%E5%8A%A1.md) | MoE 专家分配的数据流、同步和性能对照。 |
| [DSpark Decode HCA 性能优化记录](2026%E5%B9%B49%E6%9C%881%E6%97%A5-DSpark%20Decode%20HCA%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E8%AE%B0%E5%BD%95.md) | raw KV 复用、去转置和 merge/rope/pack 融合。 |
| [DSpark Decode HCA 泳道分析与优化方向](2026%E5%B9%B49%E6%9C%887%E6%97%A5-DSpark%20HCA%E6%B3%B3%E9%81%93%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96%E6%96%B9%E5%90%91.md) | 泳道诊断与后续依赖实验；各节按原测量版本阅读。 |
| [HCA 归一化优化：将除法移到矩阵广播之前](2026%E5%B9%B49%E6%9C%8813%E6%97%A5-HCA%E5%BD%92%E4%B8%80%E5%8C%96%E4%BC%98%E5%8C%96-%E5%B0%86%E9%99%A4%E6%B3%95%E7%A7%BB%E5%88%B0%E7%9F%A9%E9%98%B5%E5%B9%BF%E6%92%AD%E4%B9%8B%E5%89%8D-21081e9ca.md) | 广播前归一化及 TP1 静态候选计划；勿将计划当作测量结果。 |
| [DSpark HCA decode tuning log](dspark-hca-decode-tuning-log.md) | 持续调优记录，保留采用、否决和更正的证据。 |
| [DSpark HCA TP1：8K / 128K 性能调优计划](dspark-hca-tp1-8k-128k-tuning-plan.md) | TP1 8K/128K 的分阶段计划与执行补充。 |
| [DSpark HCA TP1：8K / 128K 调优结果](dspark-hca-tp1-8k-128k-tuning-results-20260914.md) | TP1 增量与组合测量；后续 PR 状态更新按原文日期阅读。 |
| [DeepSeek V4-Flash W8A8 performance](deepseek_v4_flash_w8a8_performance.md) | W8A8 各版本、工具链和复测口径的历史对照。 |
| [HCA TP1 CMP 任务切分实验（2026-09-15）](dspark-hca-tp1-cmp-split-results-20260915.md) | 缩短单 block 后仍未证明稳定整层收益的对照案例。 |

原文版本、性能数字和未完成计划均按各篇记录的日期阅读。HTML 文件可下载后在浏览器中打开。
