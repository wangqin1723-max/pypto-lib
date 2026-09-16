# 多卡通信与 Attention

[返回学习资料索引](../README.md)

围绕 CP/TP 数据布局、AllGather、AllToAll 和 O 投影融合。

| 资料 | 阅读重点 |
| --- | --- |
| [2026年8月28日 Decode CP 输入 AllGather 与复制 KV 实现说明](2026%E5%B9%B48%E6%9C%8828%E6%97%A5-Decode%20CP%E8%BE%93%E5%85%A5AllGather%E4%B8%8E%E5%A4%8D%E5%88%B6KV%E5%AE%9E%E7%8E%B0%E8%AF%B4%E6%98%8E.md) | 输入 AllGather、复制 KV 与 local/full-group 边界。 |
| [2026年8月29日 Decode SWA CP 实现讲解](2026%E5%B9%B48%E6%9C%8829%E6%97%A5-Decode%20SWA%20CP%E5%AE%9E%E7%8E%B0%E8%AE%B2%E8%A7%A3.md) | 按 SWA 代码追踪 CP 输入、KV 写回和本地输出。 |
| [2026年8月26日 Attention 输出与 AllToAll 融合任务](2026%E5%B9%B48%E6%9C%8826%E6%97%A5-Attention%E8%BE%93%E5%87%BA%E4%B8%8EAllToAll%E8%9E%8D%E5%90%88%E4%BB%BB%E5%8A%A1.md) | Attention 输出发布融合的数据布局与通知协议。 |
| [2026年8月26日 AllToAll 接收与 O-A 融合任务](2026%E5%B9%B48%E6%9C%8826%E6%97%A5-AllToAll%E6%8E%A5%E6%94%B6%E4%B8%8EO-A%E8%9E%8D%E5%90%88%E4%BB%BB%E5%8A%A1.md) | 从通信 window 直接计算 O-A 的方案与生命周期。 |
| [DeepSeek V4 CSA TP4 通信-计算融合方案](deepseek_v4_flash_csa_tp4_fusion_plan.md) | CSA TP4 通信计算融合的历史分阶段计划。 |

原文版本、性能数字和未完成计划均按各篇记录的日期阅读。HTML 文件可下载后在浏览器中打开。
