# 编程与调度基础

[返回学习资料索引](../README.md)

从 RMSNorm、Worker/SPMD 到数据依赖和事件同步。

| 资料 | 阅读重点 |
| --- | --- |
| [9月1日 RMSNorm 学习讨论记录](2026%E5%B9%B49%E6%9C%881%E6%97%A5-RMSNorm%E5%AD%A6%E4%B9%A0%E8%AE%A8%E8%AE%BA%E8%AE%B0%E5%BD%95.md) | RMSNorm 的动态维度、tile、流水与向量指令。 |
| [PyPTO Worker 与 SPMD 工作分配说明](2026%E5%B9%B49%E6%9C%883%E6%97%A5-PyPTO-Worker%E4%B8%8ESPMD%E5%B7%A5%E4%BD%9C%E5%88%86%E9%85%8D%E8%AF%B4%E6%98%8E.md) | 区分逻辑工作、block、worker 和物理核。 |
| [PyPTO 数据依赖与事件同步](PyPTO%E6%95%B0%E6%8D%AE%E4%BE%9D%E8%B5%96%E4%B8%8E%E4%BA%8B%E4%BB%B6%E5%90%8C%E6%AD%A5.md) | TensorMap、TaskId、汇合/扇出与 SIMD/SPMD/MPMD。 |
| [Predicated Dispatch：让运行时条件不再卡住编排线程](predicated_dispatch.html) | 运行时条件与 predicated dispatch 的交互讲解。 |

原文版本、性能数字和未完成计划均按各篇记录的日期阅读。HTML 文件可下载后在浏览器中打开。
