# PyPTO 学习资料索引

本页收录 2026-09-15 从 `pypto-lib/local_archive/` **复制**整理的 32 份学习资料和 2 张配图。
原始归档保留在原位置。资料按主题分类，正文保留原有语言，便于按问题查阅。

## 分类目录

| 分类 | 数量 | 内容 |
| --- | ---: | --- |
| [编程与调度基础](programming/README.md) | 4 | 从 RMSNorm、Worker/SPMD 到数据依赖和事件同步。 |
| [性能分析与工程方法](performance/README.md) | 7 | 学习如何读生成代码、找关键路径、删除冗余依赖并组织可追溯实验。 |
| [多卡通信与 Attention](distributed/README.md) | 5 | 围绕 CP/TP 数据布局、AllGather、AllToAll 和 O 投影融合。 |
| [模型原理与讲解](architecture/README.md) | 5 | DSpark、MTP 和目标模型 TP 验证流程的历史设计资料。 |
| [精度排查](precision/README.md) | 2 | 保留 CSA 偶发问题的假设、实验设计、证据边界和排查过程。 |
| [调优案例与实验记录](case-studies/README.md) | 9 | 按实验版本阅读 HCA、MoE、W8A8 和 TP1 调优记录。 |

## 推荐阅读顺序

1. **入门**：编程与调度基础中的 RMSNorm → Worker/SPMD → 数据依赖与事件同步。
2. **定位性能问题**：关键路径分析 → HCA 生成 Kernel → 多余依赖删除指南 → 大 kernel 调优方法。
3. **理解多卡实现**：CP 输入 AllGather → SWA CP → Attention/AllToAll 发布 → 接收与 O-A 融合。
4. **理解模型**：MTP/DSpark 对照讲解 → 原理文档 → 培训讲稿与演示。
5. **开始实验**：优化任务记录模板 → 精度排查案卷 → 与目标 kernel 对应的调优案例。
6. **关注 TP1**：先读 HCA 归一化文档中的静态候选，再对照独立的 8K/128K 计划与结果；两组记录涉及不同变换和测量基线。

## 归档与引用约定

- 各篇记录的是撰写时的实现、实验和 PR 状态；日期较晚的补充按原文保留。这里没有重新运行性能，也没有重新核实全部 PR 状态。
- 迁移整理标题、目录、来源与链接，将讨论中的姓名替换为角色称谓，不改写已有性能数字；计划、已验证结果和失败尝试仍分别标明。
- 相对代码链接在可核对 Git 对象时改为对应提交的 GitHub 链接；原文未指定版本的路径使用整理时可核对的源码快照。历史行号仍以原文为线索。
- 本地日志、泳道原始数据、golden 和构建产物不随文复制。相关链接转为“本地证据”文字路径，路径相对于原项目或原实验目录；它们不是 notes 仓库的附件。
- 所需配图放在 `programming/assets/`，HTML 演示保留交互脚本；GitHub 文件页不执行 HTML，下载后用浏览器打开。
- 本次未收录周报及其重复摘要、PR 状态清单、运行脚本、锁文件、配置、补丁、Git bundle 和原始日志；这些属于运行/备份资料，继续留在原归档。

## 仓库原有参考资料

- [PyPTO 编码规范](pypto-coding-style.md)
- [编译与运行流程](compile-runtime-workflow.md)
- [性能调优](performance-tuning.md)
- [精度调优](precision-tuning.md)
- [调试方法](debugging.md)
- [泳道阅读指南](swimlane-reading-guide.zh.md)
- [算子分类笔记](pto-isa-op-taxonomy.zh.md)

## 复制来源对照

下面的来源均相对于 `pypto-lib/local_archive/`。

| 来源 | 整理后位置 |
| --- | --- |
| `9月1日RMSNorm 学习讨论记录.md` | [programming/2026年9月1日-RMSNorm学习讨论记录.md](programming/2026%E5%B9%B49%E6%9C%881%E6%97%A5-RMSNorm%E5%AD%A6%E4%B9%A0%E8%AE%A8%E8%AE%BA%E8%AE%B0%E5%BD%95.md) |
| `2026年9月3日-PyPTO-Worker与SPMD工作分配说明.md` | [programming/2026年9月3日-PyPTO-Worker与SPMD工作分配说明.md](programming/2026%E5%B9%B49%E6%9C%883%E6%97%A5-PyPTO-Worker%E4%B8%8ESPMD%E5%B7%A5%E4%BD%9C%E5%88%86%E9%85%8D%E8%AF%B4%E6%98%8E.md) |
| `问题描述.md` | [programming/PyPTO数据依赖与事件同步.md](programming/PyPTO%E6%95%B0%E6%8D%AE%E4%BE%9D%E8%B5%96%E4%B8%8E%E4%BA%8B%E4%BB%B6%E5%90%8C%E6%AD%A5.md) |
| `20260817-main-cleanup/predicated_dispatch.html` | [programming/predicated_dispatch.html](programming/predicated_dispatch.html) |
| `2026年9月2日-HCA生成Kernel与任务结构分析.md` | [performance/2026年9月2日-HCA生成Kernel与任务结构分析.md](performance/2026%E5%B9%B49%E6%9C%882%E6%97%A5-HCA%E7%94%9F%E6%88%90Kernel%E4%B8%8E%E4%BB%BB%E5%8A%A1%E7%BB%93%E6%9E%84%E5%88%86%E6%9E%90.md) |
| `关键路径分析.html` | [performance/关键路径分析.html](performance/%E5%85%B3%E9%94%AE%E8%B7%AF%E5%BE%84%E5%88%86%E6%9E%90.html) |
| `2026年9月9日-DSpark HCA多余依赖删除指南-PR1172.md` | [performance/2026年9月9日-DSpark HCA多余依赖删除指南-PR1172.md](performance/2026%E5%B9%B49%E6%9C%889%E6%97%A5-DSpark%20HCA%E5%A4%9A%E4%BD%99%E4%BE%9D%E8%B5%96%E5%88%A0%E9%99%A4%E6%8C%87%E5%8D%97-PR1172.md) |
| `large-kernel-tuning-guide.md` | [performance/large-kernel-tuning-guide.md](performance/large-kernel-tuning-guide.md) |
| `deepseek-v4-decode-optimization-zh.md` | [performance/deepseek-v4-decode-optimization-zh.md](performance/deepseek-v4-decode-optimization-zh.md) |
| `工作流程.md` | [performance/优化任务记录模板.md](performance/%E4%BC%98%E5%8C%96%E4%BB%BB%E5%8A%A1%E8%AE%B0%E5%BD%95%E6%A8%A1%E6%9D%BF.md) |
| `20260817-main-cleanup/PERF_TRACKING.md` | [performance/性能看护原理.md](performance/%E6%80%A7%E8%83%BD%E7%9C%8B%E6%8A%A4%E5%8E%9F%E7%90%86.md) |
| `2026年8月28日-Decode CP输入AllGather与复制KV实现说明.md` | [distributed/2026年8月28日-Decode CP输入AllGather与复制KV实现说明.md](distributed/2026%E5%B9%B48%E6%9C%8828%E6%97%A5-Decode%20CP%E8%BE%93%E5%85%A5AllGather%E4%B8%8E%E5%A4%8D%E5%88%B6KV%E5%AE%9E%E7%8E%B0%E8%AF%B4%E6%98%8E.md) |
| `2026年8月29日-Decode SWA CP实现讲解.md` | [distributed/2026年8月29日-Decode SWA CP实现讲解.md](distributed/2026%E5%B9%B48%E6%9C%8829%E6%97%A5-Decode%20SWA%20CP%E5%AE%9E%E7%8E%B0%E8%AE%B2%E8%A7%A3.md) |
| `2026年8月26日-Attention输出与AllToAll融合任务.md` | [distributed/2026年8月26日-Attention输出与AllToAll融合任务.md](distributed/2026%E5%B9%B48%E6%9C%8826%E6%97%A5-Attention%E8%BE%93%E5%87%BA%E4%B8%8EAllToAll%E8%9E%8D%E5%90%88%E4%BB%BB%E5%8A%A1.md) |
| `2026年8月26日-AllToAll接收与O-A融合任务.md` | [distributed/2026年8月26日-AllToAll接收与O-A融合任务.md](distributed/2026%E5%B9%B48%E6%9C%8826%E6%97%A5-AllToAll%E6%8E%A5%E6%94%B6%E4%B8%8EO-A%E8%9E%8D%E5%90%88%E4%BB%BB%E5%8A%A1.md) |
| `20260817-main-cleanup/deepseek_v4_flash_csa_tp4_fusion_plan.md` | [distributed/deepseek_v4_flash_csa_tp4_fusion_plan.md](distributed/deepseek_v4_flash_csa_tp4_fusion_plan.md) |
| `20260817-main-cleanup/docs/models/deepseek_v4_flash_dspark.md` | [architecture/deepseek_v4_flash_dspark.md](architecture/deepseek_v4_flash_dspark.md) |
| `20260817-main-cleanup/docs/models/deepseek_v4_flash_mtp_to_dspark_tp4.md` | [architecture/deepseek_v4_flash_mtp_to_dspark_tp4.md](architecture/deepseek_v4_flash_mtp_to_dspark_tp4.md) |
| `20260817-main-cleanup/docs/models/deepseek_v4_flash_dspark_tp_training_script.md` | [architecture/deepseek_v4_flash_dspark_tp_training_script.md](architecture/deepseek_v4_flash_dspark_tp_training_script.md) |
| `20260817-main-cleanup/docs/models/deepseek_v4_flash_dspark_tp_slides.html` | [architecture/deepseek_v4_flash_dspark_tp_slides.html](architecture/deepseek_v4_flash_dspark_tp_slides.html) |
| `20260817-main-cleanup/docs/models/deepseek_v4_flash_dspark_vs_mtp.html` | [architecture/deepseek_v4_flash_dspark_vs_mtp.html](architecture/deepseek_v4_flash_dspark_vs_mtp.html) |
| `20260820-csa-precision-investigation.md` | [precision/20260820-csa-precision-investigation.md](precision/20260820-csa-precision-investigation.md) |
| `2026年8月27日-MTP CSA偶发精度问题排查方案.md` | [precision/2026年8月27日-MTP CSA偶发精度问题排查方案.md](precision/2026%E5%B9%B48%E6%9C%8827%E6%97%A5-MTP%20CSA%E5%81%B6%E5%8F%91%E7%B2%BE%E5%BA%A6%E9%97%AE%E9%A2%98%E6%8E%92%E6%9F%A5%E6%96%B9%E6%A1%88.md) |
| `2026年8月26日-MoE统计驱动专家分配任务.md` | [case-studies/2026年8月26日-MoE统计驱动专家分配任务.md](case-studies/2026%E5%B9%B48%E6%9C%8826%E6%97%A5-MoE%E7%BB%9F%E8%AE%A1%E9%A9%B1%E5%8A%A8%E4%B8%93%E5%AE%B6%E5%88%86%E9%85%8D%E4%BB%BB%E5%8A%A1.md) |
| `2026年9月1日-DSpark Decode HCA性能优化记录.md` | [case-studies/2026年9月1日-DSpark Decode HCA性能优化记录.md](case-studies/2026%E5%B9%B49%E6%9C%881%E6%97%A5-DSpark%20Decode%20HCA%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E8%AE%B0%E5%BD%95.md) |
| `2026年9月7日-DSpark HCA泳道分析与优化方向.md` | [case-studies/2026年9月7日-DSpark HCA泳道分析与优化方向.md](case-studies/2026%E5%B9%B49%E6%9C%887%E6%97%A5-DSpark%20HCA%E6%B3%B3%E9%81%93%E5%88%86%E6%9E%90%E4%B8%8E%E4%BC%98%E5%8C%96%E6%96%B9%E5%90%91.md) |
| `2026年9月13日-HCA归一化优化-将除法移到矩阵广播之前-21081e9ca.md` | [case-studies/2026年9月13日-HCA归一化优化-将除法移到矩阵广播之前-21081e9ca.md](case-studies/2026%E5%B9%B49%E6%9C%8813%E6%97%A5-HCA%E5%BD%92%E4%B8%80%E5%8C%96%E4%BC%98%E5%8C%96-%E5%B0%86%E9%99%A4%E6%B3%95%E7%A7%BB%E5%88%B0%E7%9F%A9%E9%98%B5%E5%B9%BF%E6%92%AD%E4%B9%8B%E5%89%8D-21081e9ca.md) |
| `dspark-hca-decode-tuning-log.md` | [case-studies/dspark-hca-decode-tuning-log.md](case-studies/dspark-hca-decode-tuning-log.md) |
| `dspark-hca-tp1-8k-128k-tuning-plan.md` | [case-studies/dspark-hca-tp1-8k-128k-tuning-plan.md](case-studies/dspark-hca-tp1-8k-128k-tuning-plan.md) |
| `dspark-hca-tp1-8k-128k-tuning-results-20260914.md` | [case-studies/dspark-hca-tp1-8k-128k-tuning-results-20260914.md](case-studies/dspark-hca-tp1-8k-128k-tuning-results-20260914.md) |
| `20260817-main-cleanup/docs/models/deepseek_v4_flash_w8a8_performance.md` | [case-studies/deepseek_v4_flash_w8a8_performance.md](case-studies/deepseek_v4_flash_w8a8_performance.md) |
| `dspark-hca-tp1-cmp-split-results-20260915.md` | [case-studies/dspark-hca-tp1-cmp-split-results-20260915.md](case-studies/dspark-hca-tp1-cmp-split-results-20260915.md) |
