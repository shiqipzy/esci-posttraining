# 实验结果

固定 F 包含 512 个查询、10,459 个商品对。分类并列报告 Accuracy 与 Macro-F1，并补充各类指标和 nDCG@10。各实验按自己的训练对照与评测条件报告。

| 实验 | 主要结果 | 完整报告 |
|---|---|---|
| 教师概率监督 | TA2 相对 A2 的 Macro-F1 +0.67pp；相对 G2 的 Accuracy +0.33pp，两项区间为正 | [final/RESULT.md](final/RESULT.md) |
| 半量概率融合 | S_C50 Accuracy 57.22%、Macro-F1 52.30%；相对 S 分别 +2.24 / +4.67pp | [final/RESULT.md](final/RESULT.md) |
| 融合输出蒸馏 | 独立学生 Accuracy 55.73%、Macro-F1 49.22%；相对 CE_CONT / REF_KD 的 Macro-F1 +0.91 / +1.04pp，两项区间为正 | [fusion_continue_20261021/RESULT.md](fusion_continue_20261021/RESULT.md) |

## 机器可读产物

- 教师概率监督与融合：[analysis.json](final/analysis.json)。
- 融合蒸馏：[report.json](fusion_continue_20261021/report.json)、[选点记录](fusion_continue_20261021/selected_all.json)、[运行清单](fusion_continue_20261021/run_manifest.json)、[目标来源](fusion_continue_20261021/target_manifest.json)。

融合蒸馏实验对 S 做了统一重评，使用该轮重评结果作为对照；两轮分数分别保留。模型权重、逐题预测及教师响应缓存未随仓库发布，复算要求见 [训练与评测](../docs/训练与评测.md)。
