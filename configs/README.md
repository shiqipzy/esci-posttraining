# 配置说明

| 文件 | 适用范围 |
|---|---|
| [fusion_continue_20261021.json](fusion_continue_20261021.json) | 当前 CE_CONT / REF_KD / FUSION_KD 续训的固定配置 |
| [models.lock.json](models.lock.json) | 共用官方 Qwen3-0.6B revision 和文件身份 |
| [evaluation_policy.json](evaluation_policy.json) | 当前报告导航与指标展示规则 |
| [training.json](training.json) | 当前官方底座 → S 的 SFT 配置 |
| [posttraining_data_manifest.json](posttraining_data_manifest.json) | 历史训练数据身份快照，保留当时门控计数 |

direct、subset、optimization_candidate、fusion_scope 配置各属于此前执行或拟议方案。当前入口 `python src/train.py sft` 默认读取 `training.json`，`python src/train.py fusion` 默认读取 `fusion_continue_20261021.json`，不会自动合并历史配置。

本轮实际运行的配置也冻结在 [run_manifest.json](../results/fusion_continue_20261021/run_manifest.json) 中。报告导航可以更新，实际运行配置、目标清单和原始汇总不会随文档改写。
