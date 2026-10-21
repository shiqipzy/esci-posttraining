# ESCI：底座到 S，再到融合蒸馏

基于 Amazon ESCI 数据集，以 Qwen3-0.6B + LoRA 输出 Exact / Substitute / Complement / Irrelevant 四类概率。训练分两段：先把官方底座 SFT 成原模型 S，再从固定的 S adapter 续训三组学生，把原模型与教师概率的半量融合收益蒸馏到独立学生。推理只用 0.6B 学生，不再调用教师。

| 阶段 | 入口 | 配置 | 产出 |
| --- | --- | --- | --- |
| 官方底座 → S | `python src/train.py sft` → [sft.py](src/sft.py) | `configs/training.json` | `runs/train_sft/models/S/` |
| S → CE_CONT / REF_KD / FUSION_KD | `python src/train.py fusion` → [fusion.py](src/fusion.py) | `configs/fusion_continue_20261021.json` | `selected_all.json`、`report.json` |

融合实验直接读取已发布的 `models/S/adapter`，不重训 S。教师信号来自固定缓存。

## 融合蒸馏：将系统收益迁移到独立学生

在已有半量融合收益的基础上，将融合系统的完整输出作为固定软目标 q。对原 S 低置信度的一半训练样本，在教师缓存有效时取 `q=(p_S+p_T)/2`；其他样本取 `q=p_S`。全部 10,398 条样本参与蒸馏，路由不依赖标签或教师正确性。

三组从同一 S adapter 续训：CE_CONT 仅用真实标签；REF_KD 使用真实标签与原 S 概率；FUSION_KD 使用真实标签与融合概率。两个软监督组采用相同的 `(CE + KL)/2` 损失、样本覆盖和训练控制。教师响应全部复用已有缓存。

### 独立学生结果

F 包含 512 个查询、10,459 个商品对。

| 对象 | Accuracy | Macro-F1 |
|---|---:|---:|
| S：本轮统一重评的初始学生 | 54.83% | 47.50% |
| CE_CONT：真实标签续训 | 55.88% | 48.31% |
| REF_KD：原模型软监督 | 55.65% | 48.18% |
| FUSION_KD：融合蒸馏学生 | **55.73%** | **49.22%** |

FUSION_KD 相对 CE_CONT / REF_KD 的 Macro-F1 分别提高 **0.91 / 1.04 个百分点**，查询配对 Bootstrap 95% 区间分别为 **[+0.11,+1.74] / [+0.40,+1.73]**。推理只使用 0.6B 学生，不再依赖教师调用。

本轮按 tune 独立选点，三组具有相同训练预算上限；所选 checkpoint 的曝光不同。S 与三组学生使用同一环境重新评测。本轮融合系统参考 Macro-F1 为 52.08%，学生迁移了部分融合收益。完整 Accuracy、每类 F1、nDCG@10、选点记录及区间见 [融合蒸馏报告](results/fusion_continue_20261021/RESULT.md) 和 [report.json](results/fusion_continue_20261021/report.json)。

融合蒸馏使用固定 F；该集合已在项目迭代中查看，Bootstrap 区间条件于单 seed 的已训练模型。

## 阅读与运行

```bash
python src/train.py --help
python src/train.py sft --help
python src/train.py sft --check-only
python src/train.py fusion --help
python src/train.py fusion --preflight
```

帮助命令只需标准库；SFT 只读检查核对 20,941 条 / 1,024 查询及 tune 的 5,223 条 / 256 查询，不训练、不创建输出。Fusion 预检还需要 torch 和本地 tokenizer，只做输入审计与合成 CPU 检查。完整训练命令见[训练与评测](docs/训练与评测.md)，解释与历史记录见[文档目录](docs/README.md)。

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| [train.py](src/train.py) | 22 | 唯一 CLI，只分发 sft / fusion |
| [sft.py](src/sft.py) | 161 | 官方底座 → S；输入核验、分组 SFT、tune 选点 |
| [fusion.py](src/fusion.py) | 590 | 固定目标、审计、损失与调度、三组续训、选点、评测和查询 Bootstrap |
| [evaluation.py](src/evaluation.py) | 80 | 查询隔离、教师弃权指标与 NLL |
| [common.py](src/common.py) | 249 | 数据契约、编码、指标、路由和等权融合 |
| [modeling.py](src/modeling.py) | 96 | Qwen + LoRA、四标签前向、预测与保存 |

历史轮次源码已经移除，历史文档保留并标注“历史”。`results/final/predictions/T/F_predictions.jsonl` 继续作为固定教师缓存。重构保持数值函数原文；旧 `run_manifest.json` 保留为历史证据，不用于恢复新代码运行。重新执行应使用新的 `runs/` 输出目录。

## 发布范围与数据来源

本次发布代码、配置、依赖、训练数据身份清单与汇总结果；未包含模型权重、原始数据、训练监督、逐题预测及教师响应缓存。克隆后可阅读代码、显示 CLI 参数；训练与指标复算需要另行准备对应输入。

数据来自 [Amazon ESCI 官方仓库](https://github.com/amazon-science/esci-data)，底座来自 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。数据和底座遵循各自上游许可证。本仓库未额外授予代码的开源许可证。
