# ESCI：搜索相关性后训练与融合蒸馏

基于 Amazon ESCI 数据集，使用 **Qwen3-0.6B + LoRA** 输出 Exact / Substitute / Complement / Irrelevant 四类相关性概率。先通过 SFT 得到初始学生 S，再将 S 与已有 DeepSeek 教师的半量融合输出作为固定监督，训练一个可独立推理的学生。

**最新实验已完成（2026-10-21）**：融合蒸馏学生在固定 F 上的 Macro-F1 为 **49.22%**，相对 CE 续训和原模型软监督分别提高 **0.91 / 1.04 个百分点**，两项查询配对 Bootstrap 95% 区间均为正。学生推理仅使用 0.6B 模型，本轮复用已有教师缓存，新增教师 API 请求为 0。

## 固定融合监督

已有实验观察到 S 与教师的概率融合收益。本轮研究的是将这部分收益迁移到单学生。S 的身份由 adapter SHA256 固定，三组均从同一个已训练 S adapter 开始续训，分别重置优化器与随机数状态。

仅使用 `data/posttraining/train.jsonl` 的 **10,398 条、512 个查询**；另外 20,941 条数据只属于此前训练 S 的阶段，不参加本轮三组续训。现有教师缓存中 10,387 条有效。

按固定 S 概率的最大值从低到高选取一半商品对，不使用真实标签或教师答对与否：

```text
q = (p_S + p_T) / 2    路由选中且教师概率有效
q = p_S                其余样本
```

训练集共路由 5,199 条，其中 5,195 条实际使用教师概率。全部 10,398 条都有固定目标 q，软监督组在全部样本上计算损失，目标不随学生参数更新。

## 三组对照

| 组 | 损失（温度 1，软监督系数 1） | 比较目的 |
|---|---|---|
| CE_CONT | `CE(y, p)` | 真实标签续训基线 |
| REF_KD | `(CE(y, p) + KL(p_S ∥ p)) / 2` | 控制原模型概率监督的作用 |
| FUSION_KD | `(CE(y, p) + KL(q ∥ p)) / 2` | 检验融合监督能否改善独立学生 |

KL 对四个类别求和，损失按实际 batch 样本数平均。两个软监督组的覆盖、系数及归一化一致；融合组不额外叠加另一项原模型 KL。

三组共享初始化、seed、数据顺序、学习率调度与评测条件。每组上限 650 次更新，均实际跑满；按 tune Macro-F1 独立选点，精确并列时依次比较 nDCG@10、较早步数。选中的步数为 CE_CONT / REF_KD **569**、FUSION_KD **488**。这是相同预算上限和选点规则下的比较，选中模型的训练曝光不相等。

## 独立学生结果

固定 F：512 个查询、10,459 个商品对。S 与三组学生在同一环境下重评；S_C50 是同环境重建的融合系统参考。

| 系统 | Accuracy | Macro-F1 | nDCG@10 |
|---|---:|---:|---:|
| S：初始 SFT 学生 | 54.83% | 47.50% | 0.7682 |
| CE_CONT：真实标签续训 | 55.88% | 48.31% | 0.7658 |
| REF_KD：原模型软监督 | 55.65% | 48.18% | 0.7671 |
| **FUSION_KD：融合蒸馏学生** | **55.73%** | **49.22%** | **0.7698** |
| S_C50：学生与教师半量融合参考 | 57.15% | 52.08% | 0.7804 |

10,000 次完整查询配对 Bootstrap；以下差值与 95% 区间单位均为百分点：

| 比较 | ΔAccuracy | ΔMacro-F1 |
|---|---|---|
| FUSION_KD − CE_CONT | −0.15 [−0.92, +0.61] | **+0.91 [+0.11, +1.74]** |
| FUSION_KD − REF_KD | +0.09 [−0.35, +0.52] | **+1.04 [+0.40, +1.73]** |
| FUSION_KD − S | **+0.90 [+0.24, +1.56]** | **+1.72 [+0.79, +2.72]** |

F 上的结果支持融合监督带来了超出 CE 续训及原模型软监督的 Macro-F1 收益。学生尚未达到融合系统本身的 52.08%。相对两个训练对照，Accuracy 与 nDCG@10 的区间均跨零；不能据此声称等效、非劣或排序提升。

本轮只有一个训练 seed，区间不包含训练随机性。F 已在项目迭代中查看过，属于既有固定评测集；本轮只用 tune 选点。完整结果、每类 F1、其他划分和限制见 [实验报告](results/fusion_continue_20261021/RESULT.md) 与 [report.json](results/fusion_continue_20261021/report.json)。

## 代码与运行

```bash
python src/train_fusion_continue.py --help
python src/fusion_continue.py --help
```

以上命令仅显示参数。完整输入、预检、训练、恢复与评测命令见 [运行说明](docs/RUNNING.md)。

| 功能 | 入口 |
|---|---|
| 当前三组续训、选点、恢复、统一评测与 Bootstrap | [train_fusion_continue.py](src/train_fusion_continue.py) |
| 固定目标构造、损失、学习率与停止规则 | [fusion_continue.py](src/fusion_continue.py) |
| LoRA 模型、四标签概率与推理 | [modeling.py](src/modeling.py) |
| 样本隔离检查与评测汇总 | [direct_common.py](src/direct_common.py) |
| 数据契约、指标、固定路由与融合 | [common.py](src/common.py) |
| 当前训练配置 | [fusion_continue_20261021.json](configs/fusion_continue_20261021.json) |
| 官方数据与底座准备 | [prepare.py](src/prepare.py) |

当前训练依赖源码与实际运行记录中的 SHA256 一致。训练环境、输入身份和选点记录分别见 [run_manifest.json](results/fusion_continue_20261021/run_manifest.json)、[target_manifest.json](results/fusion_continue_20261021/target_manifest.json)、[selected_all.json](results/fusion_continue_20261021/selected_all.json)。旧简历对应的门控实验报告保留在 [results/final](results/final/RESULT.md)，先前代码版本可通过 Git 提交历史查阅。

## 发布范围与来源

仓库提供代码、配置、依赖、身份清单和汇总结果；模型权重、原始数据、训练监督、逐题预测及教师响应缓存未随仓库发布。克隆后可阅读报告和查看 CLI 参数；完整训练及逐样本指标复算需要另行准备冻结输入，具体边界见运行说明。

数据来自 [Amazon ESCI](https://github.com/amazon-science/esci-data)，底座来自 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)，各自遵循上游许可证。本仓库未额外授予代码的开源许可证。
