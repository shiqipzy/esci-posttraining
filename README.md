# ESCI：搜索相关性后训练与教师概率蒸馏

基于 Amazon ESCI 数据集，以 Qwen3-0.6B + LoRA 输出 Exact / Substitute / Complement / Irrelevant 四类概率，研究教师软概率监督对单学生后训练的作用，以及低置信度样本的学生与教师协同推理。

本仓库提供训练、评测、教师缓存读取与概率融合代码，以及实验结果摘要。源码集中在 `src/`，训练统一入口为 `src/train.py`。教师信号来自已采集的固定响应缓存。

## 实验设计

- 按查询隔离数据；最终评测固定为官方 test 的 512 查询、10,459 商品对。
- S 为初始 SFT 学生；A2、TA2、G2 三组保持相同更新预算，在真实标签监督与原学生 KL 约束上比较额外监督。三组后训练各累计 650 次更新。
- TA2 对教师答对的 5,467 / 10,398 条训练样本施加教师概率监督；G2 在相同样本上增加真实标签监督；A2 为同预算基础对照。
- S_C50 根据 S 的置信度选取 5,230 条商品对，与教师概率等权融合；选择与融合不使用最终评测标签，教师不可用时回退学生。

## 结果

| 对象 | Accuracy | Macro-F1 |
|---|---:|---:|
| S：原 SFT 学生 | 54.99% | 47.63% |
| A2：同预算基础续训 | 55.73% | 48.15% |
| G2：同样本额外标签监督 | 55.58% | 48.62% |
| TA2：教师概率监督学生 | 55.91% | 48.82% |
| T：教师单独推理 | 53.23% | 50.15% |
| S_C50：原学生与教师半量融合 | 57.22% | 52.30% |

TA2 相对 A2 的 Macro-F1 增加 0.67 个百分点，相对 G2 的 Accuracy 增加 0.33 个百分点；这两项查询配对 Bootstrap 95% 区间均为正。两项指标分别报告，不把一个指标的优势推广到另一个指标。

S_C50 相对 S 的 Accuracy / Macro-F1 分别增加 2.24 / 4.67 个百分点。半量路由是使用固定教师响应缓存进行的离线评测；全量教师缓存已经采集，不能把离线选取比例当作本次采集实际节省的调用量。单学生仅一个训练种子；Bootstrap 条件于固定模型与教师响应。完整结果与限制见 [最终报告](results/final/RESULT.md) 和 [analysis.json](results/final/analysis.json)。

## 阅读与运行

```bash
python src/train.py --help
python src/prepare.py --help
python src/evaluate.py --help
```

以上命令仅显示参数，不加载模型或发送请求。完整训练、输入要求与离线复算边界见 [运行说明](docs/RUNNING.md)。

| 功能 | 源码 |
|---|---|
| SFT、三组续训与训练输出 | [train.py](src/train.py) |
| LoRA 模型与损失 | [modeling.py](src/modeling.py) |
| 教师缓存读取与有效性检查 | [train.py](src/train.py) |
| 低置信度路由与概率融合 | [common.py](src/common.py) |
| 官方数据准备、模型下载 | [prepare.py](src/prepare.py) |
| 模型评测、指标与缓存复算 | [evaluate.py](src/evaluate.py)、[common.py](src/common.py)、[reproduce_final.py](src/reproduce_final.py) |
| 同预算随机路由对照 | [evaluate_random_routing.py](src/evaluate_random_routing.py) |
| 输入、底座、adapter 与预测身份核验 | [verify_artifacts.py](src/verify_artifacts.py) |

## 发布范围与数据来源

本次发布代码、配置、依赖、训练数据身份清单与汇总结果；未包含模型权重、原始数据、训练监督、逐题预测及教师响应缓存。克隆后可阅读代码、显示 CLI 参数；训练与指标复算需要另行准备对应输入。

数据来自 [Amazon ESCI 官方仓库](https://github.com/amazon-science/esci-data)，底座来自 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)。数据和底座遵循各自上游许可证。本仓库未额外授予代码的开源许可证。
