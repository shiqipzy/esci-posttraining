# 运行说明：融合蒸馏

当前入口为 `src/train_fusion_continue.py`，配置为 `configs/fusion_continue_20261021.json`。已发布结果来自 2026-10-21 实际完成的三组续训与统一评测。

## 环境

实际运行环境：Python 3.12.3、CUDA 12.8、torch 2.8.0+cu128、transformers 4.57.6、PEFT 0.18.1、NumPy 2.3.2，GPU 为 RTX 4080 SUPER。完整版本见 [run_manifest.json](../results/fusion_continue_20261021/run_manifest.json)。

先安装适合 GPU 的 PyTorch，再安装本轮依赖：

```bash
python -m pip install -r requirements-fusion.txt
python src/train_fusion_continue.py --help
python src/fusion_continue.py --help
```

实际训练使用 BF16 backbone、FP32 loss、SDPA、关闭 TF32、确定性算法及 dropout=0。仅更新已有 LoRA 参数；S adapter 的 rank=16、alpha=32。

`requirements-training.txt` 和 `train.py` 是此前 SFT / 门控续训的工具，不能替代本轮入口或用它们重新训练的 S 冒充已发布结果所用的 S。当前续训直接读取固定 S adapter，无需重新执行 SFT。

## 冻结输入

权重、训练数据、概率目标及逐题缓存没有随 Git 仓库发布。完整运行需要如下布局：

```text
models/base/                             固定官方 Qwen3-0.6B revision
models/S/adapter/                        adapter_model.safetensors、adapter_config.json
data/posttraining/train.jsonl            10,398 条 / 512 查询
data/posttraining/teacher.jsonl          原有教师缓存，10,387 条有效
data/fusion_continue_20261021/targets.jsonl
data/fusion_continue_20261021/manifest.json
data/sft/tune.jsonl                      5,223 条 / 256 查询；唯一选点集合
data/sft/gate.jsonl                      10,438 条 / 512 查询；既有额外评测划分
data/final/input.jsonl                   F：10,459 条 / 512 查询
data/final/gold.jsonl
results/final/predictions/T/F_predictions.jsonl
```

S adapter SHA256：`96c5ccc5ea9775f5da50ad25701cc8b2b0235b00b5e73943e2d0dff1ed2752da`。底座由 [models.lock.json](../configs/models.lock.json) 固定。输入哈希见 [target_manifest.json](../results/fusion_continue_20261021/target_manifest.json)。

`targets.jsonl` 中每条记录包含样本身份、标签、`reference_probabilities`、`fusion_probabilities` 和路由/教师可用标志。固定 q 来自原 S 的概率及已有教师缓存；三组训练均只使用上述 10,398 条，不额外混入训练 S 时使用的 20,941 条。

目标构造入口为：

```bash
python src/fusion_continue.py --prepare
```

此命令写入新的目标目录，只读已有概率，不加载模型或请求教师。它需要代码中 `ARCHIVE` 指定的原始 `data/supervision.jsonl` 和 `data_manifest.json`，先核对 S、训练行及教师概率身份；这些来源文件也未随仓库发布。已有目标与 manifest 齐备时，预检和续训不再依赖原始归档目录。目标目录非空时拒绝覆盖。

发布的 `target_manifest.json` 是目标准备时的原始快照，其 `training_not_started` 仅描述当时状态；实际完成记录以 `run_manifest.json`、`selected_all.json` 和 `report.json` 为准。

## 预检、训练与恢复

输入齐备后，先做 CPU 预检和 GPU 前向检查：

```bash
python src/train_fusion_continue.py --preflight
python src/train_fusion_continue.py --gpu-check
```

CPU 预检核对数据、固定目标、底座/adapter 身份、分词长度，并用合成张量检查损失与尾批梯度；不更新模型。GPU 检查另外加载 S 做 8 条前向，不训练。

明确启动三组训练及评测，使用一个新的输出目录：

```bash
python src/train_fusion_continue.py --train --evaluate --output runs/fusion_new
```

恢复同一个作业：

```bash
python src/train_fusion_continue.py --train --evaluate --resume --output runs/fusion_new
```

仅对已经冻结选点、保留 checkpoint 的作业做统一 GPU 评测：

```bash
python src/train_fusion_continue.py --evaluate --output runs/fusion_new
```

恢复前会校验代码、配置、输入、模型哈希和 runtime，再恢复 optimizer/RNG。三组选点全部写入 `selected_all.json` 后才生成 gate/F 预测。训练中的 tune 评测与最终评测都使用 batch=8、相同精度和推理实现。模型推理为四标签 token 概率计算，关闭 thinking，无自由文本生成。

Linux 执行脚本为 `scripts/fusion_preflight.sh`、`scripts/fusion_run.sh`，可通过 `ESCI_PYTHON` 指定 Python 路径。`fusion_run.sh` 会实际启动训练，默认输出目录是 `runs/fusion_continue_20261021`，并使用 `flock` 防止同脚本并发。

## 训练与选点规则

| 参数 | 固定值 |
|---|---|
| seed | 20261001，每组重置 |
| 三组起点 | 同一 S adapter，各自新建 AdamW |
| effective / micro batch | 32 / 4；尾批 30，按实际样本数归一 |
| 预算 | 每组最多 650 updates / 2 epochs |
| 学习率 | 33 updates warmup 至 2e-5，cosine 至 2e-6 |
| weight decay / gradient clip | 0.01 / 1.0 |
| 温度 / 软监督系数 | 1 / 1 |
| tune 评测步 | 0、82、163、244、325、407、488、569、650 |
| 停止规则 | 至少 325 updates；连续 4 次评测未累计改善 0.001 Macro-F1，或耗尽预算 |
| 选点 | tune Macro-F1 最大；精确并列看 nDCG@10，再取更早步 |

本次三组都在 650 步耗尽预算，未触发早停。CE_CONT / REF_KD 选中 569 步（18,206 条次），FUSION_KD 选中 488 步（15,614 条次）。实际每组总曝光均为 20,796 条次；较早选点不代表本次实际少花了训练算力。

## 结果与复算边界

[结果目录](../results/fusion_continue_20261021/RESULT.md) 包含未经改写的机器可读报告、原始运行记录、目标身份和三组选点历史。`report.json` 还包含每类 precision/recall/F1、混淆矩阵及所有划分的比较。

10,000 次配对 Bootstrap 以完整查询为单位，seed=20260930；同一抽样用于比较双方，分类指标根据抽中的全部商品对重新计算。nDCG@10 使用 E/S/C/I 增益 1 / 0.1 / 0.01 / 0，按查询平均。单项区间未做多重比较校正；它们不估计重复训练的方差。

完整逐样本复算需要未发布的预测文件。仅凭本仓库的汇总 JSON 无法重建查询配对区间；`--evaluate` 是需要权重和 GPU 的模型评测，不是离线指标复算。旧工具 `reproduce_final.py` 对应旧门控实验，不能用其通过结果替代本轮验收。

教师缓存复用不产生新 API 请求；当前续训入口没有教师客户端。全部四个学生（S 和三组续训模型）的预测都由各自模型独立计算。
