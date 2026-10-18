# 运行说明

## 环境

Python 3.10+。模型训练参考环境为 CUDA 版 torch 2.8.0、transformers 4.57.6、PEFT 0.18.1；详细版本见 [requirements-training.txt](../requirements-training.txt)。先安装与 GPU 驱动匹配的 PyTorch，再安装训练依赖。训练要求 CUDA BF16。

```bash
python -m pip install -r requirements-training.txt
python src/prepare.py environment --gpu
```

## 输入

本次是代码发布，数据、教师缓存和模型未随仓库提供。完整训练需要如下本地布局：

```text
data/sft/                 pilot_train.jsonl、tune.jsonl、manifest.json 等
data/posttraining/        train.jsonl、teacher.jsonl、manifest.json
data/final/               input.jsonl、gold.jsonl
models/base/              固定 revision 的官方 Qwen3-0.6B
results/final/predictions/T/F_predictions.jsonl   固定教师缓存
```

训练用的教师正确性 mask 由训练标签构造，不用于选择最终测试题；推理路由仅依据学生置信度。`configs/posttraining_data_manifest.json` 记录训练样本数及输入哈希。

`prepare.py data` 根据官方 parquet 构建 tune、gate、pilot_train 三个按查询隔离的 SFT 划分；它不生成后训练教师监督或完整最终 F 输入。教师监督与最终教师响应均读取固定缓存，仓库不提供 API 采集入口。完整冻结输入包未随仓库发布，从空目录运行需要先准备这些输入。

## 训练与评测

输入齐备后，先核验，再写入一个新的运行目录：

```bash
python src/train.py --check-only
python src/train.py --model-path models/base --output runs/new_experiment
python src/evaluate.py --run runs/new_experiment --arm TA2 --split F --output runs/eval_F --device cuda
```

`train.py` 从官方底座重新训练 S，计算该 S 对训练样本的参考概率，然后从 S 分别训练 A2、TA2、G2，最后评测。三条续训臂各进行 650 次更新。`evaluate.py` 读取完整作业的 `run_manifest.json` 和 checkpoint，并向新目录输出评测结果。训练不支持断点续训；每次完整作业使用新的输出目录。仓库中的结果表是已完成实验的冻结报告，重新训练不会自动改写报告，也不承诺逐位复现原 adapter。

## 教师缓存

`train.py` 检查教师概率、样本身份和训练正确性 mask。`common.py` 按学生置信度选择半量样本并等权融合；教师响应不可用时使用学生概率。当前入口不发送 API 请求。历史实验使用 DeepSeek 教师，最终 F 的 10,459 条响应中有 4 条不可用；教师单独评测将其计为弃答，融合回退学生。

## 指标复算

安装 `requirements-offline.txt` 后，在补齐固定 F 输入、标签及全部九个对象的预测缓存时运行：

```bash
python src/reproduce_final.py
python src/reproduce_final.py --bootstrap
```

复算不需要 GPU、模型或 API，检查全部 9 个对象、固定融合结果和 13 项查询配对区间；本次未发布这些缓存，因此刚克隆后不能直接运行复算。

补齐训练输入、底座、四个学生 adapter、预测文件及其 manifest 后，可运行 `python src/verify_artifacts.py` 核验产物身份。它不校验源码哈希，也不是纯代码仓库的必过检查。
