# 源码入口与分工

训练链路为官方 Qwen3-0.6B 底座 → S → CE_CONT / REF_KD / FUSION_KD。本目录只有以下 6 个 Python 文件。

| 文件 | 行数 | 职责 |
| --- | ---: | --- |
| [train.py](train.py) | 22 | 唯一 CLI，只分发 sft / fusion |
| [sft.py](sft.py) | 161 | 官方底座 → S；输入核验、分组 SFT、tune 选点 |
| [fusion.py](fusion.py) | 590 | 固定目标、审计、损失与调度、三组续训、选点、评测和查询 Bootstrap |
| [evaluation.py](evaluation.py) | 80 | 查询隔离、教师弃权指标与 NLL |
| [common.py](common.py) | 249 | 数据契约、编码、指标、路由和等权融合 |
| [modeling.py](modeling.py) | 96 | Qwen + LoRA、四标签前向、预测与保存 |

```bash
python src/train.py --help
python src/train.py sft --check-only
python src/train.py fusion --help
python src/train.py fusion --preflight
```

`train.main()` 只取出子命令并传递剩余参数，分别调用 `sft.main()` 与 `fusion.main()`。SFT 的 seed、分组与优化器规则保留；Fusion 的 `prepare`、`audit`、`loss_rows`、`train_arm`、`final_evaluation`、`paired_bootstrap` 保持原数值实现。`prepare()` 是构造新目标目录的内部函数；当前入口复用已经冻结的目标，不自动重建。

`evaluation.normalized_text()` 是隔离检查所需的辅助函数；`common.epoch_groups()` 和 `load_split()` 仍用于 SFT。已删除无调用的历史损失与辅助代码。

新运行的 `identity()` 记录当前 6 个源文件、配置与输入；历史运行清单不做兼容，不修改已发布结果。详细函数位置见[源码导读](../docs/源码导读/README.md)，环境和完整命令见[训练与评测](../docs/训练与评测.md)。
