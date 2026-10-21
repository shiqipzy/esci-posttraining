# 运行产物

当前已完成作业为 `fusion_continue_20261021/`。本地已同步报告、预测、选点、更新日志与运行记录；权重 checkpoint 保留在原执行主机，没有下载到本地。

- `evaluation/report.json`：当前三组与初始 S 的完整评测汇总。
- `evaluation/*_<split>.jsonl`：实际预测；`*.meta.json` 记录对应身份。
- `selected_all.json` 和各组 `selected.json`：实际选点、曝光、曲线和 adapter 身份。
- 各组 `updates_*.jsonl`：训练更新日志。
- `run_manifest.json`：配置、源码/输入哈希、runtime 与初始化。

当前续训入口是 `python src/train.py fusion`；`--resume` 校验相同代码、输入和 runtime 后恢复完整作业，必须具备对应 checkpoint 和训练状态。重构前的 `run_manifest.json` 仅作历史证据，不能用于恢复当前代码；重新执行应指定新的输出目录。仅有本地报告与预测不能恢复训练。

`direct_sft_20261019/`、`subset_direct_20261020/` 是两轮历史直训输出，不是当前融合蒸馏作业。运行文件默认不进入 Git；公开摘要见 [当前结果](../results/fusion_continue_20261021/RESULT.md)，运行方法见 [训练与评测](../docs/训练与评测.md)。
