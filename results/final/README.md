# 教师概率监督与融合结果

本实验比较初始 SFT 学生 S、基础续训 A2、教师概率监督 TA2、同样本额外标签监督 G2，以及学生与教师的概率融合。

- [完整结果](RESULT.md)：Accuracy、Macro-F1、nDCG@10 和查询配对区间。
- [analysis.json](analysis.json)：机器可读指标与比较。
- [对应发布版本](https://github.com/shiqipzy/esci-posttraining/tree/7bf6e36669dab7c1c6f60b3da81e630e00f0a54f)：实验代码与配置。

后续工作将半量融合输出蒸馏到单学生，见 [融合蒸馏结果](../fusion_continue_20261021/RESULT.md)。
