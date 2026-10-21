# evaluation.py：查询隔离、弃权指标与 NLL（80 行）

本模块只依赖标准库与 `common`。`fusion.audit()` 使用隔离检查；`evaluate_point()` 与 `final_evaluation()` 使用评测指标。模型分类使用 4 类；教师弃权仍进入分类分母，混淆矩阵扩为 5×5。

| 函数 | 行号 | 作用 |
| --- | --- | --- |
| `normalized_text` | 13–14 | NFKC、合并空白、去首尾空白、转小写 |
| `check_isolation` | 17–30 | 跨集合检查 query_id、规范化查询文本、example_id |
| `evaluation_metrics` | 33–64 | 分类、排序、弃权与 NLL |
| `nll` | 67–80 | 有概率的样本计算负对数似然，并单列弃权数 |

## 1. 数据隔离

```python
def normalized_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().lower()

def check_isolation(groups):
    """Query IDs, normalized query text and example IDs must be disjoint across sets."""
    qids = {name: {r['query_id'] for r in rows} for name, rows in groups.items()}
    texts = {name: {normalized_text(r['query']) for r in rows} for name, rows in groups.items()}
    examples = {name: {r['example_id'] for r in rows} for name, rows in groups.items()}
    names = list(groups)
    receipt = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = {'query_ids': len(qids[a] & qids[b]), 'normalized_texts': len(texts[a] & texts[b]),
                       'example_ids': len(examples[a] & examples[b])}
            require(not any(overlap.values()), f'Set isolation violated: {a} vs {b} {overlap}')
            receipt[f'{a}_vs_{b}'] = overlap
    return receipt
```

不能只检查 example_id：同一查询下的不同商品也可能泄漏。三种身份都要互斥；`train/tune/gate/F` 的两两重叠必须为零。SFT 另由 `verify_inputs()` 核对自己的 pilot 与 tune 划分。

## 2. 完整预测与教师弃权

```python
def evaluation_metrics(rows):
    """Accuracy/Macro-F1 including teacher abstentions, plus nDCG@10 and NLL."""
    require(rows and len({r['example_id'] for r in rows}) == len(rows), 'Empty/duplicate predictions')
    has_abstention = any(r['prediction'] is None for r in rows)
    if not has_abstention:
        result = metrics(rows)
    else:
        result = metrics([dict(r, prediction=None) for r in rows])
        support = {label: 0 for label in LABELS}
        predicted = {label: 0 for label in LABELS}
        correct = {label: 0 for label in LABELS}
        for row in rows:
            support[row['gold']] += 1
            if row['prediction'] is not None:
                predicted[row['prediction']] += 1
                if row['prediction'] == row['gold']:
                    correct[row['gold']] += 1
        result['accuracy'] = sum(correct.values()) / len(rows)
        result['per_class'] = {label: {
            'support': support[label], 'predicted': predicted[label],
            'precision': correct[label] / max(1, predicted[label]),
            'recall': correct[label] / max(1, support[label]),
            'f1': 2 * correct[label] / max(1, support[label] + predicted[label])} for label in LABELS}
        result['macro_f1'] = sum(v['f1'] for v in result['per_class'].values()) / len(LABELS)
        result['abstentions'] = sum(row['prediction'] is None for row in rows)
        result['missing_ranking_score'] = 0.0
        result['confusion'] = [[0] * 5 for _ in range(5)]
        for row in rows:
            result['confusion'][LABELS.index(row['gold'])][
                LABELS.index(row['prediction']) if row['prediction'] else 4] += 1
    result['nll'] = nll(rows)
    return result
```

没有弃权时直接使用 `common.metrics()`。有弃权时，先将所有 `prediction` 临时置为 None，仅计算排序部分；然后以完整 gold support 重算 Accuracy 和各类 F1。弃权行不增加 predicted 或 correct，但保留在 support 和 Accuracy 分母中。

因此 `F1_k = 2 TP_k / (support_k + predicted_k)`，四类 F1 等权平均。5×5 混淆矩阵的前四列是预测 E/S/C/I，第五列是弃权；最后一行没有真实类别，保持为零。排名沿用缓存中的 score，缺失教师行的 score 为 0。

`T_cached_reference` 在 F 中共 10,459 行、4 次弃权，第五列为 `[2,1,0,1,0]`。学生每行都有四类概率，仍使用 4×4 混淆矩阵。

## 3. NLL 的分母与截断

```python
def nll(rows):
    total = 0.0
    counted = 0
    for row in rows:
        probs = row.get('probabilities')
        if probs is None:
            continue
        p = probs[row['gold']]
        require(math.isfinite(p) and p >= 0, 'Invalid probability for NLL')
        total -= math.log(max(p, NLL_CLIP))
        counted += 1
    require(counted > 0, 'No rows with probabilities for NLL')
    return {'mean': total / counted, 'rows': counted, 'abstentions': len(rows) - counted,
            'clip': NLL_CLIP, 'scope': NLL_SCOPE}
```

NLL 仅统计有概率的行，真实标签概率下限为 1e-12；弃权不进入 NLL 的分母，另记 `abstentions`。这与 Accuracy / Macro-F1 保留所有 gold support 的口径不同。`NLL_SCOPE` 原文写入输出，说明统计范围。

## 4. 只读核对

```bash
python -c "import json; m=json.load(open('results/fusion_continue_20261021/report.json',encoding='utf-8'))['metrics']['F']['T_cached_reference']; print(len(m['confusion']), [row[-1] for row in m['confusion']], m['abstentions'])"
```

输出 `5 [2, 1, 0, 1, 0] 4`。指标底层公式与排序 tie-break 见[01 common](01-common.md)，按查询重采样见[04 训练](04-training.md)，训练曝光数字见[06 数字推演](06-一次更新的数字推演.md)。
