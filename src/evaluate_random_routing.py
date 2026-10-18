"""CPU-only, matched-budget random routing on the frozen final F caches.

No model loading, training, API calls, or changes to the original final report.
The default experiment uses all 100 seeds from the historical random-routing
diagnostic. Bootstrap comparisons condition on these fixed routing masks.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
from common import (LABELS, digest, fresh_dir, fuse, low_confidence_indices, metrics,
                    read_json, read_jsonl)

METRIC_NAMES = ('accuracy', 'macro_f1', 'ndcg10')


def classification(counts):
    """Aggregate confusion matrices first, then compute accuracy and macro-F1."""
    tp = np.diagonal(counts, axis1=-2, axis2=-1)
    accuracy = tp.sum(-1) / counts.sum((-2, -1))
    denominator = counts.sum(-1) + counts.sum(-2)
    f1 = (2 * tp / np.maximum(1, denominator)).mean(-1)
    return accuracy, f1


def measured(predictions, gold, qids):
    joined = [dict(p, gold=g['label']) for p, g in zip(predictions, gold)]
    result = metrics(joined)
    assert [q['query_id'] for q in result['per_query']] == qids
    counts = np.array([q['confusion'] for q in result['per_query']], dtype=np.int64)
    ndcg = np.array([q['ndcg10'] if q['ndcg10'] is not None else np.nan
                     for q in result['per_query']])
    point = np.array([result[k] for k in METRIC_NAMES])
    acc, f1 = classification(counts.sum(0))
    np.testing.assert_allclose([acc, f1], point[:2], atol=1e-12, rtol=0)
    return point, counts, ndcg


def verify_fusion(generated, frozen):
    # Python's float summation can differ across runtime versions at the last bit.
    for a, b in zip(generated, frozen):
        assert {k: v for k, v in a.items() if k != 'score'} == {k: v for k, v in b.items() if k != 'score'}
        assert math.isclose(a['score'], b['score'], rel_tol=0, abs_tol=2e-16)


def bootstrap(low_counts, low_ndcg, random_counts, random_ndcg, weights):
    """Paired whole-query bootstrap: mean metrics over masks, not mean predictions."""
    runs, queries = random_counts.shape[:2]
    count_matrix = random_counts.transpose(1, 0, 2, 3).reshape(queries, runs * 16)
    valid = np.isfinite(low_ndcg)
    assert np.all(np.isfinite(random_ndcg) == valid[None, :])
    values = []
    for start in range(0, len(weights), 128):
        w = weights[start:start + 128]
        lc = (w @ low_counts.reshape(queries, 16)).reshape(-1, 4, 4)
        rc = (w @ count_matrix).reshape(-1, runs, 4, 4)
        la, lf = classification(lc)
        ra, rf = classification(rc)
        denominator = w @ valid.astype(float)
        ln = (w @ np.nan_to_num(low_ndcg)) / denominator
        rn = (w @ np.nan_to_num(random_ndcg).T) / denominator[:, None]
        values.append(np.column_stack((la - ra.mean(1), lf - rf.mean(1), ln - rn.mean(1))))
    return np.concatenate(values)


def mapping(values):
    return {k: float(v) for k, v in zip(METRIC_NAMES, values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/random_routing_F')
    args = parser.parse_args()
    out = args.output.resolve()
    fresh_dir(out)
    inputs = ROOT / 'data/final/input.jsonl'
    gold_path = ROOT / 'data/final/gold.jsonl'
    published = read_json(ROOT / 'results/final/analysis.json')
    rows, gold = read_jsonl(inputs), read_jsonl(gold_path)
    assert len(rows) == len(gold) == 10459
    qids = sorted({r['query_id'] for r in rows})
    assert len(qids) == 512
    assert digest(inputs) == published['dataset']['input_sha256']
    assert digest(gold_path) == published['dataset']['gold_sha256']
    paths = {n: ROOT / f'results/final/predictions/{n}/F_predictions.jsonl'
             for n in ('S', 'TA2', 'T', 'S_C50', 'TA2_C50')}
    caches = {n: read_jsonl(p) for n, p in paths.items()}
    for name, predictions in caches.items():
        assert len(predictions) == len(rows)
        for p, r, g in zip(predictions, rows, gold):
            assert all(p[k] == r[k] == g[k] for k in ('example_id', 'query_id', 'product_id'))
            if p['probabilities'] is not None:
                ps = p['probabilities']
                assert set(ps) == set(LABELS)
                assert all(math.isfinite(x) and 0 <= x <= 1 for x in ps.values())
                assert abs(sum(ps.values()) - 1) < 1e-5
        if name in published.get('student_sources', {}):
            assert digest(paths[name]) == published['student_sources'][name]
    teachers = caches['T']
    assert sum(t['probabilities'] is None for t in teachers) == 4
    seeds = list(range(1000, 1100))
    take = math.ceil(len(rows) / 2)
    protocol = {
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Supplementary replay on previously evaluated final F; not a fresh holdout.',
        'dataset': {'queries': len(qids), 'pairs': len(rows)},
        'students': ['S', 'TA2'], 'teacher': 'frozen T cache',
        'teacher_requests_per_route': take,
        'random_routing': {'seeds': seeds, 'runs': len(seeds),
                           'selection': 'uniform without replacement; exact fixed pair budget',
                           'seed_source': 'same 1000..1099 range as the historical diagnostic',
                           'shared_masks_across_students': True, 'best_seed_selection': False},
        'fusion': 'equal probability average on selected available rows; otherwise student',
        'routing_reads_gold': False,
        'bootstrap': {'draws': 2000, 'seed': 20260930, 'unit': 'whole query, paired',
                      'estimand': 'low-confidence metric minus mean random-route metric',
                      'conditioned_on': 'fixed models, teacher responses and 100 routing masks',
                      'reroute_within_bootstrap': False,
                      'intervals': 'individual percentile intervals; no multiple-comparison correction'},
        'classification_fusion_identity': 'exact probabilities and labels; score tolerance 2e-16',
        'inputs': {p.relative_to(ROOT).as_posix(): digest(p) for p in [inputs, gold_path, *paths.values()]},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
        'new_model_forwards': 0, 'new_api_requests': 0, 'optimizer_updates': 0,
    }
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rng = np.random.default_rng(20260930)
    ix = rng.integers(0, len(qids), size=(2000, len(qids)))
    weights = np.array([np.bincount(x, minlength=len(qids)) for x in ix], dtype=float)
    results = {}
    for base in ('S', 'TA2'):
        students = caches[base]
        frozen_low = caches[base + '_C50']
        verify_fusion(fuse(students, teachers, low_confidence_indices(students)), frozen_low)
        low_point, low_counts, low_ndcg = measured(frozen_low, gold, qids)
        for k, v in mapping(low_point).items():
            assert abs(v - published['metrics'][base + '_C50'][k]) < 1e-12
        points, counts, ndcgs, records = [], [], [], []
        for seed in seeds:
            chosen = np.random.default_rng(seed).choice(len(rows), take, replace=False)
            predictions = fuse(students, teachers, chosen)
            point, cc, ndcg = measured(predictions, gold, qids)
            points.append(point)
            counts.append(cc)
            ndcgs.append(ndcg)
            records.append({'seed': seed, 'selected_pairs': take,
                            'fallback_pairs': sum(teachers[int(i)]['probabilities'] is None
                                                  for i in chosen), 'metrics': mapping(point)})
        points = np.array(points)
        bootstrap_values = bootstrap(low_counts, low_ndcg, np.array(counts), np.array(ndcgs), weights)
        ci = np.quantile(bootstrap_values, [.025, .975], axis=0)
        delta = low_point - points.mean(0)
        chosen_low = low_confidence_indices(students)
        results[base] = {
            'low_confidence': {'metrics': mapping(low_point), 'selected_pairs': take,
                               'fallback_pairs': sum(teachers[i]['probabilities'] is None
                                                     for i in chosen_low)},
            'random_routing': {'mean_metrics': mapping(points.mean(0)),
                               'mask_95pct_range': {k: list(map(float, np.quantile(points[:, j], [.025, .975])))
                                                    for j, k in enumerate(METRIC_NAMES)},
                               'monte_carlo_standard_error': mapping(points.std(0, ddof=1) / math.sqrt(len(seeds))),
                               'runs': records},
            'low_minus_random_mean': {'point': mapping(delta),
                                       'query_paired_CI95': {k: list(map(float, ci[:, j]))
                                                             for j, k in enumerate(METRIC_NAMES)}},
        }
        print(base, 'low:', mapping(low_point), 'random mean:', mapping(points.mean(0)),
              'delta:', mapping(delta), flush=True)
    result = {'protocol': protocol, 'results': results}
    (out / 'analysis.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# 最终 F：同预算随机路由补充对照', '',
             '固定 F：512 查询 / 10,459 商品对。仅使用已有学生与教师概率缓存；无需 GPU、模型前向、训练或 API 调用。', '',
             'S 和 TA2 都完整报告，不按结果选择学生或随机 seed。每种路由均选择 5,230 个商品对；随机路由使用 seed 1000–1099，共 100 次。选择不读取标签，教师不可用时回退各自学生。', '',
             '| 学生 | 路由 | Accuracy | Macro-F1 | nDCG@10 |',
             '|---|---|---:|---:|---:|']
    for base, r in results.items():
        for route, ms in [('低置信度一半', r['low_confidence']['metrics']),
                          ('随机一半，100 次均值', r['random_routing']['mean_metrics'])]:
            lines.append(f"| {base} | {route} | {100*ms['accuracy']:.4f}% | {100*ms['macro_f1']:.4f}% | {ms['ndcg10']:.6f} |")
    lines += ['', '## 低置信度路由 − 随机路由均值', '',
              'Accuracy / Macro-F1 的差值与区间单位为百分点；nDCG 使用原始数值差。', '',
              '| 学生 | 指标 | 差值 | 查询配对 95% 区间 |', '|---|---|---:|---|']
    for base, r in results.items():
        comparison = r['low_minus_random_mean']
        for k in METRIC_NAMES:
            factor = 1 if k == 'ndcg10' else 100
            lo, hi = comparison['query_paired_CI95'][k]
            lines.append(f"| {base} | {k} | {factor*comparison['point'][k]:+.6f} | [{factor*lo:+.6f}, {factor*hi:+.6f}] |")
    lines += ['', '## 统计口径与边界', '',
              '- 随机基线是 100 个实际半量路由系统的指标均值；没有平均这些系统的概率后再构造一个额外集成模型，也没有挑选最优 seed。',
              '- Bootstrap 抽取完整查询，同一次抽样用于低置信度路由和全部随机路由；每个随机路由分别计算总体 Macro-F1，再对 100 个指标取均值。不能把每个查询的 F1 简单平均。',
              '- 区间条件于固定模型、教师缓存及这 100 个路由 mask。另行报告的 mask 分位范围描述随机选样波动；Monte Carlo 标准误描述有限 100 次对随机均值估计的精度，两者都不等于查询区间。',
              '- 查询重采样不重新选路由，故重采样后的教师比例可略变；它评估固定预测上的查询抽样不确定性。',
              '- 此为已查看最终 F 上的补充缓存对照，不是新的独立测试；单项区间未作多重比较校正，也不代表跨训练 seed、跨数据集或线上泛化。',
              '- 每条 5,230 次是模拟部署的调用预算。本次新增实际 API 请求为 0，历史完整教师缓存的采集费用不会因回放而减半。',
              '- 原始 final 预测、报告和审计未修改；融合概率、分类及身份严格一致，排序分数允许 2e-16 的浮点舍入差。', '',
              '完整逐 seed 指标、Monte Carlo 标准误、mask 分位范围和输入哈希见 [analysis.json](analysis.json)；输入哈希另见 [protocol.json](protocol.json)。', '',
              '复算时使用新的输出目录：', '', '```powershell',
              'python src/evaluate_random_routing.py --output results/random_routing_F_rerun', '```', '']
    (out / 'RESULT.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Saved', out, '; no model/API/training.', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
