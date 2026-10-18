"""Offline final-F verification. No API, GPU, model loading or artifact overwrite."""
import argparse
from collections import Counter
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
from common import (LABELS, digest, fuse, low_confidence_indices, metrics, read_json, read_jsonl)


def same_fusion(generated, frozen):
    if len(generated) != len(frozen):
        return False
    for a, b in zip(generated, frozen):
        if {k: v for k, v in a.items() if k != 'score'} != {k: v for k, v in b.items() if k != 'score'}:
            return False
        if not math.isclose(a['score'], b['score'], rel_tol=0, abs_tol=2e-16):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', action='store_true', help='recompute the 13 paired query intervals')
    args = parser.parse_args()
    published = read_json(ROOT / 'results/final/analysis.json')
    rows = read_jsonl(ROOT / 'data/final/input.jsonl')
    gold = read_jsonl(ROOT / 'data/final/gold.jsonl')
    assert digest(ROOT / 'data/final/input.jsonl') == published['dataset']['input_sha256']
    assert digest(ROOT / 'data/final/gold.jsonl') == published['dataset']['gold_sha256']
    qids = sorted({r['query_id'] for r in rows})
    assert len(rows) == 10459 and len(qids) == 512
    qindex = {q: i for i, q in enumerate(qids)}
    sets = {}
    matrices = {}
    measured = {}
    for name, expected in published['metrics'].items():
        path = ROOT / f'results/final/predictions/{name}/F_predictions.jsonl'
        preds = read_jsonl(path)
        assert len(preds) == len(rows)
        assert all(all(x[k] == r[k] == g[k] for k in ('example_id', 'query_id', 'product_id'))
                   for x, r, g in zip(preds, rows, gold))
        for x in preds:
            if x['probabilities'] is not None:
                assert set(x['probabilities']) == set(LABELS)
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in x['probabilities'].values())
                assert abs(sum(x['probabilities'].values()) - 1) < 1e-5
        # The operational teacher has 4 abstentions. Preserve support, don't drop rows.
        support = Counter(g['label'] for g in gold)
        predicted = Counter(x['prediction'] for x in preds)
        tp = Counter(g['label'] for x, g in zip(preds, gold) if x['prediction'] == g['label'])
        point = {'macro_f1': sum(2 * tp[k] / max(1, support[k] + predicted[k]) for k in LABELS) / 4,
                 'accuracy': sum(tp.values()) / len(rows)}
        rank = metrics([dict(x, gold=g['label'], prediction=None) for x, g in zip(preds, gold)])
        point['ndcg10'] = rank['ndcg10']
        assert all(abs(point[k] - expected[k]) < 1e-12 for k in point), name + ' changed metrics'
        confusion = np.zeros((len(qids), 4, 5), dtype=np.int64)
        for x, g in zip(preds, gold):
            confusion[qindex[x['query_id']], LABELS.index(g['label']),
                      LABELS.index(x['prediction']) if x['prediction'] else 4] += 1
        matrices[name] = (confusion, np.array([r['ndcg10'] for r in rank['per_query']]))
        sets[name] = preds
        measured[name] = point
    for name in ('S', 'TA2'):
        for suffix in ('C50', 'Full'):
            selected = low_confidence_indices(sets[name]) if suffix == 'C50' else range(len(rows))
            assert same_fusion(fuse(sets[name], sets['T'], selected), sets[name + '_' + suffix]), \
                'Changed fusion policy'
    assert all(digest(ROOT / f'results/final/predictions/{n}/F_predictions.jsonl') == h
               for n, h in published['student_sources'].items())
    policy = read_json(ROOT / 'configs/evaluation_policy.json')
    assert policy['classification_metrics'] == ['accuracy', 'macro_f1']
    print(f'Fixed F: 512 queries / 10459 pairs; all {len(measured)} models and fusion identities PASS')
    for name in measured:
        print(f"{name:12s} accuracy={measured[name]['accuracy']:.6f} "
              f"macro-F1={measured[name]['macro_f1']:.6f} nDCG@10={measured[name]['ndcg10']:.6f}")
    if args.bootstrap:
        def f1(c):
            return np.mean(2 * np.diagonal(c[:, :4]) / np.maximum(1, c.sum(1) + c[:, :4].sum(0)))
        for label, expected in published['comparisons'].items():
            a, b = label.split('-')
            ca, na = matrices[a]
            cb, nb = matrices[b]
            rng = np.random.default_rng(20260930)
            values = []
            for _ in range(2000):
                ix = rng.integers(0, len(qids), len(qids))
                x, y = ca[ix].sum(0), cb[ix].sum(0)
                values.append([f1(x) - f1(y), na[ix].mean() - nb[ix].mean(),
                               (np.trace(x[:, :4]) - np.trace(y[:, :4])) / x.sum()])
            ci = np.quantile(values, [.025, .975], axis=0)
            for i, k in enumerate(['macro_f1', 'ndcg10', 'accuracy']):
                assert np.allclose(ci[:, i], expected['CI95'][k], atol=1e-12, rtol=0), label + ' ' + k
            print(f"{label:20s} delta_accuracy={100*expected['point']['accuracy']:+.4f} pp "
                  f"CI95=[{100*ci[0,2]:+.4f}, {100*ci[1,2]:+.4f}] pp; "
                  f"delta_macro-F1={100*expected['point']['macro_f1']:+.4f} pp "
                  f"CI95=[{100*ci[0,0]:+.4f}, {100*ci[1,0]:+.4f}] pp")
        print('All 13 paired query bootstrap comparisons PASS (2000 draws, seed 20260930)')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
