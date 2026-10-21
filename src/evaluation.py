"""Query isolation and classification/ranking metrics, including teacher abstentions."""
import math
import re
import unicodedata

from common import LABELS, metrics, require

NLL_CLIP = 1e-12
NLL_SCOPE = ('mean -log(p_gold); probability clipped to [1e-12, 1]; denominator excludes '
             'rows without probabilities (teacher abstentions); abstention count reported separately')


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
