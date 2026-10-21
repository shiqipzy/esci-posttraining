"""Shared contract for the direct-SFT three-group experiment.

Covers the merged training inputs, the teacher correctness mask and weight, the
three objectives, evaluation metrics (including NLL), the plateau rule and the
checkpoint-selection rule.  This module must stay importable on CPU-only hosts:
torch is imported lazily inside the objective.
"""
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

from common import (LABELS, compact_metrics, digest, load_split, metrics, read_json,
                    read_jsonl, require)

ARMS = ('CE', 'KD', 'WCE')
NLL_CLIP = 1e-12
NLL_SCOPE = ('mean -log(p_gold); probability clipped to [1e-12, 1]; denominator excludes '
             'rows without probabilities (teacher abstentions); abstention count reported separately')


def normalized_text(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip().lower()


def teacher_weight(cfg):
    """c = teacher_available / teacher_correct, kept as an exact fraction."""
    available = cfg['inputs']['teacher_available']
    correct = cfg['inputs']['teacher_correct']
    require(available > 0 and correct > 0, 'Invalid teacher weight')
    return available, correct


def verify_teacher(rows, teacher, manifest, cfg):
    """Validate the frozen teacher cache and its one-to-one identity mapping."""
    require(len(rows) == len(teacher) == manifest['train_pairs'] == cfg['inputs']['post_rows'],
            'Post-training row count changed')
    available = correct = 0
    for row, target in zip(rows, teacher):
        require(all(row[k] == target[k] for k in ('example_id', 'query_id', 'product_id', 'label')),
                'Teacher/training identity mismatch')
        require(row['label'] in LABELS and isinstance(target['teacher_available'], bool),
                'Invalid training target')
        if target['teacher_available']:
            probs = target['probabilities']
            require(isinstance(probs, dict) and set(probs) == set(LABELS), 'Teacher probability labels changed')
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in probs.values()),
                    'Invalid teacher probability')
            require(abs(sum(probs.values()) - 1) < 1e-5, 'Unnormalized teacher probability')
            selected = max(LABELS, key=lambda k: probs[k]) == row['label']
        else:
            require(target['probabilities'] is None, 'Unavailable teacher has probabilities')
            selected = False
        require(isinstance(target['mask'], bool) and target['mask'] == selected, 'Invalid correctness mask')
        available += target['teacher_available']
        correct += selected
    require((available, correct) == (manifest['teacher_available'], manifest['teacher_correct']),
            'Teacher coverage changed')
    require((available, correct) == (cfg['inputs']['teacher_available'], cfg['inputs']['teacher_correct']),
            'Teacher coverage differs from the frozen recipe')
    require(correct > 0, 'No correct teacher rows')
    return available, correct


def load_direct_inputs(data, cfg):
    """Merge pilot SFT and post-training rows in a fixed order and map the teacher cache.

    Returns a dict with the merged rows, per-row teacher targets (None for rows without a
    cache), the boolean correctness mask, tune/gate/final splits and a receipt.
    """
    data = Path(data)
    sft_dir = data / 'sft'
    pilot, sft_manifest = load_split(sft_dir, cfg['inputs']['pilot_split'])
    tune, _ = load_split(sft_dir, 'tune')
    gate, _ = load_split(sft_dir, 'gate')
    require(len(pilot) == cfg['inputs']['pilot_rows'], 'Pilot SFT row count changed')
    require(len(tune) == cfg['inputs']['tune_rows'] and len(gate) == cfg['inputs']['gate_rows'],
            'Evaluation split count changed')

    post_dir = data / 'posttraining'
    manifest = read_json(post_dir / 'manifest.json')
    require(set(manifest['files']) == {'train.jsonl', 'teacher.jsonl'}, 'Unexpected training input')
    for name, expected in manifest['files'].items():
        require(digest(post_dir / name) == expected, 'Changed training input: ' + name)
    require(digest(post_dir / 'teacher.jsonl') == cfg['inputs']['teacher_sha256'],
            'Teacher cache differs from the frozen hash')
    post = read_jsonl(post_dir / 'train.jsonl')
    teacher = read_jsonl(post_dir / 'teacher.jsonl')
    available, correct = verify_teacher(post, teacher, manifest, cfg)

    require(len(post) == cfg['inputs']['post_rows'], 'Post-training row count changed')
    require(len({r['example_id'] for r in post}) == len(post), 'Duplicate post-training identity')
    require(len({r['query_id'] for r in post}) == cfg['inputs']['post_queries'], 'Post-training query count changed')
    require(len({r['query_id'] for r in pilot}) == cfg['inputs']['pilot_queries'], 'Pilot query count changed')

    # Fixed concatenation order: pilot rows, then post-training rows.  Never reordered here.
    rows = list(pilot) + list(post)
    targets = [None] * len(pilot) + list(teacher)
    require(len(rows) == cfg['inputs']['union_rows'], 'Union row count changed')
    require(len({r['example_id'] for r in rows}) == len(rows), 'Duplicate training identity in the union')
    require(len({r['query_id'] for r in rows}) == cfg['inputs']['union_queries'],
            'Union query count changed')

    final_input = read_jsonl(data / 'final/input.jsonl')
    final_gold = read_jsonl(data / 'final/gold.jsonl')
    require(len(final_input) == len(final_gold) == cfg['inputs']['final_rows'], 'Final split count changed')
    require(all(all(i[k] == g[k] for k in ('example_id', 'query_id', 'product_id'))
                for i, g in zip(final_input, final_gold)), 'Final labels/input identity mismatch')
    require(all(not {'gold', 'label', 'mask', 'probabilities'} & r.keys() for r in final_input),
            'Test input contains targets')

    isolation = check_isolation({'train': rows, 'tune': tune, 'gate': gate, 'final': final_input})

    receipt = {
        'pilot_rows': len(pilot), 'pilot_queries': len({r['query_id'] for r in pilot}),
        'posttraining_rows': len(post), 'posttraining_queries': len({r['query_id'] for r in post}),
        'union_rows': len(rows), 'union_queries': len({r['query_id'] for r in rows}),
        'union_order': 'pilot_train rows in file order, then posttraining/train rows in file order',
        'teacher_available': available, 'teacher_correct': correct,
        'teacher_weight': f'{available}/{correct}',
        'teacher_coverage_of_union': available / len(rows),
        'masked_share_of_union': correct / len(rows),
        'tune_rows': len(tune), 'tune_queries': len({r['query_id'] for r in tune}),
        'gate_rows': len(gate), 'gate_queries': len({r['query_id'] for r in gate}),
        'final_rows': len(final_input), 'final_queries': len({r['query_id'] for r in final_input}),
        'hashes': {
            'data/sft/pilot_train.jsonl': digest(sft_dir / 'pilot_train.jsonl'),
            'data/sft/tune.jsonl': digest(sft_dir / 'tune.jsonl'),
            'data/sft/gate.jsonl': digest(sft_dir / 'gate.jsonl'),
            'data/sft/manifest.json': digest(sft_dir / 'manifest.json'),
            'data/posttraining/train.jsonl': digest(post_dir / 'train.jsonl'),
            'data/posttraining/teacher.jsonl': digest(post_dir / 'teacher.jsonl'),
            'data/posttraining/manifest.json': digest(post_dir / 'manifest.json'),
            'data/final/input.jsonl': digest(data / 'final/input.jsonl'),
            'data/final/gold.jsonl': digest(data / 'final/gold.jsonl'),
        },
        'isolation': isolation,
    }
    return {'rows': rows, 'targets': targets, 'pilot': pilot, 'post': post, 'teacher': teacher,
            'tune': tune, 'gate': gate, 'final_input': final_input, 'final_gold': final_gold,
            'receipt': receipt, 'sft_manifest': sft_manifest, 'post_manifest': manifest}


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


def load_final_teacher(path, final_input, cfg):
    """Frozen F teacher cache: identity-checked, no request is issued."""
    path = Path(path)
    require(digest(path) == cfg['inputs']['final_teacher_sha256'],
            'Final teacher cache differs from the frozen hash')
    cached = read_jsonl(path)
    require(len(cached) == len(final_input), 'Final teacher cache length changed')
    require(all(all(p[k] == r[k] for k in ('example_id', 'query_id', 'product_id'))
                for p, r in zip(cached, final_input)), 'Final teacher cache identity mismatch')
    available = 0
    for prediction in cached:
        if prediction['probabilities'] is None:
            require(prediction['prediction'] is None and prediction['score'] == 0,
                    'Invalid unavailable teacher row')
        else:
            probs = prediction['probabilities']
            require(set(probs) == set(LABELS), 'Teacher probability labels changed')
            require(all(math.isfinite(v) and 0 <= v <= 1 for v in probs.values()), 'Invalid teacher probability')
            require(abs(sum(probs.values()) - 1) < 1e-5, 'Unnormalized teacher probability')
            require(prediction['prediction'] == max(LABELS, key=probs.get),
                    'Teacher prediction/probability mismatch')
            available += 1
    return cached, available


def sample_order(rows, cfg, epoch):
    """Epoch order shared by all arms: seed = base seed + epoch index."""
    import random
    order = list(range(len(rows)))
    random.Random(cfg['seed'] + epoch).shuffle(order)
    return order


def objective(logits, labels, teacher, mask, arm, weight):
    """Per-sample loss: CE; CE + c*m*KL(t||p); CE + c*m*CE. Temperature 1."""
    import torch
    import torch.nn.functional as F
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = F.nll_loss(logp, labels, reduction='none')
    if arm == 'CE':
        extra = ce.new_zeros(ce.shape)
    elif arm == 'KD':
        probs = teacher.clamp_min(NLL_CLIP)
        extra = (probs * (probs.log() - logp)).sum(-1)
    elif arm == 'WCE':
        extra = ce
    else:
        raise ValueError('Unknown arm: ' + str(arm))
    m = mask.to(device=logits.device, dtype=logp.dtype)
    loss = ce + weight * m * extra
    return loss, {'ce': ce, 'extra': m * extra}


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


def transition_counts(rows_a, rows_b):
    """Corrected / broken / net correct counts between two paired prediction sets."""
    require(len(rows_a) == len(rows_b), 'Unpaired prediction sets')
    corrected = broken = 0
    for a, b in zip(rows_a, rows_b):
        correct_a = a['prediction'] == a['gold']
        correct_b = b['prediction'] == b['gold']
        corrected += (not correct_b) and correct_a
        broken += correct_b and (not correct_a)
    return {'corrected': corrected, 'broken': broken, 'net': corrected - broken}


def plateau_status(history, window, threshold):
    """Operational plateau: the last `window` eval points must not raise either metric by
    `threshold` over the best value seen strictly before that window."""
    if len(history) < window + 1:
        return {'plateau': False, 'reason': 'insufficient_history', 'points': len(history)}
    recent = history[-window:]
    prior = history[:-window]
    detail = {}
    for key in ('accuracy', 'macro_f1'):
        best_prior = max(point[key] for point in prior)
        best_recent = max(point[key] for point in recent)
        gain = best_recent - best_prior
        detail[key] = {'best_before_window': best_prior, 'best_in_window': best_recent, 'gain': gain,
                       'below_threshold': bool(gain < threshold)}
    return {'plateau': all(v['below_threshold'] for v in detail.values()),
            'reason': 'checked', 'window': window, 'threshold': threshold,
            'window_steps': [p['step'] for p in recent], 'metrics': detail}


def select_checkpoint(history):
    """Tune Macro-F1 highest; exact tie -> nDCG@10 highest; exact tie -> earlier step."""
    best = None
    for point in history:
        key = (round(point['macro_f1'], 12), round(point['ndcg10'], 12), -point['step'])
        if best is None or key > best[0]:
            best = (key, point)
    return {'step': best[1]['step'], 'epoch': best[1]['epoch'],
            'tune_macro_f1': best[1]['macro_f1'], 'tune_accuracy': best[1]['accuracy'],
            'tune_ndcg10': best[1]['ndcg10'],
            'rule': 'max tune macro-F1, tie -> max tune nDCG@10, tie -> earlier step'}


def compact(result):
    return compact_metrics(result)


def file_hashes(paths, root):
    root = Path(root)
    return {str(Path(p)).replace('\\', '/'): digest(Path(p)) for p in paths}


def tree_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)).replace('\\', '/'): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def combination_label(a, b):
    return f'{a}-{b}'


def parse_comparison(label):
    a, b = label.split('-', 1)
    return a, b


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite value in output')
        return value
    return value


def write_json_strict(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2), encoding='utf-8')
