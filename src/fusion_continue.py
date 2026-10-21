"""Fixed S/C50 targets and three S-initialized continuation objectives.

This module deliberately does not change any historical loss implementation.
Preparing targets and auditing their identity requires only the standard library.
"""
import argparse
import hashlib
import math
import random
from pathlib import Path

from common import (LABELS, ROOT, digest, fresh_dir, fuse, low_confidence_indices,
                    read_json, read_jsonl, require, write_json, write_jsonl)
from direct_common import check_isolation

ARMS = ('CE_CONT', 'REF_KD', 'FUSION_KD')
CONFIG = ROOT / 'configs/fusion_continue_20261021.json'
ARCHIVE = '_archive/20260930_legacy_workspace/esci_training/teacher_correct_v1/'
PINNED = {
    'data/posttraining/train.jsonl': 'b6403a21b12ad67f5c1e63f06bd7b33291c52459842007d887b2a0919689f62e',
    'data/posttraining/teacher.jsonl': '988aede6e6d66bfd50ee402f656746bd114fe509734a3c5533bf195adc54ce6f',
    ARCHIVE + 'data/supervision.jsonl': '3f7a5d60e0b8eb9250c84bfb8e85a436ced086e7ebc2ccbfb2165a62cdb30cba',
    'models/S/adapter/adapter_model.safetensors': '96c5ccc5ea9775f5da50ad25701cc8b2b0235b00b5e73943e2d0dff1ed2752da',
}


def probs_ok(p):
    require(set(p) == set(LABELS), 'Probability labels must be E,S,C,I')
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in p.values()), 'Invalid probability')
    require(abs(sum(p.values()) - 1) < 1e-5, 'Probability sum mismatch')


def prepare():
    cfg = read_json(CONFIG)
    for name, expected in PINNED.items():
        require(digest(ROOT / name) == expected, 'Source hash mismatch: ' + name)
    old_manifest = read_json(ROOT / ARCHIVE / 'data_manifest.json')
    require(old_manifest['checkpoints']['S']['adapter_sha256'] == cfg['initial_adapter_sha256'],
            'Archived probability source is not S')
    require(old_manifest['files']['supervision.jsonl'] == PINNED[ARCHIVE + 'data/supervision.jsonl'],
            'Archived supervision manifest mismatch')
    require(old_manifest['files']['train.jsonl'] == PINNED[cfg['train']], 'Archived train mismatch')
    rows = read_jsonl(ROOT / cfg['train'])
    teachers = read_jsonl(ROOT / 'data/posttraining/teacher.jsonl')
    old = read_jsonl(ROOT / ARCHIVE / 'data/supervision.jsonl')
    require(len(rows) == len(teachers) == len(old) == 10398, 'Source coverage mismatch')
    students, target_teachers = [], []
    for r, t, s in zip(rows, teachers, old):
        require(all(r[k] == t[k] == s[k] for k in ('example_id','query_id','product_id','label')),
                'Source row identity mismatch')
        require(t['probabilities'] == s['probabilities'], 'Teacher source mismatch')
        require(bool(t['teacher_available']) == (t['probabilities'] is not None), 'Teacher availability mismatch')
        probs_ok(s['reference'])
        if t['probabilities'] is not None:
            probs_ok(t['probabilities'])
        identity = {k: r[k] for k in ('example_id','query_id','product_id')}
        students.append(dict(identity, probabilities=s['reference']))
        target_teachers.append(dict(identity, probabilities=t['probabilities']))
    route = low_confidence_indices(students)
    fused = fuse(students, target_teachers, route)
    targets = []
    for i, (r, s, t, q) in enumerate(zip(rows, students, target_teachers, fused)):
        targets.append({**{k: r[k] for k in ('example_id','query_id','product_id','label')},
                        'reference_probabilities': s['probabilities'],
                        'fusion_probabilities': q['probabilities'],
                        'route_selected': i in route,
                        'teacher_available': t['probabilities'] is not None,
                        'teacher_used': i in route and t['probabilities'] is not None})
    out = ROOT / cfg['targets']
    fresh_dir(out.parent)
    write_jsonl(out, targets)
    input_keys = ['train','tune','gate','final_input','final_gold','final_teacher']
    pinned_files = {cfg[k]: digest(ROOT / cfg[k]) for k in input_keys}
    pinned_files.update({k: v for k, v in PINNED.items() if not k.startswith('_archive/')})
    pinned_files['models/S/adapter/adapter_config.json'] = digest(ROOT / 'models/S/adapter/adapter_config.json')
    pinned_files[cfg['targets']] = digest(out)
    write_json(ROOT / cfg['target_manifest'], {
        'status': 'ready_for_S_initialized_continuation; training_not_started',
        'source_sha256': PINNED,
        'files_sha256': pinned_files,
        'source_manifest_sha256': digest(ROOT / ARCHIVE / 'data_manifest.json'),
        'reference_adapter_sha256': cfg['initial_adapter_sha256'],
        'rows': 10398, 'queries': 512, 'soft_loss_rows': 10398,
        'route_selected': 5199, 'teacher_available': 10387, 'teacher_used': 5195,
        'routed_teacher_failures': 4, 'reference_only_targets': 5203,
        'extra_training_rows': 0, 'api_calls': 0, 'model_forwards': 0,
        'route': 'lowest max(p_S), ceil(N/2); SHA256 esci-cascade-v13|example_id tie-break',
        'target': '(p_S+p_T)/2 on routed valid rows; p_S otherwise',
        'gold_or_teacher_correctness_used_in_target': False,
        'teacher_probabilities': 'existing cache only; no API',
        'initialization': 'all arms load models/S adapter; fresh optimizer and RNG',
        'generator_sha256': digest(Path(__file__)),
    })
    print(audit(cfg)['counts'])


def epoch_order(n, seed, epoch):
    order = list(range(n))
    random.Random(seed + epoch).shuffle(order)
    return order


def lr_at(step, cfg):
    """LR used by a 1-based optimizer update, with explicit nonzero cosine floor."""
    if step <= cfg['warmup_updates']:
        return cfg['learning_rate'] * step / cfg['warmup_updates']
    progress = (step - cfg['warmup_updates']) / (cfg['max_updates'] - cfg['warmup_updates'])
    return cfg['min_learning_rate'] + (cfg['learning_rate'] - cfg['min_learning_rate']) * (1 + math.cos(math.pi * progress)) / 2


def loss_rows(logits, labels, target, arm, weight=1.0):
    import torch.nn.functional as F
    require(arm in ARMS, 'Unknown arm')
    logp = F.log_softmax(logits.float(), dim=-1)
    ce = F.nll_loss(logp, labels, reduction='none')
    # kl_div implements 0*log(0)=0 without perturbing or smoothing the target.
    kl = F.kl_div(logp, target.float(), reduction='none').sum(-1)
    loss = ce if arm == 'CE_CONT' else (ce + weight * kl) / (1 + weight)
    return loss, ce, kl


def best_key(point):
    return point['macro_f1'], point['ndcg10'], -point['step']


def stop_state(history, cfg):
    reference, stale = history[0]['macro_f1'], 0
    for point in history[1:]:
        if point['macro_f1'] >= reference + cfg['early_stop_min_delta']:
            reference, stale = point['macro_f1'], 0
        else:
            stale += 1
    return {'meaningful_best': reference, 'stale_evaluations': stale,
            'stop': history[-1]['step'] >= cfg['min_updates'] and stale >= cfg['early_stop_patience']}


def audit(cfg):
    require(tuple(cfg['arms']) == ARMS, 'Wrong arms')
    require(cfg['lambda_soft'] == cfg['temperature'] == 1, 'Frozen loss contract changed')
    manifest = read_json(ROOT / cfg['target_manifest'])
    for name, expected in manifest['files_sha256'].items():
        require(digest(ROOT / name) == expected, 'Input hash mismatch: ' + name)
    require(manifest['reference_adapter_sha256'] == cfg['initial_adapter_sha256'] ==
            PINNED['models/S/adapter/adapter_model.safetensors'], 'S identity changed')
    train = read_jsonl(ROOT / cfg['train'])
    targets = read_jsonl(ROOT / cfg['targets'])
    teacher = read_jsonl(ROOT / 'data/posttraining/teacher.jsonl')
    require(len(train) == len(targets) == len(teacher) == 10398, 'Training coverage changed')
    require(len({r['example_id'] for r in train}) == 10398, 'Duplicate train rows')
    require(len({r['query_id'] for r in train}) == 512, 'Training query count changed')
    ref = [{**{k:r[k] for k in ('example_id','query_id','product_id')},
            'probabilities':t['reference_probabilities']} for r,t in zip(train,targets)]
    route = low_confidence_indices(ref)
    used, avail = 0, 0
    for i, (r,t,c) in enumerate(zip(train, targets, teacher)):
        require(all(r[k] == t[k] == c[k] for k in ('example_id','query_id','product_id','label')), 'Target identity mismatch')
        for field in ('reference_probabilities','fusion_probabilities'):
            probs_ok(t[field])
        valid = c['probabilities'] is not None
        require(t['teacher_available'] == valid == bool(c['teacher_available']), 'Availability mismatch')
        require(t['route_selected'] == (i in route), 'Route mismatch')
        mix = i in route and valid
        require(t['teacher_used'] == mix, 'Teacher-used mismatch')
        expected = {k: (t['reference_probabilities'][k] + c['probabilities'][k])/2
                    if mix else t['reference_probabilities'][k] for k in LABELS}
        require(expected == t['fusion_probabilities'], 'Fusion value mismatch')
        avail += valid
        used += mix
    require((len(route), avail, used) == (5199,10387,5195), 'Target counts changed')
    tune = read_jsonl(ROOT / cfg['tune'])
    gate = read_jsonl(ROOT / cfg['gate'])
    final_input = read_jsonl(ROOT / cfg['final_input'])
    isolation = check_isolation({'train':train,'tune':tune,'gate':gate,'F':final_input})
    require((len(tune),len(gate),len(final_input)) == (5223,10438,10459), 'Evaluation counts changed')
    require(cfg['effective_batch_size'] == 32 and cfg['micro_batch_size'] == 4, 'Batch contract changed')
    require(cfg['max_updates'] == 650 and cfg['max_epochs'] == 2, 'Budget contract changed')
    require(cfg['eval_steps'] == [0,82,163,244,325,407,488,569,650], 'Evaluation contract changed')
    orders = [epoch_order(len(train), cfg['seed'], e) for e in range(cfg['max_epochs'])]
    require(all(len(set(o)) == len(train) for o in orders), 'Epoch coverage mismatch')
    order_hashes = [hashlib.sha256(','.join(map(str,o)).encode()).hexdigest() for o in orders]
    return {'counts': {'rows':len(train),'queries':512,'soft_loss_rows':len(targets),
                       'routed':len(route),'teacher_used':used,'tail_batch':30},
            'epoch_order_sha256':order_hashes, 'isolation':isolation,
            'train':train,'targets':targets,'tune':tune}


def cpu_checks(cfg):
    """Synthetic autograd only; no model weights, dataset training, or optimizer step."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(97)
    for n in [32,30]:
        x = torch.randn(n,4,requires_grad=True)
        y = torch.arange(n) % 4
        q = torch.rand(n,4)
        q[:,3] = 0
        q /= q.sum(-1,keepdim=True)
        for arm in ARMS:
            a = x.detach().clone().requires_grad_(True)
            loss,ce,kl = loss_rows(a,y,q,arm)
            loss.mean().backward()
            expected = F.cross_entropy(a,y) if arm == 'CE_CONT' else (
                F.cross_entropy(a,y) + F.kl_div(F.log_softmax(a,dim=-1),q,reduction='batchmean'))/2
            require(torch.allclose(loss.mean(),expected,atol=1e-6), 'Loss formula mismatch')
            b = x.detach().clone().requires_grad_(True)
            for start in range(0,n,4):
                z,_,_ = loss_rows(b[start:start+4],y[start:start+4],q[start:start+4],arm)
                (z.sum()/n).backward()
            require(torch.allclose(a.grad,b.grad,atol=1e-7), 'Tail/microbatch gradients differ')
            require(torch.isfinite(a.grad).all().item(), 'Nonfinite gradient at zero target')
    history = [dict(step=i*82,macro_f1=v,ndcg10=.7) for i,v in enumerate([.4,.4004,.4008,.4012])]
    require(stop_state(history,cfg)['stale_evaluations'] == 0, 'Cumulative small improvements lost')
    require(abs(lr_at(33,cfg)-2e-5)<1e-12 and abs(lr_at(650,cfg)-2e-6)<1e-12, 'LR endpoints wrong')
    print('PASS synthetic loss, zero probabilities, 32/30-row accumulation, early stopping and LR')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--cpu-check', action='store_true')
    args = parser.parse_args()
    cfg = read_json(CONFIG)
    if args.prepare:
        prepare()
    else:
        result = audit(cfg)
        print({k:v for k,v in result.items() if k not in ('train','targets','tune')})
    if args.cpu_check:
        cpu_checks(cfg)
