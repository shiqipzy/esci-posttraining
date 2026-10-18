"""Train S, A2, TA2 and G2 from the official Qwen base in one fresh run."""
import sys
sys.dont_write_bytecode = True

import argparse
import gc
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from common import (
    ROOT, ARMS, LABELS, GAINS, check_output_path, collate, compact_metrics, digest,
    encode, epoch_groups, fresh_dir, fuse, load_split, low_confidence_indices, metrics,
    read_json, read_jsonl, require, verify_model, versions, write_json, write_jsonl,
)


def validate_probability(probabilities):
    require(isinstance(probabilities, dict) and set(probabilities) == set(LABELS),
            'Teacher probability labels changed')
    require(all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
                for value in probabilities.values()), 'Invalid teacher probability')
    require(abs(sum(probabilities.values()) - 1) < 1e-5, 'Unnormalized teacher probability')


def verify_inputs(data, cfg):
    """Validate training inputs without loading a model or reading test labels."""
    sft, _ = load_split(data / 'sft', cfg['sft']['train_split'])
    tune, _ = load_split(data / 'sft', 'tune')
    require(len({r['query_id'] for r in sft}) == cfg['sft']['train_queries'], 'SFT query count changed')

    post_dir = data / 'posttraining'
    manifest = read_json(post_dir / 'manifest.json')
    require(set(manifest['files']) == {'train.jsonl', 'teacher.jsonl'}, 'Unexpected training input')
    for name, expected in manifest['files'].items():
        require(digest(post_dir / name) == expected, 'Changed training input: ' + name)
    rows = read_jsonl(post_dir / 'train.jsonl')
    teacher = read_jsonl(post_dir / 'teacher.jsonl')
    require(len(rows) == len(teacher) == manifest['train_pairs'], 'Post-training row count changed')
    require(len({r['example_id'] for r in rows}) == len(rows), 'Duplicate training identity')
    require(len({r['query_id'] for r in rows}) == manifest['train_queries'], 'Post-training query count changed')
    available = correct = 0
    for row, target in zip(rows, teacher):
        require(all(row[k] == target[k] for k in ('example_id', 'query_id', 'product_id', 'label')),
                'Teacher/training identity mismatch')
        require(row['label'] in LABELS and isinstance(target['teacher_available'], bool),
                'Invalid training target')
        if target['teacher_available']:
            validate_probability(target['probabilities'])
            selected = max(LABELS, key=lambda k: target['probabilities'][k]) == row['label']
        else:
            require(target['probabilities'] is None, 'Unavailable teacher has probabilities')
            selected = False
        require(isinstance(target['mask'], bool) and target['mask'] == selected, 'Invalid correctness mask')
        available += target['teacher_available']
        correct += selected
    require((available, correct) == (manifest['teacher_available'], manifest['teacher_correct']),
            'Teacher coverage changed')
    require(correct > 0, 'No correct teacher rows')

    final_input = read_jsonl(data / 'final/input.jsonl')
    train_queries = {r['query_id'] for r in sft + rows}
    tune_queries = {r['query_id'] for r in tune}
    final_queries = {r['query_id'] for r in final_input}
    require(not train_queries & tune_queries, 'Training/validation query overlap')
    require(not final_queries & (train_queries | tune_queries), 'Training/test query overlap')
    require(all(not {'gold', 'label', 'mask', 'probabilities'} & r.keys() for r in final_input),
            'Test input contains targets')

    post = cfg['posttraining']
    require(tuple(post['arms']) == ARMS, 'Unexpected training arms')
    require(post['epochs'] > 0 and cfg['sft']['epochs'] > 0, 'Invalid training budget')
    require(post['effective_batch'] > 0 and post['microbatch'] > 0, 'Invalid post-training batch size')
    require(cfg['sft']['group_size'] > 0 and cfg['sft']['groups_per_update'] > 0 and cfg['sft']['micro_groups'] > 0,
            'Invalid SFT batch size')
    receipt = {
        'sft_pairs': len(sft), 'posttraining_pairs': len(rows),
        'teacher_available': available, 'teacher_correct': correct,
        'additional_weight': available / correct,
        'posttraining_steps_per_arm': math.ceil(len(rows) / post['effective_batch']) * post['epochs'],
        'sft_manifest_sha256': digest(data / 'sft/manifest.json'),
        'posttraining_manifest_sha256': digest(post_dir / 'manifest.json'),
        'final_input_sha256': digest(data / 'final/input.jsonl'),
    }
    return sft, tune, rows, teacher, final_input, receipt


def load_teacher_cache(path, final_inputs):
    cached = read_jsonl(path)
    require(len(cached) == len(final_inputs) and all(all(p[k] == r[k] for k in ('example_id', 'query_id', 'product_id'))
                for p, r in zip(cached, final_inputs)), 'Final teacher cache identity mismatch')
    for prediction in cached:
        if prediction['probabilities'] is None:
            require(prediction['prediction'] is None and prediction['score'] == 0,
                    'Invalid unavailable teacher row')
        else:
            validate_probability(prediction['probabilities'])
            require(prediction['prediction'] == max(LABELS, key=prediction['probabilities'].get),
                    'Teacher prediction/probability mismatch')
    return cached


def load_final_gold(data, final_inputs):
    gold = read_jsonl(data / 'final/gold.jsonl')
    require(len(gold) == len(final_inputs) and all(all(g[k] == r[k] for k in ('example_id', 'query_id', 'product_id'))
                for g, r in zip(gold, final_inputs)), 'Final labels/input identity mismatch')
    return gold


def reset_seed(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_gpu():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def record_update(path, entry):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry) + '\n')
    if entry['step'] % 25 == 0 or entry['step'] == entry['total_steps']:
        print(entry, flush=True)


def finish_update(params, optimizer, scheduler, cfg):
    import torch
    norm = torch.nn.utils.clip_grad_norm_(params, cfg['max_grad_norm'], error_if_nonfinite=True)
    lr = optimizer.param_groups[0]['lr']
    optimizer.step()
    scheduler.step()
    return {'gradient_norm': float(norm), 'learning_rate': lr}


def train_sft(source, output, cfg, rows, tune, log):
    import torch
    import torch.nn.functional as F
    from transformers import get_cosine_schedule_with_warmup
    from modeling import RelevanceModel, predict

    reset_seed(cfg['seed'])
    model = RelevanceModel(str(source), cfg, trainable=True)
    device = next(model.parameters()).device
    features = [encode(r, model.tokenizer, cfg['max_length']) for r in rows]
    params = [p for p in model.parameters() if p.requires_grad]
    steps_per_epoch = math.ceil(len(epoch_groups(rows, cfg['seed'], 0, cfg['group_size'])) / cfg['groups_per_update'])
    total = steps_per_epoch * cfg['epochs']
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], foreach=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
        num_warmup_steps=max(1, math.ceil(total * cfg['warmup_ratio'])), num_training_steps=total)
    best_key = best_state = None
    selected_epoch = step = 0
    curve = []
    for epoch in range(cfg['epochs']):
        groups = epoch_groups(rows, cfg['seed'], epoch, cfg['group_size'])
        model.train()
        seen = 0
        for start in range(0, len(groups), cfg['groups_per_update']):
            update = groups[start:start + cfg['groups_per_update']]
            count = sum(map(len, update))
            optimizer.zero_grad(set_to_none=True)
            update_loss = 0.
            for offset in range(0, len(update), cfg['micro_groups']):
                indices = [i for group in update[offset:offset + cfg['micro_groups']] for i in group]
                x = collate([features[i] for i in indices], model.tokenizer.pad_token_id, device)
                y = torch.tensor([LABELS.index(rows[i]['label']) for i in indices], device=device)
                loss = F.cross_entropy(model(x), y, reduction='sum') / count
                require(bool(torch.isfinite(loss)), 'Nonfinite SFT loss')
                loss.backward()
                update_loss += float(loss.detach())
            step += 1
            stat = finish_update(params, optimizer, scheduler, cfg)
            seen += count
            record_update(log, dict(stage='SFT', step=step, total_steps=total, epoch=epoch + 1,
                                    rows=count, loss=update_loss, **stat))
        require(seen == len(rows), 'Incomplete SFT epoch')
        measured = compact_metrics(metrics(predict(model, tune, cfg['eval_batch_size'], cfg['max_length'])))
        curve.append(dict(epoch=epoch + 1, **measured))
        key = (measured['macro_f1'], measured['ndcg10'], -epoch)
        if best_key is None or key > best_key:
            best_key, selected_epoch = key, epoch + 1
            best_state = {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    require(step == total, 'Incomplete SFT budget')
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(best_state[name])
    model.save(output)
    result = {'steps': step, 'selected_epoch': selected_epoch, 'validation': curve}
    del model, optimizer, scheduler, params, best_state
    release_gpu()
    return result


def probability_rows(model, rows, batch_size, max_length):
    """Inference uses query/title/brand only; target columns are never encoded."""
    from modeling import infer
    result = []
    for row, (values, _) in zip(rows, infer(model, rows, batch_size, max_length)):
        probabilities = dict(zip(LABELS, values))
        result.append({**{k: row[k] for k in ('example_id', 'query_id', 'product_id')},
                       'probabilities': probabilities,
                       'prediction': max(LABELS, key=probabilities.get),
                       'score': sum(v * g for v, g in zip(values, GAINS))})
    return result


def train_arm(arm, source, initial, output, cfg, rows, targets, reference, additional_weight, log):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from modeling import RelevanceModel, objective

    reset_seed(cfg['seed'])
    model_cfg = {**read_json(initial / 'run_config.json'), **cfg}
    model = RelevanceModel(str(source), model_cfg, trainable=True, checkpoint=initial)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    device = next(model.parameters()).device
    features = [encode(r, model.tokenizer, cfg['max_length']) for r in rows]
    labels = torch.tensor([LABELS.index(r['label']) for r in rows])
    teacher = torch.tensor([[t['probabilities'][k] for k in LABELS] if t['teacher_available'] else [.25] * 4 for t in targets])
    mask = torch.tensor([t['mask'] for t in targets], dtype=torch.bool)
    params = [p for p in model.parameters() if p.requires_grad]
    total = math.ceil(len(rows) / cfg['effective_batch']) * cfg['epochs']
    require(0 <= cfg['warmup_steps'] < total, 'Warmup exceeds training budget')
    optimizer = torch.optim.AdamW(params, lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'], foreach=False)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=cfg['warmup_steps'], num_training_steps=total)
    step = 0
    for epoch in range(cfg['epochs']):
        order = list(range(len(rows)))
        random.Random(cfg['seed'] + epoch).shuffle(order)
        model.train()
        for start in range(0, len(order), cfg['effective_batch']):
            indices = order[start:start + cfg['effective_batch']]
            optimizer.zero_grad(set_to_none=True)
            totals = {'CE': 0., 'KL_initial': 0., 'selected_extra': 0.}
            for offset in range(0, len(indices), cfg['microbatch']):
                part = indices[offset:offset + cfg['microbatch']]
                x = collate([features[i] for i in part], model.tokenizer.pad_token_id, device)
                loss, components = objective(model(x), labels[part].to(device), teacher[part].to(device),
                    reference[part].to(device), mask[part].to(device), arm, additional_weight)
                require(bool(torch.isfinite(loss).all()), 'Nonfinite post-training loss')
                (loss.sum() / len(indices)).backward()
                for name, value in components.items():
                    totals[name] += float(value.detach().sum()) / len(indices)
            step += 1
            stat = finish_update(params, optimizer, scheduler, cfg)
            record_update(log, dict(stage=arm, step=step, total_steps=total, epoch=epoch + 1,
                                    rows=len(indices), **totals, **stat))
    require(step == total, 'Incomplete post-training budget')
    model.save(output)
    del model, optimizer, scheduler, params
    release_gpu()
    return {'steps': step, 'epochs': cfg['epochs'],
            'initial_adapter_sha256': digest(initial / 'adapter/adapter_model.safetensors')}


def evaluate_final(source, output, data, cfg, rows, gold, cached_teacher, teacher_path):
    """Evaluate only after every final model has been saved."""
    from modeling import RelevanceModel

    sets = {}
    for name in ['S', *cfg['posttraining']['arms']]:
        checkpoint = output / 'models' / name
        model_cfg = read_json(checkpoint / 'run_config.json')
        model = RelevanceModel(str(source), model_cfg, checkpoint=checkpoint)
        predictions = probability_rows(model, rows, model_cfg['eval_batch_size'], model_cfg['max_length'])
        path = output / 'predictions' / name / 'F_predictions.jsonl'
        write_jsonl(path, predictions)
        write_json(path.parent / 'F_manifest.json', {'status': 'complete', 'predictions_sha256': digest(path),
            'identity': {'model_adapter_sha256': digest(checkpoint / 'adapter/adapter_model.safetensors'),
                         'input_sha256': digest(data / 'final/input.jsonl'), 'rows': len(rows)}})
        sets[name] = predictions
        del model
        release_gpu()
    sets['T'] = cached_teacher
    write_jsonl(output / 'predictions/T/F_predictions.jsonl', cached_teacher)
    for name in ('S', 'TA2'):
        sets[name + '_C50'] = fuse(sets[name], cached_teacher, low_confidence_indices(sets[name]))
        sets[name + '_Full'] = fuse(sets[name], cached_teacher, range(len(rows)))
        write_jsonl(output / 'predictions' / (name + '_C50') / 'F_predictions.jsonl', sets[name + '_C50'])
        write_jsonl(output / 'predictions' / (name + '_Full') / 'F_predictions.jsonl', sets[name + '_Full'])
    measured = {name: evaluation_metrics([dict(p, gold=g['label']) for p, g in zip(predictions, gold)])
                for name, predictions in sets.items()}
    write_json(output / 'analysis.json', {
        'dataset': {'pairs': len(rows), 'queries': len({r['query_id'] for r in rows}),
                    'input_sha256': digest(data / 'final/input.jsonl'),
                    'gold_sha256': digest(data / 'final/gold.jsonl')},
        'metrics': {name: compact_metrics(result) for name, result in measured.items()},
        'teacher_cache_sha256': digest(teacher_path)})
    for name, result in measured.items():
        print(name, {k: result[k] for k in ('accuracy', 'macro_f1', 'ndcg10')}, flush=True)


def evaluation_metrics(rows):
    """Keep teacher abstentions in classification support and the ranking pool."""
    if all(row['prediction'] in LABELS for row in rows):
        return metrics(rows)
    result = metrics([dict(row, prediction=None) for row in rows])
    support = Counter(row['gold'] for row in rows)
    predicted = Counter(row['prediction'] for row in rows)
    correct = Counter(row['gold'] for row in rows if row['prediction'] == row['gold'])
    result['accuracy'] = sum(correct.values()) / len(rows)
    result['per_class'] = {label: {'support': support[label], 'predicted': predicted[label],
        'precision': correct[label] / max(1, predicted[label]),
        'recall': correct[label] / max(1, support[label]),
        'f1': 2 * correct[label] / max(1, support[label] + predicted[label])} for label in LABELS}
    result['macro_f1'] = sum(value['f1'] for value in result['per_class'].values()) / len(LABELS)
    result['abstentions'] = sum(row['prediction'] is None for row in rows)
    result['missing_ranking_score'] = 0.0
    result['confusion'] = [[0] * 5 for _ in range(5)]
    for row in rows:
        result['confusion'][LABELS.index(row['gold'])][
            LABELS.index(row['prediction']) if row['prediction'] else 4] += 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'models/base')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/training.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/train')
    parser.add_argument('--teacher-predictions', type=Path, default=ROOT / 'results/final/predictions/T/F_predictions.jsonl')
    parser.add_argument('--check-only', action='store_true', help='verify inputs without training or creating output')
    args = parser.parse_args()
    cfg = read_json(args.config)
    output = check_output_path(args.output)
    sft, tune, rows, teacher, final_inputs, receipt = verify_inputs(args.data, cfg)
    verify_model(str(args.model_path.resolve()))
    cached_teacher = load_teacher_cache(args.teacher_predictions, final_inputs)
    gold = load_final_gold(args.data, final_inputs)
    if args.check_only:
        print(dict(status='PASS', **receipt, output=str(output), model_forwards=0, new_API_requests=0))
        return
    import torch
    from modeling import RelevanceModel
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'One CUDA GPU with BF16 support is required')
    fresh_dir(output)
    started = time.perf_counter()
    run = {'status': 'running', 'config': cfg, 'inputs': receipt, 'software': versions(),
           'base_model': str(args.model_path.resolve())}
    write_json(output / 'run_manifest.json', run)
    try:
        initial = output / 'models/S'
        run['S'] = train_sft(args.model_path, initial, cfg['sft'], sft, tune, output / 'train_log.jsonl')
        anchor = RelevanceModel(str(args.model_path), read_json(initial / 'run_config.json'), checkpoint=initial)
        reference_rows = probability_rows(anchor, rows, cfg['posttraining']['eval_batch_size'], cfg['posttraining']['max_length'])
        reference = torch.tensor([[r['probabilities'][k] for k in LABELS] for r in reference_rows], dtype=torch.float32)
        run['reference_adapter_sha256'] = digest(initial / 'adapter/adapter_model.safetensors')
        del anchor, reference_rows
        release_gpu()
        for arm in cfg['posttraining']['arms']:
            run[arm] = train_arm(arm, args.model_path, initial, output / 'models' / arm,
                cfg['posttraining'], rows, teacher, reference, receipt['additional_weight'], output / 'train_log.jsonl')
            write_json(output / 'run_manifest.json', run)
        evaluate_final(args.model_path, output, args.data, cfg, final_inputs, gold, cached_teacher, args.teacher_predictions)
        run['status'] = 'complete'
    except BaseException as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        run['elapsed_seconds'] = time.perf_counter() - started
        write_json(output / 'run_manifest.json', run)
    print('TRAINING_COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
