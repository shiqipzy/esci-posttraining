"""Train the first-stage S adapter from the official Qwen base."""
import sys
sys.dont_write_bytecode = True

import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

from common import (
    ROOT, LABELS, check_output_path, collate, compact_metrics, digest,
    encode, epoch_groups, fresh_dir, load_split, metrics, read_json, require,
    verify_model, versions, write_json,
)


def verify_inputs(data, cfg):
    """Validate SFT inputs without loading a model."""
    sft, _ = load_split(data / 'sft', cfg['train_split'])
    tune, _ = load_split(data / 'sft', 'tune')
    sft_queries = {r['query_id'] for r in sft}
    tune_queries = {r['query_id'] for r in tune}
    require(len(sft_queries) == cfg['train_queries'], 'SFT query count changed')
    require(not sft_queries & tune_queries, 'SFT/validation query overlap')
    require(cfg['epochs'] > 0, 'Invalid training budget')
    require(cfg['group_size'] > 0 and cfg['groups_per_update'] > 0 and cfg['micro_groups'] > 0,
            'Invalid SFT batch size')
    receipt = {'sft_pairs': len(sft), 'sft_queries': len(sft_queries),
               'tune_pairs': len(tune), 'tune_queries': len(tune_queries),
               'sft_manifest_sha256': digest(data / 'sft/manifest.json')}
    return sft, tune, receipt


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'models/base')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/training.json')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/train_sft')
    parser.add_argument('--check-only', action='store_true', help='verify inputs without training or creating output')
    args = parser.parse_args()
    cfg = read_json(args.config)['sft']
    output = check_output_path(args.output)
    sft, tune, receipt = verify_inputs(args.data, cfg)
    verify_model(str(args.model_path.resolve()))
    if args.check_only:
        print(dict(status='PASS', **receipt, output=str(output), model_forwards=0))
        return
    import torch
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'One CUDA GPU with BF16 support is required')
    fresh_dir(output)
    started = time.perf_counter()
    run = {'status': 'running', 'config': cfg, 'inputs': receipt, 'software': versions(),
           'base_model': str(args.model_path.resolve())}
    write_json(output / 'run_manifest.json', run)
    try:
        run['S'] = train_sft(args.model_path, output / 'models/S', cfg, sft, tune, output / 'train_log.jsonl')
        run['status'] = 'complete'
    except BaseException as exc:
        run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        run['elapsed_seconds'] = time.perf_counter() - started
        write_json(output / 'run_manifest.json', run)
    print('SFT_COMPLETE', output, flush=True)
