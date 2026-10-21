"""Continue S with fixed targets: audit, CPU preflight, three arms and evaluation.

Use python src/train.py fusion --help for the shared command-line entry point.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from common import (LABELS, ROOT, check_output_path, collate, compact_metrics, digest,
                    encode, fresh_dir, fuse, low_confidence_indices, read_json, read_jsonl,
                    require, verify_model, versions, write_json, write_jsonl)
from evaluation import check_isolation, evaluation_metrics

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


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def runtime():
    import torch
    return {'versions':versions(), 'python':__import__('sys').version,
            'cuda_available':torch.cuda.is_available(), 'cuda_version':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'eval_batch_size':8, 'attention':'sdpa', 'backbone_dtype':'bfloat16',
            'loss_dtype':'float32', 'tf32':False, 'deterministic_algorithms':True}


def gpu_required():
    import torch
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            'CUDA BF16 GPU unavailable: enable the GPU instance before training; no CPU fallback')


def load_model(cfg, checkpoint, trainable):
    import torch
    from modeling import RelevanceModel
    model = RelevanceModel(ROOT / cfg['base'], cfg, trainable=trainable, checkpoint=checkpoint)
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for adapter in model.backbone.peft_config.values():
        adapter.lora_dropout = 0.0
    if trainable:
        trainable_names = [name for name,p in model.named_parameters() if p.requires_grad]
        require(trainable_names and all('lora_' in name for name in trainable_names),
                'Only existing LoRA parameters may be trained')
    return model


def identity(cfg):
    files = ['src/train.py','src/sft.py','src/fusion.py','src/evaluation.py',
             'src/common.py','src/modeling.py','configs/models.lock.json',str(CONFIG.relative_to(ROOT)),
             cfg['target_manifest'],cfg['targets'],'models/S/adapter/adapter_model.safetensors',
             'models/S/adapter/adapter_config.json']
    return {p:digest(ROOT / p) for p in files}


def preflight(cfg, gpu=False):
    data = audit(cfg)
    cpu_checks(cfg)
    verify_model(str((ROOT / cfg['base']).resolve()))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg['base'], local_files_only=True)
    max_tokens = {}
    for name,key in [('train','train'),('tune','tune'),('gate','gate'),('F','final_input')]:
        rows = read_jsonl(ROOT / cfg[key])
        lengths = [len(encode(r,tokenizer,cfg['max_length'])['input_ids']) for r in rows]
        max_tokens[name] = max(lengths)
    require(all(len(tokenizer.encode(label,add_special_tokens=False)) == 1 for label in LABELS),
            'Labels must each be a single token')
    result = {'status':'PASS_CPU_PREFLIGHT','counts':data['counts'],
              'epoch_order_sha256':data['epoch_order_sha256'], 'isolation':data['isolation'],
              'max_input_tokens':max_tokens,'identity':identity(cfg),'runtime':runtime(),
              'optimizer_steps':0,'teacher_api_calls':0,'gpu_forward_checked':False}
    if gpu:
        gpu_required()
        seed_all(cfg['seed'])
        model = load_model(cfg,ROOT / cfg['initial_checkpoint'],False)
        from modeling import infer
        output = infer(model,data['train'][:8],cfg['eval_batch_size'],cfg['max_length'])
        require(all(all(math.isfinite(p) for p in probs) for probs,_ in output), 'GPU smoke produced nonfinite probabilities')
        result.update(status='PASS_GPU_PREFLIGHT',gpu_forward_checked=True)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


def snapshot(model, optimizer, folder, state):
    import numpy as np
    import torch
    folder = Path(folder)
    require(not folder.exists(), 'Refuse checkpoint overwrite: ' + str(folder))
    pending = folder.with_name(folder.name + '.pending-' + uuid.uuid4().hex[:8])
    model.save(pending)
    torch.save({'optimizer':optimizer.state_dict(), 'python_rng':random.getstate(),
                'numpy_rng':np.random.get_state(), 'torch_rng':torch.get_rng_state(),
                'cuda_rng':torch.cuda.get_rng_state_all()},pending / 'training_state.pt')
    state = dict(state, adapter_sha256=digest(pending / 'adapter/adapter_model.safetensors'),
                 training_state_sha256=digest(pending / 'training_state.pt'))
    write_json(pending / 'state.json',state)
    pending.rename(folder)


def restore_rng_optimizer(folder, optimizer):
    import numpy as np
    import torch
    state = read_json(folder / 'state.json')
    require(digest(folder / 'training_state.pt') == state['training_state_sha256'], 'Resume state hash mismatch')
    require(digest(folder / 'adapter/adapter_model.safetensors') == state['adapter_sha256'], 'Resume adapter hash mismatch')
    # This is our own locally generated checkpoint, pinned by the run manifest.
    saved = torch.load(folder / 'training_state.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(saved['optimizer'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return state


def evaluate_point(model, rows, cfg, step, exposures, directory):
    from modeling import predict
    predictions = predict(model,rows,cfg['eval_batch_size'],cfg['max_length'])
    result = evaluation_metrics(predictions)
    point = dict(step=step,exposures=exposures,epoch=exposures/10398,**compact_metrics(result))
    write_jsonl(directory / f'tune_step_{step:04d}.jsonl',predictions)
    return point


def train_arm(arm,cfg,data,out,resume):
    import torch
    seed_all(cfg['seed'])
    arm_dir = out / arm
    arm_dir.mkdir(parents=True,exist_ok=True)
    selected_file = arm_dir / 'selected.json'
    if selected_file.exists():
        require(resume,'Arm already completed; use --resume')
        return read_json(selected_file)
    snapshots = sorted(p for p in arm_dir.glob('step-*') if (p / 'state.json').is_file())
    require(resume or not snapshots,'Existing checkpoints; use --resume')
    initial = snapshots[-1] if snapshots else ROOT / cfg['initial_checkpoint']
    model = load_model(cfg,initial,True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    features = [encode(r,model.tokenizer,cfg['max_length']) for r in data['train']]
    labels = torch.tensor([LABELS.index(r['label']) for r in data['train']],device='cuda')
    field = 'fusion_probabilities' if arm == 'FUSION_KD' else 'reference_probabilities'
    targets = torch.tensor([[t[field][k] for k in LABELS] for t in data['targets']],
                           dtype=torch.float32,device='cuda')
    if snapshots:
        state = restore_rng_optimizer(initial,optimizer)
        step,exposures,history = state['step'],state['exposures'],state['history']
    else:
        step,exposures = 0,0
        history = [evaluate_point(model,data['tune'],cfg,0,0,arm_dir)]
        snapshot(model,optimizer,arm_dir / 'step-0000',
                 dict(arm=arm,step=0,exposures=0,history=history,epoch_order_sha256=data['epoch_order_sha256']))
    log = arm_dir / ('updates_' + time.strftime('%Y%m%dT%H%M%S') + '.jsonl')
    updates_per_epoch = math.ceil(len(features)/cfg['effective_batch_size'])
    start_time = time.monotonic()
    stopped = stop_state(history,cfg)['stop']
    while step < cfg['max_updates'] and not stopped:
        epoch,batch_no = divmod(step,updates_per_epoch)
        order = epoch_order(len(features),cfg['seed'],epoch)
        indices = order[batch_no*32:(batch_no+1)*32]
        require(indices,'Empty effective batch')
        model.train()
        optimizer.zero_grad(set_to_none=True)
        lr = lr_at(step+1,cfg)
        for group in optimizer.param_groups:
            group['lr'] = lr
        sums = [0.0,0.0,0.0]
        for start in range(0,len(indices),cfg['micro_batch_size']):
            selected = indices[start:start+cfg['micro_batch_size']]
            batch = collate([features[i] for i in selected],model.tokenizer.pad_token_id,'cuda')
            logits = model(batch)
            loss,ce,kl = loss_rows(logits,labels[selected],targets[selected],arm,cfg['lambda_soft'])
            require(torch.isfinite(loss).all().item(),'Nonfinite loss; stop run')
            (loss.sum()/len(indices)).backward()
            for j,value in enumerate((loss,ce,kl)):
                sums[j] += value.detach().sum().item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['max_grad_norm'],error_if_nonfinite=True)
        optimizer.step()
        step += 1
        exposures += len(indices)
        record = dict(step=step,epoch=epoch,rows=len(indices),exposures=exposures,
                      lr=lr,loss=sums[0]/len(indices),ce=sums[1]/len(indices),
                      kl=sums[2]/len(indices),soft_coefficient=0 if arm=='CE_CONT' else cfg['lambda_soft'],
                      grad_norm_before_clip=float(grad_norm),clip_applied=float(grad_norm)>cfg['max_grad_norm'],
                      seconds_since_process_start=time.monotonic()-start_time)
        with log.open('a',encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        if step % 20 == 0:
            print(arm,json.dumps(record),flush=True)
        if step in cfg['eval_steps']:
            point = evaluate_point(model,data['tune'],cfg,step,exposures,arm_dir)
            history.append(point)
            status = stop_state(history,cfg)
            snapshot(model,optimizer,arm_dir / f'step-{step:04d}',
                     dict(arm=arm,step=step,exposures=exposures,history=history,
                          epoch_order_sha256=data['epoch_order_sha256'],early_stop=status))
            print('TUNE',arm,json.dumps({'step':step,'macro_f1':point['macro_f1'],
                                       'accuracy':point['accuracy'],'early_stop':status}),flush=True)
            stopped = status['stop']
    best = max(history,key=best_key)
    checkpoint = arm_dir / f"step-{best['step']:04d}"
    selected = {'arm':arm,'checkpoint':str(checkpoint.relative_to(out)),
                'selected_step':best['step'],'selected_exposures':best['exposures'],
                'stopped_step':step,'stopped_exposures':exposures,
                'stop_reason':'early_stop' if stopped else 'budget_exhausted',
                'selected_adapter_sha256':digest(checkpoint / 'adapter/adapter_model.safetensors'),
                'selection_rule':'highest tune Macro-F1; exact tie highest nDCG@10; exact tie earliest step',
                'tune':best,'history':history}
    write_json(selected_file,selected)
    del model,optimizer,targets,labels
    gc.collect()
    torch.cuda.empty_cache()
    return selected


def train(cfg,out,resume):
    gpu_required()
    data = audit(cfg)
    frozen_identity = identity(cfg)
    out.mkdir(parents=True,exist_ok=True)
    run_manifest = out / 'run_manifest.json'
    current_runtime = runtime()
    if run_manifest.exists():
        require(resume,'Run exists; use --resume or choose a new output directory')
        previous = read_json(run_manifest)
        require(previous['identity'] == frozen_identity,'Code/input/config changed; cannot resume')
        require(previous['runtime'] == current_runtime,'Runtime changed; cannot mix evaluation conditions in one run')
    else:
        require(not any(out.iterdir()),'New output must be empty')
        write_json(run_manifest,{'config':cfg,'identity':frozen_identity,'runtime':current_runtime,
                                 'counts':data['counts'],'epoch_order_sha256':data['epoch_order_sha256'],
                                 'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                                 'initialization':'S adapter, reset optimizer/RNG independently per arm',
                                 'selection_data':'tune only; F repeatedly viewed, exploratory'})
    choices = {arm:train_arm(arm,cfg,data,out,resume) for arm in ARMS}
    # This file freezes all selections before any gate/F predictions are generated.
    write_json(out / 'selected_all.json',choices)
    print('ALL_SELECTIONS_FROZEN',flush=True)


def paired_bootstrap(a,b,draws,seed):
    """Resample whole queries; classification metrics use pooled sampled pairs."""
    import numpy as np
    qa,qb = a['per_query'],b['per_query']
    require([q['query_id'] for q in qa] == [q['query_id'] for q in qb],'Unpaired bootstrap queries')
    ca = np.asarray([q['confusion'] for q in qa],dtype=np.float64)
    cb = np.asarray([q['confusion'] for q in qb],dtype=np.float64)
    na = np.asarray([q['ndcg10'] if q['ndcg10'] is not None else np.nan for q in qa])
    nb = np.asarray([q['ndcg10'] if q['ndcg10'] is not None else np.nan for q in qb])
    def scores(c,n):
        tp = c.diagonal(axis1=-2,axis2=-1)
        f1 = 2*tp/np.maximum(1,c.sum(-1)+c.sum(-2))
        return np.column_stack([tp.sum(-1)/c.sum((-1,-2)),f1.mean(-1),np.nanmean(n,axis=-1),f1])
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0,draws,256):
        idx = rng.integers(0,len(qa),size=(min(256,draws-start),len(qa)))
        samples.append(scores(ca[idx].sum(1),na[idx])-scores(cb[idx].sum(1),nb[idx]))
    values = np.concatenate(samples)
    observed = [a['accuracy']-b['accuracy'],a['macro_f1']-b['macro_f1'],a['ndcg10']-b['ndcg10']]
    observed += [a['per_class'][k]['f1']-b['per_class'][k]['f1'] for k in LABELS]
    names = ['accuracy','macro_f1','ndcg10']+['f1_'+k for k in LABELS]
    return {name:{'difference':float(delta),'ci95':np.quantile(values[:,i],[.025,.975]).tolist()}
            for i,(name,delta) in enumerate(zip(names,observed))}


def final_evaluation(cfg,out):
    import torch
    gpu_required()
    audit(cfg)
    manifest = read_json(out / 'run_manifest.json')
    require(manifest['identity'] == identity(cfg),'Inputs or code changed after training')
    require(manifest['runtime'] == runtime(),'Evaluation runtime differs from training/tune')
    selected = read_json(out / 'selected_all.json')
    require(set(selected) == set(ARMS),'All arm selections must be frozen first')
    for arm,choice in selected.items():
        require(digest(out / choice['checkpoint'] / 'adapter/adapter_model.safetensors') ==
                choice['selected_adapter_sha256'],'Selected adapter hash mismatch')
    final_input = read_jsonl(ROOT / cfg['final_input'])
    gold = read_jsonl(ROOT / cfg['final_gold'])
    require(len(final_input)==len(gold),'F label coverage mismatch')
    require(all(all(a[k]==b[k] for k in ('example_id','query_id','product_id')) for a,b in zip(final_input,gold)),
            'F labels unpaired')
    splits = {'tune':read_jsonl(ROOT / cfg['tune']), 'gate':read_jsonl(ROOT / cfg['gate']),
              'F':[dict(r,label=g['label']) for r,g in zip(final_input,gold)]}
    from modeling import predict
    all_metrics = {name:{} for name in splits}
    result_dir = out / 'evaluation'
    result_dir.mkdir(exist_ok=True)
    frozen_selection_hash = digest(out / 'selected_all.json')
    teacher = read_jsonl(ROOT / cfg['final_teacher'])
    for arm in ('S',)+ARMS:
        checkpoint = ROOT / cfg['initial_checkpoint'] if arm=='S' else out / selected[arm]['checkpoint']
        seed_all(cfg['seed'])
        model = load_model(cfg,checkpoint,False)
        for split,rows in splits.items():
            filename = result_dir / f'{arm}_{split}.jsonl'
            meta = filename.with_suffix('.meta.json')
            stamp = {'selected_all_sha256':frozen_selection_hash,'arm':arm,'split':split,
                     'adapter_sha256':digest(checkpoint / 'adapter/adapter_model.safetensors')}
            if filename.exists() and meta.exists():
                previous = read_json(meta)
                require(all(previous[k]==v for k,v in stamp.items()),'Existing predictions have wrong identity')
                require(previous['predictions_sha256']==digest(filename),'Prediction file modified')
                predictions = read_jsonl(filename)
            else:
                predictions = predict(model,rows,cfg['eval_batch_size'],cfg['max_length'])
                write_jsonl(filename,predictions)
                write_json(meta,dict(stamp,predictions_sha256=digest(filename)))
            require(len(predictions)==len(rows) and all(all(p[k]==r[k] for k in ('example_id','query_id','product_id'))
                    and p['gold']==r['label'] for p,r in zip(predictions,rows)),'Evaluation identity mismatch')
            all_metrics[split][arm] = evaluation_metrics(predictions)
            if arm=='S' and split=='F':
                combined = fuse(predictions,teacher,low_confidence_indices(predictions))
                fused_rows = [dict(p,gold=r['label']) for p,r in zip(combined,rows)]
                teacher_rows = [dict(p,gold=r['label']) for p,r in zip(teacher,rows)]
                all_metrics[split]['S_C50_reference'] = evaluation_metrics(fused_rows)
                all_metrics[split]['T_cached_reference'] = evaluation_metrics(teacher_rows)
                write_jsonl(result_dir / 'S_C50_reference_F.jsonl',fused_rows)
            print('EVALUATED',arm,split,compact_metrics(all_metrics[split][arm]),flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = {}
    for split,systems in all_metrics.items():
        comparisons[split] = {f'FUSION_KD-minus-{other}':paired_bootstrap(
            systems['FUSION_KD'],systems[other],cfg['bootstrap_draws'],cfg['bootstrap_seed'])
            for other in ('CE_CONT','REF_KD','S')}
    report = {'metrics':{split:{arm:compact_metrics(m) for arm,m in systems.items()}
                         for split,systems in all_metrics.items()},
              'paired_query_bootstrap':comparisons,'bootstrap_draws':cfg['bootstrap_draws'],
              'selections':selected,'runtime':runtime(),'F_status':cfg['final_set_status'],
              'limitations':['one training seed; bootstrap does not measure seed variability',
                            'tune metrics are selected and descriptive, not held-out evidence',
                            'CI containing zero means insufficient evidence, not equivalence',
                            'S/C50 reference uses current uniform batch8 runtime; historical numbers are not substituted']}
    write_json(result_dir / 'report.json',report)
    lines = ['# Fusion continuation: measured results','',
             'All students infer independently. F has been repeatedly viewed and is exploratory.','',
             '| Split | Model | Accuracy | Macro-F1 | nDCG@10 |','|---|---|---:|---:|---:|']
    for split,systems in report['metrics'].items():
        for arm,m in systems.items():
            lines.append(f"| {split} | {arm} | {100*m['accuracy']:.2f}% | {100*m['macro_f1']:.2f}% | {m['ndcg10']:.4f} |")
    lines += ['', 'Per-class F1, query-paired intervals and actual training exposures: report.json.',
              'No claim of successful transfer is implied by target preparation or by this report existing.']
    (result_dir / 'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--gpu-check',action='store_true')
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT / 'runs/fusion_continue_20261021')
    args = parser.parse_args()
    cfg = read_json(CONFIG)
    out = check_output_path(args.output)
    if args.preflight or args.gpu_check or not (args.train or args.evaluate):
        preflight(cfg,args.gpu_check)
    if args.train:
        train(cfg,out,args.resume)
    if args.evaluate:
        final_evaluation(cfg,out)
