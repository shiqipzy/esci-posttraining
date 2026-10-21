"""Train three fixed-target continuation arms. No teacher client or network access.

--preflight: input, tokenizer, base identity and synthetic CPU checks only.
--gpu-check: additionally load S and run a small forward pass, without updates.
--train: explicitly start/resume training; --evaluate: evaluate frozen selections.
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
                    encode, fuse, low_confidence_indices, read_json, read_jsonl,
                    require, verify_model, versions, write_json, write_jsonl)
from direct_common import evaluation_metrics
from fusion_continue import (ARMS, CONFIG, audit, best_key, cpu_checks, epoch_order,
                             loss_rows, lr_at, stop_state)


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
    files = ['src/common.py','src/modeling.py','src/direct_common.py','src/fusion_continue.py',
             'src/train_fusion_continue.py','configs/models.lock.json',str(CONFIG.relative_to(ROOT)),
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


if __name__ == '__main__':
    main()
