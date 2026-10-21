# fusion.py：入口、训练循环与统一评测（fusion.py 共 590 行）

> 一句话：统一 CLI 是 `train.py`，本篇讲它分发到的 SFT 与 Fusion 实现。它把“预检 → 三组训练 → 冻结选点 → 统一评测 → 写报告”串成一条线，并把每一步的输入、输出和环境记录成可核对的 JSON。前三篇讲的所有零件，在这里被组装起来。

**读完这一篇，你应该能回答：**

1. 不带参数、`--preflight`、`--gpu-check`、`--train`、`--evaluate` 分别走到哪一行？
2. 一次参数更新里，“32 条”和“4 条”分别在哪两层循环里？尾批 30 条的分母是多少？
3. 一个 `step-0488/` 目录里有哪些文件，分别由谁写？
4. 为什么改一行源码，`--evaluate` 就拒绝在旧输出目录上运行？
5. `--evaluate` 什么时候复用旧的预测文件，什么时候重新推理？

---


## 0. 唯一 CLI 与第一阶段 SFT

`train.py` 的 `main()` 位于第 8–18 行，只做参数分发；它将剩余参数原样交给 `sft.main()` 或 `fusion.main()`。两者的原函数体保留。

```bash
python src/train.py --help
python src/train.py sft --check-only
python src/train.py fusion --help
```

| SFT 函数 | sft.py 行号 | 工作 |
| --- | --- | --- |
| `verify_inputs` | 20–34 | 核对 pilot/tune 行数、查询隔离与 manifest |
| `reset_seed` | 37–44 | seed=20260928 |
| `finish_update` | 61–67 | 梯度裁剪、AdamW、scheduler |
| `train_sft` | 70–127 | 2 epochs；查询内 2 条一组，每次更新 16 groups，micro_groups=1 |
| `main` | 130–161 | 参数与只读检查；默认写 `runs/train_sft/models/S` |

SFT 的 lr=1e-4、warmup=5%、max_length=512。每 epoch 用 tune 选点，以 Macro-F1、nDCG@10、较早 epoch 排序。`modeling.RelevanceModel.forward/save` 和推理路径与第二阶段共用。以下行号均指 `fusion.py`，除非明确标注其他文件。

## 1. 命令行走廊 `main()`（第 574–590 行）

```python
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
```

几个容易误解的地方：

- **没有 `--config`**。配置路径写死在 `fusion.CONFIG`，换配置要改文件。
- **只给 `--train` 不会先预检**：`not (args.train or args.evaluate)` 为 False。预检只在显式 `--preflight`/`--gpu-check` 或**一个动作都不给**时执行（“不给参数就预检”是刻意的默认行为）。
- **`--train --evaluate` 的顺序固定**：先 `train()` 全部完成，再 `final_evaluation()`；不是边训边评。
- **`--output` 默认指向已发布运行的目录** `runs/fusion_continue_20261021`。重构前清单与当前身份不同，不能恢复或重评；新作业明确指定新的 `--output`。同一次新作业中，已有清单时训练需 `--resume`，评测校验后复用预测。
- **`check_output_path()` 在一切之前**：输出不能落在 `data/`、`models/`、`configs/`、`src/`、`docs/`、`results/`、`_archive/` 之内或之上（[01 §8](01-common.md)）。
- `args.output` 是 `Path`，`out` 是解析后的绝对路径，后续所有文件都在 `out / ...` 下。

| 命令 | 实际调用链 | 需要什么 |
| --- | --- | --- |
| `--help` | `argparse` | 无（连 torch 都不需要） |
| 不带参数 | `preflight(cfg, False)` | 数据、配置、torch（`cpu_checks` 需要） |
| `--preflight` | 同上 | 同上 |
| `--gpu-check` | `preflight(cfg, True)` | 再加 GPU、底座、S adapter、transformers/peft |
| `--train` | `train(cfg, out, resume)` | GPU BF16、全部输入；会先 `audit()` |
| `--evaluate` | `final_evaluation(cfg, out)` | GPU、`selected_all.json`、各 checkpoint、S adapter |
| `--resume` | 与 `--train`/`--evaluate` 配合 | 旧运行目录、一致的 identity 与 runtime |

---

## 2. 启动前的环境设置（第 16–19 行）

```python
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
```

这四行必须在 `import torch` **之前**执行（文件里它们就在第 16 行，而 torch 只在函数内部 import），原因各不相同：

| 变量 | 作用 |
| --- | --- |
| `HF_HUB_OFFLINE=1` | 禁止任何 Hugging Face Hub 联网请求；一切只读本地 |
| `TRANSFORMERS_OFFLINE=1` | 同上，针对 transformers 的解析逻辑 |
| `TOKENIZERS_PARALLELISM=false` | 关闭 Rust 分词器的多线程，避免“训练时 fork 出多个线程”的不确定性与警告 |
| `CUBLAS_WORKSPACE_CONFIG=:4096:8` | 让 cuBLAS 使用固定工作区，这是 `torch.use_deterministic_algorithms(True)` 在某些矩阵乘上的前置条件 |

用 `setdefault` 而不是赋值，是为了允许外部环境覆盖（调试时可能想换工作区大小）。

这四行合起来定义一个前提：**这次运行不依赖网络，也不依赖线程时序**。任何“结果略有不同”都会在 `identity` / `runtime` 比对时暴露。

---

## 3. 确定性 `seed_all()` 与运行时快照 `runtime()`（第 228–247 行）

```python
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
```

四个随机源一起设：Python `random`、NumPy、CPU torch、所有 CUDA 设备。然后关掉三类“为了快而牺牲确定性”的开关：

- `allow_tf32 = False`（矩阵乘与 cudnn 各一处）：TF32 会降低精度且不同 GPU 结果可能不同。
- `cudnn.benchmark = False`：benchmark 会按输入形状自动挑选最快的卷积算法，不同次运行可能选到不同算法。
- `use_deterministic_algorithms(True)`：让 PyTorch 在只能用非确定算法时**报错**，而不是悄悄给出不同结果。

```python
def runtime():
    import torch
    return {'versions':versions(), 'python':__import__('sys').version,
            'cuda_available':torch.cuda.is_available(), 'cuda_version':torch.version.cuda,
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'eval_batch_size':8, 'attention':'sdpa', 'backbone_dtype':'bfloat16',
            'loss_dtype':'float32', 'tf32':False, 'deterministic_algorithms':True}
```

`runtime()` 把“这次运行的软硬件条件”压成一个字典。它不只是记录——`train()` 恢复和 `final_evaluation()` 都会要求**当前 `runtime()` 与训练时逐键相等**。少一个包、换个 GPU、甚至 Python 版本变了，都会拒绝继续。

实际记录（[run_manifest.json](../../results/fusion_continue_20261021/run_manifest.json)）：

```
torch 2.8.0+cu128, transformers 4.57.6, peft 0.18.1,
python 3.12.3, cuda 12.8, GPU NVIDIA GeForce RTX 4080 SUPER
```

`gpu_required()`（第 250–253 行）在训练和评测开头各调一次：没有 CUDA 或显卡不支持 BF16 就报错，并写明 `no CPU fallback`。训练循环里 `labels`、`targets` 的张量都直接写在 `'cuda'` 上，所以这不是“建议”，是硬性前提。

---

## 4. 身份记录 `identity()` 与 `run_manifest`

### 4.1 什么算“这次运行的身份”（第 272–277 行）

```python
def identity(cfg):
    files = ['src/train.py','src/sft.py','src/fusion.py','src/evaluation.py',
             'src/common.py','src/modeling.py','configs/models.lock.json',str(CONFIG.relative_to(ROOT)),
             cfg['target_manifest'],cfg['targets'],'models/S/adapter/adapter_model.safetensors',
             'models/S/adapter/adapter_config.json']
    return {p:digest(ROOT / p) for p in files}
```

12 个文件，缺一不可：

| 类别 | 文件 | 作用 |
| --- | --- | --- |
| 代码 | 6 个 `.py` | 改了任何一个，训练逻辑可能变 |
| 模型锁 | `configs/models.lock.json` | 底座版本 |
| 配置 | `configs/fusion_continue_20261021.json` | 所有超参数 |
| 目标 | `targets.jsonl` + `manifest.json` | 固定软监督 |
| 起点 | `models/S/adapter/*` | 三组的共同初始化 |

注意这里**不包含** `data/posttraining/train.jsonl`、`tune.jsonl` 等——它们已经通过 `target_manifest` 间接锁定（`audit()` 会逐文件核对 `files_sha256`）。

### 4.2 写入与比对（第 441–459 行）

```python
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
```

`run_manifest.json` 在**训练开始之前**就写好了。这带来两个后果：

1. 训练中途崩溃，目录里也有 `run_manifest.json`；重跑必须 `--resume`，不能当新目录用。
2. 身份在训练前冻结。训练期间如果有人改了源码，`--resume` 会拒绝，`--evaluate` 也会拒绝。

这是新运行的源码身份约束；旧清单保留为历史记录，不用于恢复重构后的运行：`identity()` 的哈希是“源码字节”的哈希，**加一行注释也会变**。

---

## 5. 加载可训练模型 `load_model()`（第 256–269 行）

```python
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
```

三件事：

1. **构造 `RelevanceModel`**（[02](02-modeling.md)），`source` 是底座目录，`checkpoint` 是 adapter 目录。
2. **把 dropout 全部压成 0**：遍历所有 `torch.nn.Dropout` 模块改 `p`，再把每个 LoRA 配置的 `lora_dropout` 改成 0。adapter 文件里记录的是 `0.05`，这里在加载后覆盖。这样做让三组、以及恢复后的运行完全确定；配置里也写了 `lora_dropout: 0.0`，两处一致。
3. **只允许训练 LoRA 参数**：可训练参数名必须**全部**包含 `lora_`。`all(...)` 保证没有漏网的底座参数被打开；`trainable_names` 非空保证确实有东西可训。

如果哪天 `get_peft_model` 插入的模块命名变了（例如不带 `lora_`），这里会直接失败，而不是悄悄训练了别的参数。

---

## 6. 预检 `preflight()`（第 280–306 行）

```python
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
```

按顺序做的事：

1. `audit()`：整份输入的契约（[03 §8](03-fusion-targets.md)）。
2. `cpu_checks()`：合成损失、梯度累积、早停、学习率的自测（[03 §9](03-fusion-targets.md)）。
3. `verify_model()`：底座文件逐个核对哈希。
4. 用**底座的分词器**把四个划分全部编码一遍，记录每个划分的最长 token 数。这一步能提前发现“某条样本超过 512”——真正的训练会在 `encode()` 里抛错，但预检阶段就报出来更省时间。
5. 再确认四个标签各是一个 token（和 `modeling` 里同一个检查，双重保险）。
6. 汇总成 `PASS_CPU_PREFLIGHT` 并打印。
7. `gpu=True` 时：检查显卡 → 设种子 → 加载 S（不可训练）→ 对训练集前 8 条做推理 → 断言概率全部有限 → 改状态为 `PASS_GPU_PREFLIGHT`。

预检**不更新任何参数**，`optimizer_steps` 与 `teacher_api_calls` 都是 0，明写在结果里。

注意 `max_tokens` 的四个键是 `train/tune/gate/F`，只能告诉你“最长多少”，不能替代训练时的逐条检查。

---

## 7. 存档 `snapshot()` 与恢复 `restore_rng_optimizer()`

### 7.1 写一个存档（第 309–322 行）

```python
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
```

**先写临时目录、再整体重命名**，这是为了原子性：如果写到一半断电，目录里只有 `step-0488.pending-1a2b3c4d/`，不会出现“看起来完整但缺文件”的 `step-0488/`。`uuid4().hex[:8]` 保证并发/重试时临时目录不撞名。`rename` 在同一文件系统内是原子的。

目录内容（`model.save()` 写前四项，`snapshot()` 写后两项）：

| 文件 / 目录 | 内容 | 谁写 |
| --- | --- | --- |
| `adapter/` | `adapter_model.safetensors` + `adapter_config.json` | `RelevanceModel.save()` |
| `tokenizer/` | 分词器文件 | 同上 |
| `run_config.json` | 构造时的整份配置 | 同上 |
| `checkpoint_manifest.json` | `adapter/`、`tokenizer/`、`run_config.json` 的 SHA256 | 同上（写在 `training_state.pt` 之前，所以不包含它） |
| `training_state.pt` | AdamW 状态 + Python/NumPy/torch/CUDA 四个随机数状态 | `snapshot()` |
| `state.json` | 步数、曝光、history、两个文件的哈希 | `snapshot()` |

`training_state.pt` 里保存四个 RNG 状态，是为了恢复后**接下来产生的随机数序列与没中断时一致**。虽然本轮 dropout 为 0、数据顺序由独立种子的 `Random(seed+epoch)` 决定，但 AdamW 之外任何潜在的随机性都要被固定住。

`state.json` 里 `adapter_sha256` 和 `training_state_sha256` 是**恢复时的校验依据**：先比对哈希再加载，避免加载一个被覆盖或截断的文件。

### 7.2 恢复（第 325–338 行）

```python
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
```

- 先读 `state.json` 里的哈希，再对文件算一遍；不一致就拒绝加载。
- `map_location='cpu'`：先加载到 CPU，再让 `optimizer.load_state_dict` 自己搬到对应设备，避免显存峰值。
- `weights_only=False`：这里保存的是优化器状态和 RNG 元组，不是纯张量，必须允许反序列化任意对象。注释 `This is our own locally generated checkpoint` 是在说明安全性依据——只加载本运行自己写的文件，且文件哈希已核对。
- 返回的 `state` 包含 `step`、`exposures`、`history`，`train_arm()` 用它接着往下走。

---

## 8. 训练一组：`train_arm()`（第 350–438 行）

这是全文最长、也最值得逐行读的函数。先看骨架：

```python
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
    ...
```

### 8.1 幂等与恢复（第 352–361 行）

- `seed_all()` 在**每组开始时**调用一次。三组共享同一个种子、同一份初始 adapter，但每组从零开始计步、重置优化器和 RNG——这就是“共同初始化、独立训练”。
- 如果 `selected.json` 已存在：有 `--resume` 就直接读出来返回（这组已完成），没有就报错。
- `snapshots` 用 `sorted()` 找 `step-*` 目录，**必须有 `state.json` 才算有效存档**（`.pending-*` 或写坏的目录被忽略）。
- `initial` 取最新存档；没有存档就从 `models/S` 开始。
- 顺序依赖 `step-0000`、`step-0082`… 的**四位补零**：字符串排序恰好等于数值排序。如果改成 `step-82`，`sorted()` 会把 `step-82` 排在 `step-650` 后面，恢复时选错起点。

### 8.2 优化器与张量（第 363–369 行）

```python
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    features = [encode(r,model.tokenizer,cfg['max_length']) for r in data['train']]
    labels = torch.tensor([LABELS.index(r['label']) for r in data['train']],device='cuda')
    field = 'fusion_probabilities' if arm == 'FUSION_KD' else 'reference_probabilities'
    targets = torch.tensor([[t[field][k] for k in LABELS] for t in data['targets']],
                           dtype=torch.float32,device='cuda')
```

- **优化器只拿可训练参数**：LoRA 的那些矩阵。`AdamW` 的学习率先给配置值，实际每步由 `lr_at()` 覆盖。
- **`features` 一次性编码 10,398 条**，存成 Python 列表（每条是 `{'input_ids': [...], 'attention_mask': [...]}`）。这是 CPU 侧的内存换时间：训练每一步只做索引和 `collate`，不再分词。
- **`labels`**：`[10398]`，int64，cuda；`LABELS.index(...)` 把字母转成 0–3。
- **`targets`**：`[10398, 4]`，float32，cuda。`field` 的选择就是三组唯一的数据差异：FUSION_KD 用融合列，CE_CONT/REF_KD 用原模型列。
- `[[t[field][k] for k in LABELS] for t in data['targets']]` 按固定列序 E、S、C、I 取字典值——**顺序在这里第三次被固定**（第一次是 `LABELS`，第二次是文件里字典的键）。

### 8.3 起点：基线评测与 `step-0000`（第 370–377 行）

```python
    if snapshots:
        state = restore_rng_optimizer(initial,optimizer)
        step,exposures,history = state['step'],state['exposures'],state['history']
    else:
        step,exposures = 0,0
        history = [evaluate_point(model,data['tune'],cfg,0,0,arm_dir)]
        snapshot(model,optimizer,arm_dir / 'step-0000',
                 dict(arm=arm,step=0,exposures=0,history=history,epoch_order_sha256=data['epoch_order_sha256']))
```

- 新训练：先在 tune 上评测一次**未训练的 S adapter**，得到第 0 步基线，再把这个“零步模型”也存成 `step-0000/`。三组的基线 Macro-F1 都是 **0.4744**——完全一致，证明它们确实从同一个起点出发。
- 恢复：从最新存档读回 `step`、`exposures`、`history`。注意存档里的 `step` 是“**已经完成**的更新数”，接下来的循环从 `step` 继续。

`evaluate_point()`（第 341–347 行）很小但关键：

```python
def evaluate_point(model, rows, cfg, step, exposures, directory):
    from modeling import predict
    predictions = predict(model,rows,cfg['eval_batch_size'],cfg['max_length'])
    result = evaluation_metrics(predictions)
    point = dict(step=step,exposures=exposures,epoch=exposures/10398,**compact_metrics(result))
    write_jsonl(directory / f'tune_step_{step:04d}.jsonl',predictions)
    return point
```

它做四件事：推理、算指标、把 5,223 条逐条预测写到 `tune_step_XXXX.jsonl`、返回一个可供 `history` 使用的摘要。`epoch=exposures/10398` 是小数形式（例如 2624/10398 ≈ 0.252），只用来看进度。`**compact_metrics(result)` 把 `per_query` 去掉，但保留 `nll`（`evaluation_metrics` 加的）。

### 8.4 外层循环：把 step 映射到 epoch 与批次（第 378–391 行）

```python
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
```

- **`updates_per_epoch = ceil(10398/32) = 325`**。每个 epoch 有 324 个满批 32 条 + 1 个尾批 30 条。
- **`divmod(step, 325)`**：循环里的 `step` 是“**已完成的更新数**”，所以它取 0–324 时属于 epoch 0，325–649 时属于 epoch 1。换句话说，**第 s 次更新用的是计数器 s−1**；更新完把计数器加 1，再判断是否评测。
- **每个 epoch 重新洗牌一次**：`epoch_order(n, seed, epoch)` 是纯函数，同一步永远得到同一个顺序。循环里每步都调用它，但结果只随 `epoch` 变——重复计算换来“无需保存 order”的简单性。
- **切片用的是字面量 32**，不是 `cfg['effective_batch_size']`。`audit()` 断言后者必须为 32，所以两者一致；如果只改配置不改代码，这里会错位——这是阅读时要留意的硬编码。
- `require(indices, ...)`：如果某步算出的切片为空（例如 `n` 变小），立刻失败。
- `model.train()` 切回训练模式；`optimizer.zero_grad(set_to_none=True)` 把上一步的梯度清成 `None`（比清零更省内存）。

**计数器到 epoch/批次/更新编号的映射**（实测）：

| 循环计数器 `step` | epoch | batch_no | 本批行数 | 对应的更新 |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 32 | 第 1 次更新，`lr_at(1)` |
| 81 | 0 | 81 | 32 | 第 82 次更新，随后评测 `step=82` |
| 324 | 0 | 324 | **30** | 第 325 次更新，epoch 0 尾批，随后评测 `step=325` |
| 325 | 1 | 0 | 32 | 第 326 次更新，epoch 1 开始、重新洗牌 |
| 649 | 1 | 324 | **30** | 第 650 次更新，最后一次，随后评测 `step=650` |
| （650） | （2） | （0） | — | 循环条件已不满足，不会执行 |

**读日志时注意**：日志里的 `step` 是自增之后的值，所以尾批出现在 `step: 325` 和 `step: 650` 两行（`rows: 30`），而不是 `step: 324`。

### 8.5 内层循环：4 条一组，梯度累积（第 389–401 行）

```python
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
```

这是“32 条更新一次”的实现细节，值得逐点拆：

1. **`lr_at(step+1)`**：当前 `step` 还是“已完成的更新数”，所以这次更新是第 `step+1` 次。`lr` 写进优化器的每个参数组。
2. **`indices` 是 32 个样本下标**（尾批 30 个）。内层按 `micro_batch_size=4` 切片：32 → 8 块；30 → 7 块 4 条 + 1 块 2 条。
3. **`collate([features[i] ...])`**：取出这 4 条的编码结果，补成 `[4, width]`（[01 §5.3](01-common.md)）。
4. **`model(batch)`**：`[4, 4]` logits（[02 §3](02-modeling.md)）。
5. **`labels[selected]` / `targets[selected]`**：`[4]` 和 `[4, 4]`，与 logits 行对齐。
6. **`loss_rows(...)`**：返回三个 `[4]` 张量：总损失、CE、KL。
7. **`(loss.sum()/len(indices)).backward()`**：关键的一行。分母是**整个有效批**的长度（32 或 30），不是这一小批的 4/2。这样每小批贡献“这 4 条损失之和 ÷ 32”，8 块加起来恰好等于“32 条的平均损失”。对尾批，分母 30，7 个 4 条块加 1 个 2 条块，总和仍等于“30 条的平均损失”。
8. **`require(torch.isfinite(loss).all())`**：一旦出现 `NaN/Inf` 立即停下，不留到优化器污染参数。
9. **`sums`**：`detach()` 后累加每个张量的和，只用于日志，不参与梯度。最后除以 `len(indices)` 得到本步平均值。

**为什么不用 `loss.mean().backward()`？** 因为每小批大小不同（4 和 2），直接取均值会让最后一块权重偏大。除以固定的 `len(indices)` 才等价于“32 条一起算”。`cpu_checks()` 用 `atol=1e-7` 验证了这一点（[03 §9](03-fusion-targets.md)）。

### 8.6 裁剪、更新与日志（第 402–414 行）

```python
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
```

- **裁剪**：所有参数梯度的全局 L2 范数超过 `max_grad_norm=1.0` 就等比缩放。`error_if_nonfinite=True` 让 `NaN` 在这里报错。`clip_grad_norm_` 返回的是**裁剪前**的范数，正好用来记录。
- **`optimizer.step()`**：真正更新 LoRA 参数。
- **`step += 1`、`exposures += len(indices)`**：步数是“更新次数”，曝光是“喂过多少条样本”。三组各 650 步、20,796 条次（= 2 × 10,398，含尾批）。
- **日志字段**：`loss/ce/kl` 是本批平均值；`soft_coefficient` 对 CE_CONT 记 0；`clip_applied` 是布尔。实际运行里三组 650 步**每一步都是 `true`**：裁剪前梯度范数最小也有 1.25（CE_CONT 2.45、REF_KD 1.25、FUSION_KD 1.27），始终高于上限 1.0。
- 日志文件按启动时间命名，用追加模式写入。**恢复时不会续写旧文件**，而是新建一个 `updates_<新时间戳>.jsonl`；旧文件保留。读日志时要按 `step` 合并，不能假设只有一个文件。

实际第一行与最后一行（`runs/fusion_continue_20261021/CE_CONT/updates_20261021T005053.jsonl`，省略了尾部字段）：

```text
{"step": 1, "epoch": 0, "rows": 32, "exposures": 32, "lr": 6.06e-07, "loss": 0.964, "ce": 0.964, "kl": 0.00059, "soft_coefficient": 0, "grad_norm_before_clip": 5.34, "clip_applied": true, ...}
{"step": 650, "epoch": 1, "rows": 30, "exposures": 20796, "lr": 2e-06, "loss": 0.841, "ce": 0.841, "kl": 0.206, "soft_coefficient": 0, "grad_norm_before_clip": 5.36, "clip_applied": true, ...}
```

注意最后一行 `epoch: 1` 而 `step: 650`：这条记录是在 `step=649` 的循环里产生的（`epoch` 在循环开头算好），更新完成后 `step` 才变成 650。读日志时不要用 `epoch` 反推步数。

### 8.7 评测点与存档（第 415–424 行）

```python
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
```

- 评测步来自配置：0、82、163、244、325、407、488、569、650。步数间隔约 81–82，正好是四分之一的 epoch（325/4 ≈ 81.25）。
- 每次评测：在 tune 上推理 5,223 条 → 写 `tune_step_XXXX.jsonl` → 追加 `history` → 计算早停状态 → **保存一个完整存档**（含优化器与 RNG）→ 打印 `TUNE` 行。
- `history` 会随存档一起保存，所以每个 `state.json` 都带着“到这一步为止的全部评测记录”。`selected.json` 里也有同样一份，这是它有 20 KB 的原因。
- 第 0 步的评测在循环外做（8.3 节），所以 `history` 的第 0 项不是这里产生的。

### 8.8 选点与 `selected.json`（第 425–438 行）

```python
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
```

- `max(history, key=best_key)`：按 Macro-F1 → nDCG@10 → 更早步 选一个。注意 `history` 里包含第 0 步基线，理论上如果训练让成绩变差，会选中“未训练”的 `step-0000`——这是一种诚实的保护。
- `checkpoint` 存的是**相对 `out` 的路径**（如 `FUSION_KD/step-0488`），便于整个目录搬迁。
- `selected_adapter_sha256` 记录权重哈希，评测时会核对。
- 三组实际结果：CE_CONT 569、REF_KD 569、FUSION_KD 488；`stop_reason` 都是 `budget_exhausted`；`stopped_step` 都是 650。
- 内存清理：删掉模型、优化器、两个大张量，回收 Python 与 CUDA 缓存，为下一组腾空间。

---

## 9. 三组调度 `train()`（第 441–463 行）

```python
    choices = {arm:train_arm(arm,cfg,data,out,resume) for arm in ARMS}
    # This file freezes all selections before any gate/F predictions are generated.
    write_json(out / 'selected_all.json',choices)
    print('ALL_SELECTIONS_FROZEN',flush=True)
```

- 字典推导按 `ARMS` 顺序**串行**训练三组。不能并行：显存只有一张卡，而且每组都要从同一份 S adapter 重新开始。
- 三组各自完成后，把三个 `selected` 合并写成一个 `selected_all.json`。注释说明它的意义：**在任何 gate/F 预测产生之前，先冻结三组选点**。这样后续评测不可能反过来影响选点。
- `ALL_SELECTIONS_FROZEN` 是给人和脚本看的里程碑。看到它意味着：训练结束、选点确定、可以进入评测。

`train()` 的第一步是 `gpu_required()`，第二步是 `audit()`，第三步是 `identity()`。任何一项不过，连输出目录都不会创建（`out.mkdir` 在三者之后）。

---

## 10. 统一评测 `final_evaluation()`（第 492–571 行）

### 10.1 前置检查（第 493–508 行）

```python
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
```

六道关：

1. GPU 与 BF16。
2. `audit()`：输入契约。
3. `identity` 与 `runtime` 必须和训练时一致——**改代码、换环境都不行**。
4. `selected_all.json` 必须三组齐全（防止只训完一组就来评测）。
5. 每个选中 adapter 的哈希必须与 `selected.json` 记录的一致。
6. F 的输入与标签必须逐行配对（F 的输入文件里没有 `label`，标签在 `gold.jsonl`）。

### 10.2 三个划分（第 509–517 行）

```python
    splits = {'tune':read_jsonl(ROOT / cfg['tune']), 'gate':read_jsonl(ROOT / cfg['gate']),
              'F':[dict(r,label=g['label']) for r,g in zip(final_input,gold)]}
    from modeling import predict
    all_metrics = {name:{} for name in splits}
    result_dir = out / 'evaluation'
    result_dir.mkdir(exist_ok=True)
    frozen_selection_hash = digest(out / 'selected_all.json')
    teacher = read_jsonl(ROOT / cfg['final_teacher'])
    for arm in ('S',)+ARMS:
```

- `splits['F']` 把 `gold.jsonl` 里的 `label` 拼进输入行，得到与 tune/gate 同构的行（都有 `label`），`predict()` 才能填 `gold`。
- `('S',) + ARMS`：先评测原模型 S，再评测三组学生。S 的结果既是基线，也用于构造两个参考系统。
- `result_dir/evaluation/` 已存在时 `exist_ok=True`，允许重复评测。

### 10.3 每个模型、每个划分：缓存复用（第 517–537 行）

```python
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
```

- 文件命名规则：`<arm>_<split>.jsonl`，配套 `.meta.json`。例如 `FUSION_KD_F.jsonl` + `FUSION_KD_F.meta.json`。
- **复用条件有四层**：文件存在、meta 存在、meta 里的三个 stamp 与当前一致、文件哈希与 meta 记录一致。任何一层不过都会走“重新推理”。
- `stamp` 包含 `selected_all_sha256`：只要选点文件变了，旧预测全部作废。
- 复用后仍要逐行核对身份与 `gold`，防止“哈希对但语义被换”（例如 meta 被伪造）。这是“先验哈希、再验内容”的双重检查。
- 即便所有预测都能复用，模型**仍然会被加载**（`load_model` 在循环前）。所以 `--evaluate` 总是需要 GPU 和 checkpoint，不能纯离线算指标。

实际目录里能看到这些文件（`runs/fusion_continue_20261021/evaluation/`）：`S_tune.jsonl`、`S_gate.jsonl`、`S_F.jsonl`、三组各自的三个文件，以及 `S_C50_reference_F.jsonl`。

### 10.4 两个参考系统（第 538–544 行）

```python
            if arm=='S' and split=='F':
                combined = fuse(predictions,teacher,low_confidence_indices(predictions))
                fused_rows = [dict(p,gold=r['label']) for p,r in zip(combined,rows)]
                teacher_rows = [dict(p,gold=r['label']) for p,r in zip(teacher,rows)]
                all_metrics[split]['S_C50_reference'] = evaluation_metrics(fused_rows)
                all_metrics[split]['T_cached_reference'] = evaluation_metrics(teacher_rows)
                write_jsonl(result_dir / 'S_C50_reference_F.jsonl',fused_rows)
```

只在 **S + F** 这一格触发，产出两个“参照系统”：

- **`S_C50_reference`**：对 S 在 F 上的预测，用 `low_confidence_indices()` 路由一半、`fuse()` 与教师缓存等权平均——就是训练目标用的同一套规则。它回答“如果不蒸馏，直接在线融合能拿多少”。
- **`T_cached_reference`**：教师缓存的原始预测。它回答“教师自己单独有多好”。

`S_C50` 用当前环境、当前 S 的预测重算；`T` 的概率直接来自冻结缓存，但它的**指标**仍是在当前划分、当前代码上重新汇总的。两者都不引用历史汇总数字——这正是报告里那句 `S/C50 reference uses current uniform batch8 runtime; historical numbers are not substituted` 的来源。

注意 `teacher` 行里可能有 `probabilities=None`（弃权）。`fuse()` 会回退到 S 的概率；而直接把 teacher 行交给 `evaluation_metrics()` 时，`prediction is None` 会触发弃权分支，混淆矩阵变成 **5×5**（第 5 列是“弃权”）。F 的教师缓存里有 4 条弃权，所以 `T_cached_reference` 的混淆矩阵是 5×5，而其他系统都是 4×4。详见 [05 evaluation](05-evaluation.md) 第 2 节。

### 10.5 汇总报告（第 549–571 行）

```python
    comparisons = {}
    for split,systems in all_metrics.items():
        comparisons[split] = {f'FUSION_KD-minus-{other}':paired_bootstrap(
            systems['FUSION_KD'],systems[other],cfg['bootstrap_draws'],cfg['bootstrap_seed'])
            for other in ('CE_CONT','REF_KD','S')}
```

三个划分各做三组比较（FUSION_KD 减 CE_CONT、REF_KD、S），共 9 组 Bootstrap。只比较 FUSION_KD，是因为它是本轮的“主方法”。

```python
    report = {'metrics':{split:{arm:compact_metrics(m) for arm,m in systems.items()}
                         for split,systems in all_metrics.items()},
              'paired_query_bootstrap':comparisons,'bootstrap_draws':cfg['bootstrap_draws'],
              'selections':selected,'runtime':runtime(),'F_status':cfg['final_set_status'],
              'limitations':['one training seed; bootstrap does not measure seed variability',
                            'tune metrics are selected and descriptive, not held-out evidence',
                            'CI containing zero means insufficient evidence, not equivalence',
                            'S/C50 reference uses current uniform batch8 runtime; historical numbers are not substituted']}
    write_json(result_dir / 'report.json',report)
```

四段结构：

| 键 | 内容 |
| --- | --- |
| `metrics[split][arm]` | 每个系统在每个划分的全部汇总指标（含 per-class、confusion、nll；不含 per_query） |
| `paired_query_bootstrap[split][比较名]` | 7 个指标的差值与 95% 区间 |
| `selections` | 三组选点（从 `selected_all.json` 原样带入） |
| `runtime` / `F_status` / `limitations` | 环境、F 的使用状态、四条限制 |

`limitations` 不是套话，它对应四个具体的解释边界：单种子、tune 是选点数据、区间含零不等于等价、参考系统是本次重算的。

最后生成 `RESULT.md`：

```python
    lines = ['# Fusion continuation: measured results','',
             'All students infer independently. F has been repeatedly viewed and is exploratory.','',
             '| Split | Model | Accuracy | Macro-F1 | nDCG@10 |','|---|---|---:|---:|---:|']
    for split,systems in report['metrics'].items():
        for arm,m in systems.items():
            lines.append(f"| {split} | {arm} | {100*m['accuracy']:.2f}% | {100*m['macro_f1']:.2f}% | {m['ndcg10']:.4f} |")
```

自动生成的是英文表格；公开仓库里那份中文 [RESULT.md](../../results/fusion_continue_20261021/RESULT.md) 是另行整理的，数字一致但排版不同。**以 `report.json` 为准**，Markdown 只是阅读层。

---

## 11. 按查询配对的 Bootstrap（第 466–489 行）

```python
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
```

### 11.1 为什么按查询重采样

同一个查询下的商品彼此相关（同一份查询意图、同一个类目）。如果按“商品对”独立重采样，会低估方差、把区间算得过窄。正确做法是把**整个查询**当作一个抽样单位：抽到某查询，就把它下面的所有商品一起带上。

`rng.integers(0, len(qa), size=(draws_per_chunk, len(qa)))` 生成 `[256, Q]` 的下标矩阵（`Q` 是该划分的查询数：tune 是 256，gate 与 F 是 512），每行是一次“有放回抽 Q 个查询”。抽到重复查询时，`ca[idx].sum(1)` 会把它们的混淆矩阵**相加**——所以分类指标是“在重采样后的商品对上合并计算”，而不是“每个查询先算 F1 再平均”。

### 11.2 `scores()` 的七个输出

`scores(c, n)` 输入一个“重采样后的数据集”：`c` 是 `[..., 4, 4]` 混淆矩阵，`n` 是 `[..., Q]` 的 nDCG 数组。返回 `[..., 7]`：

1. Accuracy：对角线之和 / 总数；
2. Macro-F1：四个类 F1 的平均；
3. nDCG@10：`np.nanmean`，忽略 `None` 转成的 `NaN`；
4. `f1_E`、5. `f1_S`、6. `f1_C`、7. `f1_I`。

`tp = c.diagonal(axis1=-2,axis2=-1)` 取每个混淆矩阵的对角线；`f1 = 2·tp / (行和 + 列和)`。`np.maximum(1, ...)` 防止全零。

### 11.3 与观测值配对

```python
    observed = [a['accuracy']-b['accuracy'], ...]
    return {name:{'difference':float(delta),'ci95':...}}
```

`difference` 用的是**原始数据上的真实差值**，不是 Bootstrap 分布的均值；`ci95` 用 2.5% 和 97.5% 分位数。两者放在一起，读者可以判断“差值是否落在自己的区间内”。

实测（F 划分，FUSION_KD − CE_CONT）：Macro-F1 差值 +0.0091，区间 `[0.0011, 0.0174]`，不包含 0；Accuracy 差值 −0.0015，区间 `[-0.0092, 0.0061]`，包含 0。报告里的 +0.91pp 就是前者乘 100。

---

## 12. 一次运行会写出哪些文件

把 `--train --evaluate --output runs/fusion_new` 的产物按时间顺序排一遍（`fusion_new` 是示例名）：

| 时刻 | 文件 | 写入者 |
| --- | --- | --- |
| 训练前 | `runs/fusion_new/run_manifest.json` | `train()` |
| 三组开始 | `<arm>/updates_<时间戳>.jsonl` | `train_arm()`（每步一行） |
| 第 0 步 | `<arm>/tune_step_0000.jsonl`、`<arm>/step-0000/` | `evaluate_point()`、`snapshot()` |
| 每个评测步 | `<arm>/tune_step_XXXX.jsonl`、`<arm>/step-XXXX/` | 同上 |
| 每组结束 | `<arm>/selected.json` | `train_arm()` |
| 三组结束 | `selected_all.json` | `train()` |
| 评测 | `evaluation/<arm>_<split>.jsonl`、`.meta.json` | `final_evaluation()` |
| 评测 | `evaluation/S_C50_reference_F.jsonl` | 同上 |
| 评测结束 | `evaluation/report.json`、`evaluation/RESULT.md` | 同上 |

本地 `runs/fusion_continue_20261021/` 保留了日志与逐条预测（`updates_*.jsonl` 各 650 行、约 200 KB；`evaluation/*.jsonl` 各 2–5 MB），但 checkpoint 目录（`step-XXXX/`）已不在本地——所以在统计结果目录时，`selected_all.json` 里记录的路径可能指向不存在的权重。这不是 bug，见[目录与复现说明 §5.2](../目录与复现说明.md)。

---

## 13. 坑与细节

| 现象 | 解释 |
| --- | --- |
| `--help` 不需要 torch | 顶层不 import modeling/torch；真正加载时才 import |
| 只给 `--train` 不预检 | `main()` 的条件是“未给 train 且未给 evaluate”才预检 |
| 源码加注释后 `--evaluate` 报 `Inputs or code changed` | `identity()` 哈希 6 个 `.py` |
| 换机器后 `--evaluate` 报 `Evaluation runtime differs` | `runtime()` 逐键比较，包括 Python 版本与 GPU 名 |
| 恢复时选错存档 | `sorted()` 依赖 `step-XXXX` 的四位补零 |
| 日志 `step: 650` 但 `epoch: 1` | `epoch` 在循环开头计算，`step` 在更新后自增 |
| 尾批日志 `rows: 30` | `10398 = 324×32 + 30` |
| `FUSION_KD_F.jsonl` 没重新推理 | meta 里的四个 stamp 全对上时复用 |
| `T_cached_reference` 的混淆矩阵是 5×5 | 教师有 4 条弃权，第 5 列是弃权 |
| `report.json` 里没有 `per_query` | `compact_metrics()` 去掉；Bootstrap 用的是内存结果 |

---

## 14. 自测

1. 如果 `--output` 指向一个已有 `run_manifest.json` 但内容不同的目录，哪一行会报错、报什么？
2. 第 325 次与第 326 次更新分别用哪个 epoch 的数据？尾批的分母是多少？
3. 为什么 `snapshot()` 要先写 `.pending-*` 再重命名？如果直接写 `step-0488/` 会有什么风险？
4. `--evaluate` 时如果 `FUSION_KD_F.jsonl` 存在但 `FUSION_KD_F.meta.json` 被删了，会发生什么？
5. `paired_bootstrap()` 为什么不能用 `report.json` 里的指标重算区间？

参考思路：1 第 451 行 `previous['identity'] == frozen_identity`，报 `Code/input/config changed; cannot resume`；2 第 325 次用 epoch 0 尾批（计数器 324），第 326 次用 epoch 1 第一批（计数器 325），尾批分母 30；3 避免断电留下“看似完整”的存档，重命名原子；4 走重新推理分支（`meta` 不存在），覆盖写；5 报告里已删 `per_query`，没有查询粒度的混淆矩阵与 nDCG。

---

## 15. 验证

```powershell
# 1) 命令行（不需要 torch）
python src/train.py fusion --help

# 2) 用已发布结果复算选点（与 selected_all.json 对照）
python -c "import json; s=json.load(open('results/fusion_continue_20261021/selected_all.json',encoding='utf-8')); print({k:v['selected_step'] for k,v in s.items()})"

# 3) 显示当前源码身份，不与历史清单比较
python -c "import sys; sys.path.insert(0,'src'); from common import digest; print(digest('src/fusion.py'))"

# 4) 数一次更新的日志行数（应为 650）
python -c "import glob; p=glob.glob('runs/fusion_continue_20261021/*/updates_*.jsonl'); print([(x, sum(1 for _ in open(x,encoding='utf-8'))) for x in p])"
```

预期：第 2 条输出 `{'CE_CONT': 569, 'REF_KD': 569, 'FUSION_KD': 488}`；第 3 条显示当前 `fusion.py` 的 SHA256；第 4 条三个文件都是 650 行。

下一站：[05 evaluation](05-evaluation.md) —— 评测指标与隔离检查的来源；最后是 [06 数字推演](06-一次更新的数字推演.md)，把整条时间线用真实数值走一遍。
