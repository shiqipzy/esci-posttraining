# common.py：数据契约、编码、指标与固定融合（249 行）

> 一句话：这个文件不碰神经网络。它规定“什么算合法数据”、把文本变成编号、把预测变成成绩、把两份概率融成一份。六个文件里只有它没有项目内依赖，是整条链路的地基。

**读完这一篇，你应该能回答：**

1. 四个标签的顺序为什么必须固定，改了会怎样？
2. 一条 JSON 记录经过 `payload()` / `encode()` / `collate()` 之后，形状和类型分别变成什么？
3. Macro-F1 和 nDCG@10 分别是怎样从预测文件算出来的？
4. 路由为什么只取一半，并列时靠什么决定顺序？

---

## 1. 它被谁调用

| 调用方                                                        | 用到的名字                                                                                                                                               | 时机                   |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- |
| [modeling.py](../../src/modeling.py)                           | `GAINS` `LABELS` `collate` `digest` `encode` `verify_model` `write_json`                                                                   | 每次加载模型、每次推理 |
| [sft.py](../../src/sft.py) | `load_split` `epoch_groups` `encode` `collate` `metrics` 等 | 第一阶段输入核验、分组 SFT 和 tune 选点 |
| [evaluation.py](../../src/evaluation.py) | `LABELS` `metrics` `require` | 指标和隔离检查 |
| [fusion.py](../../src/fusion.py) | 几乎所有名字                                                                                                                                             | 整个训练与评测流程     |

`common` 不 import 其他项目文件，可以在只有标准库的机器上导入。`fusion.audit()` 和 `sft --check-only` 也不依赖 torch；完整的 `python src/train.py fusion --preflight` 还会执行合成张量与 tokenizer 检查，需要 torch 和 transformers。

---

## 2. 模块级常量与数据契约（第 11–22 行）

```python
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs'
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
LABELS = ['E', 'S', 'C', 'I']
GAINS = [1.0, 0.1, 0.01, 0.0]
SYSTEM = '''Classify a product for a shopping query using the product title and brand.
E = Exact: the product satisfies the query intent.
S = Substitute: the product is not an exact match but can serve as an alternative.
C = Complement: an accessory or item used together with the requested product, not the requested product itself.
I = Irrelevant: neither an exact match, a substitute, nor a complement.
Return exactly one letter: E, S, C, or I.'''
```

**`ROOT` 用 `__file__` 反推，而不是用当前工作目录。** `Path(__file__).resolve()` 得到 `src/common.py` 的绝对路径，`parents[1]` 是它的上上层，也就是项目根 `ESCI/`。所以从任何目录执行脚本，`ROOT / 'data/...'` 都指向同一个文件。这也意味着**整个项目只有这一处定义“项目根在哪里”**，其余文件都复用它。

| 常量       | 含义                                       | 谁在乎它                                                               |
| ---------- | ------------------------------------------ | ---------------------------------------------------------------------- |
| `LABELS` | 类别顺序 E、S、C、I                        | 模型四列、概率字典、混淆矩阵的行列、`targets` 张量的列，全按这个顺序 |
| `GAINS`  | 排序增益：E=1.0、S=0.1、C=0.01、I=0        | `metrics()` 算 nDCG；`predict()` 算 `score`                      |
| `SYSTEM` | 固定系统提示，告诉模型四类的定义与输出格式 | `encode()` 每次把它放在第一条消息                                    |

`LABELS` 的顺序是本项目最硬的约定：一旦调换，`targets.jsonl` 里的四列概率、已训练的 adapter、已发布的混淆矩阵全部失效。代码里没有任何地方对“顺序”再做一次命名映射，全靠这个列表。

`ARMS` 只在 `fusion.py` 中定义；`common.py` 保留两阶段共用的数据和指标函数。

---

## 3. 文件与哈希工具（第 25–67 行）

### 3.1 `require()`（第 25–27 行）

```python
def require(condition, message):
    if not condition:
        raise ValueError(message)
```

整个项目的断言都用它，而不是 `assert`。原因是 `python -O` 会删除 `assert`，而这些检查是数据契约的一部分，不能在优化模式下悄悄消失。代价是报错信息只有一个字符串，没有堆栈指向具体字段；后面的 `audit()` 靠把字段名写进消息来补偿。

### 3.2 四个读写函数（第 30–48 行）

```python
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
```

四个函数合起来定义了本项目的“数据外观”：

- **`ensure_ascii=False`**：中文、西文重音字符按原字符写入，不转成 `\u4e2d`。数据里有西班牙语查询（例如 `cochecito de niño`），保持可读。
- **`write_jsonl` 用紧凑分隔符**：`separators=(',', ':')` 去掉逗号和冒号后的空格；`targets.jsonl` 有 10,398 行，紧凑写法能省下可观体积。**读的时候不能假设有空格**，所以一律用 `json.loads`，不要自己 `split(',')`。
- **`newline='\n'`**：在 Windows 上强制 LF，避免同一份内容在不同系统算出不同的 SHA256。这一点是这个项目能跨机器核对哈希的前提。
- **`read_jsonl` 逐行迭代文件对象**：不要换成 `path.read_text().splitlines()`。`str.splitlines()` 会在 `\u2028`、`\x0b` 等字符处也断行，而商品标题里出现这类字符并不是不可能。这不是假想：`data/sft/gate.jsonl` 里就有 **4 个 U+2028**（LINE SEPARATOR），`splitlines()` 会把它读成 10,442 行，而文件对象逐行读是 10,438 行，多出来的 4 行全是半截 JSON。

### 3.3 `digest()`（第 51–56 行）

```python
def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()
```

按 1 MiB 分块读，不把 1.5 GB 的 `model.safetensors` 整个读进内存。`iter(callable, sentinel)` 是 Python 的“反复调用直到返回哨兵值”写法：每次 `f.read(1MB)`，读到 `b''` 停止。返回值是 64 位十六进制字符串。

这个函数是项目里出现频率最高的名字：模型身份、数据身份、目标文件、预测文件、checkpoint 里的每个文件，全部靠它核对。

### 3.4 `stable_key()`（第 59–60 行）

```python
def stable_key(value, salt='esci-v1-20260928'):
    return hashlib.sha256(f'{salt}|{value}'.encode()).hexdigest()
```

把任意值映射成一个稳定的十六进制串，用于**并列时的确定性排序**。它不随进程、机器、Python 版本变化（`hash()` 会，且每个进程的字符串哈希种子不同）。`salt` 让不同实验可以用不同的打散方式：默认盐是 `esci-v1-20260928`，而路由用的是 `esci-cascade-v13`（下一节会看到）。

### 3.5 `fresh_dir()`（第 63–67 行）

```python
def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f'Refuse nonempty output: {path}. Preserve previous runs.')
    path.mkdir(parents=True, exist_ok=True)
```

只允许“不存在”或“存在但为空”的目录。`prepare()` 用它保护目标目录，`fresh_dir` 的报错信息直说“保留之前的运行”。注意它只检查目录非空，不检查里面是什么；这是有意的：在本项目里，输出目录非空就意味着“这里已经有一次运行”。

### 3.6 `versions()`（第 70–77 行）

```python
def versions():
    result = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'numpy', 'pandas', 'pyarrow', 'safetensors']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result
```

记录版本；没装的包写 `None`（实际运行里 `pandas`、`pyarrow` 就是 `None`）。这份字典会进入 `runtime()`，而 `runtime()` 必须在训练和评测时完全一致，否则评测拒绝执行——所以这几个键的顺序和内容也是身份的一部分。

---

## 4. 模型与数据的身份核验（第 70–110 行）

### 4.1 `verify_model()`（第 80–95 行）

```python
@lru_cache(maxsize=8)
def verify_model(path):
    lock = read_json(CONFIGS / 'models.lock.json')
    for record in lock['files']:
        p = Path(path) / record['rfilename']
        if not p.is_file() or p.stat().st_size != record['size']:
            raise ValueError(f'Missing/wrong-size official file: {p}')
        if record.get('lfs'):
            actual, expected = digest(p), record['lfs']['sha256']
        else:
            data = p.read_bytes()
            actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
            expected = record['blobId']
        if actual != expected:
            raise ValueError(f'Pinned model hash mismatch: {p}')
    return lock
```

它不只知道“文件在不在”，还知道“是不是 Hugging Face 上的那一份”。[models.lock.json](../../configs/models.lock.json) 记录了每个文件的 `size` 和两种哈希之一：

- 大权重走 Git LFS，记录的是文件内容的 **SHA256**（`lfs.sha256`）。
- 小配置走普通 Git 对象，记录的是 Git 的 **blob SHA1**。Git 的 blob 哈希不是对文件内容直接哈希，而是对 `b'blob ' + 长度 + b'\0' + 内容` 哈希——代码里手动拼出了这个前缀。

`@lru_cache(maxsize=8)` 让同一路径只校验一次：训练时每加载一个模型都会调用它，而底座有 1.5 GB，重复校验会很慢。

实际输出里，这份锁对应 Qwen3-0.6B 的某个固定 revision。**一旦底座被替换成另外一份权重（哪怕同名、同大小），训练与评测都会在这里失败。**

### 4.2 `load_split()`（第 98–110 行）

```python
def load_split(data_dir, split):
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / 'manifest.json')
    if manifest['input_contract_sha256'] != hashlib.sha256(SYSTEM.encode()).hexdigest():
        raise ValueError('Input contract changed after data freeze')
    filename = split + '.jsonl'
    meta = manifest['files'][filename]
    if digest(data_dir / filename) != meta['sha256']:
        raise ValueError(f'Modified frozen data: {filename}')
    rows = read_jsonl(data_dir / filename)
    if len(rows) != meta['pairs'] or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Example count/uniqueness mismatch')
    return rows, manifest
```

这个函数**本轮没有调用**，但它是理解“数据冻结”的钥匙，值得看懂：

1. 它把 `SYSTEM` 提示的 SHA256 和 `manifest['input_contract_sha256']` 比对。也就是说，**改系统提示会让所有冻结数据立刻不合法**——因为当年的文本输入是按那条提示编码的。这是把“提示词”当成数据契约来管理的写法。
2. 每个 split 文件的哈希、条数、`example_id` 唯一性三重检查。
3. 返回 `(rows, manifest)`。

第一阶段的 `sft.verify_inputs()` 用 `load_split()` 读取 pilot_train 和 tune，并核对 `data/sft/manifest.json`。第二阶段由 `fusion.audit()` 核对 `data/fusion_continue_20261021/manifest.json` 及其中固定的各输入哈希，见[03 目标与审计](03-fusion-targets.md)。

---

## 5. 从一行 JSON 到模型输入（第 113–140 行）

### 5.1 `payload()`（第 113–115 行）

```python
def payload(row):
    # No IDs, labels, source splits, or diagnostics enter the model input.
    return json.dumps({k: row[k] for k in ['query', 'title', 'brand']}, ensure_ascii=False)
```

三个字段、一个 JSON 字符串。注释说明了两件事：

- **只用 `query` / `title` / `brand`**。`example_id`、`query_id`、`product_id`、`label` 都不进模型输入。这防止模型通过 ID 记住样本，也防止标签泄漏。
- **字段顺序固定**：字典按列表顺序插入，`json.dumps` 保持插入顺序，所以输出永远是 `{"query": ..., "title": ..., "brand": ...}`。哪怕只是换个顺序，发给模型的字符串就变了，历史缓存概率也就不再对应。

实际数据里还有 `title_clipped`、`brand_clipped`、`source_title_chars` 等字段，它们是**截断诊断**：只有 `title`/`brand` 进入模型，但 `predict()` 会把这两个布尔值复制到预测行里，方便事后检查。别误以为它们参与了输入。

`ensure_ascii=False` 同样是为了中文和重音字符原样出现。

### 5.2 `encode()`（第 118–129 行）

```python
def encode(row, tokenizer, max_length=512, demonstrations=()):
    messages = [{'role': 'system', 'content': SYSTEM}]
    for demo in demonstrations:
        messages += [{'role': 'user', 'content': payload(demo)},
                     {'role': 'assistant', 'content': demo['label']}]
    messages.append({'role': 'user', 'content': payload(row)})
    ids = tokenizer.apply_chat_template(messages, tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
    features = {'input_ids': ids, 'attention_mask': [1] * len(ids)}
    if len(features['input_ids']) > max_length:
        raise ValueError(f"Overlength {row['example_id']}: {len(features['input_ids'])}>{max_length}; no silent truncation")
    return features
```

逐句拆：

**第 119 行**：第一条永远是 system 消息，内容就是 `SYSTEM`。这是“提示即契约”的另一半。

**第 120–122 行**：可选的 few-shot 示范，每个示范是一条 user（JSON 输入）+ 一条 assistant（答案字母）。本轮所有调用都不传 `demonstrations`，所以这段循环一次也不执行。保留它是因为历史实验用过示范。

**第 123 行**：把当前样本作为最后一条 user 消息。

**第 124–125 行**：`apply_chat_template` 把这些消息渲染成 Qwen 的对话格式（`<|im_start|>role\n...<|im_end|>`），`tokenize=True` 直接得到编号列表，`add_generation_prompt=True` 在末尾加上 `<|im_start|>assistant\n`，让模型处于“该回答下一个 token”的位置。`enable_thinking=False` 是 Qwen3 特有的开关，关掉 ` thinking` 推理段，保证下一个 token 直接是类别字母。

这三件事合起来解释了一个关键设计：**模型不是“生成一句话再解析”，而是让“下一个 token” 直接成为 E/S/C/I 之一。** 训练时看最后一个有效位置的隐藏表示，评测时对四个标签 token 的分数做 softmax。

**第 126 行**：`attention_mask` 全 1。这里没有 padding——padding 发生在 `collate()`，而且是在**右边**补（`RelevanceModel` 里设置了 `padding_side = 'right'`）。这保证了“最后一个有效位置”永远是真正的最后一条消息的末尾。

**第 127–128 行**：超长直接报错，不截断。注释写着 `no silent truncation`：一旦静默截断，样本的真实输入就变了，而参考概率是按完整输入算的，软监督会悄悄错位。所以宁可让整次运行失败。实际数据里 F 上的输入长度是 131–245 个 token，512 的上限很宽松，正常情况下不会触发。

**返回值**：`{'input_ids': [L], 'attention_mask': [L]}`，两个等长的 Python 列表，长度 L 因样本而异（131–245）。

### 5.3 `collate()`（第 132–140 行）

```python
def collate(features, pad_id, device):
    import torch
    width = max(len(x['input_ids']) for x in features)
    out = {}
    for key in features[0]:
        pad = pad_id if key == 'input_ids' else 0
        out[key] = torch.tensor([f[key] + [pad] * (width - len(f[key])) for f in features],
                                dtype=torch.long, device=device)
    return out
```

把若干条变长序列拼成一个矩形张量：

- `width` 是本批最长的那条。
- `input_ids` 用 `pad_id`（Qwen 里是 eos token 的 id）补齐；`attention_mask` 补 0。
- 每个样本 `f[key] + [pad] * (width - len(f[key]))` 是在**右边**追加，和 `padding_side='right'` 一致。
- `torch.long` 是索引类型；`device` 直接决定张量放在 CPU 还是 GPU。

形状：一批 4 条 → `input_ids: [4, width]`，`attention_mask: [4, width]`。例如宽 180 时就是 `[4, 180]`。

**这里不做任何截断、不丢样本**：宽度的上界由 `encode()` 的 512 检查保证。

### 5.4 `epoch_groups()`（第 143–155 行）

```python
def epoch_groups(rows, seed, epoch, group_size=2):
    """Label-independent query-local groups. Every row appears exactly once per epoch."""
    by_query = collections.defaultdict(list)
    for i, row in enumerate(rows):
        by_query[row['query_id']].append(i)
    rng = random.Random(seed + epoch)
    groups = []
    for query_id in sorted(by_query):
        indices = by_query[query_id][:]
        rng.shuffle(indices)
        groups.extend(indices[i:i + group_size] for i in range(0, len(indices), group_size))
    rng.shuffle(groups)
    return groups
```

`epoch_groups()` 用于第一阶段 SFT：先在查询内部打散并按 group_size=2 切组，再打散所有组；每次更新最多取 16 groups，每次前向处理 1 group。第二阶段使用 `fusion.epoch_order()` 对全部训练行洗牌，两者均以 seed+epoch 生成确定顺序。

---

## 6. 指标 `metrics()`（第 158–206 行）

这是全文最长、也最容易被误读的函数。它一次性算出分类指标（Accuracy、Macro-F1、每类 P/R/F1、混淆矩阵）和排序指标（nDCG@10、nDCG_full、随机基线、exact@1）。

### 6.1 输入契约（第 158–165 行）

```python
def metrics(rows):
    if not rows or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate predictions')
    by_query = collections.defaultdict(list)
    conf = [[0] * 4 for _ in range(4)]
    has_classes = all(r.get('prediction') in LABELS for r in rows)
    if not has_classes and any(r.get('prediction') is not None for r in rows):
        raise ValueError('Invalid or partial classification output')
```

- 每行必须有 `example_id`、`query_id`、`gold`、`score`，且 `example_id` 不重复。
- `has_classes` 的含义是“**每一行**都有合法类别预测”。注意 `all(r.get('prediction') in LABELS ...)` 用 `get`，所以 `prediction=None` 或缺失都算 False。
- 如果只有部分行有预测、部分是 `None`，直接报错。这是有意设计的：一个划分要么整体分类，要么整体弃权；混着算会让 Accuracy 的分母解释不清。
- 弃权路径（教师缓存里出现 `probabilities=None`）会走 `evaluation.evaluation_metrics()` 的另一条分支，见 [05](05-evaluation.md)。

### 6.2 计数与混淆矩阵（第 166–171 行）

```python
    for row in rows:
        if not math.isfinite(row['score']):
            raise ValueError('Nonfinite ranking score')
        by_query[row['query_id']].append(row)
        if has_classes:
            conf[LABELS.index(row['gold'])][LABELS.index(row['prediction'])] += 1
```

`conf[真实类别][预测类别]`，行是 gold、列是 prediction。`score` 必须有限：`NaN` 会让后面的排序出现不可预测的位置。

### 6.3 逐查询的排序指标（第 172–195 行）

```python
    per_query = []
    for qid, rs in sorted(by_query.items()):
        # Scores tie by product_id/example_id, never by reference label/file order.
        ranked = sorted(rs, key=lambda r: (-r['score'], r['product_id'], r['example_id']))
        gold = [GAINS[LABELS.index(r['gold'])] for r in rs]
        entry = {'query_id': qid, 'pairs': len(rs), 'confusion': [[0] * 4 for _ in range(4)]}
        for k, suffix in [(min(10, len(rs)), '10'), (len(rs), 'full')]:
            d = [1 / math.log2(i + 2) for i in range(k)]
            ideal = sum(g * w for g, w in zip(sorted(gold, reverse=True)[:k], d))
            entry['ndcg' + suffix] = (sum(GAINS[LABELS.index(r['gold'])] * w
                                        for r, w in zip(ranked[:k], d)) / ideal if ideal else None)
            entry['random_ndcg' + suffix] = sum(gold) / len(gold) * sum(d) / ideal if ideal else None
        entry['exact_at_1'] = float(ranked[0]['gold'] == 'E')
        if has_classes:
            for r in rs:
                entry['confusion'][LABELS.index(r['gold'])][LABELS.index(r['prediction'])] += 1
        per_query.append(entry)
```

要点逐条：

**排序键 `(-score, product_id, example_id)`**：分数高的在前；**同分时按 `product_id`、再按 `example_id`**，而不是按文件顺序或标签。注释特意写了 `never by reference label/file order`——如果用标签打破并列，等于偷看答案；如果用文件顺序，换一次数据排列结果就变了。

**增益 `gold`**：把真实类别映射成 `GAINS` 里的数。一条 “E” 商品算 1 分，“S” 算 0.1，“C” 算 0.01，“I” 算 0。

**折扣 `d`**：第 i 位（从 0 开始）的折扣是 `1/log2(i+2)`。i=0 → `1/log2 2 = 1`；i=1 → `1/log2 3 ≈ 0.6309`；i=2 → 0.5；i=3 → 0.4307……

**理想 DCG `ideal`**：把同一查询的增益从大到小排好，乘以同样的折扣。它表示“这个查询在完美排序下的最好成绩”。如果查询里没有任何 E/S/C（全是 I），`ideal = 0`，此时 nDCG 无法定义，代码写 `None`，后面的汇总会跳过这些查询并单独计数（`zero_idcg_queries`）。

**两个 k**：`min(10, len(rs))` 对应 nDCG@10，`len(rs)` 对应 nDCG_full（整段排序）。查询只有 8 条商品时，两者相同。

**随机基线 `random_ndcg`**：`平均增益 × 折扣之和 ÷ 理想值`。它回答“随便排能拿多少分”。这个值只取决于该查询的增益构成：F 上整体是 **0.5843**；下面那个 8 条商品的查询里有 7 条 E，它的随机基线就高达 0.9514。所以“0.77 的 nDCG”在不同查询构成下含义并不一样，基线提供了参照。

**`exact_at_1`**：排第一的商品**真实标签是不是 E**。它不是 top-1 分类准确率（top-1 是“预测的类别等于真实类别”），而是“排在第一位的商品是不是完全满足需求”。这两个数在同一份数据上可以差很多。

**`per_query` 里的混淆矩阵**：每个查询一份 4×4。它和全局 `conf` 内容重复，但**成对 Bootstrap 需要按查询重采样**，所以必须留到查询粒度。

### 6.4 汇总（第 189–206 行）

```python
    valid = [r for r in per_query if r['ndcg10'] is not None]
    result = {'pairs': len(rows), 'queries': len(by_query), 'zero_idcg_queries': len(per_query) - len(valid),
              'labels': LABELS, 'gains': GAINS, 'ndcg10': sum(r['ndcg10'] for r in valid) / len(valid),
              'ndcg_full': sum(r['ndcgfull'] for r in valid) / len(valid),
              'random_ndcg10': sum(r['random_ndcg10'] for r in valid) / len(valid),
              'exact_at_1': sum(r['exact_at_1'] for r in per_query) / len(per_query),
              'per_query': per_query}
    if has_classes:
        result['confusion'] = conf
        result['accuracy'] = sum(conf[i][i] for i in range(4)) / len(rows)
        result['per_class'] = {}
        for i, label in enumerate(LABELS):
            tp, support, predicted = conf[i][i], sum(conf[i]), sum(r[i] for r in conf)
            result['per_class'][label] = {'support': support, 'predicted': predicted,
                'precision': tp / max(1, predicted), 'recall': tp / max(1, support),
                'f1': 2 * tp / max(1, support + predicted)}
        result['macro_f1'] = sum(x['f1'] for x in result['per_class'].values()) / 4
    return result
```

- **nDCG 只在 `valid` 上平均**，即排除“没有正增益商品”的查询；这些查询的数量单独记为 `zero_idcg_queries`，不会悄悄改变分母。
- **`exact_at_1` 在全部查询上平均**（它不需要非零理想值）。
- **每类 F1 的公式**：`2·tp / (support + predicted)`。`support` 是真实类别为它的条数（行和），`predicted` 是被预测成它的条数（列和）。`max(1, …)` 防止除零。
- **Macro-F1 是四类 F1 的算术平均**，不按样本数加权。这就是为什么稀有的 C 类（训练集里只有 516 条，占 5%）对总分有很大影响：C 的 F1 从 0.30 提到 0.35，Macro-F1 就涨 1.25 个百分点。

> **命名坑**：`per_query` 里的键是 `ndcgfull`（没有下划线），顶层结果里的键是 `ndcg_full`（有下划线）。写脚本取数时别混。`ndcg10` 两处一致。

### 6.5 用真实数据手算一遍

以 FUSION_KD 在 F 上的混淆矩阵为例（来自 [report.json](../../results/fusion_continue_20261021/report.json)，行=真实、列=预测）：

| gold \ pred       |    E |    S |   C |    I | 行和（support） |
| ----------------- | ---: | ---: | --: | ---: | --------------: |
| E                 | 2933 | 1236 | 101 |  419 |            4689 |
| S                 | 1143 | 1970 |  26 |  594 |            3733 |
| C                 |  166 |   85 | 136 |   96 |             483 |
| I                 |  318 |  408 |  38 |  790 |            1554 |
| 列和（predicted） | 4560 | 3699 | 301 | 1899 |           10459 |

- Accuracy = (2933 + 1970 + 136 + 790) / 10459 = 5829 / 10459 = **0.5573**。
- F1(E) = 2×2933 / (4689 + 4560) = 5866 / 9249 = **0.6342**。
- F1(S) = 3940 / (3733 + 3699) = **0.5301**。
- F1(C) = 272 / (483 + 301) = **0.3469**。
- F1(I) = 1580 / (1554 + 1899) = **0.4576**。
- Macro-F1 = (0.6342 + 0.5301 + 0.3469 + 0.4576) / 4 = **0.4922**。

和 [RESULT.md](../../results/fusion_continue_20261021/RESULT.md) 里的 55.73% / 49.22% 完全对上。

### 6.6 nDCG 的一个真实查询

从 `runs/fusion_continue_20261021/evaluation/S_F.jsonl` 里取查询 `27256`（`cochecito de niño`，8 条商品），S 模型给出的排序是：

| 名次 | example_id | 真实标签 |  score |
| ---: | ---------: | -------- | -----: |
|    1 |     552788 | E        | 0.2927 |
|    2 |     552793 | E        | 0.2003 |
|    3 |     552794 | E        | 0.1748 |
|    4 |     552792 | E        | 0.1685 |
|    5 |     552791 | E        | 0.1594 |
|    6 |     552790 | E        | 0.1268 |
|    7 |     552787 | E        | 0.1172 |
|    8 |     552789 | C        | 0.0737 |

- 折扣：1、0.6309、0.5、0.4307、0.3869、0.3562、0.3333、0.3155（和 = 3.9535）。
- DCG = 1+0.6309+0.5+0.4307+0.3869+0.3562+0.3333 + 0.01×0.3155 = **3.6412**。
- 理想排序：前七个 E、最后一个 C，ideal 也是 3.6412，所以 **nDCG = 1.0**。
- 随机基线 = (7.01/8) × 3.9535 / 3.6412 = **0.9514**——因为 8 条里 7 条都是 E，乱排也有 0.95。

**如果把这 8 条按相反顺序排**（C 在最前）：DCG = 0.01×1 + (0.6309+…+0.3155) = 0.01 + 2.9535 = 2.9635，nDCG = 2.9635 / 3.6412 = **0.8139**。这就是“折扣”的作用：把一个 E 从第 1 位挪到第 8 位，代价约 0.19 的 nDCG。

`compact_metrics()`（第 209–210 行）只是把 `per_query` 去掉，写进报告时体积从 MB 级降到 KB 级。需要按查询做统计（Bootstrap）时，用的是内存里的完整结果，不是报告里的。

---

## 7. 路由与融合（第 213–237 行）

### 7.1 `low_confidence_indices()`（第 213–218 行）

```python
def low_confidence_indices(students):
    """Fixed half-budget route: lowest max probability, SHA256 tie-break, logits-free."""
    ordered = sorted(range(len(students)), key=lambda i: (
        max(students[i]['probabilities'].values()),
        stable_key(students[i]['example_id'], 'esci-cascade-v13')))
    return set(ordered[:math.ceil(len(students) / 2)])
```

三句话解释完：

1. **`max(p)` 作为置信度**：四类概率里最大的一类有多自信。越低说明原模型越拿不准。
2. **升序排序，取前一半**：`ceil(10398/2) = 5199`。`ceil` 而不是 `floor`，保证“恰好一半”时多选一条而不是少选。
3. **并列看哈希**：`max(p)` 完全相等的样本按 `stable_key(example_id, 'esci-cascade-v13')` 排。用哈希而不是文件顺序，是为了让路由结果不依赖输入文件的排列，换一次导出顺序结果不变。盐 `esci-cascade-v13` 固定了这套打散方式。

返回的是索引集合 `set`，后面用 `i in route` 判断。

注释里的 `logits-free` 强调：路由只用概率，不用 logits，也不看真实标签和教师是否正确——避免“用答案决定谁接受教师”。

### 7.2 `fuse()`（第 221–237 行）

```python
def fuse(students, teachers, selected):
    """Equal-weight probability fusion on `selected`; unavailable teacher falls back to student."""
    if len(students) != len(teachers):
        raise ValueError('Unpaired coverage')
    selected = set(selected)
    output = []
    for i, (s, t) in enumerate(zip(students, teachers)):
        if any(s[k] != t[k] for k in ('example_id', 'query_id', 'product_id')):
            raise ValueError('Unpaired identity')
        ps = {k: (s['probabilities'][k] + t['probabilities'][k]) / 2
              if i in selected and t['probabilities'] is not None else s['probabilities'][k]
              for k in LABELS}
        row = {k: s[k] for k in ('example_id', 'query_id', 'product_id')}
        row.update(probabilities=ps, prediction=max(LABELS, key=lambda k: ps[k]),
                   score=sum(ps[k] * g for k, g in zip(LABELS, GAINS)))
        output.append(row)
    return output
```

- **等权平均**：`(p_S + p_T) / 2`，不按置信度加权，也不做温度缩放。两边各自和为 1，平均后仍为 1，不需要再归一化。
- **教师缺失时回退到学生**：`t['probabilities'] is not None`。这正是“路由 5,199 条、实际融合 5,195 条”的 4 条差额——它们被选中了，但教师没有给出概率。
- **身份先对齐**：`example_id`、`query_id`、`product_id` 三个都得相同，否则报 `Unpaired identity`。避免“按位置 zip 了两份顺序不同的文件”。
- **输出行只有身份 + 概率 + 预测 + 分数**，不含 `gold`。`prepare()` 随后补上 `label`；`final_evaluation()` 评测 S_C50 时再从输入行补 `gold`。
- **`prediction` 用 `max(LABELS, key=...)`**：并列时取 `LABELS` 中靠前的（E > S > C > I），结果确定。

`fuse()` 在本轮出现两次：一次在 `prepare()` 里算训练目标 `q`；一次在 `final_evaluation()` 里对 F 上的 S 预测做融合，得到参考系统 `S_C50_reference`。两处用的是同一个函数和同一个路由函数，所以“训练目标”和“参考系统”在实现上是同一套规则——这是有意让参考系统可复现。

---

## 8. 输出路径保护 `check_output_path()`（第 240–249 行）

```python
def check_output_path(path):
    """Keep new run outputs outside project inputs and published results."""
    output = Path(path).resolve()
    frozen = [DATA, MODELS, CONFIGS, ROOT / 'src',
              ROOT / 'docs', ROOT / 'provenance', ROOT / 'results', ROOT / '_archive']
    for directory in frozen:
        directory = directory.resolve()
        if output == directory or output.is_relative_to(directory) or directory.is_relative_to(output):
            raise ValueError(f'Output overlaps project inputs: {output}; use runs/ or an independent directory')
    return output
```

两个方向都检查：

- `output.is_relative_to(directory)`：输出目录在 `data/`、`models/` 等里面 → 拒绝。
- `directory.is_relative_to(output)`：输出目录是这些目录的**祖先**（例如 `--output .` 或 `--output D:\Projects`）→ 也拒绝，否则一次“清空输出目录”的操作可能删掉输入。

`return output` 返回的是解析后的绝对路径，后面所有 `out / ...` 都基于它。默认输出 `runs/fusion_continue_20261021` 不在冻结列表里，合法。

---

## 9. 本文件维护的不变量

读完上面，可以把 `common.py` 的约定压缩成 8 条。任何一条被破坏，下游都会出错，而且往往不会立刻报错：

1. **类别顺序恒为 E、S、C、I**，概率字典、张量列、混淆矩阵行列一致。
2. **概率之和为 1，误差 < 1e-5**（由 `fusion.probs_ok()` 与 `audit()` 检查）。
3. **`score` 必须有限**；排序并列时按 `product_id`、`example_id`，不看标签。
4. **模型输入只有 `query`/`title`/`brand` 三个字段**，顺序固定，编码为 JSON。
5. **序列不截断**；超长即失败。
6. **JSONL 一行一条，LF 换行，UTF-8 不转义**，因此文件哈希可跨平台复现。
7. **nDCG 的平均分母只含“理想值非零”的查询**；这个数量单独报告。
8. **输出目录不得与 `data/`、`models/`、`configs/`、`src/`、`docs/`、`results/`、`_archive/` 重叠**。

---

## 10. 常见误解与坑

| 看到的现象                                                    | 实际含义                                                                 |
| ------------------------------------------------------------- | ------------------------------------------------------------------------ |
| `metrics()` 报 `Invalid or partial classification output` | 一个划分里有的行有 `prediction`、有的是 `None`。应该全部有或全部没有 |
| `KeyError: 'score'`                                         | 自己拼的预测行缺 `score`。`predict()` 会自动算，手写行要补           |
| 报告里没有 `per_query`                                      | `compact_metrics()` 为了体积删掉了；Bootstrap 用的是内存结果           |
| `exact_at_1` 比 Accuracy 高很多                             | 它衡量“第一名是不是 E”，不是分类正确率                                 |
| `random_ndcg10` 接近 1                                      | 该划分的正增益商品占比很高，不是代码算错                                 |
| 用 `splitlines()` 读 JSONL 报 `JSONDecodeError`           | 文本里有 U+2028 之类字符；用文件迭代或 `read_jsonl()`                  |
| 改了 `SYSTEM` 提示后 `load_split()` 失败                  | `input_contract_sha256` 是提示的哈希，提示属于数据契约                 |
| `stable_key` 换个盐值后路由数量变了                         | 0.5 附近有并列样本，盐值决定谁进谁出                                     |

---

## 11. 自测

1. 如果只把 `LABELS` 改成 `['E', 'C', 'S', 'I']`，程序会在哪一步先失败？为什么不是指标算错这么简单？
2. `payload()` 为什么不包含 `title_clipped`？如果包含，会改变什么？
3. 一个查询有 3 条商品，增益分别是 E、S、I，模型排成 S、E、I。手算 nDCG@10。
4. `low_confidence_indices()` 为什么返回 `set` 而不是排好序的 `list`？调用方怎么用？
5. 为什么 `metrics()` 要在每个查询里再存一份 4×4 混淆矩阵，明明顶层已经有一份？

参考思路：1 会让 `targets` 的四列对不上模型四列，损失训练出错误映射；2 会改变输入字符串但不影响标签，历史缓存概率失配；3 DCG = 0.1 + 0.6309 = 0.7309，ideal = 1 + 0.1×0.6309 = 1.0631，nDCG ≈ 0.6875；4 因为调用方只做 `i in route` 判断，集合查找 O(1)，顺序无意义；5 因为成对 Bootstrap 要按查询重采样。

---

## 12. 验证命令

只用标准库即可（把 `python` 换成你自己的解释器）：

```powershell
# 1) 标签与增益
python -c "import sys; sys.path.insert(0,'src'); from common import LABELS, GAINS; print(LABELS, GAINS)"

# 2) 文件哈希（显示当前源码身份，不与历史运行清单比较）
python -c "import sys; sys.path.insert(0,'src'); from common import digest; print(digest('src/common.py'))"

# 3) 某条预测的 score 是否等于 Σ p×gain
python -c "import json; r=[json.loads(x) for x in open('runs/fusion_continue_20261021/evaluation/S_F.jsonl',encoding='utf-8')][0]; p=r['probabilities']; print(sum(p[k]*g for k,g in zip(['E','S','C','I'],[1,.1,.01,0])), r['score'])"

# 4) 用报告里的混淆矩阵复算 Macro-F1
python -c "import json; c=json.load(open('results/fusion_continue_20261021/report.json',encoding='utf-8'))['metrics']['F']['FUSION_KD']['confusion']; f=[2*c[i][i]/(sum(c[i])+sum(r[i] for r in c)) for i in range(4)]; print([round(x,4) for x in f], round(sum(f)/4,4))"

# 5) 路由与教师使用数量
python -c "import json; t=[json.loads(x) for x in open('data/fusion_continue_20261021/targets.jsonl',encoding='utf-8')]; print(sum(r['route_selected'] for r in t), sum(r['teacher_used'] for r in t), len(t))"
```

预期输出依次是：`['E', 'S', 'C', 'I'] [1.0, 0.1, 0.01, 0.0]`；当前 `common.py` 的 SHA256；两个相同的 `0.37921982298605145`；`[0.6342, 0.5301, 0.3469, 0.4576] 0.4922`；`5199 5195 10398`。

下一站：[02 modeling](02-modeling.md) —— 这些编号怎样变成四个分数。
