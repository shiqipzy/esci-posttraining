"""Shared data contract, grouped budgets, metrics and fixed teacher fusion."""
import collections
import hashlib
import importlib.metadata
import json
import math
import random
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs'
DATA = ROOT / 'data'
MODELS = ROOT / 'models'
LABELS = ['E', 'S', 'C', 'I']
GAINS = [1.0, 0.1, 0.01, 0.0]
ARMS = ('A2', 'TA2', 'G2')
SYSTEM = '''Classify a product for a shopping query using the product title and brand.
E = Exact: the product satisfies the query intent.
S = Substitute: the product is not an exact match but can serve as an alternative.
C = Complement: an accessory or item used together with the requested product, not the requested product itself.
I = Irrelevant: neither an exact match, a substitute, nor a complement.
Return exactly one letter: E, S, C, or I.'''


def require(condition, message):
    if not condition:
        raise ValueError(message)


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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def stable_key(value, salt='esci-v1-20260928'):
    return hashlib.sha256(f'{salt}|{value}'.encode()).hexdigest()


def fresh_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f'Refuse nonempty output: {path}. Preserve previous runs.')
    path.mkdir(parents=True, exist_ok=True)


def versions():
    result = {}
    for name in ['torch', 'transformers', 'peft', 'accelerate', 'numpy', 'pandas', 'pyarrow', 'safetensors']:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


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


def payload(row):
    # No IDs, labels, source splits, or diagnostics enter the model input.
    return json.dumps({k: row[k] for k in ['query', 'title', 'brand']}, ensure_ascii=False)


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


def collate(features, pad_id, device):
    import torch
    width = max(len(x['input_ids']) for x in features)
    out = {}
    for key in features[0]:
        pad = pad_id if key == 'input_ids' else 0
        out[key] = torch.tensor([f[key] + [pad] * (width - len(f[key])) for f in features],
                                dtype=torch.long, device=device)
    return out


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


def metrics(rows):
    if not rows or len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('Empty/duplicate predictions')
    by_query = collections.defaultdict(list)
    conf = [[0] * 4 for _ in range(4)]
    has_classes = all(r.get('prediction') in LABELS for r in rows)
    if not has_classes and any(r.get('prediction') is not None for r in rows):
        raise ValueError('Invalid or partial classification output')
    for row in rows:
        if not math.isfinite(row['score']):
            raise ValueError('Nonfinite ranking score')
        by_query[row['query_id']].append(row)
        if has_classes:
            conf[LABELS.index(row['gold'])][LABELS.index(row['prediction'])] += 1
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


def compact_metrics(result):
    return {k: v for k, v in result.items() if k != 'per_query'}


def low_confidence_indices(students):
    """Fixed half-budget route: lowest max probability, SHA256 tie-break, logits-free."""
    ordered = sorted(range(len(students)), key=lambda i: (
        max(students[i]['probabilities'].values()),
        stable_key(students[i]['example_id'], 'esci-cascade-v13')))
    return set(ordered[:math.ceil(len(students) / 2)])


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
