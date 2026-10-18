"""Data preparation, pinned model downloads, and training environment checks."""
import sys
sys.dont_write_bytecode = True

import argparse
import collections
import hashlib
import unicodedata
from pathlib import Path
from common import (ROOT, SYSTEM, digest, fresh_dir, read_json, stable_key, verify_model,
                    versions, write_json, write_jsonl, require)


EXPECTED = {
    'shopping_queries_dataset_examples.parquet': '4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a',
    'shopping_queries_dataset_products.parquet': '25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265'}


def norm(s):
    return ' '.join(unicodedata.normalize('NFKC', str(s)).casefold().split())


def prepare_data(args):
    fresh_dir(args.out)
    for name, sha in EXPECTED.items():
        if digest(args.source / name) != sha:
            raise ValueError(f'Official parquet hash mismatch: {name}')
    import pandas as pd
    examples = pd.read_parquet(args.source / 'shopping_queries_dataset_examples.parquet')
    small = examples[(examples.small_version == 1) & (examples.product_locale == 'us')].copy()
    train = small[small.split == 'train']
    test = small[small.split == 'test']
    old_probe = set(read_json(ROOT / 'configs/excluded_probe.json')['query_ids'])
    qtext = train[['query_id', 'query']].drop_duplicates()
    if len(qtext) != qtext.query_id.nunique():
        raise ValueError('Multiple texts per query ID')
    if set(train.query_id) & set(test.query_id) or set(map(norm, train['query'])) & set(map(norm, test['query'])):
        raise ValueError('Cross-split query leakage')
    if small.duplicated(['query_id', 'product_id', 'product_locale']).any():
        raise ValueError('Duplicate query/product pair')
    # Normalize-equivalent queries stay together; aliases of the old probe stay excluded.
    by_text = collections.defaultdict(list)
    for row in qtext.itertuples():
        by_text[norm(row.query)].append(int(row.query_id))
    eligible = [ids for ids in by_text.values() if not set(ids) & old_probe]
    eligible.sort(key=lambda ids: stable_key(min(ids)))
    if any(len(ids) != 1 for ids in eligible):
        raise ValueError('Found normalized query aliases; freeze a group-based plan rather than split aliases')
    ordered = [ids[0] for ids in eligible]
    splits = {'tune': ordered[:256], 'gate': ordered[256:768], 'pilot_train': ordered[768:1792]}
    needed_ids = {q for ids in splits.values() for q in ids}
    selected = small[small.query_id.isin(needed_ids)].copy()
    products = pd.read_parquet(args.source / 'shopping_queries_dataset_products.parquet',
                               columns=['product_id', 'product_locale', 'product_title', 'product_brand'])
    products = products[products.product_locale == 'us']
    selected = selected.merge(products, on=['product_locale', 'product_id'], how='left',
                              validate='many_to_one', indicator=True)
    if (selected['_merge'] != 'both').any() or selected.product_title.isna().any():
        raise ValueError('Missing product/title')
    rows_by_query = collections.defaultdict(list)
    for r in selected.itertuples():
        title = str(r.product_title)
        brand = '' if pd.isna(r.product_brand) else str(r.product_brand)
        row = {'example_id': int(r.example_id), 'query_id': int(r.query_id),
               'product_id': str(r.product_id), 'query': str(r.query),
               'title': title[:640], 'brand': brand[:96], 'label': str(r.esci_label),
               'title_clipped': len(title) > 640, 'brand_clipped': len(brand) > 96,
               'source_title_chars': len(title), 'source_brand_chars': len(brand)}
        rows_by_query[row['query_id']].append(row)
    files = {}
    for split, ids in splits.items():
        rows = [r for q in ids for r in sorted(rows_by_query[q], key=lambda r: r['product_id'])]
        name = split + '.jsonl'
        write_jsonl(args.out / name, rows)
        files[name] = {'sha256': digest(args.out / name), 'pairs': len(rows), 'queries': len(ids),
                       'labels': dict(collections.Counter(r['label'] for r in rows)),
                       'title_clipped': sum(r['title_clipped'] for r in rows),
                       'brand_clipped': sum(r['brand_clipped'] for r in rows)}
    write_json(args.out / 'manifest.json', {'version': 'esci-v1',
        'input_contract_sha256': hashlib.sha256(SYSTEM.encode()).hexdigest(),
        'official_source_sha256': EXPECTED, 'files': files, 'query_ids': splits,
        'excluded_old_probe': sorted(old_probe),
        'view': {'fields': ['query', 'title', 'brand'], 'title_prefix_chars': 640, 'brand_prefix_chars': 96,
                 'token_overflow_policy': 'raise; never silently truncate or drop'},
        'split_rule': 'query IDs sorted by SHA256(esci-v1-20260928|query_id); tune256, gate512, train1024'})
    print({k: {x: v[x] for x in ['pairs', 'queries', 'labels', 'title_clipped', 'brand_clipped']}
           for k, v in files.items()})


def download_models(args):
    from huggingface_hub import hf_hub_download
    lock = read_json(ROOT / 'configs/models.lock.json')
    for f in lock['files']:
        hf_hub_download(lock['model_id'], f['rfilename'], revision=lock['revision'], local_dir=args.out)
    verify_model(str(args.out.resolve()))
    print(f'Verified {lock["model_id"]} {lock["revision"]}: {args.out}')


def check_environment(args):
    import torch
    import transformers
    import peft
    report = versions()
    require(transformers.__version__ == '4.57.6' and peft.__version__ == '0.18.1',
            'Use verified transformers4.57.6/peft0.18.1 environment; do not upgrade experiment')
    require(tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]) >= (2, 8), 'Torch>=2.8 required')
    require(report['accelerate'] is not None, 'Missing accelerate')
    report.update(CUDA_available=torch.cuda.is_available(),
                  BF16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
    if args.gpu:
        require(report['CUDA_available'] and report['BF16'], 'CUDA BF16 required; no precision fallback')
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    data = commands.add_parser('data', help='build the frozen SFT splits from official parquet')
    data.add_argument('--source', type=Path, default=ROOT / 'data/raw')
    data.add_argument('--out', type=Path, default=ROOT / 'data/sft')
    data.set_defaults(func=prepare_data)
    models = commands.add_parser('models', help='download and verify the pinned Qwen base')
    models.add_argument('--out', type=Path, required=True)
    models.set_defaults(func=download_models)
    environment = commands.add_parser('environment', help='check the training environment')
    environment.add_argument('--gpu', action='store_true')
    environment.set_defaults(func=check_environment)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
