"""Evaluate a trained adapter on the tune split or the fixed final F."""
import sys
sys.dont_write_bytecode = True

import argparse
import time
from pathlib import Path
from common import (ROOT, check_output_path, compact_metrics, digest, fresh_dir, load_split, metrics,
                    read_json, read_jsonl, require, versions, write_json, write_jsonl)


def selected_checkpoint(run_dir, arm):
    run_dir = Path(run_dir)
    run = read_json(run_dir / 'run_manifest.json')
    if run['status'] != 'complete':
        raise ValueError('Training run is not complete')
    checkpoint = run_dir / 'models' / arm
    for name, sha in read_json(checkpoint / 'checkpoint_manifest.json')['files'].items():
        if digest(checkpoint / name) != sha:
            raise ValueError(f'Modified checkpoint file: {name}')
    return checkpoint, read_json(checkpoint / 'run_config.json'), run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=ROOT / 'models/base')
    parser.add_argument('--data', type=Path, default=ROOT / 'data')
    parser.add_argument('--split', choices=['tune', 'F'], required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arm', choices=['S', 'A2', 'TA2', 'G2'], default='TA2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int)
    args = parser.parse_args()
    from transformers import set_seed
    from modeling import RelevanceModel, predict

    if args.split == 'F':
        inputs = read_jsonl(args.data / 'final/input.jsonl')
        gold = read_jsonl(args.data / 'final/gold.jsonl')
        require(len(inputs) == len(gold) and all(all(r[k] == g[k] for k in ('example_id', 'query_id', 'product_id'))
                    for r, g in zip(inputs, gold)), 'Final input/gold identity mismatch')
        rows = [dict(r, label=g['label']) for r, g in zip(inputs, gold)]
    else:
        rows, _ = load_split(args.data / 'sft', args.split)
    source_sha256 = digest(args.data / 'final/input.jsonl' if args.split == 'F'
                           else args.data / 'sft' / (args.split + '.jsonl'))

    checkpoint, cfg, training_run = selected_checkpoint(args.run, args.arm)
    require(training_run['inputs']['sft_manifest_sha256'] == digest(args.data / 'sft/manifest.json'),
            'Checkpoint trained with different SFT data')
    args.output = check_output_path(args.output)
    fresh_dir(args.output)
    set_seed(cfg['seed'])
    model = RelevanceModel(str(args.model_path), cfg, device=args.device, checkpoint=checkpoint)
    batch = args.batch_size or cfg['eval_batch_size']
    started = time.perf_counter()
    predictions = predict(model, rows, batch, cfg['max_length'])
    measured = metrics(predictions)
    write_jsonl(args.output / 'predictions.jsonl', predictions)
    write_json(args.output / 'metrics.json', measured)
    write_json(args.output / 'eval_manifest.json', {'status': 'complete', 'config': cfg, 'split': args.split,
        'source_sha256': source_sha256, 'predictions_sha256': digest(args.output / 'predictions.jsonl'),
        'checkpoint': str(checkpoint), 'training_run': str(args.run),
        'checkpoint_manifest_sha256': digest(checkpoint / 'checkpoint_manifest.json'),
        'training_run_manifest_sha256': digest(args.run / 'run_manifest.json'),
        'official_model': model.provenance, 'device': args.device, 'software': versions(),
        'batch_size': batch, 'max_length': cfg['max_length'],
        'elapsed_seconds': time.perf_counter() - started})
    print(compact_metrics(measured))


if __name__ == '__main__':
    main()
