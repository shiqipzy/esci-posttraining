"""Verify training inputs, base model and retained final artifacts."""
import sys
sys.dont_write_bytecode = True

import argparse
from common import ROOT, digest, read_json, require, verify_model
from train import verify_inputs


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    cfg = read_json(ROOT / 'configs/training.json')
    *_, receipt = verify_inputs(ROOT / 'data', cfg)
    verify_model(str(ROOT / 'models/base'))
    model_hashes = {}
    for name in ('S', 'A2', 'TA2', 'G2'):
        actual = digest(ROOT / f'models/{name}/adapter/adapter_model.safetensors')
        manifest = read_json(ROOT / f'results/final/predictions/{name}/F_manifest.json')
        require(actual == manifest['identity']['model_adapter_sha256'], 'Wrong final adapter: ' + name)
        model_hashes[name] = actual
    require(len(set(model_hashes.values())) == 4, 'Final models share an adapter')
    report = read_json(ROOT / 'results/final/analysis.json')
    for name, expected in report['student_sources'].items():
        require(digest(ROOT / f'results/final/predictions/{name}/F_predictions.jsonl') == expected,
                'Changed final predictions: ' + name)
    require(receipt['posttraining_steps_per_arm'] == 650, 'Changed final training budget')
    print({'status': 'PASS', 'training': receipt, 'retained_models': list(model_hashes),
           'evaluation_objects': len(report['metrics'])})


if __name__ == '__main__':
    main()
