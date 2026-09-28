#!/usr/bin/env python3
"""Measure full-model local canister latency with a fixed pack and corpus.

Install and warm the candidate Wasm before running this script. Run it once per
variant against the same local canister, upgrading and warming between variants.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import tempfile

from canister_infer import decode_blobs, infer
from measure_inference import Icp, require_local_network

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IDS = ('choice-natural-00', 'choice-boundary-64',
               'choice-boundary-96', 'choice-boundary-128')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--expected-wasm', required=True, type=Path)
    parser.add_argument('--network-root', required=True, type=Path)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'artifacts/int8_optimization_v4/validation-corpus.json')
    parser.add_argument('--case-id', action='append', dest='case_ids')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0:
        parser.error('repeats must be positive and warmups nonnegative')

    icp = Icp(args.network_root, 'local', 'ic-laya-int8')
    require_local_network(icp)
    status = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    wasm_hash = hashlib.sha256(args.expected_wasm.read_bytes()).hexdigest()
    if status['module_hash'] != '0x' + wasm_hash:
        raise RuntimeError('installed module does not match --expected-wasm')
    bundle = decode_blobs(icp.query('decision-engine', 'info'))
    if len(bundle) != 1 or len(bundle[0]) != 32:
        raise RuntimeError('invalid active bundle')
    corpus_raw = args.corpus.read_bytes()
    corpus = {case['id']: case for case in json.loads(corpus_raw)['cases']}
    case_ids = args.case_ids or DEFAULT_IDS
    missing = set(case_ids) - corpus.keys()
    if missing:
        raise RuntimeError(f'unknown case IDs: {sorted(missing)}')
    report = {
        'variant': args.variant,
        'measured_at': datetime.now(timezone.utc).isoformat(),
        'method': 'local icp-cli update wall time; includes CLI and local replica overhead',
        'host': platform.platform(),
        'machine': platform.machine(),
        'wasm_sha256': wasm_hash,
        'bundle_sha256': bundle[0].hex(),
        'corpus_sha256': hashlib.sha256(corpus_raw).hexdigest(),
        'repeats': args.repeats,
        'warmups': args.warmups,
        'cases': [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='laya-full-wall-') as directory:
        path = Path(directory) / 'case.json'
        for case_id in case_ids:
            case = corpus[case_id]
            path.write_text(json.dumps(case['input']))
            tokens = len(case['input']['input_ids'])
            samples = []
            for rep in range(-args.warmups, args.repeats):
                result = infer(icp, path, stepped=tokens > 100)
                if rep >= 0:
                    samples.append({
                        'rep': rep,
                        'wall_seconds': result['local_wall_seconds'],
                        'instructions': result['instructions'],
                        'logits': result['logits'],
                        'update_calls': result['inference_update_calls'],
                    })
            if len({tuple(sample['logits']) for sample in samples}) != 1:
                raise RuntimeError(f'non-deterministic logits for {case_id}')
            row = {
                'id': case_id,
                'tokens': tokens,
                'samples': samples,
                'median_wall_seconds': statistics.median(s['wall_seconds'] for s in samples),
                'median_instructions': statistics.median(s['instructions'] for s in samples),
                'min_wall_seconds': min(s['wall_seconds'] for s in samples),
                'max_wall_seconds': max(s['wall_seconds'] for s in samples),
            }
            report['cases'].append(row)
            args.output.write_text(json.dumps(report, indent=2) + '\n')
            print(args.variant, case_id, tokens, row['median_wall_seconds'], flush=True)


if __name__ == '__main__':
    main()
