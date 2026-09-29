#!/usr/bin/env python3
"""Verify the measured 16-token query allowance and 17+ rejection on local IC."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

from canister_infer import Icp, ROOT, decode_blobs, infer, network_status, require_local_network


def args_for(inp):
    ids = '; '.join(map(str, inp['input_ids']))
    markers = '; '.join(map(str, inp['markers']))
    return f"(record {{ input_ids = vec {{ {ids} }}; markers = vec {{ {markers} }}; qtype_id = {inp['qtype_id']} : nat32 }})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', default='ic-laya-int8')
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/int8_optimization_v4/query-guard-check.json')
    args = parser.parse_args()
    icp = Icp(ROOT, 'local', args.identity)
    require_local_network(icp)
    if network_status(icp) is None:
        raise RuntimeError('local network is stopped')
    module = '0x' + hashlib.sha256((ROOT / 'build/decision-engine.wasm').read_bytes()).hexdigest()
    installed = json.loads(icp.run(['canister', 'status', 'decision-engine', '-e', 'local', '--json']))
    if installed['module_hash'] != module:
        raise RuntimeError('installed module differs from build artifact')
    grid = json.loads((ROOT / 'artifacts/int8_optimization_v4/query-limit-grid.json').read_text())
    bundle = decode_blobs(icp.query('decision-engine', 'info'))[0].hex()
    if grid['bundle_sha256'] != bundle:
        raise RuntimeError('different model pack')
    rows = []
    for candidate in grid['rows']:
        if candidate['tokens'] not in (16, 17):
            continue
        reply = icp.run(['canister', 'call', 'decision-engine', 'infer_tokens_query',
                         args_for(candidate['input']), '-e', 'local',
                         '--candid', icp.did['decision-engine'], '--query'], expect_ok=False)
        row = {key: candidate[key] for key in ('tokens', 'markers', 'qtype_id', 'input_sha256')}
        if candidate['tokens'] == 16:
            cost = re.search(r'instructions\s*=\s*([\d_]+)', reply)
            logits_match = re.search(r'logits\s*=\s*vec\s*\{([^}]*)\}', reply)
            if 'Ok = record' not in reply or not cost or not logits_match:
                raise RuntimeError(reply)
            logits = [float(value) for value in re.findall(
                r'([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s*:\s*float32', logits_match[1], re.I)]
            row.update(status='ok', instructions=int(cost[1].replace('_', '')),
                       logits_match=logits == candidate['logits'])
            if not row['logits_match'] or row['instructions'] >= 5_000_000_000:
                raise RuntimeError(f'16-token query mismatch: {row}')
        else:
            if 'TooLong' not in reply or 'IC0522' in reply:
                raise RuntimeError(f'17-token input not rejected before execution: {reply}')
            row['status'] = 'too_long'
        rows.append(row)
        print(row['tokens'], row['markers'], row['qtype_id'], row['status'], row.get('instructions'), flush=True)
    with tempfile.TemporaryDirectory(prefix='laya-query-guard-') as directory:
        path = Path(directory) / 'input.json'
        path.write_text(json.dumps(next(x['input'] for x in grid['rows'] if x['tokens'] == 17)))
        update_17 = infer(icp, path)
    long_input = json.loads((ROOT / 'artifacts/laya-choice-128-input.json').read_text())
    long_reply = icp.query('decision-engine', 'infer_tokens_query', args_for(long_input))
    if 'TooLong' not in long_reply:
        raise RuntimeError('128-token query was not rejected early')
    report = {'network': 'local', 'module_hash': module, 'bundle_sha256': bundle,
              'max_accepted_tokens': 16, 'first_rejected_tokens': 17,
              'query_instruction_limit': 5_000_000_000,
              'rows': rows, 'update_17_succeeds': len(update_17['logits']) > 0,
              'query_128_rejected': True,
              'max_query_instructions': max(x['instructions'] for x in rows if x['status'] == 'ok')}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
