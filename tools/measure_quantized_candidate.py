#!/usr/bin/env python3
"""Compare exact-preserving INT8 candidates on an already uploaded local pack."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from canister_infer import Icp, ROOT, decode_blobs, require_local_network, warmup
from client_held_query import QuerySession
from measure_inference import Failure


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT/'build/client-query-project')
    p.add_argument('--identity',default='ic-laya-query-local-test')
    p.add_argument('--label',required=True)
    p.add_argument('--warmup',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();icp=Icp(args.project_root.resolve(),'local',args.identity);require_local_network(icp)
    if args.warmup:warmup(icp)
    status=json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))
    wasmhash='0x'+hashlib.sha256((ROOT/'build/decision-engine.wasm').read_bytes()).hexdigest()
    if status['module_hash']!=wasmhash:raise Failure('module does not match build')
    bundle=decode_blobs(icp.query('decision-engine','info'))[0].hex()
    corpus=json.loads((ROOT/'artifacts/int8_optimization_v4/validation-corpus.json').read_text())
    if corpus['pack_manifest_sha256']!=bundle:raise Failure('pack binding differs')
    reference=json.loads((ROOT/'artifacts/references/int8-fusion-corpus.json').read_text())
    if reference['bundle_sha256']!=bundle:raise Failure('baseline pack differs')
    baselines={c['id']:c for c in reference['cases']}
    report=dict(label=args.label,module_hash=wasmhash,bundle_sha256=bundle,network='local',measured_at=datetime.now(timezone.utc).isoformat(),query_cache_controlled=False,cases=[],source_sha256={name:hashlib.sha256((ROOT/'crates/laya-candle/src'/name).read_bytes()).hexdigest() for name in ('int8.rs','quantized.rs','int8_quant.rs','q8_ops.rs','lib.rs') if (ROOT/'crates/laya-candle/src'/name).exists()})
    pilot=json.loads((ROOT/'artifacts/references/int8-fusion-pilot.json').read_text())
    cases=[('pilot-128',json.loads((ROOT/'artifacts/laya-choice-128-input.json').read_text()),pilot)]
    for name in ('choice-natural-03','choice-boundary-65'):
        cases.append((name,next(c['input'] for c in corpus['cases'] if c['id']==name),baselines[name]))
    for name,inp,baseline in cases:
        result=QuerySession(icp,inp,2,activation_format='int8').finish()
        delta=max(abs(a-b) for a,b in zip(result['logits'],baseline['logits']))
        if delta!=0:raise Failure(f'{name} logits changed: {delta}')
        row=dict(id=name,baseline_instructions=baseline['instructions'],instruction_reduction_percent=(1-result['instructions']/baseline['instructions'])*100,max_abs_logit_difference=delta,**result)
        report['cases'].append(row);print(name,row['instructions'],row['instruction_reduction_percent'],flush=True)
    end=json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))
    if end['module_hash']!=wasmhash:raise Failure('module changed')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
