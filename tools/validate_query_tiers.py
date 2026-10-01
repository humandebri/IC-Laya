#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import json, hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import client_held_query as client
from canister_infer import Icp,decode_blobs,require_local_network
from query_transport import auto_width, AUTO_BUNDLE
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Compare calibrated query tiers with the previous conservative tiers on a warm local model.')
parser.add_argument('--output', type=Path, required=True)
args=parser.parse_args()
if args.output.exists():
    parser.error('use a new output file to preserve earlier measurements')
args.output.parent.mkdir(parents=True, exist_ok=True)
icp=Icp(root/'build/client-query-project','local','ic-laya-query-local-test')
require_local_network(icp)
module=json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))['module_hash']
bundle=decode_blobs(icp.query('decision-engine','info'))[0].hex()
def legacy_width(tokens, mode, completed, cap=16):
    width = 16 if tokens <= 16 else (9 if mode == 'f32' else 8) if tokens <= 43 else 6 if tokens <= 65 else 3
    width = min(width, cap)
    left = max(0, 28-completed)
    return min(16, cap, 32-completed) if left == 0 or left == 1 or left+2 <= width else min(width, 32-completed)
assert bundle == AUTO_BUNDLE
original = legacy_width
candidate = auto_width
report={'module':module,'bundle':bundle,'cases':[],'completed':False,'measured_at':datetime.now(timezone.utc).isoformat(),'query_cache_controlled':False,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'tools/client_held_query.py',root/'tools/query_transport.py')}}
for tokens in (66,67,68,69,70,71,72,73,79,80,81,87,88,89,90,91,95,96):
 for mode in ('f32','int8'):
  inp=dict(input_ids=[50281]+[50284]*7+[50283]*(tokens-9)+[50282],markers=list(range(1,8)),qtype_id=tokens%3)
  client.auto_width=original
  baseline=client.QuerySession(icp,inp,steps=None,activation_format=mode,compression='auto').finish()
  client.auto_width=candidate
  trial=client.QuerySession(icp,inp,steps=None,activation_format=mode,compression='auto').finish()
  assert baseline['logits']==trial['logits']
  assert not baseline['failed_query_attempts'] and not trial['failed_query_attempts']
  row={'tokens':tokens,'mode':mode,'markers':7,'qtype':inp['qtype_id'],'max_handler':trial['max_observed_query_handler_instructions'],'old_calls':baseline['inference_query_calls'],'new_calls':trial['inference_query_calls'],'payload':trial['request_candid_bytes']+trial['response_continuation_bytes'],'old_payload':baseline['request_candid_bytes']+baseline['response_continuation_bytes'],'exact_logits':True,'failures':trial['failed_query_attempts']}
  report['cases'].append(row);args.output.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
report['completed']=True
assert json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))['module_hash']==module
assert decode_blobs(icp.query('decision-engine','info'))[0].hex()==bundle
args.output.write_text(json.dumps(report,indent=2))
