#!/usr/bin/env python3
"""Calibrate first-encoder QKV on 24 cases, hold out 72, benchmark real sums.

Diagnostic only: does not change model scales. Requires an already warm local
owner canister and numpy/torch. First-layer inputs have no attention norm.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import numpy as np
import torch
from canister_infer import Icp, ROOT, decode_blobs, require_local_network
from client_held_query import QuerySession
from check_client_held_query_protocol import query_bytes
from measure_inference import Failure


def uleb(n):
    out=bytearray()
    while n>=128:
        out.append((n&127)|128);n>>=7
    out.append(n);return bytes(out)


def rounded(v):
    a=np.abs(v);w=np.trunc(a)
    return np.copysign(w+((a-w)>=np.float32(.5)),v)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT/'build/client-query-project')
    p.add_argument('--identity',default='ic-laya-query-local-test')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();icp=Icp(args.project_root.resolve(),'local',args.identity);require_local_network(icp)
    status=json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))
    manifest=json.loads((ROOT/'checkpoints/laya-int8/manifest.json').read_text())
    cfg=manifest['config'];heads=cfg['attention_heads'];groups=3*heads
    if cfg['first_layer_attention_norm'] or cfg['hidden_size']!=1024 or heads!=16:
        raise Failure('fixture requires the fixed first-layer configuration')
    corpuspath=ROOT/'artifacts/int8_optimization_v4/validation-corpus.json'
    corpus=json.loads(corpuspath.read_text())
    bundle=decode_blobs(icp.query('decision-engine','info'))[0].hex()
    if bundle!=corpus['pack_manifest_sha256']:raise Failure('pack binding differs')
    entry=next(t for t in manifest['tensors'] if t['name']=='encoder.0.qkv.weight')
    with (ROOT/'checkpoints/laya-int8/model.bin').open('rb') as f:
        f.seek(entry['offset']);raw=f.read(entry['length'])
    if hashlib.sha256(raw).digest()!=bytes(entry['sha256']):raise Failure('QKV tensor digest')
    outputs,cols=entry['shape'];width=outputs//groups
    weights=np.frombuffer(raw,dtype=np.int8,count=outputs*cols).reshape(outputs,cols).copy()
    sw=np.frombuffer(raw,dtype='<f4',offset=outputs*cols).copy()
    torch.set_num_threads(1);w=torch.from_numpy(weights.astype(np.float32)).T
    fixtures=[]
    training={f'{schema}-natural-{i:02}' for schema in ('choice','noul','score') for i in (0,1,2,4,5,6,8,9)}
    for case in corpus['cases']:
        state=QuerySession(icp,case['input'],activation_format='int8').advance()['state']
        n=len(case['input']['input_ids']);offset=60+4*(n+len(case['input']['markers']))
        sx=np.frombuffer(state,dtype='<f4',offset=offset,count=n).copy()
        x=np.frombuffer(state,dtype=np.int8,offset=offset+4*n).reshape(n,cols).copy()
        # Integer products/partial sums bounded by 1024*128^2=2^24:
        # F32 matmul represents every integer sum exactly for this fixture.
        if cols*128*128>2**24:raise Failure('F32 integer-exact bound exceeded')
        sums=(torch.from_numpy(x.astype(np.float32))@w).numpy().astype('<i4')
        physical=sums.astype(np.float32)*sx[:,None]*sw[None,:]
        dynamic=np.maximum(np.max(np.abs(physical.reshape(n,groups,width)),axis=2)/np.float32(127),np.finfo(np.float32).tiny)
        fixtures.append((case['id'],sums,sx,physical,dynamic))
        print('captured',case['id'],flush=True)
    sy=np.max(np.stack([np.max(d,axis=0) for name,_,_,_,d in fixtures if name in training]),axis=0).astype('<f4')
    metrics=[];benchmarks=[]
    for name,sums,sx,physical,dynamic in fixtures:
        n=len(sx);scales=np.repeat(sy,width)
        fixed=rounded(physical/scales[None,:]);q=np.clip(fixed,-127,127).astype(np.int8)
        dynq=np.clip(rounded(physical/np.repeat(dynamic,width,axis=1)),-127,127)
        diff=q.astype(np.float32)*scales[None,:]-physical
        dyndiff=dynq*np.repeat(dynamic,width,axis=1)-physical
        metrics.append(dict(id=name,split='calibration' if name in training else 'heldout',values=int(q.size),clamped=int(np.count_nonzero(np.abs(fixed)>127)),rmse=float(np.sqrt(np.mean(diff.astype(np.float64)**2))),dynamic_rmse=float(np.sqrt(np.mean(dyndiff.astype(np.float64)**2))),max_abs_error=float(np.max(np.abs(diff)))))
        if name in ('choice-natural-03','choice-boundary-128'):
            payload=struct.pack('<IIII',n,outputs,groups,0)+sums.tobytes()+sx.astype('<f4').tobytes()+sw.astype('<f4').tobytes()+sy.tobytes()
            # Candid: one vec nat8 argument, type table entry vec nat8.
            reply=query_bytes(icp,'benchmark_q8_epilogue',b'DIDL\x01\x6d\x7b\x01\x00'+uleb(len(payload))+payload)
            if 'Err' in reply:raise Failure(reply)
            variants=[]
            for block in re.findall(r'record\s*\{([^{}]+)\}',reply,re.S):
                nm=re.search(r'name\s*=\s*"([^"]+)"',block)
                if not nm:continue
                row={'name':nm.group(1)}
                for field in ('instructions','clamped','rmse','max_abs_error','different_from_fixed_float'):
                    v=re.search(r'\b'+field+r'\s*=\s*([-+\d_.eE]+)',block)
                    if not v:raise Failure('missing benchmark field '+field)
                    value=v.group(1).replace('_','');row[field]=float(value) if field in ('rmse','max_abs_error') else int(value)
                variants.append(row)
            if len(variants)!=3:raise Failure('expected three epilogue variants')
            prepare=int(re.search(r'prepare_instructions\s*=\s*([\d_]+)',reply).group(1).replace('_',''))
            benchmarks.append(dict(id=name,tokens=n,payload_sha256=hashlib.sha256(payload).hexdigest(),prepare_instructions=prepare,variants=variants))
    ending=json.loads(icp.run(['canister','status','decision-engine','-e','local','--json']))
    if ending['module_hash']!=status['module_hash']:raise Failure('module changed')
    report=dict(module_hash=status['module_hash'],bundle_sha256=bundle,corpus_sha256=hashlib.sha256(corpuspath.read_bytes()).hexdigest(),tensor_sha256=bytes(entry['sha256']).hex(),scope='first encoder QKV epilogue only; excludes matrix dots; not whole-model calibration or labeled accuracy',groups=groups,group_width=width,calibration_cases=sorted(training),heldout_cases=72,output_scales=sy.tolist(),cases=metrics,benchmarks=benchmarks)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(benchmarks,indent=2))

if __name__=='__main__':main()
