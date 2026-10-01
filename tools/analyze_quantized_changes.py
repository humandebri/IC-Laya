#!/usr/bin/env python3
"""Trace changed decisions and controlled phase swaps on a warmed local model.

Uses existing owner-only query APIs; does not install or modify the model.
Phase swaps are diagnostic counterfactuals, not production inference modes.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct

import numpy as np

from canister_infer import Icp, ROOT, require_local_network
from check_client_held_query_protocol import query_bytes
from client_held_query import QuerySession, decode_progress, encode_continue
from measure_inference import Failure


def unpack(state, width):
    next_step, qtype, tokens, markers = struct.unpack_from("<IIII", state, 44)
    rows = markers if next_step == 31 else tokens
    start = 60 + 4*(tokens+markers)
    if state[:4] == b"LAYQ":
        values = np.frombuffer(state, dtype="<f4", offset=start).reshape(rows, width)
    elif state[:4] == b"LAYI":
        scales = np.frombuffer(state, dtype="<f4", offset=start, count=rows)
        values = np.frombuffer(state, dtype=np.int8, offset=start+4*rows).reshape(rows,width).astype(np.float32)*scales[:,None]
    else:
        raise Failure("unsupported state format")
    return values


def convert(state, width, mode):
    values = unpack(state, width)
    if mode == "f32":
        return b"LAYQ"+struct.pack("<II",1,1)+state[12:60+4*sum(struct.unpack_from("<II",state,52))]+values.astype("<f4").tobytes()
    peak = np.max(np.abs(values),axis=1)
    scale = np.where(peak == 0, np.float32(1), np.maximum(peak/np.float32(127), np.finfo(np.float32).tiny)).astype(np.float32)
    divided = values/scale[:,None]
    magnitude = np.abs(divided)
    whole = np.trunc(magnitude)
    rounded = whole + ((magnitude-whole) >= np.float32(.5)).astype(np.float32)
    quantized = np.clip(np.copysign(rounded, divided),-127,127).astype(np.int8)
    end = 60+4*sum(struct.unpack_from("<II",state,52))
    return b"LAYI"+struct.pack("<II",1,2)+state[12:end]+scale.astype("<f4").tobytes()+quantized.tobytes()


def error(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64)
    delta=b-a
    return dict(rmse=float(np.sqrt(np.mean(delta*delta))),max_abs=float(np.max(np.abs(delta))),
                relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(a),1e-30)),
                cosine=float(np.sum(a*b)/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)))


def advance_state(icp,state,mode,steps=1):
    method="continue_token_inference_int8_query" if mode=="int8" else "continue_token_inference_query"
    return decode_progress(query_bytes(icp,method,encode_continue(state,steps)))


def finish_state(icp,state,mode):
    while True:
        result=advance_state(icp,state,mode,2)
        if result["done"]:return result["logits"]
        state=result["state"]


def winner(logits):return max(range(len(logits)),key=logits.__getitem__)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--identity",default="ic-laya-query-local-test")
    parser.add_argument("--corpus-results",type=Path,default=ROOT/"artifacts/references/int8-fusion-corpus.json")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    icp=Icp(args.project_root.resolve(),"local",args.identity);require_local_network(icp)
    status=json.loads(icp.run(["canister","status","decision-engine","-e","local","--json"]))
    measured=json.loads(args.corpus_results.read_text())
    if status["module_hash"]!=measured["module_hash"]:raise Failure("installed module differs from the analyzed corpus")
    corpus=json.loads((ROOT/"artifacts/int8_optimization_v4/validation-corpus.json").read_text())
    inputs={c["id"]:c for c in corpus["cases"]}
    manifest=json.loads((ROOT/"checkpoints/laya-int8/manifest.json").read_text());width=manifest["config"]["hidden_size"]
    if manifest["config"]["layers"]!=28 or manifest["config"]["decision_layers"]!=2:raise Failure("phase mapping differs from fixed pack")
    report=dict(module_hash=status["module_hash"],bundle_sha256=measured["bundle_sha256"],
                measured_at=datetime.now(timezone.utc).isoformat(),network="local",labeled_accuracy_measured=False,cases=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for row in measured["cases"]:
        if row["decision_matches"]:continue
        source=inputs[row["id"]];inp=source["input"]
        sessions={mode:QuerySession(icp,inp,1,activation_format=mode) for mode in ("f32","int8")}
        history={mode:{} for mode in sessions};trace=[];local=[]
        for step in range(33):
            for mode,s in sessions.items():
                s.advance()
                if not s.progress["done"]:history[mode][step]=s.progress["state"]
            if step<32:
                a=unpack(history['f32'][step],width);b=unpack(history['int8'][step],width)
                stats=error(a,b)
                if step<31:stats['marker_rmse']=error(a[inp['markers']],b[inp['markers']])['rmse']
                trace.append(dict(completed=step,**stats))
            if step>0:
                probe=advance_state(icp,convert(history['f32'][step-1],width,'int8'),'int8')
                if probe['done']:
                    local.append(dict(completed=step,logits=probe['logits'],winner=winner(probe['logits'])))
                else:
                    local.append(dict(completed=step,**error(unpack(history['f32'][step],width),unpack(probe['state'],width))))
        ref=sessions['f32'].finish()['logits'];quant=sessions['int8'].finish()['logits']
        ref_delta=max(abs(a-b) for a,b in zip(ref,row['baseline_logits']))
        q_delta=max(abs(a-b) for a,b in zip(quant,row['logits']))
        if ref_delta>0.002 or q_delta>0.002:
            raise Failure(f"traced outputs differ from original corpus: f32={ref_delta}, int8={q_delta}")
        probes={}
        for phase in [28,29,31]:
            for upstream,downstream in [('f32','int8'),('int8','f32')]:
                logits=finish_state(icp,convert(history[upstream][phase],width,downstream),downstream)
                probes[f'{upstream}_through_{phase}__{downstream}_rest']=dict(logits=logits,winner=winner(logits),matches_baseline=winner(logits)==winner(ref))
        old,new=winner(ref),winner(quant)
        result=dict(id=row['id'],schema=row['schema'],kind=row['kind'],text=source.get('text'),tokens=len(inp['input_ids']),
                    baseline_logits=ref,int8_logits=quant,baseline_winner=old,int8_winner=new,
                    original_pair_gap=ref[old]-ref[new],int8_pair_gap=quant[old]-quant[new],
                    pair_gap_change=(quant[old]-quant[new])-(ref[old]-ref[new]),
                    baseline_trace_max_abs_error=ref_delta,int8_trace_max_abs_error=q_delta,
                    trace=trace,local_phase_probes=local,counterfactuals=probes)
        report['cases'].append(result);args.output.write_text(json.dumps(report,indent=2)+'\n')
        print(f"{len(report['cases'])}/{len(measured['decision_disagreements'])} {row['id']}: " + str({k:v['matches_baseline'] for k,v in probes.items()}),flush=True)
    ending=json.loads(icp.run(["canister","status","decision-engine","-e","local","--json"]))
    if ending['module_hash']!=report['module_hash']:raise Failure("module changed during analysis")
    report['complete']=len(report['cases'])==len(measured['decision_disagreements']);args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
