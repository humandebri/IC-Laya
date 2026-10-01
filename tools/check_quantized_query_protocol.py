#!/usr/bin/env python3
"""Check owner checks, malformed states, exact retries and job isolation locally."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import struct

from canister_infer import Icp, ROOT, require_local_network
from check_client_held_query_protocol import query_bytes
from client_held_query import QuerySession, decode_progress, encode_continue, encode_input
from measure_inference import Failure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--identity",default="ic-laya-query-local-test")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    icp=Icp(args.project_root.resolve(),"local",args.identity)
    require_local_network(icp)
    status=json.loads(icp.run(["canister","status","decision-engine","-e","local","--json"]))
    inp=json.loads((ROOT/"artifacts/laya-choice-128-input.json").read_text())
    started=icp.call("decision-engine","start_token_inference", "(record { input_ids = vec { "+";".join(map(str,inp['input_ids']))+" }; markers = vec { 10;15;20 }; qtype_id = 0 : nat32 })")
    if "Ok" not in started: raise Failure(started)
    job_before=icp.query("decision-engine","token_inference_status")
    session=QuerySession(icp,inp,1,activation_format="int8")
    session.advance();state=session.progress["state"]
    scale_start=60+4*(len(inp["input_ids"])+len(inp["markers"]))
    malformed={}
    for name,offset in [("magic",0),("version",4),("revision",8),("bundle",12),("phase",44),("qtype",48),("tokens",52),("markers",56),("token_id",60)]:
        bad=bytearray(state);bad[offset:offset+4]=b"\xff"*4;malformed[name]=bytes(bad)
    for name,scale in [("zero_scale",0.),("negative_scale",-1.),("nan_scale",float('nan')),("inf_scale",float('inf')),("overflow_scale",3e38)]:
        bad=bytearray(state);bad[scale_start:scale_start+4]=struct.pack("<f",scale);malformed[name]=bytes(bad)
    malformed.update(truncated=state[:-1],trailing=state+b"\0")
    rejected=[]
    for name,bad in malformed.items():
        reply=query_bytes(icp,"continue_token_inference_int8_query",encode_continue(bad,1))
        try:decode_progress(reply)
        except Failure:rejected.append(name)
        else:raise Failure(f"accepted malformed state: {name}")
    for steps in [0,17,0xffffffff]:
        payload=encode_continue(state,1)[:-4]+struct.pack("<I",steps)
        reply=query_bytes(icp,"continue_token_inference_int8_query",payload)
        if "Err" not in reply:raise Failure("accepted invalid step count")
    payload=encode_continue(state,1)
    a=decode_progress(query_bytes(icp,"continue_token_inference_int8_query",payload))
    b=decode_progress(query_bytes(icp,"continue_token_inference_int8_query",payload))
    if a["state"]!=b["state"] or a["completed"]!=b["completed"]:raise Failure("retry changed state")
    intruder=Icp(args.project_root.resolve(),"local","ic-laya-query-local-intruder")
    for method,payload in [("begin_token_inference_int8_query",encode_input(inp)),("continue_token_inference_int8_query",encode_continue(state,1)),("profile_token_inference_int8_query",encode_continue(state,1))]:
        if "Unauthorized" not in query_bytes(intruder,method,payload):raise Failure("nonowner query was not rejected")
    if icp.query("decision-engine","token_inference_status")!=job_before:raise Failure("queries changed the update job")
    ending=json.loads(icp.run(["canister","status","decision-engine","-e","local","--json"]))
    if ending["module_hash"]!=status["module_hash"]:raise Failure("module changed during protocol checks")
    report=dict(module_hash=status["module_hash"],network="local",measured_at=datetime.now(timezone.utc).isoformat(),
                malformed_rejected=rejected,invalid_steps_rejected=True,exact_retry=True,unauthorized_rejected=True,update_job_unchanged=True)
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
