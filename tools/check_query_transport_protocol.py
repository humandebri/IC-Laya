#!/usr/bin/env python3
"""Verify compressed state bounds, exact retries, ownership and query isolation."""
import argparse
import json
from pathlib import Path
import struct
import zlib

from canister_infer import Icp, ROOT, require_local_network
from check_client_held_query_protocol import query_bytes
from client_held_query import QuerySession, decode_progress, encode_continue, encode_input
from measure_inference import Failure
from query_transport import MAX_STATE_BYTES, unpack_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    icp = Icp(ROOT / "build/client-query-project", "local", "ic-laya-query-local-test")
    intruder = Icp(icp.root, "local", "ic-laya-query-local-intruder")
    require_local_network(icp)
    module = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))["module_hash"]
    inp = json.loads((ROOT / "artifacts/laya-choice-128-input.json").read_text())
    start = icp.call("decision-engine", "start_token_inference", "(record { input_ids = vec { "+";".join(map(str,inp["input_ids"]))+" }; markers = vec { 10;15;20 }; qtype_id = 0 : nat32 })")
    if "Ok" not in start:
        raise Failure(start)
    before = icp.query("decision-engine", "token_inference_status")
    report = dict(module_hash=module, modes={})
    for mode in ("f32", "int8"):
        session = QuerySession(icp, inp, 3, activation_format=mode, compression="auto")
        session.advance()
        wire = session.progress["state"]
        if not wire.startswith(b"LAYZ"):
            raise Failure("expected compressed pilot state")
        method = "continue_token_inference_compressed_query" if mode == "f32" else "continue_token_inference_int8_compressed_query"
        begin = method.replace("continue", "begin")
        fused_begin = begin.replace("_compressed_query", "_batch_compressed_query")
        fused_payload = encode_input(inp,3)
        fused_a = decode_progress(query_bytes(icp,fused_begin,fused_payload))
        fused_b = decode_progress(query_bytes(icp,fused_begin,fused_payload))
        separated = decode_progress(query_bytes(icp,method.replace("_compressed", ""),encode_continue(unpack_state(wire),3)))
        if fused_a["completed"] != 3 or fused_a["state"] != fused_b["state"] or unpack_state(fused_a["state"]) != separated["state"]:
            raise Failure("fused begin differs from separate begin and continue")
        payload = encode_continue(wire, 3)
        a = decode_progress(query_bytes(icp, method, payload))
        b = decode_progress(query_bytes(icp, method, payload))
        if a["state"] != b["state"]:
            raise Failure("exact retry differs")
        raw_method = method.replace("_compressed", "")
        raw_result = decode_progress(query_bytes(icp, raw_method, encode_continue(unpack_state(wire), 3)))
        if unpack_state(a["state"]) != raw_result["state"]:
            raise Failure("compressed boundary bits differ")
        malformed = dict(truncated=wire[:-1], trailing=wire+b"x", checksum=wire[:-1]+bytes([wire[-1]^1]))
        for name, offset, value in [("version",4,2),("codec",8,99),("declared_size",12,60),("oversized_expansion",12,MAX_STATE_BYTES+1)]:
            bad = bytearray(wire); bad[offset:offset+4] = struct.pack("<I",value); malformed[name] = bytes(bad)
        malformed["expansion_bomb"] = b"LAYZ"+struct.pack("<III",1,1,60)+zlib.compress(b"x"*(MAX_STATE_BYTES+1))
        rejected = []
        for name, bad in malformed.items():
            response = query_bytes(icp, method, encode_continue(bad,1))
            if "Err" not in response:
                raise Failure(f"accepted malformed transport: {mode} {name}")
            rejected.append(name)
        for value in (0,17,0xffffffff):
            bad = encode_continue(wire,1)[:-4] + struct.pack("<I",value)
            if "Err" not in query_bytes(icp, method, bad):
                raise Failure("accepted invalid batch width")
            bad_begin = encode_input(inp,1)[:-4] + struct.pack("<I",value)
            if "Err" not in query_bytes(icp,fused_begin,bad_begin):
                raise Failure("accepted invalid fused begin width")
        for target, data in [(begin,encode_input(inp)),(method,encode_continue(wire,1)),(fused_begin,fused_payload),
                             (fused_begin.replace("_compressed", ""),fused_payload)]:
            if "Unauthorized" not in query_bytes(intruder,target,data):
                raise Failure("accepted non-owner compressed query")
        # Retaining the exact bytes works across both compressed query sessions.
        result = session.finish()
        report["modes"][mode] = dict(malformed_rejected=rejected, exact_retry=True, exact_boundary_bits=True,
                                    fused_begin_exact=True, fused_begin_retry_exact=True, fused_begin_bounds=True,
                                    nonowner_rejected=True, completed=True, logits=result["logits"])
    if before != icp.query("decision-engine", "token_inference_status"):
        raise Failure("compressed query modified update job")
    if module != json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))["module_hash"]:
        raise Failure("module changed")
    report["update_job_unchanged"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
