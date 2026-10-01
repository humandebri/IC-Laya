#!/usr/bin/env python3
"""Verify client-held queries against a warmed, fixed-pack local test canister.

Does not install, upload, upgrade, or reset a canister. Starts an update job to
verify query isolation and runs direct/split updates as numerical baselines.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import struct
import tempfile

from canister_infer import infer, decode_blobs
from client_held_query import QuerySession, encode_continue, encode_input
from measure_inference import Icp, ROOT, Failure, require_local_network, ensure_identity


def query_bytes(icp, method, payload, identity=True):
    with tempfile.TemporaryDirectory(prefix="laya-query-check-") as temp:
        path = Path(temp) / "args.bin"
        path.write_bytes(payload)
        return icp.run(["canister", "call", "decision-engine", method, "--query",
                        "--args-file", str(path), "--args-format", "bin", "-e", icp.env,
                        "--candid", icp.did["decision-engine"]], expect_ok=False, identity=identity)


def verify(icp, output, repeats):
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    bundle = decode_blobs(icp.query("decision-engine", "info"))[0].hex()
    intruder_identity = "ic-laya-query-local-intruder"
    if icp.identity == intruder_identity: raise Failure("owner and intruder identities must differ")
    ensure_identity(icp, intruder_identity)
    source = json.loads((ROOT / "artifacts/laya-choice-128-input.json").read_text())
    # Keep an unfinished canister-owned job alive while queries are interleaved.
    start = icp.call("decision-engine", "start_token_inference", "(record { input_ids = vec { " +
                     ";".join(map(str, source["input_ids"])) + " }; markers = vec { 10;15;20 }; qtype_id = 0 : nat32 })")
    if "Ok" not in start: raise Failure(start)
    job_before = icp.query("decision-engine", "token_inference_status")
    session = QuerySession(icp, source)
    first = session.advance()
    raw = first["state"]
    exact = query_bytes(icp, "continue_token_inference_query", encode_continue(raw, 1))
    if exact != query_bytes(icp, "continue_token_inference_query", encode_continue(raw, 1)):
        raise Failure("identical query retry changed")
    rejected = {}
    for name, offset, value in [("version", 4, 99), ("revision", 8, 99), ("bundle", 12, 0),
                                 ("step", 44, 999), ("qtype", 48, 3), ("tokens", 52, 129),
                                 ("markers", 56, 8), ("token_id", 60, 0xffffffff)]:
        broken = bytearray(raw)
        broken[offset:offset + 4] = struct.pack("<I", value)
        reply = query_bytes(icp, "continue_token_inference_query", encode_continue(broken, 1))
        if "Err" not in reply: raise Failure(f"accepted invalid {name}")
        rejected[name] = True
    hidden = 60 + 4 * (len(source["input_ids"]) + len(source["markers"]))
    for name, broken in [("truncated", raw[:-1]), ("trailing", raw + b"\x00"),
                          ("nan", raw[:hidden] + struct.pack("<f", float("nan")) + raw[hidden+4:])]:
        if "Err" not in query_bytes(icp, "continue_token_inference_query", encode_continue(broken, 1)):
            raise Failure(f"accepted invalid {name}")
        rejected[name] = True
    for steps in [0, 17]:
        payload = encode_continue(raw, 1)[:-4] + struct.pack("<I", steps)
        if "Err" not in query_bytes(icp, "continue_token_inference_query", payload):
            raise Failure("accepted invalid step count")
    unauthorized = query_bytes(icp, "begin_token_inference_query", encode_input(source), intruder_identity)
    if "Unauthorized" not in unauthorized: raise Failure("non-owner query accepted")

    # Cover all qtypes, token-length boundaries, and 2..7 marker bounds.
    cases = []
    short = json.loads((ROOT / "artifacts/int8_optimization_v4/short-inputs.json").read_text())["cases"][0]["input"]
    if "Ok" not in query_bytes(icp,"infer_tokens_query",encode_input(short)):
        raise Failure("existing 16-token query regressed")
    long_short = dict(short,input_ids=short["input_ids"]+[50282])
    if "TooLong" not in query_bytes(icp,"infer_tokens_query",encode_input(long_short)):
        raise Failure("existing 17-token guard regressed")
    for length in [16, 17]:
        inp = dict(short, input_ids=short["input_ids"] + ([50282] if length == 17 else []))
        cases.append((f"choice-{length}", inp))
    for length in [28, 64]: cases.append((f"choice-{length}", dict(source, input_ids=source["input_ids"][:length])))
    for name in ["noul", "score"]:
        cases.append((name, json.loads((ROOT / f"artifacts/laya-{name}-input.json").read_text())))
    cases.append(("seven-markers", dict(input_ids=[50281] + [50284]*7 + [50283]*7 + [50282], markers=list(range(1,8)), qtype_id=0)))
    checked = []
    with tempfile.TemporaryDirectory(prefix="laya-query-input-") as temp:
        # Two independently held states are advanced in alternating order.
        a, b = QuerySession(icp, cases[0][1], 2), QuerySession(icp, cases[-1][1], 2)
        while any(s.progress is None or not s.progress["done"] for s in (a,b)):
            for s in (a,b):
                if s.progress is None or not s.progress["done"]: s.advance()
        for name, inp in cases:
            path = Path(temp) / f"{name}.json"; path.write_text(json.dumps(inp))
            baseline = infer(icp, path)
            query = QuerySession(icp, inp, 2).finish()
            if baseline["logits"] != query["logits"]: raise Failure(f"logits differ: {name}")
            result = dict(name=name, baseline=baseline, query=query)
            (output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
            checked.append(name)
            print(f"verified {name}: {query['inference_query_calls']} queries", flush=True)
        if a.finish()["logits"] != json.loads((output / "choice-16.json").read_text())["baseline"]["logits"]:
            raise Failure("interleaved first query changed")
        if b.finish()["logits"] != json.loads((output / "seven-markers.json").read_text())["baseline"]["logits"]:
            raise Failure("interleaved second query changed")

    if icp.query("decision-engine", "token_inference_status") != job_before:
        raise Failure("queries changed canister-owned update job")
    trials = []
    path = ROOT / "artifacts/laya-choice-128-input.json"
    for trial in range(repeats):
        # Alternate ordering; repeated query caching is not controlled.
        modes = ("direct", "update_split", "query_split") if trial % 2 == 0 else ("query_split", "update_split", "direct")
        result = {"trial": trial + 1}
        for mode in modes:
            measured = QuerySession(icp, source, 2).finish() if mode == "query_split" else infer(icp, path, stepped=mode == "update_split")
            if result and any(v["logits"] != measured["logits"] for v in result.values() if isinstance(v,dict)):
                raise Failure("128-token baseline/query mismatch")
            result[mode] = measured
        trials.append(result)
        (output / "benchmark.json").write_text(json.dumps(trials, indent=2) + "\n")
        print(f"128-token trial {trial+1}/{repeats} complete", flush=True)
    after = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if after["module_hash"] != status["module_hash"] or decode_blobs(icp.query("decision-engine","info"))[0].hex() != bundle:
        raise Failure("Wasm or pack changed during verification")
    report = dict(network="local", project_root=str(icp.root), canister=status["id"], module_hash=status["module_hash"],
                  bundle_sha256=bundle, measured_at=datetime.now(timezone.utc).isoformat(),
                  malformed_rejected=rejected, unauthorized_rejected=True, exact_retry=True,
                  interleaved_queries=True, update_job_unchanged=True, logits_equal_cases=checked + ["choice-128"],
                  existing_query_guard_preserved=True,
                  trials=repeats, query_cache_controlled=False,
                  timing_scope="local end-to-end; includes CLI and parsing; repeated queries may be cached",
                  medians_seconds={mode:statistics.median(t[mode]["local_wall_seconds"] for t in trials)
                                   for mode in ("direct", "update_split", "query_split")},
                  max_query_handler_instructions=max(t["query_split"]["max_observed_query_handler_instructions"] for t in trials),
                  input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT)
    parser.add_argument("--identity",default="ic-laya-query-local-test")
    parser.add_argument("--output",type=Path,default=ROOT / "artifacts/client_held_query")
    parser.add_argument("--repeats",type=int,default=10)
    args=parser.parse_args()
    if not 1<=args.repeats<=100: parser.error("--repeats must be 1..100")
    icp=Icp(args.project_root.resolve(),"local",args.identity)
    require_local_network(icp)
    verify(icp,args.output,args.repeats)


if __name__=="__main__": main()
