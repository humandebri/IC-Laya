#!/usr/bin/env python3
"""Compare F32 and INT8 query activations on one warmed local model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from canister_infer import Icp, ROOT, decode_blobs, require_local_network
from client_held_query import QuerySession
from measure_inference import Failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--identity", default="ic-laya-query-local-test")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be 1..20")
    icp = Icp(args.project_root.resolve(), "local", args.identity)
    require_local_network(icp)
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    module = "0x"+hashlib.sha256((ROOT/"build/decision-engine.wasm").read_bytes()).hexdigest()
    if status["module_hash"] != module:
        raise Failure("installed module differs from build artifact")
    bundle = decode_blobs(icp.query("decision-engine", "info"))[0].hex()
    inp = json.loads(args.input.read_text())
    trials = []
    for i in range(args.repeats):
        trial = {}
        for mode in (("f32", "int8") if i % 2 == 0 else ("int8", "f32")):
            trial[mode] = QuerySession(icp, inp, 2, activation_format=mode).finish()
            if trial[mode]["bundle_sha256"] != bundle:
                raise Failure("model changed")
        trials.append(trial)
        print(f"trial {i+1}/{args.repeats}: f32={trial['f32']['local_wall_seconds']:.3f}s, int8={trial['int8']['local_wall_seconds']:.3f}s", flush=True)
    ending = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if ending["module_hash"] != module or decode_blobs(icp.query("decision-engine", "info"))[0].hex() != bundle:
        raise Failure("module or model changed during measurement")
    report = dict(module_hash=module, bundle_sha256=bundle, network="local",
                  measured_at=datetime.now(timezone.utc).isoformat(),
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(), trials=trials,
                  query_cache_controlled=False, timing_scope="local CLI, queries and parsing; repeated queries may be cached",
                  medians_seconds={mode:statistics.median(t[mode]["local_wall_seconds"] for t in trials) for mode in ("f32", "int8")},
                  median_instructions={mode:statistics.median(t[mode]["instructions"] for t in trials) for mode in ("f32", "int8")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "trials"},indent=2))


if __name__ == "__main__":
    main()
