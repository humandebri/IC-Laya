#!/usr/bin/env python3
"""Measure grouping and lossless compression on an already uploaded local pack."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time

from canister_infer import Icp, ROOT, decode_blobs, require_local_network, warmup
from client_held_query import QuerySession
from measure_inference import Failure


def plan_batches(costs, state_sizes, budget, maximum=16):
    """Minimize intermediate bytes using conservative single-step handler costs.

    Costs include repeated continuation decoding/encoding inside the handler,
    making their sum conservative for grouping. CDK work is excluded; budget
    must reserve room for it. This is a measured-input plan, not a universal one.
    """
    n = len(costs)
    best = {n: (0, 0, [])}
    for i in range(n - 1, -1, -1):
        candidates = []
        cost = 0
        for j in range(i + 1, min(n, i + maximum) + 1):
            cost += costs[j - 1]
            if cost > budget:
                break
            if j in best:
                tail = best[j]
                candidates.append((tail[0] + (2 * state_sizes[j] if j < n else 0),
                                   tail[1] + 1, [j - i] + tail[2]))
        if candidates:
            best[i] = min(candidates, key=lambda v: v[:2])
    if 0 not in best:
        raise Failure("a single phase exceeds the selected handler budget")
    return best[0][2]


def payload_bytes(result):
    return result["request_candid_bytes"] + result["response_continuation_bytes"]


def checked_run(icp, inp, mode, steps, expected):
    session = QuerySession(icp, inp, steps[0], activation_format=mode)
    session.advance()
    for width in steps:
        if session.progress["done"]:
            raise Failure("plan has extra phases")
        session.steps = width
        session.advance()
    if not session.progress["done"]:
        raise Failure("plan is incomplete")
    result = session.finish()
    result["requested_batch_schedule"] = steps
    if result["failed_query_attempts"]:
        raise Failure("planned inference needed a retry")
    if result["logits"] != expected:
        raise Failure("grouping changed logits")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT / "build/client-query-project")
    parser.add_argument("--identity", default="ic-laya-query-local-test")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--case", action="append", help="run only named cases")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    icp = Icp(args.project_root.resolve(), "local", args.identity)
    require_local_network(icp)
    if args.warmup:
        warmup(icp)
    module = "0x" + hashlib.sha256((ROOT / "build/decision-engine.wasm").read_bytes()).hexdigest()
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if status["module_hash"] != module:
        raise Failure("installed module differs from build")
    bundle = decode_blobs(icp.query("decision-engine", "info"))[0].hex()
    corpus = json.loads((ROOT / "artifacts/int8_optimization_v4/validation-corpus.json").read_text())
    if corpus["pack_manifest_sha256"] != bundle:
        raise Failure("pack binding differs")
    cases = [("pilot-128", json.loads((ROOT / "artifacts/laya-choice-128-input.json").read_text()))]
    for name in ("choice-natural-03", "choice-boundary-65", "score-boundary-112"):
        cases.append((name, next(c["input"] for c in corpus["cases"] if c["id"] == name)))
    if args.case:
        for length in (16, 128):
            cases.append((f"seven-markers-{length}", dict(input_ids=[50281] + [50284]*7 + [50283]*(length-9) + [50282], markers=list(range(1, 8)), qtype_id=0)))
        cases.append(("noul", json.loads((ROOT / "artifacts/laya-noul-input.json").read_text())))
        if set(args.case) - {name for name, _ in cases}:
            parser.error("unknown case")
        cases = [(name, inp) for name, inp in cases if name in args.case]
    report = dict(module_hash=module, bundle_sha256=bundle,
                  measured_at=datetime.now(timezone.utc).isoformat(), network="local",
                  query_cache_controlled=False, handler_budget=4_500_000_000,
                  payload_scope="Candid requests + continuation blobs; excludes HTTP, reply envelopes, final logits and failed requests",
                  compression_scope="local Python gzip; not implemented or instruction-measured in canister",
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, inp in cases:
        row = dict(id=name, input_sha256=hashlib.sha256(json.dumps(inp, sort_keys=True).encode()).hexdigest(),
                   input_tokens=len(inp["input_ids"]), markers=len(inp["markers"]), modes={})
        for mode in ("f32", "int8"):
            session = QuerySession(icp, inp, 1, activation_format=mode)
            states = {}
            compressed = {level: {} for level in (1, 6, 9)}
            while not session.progress or not session.progress["done"]:
                progress = session.advance()
                if progress["state"] is not None:
                    raw = progress["state"]
                    states[progress["completed"]] = len(raw)
                    for level in compressed:
                        start = time.perf_counter()
                        encoded = gzip.compress(raw, compresslevel=level, mtime=0)
                        elapsed = time.perf_counter() - start
                        if gzip.decompress(encoded) != raw:
                            raise Failure("compression round trip failed")
                        compressed[level][progress["completed"]] = dict(bytes=len(encoded), cpu_seconds=elapsed)
            single = session.finish()
            costs = [c["instructions"] for c in single["query_calls"][1:]]
            optimized = plan_batches(costs, states, report["handler_budget"])
            n = len(costs)
            schedules = {"two_steps": [min(2, n-i) for i in range(0, n, 2)], "planned": optimized}
            trials = {"single_step": single}
            for label, schedule in schedules.items():
                result = checked_run(icp, inp, mode, schedule, single["logits"])
                if result["max_observed_query_handler_instructions"] > report["handler_budget"]:
                    raise Failure("planned handler exceeded reserved budget")
                trials[label] = result
            boundaries = [0]
            step = 0
            for width in optimized:
                step += width
                if step < n:
                    boundaries.append(step)
            compression = {}
            for level, records in compressed.items():
                raw_size = 2 * sum(states[s] for s in boundaries)
                size = 2 * sum(records[s]["bytes"] for s in boundaries)
                compression[str(level)] = dict(raw_continuation_roundtrip_bytes=raw_size,
                    compressed_continuation_roundtrip_bytes=size,
                    reduction_percent=100 * (1-size/raw_size),
                    local_encode_cpu_seconds=sum(records[s]["cpu_seconds"] for s in boundaries))
            row["modes"][mode] = dict(trials=trials, compression=compression,
                                     single_phase_handler_costs=costs, state_sizes=states)
            print(name, mode, "schedule", optimized, "bytes", payload_bytes(trials["planned"]),
                  "max instructions", trials["planned"]["max_observed_query_handler_instructions"], flush=True)
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    ending = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if ending["module_hash"] != module or decode_blobs(icp.query("decision-engine", "info"))[0].hex() != bundle:
        raise Failure("module or pack changed")
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
