#!/usr/bin/env python3
"""Measure experimental INT8 queries against the fixed F32-activation corpus.

Only runs against an already warmed local test canister. Agreement with the
baseline is a compatibility measurement, not labeled-task accuracy.
"""
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=96)
    args = parser.parse_args()
    if not 1 <= args.limit <= 96:
        parser.error("--limit must be 1..96")
    icp = Icp(args.project_root.resolve(), "local", args.identity)
    require_local_network(icp)
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    module = "0x" + hashlib.sha256((ROOT / "build/decision-engine.wasm").read_bytes()).hexdigest()
    if status["module_hash"] != module:
        raise Failure("installed module differs from build artifact")
    corpus_path = ROOT / "artifacts/int8_optimization_v4/validation-corpus.json"
    baseline_path = ROOT / "artifacts/int8_optimization_v4/baseline-canister-corpus.json"
    corpus = json.loads(corpus_path.read_text())
    baseline_metadata = json.loads(baseline_path.read_text())
    baseline = {row["id"]: row["logits"] for row in baseline_metadata["cases"]}
    bundle = decode_blobs(icp.query("decision-engine", "info"))[0].hex()
    if bundle != corpus["pack_manifest_sha256"]:
        raise Failure("pack differs from corpus")
    if baseline_metadata["bundle_sha256"] != bundle or baseline_metadata["corpus_sha256"] != hashlib.sha256(corpus_path.read_bytes()).hexdigest():
        raise Failure("canister reference binding differs from corpus")
    report = dict(module_hash=module, bundle_sha256=bundle,
                  corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                  measured_at=datetime.now(timezone.utc).isoformat(), network="local",
                  query_cache_controlled=False, labeled_accuracy_measured=False,
                  baseline_scope="paired F32 queries on the same Wasm; also checked against bound historical canister results", cases=[])
    if args.output.exists():
        old = json.loads(args.output.read_text())
        for key in ("module_hash", "bundle_sha256", "corpus_sha256", "baseline_sha256"):
            if old[key] != report[key]:
                raise Failure(f"{key} changed; cannot resume")
        report["cases"] = old["cases"]
    done = {case["id"] for case in report["cases"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for case in corpus["cases"][:args.limit]:
        if case["id"] in done:
            continue
        f32 = QuerySession(icp, case["input"], 2, activation_format="f32").finish()
        measured = QuerySession(icp, case["input"], 2, activation_format="int8").finish()
        reference = f32["logits"]
        f32_reference_error = max(abs(a-b) for a,b in zip(reference,baseline[case["id"]]))
        if f32_reference_error > 0.002:
            raise Failure(f"F32 canister reference changed: {case['id']}: {f32_reference_error}")
        logits = measured["logits"]
        if len(reference) != len(logits):
            raise Failure("logit count changed")
        match = max(range(len(logits)), key=logits.__getitem__) == max(range(len(reference)), key=reference.__getitem__)
        ordered = sorted(reference, reverse=True)
        row = dict(id=case["id"], schema=case["schema"], kind=case["kind"],
                   baseline_logits=reference, max_abs_error=max(abs(a-b) for a,b in zip(reference,logits)),
                   f32_reference_max_abs_error=f32_reference_error, f32_instructions=f32["instructions"],
                   decision_matches=match, baseline_top_two_margin=ordered[0]-ordered[1], **measured)
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(f"{len(report['cases'])}/{args.limit} {case['id']}: match={match}, error={row['max_abs_error']:.6f}", flush=True)
    ending = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if ending["module_hash"] != module or decode_blobs(icp.query("decision-engine", "info"))[0].hex() != bundle:
        raise Failure("module or pack changed during measurement")
    errors = [c["max_abs_error"] for c in report["cases"]]
    report.update(complete=len(report["cases"]) == 96, worst_abs_error=max(errors), median_abs_error=statistics.median(errors),
                  worst_f32_reference_abs_error=max(c["f32_reference_max_abs_error"] for c in report["cases"]),
                  decision_disagreements=[c["id"] for c in report["cases"] if not c["decision_matches"]],
                  max_query_handler_instructions=max(c["max_observed_query_handler_instructions"] for c in report["cases"]),
                  module_unchanged=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
