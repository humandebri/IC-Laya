#!/usr/bin/env python3
"""Verify lossless query transport and automatic grouping on the warm local pack."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from canister_infer import Icp, ROOT, decode_blobs, require_local_network, warmup
from client_held_query import QuerySession
from measure_inference import Failure
from query_transport import AUTO_BUNDLE


def validation_binding(root, cases):
    """Bind reusable measurements to all inputs and the Python execution path."""
    sources = ("tools/client_held_query.py", "tools/query_transport.py",
               "tools/canister_infer.py", "tools/measure_inference.py",
               "tools/validate_query_transport.py")
    return dict(
        input_sha256={name: hashlib.sha256(json.dumps(inp, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                      for name, inp in cases},
        client_source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                              for name in sources})


def resume_cases(old, report):
    for key in ("module_hash", "bundle_sha256", "corpus_sha256", "input_sha256", "client_source_sha256"):
        if key not in old or old[key] != report[key]:
            raise Failure(f"cannot resume: {key} is missing or differs; use a new output path")
    rows = old.get("cases")
    if not isinstance(rows, list):
        raise Failure("invalid saved validation cases")
    seen = set()
    for row in rows:
        name = row.get("id")
        if name not in report["input_sha256"] or name in seen or row.get("input_sha256") != report["input_sha256"][name]:
            raise Failure("saved validation case input binding differs")
        seen.add(name)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT / "build/client-query-project")
    parser.add_argument("--identity", default="ic-laya-query-local-test")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--corpus", action="store_true", help="also run all 96 bound corpus inputs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    icp = Icp(args.project_root.resolve(), "local", args.identity)
    require_local_network(icp)
    if args.warmup:
        warmup(icp)
    module = "0x" + hashlib.sha256((ROOT / "build/decision-engine.wasm").read_bytes()).hexdigest()
    if json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))["module_hash"] != module:
        raise Failure("installed module differs from build")
    bundle = decode_blobs(icp.query("decision-engine", "info"))[0].hex()
    if bundle != AUTO_BUNDLE:
        raise Failure("pack is not calibrated")
    corpus_path = ROOT / "artifacts/int8_optimization_v4/validation-corpus.json"
    corpus = json.loads(corpus_path.read_text())
    previous = json.loads((ROOT / "artifacts/references/int8-final-corpus.json").read_text())
    historical_f32 = json.loads((ROOT / "artifacts/int8_optimization_v4/baseline-canister-corpus.json").read_text())
    if historical_f32["bundle_sha256"] != bundle or historical_f32["corpus_sha256"] != hashlib.sha256(corpus_path.read_bytes()).hexdigest() or previous["bundle_sha256"] != bundle or previous["corpus_sha256"] != hashlib.sha256(corpus_path.read_bytes()).hexdigest():
        raise Failure("reference binding differs")
    f32_reference = {c["id"]: c["logits"] for c in historical_f32["cases"]}
    int8_reference = {c["id"]: c["logits"] for c in previous["cases"]}
    cases = [("pilot-128", json.loads((ROOT / "artifacts/laya-choice-128-input.json").read_text()))]
    for name in ("choice-natural-03", "choice-boundary-65", "score-boundary-112"):
        cases.append((name, next(c["input"] for c in corpus["cases"] if c["id"] == name)))
    for length in (16, 128):
        cases.append((f"seven-markers-{length}", dict(input_ids=[50281]+[50284]*7+[50283]*(length-9)+[50282], markers=list(range(1,8)), qtype_id=0)))
    cases.append(("noul", json.loads((ROOT / "artifacts/laya-noul-input.json").read_text())))
    if args.corpus:
        cases.extend((c["id"], c["input"]) for c in corpus["cases"] if c["id"] not in {name for name, _ in cases})
    report = dict(module_hash=module, bundle_sha256=bundle, corpus_sha256=hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                  measured_at=datetime.now(timezone.utc).isoformat(), network="local", query_cache_controlled=False,
                  labeled_accuracy_measured=False, cases=[], completed=False,
                  payload_scope="successful Candid requests + continuation blobs; failed attempt requests counted separately")
    report.update(validation_binding(ROOT, cases))
    if args.output.exists():
        old = json.loads(args.output.read_text())
        report["cases"] = resume_cases(old, report)
    done = {c["id"] for c in report["cases"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, inp in cases:
        if name in done:
            continue
        row = dict(id=name, input_sha256=report["input_sha256"][name], modes={})
        for mode in ("f32", "int8"):
            compressed = QuerySession(icp, inp, steps=None, activation_format=mode, compression="auto").finish()
            raw = QuerySession(icp, inp, steps=None, activation_format=mode, compression="none").finish()
            if compressed["logits"] != raw["logits"]:
                raise Failure(f"codec changed logits: {name} {mode}")
            reference = (f32_reference if mode == "f32" else int8_reference).get(name)
            if reference is not None and compressed["logits"] != reference:
                raise Failure(f"historical logits changed: {name} {mode}")
            if compressed["failed_query_attempts"]:
                raise Failure(f"auto schedule required instruction fallback: {name} {mode}")
            details = dict(compressed=compressed, raw_auto=raw, exact_logits=True)
            if name in {n for n, _ in cases[:7]}:
                fixed = QuerySession(icp, inp, 2, activation_format=mode).finish()
                if fixed["logits"] != compressed["logits"]:
                    raise Failure("grouping changed logits")
                details["raw_two_steps"] = fixed
            row["modes"][mode] = details
            print(name, mode, "calls", compressed["inference_query_calls"], "bytes",
                  compressed["request_candid_bytes"]+compressed["response_continuation_bytes"],
                  "max", compressed["max_observed_query_handler_instructions"], flush=True)
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2)+"\n")
    if json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))["module_hash"] != module or decode_blobs(icp.query("decision-engine", "info"))[0].hex() != bundle:
        raise Failure("model/module changed")
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
