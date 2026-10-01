#!/usr/bin/env python3
"""Rerun historical BOOM DAO prompts on the warm local canister, read-only."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from types import SimpleNamespace

from tokenizers import Tokenizer
from canister_infer import Icp, ROOT, decode_blobs, require_local_network
from check_practical_laya import make_input
from client_held_query import QuerySession
from measure_inference import Failure
from sns_proposal_triage import API, _load_proposal, _laya_context, triage

SNS_ROOT = "xjngq-yaaaa-aaaaq-aabha-cai"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--project-root", type=Path, default=ROOT / "build/client-query-project")
    parser.add_argument("--identity", default="ic-laya-query-local-test")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats must be 1..20")
    out = args.output_dir
    if (out / "benchmark.json").exists():
        raise Failure("use a new output directory to preserve prior measurements")
    out.mkdir(parents=True, exist_ok=True)
    pack = ROOT / "checkpoints/laya-int8"
    manifest = (pack / "manifest.json").read_bytes()
    tokenizer_raw = (pack / "tokenizer.json").read_bytes()
    if bytes(json.loads(manifest)["tokenizer_sha256"]) != hashlib.sha256(tokenizer_raw).digest():
        raise Failure("tokenizer differs from pack")
    tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
    icp = Icp(args.project_root.resolve(), "local", args.identity)
    require_local_network(icp)
    before = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    bundle = decode_blobs(icp.query("decision-engine", "info"))[0].hex()
    if bundle != sha(manifest) or before["module_hash"] != "0x" + sha((ROOT / "build/decision-engine.wasm").read_bytes()):
        raise Failure("installed model or Wasm differs from local files")
    binary = ROOT / "target/release/laya-infer"
    report = dict(measured_at=datetime.now(timezone.utc).isoformat(), network="local",
                  canister=before["id"], module_hash=before["module_hash"], bundle_sha256=bundle,
                  tokenizer_sha256=sha(tokenizer_raw), native_binary_sha256=sha(binary.read_bytes()),
                  query_cache_controlled=False, labeled_accuracy_measured=False,
                  repeats=args.repeats, completed=False, cases=[],
                  source_sha256={f: sha((ROOT / "tools" / f).read_bytes()) for f in
                                 ("benchmark_boomdao_queries.py", "sns_proposal_triage.py", "check_practical_laya.py",
                                  "client_held_query.py", "query_transport.py", "canister_infer.py", "measure_inference.py")},
                  scope="Historical structured summaries, not full proposals; raw logits are supplementary. "
                        "Query timings include CLI and uncontrolled cache. Payload excludes reply envelopes and final logits.")
    for number in (617, 620, 653):
        proposal = _load_proposal(SimpleNamespace(input=None, sns_root=SNS_ROOT, proposal_id=str(number)))
        if str(proposal.get("id")) != str(number):
            raise Failure("API proposal ID differs")
        raw = (json.dumps(proposal, ensure_ascii=False, indent=2) + "\n").encode()
        (out / f"proposal-{number}.json").write_bytes(raw)
        result = triage(proposal)
        row = dict(proposal_id=number, source_url=f"{API}/{SNS_ROOT}/proposals/{number}",
                   snapshot_sha256=sha(raw), triage=result)
        context = _laya_context(proposal, result)
        if context is None:
            row["model_status"] = "not_applicable_to_existing_prompt"
        else:
            question, options, state = context
            inp = make_input(tokenizer, "choice", question, options, state)
            if len(inp["input_ids"]) > 128:
                raise Failure("prompt exceeds 128 tokens; do not truncate")
            input_raw = (json.dumps(inp, indent=2) + "\n").encode()
            input_path = out / f"input-{number}.json"
            input_path.write_bytes(input_raw)
            row.update(question=question, options=options, state=state,
                       input_tokens=len(inp["input_ids"]), input_sha256=sha(input_raw), modes={})
            native = json.loads(subprocess.check_output([str(binary), str(pack), str(input_path)], text=True, timeout=120))
            if native["bundle"] != bundle:
                raise Failure("native model differs")
            row["native"] = dict(logits=native["raw_logits"], label=options[max(range(len(options)), key=lambda i: native["raw_logits"][i])])
            trials = {mode: [] for mode in ("f32", "int8")}
            for repeat in range(args.repeats):
                for mode in ("f32", "int8"):
                    trial = QuerySession(icp, inp, steps=None, activation_format=mode, compression="auto").finish()
                    if trial["bundle_sha256"] != bundle or trial["failed_query_attempts"]:
                        raise Failure("unexpected model or query failure")
                    if trials[mode] and trial["logits"] != trials[mode][0]["logits"]:
                        raise Failure("logits changed between repeats")
                    trials[mode].append(trial)
                    print(number, mode, repeat + 1, "calls", trial["inference_query_calls"], "instructions", trial["instructions"], flush=True)
            for mode, results in trials.items():
                first = results[0]
                row["modes"][mode] = dict(label=options[max(range(len(options)), key=lambda i: first["logits"][i])],
                    logits=first["logits"], query_calls=first["inference_query_calls"],
                    median_instructions=statistics.median(r["instructions"] for r in results),
                    max_query_handler_instructions=max(r["max_observed_query_handler_instructions"] for r in results),
                    median_payload_bytes=statistics.median(r["request_candid_bytes"]+r["response_continuation_bytes"] for r in results),
                    median_local_wall_seconds=statistics.median(r["local_wall_seconds"] for r in results), trials=results)
            row["f32_native_max_logit_difference"] = max(abs(a-b) for a,b in zip(row["native"]["logits"], row["modes"]["f32"]["logits"]))
        report["cases"].append(row)
        (out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    after = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if after["module_hash"] != before["module_hash"] or decode_blobs(icp.query("decision-engine", "info"))[0].hex() != bundle:
        raise Failure("model or Wasm changed during benchmark")
    report["completed"] = True
    (out / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
