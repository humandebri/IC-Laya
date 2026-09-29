#!/usr/bin/env python3
"""Compare local upstream Laya F32 and runtime W8A8 on identical tokens.

Requires a reviewed local upstream common.py, a local HF checkpoint snapshot,
transformers, and a release laya-infer binary. No remote code is executed.
This small parity sample is not a language-quality or calibration benchmark.
"""
import argparse
import gc
import hashlib
import importlib.metadata
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--max-abs-error", type=float, default=0.15,
                   help="INT8 versus upstream F32 limit for this fixed sample (default: 0.15)")
    p.add_argument("--int8", type=Path, required=True)
    p.add_argument("--binary", type=Path, default=Path("target/release/laya-infer"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if not math.isfinite(args.max_abs_error) or args.max_abs_error < 0:
        p.error("--max-abs-error must be finite and nonnegative")
    spec = importlib.util.spec_from_file_location("laya_common", args.upstream)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    config = AutoConfig.from_pretrained(args.source / "encoder", local_files_only=True)
    torch.set_num_threads(4)
    with torch.device("meta"):
        encoder = AutoModel.from_config(config, attn_implementation="eager")
        model = common.DecisionModel(encoder)
    model.load_state_dict(load_file(args.source / "model.safetensors"), assign=True, strict=True)
    model = model.float().eval()
    # Rotary inverse frequencies are nonpersistent buffers; meta construction
    # leaves them uninitialized even after load_state_dict. Rebuild on CPU.
    from transformers.models.modernbert.modeling_modernbert import ModernBertRotaryEmbedding
    for name, module in list(model.named_modules()):
        if isinstance(module, ModernBertRotaryEmbedding):
            parent_name, _, attr = name.rpartition(".")
            setattr(model.get_submodule(parent_name), attr, ModernBertRotaryEmbedding(config, device="cpu"))
    tok = AutoTokenizer.from_pretrained(args.source / "tokenizer", local_files_only=True)
    questions = [
        {"t": "noul", "ins": "Does the customer ask for a refund?"},
        {"t": "choice", "ins": "Choose the handling action.", "crit": {"refund": "process refund", "reject": "reject request", "review": "human review"}},
        {"t": "score", "ins": "Rate the urgency.", "crit": ["low", "medium", "high"]},
    ]
    state = "I was charged twice. Please refund the duplicate payment."
    cases = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        samples = [(q["t"], q, state) for q in questions]
        samples.append(("choice-128", questions[1], "The customer requests a refund. " * 40))
        for label, q, sample_state in samples:
            ids, markers = common.build_sequence(tok, sample_state, q, max_len=128, head_max_len=64)
            inp = {"input_ids": ids, "markers": markers, "qtype_id": common.QTYPES[q["t"]]}
            path = args.output.parent / f"laya-{label}-input.json"
            path.write_text(json.dumps(inp) + "\n")
            logits, _ = model(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long), torch.tensor([markers]), torch.ones(1, len(markers), dtype=torch.bool), torch.tensor([inp["qtype_id"]]))
            cases.append({"case": label, "primitive": q["t"], "tokens": len(ids), "input": str(path), "upstream": logits[0].tolist()})
    del model, encoder
    gc.collect()
    for case in cases:
        for label, pack in [("int8", args.int8)]:
            result = subprocess.run([str(args.binary), str(pack), case["input"]], check=True, capture_output=True, text=True)
            values = json.loads(result.stdout)["raw_logits"]
            if len(values) != len(case["upstream"]) or not values or not all(math.isfinite(v) for v in values + case["upstream"]):
                raise SystemExit("Invalid logit shape or nonfinite values")
            case[label] = values
            case[label + "_max_abs_error"] = max(abs(a-b) for a,b in zip(case["upstream"], values))
            case[label + "_argmax_matches"] = max(range(len(values)), key=values.__getitem__) == max(range(len(values)), key=case["upstream"].__getitem__)
        print(json.dumps(case), flush=True)
    report = {"cases": cases, "quality_benchmark": False, "max_abs_error_limit": args.max_abs_error,
              "measured_at": datetime.now(timezone.utc).isoformat(),
              "source_repo": "convaiinnovations/laya-typed-decisions",
              "source_revision": json.loads((args.int8 / "manifest.json").read_text())["source_revision"],
              "upstream_common_sha256": hashlib.sha256(args.upstream.read_bytes()).hexdigest(),
              "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "safetensors", "numpy"]},
              "int8_bundle_sha256": hashlib.sha256((args.int8 / "manifest.json").read_bytes()).hexdigest()}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    if any(c["int8_max_abs_error"] > args.max_abs_error or not c["int8_argmax_matches"] for c in cases):
        raise SystemExit("INT8 upstream parity failed")


if __name__ == "__main__":
    main()
