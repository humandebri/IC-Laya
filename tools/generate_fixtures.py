#!/usr/bin/env python3
"""Generate small, redistributable random-weight fixtures and numerical vectors.

Default behaviour (no `--tier`) writes exactly the two tiny fixtures and the
numeric vectors that the test suite depends on. Output for those paths must stay
byte-identical; `MANIFEST.sha256` and `tests/test_reference.py` pin them.

`--tier` additionally writes sized synthetic INT8 packs under `fixtures/<tier>/` for
performance measurement. Those are random weights of the same architecture at a
larger scale, NOT Laya, and they say nothing about decision quality; they exist
so `tools/measure_inference.py` can measure instruction cost through the real
`laya-candle` code path without the 803 MiB upstream checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from reference_math import from_logits, score_stats
from model_reference import shapes, torch_forward, numpy_forward

ROOT = Path(__file__).resolve().parents[1]

# Vocabulary must be large enough to hold the five special tokens plus the
# distinct token ids the measurement inputs use (0..~127).
MIN_VOCAB = 128

TINY = {
    "vocab_size": 128, "hidden_size": 8, "layers": 2, "attention_heads": 2,
    "intermediate_size": 12, "norm_eps": 1e-5, "global_every": 2, "local_attention": 4,
    "global_rope_theta": 10000.0, "local_rope_theta": 1000.0,
    "first_layer_attention_norm": False, "decision_layers": 2, "decision_heads": 2,
    "decision_ff": 16, "decision_norm_eps": 1e-5, "decision_norm_first": True,
    "decision_activation": "Relu", "scorer_norm_eps": 1e-5, "qtypes": 3,
    "mask_token_id": 3,
}

# Sized tiers for measurement. hidden_size and the per-layer shapes are what drive
# cost, so `measure-l` deliberately uses the real checkpoint's hidden_size (1024)
# and rope thetas to make the layer-count extrapolation meaningful. Layer count and
# vocab are reduced so the pack stays small enough to upload over the CLI.
TIERS = {
    "measure-s": dict(TINY, hidden_size=128, layers=2, attention_heads=4,
                      intermediate_size=512, vocab_size=512, global_every=2,
                      local_attention=64, mask_token_id=3, decision_heads=4,
                      decision_ff=128),
    "measure-m": dict(TINY, hidden_size=512, layers=4, attention_heads=8,
                      intermediate_size=2048, vocab_size=2048, global_every=4,
                      local_attention=128, mask_token_id=3, decision_heads=8,
                      decision_ff=512),
    # The candidate for the conformance target: 6 layers at the real checkpoint's
    # hidden width, 16 heads, sliding window 128, and the intermediate ratio the real
    # checkpoint uses (2624/1024). vocab is reduced only to keep the pack small.
    "measure-6l768": dict(TINY, hidden_size=768, layers=6, attention_heads=16,
                          intermediate_size=1968, vocab_size=4096, global_every=3,
                          local_attention=128, global_rope_theta=160000.0,
                          local_rope_theta=10000.0, mask_token_id=3, decision_heads=16,
                          decision_ff=1024),
    "measure-l": dict(TINY, hidden_size=1024, layers=2, attention_heads=16,
                      intermediate_size=2624, vocab_size=4096, global_every=3,
                      local_attention=128, global_rope_theta=160000.0,
                      local_rope_theta=10000.0, mask_token_id=3, decision_heads=16,
                      decision_ff=1024),
}


def sha(b: bytes) -> list[int]:
    return list(hashlib.sha256(b).digest())


def build_weights(cfg: dict, seed: int) -> dict:
    torch.manual_seed(seed)
    weights = {}
    for name, shape in shapes(cfg).items():
        x = torch.randn(shape) * 0.07
        if name.endswith(".weight") and "norm" in name:
            x = 1 + x * .1
        weights[name] = x.contiguous()
    return weights


def build_tokenizer(vocab_size: int) -> bytes:
    specials = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
    vocab = {t: i for i, t in enumerate(specials)}
    vocab.update({f"t{i}": i for i in range(len(specials), vocab_size)})
    tokenizer = {
        "version": "1.0", "truncation": None, "padding": None,
        "added_tokens": [{"id": i, "content": t, "single_word": False, "lstrip": False,
                          "rstrip": False, "normalized": False, "special": True}
                         for i, t in enumerate(specials)],
        "normalizer": None, "pre_tokenizer": {"type": "WhitespaceSplit"},
        "post_processor": None, "decoder": None,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"},
    }
    return (json.dumps(tokenizer, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def write_pack(directory: Path, cfg: dict, weights: dict, tokenizer_raw: bytes,
               revision: str) -> int:
    entries = []
    blob = bytearray()
    for name, x in sorted(weights.items()):
        raw = x.numpy().astype("<f4").tobytes()
        entries.append({"name": name, "shape": list(x.shape), "offset": len(blob),
                        "length": len(raw), "sha256": sha(raw)})
        blob.extend(raw)
    manifest = {"format": "ic-laya-f32-pack-v1",
                "source_repo": "synthetic/random-weights-not-Laya",
                "source_revision": revision, "test_only": True,
                "tokenizer_sha256": sha(tokenizer_raw), "primitive_to_qtype": [0, 1, 2],
                "config": cfg, "total_bytes": len(blob), "tensors": entries}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "model.bin").write_bytes(blob)
    (directory / "tokenizer.json").write_bytes(tokenizer_raw)
    return len(blob)


def measurement_cases(cfg: dict, weights: dict) -> list[dict]:
    """Inputs covering every primitive, sized to the real Compact128 budget.

    Token ids stay below the tier's vocab_size, and markers sit on mask positions
    so `LayaModel::infer` accepts them.
    """
    specials = {"pad": 0, "cls": 1, "sep": 2, "mask": 3}
    total_tokens = min(128, 128)
    cases = []
    for qtype, options in ((0, 3), (1, 2), (2, 5)):
        ids = [specials["cls"]]
        # Fill to leave room for the final separator, then plant mask markers.
        ids += [5 + (i % (cfg["vocab_size"] - 5)) for i in range(total_tokens - 2 - options)]
        markers = []
        for _ in range(options):
            markers.append(len(ids))
            ids.append(specials["mask"])
        ids.append(specials["sep"])
        if len(ids) > 128:
            raise AssertionError("measurement input exceeds the 128-token profile")
        cases.append({"input": {"input_ids": ids, "markers": markers, "qtype_id": qtype},
                      "options": options})
    return cases


def generate_tiny() -> dict:
    """Existing behaviour: the two tiny fixtures plus numerical vectors."""
    torch.set_num_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    weights = build_weights(TINY, seed=19)
    tokenizer_raw = build_tokenizer(TINY["vocab_size"])
    metrics = []
    for pre in [True, False]:
        c = {**TINY, "decision_norm_first": pre}
        directory = ROOT / "fixtures" / ("tiny-prenorm" if pre else "tiny-postnorm")
        directory.mkdir(parents=True, exist_ok=True)
        write_pack(directory, c, weights, tokenizer_raw, "seed-19")
        cases = []
        for qtype, n, t in [(0, 3, 16), (1, 2, 13), (2, 3, 19), (2, 5, 21), (2, 7, 29)]:
            ids = [1] + [5 + i % 100 for i in range(t - 2)] + [2]
            markers = [2 + i * 2 for i in range(n)]
            for m in markers:
                ids[m] = 3
            inp = {"input_ids": ids, "markers": markers, "qtype_id": qtype}
            with torch.no_grad():
                logits = torch_forward(c, weights, inp)
            independent = numpy_forward(c, {k: v.numpy() for k, v in weights.items()}, inp)
            delta = float(np.max(np.abs(logits - independent)))
            if delta > 2e-5:
                raise AssertionError((pre, qtype, n, delta))
            cases.append({"input": inp, "expected_logits": logits.tolist(), "atol": 1e-3,
                          "rtol": 1e-3,
                          "provenance": "PyTorch synthetic reference; NOT pretrained Laya"})
            metrics.append({"pre_norm": pre, "qtype": qtype, "options": n, "tokens": t,
                            "pytorch_numpy_max_abs": delta})
        (directory / "cases.json").write_text(json.dumps(cases, indent=2) + "\n")
        (directory / "input.json").write_text(json.dumps(cases[0]["input"], indent=2) + "\n")
    vectors = []
    for logits, temp in [([0., 0.], 1.), ([0., 0., 0.], 1.), ([0.] * 5, 1.), ([0.] * 7, 1.),
                         ([4., -2., -3.], 1.), ([-4., 4.], 1.), ([5., 1., 0., -2., -4.], 1.),
                         ([2., 1., 0.], 2.), ([100., -100.], .5),
                         ([-1., -2., -3., -4., -5., -6., -7.], 1.)]:
        mass = from_logits(logits, temp)
        v = {"logits": logits, "temperature": temp, "mass_ppm": mass}
        if len(mass) >= 3:
            v["score"] = score_stats(mass)
        vectors.append(v)
    (ROOT / "fixtures/numerics.json").write_text(json.dumps(vectors, indent=2) + "\n")
    (ROOT / "artifacts/synthetic_neural_checks.json").write_text(
        json.dumps({"kind": "PyTorch versus NumPy on synthetic weights",
                    "torch": torch.__version__, "cases": metrics}, indent=2) + "\n")
    return {"synthetic_model_cases": len(metrics),
            "maximum_abs_error": max(x["pytorch_numpy_max_abs"] for x in metrics),
            "numeric_vectors": len(vectors)}


def generate_tier(name: str) -> dict:
    """Write one sized measurement pack. No PyTorch/NumPy cross-check at this size."""
    if name not in TIERS:
        raise SystemExit(f"unknown tier {name}; choose from {', '.join(TIERS)}")
    torch.set_num_threads(1)
    cfg = TIERS[name]
    weights = build_weights(cfg, seed=hash(name) % (2 ** 31))
    tokenizer_raw = build_tokenizer(cfg["vocab_size"])
    directory = ROOT / "fixtures" / name
    # Runtime packs are INT8; the F32 export exists only during conversion.
    from quantize_pack import quantize
    if directory.exists():
        raise ValueError(f"output already exists: {directory}")
    with tempfile.TemporaryDirectory(prefix="laya-tier-") as tmp:
        source = Path(tmp) / "f32"
        write_pack(source, cfg, weights, tokenizer_raw, f"tier-{name}")
        manifest = quantize(source, directory)
    total = manifest["total_bytes"]
    cases = measurement_cases(cfg, weights)
    (directory / "input.json").write_text(json.dumps(cases[0]["input"], indent=2) + "\n")
    for index, case in enumerate(cases):
        (directory / f"input-{index}.json").write_text(json.dumps(case["input"], indent=2) + "\n")
    return {"tier": name, "total_bytes": total, "tensors": len(weights),
            "inputs": len(cases), "config": {}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", action="append", default=[],
                        help=f"sized measurement pack to generate ({', '.join(TIERS)}); "
                             f"repeatable. Tiny fixtures are always written.")
    args = parser.parse_args()

    summary = generate_tiny()
    for name in args.tier:
        summary[name] = generate_tier(name)
        print(json.dumps(summary[name], indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, dict)}, indent=2))


if __name__ == "__main__":
    main()
