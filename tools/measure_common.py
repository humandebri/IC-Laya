"""Shared machinery for measuring IC-Laya inference on a local replica.

Both `tools/measure_inference.py` (totals per question) and
`tools/measure_phases.py` (per-phase breakdown) need the same sized-pack upload,
schema compilation, and Candid rendering. This module holds exactly that.

`fixture_encode` and `Canonical` are ports of the Rust fixture so the rendered
Compact128 prefix matches what the canister would compile; if
`crates/ic-laya-core/src/demo.rs` changes, these must change with it.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

# Same fixture tokenizer the integration test uses.
SPECIAL = {"cls": 1, "sep": 2, "mask": 3, "pad": 0}
SPECIAL_LITERALS = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def fixture_encode(text: str) -> list[int]:
    """Port of FixtureTokenizer::encode_piece (crates/ic-laya-core/src/demo.rs)."""
    ids = []
    for word in text.split():
        ids.append(10 + int.from_bytes(sha256(word.encode())[:2], "big"))
    return ids


class Canonical:
    """Port of src/types.rs `Canonical`: length-framed, domain-separated hashing."""

    def __init__(self, domain: str) -> None:
        self.buf = bytearray()
        self.bytes(domain.encode())

    def bytes(self, value: bytes) -> "Canonical":
        self.buf += len(value).to_bytes(8, "big") + value
        return self

    def u64(self, value: int) -> "Canonical":
        self.buf += value.to_bytes(8, "big")
        return self

    def text(self, value: str) -> "Canonical":
        return self.bytes(value.encode())

    def finish(self) -> bytes:
        return sha256(bytes(self.buf))


def blob(value: bytes) -> str:
    """Render bytes as a Candid blob literal.

    Each byte becomes `\\xx`. The quotes are added manually: `json.dumps` would
    escape the backslashes themselves and produce `\\\\xx`, which Candid reads as
    a literal backslash followed by garbage.
    """
    return 'blob "' + "".join(f"\\{byte:02x}" for byte in value) + '"'


def principal(p: str) -> str:
    return f"principal {json.dumps(p)}"


def compiled_schema(schema_id: str, primitive: str, tag: int, instructions: str,
                    options: list[tuple[str, str]], tokenizer_hash: bytes) -> dict:
    """Port of schema::compile for the fixture tokenizer."""
    prefix = [SPECIAL["cls"], *fixture_encode(f"{primitive.lower()} question: {instructions}"),
              SPECIAL["sep"]]
    markers = []
    for _, text in options:
        markers.append(len(prefix))
        prefix.append(SPECIAL["mask"])
        prefix.extend(fixture_encode(text))
    prefix.append(SPECIAL["sep"])
    h = Canonical("ic-laya/schema/v1")
    h.text(schema_id).u64(1).u64(tag).text(instructions).u64(len(options))
    for option_id, text in options:
        h.text(option_id).text(text)
    h.bytes(tokenizer_hash).text("Compact128-v1")
    return {"schema_id": schema_id, "primitive": primitive, "tag": tag,
            "instructions": instructions, "options": options,
            "prefix": prefix, "markers": markers, "schema_hash": h.finish(),
            "tokenizer_hash": tokenizer_hash}


def candid_schema(schema: dict) -> str:
    options = "; ".join("record { id = %s; text = %s }" % (json.dumps(i), json.dumps(t))
                        for i, t in schema["options"])
    return ("record { id = %s; version = 1 : nat64; primitive = variant { %s }; "
            "instructions = %s; options = vec { %s } }"
            % (json.dumps(schema["schema_id"]), schema["primitive"],
               json.dumps(schema["instructions"]), options))


def candid_compiled(schema: dict) -> str:
    special = ("record { cls = %d : nat32; sep = %d : nat32; mask = %d : nat32; "
               "pad = %d : nat32; literals = vec { %s } }"
               % (SPECIAL["cls"], SPECIAL["sep"], SPECIAL["mask"], SPECIAL["pad"],
                  "; ".join(json.dumps(x) for x in SPECIAL_LITERALS)))
    return ("record { schema = %s; schema_hash = %s; tokenizer_hash = %s; "
            "prefix = vec { %s }; markers = vec { %s }; special = %s; qtype_id = %d : nat32 }"
            % (candid_schema(schema), blob(schema["schema_hash"]), blob(schema["tokenizer_hash"]),
               "; ".join(str(v) for v in schema["prefix"]),
               "; ".join(str(v) for v in schema["markers"]), special, schema["tag"]))


def read_int(output: str) -> int | None:
    """Pull a nat result out of an icp reply, e.g. `(27 : nat64)` or `(27)`."""
    import re
    match = re.search(r"\(\s*([\d_]+)\s*(?::\s*nat(?:8|16|32|64)?\s*)?\)", output)
    return int(match.group(1).replace("_", "")) if match else None


def upload_pack(icp, tier: str, chunk_kib: int, timeout: int) -> tuple[dict, bytes]:
    """Upload fixtures/<tier> into the engine and warm it up.

    Returns (record, manifest_raw). Raises RuntimeError with the failing stage on error.
    """
    directory = ROOT / "fixtures" / tier
    if not (directory / "manifest.json").exists():
        raise RuntimeError(f"fixtures/{tier} missing; run tools/generate_fixtures.py --tier {tier}")
    manifest_raw = (directory / "manifest.json").read_bytes()
    model_raw = (directory / "model.bin").read_bytes()
    tokenizer_raw = (directory / "tokenizer.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest["format"] != "ic-laya-int8-pack-v1":
        raise RuntimeError("Runtime requires INT8; convert this legacy fixture with tools/quantize_pack.py")
    record: dict = {"tier": tier, "pack_bytes": len(model_raw),
                    "tensors": len(manifest["tensors"]), "config": manifest["config"]}

    literals = "; ".join(json.dumps(x) for x in SPECIAL_LITERALS)
    special_arg = (f"record {{ cls = {SPECIAL['cls']} : nat32; sep = {SPECIAL['sep']} : nat32; "
                   f"mask = {SPECIAL['mask']} : nat32; pad = {SPECIAL['pad']} : nat32; "
                   f"literals = vec {{ {literals} }} }}")
    started = time.monotonic()
    out = icp.call("decision-engine", "begin_upload",
                   f"({blob(manifest_raw)}, {len(tokenizer_raw)} : nat64, {special_arg})",
                   timeout=timeout, expect_ok=False)
    if "Ok" not in out:
        raise RuntimeError(f"begin_upload rejected: {out.strip()[:200]}")
    record["upload"] = {"ok": True, "seconds": round(time.monotonic() - started, 2)}

    # Chunk size is bounded by argv, not by the canister: `icp canister call` passes
    # the argument as text and Candid renders each byte as \xx, so the canister's own
    # 1 MiB limit overflows ARG_MAX. A binary client would not have this constraint.
    chunk = chunk_kib * 1024
    for start in range(0, len(model_raw) + len(tokenizer_raw), chunk):
        blob_bytes = (model_raw + tokenizer_raw)[start:start + chunk]
        result = icp.call("decision-engine", "upload_chunk",
                          f"({start} : nat64, {blob(blob_bytes)})", timeout=timeout)
        if "Err" in result:
            record["upload"]["failed_at_offset"] = start
            raise RuntimeError(f"upload_chunk at {start}: {result.strip()[:200]}")
    record["upload"]["chunks"] = (len(model_raw) + len(tokenizer_raw) + chunk - 1) // chunk

    warm = icp.call("decision-engine", "start_warmup", "()", timeout=timeout, expect_ok=False)
    if "Ok" not in warm:
        raise RuntimeError(f"start_warmup rejected: {warm.strip()[:200]}")
    started = time.monotonic()
    done = 0
    total = read_int(warm) or 0
    while True:
        step = icp.call("decision-engine", "warmup_next", "()", timeout=timeout, expect_ok=False)
        if "Err" in step:
            entry = manifest["tensors"][done] if done < len(manifest["tensors"]) else None
            record["warmup"] = {"ok": False, "tensors_done": done, "tensors_total": total,
                                "failed_tensor": entry["name"] if entry else None,
                                "bytes_loaded": sum(t["length"] for t in manifest["tensors"][:done]),
                                "response": step.strip()[:200]}
            raise RuntimeError(f"warmup_next failed at tensor {done} ({entry['name'] if entry else '?'})")
        done += 1
        if "true" in step:
            break
    record["warmup"] = {"ok": True, "tensors": total, "seconds": round(time.monotonic() - started, 2)}
    return record, manifest_raw


def register_schemas(icp, manifest_raw: bytes, tokenizer_raw: bytes, schemas, timeout: int) -> list[dict]:
    """Register the three schemas and their calibrations against the loaded bundle."""
    tokenizer_hash = sha256(tokenizer_raw)
    bundle = sha256(manifest_raw)
    compiled = [compiled_schema(*s, tokenizer_hash) for s in schemas]
    for schema in compiled:
        out = icp.call("decision-engine", "register_schema",
                       f"({candid_schema(schema)}, {schema['tag']} : nat32)",
                       timeout=timeout, expect_ok=False)
        if "Err" in out:
            raise RuntimeError(f"register_schema {schema['schema_id']}: {out.strip()[:200]}")
    # Engine throttles per minute; each tier uses its own window so the three
    # questions below never trip the quota.
    now_ns = time.time_ns() + 300_000_000_000
    for schema in compiled:
        calibration = sha256(b"measure-calibration-" + bundle + schema["schema_id"].encode())
        schema["calibration"] = calibration
        out = icp.call("decision-engine", "register_calibration",
                       "(record { id = %s; model = %s; schema = %s; tokenizer = %s; "
                       "temperature = 1.0; expires_at_ns = %d : nat64; holdout_hash = %s; "
                       "sample_count = 1 : nat64; test_only = true })"
                       % (blob(calibration), blob(bundle), blob(schema["schema_hash"]),
                          blob(tokenizer_hash), now_ns, blob(sha256(b"SYNTHETIC-NOT-REAL-HOLDOUT"))),
                       timeout=timeout, expect_ok=False)
        if "Err" in out:
            # A second tier in the same run registers under a new bundle, so the id
            # differs; a conflict here means the same bundle was measured twice.
            raise RuntimeError(f"register_calibration {schema['schema_id']}: {out.strip()[:200]}")
    return compiled, bundle, now_ns


def decision_request(schema: dict, bundle: bytes, evaluation_id: bytes, state: str, now_ns: int) -> str:
    return ("(record { evaluation_id = %s; schema_hash = %s; model = %s; calibration = opt %s; "
            "binding = null; state = %s; expires_at_ns = %d : nat64 })"
            % (blob(evaluation_id), blob(schema["schema_hash"]), blob(bundle),
               blob(schema["calibration"]), json.dumps(state), now_ns))
