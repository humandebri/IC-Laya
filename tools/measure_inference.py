#!/usr/bin/env python3
"""Measure real inference cost of a sized pack on a local replica.

Why this exists: the design's acceptance targets (20B instructions per question,
warm heap <= 2.5 GiB, cold peak <= 3.0 GiB) have never been measured. The
decision engine already records `measured_instructions` per receipt, but nothing
read it. This tool does, and it reports the numbers rather than a pass/fail
opinion.

It uses the sized synthetic packs from `generate_fixtures.py --tier`, which run
through exactly the same `laya-candle` kernels as the real checkpoint. That makes
the *cost per layer and per token* meaningful while avoiding an 803 MiB download.
It says nothing about decision quality or about upstream parity.

Two independent limits are probed, and they are not the same thing:
  * load/upload/warm-up: whether the pack can be brought into wasm memory at all,
    and where it fails if not (each `warmup_next` is one tensor, so the failing
    tensor index is the boundary);
  * inference: `measured_instructions` for one question over the Compact128
    profile.

Usage:
    python3 tools/measure_inference.py --tier measure-s --tier measure-m
    python3 tools/measure_inference.py --tier measure-l --timeout 3600

Requires a candle-enabled decision engine: `IC_LAYA_CANDLE=1 bash
tools/build_one.sh decision-engine`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"


class Failure(Exception):
    pass


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def check(condition: bool, label: str) -> None:
    print(f"  {'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        raise Failure(label)


# --- canonical hashing, ported from ic-laya-core ---------------------------------

def blob(value: bytes) -> str:
    """Render bytes as a Candid blob literal.

    Each byte becomes `\\xx`. The quotes are added manually: `json.dumps` would
    escape the backslashes themselves and produce `\\\\xx`, which Candid reads as
    a literal backslash followed by garbage.
    """
    return 'blob "' + "".join(f"\\{byte:02x}" for byte in value) + '"'


def principal(p: str) -> str:
    return f"principal {json.dumps(p)}"


def principal_of(raw: bytes) -> str:
    """Render bytes as a valid principal in text form.

    Candid principals are `crc32be(bytes) || bytes` in base32 with the standard
    alphabet, no padding, and a dash every five characters. The dashes are not
    cosmetic: the Candid parser rejects ungrouped text.
    """
    import base64
    import zlib

    body = raw[:29]
    checksum = zlib.crc32(body).to_bytes(4, "big")
    text = base64.b32encode(checksum + body).decode().lower().rstrip("=")
    return "-".join(text[index:index + 5] for index in range(0, len(text), 5))


class Icp:
    """Thin `icp` CLI wrapper bound to a fixed, non-default test identity.

    Every call passes `--identity` explicitly, so the test does not depend on
    whichever identity happens to be the default on this machine. The identity is
    created on first use if absent; see `ensure_identity`.

    ICP_HOME is deliberately *not* overridden. The test was originally written to
    point ICP_HOME at a repo-local store, but canister creation needs cycles and
    those live with the operator's funded identity, which an isolated home cannot
    reach on a local replica. `--identity` alone is what removes the ambient
    default dependency.
    """

    def __init__(self, project_root: Path, env: str, identity: str) -> None:
        self.root = project_root
        self.env = env
        self.identity = identity
        # Passing the generated .did makes replies decode with real field names
        # instead of numeric hashes, and turns argument encoding errors into
        # actionable messages rather than silent inferred types.
        self.did = {name: str(BUILD / f"{name}.did") for name in
                    ("decision-engine", "executor", "mock-ledger")}

    def run(self, args: list[str], expect_ok: bool = True, identity: "bool | str" = True,
            timeout: int = 300) -> str:
        command = ["icp", *args, "--project-root-override", str(self.root)]
        environment = dict(os.environ)
        # Only some subcommands accept --identity; `network` manages the replica
        # itself and `identity` is how a principal is discovered, so both must be
        # invoked without it.
        if identity and args[0] not in {"network", "identity"}:
            command += ["--identity", identity if isinstance(identity, str) else self.identity]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                   env=environment)
        if expect_ok and completed.returncode != 0:
            raise Failure(f"{' '.join(command)}\n{completed.stdout}\n{completed.stderr}")
        return completed.stdout + completed.stderr

    def call(self, canister: str, method: str, args: str = "()", expect_ok: bool = True,
             timeout: int = 300) -> str:
        return self.run(["canister", "call", canister, method, args, "-e", self.env,
                         "--candid", self.did[canister]], expect_ok=expect_ok, timeout=timeout)

    def query(self, canister: str, method: str, args: str = "()") -> str:
        return self.run(["canister", "call", canister, method, args, "-e", self.env,
                         "--candid", self.did[canister], "--query"])


def decode_blobs(output: str) -> list[bytes]:
    """Collect every `blob "..."` payload, decoding Candid escape sequences.

    icp prints blob bytes as `\\xx` escapes, not hex, so the escapes are the only
    thing to parse. Non-escaped characters are appended verbatim.
    """
    import re

    blobs = []
    for body in re.findall(r'blob\s+"((?:[^"\\]|\\.)*)"', output):
        collected = bytearray()
        index = 0
        while index < len(body):
            if body[index] == "\\" and index + 2 <= len(body):
                collected.append(int(body[index + 1:index + 3], 16))
                index += 3
            else:
                collected.extend(body[index].encode())
                index += 1
        blobs.append(bytes(collected))
    return blobs

# Same fixture tokenizer the integration test uses, so the rendered prefix is the
# one the design's Compact128 profile actually produces.
SPECIAL = {"cls": 1, "sep": 2, "mask": 3, "pad": 0}
SPECIAL_LITERALS = ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]

# A representative three-signal workflow (the refund plan from the design), so the
# measurement covers the real sequential shape rather than one synthetic question.
SCHEMAS = [
    ("RefundRequested", "Noul", 1, "Does the customer ask for a refund?",
     [("false", "false"), ("true", "true")]),
    ("PaymentAction", "Choice", 0, "Choose the handling action for this request.",
     [("execute", "process refund"), ("reject", "reject request"), ("review", "human review")]),
    ("PaymentRisk", "Score", 2, "Rate request risk from minimal to severe.",
     [("minimal", "minimal risk"), ("low", "low risk"), ("medium", "moderate risk"),
      ("high", "high risk"), ("severe", "severe risk")]),
]
STATE = "The customer reports a duplicate payment and requests a refund."


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


def read_instructions(receipt_output: str) -> int | None:
    import re
    match = re.search(r"measured_instructions = ([\d_]+) : nat64", receipt_output)
    return int(match.group(1).replace("_", "")) if match else None


def known_identities(icp: "Icp") -> dict[str, str]:
    """Map identity name -> principal.

    `identity list` must not carry `--identity`: it is how a principal is
    discovered in the first place.
    """
    listing = icp.run(["identity", "list", "--json"], identity=False)
    start = listing.find("{")
    if start < 0:
        raise Failure(f"could not read identity list:\n{listing}")
    parsed = json.loads(listing[start:])
    return {entry["name"]: entry["principal"] for entry in parsed["identities"]}


def ensure_identity(icp: "Icp", name: str) -> str:
    """Create `name` if absent and return its principal.

    `--storage plaintext` is used because `keyring` prompts interactively, which
    would hang a non-interactive run, so the key is stored unencrypted in the
    operator's icp home. It is a local-replica key with no funds and must never be
    reused on a real network; the `ic-laya-local-test` name marks that intent.
    """
    known = known_identities(icp)
    if name not in known:
        icp.run(["identity", "new", name, "--storage", "plaintext", "-q"], identity=False)
        known = known_identities(icp)
    if name not in known:
        raise Failure(f"identity {name} missing after creation")
    return known[name]


def canister_principal(icp: "Icp", name: str) -> str | None:
    """Read a canister's principal, or None when it does not exist yet."""
    raw = icp.run(["canister", "status", name, "-e", icp.env, "--json"], expect_ok=False)
    start = raw.find("{")
    if start < 0:
        return None
    try:
        return json.loads(raw[start:]).get("id")
    except ValueError:
        return None


def ensure_cycles(icp: "Icp", amount: str = "10t") -> None:
    """Ensure the test identity can pay for canister creation.

    A freshly created identity has no cycles, so the first canister creation fails
    with "Insufficient cycles". On a local replica `icp cycles mint` funds the
    identity without touching any real ICP; on a real network this is skipped by
    the balance check. Existing balances are left alone rather than topped up on
    every run.
    """
    balance = icp.run(["cycles", "balance", "-e", icp.env], expect_ok=False)
    match = re.search(r"([\d_]+) cycles", balance)
    if match and int(match.group(1).replace("_", "")) > 0:
        return
    print(f"  funding the test identity with {amount} cycles (local replica only)")
    icp.run(["cycles", "mint", "--cycles", amount, "-e", icp.env])


def ensure_canister(icp: "Icp", name: str) -> str:
    """Create the canister if absent, then return its principal.

    `create -q` only prints a principal on first creation, so the id is always
    re-read from `canister status` rather than parsed out of creation output.
    """
    if canister_principal(icp, name) is None:
        icp.run(["canister", "create", name, "-e", icp.env, "-q"])
    principal = canister_principal(icp, name)
    if principal is None:
        raise Failure(f"canister {name} has no principal after create")
    return principal


def any_network_status(icp: "Icp") -> dict | None:
    """Return the configured environment's status, local or not."""
    raw = icp.run(["network", "status", "-e", icp.env, "--json"], expect_ok=False)
    start = raw.find("{")
    if start < 0:
        return None
    try:
        return json.loads(raw[start:])
    except ValueError:
        return None


def require_local_network(icp: "Icp") -> None:
    """Refuse anything that could touch a real network.

    This test mints cycles, installs canisters, and reinstalls them. Pointed at a
    connected network it would attempt exactly that, and a fresh `--identity`
    means the operator would not even notice their own identity was not used. The
    `managed` flag is the only reliable local/remote discriminator, so it is
    enforced here rather than left to the docstring.
    """
    status = any_network_status(icp)
    if status and not status.get("managed"):
        raise Failure(
            f"environment '{icp.env}' points at {status.get('api_url')}, which is not a local "
            f"network. This test mints cycles, installs canisters, and wipes state, so it only "
            f"runs against a locally launched replica.")


def network_status(icp: "Icp") -> dict | None:
    """Return the running *local* network's status, or None when it is not up.

    Requires `managed: true`, which icp-cli sets only for a locally launched
    replica, so a real network is never treated as usable. `require_local_network`
    turns the resulting "not running" into an explicit refusal.
    """
    status = any_network_status(icp)
    if not status or not status.get("managed") or not status.get("api_url"):
        return None
    return status



def measure_tier(icp: Icp, tier: str, args) -> dict:
    directory = ROOT / "fixtures" / tier
    if not (directory / "manifest.json").exists():
        raise Failure(f"fixtures/{tier} missing; run tools/generate_fixtures.py --tier {tier}")
    manifest_raw = (directory / "manifest.json").read_bytes()
    model_raw = (directory / "model.bin").read_bytes()
    tokenizer_raw = (directory / "tokenizer.json").read_bytes()
    manifest = json.loads(manifest_raw)
    if manifest["format"] != "ic-laya-int8-pack-v1":
        raise RuntimeError("Runtime requires INT8; convert this legacy fixture with tools/quantize_pack.py")
    record: dict = {"tier": tier, "pack_bytes": len(model_raw), "tensors": len(manifest["tensors"]),
                    "config": manifest["config"]}

    print(f"  begin_upload ({len(model_raw) / 1024 / 1024:.1f} MiB)")
    started = time.monotonic()
    literals = "; ".join(json.dumps(x) for x in SPECIAL_LITERALS)
    special_arg = (f"record {{ cls = {SPECIAL['cls']} : nat32; sep = {SPECIAL['sep']} : nat32; "
                   f"mask = {SPECIAL['mask']} : nat32; pad = {SPECIAL['pad']} : nat32; "
                   f"literals = vec {{ {literals} }} }}")
    out = icp.call("decision-engine", "begin_upload",
                   f"({blob(manifest_raw)}, {len(tokenizer_raw)} : nat64, {special_arg})",
                   timeout=args.timeout, expect_ok=False)
    if "Ok" not in out:
        record["upload"] = {"ok": False, "response": out.strip()[:300]}
        return record
    record["upload"] = {"ok": True, "seconds": round(time.monotonic() - started, 2)}

    # Chunk size is bounded by argv, not by the canister. `icp canister call` takes
    # the argument as text and Candid renders each byte as \\xx, so a chunk expands
    # roughly 4x before it reaches the command line; 1 MiB (the canister's own
    # limit via upload_chunk) overflows ARG_MAX with "Argument list too long".
    # A real client would pass binary, but this harness goes through the CLI.
    chunk = args.chunk_kib * 1024
    print(f"  upload_chunk x{(len(model_raw) + chunk - 1) // chunk} ({args.chunk_kib} KiB each)")
    offset = 0
    while offset < len(model_raw):
        piece = model_raw[offset:offset + chunk]
        result = icp.call("decision-engine", "upload_chunk",
                          f"({offset} : nat64, {blob(piece)})", timeout=args.timeout)
        if "Err" in result:
            record["upload"]["failed_at_offset"] = offset
            record["upload"]["response"] = result.strip()[:200]
            return record
        offset += len(piece)
    # The blob area is model bytes followed by tokenizer bytes, and start_warmup
    # requires received == model_length + tokenizer_length. Uploading only the model
    # leaves it short and start_warmup answers Transition.
    tokenizer_offset = len(model_raw)
    for start in range(0, len(tokenizer_raw), chunk):
        piece = tokenizer_raw[start:start + chunk]
        result = icp.call("decision-engine", "upload_chunk",
                          f"({tokenizer_offset + start} : nat64, {blob(piece)})",
                          timeout=args.timeout)
        if "Err" in result:
            record["upload"]["failed_at_offset"] = tokenizer_offset + start
            record["upload"]["response"] = result.strip()[:200]
            return record
    record["upload"]["chunks"] = (len(model_raw) + len(tokenizer_raw) + chunk - 1) // chunk

    print("  start_warmup / warmup_next")
    warm = icp.call("decision-engine", "start_warmup", "()", timeout=args.timeout, expect_ok=False)
    if "Ok" not in warm:
        # `start_warmup` requires received == model_length + tokenizer_length; record the
        # arithmetic so a mismatch is diagnosable instead of just "Transition".
        record["warmup"] = {"ok": False, "stage": "start_warmup",
                            "model_bytes": len(model_raw), "tokenizer_bytes": len(tokenizer_raw),
                            "uploaded_bytes": offset, "expected_total": len(model_raw) + len(tokenizer_raw),
                            "response": warm.strip()[:300]}
        return record
    total = read_int(warm) or 0
    started = time.monotonic()
    done = 0
    while True:
        step = icp.call("decision-engine", "warmup_next", "()", timeout=args.timeout,
                        expect_ok=False)
        if "Err" in step:
            entry = manifest["tensors"][done] if done < len(manifest["tensors"]) else None
            record["warmup"] = {
                "ok": False, "stage": "warmup_next", "tensors_done": done, "tensors_total": total,
                "failed_tensor": entry["name"] if entry else None,
                "bytes_loaded": sum(t["length"] for t in manifest["tensors"][:done]),
                "response": step.strip()[:300]}
            return record
        done += 1
        if "true" in step:
            break
    record["warmup"] = {"ok": True, "tensors": total, "seconds": round(time.monotonic() - started, 2)}

    print("  register_schema / register_calibration")
    tokenizer_hash = sha256(tokenizer_raw)
    bundle = sha256(manifest_raw)
    compiled = [compiled_schema(*s, tokenizer_hash) for s in SCHEMAS]
    for schema in compiled:
        out = icp.call("decision-engine", "register_schema",
                       f"({candid_schema(schema)}, {schema['tag']} : nat32)",
                       timeout=args.timeout, expect_ok=False)
        if "Err" in out:
            record["registration"] = {"ok": False, "schema": schema["schema_id"],
                                      "response": out.strip()[:300]}
            return record
    # `EngineState::evaluate` rejects an expiry more than 600s ahead, and also
    # requires the calibration to cover the whole request, so both use the same
    # value derived from the replica's own clock (which is wall clock here).
    now_ns = time.time_ns() + 300_000_000_000
    for schema in compiled:
        calibration = sha256(b"measure-calibration-" + bundle + schema["schema_id"].encode())
        out = icp.call("decision-engine", "register_calibration",
                       "(record { id = %s; model = %s; schema = %s; tokenizer = %s; "
                       "temperature = 1.0; expires_at_ns = %d : nat64; holdout_hash = %s; "
                       "sample_count = 1 : nat64; test_only = true })"
                       % (blob(calibration), blob(bundle), blob(schema["schema_hash"]),
                          blob(tokenizer_hash), now_ns,
                          blob(sha256(b"SYNTHETIC-NOT-REAL-HOLDOUT"))),
                       timeout=args.timeout, expect_ok=False)
        if "Err" in out:
            record["registration"] = {"ok": False, "calibration": schema["schema_id"],
                                      "response": out.strip()[:300]}
            return record
    record["registration"] = {"ok": True, "schemas": len(compiled)}

    print("  evaluate")
    measurements = []
    for index, schema in enumerate(compiled):
        evaluation_id = sha256(f"measure-{tier}-{schema['schema_id']}".encode())
        calibration = sha256(b"measure-calibration-" + bundle + schema["schema_id"].encode())
        request = ("(record { evaluation_id = %s; schema_hash = %s; model = %s; "
                   "calibration = opt %s; binding = null; state = %s; expires_at_ns = %d : nat64 })"
                   % (blob(evaluation_id), blob(schema["schema_hash"]), blob(bundle),
                      blob(calibration), json.dumps(STATE), now_ns))
        started = time.monotonic()
        out = icp.call("decision-engine", "evaluate", request, timeout=args.timeout,
                       expect_ok=False)
        elapsed = round(time.monotonic() - started, 2)
        instructions = read_instructions(out)
        measurements.append({"schema": schema["schema_id"], "primitive": schema["primitive"],
                             "wall_seconds": elapsed, "measured_instructions": instructions,
                             "tokens": None})
        if instructions is None:
            measurements[-1]["response"] = out.strip()[:300]
        # `input_tokens` tells us the rendered length, which is what the cost scales with.
        import re
        tokens = re.search(r"input_tokens = ([\d_]+) : nat32", out)
        if tokens:
            measurements[-1]["tokens"] = int(tokens.group(1).replace("_", ""))
        print(f"    {schema['schema_id']:<18} {elapsed:>8.2f}s  "
              f"{instructions if instructions is not None else 'ERR'} instr")
    record["evaluations"] = measurements
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", action="append", default=[],
                        help="tier to measure; repeatable (default measure-s, measure-m)")
    parser.add_argument("--identity", default="ic-laya-measure")
    parser.add_argument("--env", default="local")
    parser.add_argument("--timeout", type=int, default=1800, help="per-call timeout in seconds")
    parser.add_argument("--chunk-kib", type=int, default=64,
                        help="upload chunk size; bounded by command-line limits, not by the "
                             "canister (default 64)")
    parser.add_argument("--skip-build", action="store_true",
                        help="do not rebuild the candle canister first")
    args = parser.parse_args()
    tiers = args.tier or ["measure-s", "measure-m"]

    if not args.skip_build:
        print("building decision-engine with the candle feature ...")
        target_dir = os.environ.get("CARGO_TARGET_DIR", "/tmp/target")
        subprocess.run(["bash", "tools/build_one.sh", "decision-engine"], cwd=ROOT, check=True,
                       env={**os.environ, "IC_LAYA_CANDLE": "1", "CARGO_TARGET_DIR": target_dir,
                            "PATH": f"{Path.home()}/.cargo/bin:" + os.environ["PATH"]})
    if not (BUILD / "decision-engine.wasm").exists():
        raise Failure("build/decision-engine.wasm missing")

    icp = Icp(ROOT, args.env, args.identity)
    require_local_network(icp)
    status = network_status(icp)
    started_here = status is None
    if started_here:
        print("starting local network ...")
        icp.run(["network", "start", "-d", "-e", args.env])
    else:
        print("reusing the running local network")
    try:
        owner = ensure_identity(icp, args.identity)
        ensure_cycles(icp)
        ensure_canister(icp, "decision-engine")
        # Reinstall so a previous tier's tensors cannot be mistaken for this one's.
        wasm_size = (BUILD / "decision-engine.wasm").stat().st_size / 1024 / 1024
        print(f"installing decision-engine ({wasm_size:.1f} MiB, reinstall)")
        icp.run(["canister", "install", "decision-engine", "-e", icp.env, "-y", "-m", "reinstall",
                 "--wasm", str(BUILD / "decision-engine.wasm"), "--args", f"({principal(owner)})"],
                timeout=max(args.timeout, 600))

        # `evaluate` requires the caller to be an allowed engine caller. This harness
        # calls it directly as the test identity, so that identity must be admitted
        # (admit the engine would be useless here: msg_caller is the CLI identity).
        admission = icp.run(["canister", "call", "decision-engine", "allow_caller",
                             f"({principal(owner)}, 1000 : nat32)", "-e", icp.env,
                             "--candid", str(BUILD / "decision-engine.did")], expect_ok=False)
        if "Ok" not in admission:
            raise Failure(f"could not admit the measurement identity: {admission.strip()[:200]}")

        results = []
        for tier in tiers:
            print(f"\n=== {tier} ===")
            try:
                results.append(measure_tier(icp, tier, args))
            except Failure as failure:
                results.append({"tier": tier, "harness_error": str(failure)[:400]})
                print(f"  harness error: {failure}")

        out = ROOT / "artifacts/inference_measurements.json"
        out.write_text(json.dumps({"kind": "local replica measurement of synthetic sized packs",
                                   "note": "instructions are measured by the canister; wall clock "
                                           "is a local-replica figure and is not a mainnet estimate",
                                   "targets": {"instructions_per_question": 20_000_000_000,
                                               "update_limit": 40_000_000_000,
                                               "warm_heap_bytes": 2_500_000_000,
                                               "cold_peak_bytes": 3_000_000_000},
                                   "tiers": results}, indent=2) + "\n")
        print(f"\nwrote {out.relative_to(ROOT)}")
    finally:
        if started_here:
            print("stopping local network ...")
            icp.run(["network", "stop", "-e", args.env], expect_ok=False)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as failure:
        print(f"\nFAILED: {failure}", file=sys.stderr)
        raise SystemExit(1)
