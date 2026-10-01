#!/usr/bin/env python3
"""Compare lossless state codecs locally; no compressed canister API is assumed."""
import argparse
import gzip
import json
from pathlib import Path
import struct
import time

from canister_infer import Icp, ROOT, require_local_network
from client_held_query import QuerySession
from measure_inference import Failure


def shuffle_f32(raw):
    tokens, markers = struct.unpack_from("<II", raw, 52)
    offset = 60 + 4 * (tokens + markers)
    values = raw[offset:]
    if raw[:4] != b"LAYQ" or len(values) % 4:
        raise Failure("not a F32 continuation")
    return raw[:offset] + b"".join(values[i::4] for i in range(4))


def unshuffle_f32(raw):
    tokens, markers = struct.unpack_from("<II", raw, 52)
    offset = 60 + 4 * (tokens + markers)
    planes = raw[offset:]
    n = len(planes) // 4
    restored = bytearray(len(planes))
    for i in range(4):
        restored[i::4] = planes[i*n:(i+1)*n]
    return raw[:offset] + restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    measurement = json.loads((ROOT / "artifacts/query_conditions/measurements.json").read_text())
    case = next(c for c in measurement["cases"] if c["id"] == "pilot-128")
    icp = Icp(ROOT / "build/client-query-project", "local", "ic-laya-query-local-test")
    require_local_network(icp)
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if status["module_hash"] != measurement["module_hash"]:
        raise Failure("module changed")
    inp = json.loads((ROOT / "artifacts/laya-choice-128-input.json").read_text())
    report = dict(module_hash=measurement["module_hash"], bundle_sha256=measurement["bundle_sha256"],
                  input_tokens=128, compression_scope="Python host only; canister codec not implemented", modes={})
    for mode in ("f32", "int8"):
        plan = case["modes"][mode]["trials"]["planned"]
        session = QuerySession(icp, inp, activation_format=mode)
        session.advance()
        rows = []
        for width in plan["requested_batch_schedule"]:
            raw = session.progress["state"]
            records = {}
            for shuffled in (False, True) if mode == "f32" else (False,):
                for level in (1, 6, 9):
                    start = time.perf_counter()
                    prepared = shuffle_f32(raw) if shuffled else raw
                    encoded = gzip.compress(prepared, compresslevel=level, mtime=0)
                    encode_time = time.perf_counter() - start
                    start = time.perf_counter()
                    restored = gzip.decompress(encoded)
                    restored = unshuffle_f32(restored) if shuffled else restored
                    decode_time = time.perf_counter() - start
                    if restored != raw:
                        raise Failure("codec changed state bytes")
                    records[f"{'shuffle_' if shuffled else ''}gzip_{level}"] = dict(bytes=len(encoded), local_encode_seconds=encode_time, local_decode_seconds=decode_time)
            rows.append(dict(completed=session.progress["completed"], raw_bytes=len(raw), codecs=records))
            session.steps = width
            session.advance()
        if not session.progress["done"] or session.progress["logits"] != plan["logits"] or session.bundle != report["bundle_sha256"]:
            raise Failure("inference or pack changed")
        total = 2 * sum(r["raw_bytes"] for r in rows)
        codecs = {}
        for name in rows[0]["codecs"]:
            compressed = 2 * sum(r["codecs"][name]["bytes"] for r in rows)
            codecs[name] = dict(continuation_roundtrip_bytes=compressed, reduction_percent=100*(1-compressed/total),
                local_encode_seconds=sum(r["codecs"][name]["local_encode_seconds"] for r in rows),
                local_decode_seconds=sum(r["codecs"][name]["local_decode_seconds"] for r in rows))
        report["modes"][mode] = dict(raw_continuation_roundtrip_bytes=total, codecs=codecs, states=rows,
            logits=plan["logits"], exact_codec_roundtrip=True)
        print(mode, json.dumps(codecs), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
