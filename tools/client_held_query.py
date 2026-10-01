"""Independent query inference with lossless continuation bytes held by the client."""
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time

from measure_inference import Failure
from query_transport import AUTO_BUNDLE, auto_width, inspect_state


def transient_failure(error):
    if isinstance(error, subprocess.TimeoutExpired):
        return True
    message = str(error).lower()
    statuses = re.findall(r"\b(?:status(?:\s+code)?|http(?:\s+error)?)\s*[:=]?\s*(\d{3})\b", message)
    if statuses:
        return all(int(status) in (429, 502, 503, 504) for status in statuses)
    return any(value in message for value in
               ("transport", "timed out", "connection refused", "connection reset",
                "error sending request", "temporarily unavailable"))


def uleb(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field_hash(name):
    result = 0
    for byte in name.encode():
        result = (result * 223 + byte) & 0xffffffff
    return result


def encode_input(inp, begin_steps=None):
    ids, markers, qtype = inp["input_ids"], inp["markers"], inp["qtype_id"]
    if not 1 <= len(ids) <= 128 or not 2 <= len(markers) <= 7 or qtype not in (0, 1, 2):
        raise Failure("query input is outside supported token/marker/qtype bounds")
    if any(type(v) is not int or not 0 <= v <= 0xffffffff for v in [*ids, *markers, qtype]):
        raise Failure("query input fields must be nat32 integers")
    fields = sorted((field_hash(name), name) for name in ("input_ids", "markers", "qtype_id"))
    # Type 0 = vec nat32; type 1 = record { input_ids; markers; qtype_id }.
    table = b"DIDL\x02\x6d\x79\x6c\x03"
    values = bytearray()
    for key, name in fields:
        table += uleb(key) + (b"\x79" if name == "qtype_id" else b"\x00")
        if name == "qtype_id":
            values.extend(struct.pack("<I", qtype))
        else:
            values.extend(uleb(len(inp[name])))
            for value in inp[name]:
                values.extend(struct.pack("<I", value))
    if begin_steps is None:
        return table + b"\x01\x01" + values
    if type(begin_steps) is not int or not 1 <= begin_steps <= 16:
        raise Failure("query steps must be 1..16")
    return table + b"\x02\x01\x79" + values + struct.pack("<I", begin_steps)


def encode_continue(state, steps):
    if type(steps) is not int or not 1 <= steps <= 16:
        raise Failure("query steps must be 1..16")
    # Type 0 = vec nat8; arguments = (type 0, nat32).
    return b"DIDL\x01\x6d\x7b\x02\x00\x79" + uleb(len(state)) + state + struct.pack("<I", steps)


def decode_logits(reply):
    match = re.search(r"logits\s*=\s*vec\s*\{([^}]*)\}", reply)
    if not match:
        raise Failure("missing query logits")
    logits = [float(v.strip().split(":")[0].replace("_", ""))
              for v in match[1].split(";") if v.strip()]
    if not logits or any(not math.isfinite(v) for v in logits):
        raise Failure("invalid query logits")
    return logits


def decode_progress(reply):
    # The protocol has only one blob, the opaque continuation. Decode exactly the
    # Candid blob escape syntax, never round-trip hidden values through decimals.
    from canister_infer import decode_blobs
    if not re.search(r"variant\s*\{\s*Ok\s*=", reply):
        raise Failure(reply)
    done = bool(re.search(r"variant\s*\{\s*Done\s*=", reply))
    if not done and not re.search(r"variant\s*\{\s*Continue\s*=", reply):
        raise Failure("invalid query progress variant")
    result = {"done": done}
    for key in ("completed", "total", "instructions"):
        match = re.search(rf"\b{key}\s*=\s*([\d_]+)", reply)
        if not match:
            raise Failure(f"missing query progress field: {key}")
        result[key] = int(match[1].replace("_", ""))
    if done:
        result["logits"] = decode_logits(reply)
        result["state"] = None
    else:
        blobs = decode_blobs(reply)
        if len(blobs) != 1:
            raise Failure("expected exactly one continuation blob")
        result["state"] = blobs[0]
        raw, raw_size = inspect_state(blobs[0])
        result["raw_state_bytes"] = raw_size
        if len(raw) < 60 or raw[:4] not in (b"LAYQ", b"LAYI"):
            raise Failure("invalid continuation header")
        version, revision = struct.unpack_from("<II", raw, 4)
        expected_revision = 2 if raw[:4] == b"LAYI" else 1
        if (version, revision) != (1, expected_revision):
            raise Failure("unsupported continuation revision")
        if struct.unpack_from("<I", raw, 44)[0] != result["completed"]:
            raise Failure("continuation step differs from progress")
        result["bundle"] = raw[12:44].hex()
        result["activation_format"] = "int8" if raw[:4] == b"LAYI" else "f32"
    return result


class QuerySession:
    """One client-owned inference. Multiple instances can be advanced independently."""

    def __init__(self, icp, inp, steps=1, retries=2, activation_format="f32", compression="none"):
        if steps is not None and (type(steps) is not int or not 1 <= steps <= 16):
            raise Failure("query steps must be 1..16")
        self.icp, self.inp, self.steps = icp, inp, steps or 1
        self.auto = steps is None
        self.auto_cap = 16
        if compression not in ("auto", "none"):
            raise Failure("query compression must be auto or none")
        self.compression = compression
        self.direct = self.auto and activation_format == "f32" and len(inp["input_ids"]) <= 16
        self.fused_begin = self.auto and not self.direct
        self.progress = None
        self.calls = []
        self.started = time.monotonic()
        self.bundle = None
        self.retries = retries
        self.failed_attempts = []
        if activation_format not in ("f32", "int8"):
            raise Failure("unsupported query activation format")
        self.activation_format = activation_format

    def advance(self):
        if self.progress and self.progress["done"]:
            raise Failure("query inference is already complete")
        beginning = self.progress is None
        if self.auto and not self.direct:
            self.steps = auto_width(len(self.inp["input_ids"]), self.activation_format, 0 if beginning else self.progress["completed"], self.auto_cap)
        method = "begin_token_inference_query" if beginning else "continue_token_inference_query"
        if self.activation_format == "int8":
            method = method.replace("_query", "_int8_query")
        if beginning and self.fused_begin:
            method = method.replace("_query", "_batch_query")
        if self.compression == "auto":
            method = method.replace("_query", "_compressed_query")
        if self.direct:
            method = "infer_tokens_query"
        payload = encode_input(self.inp, self.steps if self.fused_begin else None) if beginning else encode_continue(self.progress["state"], self.steps)
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="laya-query-") as temp:
            path = Path(temp) / "args.bin"
            path.write_bytes(payload)
            command = ["canister", "call", "decision-engine", method,
                       "--args-file", str(path), "--args-format", "bin", "--query",
                       "-e", self.icp.env, "--candid", self.icp.did["decision-engine"]]
            attempts = 0
            while True:
                try:
                    reply = self.icp.run(command)
                    break
                except (Failure, subprocess.TimeoutExpired) as error:
                    message = str(error).lower()
                    self.failed_attempts.append(dict(method=method, steps=self.steps,
                                                     request_candid_bytes=len(payload),
                                                     error=str(error)[:1024]))
                    # A failed query has no persistent state. Retry the same bytes
                    # only for transport errors, or reduce the batch size if
                    # it exhausts the IC instruction limit.
                    instruction_limit = "instruction" in message and any(v in message for v in ("limit", "exceeded"))
                    if instruction_limit and (not beginning or self.fused_begin) and self.steps > 1:
                        self.steps = max(1, self.steps // 2)
                        if self.auto:
                            self.auto_cap = min(self.auto_cap, self.steps)
                        payload = encode_input(self.inp, self.steps) if beginning else encode_continue(self.progress["state"], self.steps)
                        path.write_bytes(payload)
                        continue
                    if not transient_failure(error) or attempts >= self.retries:
                        raise
                    attempts += 1
                    time.sleep(0.25 * attempts)
        if self.direct:
            if not re.search(r"variant\s*\{\s*Ok\s*=", reply):
                raise Failure(reply)
            match = re.search(r"\binstructions\s*=\s*([\d_]+)", reply)
            if not match:
                raise Failure("missing short-query instruction count")
            next_progress = dict(done=True, state=None, completed=32, total=32,
                                 logits=decode_logits(reply), instructions=int(match[1].replace("_", "")))
        else:
            next_progress = decode_progress(reply)
        if self.direct:
            expected = 32
        elif beginning:
            expected = min(self.steps,next_progress["total"]) if self.fused_begin else 0
        else:
            expected = min(self.progress["completed"] + self.steps, self.progress["total"])
        if next_progress["completed"] != expected or not 0 <= expected <= next_progress["total"]:
            raise Failure("unexpected query progress")
        if not beginning and next_progress["total"] != self.progress["total"]:
            raise Failure("query phase count changed")
        if next_progress["done"] != (expected == next_progress["total"]):
            raise Failure("query completion does not match phase count")
        if next_progress["done"] and len(next_progress["logits"]) != len(self.inp["markers"]):
            raise Failure("query logit count differs from marker count")
        if self.auto and next_progress["total"] != 32:
            raise Failure("automatic query grouping requires 32 phases")
        if self.direct:
            self.bundle = AUTO_BUNDLE
        if not next_progress["done"]:
            if next_progress["activation_format"] != self.activation_format:
                raise Failure("query activation format changed")
            if self.bundle and self.bundle != next_progress["bundle"]:
                raise Failure("query model changed")
            self.bundle = next_progress["bundle"]
            if self.auto and (self.bundle != AUTO_BUNDLE or next_progress["total"] != 32):
                raise Failure("automatic query grouping requires the calibrated pack and 32 phases")
        self.calls.append(dict(method=method, completed=expected,
                               instructions=next_progress["instructions"],
                               steps=0 if beginning and not self.fused_begin else self.steps,
                               request_candid_bytes=len(payload),
                               response_continuation_bytes=len(next_progress["state"] or b""),
                               response_uncompressed_state_bytes=next_progress.get("raw_state_bytes", 0),
                               wall_seconds=time.monotonic() - started))
        # Replace the client-owned state only after a successful, valid response.
        self.progress = next_progress
        return next_progress

    def finish(self):
        while not self.progress or not self.progress["done"]:
            self.advance()
        return dict(input_tokens=len(self.inp["input_ids"]), logits=self.progress["logits"], activation_format=self.activation_format,
                    instructions=sum(c["instructions"] for c in self.calls),
                    max_observed_query_handler_instructions=max(c["instructions"] for c in self.calls),
                    instruction_scope="handler; excludes CDK argument decoding and reply encoding",
                    inference_query_calls=len(self.calls), query_steps_per_call="auto" if self.auto else self.steps,
                    compression=self.compression, direct_short_query=self.direct,
                    fused_begin_query=self.fused_begin,
                    query_calls=self.calls, bundle_sha256=self.bundle,
                    failed_query_attempts=self.failed_attempts,
                    failed_attempt_request_candid_bytes=sum(c["request_candid_bytes"] for c in self.failed_attempts),
                    request_candid_bytes=sum(c["request_candid_bytes"] for c in self.calls),
                    response_continuation_bytes=sum(c["response_continuation_bytes"] for c in self.calls),
                    response_byte_scope="continuation blobs only; excludes Candid envelope and final logits",
                    local_wall_seconds=time.monotonic() - self.started)


def infer_queries(icp, input_path: Path, steps=None, activation_format="f32", compression="auto"):
    return QuerySession(icp, json.loads(input_path.read_text()), steps, activation_format=activation_format, compression=compression).finish()
