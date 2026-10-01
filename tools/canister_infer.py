#!/usr/bin/env python3
"""Upload a Laya pack and run token-level inference on this project's local IC.

Uses 1 MiB binary Candid files, so real checkpoints do not hit argv limits.
Never reinstalls a canister. --initialize installs only on a fresh canister;
use the explicit icp upgrade workflow for an existing installed canister.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import secrets
import struct
import tempfile
import time
from measure_inference import (Icp, ROOT, BUILD, Failure, blob, principal,
                               require_local_network, network_status,
                               ensure_identity, ensure_cycles, ensure_canister)
from inference_budget import BUNDLE_HASH, MODULE_HASH, plan_inference


def uleb(value):
    out = bytearray()
    while value >= 128:
        out.append((value & 127) | 128)
        value >>= 7
    out.append(value)
    return bytes(out)


def decode_blobs(text):
    values = []
    for body in re.findall(r'blob\s+"((?:[^"\\]|\\.)*)"', text):
        value = bytearray()
        i = 0
        escapes = {"n": 10, "r": 13, "t": 9, '"': 34, "'": 39, "\\": 92}
        while i < len(body):
            if body[i] != "\\":
                value.extend(body[i].encode())
                i += 1
            elif body[i+1] in escapes:
                value.append(escapes[body[i+1]])
                i += 2
            else:
                value.append(int(body[i+1:i+3], 16))
                i += 3
        values.append(bytes(value))
    return values


def upload(icp, directory):
    raw = (directory / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    tok = (directory / "tokenizer.json").read_bytes()
    if hashlib.sha256(tok).digest() != bytes(manifest["tokenizer_sha256"]):
        raise Failure("tokenizer hash mismatch")
    tokens = {v["content"]: v["id"] for v in json.loads(tok)["added_tokens"]}
    names = {"cls": "[CLS]", "sep": "[SEP]", "mask": "[MASK]", "pad": "[PAD]"}
    special = "; ".join(f"{field} = {tokens[name]} : nat32" for field, name in names.items())
    literals = "; ".join(json.dumps(name) for name in names.values())
    result = icp.call("decision-engine", "begin_upload", f"({blob(raw)}, {len(tok)} : nat64, record {{ {special}; literals = vec {{ {literals} }} }})")
    if "Ok" not in result:
        raise Failure(result)
    offset = 0
    with tempfile.TemporaryDirectory(prefix="laya-upload-") as temp:
        argsfile = Path(temp) / "args.bin"
        for path in [directory / "model.bin", directory / "tokenizer.json"]:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    # Type table: vec nat8. Args: nat64, type-table entry 0.
                    argsfile.write_bytes(b"DIDL\x01\x6d\x7b\x02\x78\x00" + struct.pack("<Q", offset) + uleb(len(chunk)) + chunk)
                    result = icp.run(["canister", "call", "decision-engine", "upload_chunk", "--args-file", str(argsfile), "--args-format", "bin", "-e", icp.env, "--candid", icp.did["decision-engine"]], timeout=300)
                    if "Ok" not in result:
                        raise Failure(result)
                    offset += len(chunk)
                    if offset % (64 * 1024 * 1024) == 0:
                        print(f"uploaded {offset // (1024*1024)} MiB", flush=True)
    warmup(icp, len(manifest["tensors"]))
    print(f"ready: {manifest['format']}, {offset} uploaded bytes", flush=True)


def warmup(icp, count=None):
    result = icp.call("decision-engine", "start_warmup")
    if "Ok" not in result:
        raise Failure(result)
    for i in range(count or 1024):
        result = icp.call("decision-engine", "warmup_next", timeout=600)
        if "Ok" not in result:
            raise Failure(result)
        if "true" in result:
            if count is not None and i != count-1:
                raise Failure("unexpected warmup tensor count")
            return
        if (i+1) % 32 == 0:
            print(f"loaded {i+1} tensors", flush=True)
    raise Failure("warmup did not complete")


def parse_costs(reply):
    start=reply.index("costs = vec {")+len("costs = vec {")
    depth=1;end=start
    while depth:
        if reply[end]=="{": depth+=1
        elif reply[end]=="}": depth-=1
        end+=1
    body=reply[start:end-1];records=[];pos=0
    while True:
        begin=body.find("record {",pos)
        if begin<0:break
        begin+=len("record {");end=begin;depth=1
        while depth:
            if body[end]=="{":depth+=1
            elif body[end]=="}":depth-=1
            end+=1
        record=body[begin:end-1];pos=end
        name=re.search(r'name\s*=\s*"([^"]+)"',record)[1]
        shape=re.search(r'shape\s*=\s*vec\s*\{([^}]*)\}',record)[1]
        shape=[int(v.strip().split(":")[0].replace("_","")) for v in shape.split(";") if v.strip()]
        instructions=int(re.search(r'instructions\s*=\s*([\d_]+)',record)[1].replace("_",""))
        records.append(dict(name=name,shape=shape,instructions=instructions))
    return records


def infer(icp, input_path, stepped=False, profile=False, steps_per_call=16,
          request_id=None, max_update_instructions=None):
    if not 1 <= steps_per_call <= 16:
        raise Failure("steps_per_call must be 1..16")
    if profile:
        steps_per_call = 1
    inp = json.loads(input_path.read_text())
    ids = "; ".join(str(v) for v in inp["input_ids"])
    markers = "; ".join(str(v) for v in inp["markers"])
    args = f"(record {{ input_ids = vec {{ {ids} }}; markers = vec {{ {markers} }}; qtype_id = {inp['qtype_id']} : nat32 }})"
    started = time.monotonic()
    costs = []
    profiles = []
    completed_steps = []
    combined_start = stepped and not profile and steps_per_call > 1
    start_id = None
    start_instructions = None
    if stepped:
        if combined_start:
            start_id = secrets.token_bytes(32) if request_id is None else bytes.fromhex(request_id)
            if len(start_id) != 32:
                raise Failure("request_id must be 32 bytes of hex")
            print(f"start request_id: {start_id.hex()}", flush=True)
            result = icp.call("decision-engine", "start_token_inference_batch", args[:-1] + f", {blob(start_id)}, {steps_per_call} : nat32)", timeout=600)
        else:
            result = icp.call("decision-engine", "start_token_inference", args)
        if "Ok" not in result:
            raise Failure(result)
        if not combined_start:
            match = re.search(r"(?<!_)instructions\s*=\s*([\d_]+)", result)
            if not match:
                raise Failure(f"missing start instruction count: {result}")
            start_instructions = int(match[1].replace("_", ""))
            if max_update_instructions is not None and start_instructions > max_update_instructions:
                raise Failure(f"start used {start_instructions:,} instructions, above budget {max_update_instructions:,}")
        ids = decode_blobs(result)
        if len(ids) != 1 or len(ids[0]) != 32:
            raise Failure(f"invalid job id: {result}")
        job = blob(ids[0])
        total = int(re.search(r"total\s*=\s*([\d_]+)", result)[1].replace("_", ""))
        step = 0
        if combined_start:
            step = int(re.search(r"completed\s*=\s*([\d_]+)", result)[1].replace("_", ""))
            if step != min(steps_per_call,total):
                raise Failure("unexpected initial batch progress")
            costs.append(int(re.search(r"(?<!_)instructions\s*=\s*([\d_]+)", result)[1].replace("_", "")))
            if max_update_instructions is not None and costs[-1] > max_update_instructions:
                raise Failure(f"update used {costs[-1]:,} instructions, above budget {max_update_instructions:,}")
            completed_steps.append(step)
            print(f"start + steps 1-{step}/{total}: {costs[-1]} instructions", flush=True)
        while step < total:
            method = "profile_token_step" if profile else ("step_token_inference" if steps_per_call == 1 else "step_token_inference_batch")
            call_args = f"({job}, {step} : nat32)" if steps_per_call == 1 else f"({job}, {step} : nat32, {steps_per_call} : nat32)"
            result = icp.call("decision-engine", method, call_args, timeout=600)
            if profile and "Ok" in result:
                profiles.append(parse_costs(result))
                result=result[result.index("progress = record"):]
                result="Ok " + result
            if "Ok" not in result:
                raise Failure(result)
            costs.append(int(re.search(r"(?<!_)instructions\s*=\s*([\d_]+)", result)[1].replace("_", "")))
            if max_update_instructions is not None and costs[-1] > max_update_instructions:
                raise Failure(f"update used {costs[-1]:,} instructions, above budget {max_update_instructions:,}")
            completed = int(re.search(r"completed\s*=\s*([\d_]+)", result)[1].replace("_", ""))
            if completed != min(step + steps_per_call, total):
                raise Failure(f"unexpected batch progress: {step} -> {completed}")
            completed_steps.append(completed)
            print(f"steps {step+1}-{completed}/{total}: {costs[-1]} instructions", flush=True)
            step = completed
    else:
        result = icp.call("decision-engine", "infer_tokens", args, timeout=600)
    if "Ok" not in result:
        raise Failure(result)
    match = re.search(r"logits\s*=\s*(?:opt\s+)?vec\s*\{([^}]*)\}", result)
    instructions = re.search(r"cumulative_instructions\s*=\s*([\d_]+)" if stepped else r"instructions\s*=\s*([\d_]+)", result)
    if not match or not instructions:
        raise Failure(f"unexpected inference result: {result}")
    measured = int(instructions[1].replace("_", ""))
    if max_update_instructions is not None and not stepped and measured > max_update_instructions:
        raise Failure(f"update used {measured:,} instructions, above budget {max_update_instructions:,}")
    logits = [float(v.strip().split(":")[0].replace("_", "")) for v in match[1].split(";") if v.strip()]
    return {"input_tokens": len(inp["input_ids"]), "logits": logits,
            "instructions": measured, "step_instructions": costs,
            "start_instructions": start_instructions,
            "completed_steps": completed_steps, "steps_per_call": steps_per_call if stepped else None,
            "inference_update_calls": (len(costs) if combined_start else 1 + len(costs)) if stepped else 1,
            "start_request_id": start_id.hex() if start_id else None, "profiles": profiles, "local_wall_seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, help="upload and warm this pack before inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", action="store_true", help="record detailed spans for each step")
    parser.add_argument("--query-int8", action="store_true", help="experimental INT8 activation engine; requires --query-stepped")
    parser.add_argument("--stepped", action="store_true", help="resumable inference, up to 16 phases per update")
    parser.add_argument("--query-stepped", action="store_true", help="independent queries; continuation state stays on the client")
    parser.add_argument("--query-steps-per-call", type=int, choices=range(1,17), help="fixed phases per query; omit for input-aware grouping")
    parser.add_argument("--query-compression", choices=("auto", "none"), default="auto", help="lossless query state compression (default: auto)")
    parser.add_argument("--steps-per-call", type=int, choices=range(1,17), default=16, help="batch size for --stepped; --profile always uses 1")
    parser.add_argument("--max-update-instructions", type=int,
                        help="empirical per-update budget; auto-select direct or split for the calibrated Wasm and pack")
    parser.add_argument("--warmup", action="store_true", help="rebuild model from already uploaded stable bytes")
    parser.add_argument("--initialize", action="store_true", help="create and install a fresh local canister")
    parser.add_argument("--request-id", help="32-byte hex ID for retrying the current combined-start job")
    parser.add_argument("--identity", default="ic-laya-int8")
    parser.add_argument("--project-root", type=Path, default=ROOT, help="local icp project to use (default: repository root)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.query_int8 and not args.query_stepped:
        parser.error("--query-int8 requires --query-stepped")
    if args.query_stepped and (args.stepped or args.profile or args.max_update_instructions is not None or args.request_id or args.steps_per_call != 16):
        parser.error("--query-stepped cannot be combined with update inference options")
    if not args.query_stepped and args.query_steps_per_call is not None:
        parser.error("--query-steps-per-call requires --query-stepped")
    if not args.query_stepped and args.query_compression != "auto":
        parser.error("--query-compression requires --query-stepped")
    if args.max_update_instructions is not None and (args.stepped or args.profile or args.steps_per_call != 16):
        parser.error("--max-update-instructions selects the inference route; omit --stepped, --profile and --steps-per-call")
    if args.max_update_instructions is not None and args.pack:
        if hashlib.sha256((args.pack / "manifest.json").read_bytes()).hexdigest() != BUNDLE_HASH:
            raise Failure("budget calibration does not match the requested pack")
    if args.max_update_instructions is not None and args.initialize:
        if "0x" + hashlib.sha256((BUILD / "decision-engine.wasm").read_bytes()).hexdigest() != MODULE_HASH:
            raise Failure("budget calibration does not match the Wasm to install")
    icp = Icp(args.project_root.resolve(), "local", args.identity)
    require_local_network(icp)
    if network_status(icp) is None:
        raise Failure("start this project's local network first")
    if args.initialize:
        owner = ensure_identity(icp, args.identity)
        ensure_cycles(icp)
        ensure_canister(icp, "decision-engine")
        icp.run(["canister", "install", "decision-engine", "-e", "local", "-y", "-m", "install", "--wasm", str(BUILD / "decision-engine.wasm"), "--args", f"({principal(owner)})"], timeout=600)
    if args.pack:
        upload(icp, args.pack)
    elif args.warmup:
        warmup(icp)
    plan = None
    if args.max_update_instructions is not None:
        before = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
        active_before = decode_blobs(icp.query("decision-engine", "info"))
        if len(active_before) != 1 or len(active_before[0]) != 32:
            raise Failure("invalid active bundle response")
        inp = json.loads(args.input.read_text())
        try:
            plan = plan_inference(len(inp["input_ids"]), len(inp["markers"]), inp["qtype_id"],
                                  args.max_update_instructions, before["module_hash"], active_before[0].hex())
        except (KeyError, TypeError, ValueError) as error:
            raise Failure(f"cannot route budgeted inference: {error}") from error
        if args.request_id and plan.steps_per_call in (None, 1):
            raise Failure("--request-id requires split inference with at least 2 steps per call")
        print(f"budget route: {plan.mode}, estimate={plan.estimated_direct_instructions:,}, "
              f"per-update budget={plan.effective_budget:,}, steps={plan.steps_per_call}", flush=True)
    if args.query_stepped:
        from client_held_query import infer_queries
        query_before = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
        result = infer_queries(icp, args.input, args.query_steps_per_call, "int8" if args.query_int8 else "f32", args.query_compression)
    else:
        result = infer(icp, args.input,
                       stepped=plan.mode == "stepped" if plan else (args.stepped or args.profile),
                       profile=args.profile,
                       steps_per_call=plan.steps_per_call if plan and plan.steps_per_call else args.steps_per_call,
                       request_id=args.request_id,
                       max_update_instructions=plan.effective_budget if plan else None)
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    result.update(network="local", measured_at=datetime.now(timezone.utc).isoformat(),
                  canister=status["id"], module_hash=status["module_hash"],
                  total_canister_memory_bytes=int(status["memory_size"].replace("_", "")),
                  input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    active = decode_blobs(icp.query("decision-engine", "info"))
    if len(active) != 1 or len(active[0]) != 32:
        raise Failure("invalid active bundle response")
    if args.query_stepped and (status["module_hash"] != query_before["module_hash"] or
                              result["bundle_sha256"] != active[0].hex()):
        raise Failure("installed Wasm or pack changed during query inference")
    result["bundle_sha256"] = active[0].hex()
    if plan:
        if status["module_hash"] != before["module_hash"] or result["bundle_sha256"] != active_before[0].hex():
            raise Failure("installed Wasm or pack changed during budgeted inference")
        result["budget_route"] = plan.mode
        result["requested_max_update_instructions"] = plan.requested_budget
        result["effective_max_update_instructions"] = plan.effective_budget
        result["estimated_direct_instructions"] = plan.estimated_direct_instructions
        updates = result["step_instructions"][:]
        if result["start_instructions"] is not None:
            updates.append(result["start_instructions"])
        result["max_observed_update_instructions"] = max(updates, default=result["instructions"])
    if args.pack and result["bundle_sha256"] != hashlib.sha256((args.pack / "manifest.json").read_bytes()).hexdigest():
        raise Failure("active bundle does not match uploaded manifest")
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
