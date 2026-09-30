#!/usr/bin/env python3
"""Capture inclusive operator spans for experimental INT8 local queries."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from canister_infer import Icp, ROOT, parse_costs, require_local_network
from client_held_query import QuerySession
from measure_inference import Failure


def unwrap_progress(reply):
    start = reply.index("progress =") + len("progress =")
    body = reply[start:].lstrip()
    depth = 0
    quoted = escaped = False
    for i, char in enumerate(body):
        if quoted:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == '"': quoted = False
        elif char == '"': quoted = True
        elif char == "{": depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "(variant { Ok = " + body[:i+1] + " })"
    raise Failure("invalid profiled query progress")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--identity", default="ic-laya-query-local-test")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    icp = Icp(args.project_root.resolve(), "local", args.identity)
    require_local_network(icp)
    status = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    module = "0x"+hashlib.sha256((ROOT/"build/decision-engine.wasm").read_bytes()).hexdigest()
    if status["module_hash"] != module:
        raise Failure("installed module differs from build artifact")
    profiles = []

    class RecordingIcp:
        env = icp.env
        did = icp.did

        def run(self, command):
            command = list(command)
            if command[3] == "continue_token_inference_int8_query":
                command[3] = "profile_token_inference_int8_query"
            reply = icp.run(command)
            if command[3] == "profile_token_inference_int8_query":
                profiles.append(parse_costs(reply))
                # The Candid record prints costs before progress. Keep cost
                # instructions out of QuerySession's progress metric parser.
                reply = unwrap_progress(reply)
            return reply

    result = QuerySession(RecordingIcp(), json.loads(args.input.read_text()), 1, activation_format="int8").finish()
    costs = Counter()
    for phase in profiles:
        for cost in phase:
            costs[cost["name"]] += cost["instructions"]
    result.update(module_hash=module, profiles=profiles, inclusive_costs=dict(costs),
                  profile_scope="inclusive; q8.linear contains q8.fused_dot_writeback and q8.output_quantize; embedding is not profiled")
    ending = json.loads(icp.run(["canister", "status", "decision-engine", "-e", "local", "--json"]))
    if ending["module_hash"] != module:
        raise Failure("module changed during profiling")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(instructions=result["instructions"], inclusive_costs=dict(costs)), indent=2))


if __name__ == "__main__":
    main()
