#!/usr/bin/env python3
"""Experimental, read-only SNS proposal triage from Dashboard proposal JSON.

The structured action and historical parameter values decide mandatory review.
Optional Laya inference is supplementary and never changes that decision.
"""

import argparse
import json
import re
import subprocess
import tempfile
from decimal import Decimal
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
API = "https://sns-api.internetcomputer.org/api/v1/snses"
EXTREME_FACTOR = 1000
REVIEW_FACTOR = 10
PARAMETERS = {
    "neuron_minimum_dissolve_delay_to_vote_seconds": "minimum voting dissolve delay",
    "max_dissolve_delay_seconds": "maximum dissolve delay",
    "neuron_minimum_stake_e8s": "minimum neuron stake",
}


def _positive_int(value):
    return type(value) is int and value > 0


def _tokens(e8s):
    return format(Decimal(e8s) / Decimal(100_000_000), ",.8f").rstrip("0").rstrip(".")


def _current_parameter(rendering, name):
    marker = "## Current nervous system parameters:"
    if not isinstance(rendering, str) or marker not in rendering:
        return None
    current = rendering.split(marker, 1)[1].split("\n## ", 1)[0]
    match = re.search(r"\b" + re.escape(name) + r":\s*Some\(\s*(\d+)", current)
    return int(match.group(1)) if match else None


def triage(proposal):
    """Return a review priority; never interpret a model output as an execution gate."""
    if not isinstance(proposal, dict):
        raise ValueError("proposal must be a JSON object")
    action = proposal.get("proposal_action_type")
    payload = proposal.get("proposal_action_payload")
    reasons = []
    changes = []
    priority = "standard"

    if action == "ManageNervousSystemParameters":
        if not isinstance(payload, dict):
            priority, reasons = "review", ["parameter payload is unavailable"]
        elif not any(value is not None for value in payload.values()):
            priority, reasons = "review", ["parameter payload contains no proposed values"]
        else:
            for key, label in PARAMETERS.items():
                new = payload.get(key)
                if new is None:
                    continue
                old = _current_parameter(proposal.get("payload_text_rendering"), key)
                if not _positive_int(old) or not _positive_int(new):
                    if priority != "critical_review":
                        priority = "review"
                    reasons.append(f"{label}: historical value or proposed value is unavailable")
                    continue
                direction = "increase" if new > old else "decrease" if new < old else "unchanged"
                factor = max(new / old, old / new)
                changes.append({"field": key, "old": old, "new": new,
                                "direction": direction, "change_factor": factor})
                if new >= old * EXTREME_FACTOR or old >= new * EXTREME_FACTOR:
                    priority = "critical_review"
                    reasons.append(f"{label}: {old} to {new} ({factor:g}x {direction})")
                elif new >= old * REVIEW_FACTOR or old >= new * REVIEW_FACTOR:
                    if priority != "critical_review":
                        priority = "review"
                    reasons.append(f"{label}: {old} to {new} ({factor:g}x {direction})")
            for key, new in payload.items():
                if new is None or key in PARAMETERS:
                    continue
                old = _current_parameter(proposal.get("payload_text_rendering"), key)
                if type(new) is not int or old is None:
                    if priority != "critical_review":
                        priority = "review"
                    reasons.append(f"{key}: change cannot be verified from historical values")
                elif new != old:
                    if priority != "critical_review":
                        priority = "review"
                    reasons.append(f"{key}: {old} to {new}")
    elif action == "MintSnsTokens":
        priority = "review"
        amount = payload.get("amount_e8s") if isinstance(payload, dict) else None
        if _positive_int(amount):
            reasons.append(f"token mint: {_tokens(amount)} tokens to one account; supply comparison unavailable")
        else:
            reasons.append("token mint: amount is unavailable")
    elif action == "ManageSnsMetadata":
        if not isinstance(payload, dict):
            priority, reasons = "review", ["metadata payload is unavailable"]
    else:
        priority = "review"
        reasons.append(f"unsupported action type: {action or 'missing'}")

    return {"proposal_id": proposal.get("id"), "action_type": action,
            "priority": priority, "extra_human_review_required": priority != "standard",
            "reasons": reasons, "measured_changes": changes,
            "policy": (f"experimental: >= {EXTREME_FACTOR}x change in selected governance parameters "
                       f"requires critical review; >= {REVIEW_FACTOR}x and changes in other fields, "
                       "token mints, and unknown actions require review; standard is not a safety certification")}


def _load_proposal(args):
    if args.input:
        return json.loads(args.input.read_text())
    if not args.sns_root:
        raise ValueError("--sns-root is required with --proposal-id")
    if not re.fullmatch(r"[a-z0-9-]+", args.sns_root) or not re.fullmatch(r"[0-9]+", args.proposal_id):
        raise ValueError("invalid SNS root or proposal ID")
    url = f"{API}/{args.sns_root}/proposals/{args.proposal_id}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "IC-Laya proposal triage"})
    with urlopen(request, timeout=20) as response:
        raw = response.read(4_000_001)
    if len(raw) > 4_000_000:
        raise ValueError("proposal response is too large")
    return json.loads(raw)


def _laya_context(proposal, result):
    action = result["action_type"]
    if action == "ManageNervousSystemParameters":
        question = "Could this change reduce voter participation or concentrate voting power?"
        changes = [c for c in result["measured_changes"]
                   if c["field"] == "neuron_minimum_dissolve_delay_to_vote_seconds"]
        if not changes or changes[0]["direction"] != "increase":
            return None
        change = changes[0]
        old_days = change["old"] / 86400
        new_days = change["new"] / 86400
        old_unit = "day" if old_days == 1 else "days"
        new_unit = "day" if new_days == 1 else "days"
        state = (f"Minimum voting dissolve delay changes from {old_days:g} {old_unit} to "
                 f"{new_days:g} {new_unit}, a {change['change_factor']:g}-fold increase.")
    elif action == "MintSnsTokens":
        payload = proposal.get("proposal_action_payload")
        amount = payload.get("amount_e8s") if isinstance(payload, dict) else None
        if not _positive_int(amount):
            return None
        question = "Could this token mint concentrate token control?"
        state = f"This proposal mints {_tokens(amount)} tokens to one account."
    else:
        return None
    return question, ("unlikely", "possible", "likely"), state


def laya_advisory(proposal, result, pack, binary):
    context = _laya_context(proposal, result)
    if context is None:
        return {"status": "not_applicable"}
    from tokenizers import Tokenizer
    from check_practical_laya import make_input

    question, options, state = context
    tokenizer = Tokenizer.from_file(str(pack / "tokenizer.json"))
    model_input = make_input(tokenizer, "choice", question, options, state)
    if len(model_input["input_ids"]) > 128:
        return {"status": "unavailable", "reason": "input exceeds 128 tokens"}
    with tempfile.TemporaryDirectory(prefix="laya-sns-triage-") as directory:
        path = Path(directory) / "input.json"
        path.write_text(json.dumps(model_input))
        output = subprocess.check_output([str(binary), str(pack), str(path)], text=True)
    inference = json.loads(output)
    logits = inference["raw_logits"]
    if len(logits) != len(options):
        raise ValueError("Laya returned an unexpected number of logits")
    return {"status": "ok", "question": question, "state": state,
            "label": options[max(range(len(logits)), key=logits.__getitem__)],
            "raw_logits": dict(zip(options, logits)), "model_bundle": inference["bundle"],
            "note": "uncalibrated supplementary classification; does not change priority"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Dashboard proposal JSON file")
    source.add_argument("--proposal-id", help="proposal number to fetch from Dashboard API")
    parser.add_argument("--sns-root", help="SNS root canister ID for --proposal-id")
    parser.add_argument("--with-laya", action="store_true", help="run supplementary local Laya inference")
    parser.add_argument("--pack", type=Path, default=ROOT / "checkpoints/laya-int8")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/laya-infer")
    args = parser.parse_args()
    proposal = _load_proposal(args)
    result = triage(proposal)
    if args.with_laya:
        try:
            result["laya"] = laya_advisory(proposal, result, args.pack, args.binary)
        except (OSError, ValueError, ImportError, subprocess.CalledProcessError) as exc:
            result["laya"] = {"status": "unavailable", "reason": str(exc)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
