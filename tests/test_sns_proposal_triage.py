"""Review escalation from structured SNS proposal data."""

import sys
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sns_proposal_triage import triage, _laya_contexts, laya_advisory  # noqa: E402


def parameters(old, new, rendering=True):
    current = ("## Current nervous system parameters:\n"
               "NervousSystemParameters {\n"
               f"    neuron_minimum_dissolve_delay_to_vote_seconds: Some({old}),\n"
               "}\n## Proposed nervous system parameters:\n"
               f"neuron_minimum_dissolve_delay_to_vote_seconds: Some({new})")
    return {"id": "test", "proposal_action_type": "ManageNervousSystemParameters",
            "proposal_action_payload": {"neuron_minimum_dissolve_delay_to_vote_seconds": new},
            "payload_text_rendering": current if rendering else ""}


class ProposalTriageTests(unittest.TestCase):
    def test_each_change_retains_evidence_and_coverage(self):
        proposal = parameters(86400, 1728000000)
        proposal["proposal_action_payload"].update(max_dissolve_delay_seconds=2629800000,
                                                    initial_voting_period_seconds=3600)
        contexts = _laya_contexts(proposal, triage(proposal))
        indexed = {c["fact"]["field"]: c for c in contexts}
        self.assertEqual(len(indexed), 3)
        maximum = indexed["max_dissolve_delay_seconds"]
        self.assertIsNone(maximum["fact"]["old"])
        self.assertFalse(maximum["fact"]["historical_value_available"])
        self.assertIn("unknown", maximum["state"])
        self.assertEqual(indexed["initial_voting_period_seconds"]["status"], "unsupported")
        self.assertNotEqual(maximum["question"], indexed["neuron_minimum_dissolve_delay_to_vote_seconds"]["question"])

    def test_decrease_and_unchanged_are_not_silently_discarded(self):
        for old, new, expected in ((86400, 0, "ready"), (86400, 43200, "ready"), (86400, 86400, "unchanged")):
            proposal = parameters(old, new)
            rows = _laya_contexts(proposal, triage(proposal))
            self.assertEqual(rows[0]["status"], expected)
            self.assertEqual(rows[0]["fact"]["new"], new)

    def test_advisory_runs_each_supported_field_and_keeps_unsupported_evidence(self):
        proposal = parameters(86400, 1728000000)
        proposal["proposal_action_payload"].update(max_dissolve_delay_seconds=2629800000,
                                                    initial_voting_period_seconds=3600)
        result = triage(proposal)
        before = json.dumps(result, sort_keys=True)
        tokenizer = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda p: object()))
        builder = SimpleNamespace(make_input=lambda *a: dict(input_ids=[1]*68, markers=[1,2,3], qtype_id=0))
        reply = json.dumps(dict(raw_logits=[0., 1., 2.], bundle="test"))
        with patch.dict(sys.modules, {"tokenizers": tokenizer, "check_practical_laya": builder}), \
                patch("sns_proposal_triage.subprocess.check_output", return_value=reply) as run:
            advisory = laya_advisory(proposal, result, Path("pack"), Path("model"))
        self.assertEqual(run.call_count, 2)
        self.assertEqual(advisory["status"], "partial")
        self.assertIsNone(advisory["aggregate_label"])
        self.assertFalse(advisory["coverage_complete"])
        self.assertEqual(len(advisory["assessments"]), 3)
        self.assertEqual(json.dumps(result, sort_keys=True), before)

    def test_overlong_input_is_reported_without_truncating_or_running_model(self):
        proposal = parameters(86400, 1728000000)
        tokenizer = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda p: object()))
        builder = SimpleNamespace(make_input=lambda *a: dict(input_ids=[1]*129, markers=[1,2,3], qtype_id=0))
        with patch.dict(sys.modules, {"tokenizers": tokenizer, "check_practical_laya": builder}), \
                patch("sns_proposal_triage.subprocess.check_output") as run:
            advisory = laya_advisory(proposal, triage(proposal), Path("pack"), Path("model"))
        run.assert_not_called()
        self.assertEqual(advisory["assessments"][0]["fact"]["new"], 1728000000)
        self.assertEqual(advisory["status"], "unavailable")

    def test_model_failure_does_not_erase_other_changes(self):
        proposal = parameters(86400, 1728000000)
        proposal["proposal_action_payload"]["max_dissolve_delay_seconds"] = 2629800000
        tokenizer = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda p: object()))
        builder = SimpleNamespace(make_input=lambda *a: dict(input_ids=[1]*68, markers=[1,2,3], qtype_id=0))
        with patch.dict(sys.modules, {"tokenizers": tokenizer, "check_practical_laya": builder}), \
                patch("sns_proposal_triage.subprocess.check_output", side_effect=[
                    OSError("model failed"), json.dumps(dict(raw_logits=[0.,1.,2.],bundle="test"))]):
            advisory = laya_advisory(proposal, triage(proposal), Path("pack"), Path("model"))
        self.assertEqual(advisory["status"], "partial")
        self.assertEqual([c["status"] for c in advisory["assessments"]], ["unavailable", "ok"])
        self.assertEqual(len(advisory["assessments"]), 2)

    def test_boom_617_requires_critical_review_and_620_does_not(self):
        extreme = triage(parameters(86400, 1728000000))
        ordinary = triage(parameters(86400, 172800))
        self.assertEqual(extreme["priority"], "critical_review")
        self.assertTrue(extreme["extra_human_review_required"])
        self.assertEqual(extreme["measured_changes"][0]["change_factor"], 20000)
        self.assertEqual(ordinary["priority"], "standard")

    def test_threshold_is_exact_and_missing_history_requires_review(self):
        self.assertEqual(triage(parameters(100, 999))["priority"], "standard")
        self.assertEqual(triage(parameters(100, 1_000))["priority"], "review")
        self.assertEqual(triage(parameters(100, 99_999))["priority"], "review")
        self.assertEqual(triage(parameters(100, 100_000))["priority"], "critical_review")
        self.assertEqual(triage(parameters(100_000, 100))["priority"], "critical_review")
        self.assertEqual(triage(parameters(100, 100_000, rendering=False))["priority"], "review")
        self.assertEqual(triage(parameters(0, 100_000))["priority"], "review")

    def test_other_changed_parameter_requires_review(self):
        proposal = parameters(86400, 86400)
        proposal["proposal_action_payload"]["initial_voting_period_seconds"] = 3600
        proposal["payload_text_rendering"] = proposal["payload_text_rendering"].replace(
            "    neuron_minimum", "    initial_voting_period_seconds: Some(86400),\n    neuron_minimum")
        result = triage(proposal)
        self.assertEqual(result["priority"], "review")
        self.assertIn("initial_voting_period_seconds: 86400 to 3600", result["reasons"])

    def test_mint_and_unknown_actions_are_reviewed_without_supply_claim(self):
        mint = triage({"proposal_action_type": "MintSnsTokens",
                       "proposal_action_payload": {"amount_e8s": 25000000000000000}})
        self.assertEqual(mint["priority"], "review")
        self.assertIn("250,000,000 tokens", mint["reasons"][0])
        self.assertIn("supply comparison unavailable", mint["reasons"][0])
        self.assertEqual(triage({"proposal_action_type": "Other"})["priority"], "review")

    def test_metadata_has_no_numeric_alert_and_malformed_payload_is_reviewed(self):
        proposal = {"proposal_action_type": "ManageSnsMetadata",
                    "proposal_action_payload": {"name": "BOOM DAO"}}
        self.assertEqual(triage(proposal)["priority"], "standard")
        proposal["proposal_action_payload"] = None
        self.assertEqual(triage(proposal)["priority"], "review")


if __name__ == "__main__":
    unittest.main()
