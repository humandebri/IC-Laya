"""Review escalation from structured SNS proposal data."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from sns_proposal_triage import triage  # noqa: E402


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
