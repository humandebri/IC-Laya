import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from measure_inference import Failure
from validate_query_transport import resume_cases, validation_binding


class ValidationResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tools").mkdir()
        for name in ("client_held_query", "query_transport", "canister_infer", "measure_inference", "validate_query_transport"):
            (self.root / f"tools/{name}.py").write_text("original source")
        self.inputs = [("pilot-128", dict(input_ids=[1, 2], markers=[0, 1], qtype_id=0)),
                       ("noul", dict(input_ids=[3, 4], markers=[0, 1], qtype_id=1))]
        self.report = dict(module_hash="wasm", bundle_sha256="pack", corpus_sha256="corpus",
                           **validation_binding(self.root, self.inputs))
        self.old = copy.deepcopy(self.report)
        self.old["cases"] = [dict(id=name, input_sha256=self.report["input_sha256"][name], modes={})
                             for name, _ in self.inputs]

    def test_unchanged_and_partial_results_resume(self):
        self.assertEqual(resume_cases(self.old, self.report), self.old["cases"])
        self.old["cases"] = self.old["cases"][:1]
        self.assertEqual(len(resume_cases(self.old, self.report)), 1)

    def test_changed_extra_input_is_rejected(self):
        for index in (0, 1):
            inputs = copy.deepcopy(self.inputs)
            inputs[index][1]["qtype_id"] = 2
            report = dict(self.report, **validation_binding(self.root, inputs))
            with self.assertRaisesRegex(Failure, "input_sha256"):
                resume_cases(self.old, report)

    def test_changed_client_or_validator_is_rejected(self):
        for filename in self.report["client_source_sha256"]:
            path = self.root / filename
            path.write_text("changed source")
            report = dict(self.report, **validation_binding(self.root, self.inputs))
            with self.assertRaisesRegex(Failure, "client_source_sha256"):
                resume_cases(self.old, report)
            path.write_text("original source")

    def test_legacy_missing_binding_and_changed_scope_are_rejected(self):
        for key in ("input_sha256", "client_source_sha256"):
            old = copy.deepcopy(self.old)
            del old[key]
            with self.assertRaises(Failure):
                resume_cases(old, self.report)
        report = dict(self.report, **validation_binding(self.root, self.inputs[:1]))
        with self.assertRaises(Failure):
            resume_cases(self.old, report)

    def test_mismatched_case_binding_or_duplicate_case_is_rejected(self):
        old = copy.deepcopy(self.old)
        old["cases"][0]["input_sha256"] = "wrong-input"
        with self.assertRaises(Failure):
            resume_cases(old, self.report)
        old = copy.deepcopy(self.old)
        old["cases"].append(old["cases"][0])
        with self.assertRaises(Failure):
            resume_cases(old, self.report)


if __name__ == "__main__":
    unittest.main()
