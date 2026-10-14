"""Pinned checkpoint evidence must not be mistaken for runtime or task success."""

import copy
import unittest
from unittest.mock import patch

from g1_vla.weight_audit import CATALOG, WeightAuditError, audit_public_weights, fetch_checkpoint_metadata


def fixture(candidate, *, parameters=1000000, dtype="F32", config=None):
    return {"hub": {"sha": candidate["revision"], "gated": False,
                    "cardData": {"license": "apache-2.0"},
                    "safetensors": {"parameters": {dtype: parameters}, "total": parameters},
                    "siblings": [{"rfilename": "model.safetensors", "size": parameters * 4}]},
            "documents": {"config.json": {"value": config or {}, "sha256": "a" * 64}}}


HARDWARE = {"gpu_total_mib": 8188, "gpu_free_mib": 6912,
            "python": "3.10.20", "torch": "2.6.0+cu124", "cuda": "12.4"}


class WeightAuditTests(unittest.TestCase):
    def _audit(self, candidate, evidence, hardware=None):
        return audit_public_weights(".", {"public_weights": {"candidates": [candidate]}},
                                    hardware=hardware or HARDWARE, metadata={candidate["repo_id"]: evidence})

    def test_refuses_mutable_or_missing_checkpoint_identity(self):
        for candidate in ({"repo_id": "owner/model", "revision": "main"},
                          {"repo_id": "../model", "revision": "a" * 40},
                          {"repo_id": "owner/model", "revision": "A" * 40}):
            with self.subTest(candidate=candidate), self.assertRaises(WeightAuditError):
                audit_public_weights(".", {"public_weights": {"candidates": [candidate]}}, metadata={})

    def test_n17_reports_gated_dependency_software_and_memory_blockers(self):
        candidate = CATALOG[0]
        evidence = fixture(candidate, parameters=3455180928, dtype="BF16")
        report = self._audit(candidate, evidence)["candidates"][0]
        self.assertEqual(report["status"], "blocked_for_native_probe")
        self.assertIn("gated_dependency:nvidia/Cosmos-Reason2-2B", report["blockers"])
        self.assertIn("insufficient_estimated_native_gpu_headroom", report["blockers"])
        self.assertTrue(any(item.startswith("python_") for item in report["blockers"]))
        self.assertFalse(report["inference_passed"])

    def test_n16_lower_bound_does_not_claim_8gb_fits(self):
        candidate = CATALOG[1]
        evidence = fixture(candidate, parameters=3286608832, dtype="BF16")
        report = self._audit(candidate, evidence)["candidates"][0]
        self.assertFalse(report["memory"]["runtime_measured"])
        self.assertEqual(report["memory"]["bf16_parameter_lower_bound_bytes"], 6573217664)
        self.assertEqual(report["memory"]["estimated_status"], "insufficient_estimated_native_headroom")
        self.assertFalse(report["runtime"]["native_cpu_offload"])

    def test_compact_act_keeps_native_dimensions_and_never_marks_project_mapping(self):
        candidate = CATALOG[-1]
        config = {"type": "act", "chunk_size": 100, "n_obs_steps": 1,
                  "input_features": {"observation.state": {"type": "STATE", "shape": [29]},
                                     "observation.images.global_view": {"type": "VISUAL", "shape": [3, 480, 640]}},
                  "output_features": {"action": {"type": "ACTION", "shape": [20]}}}
        report = self._audit(candidate, fixture(candidate, parameters=51706516, config=config))
        selected = report["candidates"][0]
        self.assertEqual(selected["native_protocol"]["action_dim"], 20)
        self.assertEqual(selected["native_protocol"]["state_dim"], 29)
        self.assertEqual(selected["memory"]["estimated_status"], "estimated_room_for_native_probe")
        self.assertFalse(selected["project_adapter_validated"])
        self.assertFalse(selected["training_eligible"])
        self.assertFalse(report["model_tensors_downloaded"])

    def test_archive_size_does_not_invent_parameter_memory(self):
        candidate = CATALOG[3]
        evidence = {"hub": {"sha": candidate["revision"], "siblings": [
            {"rfilename": "checkpoints/pytorch_model.pt", "size": 18982885927}]}, "documents": {}}
        report = self._audit(candidate, evidence)["candidates"][0]
        self.assertIsNone(report["memory"]["bf16_parameter_lower_bound_bytes"])
        self.assertEqual(report["memory"]["estimated_status"], "insufficient_estimated_native_headroom")
        self.assertEqual(report["memory"]["dependency_bf16_lower_bound_bytes"], 16584333312)
        self.assertEqual(report["memory"]["download_bytes"], 18982885927)

    def test_revision_change_drops_catalog_compatibility_assertions(self):
        candidate = {"repo_id": CATALOG[-1]["repo_id"], "revision": "a" * 40}
        report = self._audit(candidate, fixture(candidate))["candidates"][0]
        self.assertEqual(report["role"], "unreviewed")
        self.assertEqual(report["native_protocol"], {})

    def test_mismatched_revision_missing_weights_and_invalid_hardware_are_not_success(self):
        candidate = CATALOG[-1]
        evidence = fixture(candidate)
        evidence["hub"]["sha"] = "b" * 40
        self.assertEqual(self._audit(candidate, evidence)["candidates"][0]["status"], "metadata_unavailable")
        evidence = fixture(candidate)
        evidence["hub"]["siblings"] = []
        self.assertEqual(self._audit(candidate, evidence)["candidates"][0]["status"], "metadata_unavailable")
        evidence = fixture(candidate)
        machine = dict(HARDWARE, gpu_free_mib=float("nan"))
        self.assertEqual(self._audit(candidate, evidence, machine)["candidates"][0]["status"], "metadata_unavailable")

    def test_bounded_allowlist_never_requests_model_payload(self):
        candidate = CATALOG[-1]
        hub = fixture(candidate)["hub"]
        hub["siblings"] += [{"rfilename": "config.json"}, {"rfilename": "optimizer.pt"}]
        import json
        with patch("g1_vla.weight_audit._get_bytes", side_effect=[json.dumps(hub).encode(), b'{}']) as reader:
            evidence = fetch_checkpoint_metadata(candidate)
        self.assertEqual(evidence["hub"]["sha"], candidate["revision"])
        self.assertEqual(len(reader.call_args_list), 2)
        self.assertFalse(any("safetensors" in call.args[0] or "optimizer" in call.args[0] for call in reader.call_args_list))

    def test_duplicate_identity_rejected_and_inputs_unchanged(self):
        candidate = CATALOG[-1]
        config = {"public_weights": {"candidates": [candidate, candidate]}}
        original = copy.deepcopy(config)
        with self.assertRaisesRegex(WeightAuditError, "duplicate"):
            audit_public_weights(".", config, hardware=HARDWARE, metadata={candidate["repo_id"]: fixture(candidate)})
        self.assertEqual(config, original)


if __name__ == "__main__":
    unittest.main()
