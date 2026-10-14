"""Protocol regressions run without torch or simulator dependencies."""
from __future__ import annotations

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path

from g1_vla.config import load_config
from g1_vla.contracts import ActionContract, ContractError, observation_spec_from_config, read_joint_order
from g1_vla.policy import (TensorObservation, make_checkpoint_metadata, ordered_vocabulary,
                           validate_checkpoint_metadata, validate_stats, validate_vocab,
                           validate_warm_start, validate_tensor_observation,
                           validate_model_metadata, torch)
from g1_vla.policy import contract_from_metadata
from g1_vla.training import train

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    config = load_config(ROOT / "configs/experiment.json")
    names = read_joint_order(json.loads(Path(config["action"]["joint_metadata_path"]).read_text()))
    contract = ActionContract(names, 20.0, .0025, (.08,) * 31, (1.6,) * 31)
    vocab = {"<pad>": 0, "<unk>": 1, "green": 2, "cube": 3}
    stats = {}
    for prefix, dimension in (("state", 31), ("velocity", 31), ("contact_sites", 108), ("action_delta", 31)):
        stats[prefix + "_mean"] = [0.0] * dimension
        stats[prefix + "_std"] = [1.0] * dimension
    spec = observation_spec_from_config(config)
    metadata = make_checkpoint_metadata(contract, vocab, stats, observation_spec=spec)
    checkpoint = {"metadata": metadata, "vocab": vocab, "stats": stats,
                  "data_provenance": {"manifest_sha256": "a" * 64, "audit_sha256": "b" * 64}}
    return contract, vocab, stats, metadata, checkpoint


class PolicyMetadataTest(unittest.TestCase):
    def test_execution_gains_are_part_of_checkpoint_identity(self):
        contract, vocab, stats, _, _ = fixture()
        from dataclasses import replace
        changed = replace(contract, body_gain_scale=4.0, hand_gain_scale=5.0)
        metadata = make_checkpoint_metadata(changed, vocab, stats, observation_spec=fixture()[3]["observation_spec"])
        self.assertEqual(contract_from_metadata(metadata).body_gain_scale, 4.0)
        self.assertNotEqual(contract.contract_sha256, changed.contract_sha256)

    def test_metadata_uses_token_id_order(self):
        _, vocab, stats, metadata, _ = fixture()
        shuffled = {"cube": 3, "green": 2, "<unk>": 1, "<pad>": 0}
        self.assertEqual(ordered_vocabulary(shuffled), ["<pad>", "<unk>", "green", "cube"])
        validate_checkpoint_metadata(metadata, vocab=shuffled, stats=stats)

    def test_same_shape_reordered_vocabulary_rejected(self):
        _, vocab, stats, metadata, _ = fixture()
        vocab["green"], vocab["cube"] = vocab["cube"], vocab["green"]
        with self.assertRaisesRegex(ContractError, "vocabulary"):
            validate_checkpoint_metadata(metadata, vocab=vocab, stats=stats)

    def test_normalization_changes_rejected(self):
        _, vocab, stats, metadata, checkpoint = fixture()
        changed = copy.deepcopy(stats)
        changed["state_mean"][0] = .01
        with self.assertRaisesRegex(ContractError, "normalization"):
            validate_checkpoint_metadata(metadata, vocab=vocab, stats=changed)
        with self.assertRaisesRegex(ContractError, "normalization"):
            validate_warm_start(checkpoint, vocab, changed, metadata)

    def test_action_contract_joint_order_mutation_rejected(self):
        _, vocab, stats, metadata, _ = fixture()
        changed = copy.deepcopy(metadata)
        changed["contract"]["joint_names"].reverse()
        with self.assertRaises(ContractError):
            validate_checkpoint_metadata(changed, vocab=vocab, stats=stats)

    def test_observation_fields_mutation_rejected(self):
        _, vocab, stats, metadata, _ = fixture()
        changed = copy.deepcopy(metadata)
        changed["observation_fields"].append("object_pose")
        with self.assertRaises(ContractError):
            validate_checkpoint_metadata(changed, vocab=vocab, stats=stats)

    def test_camera_order_and_tactile_spec_mismatch_rejected(self):
        _, vocab, stats, metadata, _ = fixture()
        changed = copy.deepcopy(metadata)
        changed["observation_spec"]["camera_keys"].reverse()
        changed["observation_spec"]["camera_ids"].reverse()
        with self.assertRaisesRegex(ContractError, "observation_spec"):
            validate_checkpoint_metadata(changed, vocab=vocab, stats=stats)
        changed = copy.deepcopy(metadata)
        changed["observation_spec"]["tactile_protocol"] = "legacy_target_specific_object_contact"
        with self.assertRaisesRegex(ContractError, "tactile"):
            validate_checkpoint_metadata(changed, vocab=vocab, stats=stats)

    def test_nonfinite_and_nonpositive_statistics_rejected(self):
        _, _, stats, _, _ = fixture()
        changed = copy.deepcopy(stats)
        changed["state_std"][0] = 0.0
        with self.assertRaises(ContractError):
            validate_stats(changed)
        changed["state_std"][0] = float("nan")
        with self.assertRaises(ContractError):
            validate_stats(changed)

    def test_model_dimensions_must_match_observation_and_action_protocol(self):
        _, _, _, metadata, _ = fixture()
        config = {"vocab_size": 4, "contact_site_dim": 108, "history_frames": 4,
                  "action_chunk": 8, "hidden_dim": 384, "factorized_actions": True}
        validate_model_metadata(config, metadata)
        config["contact_site_dim"] = 72
        with self.assertRaisesRegex(ContractError, "contact-site"):
            validate_model_metadata(config, metadata)
        config["contact_site_dim"] = 108
        config["action_chunk"] = 4
        with self.assertRaisesRegex(ContractError, "temporal"):
            validate_model_metadata(config, metadata)

    def test_bad_vocab_and_incomplete_stats_rejected(self):
        with self.assertRaises(ContractError):
            validate_vocab({"<pad>": 0, "<unk>": 3})
        with self.assertRaises(ContractError):
            validate_stats({"state_mean": [0.0] * 31})

    def test_legacy_warm_start_and_missing_provenance_rejected(self):
        _, vocab, stats, metadata, checkpoint = fixture()
        legacy = {"vocab": vocab, "stats": stats}
        with self.assertRaisesRegex(ContractError, "Historical"):
            validate_warm_start(legacy, vocab, stats, metadata)
        checkpoint.pop("data_provenance")
        with self.assertRaisesRegex(ContractError, "provenance"):
            validate_warm_start(checkpoint, vocab, stats, metadata)

    def test_warm_start_preserves_complete_token_and_normalization_space(self):
        _, vocab, stats, metadata, checkpoint = fixture()
        validate_warm_start(checkpoint, vocab, stats, metadata)
        changed = dict(vocab)
        changed["green"], changed["cube"] = changed["cube"], changed["green"]
        with self.assertRaisesRegex(ContractError, "vocabulary"):
            validate_warm_start(checkpoint, changed, stats, metadata)
        with self.assertRaisesRegex(ContractError, "current audited"):
            validate_warm_start(checkpoint, vocab, stats, None)

    def test_policy_boundary_cannot_accept_simulator_context(self):
        with self.assertRaises(TypeError):
            TensorObservation(images=None, state=None, velocity=None, sites=None,
                              tactile=None, ids=None, mask=None, simulator_context={})
        for name in ("policy.py", "training.py"):
            tree = ast.parse((ROOT / "src/g1_vla" / name).read_text())
            imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
            self.assertFalse(any(value and (value.startswith("scripts") or value.startswith("g1_")) for value in imports))

    def test_training_gate_records_disabled_without_tensor_runtime(self):
        config = load_config(ROOT / "configs/experiment.json")
        config["train"]["enabled"] = False
        with tempfile.TemporaryDirectory() as temp:
            result = train(config, Path(temp))
            self.assertEqual(result["status"], "disabled")
            self.assertFalse(result["training_started"])
            self.assertFalse(result["optimizer_created"])
            self.assertFalse(result["weights_updated"])
            self.assertFalse(result["readiness"]["ready"])
            stored = json.loads((Path(temp) / "reports/training_gate.json").read_text())
            self.assertEqual(stored, result)
            self.assertEqual(list(Path(temp).rglob("*.pt")), [])

    def test_editing_enabled_does_not_create_an_optimizer(self):
        config = load_config(ROOT / "configs/experiment.json")
        config["train"]["enabled"] = True
        with tempfile.TemporaryDirectory() as temp:
            result = train(config, Path(temp))
            self.assertEqual(result["status"], "blocked")
            self.assertFalse(result["training_started"])
            self.assertFalse(result["optimizer_created"])

    @unittest.skipUnless(torch is not None, "tensor runtime unavailable; run on the WSL CPU runtime")
    def test_tensor_boundary_rejects_nonfinite_sensors_and_bad_tokens(self):
        def observation():
            return TensorObservation(torch.zeros(1, 4, 3, 3, 8, 8), torch.zeros(1, 4, 31),
                                     torch.zeros(1, 4, 31), torch.zeros(1, 4, 108),
                                     torch.zeros(1, 4, 1), torch.zeros(1, 16, dtype=torch.long),
                                     torch.zeros(1, 16, dtype=torch.bool))
        good = observation()
        validate_tensor_observation(good, 4)
        bad = observation()
        bad.state[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_tensor_observation(bad, 4)
        bad = observation()
        bad.ids[0, 0] = 4
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            validate_tensor_observation(bad, 4)
        bad = observation()
        bad.images[0, 0, 0, 0, 0, 0] = 255
        with self.assertRaisesRegex(ValueError, "normalized"):
            validate_tensor_observation(bad, 4)


if __name__ == "__main__":
    unittest.main()
