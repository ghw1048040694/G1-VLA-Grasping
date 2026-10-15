"""The frozen-state audit must change network input without changing its anchor."""

from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from g1_vla import state_conditioning as conditioning
from g1_vla import expanded_residual as expanded
from g1_vla import grasp_act as native
from g1_vla import state_residual as residual

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def fixture(root="."):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "experiment.json").read_text())
    config["paths"]["root"] = str(root)
    config["state_conditioning"] = {"backend": conditioning.BACKEND, "variants": list(conditioning.VARIANTS),
        "modes": list(conditioning.MODES), "eval_splits": ["train", "validation"], "eval_stride": 30,
        "eval_batch_size": 8, "within_episode_rule": conditioning.WITHIN_RULE,
        "cross_episode_rule": conditioning.CROSS_RULE, "real_reproduction_atol": 1e-6,
        "optimizer_steps": 0, "model_output_constant_method": conditioning.MODEL_CONSTANT_METHOD,
        "max_eval_wall_s": 600, "max_gpu_reserved_bytes": config["expanded_residual"]["max_gpu_reserved_bytes"],
        "tie_tolerance_rad": 0.0, "output_dir": "state_conditioning_outputs",
        "reference_expanded_report_path": config["constant_residual"]["reference_expanded_report_path"],
        "reference_expanded_report_sha256": config["constant_residual"]["reference_expanded_report_sha256"],
        "reference_constant_report_path": "constant.json", "reference_constant_report_sha256": "c" * 64}
    return config


class ConditioningProtocolTests(unittest.TestCase):
    def test_protocol_disallows_budget_training_selection_and_intervention_drift(self):
        config = fixture()
        self.assertEqual(conditioning.validate_state_conditioning_settings(config)["optimizer_steps"], 0)
        changes = (("optimizer_steps", True), ("optimizer_steps", 1), ("variants", ["state_only_residual"]),
            ("modes", ["real"]), ("eval_splits", ["validation"]), ("eval_stride", 60),
            ("eval_batch_size", 4), ("within_episode_rule", "choose_best_shift"),
            ("cross_episode_rule", "nearest_label"), ("real_reproduction_atol", 1e-3),
            ("model_output_constant_method", "validation_median"), ("max_eval_wall_s", 601),
            ("max_eval_wall_s", float("nan")), ("max_gpu_reserved_bytes", 8 * 1024 ** 3),
            ("tie_tolerance_rad", .001), ("reference_expanded_report_sha256", "A" * 64),
            ("reference_expanded_report_path", "changed.json"), ("output_dir", "../escape"))
        for key, value in changes:
            changed = copy.deepcopy(config)
            changed["state_conditioning"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((conditioning.StateConditioningError, native.GraspACTError)):
                conditioning.validate_state_conditioning_settings(changed)

    def test_failed_reference_is_saved_without_optimizer_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            # The package import remains independent of optional Torch. A failed
            # runtime dependency must also produce the immutable failure record.
            with patch.object(conditioning, "_prepare", side_effect=conditioning.StateConditioningError("unverified checkpoint")):
                report = conditioning.run_state_conditioning_audit(config, directory)
            self.assertFalse(report["passed"])
            self.assertFalse(report["optimizer_created"])
            self.assertFalse(report["training_started"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertFalse(report["test_payloads_opened"])
            saved = Path(directory) / "reports" / "state_conditioning.json"
            self.assertEqual(json.loads(saved.read_text())["error"], report["error"])
            with self.assertRaises(conditioning.StateConditioningError):
                conditioning.run_state_conditioning_audit(config, directory)


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for state intervention and decoding tests")
class ConditioningArrayTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np
        self.ids = np.asarray([(20, 0), (20, 30), (20, 60), (20, 90), (38, 0), (38, 30), (75, 0), (75, 30), (75, 60)], dtype=np.int64)
        self.states = np.arange(len(self.ids) * 28, dtype=np.float32).reshape(-1, 28) / 100
        self.mean = np.full(28, .2, dtype=np.float32)
        self.std = np.full(28, .5, dtype=np.float32)

    def test_within_roll_is_nonidentity_with_exact_odd_and_even_directions(self):
        actual = conditioning.state_donor_indices(self.ids, "within_episode_shift", self.np)
        self.assertEqual(actual.tolist(), [2, 3, 0, 1, 5, 4, 8, 6, 7])
        self.np.testing.assert_array_equal(self.ids[actual, 0], self.ids[:, 0])
        self.assertTrue(self.np.all(actual != self.np.arange(len(actual))))

    def test_test_split_is_rejected_before_enumerating_or_opening_any_payload(self):
        dataset = SimpleNamespace(split="test", get_eval_indices=unittest.mock.Mock(side_effect=AssertionError("test must not be read")))
        with self.assertRaisesRegex(conditioning.StateConditioningError, "train and validation"):
            conditioning._cohort(dataset, 30, self.np)
        dataset.get_eval_indices.assert_not_called()

    def test_cross_episode_mapping_cycles_sorted_episodes_with_relative_rank(self):
        actual = conditioning.state_donor_indices(self.ids, "cross_episode_swap", self.np)
        self.assertEqual(actual.tolist(), [4, 4, 5, 5, 6, 8, 0, 2, 3])
        self.assertTrue(self.np.all(self.ids[actual, 0] != self.ids[:, 0]))
        # Donors depend on identities only, and have no values/labels argument.
        self.assertEqual(conditioning.state_donor_indices(self.ids.copy(), "cross_episode_swap", self.np).tolist(), actual.tolist())

    def test_invalid_degenerate_and_unsorted_donor_populations_fail(self):
        np = self.np
        cases = ((self.ids.astype(np.float64), "real"), (self.ids[:1], "within_episode_shift"),
            (self.ids[:4], "cross_episode_swap"), (self.ids[[1, 0, 2, 3]], "within_episode_shift"),
            (np.asarray([(20, 0), (20, 0)], dtype=np.int64), "real"),
            (np.asarray([(20, 1), (38, 0)], dtype=np.int64), "real"))
        for ids, mode in cases:
            with self.subTest(ids=ids.tolist(), mode=mode), self.assertRaises(conditioning.StateConditioningError):
                conditioning.state_donor_indices(ids, mode, np)

    def test_mean_input_is_exact_zero_without_source_donor_or_anchor_mutation(self):
        np = self.np
        before = self.states.tobytes()
        normalized, raw, ids, offsets = conditioning.network_state_intervention(self.states, self.ids, self.mean, self.std, "train_mean", np)
        self.assertEqual(normalized.tobytes(), np.zeros_like(self.states).tobytes())
        self.assertTrue(np.all(raw == self.mean))
        self.assertTrue(np.all(ids == -1))
        self.assertTrue(np.all(offsets == -1))
        self.assertEqual(self.states.tobytes(), before)

    def test_donor_normalization_uses_train_stats_and_does_not_replace_anchor(self):
        np = self.np
        before = self.states.copy()
        network, raw, ids, offsets = conditioning.network_state_intervention(self.states, self.ids, self.mean, self.std, "within_episode_shift", np)
        np.testing.assert_array_equal(raw, before[offsets])
        np.testing.assert_array_equal(network, (before[offsets] - self.mean) / self.std)
        np.testing.assert_array_equal(ids, self.ids[offsets])
        np.testing.assert_array_equal(self.states, before)
        network[:] = 999
        np.testing.assert_array_equal(self.states, before)

    def arrays(self):
        np = self.np
        q = self.states.copy()
        q[0, 0] = np.float32(-0.0)
        z = np.repeat((q * np.float32(.1))[:, None, :], 16, axis=1)
        std = np.full(28, .5, dtype=np.float32)
        return {"observation_state": q, "sample_episode_frame": self.ids.copy(),
            "target_source_recorded_actions": np.zeros_like(z), "action_std": std,
            "predictions_scaled_state_residual": z, "predictions_raw_state_residual": z * std,
            "predictions_source_actions": residual.decode_residual(z, q, std, np),
            "original_rgb_sample_episode_frame": self.ids.copy(), "rgb_input_sample_episode_frame": self.ids.copy(),
            "rgb_input_bytes_sha256": np.asarray(["a" * 64] * len(q), dtype="U64"),
            "network_state_donor_eval_offset": np.arange(len(q), dtype=np.int64),
            "network_input_normalized_state": q.copy()}

    def test_consistency_rejects_anchor_labels_rgb_or_decode_changes(self):
        real = self.arrays()
        for key in ("observation_state", "target_source_recorded_actions", "rgb_input_sample_episode_frame", "predictions_source_actions"):
            changed = {name: value.copy() for name, value in real.items()}
            changed[key].flat[-1] += 1
            with self.subTest(key=key), self.assertRaises(conditioning.StateConditioningError):
                conditioning.intervention_consistency(changed, real, "within_episode_shift", "rgb_residual_scratch", self.np)
        changed = {name: value.copy() for name, value in real.items()}
        changed["rgb_input_bytes_sha256"][0] = "b" * 64
        with self.assertRaisesRegex(conditioning.StateConditioningError, "camera"):
            conditioning.intervention_consistency(changed, real, "within_episode_shift", "rgb_residual_scratch", self.np)

    def test_state_only_donor_equivariance_is_checked_in_network_output_not_anchor(self):
        np = self.np
        real = self.arrays()
        changed = {name: value.copy() for name, value in real.items()}
        donors = conditioning.state_donor_indices(self.ids, "cross_episode_swap", np)
        changed["network_state_donor_eval_offset"] = donors
        changed["predictions_scaled_state_residual"] = real["predictions_scaled_state_residual"][donors].copy()
        changed["predictions_raw_state_residual"] = changed["predictions_scaled_state_residual"] * real["action_std"]
        changed["predictions_source_actions"] = residual.decode_residual(changed["predictions_scaled_state_residual"], real["observation_state"], real["action_std"], np)
        result = conditioning.intervention_consistency(changed, real, "cross_episode_swap", "state_only_residual", np)
        self.assertTrue(result["state_only_donor_equivariance_checked"])
        changed["predictions_scaled_state_residual"][0, 0, 0] += np.float32(.01)
        changed["predictions_source_actions"] = residual.decode_residual(changed["predictions_scaled_state_residual"], real["observation_state"], real["action_std"], np)
        with self.assertRaisesRegex(conditioning.StateConditioningError, "donor"):
            conditioning.intervention_consistency(changed, real, "cross_episode_swap", "state_only_residual", np)

    def test_state_only_mean_residual_must_be_constant_while_raw_actions_follow_original_q(self):
        np = self.np
        real = self.arrays()
        changed = {name: value.copy() for name, value in real.items()}
        z = np.zeros_like(real["predictions_scaled_state_residual"])
        changed.update(predictions_scaled_state_residual=z, predictions_raw_state_residual=z.copy(),
            network_input_normalized_state=np.zeros_like(real["observation_state"]),
            predictions_source_actions=residual.decode_residual(z, real["observation_state"], real["action_std"], np))
        result = conditioning.intervention_consistency(changed, real, "train_mean", "state_only_residual", np)
        self.assertTrue(result["state_only_mean_residual_constant_checked"])
        self.assertEqual(changed["predictions_source_actions"].tobytes(), np.repeat(real["observation_state"][:, None, :], 16, axis=1).tobytes())
        changed["predictions_raw_state_residual"][0, 0, 0] = .01
        with self.assertRaisesRegex(conditioning.StateConditioningError, "constant"):
            conditioning.intervention_consistency(changed, real, "train_mean", "state_only_residual", np)

    def test_real_reproduction_checks_source_bits_and_numerical_tolerance(self):
        np = self.np
        from g1_vla.artifacts import sha256_file
        real = self.arrays()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.npz"
            np.savez(path, **real)
            record = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            self.assertTrue(conditioning._reproduction(real, record, np)["passed"])
            drift = {name: value.copy() for name, value in real.items()}
            drift["predictions_source_actions"][0, 0, 1] += np.float32(2e-6)
            with self.assertRaisesRegex(conditioning.StateConditioningError, "reproduce"):
                conditioning._reproduction(drift, record, np)
            drift = {name: value.copy() for name, value in real.items()}
            drift["observation_state"][0, 0] = np.float32(+0.0)
            with self.assertRaisesRegex(conditioning.StateConditioningError, "source bytes"):
                conditioning._reproduction(drift, record, np)

    def test_mean_state_constant_must_be_shared_across_both_splits(self):
        np = self.np
        train = np.full((4, 16, 28), .03125, dtype=np.float32)
        validation = np.full((3, 16, 28), .03125, dtype=np.float32)
        receipt = conditioning.verify_mean_state_cross_split(train, validation, np)
        self.assertEqual(receipt["combined_anchor_count"], 7)
        validation[:] += .001
        with self.assertRaisesRegex(conditioning.StateConditioningError, "across train and validation"):
            conditioning.verify_mean_state_cross_split(train, validation, np)

    def test_actual_inference_batch_has_only_intervened_q_and_unchanged_rgb_no_labels(self):
        np = self.np
        class Tensor:
            def __init__(self, values): self.values = values
            def to(self, *_): return self
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.values
        class Policy:
            def __init__(self): self.batches = []
            def eval(self): return self
            def predict_action_chunk(self, batch):
                self.batches.append({key: value.numpy().copy() for key, value in batch.items()})
                return Tensor(np.repeat(batch["observation.state"].numpy()[:, None, :], 16, axis=1))
        torch = SimpleNamespace(from_numpy=Tensor, inference_mode=nullcontext)
        stats = {key: (np.full(shape, .2, dtype=np.float32), np.full(shape, .5, dtype=np.float32))
            for key, shape in (("observation.state", (28,)), ("action", (28,)), (native.CAMERA, (3, 1, 1)))}
        targets = np.full((len(self.ids), 16, 28), 123.0, dtype=np.float32)
        def batch(_dataset, indices, _stats, _np, _torch, _device, include_actions):
            self.assertFalse(include_actions)
            samples = [{"observation.state": self.states[index].copy(), "action": targets[index].copy()} for index in indices]
            return {"observation.state": Tensor((self.states[indices] - self.mean) / self.std),
                native.CAMERA: Tensor(np.full((len(indices), 3, 2, 2), .375, dtype=np.float32))}, samples
        cohort = (list(range(len(self.ids))), self.ids, self.states, np.full(len(self.ids), np.nan, dtype=np.float64))
        for variant in conditioning.VARIANTS:
            policy = Policy()
            with patch.object(expanded, "_stats", return_value=stats), patch.object(native, "_batch", side_effect=batch), patch.object(native, "_check_gpu"):
                actual = conditioning._evaluate(policy, object(), cohort, {}, {"eval_batch_size": 8}, "train_mean", variant, np, torch, "cpu")
            for received in policy.batches:
                self.assertNotIn("action", received)
                self.assertNotIn("action_is_pad", received)
                self.assertTrue(np.all(received["observation.state"] == 0))
                self.assertEqual(native.CAMERA in received, variant == "rgb_residual_scratch")
                if native.CAMERA in received:
                    self.assertTrue(np.all(received[native.CAMERA] == .375))
            self.assertEqual(actual["observation_state"].tobytes(), self.states.tobytes())
            self.assertEqual(actual["target_source_recorded_actions"].tobytes(), targets.tobytes())
            self.assertEqual(actual["predictions_source_actions"].tobytes(), np.repeat(self.states[:, None, :], 16, axis=1).tobytes())


if __name__ == "__main__":
    unittest.main()
