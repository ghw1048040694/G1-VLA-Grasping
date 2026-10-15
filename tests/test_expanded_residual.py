"""Expanded residual admission protects frozen validation and fresh budgets."""

from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from g1_vla import expanded_residual as expanded
from g1_vla import grasp_act as native
from g1_vla.artifacts import sha256_file


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def fixture(root="."):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "experiment.json").read_text())
    config["paths"]["root"] = str(root)
    previous = config["state_residual"]
    settings = {key: previous[key] for key in ("variants", "use_vae", "target", "decode", "zero_head_init",
        "include_step_zero_in_selection", "seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size",
        "max_train_wall_s", "max_gpu_reserved_bytes")}
    settings.update(backend=expanded.BACKEND, data_manifest_path="expanded/manifest.json", data_manifest_sha256="0" * 64,
        reference_residual_report_path="reference.json", reference_residual_report_sha256="1" * 64,
        admission_report_path="admission.json", output_dir="new_outputs")
    config["expanded_residual"] = settings
    config.setdefault("train_expansion", {})["build_dir"] = "expanded"
    return config


class ExpandedProtocolTests(unittest.TestCase):
    def test_frozen_training_budget_and_origin_cannot_drift(self):
        config = fixture()
        self.assertEqual(expanded.validate_expanded_residual_settings(config)["variants"], list(expanded.VARIANTS))
        changes = (("variants", list(reversed(expanded.VARIANTS))), ("use_vae", True),
            ("zero_head_init", False), ("include_step_zero_in_selection", False), ("target", "absolute"),
            ("decode", "inverse_normalized_q"), ("max_steps", 4000), ("max_steps", 2000.0),
            ("seed", True), ("eval_batch_size", 4), ("eval_every_steps", 100),
            ("max_train_wall_s", float("nan")), ("data_manifest_sha256", "A" * 64),
            ("data_manifest_path", "../escape"), ("admission_report_path", ""))
        for key, value in changes:
            changed = copy.deepcopy(config)
            changed["expanded_residual"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((expanded.ExpandedResidualError, native.GraspACTError)):
                expanded.validate_expanded_residual_settings(changed)

    def test_disabled_training_stops_before_reference_data_or_model(self):
        for function in (expanded.run_expanded_residual_admission, expanded.train_expanded_residual):
            with tempfile.TemporaryDirectory() as directory:
                config = fixture(directory)
                config["train"]["enabled"] = False
                with patch.object(expanded, "_prepare") as prepare:
                    report = function(config, directory)
                prepare.assert_not_called()
                self.assertFalse(report["passed"])
                self.assertFalse(report["training_started"])
                self.assertEqual(report["optimizer_steps"], 0)
                self.assertEqual(report["discarded_probe_optimizer_steps"], 0)
                self.assertFalse(report["test_payloads_opened"])

    def test_failed_manifest_admission_is_saved_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            with patch.object(expanded, "_prepare", side_effect=expanded.ExpandedResidualError("candidate build is unverified")):
                report = expanded.run_expanded_residual_admission(config, directory)
            self.assertFalse(report["training_started"])
            self.assertEqual(report["optimizer_steps_including_discarded_probes"], 0)
            saved = Path(directory) / "reports" / "expanded_residual_admission.json"
            self.assertEqual(json.loads(saved.read_text())["error"], report["error"])
            with self.assertRaises(expanded.ExpandedResidualError):
                expanded.run_expanded_residual_admission(config, directory)

    def test_unbound_manifest_rejected_before_payload_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            build = Path(directory) / "expanded"
            build.mkdir()
            (build / "manifest.json").write_text("{}")
            with self.assertRaisesRegex(expanded.ExpandedResidualError, "full SHA256"):
                expanded.load_expanded_data(config, config["expanded_residual"], {})

    def test_current_candidate_audit_session_file_and_failure_marker_are_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = fixture(directory)
            candidates_root = root / "candidates"
            candidates_root.mkdir()
            evidence = root / "session.json"
            evidence.write_text('{"sessions": "unknown"}')
            binding = {"session_evidence_sha256": sha256_file(evidence)}
            settings = {"audit_report_path": "audit.json", "candidate_build_dir": "candidates",
                        "session_evidence_path": "session.json", "session_evidence_sha256": sha256_file(evidence)}
            items = [{"episode_id": identity, **{key: {"sha256": str(identity)} for key in
                      ("arrays", "rgb", "decoded_receipt")}} for identity in expanded.CANDIDATES]
            candidate_path = candidates_root / "candidate_manifest.json"
            candidate_path.write_text(json.dumps({"passed": True, "bindings": binding, "episodes": items,
                "episode_ids": list(expanded.CANDIDATES), "independent_session_data_admitted": False,
                "admission_class": "same_source_coverage_only"}))
            audit = {"passed": True, "status": "passed_complete_candidate_protocol_same_source_coverage",
                "settings": settings, "bindings": binding, "candidate_episode_ids": list(expanded.CANDIDATES),
                "candidate_manifest_sha256": sha256_file(candidate_path), "candidate_manifest_path": str(candidate_path),
                "admission_class": "same_source_coverage_only"}
            for key in ("test_payloads_opened", "test_metrics_read", "independent_session_data_admitted",
                        "exact_state_action_duplicate_trajectories_found", "capture_session_mapping_available",
                        "scene_operator_independence_verified"):
                audit[key] = False
            for key in ("all_candidates_required_no_performance_filtering", "all_camera_direct_frozen_file_disjoint",
                        "source_label_protocol_preserved"):
                audit[key] = True
            audit_path = root / "audit.json"
            audit_path.write_text(json.dumps(audit))
            manifest = {"settings": settings, "bindings": binding, "episodes": items,
                        "candidate_audit_path": str(audit_path),
                        "candidate_audit_sha256": sha256_file(audit_path),
                        "candidate_manifest_sha256": sha256_file(candidate_path)}
            module = SimpleNamespace(validate_train_expansion_settings=lambda _: settings,
                                     _base=lambda *_: (None, None, None, binding))
            result = expanded._verify_expansion_audit(config, manifest, module)
            self.assertFalse(result["session_independence_verified"])
            self.assertTrue(result["actual_audit_and_candidate_files_reopened"])
            evidence.write_text('{"sessions": "changed"}')
            with self.assertRaisesRegex(expanded.ExpandedResidualError, "session evidence changed"):
                expanded._verify_expansion_audit(config, manifest, module)
            evidence.write_text('{"sessions": "unknown"}')
            (candidates_root / "failed.json").write_text("{}")
            with self.assertRaisesRegex(expanded.ExpandedResidualError, "partial-failure"):
                expanded._verify_expansion_audit(config, manifest, module)
            (candidates_root / "failed.json").unlink()
            audit["independent_session_data_admitted"] = True
            audit_path.write_text(json.dumps(audit))
            manifest["candidate_audit_sha256"] = sha256_file(audit_path)
            with self.assertRaisesRegex(expanded.ExpandedResidualError, "same-source-only"):
                expanded._verify_expansion_audit(config, manifest, module)


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for real expanded normalization and raw decode")
class ExpandedArrayTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np

    def statistics(self):
        np = self.np
        return {"training_episode_ids": list(expanded.TRAIN), "validation_or_test_used": False,
            "statistics_include_held_out": False, "std_minimum": 0.001,
            "features": {key: {"mean": np.full(shape, 0.123456789).tolist(), "std": np.full(shape, 0.27182818).tolist()}
                for key, shape in (("observation.state", (28,)), ("action", (28,)), (native.CAMERA, (3, 1, 1)))}}

    def test_expanded_stats_are_real_fourteen_episode_fit_not_renamed_old_six(self):
        statistics = self.statistics()
        with patch.object(native, "_stats", side_effect=AssertionError("hard-coded old IDs must not be used")):
            result = expanded._stats(statistics, self.np)
        self.assertEqual(result["action"][1].dtype, self.np.float32)
        for key, value in (("training_episode_ids", list(expanded.ORIGINAL_TRAIN)),
                           ("training_episode_ids", list(expanded.TRAIN) + [180]),
                           ("validation_or_test_used", True), ("statistics_include_held_out", True)):
            bad = copy.deepcopy(statistics)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(expanded.ExpandedResidualError):
                expanded._stats(bad, self.np)

    def test_two_expanded_arms_share_new_sampling_and_keep_four_thousand_draw_budget(self):
        np = self.np
        class Samples:
            def __len__(self):
                return len(self.samples)
        dataset = Samples()
        dataset.samples = [(int(expanded.TRAIN[index % 14]), index) for index in range(13835)]
        settings = fixture()["expanded_residual"]
        first = expanded.diagnostics.anchor_schedule(dataset, settings, 2, 20261014, np)
        second = expanded.diagnostics.anchor_schedule(dataset, settings, 2, 20261014, np)
        self.assertTrue(np.array_equal(first[0], second[0]))
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[2]["sample_draw_count"], 4000)
        self.assertEqual(first[2]["available_train_anchor_count"], 13835)
        self.assertLess(first[2]["unique_train_anchor_fraction"], 0.3)

    def test_raw_zero_identity_and_label_free_inference_survive_changed_statistics(self):
        np = self.np
        states = np.linspace(-0.731, 0.819, 84, dtype=np.float32).reshape(3, 28)
        states[0, 0], states[1, 0] = np.float32(-0.0), np.float32(0.0)
        actions = np.repeat((states + np.float32(0.01))[:, None, :], 16, axis=1)
        samples = [{"observation.state": state.copy(), "action": action.copy()} for state, action in zip(states, actions)]
        dataset = SimpleNamespace(split="validation", samples=[(180, 0), (181, 0), (182, 0)],
                                  get_eval_indices=lambda stride: [0, 1, 2])
        policy = SimpleNamespace(eval=lambda: None)
        seen = []
        def predict(batch):
            seen.append(set(batch))
            self.assertEqual(set(batch), {"observation.state"})
            result = np.zeros((3, 16, 28), dtype=np.float32)
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: result))
        policy.predict_action_chunk = predict
        def batch(*args, **kwargs):
            self.assertIs(kwargs["include_actions"], False)
            return {"observation.state": object(), native.CAMERA: object()}, samples
        torch = SimpleNamespace(inference_mode=nullcontext)
        settings = {"eval_stride": 30, "eval_batch_size": 8, "chunk_size": 16}
        with patch.object(native, "_batch", side_effect=batch), patch.object(native, "_check_gpu"), \
             patch.object(native, "_stats", side_effect=AssertionError("old stats forbidden")):
            metrics, arrays = expanded._evaluate(policy, dataset, self.statistics(), settings, np, torch, "unused",
                                                 variant="state_only_residual", require_zero=True)
        self.assertEqual(arrays["predictions_source_actions"].tobytes(), np.repeat(states[:, None, :], 16, axis=1).tobytes())
        self.assertEqual(metrics["policy"], metrics["state_hold"])
        self.assertEqual(len(seen), 1)

    def test_old_six_and_new_eight_reporting_keeps_same_raw_denominators(self):
        np = self.np
        ids = np.array([(episode, 0) for episode in expanded.TRAIN], dtype=np.int64)
        state = np.zeros((14, 28), dtype=np.float32)
        target = np.ones((14, 16, 28), dtype=np.float32)
        prediction = target * np.float32(0.75)
        arrays = {"sample_episode_frame": ids, "observation_state": state,
                  "target_source_recorded_actions": target, "predictions_source_actions": prediction}
        cohorts = expanded._train_cohorts(arrays, expanded._stats(self.statistics(), np), np)
        self.assertEqual(cohorts["original_six"]["anchor_count"], 6)
        self.assertEqual(cohorts["added_eight"]["anchor_count"], 8)
        for value in cohorts.values():
            self.assertFalse(value["used_for_checkpoint_selection"])
            self.assertEqual(value["metrics"]["policy"]["all_28"]["mae"], 0.25)

    def test_rgb_nonzero_decode_uses_expanded_scale_and_direct_anchor_without_action_inputs(self):
        np = self.np
        state = np.linspace(-0.731, 0.819, 28, dtype=np.float32)[None, :]
        action = np.repeat(state[:, None, :], 16, axis=1)
        residual = np.full((1, 16, 28), 0.037, dtype=np.float32)
        statistics = self.statistics()
        statistics["features"]["action"]["std"] = np.linspace(0.04, 0.72, 28).tolist()
        samples = [{"observation.state": state[0], "action": action[0]}]
        dataset = SimpleNamespace(split="validation", samples=[(180, 0)], get_eval_indices=lambda stride: [0])
        policy = SimpleNamespace(eval=lambda: None)
        def predict(batch):
            self.assertEqual(set(batch), {"observation.state", native.CAMERA})
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: residual))
        policy.predict_action_chunk = predict
        with patch.object(native, "_batch", return_value=({"observation.state": object(), native.CAMERA: object()}, samples)), \
             patch.object(native, "_check_gpu"):
            _, arrays = expanded._evaluate(policy, dataset, statistics,
                {"eval_stride": 30, "eval_batch_size": 8, "chunk_size": 16}, np,
                SimpleNamespace(inference_mode=nullcontext), "unused", variant="rgb_residual_scratch")
        expected = state[:, None, :] + np.asarray(statistics["features"]["action"]["std"], dtype=np.float32) * residual
        self.assertEqual(arrays["predictions_source_actions"].tobytes(), expected.tobytes())

    def test_motion_strata_use_previous_observation_and_partition_the_original_denominator(self):
        np = self.np
        ids = np.array([[180, 0], [180, 1], [181, 1], [182, 1]], dtype=np.int64)
        source = {180: (np.zeros((2, 28), dtype=np.float32), None),
                  181: (np.repeat(np.array([0, .002], dtype=np.float32)[:, None], 28, axis=1), None),
                  182: (np.repeat(np.array([0, .02], dtype=np.float32)[:, None], 28, axis=1), None)}
        states = np.stack([source[int(episode)][0][int(frame)] for episode, frame in ids])
        predictions = np.repeat(states[:, None, :], 16, axis=1)
        arrays = {"sample_episode_frame": ids, "observation_state": states,
                  "predictions_source_actions": predictions,
                  "target_source_recorded_actions": np.full_like(predictions, 99.0)}
        result = expanded._motion_strata(SimpleNamespace(_arrays=source), arrays, self.statistics(),
            {"native_diagnostics": {"stationary_speed_rad_s": .02, "moving_speed_rad_s": .1}}, np)
        self.assertEqual({key: value["anchor_count"] for key, value in result["strata"].items()},
            {"stationary": 1, "moving": 1, "intermediate": 1, "unknown_first_row": 1})
        self.assertEqual(result["partition_anchor_count"], 4)
        self.assertFalse(result["used_as_model_input"])
        self.assertTrue(np.isnan(arrays["anchor_max_joint_speed_rad_s"][0]))


if __name__ == "__main__":
    unittest.main()
