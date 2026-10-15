"""The constant diagnostic fits train labels once and preserves raw inference."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1_vla import constant_residual as constant
from g1_vla import expanded_residual as expanded
from g1_vla import grasp_act as native
from g1_vla.artifacts import sha256_file

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def fixture(root="."):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "experiment.json").read_text())
    config["paths"]["root"] = str(root)
    config["constant_residual"] = {"backend": constant.BACKEND, "method": constant.METHOD,
        "decode": constant.DECODE, "fit_cohorts": list(constant.FIT_COHORTS), "eval_stride": 30,
        "output_dir": "constant_outputs", "reference_expanded_report_path": "expanded_report.json",
        "reference_expanded_report_sha256": "0" * 64}
    return config


class ConstantProtocolTests(unittest.TestCase):
    def test_predeclared_median_and_frozen_validation_cannot_be_changed(self):
        config = fixture()
        self.assertEqual(constant.validate_constant_residual_settings(config)["fit_cohorts"], list(constant.FIT_COHORTS))
        for key, value in (("backend", "other"), ("method", "validation_best_bias"), ("decode", "inverse_state_normalization"),
                           ("fit_cohorts", ["expanded_fourteen"]), ("eval_stride", True), ("eval_stride", 30.0),
                           ("eval_stride", 15), ("reference_expanded_report_sha256", "A" * 64),
                           ("output_dir", "../escape"), ("output_dir", "")):
            changed = copy.deepcopy(config)
            changed["constant_residual"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((constant.ConstantResidualError, native.GraspACTError)):
                constant.validate_constant_residual_settings(changed)
        changed = copy.deepcopy(config)
        changed["native_diagnostics"]["stationary_speed_rad_s"] = .03
        with self.assertRaises(constant.ConstantResidualError):
            constant.validate_constant_residual_settings(changed)

    def test_failed_binding_is_saved_without_training_or_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            with patch.object(constant, "_prepare", side_effect=constant.ConstantResidualError("source changed")):
                result = constant.run_constant_residual_audit(config, directory)
            self.assertFalse(result["passed"])
            self.assertFalse(result["training_started"])
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertFalse(result["closed_form_train_fit_performed"])
            self.assertFalse(result["test_payloads_opened"])
            target = Path(directory) / "reports" / "constant_residual.json"
            self.assertEqual(json.loads(target.read_text())["error"], result["error"])
            with self.assertRaises(constant.ConstantResidualError):
                constant.run_constant_residual_audit(config, directory)


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for real closed-form fitting and raw decoding")
class ConstantArrayTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np

    def episode(self, count=40, state=0., residual=0.):
        np = self.np
        states = np.full((count, 28), state, dtype=np.float32)
        actions = states + np.float32(residual)
        return states, actions

    def test_median_pools_anchors_and_does_not_average_episode_medians(self):
        np = self.np
        episodes = {20: self.episode(count=16, residual=100.), 21: self.episode(count=18, residual=2.),
                    180: self.episode(count=100, residual=-1000.)}
        parameter, report = constant.fit_constant_residual(episodes, [20, 21], np)
        self.assertTrue(np.array_equal(parameter, np.full((16, 28), 2., dtype=np.float32)))
        self.assertEqual(report["complete_anchor_count"], 4)
        self.assertEqual(report["complete_anchor_count_by_episode"], {"20": 1, "21": 3})
        self.assertEqual(report["scalar_label_count"], 4 * 16 * 28)
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(report["compute_matched_to_neural_training"])
        episodes[180] = self.episode(count=100, residual=9999.)
        other, _ = constant.fit_constant_residual(episodes, [20, 21], np)
        self.assertEqual(parameter.tobytes(), other.tobytes())

    def test_even_population_uses_the_declared_central_midpoint(self):
        np = self.np
        parameter, report = constant.fit_constant_residual(
            {20: self.episode(16, residual=2.), 21: self.episode(16, residual=8.)}, [20, 21], np)
        self.assertTrue(np.array_equal(parameter, np.full((16, 28), 5., dtype=np.float32)))
        self.assertEqual(report["even_population_rule"], "midpoint_of_two_central_values")
        self.assertEqual(report["training_raw_residual_l1_at_float64_median"], 3.)
        self.assertEqual(report["parameter_bytes_sha256"], hashlib.sha256(parameter.tobytes()).hexdigest())

    def test_every_horizon_uses_the_same_raw_anchor_q(self):
        np = self.np
        states = np.arange(16, dtype=np.float32)[:, None] * np.ones((1, 28), dtype=np.float32)
        actions = (states * np.float32(2)) + np.float32(3)
        parameter, _ = constant.fit_constant_residual({20: (states, actions)}, [20], np)
        self.assertTrue(np.array_equal(parameter[:, 0], np.arange(16, dtype=np.float32) * 2 + 3))
        # A rolling q[t+h] subtraction would instead produce h+3.
        self.assertNotEqual(float(parameter[15, 0]), 18.)

    def test_float64_source_difference_then_one_float32_quantization_is_explicit(self):
        np = self.np
        state, action = self.episode(16)
        state[:] = np.float32(.123456789)
        action[:] = np.float32(.987654321)
        with patch.object(np, "median", wraps=np.median) as median:
            parameter, report = constant.fit_constant_residual({20: (state, action)}, [20], np)
        self.assertEqual(median.call_args.args[0].dtype, np.float64)
        ideal = float(action[0, 0]) - float(state[0, 0])
        self.assertEqual(float(parameter[0, 0]), float(np.float32(ideal)))
        self.assertEqual(report["subtraction_and_median_dtype"], "float64")
        self.assertEqual(report["parameter_dtype"], "float32")
        self.assertEqual(report["parameter_quantization_max_abs_rad"], abs(ideal - float(np.float32(ideal))))

    def test_zero_and_negative_zero_parameters_preserve_every_anchor_bit(self):
        np = self.np
        states = np.full((2, 28), .123456789, dtype=np.float32)
        states[0, ::2] = np.float32(-0.)
        states[1, 1::2] = np.float32(-0.)
        parameter = np.zeros((16, 28), dtype=np.float32)
        parameter[::2, ::2] = np.float32(-0.)
        decoded = constant.decode_constant_residual(states, parameter, np)
        self.assertEqual(decoded.tobytes(), np.broadcast_to(states[:, None, :], (2, 16, 28)).copy().tobytes())
        self.assertEqual(decoded.dtype, np.float32)
        self.assertTrue(decoded.flags.c_contiguous)

    def test_nonzero_parameter_decodes_direct_raw_state_without_normalization(self):
        np = self.np
        state = np.full((2, 28), .1, dtype=np.float32)
        parameter = np.full((16, 28), .015, dtype=np.float32)
        decoded = constant.decode_constant_residual(state, parameter, np)
        self.assertEqual(decoded.tobytes(), np.broadcast_to(state[:, None, :] + parameter, (2, 16, 28)).copy().tobytes())

    def test_invalid_fits_and_decodes_are_rejected(self):
        np = self.np
        episodes = {20: self.episode(16)}
        for ids in ([], [20, 20], [180]):
            with self.subTest(ids=ids), self.assertRaises(constant.ConstantResidualError):
                constant.fit_constant_residual(episodes, ids, np)
        for state, action in ((np.zeros((16, 28)), np.zeros((16, 28))),
                              (np.zeros((15, 28), dtype=np.float32), np.zeros((15, 28), dtype=np.float32)),
                              (np.zeros((16, 27), dtype=np.float32), np.zeros((16, 27), dtype=np.float32))):
            with self.assertRaises(constant.ConstantResidualError):
                constant.fit_constant_residual({20: (state, action)}, [20], np)
        states = np.zeros((2, 28), dtype=np.float32)
        for parameter in (np.zeros((16, 28)), np.zeros((15, 28), dtype=np.float32),
                          np.full((16, 28), np.nan, dtype=np.float32)):
            with self.assertRaises(constant.ConstantResidualError):
                constant.decode_constant_residual(states, parameter, np)

    def test_evaluation_predictor_never_receives_labels_and_keeps_identical_denominators(self):
        np = self.np
        episodes = {20: self.episode(77, residual=.1), 21: self.episode(47, residual=.2)}
        parameter = np.full((16, 28), .05, dtype=np.float32)
        real_decode = constant.decode_constant_residual
        with patch.object(constant, "decode_constant_residual", wraps=real_decode) as decode:
            metrics, arrays = constant.evaluate_constant_residual(episodes, [20, 21], parameter, 30, np)
        self.assertEqual(decode.call_count, 2)  # actual fixed parameter and exact-zero identity audit
        for call in decode.call_args_list:
            self.assertEqual(len(call.args), 3)
            self.assertEqual(call.args[0].shape[1], 28)
            self.assertEqual(call.args[1].shape, (16, 28))
        self.assertEqual(arrays["sample_episode_frame"].tolist(), [[20, 0], [20, 30], [20, 60], [21, 0], [21, 30]])
        self.assertEqual(metrics["policy"]["valid_scalar_count"], 5 * 16 * 28)
        self.assertEqual(metrics["state_hold"]["valid_scalar_count"], 5 * 16 * 28)
        self.assertEqual(len(metrics["policy"]["mae_by_joint"]), 28)
        self.assertEqual(len(metrics["policy"]["mae_by_horizon"]), 16)
        self.assertEqual(set(metrics["per_episode"]), {"20", "21"})
        self.assertFalse(metrics["inference_received_labels"])
        self.assertFalse(metrics["inference_received_rgb"])

    def test_motion_strata_use_previous_raw_state_only_and_partition_denominator(self):
        np = self.np
        state, action = self.episode(19)
        # Four evaluated anchors 0..3: unknown, stationary, intermediate, moving.
        state[1] = .0001
        state[2] = .002
        state[3:] = .1
        metrics, arrays = constant.evaluate_constant_residual({20: (state, action)}, [20],
            np.zeros((16, 28), dtype=np.float32), 1, np)
        strata = metrics["motion_strata"]["strata"]
        self.assertEqual({name: value["anchor_count"] for name, value in strata.items()},
                         {"stationary": 1, "moving": 1, "intermediate": 1, "unknown_first_row": 1})
        self.assertFalse(arrays["anchor_speed_known"][0])
        self.assertTrue(np.isnan(arrays["anchor_max_joint_speed_rad_s"][0]))
        self.assertEqual(sum(item["anchor_count"] for item in strata.values()), 4)
        self.assertFalse(metrics["motion_strata"]["used_as_model_input"])

    def test_source_hash_is_rechecked_before_npz_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "arrays.npz"
            source.write_bytes(b"changed source")
            item = {"episode_id": 20, "row_count": 16, "sample_count": 1,
                    "arrays": {"relative_path": "arrays.npz", "size_bytes": source.stat().st_size, "sha256": "0" * 64}}
            with patch.object(self.np, "load", side_effect=AssertionError("unverified arrays opened")) as load:
                with self.assertRaises(ValueError):
                    constant._load_pair(root, item, self.np)
            load.assert_not_called()

    def test_complete_run_reports_fit_membership_and_never_compute_matches(self):
        np = self.np
        # The validation lengths reproduce the immutable 85-anchor stride-30 cohort.
        episodes = {identity: self.episode(46, residual=.1) for identity in expanded.TRAIN}
        for identity, length in zip(expanded.VALIDATION, (866, 866, 801)):
            episodes[identity] = self.episode(length, residual=.1)
        binding = {"source_arrays": {str(identity): {"sha256": str(identity)} for identity in episodes}}
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            config["train"]["enabled"] = False  # no optimizer/runtime exists in this closed-form audit
            with patch.object(constant, "_prepare", return_value=(episodes, binding)):
                report = constant.run_constant_residual_audit(config, directory)
            self.assertTrue(report["passed"], report.get("error"))
            self.assertEqual(report["status"], "completed_train_only_constant_residual_audit")
            self.assertFalse(report["training_started"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertTrue(report["closed_form_train_fit_performed"])
            self.assertFalse(report["compute_matched_to_neural_training"])
            self.assertEqual(set(report["fits"]), set(constant.FIT_COHORTS))
            first = report["fits"]["original_six"]
            second = report["fits"]["expanded_fourteen"]
            self.assertEqual(set(first["source_arrays"]), set(map(str, expanded.ORIGINAL_TRAIN)))
            self.assertEqual(first["evaluations"]["added_eight"]["fit_membership"], "excluded_from_fit_diagnostic_train_pool")
            self.assertEqual(first["evaluations"]["expanded_fourteen"]["fit_membership"], "mixed_fit_membership_diagnostic_train_pool")
            self.assertEqual(second["evaluations"]["added_eight"]["fit_membership"], "included_in_fit")
            for fit in report["fits"].values():
                self.assertEqual(fit["evaluations"]["validation"]["fit_membership"], "held_out_validation_never_fitted")
                self.assertEqual(fit["evaluations"]["validation"]["metrics"]["policy"]["sample_count"], 85)
                parameter_path = Path(fit["parameter_artifact"]["path"])
                self.assertEqual(sha256_file(parameter_path), fit["parameter_artifact"]["sha256"])
                with np.load(parameter_path, allow_pickle=False) as saved:
                    self.assertEqual(saved["constant_raw_residual"].dtype, np.float32)
                validation_path = fit["evaluations"]["validation"]["prediction_artifact"]["path"]
                with np.load(validation_path, allow_pickle=False) as saved:
                    self.assertEqual(saved["sample_episode_frame"].shape, (85, 2))

    def test_preparation_opens_only_selected_seventeen_npz_and_never_rgb_or_test(self):
        from g1_vla import grasp_data, train_expansion
        np = self.np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixture(directory)
            old_root, new_root = root / "original", root / "expanded"
            old_root.mkdir()
            new_root.mkdir()
            old_path, new_path = old_root / "manifest.json", new_root / "manifest.json"
            old_path.write_text('{"frozen":"original"}')
            new_path.write_text('{"frozen":"expanded"}')
            config["grasp_act"]["dataset_manifest_path"] = "original/manifest.json"
            config["expanded_residual"]["data_manifest_path"] = "expanded/manifest.json"
            config["expanded_residual"]["data_manifest_sha256"] = sha256_file(new_path)
            config["train_expansion"]["build_dir"] = "expanded"
            counts = dict(zip(expanded.ORIGINAL_TRAIN, (1000, 1000, 1000, 1000, 1000, 741)))
            counts.update(zip(expanded.CANDIDATES, (1000, 1000, 1000, 1000, 1000, 1000, 1000, 1304)))
            counts.update(zip(expanded.VALIDATION, (866, 866, 801)))
            items = []
            for identity, count in counts.items():
                state, action = self.episode(count, state=.1, residual=.01)
                path = new_root / (str(identity) + ".source.npz")
                with path.open("xb") as stream:
                    np.savez(stream, source_state=state, source_action=action,
                             source_frame_index=np.arange(count), source_episode_index=np.full(count, identity))
                record = {"relative_path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                items.append({"episode_id": identity, "split": "validation" if identity in expanded.VALIDATION else "train",
                              "row_count": count, "sample_count": count - 15, "arrays": record,
                              "rgb": {"relative_path": "MUST_NOT_OPEN.rgb.npy"}})
            original_items = [item for item in items if item["episode_id"] in expanded.ORIGINAL_TRAIN + expanded.VALIDATION]
            # Test metadata must remain ignorable even when the test payload path is missing.
            original_items.append({"episode_id": 280, "split": "test", "arrays": {"relative_path": "MUST_NOT_OPEN_TEST.npz"}})
            original = {"episodes": original_items}
            manifest = {"episodes": items}
            admission = {"audit": "fixed", "session_independence_verified": False}
            reference = {"passed": True, "status": "completed_expanded_train_validation_state_residual_comparison",
                "test_payloads_opened": False, "test_used_for_selection": False,
                "settings": copy.deepcopy(config["expanded_residual"]),
                "bindings": {"original_training_module_sha256": sha256_file(native.__file__),
                    "expanded_residual_module_sha256": sha256_file(expanded.__file__),
                    "expanded_data_module_sha256": sha256_file(train_expansion.__file__),
                    "original_manifest_sha256": sha256_file(old_path), "manifest_sha256": sha256_file(new_path),
                    "candidate_admission": admission,
                    "selected_artifact_sha256": {item["arrays"]["relative_path"]: item["arrays"]["sha256"] for item in items}}}
            reference_path = root / "expanded_report.json"
            reference_path.write_text(json.dumps(reference))
            config["constant_residual"]["reference_expanded_report_sha256"] = sha256_file(reference_path)
            real_load = np.load
            opened = []
            def restricted_load(path, *args, **kwargs):
                opened.append(str(path))
                self.assertTrue(str(path).endswith(".source.npz"))
                self.assertNotIn("MUST_NOT_OPEN", str(path))
                return real_load(path, *args, **kwargs)
            with patch.object(grasp_data, "load_grasp_manifest", return_value=original) as original_loader, \
                    patch.object(train_expansion, "load_expanded_manifest", return_value=manifest) as expanded_loader, \
                    patch.object(expanded, "_verify_expansion_audit", return_value=admission), \
                    patch.object(np, "load", side_effect=restricted_load):
                episodes, binding = constant._prepare(config, config["constant_residual"], np)
            self.assertEqual(set(episodes), set(expanded.TRAIN + expanded.VALIDATION))
            self.assertEqual(len(opened), 17)
            self.assertEqual(binding["complete_anchor_counts"], {"original_six": 5651, "expanded_fourteen": 13835, "validation": 2488})
            self.assertFalse(binding["rgb_payloads_opened"])
            self.assertFalse(binding["test_payloads_opened"])
            self.assertFalse(original_loader.call_args.kwargs["verify_files"])
            self.assertFalse(expanded_loader.call_args.kwargs["verify_files"])


if __name__ == "__main__":
    unittest.main()
