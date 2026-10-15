"""Old-six statistics are an explicit treatment on unchanged expanded data."""

from contextlib import nullcontext
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from g1_vla import normalization_attribution as norm
from g1_vla import expanded_residual as expanded
from g1_vla import grasp_act as native
from g1_vla.artifacts import sha256_file


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def fixture(root="."):
    config = json.loads((Path(__file__).resolve().parents[1] / "configs" / "experiment.json").read_text())
    config["paths"]["root"] = str(root)
    settings = copy.deepcopy(config["expanded_residual"])
    settings.update(backend=norm.BACKEND, model_statistics_source=norm.MODEL_STATISTICS_SOURCE,
        reference_expanded_report_path="expanded_reference.json", reference_expanded_report_sha256="1" * 64,
        reference_statistics_manifest_path=config["grasp_act"]["dataset_manifest_path"],
        reference_statistics_manifest_sha256="2" * 64, reference_statistics_sha256="3" * 64,
        admission_report_path="normalization_admission.json", output_dir="normalization_outputs")
    config["normalization_attribution"] = settings
    return config


class NormalizationProtocolTests(unittest.TestCase):
    def test_only_explicit_old_statistics_and_frozen_budget_are_accepted(self):
        config = fixture()
        self.assertEqual(norm.validate_normalization_settings(config)["max_steps"], 2000)
        for key, value in (("model_statistics_source", "new_fourteen"), ("max_steps", 4000),
                ("max_steps", 2000.0), ("seed", True), ("max_train_wall_s", float("nan")),
                ("variants", list(reversed(norm.VARIANTS))), ("zero_head_init", False),
                ("include_step_zero_in_selection", False), ("use_vae", True),
                ("data_manifest_sha256", "0" * 64), ("reference_statistics_sha256", "A" * 64),
                ("reference_statistics_manifest_path", "../escaped"),
                ("reference_statistics_manifest_path", "wrong/manifest.json"), ("output_dir", "")):
            bad = copy.deepcopy(config)
            bad["normalization_attribution"][key] = value
            with self.subTest(key=key), self.assertRaises((norm.NormalizationAttributionError, native.GraspACTError)):
                norm.validate_normalization_settings(bad)

    def test_disabled_training_never_reaches_reference_data_or_model(self):
        for function in (norm.run_normalization_admission, norm.train_normalization_attribution):
            with tempfile.TemporaryDirectory() as directory:
                config = fixture(directory)
                config["train"]["enabled"] = False
                with patch.object(norm, "_prepare") as prepare:
                    report = function(config, directory)
                prepare.assert_not_called()
                self.assertFalse(report["passed"])
                self.assertFalse(report["training_started"])
                self.assertEqual(report["optimizer_steps_including_discarded_probes"], 0)
                self.assertFalse(report["test_payloads_opened"])

    def test_admission_failure_is_preserved_and_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            with patch.object(norm, "_prepare", side_effect=norm.NormalizationAttributionError("stats source mismatch")):
                report = norm.run_normalization_admission(config, directory)
            path = Path(directory) / "reports" / "normalization_admission.json"
            self.assertEqual(json.loads(path.read_text())["error"], report["error"])
            with self.assertRaises(norm.NormalizationAttributionError):
                norm.run_normalization_admission(config, directory)

    def test_reference_hash_fails_before_checkpoint_or_payload_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            (Path(directory) / "expanded_reference.json").write_text("{}")
            with patch.object(native, "_read", side_effect=AssertionError("unbound JSON read")):
                with self.assertRaisesRegex(norm.NormalizationAttributionError, "SHA256"):
                    norm._verify_expanded_reference(config, config["normalization_attribution"])

    def test_selection_preserves_step_zero_best_trained_and_last(self):
        points = [{"step": step, "metrics": {"policy": {"all_28": {"mae": mae}}}}
                  for step, mae in ((0, 0.04), (50, 0.05), (100, 0.045), (150, 0.06))]
        selection = norm.select_checkpoint(points)
        self.assertEqual(selection["selected_step"], 0)
        self.assertEqual(selection["best_trained_step"], 100)
        self.assertEqual(selection["last_step"], 150)
        self.assertFalse(selection["test_used_for_selection"])


@unittest.skipUnless(HAS_NUMPY, "NumPy is needed to verify actual statistical treatment")
class NormalizationArrayTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np

    def statistics(self):
        np = self.np
        return {"training_episode_ids": list(norm.ORIGINAL_TRAIN), "validation_or_test_used": False,
            "statistics_include_held_out": False, "std_minimum": 0.001,
            "features": {key: {"mean": np.full(shape, 0.123456789).tolist(),
                              "std": np.full(shape, 0.27182818).tolist()}
                for key, shape in (("observation.state", (28,)), ("action", (28,)), (native.CAMERA, (3, 1, 1)))}}

    def fake_torch(self):
        return SimpleNamespace(from_numpy=lambda value: SimpleNamespace(to=lambda _: value), inference_mode=nullcontext)

    def test_old_stats_reader_rejects_fourteen_fit_and_any_held_out_fit(self):
        stats = self.statistics()
        self.assertEqual(norm._stats(stats, self.np)["action"][1].dtype, self.np.float32)
        for key, value in (("training_episode_ids", list(norm.TRAIN)),
                           ("training_episode_ids", list(norm.ORIGINAL_TRAIN) + [180]),
                           ("validation_or_test_used", True), ("statistics_include_held_out", True)):
            bad = copy.deepcopy(stats)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(native.GraspACTError):
                norm._stats(bad, self.np)
        with self.assertRaises(expanded.ExpandedResidualError):
            expanded._stats(stats, self.np)

    def test_actual_training_batch_uses_old_q_rgb_and_action_scales(self):
        np = self.np
        statistics = self.statistics()
        statistics["features"]["action"]["std"] = np.linspace(0.04, 0.72, 28).tolist()
        state = np.linspace(-0.731, 0.819, 28, dtype=np.float32)
        action = np.repeat((state + np.float32(0.031))[None, :], 16, axis=0)
        image = np.full((3, 2, 2), 0.75, dtype=np.float32)
        sample = {"observation.state": state, "action": action, native.CAMERA: image,
                  "action_is_pad": np.zeros(16, dtype=np.bool_)}
        stats = norm._stats(statistics, np)
        with patch.object(expanded, "_stats", side_effect=AssertionError("expanded fit must never normalize this model")):
            for variant in norm.VARIANTS:
                batch, _ = norm._training_batch([sample], [0], stats, variant, np, self.fake_torch(), "unused")
                self.assertTrue(np.array_equal(batch["observation.state"],
                    np.asarray((state[None, :] - stats["observation.state"][0]) / stats["observation.state"][1], dtype=np.float32)))
                expected = (action[None, :, :] - state[None, None, :]) / stats["action"][1]
                self.assertEqual(batch["action"].tobytes(), expected.tobytes())
                self.assertFalse(batch["action_is_pad"].any())
                if variant == "state_only_residual":
                    self.assertNotIn(native.CAMERA, batch)
                else:
                    expected_image = np.asarray((image[None, :, :, :] - stats[native.CAMERA][0]) / stats[native.CAMERA][1], dtype=np.float32)
                    self.assertEqual(batch[native.CAMERA].tobytes(), expected_image.tobytes())

    def test_actual_inference_is_label_free_and_zero_decode_preserves_signed_zero(self):
        np = self.np
        states = np.linspace(-0.731, 0.819, 84, dtype=np.float32).reshape(3, 28)
        states[0, 0], states[1, 0] = np.float32(-0.0), np.float32(0.0)
        actions = np.repeat((states + np.float32(0.01))[:, None, :], 16, axis=1)
        samples = [{"observation.state": state, "action": action} for state, action in zip(states, actions)]
        dataset = SimpleNamespace(split="validation", samples=[(180, 0), (181, 0), (182, 0)],
            get_eval_indices=lambda stride: [0, 1, 2])
        def predict(batch):
            self.assertEqual(set(batch), {"observation.state"})
            value = np.zeros((3, 16, 28), dtype=np.float32)
            return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: value))
        policy = SimpleNamespace(eval=lambda: None, predict_action_chunk=predict)
        with patch.object(native, "_batch", return_value=({"observation.state": object(), native.CAMERA: object()}, samples)) as batch, \
             patch.object(native, "_check_gpu"), patch.object(expanded, "_stats", side_effect=AssertionError("new fit forbidden")):
            metrics, arrays = norm._evaluate(policy, dataset, self.statistics(),
                {"eval_stride": 30, "eval_batch_size": 8, "chunk_size": 16}, np, self.fake_torch(), "unused",
                variant="state_only_residual", require_zero=True)
        self.assertIs(batch.call_args.kwargs["include_actions"], False)
        self.assertEqual(arrays["predictions_source_actions"].tobytes(), np.repeat(states[:, None, :], 16, axis=1).tobytes())
        self.assertEqual(metrics["policy"], metrics["state_hold"])
        self.assertEqual(metrics["model_statistics_source"], norm.MODEL_STATISTICS_SOURCE)

    def test_expanded_cohort_denominators_do_not_revert_to_six_train(self):
        np = self.np
        states = np.zeros((14, 28), dtype=np.float32)
        actions = np.ones((14, 16, 28), dtype=np.float32)
        arrays = {"sample_episode_frame": np.asarray([(episode, 0) for episode in norm.TRAIN]),
            "observation_state": states, "target_source_recorded_actions": actions,
            "predictions_source_actions": actions * np.float32(0.75)}
        with patch.object(norm.residual, "_evaluate", return_value=({}, arrays)):
            metrics, _ = norm._evaluate(None, SimpleNamespace(split="train"), self.statistics(), {}, np,
                None, None, variant="state_only_residual")
        self.assertEqual(metrics["train_cohorts"]["original_six"]["anchor_count"], 6)
        self.assertEqual(metrics["train_cohorts"]["added_eight"]["anchor_count"], 8)
        self.assertEqual(sum(value["anchor_count"] for value in metrics["train_cohorts"].values()), 14)

    def test_original_statistics_manifest_and_content_must_match_source_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            settings = config["normalization_attribution"]
            path = native._resolve(config, settings["reference_statistics_manifest_path"])
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            settings["reference_statistics_manifest_sha256"] = sha256_file(path)
            binding = {"manifest_sha256": sha256_file(path), "statistics_sha256": settings["reference_statistics_sha256"],
                "selected_artifact_sha256": {"train_only.npy": "sha"}}
            frozen = {"bindings": copy.deepcopy(binding)}
            with patch.object(norm.diagnostics, "load_diagnostic_data", return_value=({}, self.statistics(), binding, {})):
                statistics, origin = norm._load_original_statistics(config, settings, frozen, self.np)
                self.assertEqual(statistics["training_episode_ids"], list(norm.ORIGINAL_TRAIN))
                self.assertTrue(origin["actual_train_rows_and_pixels_recomputed"])
                binding["statistics_sha256"] = "9" * 64
                with self.assertRaisesRegex(norm.NormalizationAttributionError, "frozen original-six"):
                    norm._load_original_statistics(config, settings, frozen, self.np)

    def test_checkpoint_keeps_true_fourteen_manifest_and_bound_six_model_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            statistics = self.statistics()
            manifest = {"split_episode_ids": {"train": list(norm.TRAIN), "validation": list(norm.VALIDATION)},
                "statistics": {"sha256": "a" * 64}}
            config = {"normalization_attribution": {"model_statistics_source": norm.MODEL_STATISTICS_SOURCE}}
            for name, value in (("manifest.json", manifest), ("train_stats.json", statistics), ("experiment_config.json", config)):
                (root / name).write_text(json.dumps(value))
            binding = {"model_statistics_source": norm.MODEL_STATISTICS_SOURCE,
                "dataset_manifest_statistics_not_modified": True, "model_statistics_are_train_subset": True,
                "model_statistics_training_episode_ids": list(norm.ORIGINAL_TRAIN), "dataset_training_episode_ids": list(norm.TRAIN),
                "dataset_statistics_sha256": "a" * 64, "data_statistics_sha256": "a" * 64,
                "statistics_sha256": "b" * 64, "model_statistics_sha256": "b" * 64,
                "model_statistics_source_file_sha256": "b" * 64, "normalization_statistics_sha256": "b" * 64,
                "model_statistics_content_sha256": native._digest(statistics)}
            with patch.object(native, "verify_grasp_checkpoint", return_value={"passed": True}) as verify:
                self.assertTrue(norm.verify_normalization_checkpoint(root, binding)["passed"])
                verify.assert_called_once_with(root, binding)
                for key, value in (("dataset_statistics_sha256", "b" * 64),
                                   ("model_statistics_content_sha256", "c" * 64),
                                   ("model_statistics_training_episode_ids", list(norm.TRAIN))):
                    bad = copy.deepcopy(binding)
                    bad[key] = value
                    with self.subTest(key=key), self.assertRaises(norm.NormalizationAttributionError):
                        norm.verify_normalization_checkpoint(root, bad)


if __name__ == "__main__":
    unittest.main()
