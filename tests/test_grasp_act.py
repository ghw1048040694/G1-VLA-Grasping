"""Admission and metric tests for the native 28-joint bounded ACT backend."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1_vla.grasp_act import (
    CAMERA, REPO, REVISION, SPLITS, GraspACTError, _digest, _stats,
    compute_action_metrics, eval_grasp_act, run_grasp_act_memory, train_grasp_act,
    validate_grasp_manifest, validate_grasp_settings, validate_memory_admission,
    verify_grasp_checkpoint,
)
from g1_vla.artifacts import sha256_file


HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def fixture(root="."):
    settings = {"backend": "lerobot_act_grasp_square", "dataset_manifest_path": "data/manifest.json",
                "memory_report_path": "runs/memory/grasp_act_memory.json", "checkpoint_path": "runs/train/best",
                "device": "cuda", "chunk_size": 16, "dim_model": 256, "n_heads": 8,
                "dim_feedforward": 1024, "n_encoder_layers": 2, "n_decoder_layers": 2,
                "n_vae_encoder_layers": 2, "latent_dim": 32, "dropout": 0.1,
                "weight_decay": 1e-4, "kl_weight": 10, "max_steps": 200, "eval_every_steps": 50,
                "max_train_wall_s": 1800, "max_gpu_reserved_bytes": 7 * 1024 ** 3,
                "eval_stride": 30, "eval_batch_size": 8}
    return {"paths": {"root": str(root)}, "grasp_act": settings,
            "train": {"backend": "grasp_square_act", "enabled": True, "authorization": "conditional",
                      "epochs": None, "seed": 20261014, "batch_size": 2, "learning_rate": 1e-4}}


def memory_fixture():
    config = fixture()
    binding = {"manifest_sha256": "a" * 64, "statistics_sha256": "b" * 64,
               "experiment_config_sha256": _digest(config), "module_sha256": "c" * 64,
               "train_batch_size": 2, "eval_batch_size": 8}
    return binding, {"status": "passed_memory_probe", "memory_passed": True, "bindings": binding,
                     "training_started": False, "probe_optimizer_steps": 1,
                     "temporary_parameters_discarded": True, "finite_gradients": True,
                     "eval_batch_probe_passed": True, "gpu": {"peak_reserved_bytes": 512 * 1024 ** 2}}


class GraspACTAdmissionTests(unittest.TestCase):
    def test_fixed_native_backend_rejects_other_protocol_and_invalid_budget(self):
        self.assertEqual(validate_grasp_settings(fixture())["chunk_size"], 16)
        for key, value in (("device", "cpu"), ("chunk_size", 8), ("dim_model", 255),
                           ("max_steps", True), ("dropout", float("nan")),
                           ("max_gpu_reserved_bytes", 7 * 1024 ** 3 + 1)):
            config = fixture()
            config["grasp_act"][key] = value
            with self.subTest(key=key), self.assertRaises(GraspACTError):
                validate_grasp_settings(config)
        for key, value in (("backend", "legacy"), ("epochs", 5), ("batch_size", 0),
                           ("learning_rate", float("inf")), ("seed", False)):
            config = fixture()
            config["train"][key] = value
            with self.subTest(key=key), self.assertRaises(GraspACTError):
                validate_grasp_settings(config)

    def test_memory_admission_requires_real_step_finite_grads_and_exact_full_binding(self):
        binding, report = memory_fixture()
        self.assertIs(validate_memory_admission(report, binding, fixture()["grasp_act"]), report)
        mutations = [lambda item: item.update(memory_passed=False),
                     lambda item: item.update(probe_optimizer_steps=0),
                     lambda item: item.update(training_started=True),
                     lambda item: item.update(finite_gradients=False),
                     lambda item: item.update(eval_batch_probe_passed=False),
                     lambda item: item.update(temporary_parameters_discarded=False),
                     lambda item: item["gpu"].update(peak_reserved_bytes=7 * 1024 ** 3 + 1),
                     lambda item: item["bindings"].update(manifest_sha256="d" * 64),
                     lambda item: item["bindings"].update(train_batch_size=4),
                     lambda item: item["bindings"].update(module_sha256="d" * 64)]
        for mutation in mutations:
            changed = copy.deepcopy(report)
            mutation(changed)
            with self.assertRaises(GraspACTError):
                validate_memory_admission(changed, binding, fixture()["grasp_act"])

    def test_failed_memory_stage_is_recorded_and_never_creates_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            with patch("g1_vla.grasp_act._data", side_effect=GraspACTError("incomplete source RGB")):
                report = run_grasp_act_memory(config, directory)
            self.assertFalse(report["memory_passed"])
            self.assertFalse(report["training_started"])
            self.assertEqual(report["probe_optimizer_steps"], 0)
            self.assertIn("incomplete source RGB", report["error"]["message"])
            self.assertFalse((Path(directory) / "runs/train/best").exists())
            self.assertEqual(json.loads((Path(directory) / "reports/grasp_act_memory.json").read_text())["status"], "failed")
            with self.assertRaises(GraspACTError):
                run_grasp_act_memory(config, directory)

    def test_disabled_training_never_reads_data_or_constructs_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            config["train"]["enabled"] = False
            with patch("g1_vla.grasp_act._data") as data, patch("g1_vla.grasp_act._runtime") as runtime:
                report = train_grasp_act(config, directory)
            data.assert_not_called(); runtime.assert_not_called()
            self.assertFalse(report["training_started"])
            self.assertEqual(report["optimizer_steps"], 0)

    def test_training_refuses_missing_memory_evidence_before_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            with patch("g1_vla.grasp_act._data", return_value=({}, {}, {}, {})), patch("g1_vla.grasp_act._runtime") as runtime:
                report = train_grasp_act(config, directory)
            runtime.assert_not_called()
            self.assertFalse(report["training_started"])
            self.assertIn("FileNotFoundError", report["error"]["type"])

    def _checkpoint(self, root):
        binding, _ = memory_fixture()
        names = ("model.safetensors", "config.json", "experiment_config.json", "manifest.json",
                 "train_stats.json", "training_summary.json")
        files = {}
        for name in names:
            path = root / name
            path.write_bytes(b"immutable full bytes for hash-admission test")
            files[name] = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        provenance = {"passed": True, "bindings": copy.deepcopy(binding), "test_used_for_selection": False, "files": files}
        (root / "provenance.json").write_text(json.dumps(provenance))
        return binding, provenance

    def test_checkpoint_requires_all_independent_metadata_and_full_file_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding, _ = self._checkpoint(root)
            self.assertTrue(verify_grasp_checkpoint(root, binding)["passed"])
            (root / "train_stats.json").write_bytes(b"held-out replacement")
            with self.assertRaises(GraspACTError):
                verify_grasp_checkpoint(root, binding)
        for mutation in (lambda item: item.update(test_used_for_selection=True),
                         lambda item: item["files"].pop("experiment_config.json"),
                         lambda item: item["bindings"].update(statistics_sha256="e" * 64)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                binding, provenance = self._checkpoint(root)
                mutation(provenance)
                (root / "provenance.json").write_text(json.dumps(provenance))
                with self.assertRaises(GraspACTError):
                    verify_grasp_checkpoint(root, binding)

    def test_second_test_evaluation_is_refused_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            checkpoint = Path(directory) / "runs/train/best"
            checkpoint.mkdir(parents=True)
            (checkpoint / "test_evaluation_claim.json").write_text('{"already":"evaluated"}')
            with patch("g1_vla.grasp_act._data", return_value=({}, {}, {}, {})), \
                    patch("g1_vla.grasp_act.verify_grasp_checkpoint", return_value={}), \
                    patch("g1_vla.grasp_act._runtime") as runtime:
                report = eval_grasp_act(config, directory)
            runtime.assert_not_called()
            self.assertFalse(report["evaluation_passed"])
            self.assertIn("already has a test-evaluation claim", report["error"]["message"])


@unittest.skipUnless(HAS_NUMPY, "optional audit NumPy runtime is unavailable")
class GraspACTArrayTests(unittest.TestCase):
    def _manifest(self):
        from g1_vla.external_data import SOURCE_JOINT_NAMES
        return {"passed": True, "source_repo_id": REPO, "source_revision": REVISION,
                "source_state_joint_names": list(SOURCE_JOINT_NAMES), "source_action_joint_names": list(SOURCE_JOINT_NAMES),
                "action_dim": 28, "source_fps": 30, "chunk_steps": 16, "camera_key": CAMERA,
                "image_shape_hwc": [240, 320, 3], "split_episode_ids": copy.deepcopy(SPLITS),
                "episodes": [{"episode_id": episode, "split": split, "row_count": 50, "sample_count": 35}
                             for split, episodes in SPLITS.items() for episode in episodes]}

    def test_manifest_rejects_padded_joints_relabelled_camera_and_changed_holdout(self):
        self.assertTrue(validate_grasp_manifest(self._manifest(), fixture()["grasp_act"])["passed"])
        for mutation in (lambda item: item.update(action_dim=31), lambda item: item.update(source_revision="main"),
                         lambda item: item.update(camera_key="observation.images.head"),
                         lambda item: item["source_action_joint_names"].reverse(),
                         lambda item: item["split_episode_ids"]["test"].append(0),
                         lambda item: item["episodes"][0].update(sample_count=50),
                         lambda item: item["episodes"][0].update(split="test")):
            changed = self._manifest()
            mutation(changed)
            with self.assertRaises(GraspACTError):
                validate_grasp_manifest(changed, fixture()["grasp_act"])

    def test_metrics_keep_identical_full_chunk_denominator_and_joint_groups(self):
        import numpy as np
        target = np.zeros((2, 16, 28), dtype=np.float32)
        prediction = target.copy()
        prediction[:, :, :14] = 1
        prediction[:, :, 14:] = 2
        metrics = compute_action_metrics(prediction, target, np)
        self.assertEqual(metrics["valid_scalar_count"], 2 * 16 * 28)
        self.assertEqual(metrics["all_28"]["mae"], 1.5)
        self.assertAlmostEqual(metrics["all_28"]["rmse"], np.sqrt(2.5))
        self.assertEqual(metrics["arms_14"]["mae"], 1)
        self.assertEqual(metrics["hands_14"]["mae"], 2)
        self.assertEqual(metrics["first_step_28"]["mae"], 1.5)
        for invalid in (np.zeros((2, 16, 31)), np.full_like(target, np.nan)):
            with self.assertRaises(GraspACTError):
                compute_action_metrics(invalid, target, np)

    def test_normalization_rejects_heldout_contamination_or_nonpositive_std(self):
        import numpy as np
        statistics = {"training_episode_ids": SPLITS["train"], "validation_or_test_used": False,
                      "statistics_include_held_out": False, "std_minimum": .001, "features": {
                          "observation.state": {"mean": [0] * 28, "std": [1] * 28},
                          "action": {"mean": [0] * 28, "std": [1] * 28},
                          CAMERA: {"mean": [[[0]], [[0]], [[0]]], "std": [[[1]], [[1]], [[1]]]}}}
        self.assertEqual(_stats(statistics, np)["action"][0].shape, (28,))
        for mutation in (lambda item: item.update(validation_or_test_used=True),
                         lambda item: item.update(statistics_include_held_out=True),
                         lambda item: item.update(std_minimum=.0001),
                         lambda item: item["training_episode_ids"].append(180),
                         lambda item: item["features"]["action"]["std"].__setitem__(0, 0),
                         lambda item: item["features"][CAMERA].update(mean=[0, 0, 0])):
            changed = copy.deepcopy(statistics)
            mutation(changed)
            with self.assertRaises(GraspACTError):
                _stats(changed, np)


if __name__ == "__main__":
    unittest.main()
