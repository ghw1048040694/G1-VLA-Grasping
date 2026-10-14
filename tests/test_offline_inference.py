"""Native evidence gates: no dimensional padding, provenance forgery or fake RGB."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1_vla.offline_inference import (
    CAMERA, DATASET_REPO, DATASET_REVISION, MODEL_REPO, MODEL_REVISION,
    OfflineInferenceError, run_native_act_inference, validate_native_act_metadata,
    verify_native_provenance, _parse_act_config,
)


def fixture():
    arms = ["arm_{}".format(index) for index in range(14)]
    names = ["body_{}".format(index) for index in range(15)] + arms
    actions = arms + ["remote.lx", "remote.ly", "remote.rx", "remote.ry", "left_hand_open", "right_hand_open"]
    inputs = {"observation.state": {"type": "STATE", "shape": [29]}, CAMERA: {"type": "VISUAL", "shape": [3, 480, 640]}}
    output = {"action": {"type": "ACTION", "shape": [20]}}
    norm_map = {"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}
    model = {"type": "act", "n_obs_steps": 1, "chunk_size": 100, "input_features": inputs,
             "output_features": output, "normalization_mapping": norm_map}
    info = {"codebase_version": "v3.0", "fps": 30, "features": {
        "observation.state": {"shape": [29], "names": names}, "action": {"shape": [20], "names": actions},
        CAMERA: {"shape": [480, 640, 3]}}}
    pre = {"steps": [{"registry_name": "rename_observations_processor", "config": {"rename_map": {}}},
                     {"registry_name": "to_batch_processor", "config": {}},
                     {"registry_name": "device_processor", "config": {"device": "cuda"}},
                     {"registry_name": "normalizer_processor", "config": {
                         "features": {**inputs, **output}, "norm_map": norm_map, "eps": 1e-8}, "state_file": "pre.safetensors"}]}
    post = {"steps": [{"registry_name": "unnormalizer_processor", "config": {
        "features": output, "norm_map": norm_map, "eps": 1e-8}, "state_file": "post.safetensors"},
                      {"registry_name": "device_processor", "config": {"device": "cpu"}}]}
    settings = {"backend": "lerobot_act_native", "repo_id": MODEL_REPO, "revision": MODEL_REVISION,
                "dataset_repo_id": DATASET_REPO, "dataset_revision": DATASET_REVISION,
                "device": "cuda", "frame_indices": [0, 30, 60], "episode_id": 0,
                "warmup_runs": 1, "measured_runs": 3}
    return model, info, pre, post, settings


class OfflineInferenceTests(unittest.TestCase):
    def test_saved_type_discriminator_uses_generic_registry_loader(self):
        class NativeACT:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise AssertionError("concrete loader cannot decode the saved type discriminator")
        class GenericConfig:
            calls = []
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                cls.calls.append((args, kwargs))
                return NativeACT()
        result = _parse_act_config("/audited/model", NativeACT, GenericConfig)
        self.assertIsInstance(result, NativeACT)
        self.assertEqual(GenericConfig.calls, [(("/audited/model",), {"local_files_only": True})])
        class WrongConfig:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return object()
        with self.assertRaises(OfflineInferenceError):
            _parse_act_config("/audited/model", NativeACT, WrongConfig)

    def test_native_protocol_retains_full_named_action_tail(self):
        result = validate_native_act_metadata(*fixture())
        self.assertEqual(result["action_dim"], 20)
        self.assertEqual(result["state_dim"], 29)
        self.assertEqual(result["action_names"][-2:], ["left_hand_open", "right_hand_open"])

    def test_rejects_padded_state_reordered_action_or_alias_camera(self):
        for index, mutation in ((0, lambda model: model["input_features"]["observation.state"].update(shape=[31])),
                                (1, lambda info: info["features"]["action"]["names"].reverse()),
                                (0, lambda model: model["input_features"].update({"observation.images.head": model["input_features"].pop(CAMERA)}))):
            values = fixture()
            mutation(values[index])
            with self.subTest(index=index), self.assertRaises(OfflineInferenceError):
                validate_native_act_metadata(*values)

    def test_rejects_substituted_grasp_square_data_and_mutable_model_revision(self):
        for key, value in (("dataset_repo_id", "unitreerobotics/G1_Dex3_GraspSquare_Dataset"), ("revision", "main")):
            values = fixture()
            values[-1][key] = value
            with self.assertRaises(OfflineInferenceError):
                validate_native_act_metadata(*values)

    def test_rejects_replaced_stats_camera_rename_or_nonfinite_epsilon(self):
        for mutation in (lambda pre: pre["steps"][0]["config"].update(rename_map={CAMERA: "head"}),
                         lambda pre: pre["steps"][3]["config"].update(eps=float("nan")),
                         lambda pre: pre["steps"][3].update(state_file="../new_stats.safetensors"),
                         lambda pre: pre["steps"][3]["config"]["norm_map"].update(STATE="MIN_MAX")):
            values = copy.deepcopy(fixture())
            mutation(values[2])
            with self.assertRaises(OfflineInferenceError):
                validate_native_act_metadata(*values)

    def test_rejects_empty_duplicate_out_of_type_frames_and_no_measured_runs(self):
        for key, value in (("frame_indices", []), ("frame_indices", [0, 0]),
                           ("frame_indices", [False]), ("measured_runs", 0)):
            values = fixture()
            values[-1][key] = value
            with self.assertRaises(OfflineInferenceError):
                validate_native_act_metadata(*values)

    def _receipt(self, root):
        file = root / "config.json"
        file.write_text('{"real":true}')
        receipt = {"repo_id": MODEL_REPO, "revision": MODEL_REVISION, "files": [{
            "relative_path": "config.json", "path": str(file), "size_bytes": file.stat().st_size,
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "full_file_verified": True}]}
        (root / "provenance.json").write_text(json.dumps(receipt))
        return file, receipt

    def test_full_provenance_requires_exact_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file, receipt = self._receipt(root)
            report = verify_native_provenance(root, MODEL_REPO, MODEL_REVISION, ["config.json"])
            self.assertEqual(report["config.json"]["size_bytes"], file.stat().st_size)
            file.write_text('{"fake":true}')
            with self.assertRaises(OfflineInferenceError):
                verify_native_provenance(root, MODEL_REPO, MODEL_REVISION, ["config.json"])

    def test_provenance_rejects_traversal_identity_mismatch_or_incomplete_file(self):
        for mutation in (lambda receipt: receipt.update(revision="main"),
                         lambda receipt: receipt["files"][0].update(relative_path="../outside.json"),
                         lambda receipt: receipt["files"][0].update(full_file_verified=False),
                         lambda receipt: receipt["files"][0].update(expected_lfs_sha256="a" * 64)):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, receipt = self._receipt(root)
                mutation(receipt)
                (root / "provenance.json").write_text(json.dumps(receipt))
                with self.assertRaises(OfflineInferenceError):
                    verify_native_provenance(root, MODEL_REPO, MODEL_REVISION, ["config.json"])

    def test_missing_native_runtime_is_recorded_as_failure_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"offline_policy": fixture()[-1]}
            real_import = __import__
            def fail_torch(name, *args, **kwargs):
                if name == "numpy":
                    raise ImportError("probe scientific runtime missing")
                return real_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=fail_torch):
                result = run_native_act_inference(config, root)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["inference_passed"])
            self.assertFalse(result["training_started"])
            self.assertTrue((root / "offline_inference.json").is_file())
            self.assertFalse((root / "native_act_predictions.npz").exists())
            with self.assertRaises(OfflineInferenceError):
                run_native_act_inference(config, root)


if __name__ == "__main__":
    unittest.main()
