"""Diagnostic protocol, paired anchors and true state-only ACT admission."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from g1_vla import act_diagnostics as diagnostic
from g1_vla import grasp_act as native


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_ACT = HAS_NUMPY and importlib.util.find_spec("torch") is not None and importlib.util.find_spec("lerobot") is not None


def fixture(root="."):
    return {"paths": {"root": str(root)},
            "train": {"backend": "grasp_square_act", "enabled": True, "authorization": "conditional", "epochs": None,
                      "seed": 20261014, "batch_size": 2, "learning_rate": 1e-4},
            "grasp_act": {"backend": "lerobot_act_grasp_square", "dataset_manifest_path": "data/manifest.json",
                          "memory_report_path": "memory/report.json", "checkpoint_path": "reference", "device": "cuda",
                          "chunk_size": 16, "dim_model": 32, "n_heads": 4, "dim_feedforward": 64,
                          "n_encoder_layers": 1, "n_decoder_layers": 1, "n_vae_encoder_layers": 1,
                          "latent_dim": 8, "dropout": .1, "weight_decay": 1e-4, "kl_weight": 10,
                          "max_steps": 200, "eval_every_steps": 50, "max_train_wall_s": 1800,
                          "max_gpu_reserved_bytes": 7 * 1024 ** 3, "eval_stride": 30, "eval_batch_size": 8},
            "act_diagnostics": {"backend": diagnostic.BACKEND, "reference_checkpoint_path": "reference",
                                "reference_config_path": "reference/experiment_config.json", "output_dir": "ablation",
                                "visual_modes": list(diagnostic.VISUAL_MODES), "variants": list(diagnostic.VARIANTS),
                                "image_shift_fraction": .5, "seed": 20261014, "max_steps": 200,
                                "eval_every_steps": 50, "eval_stride": 30, "eval_batch_size": 8,
                                "max_train_wall_s": 1800, "max_gpu_reserved_bytes": 7 * 1024 ** 3}}


class DiagnosticAdmissionTests(unittest.TestCase):
    def test_declared_comparators_budget_and_common_seed_are_required(self):
        self.assertEqual(diagnostic.validate_diagnostic_settings(fixture())["max_steps"], 200)
        for key, value in (("backend", "legacy"), ("visual_modes", ["real"]), ("variants", ["rgb"]),
                           ("image_shift_fraction", .1), ("seed", 0), ("max_steps", 201),
                           ("eval_batch_size", True), ("max_gpu_reserved_bytes", 8 * 1024 ** 3),
                           ("max_train_wall_s", float("nan"))):
            config = fixture()
            config["act_diagnostics"][key] = value
            with self.subTest(key=key), self.assertRaises(diagnostic.ACTDiagnosticsError):
                diagnostic.validate_diagnostic_settings(config)

    def test_failures_are_immutable_and_do_not_start_optimizer(self):
        for function, name in ((diagnostic.run_act_diagnostics, "act_diagnostics"),
                               (diagnostic.train_act_ablation, "act_ablation")):
            with tempfile.TemporaryDirectory() as directory:
                config = fixture(directory)
                with patch.object(diagnostic, "load_diagnostic_data", side_effect=diagnostic.ACTDiagnosticsError("train artifact changed")), \
                        patch.object(native, "_runtime") as runtime:
                    result = function(config, directory)
                runtime.assert_not_called()
                self.assertFalse(result["passed"])
                self.assertFalse(result["training_started"])
                self.assertEqual(result["optimizer_steps"], 0)
                self.assertFalse(result["test_payloads_opened"])
                self.assertEqual(json.loads((Path(directory) / "reports" / (name + ".json")).read_text())["error"]["message"], "train artifact changed")
                with self.assertRaises(diagnostic.ACTDiagnosticsError):
                    function(config, directory)

    def test_disabled_ablation_does_not_read_data_or_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            config["train"]["enabled"] = False
            with patch.object(diagnostic, "load_diagnostic_data") as data:
                report = diagnostic.train_act_ablation(config, directory)
            data.assert_not_called()
            self.assertFalse(report["passed"])
            self.assertEqual(report["discarded_probe_optimizer_steps"], 0)
            self.assertFalse((Path(directory) / "ablation").exists())

    def test_qstate_size_alias_cannot_accept_images_or_a_second_state_source(self):
        state = SimpleNamespace(shape=(2, 28))
        batch = {"observation.state": state}
        aliased = diagnostic._qstate_size_alias(None, (batch,))[0]
        self.assertIs(aliased["observation.state"], state)
        self.assertIs(aliased["observation.environment_state"], state)
        self.assertEqual(set(batch), {"observation.state"})
        for key in (native.CAMERA, "observation.environment_state", "observation.images"):
            with self.assertRaises(diagnostic.ACTDiagnosticsError):
                diagnostic._qstate_size_alias(None, ({**batch, key: state},))
        with self.assertRaises(diagnostic.ACTDiagnosticsError):
            diagnostic._qstate_size_alias(None, ({"observation.state": SimpleNamespace(shape=(2, 31))},))

    def test_loss_windows_keep_l1_and_kl_separate(self):
        steps = [{"loss": 11.0, "loss_components": {"l1_loss": 1.0, "kld_loss": 1.0}} for _ in range(20)]
        windows = diagnostic._loss_windows(steps)
        self.assertEqual(windows["last_20"]["mean_total_loss"], 11)
        self.assertEqual(windows["last_20"]["mean_training_l1_loss"], 1)
        self.assertEqual(windows["last_20"]["mean_kld_loss"], 1)

    def test_rgb_replay_rejects_changed_optimization_or_validation(self):
        step = {"step": 1, "loss": 11., "loss_components": {"l1_loss": 1., "kld_loss": 1.}}
        validation = {"step": 1, "metrics": {"policy": {"all_28": {"mae": .2}}}}
        history = {"training_passed": True, "optimizer_steps": 1, "steps": [step], "validation": [validation]}
        actual = {"steps": [copy.deepcopy(step)], "validation": [copy.deepcopy(validation)]}
        self.assertTrue(diagnostic.verify_rgb_replay(actual, history)["passed"])
        for mutation in (lambda value: value["steps"][0].update(loss=12.),
                         lambda value: value["steps"][0]["loss_components"].update(kld_loss=2.),
                         lambda value: value["validation"][0]["metrics"]["policy"]["all_28"].update(mae=.21),
                         lambda value: value.update(validation=[])):
            changed = copy.deepcopy(actual)
            mutation(changed)
            with self.assertRaises(diagnostic.ACTDiagnosticsError):
                diagnostic.verify_rgb_replay(changed, history)


@unittest.skipUnless(HAS_NUMPY, "optional NumPy runtime is unavailable")
class DiagnosticArrayTests(unittest.TestCase):
    def test_interventions_are_fixed_derangements_without_crossing_label_identities(self):
        import numpy as np
        ids = np.array([[180, 0], [180, 30], [180, 60], [181, 0], [181, 30], [182, 0], [182, 30]])
        own = np.arange(len(ids))
        np.testing.assert_array_equal(diagnostic.image_donor_plan(ids, "real", np), own)
        np.testing.assert_array_equal(diagnostic.image_donor_plan(ids, "mean_image", np), np.full(len(ids), -1))
        shifted = diagnostic.image_donor_plan(ids, "within_episode_shift", np)
        self.assertTrue(np.all(shifted != own))
        np.testing.assert_array_equal(ids[shifted, 0], ids[:, 0])
        cross = diagnostic.image_donor_plan(ids, "cross_episode_swap", np)
        self.assertTrue(np.all(ids[cross, 0] != ids[:, 0]))
        np.testing.assert_array_equal(cross, diagnostic.image_donor_plan(ids, "cross_episode_swap", np))
        with self.assertRaises(diagnostic.ACTDiagnosticsError):
            diagnostic.image_donor_plan([[180, 0]], "within_episode_shift", np)
        with self.assertRaises(diagnostic.ACTDiagnosticsError):
            diagnostic.image_donor_plan([[180, 0], [180, 30]], "cross_episode_swap", np)
        with self.assertRaises(diagnostic.ACTDiagnosticsError):
            diagnostic.image_donor_plan([[180, 0], [180, 0]], "real", np)

    def test_metrics_use_identical_full_horizon_denominator_and_prior_hold(self):
        import numpy as np
        target = np.zeros((2, 16, 28), dtype=np.float32)
        prediction = np.ones_like(target)
        prediction[:, :, 14:] = 2
        state = np.zeros((2, 28), dtype=np.float32)
        report = diagnostic.diagnostic_metrics(prediction, target, state, np.zeros(28), np.ones(28), [[180, 0], [181, 0]], np)
        self.assertEqual(report["policy"]["all_28"]["mae"], 1.5)
        self.assertEqual(report["policy"]["mae_by_horizon"], [1.5] * 16)
        self.assertEqual(report["state_hold"]["all_28"]["mae"], 0)
        self.assertIsNone(report["policy_to_state_hold_mae_ratio"])
        self.assertEqual(report["policy"]["valid_scalar_count"], 2 * 16 * 28)
        self.assertEqual(set(report["per_episode"]), {"180", "181"})
        with self.assertRaises(diagnostic.ACTDiagnosticsError):
            diagnostic.diagnostic_metrics(prediction, target, np.zeros((2, 31)), np.zeros(28), np.ones(28), [[180, 0], [181, 0]], np)

    def test_anchor_schedule_exactly_replays_native_per_step_rng_sampling(self):
        import numpy as np
        class Dataset:
            samples = [(20, index) for index in range(90)] + [(21, index) for index in range(40)]
            def __len__(self):
                return len(self.samples)
        data, settings = Dataset(), fixture()["act_diagnostics"]
        schedule, ids, coverage = diagnostic.anchor_schedule(data, settings, 2, settings["seed"], np)
        rng = np.random.default_rng(settings["seed"])
        original = np.asarray([rng.integers(0, len(data), size=2) for _ in range(settings["max_steps"])])
        np.testing.assert_array_equal(schedule, original)
        self.assertEqual(coverage["sample_draw_count"], 400)
        self.assertEqual(coverage["unique_train_anchor_count"], len(set(original.ravel())))
        self.assertEqual(coverage["sample_identity_shape"], [200, 2, 2])
        self.assertEqual(coverage["unique_train_anchor_count"], sum(coverage["unique_anchors_by_episode"].values()))
        self.assertTrue(np.all(ids[:, :, 0] < 180))

    def test_loader_verifies_only_train_validation_payloads(self):
        import numpy as np
        from g1_vla.external_data import SOURCE_JOINT_NAMES
        with tempfile.TemporaryDirectory() as directory:
            root, config = Path(directory), fixture(directory)
            reference, data = root / "reference", root / "data"
            reference.mkdir()
            data.mkdir()
            (reference / "experiment_config.json").write_text(json.dumps(config))
            (data / "manifest.json").write_text("{}")
            original = native._digest(config)
            statistics = {"training_episode_ids": native.SPLITS["train"], "validation_or_test_used": False,
                          "statistics_include_held_out": False, "std_minimum": .001, "features": {}}
            for key, shape in (("observation.state", (28,)), ("action", (28,)), (native.CAMERA, (3, 1, 1))):
                statistics["features"][key] = {"mean": np.zeros(shape).tolist(), "std": np.ones(shape).tolist(), "unclamped_std": np.ones(shape).tolist()}
            (data / "statistics.json").write_text(json.dumps(statistics))
            binding = {"experiment_config_sha256": original, "module_sha256": native.sha256_file(native.__file__),
                       "manifest_sha256": native.sha256_file(data / "manifest.json"), "statistics_sha256": native.sha256_file(data / "statistics.json")}
            (reference / "provenance.json").write_text(json.dumps({"bindings": binding, "files": {"model.safetensors": {"sha256": "f" * 64}}}))
            manifest = {"episodes": [{"split": split, "episode_id": episode, "arrays": {"relative_path": split + "/arrays", "sha256": "a"},
                                      "rgb": {"relative_path": split + "/rgb", "sha256": "b"},
                                      "decoded_receipt": {"relative_path": split + "/receipt", "sha256": "c"}}
                                     for split, episodes in native.SPLITS.items() for episode in episodes],
                        "statistics": {"relative_path": "statistics.json"}}
            verified = []
            def verify(directory, record):
                self.assertFalse(record["relative_path"].startswith("test/"))
                verified.append(record["relative_path"])
                return data / record["relative_path"]
            config["source_native"], config["external_data"] = {}, {}
            # The saved full configuration is intentionally the original one;
            # adding diagnostics settings must not rebind old provenance.
            with patch.object(native, "verify_grasp_checkpoint", return_value={}), \
                    patch.object(native, "validate_grasp_manifest"), \
                    patch("g1_vla.grasp_data.load_grasp_manifest", return_value=manifest) as loader, \
                    patch("g1_vla.grasp_data._verify_artifact", side_effect=verify), \
                    patch("g1_vla.grasp_data.train_only_statistics", return_value=statistics) as recompute, \
                    patch("g1_vla.grasp_data.GraspSquareDataset", side_effect=lambda directory, split, **kwargs: split) as dataset:
                _, _, result, datasets = diagnostic.load_diagnostic_data(config)
            self.assertEqual(set(datasets), {"train", "validation"})
            self.assertFalse(result["test_payloads_opened"])
            self.assertFalse(loader.call_args.kwargs["verify_files"])
            self.assertTrue(all(item["split"] != "test" for item in recompute.call_args.args[1]))
            self.assertEqual(len(verified), 9 * 3 + 1)
            self.assertEqual(dataset.call_count, 2)
            for key in ("seed", "batch_size", "learning_rate"):
                changed = copy.deepcopy(config)
                changed["train"][key] += 1
                if key == "seed":
                    changed["act_diagnostics"]["seed"] = changed["train"][key]
                with patch.object(native, "verify_grasp_checkpoint", return_value={}), self.assertRaisesRegex(diagnostic.ACTDiagnosticsError, "differs from the original RGB"):
                    diagnostic.load_diagnostic_data(changed)


@unittest.skipUnless(HAS_ACT, "optional torch and LeRobot ACT runtime is unavailable")
class TrueStateOnlyTests(unittest.TestCase):
    def test_real_state_only_network_uses_no_camera_or_environment_token_and_shares_initial_weights(self):
        import numpy as np
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.configs.types import PolicyFeature, FeatureType
        config, settings = fixture(), fixture()["grasp_act"]
        kwargs = {key: settings[key] for key in ("chunk_size", "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers",
                                               "n_decoder_layers", "n_vae_encoder_layers", "latent_dim", "dropout", "kl_weight")}
        blueprint_policy = ACTPolicy(ACTConfig(**kwargs, n_action_steps=16, pretrained_backbone_weights=None,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (28,)), native.CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 240, 320))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (28,))}))
        blueprint = {key: value.detach().cpu().clone() for key, value in blueprint_policy.state_dict().items()}
        rng_state = {"cpu": torch.get_rng_state().clone(), "cuda": []}
        state_only, metadata = diagnostic._variant_policy(config, settings, "state_only", blueprint, np, torch, torch.device("cpu"), rng_state)
        self.assertEqual(set(state_only.config.input_features), {"observation.state"})
        self.assertIsNone(state_only.config.env_state_feature)
        self.assertFalse(state_only.config.image_features)
        self.assertTrue(metadata["same_common_initial_tensor_values_verified"])
        self.assertFalse(metadata["alias_adds_projection_token_or_privileged_information"])
        self.assertFalse(hasattr(state_only.model, "backbone"))
        self.assertFalse(hasattr(state_only.model, "encoder_env_state_input_proj"))
        batch = {"observation.state": torch.zeros((2, 28)), "action": torch.zeros((2, 16, 28)), "action_is_pad": torch.zeros((2, 16), dtype=torch.bool)}
        state_only.train()
        loss, components = state_only(batch)
        loss.backward()
        self.assertTrue(diagnostic._gradient_report(state_only, torch)["finite"])
        state_only.eval()
        prediction = state_only.predict_action_chunk({"observation.state": batch["observation.state"]})
        self.assertEqual(tuple(prediction.shape), (2, 16, 28))
        self.assertTrue(torch.isfinite(prediction).all())
        no_vae, no_vae_metadata = diagnostic._variant_policy(config, settings, "rgb_no_vae", blueprint, np, torch, torch.device("cpu"), rng_state)
        self.assertFalse(no_vae.config.use_vae)
        self.assertFalse(hasattr(no_vae.model, "vae_encoder"))
        self.assertTrue(no_vae_metadata["same_common_initial_tensor_values_verified"])


if __name__ == "__main__":
    unittest.main()
