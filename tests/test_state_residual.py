"""State residual admission must preserve the exact raw-state hold baseline."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1_vla import act_diagnostics as diagnostic
from g1_vla import grasp_act as native
from g1_vla import state_residual as residual


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_TORCH = HAS_NUMPY and importlib.util.find_spec("torch") is not None
HAS_ACT = HAS_TORCH and importlib.util.find_spec("lerobot") is not None and importlib.util.find_spec("torchvision") is not None


def fixture(root="."):
    path = Path(__file__).resolve().parents[1] / "configs" / "experiment.json"
    config = json.loads(path.read_text())
    config["paths"]["root"] = str(root)
    return config


class ResidualAdmissionTests(unittest.TestCase):
    def test_protocol_has_two_no_vae_controls_and_the_previous_complete_budget(self):
        config = fixture()
        settings = residual.validate_state_residual_settings(config)
        self.assertEqual(tuple(settings["variants"]), residual.VARIANTS)
        self.assertIs(settings["use_vae"], False)
        self.assertIs(settings["zero_head_init"], True)
        self.assertIs(settings["include_step_zero_in_selection"], True)
        for name in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_train_wall_s", "max_gpu_reserved_bytes"):
            self.assertEqual(settings[name], config["visual_pretraining"][name], name)
        self.assertEqual(settings["target"], residual.TARGET)
        self.assertEqual(settings["decode"], residual.DECODE)

    def test_changed_semantics_controls_and_training_budget_are_rejected(self):
        mutations = (("backend", "absolute_action"), ("variants", list(reversed(residual.VARIANTS))),
                     ("variants", [residual.VARIANTS[0]]), ("use_vae", True),
                     ("zero_head_init", False), ("include_step_zero_in_selection", False),
                     ("target", "residual_with_mean"), ("decode", "inverse_normalized_q"),
                     ("seed", 0), ("max_steps", 1999), ("max_steps", 2000.0),
                     ("eval_every_steps", 100), ("eval_batch_size", True),
                     ("max_gpu_reserved_bytes", 8 * 1024 ** 3), ("max_train_wall_s", float("nan")),
                     ("reference_visual_report_sha256", "f" * 63),
                     ("reference_visual_report_sha256", "A" * 64),
                     ("output_dir", "../escape"), ("admission_report_path", ""))
        for key, value in mutations:
            config = fixture()
            config["state_residual"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises((residual.StateResidualError, native.GraspACTError)):
                residual.validate_state_residual_settings(config)

    def test_disabled_training_never_opens_payloads_or_constructs_optimizer_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            config["train"]["enabled"] = False
            with patch.object(diagnostic, "load_diagnostic_data") as loader, \
                    patch.object(native, "_runtime") as runtime:
                report = residual.train_state_residual(config, directory)
            loader.assert_not_called()
            runtime.assert_not_called()
            self.assertFalse(report["passed"])
            self.assertFalse(report["training_started"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["discarded_probe_optimizer_steps"], 0)
            self.assertFalse(report["test_payloads_opened"])
            self.assertFalse((Path(directory) / config["state_residual"]["output_dir"]).exists())

    def test_failed_selected_data_admission_is_immutable_and_does_not_start_training(self):
        for function, name in ((residual.run_residual_admission, "state_residual_admission"),
                               (residual.train_state_residual, "state_residual_training")):
            with self.subTest(stage=name), tempfile.TemporaryDirectory() as directory:
                config = fixture(directory)
                with patch.object(diagnostic, "load_diagnostic_data", side_effect=diagnostic.ACTDiagnosticsError("selected train data changed")), \
                        patch.object(native, "_runtime") as runtime:
                    report = function(config, directory)
                runtime.assert_not_called()
                self.assertFalse(report["passed"])
                self.assertFalse(report["training_started"])
                self.assertEqual(report["optimizer_steps"], 0)
                self.assertFalse(report["test_payloads_opened"])
                self.assertFalse(report["test_used_for_selection"])
                self.assertIn("selected train data changed", report["error"]["message"])
                saved = Path(directory) / "reports" / (name + ".json")
                self.assertEqual(json.loads(saved.read_text())["error"], report["error"])
                with self.assertRaises(residual.StateResidualError):
                    function(config, directory)

    def test_prior_report_hash_is_checked_before_trusting_json(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            settings = config["state_residual"]
            path = Path(directory) / settings["reference_visual_report_path"]
            path.parent.mkdir(parents=True)
            path.write_bytes(b'{"passed": true, "untrusted": "changed bytes"}')
            with patch.object(native, "_read") as reader, self.assertRaises(residual.StateResidualError):
                residual._verify_reference(config, settings, {})
            reader.assert_not_called()


class ResidualSelectionTests(unittest.TestCase):
    @staticmethod
    def point(step, mae):
        return {"step": step, "metrics": {"policy": {"all_28": {"mae": mae}}}}

    def test_hold_remains_selected_when_every_trained_checkpoint_regresses(self):
        report = residual.select_checkpoint([self.point(0, .041), self.point(50, .05),
                                             self.point(100, .047), self.point(150, .06)])
        self.assertEqual(report["selected_step"], 0)
        self.assertEqual(report["selected_mae"], .041)
        self.assertEqual(report["best_trained_step"], 100)
        self.assertEqual(report["best_trained_mae"], .047)
        self.assertEqual(report["last_step"], 150)
        self.assertEqual(report["last_mae"], .06)
        self.assertFalse(report["test_used_for_selection"])

    def test_strict_improvement_selects_trained_checkpoint_and_earliest_tie(self):
        report = residual.select_checkpoint([self.point(0, .041), self.point(50, .041),
                                             self.point(100, .03), self.point(150, .03)])
        self.assertEqual(report["selected_step"], 100)
        self.assertEqual(report["best_trained_step"], 100)
        self.assertEqual(report["last_step"], 150)
        held = residual.select_checkpoint([self.point(0, .041), self.point(50, .041)])
        self.assertEqual(held["selected_step"], 0)
        self.assertEqual(held["best_trained_step"], 50)

    def test_missing_origin_nonincreasing_steps_and_invalid_metrics_are_rejected(self):
        invalid = ([], [self.point(50, .03)], [self.point(0, .04), self.point(0, .03)],
                   [self.point(0, .04), self.point(100, .03), self.point(50, .02)],
                   [self.point(0, .04), self.point(True, .02)],
                   [self.point(0, float("nan"))], [self.point(0, float("inf"))],
                   [self.point(0, -.01)], [self.point(0, True)],
                   [{"step": 0, "metrics": {}}])
        for points in invalid:
            with self.subTest(points=points), self.assertRaises(residual.StateResidualError):
                residual.select_checkpoint(points)


@unittest.skipUnless(HAS_NUMPY, "optional NumPy runtime is unavailable")
class ResidualArrayTests(unittest.TestCase):
    def test_zero_prediction_is_bitwise_raw_hold_for_signed_zero_and_arbitrary_state(self):
        import numpy as np
        rng = np.random.default_rng(1357)
        state = rng.uniform(-3.0, 3.0, (4, 28)).astype(np.float32)
        state[0, :4] = np.asarray([0.0, -0.0, np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny], dtype=np.float32)
        standard_deviation = np.exp(rng.uniform(-5, 2, 28)).astype(np.float32)
        prediction = np.zeros((4, 16, 28), dtype=np.float32)
        prediction[1::2] = np.float32(-0.0)
        decoded = residual.decode_residual(prediction, state, standard_deviation, np)
        expected = np.broadcast_to(state[:, None, :], prediction.shape)
        self.assertEqual(decoded.dtype, np.float32)
        np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))
        # No mutation is allowed: the same raw state is also the hold comparator.
        np.testing.assert_array_equal(state[0, :4].view(np.uint32), np.asarray([0., -0., np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny], dtype=np.float32).view(np.uint32))

    def test_residuals_use_one_raw_anchor_for_every_future_label_without_action_mean(self):
        import numpy as np
        state = np.linspace(-1.5, 2.0, 56, dtype=np.float32).reshape(2, 28)
        standard_deviation = np.linspace(.001, 1.5, 28, dtype=np.float32)
        horizon_offset = np.arange(16, dtype=np.float32)[None, :, None] * np.float32(.0125)
        labels = (state[:, None, :] + horizon_offset + np.float32(.25)).astype(np.float32)
        original_state, original_labels, original_std = state.copy(), labels.copy(), standard_deviation.copy()
        expected = (labels - state[:, None, :]) / standard_deviation
        encoded = residual.residual_targets(labels, state, standard_deviation, np)
        np.testing.assert_array_equal(encoded, expected)
        self.assertEqual(encoded.dtype, np.float32)
        np.testing.assert_allclose(residual.decode_residual(encoded, state, standard_deviation, np), labels,
                                   rtol=0, atol=3e-7)
        np.testing.assert_array_equal(state, original_state)
        np.testing.assert_array_equal(labels, original_labels)
        np.testing.assert_array_equal(standard_deviation, original_std)

    def test_nonzero_decode_does_not_reconstruct_anchor_through_normalized_state(self):
        import numpy as np
        # q contains values for which (q - mean) / std followed by inversion
        # need not be bitwise exact. Decode must add to the provided raw q.
        state = np.full((2, 28), np.float32(.10000001), dtype=np.float32)
        standard_deviation = np.full(28, np.float32(.013), dtype=np.float32)
        prediction = np.full((2, 16, 28), np.float32(.125), dtype=np.float32)
        prediction[:, 3, 2] = 0
        decoded = residual.decode_residual(prediction, state, standard_deviation, np)
        expected = state[:, None, :] + prediction * standard_deviation
        np.testing.assert_array_equal(decoded, expected)
        np.testing.assert_array_equal(decoded[:, 3, 2].view(np.uint32), state[:, 2].view(np.uint32))

    def test_equal_raw_action_and_anchor_produce_zero_targets_without_mean_bias(self):
        import numpy as np
        state = np.arange(56, dtype=np.float32).reshape(2, 28) * np.float32(.1)
        labels = np.broadcast_to(state[:, None, :], (2, 16, 28)).copy()
        encoded = residual.residual_targets(labels, state, np.linspace(.001, 2, 28, dtype=np.float32), np)
        np.testing.assert_array_equal(encoded, np.zeros_like(encoded))

    def test_malformed_dtype_dimensions_and_nonfinite_arrays_are_rejected(self):
        import numpy as np
        good = (np.zeros((2, 16, 28), dtype=np.float32), np.zeros((2, 28), dtype=np.float32), np.ones(28, dtype=np.float32))
        bad = [
            (good[0].astype(np.float64), good[1], good[2]),
            (good[0], good[1].astype(np.float64), good[2]),
            (good[0], good[1], good[2].astype(np.float64)),
            (good[0].tolist(), good[1], good[2]),
            (good[0][:, :, :27], good[1], good[2]),
            (good[0], good[1][:1], good[2]),
            (good[0], good[1][:, :27], good[2]),
            (good[0], good[1], good[2][None, :]),
            (good[0][:, 0, :], good[1], good[2]),
        ]
        for column in range(3):
            for value in (float("nan"), float("inf"), float("-inf")):
                changed = [item.copy() for item in good]
                changed[column].flat[0] = value
                bad.append(tuple(changed))
        for value in (0., -.1):
            changed = good[2].copy()
            changed[0] = value
            bad.append((good[0], good[1], changed))
        for arrays in bad:
            for function in (residual.residual_targets, residual.decode_residual):
                with self.subTest(function=function.__name__, shapes=[getattr(item, "shape", None) for item in arrays]), self.assertRaises(residual.StateResidualError):
                    function(*arrays, np)

    def test_finite_inputs_that_overflow_residual_arithmetic_fail_closed(self):
        import numpy as np
        maximum = np.finfo(np.float32).max
        state = np.full((1, 28), -maximum, dtype=np.float32)
        labels = np.full((1, 16, 28), maximum, dtype=np.float32)
        standard_deviation = np.ones(28, dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(residual.StateResidualError):
            residual.residual_targets(labels, state, standard_deviation, np)
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(residual.StateResidualError):
            residual.decode_residual(labels, -state, np.full(28, maximum, dtype=np.float32), np)

    def test_motion_strata_partition_the_identical_anchor_and_label_denominators(self):
        import numpy as np
        _, statistics, _, _ = ResidualBatchBoundaryTests._data()
        states = np.zeros((4, 28), dtype=np.float32)
        states[:, 0] = np.cumsum(np.asarray([0., .01, .05, .2], dtype=np.float32)) / np.float32(30.)
        target = np.ones((4, 16, 28), dtype=np.float32)
        predictions = np.arange(4, dtype=np.float32)[:, None, None] * np.ones_like(target)
        arrays = {"sample_episode_frame": np.asarray([[180, index] for index in range(4)], dtype=np.int64),
                  "predictions_source_actions": predictions, "target_source_recorded_actions": target,
                  "observation_state": states}
        dataset = type("Dataset", (), {"_arrays": {180: (states, np.zeros_like(states))}})()
        report = residual._motion_strata(dataset, arrays, statistics,
            {"native_diagnostics": {"stationary_speed_rad_s": .02, "moving_speed_rad_s": .1}}, np)
        self.assertEqual(report["partition_anchor_count"], 4)
        self.assertEqual({name: value["anchor_count"] for name, value in report["strata"].items()},
                         {"stationary": 1, "moving": 1, "intermediate": 1, "unknown_first_row": 1})
        self.assertFalse(report["used_as_model_input"])
        self.assertTrue(np.isnan(arrays["anchor_max_joint_speed_rad_s"][0]))
        np.testing.assert_array_equal(arrays["anchor_speed_known"], [False, True, True, True])
        scalar_count = sum(value["metrics"]["policy"]["valid_scalar_count"] for value in report["strata"].values())
        self.assertEqual(scalar_count, target.size)
        weighted_error = sum(value["anchor_count"] * value["metrics"]["policy"]["all_28"]["mae"] for value in report["strata"].values()) / 4
        self.assertEqual(weighted_error, float(np.abs(predictions - target).mean()))


@unittest.skipUnless(HAS_TORCH, "optional Torch runtime is unavailable")
class ResidualHeadTests(unittest.TestCase):
    def test_zero_head_changes_no_other_parameter_or_buffer(self):
        import torch
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.encoder = torch.nn.Linear(5, 7)
                self.model.action_head = torch.nn.Linear(7, 28)
                self.model.register_buffer("unrelated", torch.arange(4, dtype=torch.float32))
        policy = Policy()
        before = {name: value.clone() for name, value in policy.state_dict().items()}
        receipt = residual.initialize_residual_head(policy, torch)
        self.assertTrue(receipt)
        for name, value in policy.state_dict().items():
            if name.startswith("model.action_head."):
                self.assertEqual(int(torch.count_nonzero(value)), 0)
            else:
                self.assertTrue(torch.equal(value, before[name]), name)
        output = policy.model.action_head(policy.model.encoder(torch.randn(3, 5)))
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_first_backward_reaches_head_then_second_backward_can_reach_encoder(self):
        import torch
        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.encoder = torch.nn.Linear(5, 7)
                self.model.action_head = torch.nn.Linear(7, 28)
            def forward(self, value):
                return self.model.action_head(self.model.encoder(value).tanh())
        torch.manual_seed(707)
        policy = Policy()
        residual.initialize_residual_head(policy, torch)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3, weight_decay=1e-4)
        inputs = torch.randn(4, 5)
        targets = torch.full((4, 28), .25)
        (policy(inputs) - targets).abs().mean().backward()
        self.assertGreater(float(policy.model.action_head.bias.grad.abs().sum()), 0)
        for parameter in policy.model.encoder.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(float(parameter.grad.abs().sum()), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        (policy(inputs) - targets).abs().mean().backward()
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in policy.model.encoder.parameters()), 0)


@unittest.skipUnless(HAS_TORCH, "optional Torch runtime is unavailable")
class ResidualBatchBoundaryTests(unittest.TestCase):
    @staticmethod
    def _data():
        import numpy as np
        state = np.linspace(-1., 1., 56, dtype=np.float32).reshape(2, 28)
        state[0, 0] = np.float32(-0.)
        samples = [{"observation.state": q.copy(), native.CAMERA: np.zeros((3, 4, 4), dtype=np.float32),
                    "action": np.repeat(q[None, :], 16, axis=0) + np.float32(.025),
                    "action_is_pad": np.zeros(16, dtype=np.bool_)} for q in state]
        class Dataset:
            split = "validation"
            def __init__(self):
                self.samples = [(180, 0), (181, 30)]
            def __getitem__(self, index):
                return samples[index]
            def get_eval_indices(self, stride):
                return [0, 1]
        statistics = {"training_episode_ids": native.SPLITS["train"], "validation_or_test_used": False,
                      "statistics_include_held_out": False, "std_minimum": .001, "features": {
                          "observation.state": {"mean": [13.] * 28, "std": [.3] * 28},
                          "action": {"mean": [123.] * 28, "std": [.25] * 28},
                          native.CAMERA: {"mean": [[[.4]], [[.5]], [[.6]]], "std": [[[.2]], [[.25]], [[.3]]]}}}
        return Dataset(), statistics, state, samples

    def test_residual_batch_replaces_labels_from_raw_samples_and_rejects_padding(self):
        import numpy as np
        import torch
        dataset, statistics, state, samples = self._data()
        stats = native._stats(statistics, np)
        with patch.object(native, "_batch", wraps=native._batch) as loader:
            batch, _ = residual._training_batch(dataset, [0, 1], stats, "state_only_residual", np, torch, torch.device("cpu"))
        self.assertIs(loader.call_args.kwargs["include_actions"], False)
        self.assertEqual(set(batch), {"observation.state", "action", "action_is_pad"})
        expected = residual.residual_targets(np.stack([sample["action"] for sample in samples]), state, stats["action"][1], np)
        np.testing.assert_array_equal(batch["action"].numpy(), expected)
        np.testing.assert_array_equal(batch["observation.state"].numpy(), (state - np.float32(13.)) / np.float32(.3))
        self.assertFalse(bool(batch["action_is_pad"].any()))
        samples[0]["action_is_pad"][0] = True
        with self.assertRaises(residual.StateResidualError):
            residual._training_batch(dataset, [0, 1], stats, "state_only_residual", np, torch, torch.device("cpu"))

    def test_actual_inference_boundary_uses_no_labels_and_decodes_direct_raw_anchor(self):
        import numpy as np
        import torch
        dataset, statistics, state, _ = self._data()
        settings = {"eval_stride": 30, "eval_batch_size": 8, "chunk_size": 16}
        for variant in residual.VARIANTS:
            class SpyPolicy:
                def __init__(self):
                    self.calls = []
                def eval(self):
                    return self
                def predict_action_chunk(self, batch):
                    self.calls.append(batch)
                    self_keys = {"observation.state"} if variant == "state_only_residual" else {"observation.state", native.CAMERA}
                    if set(batch) != self_keys:
                        raise AssertionError("model received an undeclared input or future label")
                    return torch.zeros((batch["observation.state"].shape[0], 16, 28), dtype=torch.float32)
            policy = SpyPolicy()
            with self.subTest(variant=variant), patch.object(native, "_check_gpu"):
                metrics, arrays = residual._evaluate(policy, dataset, statistics, settings, np, torch, torch.device("cpu"),
                                                      variant=variant, require_zero=True)
            self.assertTrue(policy.calls)
            self.assertEqual(metrics["policy"], metrics["state_hold"])
            expected = np.repeat(state[:, None, :], 16, axis=1)
            np.testing.assert_array_equal(arrays["predictions_source_actions"].view(np.uint32), expected.view(np.uint32))
            self.assertFalse(metrics["raw_anchor_reconstructed_from_normalized_state"])


@unittest.skipUnless(HAS_ACT, "optional TorchVision and LeRobot ACT runtime is unavailable")
class ActualResidualModelTests(unittest.TestCase):
    @staticmethod
    def _blueprint():
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        config = fixture()
        config["grasp_act"].update(device="cpu", dim_model=32, n_heads=4, dim_feedforward=64,
                                   n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
                                   latent_dim=8, dropout=0.)
        settings = {**config["grasp_act"], **config["state_residual"], "device": "cpu"}
        keys = ("chunk_size", "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers",
                "n_decoder_layers", "n_vae_encoder_layers", "latent_dim", "dropout", "kl_weight")
        torch.manual_seed(config["train"]["seed"])
        policy = ACTPolicy(ACTConfig(**{key: settings[key] for key in keys}, n_action_steps=16,
            pretrained_backbone_weights=None, use_vae=True,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (28,)),
                            native.CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 240, 320))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (28,))}))
        blueprint = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
        rng = {"cpu": torch.get_rng_state().clone(), "cuda": []}
        return config, settings, blueprint, rng

    def test_both_real_models_have_no_vae_zero_prior_and_common_nonhead_initialization(self):
        import numpy as np
        import torch
        config, settings, blueprint, rng = self._blueprint()
        heads = {"model.action_head.weight", "model.action_head.bias"}
        initial_states = {}
        for variant in residual.VARIANTS:
            with self.subTest(variant=variant):
                policy, metadata = residual._variant_policy(config, settings, variant, blueprint, np, torch,
                    torch.device("cpu"), rng)
                self.assertFalse(policy.config.use_vae)
                self.assertFalse(any("vae_" in name for name in policy.state_dict()))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng["cpu"]))
                self.assertTrue(metadata["residual_head"]["zero_final_action_head_verified"])
                self.assertIsNone(policy.config.env_state_feature)
                self.assertEqual(set(policy.config.input_features), {"observation.state"} if variant == "state_only_residual" else {"observation.state", native.CAMERA})
                if variant == "state_only_residual":
                    for name in ("backbone", "encoder_img_feat_input_proj", "encoder_env_state_input_proj", "encoder_cam_feat_pos_embed"):
                        self.assertFalse(hasattr(policy.model, name), name)
                state = policy.state_dict()
                for name, tensor in state.items():
                    self.assertTrue(torch.equal(tensor.cpu(), torch.zeros_like(tensor.cpu()) if name in heads else blueprint[name]), name)
                batch = {"observation.state": torch.randn(1, 28)}
                if variant != "state_only_residual":
                    batch[native.CAMERA] = torch.randn(1, 3, 240, 320)
                policy.eval()
                with torch.inference_mode():
                    prediction = policy.predict_action_chunk(batch)
                self.assertEqual(tuple(prediction.shape), (1, 16, 28))
                self.assertEqual(int(torch.count_nonzero(prediction)), 0)
                initial_states[variant] = {name: value.detach().cpu().clone() for name, value in state.items()}
                del policy
        common = set(initial_states[residual.VARIANTS[0]]) & set(initial_states[residual.VARIANTS[1]])
        for name in common:
            self.assertTrue(torch.equal(initial_states[residual.VARIANTS[0]][name], initial_states[residual.VARIANTS[1]][name]), name)


if __name__ == "__main__":
    unittest.main()
