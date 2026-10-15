"""Controlled visual initialization must preserve the native no-VAE protocol."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from g1_vla import act_diagnostics as diagnostic
from g1_vla import grasp_act as native
from g1_vla import vision_pretraining as vision


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_ACT = (HAS_NUMPY and importlib.util.find_spec("torch") is not None
           and importlib.util.find_spec("lerobot") is not None
           and importlib.util.find_spec("torchvision") is not None)


def fixture(root="."):
    config_path = Path(__file__).resolve().parents[1] / "configs" / "experiment.json"
    config = json.loads(config_path.read_text())
    config["paths"]["root"] = str(root)
    # Fetch may precede the full pin. Unit tests of training admission always
    # provide a full identity; they never download an actual public checkpoint.
    if config["visual_pretraining"].get("weights_sha256") is None:
        config["visual_pretraining"]["weights_sha256"] = "f37072fd" + "a" * 56
    return config


def replay_fixture():
    step = {"step": 1, "loss": .4, "loss_components": {"l1_loss": .4}}
    validation = {"step": 1, "metrics": {"policy": {"all_28": {"mae": .2}}}}
    reference = {"status": "completed_matched_budget_training", "optimizer_steps": 1,
                 "steps": [step], "validation": [validation], "initialization": {"use_vae": False}}
    return {"steps": [copy.deepcopy(step)], "validation": [copy.deepcopy(validation)]}, reference


class VisualPretrainingAdmissionTests(unittest.TestCase):
    def test_all_four_controls_share_the_fixed_no_vae_budget_and_preprocessing(self):
        config = fixture()
        settings = vision.validate_visual_pretraining_settings(config)
        self.assertEqual(tuple(settings["variants"]), ("rgb_scratch", "rgb_pretrained", "rgb_mean", "state_only"))
        self.assertEqual(settings["max_steps"], config["act_diagnostics"]["max_steps"])
        self.assertEqual(settings["seed"], config["train"]["seed"])
        self.assertIs(settings["use_vae"], False)
        self.assertEqual(settings["image_preprocessing"], "unchanged_train_rgb_statistics_v1")
        self.assertEqual(settings["eval_stride"], config["act_diagnostics"]["eval_stride"])
        self.assertEqual(settings["eval_batch_size"], config["act_diagnostics"]["eval_batch_size"])

    def test_changed_controls_prior_mechanism_or_partial_budgets_are_rejected(self):
        mutations = (("backend", "legacy"), ("variants", ["rgb_scratch", "rgb_pretrained"]),
                     ("variants", list(reversed(vision.VARIANTS))), ("use_vae", True),
                     ("image_preprocessing", "imagenet_center_crop"), ("seed", 0),
                     ("max_steps", 1999), ("eval_every_steps", 250),
                     ("eval_batch_size", True), ("max_gpu_reserved_bytes", 8 * 1024 ** 3),
                     ("max_train_wall_s", float("nan")))
        for key, value in mutations:
            config = fixture()
            config["visual_pretraining"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(vision.VisionPretrainingError):
                vision.validate_visual_pretraining_settings(config)

    def test_training_requires_full_weight_identity_and_fetch_only_allows_the_official_pin(self):
        config = fixture()
        config["visual_pretraining"]["weights_sha256"] = None
        with self.assertRaises(vision.VisionPretrainingError):
            vision.validate_visual_pretraining_settings(config)
        vision.validate_visual_pretraining_settings(config, allow_unpinned_weights=True)
        for key, value in (("weights_url", "http://download.pytorch.org/models/resnet18-f37072fd.pth"),
                           ("weights_url", "https://example.com/resnet18-f37072fd.pth"),
                           ("weights_url", "https://download.pytorch.org/models/resnet18-latest.pth"),
                           ("weights_sha256_prefix", "00000000"),
                           ("weights_sha256", "f37072fd"),
                           ("weights_sha256", "F37072FD" + "A" * 56),
                           ("weights_max_bytes", True)):
            changed = fixture()
            changed["visual_pretraining"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(vision.VisionPretrainingError):
                vision.validate_visual_pretraining_settings(changed, allow_unpinned_weights=True)

    def test_scratch_replay_rejects_changed_losses_labels_and_validation_plan(self):
        actual, reference = replay_fixture()
        self.assertTrue(vision.verify_scratch_replay(actual, reference)["passed"])
        mutations = (lambda value: value["steps"][0].update(step=2),
                     lambda value: value["steps"][0].update(loss=.41),
                     lambda value: value["steps"][0]["loss_components"].update(l1_loss=.41),
                     lambda value: value["steps"][0]["loss_components"].update(kld_loss=0.),
                     lambda value: value["validation"][0]["metrics"]["policy"]["all_28"].update(mae=.21),
                     lambda value: value.update(validation=[]),
                     lambda value: value.update(steps=[]))
        for mutation in mutations:
            changed = copy.deepcopy(actual)
            mutation(changed)
            with self.subTest(actual=changed), self.assertRaises(vision.VisionPretrainingError):
                vision.verify_scratch_replay(changed, reference)

    def test_disabled_training_never_reads_data_weights_or_constructs_a_model(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            config["train"]["enabled"] = False
            with patch.object(diagnostic, "load_diagnostic_data") as loader, \
                    patch.object(native, "_runtime") as runtime, \
                    patch.object(vision, "_load_verified_weights") as weights:
                report = vision.train_visual_pretraining(config, directory)
            loader.assert_not_called()
            runtime.assert_not_called()
            weights.assert_not_called()
            self.assertFalse(report["passed"])
            self.assertFalse(report["training_started"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["discarded_probe_optimizer_steps"], 0)
            self.assertFalse((Path(directory) / config["visual_pretraining"]["output_dir"]).exists())

    def test_selected_data_integrity_failure_is_immutable_and_never_starts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            with patch.object(diagnostic, "load_diagnostic_data", side_effect=diagnostic.ACTDiagnosticsError("selected train artifact changed")), \
                    patch.object(native, "_runtime") as runtime:
                report = vision.train_visual_pretraining(config, directory)
            runtime.assert_not_called()
            self.assertFalse(report["passed"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertFalse(report["test_payloads_opened"])
            self.assertFalse(report["test_used_for_selection"])
            self.assertIn("selected train artifact changed", report["error"]["message"])
            saved = Path(directory) / "reports" / "visual_pretraining.json"
            self.assertEqual(json.loads(saved.read_text())["error"], report["error"])
            with self.assertRaises(vision.VisionPretrainingError):
                vision.train_visual_pretraining(config, directory)

    def test_reference_report_hash_is_verified_before_its_json_is_trusted(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            settings = config["visual_pretraining"]
            reference = Path(directory) / settings["reference_ablation_report_path"]
            reference.parent.mkdir(parents=True)
            reference.write_bytes(b'{"passed": true, "untrusted": "changed report"}')
            with patch.object(native, "_read") as reader, self.assertRaises(vision.VisionPretrainingError):
                vision._verify_reference(config, settings, {})
            reader.assert_not_called()

    def test_no_vae_admission_rejects_active_or_leftover_posterior_mechanisms(self):
        def policy(use_vae=False, module_names=(), tensor_names=()):
            return SimpleNamespace(config=SimpleNamespace(use_vae=use_vae),
                                   named_modules=lambda: [(name, None) for name in module_names],
                                   state_dict=lambda: {name: None for name in tensor_names})
        self.assertFalse(vision.assert_no_vae(policy())["posterior_module_or_tensor_present"])
        for malformed in (policy(use_vae=True), policy(module_names=("model.vae_encoder",)),
                          policy(tensor_names=("model.vae_encoder_latent_output_proj.weight",))):
            with self.assertRaises(vision.VisionPretrainingError):
                vision.assert_no_vae(malformed)


class WeightIntegrityTests(unittest.TestCase):
    def test_existing_cache_without_receipt_is_preserved_and_never_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            config = fixture(directory)
            config["visual_pretraining"]["weights_sha256"] = None
            destination = Path(directory) / config["visual_pretraining"]["weights_cache_path"]
            destination.parent.mkdir(parents=True)
            original = b"unattributed existing cache must not acquire download provenance"
            destination.write_bytes(original)
            with patch.object(vision.urllib.request, "urlopen") as network, \
                    patch.object(vision, "safe_load_weights") as deserialize, \
                    patch.object(native, "_runtime") as runtime:
                report = vision.fetch_visual_backbone(config, directory)
            network.assert_not_called()
            deserialize.assert_not_called()
            runtime.assert_not_called()
            self.assertFalse(report["passed"])
            self.assertFalse(report["training_started"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertIn("without its observed download receipt", report["error"]["message"])
            self.assertEqual(destination.read_bytes(), original)
            receipt = Path(directory) / config["visual_pretraining"]["weights_receipt_path"]
            self.assertFalse(receipt.exists())

    def test_full_file_hash_and_size_are_checked_before_deserialization(self):
        payload = b"a deterministic fake checkpoint used only for integrity admission"
        digest = hashlib.sha256(payload).hexdigest()
        settings = {"weights_sha256": digest, "weights_sha256_prefix": digest[:8],
                    "weights_max_bytes": len(payload)}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.pth"
            path.write_bytes(payload)
            receipt = vision.validate_weight_file(path, settings)
            self.assertEqual(receipt["sha256"], digest)
            self.assertEqual(receipt["size_bytes"], len(payload))
            for corrupted in (payload[:-1], payload + b"x", bytes([payload[0] ^ 1]) + payload[1:]):
                path.write_bytes(corrupted)
                with self.subTest(size=len(corrupted)), self.assertRaises(vision.VisionPretrainingError):
                    vision.validate_weight_file(path, settings)
            path.write_bytes(payload)
            with self.assertRaises(vision.VisionPretrainingError):
                vision.validate_weight_file(path, {**settings, "weights_sha256": "0" * 64})
            with self.assertRaises(vision.VisionPretrainingError):
                vision.validate_weight_file(path, {**settings, "weights_sha256": None})
            fetched = vision.validate_weight_file(path, {**settings, "weights_sha256": None}, allow_unpinned=True)
            self.assertEqual(fetched["sha256"], digest)
            alias = Path(directory) / "linked.pth"
            alias.symlink_to(path)
            with self.assertRaises(vision.VisionPretrainingError):
                vision.validate_weight_file(alias, settings)

    def test_safe_loader_uses_cpu_weights_only_without_an_unsafe_fallback(self):
        tensors = {"conv1.weight": object()}
        torch = SimpleNamespace(load=Mock(return_value=tensors))
        self.assertIs(vision.safe_load_weights(Path("weights.pth"), torch), tensors)
        torch.load.assert_called_once_with("weights.pth", map_location="cpu", weights_only=True)
        for error in (TypeError("old runtime lacks weights_only"), RuntimeError("unsafe globals")):
            torch = SimpleNamespace(load=Mock(side_effect=error))
            with self.subTest(error=error), self.assertRaises(Exception):
                vision.safe_load_weights(Path("weights.pth"), torch)
            self.assertEqual(torch.load.call_count, 1)
            self.assertIs(torch.load.call_args.kwargs["weights_only"], True)

    def test_fetch_rejects_redirects_corruption_and_declared_or_streaming_size_overruns(self):
        class Response(io.BytesIO):
            def __init__(self, payload, url, headers):
                super().__init__(payload)
                self.url, self.headers = url, headers
            def geturl(self):
                return self.url
        cases = ((b"bad checkpoint", vision.WEIGHTS_URL, {}),
                 (b"", "https://example.com/redirect.pth", {}),
                 (b"", vision.WEIGHTS_URL, {"Content-Length": "33"}),
                 (b"x" * 33, vision.WEIGHTS_URL, {}))
        for payload, url, headers in cases:
            with self.subTest(url=url, headers=headers, size=len(payload)), tempfile.TemporaryDirectory() as directory:
                config = fixture(directory)
                config["visual_pretraining"].update(weights_sha256=None, weights_max_bytes=32)
                with patch.object(vision.urllib.request, "urlopen", return_value=Response(payload, url, headers)) as opener:
                    report = vision.fetch_visual_backbone(config, directory)
                self.assertFalse(report["passed"])
                self.assertFalse(report["training_started"])
                self.assertEqual(report["optimizer_steps"], 0)
                opener.assert_called_once()
                destination = Path(directory) / config["visual_pretraining"]["weights_cache_path"]
                self.assertFalse(destination.exists())
                self.assertEqual(list(destination.parent.glob("*.partial")), [])
                receipt = Path(directory) / config["visual_pretraining"]["weights_receipt_path"]
                self.assertFalse(receipt.exists())


@unittest.skipUnless(HAS_ACT, "optional torch, torchvision and LeRobot ACT runtime is unavailable")
class ControlledVisualModelTests(unittest.TestCase):
    @staticmethod
    def _blueprint():
        import torch
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        config = fixture()
        config["grasp_act"].update(device="cpu", dim_model=32, n_heads=4, dim_feedforward=64,
                                   n_encoder_layers=1, n_decoder_layers=1, n_vae_encoder_layers=1,
                                   latent_dim=8)
        settings = {**config["grasp_act"], **config["visual_pretraining"], "device": "cpu"}
        keys = ("chunk_size", "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers",
                "n_decoder_layers", "n_vae_encoder_layers", "latent_dim", "dropout", "kl_weight")
        torch.manual_seed(config["train"]["seed"])
        model = ACTPolicy(ACTConfig(**{key: settings[key] for key in keys}, n_action_steps=16,
            pretrained_backbone_weights=None, use_vae=True,
            input_features={"observation.state": PolicyFeature(FeatureType.STATE, (28,)),
                            native.CAMERA: PolicyFeature(FeatureType.VISUAL, (3, 240, 320))},
            output_features={"action": PolicyFeature(FeatureType.ACTION, (28,))}))
        blueprint = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        rng = {"cpu": torch.get_rng_state().clone(), "cuda": []}
        return config, settings, blueprint, rng

    def test_resnet_mapping_is_complete_and_leaves_action_and_state_heads_untouched(self):
        import torch
        import torchvision
        _, _, target, _ = self._blueprint()
        source = torchvision.models.resnet18(weights=None).state_dict()
        original = {key: tensor.clone() for key, tensor in target.items()}
        mapped, receipt = vision.map_resnet18_backbone(source, target, torch)
        self.assertEqual(set(mapped), {key for key in target if key.startswith("model.backbone.")})
        self.assertTrue(mapped)
        for key, tensor in mapped.items():
            source_key = key[len("model.backbone."):]
            if source_key.startswith("body."):
                source_key = source_key[len("body."):]
            self.assertTrue(torch.equal(tensor, source[source_key]), key)
            self.assertEqual(tensor.dtype, target[key].dtype)
        self.assertTrue(receipt)
        self.assertTrue(all(torch.equal(value, original[key]) for key, value in target.items()))
        self.assertFalse(any("action_head" in key or "robot_state" in key for key in mapped))

    def test_canonical_102_and_modern_122_tensor_layouts_map_identically_but_partial_counters_fail(self):
        import torch
        import torchvision
        _, _, target, _ = self._blueprint()
        modern = torchvision.models.resnet18(weights=None).state_dict()
        counters = sorted(key for key in modern if key.endswith(".num_batches_tracked"))
        self.assertEqual(len(counters), 20)
        canonical = {key: tensor for key, tensor in modern.items() if key not in counters}
        self.assertEqual(len(canonical), 102)
        modern_mapping, modern_receipt = vision.map_resnet18_backbone(modern, target, torch)
        canonical_mapping, canonical_receipt = vision.map_resnet18_backbone(canonical, target, torch)
        self.assertEqual(len(canonical_mapping), 100)
        self.assertEqual(set(canonical_mapping), set(modern_mapping))
        for key, tensor in canonical_mapping.items():
            self.assertTrue(torch.equal(tensor, modern_mapping[key]), key)
        self.assertEqual(canonical_receipt["source_tensor_count"], 102)
        self.assertEqual(canonical_receipt["batch_counter_keys_not_stored"], counters)
        self.assertEqual(canonical_receipt["ignored_batchnorm_tracking_keys"], [])
        self.assertEqual(modern_receipt["source_tensor_count"], 122)
        self.assertEqual(modern_receipt["batch_counter_keys_not_stored"], [])
        self.assertEqual(modern_receipt["ignored_batchnorm_tracking_keys"], counters)
        # Both historical layouts are complete. Accepting arbitrary omissions
        # would hide damaged checkpoints or silently broaden the architecture.
        partial_layouts = ({key: tensor for key, tensor in modern.items() if key != counters[0]},
                           {**canonical, counters[0]: modern[counters[0]]})
        for partial in partial_layouts:
            with self.subTest(source_tensor_count=len(partial)), self.assertRaises(vision.VisionPretrainingError):
                vision.map_resnet18_backbone(partial, target, torch)
        for required_key in ("conv1.weight", "bn1.running_mean"):
            damaged = {key: tensor for key, tensor in canonical.items() if key != required_key}
            with self.subTest(missing_required=required_key), self.assertRaises(vision.VisionPretrainingError):
                vision.map_resnet18_backbone(damaged, target, torch)

    def test_missing_unexpected_wrong_shape_or_nonfinite_source_tensors_fail_closed(self):
        import torch
        import torchvision
        _, _, target, _ = self._blueprint()
        source = torchvision.models.resnet18(weights=None).state_dict()
        mutations = (lambda value: value.pop("conv1.weight"),
                     lambda value: value.update(unreviewed_extra=torch.zeros(1)),
                     lambda value: value.update({"conv1.weight": torch.zeros(1)}),
                     lambda value: value.update({"fc.weight": torch.zeros(1)}),
                     lambda value: value.update({"conv1.weight": value["conv1.weight"].double()}),
                     lambda value: value.update({"bn1.running_mean": torch.full_like(value["bn1.running_mean"], float("nan"))}))
        for mutation in mutations:
            changed = {key: tensor.clone() for key, tensor in source.items()}
            mutation(changed)
            with self.subTest(keys=len(changed)), self.assertRaises(vision.VisionPretrainingError):
                vision.map_resnet18_backbone(changed, target, torch)
        changed_target = {key: tensor for key, tensor in target.items() if not key.startswith("model.backbone.")}
        with self.assertRaises(vision.VisionPretrainingError):
            vision.map_resnet18_backbone(source, changed_target, torch)

    def test_all_four_models_remove_vae_and_share_heads_rng_and_only_pretrained_backbone_changes(self):
        import numpy as np
        import torch
        import torchvision
        config, settings, blueprint, rng = self._blueprint()
        pretrained = torchvision.models.resnet18(weights=None).state_dict()
        expected, _ = vision.map_resnet18_backbone(pretrained, blueprint, torch)
        self.assertTrue(any(not torch.equal(tensor, blueprint[key]) for key, tensor in expected.items()))
        for variant in vision.VARIANTS:
            with self.subTest(variant=variant):
                policy, metadata = vision._variant_policy(config, settings, variant, blueprint, np, torch,
                    torch.device("cpu"), rng, pretrained_state=pretrained if variant == "rgb_pretrained" else None)
                vision.assert_no_vae(policy)
                self.assertFalse(policy.config.use_vae)
                self.assertFalse(any("vae_" in key for key in policy.state_dict()))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng["cpu"]))
                self.assertEqual(set(policy.config.input_features),
                                 {"observation.state"} if variant == "state_only" else {"observation.state", native.CAMERA})
                self.assertIsNone(policy.config.env_state_feature)
                self.assertEqual(hasattr(policy.model, "backbone"), variant != "state_only")
                if variant == "state_only":
                    for name in ("encoder_img_feat_input_proj", "encoder_env_state_input_proj", "encoder_cam_feat_pos_embed"):
                        self.assertFalse(hasattr(policy.model, name), name)
                for key, tensor in policy.state_dict().items():
                    wanted = expected[key] if variant == "rgb_pretrained" and key.startswith("model.backbone.") else blueprint[key]
                    self.assertTrue(torch.equal(tensor.cpu(), wanted), key)
                self.assertTrue(metadata)
                if variant == "state_only":
                    batch = {"observation.state": torch.zeros((2, 28)), "action": torch.zeros((2, 16, 28)),
                             "action_is_pad": torch.zeros((2, 16), dtype=torch.bool)}
                    policy.train()
                    loss, components = policy(batch)
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertNotIn("kld_loss", components)
                    policy.eval()
                    prediction = policy.predict_action_chunk({"observation.state": batch["observation.state"]})
                    self.assertEqual(tuple(prediction.shape), (2, 16, 28))
                    self.assertTrue(torch.isfinite(prediction).all())
                del policy
        for variant, weights in (("rgb_scratch", pretrained), ("rgb_mean", pretrained),
                                 ("state_only", pretrained), ("rgb_pretrained", None)):
            with self.subTest(invalid_weight_arm=variant), self.assertRaises(vision.VisionPretrainingError):
                vision._variant_policy(config, settings, variant, blueprint, np, torch,
                    torch.device("cpu"), rng, pretrained_state=weights)


if __name__ == "__main__":
    unittest.main()
