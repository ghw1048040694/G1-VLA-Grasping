"""Matched, no-VAE ACT comparison of random and ImageNet visual initialization.

Only the visual backbone is replaced. The frozen native train/validation data,
train RGB statistics, action labels, optimizer budget and sampled anchors stay
the same. This module never constructs a test dataset or opens test payloads.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
import urllib.request

from .artifacts import sha256_file
from . import act_diagnostics as diagnostics
from . import grasp_act as native


BACKEND = "grasp_square_visual_pretraining_v1"
VARIANTS = ("rgb_scratch", "rgb_pretrained", "rgb_mean", "state_only")
WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHTS_SHA256_PREFIX = "f37072fd"
IMAGE_PREPROCESSING = "unchanged_train_rgb_statistics_v1"


class VisionPretrainingError(ValueError):
    """The matched comparison or its visual-weight provenance was not admitted."""


def validate_visual_pretraining_settings(config, *, allow_unpinned_weights=False):
    settings = copy.deepcopy(config["visual_pretraining"])
    if settings.get("backend") != BACKEND or settings.get("variants") != list(VARIANTS):
        raise VisionPretrainingError("all four predeclared no-VAE variants are required in order")
    if settings.get("use_vae") is not False or settings.get("image_preprocessing") != IMAGE_PREPROCESSING:
        raise VisionPretrainingError("every variant must disable VAE and retain the original train RGB normalization")
    if settings.get("weights_url") != WEIGHTS_URL or settings.get("weights_sha256_prefix") != WEIGHTS_SHA256_PREFIX:
        raise VisionPretrainingError("only the declared official ResNet18 ImageNet1K V1 checkpoint is admitted")
    full_hash = settings.get("weights_sha256")
    if full_hash is None and allow_unpinned_weights:
        pass
    elif (not isinstance(full_hash, str) or len(full_hash) != 64
          or any(char not in "0123456789abcdef" for char in full_hash)
          or not full_hash.startswith(WEIGHTS_SHA256_PREFIX)):
        raise VisionPretrainingError("training requires the complete downloaded weight SHA256 frozen in configuration")
    for name in ("eval_stride", "eval_batch_size", "max_steps", "eval_every_steps", "max_gpu_reserved_bytes", "weights_max_bytes"):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise VisionPretrainingError(name + " must be a positive integer")
    if (settings["max_steps"] != 2000 or settings["eval_every_steps"] != 50
            or settings["eval_stride"] != 30 or settings["eval_batch_size"] != 8
            or settings["max_gpu_reserved_bytes"] > 7 * 1024 ** 3
            or settings["weights_max_bytes"] > 64 * 1024 ** 2):
        raise VisionPretrainingError("the frozen 2000-step budget, validation cadence and memory/download caps must be retained")
    wall = settings.get("max_train_wall_s")
    if isinstance(wall, bool) or not isinstance(wall, (float, int)) or not math.isfinite(wall) or wall != 1800:
        raise VisionPretrainingError("the unchanged per-variant wall budget must be 1800 seconds")
    if (settings.get("seed") != 20261014 or isinstance(settings.get("seed"), bool)
            or config["train"].get("seed") != settings["seed"]
            or config["train"].get("batch_size") != 2 or config["train"].get("learning_rate") != 0.0001
            or config["grasp_act"].get("weight_decay") != 0.0001):
        raise VisionPretrainingError("seed, batch size, learning rate and weight decay must match the frozen trial")
    reference_hash = settings.get("reference_ablation_report_sha256")
    if (not isinstance(reference_hash, str) or len(reference_hash) != 64
            or any(char not in "0123456789abcdef" for char in reference_hash)):
        raise VisionPretrainingError("the train/validation-only reference ablation report must have a complete SHA256")
    for name in ("weights_cache_path", "weights_receipt_path", "reference_ablation_report_path", "output_dir"):
        if not isinstance(settings.get(name), str) or not settings[name]:
            raise VisionPretrainingError(name + " must name a declared project path")
        native._resolve(config, settings[name])
    previous = diagnostics.validate_diagnostic_settings(config)
    for name in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_train_wall_s", "max_gpu_reserved_bytes"):
        if settings[name] != previous[name]:
            raise VisionPretrainingError("the visual comparison differs from the unchanged stage5 " + name)
    return settings


def validate_weight_file(path, settings, *, allow_unpinned=False):
    """Check bytes without importing pickle, torch, or any model implementation."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise VisionPretrainingError("the visual weight cache must be a regular non-symlink file")
    size = path.stat().st_size
    if not 0 < size <= settings["weights_max_bytes"]:
        raise VisionPretrainingError("the visual weight file is empty or exceeds its declared byte cap")
    digest = sha256_file(path)
    expected = settings.get("weights_sha256")
    if not digest.startswith(settings["weights_sha256_prefix"]):
        raise VisionPretrainingError("downloaded weights do not match the official published SHA256 prefix")
    if expected is None and not allow_unpinned:
        raise VisionPretrainingError("a full weight SHA256 must be configured before deserialization or training")
    if expected is not None and digest != expected:
        raise VisionPretrainingError("the visual weight file differs from the configured complete SHA256")
    return {"sha256": digest, "size_bytes": size}


def safe_load_weights(path, torch):
    """Only deserialize a separately verified file, with torch's restricted loader."""
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(isinstance(key, str) for key in state):
        raise VisionPretrainingError("the public visual file must contain one tensor state dictionary")
    return state


def _resnet18_keys():
    keys, tracked = {"conv1.weight", "fc.weight", "fc.bias"}, set()
    def bn(prefix):
        keys.update(prefix + "." + suffix for suffix in ("weight", "bias", "running_mean", "running_var"))
        tracked.add(prefix + ".num_batches_tracked")
    bn("bn1")
    for layer in range(1, 5):
        for block in range(2):
            prefix = "layer{}.{}".format(layer, block)
            for conv in (1, 2):
                keys.add(prefix + ".conv{}.weight".format(conv))
                bn(prefix + ".bn{}".format(conv))
            if layer > 1 and block == 0:
                keys.add(prefix + ".downsample.0.weight")
                bn(prefix + ".downsample.1")
    return keys | tracked, tracked


def map_resnet18_backbone(source, target, torch):
    """Map every visual conv/FrozenBN tensor; never load an ACT head or ImageNet FC."""
    expected, tracked = _resnet18_keys()
    if set(source) != expected:
        raise VisionPretrainingError("the public state keys do not exactly match TorchVision ResNet18 ImageNet1K V1")
    if any(not isinstance(value, torch.Tensor) for value in source.values()):
        raise VisionPretrainingError("the public state dictionary contains a non-tensor value")
    for name in tracked:
        value = source[name]
        if (not isinstance(value, torch.Tensor) or value.dtype != torch.int64 or tuple(value.shape) != ()
                or int(value) < 0):
            raise VisionPretrainingError("a BatchNorm tracking buffer has an invalid type, shape or value")
    if (tuple(source["fc.weight"].shape) != (1000, 512) or tuple(source["fc.bias"].shape) != (1000,)):
        raise VisionPretrainingError("the ignored ImageNet classifier does not have its canonical shape")
    backbone = {name: value for name, value in target.items() if name.startswith("model.backbone.")}
    required = expected - tracked - {"fc.weight", "fc.bias"}
    mapped, mapping = {}, {}
    for name, target_value in backbone.items():
        original = name[len("model.backbone."):]
        if original.startswith("body."):
            original = original[len("body."):]
        if original not in required or original in mapping:
            raise VisionPretrainingError("the ACT backbone is not a complete unambiguous FrozenBN ResNet18")
        value = source[original]
        if (not isinstance(value, torch.Tensor) or not isinstance(target_value, torch.Tensor) or value.dtype != torch.float32
                or target_value.dtype != torch.float32 or value.shape != target_value.shape
                or not bool(torch.isfinite(value).all())):
            raise VisionPretrainingError("a visual tensor has the wrong dtype, shape or finite values")
        mapped[name] = value.detach().cpu().contiguous().clone()
        mapping[original] = name
    if set(mapping) != required or len(mapped) != 100:
        raise VisionPretrainingError("every one of the 100 convolution and FrozenBN tensors must map exactly")
    for name in ("fc.weight", "fc.bias"):
        if source[name].dtype != torch.float32 or not bool(torch.isfinite(source[name]).all()):
            raise VisionPretrainingError("the ignored classifier must still contain finite FP32 tensors")
    return mapped, {"mapped_tensor_count": len(mapped), "mapped_source_to_target": mapping,
                    "ignored_classifier_keys": ["fc.bias", "fc.weight"],
                    "ignored_batchnorm_tracking_keys": sorted(tracked),
                    "all_frozen_batchnorm_affine_and_running_buffers_loaded": True,
                    "non_backbone_tensor_loaded": False,
                    "mapped_tensor_sha256": diagnostics._tensor_digest(mapped)}


def _stage(config, run_dir, name, *, allow_unpinned_weights=False):
    settings = validate_visual_pretraining_settings(config, allow_unpinned_weights=allow_unpinned_weights)
    directory = Path(run_dir)
    path = directory / "reports" / (name + ".json")
    if not directory.is_dir() or path.exists():
        raise VisionPretrainingError("the stage requires an existing run and a new immutable report")
    path.parent.mkdir(exist_ok=True)
    report = {"schema_version": 1, "status": "failed", "passed": False, "settings": settings,
              "training_started": False, "optimizer_steps": 0, "test_payloads_opened": False,
              "test_used_for_selection": False, "physical_success_evaluated": False,
              "project_adapter_validated": False, "closed_loop_control": False}
    return settings, path, report


def fetch_visual_backbone(config, run_dir):
    """Fetch the one declared official file; publish its measured full hash first."""
    settings, path, report = _stage(config, run_dir, "visual_backbone_fetch", allow_unpinned_weights=True)
    started, partial = time.perf_counter(), None
    try:
        destination = native._resolve(config, settings["weights_cache_path"])
        receipt_path = native._resolve(config, settings["weights_receipt_path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        fetched = False
        if destination.exists() and not receipt_path.is_file():
            raise VisionPretrainingError("an existing weight cache without its observed download receipt cannot be adopted")
        if not destination.exists():
            request = urllib.request.Request(settings["weights_url"], headers={"User-Agent": "g1-vla-stage6/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.geturl() != settings["weights_url"]:
                    raise VisionPretrainingError("the official weight download unexpectedly redirected")
                declared_length = response.headers.get("Content-Length")
                if declared_length is not None and not 0 < int(declared_length) <= settings["weights_max_bytes"]:
                    raise VisionPretrainingError("the announced weight size exceeds its declared cap")
                with tempfile.NamedTemporaryFile(prefix="resnet18-download-", suffix=".partial", dir=destination.parent, delete=False) as stream:
                    partial = Path(stream.name)
                    count = 0
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        count += len(block)
                        if count > settings["weights_max_bytes"]:
                            raise VisionPretrainingError("the downloaded weight size exceeds its declared cap")
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
            validate_weight_file(partial, settings, allow_unpinned=True)
            os.link(partial, destination)
            partial.unlink()
            partial = None
            fetched = True
        verified = validate_weight_file(destination, settings, allow_unpinned=True)
        receipt = {"schema_version": 1, "passed": True, "source": "TorchVision ResNet18_Weights.IMAGENET1K_V1",
                   "url": settings["weights_url"], "published_sha256_prefix": WEIGHTS_SHA256_PREFIX,
                   "requested_url": settings["weights_url"], "effective_url": settings["weights_url"],
                   "sha256": verified["sha256"], "size_bytes": verified["size_bytes"], "path": str(destination),
                   "full_sha256_independently_published_by_source": False,
                   "trust_basis": "official fixed HTTPS URL and published eight-character filename SHA256 prefix; full SHA256 measured locally",
                   "public_task_policy_weights_loaded": False, "deserialized_during_fetch": False}
        if receipt_path.exists():
            if native._read(receipt_path) != receipt:
                raise VisionPretrainingError("an immutable weight receipt already exists with different content")
        else:
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            native._write(receipt_path, receipt)
        report.update(status="passed_official_visual_backbone_fetch", passed=True, downloaded_this_run=fetched,
                      weight_file=verified, weight_receipt={"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
                      provenance=receipt, training_requires_full_sha256_in_configuration=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        if partial is not None and partial.exists():
            partial.unlink()
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report


def _load_verified_weights(config, settings, torch):
    path = native._resolve(config, settings["weights_cache_path"])
    verified = validate_weight_file(path, settings)
    receipt_path = native._resolve(config, settings["weights_receipt_path"])
    receipt = native._read(receipt_path)
    if (receipt.get("passed") is not True or receipt.get("url") != WEIGHTS_URL
            or receipt.get("requested_url") != WEIGHTS_URL or receipt.get("effective_url") != WEIGHTS_URL
            or receipt.get("published_sha256_prefix") != WEIGHTS_SHA256_PREFIX
            or receipt.get("sha256") != verified["sha256"] or receipt.get("size_bytes") != verified["size_bytes"]
            or receipt.get("path") != str(path) or receipt.get("deserialized_during_fetch") is not False):
        raise VisionPretrainingError("the complete configured visual file does not match its immutable fetch receipt")
    state = safe_load_weights(path, torch)
    return state, {"path": str(path), **verified, "receipt_path": str(receipt_path),
                   "receipt_sha256": sha256_file(receipt_path), "safe_weights_only_cpu_deserialization": True,
                   "public_task_policy_loaded": False, "source": receipt["source"], "trust_basis": receipt["trust_basis"]}


def assert_no_vae(policy):
    if policy.config.use_vae is not False:
        raise VisionPretrainingError("every formal and temporary policy must have use_vae=False")
    forbidden = [name for name, _ in policy.named_modules() if "vae_encoder" in name]
    forbidden += [name for name in policy.state_dict() if "vae_encoder" in name]
    if forbidden:
        raise VisionPretrainingError("a no-VAE variant retained posterior modules or tensors")
    return {"use_vae": False, "posterior_module_or_tensor_present": False,
            "future_action_conditioned_latent": False, "kl_loss_active": False}


def _variant_policy(config, settings, variant, blueprint, np, torch, device, rng_state, pretrained_state=None):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

    if variant not in VARIANTS:
        raise VisionPretrainingError("unknown visual pretraining variant")

    class QStateOnlyACTConfig(ACTConfig):
        def validate_features(self):
            if (self.image_features or self.env_state_feature or set(self.input_features) != {"observation.state"}
                    or self.robot_state_feature is None or tuple(self.robot_state_feature.shape) != (28,)
                    or tuple(self.action_feature.shape) != (28,)):
                raise VisionPretrainingError("state-only ACT must contain only the original 28-joint state feature")

    torch.manual_seed(config["train"]["seed"])
    torch.cuda.manual_seed_all(config["train"]["seed"])
    keys = ("chunk_size", "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers",
            "n_decoder_layers", "n_vae_encoder_layers", "latent_dim", "dropout", "kl_weight")
    kwargs = {key: settings[key] for key in keys}
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (28,))}
    if variant != "state_only":
        features[native.CAMERA] = PolicyFeature(FeatureType.VISUAL, (3, 240, 320))
    config_type = QStateOnlyACTConfig if variant == "state_only" else ACTConfig
    act_config = config_type(**kwargs, n_obs_steps=1, n_action_steps=settings["chunk_size"], input_features=features,
                            output_features={"action": PolicyFeature(FeatureType.ACTION, (28,))},
                            normalization_mapping={key: NormalizationMode.MEAN_STD for key in ("STATE", "VISUAL", "ACTION")},
                            pretrained_backbone_weights=None, vision_backbone="resnet18", use_vae=False,
                            device=settings["device"], optimizer_lr=config["train"]["learning_rate"],
                            optimizer_lr_backbone=config["train"]["learning_rate"], optimizer_weight_decay=settings["weight_decay"])
    policy = ACTPolicy(act_config).to(device=device, dtype=torch.float32)
    if variant == "state_only":
        for name in ("backbone", "encoder_img_feat_input_proj", "encoder_env_state_input_proj", "encoder_cam_feat_pos_embed"):
            if hasattr(policy.model, name):
                raise VisionPretrainingError("state-only ACT retained the image or environment branch " + name)
        policy.model.register_forward_pre_hook(diagnostics._qstate_size_alias)
    initial = policy.state_dict()
    if any(name not in blueprint or blueprint[name].shape != value.shape for name, value in initial.items()):
        raise VisionPretrainingError("a no-VAE variant has an initial tensor without an original blueprint counterpart")
    selected = {name: blueprint[name] for name in initial}
    common_head = {name: value for name, value in selected.items() if not name.startswith("model.backbone.")}
    metadata = {"variant": variant, "parameter_count": sum(value.numel() for value in policy.parameters()),
                "image_branch_present": variant != "state_only", "environment_feature_present": False,
                "qstate_only_batch_size_alias": variant == "state_only", "batch_size_alias_is_same_qstate_tensor": variant == "state_only",
                "alias_adds_projection_token_or_privileged_information": False,
                "common_fresh_initial_tensor_sha256": diagnostics._tensor_digest(selected),
                "non_backbone_fresh_initial_tensor_sha256": diagnostics._tensor_digest(common_head),
                "common_head_initial_values_verified": True, "public_task_policy_loaded": False,
                "public_visual_backbone_loaded": variant == "rgb_pretrained",
                "input_preprocessing": IMAGE_PREPROCESSING,
                "architecture": json.loads(json.dumps(dataclasses.asdict(act_config), allow_nan=False))}
    metadata["architecture"]["type"] = "act"
    if variant == "rgb_pretrained":
        if pretrained_state is None:
            raise VisionPretrainingError("pretrained RGB requires a separately hash-verified official state dictionary")
        replacement, mapping = map_resnet18_backbone(pretrained_state, initial, torch)
        mapping["tensor_values_changed_from_scratch_count"] = sum(not torch.equal(value, blueprint[name]) for name, value in replacement.items())
        if not mapping["tensor_values_changed_from_scratch_count"]:
            raise VisionPretrainingError("the declared pretrained backbone is identical to random initialization")
        selected.update(replacement)
        metadata["backbone_mapping"] = mapping
    elif pretrained_state is not None:
        raise VisionPretrainingError("public weights must be supplied only to the pretrained visual arm")
    loaded = policy.load_state_dict(selected, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise VisionPretrainingError("the no-VAE initial model did not load completely")
    actual = policy.state_dict()
    if any(not torch.equal(value.detach().cpu(), selected[name]) for name, value in actual.items()):
        raise VisionPretrainingError("initial no-VAE model tensors differ from their exact declared values")
    if any(not torch.equal(actual[name].detach().cpu(), value) for name, value in common_head.items()):
        raise VisionPretrainingError("public visual initialization changed a non-backbone head tensor")
    metadata["no_vae_admission"] = assert_no_vae(policy)
    metadata["use_vae"] = False
    metadata["loaded_initial_tensor_sha256"] = diagnostics._tensor_digest(selected)
    torch.set_rng_state(rng_state["cpu"])
    torch.cuda.set_rng_state_all(rng_state["cuda"])
    metadata["post_original_rgb_blueprint_rng_state_restored"] = True
    metadata["same_stochastic_gradients_assumed"] = False
    return policy, metadata


def verify_scratch_replay(entry, reference_variant):
    """Require the entire previous no-VAE scratch trajectory, not just its score."""
    if (reference_variant.get("status") != "completed_matched_budget_training"
            or reference_variant.get("initialization", {}).get("use_vae") is not False):
        raise VisionPretrainingError("the replay reference must be a completed no-VAE scratch variant")
    reference = dict(reference_variant, training_passed=True)
    try:
        result = diagnostics.verify_rgb_replay(entry, reference)
    except diagnostics.ACTDiagnosticsError as exc:
        raise VisionPretrainingError("no-VAE scratch replay failed: " + str(exc)) from exc
    result["reference_variant"] = "rgb_no_vae"
    result["same_train_rgb_preprocessing"] = True
    return result


def _probe_variant(config, settings, variant, blueprint, datasets, statistics, schedule, np, torch, device,
                   rng_state, pretrained_state, receipt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    receipt.update(passed=False, probe_optimizer_steps=0, temporary_parameters_discarded=False, formal_training_steps=0)
    policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng_state, pretrained_state)
    stats = native._stats(statistics, np)
    mode = "state_only" if variant == "state_only" else "rgb_mean" if variant == "rgb_mean" else "rgb"
    batch, _ = diagnostics._training_batch(datasets["train"], schedule[0], stats, mode, np, torch, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
    policy.train()
    loss, components = policy(batch)
    if not bool(torch.isfinite(loss)) or "kld_loss" in components:
        raise VisionPretrainingError("temporary no-VAE loss is nonfinite or contains KL")
    loss.backward()
    gradients = diagnostics._gradient_report(policy, torch)
    if not gradients["finite"]:
        raise VisionPretrainingError("temporary no-VAE gradients are missing or nonfinite")
    optimizer.step()
    receipt["probe_optimizer_steps"] = 1
    if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
        raise VisionPretrainingError("temporary no-VAE optimizer produced nonfinite parameters")
    del batch, loss
    indices = datasets["validation"].get_eval_indices(settings["eval_stride"])[:settings["eval_batch_size"]]
    if len(indices) != settings["eval_batch_size"]:
        raise VisionPretrainingError("the full configured validation memory-probe batch is unavailable")
    batch, _ = native._batch(datasets["validation"], indices, stats, np, torch, device, include_actions=False)
    if variant == "state_only":
        del batch[native.CAMERA]
    elif variant == "rgb_mean":
        batch[native.CAMERA] = torch.zeros_like(batch[native.CAMERA])
    policy.eval()
    with torch.inference_mode():
        prediction = policy.predict_action_chunk(batch)
    torch.cuda.synchronize()
    if tuple(prediction.shape) != (settings["eval_batch_size"], settings["chunk_size"], 28) or not bool(torch.isfinite(prediction).all()):
        raise VisionPretrainingError("temporary no-VAE validation prediction is invalid")
    native._check_gpu(torch, settings)
    receipt.update(passed=True, finite_gradients=True, temporary_parameters_discarded=True,
                   full_validation_batch_passed=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), initialization=initialization,
                   probe_loss_components=components)
    del optimizer, policy, prediction, batch
    torch.cuda.empty_cache()


def _verify_reference(config, settings, binding):
    path = native._resolve(config, settings["reference_ablation_report_path"])
    if sha256_file(path) != settings["reference_ablation_report_sha256"]:
        raise VisionPretrainingError("the frozen stage5 train/validation report hash changed")
    reference = native._read(path)
    if (reference.get("passed") is not True or reference.get("test_payloads_opened") is not False
            or reference.get("test_used_for_selection") is not False
            or reference.get("variants", {}).get("rgb_no_vae", {}).get("optimizer_steps") != settings["max_steps"]):
        raise VisionPretrainingError("the reference must be the complete prior train/validation-only 2000-step ablation")
    for name in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_train_wall_s", "max_gpu_reserved_bytes"):
        if reference.get("settings", {}).get(name) != settings[name]:
            raise VisionPretrainingError("the fixed comparison differs from the prior report's " + name)
    replay = reference["variants"]["rgb_no_vae"]
    if (replay.get("status") != "completed_matched_budget_training" or len(replay.get("steps", [])) != settings["max_steps"]
            or replay.get("initialization", {}).get("use_vae") is not False
            or [item.get("step") for item in replay.get("validation", [])] != list(range(settings["eval_every_steps"], settings["max_steps"] + 1, settings["eval_every_steps"]))):
        raise VisionPretrainingError("the prior no-VAE trajectory must contain all 2000 losses and 40 validation points")
    for name in ("original_training_module_sha256", "manifest_sha256", "statistics_sha256",
                 "reference_provenance_sha256", "reference_model_sha256", "selected_artifact_sha256"):
        if reference.get("bindings", {}).get(name) != binding[name]:
            raise VisionPretrainingError("the stage5 reference differs in " + name)
    return reference, {"path": str(path), "sha256": sha256_file(path), "variant": "rgb_no_vae",
                       "test_payloads_or_metrics_read": False}


def train_visual_pretraining(config, run_dir):
    """Run exactly four fixed no-VAE budgets, selecting checkpoints on validation."""
    declared, path, report = _stage(config, run_dir, "visual_pretraining")
    started, torch = time.perf_counter(), None
    report.update(variants={}, matched_anchor_schedule=True, optimizer_steps_including_discarded_probes=0)
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise VisionPretrainingError("the unified conditional training admission must be enabled")
        output = native._resolve(config, declared["output_dir"])
        if output.exists():
            raise VisionPretrainingError("the output directory exists; comparison variants never overwrite or resume")
        manifest, statistics, binding, datasets = diagnostics.load_diagnostic_data(config)
        binding = dict(binding, visual_pretraining_module_sha256=sha256_file(__file__))
        report["bindings"] = binding
        reference, report["scratch_replay_reference"] = _verify_reference(config, declared, binding)
        settings = diagnostics._settings(config, declared)
        np, torch, device, blueprint_policy = native._runtime(config, settings, report)
        diagnostics._verify_reference_runtime(config, config["act_diagnostics"], report)
        if any(report[name] != reference.get(name) for name in ("environment", "runtime_sources", "architecture")):
            raise VisionPretrainingError("the runtime or original construction architecture differs from stage5")
        rng_state = {"cpu": torch.get_rng_state().clone(), "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}
        blueprint = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in blueprint_policy.state_dict().items()}
        report["full_rgb_initial_blueprint_sha256"] = diagnostics._tensor_digest(blueprint)
        report["post_rgb_blueprint_rng_sha256"] = {"cpu": hashlib.sha256(rng_state["cpu"].numpy().tobytes()).hexdigest(),
                                                  "cuda": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in rng_state["cuda"]]}
        if (report["full_rgb_initial_blueprint_sha256"] != reference.get("full_rgb_initial_blueprint_sha256")
                or report["post_rgb_blueprint_rng_sha256"] != reference.get("post_rgb_blueprint_rng_sha256")):
            raise VisionPretrainingError("the common initial blueprint or post-construction RNG differs from stage5")
        del blueprint_policy
        torch.cuda.empty_cache()
        pretrained_state, report["visual_backbone_weights"] = _load_verified_weights(config, declared, torch)
        _, report["visual_backbone_mapping_admission"] = map_resnet18_backbone(pretrained_state, blueprint, torch)
        binding["visual_backbone_file_sha256"] = report["visual_backbone_weights"]["sha256"]
        binding["visual_backbone_receipt_sha256"] = report["visual_backbone_weights"]["receipt_sha256"]
        schedule, identities, schedule_report = diagnostics.anchor_schedule(datasets["train"], settings, config["train"]["batch_size"], declared["seed"], np)
        if schedule_report != reference.get("anchor_schedule"):
            raise VisionPretrainingError("the exact sampled anchor schedule differs from stage5")
        output.mkdir(parents=True, exist_ok=False)
        report["anchor_schedule"] = schedule_report
        report["anchor_schedule_artifact"] = diagnostics._save_arrays(output / "training_anchor_schedule.npz", {"sample_episode_frame": identities, "dataset_sample_indices": schedule}, np)
        stats = native._stats(statistics, np)
        report["all_formal_and_probe_models_use_vae"] = False
        report["original_cvae_blueprint_constructed_only_for_identical_initialization_and_rng"] = True
        report["architecture_scope"] = "original untrained construction blueprint; formal per-variant architectures are saved under variants.*.initialization.architecture"
        report["construction_blueprint_initialization"] = report.pop("model_initialization")
        report["model_initialization"] = {"formal_use_vae": False, "public_task_policy_loaded": False,
                                          "public_visual_backbone_loaded_only_for": ["rgb_pretrained"],
                                          "same_non_backbone_fresh_initialization": True}
        report["preprocessing_comparison"] = {"all_arms": IMAGE_PREPROCESSING, "imagenet_crop_resize_or_normalization_added": False,
                                               "source_image_resize": "same prebuilt 320x240 bilinear dataset", "public_weight_recommended_transform_not_applied": True}
        for variant in VARIANTS:
            entry = {"status": "started", "training_started": False, "optimizer_steps": 0, "steps": [], "validation": [],
                     "selection_split": "validation", "test_payloads_opened": False, "test_used_for_selection": False}
            report["variants"][variant] = entry
            variant_started = time.perf_counter()
            entry["memory_probe"] = {}
            weights = pretrained_state if variant == "rgb_pretrained" else None
            _probe_variant(config, settings, variant, blueprint, datasets, statistics, schedule, np, torch, device, rng_state, weights, entry["memory_probe"])
            torch.cuda.reset_peak_memory_stats()
            policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng_state, weights)
            entry["initialization"] = initialization
            if initialization != entry["memory_probe"]["initialization"]:
                raise VisionPretrainingError("the formal model differs from its discarded no-VAE memory probe")
            if variant == "rgb_pretrained" and initialization["non_backbone_fresh_initial_tensor_sha256"] != report["variants"]["rgb_scratch"]["initialization"]["non_backbone_fresh_initial_tensor_sha256"]:
                raise VisionPretrainingError("the pretrained visual arm changed the common non-backbone initialization")
            optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
            best_state, best_metrics, best_mae = None, None, math.inf
            deadline = variant_started + settings["max_train_wall_s"]
            mode = "state_only" if variant == "state_only" else "rgb_mean" if variant == "rgb_mean" else "rgb"
            for offset, indices in enumerate(schedule):
                if time.perf_counter() >= deadline:
                    raise VisionPretrainingError("a variant exhausted its unchanged wall budget before all declared steps")
                step = offset + 1
                policy.train()
                batch, _ = diagnostics._training_batch(datasets["train"], indices, stats, mode, np, torch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, components = policy(batch)
                if not bool(torch.isfinite(loss)) or "kld_loss" in components:
                    raise VisionPretrainingError("a no-VAE training loss is nonfinite or contains KL")
                loss.backward()
                gradients = diagnostics._gradient_report(policy, torch)
                if not gradients["finite"]:
                    raise VisionPretrainingError("a no-VAE training gradient is missing or nonfinite")
                optimizer.step()
                report["training_started"], entry["training_started"] = True, True
                report["optimizer_steps"] += 1
                entry["optimizer_steps"] = step
                torch.cuda.synchronize()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
                    raise VisionPretrainingError("a no-VAE optimizer produced nonfinite parameters")
                native._check_gpu(torch, settings)
                entry["steps"].append({"step": step, "loss": float(loss.detach().cpu()), "loss_components": components,
                                       "gradients": gradients, "elapsed_s": time.perf_counter() - variant_started})
                del batch, loss
                if step % settings["eval_every_steps"] == 0:
                    image_mode = "mean_image" if variant == "rgb_mean" else "real"
                    metrics, _ = diagnostics._evaluate(policy, datasets["validation"], statistics, settings, np, torch, device,
                                                       image_mode=image_mode, state_only=variant == "state_only", deadline=deadline)
                    metrics["inference_latent"] = "zero_latent_no_vae_no_labels"
                    entry["validation"].append({"step": step, "metrics": metrics})
                    mae = metrics["policy"]["all_28"]["mae"]
                    if mae < best_mae:
                        best_mae, best_metrics, entry["selected_step"] = mae, metrics, step
                        best_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
            if variant == "rgb_scratch":
                entry["stage5_no_vae_scratch_replay"] = verify_scratch_replay(entry, reference["variants"]["rgb_no_vae"])
            if best_state is None:
                raise VisionPretrainingError("a no-VAE variant did not complete a selectable validation")
            entry["checkpoint"] = diagnostics._save_variant_checkpoint(output / variant / "checkpoint_best", config, manifest, statistics, binding,
                                                                         initialization, best_state, entry["selected_step"], best_metrics)
            policy.load_state_dict(best_state, strict=True)
            policy.requires_grad_(False).eval()
            entry["selected_prior_inference"] = {}
            for split in ("train", "validation"):
                image_mode = "mean_image" if variant == "rgb_mean" else "real"
                metrics, arrays = diagnostics._evaluate(policy, datasets[split], statistics, settings, np, torch, device,
                                                        image_mode=image_mode, state_only=variant == "state_only", deadline=deadline)
                metrics["inference_latent"] = "zero_latent_no_vae_no_labels"
                entry["selected_prior_inference"][split] = {"metrics": metrics,
                    "predictions": diagnostics._save_arrays(output / variant / (split + "_predictions.npz"), arrays, np)}
            entry["loss_windows"] = diagnostics._loss_windows(entry["steps"])
            entry["gpu"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            entry["total_elapsed_s"] = time.perf_counter() - variant_started
            entry["status"] = "completed_matched_budget_no_vae_training"
            del optimizer, policy, best_state
            torch.cuda.empty_cache()
        report["status"], report["passed"] = "completed_train_validation_visual_pretraining_comparison", True
        report["limitations"] = ["ImageNet visual weights alone are compared; no pretrained action head or public task policy is used.",
                                  "All arms retain original train RGB statistics; the ImageNet recommended preprocessing is deliberately not introduced as another variable.",
                                  "All models disable the entire VAE mechanism; state-only has fewer parameters and FLOPs than image models.",
                                  "One seed and three validation episodes support a development comparison, not a generalization or physical-success claim.",
                                  "Test arrays, images, predictions and metrics are excluded; no controller or robot is evaluated."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["discarded_probe_optimizer_steps"] = sum(entry.get("memory_probe", {}).get("probe_optimizer_steps", 0) for entry in report["variants"].values())
        report["optimizer_steps_including_discarded_probes"] = report["optimizer_steps"] + report["discarded_probe_optimizer_steps"]
        native._gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report
