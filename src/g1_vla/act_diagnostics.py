"""Train/validation-only ACT diagnostics under one declared native protocol.

Image interventions keep the original state, labels and validation denominator.
Matched-budget variants share all compatible initial parameters and the exact
sampled training anchors.  Neither function opens test arrays, RGB or metrics.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
import traceback

from .artifacts import sha256_file
from . import grasp_act as native


BACKEND = "grasp_square_act_diagnostics_v1"
VISUAL_MODES = ("real", "mean_image", "within_episode_shift", "cross_episode_swap")
VARIANTS = ("rgb", "rgb_mean", "state_only", "rgb_no_vae")


class ACTDiagnosticsError(ValueError):
    """A diagnostic input, intervention or matched-budget admission failed."""


def validate_diagnostic_settings(config):
    settings = copy.deepcopy(config["act_diagnostics"])
    if settings.get("backend") != BACKEND:
        raise ACTDiagnosticsError("the native ACT diagnostic backend must be explicit")
    if settings.get("visual_modes") != list(VISUAL_MODES) or settings.get("variants") != list(VARIANTS):
        raise ACTDiagnosticsError("all four predeclared interventions and training variants are required in order")
    for key in ("eval_stride", "eval_batch_size", "max_steps", "eval_every_steps", "max_gpu_reserved_bytes"):
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ACTDiagnosticsError(key + " must be a positive integer")
    if settings["max_steps"] % settings["eval_every_steps"]:
        raise ACTDiagnosticsError("each matched budget must end at a declared validation point")
    if settings["max_gpu_reserved_bytes"] > 7 * 1024 ** 3:
        raise ACTDiagnosticsError("the GPU allocator cap must not exceed 7 GiB")
    if settings.get("image_shift_fraction") != 0.5:
        raise ACTDiagnosticsError("the within-episode image shift must remain the predeclared half-episode shift")
    wall = settings.get("max_train_wall_s")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise ACTDiagnosticsError("max_train_wall_s must be finite and positive")
    if settings.get("seed") != config["train"].get("seed") or isinstance(settings.get("seed"), bool):
        raise ACTDiagnosticsError("all variants must use the unified training seed")
    native.validate_grasp_settings(config)
    return settings


def load_diagnostic_data(config):
    """Verify selected artifacts and normalization, without hashing test payloads."""
    import numpy as np
    from .grasp_data import (GraspSquareDataset, load_grasp_manifest,
                             train_only_statistics, _verify_artifact)
    settings = validate_diagnostic_settings(config)
    reference = native._resolve(config, settings["reference_checkpoint_path"])
    reference_config_path = native._resolve(config, settings["reference_config_path"])
    if reference_config_path != reference / "experiment_config.json":
        raise ACTDiagnosticsError("reference_config_path must name the independently saved checkpoint configuration")
    reference_config = native._read(reference_config_path)
    provenance = native._read(reference / "provenance.json")
    saved_binding = provenance.get("bindings", {})
    native.verify_grasp_checkpoint(reference, saved_binding)
    if native._digest(reference_config) != saved_binding.get("experiment_config_sha256"):
        raise ACTDiagnosticsError("reference configuration does not match its original full binding")
    for key in ("seed", "batch_size", "learning_rate"):
        if config["train"].get(key) != reference_config["train"].get(key):
            raise ACTDiagnosticsError("the diagnostic train.{} differs from the original RGB trial".format(key))
    if sha256_file(native.__file__) != saved_binding.get("module_sha256"):
        raise ACTDiagnosticsError("the original ACT implementation changed since the reference checkpoint")
    manifest_path = native._resolve(config, config["grasp_act"]["dataset_manifest_path"])
    if manifest_path.name != "manifest.json" or sha256_file(manifest_path) != saved_binding.get("manifest_sha256"):
        raise ACTDiagnosticsError("the diagnostic native manifest differs from the reference checkpoint")
    manifest = load_grasp_manifest(manifest_path.parent, expected_settings=config["source_native"],
                                   expected_source=config["external_data"], verify_files=False)
    native.validate_grasp_manifest(manifest, config["grasp_act"])
    selected = [item for item in manifest["episodes"] if item["split"] in ("train", "validation")]
    files = {}
    for item in selected:
        for key in ("arrays", "rgb", "decoded_receipt"):
            path = _verify_artifact(manifest_path.parent, item[key])
            files[item[key]["relative_path"]] = item[key]["sha256"]
    statistics_path = _verify_artifact(manifest_path.parent, manifest["statistics"])
    if sha256_file(statistics_path) != saved_binding.get("statistics_sha256"):
        raise ACTDiagnosticsError("train-only normalization differs from the reference checkpoint")
    statistics = native._read(statistics_path)
    native._stats(statistics, np)
    actual = train_only_statistics(manifest_path.parent, selected, native.CAMERA)
    for key in ("observation.state", "action", native.CAMERA):
        for suffix in ("mean", "std", "unclamped_std"):
            saved, recomputed = np.asarray(statistics["features"][key][suffix]), np.asarray(actual["features"][key][suffix])
            if saved.shape != recomputed.shape or not np.allclose(saved, recomputed, rtol=1e-12, atol=1e-12):
                raise ACTDiagnosticsError("saved statistics differ from actual train-only observations")
    datasets = {split: GraspSquareDataset(manifest_path.parent, split, manifest=manifest, verify=False)
                for split in ("train", "validation")}
    binding = {"experiment_config_sha256": native._digest(config), "diagnostic_module_sha256": sha256_file(__file__),
               "original_training_module_sha256": sha256_file(native.__file__),
               "manifest_sha256": sha256_file(manifest_path), "statistics_sha256": sha256_file(statistics_path),
               "reference_provenance_sha256": sha256_file(reference / "provenance.json"),
               "reference_model_sha256": provenance["files"]["model.safetensors"]["sha256"],
               "verified_payload_splits": ["train", "validation"], "test_payloads_opened": False,
               "selected_artifact_sha256": files}
    return manifest, statistics, binding, datasets


def image_donor_plan(identities, mode, np):
    """Return fixed donor indices; every non-real donor differs from its anchor."""
    ids = np.asarray(identities, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[1] != 2 or not len(ids) or mode not in VISUAL_MODES:
        raise ACTDiagnosticsError("image interventions require a nonempty episode/frame identity plan")
    if len({tuple(item) for item in ids.tolist()}) != len(ids):
        raise ACTDiagnosticsError("evaluation anchor identities must be unique")
    if mode == "mean_image":
        return np.full(len(ids), -1, dtype=np.int64)
    plan = np.arange(len(ids), dtype=np.int64)
    if mode == "real":
        return plan
    episodes = sorted(set(ids[:, 0].tolist()))
    if mode == "cross_episode_swap" and len(episodes) < 2:
        raise ACTDiagnosticsError("cross-episode swapping needs at least two validation episodes")
    for episode_offset, episode in enumerate(episodes):
        positions = np.flatnonzero(ids[:, 0] == episode)
        if mode == "within_episode_shift":
            if len(positions) < 2:
                raise ACTDiagnosticsError("within-episode derangement needs at least two anchors per episode")
            plan[positions] = np.roll(positions, max(1, len(positions) // 2))
        else:
            other = episodes[(episode_offset + 1) % len(episodes)]
            donors = np.flatnonzero(ids[:, 0] == other)
            plan[positions] = donors[np.arange(len(positions)) % len(donors)]
    if np.any(plan == np.arange(len(ids))):
        raise ACTDiagnosticsError("an image intervention retained an anchor's own image")
    return plan


def diagnostic_metrics(prediction, target, state, action_mean, action_std, identities, np):
    """Same raw-label denominator for policy, state hold and train mean."""
    prediction, target, state = np.asarray(prediction), np.asarray(target), np.asarray(state)
    ids = np.asarray(identities, dtype=np.int64)
    if state.shape != (len(target), 28) or ids.shape != (len(target), 2):
        raise ACTDiagnosticsError("metric states and identities must match every action chunk")
    if not np.isfinite(state).all():
        raise ACTDiagnosticsError("state-hold observations must be finite")
    held = np.repeat(state[:, None, :], target.shape[1], axis=1)
    constant = np.broadcast_to(action_mean, target.shape)
    def compared(values, labels):
        result = native.compute_action_metrics(values, labels, np)
        difference = values.astype(np.float64) - labels.astype(np.float64)
        result["normalized_l1"] = float(np.abs(difference / action_std).mean())
        result["mae_by_horizon"] = np.abs(difference).mean(axis=(0, 2)).tolist()
        result["rmse_by_horizon"] = np.sqrt(np.square(difference).mean(axis=(0, 2))).tolist()
        return result
    result = {"policy": compared(prediction, target), "state_hold": compared(held, target),
              "train_mean": compared(constant, target), "anchor_independence_assumed": False,
              "physical_success_evaluated": False, "selection_criterion": "policy.all_28.mae"}
    result["per_episode"] = {str(int(episode)): {
        "policy": compared(prediction[ids[:, 0] == episode], target[ids[:, 0] == episode]),
        "state_hold": compared(held[ids[:, 0] == episode], target[ids[:, 0] == episode]),
        "train_mean": compared(constant[ids[:, 0] == episode], target[ids[:, 0] == episode])}
        for episode in np.unique(ids[:, 0])}
    result["policy_to_state_hold_mae_ratio"] = (result["policy"]["all_28"]["mae"] / result["state_hold"]["all_28"]["mae"]
                                                 if result["state_hold"]["all_28"]["mae"] > 0 else None)
    return result


def _stage(config, run_dir, name):
    settings = validate_diagnostic_settings(config)
    directory = Path(run_dir)
    path = directory / "reports" / (name + ".json")
    if not directory.is_dir() or path.exists():
        raise ACTDiagnosticsError("diagnostics require an existing run and a new immutable report")
    path.parent.mkdir(exist_ok=True)
    return settings, path, {"schema_version": 1, "status": "failed", "passed": False, "settings": settings,
                            "training_started": False, "optimizer_steps": 0, "test_payloads_opened": False,
                            "test_used_for_selection": False, "physical_success_evaluated": False,
                            "project_adapter_validated": False, "closed_loop_control": False}


def _settings(config, diagnostics):
    result = copy.deepcopy(config["grasp_act"])
    for key in ("eval_stride", "eval_batch_size", "max_steps", "eval_every_steps", "max_train_wall_s", "max_gpu_reserved_bytes"):
        result[key] = diagnostics[key]
    return result


def _save_arrays(path, arrays, np):
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return {"path": str(path), "sha256": sha256_file(path), "size_bytes": Path(path).stat().st_size}


def _verify_reference_runtime(config, diagnostics, report):
    reference = native._resolve(config, diagnostics["reference_checkpoint_path"])
    original = native._read(reference / "experiment_config.json")
    provenance = native._read(reference / "provenance.json")
    memory_path = native._resolve(config, original["grasp_act"]["memory_report_path"])
    memory = native.validate_memory_admission(native._read(memory_path), provenance["bindings"], original["grasp_act"])
    if (report["environment"] != memory.get("environment") or report["runtime_sources"] != memory.get("runtime_sources")
            or report["architecture"] != memory.get("architecture")
            or report["architecture"] != native._read(reference / "config.json")):
        raise ACTDiagnosticsError("the reference architecture, installed ACT sources or runtime environment changed")
    report["reference_compatibility"] = {
        "original_full_config_sha256_verified": True,
        "current_configuration_is_original": native._digest(config) == native._digest(original),
        "same_native_data_statistics_architecture_runtime": True,
        "original_memory_report_path": str(memory_path), "original_memory_report_sha256": sha256_file(memory_path),
        "reference_checkpoint_or_test_claim_modified": False}


def _evaluate(policy, dataset, statistics, settings, np, torch, device, *, image_mode="real", state_only=False, deadline=None):
    indices = dataset.get_eval_indices(settings["eval_stride"])
    ids = np.asarray([dataset.samples[index] for index in indices], dtype=np.int64)
    donors = image_donor_plan(ids, image_mode, np)
    stats = native._stats(statistics, np)
    predictions, targets, states = [], [], []
    policy.eval()
    for start in range(0, len(indices), settings["eval_batch_size"]):
        if deadline is not None and time.perf_counter() >= deadline:
            raise ACTDiagnosticsError("diagnostic wall-clock budget expired before complete validation")
        positions = list(range(start, min(start + settings["eval_batch_size"], len(indices))))
        selected = [indices[position] for position in positions]
        batch, samples = native._batch(dataset, selected, stats, np, torch, device, include_actions=False)
        if state_only:
            del batch[native.CAMERA]
        elif image_mode == "mean_image":
            batch[native.CAMERA] = torch.zeros_like(batch[native.CAMERA])
        elif image_mode != "real":
            donor_samples = [dataset[indices[int(donors[position])]] for position in positions]
            raw = np.stack([sample[native.CAMERA] for sample in donor_samples])
            batch[native.CAMERA] = torch.from_numpy(np.asarray((raw - stats[native.CAMERA][0]) / stats[native.CAMERA][1], dtype=np.float32)).to(device)
        if "action" in batch:
            raise ACTDiagnosticsError("inference must not receive a recorded action or posterior latent")
        with torch.inference_mode():
            prediction = policy.predict_action_chunk(batch).cpu().numpy()
        if prediction.shape != (len(samples), settings["chunk_size"], 28) or not np.isfinite(prediction).all():
            raise ACTDiagnosticsError("diagnostic output is not a complete finite original action chunk")
        predictions.extend(prediction * stats["action"][1] + stats["action"][0])
        targets.extend(sample["action"] for sample in samples)
        states.extend(sample["observation.state"] for sample in samples)
        native._check_gpu(torch, settings)
    arrays = {"predictions_source_actions": np.asarray(predictions), "target_source_recorded_actions": np.asarray(targets),
              "observation_state": np.asarray(states), "sample_episode_frame": ids,
              "image_donor_episode_frame": np.asarray([ids[index] if index >= 0 else [-1, -1] for index in donors], dtype=np.int64)}
    metrics = diagnostic_metrics(arrays["predictions_source_actions"], arrays["target_source_recorded_actions"],
                                 arrays["observation_state"], stats["action"][0], stats["action"][1], ids, np)
    metrics.update(image_mode=image_mode, state_only=state_only, inference_latent="zero_prior_no_labels",
                   eval_stride_source_frames=settings["eval_stride"], evaluated_split=dataset.split,
                   intervention_state_or_label_changed=False, camera_input_present=not state_only)
    return metrics, arrays


def run_act_diagnostics(config, run_dir):
    """Evaluate the original selected checkpoint; no optimizer is constructed."""
    settings, path, report = _stage(config, run_dir, "act_diagnostics")
    started, torch = time.perf_counter(), None
    try:
        _, statistics, binding, datasets = load_diagnostic_data(config)
        report["bindings"] = binding
        native_settings = _settings(config, settings)
        np, torch, device, policy = native._runtime(config, native_settings, report)
        _verify_reference_runtime(config, settings, report)
        from safetensors.torch import load_file
        checkpoint = native._resolve(config, settings["reference_checkpoint_path"])
        loaded = policy.load_state_dict(load_file(str(checkpoint / "model.safetensors"), device="cpu"), strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise ACTDiagnosticsError("the original checkpoint did not load completely")
        policy.requires_grad_(False).eval()
        report["checkpoint_loaded_strictly"] = True
        report["model_initialization"] = {"checkpoint_loaded_strictly": True, "public_checkpoint_loaded": False,
                                          "original_training_initialization": "from_scratch",
                                          "reference_model_sha256": binding["reference_model_sha256"]}
        report["optimizer_created"] = False
        report["interventions"] = {}
        for mode in VISUAL_MODES:
            metrics, arrays = _evaluate(policy, datasets["validation"], statistics, native_settings, np, torch, device, image_mode=mode)
            report["interventions"][mode] = {"metrics": metrics, "predictions": _save_arrays(Path(run_dir) / ("act_validation_" + mode + ".npz"), arrays, np)}
        metrics, arrays = _evaluate(policy, datasets["train"], statistics, native_settings, np, torch, device)
        report["fixed_train_prior_inference"] = {"metrics": metrics, "predictions": _save_arrays(Path(run_dir) / "act_train_prior_predictions.npz", arrays, np)}
        report["status"], report["passed"] = "passed_validation_image_diagnostics", True
        report["limitations"] = ["Image interventions test this selected undertrained checkpoint, not the general usefulness of vision.",
                                  "Normalized-zero images represent the training mean image and may be out of distribution.",
                                  "Only train/validation recorded-action errors are measured; test payloads and robot control are excluded."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        native._gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report


def _tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _qstate_size_alias(module, args):
    """Work around LeRobot 0.4.3's batch-size branch, with no env feature token.

    Stock ACT.forward chooses batch size from image or environment state even
    when its transformer supports the robot-state token alone.  This local
    alias is not declared as an input feature, hence no environment projection
    or token exists.  It cannot carry any information beyond the same qstate.
    """
    batch = args[0]
    if (set(batch) - {"observation.state", "action", "action_is_pad"}
            or "observation.state" not in batch or batch["observation.state"].shape[-1] != 28):
        raise ACTDiagnosticsError("state-only ACT admits exactly the original 28 joint observations and optional train labels")
    aliased = dict(batch)
    aliased["observation.environment_state"] = batch["observation.state"]
    return (aliased,)


def _variant_policy(config, settings, variant, blueprint, np, torch, device, rng_state):
    """Build a true reduced ACT, then strictly copy every common initial tensor."""
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

    class QStateOnlyACTConfig(ACTConfig):
        def validate_features(self):
            if (self.image_features or self.env_state_feature or set(self.input_features) != {"observation.state"}
                    or self.robot_state_feature is None or tuple(self.robot_state_feature.shape) != (28,)
                    or tuple(self.action_feature.shape) != (28,)):
                raise ACTDiagnosticsError("the state-only configuration must have one 28-joint state token and no environment or image feature")

    torch.manual_seed(config["train"]["seed"])
    torch.cuda.manual_seed_all(config["train"]["seed"])
    architecture_keys = ("chunk_size", "dim_model", "n_heads", "dim_feedforward", "n_encoder_layers",
                         "n_decoder_layers", "n_vae_encoder_layers", "latent_dim", "dropout", "kl_weight")
    kwargs = {key: settings[key] for key in architecture_keys}
    features = {"observation.state": PolicyFeature(FeatureType.STATE, (28,))}
    if variant != "state_only":
        features[native.CAMERA] = PolicyFeature(FeatureType.VISUAL, (3, 240, 320))
    config_type = QStateOnlyACTConfig if variant == "state_only" else ACTConfig
    act_config = config_type(**kwargs, n_obs_steps=1, n_action_steps=settings["chunk_size"], input_features=features,
                            output_features={"action": PolicyFeature(FeatureType.ACTION, (28,))},
                            normalization_mapping={key: NormalizationMode.MEAN_STD for key in ("STATE", "VISUAL", "ACTION")},
                            pretrained_backbone_weights=None, vision_backbone="resnet18", use_vae=variant != "rgb_no_vae",
                            device=settings["device"], optimizer_lr=config["train"]["learning_rate"],
                            optimizer_lr_backbone=config["train"]["learning_rate"], optimizer_weight_decay=settings["weight_decay"])
    policy = ACTPolicy(act_config).to(device=device, dtype=torch.float32)
    if variant == "state_only":
        for name in ("backbone", "encoder_img_feat_input_proj", "encoder_env_state_input_proj", "encoder_cam_feat_pos_embed"):
            if hasattr(policy.model, name):
                raise ACTDiagnosticsError("the true state-only architecture unexpectedly retained " + name)
        policy.model.register_forward_pre_hook(_qstate_size_alias)
    initial = policy.state_dict()
    if any(name not in blueprint or blueprint[name].shape != value.shape for name, value in initial.items()):
        raise ACTDiagnosticsError("a variant has an initial tensor without an exact common RGB blueprint counterpart")
    selected = {name: blueprint[name] for name in initial}
    loaded = policy.load_state_dict(selected, strict=True)
    if loaded.missing_keys or loaded.unexpected_keys or not all(torch.equal(value.detach().cpu(), blueprint[name]) for name, value in policy.state_dict().items()):
        raise ACTDiagnosticsError("a variant did not exactly inherit all common RGB initial tensors")
    metadata = {"variant": variant, "parameter_count": sum(value.numel() for value in policy.parameters()),
                "use_vae": act_config.use_vae, "image_branch_present": variant != "state_only",
                "environment_feature_present": False, "qstate_only_batch_size_alias": variant == "state_only",
                "batch_size_alias_is_same_qstate_tensor": variant == "state_only",
                "alias_adds_projection_token_or_privileged_information": False,
                "same_common_initial_tensor_values_verified": True, "copied_initial_tensor_count": len(selected),
                "copied_initial_tensor_sha256": _tensor_digest(selected), "public_checkpoint_loaded": False,
                "architecture": json.loads(json.dumps(dataclasses.asdict(act_config), allow_nan=False))}
    metadata["architecture"]["type"] = "act"
    # Restore the exact state immediately after the original full RGB model's
    # construction. Probe/model construction must not perturb formal dropout
    # or VAE draws. Different architectures still consume randomness differently.
    torch.set_rng_state(rng_state["cpu"])
    torch.cuda.set_rng_state_all(rng_state["cuda"])
    metadata["post_rgb_blueprint_rng_state_restored"] = True
    metadata["same_stochastic_gradients_assumed"] = False
    return policy, metadata


def _training_batch(dataset, indices, stats, variant, np, torch, device):
    batch, samples = native._batch(dataset, indices, stats, np, torch, device)
    if variant == "state_only":
        del batch[native.CAMERA]
    elif variant == "rgb_mean":
        batch[native.CAMERA] = torch.zeros_like(batch[native.CAMERA])
    return batch, samples


def _gradient_report(policy, torch):
    groups = {}
    finite, total_squared, parameter_count = True, 0.0, 0
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        finite = finite and bool(torch.isfinite(parameter.grad).all())
        squared = float(parameter.grad.detach().float().square().sum().cpu())
        total_squared += squared
        parameter_count += 1
        group = "backbone" if name.startswith("model.backbone") else "vae" if name.startswith("model.vae_") else "state_projection" if name.startswith("model.encoder_robot_state") else "action_head" if name.startswith("model.action_head") else "transformer_other"
        groups[group] = groups.get(group, 0.0) + squared
    return {"finite": finite and parameter_count > 0, "parameter_tensor_count": parameter_count,
            "global_l2_norm": math.sqrt(total_squared), "l2_norm_by_module_group": {key: math.sqrt(value) for key, value in groups.items()}}


def anchor_schedule(dataset, settings, batch_size, seed, np):
    schedule = np.random.default_rng(seed).integers(0, len(dataset), size=(settings["max_steps"], batch_size))
    identities = np.asarray([[dataset.samples[int(index)] for index in batch] for batch in schedule], dtype=np.int64)
    unique = len(set(int(value) for value in schedule.ravel()))
    episodes = sorted(set(int(identity[0]) for batch in identities for identity in batch))
    report = {"seed": seed, "steps": settings["max_steps"], "batch_size": batch_size,
              "sample_draw_count": int(schedule.size), "available_train_anchor_count": len(dataset),
              "unique_train_anchor_count": unique, "unique_train_anchor_fraction": unique / len(dataset),
              "sampled_episode_ids": episodes, "sampling_with_replacement": True,
              "sample_identity_sha256": hashlib.sha256(identities.tobytes()).hexdigest(),
              "sample_identity_shape": list(identities.shape)}
    report["unique_anchors_by_episode"] = {str(episode): len({int(frame) for item in identities for identity, frame in item if int(identity) == episode}) for episode in episodes}
    return schedule, identities, report


def _probe_variant(config, settings, variant, blueprint, datasets, statistics, schedule, np, torch, device, rng_state, receipt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    receipt.update(passed=False, probe_optimizer_steps=0, temporary_parameters_discarded=False, formal_training_steps=0)
    policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng_state)
    stats = native._stats(statistics, np)
    batch, _ = _training_batch(datasets["train"], schedule[0], stats, variant, np, torch, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
    policy.train()
    loss, components = policy(batch)
    if not bool(torch.isfinite(loss)):
        raise ACTDiagnosticsError("temporary matched-variant loss is nonfinite")
    loss.backward()
    gradients = _gradient_report(policy, torch)
    if not gradients["finite"]:
        raise ACTDiagnosticsError("temporary matched-variant gradients are missing or nonfinite")
    optimizer.step()
    receipt["probe_optimizer_steps"] = 1
    if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
        raise ACTDiagnosticsError("temporary variant optimizer produced nonfinite parameters")
    del batch, loss
    indices = datasets["validation"].get_eval_indices(settings["eval_stride"])[:settings["eval_batch_size"]]
    if len(indices) != settings["eval_batch_size"]:
        raise ACTDiagnosticsError("the configured full validation memory-probe batch is unavailable")
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
        raise ACTDiagnosticsError("temporary validation memory-probe prediction is invalid")
    native._check_gpu(torch, settings)
    receipt.update({"passed": True, "probe_optimizer_steps": 1, "finite_gradients": gradients["finite"],
               "temporary_parameters_discarded": True, "formal_training_steps": 0,
               "full_validation_batch_passed": True, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "initialization": initialization,
               "probe_loss_components": components})
    del optimizer, policy, prediction, batch
    torch.cuda.empty_cache()
    return receipt


def _save_variant_checkpoint(directory, config, manifest, statistics, binding, initialization, best_state, selected_step, metrics):
    from safetensors.torch import save_file
    directory.mkdir(parents=True, exist_ok=False)
    save_file(best_state, str(directory / "model.safetensors"))
    native._write(directory / "config.json", initialization["architecture"])
    native._write(directory / "experiment_config.json", config)
    native._write(directory / "manifest.json", manifest)
    native._write(directory / "train_stats.json", statistics)
    native._write(directory / "training_summary.json", {"selected_step": selected_step, "validation": metrics,
                                                        "selection_split": "validation", "test_used_for_selection": False,
                                                        "resumable_optimizer_state_saved": False})
    files = {path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in directory.iterdir() if path.is_file()}
    provenance = {"schema_version": 1, "passed": True, "bindings": binding, "files": files,
                  "variant_initialization": initialization, "selected_step": selected_step,
                  "selection_split": "validation", "test_used_for_selection": False,
                  "physical_success_evaluated": False, "resumable_optimizer_state_saved": False}
    native._write(directory / "provenance.json", provenance)
    return {"path": str(directory), "provenance_sha256": sha256_file(directory / "provenance.json"), "selected_step": selected_step}


def _loss_windows(steps):
    result = {}
    for name, window in (("first_20", steps[:20]), ("last_20", steps[-20:])):
        if not window:
            continue
        result[name] = {"count": len(window), "mean_total_loss": sum(item["loss"] for item in window) / len(window),
                        "mean_training_l1_loss": sum(item["loss_components"]["l1_loss"] for item in window) / len(window),
                        "mean_kld_loss": sum(item["loss_components"].get("kld_loss", 0.0) for item in window) / len(window)}
    return result


def verify_rgb_replay(entry, reference):
    """Reject a changed first 200-step trajectory before interpreting extensions."""
    count = reference.get("optimizer_steps")
    if (reference.get("training_passed") is not True or isinstance(count, bool) or not isinstance(count, int)
            or count <= 0 or len(reference.get("steps", [])) != count or len(entry.get("steps", [])) < count):
        raise ACTDiagnosticsError("the original RGB optimization trajectory is incomplete")
    maximum_loss_difference = 0.0
    for actual, expected in zip(entry["steps"][:count], reference["steps"]):
        if actual["step"] != expected["step"]:
            raise ACTDiagnosticsError("RGB replay optimizer step identities changed")
        pairs = [(actual["loss"], expected["loss"])]
        if set(actual["loss_components"]) != set(expected["loss_components"]):
            raise ACTDiagnosticsError("RGB replay training loss component identities changed")
        pairs.extend((actual["loss_components"][key], expected["loss_components"][key]) for key in expected["loss_components"])
        for observed, original in pairs:
            if not math.isclose(observed, original, rel_tol=1e-6, abs_tol=1e-6):
                raise ACTDiagnosticsError("RGB replay training trajectory differs from the original trial")
            maximum_loss_difference = max(maximum_loss_difference, abs(observed - original))
    validations = {item["step"]: item for item in entry["validation"]}
    maximum_validation_difference = 0.0
    for expected in reference["validation"]:
        if expected["step"] not in validations:
            raise ACTDiagnosticsError("RGB replay omitted an original validation point")
        observed = validations[expected["step"]]["metrics"]["policy"]["all_28"]["mae"]
        original = expected["metrics"]["policy"]["all_28"]["mae"]
        if not math.isclose(observed, original, rel_tol=1e-6, abs_tol=1e-7):
            raise ACTDiagnosticsError("RGB replay validation differs from the original trial")
        maximum_validation_difference = max(maximum_validation_difference, abs(observed - original))
    return {"passed": True, "replayed_optimizer_steps": count, "replayed_validation_points": len(reference["validation"]),
            "maximum_absolute_loss_difference": maximum_loss_difference,
            "maximum_absolute_validation_mae_difference": maximum_validation_difference,
            "relative_tolerance": 1e-6, "loss_absolute_tolerance": 1e-6, "validation_mae_absolute_tolerance": 1e-7}


def validate_reference_replay_budget(settings, reference):
    """Every original validation point must be reached before any extension."""
    count = reference.get("optimizer_steps")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ACTDiagnosticsError("the reference optimization budget is invalid")
    points = [item.get("step") for item in reference.get("validation", [])]
    if (not points or any(isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= count for step in points)
            or points != sorted(set(points)) or count not in points):
        raise ACTDiagnosticsError("the reference validation plan must include its final optimization step")
    if settings["max_steps"] < count:
        raise ACTDiagnosticsError("matched diagnostics must retain the complete original RGB step budget")
    if any(step % settings["eval_every_steps"] for step in points):
        raise ACTDiagnosticsError("the configured validation cadence omits an original RGB replay point")
    return {"reference_optimizer_steps": count, "required_replay_validation_steps": points,
            "configured_validation_cadence_reaches_all_reference_points": True}


def _require_completed_rgb_replay(entry):
    if entry.get("original_rgb_replay", {}).get("passed") is not True:
        raise ACTDiagnosticsError("RGB training cannot complete without its original trajectory replay guard")


def train_act_ablation(config, run_dir):
    """Four predeclared native models; no test loading or automatic budget extension."""
    diagnostics, path, report = _stage(config, run_dir, "act_ablation")
    started, torch = time.perf_counter(), None
    report.update(variants={}, matched_anchor_schedule=True, optimizer_steps_including_discarded_probes=0)
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise ACTDiagnosticsError("the unified conditional training admission must be enabled")
        output = native._resolve(config, diagnostics["output_dir"])
        if output.exists():
            raise ACTDiagnosticsError("the diagnostic output directory already exists; variants never overwrite or resume")
        manifest, statistics, binding, datasets = load_diagnostic_data(config)
        report["bindings"] = binding
        settings = _settings(config, diagnostics)
        reference_dir = native._resolve(config, diagnostics["reference_checkpoint_path"])
        reference_training_path = reference_dir.parent / "reports" / "grasp_act_training.json"
        reference_training = native._read(reference_training_path)
        if (reference_training.get("bindings") != native._read(reference_dir / "provenance.json")["bindings"]
                or reference_training.get("checkpoint", {}).get("provenance_sha256") != binding["reference_provenance_sha256"]
                or reference_training.get("training_passed") is not True):
            raise ACTDiagnosticsError("the reference training trajectory is not bound to the original selected checkpoint")
        report["reference_replay_budget_admission"] = validate_reference_replay_budget(settings, reference_training)
        report["reference_training_report"] = {"path": str(reference_training_path), "sha256": sha256_file(reference_training_path)}
        np, torch, device, blueprint_policy = native._runtime(config, settings, report)
        _verify_reference_runtime(config, diagnostics, report)
        rng_state = {"cpu": torch.get_rng_state().clone(), "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}
        blueprint = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in blueprint_policy.state_dict().items()}
        report["full_rgb_initial_blueprint_sha256"] = _tensor_digest(blueprint)
        report["post_rgb_blueprint_rng_sha256"] = {"cpu": hashlib.sha256(rng_state["cpu"].numpy().tobytes()).hexdigest(),
                                                  "cuda": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in rng_state["cuda"]]}
        del blueprint_policy
        torch.cuda.empty_cache()
        schedule, identities, schedule_report = anchor_schedule(datasets["train"], settings, config["train"]["batch_size"], diagnostics["seed"], np)
        output.mkdir(parents=True, exist_ok=False)
        report["anchor_schedule"] = schedule_report
        report["anchor_schedule_artifact"] = _save_arrays(output / "training_anchor_schedule.npz", {"sample_episode_frame": identities, "dataset_sample_indices": schedule}, np)
        stats = native._stats(statistics, np)
        for variant in VARIANTS:
            entry = {"status": "started", "training_started": False, "optimizer_steps": 0, "steps": [], "validation": [],
                     "selection_split": "validation", "test_payloads_opened": False, "test_used_for_selection": False}
            report["variants"][variant] = entry
            variant_started = time.perf_counter()
            entry["memory_probe"] = {}
            _probe_variant(config, settings, variant, blueprint, datasets, statistics, schedule, np, torch, device, rng_state, entry["memory_probe"])
            torch.cuda.reset_peak_memory_stats()
            policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng_state)
            entry["initialization"] = initialization
            if initialization != entry["memory_probe"]["initialization"]:
                raise ACTDiagnosticsError("the formal model differs from its discarded temporary memory probe")
            optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
            best_state, best_metrics, best_mae = None, None, math.inf
            deadline = variant_started + settings["max_train_wall_s"]
            for offset, indices in enumerate(schedule):
                if time.perf_counter() >= deadline:
                    raise ACTDiagnosticsError("a matched variant exhausted its wall-clock budget before all declared steps")
                step = offset + 1
                policy.train()
                batch, _ = _training_batch(datasets["train"], indices, stats, variant, np, torch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, components = policy(batch)
                if not bool(torch.isfinite(loss)):
                    raise ACTDiagnosticsError("a matched variant produced a nonfinite loss")
                loss.backward()
                gradients = _gradient_report(policy, torch)
                if not gradients["finite"]:
                    raise ACTDiagnosticsError("a matched variant produced missing or nonfinite gradients")
                optimizer.step()
                report["training_started"], entry["training_started"] = True, True
                report["optimizer_steps"] += 1
                entry["optimizer_steps"] = step
                torch.cuda.synchronize()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
                    raise ACTDiagnosticsError("a matched variant produced nonfinite parameters")
                native._check_gpu(torch, settings)
                entry["steps"].append({"step": step, "loss": float(loss.detach().cpu()), "loss_components": components,
                                       "gradients": gradients, "elapsed_s": time.perf_counter() - variant_started})
                del batch, loss
                if step % settings["eval_every_steps"] == 0:
                    mode = "mean_image" if variant == "rgb_mean" else "real"
                    metrics, _ = _evaluate(policy, datasets["validation"], statistics, settings, np, torch, device,
                                            image_mode=mode, state_only=variant == "state_only", deadline=deadline)
                    entry["validation"].append({"step": step, "metrics": metrics})
                    mae = metrics["policy"]["all_28"]["mae"]
                    if mae < best_mae:
                        best_mae, best_metrics, entry["selected_step"] = mae, metrics, step
                        best_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
                    if variant == "rgb" and step == reference_training["optimizer_steps"]:
                        entry["original_rgb_replay"] = verify_rgb_replay(entry, reference_training)
            if variant == "rgb":
                _require_completed_rgb_replay(entry)
            if best_state is None:
                raise ACTDiagnosticsError("a variant did not complete a selectable validation")
            entry["checkpoint"] = _save_variant_checkpoint(output / variant / "checkpoint_best", config, manifest, statistics, binding,
                                                            initialization, best_state, entry["selected_step"], best_metrics)
            policy.load_state_dict(best_state, strict=True)
            policy.requires_grad_(False).eval()
            entry["selected_prior_inference"] = {}
            for split in ("train", "validation"):
                mode = "mean_image" if variant == "rgb_mean" else "real"
                metrics, arrays = _evaluate(policy, datasets[split], statistics, settings, np, torch, device,
                                            image_mode=mode, state_only=variant == "state_only", deadline=deadline)
                entry["selected_prior_inference"][split] = {"metrics": metrics,
                    "predictions": _save_arrays(output / variant / (split + "_predictions.npz"), arrays, np)}
            entry["loss_windows"] = _loss_windows(entry["steps"])
            entry["gpu"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            entry["total_elapsed_s"] = time.perf_counter() - variant_started
            entry["status"] = "completed_matched_budget_training"
            del optimizer, policy, best_state
            torch.cuda.empty_cache()
        report["status"], report["passed"] = "completed_train_validation_ablation", True
        report["limitations"] = ["All variants share their exact compatible initial tensors and sampled anchors, but parameter counts and FLOPs differ.",
                                  "The no-VAE variant removes the complete variational mechanism; it does not isolate KL weight alone.",
                                  "Posterior-conditioned stochastic train L1 is distinct from zero-prior label-free inference error.",
                                  "Checkpoint selection uses validation only; no test payload, physical success or current controller is evaluated."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["discarded_probe_optimizer_steps"] = sum(entry.get("memory_probe", {}).get("probe_optimizer_steps", 0)
                                                        for entry in report["variants"].values())
        report["optimizer_steps_including_discarded_probes"] = report["optimizer_steps"] + report["discarded_probe_optimizer_steps"]
        native._gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report
