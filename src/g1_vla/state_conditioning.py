"""Frozen-checkpoint tests of state conditioning, with the raw decoder held fixed.

The network state can be replaced; the source anchor, camera and labels cannot.
No optimizer is constructed, no checkpoint is selected, and no test payload is
read. Intervention failures describe this development set, including possible
out-of-distribution inputs, rather than a physical or causal success criterion.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time
import traceback

from .artifacts import sha256_file
from . import act_diagnostics as diagnostics
from . import constant_residual as constant
from . import expanded_residual as expanded
from . import grasp_act as native
from . import state_residual as residual
from . import vision_pretraining as vision

BACKEND = "grasp_square_frozen_state_conditioning_v1"
MODES = ("real", "train_mean", "within_episode_shift", "cross_episode_swap")
VARIANTS = expanded.VARIANTS
SPLITS = ("train", "validation")
WITHIN_RULE = "half_length_cyclic_shift_v1"
CROSS_RULE = "next_sorted_episode_same_relative_rank_v1"
MODEL_CONSTANT_METHOD = "train_eval_model_raw_residual_coordinate_median_fp32scale_fp64fit_v1"
ATOL = 1e-6
SELECTED_STEP = 1650


class StateConditioningError(ValueError):
    """The fixed checkpoint, data, or conditioning protocol changed."""


def validate_state_conditioning_settings(config):
    settings = copy.deepcopy(config["state_conditioning"])
    for key, expected in (("backend", BACKEND), ("variants", list(VARIANTS)),
                          ("modes", list(MODES)), ("eval_splits", list(SPLITS)),
                          ("within_episode_rule", WITHIN_RULE), ("cross_episode_rule", CROSS_RULE),
                          ("real_reproduction_atol", ATOL), ("optimizer_steps", 0),
                          ("model_output_constant_method", MODEL_CONSTANT_METHOD), ("tie_tolerance_rad", 0.0)):
        if settings.get(key) != expected or isinstance(settings.get(key), bool):
            raise StateConditioningError("the fixed conditioning protocol changed: " + key)
    for key, value in (("eval_stride", 30), ("eval_batch_size", 8)):
        if type(settings.get(key)) is not int or settings[key] != value:
            raise StateConditioningError("the frozen evaluation denominator or batch changed: " + key)
        if settings[key] != config["expanded_residual"][key]:
            raise StateConditioningError("the evaluation differs from the frozen expanded comparison")
    if (type(settings.get("max_eval_wall_s")) not in (float, int)
            or not 0 < settings["max_eval_wall_s"] <= 600
            or type(settings.get("max_gpu_reserved_bytes")) is not int
            or settings["max_gpu_reserved_bytes"] != config["expanded_residual"]["max_gpu_reserved_bytes"]):
        raise StateConditioningError("the inference wall or unchanged GPU budget is invalid")
    if (config["native_diagnostics"]["stationary_speed_rad_s"] != .02
            or config["native_diagnostics"]["moving_speed_rad_s"] != .1):
        raise StateConditioningError("the analysis-only motion thresholds changed")
    for key in ("reference_expanded_report_path", "reference_constant_report_path", "output_dir"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise StateConditioningError(key + " must be an explicit project path")
        native._resolve(config, settings[key])
    for key in ("reference_expanded_report_sha256", "reference_constant_report_sha256"):
        digest = settings.get(key)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise StateConditioningError(key + " must be a complete lowercase SHA256")
    for key in ("reference_expanded_report_path", "reference_expanded_report_sha256"):
        if settings[key] != config["constant_residual"][key]:
            raise StateConditioningError("conditioning and the train-label median must share the frozen expanded reference")
    return settings


def state_donor_indices(identities, mode, np):
    """Use only fixed evaluation identities, never states, labels or errors."""
    ids = np.asarray(identities)
    if (ids.dtype != np.int64 or ids.ndim != 2 or ids.shape[1] != 2 or len(ids) == 0
            or len({tuple(row) for row in ids.tolist()}) != len(ids)
            or np.any(ids < 0) or np.any(ids[:, 1] % 30 != 0)):
        raise StateConditioningError("donors require unique nonnegative stride-30 int64 episode/frame identities")
    if mode not in MODES:
        raise StateConditioningError("unknown state intervention")
    if mode == "train_mean":
        return np.full(len(ids), -1, dtype=np.int64)
    if mode == "real":
        return np.arange(len(ids), dtype=np.int64)
    groups = {int(episode): np.flatnonzero(ids[:, 0] == episode) for episode in np.unique(ids[:, 0])}
    for indices in groups.values():
        if not np.all(np.diff(ids[indices, 1]) > 0):
            raise StateConditioningError("each evaluation episode must retain increasing source-frame order")
    donors = np.empty(len(ids), dtype=np.int64)
    if mode == "within_episode_shift":
        for indices in groups.values():
            if len(indices) < 2:
                raise StateConditioningError("half-episode intervention requires at least two anchors")
            donors[indices] = np.roll(indices, len(indices) // 2)
        if np.any(donors == np.arange(len(ids))) or not np.array_equal(ids[donors, 0], ids[:, 0]):
            raise StateConditioningError("within-episode donors must be nonidentity and retain the episode")
    else:
        episodes = sorted(groups)
        if len(episodes) < 2:
            raise StateConditioningError("cross-episode intervention requires at least two episodes")
        for offset, episode in enumerate(episodes):
            source = groups[episode]
            destination = groups[episodes[(offset + 1) % len(episodes)]]
            # Integer arithmetic implements floor((rank + .5) * M / N).
            ranks = ((2 * np.arange(len(source), dtype=np.int64) + 1) * len(destination)) // (2 * len(source))
            donors[source] = destination[ranks]
        if np.any(ids[donors, 0] == ids[:, 0]):
            raise StateConditioningError("cross-episode donors must change the source episode")
    return donors


def network_state_intervention(raw_states, identities, state_mean, state_std, mode, np):
    """Return network inputs and donor provenance; never replace decoder anchors."""
    states = np.asarray(raw_states)
    mean, std = np.asarray(state_mean), np.asarray(state_std)
    if (states.dtype != np.float32 or states.shape != (len(identities), 28)
            or mean.dtype != np.float32 or std.dtype != np.float32 or mean.shape != (28,) or std.shape != (28,)
            or not np.isfinite(states).all() or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)):
        raise StateConditioningError("state intervention requires finite FP32 raw states and train statistics")
    donors = state_donor_indices(identities, mode, np)
    if mode == "train_mean":
        network_raw = np.broadcast_to(mean, states.shape).copy()
        normalized = np.zeros_like(states)
        donor_ids = np.full((len(states), 2), -1, dtype=np.int64)
    else:
        network_raw = states[donors].copy()
        normalized = np.ascontiguousarray((network_raw - mean) / std, dtype=np.float32)
        donor_ids = np.asarray(identities)[donors].copy()
    return normalized, network_raw, donor_ids, donors


def _verified_artifact(record):
    path = Path(record["path"])
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != record["size_bytes"]
            or sha256_file(path) != record["sha256"]):
        raise StateConditioningError("a frozen inference artifact changed: " + str(path))
    return path


def _verify_references(config, declared):
    path = native._resolve(config, declared["reference_expanded_report_path"])
    if sha256_file(path) != declared["reference_expanded_report_sha256"]:
        raise StateConditioningError("the frozen expanded comparison SHA256 changed")
    reference = native._read(path)
    if (reference.get("passed") is not True
            or reference.get("status") != "completed_expanded_train_validation_state_residual_comparison"
            or reference.get("settings") != config["expanded_residual"]
            or reference.get("test_payloads_opened") is not False or reference.get("test_used_for_selection") is not False
            or reference.get("optimizer_steps") != 4000):
        raise StateConditioningError("the reference must retain the completed frozen expanded comparison")
    for key, module in (("original_training_module_sha256", native), ("diagnostic_module_sha256", diagnostics),
                        ("prior_no_vae_builder_sha256", vision), ("state_residual_module_sha256", residual),
                        ("expanded_residual_module_sha256", expanded)):
        if reference["bindings"].get(key) != sha256_file(module.__file__):
            raise StateConditioningError("a frozen source implementation changed: " + key)
    for variant in VARIANTS:
        entry = reference.get("variants", {}).get(variant, {})
        if (entry.get("selected_step") != SELECTED_STEP or entry.get("optimizer_steps") != 2000
                or entry.get("status") != "completed_matched_budget_state_residual_training"
                or entry.get("checkpoint", {}).get("selected_step") != SELECTED_STEP
                or residual.select_checkpoint(entry["validation"]) != entry.get("selection_summary")):
            raise StateConditioningError("the stage8 selected step 1650 and original budget must remain fixed")
        checkpoint = entry["checkpoint"]
        directory = Path(checkpoint["path"])
        if sha256_file(directory / "provenance.json") != checkpoint["provenance_sha256"]:
            raise StateConditioningError("the selected checkpoint provenance changed")
        provenance = native.verify_grasp_checkpoint(directory, reference["bindings"])
        if (provenance.get("selected_step") != SELECTED_STEP
                or provenance.get("variant_initialization") != entry.get("initialization")
                or entry["initialization"].get("use_vae") is not False):
            raise StateConditioningError("the selected no-VAE checkpoint architecture or step changed")
        prior_config = native._read(directory / "experiment_config.json")
        if (native._digest(prior_config) != reference["bindings"]["experiment_config_sha256"]
                or prior_config["grasp_act"] != config["grasp_act"]
                or prior_config["expanded_residual"] != config["expanded_residual"]):
            raise StateConditioningError("the checkpoint construction or expanded source configuration changed")
        for split in SPLITS:
            _verified_artifact(entry["selected_prior_inference"][split]["predictions"])
    constant_path = native._resolve(config, declared["reference_constant_report_path"])
    if sha256_file(constant_path) != declared["reference_constant_report_sha256"]:
        raise StateConditioningError("the frozen train-label median report changed")
    label_reference = native._read(constant_path)
    if (label_reference.get("passed") is not True
            or label_reference.get("status") != "completed_train_only_constant_residual_audit"
            or label_reference.get("test_payloads_opened") is not False
            or label_reference.get("test_metrics_read") is not False
            or label_reference.get("optimizer_steps") != 0
            or label_reference["bindings"].get("reference_expanded_report_sha256") != sha256_file(path)
            or label_reference["bindings"].get("module_sha256") != sha256_file(constant.__file__)):
        raise StateConditioningError("the fixed train-label median source binding changed")
    fit = label_reference.get("fits", {}).get("expanded_fourteen", {})
    if (set(fit.get("episode_ids", [])) != set(expanded.TRAIN)
            or fit.get("complete_anchor_count") != 13835 or fit.get("validation_or_test_used_for_fit") is not False):
        raise StateConditioningError("the label median must use all fourteen training trajectories only")
    return reference, label_reference


def _prepare(config, declared, report):
    reference, label_reference = _verify_references(config, declared)
    stage7, _ = expanded._verify_reference(config, config["expanded_residual"])
    manifest, statistics, binding, datasets = expanded.load_expanded_data(config, config["expanded_residual"], stage7)
    for key in ("manifest_sha256", "statistics_sha256", "selected_artifact_sha256", "candidate_admission"):
        if binding[key] != reference["bindings"][key]:
            raise StateConditioningError("the verified fourteen-train data differ from the frozen checkpoint: " + key)
    for variant in VARIANTS:
        checkpoint = Path(reference["variants"][variant]["checkpoint"]["path"])
        if (native._read(checkpoint / "manifest.json") != manifest
                or native._read(checkpoint / "train_stats.json") != statistics
                or native._read(checkpoint / "config.json") != reference["variants"][variant]["initialization"]["architecture"]):
            raise StateConditioningError("the selected checkpoint manifest, normalization or architecture differs from its frozen source")
    settings = diagnostics._settings(config, config["expanded_residual"])
    np, torch, device, blueprint_policy = native._runtime(config, settings, report)
    for key in ("environment", "runtime_sources", "architecture"):
        if report[key] != reference[key]:
            raise StateConditioningError("the installed frozen inference runtime changed: " + key)
    rng = {"cpu": torch.get_rng_state().clone(), "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}
    blueprint = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in blueprint_policy.state_dict().items()}
    digest = diagnostics._tensor_digest(blueprint)
    rng_digest = {"cpu": hashlib.sha256(rng["cpu"].numpy().tobytes()).hexdigest(),
                  "cuda": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in rng["cuda"]]}
    if digest != reference["full_rgb_initial_blueprint_sha256"] or rng_digest != reference["post_rgb_blueprint_rng_sha256"]:
        raise StateConditioningError("the architecture construction blueprint or RNG state changed")
    del blueprint_policy
    torch.cuda.empty_cache()
    with np.load(_verified_artifact(label_reference["fits"]["expanded_fourteen"]["parameter_artifact"]), allow_pickle=False) as saved:
        label_parameter = saved["constant_raw_residual"].copy()
        if set(saved["fit_episode_ids"].tolist()) != set(expanded.TRAIN):
            raise StateConditioningError("the frozen label-median parameter has different fit episodes")
    if label_parameter.dtype != np.float32 or label_parameter.shape != (16, 28) or not np.isfinite(label_parameter).all():
        raise StateConditioningError("the fixed train-label median parameter is invalid")
    report["bindings"] = {"experiment_config_sha256": native._digest(config), "module_sha256": sha256_file(__file__),
        "reference_expanded_report_sha256": declared["reference_expanded_report_sha256"],
        "reference_constant_report_sha256": declared["reference_constant_report_sha256"],
        "expanded_data_bindings": binding, "selected_checkpoint_steps": {name: SELECTED_STEP for name in VARIANTS},
        "frozen_checkpoint_provenance_sha256": {name: reference["variants"][name]["checkpoint"]["provenance_sha256"] for name in VARIANTS},
        "full_rgb_construction_blueprint_sha256": digest, "post_blueprint_rng_sha256": rng_digest}
    report["construction_blueprint_initialization"] = report.pop("model_initialization")
    report["construction_blueprint_used_for_training"] = False
    return reference, statistics, datasets, settings, label_parameter, np, torch, device, blueprint, rng


def _cohort(dataset, stride, np):
    if dataset.split not in SPLITS:
        raise StateConditioningError("conditioning may open train and validation payloads only")
    indices = dataset.get_eval_indices(stride)
    ids = np.asarray([dataset.samples[index] for index in indices], dtype=np.int64)
    states, speeds = [], []
    for index, (episode, frame) in zip(indices, ids):
        sample = dataset[int(index)]
        states.append(sample["observation.state"])
        source = dataset._arrays[int(episode)][0]
        speeds.append(float(np.max(np.abs(source[int(frame)].astype(np.float64) - source[int(frame) - 1].astype(np.float64)))) * 30
                      if frame else np.nan)
    expected = 468 if dataset.split == "train" else 85
    if len(ids) != expected or set(ids[:, 0].tolist()) != set(expanded.TRAIN if dataset.split == "train" else expanded.VALIDATION):
        raise StateConditioningError("the fixed stride-30 train or validation population changed")
    return indices, ids, np.asarray(states, dtype=np.float32), np.asarray(speeds, dtype=np.float64)


def _evaluate(policy, dataset, cohort, statistics, settings, mode, variant, np, torch, device, *, deadline=None):
    indices, ids, raw_states, speeds = cohort
    stats = expanded._stats(statistics, np)
    network, network_raw, donor_ids, donor_offsets = network_state_intervention(raw_states, ids, *stats["observation.state"], mode, np)
    predictions, scaled, targets, camera_hashes = [], [], [], []
    policy.eval()
    for start in range(0, len(indices), settings["eval_batch_size"]):
        if deadline is not None and time.perf_counter() >= deadline:
            raise StateConditioningError("the fixed inference wall budget expired")
        stop = min(start + settings["eval_batch_size"], len(indices))
        batch, samples = native._batch(dataset, indices[start:stop], stats, np, torch, device, include_actions=False)
        actual_states = np.stack([sample["observation.state"] for sample in samples])
        if actual_states.tobytes() != raw_states[start:stop].tobytes():
            raise StateConditioningError("an intervention changed the original source decoder anchor")
        batch["observation.state"] = torch.from_numpy(network[start:stop].copy()).to(device)
        if variant == "state_only_residual":
            del batch[native.CAMERA]
            camera_hashes.extend([""] * len(samples))
        else:
            camera = batch[native.CAMERA].detach().cpu().numpy()
            camera_hashes.extend(hashlib.sha256(value.tobytes()).hexdigest() for value in camera)
        if set(batch) != ({"observation.state"} if variant == "state_only_residual" else {"observation.state", native.CAMERA}):
            raise StateConditioningError("inference may receive only the declared network state and unchanged camera")
        with torch.inference_mode():
            value = policy.predict_action_chunk(batch).detach().cpu().numpy()
        if value.dtype != np.float32 or value.shape != (len(samples), 16, 28) or not np.isfinite(value).all():
            raise StateConditioningError("the frozen network must produce finite FP32 residual chunks")
        scaled.extend(value)
        predictions.extend(residual.decode_residual(value, actual_states, stats["action"][1], np))
        targets.extend(sample["action"] for sample in samples)
        native._check_gpu(torch, settings)
    z = np.asarray(scaled, dtype=np.float32)
    arrays = {"predictions_source_actions": np.asarray(predictions, dtype=np.float32),
        "predictions_scaled_state_residual": z, "predictions_raw_state_residual": np.multiply(z, stats["action"][1], dtype=np.float32),
        "target_source_recorded_actions": np.asarray(targets, dtype=np.float32), "observation_state": raw_states.copy(),
        "sample_episode_frame": ids.copy(), "action_std": stats["action"][1].copy(),
        "network_input_normalized_state": network, "network_input_raw_state": network_raw,
        "network_state_donor_episode_frame": donor_ids, "network_state_donor_eval_offset": donor_offsets,
        "original_rgb_sample_episode_frame": ids.copy(),
        "rgb_input_sample_episode_frame": ids.copy() if variant != "state_only_residual" else np.full_like(ids, -1),
        "rgb_input_bytes_sha256": np.asarray(camera_hashes, dtype="U64"),
        "anchor_max_joint_speed_rad_s": speeds.copy(), "anchor_speed_known": np.isfinite(speeds)}
    return arrays


def _reproduction(arrays, record, np):
    with np.load(_verified_artifact(record), allow_pickle=False) as saved:
        for name in ("observation_state", "sample_episode_frame", "target_source_recorded_actions", "action_std"):
            if arrays[name].dtype != saved[name].dtype or arrays[name].shape != saved[name].shape or arrays[name].tobytes() != saved[name].tobytes():
                raise StateConditioningError("real frozen inference changed original source bytes: " + name)
        maximum = {}
        for name in ("predictions_source_actions", "predictions_scaled_state_residual"):
            difference = float(np.max(np.abs(arrays[name].astype(np.float64) - saved[name].astype(np.float64))))
            if not np.isfinite(difference) or difference > ATOL:
                raise StateConditioningError("real inference did not reproduce the frozen selected checkpoint: " + name)
            maximum[name] = difference
    return {"passed": True, "absolute_tolerance": ATOL, "maximum_absolute_difference": maximum,
            "source_anchor_labels_identities_action_scale_bitwise_equal": True, "reference_artifact": record}


def intervention_consistency(arrays, real, mode, variant, np):
    """Verify that only the declared network input was changed."""
    for name in ("observation_state", "sample_episode_frame", "target_source_recorded_actions", "action_std",
                 "original_rgb_sample_episode_frame", "rgb_input_sample_episode_frame", "rgb_input_bytes_sha256"):
        if arrays[name].dtype != real[name].dtype or arrays[name].shape != real[name].shape or arrays[name].tobytes() != real[name].tobytes():
            raise StateConditioningError("state intervention changed a fixed anchor, label, scale or camera: " + name)
    decoded = residual.decode_residual(arrays["predictions_scaled_state_residual"], arrays["observation_state"], arrays["action_std"], np)
    if decoded.tobytes() != arrays["predictions_source_actions"].tobytes():
        raise StateConditioningError("intervention predictions must decode from the original raw source anchor")
    result = {"original_anchor_labels_and_rgb_fixed": True, "decoder_recomputed_bitwise": True,
              "network_state_only_intervention": True, "state_only_donor_equivariance_checked": False,
              "state_only_mean_residual_constant_checked": False}
    if variant == "state_only_residual" and mode in ("within_episode_shift", "cross_episode_swap"):
        expected = real["predictions_scaled_state_residual"][arrays["network_state_donor_eval_offset"]]
        maximum = float(np.max(np.abs(arrays["predictions_scaled_state_residual"].astype(np.float64) - expected.astype(np.float64))))
        if maximum > ATOL:
            raise StateConditioningError("pure state-only residual output did not follow its donor network state")
        result.update(state_only_donor_equivariance_checked=True, donor_scaled_residual_max_abs_difference=maximum)
    if variant == "state_only_residual" and mode == "train_mean":
        values = arrays["predictions_raw_state_residual"].astype(np.float64)
        maximum = float(np.max(np.max(values, axis=0) - np.min(values, axis=0)))
        if np.any(arrays["network_input_normalized_state"] != 0) or maximum > ATOL:
            raise StateConditioningError("state-only normalized-zero input did not produce a numerically constant residual")
        result.update(state_only_mean_residual_constant_checked=True, raw_residual_max_range_rad=maximum)
    return result


def verify_mean_state_cross_split(train_raw_residual, validation_raw_residual, np):
    """A fixed state-only input must retain one constant across both split loops."""
    values = np.concatenate((np.asarray(train_raw_residual), np.asarray(validation_raw_residual)), axis=0)
    if values.ndim != 3 or values.shape[1:] != (16, 28) or not np.isfinite(values).all():
        raise StateConditioningError("cross-split constant audit requires finite residual chunks")
    values = values.astype(np.float64)
    maximum = float(np.max(values.max(axis=0) - values.min(axis=0)))
    if maximum > ATOL:
        raise StateConditioningError("state-only mean input produced different constants across train and validation")
    return {"passed": True, "raw_residual_max_range_rad": maximum, "absolute_tolerance_rad": ATOL,
            "combined_anchor_count": int(len(values)), "new_inference_forwards_performed": False}


def run_state_conditioning_audit(config, run_dir):
    """Measure four fixed state interventions on two already selected policies."""
    declared = validate_state_conditioning_settings(config)
    directory = Path(run_dir)
    path = directory / "reports" / "state_conditioning.json"
    if not directory.is_dir() or path.exists():
        raise StateConditioningError("conditioning requires an existing run and a new immutable report")
    path.parent.mkdir(exist_ok=True)
    report = {"schema_version": 1, "settings": declared, "status": "failed", "passed": False,
        "training_started": False, "optimizer_created": False, "optimizer_steps": 0,
        "discarded_probe_optimizer_steps": 0, "test_payloads_opened": False, "test_metrics_read": False,
        "test_used_for_selection": False, "checkpoint_selection_performed": False,
        "closed_form_model_output_fit_performed": False, "physical_success_evaluated": False,
        "closed_loop_control": False, "variants": {}}
    started, torch = time.perf_counter(), None
    deadline = started + declared["max_eval_wall_s"]
    try:
        from safetensors.torch import load_file
        from . import conditioning_metrics as metrics
        output = native._resolve(config, declared["output_dir"])
        if output.exists():
            raise StateConditioningError("conditioning never overwrites an existing artifact directory")
        reference, statistics, datasets, settings, label_parameter, np, torch, device, blueprint, rng = _prepare(config, declared, report)
        report["bindings"]["conditioning_metrics_module_sha256"] = sha256_file(metrics.__file__)
        output.mkdir(parents=True)
        cohorts = {split: _cohort(datasets[split], declared["eval_stride"], np) for split in SPLITS}
        stats = expanded._stats(statistics, np)
        report["protocol"] = {"network_state_may_change": True, "raw_decoder_anchor_labels_rgb_fixed": True,
            "network_inference_receives_labels": False, "state_mean_from_fourteen_train_only": True,
            "selected_step_fixed": SELECTED_STEP, "new_neural_optimizer_steps": 0,
            "donors_from_same_split_evaluation_anchor_set": True, "donor_mapping_uses_labels_or_errors": False,
            "within_shift_direction": "np.roll(group_eval_indices, len(group_eval_indices)//2)",
            "cross_episode_rank": "floor((rank+0.5)*donor_episode_anchor_count/source_episode_anchor_count)",
            "model_output_constant_fit_cohort": "468_stride30_original_fourteen_train_anchors_only",
            "model_output_constant_fit_receives_labels": False, "interventions_used_for_checkpoint_or_parameter_selection": False}
        for variant in VARIANTS:
            frozen = reference["variants"][variant]
            policy, initialization = residual._variant_policy(config, settings, variant, blueprint, np, torch, device, rng)
            if initialization != frozen["initialization"]:
                raise StateConditioningError("the reconstructed no-VAE architecture differs from its frozen selected checkpoint")
            checkpoint = Path(frozen["checkpoint"]["path"])
            state = load_file(str(checkpoint / "model.safetensors"), device="cpu")
            loaded = policy.load_state_dict(state, strict=True)
            if loaded.missing_keys or loaded.unexpected_keys or any(not torch.equal(value.detach().cpu(), state[name]) for name, value in policy.state_dict().items()):
                raise StateConditioningError("strict selected-checkpoint load did not preserve every saved tensor")
            del state
            policy.requires_grad_(False).eval()
            entry = {"selected_checkpoint": frozen["checkpoint"], "selected_step": SELECTED_STEP,
                "initialization_architecture_reproduced": True, "checkpoint_loaded_strictly": True,
                "model_parameters_require_grad": False, "model_eval_mode": True,
                "optimizer_created": False, "optimizer_steps": 0, "splits": {}}
            report["variants"][variant] = entry
            variant_dir = output / variant
            variant_dir.mkdir()
            parameter = None
            mean_raw_residual_by_split = {}
            for split in SPLITS:
                data = datasets[split]
                evaluated = {}
                for mode in MODES:
                    arrays = _evaluate(policy, data, cohorts[split], statistics, settings, mode, variant, np, torch, device,
                                       deadline=deadline)
                    if mode == "real":
                        reproduction = _reproduction(arrays, frozen["selected_prior_inference"][split]["predictions"], np)
                        real = arrays
                        if split == "train":
                            parameter, fit = metrics.fit_model_output_constant(arrays["predictions_scaled_state_residual"],
                                arrays["sample_episode_frame"], stats["action"][1], expanded.TRAIN, np,
                                expected_train_identities=cohorts["train"][1])
                            fit["parameter_artifact"] = diagnostics._save_arrays(variant_dir / "model_output_constant_parameters.npz",
                                {"constant_raw_residual": parameter, "fit_sample_episode_frame": arrays["sample_episode_frame"]}, np)
                            entry["model_output_constant_fit"] = fit
                            report["closed_form_model_output_fit_performed"] = True
                    if variant == "state_only_residual" and mode == "train_mean":
                        mean_raw_residual_by_split[split] = arrays["predictions_raw_state_residual"].copy()
                    item = {"metrics": metrics.raw_conditioning_metrics(arrays["predictions_source_actions"],
                        arrays["target_source_recorded_actions"], arrays["observation_state"], arrays["sample_episode_frame"], np,
                        speeds=arrays["anchor_max_joint_speed_rad_s"]),
                        "consistency": intervention_consistency(arrays, real, mode, variant, np)}
                    if mode == "real":
                        item["frozen_real_reproduction"] = reproduction
                    item["paired_against_real"] = metrics.paired_conditioning_metrics(real["predictions_source_actions"],
                        arrays["predictions_source_actions"], arrays["target_source_recorded_actions"], arrays["sample_episode_frame"], np,
                        speeds=arrays["anchor_max_joint_speed_rad_s"])
                    item["residual_variation"] = metrics.model_residual_variation(arrays["predictions_raw_state_residual"], parameter,
                        arrays["sample_episode_frame"], np)
                    evaluated[mode] = item
                    arrays["paired_mae_delta_mode_minus_real_rad"] = np.abs(arrays["predictions_source_actions"].astype(np.float64)
                        - arrays["target_source_recorded_actions"].astype(np.float64)).mean(axis=(1, 2)) - np.abs(real["predictions_source_actions"].astype(np.float64)
                        - real["target_source_recorded_actions"].astype(np.float64)).mean(axis=(1, 2))
                    item["predictions"] = diagnostics._save_arrays(variant_dir / (split + "_" + mode + "_predictions.npz"), arrays, np)
                controls = {}
                for name, fixed in (("model_output_constant", parameter), ("train_label_median_constant", label_parameter),
                                    ("state_hold", np.zeros((16, 28), dtype=np.float32))):
                    prediction = constant.decode_constant_residual(real["observation_state"], fixed, np)
                    arrays = {name: value.copy() for name, value in real.items() if name in
                        ("observation_state", "sample_episode_frame", "target_source_recorded_actions", "anchor_max_joint_speed_rad_s", "anchor_speed_known")}
                    arrays.update(predictions_source_actions=prediction, constant_raw_residual=fixed.copy())
                    controls[name] = {"metrics": metrics.raw_conditioning_metrics(prediction, arrays["target_source_recorded_actions"],
                        arrays["observation_state"], arrays["sample_episode_frame"], np, speeds=arrays["anchor_max_joint_speed_rad_s"]),
                        "paired_against_real": metrics.paired_conditioning_metrics(real["predictions_source_actions"], prediction,
                            arrays["target_source_recorded_actions"], arrays["sample_episode_frame"], np, speeds=arrays["anchor_max_joint_speed_rad_s"]),
                        "predictions": diagnostics._save_arrays(variant_dir / (split + "_" + name + "_predictions.npz"), arrays, np),
                        "optimizer_steps": 0, "compute_matched_to_neural_training": False,
                        "new_closed_form_fit": name == "model_output_constant",
                        "frozen_prior_train_label_fit": name == "train_label_median_constant",
                        "no_parameters_fitted": name == "state_hold"}
                entry["splits"][split] = {"anchor_count": len(real["sample_episode_frame"]), "modes": evaluated, "controls": controls}
            if variant == "state_only_residual":
                entry["mean_state_cross_split_consistency"] = verify_mean_state_cross_split(
                    mean_raw_residual_by_split["train"], mean_raw_residual_by_split["validation"], np)
            del policy
            torch.cuda.empty_cache()
        report.update(status="completed_frozen_state_conditioning_audit", passed=True)
        report["limitations"] = [
            "These fixed checkpoints remain selected using three repeatedly used development-validation trajectories; no independent test or physical success is measured.",
            "Network-state interventions preserve the original raw decoder anchor and RGB; resulting state/anchor or state/RGB combinations may be out of distribution.",
            "Sensitivity, donor equivariance, and lower raw error than collapsed constants describe state conditioning, not unique causal dynamics or robust physical generalization.",
            "The model-output constant pools only 468 stride-30 training outputs without labels; the frozen label median used all 13835 complete train anchors with labels.",
            "Anchor-level paired differences are descriptive, temporally dependent observations; no independent-anchor confidence intervals or significance claims are made."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=10)}
    finally:
        native._gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report
