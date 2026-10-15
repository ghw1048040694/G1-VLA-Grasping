"""Matched residual comparison after audited, train-only coverage expansion.

The previous residual implementation remains frozen.  This module validates
an immutable expanded manifest rather than pretending the original six-episode
statistics or sampled-anchor schedule still apply.  Validation bytes remain
unchanged and only raw-unit errors are compared across normalization fits.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import time
import traceback

from .artifacts import sha256_file
from . import act_diagnostics as diagnostics
from . import grasp_act as native
from . import state_residual as residual
from . import vision_pretraining as vision

BACKEND = "grasp_square_expanded_state_residual_v1"
VARIANTS, TARGET, DECODE = residual.VARIANTS, residual.TARGET, residual.DECODE
ORIGINAL_TRAIN = (20, 21, 75, 76, 125, 126)
CANDIDATES = (12, 38, 62, 73, 90, 98, 129, 147)
VALIDATION = (180, 181, 182)
TRAIN = tuple(sorted(ORIGINAL_TRAIN + CANDIDATES))


class ExpandedResidualError(ValueError):
    """Expanded data or the frozen residual comparison did not pass admission."""


# Pure construction/math helpers retain exactly the reviewed stage7 protocol.
_variant_policy = residual._variant_policy
_training_batch = residual._training_batch
_save_checkpoint = residual._save_checkpoint
select_checkpoint = residual.select_checkpoint
residual_targets = residual.residual_targets
decode_residual = residual.decode_residual


def validate_expanded_residual_settings(config):
    settings = copy.deepcopy(config["expanded_residual"])
    previous = residual.validate_state_residual_settings(config)
    if settings.get("backend") != BACKEND or settings.get("variants") != list(VARIANTS):
        raise ExpandedResidualError("the frozen state-only and scratch-RGB residual arms are required in order")
    for key, expected in (("use_vae", False), ("zero_head_init", True),
                          ("include_step_zero_in_selection", True), ("target", TARGET), ("decode", DECODE)):
        if settings.get(key) != expected or (isinstance(expected, bool) and settings.get(key) is not expected):
            raise ExpandedResidualError("the frozen residual origin and semantics changed: " + key)
    for key in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size",
                "max_train_wall_s", "max_gpu_reserved_bytes"):
        if settings.get(key) != previous[key] or isinstance(settings.get(key), bool):
            raise ExpandedResidualError("the expanded comparison must retain the frozen residual budget " + key)
        if key != "max_train_wall_s" and not isinstance(settings[key], int):
            raise ExpandedResidualError("the expanded integer budget changed its type: " + key)
    for key in ("data_manifest_path", "reference_residual_report_path", "admission_report_path", "output_dir"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ExpandedResidualError(key + " must name an explicit project path")
        native._resolve(config, settings[key])
    for key in ("data_manifest_sha256", "reference_residual_report_sha256"):
        digest = settings.get(key)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ExpandedResidualError(key + " must contain a full lowercase SHA256")
    return settings


def _stage(config, run_dir, name):
    settings = validate_expanded_residual_settings(config)
    directory = Path(run_dir)
    path = directory / "reports" / (name + ".json")
    if not directory.is_dir() or path.exists():
        raise ExpandedResidualError("the stage requires an existing run and a new immutable report")
    path.parent.mkdir(exist_ok=True)
    return settings, path, {"schema_version": 1, "status": "failed", "passed": False, "settings": settings,
                            "training_started": False, "optimizer_steps": 0, "test_payloads_opened": False,
                            "test_used_for_selection": False, "physical_success_evaluated": False,
                            "project_adapter_validated": False, "closed_loop_control": False}


def _stats(statistics, np):
    """Read the expanded train-only normalization without changing old helpers."""
    ids = statistics.get("training_episode_ids")
    if (not isinstance(ids, list) or len(ids) != len(TRAIN) or set(ids) != set(TRAIN)
            or statistics.get("validation_or_test_used") is not False
            or statistics.get("statistics_include_held_out") is not False
            or statistics.get("std_minimum") != 0.001):
        raise ExpandedResidualError("normalization must use exactly the fourteen admitted train episodes")
    result = {}
    for key, shape in (("observation.state", (28,)), ("action", (28,)), (native.CAMERA, (3, 1, 1))):
        values = statistics.get("features", {}).get(key)
        if not isinstance(values, dict):
            raise ExpandedResidualError("expanded train statistics lack " + key)
        mean, std = np.asarray(values["mean"], dtype=np.float32), np.asarray(values["std"], dtype=np.float32)
        if (mean.shape != shape or std.shape != shape or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or np.any(std < np.float32(0.001))):
            raise ExpandedResidualError("expanded train statistics have invalid FP32 shape or scale")
        result[key] = mean, std
    return result


def _verify_reference(config, declared):
    path = native._resolve(config, declared["reference_residual_report_path"])
    if sha256_file(path) != declared["reference_residual_report_sha256"]:
        raise ExpandedResidualError("the frozen stage7 residual comparison SHA256 changed")
    reference = native._read(path)
    if (reference.get("passed") is not True
            or reference.get("status") != "completed_train_validation_state_residual_comparison"
            or reference.get("test_payloads_opened") is not False or reference.get("test_used_for_selection") is not False):
        raise ExpandedResidualError("the reference must be the complete train/validation-only residual comparison")
    for key in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size",
                "max_train_wall_s", "max_gpu_reserved_bytes", "use_vae", "zero_head_init",
                "include_step_zero_in_selection", "target", "decode", "variants"):
        if reference.get("settings", {}).get(key) != declared[key]:
            raise ExpandedResidualError("the frozen residual comparison differs in " + key)
    bindings = reference.get("bindings", {})
    for key, module in (("original_training_module_sha256", native), ("diagnostic_module_sha256", diagnostics),
                        ("prior_no_vae_builder_sha256", vision), ("state_residual_module_sha256", residual)):
        if bindings.get(key) != sha256_file(module.__file__):
            raise ExpandedResidualError("the frozen original implementation changed: " + key)
    controls = {}
    for name in VARIANTS:
        variant = reference.get("variants", {}).get(name, {})
        if (variant.get("status") != "completed_matched_budget_state_residual_training"
                or variant.get("optimizer_steps") != declared["max_steps"]
                or len(variant.get("steps", [])) != declared["max_steps"]
                or [item.get("step") for item in variant.get("validation", [])]
                   != list(range(0, declared["max_steps"] + 1, declared["eval_every_steps"]))):
            raise ExpandedResidualError("the frozen residual controls must have their complete budgets and selection points")
        checkpoint = variant["checkpoint"]
        directory = Path(checkpoint["path"])
        if sha256_file(directory / "provenance.json") != checkpoint["provenance_sha256"]:
            raise ExpandedResidualError("a frozen residual checkpoint provenance changed")
        native.verify_grasp_checkpoint(directory, bindings)
        prior_config = native._read(directory / "experiment_config.json")
        for key in ("seed", "batch_size", "learning_rate"):
            if config["train"][key] != prior_config["train"][key]:
                raise ExpandedResidualError("the expanded trial changed original training " + key)
        for key, value in config["grasp_act"].items():
            if prior_config["grasp_act"].get(key) != value:
                raise ExpandedResidualError("the expanded trial changed the frozen ACT construction setting " + key)
        controls[name] = {"selected_step": variant["selected_step"], "selection_summary": variant["selection_summary"],
                          "selected_train_original_six": variant["selected_prior_inference"]["train"]["metrics"],
                          "selected_validation": variant["selected_prior_inference"]["validation"]["metrics"],
                          "optimizer_steps": variant["optimizer_steps"], "reference_only_no_new_training": True}
    return reference, {"path": str(path), "sha256": sha256_file(path), "residual_controls": controls,
                       "old_statistics_retained_for_reference_scores": True,
                       "raw_validation_metric_comparison_only": True, "test_payloads_or_metrics_read": False}


def load_expanded_data(config, declared, reference):
    """Load only audited expanded train and the byte-identical frozen validation."""
    import numpy as np
    from .grasp_data import GraspSquareDataset, _verify_artifact
    from . import train_expansion

    path = native._resolve(config, declared["data_manifest_path"])
    build = native._resolve(config, config["train_expansion"]["build_dir"])
    if (path != build / "manifest.json" or not path.is_file()
            or sha256_file(path) != declared["data_manifest_sha256"]):
        raise ExpandedResidualError("the expanded manifest must match the final admitted full SHA256 and declared build")
    manifest = train_expansion.load_expanded_manifest(build, verify_files=True)
    audit_binding = _verify_expansion_audit(config, manifest, train_expansion)
    split_ids = manifest.get("split_episode_ids", {})
    if (set(split_ids) != {"train", "validation"} or split_ids.get("validation") != list(VALIDATION)
            or len(split_ids.get("train", [])) != len(TRAIN) or set(split_ids["train"]) != set(TRAIN)
            or manifest.get("camera_key") != native.CAMERA or manifest.get("chunk_steps") != config["grasp_act"]["chunk_size"]
            or manifest.get("image_shape_hwc") != [240, 320, 3]):
        raise ExpandedResidualError("the expanded manifest changed the admitted fourteen train or frozen three validation episodes")
    prior_binding = reference["bindings"]
    for key, expected in (("original_manifest_sha256", prior_binding["manifest_sha256"]),
                          ("original_statistics_sha256", prior_binding["statistics_sha256"]),
                          ("prior_residual_report_sha256", declared["reference_residual_report_sha256"])):
        if manifest.get(key) != expected:
            raise ExpandedResidualError("expanded build differs from its frozen original reference: " + key)
    old_path = native._resolve(config, config["grasp_act"]["dataset_manifest_path"])
    if sha256_file(old_path) != prior_binding["manifest_sha256"]:
        raise ExpandedResidualError("the original native manifest changed")
    original = native._read(old_path)
    old_items = {item["episode_id"]: item for item in original["episodes"] if item["split"] in ("train", "validation")}
    files, preserved = {}, {}
    for item in manifest["episodes"]:
        if item["split"] not in ("train", "validation") or item["episode_id"] not in split_ids[item["split"]]:
            raise ExpandedResidualError("expanded data contains an undeclared split or episode")
        for key in ("arrays", "rgb", "decoded_receipt"):
            actual_path = _verify_artifact(build, item[key])
            files[item[key]["relative_path"]] = item[key]["sha256"]
            if item["episode_id"] in old_items:
                prior = old_items[item["episode_id"]][key]
                if any(item[key][field] != prior[field] for field in ("sha256", "size_bytes")):
                    raise ExpandedResidualError("the expanded build modified an original raw train/validation payload")
                if item["split"] == "validation":
                    preserved[str(item["episode_id"]) + ":" + key] = {"sha256": sha256_file(actual_path),
                                                                           "size_bytes": actual_path.stat().st_size}
        if item["episode_id"] in old_items:
            for key in ("split", "row_count", "sample_count", "all_camera_video_relative_paths"):
                if item.get(key) != old_items[item["episode_id"]].get(key):
                    raise ExpandedResidualError("an original episode identity, row count or source grouping changed")
    statistics_path = _verify_artifact(build, manifest["statistics"])
    statistics = native._read(statistics_path)
    _stats(statistics, np)
    if statistics["training_episode_ids"] != [item["episode_id"] for item in manifest["episodes"] if item["split"] == "train"]:
        raise ExpandedResidualError("expanded normalization differs from the actual declared train episode order")
    datasets = {split: GraspSquareDataset(build, split, manifest=manifest, verify=False) for split in ("train", "validation")}
    if len(datasets["validation"]) != 2488 or len(datasets["train"]) != 13835:
        raise ExpandedResidualError("expanded/frozen complete-anchor counts differ from the audited source")
    binding = {"experiment_config_sha256": native._digest(config), "manifest_sha256": sha256_file(path),
               "statistics_sha256": sha256_file(statistics_path), "selected_artifact_sha256": files,
               "original_manifest_sha256": prior_binding["manifest_sha256"],
               "original_statistics_sha256": prior_binding["statistics_sha256"],
               "reference_residual_report_sha256": declared["reference_residual_report_sha256"],
               "expanded_data_module_sha256": sha256_file(train_expansion.__file__),
               "expanded_residual_module_sha256": sha256_file(__file__),
               "original_training_module_sha256": sha256_file(native.__file__),
               "diagnostic_module_sha256": sha256_file(diagnostics.__file__),
               "prior_no_vae_builder_sha256": sha256_file(vision.__file__),
               "state_residual_module_sha256": sha256_file(residual.__file__),
               "candidate_admission": audit_binding,
               "verified_payload_splits": ["train", "validation"], "test_payloads_opened": False,
               "validation_payloads_byte_identical_to_stage7": preserved}
    return manifest, statistics, binding, datasets


def _verify_expansion_audit(config, manifest, module):
    """Reopen the actual candidate/session admission rather than trusting flags."""
    settings = module.validate_train_expansion_settings(config)
    _, _, _, binding = module._base(config, settings)
    if manifest.get("settings") != settings or manifest.get("bindings") != binding:
        raise ExpandedResidualError("the expanded manifest does not bind the current declared candidate/session audit")
    audit_path = native._resolve(config, settings["audit_report_path"])
    audit_sha = sha256_file(audit_path)
    audit = native._read(audit_path)
    if (str(audit_path) != manifest.get("candidate_audit_path")
            or audit_sha != manifest.get("candidate_audit_sha256")
            or audit.get("passed") is not True
            or audit.get("status") != "passed_complete_candidate_protocol_same_source_coverage"
            or audit.get("settings") != settings or audit.get("bindings") != binding
            or audit.get("candidate_episode_ids") != list(CANDIDATES)
            or audit.get("test_payloads_opened") is not False or audit.get("test_metrics_read") is not False
            or audit.get("independent_session_data_admitted") is not False
            or audit.get("admission_class") != "same_source_coverage_only"
            or audit.get("all_candidates_required_no_performance_filtering") is not True
            or audit.get("all_camera_direct_frozen_file_disjoint") is not True
            or audit.get("source_label_protocol_preserved") is not True
            or audit.get("exact_state_action_duplicate_trajectories_found") is not False
            or audit.get("capture_session_mapping_available") is not False
            or audit.get("scene_operator_independence_verified") is not False):
        raise ExpandedResidualError("the full candidate audit no longer satisfies the same-source-only training conditions")
    candidate_root = native._resolve(config, settings["candidate_build_dir"])
    candidate_path = candidate_root / "candidate_manifest.json"
    candidate_sha = sha256_file(candidate_path)
    if ((candidate_root / "failed.json").exists() or candidate_sha != manifest.get("candidate_manifest_sha256")
            or candidate_sha != audit.get("candidate_manifest_sha256")
            or str(candidate_path) != audit.get("candidate_manifest_path")):
        raise ExpandedResidualError("the admitted candidate manifest changed or has a partial-failure marker")
    candidates = native._read(candidate_path)
    if (candidates.get("passed") is not True or candidates.get("bindings") != binding
            or candidates.get("episode_ids") != list(CANDIDATES)
            or candidates.get("independent_session_data_admitted") is not False
            or candidates.get("admission_class") != "same_source_coverage_only"):
        raise ExpandedResidualError("the candidate manifest differs from its audited origin")
    expanded_items = {item["episode_id"]: item for item in manifest["episodes"]}
    if [item.get("episode_id") for item in candidates.get("episodes", [])] != list(CANDIDATES):
        raise ExpandedResidualError("the complete eight candidate records are required")
    for item in candidates["episodes"]:
        if any(expanded_items[item["episode_id"]][key] != item[key] for key in ("arrays", "rgb", "decoded_receipt")):
            raise ExpandedResidualError("the expanded candidate bytes differ from the original audited payloads")
    evidence_path = native._resolve(config, settings["session_evidence_path"])
    evidence_sha = sha256_file(evidence_path)
    if evidence_sha != settings["session_evidence_sha256"] or evidence_sha != binding["session_evidence_sha256"]:
        raise ExpandedResidualError("the current session evidence changed after candidate admission")
    return {"audit_report_path": str(audit_path), "audit_report_sha256": audit_sha,
            "candidate_manifest_path": str(candidate_path), "candidate_manifest_sha256": candidate_sha,
            "session_evidence_path": str(evidence_path), "session_evidence_sha256": evidence_sha,
            "all_candidates_required_no_performance_filtering": True,
            "actual_audit_and_candidate_files_reopened": True, "session_independence_verified": False}


def _prepare(config, declared, report):
    reference, report["frozen_residual_reference"] = _verify_reference(config, declared)
    manifest, statistics, binding, datasets = load_expanded_data(config, declared, reference)
    report["bindings"] = binding
    settings = diagnostics._settings(config, declared)
    settings["motion_thresholds"] = {key: config["native_diagnostics"][key]
                                      for key in ("stationary_speed_rad_s", "moving_speed_rad_s")}
    np, torch, device, blueprint_policy = native._runtime(config, settings, report)
    if any(report[name] != reference.get(name) for name in ("environment", "runtime_sources", "architecture")):
        raise ExpandedResidualError("the expanded comparison changed the frozen runtime or ACT architecture")
    rng = {"cpu": torch.get_rng_state().clone(), "cuda": [value.clone() for value in torch.cuda.get_rng_state_all()]}
    blueprint = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in blueprint_policy.state_dict().items()}
    report["full_rgb_initial_blueprint_sha256"] = diagnostics._tensor_digest(blueprint)
    report["post_rgb_blueprint_rng_sha256"] = {"cpu": hashlib.sha256(rng["cpu"].numpy().tobytes()).hexdigest(),
                                              "cuda": [hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest() for value in rng["cuda"]]}
    for key in ("full_rgb_initial_blueprint_sha256", "post_rgb_blueprint_rng_sha256"):
        if report[key] != reference.get(key):
            raise ExpandedResidualError("the common initialization or dropout RNG changed: " + key)
    del blueprint_policy
    torch.cuda.empty_cache()
    schedule, identities, schedule_report = diagnostics.anchor_schedule(datasets["train"], settings, config["train"]["batch_size"], declared["seed"], np)
    report["anchor_schedule"] = schedule_report
    report["schedule_changed_because_train_membership_changed"] = True
    report["construction_blueprint_initialization"] = report.pop("model_initialization")
    report["model_initialization"] = {"formal_use_vae": False, "public_task_policy_loaded": False,
                                      "public_visual_backbone_loaded": False, "all_formal_heads_zero_initialized": True,
                                      "previous_checkpoint_weights_resumed": False}
    report["architecture_scope"] = "Original untrained RGB construction blueprint; fresh no-VAE residual architectures are saved per variant."
    report["data_expansion_treatment"] = {"original_train_episode_ids": list(ORIGINAL_TRAIN),
        "added_train_episode_ids": list(CANDIDATES), "frozen_validation_episode_ids": list(VALIDATION),
        "train_membership_changed": True, "train_only_statistics_recomputed": True,
        "residual_scale_is_expanded_train_action_std": True, "raw_validation_bytes_unchanged": True,
        "session_independence_verified": False, "data_scope": "audited_same_source_coverage_expansion",
        "cross_trial_comparison_metric": "raw_action_MAE_rad_on_unchanged_validation",
        "normalized_losses_across_statistics_are_not_directly_comparable": True}
    report["target_protocol"] = {"target": TARGET, "decode": DECODE, "anchor": "direct_raw_observation_state_at_sample",
        "scale": "expanded_train_only_action_std", "action_mean_subtracted": False, "residual_statistics_fitted": False,
        "source_labels_modified": False, "normalized_state_inverted_for_anchor": False, "network_inference_receives_labels": False}
    return settings, manifest, statistics, datasets, binding, np, torch, device, blueprint, rng, schedule, identities

def _evaluate(policy, dataset, statistics, settings, np, torch, device, *, variant, deadline=None, require_zero=False):
    indices = dataset.get_eval_indices(settings["eval_stride"])
    ids = np.asarray([dataset.samples[index] for index in indices], dtype=np.int64)
    stats = _stats(statistics, np)
    predictions, residuals, targets, states = [], [], [], []
    policy.eval()
    for start in range(0, len(indices), settings["eval_batch_size"]):
        if deadline is not None and time.perf_counter() >= deadline:
            raise ExpandedResidualError("the unchanged wall budget expired before complete validation")
        batch, samples = native._batch(dataset, indices[start:start + settings["eval_batch_size"]], stats, np, torch, device, include_actions=False)
        if variant == "state_only_residual":
            del batch[native.CAMERA]
        if "action" in batch or "action_is_pad" in batch:
            raise ExpandedResidualError("residual inference must not receive labels or action masks")
        with torch.inference_mode():
            residual = policy.predict_action_chunk(batch).cpu().numpy()
        state = np.stack([sample["observation.state"] for sample in samples])
        if residual.shape != (len(samples), settings["chunk_size"], 28):
            raise ExpandedResidualError("the residual network must predict one complete original action chunk")
        prediction = decode_residual(residual, state, stats["action"][1], np)
        if require_zero:
            held = np.repeat(state[:, None, :], settings["chunk_size"], axis=1)
            if np.any(residual != 0) or prediction.tobytes() != held.tobytes():
                raise ExpandedResidualError("the zero-head actual network did not decode bit-for-bit to raw state-hold")
        residuals.extend(residual)
        predictions.extend(prediction)
        targets.extend(sample["action"] for sample in samples)
        states.extend(state)
        native._check_gpu(torch, settings)
    arrays = {"predictions_source_actions": np.asarray(predictions, dtype=np.float32),
              "predictions_scaled_state_residual": np.asarray(residuals, dtype=np.float32),
              "target_source_recorded_actions": np.asarray(targets, dtype=np.float32),
              "observation_state": np.asarray(states, dtype=np.float32), "sample_episode_frame": ids,
              "action_std": stats["action"][1]}
    metrics = diagnostics.diagnostic_metrics(arrays["predictions_source_actions"], arrays["target_source_recorded_actions"],
                                               arrays["observation_state"], stats["action"][0], stats["action"][1], ids, np)
    metrics.update(inference_latent="zero_latent_no_vae_no_labels", evaluated_split=dataset.split,
                   eval_stride_source_frames=settings["eval_stride"], state_only=variant == "state_only_residual",
                   camera_input_present=variant != "state_only_residual", target_protocol=TARGET,
                   raw_anchor_reconstructed_from_normalized_state=False, zero_origin_verified=require_zero)
    if require_zero and metrics["policy"] != metrics["state_hold"]:
        raise ExpandedResidualError("measured zero-head policy and state-hold metrics differ")
    if "motion_thresholds" in settings:
        metrics["motion_strata"] = _motion_strata(dataset, arrays, statistics,
                                                  {"native_diagnostics": settings["motion_thresholds"]}, np)
    if dataset.split == "train":
        metrics["train_cohorts"] = _train_cohorts(arrays, stats, np)
    return metrics, arrays


def _train_cohorts(arrays, stats, np):
    """Describe the old-six and added-eight train subsets without selecting on them."""
    ids = arrays["sample_episode_frame"]
    cohorts = {}
    for name, episodes in (("original_six", ORIGINAL_TRAIN), ("added_eight", CANDIDATES)):
        mask = np.isin(ids[:, 0], episodes)
        count = int(mask.sum())
        if not count:
            raise ExpandedResidualError("both predeclared expanded train reporting cohorts must be represented")
        cohorts[name] = {"episode_ids": list(episodes), "anchor_count": count,
            "used_for_checkpoint_selection": False,
            "metrics": diagnostics.diagnostic_metrics(arrays["predictions_source_actions"][mask],
                arrays["target_source_recorded_actions"][mask], arrays["observation_state"][mask],
                stats["action"][0], stats["action"][1], ids[mask], np)}
    if sum(item["anchor_count"] for item in cohorts.values()) != len(ids):
        raise ExpandedResidualError("train cohorts must partition the full evaluation anchor denominator")
    return cohorts


def _motion_strata(dataset, arrays, statistics, config, np):
    """Analysis-only anchor speed uses the previous raw row, never a model input."""
    ids = arrays["sample_episode_frame"]
    known = ids[:, 1] > 0
    speeds = np.full(len(ids), np.nan, dtype=np.float64)
    for offset, (episode, frame) in enumerate(ids):
        if frame > 0:
            # _evaluate already loaded exactly these selected train/val episodes.
            # No call to a test dataset, future row, action or vision is needed.
            states = dataset._arrays[int(episode)][0]
            speeds[offset] = float(np.max(np.abs(states[int(frame)].astype(np.float64)
                                                - states[int(frame) - 1].astype(np.float64))) * 30.0)
    thresholds = config["native_diagnostics"]
    stationary, moving = thresholds["stationary_speed_rad_s"], thresholds["moving_speed_rad_s"]
    masks = {"stationary": known & (speeds <= stationary), "moving": known & (speeds >= moving),
             "intermediate": known & (speeds > stationary) & (speeds < moving), "unknown_first_row": ~known}
    arrays["anchor_max_joint_speed_rad_s"] = speeds
    arrays["anchor_speed_known"] = known
    action_mean, action_std = _stats(statistics, np)["action"]
    result = {"speed_definition": "30*max_joint_abs(raw_q[t]-raw_q[t-1]); t=0 unknown",
              "stationary_max_speed_rad_s": stationary, "moving_min_speed_rad_s": moving,
              "uses_only_current_and_previous_observed_state": True, "used_as_model_input": False,
              "partition_anchor_count": len(ids), "strata": {}}
    for name, mask in masks.items():
        result["strata"][name] = {"anchor_count": int(mask.sum()), "metrics": None}
        if mask.any():
            result["strata"][name]["metrics"] = diagnostics.diagnostic_metrics(
                arrays["predictions_source_actions"][mask], arrays["target_source_recorded_actions"][mask],
                arrays["observation_state"][mask], action_mean, action_std, ids[mask], np)
    if sum(item["anchor_count"] for item in result["strata"].values()) != len(ids):
        raise ExpandedResidualError("motion strata must partition the same evaluated anchor denominator exactly")
    return result

def _roundtrip_audit(datasets, statistics, np):
    std = _stats(statistics, np)["action"][1]
    result = {}
    for split, dataset in datasets.items():
        maxima, count = [], 0
        for episode, metadata in dataset.episodes.items():
            # One accessor call per episode admits the original FP32 arrays;
            # the remaining audit never materializes every RGB frame again.
            first = next(index for index, identity in enumerate(dataset.samples) if identity[0] == episode)
            dataset[first]
            source_state, source_action = dataset._arrays[episode]
            for offset in range(0, metadata["sample_count"], 128):
                stop = min(offset + 128, metadata["sample_count"])
                states = source_state[offset:stop]
                actions = np.stack([source_action[frame:frame + dataset.chunk_steps] for frame in range(offset, stop)])
                encoded = residual_targets(actions, states, std, np)
                decoded = decode_residual(encoded, states, std, np)
                maximum = float(np.max(np.abs(decoded.astype(np.float64) - actions.astype(np.float64))))
                if maximum > 1e-6:
                    raise ExpandedResidualError("the declared scaled residual roundtrip exceeds 1e-6 rad FP32 tolerance")
                zero = decode_residual(np.zeros_like(actions), states, std, np)
                if zero.tobytes() != np.broadcast_to(states[:, None, :], actions.shape).copy().tobytes():
                    raise ExpandedResidualError("raw state-hold bytes were not preserved for every admitted anchor")
                maxima.append(maximum)
                count += stop - offset
        result[split] = {"complete_anchor_count": count, "max_encode_decode_error_rad": max(maxima),
                         "all_anchor_zero_residual_identity_bitwise": True, "roundtrip_tolerance_rad": 1e-6}
    return result

def _probe_variant(config, settings, variant, blueprint, datasets, statistics, schedule, np, torch, device, rng, receipt):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    receipt.update(passed=False, optimizer_created=False, probe_optimizer_steps=0,
                   temporary_parameters_discarded=False, formal_training_steps=0)
    policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng)
    stats = _stats(statistics, np)
    batch, _ = _training_batch(datasets["train"], schedule[0], stats, variant, np, torch, device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
    receipt["optimizer_created"] = True
    policy.train()
    loss, components = policy(batch)
    if not bool(torch.isfinite(loss)) or "kld_loss" in components:
        raise ExpandedResidualError("the residual memory probe has nonfinite loss or an active VAE")
    loss.backward()
    gradients = diagnostics._gradient_report(policy, torch)
    if not gradients["finite"]:
        raise ExpandedResidualError("the residual memory probe gradients are missing or nonfinite")
    optimizer.step()
    receipt["probe_optimizer_steps"] = 1
    if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
        raise ExpandedResidualError("the temporary residual optimizer produced nonfinite parameters")
    del batch, loss
    # One complete configured validation batch exercises both image inference
    # and direct-anchor decoding at the same shapes used in formal validation.
    indices = datasets["validation"].get_eval_indices(settings["eval_stride"])[:settings["eval_batch_size"]]
    if len(indices) != settings["eval_batch_size"]:
        raise ExpandedResidualError("the full validation memory-probe batch is unavailable")
    batch, samples = native._batch(datasets["validation"], indices, stats, np, torch, device, include_actions=False)
    if variant == "state_only_residual":
        del batch[native.CAMERA]
    policy.eval()
    with torch.inference_mode():
        residual = policy.predict_action_chunk(batch).cpu().numpy()
    decoded = decode_residual(residual, np.stack([sample["observation.state"] for sample in samples]), stats["action"][1], np)
    torch.cuda.synchronize()
    if decoded.shape != (settings["eval_batch_size"], settings["chunk_size"], 28):
        raise ExpandedResidualError("the temporary residual validation output shape changed")
    native._check_gpu(torch, settings)
    receipt.update(passed=True, finite_gradients=True, temporary_parameters_discarded=True,
                   full_validation_batch_passed=True, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), initialization=initialization,
                   probe_loss_components=components, first_backward_gradients=gradients,
                   zero_head_first_backward_blocks_upstream_data_gradient_expected=True,
                   adamw_weight_decay_can_still_update_zero_gradient_parameters=True)
    del optimizer, policy, residual, decoded, batch
    torch.cuda.empty_cache()

def run_expanded_residual_admission(config, run_dir):
    """Audit all raw anchors, actual step-zero inference, and discarded probes."""
    declared, path, report = _stage(config, run_dir, "expanded_residual_admission")
    started, torch = time.perf_counter(), None
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise ExpandedResidualError("discarded residual optimizer probes require the unified conditional training admission")
        prepared = _prepare(config, declared, report)
        settings, _, statistics, datasets, _, np, torch, device, blueprint, rng, schedule, _ = prepared
        report.update(optimizer_created=False, formal_optimizer_created=False, variants={})
        report["roundtrip_audit"] = _roundtrip_audit(datasets, statistics, np)
        for variant in VARIANTS:
            policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng)
            entry = {"initialization": initialization, "step_zero": {}}
            report["variants"][variant] = entry
            for split in ("train", "validation"):
                metrics, arrays = _evaluate(policy, datasets[split], statistics, settings, np, torch, device,
                                             variant=variant, require_zero=True)
                entry["step_zero"][split] = {"metrics": metrics, "predictions": diagnostics._save_arrays(
                    Path(run_dir) / (variant + "_" + split + "_step_zero.npz"), arrays, np)}
            del policy
            torch.cuda.empty_cache()
            entry["memory_probe"] = {}
            _probe_variant(config, settings, variant, blueprint, datasets, statistics, schedule,
                           np, torch, device, rng, entry["memory_probe"])
            if entry["memory_probe"]["initialization"] != initialization:
                raise ExpandedResidualError("the temporary memory model differs from the measured zero-origin model")
        report.update(status="passed_expanded_exact_zero_state_hold_admission", passed=True,
                      all_anchor_zero_residual_identity_bitwise=True,
                      actual_network_step_zero_train_validation_equals_state_hold=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["optimizer_created"] = any(entry.get("memory_probe", {}).get("optimizer_created", False)
                                           for entry in report.get("variants", {}).values())
        report["discarded_probe_optimizer_steps"] = sum(entry.get("memory_probe", {}).get("probe_optimizer_steps", 0)
                                                        for entry in report.get("variants", {}).values())
        report["optimizer_steps_including_discarded_probes"] = report["optimizer_steps"] + report["discarded_probe_optimizer_steps"]
        native._gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report

def _verify_admission(config, declared, binding, report, datasets):
    path = native._resolve(config, declared["admission_report_path"])
    admission = native._read(path)
    if (admission.get("passed") is not True
            or admission.get("status") != "passed_expanded_exact_zero_state_hold_admission"
            or admission.get("bindings") != binding or admission.get("settings") != declared
            or admission.get("formal_optimizer_created") is not False or admission.get("optimizer_steps") != 0
            or admission.get("discarded_probe_optimizer_steps") != len(VARIANTS)
            or admission.get("optimizer_steps_including_discarded_probes") != len(VARIANTS)
            or admission.get("test_payloads_opened") is not False or admission.get("test_used_for_selection") is not False
            or admission.get("all_anchor_zero_residual_identity_bitwise") is not True
            or admission.get("actual_network_step_zero_train_validation_equals_state_hold") is not True):
        raise ExpandedResidualError("formal residual training requires the exact current config/code/data zero-origin admission")
    for name in ("environment", "runtime_sources", "architecture", "full_rgb_initial_blueprint_sha256", "post_rgb_blueprint_rng_sha256", "anchor_schedule"):
        if report[name] != admission.get(name):
            raise ExpandedResidualError("the current residual runtime differs from its zero-origin admission " + name)
    for split in ("train", "validation"):
        expected = len(datasets[split])
        audit = admission.get("roundtrip_audit", {}).get(split, {})
        error = audit.get("max_encode_decode_error_rad")
        if (audit.get("complete_anchor_count") != expected or audit.get("all_anchor_zero_residual_identity_bitwise") is not True
                or isinstance(error, bool) or not isinstance(error, (int, float))
                or not math.isfinite(error) or not 0 <= error <= 1e-6):
            raise ExpandedResidualError("admission must audit every admitted expanded-train/frozen-validation complete raw anchor")
    for variant in VARIANTS:
        entry = admission.get("variants", {}).get(variant, {})
        if entry.get("initialization", {}).get("residual_head", {}).get("zero_final_action_head_verified") is not True:
            raise ExpandedResidualError("the admitted residual output head is not exactly zero")
        probe = entry.get("memory_probe", {})
        reserved = probe.get("peak_reserved_bytes")
        if (probe.get("passed") is not True or probe.get("probe_optimizer_steps") != 1
                or probe.get("formal_training_steps") != 0 or probe.get("temporary_parameters_discarded") is not True
                or probe.get("finite_gradients") is not True or probe.get("full_validation_batch_passed") is not True
                or probe.get("initialization") != entry["initialization"]
                or isinstance(reserved, bool) or not isinstance(reserved, int)
                or not 0 < reserved <= declared["max_gpu_reserved_bytes"]):
            raise ExpandedResidualError("each residual arm requires its exact discarded optimizer/evaluation memory probe")
        for split in ("train", "validation"):
            item = entry.get("step_zero", {}).get(split, {})
            artifact = item.get("predictions", {})
            if sha256_file(artifact.get("path", "")) != artifact.get("sha256"):
                raise ExpandedResidualError("an admitted step-zero prediction artifact changed")
            if (item.get("metrics", {}).get("zero_origin_verified") is not True
                    or item["metrics"]["policy"] != item["metrics"]["state_hold"]):
                raise ExpandedResidualError("the admitted step-zero policy is not the exact state-hold control")
    report["zero_origin_admission"] = {"path": str(path), "sha256": sha256_file(path), "same_full_binding_verified": True,
                                       "formal_optimizer_steps": 0, "discarded_probe_optimizer_steps": len(VARIANTS),
                                       "prediction_files_rehashed": True}
    return admission

def train_expanded_residual(config, run_dir):
    """Run the fixed paired residual budgets; explicitly retain hold as step zero."""
    declared, path, report = _stage(config, run_dir, "expanded_residual_training")
    started, torch = time.perf_counter(), None
    report.update(variants={}, matched_anchor_schedule=True, optimizer_steps_including_discarded_probes=0)
    try:
        if config["train"].get("enabled") is not True or config["train"].get("authorization") != "conditional":
            raise ExpandedResidualError("the unified conditional training admission must be enabled")
        output = native._resolve(config, declared["output_dir"])
        if output.exists():
            raise ExpandedResidualError("residual variants never overwrite existing output or resume prior optimizer state")
        prepared = _prepare(config, declared, report)
        settings, manifest, statistics, datasets, binding, np, torch, device, blueprint, rng, schedule, identities = prepared
        admission = _verify_admission(config, declared, binding, report, datasets)
        report["prior_admission_probe_optimizer_steps"] = admission["discarded_probe_optimizer_steps"]
        report["discarded_probes_performed_this_run"] = 0
        output.mkdir(parents=True, exist_ok=False)
        report["anchor_schedule_artifact"] = diagnostics._save_arrays(output / "training_anchor_schedule.npz",
            {"sample_episode_frame": identities, "dataset_sample_indices": schedule}, np)
        report["checkpoint_selection"] = {"include_measured_step_zero": True, "criterion": "validation.policy.all_28.mae",
                                           "best_trained_and_last_step_reported_separately": True, "test_used": False}
        stats = _stats(statistics, np)
        for variant in VARIANTS:
            entry = {"status": "started", "training_started": False, "optimizer_steps": 0, "steps": [], "validation": [],
                     "selection_split": "validation", "test_payloads_opened": False, "test_used_for_selection": False}
            report["variants"][variant] = entry
            (output / variant).mkdir(exist_ok=False)
            variant_started = time.perf_counter()
            deadline = variant_started + settings["max_train_wall_s"]
            entry["prior_admission_memory_probe"] = copy.deepcopy(admission["variants"][variant]["memory_probe"])
            entry["discarded_probes_performed_this_run"] = 0
            torch.cuda.reset_peak_memory_stats()
            policy, initialization = _variant_policy(config, settings, variant, blueprint, np, torch, device, rng)
            entry["initialization"] = initialization
            if (initialization != entry["prior_admission_memory_probe"]["initialization"]
                    or initialization != admission["variants"][variant]["initialization"]):
                raise ExpandedResidualError("the formal residual initialization differs from its probe or zero-origin admission")
            metrics, _ = _evaluate(policy, datasets["validation"], statistics, settings, np, torch, device,
                                    variant=variant, deadline=deadline, require_zero=True)
            if metrics != admission["variants"][variant]["step_zero"]["validation"]["metrics"]:
                raise ExpandedResidualError("formal step-zero validation differs from the admitted exact state-hold identity")
            entry["validation"].append({"step": 0, "metrics": metrics})
            best_mae, best_metrics, entry["selected_step"] = metrics["policy"]["all_28"]["mae"], metrics, 0
            best_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
            trained_state, trained_mae, trained_metrics = None, math.inf, None
            optimizer = torch.optim.AdamW(policy.parameters(), lr=config["train"]["learning_rate"], weight_decay=settings["weight_decay"])
            # Step-zero evaluation is label-free and consumes no training RNG.
            # Reapply the declared post-blueprint state before the first formal
            # dropout draw, after probe rebuilding and origin verification.
            torch.set_rng_state(rng["cpu"])
            torch.cuda.set_rng_state_all(rng["cuda"])
            for offset, indices in enumerate(schedule):
                if time.perf_counter() >= deadline:
                    raise ExpandedResidualError("a residual arm exhausted its unchanged wall budget before all 2000 steps")
                step = offset + 1
                policy.train()
                batch, _ = _training_batch(datasets["train"], indices, stats, variant, np, torch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, components = policy(batch)
                if not bool(torch.isfinite(loss)) or "kld_loss" in components:
                    raise ExpandedResidualError("residual training loss is nonfinite or includes KL")
                loss.backward()
                gradients = diagnostics._gradient_report(policy, torch)
                if not gradients["finite"]:
                    raise ExpandedResidualError("residual training gradients are missing or nonfinite")
                optimizer.step()
                report["training_started"], entry["training_started"] = True, True
                report["optimizer_steps"] += 1
                entry["optimizer_steps"] = step
                torch.cuda.synchronize()
                if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
                    raise ExpandedResidualError("a formal residual optimizer produced nonfinite parameters")
                native._check_gpu(torch, settings)
                entry["steps"].append({"step": step, "loss": float(loss.detach().cpu()), "loss_components": components,
                                       "gradients": gradients, "elapsed_s": time.perf_counter() - variant_started})
                del batch, loss
                if step % settings["eval_every_steps"] == 0:
                    metrics, validation_arrays = _evaluate(policy, datasets["validation"], statistics, settings, np, torch, device,
                                                           variant=variant, deadline=deadline)
                    entry["validation"].append({"step": step, "metrics": metrics})
                    mae = metrics["policy"]["all_28"]["mae"]
                    if mae < best_mae:
                        best_mae, best_metrics, entry["selected_step"] = mae, metrics, step
                        best_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
                    if mae < trained_mae:
                        trained_mae, trained_metrics, entry["best_trained_step"] = mae, metrics, step
                        trained_state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in policy.state_dict().items()}
                    if step == settings["max_steps"]:
                        entry["last_step_validation_predictions"] = diagnostics._save_arrays(
                            output / variant / "validation_last_step_predictions.npz", validation_arrays, np)
            selection = select_checkpoint(entry["validation"])
            if selection["selected_step"] != entry["selected_step"] or selection["best_trained_step"] != entry["best_trained_step"]:
                raise ExpandedResidualError("online checkpoint tracking differs from independent step-zero-aware selection")
            entry["selection_summary"] = selection
            entry["last_step_validation"] = entry["validation"][-1]
            entry["checkpoint"] = _save_checkpoint(output / variant / "checkpoint_best", config, manifest, statistics, binding,
                                                    initialization, best_state, entry["selected_step"], best_metrics)
            entry["best_trained_checkpoint"] = _save_checkpoint(output / variant / "checkpoint_best_trained", config, manifest,
                statistics, binding, initialization, trained_state, entry["best_trained_step"], trained_metrics)
            entry["selected_prior_inference"] = {}
            policy.load_state_dict(best_state, strict=True)
            policy.requires_grad_(False).eval()
            for split in ("train", "validation"):
                metrics, arrays = _evaluate(policy, datasets[split], statistics, settings, np, torch, device, variant=variant,
                                            deadline=deadline, require_zero=entry["selected_step"] == 0)
                entry["selected_prior_inference"][split] = {"metrics": metrics, "predictions": diagnostics._save_arrays(
                    output / variant / (split + "_predictions.npz"), arrays, np)}
            entry["best_trained_prior_inference"] = {}
            policy.load_state_dict(trained_state, strict=True)
            for split in ("train", "validation"):
                metrics, arrays = _evaluate(policy, datasets[split], statistics, settings, np, torch, device,
                                            variant=variant, deadline=deadline)
                entry["best_trained_prior_inference"][split] = {"metrics": metrics, "predictions": diagnostics._save_arrays(
                    output / variant / (split + "_best_trained_predictions.npz"), arrays, np)}
            entry["loss_windows"] = diagnostics._loss_windows(entry["steps"])
            entry["gpu"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            entry["total_elapsed_s"] = time.perf_counter() - variant_started
            entry["status"] = "completed_matched_budget_state_residual_training"
            del optimizer, policy, best_state, trained_state
            torch.cuda.empty_cache()
        report["status"], report["passed"] = "completed_expanded_train_validation_state_residual_comparison", True
        report["limitations"] = ["Only train membership, its fitted state/action/RGB statistics, and the corresponding sampled train anchors change; both arms retain the frozen 2000-step budget and raw validation bytes.",
                                  "State hold is selectable as the measured zero-step checkpoint; best trained and final validation scores are also reported without hiding regressions.",
                                  "Residual heads are deliberately zero initialized, so the prediction parameterization and output-head origin both differ from previous absolute controls.",
                                  "Scratch RGB and state-only have different parameter counts and FLOPs; no new pretrained arm is introduced in this isolated residual comparison.",
                                  "One seed and three repeatedly used validation trajectories support development decisions, not independent-test or physical-success claims.",
                                  "The expanded action standard deviation changes the residual scale; normalized losses/errors across old and expanded fits are not a controlled comparison.",
                                  "Storage-file separation does not verify collection session independence; candidate addition improves same-source coverage only.",
                                  "This checkpoint predicts scaled residuals and requires its bound raw-anchor decoder; the original absolute-action deployment loader is incompatible."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["discarded_probe_optimizer_steps"] = 0
        report["optimizer_steps_including_discarded_probes"] = report["optimizer_steps"] + report["discarded_probe_optimizer_steps"]
        native._gpu_finalize(torch, report)
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(path, report)
    return report
