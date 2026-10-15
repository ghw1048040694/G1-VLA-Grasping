"""Train-only coordinate medians diagnose a residual model's global bias.

This closed-form diagnostic sees every complete training anchor. It is not
compute matched to the step-bounded neural policies. It never reads RGB or
held-out test payloads, and validation cannot change the fitted predictor.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import time
import traceback

from .artifacts import sha256_file
from . import grasp_act as native
from . import expanded_residual as expanded

BACKEND = "grasp_square_train_median_residual_v1"
METHOD = "per_horizon_joint_raw_residual_median_all_complete_train_anchors_v1"
DECODE = "raw_anchor_q_plus_raw_constant_zero_exact_v1"
FIT_COHORTS = ("original_six", "expanded_fourteen")


class ConstantResidualError(ValueError):
    """The fixed train-only diagnostic or its source binding changed."""


def validate_constant_residual_settings(config):
    settings = copy.deepcopy(config["constant_residual"])
    for key, expected in (("backend", BACKEND), ("method", METHOD), ("decode", DECODE),
                          ("fit_cohorts", list(FIT_COHORTS))):
        if settings.get(key) != expected:
            raise ConstantResidualError("the declared train-only median protocol changed: " + key)
    stride = settings.get("eval_stride")
    if (isinstance(stride, bool) or not isinstance(stride, int) or stride != 30
            or stride != config["expanded_residual"]["eval_stride"]):
        raise ConstantResidualError("the frozen stride-30 evaluation cohort must remain unchanged")
    if (config["native_diagnostics"].get("stationary_speed_rad_s") != .02
            or config["native_diagnostics"].get("moving_speed_rad_s") != .1):
        raise ConstantResidualError("the original analysis-only motion thresholds must be retained")
    for key in ("output_dir", "reference_expanded_report_path"):
        if not isinstance(settings.get(key), str) or not settings[key]:
            raise ConstantResidualError(key + " must be an explicit project path")
        native._resolve(config, settings[key])
    digest = settings.get("reference_expanded_report_sha256")
    if (not isinstance(digest, str) or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)):
        raise ConstantResidualError("the frozen expanded report requires a full lowercase SHA256")
    return settings


def _source_pair(state, action, np):
    state, action = np.asarray(state), np.asarray(action)
    if (state.dtype != np.float32 or action.dtype != np.float32
            or state.ndim != 2 or state.shape[1] != 28 or action.shape != state.shape
            or len(state) < 16 or not np.isfinite(state).all() or not np.isfinite(action).all()):
        raise ConstantResidualError("source state and action must be matching finite FP32 [rows>=16,28]")
    return state, action


def fit_constant_residual(episodes, episode_ids, np):
    """One raw-unit median per horizon/joint over all complete train anchors.

    FP32 source angles are promoted before subtraction. NumPy's coordinate
    median takes the midpoint of the two central values for even populations;
    that midpoint belongs to the raw-L1 optimum interval. The deployed
    diagnostic parameter is then explicitly quantized once to FP32.
    """
    ids = list(episode_ids)
    if (not ids or len(set(ids)) != len(ids) or any(identity not in episodes for identity in ids)):
        raise ConstantResidualError("fit identities must be a nonempty unique subset of admitted episodes")
    values, counts = [], {}
    for identity in ids:
        state, action = _source_pair(*episodes[identity], np)
        count = len(state) - 15
        counts[str(identity)] = count
        chunks = np.stack([action[offset:offset + count] for offset in range(16)], axis=1)
        values.append(chunks.astype(np.float64) - state[:count, None, :].astype(np.float64))
    all_residuals = np.concatenate(values, axis=0)
    ideal = np.median(all_residuals, axis=0)
    constant = ideal.astype(np.float32)
    if constant.shape != (16, 28) or not np.isfinite(constant).all():
        raise ConstantResidualError("the fitted constant must retain finite FP32 [16,28] parameters")
    report = {"episode_ids": ids, "complete_anchor_count": int(len(all_residuals)),
              "complete_anchor_count_by_episode": counts, "scalar_label_count": int(all_residuals.size),
              "fit_axis": "all_complete_train_anchors_pooled_across_episodes",
              "source_dtype": "float32", "subtraction_and_median_dtype": "float64",
              "parameter_dtype": "float32", "even_population_rule": "midpoint_of_two_central_values",
              "training_raw_residual_l1_at_float64_median": float(np.abs(all_residuals - ideal).mean()),
              "training_raw_residual_l1_at_saved_float32_parameter": float(np.abs(all_residuals - constant).mean()),
              "parameter_quantization_max_abs_rad": float(np.abs(ideal - constant).max()),
              "parameter_bytes_sha256": hashlib.sha256(constant.tobytes()).hexdigest(),
              "validation_or_test_used_for_fit": False, "per_episode_parameters_fitted": False,
              "action_mean_or_std_used": False, "validation_scaling_or_selection": False,
              "closed_form_fit_performed": True, "optimizer_steps": 0,
              "compute_matched_to_neural_training": False}
    return constant, report


def decode_constant_residual(raw_state, constant, np):
    """Inference receives the current raw state and fixed trained parameters."""
    state, parameter = np.asarray(raw_state), np.asarray(constant)
    if (state.dtype != np.float32 or parameter.dtype != np.float32
            or state.ndim != 2 or state.shape[1] != 28 or parameter.shape != (16, 28)
            or not np.isfinite(state).all() or not np.isfinite(parameter).all()):
        raise ConstantResidualError("decode requires finite FP32 raw states [B,28] and a fixed [16,28] parameter")
    anchor = np.broadcast_to(state[:, None, :], (len(state), 16, 28))
    return np.ascontiguousarray(np.where(parameter == np.float32(0), anchor, anchor + parameter), dtype=np.float32)


def _raw_metrics(prediction, target, state, identities, np):
    ids = np.asarray(identities, dtype=np.int64)
    state = np.asarray(state)
    if ids.shape != (len(target), 2) or state.shape != (len(target), 28):
        raise ConstantResidualError("evaluation identities and raw anchors must match the complete target denominator")
    held = np.broadcast_to(state[:, None, :], target.shape).copy()
    def compare(values, labels):
        metric = native.compute_action_metrics(values, labels, np)
        difference = values.astype(np.float64) - labels.astype(np.float64)
        metric["mae_by_horizon"] = np.abs(difference).mean(axis=(0, 2)).tolist()
        metric["rmse_by_horizon"] = np.sqrt(np.square(difference).mean(axis=(0, 2))).tolist()
        return metric
    result = {"policy": compare(prediction, target), "state_hold": compare(held, target),
              "anchor_independence_assumed": False, "physical_success_evaluated": False,
              "used_for_parameter_selection": False, "normalization_used_for_metrics": False}
    result["per_episode"] = {str(int(identity)): {
        "policy": compare(prediction[ids[:, 0] == identity], target[ids[:, 0] == identity]),
        "state_hold": compare(held[ids[:, 0] == identity], target[ids[:, 0] == identity])}
        for identity in np.unique(ids[:, 0])}
    hold_mae = result["state_hold"]["all_28"]["mae"]
    result["policy_to_state_hold_mae_ratio"] = result["policy"]["all_28"]["mae"] / hold_mae if hold_mae else None
    return result


def evaluate_constant_residual(episodes, episode_ids, constant, stride, np):
    """Apply the immutable predictor; labels are supplied only to scoring."""
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ConstantResidualError("evaluation stride must be a positive integer")
    episode_ids = list(episode_ids)
    if (not episode_ids or len(set(episode_ids)) != len(episode_ids)
            or any(identity not in episodes for identity in episode_ids)):
        raise ConstantResidualError("evaluation requires unique admitted episode identities")
    states, targets, identities, speeds = [], [], [], []
    for identity in episode_ids:
        state, action = _source_pair(*episodes[identity], np)
        for frame in range(0, len(state) - 15, stride):
            states.append(state[frame])
            targets.append(action[frame:frame + 16])
            identities.append((identity, frame))
            speeds.append(float(np.max(np.abs(state[frame].astype(np.float64) - state[frame - 1].astype(np.float64)))) * 30
                          if frame else np.nan)
    state = np.asarray(states, dtype=np.float32)
    target = np.asarray(targets, dtype=np.float32)
    ids = np.asarray(identities, dtype=np.int64)
    prediction = decode_constant_residual(state, constant, np)
    speed = np.asarray(speeds, dtype=np.float64)
    known = np.isfinite(speed)
    zero = decode_constant_residual(state, np.zeros((16, 28), dtype=np.float32), np)
    if zero.tobytes() != np.broadcast_to(state[:, None, :], target.shape).copy().tobytes():
        raise ConstantResidualError("the exact-zero predictor must retain all raw anchor bits")
    metrics = _raw_metrics(prediction, target, state, ids, np)
    masks = {"stationary": known & (speed <= .02), "moving": known & (speed >= .1),
             "intermediate": known & (speed > .02) & (speed < .1), "unknown_first_row": ~known}
    motion = {"speed_definition": "30*max_joint_abs(raw_q[t]-raw_q[t-1]); t=0 unknown",
              "stationary_max_speed_rad_s": .02, "moving_min_speed_rad_s": .1,
              "used_as_model_input": False, "partition_anchor_count": len(ids), "strata": {}}
    for name, mask in masks.items():
        motion["strata"][name] = {"anchor_count": int(mask.sum()), "metrics":
            _raw_metrics(prediction[mask], target[mask], state[mask], ids[mask], np) if mask.any() else None}
    if sum(item["anchor_count"] for item in motion["strata"].values()) != len(ids):
        raise ConstantResidualError("motion groups must partition the exact evaluation denominator")
    metrics.update(motion_strata=motion, eval_stride_source_frames=stride,
                   all_evaluated_zero_residual_identity_bitwise=True, inference_received_labels=False,
                   inference_received_rgb=False)
    arrays = {"predictions_source_actions": prediction, "target_source_recorded_actions": target,
              "observation_state": state, "sample_episode_frame": ids,
              "constant_raw_residual": constant, "anchor_max_joint_speed_rad_s": speed,
              "anchor_speed_known": known}
    return metrics, arrays


def _load_pair(root, item, np):
    from .grasp_data import _verify_artifact
    path = _verify_artifact(root, item["arrays"])
    with np.load(path, allow_pickle=False) as saved:
        state, action = _source_pair(saved["source_state"], saved["source_action"], np)
        if (len(state) != item["row_count"] or item["sample_count"] != len(state) - 15
                or not np.array_equal(saved["source_frame_index"], np.arange(len(state)))
                or not np.all(saved["source_episode_index"] == item["episode_id"])):
            raise ConstantResidualError("raw source arrays differ from their admitted complete episode identity")
        state, action = state.copy(), action.copy()
    return (state, action), {"path": str(path), "sha256": sha256_file(path),
                            "size_bytes": path.stat().st_size, "row_count": len(state),
                            "complete_anchor_count": len(state) - 15,
                            "state_bytes_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                            "action_bytes_sha256": hashlib.sha256(action.tobytes()).hexdigest()}


def _prepare(config, settings, np):
    from . import grasp_data, train_expansion
    reference_path = native._resolve(config, settings["reference_expanded_report_path"])
    if sha256_file(reference_path) != settings["reference_expanded_report_sha256"]:
        raise ConstantResidualError("the completed expanded residual report changed")
    reference = native._read(reference_path)
    if (reference.get("passed") is not True
            or reference.get("status") != "completed_expanded_train_validation_state_residual_comparison"
            or reference.get("test_payloads_opened") is not False or reference.get("test_used_for_selection") is not False):
        raise ConstantResidualError("the reference must be the completed train/validation-only expanded comparison")
    previous = expanded.validate_expanded_residual_settings(config)
    for key in ("data_manifest_path", "data_manifest_sha256", "eval_stride"):
        if previous[key] != reference.get("settings", {}).get(key):
            raise ConstantResidualError("current expanded source differs from the frozen report: " + key)
    binding = reference["bindings"]
    for key, module in (("original_training_module_sha256", native), ("expanded_residual_module_sha256", expanded),
                        ("expanded_data_module_sha256", train_expansion)):
        if binding.get(key) != sha256_file(module.__file__):
            raise ConstantResidualError("the frozen source implementation changed: " + key)
    old_path = native._resolve(config, config["grasp_act"]["dataset_manifest_path"])
    new_path = native._resolve(config, previous["data_manifest_path"])
    if (sha256_file(old_path) != binding["original_manifest_sha256"]
            or sha256_file(new_path) != binding["manifest_sha256"]
            or sha256_file(new_path) != previous["data_manifest_sha256"]):
        raise ConstantResidualError("the frozen original or expanded manifest hash changed")
    original = grasp_data.load_grasp_manifest(old_path.parent, expected_settings=config["source_native"],
                                            expected_source=config["external_data"], verify_files=False)
    manifest = train_expansion.load_expanded_manifest(new_path.parent, verify_files=False)
    admission = expanded._verify_expansion_audit(config, manifest, train_expansion)
    if admission != binding["candidate_admission"]:
        raise ConstantResidualError("candidate/session admission differs from the complete expanded reference")
    old_items = {item["episode_id"]: item for item in original["episodes"] if item["split"] in ("train", "validation")}
    expected = set(expanded.TRAIN + expanded.VALIDATION)
    if {item["episode_id"] for item in manifest["episodes"]} != expected:
        raise ConstantResidualError("only fourteen train and the unchanged three validation episodes are admitted")
    episodes, sources = {}, {}
    for item in manifest["episodes"]:
        identity = item["episode_id"]
        if identity in old_items and item["arrays"] != old_items[identity]["arrays"]:
            raise ConstantResidualError("an original train or validation source array was modified")
        relative = item["arrays"]["relative_path"]
        if binding["selected_artifact_sha256"].get(relative) != item["arrays"]["sha256"]:
            raise ConstantResidualError("the selected source array was not present in the completed expanded comparison")
        episodes[identity], sources[str(identity)] = _load_pair(new_path.parent, item, np)
    counts = {"original_six": sum(len(episodes[item][0]) - 15 for item in expanded.ORIGINAL_TRAIN),
              "expanded_fourteen": sum(len(episodes[item][0]) - 15 for item in expanded.TRAIN),
              "validation": sum(len(episodes[item][0]) - 15 for item in expanded.VALIDATION)}
    if counts != {"original_six": 5651, "expanded_fourteen": 13835, "validation": 2488}:
        raise ConstantResidualError("the frozen complete-anchor populations changed")
    bindings = {"experiment_config_sha256": native._digest(config), "module_sha256": sha256_file(__file__),
                "reference_expanded_report_path": str(reference_path),
                "reference_expanded_report_sha256": sha256_file(reference_path),
                "original_manifest_sha256": sha256_file(old_path), "expanded_manifest_sha256": sha256_file(new_path),
                "candidate_admission": admission, "source_arrays": sources, "complete_anchor_counts": counts,
                "test_payloads_opened": False, "test_metrics_read": False, "rgb_payloads_opened": False}
    return episodes, bindings


def run_constant_residual_audit(config, run_dir):
    """Fit two predeclared train-only medians and score the frozen raw cohorts."""
    settings = validate_constant_residual_settings(config)
    directory = Path(run_dir)
    target = directory / "reports" / "constant_residual.json"
    if not directory.is_dir() or target.exists():
        raise ConstantResidualError("the diagnostic requires an existing run and a new immutable report")
    target.parent.mkdir(exist_ok=True)
    report = {"schema_version": 1, "settings": settings, "status": "failed", "passed": False,
              "training_started": False, "optimizer_steps": 0, "closed_form_train_fit_performed": False,
              "compute_matched_to_neural_training": False, "test_payloads_opened": False,
              "test_metrics_read": False, "test_used_for_selection": False, "rgb_payloads_opened": False,
              "physical_success_evaluated": False, "closed_loop_control": False, "fits": {}}
    started = time.perf_counter()
    try:
        import numpy as np
        output = native._resolve(config, settings["output_dir"])
        if output.exists():
            raise ConstantResidualError("the diagnostic never overwrites or resumes an existing output")
        episodes, report["bindings"] = _prepare(config, settings, np)
        output.mkdir(parents=True)
        from .act_diagnostics import _save_arrays
        for name, identities in zip(FIT_COHORTS, (expanded.ORIGINAL_TRAIN, expanded.TRAIN)):
            constant, fit = fit_constant_residual(episodes, identities, np)
            report["closed_form_train_fit_performed"] = True
            fit["source_arrays"] = {str(identity): report["bindings"]["source_arrays"][str(identity)] for identity in identities}
            fit["parameter_artifact"] = _save_arrays(output / (name + "_parameters.npz"),
                {"constant_raw_residual": constant, "fit_episode_ids": np.asarray(identities, dtype=np.int64)}, np)
            fit["evaluations"] = {}
            fit["status"] = "train_only_fit_completed_evaluation_pending"
            report["fits"][name] = fit
            for cohort, ids in (("original_six", expanded.ORIGINAL_TRAIN), ("added_eight", expanded.CANDIDATES),
                                ("expanded_fourteen", expanded.TRAIN), ("validation", expanded.VALIDATION)):
                metrics, arrays = evaluate_constant_residual(episodes, ids, constant, settings["eval_stride"], np)
                membership = ("held_out_validation_never_fitted" if cohort == "validation" else
                              "included_in_fit" if set(ids).issubset(identities) else
                              "excluded_from_fit_diagnostic_train_pool" if not set(ids).intersection(identities) else
                              "mixed_fit_membership_diagnostic_train_pool")
                fit["evaluations"][cohort] = {"episode_ids": list(ids), "metrics": metrics,
                    "fit_membership": membership,
                    "prediction_artifact": _save_arrays(output / (name + "_" + cohort + "_predictions.npz"), arrays, np)}
            if fit["evaluations"]["validation"]["metrics"]["policy"]["sample_count"] != 85:
                raise ConstantResidualError("the frozen validation evaluation denominator changed")
            fit["status"] = "completed_predeclared_train_fit_and_raw_cohort_scoring"
        report.update(status="completed_train_only_constant_residual_audit", passed=True,
                      parameter_choice_predeclared_without_validation=True)
        report["limitations"] = [
            "Each constant is fitted to all complete training anchors, unlike the neural model's 4000 sampled draws; this is not a compute-matched comparison.",
            "The fixed median predictor is a diagnostic of global horizon/joint bias; it does not establish that a learned policy lacks state-dependent dynamics.",
            "The original three validation episodes are reused only for development scoring; no independent test or closed-loop success is claimed.",
            "The eight added episodes expand same-source coverage; their individual acquisition sessions remain unverified."]
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)}
    finally:
        report["total_elapsed_s"] = time.perf_counter() - started
        native._write(target, report)
    return report
