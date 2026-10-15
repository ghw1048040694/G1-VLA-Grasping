"""Raw-unit, label-free collapse and paired scoring for frozen residual policies.

The model-output median is fitted from frozen train predictions, not labels.
It measures whether retaining the policy's input-dependent output helps on the
same recorded anchors. It is distinct from the train-label median residual.
No optimizer, checkpoint update, filesystem access or significance test lives
in these helpers.
"""

from __future__ import annotations

import hashlib

from . import grasp_act as native
from .expanded_residual import TRAIN


class ConditioningMetricsError(ValueError):
    """A raw prediction, fixed fit cohort or paired denominator changed."""


def _chunks(values, np, name):
    values = np.asarray(values)
    if (values.dtype != np.float32 or values.ndim != 3 or values.shape[1:] != (16, 28)
            or not len(values) or not np.isfinite(values).all()):
        raise ConditioningMetricsError(name + " must be finite FP32 [anchors>0,16,28]")
    return values


def _identities(identities, count, np):
    ids = np.asarray(identities)
    if (ids.shape != (count, 2) or not np.issubdtype(ids.dtype, np.integer)
            or np.issubdtype(ids.dtype, np.bool_) or np.any(ids < 0)
            or np.any(ids > np.iinfo(np.int64).max)):
        raise ConditioningMetricsError("every anchor requires nonnegative integer [episode,frame] identity")
    ids = np.asarray(ids, dtype=np.int64)
    if len(np.unique(ids, axis=0)) != len(ids):
        raise ConditioningMetricsError("evaluation anchor identities must be unique")
    return ids


def _parameter(constant, np):
    parameter = np.asarray(constant)
    if parameter.dtype != np.float32 or parameter.shape != (16, 28) or not np.isfinite(parameter).all():
        raise ConditioningMetricsError("the frozen raw constant must be finite FP32 [16,28]")
    return parameter


def fit_model_output_constant(train_scaled_residual, train_ids, action_std,
                              allowed_train_episode_ids, np, *, expected_train_identities):
    """Collapse frozen train outputs using their pooled coordinate median.

    Multiplication uses FP32, exactly as the raw residual term in the original
    decoder. Only then is it promoted to FP64 for the median; even populations
    use the midpoint. The result is quantized once to FP32. The caller supplies
    the full stride-30 train identities derived from the admitted dataset.
    Targets, validation predictions and validation labels are not arguments.
    """
    scaled = _chunks(train_scaled_residual, np, "frozen train scaled residual")
    ids = _identities(train_ids, len(scaled), np)
    expected = _identities(expected_train_identities, len(scaled), np)
    declared = tuple(allowed_train_episode_ids)
    if (declared != TRAIN or any(not isinstance(identity, (int, np.integer))
                                or isinstance(identity, (bool, np.bool_)) for identity in declared)
            or set(ids[:, 0].tolist()) != set(TRAIN)):
        raise ConditioningMetricsError("model-output fitting requires exactly the frozen fourteen train episode identities")
    if not np.array_equal(ids, expected):
        raise ConditioningMetricsError("model-output fitting must retain the complete admitted train evaluation identity order")
    counts = {}
    for episode in TRAIN:
        frames = ids[ids[:, 0] == episode, 1]
        if not np.array_equal(np.sort(frames), np.arange(0, int(frames.max()) + 1, 30, dtype=np.int64)):
            raise ConditioningMetricsError("each fitted episode must retain its full frame-zero stride-30 anchor sequence")
        counts[str(episode)] = int(len(frames))
    std = np.asarray(action_std)
    if (std.dtype != np.float32 or std.shape != (28,) or not np.isfinite(std).all()
            or np.any(std <= 0)):
        raise ConditioningMetricsError("the saved model action scale must be finite positive FP32 [28]")
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.multiply(scaled, std, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ConditioningMetricsError("scaled model output overflowed its original FP32 raw-residual multiplication")
    promoted = raw.astype(np.float64)
    ideal = np.median(promoted, axis=0)
    parameter = np.ascontiguousarray(ideal.astype(np.float32))
    report = {
        "method": "pooled_frozen_train_output_coordinate_median_v1",
        "fit_source": "frozen_model_outputs_on_all_admitted_stride30_train_anchors",
        "episode_ids": list(TRAIN), "pooled_model_output_count": int(len(ids)),
        "model_output_count_by_episode": counts, "eval_stride_source_frames": 30,
        "fit_axis": "train_anchors_pooled_across_episodes_per_horizon_joint",
        "source_scaled_output_dtype": "float32", "raw_scale_multiply_dtype": "float32",
        "median_dtype": "float64", "parameter_dtype": "float32",
        "even_population_rule": "midpoint_of_two_central_values",
        "train_identity_order_exactly_matches_admitted_dataset": True,
        "train_identity_bytes_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "train_raw_model_output_bytes_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        "parameter_bytes_sha256": hashlib.sha256(parameter.tobytes()).hexdigest(),
        "parameter_quantization_max_abs_rad": float(np.abs(ideal - parameter).max()),
        "train_model_output_l1_at_float64_median_rad": float(np.abs(promoted - ideal).mean()),
        "train_model_output_l1_at_saved_float32_parameter_rad": float(np.abs(promoted - parameter).mean()),
        "labels_used_for_fit": False, "validation_or_test_used_for_fit": False,
        "train_label_residual_l1_optimum_claimed": False, "closed_form_train_output_fit_performed": True,
        "optimizer_created": False, "optimizer_steps": 0, "checkpoint_parameters_modified": False,
        "new_learned_parameters_in_checkpoint": 0, "compute_matched_to_neural_training": False,
    }
    return parameter, report


def _compared(prediction, target, np):
    result = native.compute_action_metrics(prediction, target, np)
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    result["mae_by_horizon"] = np.abs(difference).mean(axis=(0, 2)).tolist()
    result["rmse_by_horizon"] = np.sqrt(np.square(difference).mean(axis=(0, 2))).tolist()
    return result


def _metric_group(prediction, target, state, ids, np):
    hold = np.broadcast_to(state[:, None, :], target.shape).copy()
    result = {"policy": _compared(prediction, target, np), "state_hold": _compared(hold, target, np)}
    hold_mae = result["state_hold"]["all_28"]["mae"]
    result["policy_to_state_hold_mae_ratio"] = result["policy"]["all_28"]["mae"] / hold_mae if hold_mae else None
    return result


def _motion(speeds, ids, np):
    speed = np.asarray(speeds)
    if (speed.shape != (len(ids),) or not np.issubdtype(speed.dtype, np.floating)
            or np.isinf(speed).any() or np.any(speed[np.isfinite(speed)] < 0)):
        raise ConditioningMetricsError("analysis speed must be a nonnegative finite float per anchor or NaN for frame zero")
    known = np.isfinite(speed)
    if not np.array_equal(known, ids[:, 1] > 0):
        raise ConditioningMetricsError("only each episode's frame-zero speed may be unknown")
    masks = {"stationary": known & (speed <= .02), "moving": known & (speed >= .1),
             "intermediate": known & (speed > .02) & (speed < .1), "unknown_first_row": ~known}
    result = {"speed_definition": "30*max_joint_abs(raw_q[t]-raw_q[t-1]); t=0 unknown",
              "stationary_max_speed_rad_s": .02, "moving_min_speed_rad_s": .1,
              "used_as_model_input": False, "partition_anchor_count": int(len(ids)), "strata": {}}
    if sum(int(mask.sum()) for mask in masks.values()) != len(ids):
        raise ConditioningMetricsError("motion strata must partition the unchanged evaluation denominator")
    return result, masks


def raw_conditioning_metrics(predictions, targets, raw_states, ids, np, *, speeds=None):
    """Score recorded raw actions with one identical complete-chunk denominator."""
    prediction = _chunks(predictions, np, "predictions")
    target = _chunks(targets, np, "targets")
    state = np.asarray(raw_states)
    identities = _identities(ids, len(prediction), np)
    if (prediction.shape != target.shape or state.dtype != np.float32
            or state.shape != (len(prediction), 28) or not np.isfinite(state).all()):
        raise ConditioningMetricsError("raw states and labels must match every finite FP32 predicted action chunk")
    result = _metric_group(prediction, target, state, identities, np)
    result["per_episode"] = {str(int(episode)): _metric_group(
        prediction[identities[:, 0] == episode], target[identities[:, 0] == episode],
        state[identities[:, 0] == episode], identities[identities[:, 0] == episode], np)
        for episode in np.unique(identities[:, 0])}
    result.update(anchor_independence_assumed=False, physical_success_evaluated=False,
                  used_for_parameter_selection=False, normalization_used_for_metrics=False,
                  error_arithmetic_dtype="float64", evaluated_anchor_count=int(len(identities)))
    if speeds is not None:
        motion, masks = _motion(speeds, identities, np)
        for name, mask in masks.items():
            motion["strata"][name] = {"anchor_count": int(mask.sum()), "metrics": _metric_group(
                prediction[mask], target[mask], state[mask], identities[mask], np) if mask.any() else None}
        result["motion_strata"] = motion
    return result


def _paired_group(full, comparator, target, ids, np):
    full_error = np.abs(full.astype(np.float64) - target.astype(np.float64))
    other_error = np.abs(comparator.astype(np.float64) - target.astype(np.float64))
    full_mae, other_mae = full_error.mean(axis=(1, 2)), other_error.mean(axis=(1, 2))
    delta = other_mae - full_mae
    return {"full_mae_rad": float(full_error.mean()), "comparator_mae_rad": float(other_error.mean()),
            "comparator_minus_full_mae_rad": float(other_error.mean() - full_error.mean()),
            "anchor_count": int(len(ids)), "valid_scalar_count": int(target.size),
            "anchor_comparison_counts": {"improved": int((delta > 0).sum()),
                                         "worsened": int((delta < 0).sum()), "tied": int((delta == 0).sum())},
            "comparator_minus_full_mae_by_horizon_rad": (other_error.mean(axis=(0, 2)) - full_error.mean(axis=(0, 2))).tolist(),
            "comparator_minus_full_mae_by_joint_rad": (other_error.mean(axis=(0, 1)) - full_error.mean(axis=(0, 1))).tolist(),
            "arms_14_comparator_minus_full_mae_rad": float(other_error[:, :, :14].mean() - full_error[:, :, :14].mean()),
            "hands_14_comparator_minus_full_mae_rad": float(other_error[:, :, 14:].mean() - full_error[:, :, 14:].mean()),
            "per_anchor": [{"episode_id": int(identity[0]), "frame_index": int(identity[1]),
                            "full_mae_rad": float(first), "comparator_mae_rad": float(second),
                            "comparator_minus_full_mae_rad": float(change)}
                           for identity, first, second, change in zip(ids, full_mae, other_mae, delta)]}


def paired_conditioning_metrics(full_prediction, comparator_prediction, targets, ids, np, *,
                                tie_tolerance_rad=0.0, speeds=None):
    """A positive paired difference means the full prediction has smaller MAE.

    Counts describe correlated anchors. No independence, p-value, confidence
    interval or episode population inference is asserted.
    """
    if isinstance(tie_tolerance_rad, bool) or tie_tolerance_rad != 0.0:
        raise ConditioningMetricsError("paired ties use exactly zero tolerance; no threshold is selected")
    full = _chunks(full_prediction, np, "full prediction")
    other = _chunks(comparator_prediction, np, "comparator prediction")
    target = _chunks(targets, np, "targets")
    identities = _identities(ids, len(full), np)
    if full.shape != other.shape or full.shape != target.shape:
        raise ConditioningMetricsError("paired predictions must share every complete recorded label")
    result = _paired_group(full, other, target, identities, np)
    result["per_episode"] = {str(int(episode)): _paired_group(
        full[identities[:, 0] == episode], other[identities[:, 0] == episode],
        target[identities[:, 0] == episode], identities[identities[:, 0] == episode], np)
        for episode in np.unique(identities[:, 0])}
    episode_deltas = [value["comparator_minus_full_mae_rad"] for value in result["per_episode"].values()]
    result["episode_comparison_counts"] = {"improved": sum(value > 0 for value in episode_deltas),
                                           "worsened": sum(value < 0 for value in episode_deltas),
                                           "tied": sum(value == 0 for value in episode_deltas)}
    result.update(tie_tolerance_rad=0.0, positive_difference_means="full_prediction_has_smaller_mae",
                  anchor_independence_assumed=False, significance_test_performed=False,
                  used_for_parameter_selection=False, error_arithmetic_dtype="float64")
    if speeds is not None:
        motion, masks = _motion(speeds, identities, np)
        for name, mask in masks.items():
            motion["strata"][name] = {"anchor_count": int(mask.sum()), "metrics": _paired_group(
                full[mask], other[mask], target[mask], identities[mask], np) if mask.any() else None}
        result["motion_strata"] = motion
    return result


def model_residual_variation(raw_residual, constant, ids, np):
    """Measure pre-anchor raw output variation without subtracting decoded q."""
    residual = _chunks(raw_residual, np, "pre-anchor raw model residual")
    parameter = _parameter(constant, np)
    identities = _identities(ids, len(residual), np)
    values = residual.astype(np.float64)
    distance = values - parameter.astype(np.float64)
    mean, std, span = values.mean(axis=0), values.std(axis=0), np.ptp(values, axis=0)
    result = {
        "anchor_count": int(len(residual)), "units": "source_recorded_radian",
        "raw_residual_obtained_before_anchor_addition": True, "moment_arithmetic_dtype": "float64",
        "population_std_ddof": 0, "mean_by_horizon_joint_rad": mean.tolist(),
        "std_by_horizon_joint_rad": std.tolist(), "range_by_horizon_joint_rad": span.tolist(),
        "mean_absolute_raw_residual_rad": float(np.abs(values).mean()),
        "mean_coordinate_std_rad": float(std.mean()), "max_coordinate_std_rad": float(std.max()),
        "mean_coordinate_range_rad": float(span.mean()), "max_coordinate_range_rad": float(span.max()),
        "coordinates_with_exactly_nonzero_range": int((span > 0).sum()),
        "distance_to_train_output_constant_mae_rad": float(np.abs(distance).mean()),
        "distance_to_train_output_constant_rmse_rad": float(np.sqrt(np.square(distance).mean())),
        "distance_to_train_output_constant_max_abs_rad": float(np.abs(distance).max()),
        "distance_to_train_output_constant_mae_by_horizon_rad": np.abs(distance).mean(axis=(0, 2)).tolist(),
        "distance_to_train_output_constant_mae_by_joint_rad": np.abs(distance).mean(axis=(0, 1)).tolist(),
        "distance_to_train_output_constant_arms_14_mae_rad": float(np.abs(distance[:, :, :14]).mean()),
        "distance_to_train_output_constant_hands_14_mae_rad": float(np.abs(distance[:, :, 14:]).mean()),
        "per_episode": {}, "input_output_variation_proves_useful_state_matching": False,
        "physical_success_evaluated": False, "anchor_independence_assumed": False,
    }
    for episode in np.unique(identities[:, 0]):
        mask = identities[:, 0] == episode
        part, change = values[mask], distance[mask]
        result["per_episode"][str(int(episode))] = {
            "anchor_count": int(mask.sum()), "mean_coordinate_std_rad": float(part.std(axis=0).mean()),
            "max_coordinate_range_rad": float(np.ptp(part, axis=0).max()),
            "distance_to_train_output_constant_mae_rad": float(np.abs(change).mean()),
            "distance_to_train_output_constant_rmse_rad": float(np.sqrt(np.square(change).mean()))}
    return result
