"""Lossless, masked native labels; this protocol grants no training admission.

A teacher replans every interval. Only the contiguous prefix executable against
one recorded planning anchor is a label for the shared chunk controller. Padding
holds the last real target for shape compatibility and is never supervision.
"""
import json

import numpy as np

from .contracts import ContractError, canonical_sha256


LABEL_PROTOCOL = "common_anchor_causal_applied_prefix_v1"
LABEL_BACKEND = "native_causal_prefix_v1"


class LabelProtocolError(ContractError):
    """Evidence or a consumer does not support this explicit label protocol."""


def _mask(action_is_pad, shape):
    value = np.asarray(action_is_pad)
    if value.dtype.kind != "b" or value.shape != shape:
        raise LabelProtocolError("action_is_pad must be explicit boolean [samples,horizon]")
    if np.any(np.diff(value.astype(np.int8), axis=1) < 0) or np.any(value.all(axis=1)):
        raise LabelProtocolError("real actions must form one nonempty contiguous prefix")
    return value


def encode_causal_prefixes(episode, contract):
    """Copy real commands unchanged; stop at the first contract/evidence failure.

    Previous command is a past recorded command, never a future state. An anchor
    without that evidence is excluded. The first label is the command starting
    at this observation, not the command following its post-state.
    """
    names = episode["joint_names"].tolist()
    contract.assert_joint_order(names)
    q = np.asarray(episode["observation_joint_position_rad"], dtype=np.float64)
    actions = np.asarray(episode["action_joint_position_rad"], dtype=np.float64)
    count, horizon, dim = len(q), contract.chunk_steps, contract.action_dim
    if q.shape != (count, dim) or actions.shape != q.shape or not np.isfinite(q).all() or not np.isfinite(actions).all():
        raise LabelProtocolError("observations and real commands must be finite action31 matrices")
    eligible = np.asarray(episode["pairing_valid"])
    continuity = np.asarray(episode["continuity_from_previous"])
    segments = np.asarray(episode["state_segment_id"])
    sequence = np.asarray(episode["command_sequence_index"])
    starts, ends = np.asarray(episode["command_start_time_s"]), np.asarray(episode["command_end_time_s"])
    for value in (eligible, continuity, segments, sequence, starts, ends):
        if value.shape != (count,):
            raise LabelProtocolError("per-command evidence shape differs")
    if eligible.dtype.kind != "b" or continuity.dtype.kind != "b":
        raise LabelProtocolError("pairing and continuity evidence must be boolean")
    if segments.dtype.kind not in "iu" or sequence.dtype.kind not in "iu" or np.any(segments < 0) or np.any(sequence < 0):
        raise LabelProtocolError("segment and source sequence indices must be nonnegative integers")
    observation_times = np.asarray(episode["observation_time_s"])
    if (observation_times.shape != (count,) or not np.isfinite(observation_times).all()
            or not np.isfinite(starts).all() or not np.isfinite(ends).all()
            or np.any(ends <= starts) or np.any(np.abs(observation_times - starts) > 1e-9)
            or np.any(np.abs(ends - starts - 1.0 / contract.control_hz) > 1e-9)):
        raise LabelProtocolError("observation and interval times must be finite actual pre-command timestamps")
    amplitude = np.asarray(contract.max_anchor_delta_rad)
    step_limit = np.asarray(contract.max_velocity_rad_s) / contract.control_hz
    rows, excluded, histogram = [], [], {}
    for start in np.flatnonzero(eligible):
        start = int(start)
        previous_known = (start > 0 and continuity[start] and segments[start] == segments[start - 1]
                          and sequence[start] == sequence[start - 1] + 1
                          and abs(starts[start] - ends[start - 1]) <= 1e-9)
        prefix, reason = 0, "horizon_complete"
        previous = actions[start - 1].copy() if previous_known else None
        if not previous_known:
            reason = "previous_applied_command_unproven"
        else:
            for offset in range(horizon):
                index = start + offset
                if index >= count:
                    reason = "source_end"
                    break
                if (not eligible[index] or segments[index] != segments[start]
                        or offset and (not continuity[index] or sequence[index] != sequence[index - 1] + 1
                                       or abs(starts[index] - ends[index - 1]) > 1e-9)):
                    reason = "pairing_or_segment_boundary"
                    break
                target = actions[index]
                if np.any(np.abs(target - q[start]) > amplitude + 1e-12):
                    reason = "common_anchor_amplitude"
                    break
                if np.any(np.abs(target - previous) > step_limit + 1e-12):
                    reason = "adjacent_command_velocity"
                    break
                previous = target
                prefix += 1
        histogram[str(prefix)] = histogram.get(str(prefix), 0) + 1
        if prefix < contract.execute_steps:
            excluded.append({"anchor_record_index": start, "valid_prefix_length": prefix, "stop_reason": reason})
            continue
        targets = np.repeat(actions[start + prefix - 1][None], horizon, axis=0)
        targets[:prefix] = actions[start:start + prefix]
        pad = np.arange(horizon) >= prefix
        source_indices = np.where(pad, -1, start + np.arange(horizon))
        delta = np.asarray(contract.encode_chunk(q[start].tolist(), targets.tolist(), joint_names=names))
        decoded = np.asarray(contract.decode_chunk(q[start].tolist(), delta.tolist(), joint_names=names))
        limited = np.asarray(contract.limit_targets(q[start].tolist(), decoded.tolist(), joint_names=names,
                                                   previous_target=actions[start - 1].tolist()))
        # Every admitted row really is consumable by the unchanged codec. A
        # padding convention cannot rescue an invalid first execution horizon.
        if not np.allclose(limited, targets, rtol=0, atol=1e-12):
            raise LabelProtocolError("padded prefix differs under the shared action limiter")
        first = start
        while first > 0 and continuity[first] and segments[first - 1] == segments[start]:
            first -= 1
        history = np.maximum(np.arange(start - contract.history_steps + 1, start + 1), first)
        rows.append((start, prefix, reason, targets, delta, pad, source_indices, history))
    size = len(rows)
    arrays = {
        "label_protocol": np.asarray(LABEL_PROTOCOL),
        "joint_names": np.asarray(names),
        "anchor_record_index": np.asarray([r[0] for r in rows], dtype=np.int64),
        "valid_prefix_length": np.asarray([r[1] for r in rows], dtype=np.int64),
        "prefix_stop_reason": np.asarray([r[2] for r in rows], dtype="U64"),
        "action_absolute_target_rad": np.asarray([r[3] for r in rows], dtype=np.float64).reshape(size, horizon, dim),
        "action_delta_chunk_rad": np.asarray([r[4] for r in rows], dtype=np.float64).reshape(size, horizon, dim),
        "action_is_pad": np.asarray([r[5] for r in rows], dtype=np.bool_).reshape(size, horizon),
        "source_record_index": np.asarray([r[6] for r in rows], dtype=np.int64).reshape(size, horizon),
        "observation_history_record_index": np.asarray([r[7] for r in rows], dtype=np.int64).reshape(size, contract.history_steps),
    }
    anchors = arrays["anchor_record_index"]
    arrays["anchor_joint_position_rad"] = q[anchors].copy()
    arrays["previous_applied_record_index"] = anchors - 1
    arrays["previous_applied_target_rad"] = actions[anchors - 1].copy()
    arrays["anchor_observation_time_s"] = np.asarray(episode["observation_time_s"])[anchors].copy()
    metadata = {"protocol": LABEL_PROTOCOL, "training_eligible": False, "split": "diagnostic",
                "chunk_steps": horizon, "execute_steps": contract.execute_steps,
                "minimum_valid_prefix": contract.execute_steps, "contract": contract.metadata(),
                "padding_semantics": "repeat_last_real_target_shape_only_masked_not_truth",
                "real_labels_modified": False, "future_state_is_policy_input": False,
                "previous_command_semantics": "recorded_applied_command_at_anchor_minus_one",
                "consumer_cache_bridge": "not_implemented_training_fail_closed"}
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))
    return {"arrays": arrays, "excluded": excluded,
            "summary": {"raw_control_count": count, "candidate_anchor_count": int(eligible.sum()),
                        "admitted_anchor_count": size, "excluded_anchor_count": len(excluded),
                        "valid_prefix_histogram": histogram,
                        "real_label_position_count": int(arrays["valid_prefix_length"].sum()),
                        "unique_real_command_count": len(set(arrays["source_record_index"][~arrays["action_is_pad"]].tolist())),
                        "padding_position_count": int(arrays["action_is_pad"].sum()),
                        "training_eligible_count": 0}}


def validate_masked_labels(labels, contract):
    """Validate a derived artifact before a mask-aware diagnostic consumer uses it."""
    if str(labels.get("label_protocol", "")) != LABEL_PROTOCOL:
        raise LabelProtocolError("unknown native action label protocol")
    contract.assert_joint_order(labels["joint_names"].tolist())
    action = np.asarray(labels["action_delta_chunk_rad"])
    if action.ndim != 3 or action.shape[1:] != (contract.chunk_steps, contract.action_dim) or not np.isfinite(action).all():
        raise LabelProtocolError("masked action labels must be finite [samples,8,31]")
    pad = _mask(labels["action_is_pad"], action.shape[:2])
    lengths = (~pad).sum(axis=1)
    if np.any(lengths < contract.execute_steps) or not np.array_equal(lengths, labels["valid_prefix_length"]):
        raise LabelProtocolError("prefix must contain the full unchanged four-command execution")
    indices = np.asarray(labels["source_record_index"])
    anchors = np.asarray(labels["anchor_record_index"])
    if anchors.shape != (len(action),) or anchors.dtype.kind not in "iu" or np.any(anchors < 1) or indices.dtype.kind not in "iu":
        raise LabelProtocolError("anchor/source record indices must be explicit nonnegative integers")
    expected = anchors[:, None] + np.arange(contract.chunk_steps)
    if indices.shape != pad.shape or not np.array_equal(indices, np.where(pad, -1, expected)):
        raise LabelProtocolError("real source indices must be contiguous; padding has no source")
    histories = np.asarray(labels["observation_history_record_index"])
    if (histories.shape != (len(action), contract.history_steps) or histories.dtype.kind not in "iu"
            or np.any(histories > anchors[:, None]) or np.any(histories < 0)
            or np.any(np.diff(histories, axis=1) < 0) or not np.array_equal(histories[:, -1], anchors)):
        raise LabelProtocolError("policy observation history cannot contain future records")
    absolute = np.asarray(labels["action_absolute_target_rad"])
    q = np.asarray(labels["anchor_joint_position_rad"])
    previous = np.asarray(labels["previous_applied_target_rad"])
    if absolute.shape != action.shape or q.shape != (len(action), 31) or previous.shape != q.shape:
        raise LabelProtocolError("anchor, previous command and absolute label shapes differ")
    if not np.array_equal(labels["previous_applied_record_index"], anchors - 1):
        raise LabelProtocolError("previous command must be past evidence")
    for row, length in enumerate(lengths):
        if not np.array_equal(absolute[row, length:], np.repeat(absolute[row, length - 1][None], contract.chunk_steps - length, axis=0)):
            raise LabelProtocolError("padding must only hold the last real target")
        decoded = np.asarray(contract.decode_chunk(q[row].tolist(), action[row].tolist(), joint_names=contract.joint_names))
        limited = np.asarray(contract.limit_targets(q[row].tolist(), decoded.tolist(), joint_names=contract.joint_names,
                                                   previous_target=previous[row].tolist()))
        if not np.allclose(decoded, absolute[row], rtol=0, atol=1e-12) or not np.allclose(limited, absolute[row], rtol=0, atol=1e-12):
            raise LabelProtocolError("masked labels disagree with the unchanged shared codec")
    metadata = json.loads(str(labels["metadata_json"]))
    if metadata.get("training_eligible") is not False or metadata.get("protocol") != LABEL_PROTOCOL:
        raise LabelProtocolError("diagnostic labels cannot grant training admission")
    if (canonical_sha256(metadata.get("contract")) != contract.contract_sha256
            or metadata.get("chunk_steps") != contract.chunk_steps or metadata.get("execute_steps") != contract.execute_steps
            or metadata.get("minimum_valid_prefix") != contract.execute_steps):
        raise LabelProtocolError("label metadata differs from the unchanged execution contract")
    return action.copy(), pad.copy()


def masked_action_stats(action, action_is_pad):
    """Population stats over real label positions only, preserving joint axes."""
    values = np.asarray(action, dtype=np.float64)
    if values.ndim != 3:
        raise LabelProtocolError("actions must be [samples,horizon,joints]")
    pad = _mask(action_is_pad, values.shape[:2])
    real = values[~pad]
    if not len(real) or not np.isfinite(real).all():
        raise LabelProtocolError("real action labels must be nonempty and finite")
    return {"mean": real.mean(axis=0), "std": real.std(axis=0), "real_position_count": len(real)}


def masked_action_diagnostics(labels, contract, prediction=None):
    """Concrete artifact consumer; stats/errors exclude every padding position."""
    action, pad = validate_masked_labels(labels, contract)
    stats = masked_action_stats(action, pad)
    result = {"protocol": LABEL_PROTOCOL, "training_eligible": False,
              "action_mean": stats["mean"].tolist(), "action_std": stats["std"].tolist(),
              "real_position_count": stats["real_position_count"], "sample_count": len(action)}
    if prediction is not None:
        result["errors"] = masked_action_errors(prediction, action, pad)
    return result


def masked_action_errors(prediction, target, action_is_pad):
    predicted, actual = np.asarray(prediction), np.asarray(target)
    if predicted.shape != actual.shape or actual.ndim != 3:
        raise LabelProtocolError("prediction and target shapes must match [samples,horizon,joints]")
    pad = _mask(action_is_pad, actual.shape[:2])
    errors = predicted[~pad] - actual[~pad]
    if not errors.size or not np.isfinite(errors).all():
        raise LabelProtocolError("real prediction errors must be finite and nonempty")
    last = (~pad).sum(axis=1) - 1
    endpoint = predicted[np.arange(len(last)), last] - actual[np.arange(len(last)), last]
    return {"mae_rad": float(np.abs(errors).mean()), "rmse_rad": float(np.sqrt(np.square(errors).mean())),
            "last_real_endpoint_mae_rad": float(np.abs(endpoint).mean()),
            "real_position_count": int((~pad).sum())}


def masked_smooth_l1_loss(prediction, target, action_is_pad):
    """Torch adapter with zero value/gradient contribution from padding slots."""
    import torch
    import torch.nn.functional as functional
    if prediction.shape != target.shape or target.ndim != 3:
        raise LabelProtocolError("prediction and target must match [samples,horizon,joints]")
    if action_is_pad.dtype != torch.bool or tuple(action_is_pad.shape) != tuple(target.shape[:2]):
        raise LabelProtocolError("torch padding mask must be explicit boolean [samples,horizon]")
    pad = action_is_pad
    if torch.any((~pad[:, 1:]) & pad[:, :-1]) or torch.any(pad.all(dim=1)):
        raise LabelProtocolError("real actions must form a nonempty contiguous prefix")
    real_prediction, real_target = prediction[~pad], target[~pad]
    if real_target.numel() == 0 or not torch.isfinite(real_prediction).all() or not torch.isfinite(real_target).all():
        raise LabelProtocolError("real loss values must be finite and nonempty")
    return functional.smooth_l1_loss(real_prediction, real_target)
