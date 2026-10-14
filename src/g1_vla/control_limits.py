"""Source-derived simulation command limits and shared-contract diagnostics.

Recorded commands, finite-difference motion and measured qvel are deliberately
reported separately. Historical render-rate samples cannot establish a motor's
hardware velocity limit or the true peak between samples. This module performs
no physics, collection or training and never changes the active configuration.
"""

import copy
import json
import math
from pathlib import Path

from .artifacts import sha256_file
from .contracts import ActionContract, ContractError, canonical_sha256, read_joint_order


class ControlLimitError(ValueError):
    """A source cannot justify the requested command-limit audit."""


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ControlLimitError("NumPy is required for raw command-limit audits") from exc
    return np


def _within(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if path.is_symlink():
        raise ControlLimitError("audit sources must not be symlinks")
    resolved = path.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ControlLimitError("audit source must remain inside the project") from exc
    return resolved


def _stats(values):
    np = _numpy()
    if not len(values):
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": len(values),
        **{label: np.percentile(values, percentile, axis=0).tolist()
           for label, percentile in (("p50", 50), ("p95", 95), ("p99", 99))},
        "max": np.max(values, axis=0).tolist(),
    }


def _group_summary(values, joint_names):
    np = _numpy()
    result = {}
    for label, predicate in (
            ("body_arm", lambda name: "hand_" not in name),
            ("hand", lambda name: "hand_" in name),
            ("right_arm", lambda name: name.startswith("right_") and "hand_" not in name),
            ("right_hand", lambda name: name.startswith("right_") and "hand_" in name)):
        indices = [index for index, name in enumerate(joint_names) if predicate(name)]
        selected = values[:, indices].ravel() if len(values) else np.array([])
        result[label] = _stats(selected)
    return result


def command_motion_arrays(arrays, joint_names):
    """Validate and map historical NPZ by identity, with actual sample deltas.

    time_s labels in the historical exporter precede mj_step while qpos is
    captured afterwards. Adjacent command speeds remain diagnostic evidence;
    these arrays are not asserted to be next-control-step training pairs.
    """
    np = _numpy()
    required = ("joint_names", "time_s", "joint_position_rad", "joint_velocity_rad_s",
                "action_joint_position_rad", "task_phase", "assist_active")
    if any(key not in arrays for key in required):
        raise ControlLimitError("raw source is missing command/state/timing/phase evidence")
    names = [str(value) for value in np.asarray(arrays["joint_names"]).tolist()]
    if len(set(names)) != len(names) or any(name not in names for name in joint_names):
        raise ControlLimitError("raw source joint identities are missing or duplicated")
    indices = [names.index(name) for name in joint_names]
    time = np.asarray(arrays["time_s"], dtype=np.float64)
    if time.ndim != 1 or len(time) < 2 or not np.all(np.isfinite(time)) or time[0] < 0:
        raise ControlLimitError("raw time_s must contain at least two finite non-negative samples")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ControlLimitError("raw timestamps must be strictly increasing")
    count = len(time)
    mapped = {}
    for key in ("joint_position_rad", "joint_velocity_rad_s", "action_joint_position_rad"):
        matrix = np.asarray(arrays[key], dtype=np.float64)
        if matrix.shape != (count, len(names)) or not np.all(np.isfinite(matrix)):
            raise ControlLimitError("{} must match timestamps and joint identities with finite values".format(key))
        mapped[key] = matrix[:, indices]
    phase = np.asarray(arrays["task_phase"])
    assist = np.asarray(arrays["assist_active"])
    if phase.shape != (count,) or not np.issubdtype(phase.dtype, np.integer) or np.any((phase < 0) | (phase > 7)):
        raise ControlLimitError("historical task_phase must contain integer phases 0 through 7")
    if assist.shape != (count,) or np.any(assist != 0):
        raise ControlLimitError("assisted grasp evidence cannot calibrate a strict command envelope")
    command, position, velocity = (mapped[key] for key in (
        "action_joint_position_rad", "joint_position_rad", "joint_velocity_rad_s"))
    delta = np.diff(command, axis=0)
    # Phase zero contains recovery motion as well as waiting. Split by recorded
    # adjacent command changes rather than claiming all phase zero is idle.
    startup = phase[1:] == 0
    changing = np.any(np.abs(delta) > 1e-8, axis=1)
    masks = {
        "all": np.ones(count - 1, dtype=bool),
        "startup_wait": startup & ~changing,
        "startup_recovery_motion": startup & changing,
        "effective_grasp_lift": np.isin(phase[1:], (1, 2, 3, 4)),
        "place_retreat": np.isin(phase[1:], (5, 6, 7)),
    }
    values = {
        "adjacent_command_delta_rad": np.abs(delta),
        "sampled_command_velocity_rad_s": np.abs(delta / dt[:, None]),
        "finite_difference_tracking_velocity_rad_s": np.abs(np.diff(position, axis=0) / dt[:, None]),
        "recorded_qvel_rad_s": np.abs(velocity[1:]),
        "same_frame_command_tracking_error_rad": np.abs(command[1:] - position[1:]),
    }
    return {"time_s": time, "dt_s": dt, "command": command, "position": position,
            "masks": masks, "values": values}


def summarize_command_motion(arrays, joint_names):
    mapped = command_motion_arrays(arrays, joint_names)
    result = {
        "samples": len(mapped["time_s"]),
        "sample_period_s": _stats(mapped["dt_s"]),
        "groups": {},
    }
    for label, mask in mapped["masks"].items():
        result["groups"][label] = {
            "interval_count": int(mask.sum()),
            "per_joint": {key: _stats(values[mask]) for key, values in mapped["values"].items()},
            "arm_hand": {key: _group_summary(values[mask], joint_names)
                         for key, values in mapped["values"].items()},
        }
    return result


def _contract_application(mapped, joint_names, config, root):
    """Audit clipping with the same encode/execute method used by PDController.

    This is a sampled replay diagnostic and does not certify historical training
    alignment. All 8 future targets share the current measured qpos anchor; only
    the first 4 limited commands enter the comparison.
    """
    np = _numpy()
    try:
        contract = ActionContract.from_config(config, project_root=root)
    except (ContractError, ValueError) as exc:
        return {"status": "unresolved", "message": str(exc)}
    counts = {"chunks": 0, "conflicting_chunks": 0, "executed_commands": 0,
              "commands_clipped": 0, "velocity_violations": 0, "anchor_violations": 0}
    max_change = 0.0
    messages = []
    mask = mapped["masks"]["effective_grasp_lift"]
    for index in range(0, len(mapped["time_s"]) - contract.chunk_steps, contract.execute_steps):
        if not bool(mask[index]):
            continue
        counts["chunks"] += 1
        anchor = mapped["position"][index].tolist()
        previous = mapped["command"][index].tolist()
        requested = mapped["command"][index + 1:index + 1 + contract.chunk_steps].tolist()
        delta = contract.encode_chunk(anchor, requested, joint_names=joint_names)
        try:
            actual = contract.executable_targets(anchor, delta, joint_names=joint_names, previous_target=previous)
        except ContractError as exc:
            counts["conflicting_chunks"] += 1
            if len(messages) < 4:
                messages.append({"sample": index, "message": str(exc)})
            continue
        for target, wanted in zip(actual, requested):
            movement = np.abs(np.asarray(target) - previous)
            amplitude = np.abs(np.asarray(target) - anchor)
            difference = np.abs(np.asarray(target) - wanted)
            counts["executed_commands"] += 1
            counts["commands_clipped"] += int(np.any(difference > 1e-7))
            counts["velocity_violations"] += int(np.any(movement > np.asarray(contract.max_velocity_rad_s) / contract.control_hz + 1e-10))
            counts["anchor_violations"] += int(np.any(amplitude > np.asarray(contract.max_anchor_delta_rad) + 1e-10))
            max_change = max(max_change, float(difference.max()))
            previous = list(target)
    return {"status": "passed" if not any(counts[key] for key in (
            "conflicting_chunks", "velocity_violations", "anchor_violations")) else "failed",
            **counts, "maximum_clipping_rad": max_change,
            "conflicts": messages, "contract_sha256": contract.contract_sha256}


def audit_control_limits(root, config):
    """Return historical green/right diagnostics and a conservative envelope.

    The envelope is each joint's observed successful grasp/lift command peak
    rounded upward to 0.01 rad/s. Exactly uncommanded joints receive zero, which
    means hold the prior command. No guessed minimum or hardware spec is added.
    Failed and startup/recovery/transport segments remain separate diagnostics.
    """
    np = _numpy()
    root = Path(root).resolve()
    if config["task"]["hand"] != "right" or config["task"]["object_ids"] != ["green_cube"] or config["task"]["mode"] != "grasp_lift":
        raise ControlLimitError("this source-derived envelope applies only to green/right grasp_lift")
    metadata_path = _within(root, config["action"]["joint_metadata_path"])
    joint_names = read_joint_order(json.loads(metadata_path.read_text(encoding="utf-8")))
    if canonical_sha256(list(joint_names)) != config["action"]["joint_order_sha256"]:
        raise ControlLimitError("audit joint order differs from the frozen action metadata")
    registry_path = _within(root, config["data"]["registry_path"])
    rows = [json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reports, source_arrays, rejected = [], [], []
    candidates = [row for row in rows if row.get("raw_role") == "expert_trajectory"
                  and row.get("target_object") == "green_cube"]
    skipped_other_hand = 0
    for row in candidates:
        try:
            relative = row.get("relative_raw_path")
            raw_path = _within(root, relative or row["raw_path"])
            # Read the sibling summary after project relocation. Absolute legacy
            # references in the registry need not be trusted or followed.
            summary_path = raw_path.parent / "language_pick_place_summary.json"
            summary_path = _within(root, summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("active_arm") != "right":
                skipped_other_hand += 1
                continue
            if summary.get("target_object") != "green_cube" or summary.get("grasp_mode") != "contact":
                raise ControlLimitError("source summary does not establish green/right strict contact")
            actual_hash = sha256_file(raw_path)
            if actual_hash != row.get("raw_sha256"):
                raise ControlLimitError("source NPZ hash differs from the protected raw registry")
            with np.load(raw_path, allow_pickle=False) as archive:
                arrays = {key: archive[key].copy() for key in (
                    "joint_names", "time_s", "joint_position_rad", "joint_velocity_rad_s",
                    "action_joint_position_rad", "task_phase", "assist_active", "active_arm_index")}
                embedded_summary = json.loads(str(archive["metadata_json"].item()))
            for key in ("active_arm", "target_object", "grasp_mode", "passed", "grasp_physics_audit"):
                if embedded_summary.get(key) != summary.get(key):
                    raise ControlLimitError("sibling summary differs from protected NPZ metadata for {}".format(key))
            if arrays["active_arm_index"].shape != arrays["time_s"].shape or np.any(arrays["active_arm_index"] != 1):
                raise ControlLimitError("raw active_arm_index differs from the right-hand summary")
            mapped = command_motion_arrays(arrays, joint_names)
            successful = bool(summary.get("passed")) and bool(summary.get("grasp_physics_audit", {}).get("physical_grasp_passed"))
            report = {
                "path": str(raw_path.relative_to(root)), "sha256": actual_hash,
                "summary_sha256": sha256_file(summary_path),
                "regime": row.get("regime"), "historical_outcome": row.get("historical_outcome"),
                "historical_successful_contact_task": successful,
                "new_strict_teacher_gate_certified": False,
                "training_eligible": False, "controller_gain_scale": summary.get("controller_gain_scale"),
                "initial_state_recovery": summary.get("initial_state_recovery"),
                "motion": summarize_command_motion(arrays, joint_names),
            }
            reports.append(report)
            source_arrays.append((report, mapped))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            rejected.append({"path": row.get("relative_raw_path", row.get("raw_path")), "reason": str(exc)})
    successful = [(report, mapped) for report, mapped in source_arrays if report["historical_successful_contact_task"]]
    peak_samples = [mapped["values"]["sampled_command_velocity_rad_s"][mapped["masks"]["effective_grasp_lift"]]
                    for _, mapped in successful]
    peak_samples = [samples for samples in peak_samples if len(samples)]
    recommendation = None
    if peak_samples:
        peaks = np.concatenate(peak_samples).max(axis=0)
        envelope = (np.ceil(peaks * 100.0) / 100.0).tolist()
        recommendation = {
            "max_velocity_rad_s": envelope,
            "observed_peak_rad_s": peaks.tolist(),
            "frozen_joints": [name for name, value in zip(joint_names, envelope) if value == 0],
            "source_paths": [report["path"] for report, _ in successful],
            "rule": "successful_contact_grasp_lift_phases_1_to_4_peak_ceil_0.01_rad_s_zero_means_hold",
            "hardware_calibration": False,
        }
        candidate_config = copy.deepcopy(config)
        candidate_config["control"]["max_velocity_rad_s"] = envelope
        for report, mapped in source_arrays:
            report["recommended_contract_application"] = _contract_application(mapped, joint_names, candidate_config, root)
            report["configured_contract_application"] = _contract_application(mapped, joint_names, config, root)
    return {
        "schema_version": 1, "protocol": "g1_source_command_envelope_v1",
        "status": "passed" if recommendation is not None and not rejected else "incomplete",
        "joint_names": list(joint_names), "joint_order_sha256": canonical_sha256(list(joint_names)),
        "registry_sha256": sha256_file(registry_path), "candidate_green_sources": len(candidates),
        "skipped_other_hand_sources": skipped_other_hand,
        "audited_right_sources": len(reports), "successful_contact_sources": len(successful),
        "rejected_sources": rejected, "sources": reports, "recommendation": recommendation,
        "scope": "simulation_command_envelope_diagnostic_only", "training_started": False,
        "limitations": [
            "Historical right-hand green successes are recovery trajectories, not a home-start teacher gate.",
            "Render-rate samples measure averaged command velocity; substep peaks and hardware limits are not calibrated.",
            "Historical qpos is sampled after mj_step with a step-start time label; this audit certifies no training pair alignment.",
            "Historical success lacks the new independently measured 0.5-second lift hold gate.",
            "The migrated motor force bounds are a changed simulation profile, not velocity calibration.",
            "Anchor amplitude 0.08 rad can clip teacher commands even when the source-derived adjacent speed envelope is respected.",
        ],
    }
