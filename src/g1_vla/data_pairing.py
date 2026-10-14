"""Read-only raw pairing audit and the exact control-interval episode format.

Historical expert NPZs record *after* a physics step and only at video cadence,
while their controller changes target every physics tick. Shifting those rows
does not recover the next actually applied command. They remain diagnostic.
New episodes capture a pre-command sensor state, the limited applied target,
the real interval, and its post-state without resampling or clipping labels.
"""

import json
import math
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from .artifacts import sha256_file
from .contracts import canonical_sha256, read_joint_order
from .data_ingest import ingest_raw_registry
from .data_registry import detect_split_leakage


PAIRED_SCHEMA_VERSION = 1
PAIRED_PROTOCOL = "pre_command_observation_applied_target_interval_v1"
MAX_RAW_BYTES = 256 * 1024 * 1024


class PairingError(ValueError):
    """Raw evidence cannot establish the requested control correspondence."""


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise PairingError("raw pairing requires the simulation numpy dependency") from exc
    return np


def _matrix(value, rows, columns, name):
    np = _numpy()
    result = np.asarray(value)
    if result.shape != (rows, columns) or result.dtype.kind not in "fiu":
        raise PairingError("{} must be a numeric {} x {} matrix".format(name, rows, columns))
    if not np.isfinite(result).all():
        raise PairingError("{} contains non-finite values".format(name))
    return result.astype(np.float64, copy=True)


def _names(names, *, canonical=False):
    values = names.tolist() if hasattr(names, "tolist") else list(names)
    if (not values or any(not isinstance(name, str) or not name.strip() for name in values)
            or len(set(values)) != len(values) or canonical and len(values) != 31):
        raise PairingError("joint names must be unique strings and action31 must contain 31 names")
    return values


def select_action31(values, joint_names, action_joint_names):
    """Select by unique names, never positional slicing of 43D evidence."""
    np = _numpy()
    source, ordered = _names(joint_names), _names(action_joint_names, canonical=True)
    absent = [name for name in ordered if name not in source]
    if absent:
        raise PairingError("raw joint names omit {}".format(", ".join(absent)))
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != len(source) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise PairingError("joint evidence must be a finite matrix matching its names")
    return array[:, [source.index(name) for name in ordered]].copy()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PairingError("{} must be finite numeric evidence".format(name))
    return float(value)


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def build_paired_episode(observations: Sequence[Dict], command_targets: Sequence,
                         timing: Sequence[Dict], joint_names: Sequence[str], source: Dict,
                         *, physical_evidence: Optional[Sequence[Dict]] = None,
                         expected_interval_s: Optional[float] = None) -> Dict[str, Any]:
    """Return NPZ-safe arrays for actual, unmodified pre-state/applied-command pairs.

    ``timing`` rows require start_s/end_s/sequence_index/state_segment_id and
    restart_wait. post_joint_position/post_joint_velocity/post_time_s establish
    continuity. A changed segment invalidates cross-segment chunk construction;
    no interpolation, time normalization, target clipping or state replacement
    is performed. Sensor-incomplete acceptance runs may still be audited, but
    every episode remains training_eligible=False until a future reviewed gate.
    """
    np = _numpy()
    count, names = len(observations), _names(joint_names, canonical=True)
    if count < 1 or len(timing) != count or not isinstance(source, dict):
        raise PairingError("observations, timing and source must describe one nonempty episode")
    qpos = _matrix([row["joint_position"] for row in observations], count, 31, "observed position")
    qvel = _matrix([row["joint_velocity"] for row in observations], count, 31, "observed velocity")
    targets = _matrix(command_targets, count, 31, "applied target")
    times = np.asarray([_number(row["timestamp_s"], "observation time") for row in observations], dtype=np.float64)
    starts = np.asarray([_number(row["start_s"], "command start") for row in timing], dtype=np.float64)
    ends = np.asarray([_number(row["end_s"], "command end") for row in timing], dtype=np.float64)
    if np.any(ends <= starts) or np.any(np.abs(times - starts) > 1e-9):
        raise PairingError("observations must precede their command at the exact interval start")
    if count > 1 and (np.any(np.diff(starts) <= 0) or np.any(starts[1:] < ends[:-1] - 1e-9)):
        raise PairingError("control intervals must be ordered and cannot overlap")
    if expected_interval_s is not None:
        interval = _number(expected_interval_s, "expected interval")
        if interval <= 0 or not np.allclose(ends - starts, interval, rtol=0, atol=1e-9):
            raise PairingError("actual control duration differs from the configured interval")
    for key in ("sequence_index", "state_segment_id"):
        if any(isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] < 0 for row in timing):
            raise PairingError("{} must be an explicit nonnegative integer".format(key))
    if any(not isinstance(row.get("restart_wait"), bool) for row in timing):
        raise PairingError("restart_wait must be explicitly recorded for every command")
    sequence = np.asarray([row["sequence_index"] for row in timing], dtype=np.int64)
    segments = np.asarray([row["state_segment_id"] for row in timing], dtype=np.int64)
    waits = np.asarray([row["restart_wait"] for row in timing], dtype=np.bool_)
    if count > 1 and np.any(np.diff(sequence) != 1):
        raise PairingError("missing control commands cannot be silently skipped")
    post_complete = all(all(key in row for key in ("post_joint_position", "post_joint_velocity", "post_time_s")) for row in timing)
    continuity = np.zeros(count, dtype=np.bool_)
    arrays = {
        "paired_schema_version": np.asarray(PAIRED_SCHEMA_VERSION, dtype=np.int64),
        "joint_names": np.asarray(names), "observation_joint_position_rad": qpos,
        "observation_joint_velocity_rad_s": qvel, "action_joint_position_rad": targets,
        "observation_time_s": times, "command_start_time_s": starts, "command_end_time_s": ends,
        "command_sequence_index": sequence, "state_segment_id": segments, "restart_wait": waits,
    }
    issues = []
    if post_complete:
        post_q = _matrix([row["post_joint_position"] for row in timing], count, 31, "post position")
        post_v = _matrix([row["post_joint_velocity"] for row in timing], count, 31, "post velocity")
        post_t = np.asarray([_number(row["post_time_s"], "post time") for row in timing])
        if np.any(np.abs(post_t - ends) > 1e-9):
            raise PairingError("post-state timestamp must equal the actual interval end")
        continuity[:] = True
        for index in range(1, count):
            if segments[index] != segments[index - 1]:
                continuity[index] = False
                issues.append("explicit_state_segment_boundary")
            elif (abs(starts[index] - ends[index - 1]) > 1e-9
                  or not np.array_equal(qpos[index], post_q[index - 1])
                  or not np.array_equal(qvel[index], post_v[index - 1])):
                raise PairingError("hidden state jump or control gap at command {}".format(index))
        arrays.update(post_joint_position_rad=post_q, post_joint_velocity_rad_s=post_v, post_time_s=post_t)
    else:
        issues.append("post_state_continuity_unproven")
    # Movement diagnostics do not manufacture a command for a held-start row.
    correction = np.max(np.abs(targets - qpos), axis=1) > 1e-7
    arrays["correction_nonzero"] = correction
    arrays["pairing_valid"] = continuity & ~waits
    arrays["continuity_from_previous"] = continuity
    if bool(waits.any()):
        issues.append("restart_wait_excluded")
    if not bool(correction[0]):
        issues.append("first_command_has_no_motion_correction")
    sensor_complete = all(all(key in row for key in ("images", "finger_sites", "tactile", "language")) for row in observations)
    if sensor_complete:
        image_keys = sorted(observations[0]["images"])
        if len(image_keys) != 3 or any(sorted(row["images"]) != image_keys for row in observations):
            raise PairingError("complete sensor recordings require the same three camera keys")
        for index, key in enumerate(image_keys):
            images = np.stack([np.asarray(row["images"][key]) for row in observations])
            if images.dtype != np.uint8 or images.ndim != 4 or min(images.shape[1:3]) < 1 or images.shape[-1] != 3:
                raise PairingError("camera pixels must be uint8 T x H x W x RGB")
            arrays["images_camera_{}".format(index)] = images
        arrays["camera_keys"] = np.asarray(image_keys)
        arrays["finger_sites"] = _matrix([row["finger_sites"] for row in observations], count, 108, "finger sites")
        arrays["tactile"] = _matrix([row["tactile"] for row in observations], count, 1, "tactile")
        if any(not isinstance(row["language"], str) or not row["language"].strip() for row in observations):
            raise PairingError("language must be nonempty for each sensor frame")
        arrays["language_instruction"] = np.asarray([row["language"] for row in observations])
    else:
        issues.append("sensor_observation_incomplete")
    lineage_complete = all(isinstance(source.get(key), str) and source[key].strip()
                           for key in ("parent_root", "source_rollout", "parent_rollout_id", "layout_fingerprint", "regime"))
    fingerprint = source.get("layout_fingerprint")
    lineage_complete = bool(lineage_complete and len(fingerprint) == 64
                            and all(ch in "0123456789abcdef" for ch in fingerprint))
    if source.get("regime") in ("direct_takeover", "direct_continuation", "waypoint_recovery", "safe_waypoint"):
        frame = source.get("takeover_frame")
        lineage_complete &= isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0
    elif source.get("regime") != "expert":
        lineage_complete = False
    if not lineage_complete:
        issues.append("lineage_incomplete")
    contact = np.zeros(count, dtype=np.bool_)
    lift = np.zeros(count, dtype=np.float64)
    hold = np.zeros(count, dtype=np.float64)
    assisted = np.ones(count, dtype=np.bool_)
    finite = np.zeros(count, dtype=np.bool_)
    joint_safe = np.zeros(count, dtype=np.bool_)
    safety_complete = physical_evidence is not None
    if physical_evidence is not None:
        if len(physical_evidence) != count:
            raise PairingError("physical evidence must correspond to every control interval")
        for index, row in enumerate(physical_evidence):
            if not isinstance(row.get("contact"), bool) or not isinstance(row.get("assisted"), bool):
                raise PairingError("contact and assisted must be measured booleans")
            contact[index], assisted[index] = row["contact"], row["assisted"]
            lift[index] = _number(row["lift_height_m"], "measured lift")
            hold[index] = _number(row["hold_seconds"], "measured continuous hold")
            if hold[index] < 0:
                raise PairingError("measured hold cannot be negative")
            for field, values in (("finite", finite), ("joint_safe", joint_safe)):
                if field not in row:
                    safety_complete = False
                    continue
                if not isinstance(row[field], bool):
                    raise PairingError("{} must be an explicit measured boolean".format(field))
                values[index] = row[field]
    else:
        issues.append("physical_measurements_missing")
    if not safety_complete:
        issues.append("physical_safety_unproven")
    positive = contact & (lift >= 0.1) & (hold >= 0.5) & ~assisted & finite & joint_safe
    arrays.update(physical_contact_active=contact, measured_lift_height_m=lift,
                  measured_hold_seconds=hold, assist_active=assisted,
                  physical_numeric_finite=finite, physical_joint_safe=joint_safe,
                  frame_physical_positive=positive)
    metadata = {
        "schema_version": PAIRED_SCHEMA_VERSION, "pairing_protocol": PAIRED_PROTOCOL,
        "joint_order_sha256": canonical_sha256(names), "source": source,
        "observation_sampling": "before_applied_control_command", "command_semantics": "limited_absolute_joint_target",
        "command_held_for_interval": True, "time_resampled": False, "labels_modified": False,
        "post_state_complete": post_complete, "sensor_complete": sensor_complete,
        "physical_safety_complete": bool(safety_complete),
        "lineage_complete": lineage_complete, "pairing_eligible": bool(post_complete and lineage_complete),
        "training_eligible": False, "split": "diagnostic", "review_issues": sorted(set(issues)),
        "frame_positive_scope": "measured_contact_lift_continuous_hold_numeric_finite_joint_safe_for_this_interval_only",
        "episode_strict_physical_success": bool(positive.any()),
    }
    arrays["metadata_json"] = np.asarray(_json(metadata))
    return arrays


def _load_npz(path: Path):
    np = _numpy()
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_RAW_BYTES:
        raise PairingError("raw file is missing, a symlink or exceeds the bounded read limit")
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_RAW_BYTES:
            raise PairingError("uncompressed raw archive exceeds the bounded read limit")
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    if sum(value.nbytes for value in result.values()) > MAX_RAW_BYTES:
        raise PairingError("uncompressed raw arrays exceed the bounded read limit")
    return result


def encode_paired_chunks(episode: Dict, contract) -> Dict[str, Any]:
    """Encode real future targets with the same ActionContract as execution.

    Only fully recorded chunks within a continuous segment are returned. This
    is an in-memory protocol helper; it neither creates training caches nor
    pads the end of an episode with invented actions.
    """
    np = _numpy()
    names = episode["joint_names"].tolist()
    contract.assert_joint_order(names)
    count = len(episode["observation_joint_position_rad"])
    indices, chunks, consistent = [], [], []
    for start in range(max(0, count - contract.chunk_steps + 1)):
        end = start + contract.chunk_steps
        if (not bool(episode["pairing_valid"][start:end].all())
                or not bool(episode["continuity_from_previous"][start + 1:end].all())
                or len(set(episode["state_segment_id"][start:end].tolist())) != 1):
            continue
        indices.append(start)
        anchor = episode["observation_joint_position_rad"][start].tolist()
        targets = episode["action_joint_position_rad"][start:end].tolist()
        chunks.append(contract.encode_chunk(anchor, targets, joint_names=names))
        previous = episode["action_joint_position_rad"][start - 1].tolist() if start else anchor
        try:
            executable = contract.limit_targets(anchor, targets, joint_names=names, previous_target=previous)
            consistent.append(bool(np.allclose(executable, targets, rtol=0, atol=1e-12)))
        except ValueError:
            consistent.append(False)
    return {"anchor_record_index": np.asarray(indices, dtype=np.int64),
            "action_delta_chunk_rad": np.asarray(chunks, dtype=np.float64).reshape(-1, contract.chunk_steps, 31),
            "action_contract_consistent": np.asarray(consistent, dtype=np.bool_)}


def _exact_error(first, second):
    np = _numpy()
    return {"exact": bool(np.array_equal(first, second)), "max_abs_error": float(np.max(np.abs(first - second)))}


def _takeover_evidence(raw_path, arrays, names, base, row):
    np = _numpy()
    visited_path = raw_path.parent / "policy_visited_state.npz"
    if not visited_path.is_file():
        return None
    visited = _load_npz(visited_path)
    vq = select_action31(visited["observation_joint_position_rad"], visited["joint_names"], names)[0]
    vv = select_action31(visited["observation_joint_velocity_rad_s"], visited["joint_names"], names)[0]
    q_key = "observation_joint_position_rad" if "observation_joint_position_rad" in arrays else "joint_position_rad"
    v_key = "observation_joint_velocity_rad_s" if "observation_joint_velocity_rad_s" in arrays else "joint_velocity_rad_s"
    q = select_action31(arrays[q_key], arrays["joint_names"], names)
    v = select_action31(arrays[v_key], arrays["joint_names"], names)
    command = select_action31(arrays["action_joint_position_rad"], arrays["joint_names"], names)
    evidence = {"visited_raw_path": str(visited_path), "visited_sha256": sha256_file(visited_path),
                "teacher_observed_start_position": _exact_error(q[0], vq),
                "teacher_observed_start_velocity": _exact_error(v[0], vv),
                "teacher_first_target_max_delta_rad": float(np.max(np.abs(command[0] - vq))),
                "next_recorded_target_max_delta_rad": float(np.max(np.abs(command[1] - vq))) if len(command) > 1 else None,
                "visited_start_preserved": False, "parent_exact": False}
    source = row.get("source_rollout")
    if source:
        path = Path(source)
        if not path.is_absolute():
            path = base / path
        try:
            path.resolve().relative_to(base.resolve())
            if path.is_dir():
                path = path / "trajectory.npz"
            parent = _load_npz(path)
            frame = row.get("takeover_frame")
            if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
                raise PairingError("source takeover frame is unresolved")
            pq = select_action31(parent["observation_joint_position_rad"], parent["joint_names"], names)[frame]
            pv = select_action31(parent["observation_joint_velocity_rad_s"], parent["joint_names"], names)[frame]
            evidence["parent_visited_position"] = _exact_error(pq, vq)
            evidence["parent_visited_velocity"] = _exact_error(pv, vv)
            evidence["parent_exact"] = evidence["parent_visited_position"]["exact"] and evidence["parent_visited_velocity"]["exact"]
            if "source_object_position_m" in visited and "object_position_m" in parent:
                # Compare the saved policy precision without rounding poses.
                parent_objects = parent["object_position_m"][frame]
                evidence["parent_visited_objects"] = _exact_error(parent_objects, visited["source_object_position_m"].astype(parent_objects.dtype))
                evidence["parent_exact"] &= evidence["parent_visited_objects"]["exact"]
        except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile) as error:
            evidence["parent_error"] = str(error)
    evidence["visited_start_preserved"] = bool(evidence["teacher_observed_start_position"]["exact"]
                                               and evidence["teacher_observed_start_velocity"]["exact"])
    return evidence


def _new_schema_audit(arrays, names, interval):
    np = _numpy()
    metadata = json.loads(str(arrays["metadata_json"]))
    if metadata.get("pairing_protocol") != PAIRED_PROTOCOL or int(arrays["paired_schema_version"]) != PAIRED_SCHEMA_VERSION:
        raise PairingError("unknown paired raw protocol")
    ordered = _names(arrays["joint_names"], canonical=True)
    if ordered != list(names) or metadata.get("joint_order_sha256") != canonical_sha256(list(names)):
        raise PairingError("paired raw action31 order differs from configured metadata")
    q = arrays["observation_joint_position_rad"]
    count = len(q)
    observations = [{"joint_position": q[i], "joint_velocity": arrays["observation_joint_velocity_rad_s"][i],
                     "timestamp_s": float(arrays["observation_time_s"][i])} for i in range(count)]
    if metadata.get("sensor_complete"):
        keys = arrays["camera_keys"].tolist()
        for index, observation in enumerate(observations):
            observation.update(images={key: arrays["images_camera_{}".format(k)][index] for k, key in enumerate(keys)},
                               finger_sites=arrays["finger_sites"][index], tactile=arrays["tactile"][index],
                               language=str(arrays["language_instruction"][index]))
    timing = [{"start_s": float(arrays["command_start_time_s"][i]), "end_s": float(arrays["command_end_time_s"][i]),
               "sequence_index": int(arrays["command_sequence_index"][i]), "state_segment_id": int(arrays["state_segment_id"][i]),
               "restart_wait": bool(arrays["restart_wait"][i])} for i in range(count)]
    if metadata.get("post_state_complete"):
        for index, item in enumerate(timing):
            item.update(post_joint_position=arrays["post_joint_position_rad"][index],
                        post_joint_velocity=arrays["post_joint_velocity_rad_s"][index],
                        post_time_s=float(arrays["post_time_s"][index]))
    physical = [{"contact": bool(arrays["physical_contact_active"][i]), "assisted": bool(arrays["assist_active"][i]),
                 "lift_height_m": float(arrays["measured_lift_height_m"][i]),
                 "hold_seconds": float(arrays["measured_hold_seconds"][i])} for i in range(count)]
    for key, field in (("physical_numeric_finite", "finite"), ("physical_joint_safe", "joint_safe")):
        if key not in arrays:
            # Earlier paired archives remain diagnostic. Missing evidence never
            # becomes a true default and the source archive is not rewritten.
            continue
        values = arrays[key]
        if values.shape != (count,) or values.dtype.kind != "b":
            raise PairingError("{} must contain one explicit boolean per interval".format(key))
        for index, row in enumerate(physical):
            row[field] = bool(values[index])
    verified = build_paired_episode(observations, arrays["action_joint_position_rad"], timing, names,
                                    metadata["source"], physical_evidence=physical, expected_interval_s=interval)
    if metadata.get("physical_safety_complete") is False:
        # False masks encode missing evidence conservatively. Preserve that
        # original provenance warning rather than calling the masks measured.
        verified_metadata = json.loads(str(verified["metadata_json"]))
        verified_metadata["physical_safety_complete"] = False
        verified_metadata["review_issues"] = sorted(set(verified_metadata["review_issues"] + ["physical_safety_unproven"]))
        verified["metadata_json"] = np.asarray(_json(verified_metadata))
    for key in ("pairing_valid", "correction_nonzero", "continuity_from_previous", "frame_physical_positive",
                "physical_numeric_finite", "physical_joint_safe"):
        if key not in arrays and key in ("physical_numeric_finite", "physical_joint_safe"):
            continue
        if not np.array_equal(arrays[key], verified[key]):
            raise PairingError("stored {} disagrees with actual raw evidence".format(key))
    return verified


def audit_raw_pairing(root: Union[str, Path], config: Dict, *, run_dir: Optional[Union[str, Path]] = None,
                      records: Optional[Sequence[Dict]] = None) -> Dict[str, Any]:
    """Audit every registered raw without mutating source or its registry.

    Optional run_dir stores exclusive JSON evidence and unmodified action31
    diagnostic samples. Legacy next-sampled targets are explicitly named as
    such; no next-actual-control labels are invented. New schema is revalidated
    and checked for parent/layout split leakage before any eligibility report.
    """
    np = _numpy()
    base = Path(root).expanduser().resolve()
    path = Path(config["action"]["joint_metadata_path"])
    if not path.is_absolute():
        path = base / path
    names = list(read_joint_order(json.loads(path.read_text(encoding="utf-8"))))
    expected_hash = config["action"].get("joint_order_sha256")
    if expected_hash and expected_hash != canonical_sha256(names):
        raise PairingError("configured joint order hash differs from source metadata")
    interval = 1.0 / float(config["control"]["control_hz"])
    registered = list(records) if records is not None else ingest_raw_registry(base)["records"]
    destination = None
    if run_dir is not None:
        destination = Path(run_dir).expanduser().resolve()
        try:
            destination.relative_to(base)
        except ValueError:
            pass
        if destination == base or destination.name == "outputs":
            raise PairingError("pairing output must be an independent run directory")
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / "raw_pairing.json").exists() or (destination / "paired_evidence").exists():
            raise PairingError("pairing results are immutable; choose a new run directory")
        (destination / "paired_evidence").mkdir()
    results, errors, new_rows = [], [], []
    for record in registered:
        raw = Path(record["raw_path"])
        if not raw.is_absolute():
            raw = base / raw
        result = {"record_id": record.get("record_id"), "raw_path": str(raw), "raw_role": record.get("raw_role"),
                  "training_eligible": False, "split": record.get("split", "diagnostic"),
                  "group_id": record.get("group_id"), "layout_fingerprint": record.get("layout_fingerprint"),
                  "pairing_eligible": False, "frame_positive_count": 0, "review_issues": []}
        try:
            raw.resolve().relative_to(base)
            before = raw.stat()
            arrays = _load_npz(raw)
            digest = sha256_file(raw)
            result["raw_sha256"] = digest
            if record.get("raw_sha256") and record["raw_sha256"] != digest:
                raise PairingError("raw differs from protected registry hash")
            if raw.name == "policy_visited_state.npz":
                qkey, vkey = "observation_joint_position_rad", "observation_joint_velocity_rad_s"
                evidence = {"joint_names": np.asarray(names),
                            qkey: select_action31(arrays[qkey], arrays["joint_names"], names),
                            vkey: select_action31(arrays[vkey], arrays["joint_names"], names)}
                result.update(pairing_status="saved_state_without_command", sample_count=len(evidence[qkey]),
                              review_issues=["visited_state_is_not_a_command_label"])
            elif "paired_schema_version" in arrays:
                evidence = _new_schema_audit(arrays, names, interval)
                metadata = json.loads(str(evidence["metadata_json"]))
                result.update(pairing_status="exact_control_interval", sample_count=len(evidence["pairing_valid"]),
                              pairing_eligible=metadata["pairing_eligible"], valid_pair_count=int(evidence["pairing_valid"].sum()),
                              frame_positive_count=int(evidence["frame_physical_positive"].sum()),
                              sensor_complete=metadata["sensor_complete"], review_issues=metadata["review_issues"],
                              source=metadata["source"], episode_strict_physical_success=metadata["episode_strict_physical_success"])
                source = metadata["source"]
                if source.get("parent_root") != str(base):
                    result["pairing_eligible"] = False
                    result["review_issues"].append("source_root_does_not_match_raw_project")
                source_path = Path(source.get("source_rollout", ""))
                if not source_path.is_absolute():
                    source_path = base / source_path
                try:
                    source_path.resolve().relative_to(base)
                    source_exists = source_path.is_file() and not source_path.is_symlink()
                    if source_path.is_dir() and not source_path.is_symlink():
                        source_exists = any((source_path / filename).is_file() for filename in ("trajectory.npz", "expert_pick_place_episode.npz"))
                except ValueError:
                    source_exists = False
                if not source_exists:
                    result["pairing_eligible"] = False
                    result["review_issues"].append("new_parent_source_unresolved")
                if record.get("layout_fingerprint") != source.get("layout_fingerprint"):
                    result["pairing_eligible"] = False
                    result["review_issues"].append("new_layout_identity_not_independently_verified")
                new_rows.append({**record, "parent_root": source.get("parent_root"), "source_rollout": source.get("source_rollout"),
                                 "layout_fingerprint": source.get("layout_fingerprint"), "split": source.get("split", "diagnostic")})
                result["takeover_audit"] = _takeover_evidence(raw, arrays, names, base, record)
                if result["takeover_audit"] and not (result["takeover_audit"]["parent_exact"] and result["takeover_audit"]["visited_start_preserved"]):
                    result["pairing_eligible"] = False
                    result["review_issues"].append("takeover_start_not_exact")
            else:
                qkey = "observation_joint_position_rad" if "observation_joint_position_rad" in arrays else "joint_position_rad"
                vkey = "observation_joint_velocity_rad_s" if "observation_joint_velocity_rad_s" in arrays else "joint_velocity_rad_s"
                q, v = select_action31(arrays[qkey], arrays["joint_names"], names), select_action31(arrays[vkey], arrays["joint_names"], names)
                commands = select_action31(arrays["action_joint_position_rad"], arrays["joint_names"], names)
                if q.shape != v.shape or q.shape != commands.shape:
                    raise PairingError("position, velocity and command arrays have different sample counts")
                evidence = {"joint_names": np.asarray(names), "recorded_joint_position_rad": q,
                            "recorded_joint_velocity_rad_s": v, "recorded_command_target_rad": commands,
                            "observation_record_index": np.arange(max(0, len(q) - 1), dtype=np.int64),
                            "next_sampled_command_record_index": np.arange(1, len(q), dtype=np.int64),
                            "next_sampled_command_target_rad": commands[1:],
                            "next_sampled_command_delta_rad": commands[1:] - q[:-1]}
                result.update(pairing_status="historical_sampled_only", sample_count=len(q),
                              review_issues=["next_actual_control_command_not_recorded", "post_state_continuity_unproven", "historical_diagnostic_only"])
                if "time_s" in arrays:
                    times = np.asarray(arrays["time_s"])
                    if times.shape != (len(q),) or not np.isfinite(times).all() or len(times) > 1 and np.any(np.diff(times) <= 0):
                        raise PairingError("historical timestamps are invalid or unordered")
                    evidence["recorded_time_s"] = times.copy()
                    result["recorded_interval_min_s"] = float(np.diff(times).min()) if len(times) > 1 else None
                    result["recorded_interval_max_s"] = float(np.diff(times).max()) if len(times) > 1 else None
                    result["review_issues"].append("recorded_time_is_pre_tick_label_for_post_tick_state")
                else:
                    result["review_issues"].append("actual_control_timestamps_missing")
                takeover = _takeover_evidence(raw, arrays, names, base, record)
                if takeover:
                    result["takeover_audit"] = takeover
                    if not takeover["visited_start_preserved"]:
                        result["review_issues"].append("teacher_start_changes_visited_state")
                    if not takeover["parent_exact"]:
                        result["review_issues"].append("visited_state_parent_exact_unproven")
                    if takeover["teacher_first_target_max_delta_rad"] < 1e-7:
                        result["review_issues"].append("first_command_is_hold_not_correction")
            after = raw.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise PairingError("raw changed while auditing")
            if destination is not None:
                evidence_path = destination / "paired_evidence" / (canonical_sha256(str(raw))[:24] + ".npz")
                evidence["audit_metadata_json"] = np.asarray(_json(result))
                with evidence_path.open("xb") as stream:
                    np.savez_compressed(stream, **evidence)
                result["derived_evidence_path"] = str(evidence_path)
                result["derived_evidence_sha256"] = sha256_file(evidence_path)
        except (OSError, ValueError, KeyError, IndexError, TypeError, zipfile.BadZipFile) as error:
            result["pairing_status"] = "unproven"
            result["pairing_eligible"] = False
            result["review_issues"].append("raw_pairing_unproven")
            errors.append({"raw_path": str(raw), "error_type": type(error).__name__, "message": str(error)})
        result["review_issues"] = sorted(set(result["review_issues"]))
        results.append(result)
    leakage = detect_split_leakage(new_rows)
    if leakage:
        for item in results:
            if item["pairing_status"] == "exact_control_interval":
                item["pairing_eligible"] = False
                item["review_issues"].append("new_record_split_leakage")
    summary = {"record_count": len(results), "error_count": len(errors),
               "pairing_status_counts": dict(Counter(row["pairing_status"] for row in results)),
               "pairing_eligible_count": sum(row["pairing_eligible"] for row in results),
               "training_eligible_count": 0, "frame_positive_count": sum(row["frame_positive_count"] for row in results),
               "takeover_start_mismatch_count": sum((row.get("takeover_audit") or {}).get("visited_start_preserved") is False for row in results)}
    report = {"schema_version": 1, "root": str(base), "read_only_raw": True, "training_started": False,
              "protocol": PAIRED_PROTOCOL, "action_joint_names": names, "joint_order_sha256": canonical_sha256(names),
              "configured_interval_s": interval, "records": results, "summary": summary, "errors": errors,
              "split_leakage": leakage,
              "historical_regression_note": "G1LANG-660 replaces visited frame then uses sampled teacher action[1] with clipping; its zero-error statistic proves copied state, not restoration of the teacher initial state or the next actual command."}
    if destination is not None:
        with (destination / "raw_pairing.json").open("x", encoding="utf-8") as stream:
            stream.write(_json(report) + "\n")
        with (destination / "raw_pairing.jsonl").open("x", encoding="utf-8") as stream:
            for row in results:
                stream.write(_json(row) + "\n")
    return report
