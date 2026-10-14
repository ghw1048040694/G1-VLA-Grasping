"""Register existing raw evidence without reading NPZ arrays or training anything.

All imported evidence remains outside the new training set.  Camera or robot
changes do not affect layout identities: fingerprints contain named object poses
only.  Missing provenance is reported explicitly rather than invented.
"""

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .artifacts import experiment_ids, sha256_file
from .data_registry import group_key, strict_physical_success


RAW_NAMES = frozenset(("expert_pick_place_episode.npz", "trajectory.npz", "policy_visited_state.npz"))
OBJECT_NAMES = frozenset(("red_triangle", "yellow_rod", "green_cube"))
RECEPTACLE_NAMES = frozenset(("blue_receptacle", "blue_box", "blue_container"))
MAX_METADATA_BYTES = 16 * 1024 * 1024


def _absolute(path: Union[str, Path]) -> Path:
    return Path(os.path.abspath(str(Path(path).expanduser())))


def _read_json(path: Path) -> Dict[str, Any]:
    if path.is_symlink():
        raise ValueError("Metadata symlinks are not followed.")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Metadata exceeds bounded read limit.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Metadata must be a JSON object.")
    return payload


def _candidate_paths(value: str, root: Path, sibling: Path) -> List[Path]:
    path = Path(value.replace("\\", "/"))
    candidates = [path] if path.is_absolute() else [root / path, sibling / path]
    # Retired project names have a known mapping, but only an actually existing
    # target may be accepted.  No global basename search is used.
    for index, part in enumerate(path.parts):
        compact = part.lower().replace("-", "").replace("_", "")
        if compact in ("g1upperbody", "g1vlagrasping"):
            candidates.append(root.joinpath(*path.parts[index + 1:]))
    unique = []
    for candidate in candidates:
        normalized = _absolute(candidate)
        if normalized not in unique:
            unique.append(normalized)
    return unique


def _find_path(value: Any, root: Path, sibling: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    for candidate in _candidate_paths(value.strip(), root, sibling):
        # Metadata may name an older project.  After canonicalisation, registered
        # lineage must remain within this project, never an unrelated filesystem.
        try:
            candidate.relative_to(root)
        except ValueError:
            continue
        if candidate.exists() and not candidate.is_symlink():
            return candidate
    return None


def _parent_trajectory(spec: Dict[str, Any], collection: Dict[str, Any], root: Path,
                       sibling: Path) -> Optional[Path]:
    source = spec.get("source_rollout")
    episode = spec.get("source_episode")
    references = []
    if isinstance(source, str) and source.strip():
        source_path = Path(source.replace("\\", "/"))
        # Full rollout episode paths outrank source_episode, whose basename alone
        # is not globally unique.
        references.append(source)
        if source_path.name != episode and source_path.suffix != ".npz" and isinstance(episode, str) and episode.strip():
            references.append(str(source_path / episode))
    if isinstance(episode, str) and ("/" in episode or "\\" in episode):
        references.append(episode)
    rollout_root = collection.get("source_rollout_root")
    if isinstance(rollout_root, str) and isinstance(episode, str) and episode:
        references.append(str(Path(rollout_root) / episode))
    for reference in references:
        candidate = _find_path(reference, root, sibling)
        if candidate is None:
            continue
        if candidate.is_file() and candidate.name in RAW_NAMES and candidate.name != "policy_visited_state.npz":
            return candidate
        if candidate.is_dir():
            for name in ("trajectory.npz", "expert_pick_place_episode.npz"):
                raw = candidate / name
                if raw.is_file() and not raw.is_symlink():
                    return raw
    return None


def _vector(value: Any, length: int) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    values = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item):
            return None
        number = round(float(item), 8)
        values.append(0.0 if number == 0 else number)
    return values


def _quat(value: Any) -> Optional[List[float]]:
    values = _vector(value, 4)
    if values is None:
        return None
    norm = math.sqrt(sum(number * number for number in values))
    if norm < 1e-12:
        return None
    values = [number / norm for number in values]
    # q and -q represent the same pose.
    for number in values:
        if abs(number) > 1e-12:
            if number < 0:
                values = [-number for number in values]
            break
    return [0.0 if round(number, 8) == 0 else round(number, 8) for number in values]


def _object_pose_layout(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    objects = payload.get("objects")
    if not isinstance(objects, dict):
        return None
    poses = {}
    for name, value in objects.items():
        if name not in OBJECT_NAMES | RECEPTACLE_NAMES or not isinstance(value, dict):
            continue
        position = _vector(value.get("pos", value.get("position")), 3)
        quaternion = _quat(value.get("quat", value.get("quaternion")))
        if quaternion is None and payload.get("orientation_default") == "identity":
            quaternion = [1.0, 0.0, 0.0, 0.0]
        if position is None or quaternion is None:
            return None
        poses[name] = {"pos": position, "quat": quaternion}
    return {"objects": dict(sorted(poses.items()))} if set(poses) & OBJECT_NAMES else None


def _xml_layout(path: Path) -> Optional[Dict[str, Any]]:
    if path.is_symlink() or path.stat().st_size > MAX_METADATA_BYTES:
        return None
    tree = ET.parse(path)
    poses = {}
    # Only top-level task objects are accepted, so local nested body transforms
    # are not accidentally treated as world positions.
    for body in tree.findall("./worldbody/body"):
        name = body.get("name")
        if name not in OBJECT_NAMES | RECEPTACLE_NAMES:
            continue
        position = _vector([float(item) for item in body.get("pos", "").split()], 3)
        if any(body.get(key) is not None for key in ("euler", "axisangle", "xyaxes", "zaxis")):
            return None
        quaternion = _quat([float(item) for item in body.get("quat", "1 0 0 0").split()])
        if position is None or quaternion is None:
            return None
        poses[name] = {"pos": position, "quat": quaternion}
    return {"objects": dict(sorted(poses.items()))} if set(poses) & OBJECT_NAMES else None


def _layout(raw: Path, summary: Dict[str, Any], root: Path,
            errors: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str]]:
    candidates = []
    referenced = _find_path(summary.get("scene"), root, raw.parent)
    if referenced is not None and referenced.is_file():
        candidates.append(referenced)
    candidates.extend(raw.parent / name for name in ("scene.xml", "g1_language_pick_place.xml"))
    seen = set()
    for path in candidates:
        if str(path) in seen or not path.is_file() or path.is_symlink():
            continue
        seen.add(str(path))
        try:
            value = _xml_layout(path)
            initial = summary.get("object_initial_position_m")
            expected = set(initial) & OBJECT_NAMES if isinstance(initial, dict) else set()
            if value and expected and not expected.issubset(value["objects"]):
                value = None
            if value:
                serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
                return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), value, str(path)
        except (OSError, ValueError, ET.ParseError) as error:
            errors.append({"path": str(path), "operation": "layout", "error_type": type(error).__name__, "message": str(error)})
    for filename in ("layout.json", "initial_layout.json"):
        path = raw.parent / filename
        if not path.is_file() or path.is_symlink():
            continue
        try:
            value = _object_pose_layout(_read_json(path))
            if value:
                serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
                return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), value, str(path)
        except (OSError, ValueError, UnicodeError) as error:
            errors.append({"path": str(path), "operation": "layout", "error_type": type(error).__name__, "message": str(error)})
    positions = summary.get("object_initial_position_m")
    quaternions = summary.get("object_initial_quaternion", summary.get("object_initial_quaternion_wxyz"))
    if isinstance(positions, dict) and isinstance(quaternions, dict):
        value = _object_pose_layout({"objects": {name: {"pos": position, "quat": quaternions.get(name)}
                                                 for name, position in positions.items()}})
        if value:
            serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
            return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), value, "summary_object_initial_pose"
    value = _object_pose_layout(summary.get("layout", {})) if isinstance(summary.get("layout"), dict) else None
    if value:
        serialized = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), value, "summary_layout"
    return None, None, None


def _physical_gate(summary: Dict[str, Any], collection: Dict[str, Any]) -> Dict[str, Any]:
    audit = summary.get("grasp_physics_audit")
    audit = audit if isinstance(audit, dict) else {}
    explicit = summary.get("physical_gate")
    explicit = explicit if isinstance(explicit, dict) else {}
    contact = explicit.get("contact")
    if not isinstance(contact, bool):
        steps = audit.get("physical_contact_steps")
        contact = steps > 0 if isinstance(steps, (int, float)) and not isinstance(steps, bool) else None
    lift = explicit.get("lift_height_m", audit.get("target_lift_m"))
    # Configured lift_height_m/lift_hold_duration_s are desired commands, not
    # measured outcomes.  They must never fill missing physical measurements.
    hold = explicit.get("hold_seconds", audit.get("measured_hold_seconds", audit.get("target_hold_duration_s")))
    assisted = explicit.get("assisted")
    if not isinstance(assisted, bool):
        configured = summary.get("assisted_grasp_enabled", collection.get("assisted_grasp_enabled"))
        active_steps = audit.get("assisted_constraint_active_steps")
        if configured is True or isinstance(active_steps, (int, float)) and active_steps > 0:
            assisted = True
        elif configured is False or isinstance(active_steps, (int, float)) and active_steps == 0:
            assisted = False
        else:
            assisted = None
    historical = {key: summary[key] for key in ("passed", "task_success", "target_in_box", "wrong_objects_in_box",
                                               "hand_object_contact_candidate_steps", "hand_object_contact_candidate_pairs",
                                               "max_hand_object_contacts", "assist_activation_frame", "assist_release_frame",
                                               "lift_height_m", "lift_hold_duration_s") if key in summary}
    historical["grasp_physics_audit"] = audit
    return {"contact": contact, "lift_height_m": lift, "hold_seconds": hold, "assisted": assisted,
            "in_box": summary.get("target_in_box", explicit.get("in_box")), "released": explicit.get("released"),
            "historical_evidence": historical}


def ingest_raw_registry(root: Union[str, Path]) -> Dict[str, Any]:
    """Build registration rows from actual raw paths, metadata, and object poses.

    The returned rows can be persisted by write_registry.  No writes occur here,
    NPZs are only hashed, and every imported row has training_eligible=False and
    is_positive=False until a separate review and new-data gate approves it.
    """
    base = _absolute(root)
    output = base / "outputs"
    errors = []
    paths = []
    if not output.is_dir() or output.is_symlink():
        return {"schema_version": 1, "records": [], "summary": {"record_count": 0, "error_count": 1},
                "errors": [{"path": str(output), "operation": "scan", "error_type": "MissingPath", "message": "outputs is missing or a symlink."}]}

    def walk_error(error: OSError) -> None:
        errors.append({"path": error.filename, "operation": "scan", "error_type": type(error).__name__, "message": str(error)})

    for directory, dirnames, filenames in os.walk(str(output), followlinks=False, onerror=walk_error):
        parent = Path(directory)
        dirnames[:] = [name for name in dirnames if not (parent / name).is_symlink()
                       and not (parent == output and name == "rebuild")]
        for name in filenames:
            path = parent / name
            if name in RAW_NAMES and not path.is_symlink():
                paths.append(path)
    cache = {}

    def read_metadata(path: Path) -> Dict[str, Any]:
        key = str(path)
        if key not in cache:
            try:
                cache[key] = _read_json(path)
            except (OSError, ValueError, UnicodeError, RecursionError) as error:
                errors.append({"path": key, "operation": "read_metadata", "error_type": type(error).__name__, "message": str(error)})
                cache[key] = {}
        return cache[key]

    records = []
    for raw in sorted(paths):
        issues = []
        summary_path = next((raw.parent / name for name in ("language_pick_place_summary.json", "summary.json")
                             if (raw.parent / name).is_file() and not (raw.parent / name).is_symlink()), None)
        summary = read_metadata(summary_path) if summary_path else {}
        if summary_path is None or not summary:
            issues.append("summary_missing_or_unreadable")
        recovery_path = raw.parent / "recovery_spec.json"
        has_recovery = recovery_path.is_file() and not recovery_path.is_symlink()
        recovery = read_metadata(recovery_path) if has_recovery else {}
        collection_path = raw.parent.parent / "summary.json"
        collection = read_metadata(collection_path) if collection_path.is_file() and not collection_path.is_symlink() else {}
        parent_path = None
        frame = None
        if has_recovery:
            parent_path = _parent_trajectory(recovery, collection, base, raw.parent)
            if parent_path is None:
                issues.append("parent_unresolved")
            candidate_frame = recovery.get("source_frame", recovery.get("takeover_frame"))
            if isinstance(candidate_frame, int) and not isinstance(candidate_frame, bool) and candidate_frame >= 0:
                frame = candidate_frame
            else:
                issues.append("takeover_frame_unresolved")
            if recovery.get("safe_waypoint_recovery") is True:
                regime = "safe_waypoint"
            elif recovery.get("safe_waypoint_recovery") is False and isinstance(summary.get("initial_direct_continuation"), dict) and summary["initial_direct_continuation"].get("enabled") is True:
                regime = "direct_continuation"
            else:
                regime = "unknown_recovery"
                issues.append("recovery_regime_unresolved")
        elif isinstance(summary.get("initial_state_recovery"), dict) and summary["initial_state_recovery"].get("source"):
            # This field often points to the copied visited-state NPZ, which is
            # insufficient to identify the original rollout.  Do not group by it.
            regime = "unknown_recovery"
            issues.append("parent_unresolved")
        else:
            parent_path = raw
            regime = "policy_rollout" if raw.name == "trajectory.npz" else "expert"
        fingerprint, layout, layout_source = _layout(raw, summary, base, errors)
        if fingerprint is None:
            issues.append("layout_unresolved")
        gate = _physical_gate(summary, collection)
        if not isinstance(gate["hold_seconds"], (int, float)) or isinstance(gate["hold_seconds"], bool) or not math.isfinite(gate["hold_seconds"]):
            issues.append("measured_hold_missing")
        if gate["contact"] is None or gate["assisted"] is None or gate["lift_height_m"] is None:
            issues.append("physical_measurements_incomplete")
        outcome = summary.get("passed", summary.get("task_success"))
        role = "visited_state" if raw.name == "policy_visited_state.npz" else "policy_trajectory" if raw.name == "trajectory.npz" else "expert_trajectory"
        row = {"record_id": hashlib.sha256(str(raw).encode("utf-8")).hexdigest(), "raw_path": str(raw),
               "relative_raw_path": str(raw.relative_to(base)), "raw_role": role,
               "experiment_ids": experiment_ids(raw), "parent_root": str(base),
               "source_rollout": str(parent_path) if parent_path is not None else None,
               "parent_resolution": "resolved" if parent_path is not None else "parent_unresolved",
               "layout_fingerprint": fingerprint, "layout": layout, "layout_source": layout_source,
               "takeover_frame": frame, "regime": regime, "physical_gate": gate,
               "task": summary.get("task", "historical_pick_place"),
               "language_instruction": summary.get("language_instruction", recovery.get("instruction")),
               "target_object": summary.get("target_object", recovery.get("target_object")),
               "historical_outcome": "success" if outcome is True else "failure" if outcome is False else "unknown",
               "training_eligible": False, "is_positive": False,
               "quality_status": "needs_review" if issues else "historical_only",
               "split": "diagnostic" if issues else "historical_regression", "review_issues": sorted(set(issues)),
               "summary_path": str(summary_path) if summary_path else None,
               "recovery_spec_path": str(recovery_path) if has_recovery else None,
               "raw_sha256": None, "size_bytes": None}
        if role == "expert_trajectory" and isinstance(outcome, bool):
            row["expert_success"] = outcome
        row["strict_physical_success"] = strict_physical_success(row)
        row["group_id"] = group_key(row) if parent_path is not None and fingerprint is not None else None
        try:
            stat = raw.stat()
            row["size_bytes"] = stat.st_size
            row["raw_sha256"] = sha256_file(raw)
            after = raw.stat()
            if (after.st_size, after.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                row["raw_sha256"] = None
                row["quality_status"] = "needs_review"
                row["split"] = "diagnostic"
                row["review_issues"].append("raw_changed_during_hash")
        except OSError as error:
            errors.append({"path": str(raw), "operation": "hash_raw", "error_type": type(error).__name__, "message": str(error)})
            row["quality_status"] = "needs_review"
            row["split"] = "diagnostic"
            row["review_issues"].append("raw_unreadable")
        records.append(row)
    split_counts = Counter(row["split"] for row in records)
    regimes = Counter(row["regime"] for row in records)
    outcomes = Counter(row["historical_outcome"] for row in records)
    return {"schema_version": 1, "root": str(base), "read_only": True, "records": records, "errors": errors,
            "summary": {"record_count": len(records), "error_count": len(errors), "split_counts": dict(split_counts),
                        "regime_counts": dict(regimes), "historical_outcomes": dict(outcomes),
                        "parent_unresolved_count": sum(row["parent_resolution"] == "parent_unresolved" for row in records),
                        "layout_unresolved_count": sum(row["layout_fingerprint"] is None for row in records),
                        "training_eligible_count": 0, "positive_label_count": 0}}
