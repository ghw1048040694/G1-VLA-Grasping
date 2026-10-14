"""Strict, dependency-free configuration for the G1 rebuild preparation.

Unverified physical settings remain null in the committed configuration.  They
must be filled from the source/asset audit before creating an ActionContract.
"""

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Union


class ConfigError(ValueError):
    """Configuration cannot be used without correction."""


def _error(path: str, message: str) -> None:
    raise ConfigError("{}: {}".format(path, message))


def _text(value: Any, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _error(path, "must be a non-empty string")


def _boolean(value: Any, path: str) -> None:
    if not isinstance(value, bool):
        _error(path, "must be a boolean")


def _positive(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(path, "must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        _error(path, "must be a finite positive number")


def _integer(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _error(path, "must be a non-negative integer")


def _positive_integer(value: Any, path: str) -> None:
    _integer(value, path)
    if value == 0:
        _error(path, "must be greater than zero")


def _fraction(value: Any, path: str) -> None:
    _positive(value, path)
    if value >= 1:
        _error(path, "must be less than one")


def _probability(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(path, "must be a finite number between zero and one")
    if not math.isfinite(value) or not 0 <= value <= 1:
        _error(path, "must be a finite number between zero and one")


def _strings(value: Any, path: str) -> None:
    if not isinstance(value, list) or not value:
        _error(path, "must be a non-empty list of strings")
    for index, item in enumerate(value):
        _text(item, "{}[{}]".format(path, index))
    if len(set(value)) != len(value):
        _error(path, "must not contain duplicates")


def _literal(expected: Any):
    def validate(value: Any, path: str) -> None:
        if type(value) is not type(expected) or value != expected:
            _error(path, "must equal {!r}".format(expected))
    return validate


def _choice(*choices: str):
    def validate(value: Any, path: str) -> None:
        if value not in choices:
            _error(path, "must be one of {}".format(", ".join(choices)))
    return validate


def _optional(validator):
    def validate(value: Any, path: str) -> None:
        if value is not None:
            validator(value, path)
    return validate


def _limit(value: Any, path: str) -> None:
    if isinstance(value, list):
        if len(value) != 31:
            _error(path, "joint limit vector must contain exactly 31 values")
        for index, item in enumerate(value):
            _positive(item, "{}[{}]".format(path, index))
    else:
        _positive(value, path)


def _velocity_limit(value: Any, path: str) -> None:
    values = value if isinstance(value, list) else [value]
    if isinstance(value, list) and len(value) != 31:
        _error(path, "joint velocity vector must contain exactly 31 values")
    for index, item in enumerate(values):
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            _error("{}[{}]".format(path, index), "velocity must be finite and non-negative; zero freezes a command")


def _hash(value: Any, path: str) -> None:
    _text(value, path)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        _error(path, "must be a lowercase SHA-256 hex digest")


def _finite(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _error(path, "must be a finite number")


def _triple(value: Any, path: str) -> None:
    if not isinstance(value, list) or len(value) != 3:
        _error(path, "must contain three finite numbers")
    for index, item in enumerate(value):
        _finite(item, "{}[{}]".format(path, index))


def _indices(value: Any, path: str) -> None:
    if not isinstance(value, list) or not value or len(value) != len(set(value)):
        _error(path, "must contain unique integer indices")
    for item in value:
        _integer(item, path)


def _revision(value: Any, path: str) -> None:
    _text(value, path)
    if len(value) != 40 or any(ch not in "0123456789abcdef" for ch in value):
        _error(path, "must be a pinned 40-character Hub commit SHA")


def _candidates(value: Any, path: str) -> None:
    if not isinstance(value, list) or not value or len(value) > 5:
        _error(path, "must contain one to five pinned model candidates")
    for index, item in enumerate(value):
        _object(item, {"repo_id": _text, "revision": _revision}, "{}[{}]".format(path, index))


def _object(value: Any, schema: Dict[str, Any], path: str) -> None:
    if not isinstance(value, dict):
        _error(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        _error(path, "object keys must be strings")
    unknown = sorted(set(value) - set(schema))
    if unknown:
        _error(path, "unknown keys: {}".format(", ".join(unknown)))
    missing = sorted(set(schema) - set(value))
    if missing:
        _error(path, "missing keys: {}".format(", ".join(missing)))
    for key, validator in schema.items():
        child_path = "{}.{}".format(path, key)
        if isinstance(validator, dict):
            _object(value[key], validator, child_path)
        else:
            validator(value[key], child_path)


SCHEMA = {
    "schema_version": _literal(1),
    "paths": {"root": _text},
    "task": {
        "name": _text, "mode": _choice("grasp_lift", "pick_place"),
        "object_ids": _strings, "hand": _choice("right", "left", "both"),
    },
    "scene": {
        "backend": _literal("mujoco"), "asset_path": _optional(_text),
        "fixed_stance": _literal(True), "assisted_grasp": _literal(False),
        "seed": _integer,
    },
    "observation": {
        "fields": _strings, "camera_ids": _optional(_strings),
        "camera_keys": _strings,
        "image_width": _positive_integer, "image_height": _positive_integer,
        "tactile_definition": _literal("sensor_equivalent"),
    },
    "control": {
        "control_hz": _optional(_positive),
        "nominal_control_hz": _optional(_positive),
        "physics_dt_s": _optional(_positive),
        "physics_substeps": _optional(_positive_integer),
        "body_gain_scale": _positive, "hand_gain_scale": _positive,
        "max_anchor_delta_rad": _optional(_limit),
        "max_velocity_rad_s": _optional(_velocity_limit),
    },
    "action": {
        "protocol": _literal("anchor_relative_target_chunk"),
        "action_dim": _literal(31), "history_steps": _literal(4),
        "chunk_steps": _literal(8), "execute_steps": _literal(4),
        "anchor": _literal("qpos_at_plan"),
        "joint_metadata_path": _text,
        "joint_order_sha256": _optional(_hash),
    },
    "expert": {
        "label_alignment": _literal("next_control_command"),
        "include_restart_wait_labels": _literal(False),
        "separate_waypoint_recovery": _literal(True),
        "teacher": {
            "profile": _literal("green_right_recovery_v1"),
            "grasp_yaw_deg": _finite, "closure_multiplier": _positive,
            "finger_scale": _triple, "contact_compensation_m": _triple,
            "pregrasp_height_m": _positive, "pregrasp_lateral_m": _positive,
            "high_pregrasp_height_m": _positive, "home_settle_s": _positive,
            "high_approach_s": _positive, "pregrasp_s": _positive,
            "contact_approach_s": _positive, "close_duration_s": _positive,
            "grasp_hold_s": _positive, "lift_height_m": _positive,
            "lift_duration_s": _positive, "final_hold_s": _positive,
            "adaptive_regrasp": _boolean, "ik_max_error_m": _positive,
            "record_rgb": _literal(True), "palm_position_tolerance_m": _positive,
            "joint_target_tolerance_rad": _positive, "phase_timeout_scale": _positive,
        },
    },
    "data": {
        "registry_path": _text, "source_roots": _strings,
        "build_dir": _text, "split_group_fields": _strings,
        "validation_fraction": _fraction, "test_fraction": _fraction,
        "seed": _integer, "historical_test_role": _literal("regression"),
    },
    "policy": {
        "name": _literal("continuous_dagger"),
        "input_boundary": _literal("observation_only"),
        "arm_hand_heads": _literal(True),
    },
    "train": {
        "enabled": _boolean, "authorization": _literal("conditional"), "seed": _integer,
        "batch_size": _optional(_positive_integer),
        "epochs": _optional(_positive_integer),
        "learning_rate": _optional(_positive),
        "resume_checkpoint": _optional(_text),
        "warm_start_checkpoint": _optional(_text),
    },
    "eval": {
        "checkpoint": _optional(_text), "layouts_path": _optional(_text),
        "num_layouts": _positive_integer,
        "lift_height_m": _positive, "hold_duration_s": _positive,
        "joint_limit_tolerance_rad": _positive,
        "assisted_grasp": _literal(False),
        "gates": {"expert_lift_rate": _probability,
                  "policy_contact_rate": _probability,
                  "policy_lift_rate": _probability},
    },
    "artifacts": {
        "runs_dir": _text, "inventory_path": _text,
        "protection_manifest_path": _text,
    },
    "environment": {
        "python_min": _text, "mujoco_expected": _optional(_text),
        "torch_expected": _optional(_text), "cuda_expected": _optional(_text),
    },
    "wm": {"enabled": _literal(False)},
    "external_data": {
        "repo_id": _literal("unitreerobotics/G1_Dex3_GraspSquare_Dataset"),
        "revision": _revision, "local_dir": _text,
        "episode_ids": _indices, "source_fps": _positive,
        "max_episode_rows": _positive_integer, "max_download_bytes": _positive_integer,
        "timestamp_tolerance_s": _positive, "semantic_evidence_path": _optional(_text),
        "audit_report_paths": _strings,
        "probe_videos": _boolean,
        "video": {"backend": _literal("pyav_http_range"), "samples_per_episode": _positive_integer,
                  "max_read_bytes": _positive_integer},
    },
    "public_weights": {"cache_dir": _text, "candidates": _candidates},
    "offline_policy": {
        "backend": _literal("lerobot_act_native"), "repo_id": _text, "revision": _revision,
        "model_dir": _text, "dataset_repo_id": _text, "dataset_revision": _revision,
        "dataset_dir": _text, "episode_id": _integer, "frame_indices": _indices,
        "device": _literal("cuda"), "warmup_runs": _positive_integer,
        "measured_runs": _positive_integer, "max_download_bytes": _positive_integer,
    },
}

OBSERVATION_FIELDS = frozenset({
    "images", "joint_position", "joint_velocity", "finger_sites",
    "language", "tactile", "timestamp_s", "history"
})


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reject unknown keys, unsafe preparation settings and inconsistent units."""
    _object(config, SCHEMA, "config")
    unknown_fields = set(config["observation"]["fields"]) - OBSERVATION_FIELDS
    if unknown_fields:
        _error("config.observation.fields", "unsupported fields: {}".format(
            ", ".join(sorted(unknown_fields))))
    required_fields = {"joint_position", "images", "language"}
    if not required_fields <= set(config["observation"]["fields"]):
        _error("config.observation.fields", "must include images, joint_position and language")
    control = config["control"]
    if control["control_hz"] is not None and control["physics_dt_s"] is not None:
        steps = 1.0 / (control["control_hz"] * control["physics_dt_s"])
        if steps < 1 or not math.isclose(steps, round(steps), rel_tol=1e-8, abs_tol=1e-8):
            _error("config.control", "control period must contain an integer number of physics steps")
        if control["physics_substeps"] is not None and not math.isclose(
                steps, control["physics_substeps"], rel_tol=1e-8, abs_tol=1e-8):
            _error("config.control.physics_substeps", "must match control_hz and physics_dt_s")
    cameras = config["observation"]
    if cameras["camera_ids"] is not None and len(cameras["camera_ids"]) != len(cameras["camera_keys"]):
        _error("config.observation", "camera_ids and camera_keys must have matching lengths")
    bindings = {"observation.images.head": "head_camera", "observation.images.left_wrist": "left_wrist_camera",
                "observation.images.right_wrist": "right_wrist_camera"}
    if set(cameras["camera_keys"]) != set(bindings):
        _error("config.observation.camera_keys", "must name exactly the three known camera keys")
    if cameras["camera_ids"] is not None and cameras["camera_ids"] != [bindings[key] for key in cameras["camera_keys"]]:
        _error("config.observation.camera_ids", "must match the configured camera key order")
    if set(cameras["fields"]) != OBSERVATION_FIELDS:
        _error("config.observation.fields", "must include the complete sensor whitelist")
    data = config["data"]
    if data["validation_fraction"] + data["test_fraction"] >= 1:
        _error("config.data", "validation and test fractions must leave training groups")
    if not {"layout_fingerprint", "parent_rollout_id"} <= set(data["split_group_fields"]):
        _error("config.data.split_group_fields", "must include layout_fingerprint and parent_rollout_id")
    if config["train"]["resume_checkpoint"] and config["train"]["warm_start_checkpoint"]:
        _error("config.train", "resume_checkpoint and warm_start_checkpoint are mutually exclusive")
    teacher = config["expert"]["teacher"]
    if any(value <= 0 for value in teacher["finger_scale"]):
        _error("config.expert.teacher.finger_scale", "must be positive")
    if teacher["phase_timeout_scale"] < 1:
        _error("config.expert.teacher.phase_timeout_scale", "must be at least one")
    if teacher["final_hold_s"] < config["eval"]["hold_duration_s"]:
        _error("config.expert.teacher.final_hold_s", "must cover the acceptance hold")
    return copy.deepcopy(config)


_PATH_FIELDS = {
    "scene": ("asset_path",), "action": ("joint_metadata_path",),
    "data": ("registry_path", "build_dir"),
    "train": ("resume_checkpoint", "warm_start_checkpoint"),
    "eval": ("checkpoint", "layouts_path"),
    "artifacts": ("runs_dir", "inventory_path", "protection_manifest_path"),
    "external_data": ("local_dir", "semantic_evidence_path"),
    "public_weights": ("cache_dir",),
    "offline_policy": ("model_dir", "dataset_dir"),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError("duplicate JSON key: {}".format(key))
        result[key] = value
    return result


def load_config(path: Union[str, Path], project_root: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load one JSON configuration and resolve relative paths from the project root.

    An explicit project_root wins. Otherwise a config in ``configs/`` belongs to
    its parent project; for other locations the config's directory is the root.
    paths.root is resolved relative to that base. Files need not exist for plan
    or inventory; execution readiness is a separate contract audit.
    """
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            raw = json.load(stream, object_pairs_hook=_unique_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError("cannot read {}: {}".format(config_path, exc)) from exc
    result = validate_config(raw)
    if project_root is None:
        base = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    else:
        base = Path(project_root).expanduser().resolve()
    configured_root = Path(result["paths"]["root"]).expanduser()
    root = configured_root.resolve() if configured_root.is_absolute() else (base / configured_root).resolve()
    result["paths"]["root"] = str(root)

    def resolve(value):
        if value is None:
            return None
        item = Path(value).expanduser()
        return str(item.resolve() if item.is_absolute() else (root / item).resolve())

    for section, keys in _PATH_FIELDS.items():
        for key in keys:
            result[section][key] = resolve(result[section][key])
    result["data"]["source_roots"] = [resolve(item) for item in result["data"]["source_roots"]]
    result["external_data"]["audit_report_paths"] = [resolve(item) for item in result["external_data"]["audit_report_paths"]]
    return result
