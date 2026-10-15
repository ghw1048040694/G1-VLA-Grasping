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


def _lags(value: Any, path: str) -> None:
    if (not isinstance(value, list) or not value or
            any(isinstance(item, bool) or not isinstance(item, int) or abs(item) > 30 for item in value) or
            len(value) != len(set(value)) or 0 not in value):
        _error(path, "must contain unique integer frame lags within [-30, 30], including zero")


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
            "profile": _choice("green_right_recovery_v1", "green_right_geometry_feedback_v2"),
            "grasp_yaw_deg": _finite, "closure_multiplier": _positive,
            "finger_scale": _triple, "contact_compensation_m": _triple,
            "pregrasp_height_m": _positive, "pregrasp_lateral_m": _positive,
            "high_pregrasp_height_m": _positive, "home_settle_s": _positive,
            "high_approach_s": _positive, "pregrasp_s": _positive,
            "contact_approach_s": _positive, "close_duration_s": _positive,
            "grasp_hold_s": _positive, "lift_height_m": _positive,
            "lift_duration_s": _positive, "final_hold_s": _positive,
            "adaptive_regrasp": _boolean, "ik_max_error_m": _positive,
            "ik_collision_margin_m": _positive, "ik_collision_weight": _positive,
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
        "backend": _choice("current_action31", "grasp_square_act"),
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
    "source_native": {
        "backend": _literal("grasp_square_native_v1"), "episode_ids": _indices,
        "split_episode_ids": {"train": _indices, "validation": _indices, "test": _indices},
        "split_unit": _literal("source_video_file_group"),
        "camera_key": _literal("observation.images.cam_left_high"),
        "source_fps": _literal(30), "chunk_steps": _positive_integer,
        "image_width": _positive_integer, "image_height": _positive_integer,
        "image_transform": _literal("bilinear_resize_rgb_v1"),
        "build_dir": _text, "video_dir": _text, "timestamp_tolerance_s": _positive,
        "max_episode_rows": _positive_integer, "max_total_rows": _positive_integer,
        "max_derived_bytes": _positive_integer, "max_video_download_bytes": _positive_integer,
    },
    "grasp_act": {
        "backend": _literal("lerobot_act_grasp_square"), "dataset_manifest_path": _text,
        "memory_report_path": _text, "checkpoint_path": _text, "device": _literal("cuda"),
        "chunk_size": _positive_integer, "dim_model": _positive_integer, "n_heads": _positive_integer,
        "dim_feedforward": _positive_integer, "n_encoder_layers": _positive_integer,
        "n_decoder_layers": _positive_integer, "n_vae_encoder_layers": _positive_integer,
        "latent_dim": _positive_integer, "dropout": _probability, "weight_decay": _positive,
        "kl_weight": _positive, "max_steps": _positive_integer, "eval_every_steps": _positive_integer,
        "max_train_wall_s": _positive, "max_gpu_reserved_bytes": _positive_integer,
        "eval_stride": _positive_integer, "eval_batch_size": _positive_integer,
    },
    "native_diagnostics": {
        "training_report_path": _text, "lag_frames": _lags,
        "stationary_speed_rad_s": _positive, "moving_speed_rad_s": _positive,
        "loss_window_steps": _positive_integer, "verify_raw_source": _literal(True),
    },
    "act_diagnostics": {
        "backend": _literal("grasp_square_act_diagnostics_v1"),
        "reference_checkpoint_path": _text, "reference_config_path": _text,
        "eval_stride": _positive_integer, "eval_batch_size": _positive_integer,
        "visual_modes": _strings, "image_shift_fraction": _fraction, "seed": _integer,
        "max_steps": _positive_integer, "eval_every_steps": _positive_integer,
        "max_train_wall_s": _positive, "max_gpu_reserved_bytes": _positive_integer,
        "output_dir": _text, "variants": _strings,
    },
    "visual_pretraining": {
        "backend": _literal("grasp_square_visual_pretraining_v1"),
        "variants": _strings, "use_vae": _literal(False),
        "image_preprocessing": _literal("unchanged_train_rgb_statistics_v1"),
        "seed": _integer, "max_steps": _positive_integer,
        "eval_every_steps": _positive_integer, "eval_stride": _positive_integer,
        "eval_batch_size": _positive_integer, "max_train_wall_s": _positive,
        "max_gpu_reserved_bytes": _positive_integer, "output_dir": _text,
        "reference_ablation_report_path": _text, "reference_ablation_report_sha256": _hash,
        "weights_url": _literal("https://download.pytorch.org/models/resnet18-f37072fd.pth"),
        "weights_sha256_prefix": _literal("f37072fd"), "weights_sha256": _optional(_hash),
        "weights_max_bytes": _positive_integer,
        "weights_cache_path": _text, "weights_receipt_path": _text,
    },
    "state_residual": {
        "backend": _literal("grasp_square_state_residual_v1"),
        "variants": _strings, "use_vae": _literal(False),
        "target": _literal("raw_action_minus_anchor_q_over_train_action_std_v1"),
        "decode": _literal("raw_anchor_q_plus_train_action_std_times_residual_zero_exact_v1"),
        "zero_head_init": _literal(True), "include_step_zero_in_selection": _literal(True),
        "seed": _integer, "max_steps": _positive_integer,
        "eval_every_steps": _positive_integer, "eval_stride": _positive_integer,
        "eval_batch_size": _positive_integer, "max_train_wall_s": _positive,
        "max_gpu_reserved_bytes": _positive_integer, "output_dir": _text,
        "reference_visual_report_path": _text, "reference_visual_report_sha256": _hash,
        "admission_report_path": _text,
    },
    "trajectory_coverage": {
        "backend": _literal("grasp_square_trajectory_coverage_v1"),
        "reference_training_report_path": _text, "reference_training_report_sha256": _hash,
        "stationary_speed_rad_s": _positive, "moving_speed_rad_s": _positive,
        "quarter_bins": _literal(4), "expansion_group_limit": _positive_integer,
        "expansion_episodes_per_group": _positive_integer,
    },
    "train_expansion": {
        "backend": _literal("grasp_square_train_expansion_v1"),
        "candidate_episode_ids": _indices,
        "reference_coverage_report_path": _text, "reference_coverage_report_sha256": _hash,
        "reference_residual_report_path": _text, "reference_residual_report_sha256": _hash,
        "session_evidence_path": _text, "session_evidence_sha256": _hash,
        "candidate_build_dir": _text, "build_dir": _text, "audit_report_path": _text,
        "max_total_rows": _positive_integer, "max_candidate_derived_bytes": _positive_integer,
        "max_derived_bytes": _positive_integer, "allow_same_source_coverage": _literal(True),
        "require_independent_sessions": _literal(False),
    },
    "expanded_residual": {
        "backend": _literal("grasp_square_expanded_state_residual_v1"),
        "variants": _strings, "use_vae": _literal(False),
        "target": _literal("raw_action_minus_anchor_q_over_train_action_std_v1"),
        "decode": _literal("raw_anchor_q_plus_train_action_std_times_residual_zero_exact_v1"),
        "zero_head_init": _literal(True), "include_step_zero_in_selection": _literal(True),
        "seed": _integer, "max_steps": _positive_integer,
        "eval_every_steps": _positive_integer, "eval_stride": _positive_integer,
        "eval_batch_size": _positive_integer, "max_train_wall_s": _positive,
        "max_gpu_reserved_bytes": _positive_integer, "output_dir": _text,
        "data_manifest_path": _text, "data_manifest_sha256": _hash,
        "reference_residual_report_path": _text, "reference_residual_report_sha256": _hash,
        "admission_report_path": _text,
    },
    "normalization_attribution": {
        "backend": _literal("grasp_square_normalization_attribution_v1"),
        "variants": _strings, "use_vae": _literal(False),
        "target": _literal("raw_action_minus_anchor_q_over_train_action_std_v1"),
        "decode": _literal("raw_anchor_q_plus_train_action_std_times_residual_zero_exact_v1"),
        "zero_head_init": _literal(True), "include_step_zero_in_selection": _literal(True),
        "seed": _integer, "max_steps": _positive_integer,
        "eval_every_steps": _positive_integer, "eval_stride": _positive_integer,
        "eval_batch_size": _positive_integer, "max_train_wall_s": _positive,
        "max_gpu_reserved_bytes": _positive_integer, "output_dir": _text,
        "data_manifest_path": _text, "data_manifest_sha256": _hash,
        "reference_residual_report_path": _text, "reference_residual_report_sha256": _hash,
        "reference_expanded_report_path": _text, "reference_expanded_report_sha256": _hash,
        "reference_statistics_manifest_path": _text, "reference_statistics_manifest_sha256": _hash,
        "reference_statistics_sha256": _hash,
        "model_statistics_source": _literal("original_six_train_only_v1"),
        "admission_report_path": _text,
    },
    "constant_residual": {
        "backend": _literal("grasp_square_train_median_residual_v1"),
        "method": _literal("per_horizon_joint_raw_residual_median_all_complete_train_anchors_v1"),
        "decode": _literal("raw_anchor_q_plus_raw_constant_zero_exact_v1"),
        "fit_cohorts": _strings, "eval_stride": _positive_integer, "output_dir": _text,
        "reference_expanded_report_path": _text, "reference_expanded_report_sha256": _hash,
    },
    "state_conditioning": {
        "backend": _literal("grasp_square_frozen_state_conditioning_v1"),
        "variants": _strings, "modes": _strings, "eval_splits": _strings,
        "eval_stride": _positive_integer, "eval_batch_size": _positive_integer,
        "within_episode_rule": _literal("half_length_cyclic_shift_v1"),
        "cross_episode_rule": _literal("next_sorted_episode_same_relative_rank_v1"),
        "real_reproduction_atol": _literal(1e-6), "optimizer_steps": _literal(0),
        "model_output_constant_method": _literal("train_eval_model_raw_residual_coordinate_median_fp32scale_fp64fit_v1"),
        "tie_tolerance_rad": _literal(0.0), "max_eval_wall_s": _positive,
        "max_gpu_reserved_bytes": _positive_integer,
        "reference_expanded_report_path": _text, "reference_expanded_report_sha256": _hash,
        "reference_constant_report_path": _text, "reference_constant_report_sha256": _hash,
        "output_dir": _text,
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
    native, act = config["source_native"], config["grasp_act"]
    split_ids = [item for name in ("train", "validation", "test") for item in native["split_episode_ids"][name]]
    if len(split_ids) != len(set(split_ids)) or set(split_ids) != set(native["episode_ids"]):
        _error("config.source_native.split_episode_ids", "must partition every selected episode exactly once")
    if set(native["episode_ids"]) & {0, 150, 300}:
        _error("config.source_native.episode_ids", "exclude the prior discovery-audit episodes 0/150/300")
    if native["chunk_steps"] != act["chunk_size"] or native["chunk_steps"] < 2:
        _error("config.grasp_act.chunk_size", "must agree with the native future horizon and be at least two")
    if act["dim_model"] % act["n_heads"] or act["dropout"] >= 1:
        _error("config.grasp_act", "attention dimension must divide into heads and dropout must be below one")
    if act["eval_every_steps"] > act["max_steps"] or act["eval_stride"] < act["chunk_size"]:
        _error("config.grasp_act", "validation interval must fit the step budget; evaluation anchors must not overlap future windows")
    if config["train"]["backend"] == "grasp_square_act" and (config["train"]["resume_checkpoint"] or config["train"]["warm_start_checkpoint"]):
        _error("config.train", "the admitted native ACT baseline starts from scratch")
    diagnostics, timing = config["act_diagnostics"], config["native_diagnostics"]
    if set(diagnostics["visual_modes"]) != {"real", "mean_image", "within_episode_shift", "cross_episode_swap"}:
        _error("config.act_diagnostics.visual_modes", "must retain all four predeclared visual conditions")
    if not set(diagnostics["variants"]) <= {"rgb", "rgb_mean", "state_only", "rgb_no_vae"}:
        _error("config.act_diagnostics.variants", "unsupported native ACT ablation variant")
    if diagnostics["eval_stride"] < act["chunk_size"] or diagnostics["eval_every_steps"] > diagnostics["max_steps"]:
        _error("config.act_diagnostics", "evaluation must use complete nonoverlapping windows within the step budget")
    if diagnostics["seed"] != config["train"]["seed"] or diagnostics["max_gpu_reserved_bytes"] > 7 * 1024**3:
        _error("config.act_diagnostics", "paired trials must retain the training seed and 7 GiB allocator bound")
    if timing["stationary_speed_rad_s"] >= timing["moving_speed_rad_s"]:
        _error("config.native_diagnostics", "stationary speed threshold must be lower than moving threshold")
    visual = config["visual_pretraining"]
    if visual["variants"] != ["rgb_scratch", "rgb_pretrained", "rgb_mean", "state_only"]:
        _error("config.visual_pretraining.variants", "must retain the ordered no-VAE RGB, pretrained, mean-image and state-only controls")
    for key in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_train_wall_s", "max_gpu_reserved_bytes"):
        if visual[key] != diagnostics[key]:
            _error("config.visual_pretraining." + key, "must retain the declared stage5 paired training budget")
    if visual["max_steps"] % visual["eval_every_steps"]:
        _error("config.visual_pretraining", "the training budget must end at a validation point")
    if visual["weights_max_bytes"] > 64 * 1024**2:
        _error("config.visual_pretraining.weights_max_bytes", "must not exceed the 64 MiB pinned ResNet18 download budget")
    if visual["weights_sha256"] is not None and not visual["weights_sha256"].startswith(visual["weights_sha256_prefix"]):
        _error("config.visual_pretraining.weights_sha256", "must agree with the trusted TorchVision URL hash prefix")
    residual = config["state_residual"]
    if residual["variants"] != ["state_only_residual", "rgb_residual_scratch"]:
        _error("config.state_residual.variants", "must retain the ordered state-only and fresh RGB residual controls")
    for key in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_train_wall_s", "max_gpu_reserved_bytes"):
        if residual[key] != visual[key]:
            _error("config.state_residual." + key, "must retain the declared stage6 training budget")
    coverage = config["trajectory_coverage"]
    for key in ("stationary_speed_rad_s", "moving_speed_rad_s"):
        if coverage[key] != timing[key]:
            _error("config.trajectory_coverage." + key, "must retain the original motion strata")
    if coverage["expansion_group_limit"] > 4 or coverage["expansion_episodes_per_group"] > 2:
        _error("config.trajectory_coverage", "the metadata-only next-batch proposal is bounded to four groups and two episodes per group")
    expansion, expanded = config["train_expansion"], config["expanded_residual"]
    if expansion["candidate_episode_ids"] != [12, 38, 62, 73, 90, 98, 129, 147]:
        _error("config.train_expansion.candidate_episode_ids", "must retain the eight metadata-predeclared candidates")
    if config["source_native"]["split_episode_ids"] != {
            "train": [20, 21, 75, 76, 125, 126], "validation": [180, 181, 182], "test": [280, 281, 282]}:
        _error("config.train_expansion", "the parent split and original validation identities must remain frozen")
    if (expansion["max_total_rows"] > 20000 or expansion["max_candidate_derived_bytes"] > 2560 * 1024**2
            or expansion["max_derived_bytes"] > 5 * 1024**3):
        _error("config.train_expansion", "the bounded candidate and expanded-data storage/row limits were exceeded")
    if expanded["variants"] != residual["variants"]:
        _error("config.expanded_residual.variants", "must retain the state-only and scratch RGB residual controls")
    for key in ("seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size", "max_train_wall_s", "max_gpu_reserved_bytes"):
        if expanded[key] != residual[key]:
            _error("config.expanded_residual." + key, "must retain the fixed stage7 residual training budget")
    for key in ("reference_residual_report_path", "reference_residual_report_sha256"):
        if expanded[key] != expansion[key]:
            _error("config.expanded_residual." + key, "must bind the same frozen residual comparison as the data audit")
    if Path(expanded["data_manifest_path"]) != Path(expansion["build_dir"]) / "manifest.json":
        _error("config.expanded_residual.data_manifest_path", "must name the new expanded-data manifest")
    attribution, constant = config["normalization_attribution"], config["constant_residual"]
    for key in ("variants", "seed", "max_steps", "eval_every_steps", "eval_stride", "eval_batch_size",
                "max_train_wall_s", "max_gpu_reserved_bytes", "data_manifest_path", "data_manifest_sha256",
                "reference_residual_report_path", "reference_residual_report_sha256"):
        if attribution[key] != expanded[key]:
            _error("config.normalization_attribution." + key, "must retain the frozen expanded data and optimizer protocol")
    if Path(attribution["reference_statistics_manifest_path"]) != Path(config["grasp_act"]["dataset_manifest_path"]):
        _error("config.normalization_attribution.reference_statistics_manifest_path", "must identify the original six-train statistics source")
    if constant["fit_cohorts"] != ["original_six", "expanded_fourteen"]:
        _error("config.constant_residual.fit_cohorts", "must retain both predeclared train-only fits")
    if constant["eval_stride"] != expanded["eval_stride"]:
        _error("config.constant_residual.eval_stride", "must retain the original validation anchor cohort")
    for key in ("reference_expanded_report_path", "reference_expanded_report_sha256"):
        if constant[key] != attribution[key]:
            _error("config.constant_residual." + key, "must bind the same frozen fourteen-train comparison")
    conditioning = config["state_conditioning"]
    if conditioning["variants"] != expanded["variants"]:
        _error("config.state_conditioning.variants", "must retain both frozen fourteen-train residual models")
    if conditioning["modes"] != ["real", "train_mean", "within_episode_shift", "cross_episode_swap"]:
        _error("config.state_conditioning.modes", "must retain the four ordered predeclared network-state interventions")
    if conditioning["eval_splits"] != ["train", "validation"]:
        _error("config.state_conditioning.eval_splits", "only train fitting and frozen validation diagnostics are admitted")
    for key in ("eval_stride", "eval_batch_size", "max_gpu_reserved_bytes"):
        if conditioning[key] != expanded[key]:
            _error("config.state_conditioning." + key, "must retain the frozen expanded evaluation protocol")
    for key in ("reference_expanded_report_path", "reference_expanded_report_sha256"):
        if conditioning[key] != attribution[key]:
            _error("config.state_conditioning." + key, "must bind the frozen fourteen-train/new-statistics reference")
    if conditioning["max_eval_wall_s"] > 600:
        _error("config.state_conditioning.max_eval_wall_s", "the inference-only diagnostic wall bound is 600 seconds")
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
    "source_native": ("build_dir", "video_dir"),
    "grasp_act": ("dataset_manifest_path", "memory_report_path", "checkpoint_path"),
    "native_diagnostics": ("training_report_path",),
    "act_diagnostics": ("reference_checkpoint_path", "reference_config_path", "output_dir"),
    "visual_pretraining": ("reference_ablation_report_path", "weights_cache_path", "weights_receipt_path", "output_dir"),
    "state_residual": ("reference_visual_report_path", "admission_report_path", "output_dir"),
    "trajectory_coverage": ("reference_training_report_path",),
    "train_expansion": ("reference_coverage_report_path", "reference_residual_report_path", "session_evidence_path", "candidate_build_dir", "build_dir", "audit_report_path"),
    "expanded_residual": ("data_manifest_path", "reference_residual_report_path", "admission_report_path", "output_dir"),
    "normalization_attribution": ("data_manifest_path", "reference_residual_report_path", "reference_expanded_report_path", "reference_statistics_manifest_path", "admission_report_path", "output_dir"),
    "constant_residual": ("reference_expanded_report_path", "output_dir"),
    "state_conditioning": ("reference_expanded_report_path", "reference_constant_report_path", "output_dir"),
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
