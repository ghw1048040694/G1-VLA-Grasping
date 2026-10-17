"""Dependency-free contracts for a separately versioned privileged RL teacher."""

import hashlib
import json
import math
from pathlib import Path


RL_PROTOCOL = "g1_official_dex3_fixed_root_pregrasp_privileged_v2"
RL_PHYSICS = "physx_unitree_official_usd_fixed_root_kinematic_table_tgs_v3"
RL_PHYSICS_FIXED_PALM_ADJACENCY = "physx_unitree_official_usd_fixed_root_kinematic_table_tgs_v4_fixed_palm_adjacency"
RL_PHYSICS_EFFORT_GRASP = "physx_unitree_official_usd_fixed_root_effort_grasp_v5"
RIGHT_ACTION_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]
RL_TEACHER_DEFAULTS = {
    "enabled": False, "backend": "isaaclab_rsl_ppo_v1", "task_protocol": RL_PROTOCOL,
    "physics_protocol": RL_PHYSICS, "runtime_windows_root": "D:/G1-RL-Teacher",
    "isaac_python_bat": "D:/isaacsim-6.0.1/python.bat", "isaaclab_path": "D:/IsaacLab",
    "git_executable": "D:/tools/MinGit/cmd/git.exe",
    "asset_path": ".runtime/rebuild/rl_assets/official_v1/robot.usd",
    "seed_state_path": ".runtime/rebuild/rl_assets/official_v1/seed_state.json",
    "asset_manifest_path": ".runtime/rebuild/rl_assets/official_v1/asset_manifest.json",
    "asset_manifest_sha256": None, "seed_state_sha256": None,
    "checkpoint_path": None, "checkpoint_sha256": None,
    "action_joint_names": RIGHT_ACTION_JOINTS, "device": "cuda:0", "num_envs": 256,
    "smoke_env_counts": [1, 8, 32, 128, 256], "smoke_steps": 128, "seed": 2026101601,
    "evaluation_seed": 2026101602, "smoke_seed": 2026101603, "episode_length_s": 6.0,
    "steps_per_env": 32, "max_iterations": 400, "max_wall_time_s": 3600,
    "eval_interval": 50, "eval_episodes": 32, "eval_num_envs": 8, "save_interval": 50,
    "hidden_dims": [128, 128], "learning_rate": 0.0003, "num_learning_epochs": 4,
    "num_mini_batches": 4, "gamma": 0.99, "lam": 0.95, "clip_param": 0.2,
    "entropy_coef": 0.01, "init_noise_std": 0.5, "max_grad_norm": 1.0,
    "action_scale_rad": 0.08, "training_position_jitter_m": 0.0002,
    "training_yaw_jitter_rad": 0.001,
    "validation_offsets_m": [[-0.0004, -0.0002], [-0.0004, 0.0002], [-0.0002, -0.0004],
        [-0.0002, 0.0004], [0.0002, -0.0004], [0.0002, 0.0004],
        [0.0004, -0.0002], [0.0004, 0.0002]],
    "target_lift_height_m": 0.006, "probe_height_cap_m": 0.006,
    "probe_overshoot_tolerance_m": 0.0005, "physics_dt_s": 0.002, "decimation": 33,
    "table_top_m": 0.75, "cube_side_m": 0.05, "cube_mass_kg": 0.08,
    "position_iterations": 16, "velocity_iterations": 4, "contact_offset_m": 0.001,
    "rest_offset_m": 0.0, "friction_combine_mode": "max", "hand_friction": 2.2,
    "cube_friction": 1.4, "table_friction": 1.0, "qualification_required": True,
    "palm_friction": 2.0, "max_depenetration_velocity_m_s": 0.2,
    "migration_position_tolerance_m": 0.001, "migration_rotation_tolerance_rad": 0.01,
    "recursive_leaf_pattern_match": False,
    "contact_extraction_backend": "cube_partner_once_v3",
    "reward_reach": 1.0, "reward_fingertip": 0.5, "reward_contact": 1.0,
    "reward_clearance": 4.0, "reward_support": 2.0, "reward_lift": 5.0,
    "reward_success": 10.0, "penalty_action_change": 0.01,
    "penalty_object_speed": 0.05, "penalty_force": 1.0, "penalty_penetration": 2.0,
    "scope": "local_pregrasp_course_only_not_original20_acceptance",
    "training_authorization": "explicit_user_authorized_20261016",
    "reset_protocol": "source15_joint_seed_official_geometry_jitter_v2",
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rl_settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(RL_TEACHER_DEFAULTS):
        raise ValueError("rl_teacher requires its complete declaration with no unknown fields")
    for key, default in RL_TEACHER_DEFAULTS.items():
        value = settings[key]
        if isinstance(default, bool):
            if type(value) is not bool:
                raise ValueError(key + " requires a boolean")
        elif isinstance(default, (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(key + " requires a finite number")
            if isinstance(default, int) and not isinstance(value, int):
                raise ValueError(key + " requires an integer")
        elif isinstance(default, str):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(key + " requires a nonempty string")
        elif isinstance(default, list):
            if not isinstance(value, list) or not value:
                raise ValueError(key + " requires a nonempty list")
    for key in ("backend", "task_protocol", "scope", "reset_protocol",
                "training_authorization", "action_joint_names", "physics_dt_s", "decimation",
                "cube_mass_kg", "cube_side_m", "table_top_m", "friction_combine_mode",
                "hand_friction", "cube_friction", "table_friction", "target_lift_height_m",
                "probe_height_cap_m", "probe_overshoot_tolerance_m", "qualification_required"):
        if settings[key] != RL_TEACHER_DEFAULTS[key]:
            raise ValueError(key + " requires an explicitly new protocol before it can change")
    if settings["physics_protocol"] not in (RL_PHYSICS, RL_PHYSICS_FIXED_PALM_ADJACENCY, RL_PHYSICS_EFFORT_GRASP):
        raise ValueError("physics_protocol requires an explicitly supported physical model")
    for key in ("palm_friction", "max_depenetration_velocity_m_s",
                "migration_position_tolerance_m", "migration_rotation_tolerance_rad", "recursive_leaf_pattern_match",
                "contact_extraction_backend"):
        if settings[key] != RL_TEACHER_DEFAULTS[key]:
            raise ValueError(key + " requires an explicitly new physical protocol")
    if any(settings[key] < 0 for key in RL_TEACHER_DEFAULTS if key.startswith(("reward_", "penalty_"))):
        raise ValueError("RL reward and penalty weights must be nonnegative")
    positive = ("num_envs", "smoke_steps", "steps_per_env", "max_iterations", "max_wall_time_s",
        "eval_interval", "eval_episodes", "eval_num_envs", "save_interval", "learning_rate",
        "num_learning_epochs", "num_mini_batches", "init_noise_std", "max_grad_norm",
        "episode_length_s", "action_scale_rad", "position_iterations", "velocity_iterations")
    if any(settings[key] <= 0 for key in positive):
        raise ValueError("RL budgets and learning parameters must be positive")
    if settings["num_envs"] > 512 or settings["max_iterations"] > 400 or settings["max_wall_time_s"] > 3600:
        raise ValueError("scaled RL budget is at most512 envs,400 updates and3600 seconds")
    if settings["num_envs"] * settings["steps_per_env"] * settings["max_iterations"] > 6553600:
        raise ValueError("scaled RL budget permits at most6553600 training transitions")
    if settings["smoke_steps"] * .066 < settings["episode_length_s"]:
        raise ValueError("smoke step budget must span at least one full episode duration")
    if settings["action_scale_rad"] > .08 or settings["eval_num_envs"] > settings["num_envs"]:
        raise ValueError("action or evaluation selection exceeds its declared bound")
    for key in ("gamma", "lam", "clip_param"):
        if not 0 < settings[key] < 1:
            raise ValueError(key + " must lie strictly between zero and one")
    for key in ("training_position_jitter_m", "training_yaw_jitter_rad", "contact_offset_m",
                "rest_offset_m", "entropy_coef"):
        if settings[key] < 0:
            raise ValueError(key + " cannot be negative")
    for key in ("seed", "evaluation_seed", "smoke_seed"):
        if settings[key] < 0:
            raise ValueError(key + " cannot be negative")
    if len({settings[key] for key in ("seed", "evaluation_seed", "smoke_seed")}) != 3:
        raise ValueError("train, validation and smoke seeds must differ")
    if (settings["smoke_env_counts"] != sorted(set(settings["smoke_env_counts"]))
            or any(type(n) is not int or n <= 0 or n > 512 for n in settings["smoke_env_counts"])
            or settings["num_envs"] not in settings["smoke_env_counts"]):
        raise ValueError("training env count requires a declared monotonic smoke scale")
    if any(type(n) is not int or n <= 0 or n > 256 for n in settings["hidden_dims"]):
        raise ValueError("the initial state policy requires small positive hidden dimensions")
    offsets = settings["validation_offsets_m"]
    if (len(offsets) != settings["eval_num_envs"] or len({tuple(v) for v in offsets}) != len(offsets)
            or any(len(v) != 2 or any(isinstance(n, bool) or not isinstance(n, (int, float))
                or not math.isfinite(n) for n in v) for v in offsets)
            or any(max(map(abs, v)) <= settings["training_position_jitter_m"] for v in offsets)
            or settings["eval_episodes"] % len(offsets)):
        raise ValueError("validation requires fixed, separate offsets and equal episode quotas")
    for key in ("asset_manifest_sha256", "seed_state_sha256", "checkpoint_sha256"):
        value = settings[key]
        if value is not None and (not isinstance(value, str) or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)):
            raise ValueError(key + " requires a full SHA256")
    if bool(settings["checkpoint_path"]) != bool(settings["checkpoint_sha256"]):
        raise ValueError("checkpoint path and full SHA256 must be selected together")
    if settings["enabled"] and not all(settings[k] for k in ("asset_manifest_sha256", "seed_state_sha256")):
        raise ValueError("enabled RL requires independently pinned seed and asset manifest")


def verify_rl_assets(config):
    opts = config["rl_teacher"]
    validate_rl_settings(opts)
    if not opts["enabled"]:
        raise ValueError("RL teacher is disabled")
    root = Path(config["paths"]["root"]).resolve()
    def resolved(key):
        item = Path(opts[key])
        return item if item.is_absolute() else root / item
    manifest_path, seed_path = resolved("asset_manifest_path"), resolved("seed_state_path")
    if file_hash(manifest_path) != opts["asset_manifest_sha256"] or file_hash(seed_path) != opts["seed_state_sha256"]:
        raise ValueError("RL asset or reset seed does not match its pinned SHA256")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    asset_root = manifest_path.parent.resolve()
    for item in manifest["files"]:
        original_path = asset_root / item["path"]
        if original_path.is_symlink():
            raise ValueError("RL assets cannot contain symbolic file substitutions")
        path = original_path.resolve()
        path.relative_to(asset_root)
        if path.is_symlink() or path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError("RL asset file mismatch: " + item["path"])
    entrypoint = manifest.get("entrypoint", "robot.xml")
    pinned_robot = [item for item in manifest["files"] if item["path"] == entrypoint]
    if (len(pinned_robot) != 1 or resolved("asset_path").resolve() != asset_root / entrypoint
            or not entrypoint.endswith((".usd", ".usda", ".usdc"))):
        raise ValueError("RL must load the exact official USD entrypoint in its pinned manifest")
    if (manifest.get("asset_protocol") != "unitree_official_dex3_fixed_base_usd_v1"
            or manifest.get("source", {}).get("owner") != "unitreerobotics"):
        raise ValueError("the active native RL protocol requires its pinned official Unitree provenance")
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    metadata = Path(config["action"]["joint_metadata_path"])
    if not metadata.is_absolute():
        metadata = root / metadata
    names = json.loads(metadata.read_text(encoding="utf-8"))["features"]["action"]["names"]
    if seed["upper31_joint_names"] != names or opts["action_joint_names"] != names[17:31]:
        raise ValueError("RL action bindings differ from the original31 metadata")
    if len(seed["joint_positions_rad"]) != 43 or set(seed["joint_positions_rad"]) != set(seed["actuators"]):
        raise ValueError("the RL robot requires all43 explicitly bound joints")
    return {"manifest": manifest, "seed": seed, "asset_path": str(resolved("asset_path")),
            "entrypoint": entrypoint,
            "manifest_sha256": opts["asset_manifest_sha256"], "seed_sha256": opts["seed_state_sha256"]}


