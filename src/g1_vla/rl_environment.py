"""Privileged-state G1 reinforcement-learning curriculum in Isaac Lab 3.

Isaac, torch and USD are imported only by ``build_environment`` after AppLauncher.
This PhysX curriculum has its own physics identity. It never certifies a MuJoCo
regression layout, exports demonstrations, or interprets reward as task success.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .control import LOWER_BODY_JOINT_NAMES
from .contracts import canonical_sha256, read_joint_order


ENVIRONMENT_VERSION = "g1_physx_official_precontact_rl_v2"
CONTACT_COLLECTION_VERSION = "cube_partner_once_v3"
SUPPORT_GATE_VERSION = "physx_complete_partner_manifold_actual_contact_v4"
TABLE_BODY_PROTOCOL = {"version": "kinematic_rigid_table_contact_channel_v3",
                       "kinematic_enabled": True, "disable_gravity": True,
                       "size_m": (0.84, 1.1, 0.06), "position_m": (0.52, 0.0, 0.72)}
CONTACT_CROSSCHECK = {"num_envs": 1, "physics_samples": 33, "minimum_nonzero_contact_samples": 1,
                      "force_atol_n": 1e-5, "force_rtol": 1e-4,
                      "position_atol_m": 1e-6, "normal_atol": 1e-5}
OBSERVATION_VERSION = "privileged_state_xyzw_81_v1"
ACTION_VERSION = "right_arm_hand_velocity_limited_14_v1"
OBSERVATION_DIM = 81
RIGHT_ARM = tuple("right_{}_joint".format(name) for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw"))
RIGHT_HAND = tuple("right_hand_{}_joint".format(name) for name in (
    "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"))
ACTION_JOINT_NAMES = RIGHT_ARM + RIGHT_HAND
HAND_BODIES = tuple(name.replace("_joint", "_link") for name in RIGHT_HAND)
HAND_SENSOR_BODIES = HAND_BODIES + ("right_wrist_yaw_link",)
DEFAULT_SETTINGS = {
    "environment_version": ENVIRONMENT_VERSION,
    "episode_length_s": 6.0,
    "cube_size_m": 0.05,
    "cube_mass_kg": 0.08,
    "table_top_z_m": 0.75,
    "hand_friction": 2.2,
    "cube_friction": 1.4,
    "table_friction": 1.0,
    "palm_friction": 2.0,
    "max_depenetration_velocity_m_s": 0.2,
    "friction_combine_mode": "max",
    "solver_position_iterations": 16,
    "solver_velocity_iterations": 4,
    "contact_offset_m": 0.001,
    "rest_offset_m": 0.0,
    "train_xy_jitter_m": 0.0002,
    "train_yaw_jitter_rad": 0.001,
    "eval_xy_jitter_m": 0.0,
    "eval_yaw_jitter_rad": 0.0,
    "eval_seed": 2026101602,
    "probe_clearance_max_m": 0.006,
    "probe_overshoot_tolerance_m": 0.0005,
    "support_clearance_min_m": 0.002,
    "support_window_s": 0.198,
    "lift_height_m": 0.006,
    "max_linear_speed_m_s": 0.02,
    "max_angular_speed_rad_s": 0.2,
    "grip_drift_m": 0.003,
    "group_force_min_n": 0.02,
    "group_force_max_n": 1.5,
    "max_penetration_m": 0.003,
    "max_abs_normal_z": 0.6,
    "opposition_dot_max": -0.6,
    "reward_reach": 1.0,
    "reward_fingertip": 0.5,
    "reward_contact": 1.0,
    "reward_clearance": 4.0,
    "reward_support": 2.0,
    "reward_lift": 5.0,
    "reward_success": 10.0,
    "penalty_action_change": 0.01,
    "penalty_object_speed": 0.05,
    "penalty_force": 1.0,
    "penalty_penetration": 2.0,
    "migration_position_tolerance_m": 0.001,
    "migration_rotation_tolerance_rad": 0.01,
}


class RLEnvironmentError(ValueError):
    """An unsafe or incompatible RL environment specification."""


class ContactEquivalenceError(AssertionError):
    """A same-step native contact-view crosscheck failed; never unsupported."""


def control_probe_profile(profile):
    """Validate an explicitly enabled, non-training controller diagnostic."""
    if profile is None:
        return None
    if not isinstance(profile, dict):
        raise RLEnvironmentError("control_probe must be a profile dictionary or None")
    required = ("name", "action_source", "control_guard_version")
    if any(not isinstance(profile.get(key), str) or not profile[key].strip() for key in required):
        raise RLEnvironmentError("control_probe needs name, action_source and control_guard_version")
    if profile["control_guard_version"] not in ("legacy_scalar_v1", "box_halfspace_projected_v2"):
        raise RLEnvironmentError("unknown diagnostic control guard")
    return {key: profile[key] for key in required}


def offset_point_jacobian(torch, link_jacobian, offset_world):
    """Shift a world [linear; angular] Jacobian from link origin to a site."""
    if (link_jacobian.ndim != 3 or link_jacobian.shape[1] != 6 or
            offset_world.shape != (link_jacobian.shape[0], 3)):
        raise RLEnvironmentError("site Jacobian requires (N,6,J) and (N,3)")
    result = link_jacobian.clone()
    correction = torch.cross(link_jacobian[:, 3:].transpose(1, 2),
                             offset_world[:, None, :].expand(-1, link_jacobian.shape[-1], -1), dim=-1)
    result[:, :3] += correction.transpose(1, 2)
    return result


def _probe_payload(torch, value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _probe_payload(torch, item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_probe_payload(torch, item) for item in value]
    return value


def preload_contact_ready(torch, metrics, num_envs, device):
    """Contact prerequisites for a guarded probe; this is not load support."""
    ready = torch.ones(num_envs, dtype=torch.bool, device=device)
    for key in ("contact_safe", "grip_reference_locked", "no_positive_gap_hand_load",
                "friction_partner_manifold_certified", "finite", "joint_safe"):
        value = metrics.get(key)
        if value is None:
            return torch.zeros_like(ready)
        ready &= value
    return ready


class TableContactChannel:
    """Require a native rigid binding and a positive measured load witness."""

    def __init__(self):
        self._bound = False
        self._failed = False
        self._state = {"supported": False, "table_kinematic": True, "disable_gravity": True,
                       "valid_physics_samples": 0, "positive_force_physics_samples": 0,
                       "positive_force_physics_samples_semantics": "minimum per-environment positive 2ms sample count; positive means every Table channel witnessed",
                       "positive_force_any_env_physics_samples": 0, "bound_env_count": 0,
                       "positive_force_env_samples": 0, "max_force_n": 0.0,
                       "unavailable_reason": "awaiting_native_table_binding_and_positive_force_witness"}

    def bind(self, paths, num_envs, *, kinematic, disable_gravity):
        if (not kinematic or not disable_gravity or len(paths) != num_envs or len(set(paths)) != num_envs
                or any(path.rsplit("/", 1)[-1] != "Table" for path in paths)):
            self.unavailable("native Table binding or kinematic/gravity declaration invalid")
            raise RLEnvironmentError(self._state["unavailable_reason"])
        self._bound = True
        self._state["bound_env_count"] = num_envs
        self._state["unavailable_reason"] = "awaiting_positive_normal_force_witness"

    def observe(self, positive_env_count, max_force_n, min_per_env_positive_samples):
        if (not isinstance(positive_env_count, int) or isinstance(positive_env_count, bool) or positive_env_count < 0
                or not isinstance(min_per_env_positive_samples, int) or isinstance(min_per_env_positive_samples, bool)
                or min_per_env_positive_samples < self._state["positive_force_physics_samples"]
                or not math.isfinite(float(max_force_n)) or max_force_n < 0):
            raise FloatingPointError("invalid or nonfinite native Table load summary")
        if not self._bound or self._failed:
            self.unavailable("Table force read without a valid native rigid channel")
            return
        self._state["valid_physics_samples"] += 1
        self._state["positive_force_any_env_physics_samples"] += int(positive_env_count > 0)
        self._state["positive_force_physics_samples"] = min_per_env_positive_samples
        self._state["positive_force_env_samples"] += positive_env_count
        self._state["max_force_n"] = max(self._state["max_force_n"], float(max_force_n))
        if self._state["positive_force_physics_samples"]:
            self._state["supported"] = True
            self._state["unavailable_reason"] = None

    def unavailable(self, reason):
        self._failed = True
        self._state.update(supported=False, unavailable_reason=str(reason))

    def snapshot(self):
        return dict(self._state)


def wxyz_to_xyzw(quaternion):
    """Convert a finite unit quaternion from MuJoCo to Isaac Lab 3 order."""
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise RLEnvironmentError("quaternion must contain four finite values")
    if not math.isclose(sum(value * value for value in values), 1.0, abs_tol=1e-6):
        raise RLEnvironmentError("quaternion must have unit norm")
    return values[1:] + values[:1]


def minimum_box_clearance(position, quaternion_xyzw, *, size_m=0.05, table_top_z_m=0.75):
    """Return the actual minimum rotated box-corner clearance [m]."""
    x, y, z, w = tuple(float(value) for value in quaternion_xyzw)
    if not math.isclose(x*x + y*y + z*z + w*w, 1.0, abs_tol=1e-6):
        raise RLEnvironmentError("quaternion must have unit norm")
    row_z = (2.0 * (x*z - w*y), 2.0 * (y*z + w*x), 1.0 - 2.0 * (x*x + y*y))
    return float(position[2]) - 0.5 * float(size_m) * sum(abs(v) for v in row_z) - float(table_top_z_m)


def consecutive_window_samples(window_s, physics_dt_s):
    """Samples needed so the first and last qualifying times span [s]."""
    if not math.isfinite(window_s) or not math.isfinite(physics_dt_s) or window_s <= 0 or physics_dt_s <= 0:
        raise RLEnvironmentError("positive finite timing is required")
    return int(math.ceil(window_s / physics_dt_s - 1e-10)) + 1


def migration_pose_errors(reference_position, reference_rotation_matrix, actual_position, actual_quaternion_xyzw):
    """Compare imported forward kinematics to the source pose [m, rad]."""
    x, y, z, w = tuple(float(v) for v in actual_quaternion_xyzw)
    wxyz_to_xyzw((w, x, y, z))
    actual_rotation = (
        (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)),
    )
    cosine = (sum(float(reference_rotation_matrix[i][j])*actual_rotation[i][j]
                  for i in range(3) for j in range(3))-1.0)/2.0
    position = math.sqrt(sum((float(a)-float(b))**2 for a, b in zip(reference_position, actual_position)))
    return {"position_error_m": position, "rotation_error_rad": math.acos(min(1.0, max(-1.0, cosine)))}


def policy_velocity_limits(config):
    """Resolve 14 action limits [rad/s] by names, never a positional slice."""
    names = _source_joint_names(config)
    limits = config["control"]["max_velocity_rad_s"]
    if len(names) != 31 or len(set(names)) != 31 or len(limits) != 31:
        raise RLEnvironmentError("the source 31-joint action mapping is required")
    result = tuple(float(limits[names.index(name)]) for name in ACTION_JOINT_NAMES)
    if not all(math.isfinite(value) and value > 0 for value in result):
        raise RLEnvironmentError("every right-arm/hand policy joint needs a positive limit")
    return result


def _source_joint_names(config):
    action = config["action"]
    if "joint_names" in action:
        names = tuple(action["joint_names"])
    else:
        path = Path(action["joint_metadata_path"])
        if not path.is_absolute():
            path = Path(config["paths"]["root"]) / path
        names = read_joint_order(json.loads(path.read_text(encoding="utf-8")))
    if action.get("joint_order_sha256") and canonical_sha256(list(names)) != action["joint_order_sha256"]:
        raise RLEnvironmentError("source 31-joint metadata hash mismatch")
    return names


def resolve_settings(config):
    """Resolve the separate RL scene without changing the MuJoCo configuration."""
    settings = dict(DEFAULT_SETTINGS)
    settings.update(config.get("rl_teacher", {}).get("environment", {}))
    rl = config.get("rl_teacher", {})
    if rl.get("contact_extraction_backend", CONTACT_COLLECTION_VERSION) != CONTACT_COLLECTION_VERSION:
        raise RLEnvironmentError("RL contact extraction backend differs from this source protocol")
    for key in DEFAULT_SETTINGS:
        if key in rl:
            settings[key] = rl[key]
    aliases = {"episode_length_s": "episode_length_s", "cube_size_m": "cube_side_m",
               "table_top_z_m": "table_top_m", "train_xy_jitter_m": "training_position_jitter_m",
               "train_yaw_jitter_rad": "training_yaw_jitter_rad", "eval_seed": "evaluation_seed",
               "lift_height_m": "target_lift_height_m", "solver_position_iterations": "position_iterations",
               "solver_velocity_iterations": "velocity_iterations", "probe_clearance_max_m": "probe_height_cap_m"}
    for destination, source in aliases.items():
        if source in rl:
            settings[destination] = rl[source]
    if settings["environment_version"] != ENVIRONMENT_VERSION:
        raise RLEnvironmentError("unsupported RL environment version")
    control = config["control"]
    if not math.isclose(float(control["physics_dt_s"]), 0.002, abs_tol=1e-12) or control["physics_substeps"] != 33:
        raise RLEnvironmentError("RL curriculum requires explicit 2 ms physics and 33-step control")
    for key, value in settings.items():
        if isinstance(DEFAULT_SETTINGS[key], (int, float)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise RLEnvironmentError("finite numeric RL setting required: " + key)
    if settings["cube_size_m"] != 0.05 or settings["cube_mass_kg"] != 0.08 or settings["table_top_z_m"] != 0.75:
        raise RLEnvironmentError("version 1 pins the original cube mass, size, and table height")
    if settings["friction_combine_mode"] != "max":
        raise RLEnvironmentError("version 1 requires explicitly declared max friction combination")
    if settings["lift_height_m"] != 0.006:
        raise RLEnvironmentError("this local curriculum permits only the 6 mm probe target")
    if rl.get("action_joint_names", list(ACTION_JOINT_NAMES)) != list(ACTION_JOINT_NAMES):
        raise RLEnvironmentError("RL policy action joint order differs from the explicit 14-joint contract")
    policy_velocity_limits(config)
    return settings


def _seed_state(config):
    rl = config["rl_teacher"]
    state = rl.get("initial_state")
    if state is None:
        seed_path = rl.get("seed_state_path") or rl.get("initial_state_path")
        if not seed_path:
            raise RLEnvironmentError("a versioned physical contact-entry seed state is required")
        path = Path(seed_path)
        if not path.is_absolute():
            path = Path(config["paths"]["root"]) / path
        payload = path.read_bytes()
        expected = rl.get("seed_state_sha256")
        if expected and hashlib.sha256(payload).hexdigest() != expected:
            raise RLEnvironmentError("RL seed-state hash mismatch")
        state = json.loads(payload)
    joints = state.get("joint_positions_rad", state.get("joint_position_rad"))
    if not isinstance(joints, dict):
        raise RLEnvironmentError("seed state must explicitly name all 43 joint positions")
    expected_names = set(_source_joint_names(config)) | set(LOWER_BODY_JOINT_NAMES)
    if set(joints) != expected_names or len(expected_names) != 43:
        raise RLEnvironmentError("seed state and source 43-joint identity differ")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in joints.values()):
        raise RLEnvironmentError("seed joint state must be finite")
    root_pose = state.get("root_pose_xyzw")
    if root_pose is None and "root_pose_wxyz" in state:
        root_pose = state["root_pose_wxyz"][:3] + list(wxyz_to_xyzw(state["root_pose_wxyz"][3:]))
    cube_pose = state.get("cube_pose_xyzw")
    if cube_pose is None and "cube_pose_wxyz" in state:
        cube_pose = state["cube_pose_wxyz"][:3] + list(wxyz_to_xyzw(state["cube_pose_wxyz"][3:]))
    if cube_pose is None and "object_position_world_m" in state:
        if "object_quaternion_xyzw" not in state:
            raise RLEnvironmentError("actual seed object quaternion must be recorded explicitly")
        cube_pose = list(state["object_position_world_m"]) + list(state["object_quaternion_xyzw"])
    for name, pose in (("root_pose_xyzw", root_pose), ("cube_pose_xyzw", cube_pose)):
        if not isinstance(pose, (list, tuple)) or len(pose) != 7 or not all(math.isfinite(float(v)) for v in pose):
            raise RLEnvironmentError("seed needs a finite seven-value " + name)
        wxyz_to_xyzw([pose[6], pose[3], pose[4], pose[5]])
    return dict(state, joint_positions_rad=joints, root_pose_xyzw=list(root_pose), cube_pose_xyzw=list(cube_pose))


def _actuator_specifications(initial):
    values = initial.get("actuators")
    if isinstance(values, dict):
        values = [dict(item, name=name) for name, item in values.items()]
    if not isinstance(values, list):
        raise RLEnvironmentError("source 43-actuator PD/force/dynamics audit is required")
    result = {}
    for item in values:
        name = item["name"]
        force_range = item.get("force_range", item.get("force_range_nm"))
        if not isinstance(force_range, (list, tuple)) or len(force_range) != 2:
            raise RLEnvironmentError("actuator needs original finite force_range")
        if not math.isclose(float(force_range[0]), -float(force_range[1]), abs_tol=1e-8):
            raise RLEnvironmentError("asymmetric original effort limits need an explicit new actuator implementation")
        result[name] = {"kp": float(item["kp"]), "kd": float(item["kd"]),
                        "effort_limit_nm": float(force_range[1]),
                        "armature": float(item.get("armature", item.get("armature_kg_m2"))),
                        "dof_damping": float(item.get("dof_damping", item.get("damping_nm_s_rad", item.get("damping"))))}
        if not all(math.isfinite(value) and value >= 0 for value in result[name].values()):
            raise RLEnvironmentError("actuator settings must be finite and nonnegative")
    if set(result) != set(initial["joint_positions_rad"]):
        raise RLEnvironmentError("source actuator identity differs from the 43 reset joints")
    return result


def audit_asset_manifest(config, asset_path):
    """Verify the pinned robot, reset and every referenced mesh byte."""
    rl = config["rl_teacher"]
    manifest_path = Path(rl["asset_manifest_path"])
    if not manifest_path.is_absolute():
        manifest_path = Path(config["paths"]["root"]) / manifest_path
    payload = manifest_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != rl["asset_manifest_sha256"]:
        raise RLEnvironmentError("RL asset manifest hash mismatch")
    manifest = json.loads(payload)
    seen, asset_verified = set(), False
    for entry in manifest["files"]:
        path = (manifest_path.parent / entry["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent.resolve()) or path in seen:
            raise RLEnvironmentError("asset manifest path escapes or duplicates its asset root")
        seen.add(path)
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise RLEnvironmentError("RL asset manifest file mismatch: " + entry["path"])
        asset_verified |= path == Path(asset_path).resolve()
    if not asset_verified:
        raise RLEnvironmentError("prepared robot is not included in its immutable asset manifest")
    return {"manifest_sha256": hashlib.sha256(payload).hexdigest(), "verified_file_count": len(seen)}


def audit_fixed_root_joints(joints):
    """Require one enabled world-to-pelvis joint, without creating another."""
    world_joints = [item for item in joints if bool(item["body0"]) != bool(item["body1"])]
    if len(world_joints) != 1:
        raise RLEnvironmentError("robot asset must contain exactly one world-fixed root joint")
    fixed = world_joints[0]
    attached = fixed["body0"] or fixed["body1"]
    if not fixed["enabled"] or len(attached) != 1 or attached[0].rsplit("/", 1)[-1] != "pelvis":
        raise RLEnvironmentError("world root joint must be enabled and attached only to the pelvis")
    return {"world_fixed_joint_count": 1, "fixed_joint": fixed}


def sensor_body_path_map(entries, required_names):
    """Resolve rigid bodies without allowing same-named render meshes to win."""
    result = {}
    for name, path, is_rigid_body in entries:
        if name in required_names and is_rigid_body:
            if name in result:
                raise RLEnvironmentError("duplicate rigid sensor body identity: " + name)
            result[name] = path
    missing = set(required_names)-set(result)
    if missing:
        raise RLEnvironmentError("asset lacks rigid sensor bodies: " + repr(sorted(missing)))
    return result


def cube_contact_partner_layout(robot_body_paths, hand_contact_names):
    """Map exact native partners; every non-hand robot body remains explicit."""
    if not set(hand_contact_names).issubset(robot_body_paths):
        raise RLEnvironmentError("cube partner ledger is missing a declared hand contact body")
    names = ["Table"] + sorted(robot_body_paths)
    return names, [index for index, name in enumerate(names) if index > 0 and name not in hand_contact_names]


def _ragged_contact_indices(torch, counts, starts, capacity):
    """Copy shared SDK indices, validate once, and unpack all pairs together."""
    # Native normal/friction getters reuse the same counts/starts storage.
    # Conversion copies both arrays before a subsequent native getter runs.
    counts, starts = counts.to(dtype=torch.int64).clone(), starts.to(dtype=torch.int64).clone()
    flat_count, flat_start = counts.flatten(), starts.flatten()
    valid = (flat_count >= 0) & ((flat_count == 0) | ((flat_start >= 0) & (flat_start+flat_count <= capacity)))
    if not bool(valid.all()):
        raise RLEnvironmentError("native contact stream overflow or invalid pair indices")
    bucket = torch.repeat_interleave(torch.arange(flat_count.numel(), device=counts.device), flat_count)
    offset = torch.repeat_interleave(torch.cumsum(flat_count, 0)-flat_count, flat_count)
    local = torch.arange(bucket.numel(), device=counts.device)-offset
    indices = torch.repeat_interleave(flat_start, flat_count)+local
    return counts, starts, bucket, indices, local


def _reduce_friction_pairs(torch, forces, points, counts, starts):
    """Reduce the installed SDK's (forces, points, counts, starts) tuple."""
    counts, starts, bucket, indices, local = _ragged_contact_indices(torch, counts, starts, forces.shape[0])
    selected_force, selected_points = forces[indices], points[indices]
    if not bool(torch.isfinite(torch.cat((selected_force.flatten(), selected_points.flatten()))).all()):
        raise FloatingPointError("nonfinite native friction force/position")
    if bool(counts.sum() >= forces.shape[0]):
        raise RLEnvironmentError("saturated friction buffer cannot certify complete patch data")
    result = torch.zeros(counts.numel(), 3, dtype=forces.dtype, device=forces.device)
    result.index_add_(0, bucket, selected_force)
    nonzero_patch_count = torch.zeros(counts.numel(), dtype=torch.int64, device=forces.device)
    nonzero_patch_count.scatter_add_(0, bucket, selected_force.ne(0).any(-1).long())
    return result.reshape(*counts.shape, 3), {"bucket": bucket, "indices": indices,
                                             "force": selected_force, "point": selected_points,
                                             "counts": counts, "starts": starts,
                                             "nonzero_patch_count": nonzero_patch_count.reshape(counts.shape)}


def _reduce_contact_pairs(torch, *, normal, friction, net_normal, force, points, normals,
                          separation, counts, starts, hand_indices, force_threshold, hand_inverse=None,
                          object_up_world=None, friction_nonzero_patch_count=None):
    """One sparse GPU reduction for every Cube partner, plus all eight hands.

    Aggregate hand vectors act on the hand. Native point normals are shared
    canonical directions between reversed views, while their signed scalar
    forces reverse. Cube raw positions, separations and counts stay unchanged.
    No view ordering is assumed: hand_indices explicitly maps named bodies.
    """
    num_envs, width = counts.shape
    counts, starts, bucket, indices, local = _ragged_contact_indices(torch, counts, starts, force.numel())
    raw_force, raw_normals = force[indices], normals[indices]
    raw_separation, raw_points = separation[indices], points[indices]
    finite_values = torch.cat(tuple(value.flatten() for value in
                                   (normal, friction, net_normal, raw_force, raw_normals, raw_separation, raw_points)))
    if not bool(torch.isfinite(finite_values).all()):
        raise FloatingPointError("nonfinite native cube partner contact stream")
    if bool((counts.sum() >= force.numel()) | (counts.sum(-1) >= 256).any()):
        raise RLEnvironmentError("saturated normal contact buffer cannot certify a complete partner manifold")
    fn = torch.zeros(num_envs*width, dtype=normal.dtype, device=normal.device)
    depth, vertical = torch.zeros_like(fn), torch.zeros_like(fn)
    physically_loaded = torch.zeros(num_envs*width, dtype=torch.int64, device=normal.device)
    positive_gap = torch.zeros_like(physically_loaded)
    positive_gap_force = torch.zeros_like(physically_loaded)
    fn.scatter_add_(0, bucket, raw_force.abs())
    depth.scatter_reduce_(0, bucket, (-raw_separation).clamp_min(0), reduce="amax", include_self=True)
    physical_point = (raw_force.abs() > 0) & (raw_separation <= 0)
    loaded = (raw_force.abs() > force_threshold) & physical_point
    owners = torch.div(bucket, width, rounding_mode="floor")
    side_up = raw_normals[:, 2].abs()
    if object_up_world is not None:
        if not bool(torch.isfinite(object_up_world).all()):
            raise FloatingPointError("nonfinite actual cube up direction")
        side_up = torch.maximum(side_up, (raw_normals*object_up_world[owners]).sum(-1).abs())
    vertical.scatter_reduce_(0, bucket, torch.where(physical_point, side_up, 0), reduce="amax", include_self=True)
    physically_loaded.scatter_add_(0, bucket, physical_point.long())
    positive_gap.scatter_add_(0, bucket, (raw_separation > 0).long())
    positive_gap_force.scatter_add_(0, bucket, ((raw_force.abs() > 0) & (raw_separation > 0)).long())
    fn, depth, vertical, physically_loaded, positive_gap, positive_gap_force = (
        value.reshape(num_envs, width) for value in
        (fn, depth, vertical, physically_loaded, positive_gap, positive_gap_force))
    if not hasattr(hand_indices, "device") and len(set(hand_indices)) != 8:
        raise RLEnvironmentError("eight distinct named hand partners are required")
    hand_indices = torch.as_tensor(hand_indices, device=normal.device, dtype=torch.int64)
    if hand_indices.numel() != 8:
        raise RLEnvironmentError("eight distinct named hand partners are required")
    if bool((counts[:, hand_indices] > 32).any()):
        raise RLEnvironmentError("hand contact count exceeds the unchanged 32-point capacity")
    inverse = hand_inverse
    if inverse is None:
        inverse = torch.full((width,), -1, device=normal.device, dtype=torch.int64)
        inverse[hand_indices] = torch.arange(8, device=normal.device)
    hand_id = inverse[bucket % width]
    selected = hand_id >= 0
    owner = owners[selected]
    hand_id, hand_local = hand_id[selected], local[selected]
    padded_normal = torch.zeros(num_envs, 8, 32, 3, device=normal.device, dtype=normal.dtype)
    padded_point = torch.zeros_like(padded_normal)
    padded_separation = torch.zeros(num_envs, 8, 32, device=normal.device, dtype=normal.dtype)
    padded_force = torch.zeros_like(padded_separation)
    padded_valid = torch.zeros_like(padded_separation, dtype=torch.bool)
    padded_loaded = torch.zeros_like(padded_valid)
    padded_normal[owner, hand_id, hand_local] = raw_normals[selected]
    padded_point[owner, hand_id, hand_local] = raw_points[selected]
    padded_separation[owner, hand_id, hand_local] = raw_separation[selected]
    padded_force[owner, hand_id, hand_local] = -raw_force[selected]
    padded_valid[owner, hand_id, hand_local] = True
    padded_loaded[owner, hand_id, hand_local] = loaded[selected]
    hand = {"normal": -normal[:, hand_indices], "friction": -friction[:, hand_indices],
            "fn": fn[:, hand_indices], "depth": depth[:, hand_indices], "vertical": vertical[:, hand_indices],
            "physical_loaded_count": physically_loaded[:, hand_indices],
            "point_normal": padded_normal, "point_loaded": padded_loaded, "point_valid": padded_valid,
            "point": padded_point, "separation": padded_separation, "point_force": padded_force}
    if friction_nonzero_patch_count is None:
        if bool(friction.ne(0).any()):
            raise RLEnvironmentError("nonzero friction needs explicit raw patch counts for manifold certification")
        friction_nonzero_patch_count = torch.zeros_like(counts)
    hand.update(raw_point_count=counts[:, hand_indices], positive_gap_point_count=positive_gap[:, hand_indices],
                positive_gap_force_count=positive_gap_force[:, hand_indices],
                friction_nonzero_patch_count=friction_nonzero_patch_count[:, hand_indices])
    return {"normal": normal, "friction": friction, "fn": fn, "depth": depth,
            "physical_loaded_count": physically_loaded, "hand": hand,
            "unfiltered_normal_residual_n": torch.linalg.vector_norm(net_normal-normal.sum(dim=1), dim=-1),
            "raw": {"bucket": bucket, "force": raw_force, "normal": raw_normals,
                    "separation": raw_separation, "point": raw_points, "counts": counts, "starts": starts}}


def _summarize_hand_contacts(torch, hand):
    """Apply the unchanged per-finger/group and raw distal-normal definitions."""
    fn = hand["fn"]
    groups = torch.stack((fn[:, :3].sum(dim=1), fn[:, 3:5].sum(dim=1), fn[:, 5:7].sum(dim=1)), dim=-1)
    physical = hand["physical_loaded_count"]
    group_physical = torch.stack((physical[:, :3].sum(dim=1), physical[:, 3:5].sum(dim=1),
                                  physical[:, 5:7].sum(dim=1)), dim=-1)
    no_positive_gap_load = (hand["positive_gap_force_count"] == 0).all(-1)
    friction_active = hand["friction_nonzero_patch_count"] > 0
    friction_manifold = ((~friction_active) | ((hand["raw_point_count"] > 0) &
                                              (hand["positive_gap_point_count"] == 0))).all(-1)
    distal_normal, distal_loaded = hand["point_normal"][:, (2, 4, 6)], hand["point_loaded"][:, (2, 4, 6)]
    dots = torch.einsum("nki,ngli->ngkl", distal_normal[:, 0], distal_normal[:, 1:])
    valid = distal_loaded[:, 0, None, :, None] & distal_loaded[:, 1:, None, :]
    values = torch.where(valid, dots, -torch.ones_like(dots)).amax(dim=(-1, -2))
    opposition = torch.where(valid.any(dim=(-1, -2)), values, torch.ones_like(values)).amax(dim=-1)
    return {"normal_sum": -hand["normal"].sum(dim=1),
            "total_sum": -(hand["normal"]+hand["friction"]).sum(dim=1),
            "groups": groups, "group_physical_loaded_count": group_physical, "depth": hand["depth"].amax(dim=1),
            "no_positive_gap_hand_load": no_positive_gap_load, "friction_manifold_certified": friction_manifold,
            "normal_vertical": hand["vertical"].amax(dim=1),
            "opposition": opposition}


def _compare_hand_contacts(torch, actual, reference, tolerances=CONTACT_CROSSCHECK):
    """Compare independent views without assuming their raw point ordering."""
    differences, valid = {}, []
    for name in ("normal", "friction", "fn", "depth", "vertical", "physical_loaded_count"):
        left, right = actual[name], reference[name]
        differences[name] = float((left-right).abs().amax().item())
        if name == "physical_loaded_count":
            valid.append(torch.equal(left, right))
        else:
            is_force = name in ("normal", "friction", "fn")
            atol = tolerances["force_atol_n"] if is_force else (
                tolerances["position_atol_m"] if name == "depth" else tolerances["normal_atol"])
            valid.append(bool(torch.isclose(left, right, atol=atol,
                                           rtol=tolerances["force_rtol"] if is_force else 0).all()))
    valid.append(torch.equal(actual["point_valid"].sum(-1), reference["point_valid"].sum(-1)))
    position_difference = torch.linalg.vector_norm(actual["point"][:, :, :, None]-reference["point"][:, :, None, :], dim=-1)
    normal_difference = torch.linalg.vector_norm(actual["point_normal"][:, :, :, None]-reference["point_normal"][:, :, None, :], dim=-1)
    separation_difference = (actual["separation"][:, :, :, None]-reference["separation"][:, :, None, :]).abs()
    point_force_close = torch.isclose(actual["point_force"][:, :, :, None], reference["point_force"][:, :, None, :],
                                     atol=tolerances["force_atol_n"], rtol=tolerances["force_rtol"])
    matches = (position_difference <= tolerances["position_atol_m"]) & (normal_difference <= tolerances["normal_atol"])
    matches &= (separation_difference <= tolerances["position_atol_m"]) & point_force_close
    matches &= actual["point_valid"][:, :, :, None] & reference["point_valid"][:, :, None, :]
    valid.append(bool(((~actual["point_valid"]) | matches.any(-1)).all()))
    valid.append(bool(((~reference["point_valid"]) | matches.any(-2)).all()))
    # The actual gates are exact Boolean comparisons: numeric tolerances never
    # authorize a disagreement at one of the original physical thresholds.
    return {"passed": all(valid), "maximum_absolute_differences": differences,
            "unordered_raw_points_match": valid[-1] and valid[-2] and valid[-3]}


def build_environment(config, *, num_envs, device, asset_path, train_seed, eval_mode=False, output_dir=None,
                      control_probe=None):
    """Build the state-only local grasp curriculum after Isaac AppLauncher.

    The fixed robot and free cube are reset only at episode boundaries. Force,
    separation and normal data must all be available for independent success.
    """
    probe_profile = control_probe_profile(control_probe)
    if probe_profile is not None and (not eval_mode or output_dir is None or num_envs > 8):
        raise RLEnvironmentError("control diagnostics require standalone evaluation, output and at most eight environments")
    settings = resolve_settings(config)
    initial = _seed_state(config)
    actuator_specs = _actuator_specifications(initial)
    asset_path = Path(asset_path).resolve()
    if not asset_path.is_file() or asset_path.suffix.lower() not in (".usd", ".usda", ".usdc", ".xml"):
        raise RLEnvironmentError("a prepared robot-only MJCF or USD asset is required")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise RLEnvironmentError("num_envs must be a positive integer")
    asset_sha = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    asset_audit = audit_asset_manifest(config, asset_path)
    native_official_asset = asset_path.suffix.lower() in (".usd", ".usda", ".usdc")
    if native_official_asset:
        reference = initial.get("official_native_fk_reference")
        if not reference or reference.get("usd_entry_sha256") != asset_sha:
            raise RLEnvironmentError("the official USD requires its independently computed and asset-pinned native FK reference")
    palm_site_body = initial["palm_site_body_name"]
    palm_contact_body = initial.get("palm_contact_body_name", "right_wrist_yaw_link")
    sensor_body_names = HAND_BODIES + (palm_contact_body,)
    if len(set(sensor_body_names)) != 8:
        raise RLEnvironmentError("the seven finger and one palm contact bodies must be distinct")
    output = Path(output_dir).resolve() if output_dir else None
    # These imports intentionally follow AppLauncher, not the project's CLI.
    import torch
    import warp as wp
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab.sim.utils.prims import bind_physics_material
    from pxr import UsdPhysics
    from isaaclab.utils.configclass import configclass
    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul
    from isaaclab_physx.physics import PhysxCfg
    import carb
    tensor_settings = carb.settings.get_settings()
    if config["rl_teacher"].get("recursive_leaf_pattern_match", False) is not False:
        raise RLEnvironmentError("the exact-body contact protocol requires nonrecursive leaf matching")
    tensor_settings.set_bool("/physics/tensors/recursiveLeafPatternMatch", False)
    if tensor_settings.get("/physics/tensors/recursiveLeafPatternMatch") is not False:
        raise RLEnvironmentError("native physics tensor leaf matching setting did not apply")

    def tensor(value):
        return value.torch if hasattr(value, "torch") else (wp.to_torch(value) if isinstance(value, wp.array) else value)

    def material(friction):
        return sim_utils.RigidBodyMaterialCfg(static_friction=friction, dynamic_friction=friction,
                                             restitution=0.0, friction_combine_mode="max",
                                             restitution_combine_mode="max")

    rigid = sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, linear_damping=0.0, angular_damping=0.0,
                                            max_depenetration_velocity=settings["max_depenetration_velocity_m_s"],
                                            solver_position_iteration_count=settings["solver_position_iterations"],
                                            solver_velocity_iteration_count=settings["solver_velocity_iterations"])
    collision = sim_utils.CollisionPropertiesCfg(contact_offset=settings["contact_offset_m"],
                                                 rest_offset=settings["rest_offset_m"])
    spawn_args = dict(activate_contact_sensors=True, rigid_props=rigid,
                      collision_props=collision,
                      articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                          # MjcfFileCfg.fix_base already creates a sibling
                          # fix_base_joint. Geometry is the articulation root,
                          # not a rigid body: a second generic fixer is invalid.
                          enabled_self_collisions=True,
                          solver_position_iteration_count=settings["solver_position_iterations"],
                          solver_velocity_iteration_count=settings["solver_velocity_iterations"]))
    if asset_path.suffix.lower() == ".xml":
        cache_dir = asset_path.parent / "isaac_physx_cache_v1"
        cache_dir.mkdir(parents=True, exist_ok=True)
        robot_spawn = sim_utils.MjcfFileCfg(asset_path=str(asset_path), usd_dir=str(cache_dir),
                                           usd_file_name="g1_rl_physx_v1.usd", fix_base=True,
                                           self_collision=True, import_physics_scene=False,
                                           collision_from_visuals=False, collision_type="Convex Hull", **spawn_args)
    else:
        robot_spawn = sim_utils.UsdFileCfg(usd_path=str(asset_path), **spawn_args)

    @configclass
    class EnvironmentCfg(DirectRLEnvCfg):
        decimation = 33
        episode_length_s = float(settings["episode_length_s"])
        action_space = len(ACTION_JOINT_NAMES)
        observation_space = OBSERVATION_DIM
        state_space = 0
        seed = int(train_seed)
        sim = SimulationCfg(dt=0.002, device=device, render_interval=33,
                            physics=PhysxCfg(solver_type=1, enable_stabilization=False,
                                             solve_articulation_contact_last=True,
                                             gpu_max_rigid_contact_count=2**19,
                                             gpu_max_rigid_patch_count=2**15,
                                             gpu_found_lost_pairs_capacity=2**19,
                                             gpu_found_lost_aggregate_pairs_capacity=2**20,
                                             gpu_total_aggregate_pairs_capacity=2**20),
                            physics_material=material(settings["table_friction"]))
        scene = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True)

    class Environment(DirectRLEnv):
        """An explicitly local PhysX curriculum, with fail-closed physical scoring."""

        def __init__(self):
            self._source_config = config
            self._initial = initial
            self._settings = settings
            self._eval_mode = bool(eval_mode)
            self._evaluation_seed = int(settings["eval_seed"])
            self._outcomes = []
            self._resets = 0
            self._interrupted = 0
            self._physics_samples = 0
            self._pending_physics_sample = False
            self._force_unavailable = None
            self._qualification_supported = False
            self._table_contact_channel = TableContactChannel()
            self._standalone_video = bool(eval_mode)
            self._evaluation_camera = None
            self._video_status = {"requested": bool(eval_mode), "enabled": False,
                                  "frames": 0, "failure": None, "camera_scope": "fixed external env0 audit view"}
            self._video_manifest = None
            self._last_video_native_step = None
            self._last_camera_frame = None
            self._next_action_origin = "policy_or_zero_target_hold"
            self._action_origin = "not_started"
            self._state_hold_request_clipped = 0
            self._control_probe_enabled = probe_profile is not None
            self._control_probe_profile = probe_profile
            self._control_probe_stream = None
            self._control_command_id = 0
            self._control_transition_snapshot = None
            self._control_transition_recorded = 0
            self._control_guard_correction_count = 0
            self._control_guard_unsafe_count = 0
            self._contact_crosscheck = {"protocol": CONTACT_COLLECTION_VERSION,
                                       "required_physics_samples": CONTACT_CROSSCHECK["physics_samples"],
                                       "enabled": num_envs == CONTACT_CROSSCHECK["num_envs"],
                                       "completed_physics_samples": 0, "completed_nonzero_samples": 0, "started": False,
                                       "status": "pending_nonzero_contact" if num_envs == 1 else "not_requested_multi_env",
                                       "maximum_absolute_differences": {}, "first_failure": None}
            self._crosscheck_record = None
            super().__init__(EnvironmentCfg(), render_mode=None)
            self._contact_body_binding_audit = {}
            binding_names = (list(sensor_body_names) if self._hand_sensors else []) + ["Cube"]
            for sensor, name in zip(self._hand_sensors + [self._cube_sensor], binding_names):
                paths = list(sensor.body_physx_view.prim_paths)
                if len(paths) != self.num_envs or any(path.rsplit("/", 1)[-1] != name for path in paths):
                    raise RLEnvironmentError("native contact sensor did not bind exactly one declared rigid body per environment: " + name)
                self._contact_body_binding_audit[name] = paths
            table_paths = list(self.table.root_view.prim_paths)
            self._table_contact_channel.bind(table_paths, self.num_envs,
                                             kinematic=self._table_schema_audit["kinematic_enabled"],
                                             disable_gravity=self._table_schema_audit["disable_gravity"])
            self._contact_body_binding_audit["Table"] = table_paths
            # The full Cube ledger explicitly names every hand. Only the
            # bounded one-environment crosscheck creates the old eight views.
            self._cube_hand_filter_indices = [self._cube_filter_body_names.index(name) for name in sensor_body_names]
            self._cube_hand_index_tensor = torch.tensor(self._cube_hand_filter_indices, device=self.device)
            self._cube_hand_inverse = torch.full((len(self._cube_filter_body_names),), -1, dtype=torch.int64, device=self.device)
            self._cube_hand_inverse[self._cube_hand_index_tensor] = torch.arange(8, device=self.device)
            if not self.robot.is_fixed_base or self.robot.num_base_dofs != 0:
                raise RLEnvironmentError("native physics did not recognize the imported robot as fixed-root")
            if self._evaluation_camera is not None:
                eye = self.scene.env_origins[0] + torch.tensor([1.05, -0.65, 1.30], device=self.device)
                target = self.scene.env_origins[0] + torch.tensor([0.35, 0.055, 0.80], device=self.device)
                self._evaluation_camera.set_world_poses_from_view(eye[None], target[None])
                self._video_status.update(enabled=True, eye_world_m=eye.detach().cpu().tolist(),
                                          target_world_m=target.detach().cpu().tolist())
            self._action_ids = [self.robot.joint_names.index(name) for name in ACTION_JOINT_NAMES]
            if len(self.robot.joint_names) != 43 or set(self.robot.joint_names) != set(initial["joint_positions_rad"]):
                raise RLEnvironmentError("converted asset does not contain the exact 43 robot joints")
            self._palm_id = self.robot.body_names.index(palm_site_body)
            self._tip_ids = [self.robot.body_names.index(name) for name in (
                "right_hand_thumb_2_link", "right_hand_middle_1_link", "right_hand_index_1_link")]
            self._limits = tensor(self.robot.data.joint_pos_limits)
            self._source_joint_limit_differences = {}
            for index, name in enumerate(self.robot.joint_names):
                source_range = initial.get("original_source_joint_ranges_rad", {}).get(
                    name, initial["actuators"][name]["joint_range_rad"])
                expected_range = [math.radians(self._joint_schema_audit[name][key])
                                  for key in ("lower_limit_degrees", "upper_limit_degrees")]
                actual_range = self._limits[0, index].detach().cpu().tolist()
                if any(not math.isclose(a, float(b), abs_tol=1e-5, rel_tol=0) for a, b in zip(actual_range, expected_range)):
                    raise RLEnvironmentError("native radian mechanical range differs from USD degree schema: " + name)
                self._source_joint_limit_differences[name] = [a-float(b) for a, b in zip(actual_range, source_range)]
                tolerance = float(config["eval"]["joint_limit_tolerance_rad"])
                if not actual_range[0]-tolerance <= initial["joint_positions_rad"][name] <= actual_range[1]+tolerance:
                    raise RLEnvironmentError("source joint seed is outside the official asset's mechanical range: " + name)
            self._velocity = torch.tensor(policy_velocity_limits(config), device=self.device, dtype=torch.float32)
            self._initial_q = torch.tensor([initial["joint_positions_rad"][name] for name in self.robot.joint_names],
                                           device=self.device, dtype=torch.float32)
            self._palm_local = torch.tensor(initial["palm_site_position_m"], device=self.device,
                                            dtype=torch.float32).repeat(self.num_envs, 1)
            self._palm_local_rotation = torch.tensor(initial["palm_site_quaternion_xyzw"], device=self.device,
                                                     dtype=torch.float32).repeat(self.num_envs, 1)
            self._targets = self._initial_q.repeat(self.num_envs, 1)
            self._actions = torch.zeros(self.num_envs, 14, device=self.device)
            self._previous_actions = torch.zeros_like(self._actions)
            self._episode_ids = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self._table_positive_per_env = torch.zeros_like(self._episode_ids)
            self._episode_return = torch.zeros(self.num_envs, device=self.device)
            self._support_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self._support_qualified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._local_success = torch.zeros_like(self._support_qualified)
            self._unsafe = torch.zeros_like(self._support_qualified)
            self._probe_overshoot = torch.zeros_like(self._support_qualified)
            self._grip_locked = torch.zeros_like(self._support_qualified)
            self._grip_reference = torch.zeros(self.num_envs, 3, device=self.device)
            self._grip_rotation = torch.zeros(self.num_envs, 4, device=self.device)
            self._grip_rotation[:, 3] = 1.0
            self._start_cube_z = torch.zeros(self.num_envs, device=self.device)
            self._start_palm_z = torch.full((self.num_envs,), float("nan"), device=self.device)
            self._shield_count = 0
            self._max_clearance = torch.full((self.num_envs,), -float("inf"), device=self.device)
            self._max_support_samples = torch.zeros_like(self._support_count)
            self._last_metrics = {}
            self._reward_terms = {}
            self._train_generator = torch.Generator(device=self.device).manual_seed(int(train_seed))
            self._eval_generator = torch.Generator(device=self.device).manual_seed(int(settings["eval_seed"]))
            self._record = None
            self._record_index = 0
            self._raw_contact_tensors = {}
            self._mechanical_migration = None
            self._expected_reset_cube = torch.zeros(self.num_envs, 7, device=self.device)
            self._compiled_files = []
            if asset_path.suffix.lower() == ".xml":
                for path in sorted(cache_dir.rglob("*")):
                    if path.is_file() and path.suffix.lower() in (".usd", ".usda", ".usdc", ".json"):
                        self._compiled_files.append({"path": str(path.relative_to(cache_dir)),
                                                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if output:
                output.mkdir(parents=True, exist_ok=True)
                if self._control_probe_enabled:
                    self._control_probe_stream = (output / "control_commands.jsonl").open("x", encoding="utf-8")
                if eval_mode:
                    self._open_evaluation_recording()
                    self._video_frame_dir = output / "rgb_frames"
                    self._video_frame_dir.mkdir(exist_ok=False)
                    self._video_manifest = (output / "video_frames.jsonl").open("x", encoding="utf-8")
                    self._video_manifest.write(json.dumps({"event": "camera_header", "width": 640, "height": 480,
                                                          "fps_numerator": 500, "fps_denominator": 33,
                                                          "scope": "actual env0 pre-reset post-physics external camera; no policy image input",
                                                          "eye_world_m": self._video_status.get("eye_world_m"),
                                                          "target_world_m": self._video_status.get("target_world_m"),
                                                          **({"control_probe": dict(self._control_probe_profile)} if self._control_probe_enabled else {})}) + "\n")
            if self._control_probe_enabled:
                self.configure_control_probe(self._control_probe_profile)

        def _setup_scene(self):
            self.robot = Articulation(ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot", spawn=robot_spawn,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(initial["root_pose_xyzw"][:3]), rot=tuple(initial["root_pose_xyzw"][3:]),
                    joint_pos=initial["joint_positions_rad"], joint_vel={".*": 0.0}),
                actuators={"shared_pd_without_mujoco_bias_v1": ImplicitActuatorCfg(
                    joint_names_expr=[".*"], stiffness={name: value["kp"] for name, value in actuator_specs.items()},
                    damping={name: value["kd"] for name, value in actuator_specs.items()},
                    effort_limit_sim={name: value["effort_limit_nm"] for name, value in actuator_specs.items()},
                    armature={name: value["armature"] for name, value in actuator_specs.items()},
                    viscous_friction={name: value["dof_damping"] for name, value in actuator_specs.items()})}))
            self.cube = RigidObject(RigidObjectCfg(
                prim_path="/World/envs/env_.*/Cube",
                spawn=sim_utils.CuboidCfg(size=(0.05, 0.05, 0.05), rigid_props=rigid,
                                          mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
                                          collision_props=collision, activate_contact_sensors=True,
                                          physics_material=material(settings["cube_friction"]),
                                          visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1))),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(initial["cube_pose_xyzw"][:3]), rot=tuple(initial["cube_pose_xyzw"][3:]))))
            table = sim_utils.CuboidCfg(size=TABLE_BODY_PROTOCOL["size_m"], collision_props=collision,
                                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                                        activate_contact_sensors=True,
                                        physics_material=material(settings["table_friction"]))
            self.table = RigidObject(RigidObjectCfg(prim_path="/World/envs/env_.*/Table", spawn=table,
                                                   init_state=RigidObjectCfg.InitialStateCfg(pos=TABLE_BODY_PROTOCOL["position_m"])))
            self.scene.articulations["robot"] = self.robot
            self.scene.rigid_objects["cube"] = self.cube
            self.scene.rigid_objects["table"] = self.table
            stage = get_current_stage()
            table_prim = stage.GetPrimAtPath("/World/envs/env_0/Table")
            self._table_schema_audit = {"rigid_body_api": table_prim.HasAPI(UsdPhysics.RigidBodyAPI),
                                        "kinematic_enabled": bool(table_prim.GetAttribute("physics:kinematicEnabled").Get()),
                                        "disable_gravity": bool(table_prim.GetAttribute("physxRigidBody:disableGravity").Get())}
            if not all(self._table_schema_audit.values()):
                raise RLEnvironmentError("Table must be a gravity-disabled native kinematic rigid body")
            fixed_joints = []
            self._joint_schema_audit = {}
            task_names = {"floor", "calibration_table", "green_cube", "red_triangle", "yellow_rod", "warehouse_tote"}
            for prim in stage.Traverse():
                if not str(prim.GetPath()).startswith("/World/envs/env_0/Robot/"):
                    continue
                if prim.GetName() in task_names:
                    raise RLEnvironmentError("robot-only asset contains an old task entity: " + str(prim.GetPath()))
                if prim.IsA(UsdPhysics.FixedJoint):
                    joint = UsdPhysics.FixedJoint(prim)
                    fixed_joints.append({"path": str(prim.GetPath()).replace("/env_0/", "/env_*/"),
                                         "body0": [str(path).replace("/env_0/", "/env_*/") for path in joint.GetBody0Rel().GetTargets()],
                                         "body1": [str(path).replace("/env_0/", "/env_*/") for path in joint.GetBody1Rel().GetTargets()],
                                         "enabled": bool(joint.GetJointEnabledAttr().Get())})
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    joint = UsdPhysics.RevoluteJoint(prim)
                    def quat_values(value):
                        return [float(v) for v in value.GetImaginary()] + [float(value.GetReal())]
                    item = {"axis": str(joint.GetAxisAttr().Get()),
                            "lower_limit_degrees": float(joint.GetLowerLimitAttr().Get()),
                            "upper_limit_degrees": float(joint.GetUpperLimitAttr().Get()),
                            "local_position0_m": [float(v) for v in joint.GetLocalPos0Attr().Get()],
                            "local_position1_m": [float(v) for v in joint.GetLocalPos1Attr().Get()],
                            "local_quaternion0_xyzw": quat_values(joint.GetLocalRot0Attr().Get()),
                            "local_quaternion1_xyzw": quat_values(joint.GetLocalRot1Attr().Get()),
                            "body0": [str(path).replace("/env_0/", "/env_*/") for path in joint.GetBody0Rel().GetTargets()],
                            "body1": [str(path).replace("/env_0/", "/env_*/") for path in joint.GetBody1Rel().GetTargets()]}
                    if item["axis"] not in ("X", "Y", "Z") or not all(math.isfinite(item[key]) for key in ("lower_limit_degrees", "upper_limit_degrees")):
                        raise RLEnvironmentError("invalid official joint axis or degree-limit schema: " + prim.GetName())
                    if item["lower_limit_degrees"] >= item["upper_limit_degrees"]:
                        raise RLEnvironmentError("invalid official mechanical limits: " + prim.GetName())
                    for key in ("local_quaternion0_xyzw", "local_quaternion1_xyzw"):
                        q = item[key]
                        wxyz_to_xyzw((q[3], q[0], q[1], q[2]))
                    if prim.GetName() in self._joint_schema_audit:
                        raise RLEnvironmentError("duplicate official revolute joint identity: " + prim.GetName())
                    self._joint_schema_audit[prim.GetName()] = item
            self._fixed_root_audit = audit_fixed_root_joints(fixed_joints)
            if set(self._joint_schema_audit) != set(initial["joint_positions_rad"]):
                raise RLEnvironmentError("official USD revolute-joint schema differs from the exact 43 source names")
            hand_material = material(settings["hand_friction"])
            hand_material.func("/World/RLHandContactMaterial", hand_material)
            palm_material = material(settings["palm_friction"])
            palm_material.func("/World/RLPalmContactMaterial", palm_material)
            self._material_bindings = []
            self._instance_overrides = []
            for prim in list(stage.Traverse()):
                if not str(prim.GetPath()).startswith("/World/envs/env_0/Robot/") or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                ancestor = prim
                while ancestor.IsValid() and not (ancestor.GetName() in sensor_body_names and ancestor.HasAPI(UsdPhysics.RigidBodyAPI)):
                    ancestor = ancestor.GetParent()
                if ancestor.IsValid():
                    collider_path = str(prim.GetPath())
                    instance = prim
                    while instance.IsValid() and not instance.IsInstance():
                        instance = instance.GetParent()
                    if instance.IsValid():
                        instance_path = str(instance.GetPath())
                        instance.SetInstanceable(False)
                        self._instance_overrides.append(instance_path.replace("/env_0/", "/env_*/"))
                        prim = stage.GetPrimAtPath(collider_path)
                    material_path = "/World/RLPalmContactMaterial" if ancestor.GetName() == palm_contact_body else "/World/RLHandContactMaterial"
                    bind_physics_material(str(prim.GetPath()), material_path, stage=stage)
                    sim_utils.modify_collision_properties(str(prim.GetPath()), collision, stage=stage)
                    bound = [str(path) for path in prim.GetRelationship("material:binding:physics").GetTargets()]
                    if bound != [material_path]:
                        raise RLEnvironmentError("explicit physics material relationship could not be authored: " + collider_path)
                    self._material_bindings.append({"collider": str(prim.GetPath()).replace("/env_0/", "/env_*/"),
                                                    "body": ancestor.GetName(), "material": material_path})
            if {item["body"] for item in self._material_bindings} != set(sensor_body_names):
                raise RLEnvironmentError("explicit physics material binding failed for one or more original right-hand bodies")
            path_entries = []
            for prim in stage.Traverse():
                text = str(prim.GetPath())
                if text.startswith("/World/envs/env_0/Robot/") and prim.GetName() in sensor_body_names:
                    path_entries.append((prim.GetName(), text.replace("/env_0/", "/env_.*/"),
                                         prim.HasAPI(UsdPhysics.RigidBodyAPI)))
            paths = sensor_body_path_map(path_entries, sensor_body_names)
            all_body_entries = []
            for prim in stage.Traverse():
                text = str(prim.GetPath())
                if text.startswith("/World/envs/env_0/Robot/") and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    all_body_entries.append((prim.GetName(), text.replace("/env_0/", "/env_.*/"), True))
            all_body_paths = sensor_body_path_map(all_body_entries, [entry[0] for entry in all_body_entries])
            self._cube_filter_body_names, self._cube_other_filter_indices = cube_contact_partner_layout(all_body_paths, sensor_body_names)
            self._hand_sensors = []
            for body in sensor_body_names if self._contact_crosscheck["enabled"] else ():
                sensor = ContactSensor(ContactSensorCfg(
                    prim_path=paths[body], update_period=0.0,
                    filter_prim_paths_expr=["/World/envs/env_.*/Cube"],
                    track_friction_forces=True, track_contact_points=True,
                    max_contact_data_count_per_prim=32))
                self.scene.sensors[body] = sensor
                self._hand_sensors.append(sensor)
            self._cube_sensor = ContactSensor(ContactSensorCfg(
                prim_path="/World/envs/env_.*/Cube", update_period=0.0,
                filter_prim_paths_expr=["/World/envs/env_.*/Table"] +
                    [all_body_paths[name] for name in self._cube_filter_body_names[1:]],
                track_friction_forces=True, max_contact_data_count_per_prim=256))
            self.scene.sensors["cube_table"] = self._cube_sensor
            if self._standalone_video:
                from isaaclab.sensors import Camera, CameraCfg
                from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import IsaacRtxRendererCfg
                self._evaluation_camera = Camera(CameraCfg(
                    prim_path="/World/RLEvaluationCamera", update_period=0.066,
                    width=640, height=480, data_types=["rgb"],
                    renderer_cfg=IsaacRtxRendererCfg(),
                    spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=32.0,
                                                     clipping_range=(0.01, 20.0))))
                # One global camera is updated explicitly. Registering it in
                # the N-env scene would pass partial reset IDs > 0 to a camera
                # with only one instance and corrupt its reset bookkeeping.
                light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
                light.func("/World/RLEvaluationLight", light)
            self.scene.clone_environments(copy_from_source=False)
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=[])

        def _pre_physics_step(self, actions):
            if tuple(actions.shape) != (self.num_envs, 14) or not bool(torch.isfinite(actions).all()):
                raise RLEnvironmentError("nonfinite or incorrectly shaped policy action")
            if self._control_probe_enabled and self._control_transition_snapshot is not None:
                raise RLEnvironmentError("previous diagnostic transition must be recorded before the next command")
            self._previous_actions[:] = self._actions
            self._action_origin, self._next_action_origin = self._next_action_origin, "policy_or_zero_target_hold"
            self._actions[:] = actions.clamp(-1.0, 1.0)
            per_step = torch.minimum(self._velocity * self.step_dt,
                                     torch.full_like(self._velocity, float(config["rl_teacher"]["action_scale_rad"])))
            previous = self._targets[:, self._action_ids]
            desired = previous + self._actions * per_step
            # The exact applied command still obeys the source anchor delta.
            current = tensor(self.robot.data.joint_pos)[:, self._action_ids]
            anchor = float(config["control"]["max_anchor_delta_rad"])
            low = torch.maximum(torch.maximum(previous-per_step, current-anchor), self._limits[:, self._action_ids, 0])
            high = torch.minimum(torch.minimum(previous+per_step, current+anchor), self._limits[:, self._action_ids, 1])
            infeasible = (low > high).any(dim=-1)
            self._unsafe |= infeasible
            desired = torch.maximum(torch.minimum(desired, high), low)
            desired[infeasible] = previous[infeasible]
            # Conservative first-order task-point shield; true 2 ms overshoot
            # remains a separate terminating measurement, never a claimed proof
            # that the physical cube cannot overshoot.
            body_rot = tensor(self.robot.data.body_quat_w)[:, self._palm_id]
            offset_w = quat_apply(body_rot, self._palm_local)
            palm_z = tensor(self.robot.data.body_pos_w)[:, self._palm_id, 2] + offset_w[:, 2]
            self._start_palm_z = torch.where(torch.isnan(self._start_palm_z), palm_z, self._start_palm_z)
            jacobian = tensor(self.robot.data.body_link_jacobian_w)
            if jacobian.shape[1] != self.robot.num_bodies - 1 or jacobian.shape[-1] != 43:
                raise RLEnvironmentError("probe shield requires the verified fixed-root 43-joint Jacobian")
            palm_jac = jacobian[:, self._palm_id-1][:, :, self._action_ids[:7]]
            point_jz = palm_jac[:, 2] + torch.cross(
                palm_jac[:, 3:].transpose(1, 2), offset_w[:, None, :].expand(-1, 7, -1), dim=-1)[:, :, 2]
            current_arm = current[:, :7]
            previous_arm = previous[:, :7]
            increment = desired[:, :7] - previous_arm
            previous_prediction = palm_z + torch.sum(point_jz*(previous_arm-current_arm), dim=-1)
            predicted_increment = torch.sum(point_jz*increment, dim=-1)
            allowance = (self._start_palm_z + settings["probe_clearance_max_m"] - previous_prediction).clamp_min(0)
            factor = torch.where(predicted_increment > allowance,
                                 allowance/predicted_increment.clamp_min(1e-12), torch.ones_like(allowance)).clamp(0, 1)
            if not self._control_probe_enabled:
                # Preserve the frozen training path expression for expression.
                self._shield_count += int((factor < 1).sum().item())
                desired[:, :7] = previous_arm + increment*factor[:, None]
            else:
                from .rl_probe_control import apply_probe_guard, LEGACY_JOINT_ROUNDOFF_TOLERANCE_RAD
                candidate = desired.clone()
                version = self._control_probe_profile["control_guard_version"]
                guarded, evidence = apply_probe_guard(
                    torch, desired[:, :7], previous_arm, current_arm, low[:, :7], high[:, :7],
                    point_jz, palm_z, self._start_palm_z, settings["probe_clearance_max_m"], mode=version)
                desired[:, :7] = guarded
                bound_tolerance = LEGACY_JOINT_ROUNDOFF_TOLERANCE_RAD if version == "legacy_scalar_v1" else 0.0
                lower_violation = (low-desired).clamp_min(0)
                upper_violation = (desired-high).clamp_min(0)
                bounds_ok = ((lower_violation <= bound_tolerance) & (upper_violation <= bound_tolerance)).all(-1)
                guard_unsafe = evidence["requires_unsafe"] | ~bounds_ok
                self._unsafe |= guard_unsafe
                corrected = (evidence["target_correction"] != 0).any(-1)
                self._control_guard_correction_count += int(corrected.sum().item())
                self._control_guard_unsafe_count += int(guard_unsafe.sum().item())
                self._shield_count += int((evidence["factor"] < 1).sum().item()) if version == "legacy_scalar_v1" else int(corrected.sum().item())
                self._control_command_id += 1
                self._write_control_probe({
                    "event": "control_command", "command_id": self._control_command_id,
                    "native_physics_step_count": int(self.sim.get_physics_step_count()),
                    "episode_id": self._episode_ids, "episode_control_steps": self.episode_length_buf,
                    "raw_action": actions, "clipped_action": self._actions,
                    "previous_target_action_rad": previous, "measured_q_action_rad": current,
                    "measured_qd_action_rad_s": tensor(self.robot.data.joint_vel)[:, self._action_ids],
                    "per_step_action_rad": per_step, "lower_action_bound_rad": low, "upper_action_bound_rad": high,
                    "unbounded_candidate_action_rad": previous+self._actions*per_step,
                    "limited_candidate_action_rad": candidate, "post_guard_target_action_rad": desired,
                    "point_jz": point_jz, "palm_position_world_m": tensor(self.robot.data.body_pos_w)[:, self._palm_id]+offset_w,
                    "palm_point_jacobian_w": offset_point_jacobian(torch, palm_jac, offset_w),
                    "start_palm_z_m": self._start_palm_z, "guard_evidence": evidence,
                    "pre_guard_infeasible": infeasible, "post_lower_bound_violation_rad": lower_violation,
                    "post_upper_bound_violation_rad": upper_violation, "post_bound_tolerance_rad": bound_tolerance,
                    "post_bounds_passed": bounds_ok, "requires_unsafe": guard_unsafe,
                    "legacy_diagnostic_difference": "post-guard box/height violations are explicit unsafe; roundoff-only legacy residuals remain logged"})
            self._targets[:, self._action_ids] = desired

        def _apply_action(self):
            if self._pending_physics_sample:
                self._sample_physics()
            self.robot.set_joint_position_target_index(target=self._targets)
            self._pending_physics_sample = True

        def _contact_values(self, sensor, object_up_world, filter_index=0):
            data = sensor.data
            normal = tensor(data.force_matrix_w).reshape(self.num_envs, -1, 3)[:, filter_index]
            friction = tensor(data.friction_forces_w).reshape(self.num_envs, -1, 3)[:, filter_index]
            friction_force, friction_points, friction_counts, friction_starts = sensor.contact_view.get_friction_data(dt=self.physics_dt)
            _, friction_raw = _reduce_friction_pairs(
                torch, tensor(friction_force).reshape(-1, 3), tensor(friction_points).reshape(-1, 3),
                tensor(friction_counts).reshape(self.num_envs, -1), tensor(friction_starts).reshape(self.num_envs, -1))
            force, points, normals, separation, counts, starts = sensor.contact_view.get_contact_data(dt=self.physics_dt)
            force, normals, separation = tensor(force).flatten(), tensor(normals).reshape(-1, 3), tensor(separation).flatten()
            counts, starts = tensor(counts).long().reshape(self.num_envs, -1), tensor(starts).long().reshape(self.num_envs, -1)
            if counts.shape[1] != len(sensor.cfg.filter_prim_paths_expr):
                raise RLEnvironmentError("native filter count differs from declared contact-partner expressions")
            if filter_index >= counts.shape[1]:
                raise RLEnvironmentError("contact filter did not resolve the declared collider")
            counts, starts = counts[:, filter_index], starts[:, filter_index]
            if bool((counts < 0).any()) or bool((starts[counts > 0] + counts[counts > 0] > len(force)).any()):
                raise RLEnvironmentError("contact stream overflow or invalid contact indices")
            if bool((counts > 32).any()):
                raise RLEnvironmentError("raw contact count exceeds the declared per-body capacity")
            owners = torch.repeat_interleave(torch.arange(self.num_envs, device=self.device), counts)
            repeated_offsets = torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
            indices = torch.repeat_interleave(starts, counts) + torch.arange(len(owners), device=self.device) - repeated_offsets
            arrays = (normal, friction, force[indices], normals[indices], separation[indices],
                      tensor(points).reshape(-1, 3)[indices])
            if not all(bool(torch.isfinite(value).all()) for value in arrays):
                # Numerical corruption is a hard run failure, never an optional
                # API capability that training can continue without.
                raise FloatingPointError("nonfinite native raw contact force/normal/separation/position")
            fn = torch.zeros(self.num_envs, device=self.device)
            depth = torch.zeros_like(fn)
            vertical = torch.zeros_like(fn)
            loaded_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            padded_normal = torch.zeros(self.num_envs, 32, 3, device=self.device)
            padded_point = torch.zeros_like(padded_normal)
            padded_separation = torch.zeros(self.num_envs, 32, device=self.device)
            padded_force = torch.zeros_like(padded_separation)
            padded_valid = torch.zeros(self.num_envs, 32, dtype=torch.bool, device=self.device)
            padded_loaded = torch.zeros(self.num_envs, 32, dtype=torch.bool, device=self.device)
            positive_gap = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            positive_gap_force = torch.zeros_like(positive_gap)
            if len(indices):
                fn.scatter_add_(0, owners, torch.abs(force[indices]))
                depth.scatter_reduce_(0, owners, torch.clamp(-separation[indices], min=0), reduce="amax", include_self=True)
                loaded = (torch.abs(force[indices]) > settings["group_force_min_n"]) & (separation[indices] <= 0)
                # Physical external loading uses the source active-contact
                # definition, separate from the 20 mN finger-side threshold.
                physically_loaded = (torch.abs(force[indices]) > 0) & (separation[indices] <= 0)
                side_up = torch.maximum(normals[indices, 2].abs(),
                                        (normals[indices]*object_up_world[owners]).sum(-1).abs())
                vertical.scatter_reduce_(0, owners, torch.where(physically_loaded, side_up, 0.0),
                                          reduce="amax", include_self=True)
                loaded_count.scatter_add_(0, owners, physically_loaded.long())
                positive_gap.scatter_add_(0, owners, (separation[indices] > 0).long())
                positive_gap_force.scatter_add_(0, owners, ((force[indices].abs() > 0) & (separation[indices] > 0)).long())
                local_ids = torch.arange(len(owners), device=self.device)-repeated_offsets
                padded_normal[owners, local_ids] = normals[indices]
                padded_point[owners, local_ids] = tensor(points).reshape(-1, 3)[indices]
                padded_separation[owners, local_ids] = separation[indices]
                padded_force[owners, local_ids] = force[indices]
                padded_valid[owners, local_ids] = True
                padded_loaded[owners, local_ids] = loaded
            if self._record:
                selected = owners < min(8, self.num_envs)
                selected_indices = indices[selected]
                label = sensor.cfg.prim_path.rsplit("/", 1)[-1] + ":filter_" + str(filter_index)
                self._raw_contact_tensors[label] = {
                    "env_id": owners[selected], "normal_force_n": force[selected_indices],
                    "separation_m": separation[selected_indices], "normal_w": normals[selected_indices],
                    "position_w_m": tensor(points).reshape(-1, 3)[selected_indices],
                    "normal_force_on_sensor_w_n": normal[:min(8, self.num_envs)],
                    "friction_force_on_sensor_w_n": friction[:min(8, self.num_envs)],
                    "physical_loaded_count": loaded_count[:min(8, self.num_envs)],
                }
            return {"normal": normal, "friction": friction, "fn": fn, "depth": depth,
                    "vertical": vertical, "physical_loaded_count": loaded_count,
                    "point_normal": padded_normal, "point_loaded": padded_loaded,
                    "point": padded_point, "separation": padded_separation,
                    "point_force": padded_force, "point_valid": padded_valid,
                    "raw_point_count": counts, "positive_gap_point_count": positive_gap,
                    "positive_gap_force_count": positive_gap_force,
                    "friction_nonzero_patch_count": friction_raw["nonzero_patch_count"][:, filter_index]}

        def _cube_contact_values(self, object_up_world):
            """Read one native view; never trigger ContactSensor.data updates."""
            sensor = self._cube_sensor
            view = sensor.contact_view
            width = len(self._cube_filter_body_names)
            normal = tensor(view.get_contact_force_matrix(dt=self.physics_dt)).reshape(self.num_envs, width, 3)
            cube_normal = tensor(view.get_net_contact_forces(dt=self.physics_dt)).reshape(self.num_envs, 3)
            friction_force, friction_points, friction_counts, friction_starts = view.get_friction_data(dt=self.physics_dt)
            friction, friction_raw = _reduce_friction_pairs(
                torch, tensor(friction_force).reshape(-1, 3), tensor(friction_points).reshape(-1, 3),
                tensor(friction_counts).reshape(self.num_envs, width), tensor(friction_starts).reshape(self.num_envs, width))
            # The preceding reducer copied/reduced the shared counts/starts.
            # This final getter may overwrite the SDK's shared index buffers.
            force, points, normals, separation, counts, starts = view.get_contact_data(dt=self.physics_dt)
            force, points = tensor(force).flatten(), tensor(points).reshape(-1, 3)
            normals, separation = tensor(normals).reshape(-1, 3), tensor(separation).flatten()
            counts, starts = tensor(counts).reshape(self.num_envs, -1), tensor(starts).reshape(self.num_envs, -1)
            if counts.shape[1] != width:
                raise RLEnvironmentError("cube partner columns differ from explicit rigid-body filters")
            result = _reduce_contact_pairs(torch, normal=normal, friction=friction, net_normal=cube_normal,
                                          force=force, points=points, normals=normals, separation=separation,
                                          counts=counts, starts=starts, hand_indices=self._cube_hand_index_tensor,
                                          hand_inverse=self._cube_hand_inverse,
                                          object_up_world=object_up_world,
                                          friction_nonzero_patch_count=friction_raw["nonzero_patch_count"],
                                          force_threshold=settings["group_force_min_n"])
            fn, depth, loaded, raw = result["fn"], result["depth"], result["physical_loaded_count"], result["raw"]
            if self._record:
                owners = torch.div(raw["bucket"], width, rounding_mode="floor")
                selected = owners < min(8, self.num_envs)
                self._raw_contact_tensors["Cube:all_explicit_partners"] = {
                    "env_id": owners[selected], "partner_filter_index": raw["bucket"][selected] % width,
                    "normal_force_n": raw["force"][selected], "separation_m": raw["separation"][selected],
                    "normal_w": raw["normal"][selected], "position_w_m": raw["point"][selected],
                    "normal_force_on_sensor_by_partner_w_n": normal[:min(8, self.num_envs)],
                    "friction_force_on_sensor_by_partner_w_n": friction[:min(8, self.num_envs)],
                    "physical_loaded_count_by_partner": loaded[:min(8, self.num_envs)]}
                friction_owner = torch.div(friction_raw["bucket"], width, rounding_mode="floor")
                selected = friction_owner < min(8, self.num_envs)
                self._raw_contact_tensors["Cube:friction_explicit_partners"] = {
                    "env_id": friction_owner[selected], "partner_filter_index": friction_raw["bucket"][selected] % width,
                    "friction_force_on_cube_w_n": friction_raw["force"][selected],
                    "friction_point_world_m": friction_raw["point"][selected]}
                for index, name in enumerate(sensor_body_names):
                    hand = result["hand"]
                    env_ids, point_ids = torch.where(hand["point_valid"][:min(8, self.num_envs), index])
                    self._raw_contact_tensors[name+":derived_from_cube"] = {
                        "env_id": env_ids, "normal_force_n": hand["point_force"][env_ids, index, point_ids],
                        "separation_m": hand["separation"][env_ids, index, point_ids],
                        "normal_w": hand["point_normal"][env_ids, index, point_ids],
                        "position_w_m": hand["point"][env_ids, index, point_ids],
                        "normal_force_on_sensor_w_n": hand["normal"][:min(8, self.num_envs), index],
                        "friction_force_on_sensor_w_n": hand["friction"][:min(8, self.num_envs), index],
                        "physical_loaded_count": hand["physical_loaded_count"][:min(8, self.num_envs), index]}
            other = self._cube_other_filter_indices
            return {"table_normal": normal[:, 0], "table_fn": fn[:, 0], "table_loaded_count": loaded[:, 0],
                    "robot_normal": normal[:, 1:].sum(dim=1), "robot_fn": fn[:, 1:].sum(dim=1),
                    "maximum_penetration_m": depth.amax(dim=1),
                    "other_fn": fn[:, other].sum(dim=1), "other_loaded_count": loaded[:, other].sum(dim=1),
                    "unfiltered_normal_residual_n": result["unfiltered_normal_residual_n"],
                    "hand": result["hand"]}

        def _crosscheck_hand_views(self, hand, object_up_world):
            audit = self._contact_crosscheck
            if not audit["enabled"] or audit["completed_physics_samples"] >= audit["required_physics_samples"]:
                return None
            has_contact = bool((hand["fn"] > 0).any())
            if not audit["started"] and not has_contact:
                return None
            audit["started"] = True
            references = [self._contact_values(sensor, object_up_world) for sensor in self._hand_sensors]
            reference = {name: torch.stack([item[name] for item in references], dim=1) for name in hand}
            self._crosscheck_current_raw = (hand, reference)
            comparison = _compare_hand_contacts(torch, hand, reference)
            audit["completed_physics_samples"] += 1
            if has_contact:
                audit["completed_nonzero_samples"] += 1
            for name, value in comparison["maximum_absolute_differences"].items():
                audit["maximum_absolute_differences"][name] = max(value, audit["maximum_absolute_differences"].get(name, 0))
            audit["status"] = ("passed" if comparison["passed"] and
                               audit["completed_physics_samples"] >= audit["required_physics_samples"] and
                               audit["completed_nonzero_samples"] >= CONTACT_CROSSCHECK["minimum_nonzero_contact_samples"] else "checking")
            row = {"event": "same_physics_step_contact_crosscheck", "physics_sample": self._physics_samples+1,
                   "native_physics_step_count": int(self.sim.get_physics_step_count()),
                   "has_nonzero_hand_contact": has_contact, "hand_names": list(sensor_body_names),
                   "comparison": comparison, "tolerances": CONTACT_CROSSCHECK,
                   "completed_physics_samples": audit["completed_physics_samples"],
                   "completed_nonzero_samples": audit["completed_nonzero_samples"]}
            if not comparison["passed"]:
                row["cube_derived_hand"] = {name: value.detach().cpu().tolist() for name, value in hand.items()}
                row["independent_hand_sensor"] = {name: value.detach().cpu().tolist() for name, value in reference.items()}
                audit["status"], audit["first_failure"] = "failed", row
            if output:
                if self._crosscheck_record is None:
                    self._crosscheck_record = (output / "contact_collection_crosscheck.jsonl").open("x", encoding="utf-8")
                self._crosscheck_record.write(json.dumps(_json_finite(row), allow_nan=False)+"\n")
                self._crosscheck_record.flush()
            if not comparison["passed"]:
                raise ContactEquivalenceError("Cube-derived and independent hand contact views disagree")
            return _summarize_hand_contacts(torch, reference)

        def _check_crosscheck_gates(self, actual, reference):
            if reference is None:
                return
            different = {name: bool((actual[name] != reference[name]).any()) for name in actual}
            if any(different.values()):
                row = {"event": "contact_gate_crosscheck_failure", "physics_sample": self._physics_samples+1,
                       "native_physics_step_count": int(self.sim.get_physics_step_count()),
                       "different_gates": different,
                       "actual": {name: value.detach().cpu().tolist() for name, value in actual.items()},
                       "reference": {name: value.detach().cpu().tolist() for name, value in reference.items()}}
                for label, values in zip(("cube_derived_hand", "independent_hand_sensor"), self._crosscheck_current_raw):
                    row[label] = {name: value.detach().cpu().tolist() for name, value in values.items()}
                self._contact_crosscheck["status"], self._contact_crosscheck["first_failure"] = "failed", row
                if self._crosscheck_record:
                    self._crosscheck_record.write(json.dumps(row)+"\n")
                    self._crosscheck_record.flush()
                raise ContactEquivalenceError("same-step contact gate Boolean mismatch")

        def _physical_state(self):
            cube_position = tensor(self.cube.data.root_pos_w)
            cube_rotation = tensor(self.cube.data.root_quat_w)
            cube_velocity = tensor(self.cube.data.root_lin_vel_w)
            cube_angular = tensor(self.cube.data.root_ang_vel_w)
            body_rotation = tensor(self.robot.data.body_quat_w)[:, self._palm_id]
            palm_position = tensor(self.robot.data.body_pos_w)[:, self._palm_id] + quat_apply(body_rotation, self._palm_local)
            palm_rotation = quat_mul(body_rotation, self._palm_local_rotation)
            relative_position = quat_apply(quat_conjugate(palm_rotation), cube_position - palm_position)
            relative_rotation = quat_mul(quat_conjugate(palm_rotation), cube_rotation)
            x, y, z, w = cube_rotation.unbind(-1)
            extent = 0.025 * (torch.abs(2*(x*z-w*y)) + torch.abs(2*(y*z+w*x)) + torch.abs(1-2*(x*x+y*y)))
            clearance = cube_position[:, 2] - extent - 0.75
            return cube_position, cube_rotation, cube_velocity, cube_angular, relative_position, relative_rotation, clearance

        def _sample_physics(self):
            if self._record:
                self._raw_contact_tensors = {}
            state = self._physical_state()
            position, rotation, velocity, angular, relative, relative_rotation, clearance = state
            finite = torch.isfinite(torch.cat((tensor(self.robot.data.joint_pos), tensor(self.robot.data.joint_vel),
                                               position, rotation, velocity, angular), dim=-1)).all(dim=-1)
            joint_tolerance = float(config["eval"]["joint_limit_tolerance_rad"])
            joint_safe = ((tensor(self.robot.data.joint_pos) >= self._limits[..., 0] - joint_tolerance) &
                          (tensor(self.robot.data.joint_pos) <= self._limits[..., 1] + joint_tolerance)).all(dim=-1)
            total_sum = torch.zeros(self.num_envs, 3, device=self.device)
            groups = torch.zeros(self.num_envs, 3, device=self.device)
            group_physical = torch.zeros(self.num_envs, 3, dtype=torch.int64, device=self.device)
            depth = torch.zeros(self.num_envs, device=self.device)
            normal_vertical = torch.zeros_like(depth)
            opposition = torch.ones(self.num_envs, device=self.device)
            no_positive_gap_load = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            friction_manifold = torch.zeros_like(no_positive_gap_load)
            reference_summary = None
            contact_available = self._force_unavailable is None
            try:
                x, y, z, w = rotation.unbind(-1)
                object_up_world = torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), dim=-1)
                cube_contacts = self._cube_contact_values(object_up_world)
                summary = _summarize_hand_contacts(torch, cube_contacts["hand"])
                reference_summary = self._crosscheck_hand_views(cube_contacts["hand"], object_up_world)
                total_sum, groups = summary["total_sum"], summary["groups"]
                group_physical = summary["group_physical_loaded_count"]
                no_positive_gap_load = summary["no_positive_gap_hand_load"]
                friction_manifold = summary["friction_manifold_certified"]
                depth, normal_vertical = summary["depth"], summary["normal_vertical"]
                opposition = summary["opposition"]
                table_fn, table_loaded_count = cube_contacts["table_fn"], cube_contacts["table_loaded_count"]
                table_positive = table_fn > 0
                self._table_positive_per_env += table_positive.long()
                table_stats = torch.stack((table_positive.sum().to(table_fn.dtype), table_fn.amax(),
                                           self._table_positive_per_env.amin().to(table_fn.dtype))).detach().cpu().tolist()
                self._table_contact_channel.observe(int(table_stats[0]), table_stats[1], int(table_stats[2]))
                depth = torch.maximum(depth, cube_contacts["maximum_penetration_m"])
                other_force = cube_contacts["other_fn"] + cube_contacts["unfiltered_normal_residual_n"]
                other_loaded_count = cube_contacts["other_loaded_count"]
                table_supported = self._table_contact_channel.snapshot()["supported"]
                self._qualification_supported = self._force_unavailable is None and table_supported
                contact_available &= table_supported
            except (AttributeError, TypeError, RuntimeError, RLEnvironmentError) as exc:
                self._force_unavailable = type(exc).__name__ + ": " + str(exc)
                self._qualification_supported = False
                self._table_contact_channel.unavailable(self._force_unavailable)
                contact_available = False
                table_fn = torch.full_like(depth, float("inf"))
                other_force = torch.full_like(depth, float("inf"))
                table_loaded_count = torch.ones(self.num_envs, dtype=torch.int64, device=self.device)
                other_loaded_count = torch.ones_like(table_loaded_count)
            loaded = (groups > settings["group_force_min_n"]).all(dim=-1) & (group_physical > 0).all(dim=-1)
            loaded &= no_positive_gap_load & friction_manifold
            new_lock = loaded & ~self._grip_locked
            self._grip_reference[new_lock] = relative[new_lock]
            self._grip_rotation[new_lock] = relative_rotation[new_lock]
            self._grip_locked |= new_lock
            drift = torch.linalg.vector_norm(relative - self._grip_reference, dim=-1)
            cosine = torch.abs(torch.sum(relative_rotation * self._grip_rotation, dim=-1)).clamp(0.0, 1.0)
            rotation_error = 2 * torch.acos(cosine)
            weight = 0.08 * 9.81
            upward = total_sum[:, 2]
            speed = torch.linalg.vector_norm(velocity, dim=-1)
            spin = torch.linalg.vector_norm(angular, dim=-1)
            contact_safe = loaded & (groups <= settings["group_force_max_n"]).all(dim=-1)
            contact_safe &= (depth <= settings["max_penetration_m"]) & (normal_vertical <= settings["max_abs_normal_z"])
            contact_safe &= opposition <= settings["opposition_dot_max"]
            support = finite & joint_safe & contact_safe & bool(contact_available)
            support &= (table_fn <= 0.05*weight) & (other_force <= 0.02)
            support &= (table_loaded_count == 0) & (other_loaded_count == 0)
            support &= (upward >= 0.8*weight) & (upward <= 1.2*weight)
            support &= (clearance >= settings["support_clearance_min_m"])
            support &= (speed <= settings["max_linear_speed_m_s"]) & (spin <= settings["max_angular_speed_rad_s"])
            support &= self._grip_locked & (drift <= settings["grip_drift_m"])
            if reference_summary is not None:
                ref_groups = reference_summary["groups"]
                ref_depth = torch.maximum(reference_summary["depth"], cube_contacts["maximum_penetration_m"])
                ref_upward = reference_summary["total_sum"][:, 2]
                ref_physical = reference_summary["group_physical_loaded_count"]
                ref_loaded = (ref_groups > settings["group_force_min_n"]).all(dim=-1) & (ref_physical > 0).all(dim=-1)
                ref_loaded &= reference_summary["no_positive_gap_hand_load"] & reference_summary["friction_manifold_certified"]
                ref_safe = ref_loaded & (ref_groups <= settings["group_force_max_n"]).all(dim=-1)
                ref_safe &= (ref_depth <= settings["max_penetration_m"])
                ref_safe &= reference_summary["normal_vertical"] <= settings["max_abs_normal_z"]
                ref_safe &= reference_summary["opposition"] <= settings["opposition_dot_max"]
                ref_support = finite & joint_safe & ref_safe & bool(contact_available)
                ref_support &= (table_fn <= 0.05*weight) & (other_force <= 0.02)
                ref_support &= (table_loaded_count == 0) & (other_loaded_count == 0)
                ref_support &= (ref_upward >= 0.8*weight) & (ref_upward <= 1.2*weight)
                ref_support &= clearance >= settings["support_clearance_min_m"]
                ref_support &= (speed <= settings["max_linear_speed_m_s"]) & (spin <= settings["max_angular_speed_rad_s"])
                ref_support &= self._grip_locked & (drift <= settings["grip_drift_m"])
                self._check_crosscheck_gates(
                    {"group_loaded": groups > settings["group_force_min_n"],
                     "no_positive_gap_hand_load": no_positive_gap_load, "friction_manifold_certified": friction_manifold,
                     "group_physically_loaded": group_physical > 0,
                     "group_force_safe": groups <= settings["group_force_max_n"],
                     "penetration_safe": depth <= settings["max_penetration_m"],
                     "side_normals_safe": normal_vertical <= settings["max_abs_normal_z"],
                     "opposition_safe": opposition <= settings["opposition_dot_max"],
                     "upward_load_min": upward >= 0.8*weight, "upward_load_max": upward <= 1.2*weight,
                     "loaded": loaded, "contact_safe": contact_safe, "support": support},
                    {"group_loaded": ref_groups > settings["group_force_min_n"],
                     "no_positive_gap_hand_load": reference_summary["no_positive_gap_hand_load"],
                     "friction_manifold_certified": reference_summary["friction_manifold_certified"],
                     "group_physically_loaded": ref_physical > 0,
                     "group_force_safe": ref_groups <= settings["group_force_max_n"],
                     "penetration_safe": ref_depth <= settings["max_penetration_m"],
                     "side_normals_safe": reference_summary["normal_vertical"] <= settings["max_abs_normal_z"],
                     "opposition_safe": reference_summary["opposition"] <= settings["opposition_dot_max"],
                     "upward_load_min": ref_upward >= 0.8*weight, "upward_load_max": ref_upward <= 1.2*weight,
                     "loaded": ref_loaded, "contact_safe": ref_safe, "support": ref_support})
            self._support_count = torch.where(support, self._support_count + 1, 0)
            self._max_support_samples = torch.maximum(self._max_support_samples, self._support_count)
            qualified_now = self._support_count >= consecutive_window_samples(settings["support_window_s"], self.physics_dt)
            self._support_qualified = qualified_now
            lift_height = position[:, 2] - self._start_cube_z
            probe_limit = settings["probe_clearance_max_m"] + settings["probe_overshoot_tolerance_m"]
            self._probe_overshoot |= (clearance > probe_limit) | (lift_height > probe_limit)
            self._unsafe |= ~finite | ~joint_safe | self._probe_overshoot | (depth > settings["max_penetration_m"])
            # This version terminates at qualified 2--6 mm support. It never
            # executes or claims the original 0.1 m lift / 0.5 s carry gate.
            self._local_success = qualified_now & ~self._unsafe
            self._max_clearance = torch.maximum(self._max_clearance, clearance)
            table_supported = self._table_contact_channel.snapshot()["supported"]
            unknown_table = torch.full_like(table_fn, float("nan"))
            self._last_metrics = {"clearance_m": clearance, "object_speed_m_s": speed,
                                  "object_lift_height_m": lift_height,
                                  "object_angular_speed_rad_s": spin, "grip_drift_m": drift,
                                  "grip_reference_locked": self._grip_locked,
                                  "relative_rotation_rad": rotation_error,
                                  "table_normal_force_n": table_fn if table_supported else unknown_table,
                                  "table_pair_measurement_supported": torch.full((self.num_envs,), table_supported, dtype=torch.bool, device=self.device),
                                  "hand_upward_force_n": upward, "other_normal_force_n": other_force,
                                  "no_loaded_table_contact": (table_loaded_count == 0) if table_supported else unknown_table,
                                  "no_other_loaded_external_contact": other_loaded_count == 0,
                                  "group_normal_force_n": groups, "maximum_penetration_m": depth,
                                  "group_physical_loaded_contact_count": group_physical,
                                  "required_groups_physically_loaded": (group_physical > 0).all(dim=-1),
                                  "no_positive_gap_hand_load": no_positive_gap_load,
                                  "friction_partner_manifold_certified": friction_manifold,
                                  "maximum_abs_normal_z": normal_vertical, "opposition_dot": opposition,
                                  "finite": finite, "joint_safe": joint_safe, "contact_safe": contact_safe,
                                  "support_current": support, "support_qualified": self._support_qualified,
                                  "unsafe_probe_overshoot": self._probe_overshoot, "success": self._local_success}
            self._physics_samples += 1
            if self._record:
                count = min(8, self.num_envs)
                row = {"event": "post_physics", "physics_sample": self._physics_samples,
                       "native_physics_step_count": int(self.sim.get_physics_step_count()),
                       "actual_physics_dt_s": float(self.sim.get_physics_dt()),
                       "episode_id": self._episode_ids[:count].detach().cpu().tolist(),
                       "env_ids": list(range(count)), "force_unavailable": self._force_unavailable,
                       "qualification_supported": self._qualification_supported,
                       "table_contact_channel": self._table_contact_channel.snapshot(),
                       "main_lift_authorized": False,
                       "metrics": {name: value[:count].detach().cpu().tolist() for name, value in self._last_metrics.items()},
                       "joint_position_rad": tensor(self.robot.data.joint_pos)[:count].detach().cpu().tolist(),
                       "joint_velocity_rad_s": tensor(self.robot.data.joint_vel)[:count].detach().cpu().tolist(),
                       "requested_policy_action": self._actions[:count].detach().cpu().tolist(),
                       "action_origin": self._action_origin,
                       "applied_joint_position_target_rad": self._targets[:count].detach().cpu().tolist(),
                       "cube_position_world_m": position[:count].detach().cpu().tolist(),
                       "cube_quaternion_xyzw": rotation[:count].detach().cpu().tolist(),
                       "cube_linear_velocity_world_m_s": velocity[:count].detach().cpu().tolist(),
                       "cube_angular_velocity_world_rad_s": angular[:count].detach().cpu().tolist(),
                       "cube_position_in_palm_m": relative[:count].detach().cpu().tolist(),
                       "cube_rotation_in_palm_xyzw": relative_rotation[:count].detach().cpu().tolist(),
                       "raw_contacts": {name: {key: value.detach().cpu().tolist() for key, value in raw.items()}
                                        for name, raw in self._raw_contact_tensors.items()}}
                self._write_evaluation_row(row)

        def _get_dones(self):
            if self._pending_physics_sample:
                self._sample_physics()
                self._pending_physics_sample = False
            if self._control_probe_enabled:
                # DirectRLEnv resets terminal environments before returning
                # from step. Keep the actual pre-reset transition for the
                # runner's explicit post-step recording call.
                self._control_transition_snapshot = {
                    "event": "control_transition", "command_id": self._control_command_id,
                    "profile": dict(self._control_probe_profile),
                    "native_physics_step_count": int(self.sim.get_physics_step_count()),
                    "episode_id": self._episode_ids.detach().clone(),
                    "episode_control_steps": self.episode_length_buf.detach().clone(),
                    "measured_q_rad": tensor(self.robot.data.joint_pos).detach().clone(),
                    "measured_qd_rad_s": tensor(self.robot.data.joint_vel).detach().clone(),
                    "applied_target_rad": self._targets.detach().clone(),
                    "cube_position_world_m": tensor(self.cube.data.root_pos_w).detach().clone(),
                    "cube_quaternion_xyzw": tensor(self.cube.data.root_quat_w).detach().clone(),
                    "metrics": {name: value.detach().clone() for name, value in self._last_metrics.items()},
                    "unsafe_latched": self._unsafe.detach().clone(),
                    "main_lift_authorized": False, "pre_reset": True}
            if self._standalone_video:
                self._capture_evaluation_frame()
            dropped = tensor(self.cube.data.root_pos_w)[:, 2] < 0.70
            terminated = self._unsafe | self._local_success | dropped
            truncated = self.episode_length_buf >= self.max_episode_length
            return terminated, truncated

        def _get_rewards(self):
            _, _, velocity, angular, relative, _, clearance = self._physical_state()
            grasp = torch.tensor(config["expert"]["teacher"]["grasp_center_in_palm_m"], device=self.device)
            reach = torch.exp(-20.0*torch.linalg.vector_norm(relative-grasp, dim=-1))
            tips = tensor(self.robot.data.body_pos_w)[:, self._tip_ids]
            cube = tensor(self.cube.data.root_pos_w)
            fingertip = torch.exp(-30.0*torch.linalg.vector_norm(tips-cube[:, None], dim=-1)).mean(dim=-1)
            group_force = self._last_metrics["group_normal_force_n"]
            contacts = (group_force > settings["group_force_min_n"]).float().mean(dim=-1)
            depth = self._last_metrics["maximum_penetration_m"]
            terms = {
                "reach": settings["reward_reach"]*reach,
                "fingertip": settings["reward_fingertip"]*fingertip,
                "contact": settings["reward_contact"]*contacts,
                "clearance": settings["reward_clearance"]*(clearance/0.006).clamp(0, 1)*contacts,
                "support": settings["reward_support"]*self._last_metrics["support_current"].float(),
                "lift": settings["reward_lift"]*(clearance/settings["lift_height_m"]).clamp(0, 1)*self._support_qualified,
                "success": settings["reward_success"]*self._local_success.float(),
                "action_change": -settings["penalty_action_change"]*((self._actions-self._previous_actions)**2).mean(dim=-1),
                "object_speed": -settings["penalty_object_speed"]*(torch.linalg.vector_norm(velocity, dim=-1) +
                                                                           0.1*torch.linalg.vector_norm(angular, dim=-1)).clamp(max=10),
                "force": -settings["penalty_force"]*(group_force-settings["group_force_max_n"]).clamp(min=0).sum(dim=-1),
                "penetration": -settings["penalty_penetration"]*(depth/settings["max_penetration_m"]).clamp(0, 10),
            }
            self._reward_terms = terms
            reward = sum(terms.values())
            if not bool(torch.isfinite(reward).all()):
                raise RLEnvironmentError("nonfinite physical state or reward")
            self._episode_return += reward
            self.extras["reward_components"] = {name: float(value.mean().item()) for name, value in terms.items()}
            return reward

        def _get_observations(self):
            position, rotation, velocity, angular, relative, rel_rotation, clearance = self._physical_state()
            q = tensor(self.robot.data.joint_pos)[:, self._action_ids]
            qv = tensor(self.robot.data.joint_vel)[:, self._action_ids]
            body_rot = tensor(self.robot.data.body_quat_w)[:, self._palm_id]
            palm = tensor(self.robot.data.body_pos_w)[:, self._palm_id] + quat_apply(body_rot, self._palm_local)
            palm_rot = quat_mul(body_rot, self._palm_local_rotation)
            tips = tensor(self.robot.data.body_pos_w)[:, self._tip_ids] - position[:, None]
            remaining = (self.max_episode_length-self.episode_length_buf).float()/self.max_episode_length
            obs = torch.cat((q, qv*0.1, position-self.scene.env_origins, rotation, velocity, angular,
                             relative, rel_rotation, palm-self.scene.env_origins, palm_rot,
                             tips.reshape(self.num_envs, 9), self._actions,
                             clearance[:, None], self._support_qualified.float()[:, None], remaining[:, None]), dim=-1)
            if obs.shape != (self.num_envs, OBSERVATION_DIM) or not bool(torch.isfinite(obs).all()):
                raise RLEnvironmentError("invalid privileged observation shape or values")
            if self._mechanical_migration is None and bool((self._episode_ids > 0).all()):
                self._verify_initial_reset_fk(palm, palm_rot)
            return {"policy": obs}

        def _verify_initial_reset_fk(self, palm_position, palm_rotation):
            rows = []
            origins = self.scene.env_origins.detach().cpu().tolist()
            palms = palm_position.detach().cpu().tolist()
            rotations = palm_rotation.detach().cpu().tolist()
            cube = tensor(self.cube.data.root_pos_w).detach().cpu().tolist()
            cube_rotation = tensor(self.cube.data.root_quat_w).detach().cpu().tolist()
            roots = tensor(self.robot.data.root_pos_w).detach().cpu().tolist()
            root_rotation = tensor(self.robot.data.root_quat_w).detach().cpu().tolist()
            actual_joints = tensor(self.robot.data.joint_pos).detach().cpu().tolist()
            expected_cube = self._expected_reset_cube.detach().cpu().tolist()
            official_reference = initial.get("official_native_fk_reference") if native_official_asset else None
            for index in range(self.num_envs):
                expected_palm = [float(value)+float(origin) for value, origin in
                                 zip(initial["source_palm_world_position_m"], origins[index])]
                expected_palm[2] += float(initial["reset_vertical_lift_m"])
                legacy_error = migration_pose_errors(expected_palm, initial["source_palm_rotation_world"],
                                                       palms[index], rotations[index])
                expected_palm_rotation = initial["source_palm_rotation_world"]
                if official_reference:
                    expected_palm = [value+origin for value, origin in zip(official_reference["palm_world_position_m"], origins[index])]
                    expected_palm_rotation = official_reference["palm_rotation_world"]
                palm_errors = migration_pose_errors(expected_palm, expected_palm_rotation,
                                                     palms[index], rotations[index])
                cube_position_error = math.sqrt(sum((a-b)**2 for a, b in zip(cube[index], expected_cube[index][:3])))
                dot = abs(sum(a*b for a, b in zip(cube_rotation[index], expected_cube[index][3:])))
                cube_rotation_error = 2*math.acos(min(1.0, max(-1.0, dot)))
                expected_root = [value+origin for value, origin in zip(initial["root_pose_xyzw"][:3], origins[index])]
                root_error = math.sqrt(sum((a-b)**2 for a, b in zip(roots[index], expected_root)))
                root_dot = abs(sum(a*b for a, b in zip(root_rotation[index], initial["root_pose_xyzw"][3:])))
                root_rotation_error = 2*math.acos(min(1.0, max(-1.0, root_dot)))
                joint_error = max(abs(a-float(initial["joint_positions_rad"][name]))
                                  for a, name in zip(actual_joints[index], self.robot.joint_names))
                rows.append({"env_id": index, "palm": palm_errors, "cube_position_error_m": cube_position_error,
                             "legacy_source_palm_difference_diagnostic": legacy_error,
                             "cube_rotation_error_rad": cube_rotation_error, "root_position_error_m": root_error,
                             "root_rotation_error_rad": root_rotation_error, "reset_joint_position_error_max_rad": joint_error,
                             "actual_palm_world_position_m": palms[index], "actual_palm_world_quaternion_xyzw": rotations[index],
                             "expected_palm_world_position_m": expected_palm,
                             "actual_cube_position_world_m": cube[index], "actual_cube_quaternion_xyzw": cube_rotation[index],
                             "expected_cube_pose_xyzw": expected_cube[index]})
            enforce_palm = not native_official_asset or bool(official_reference)
            position_max = max(max(row["palm"]["position_error_m"] if enforce_palm else 0,
                                   row["cube_position_error_m"], row["root_position_error_m"]) for row in rows)
            rotation_max = max(max(row["palm"]["rotation_error_rad"] if enforce_palm else 0,
                                   row["cube_rotation_error_rad"], row["root_rotation_error_rad"]) for row in rows)
            joint_max = max(row["reset_joint_position_error_max_rad"] for row in rows)
            passed = (position_max <= settings["migration_position_tolerance_m"] and
                      rotation_max <= settings["migration_rotation_tolerance_rad"] and joint_max <= 1e-5)
            self._mechanical_migration = {"passed": passed, "maximum_position_error_m": position_max,
                                          "maximum_rotation_error_rad": rotation_max,
                                          "maximum_reset_joint_error_rad": joint_max,
                                          "palm_fk_reference_scope": "official_independent_reference" if official_reference else
                                              "legacy_source_geometry_diagnostic_only" if native_official_asset else "same_source_conversion_equivalence",
                                          "official_independent_palm_fk_reference_pending": native_official_asset and not bool(official_reference),
                                          "legacy_palm_difference_blocks_native_training": False if native_official_asset else True,
                                          "position_tolerance_m": settings["migration_position_tolerance_m"],
                                          "rotation_tolerance_rad": settings["migration_rotation_tolerance_rad"],
                                          "physics_steps_executed_before_check": self._physics_samples, "environments": rows}
            if output:
                path = output / "initial_reset_fk_audit.json"
                with path.open("x", encoding="utf-8") as stream:
                    json.dump(self._mechanical_migration, stream, indent=2, allow_nan=False)
            self._write_evaluation_row({"event": "initial_reset_fk", "audit": self._mechanical_migration})
            if not passed:
                raise RLEnvironmentError("mechanical migration FK failed: position {:.9f} m, rotation {:.9f} rad".format(position_max, rotation_max))

        def _reset_idx(self, env_ids):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
            # Constructor has not yet allocated curriculum buffers.
            if not hasattr(self, "_episode_ids"):
                super()._reset_idx(env_ids)
                return
            terminal = self.reset_buf[env_ids] & (self.episode_length_buf[env_ids] > 0)
            terminal_ids = env_ids[terminal]
            terminal_indices = terminal_ids.detach().cpu().tolist()
            # Copy each field once for the complete terminal batch. Per-episode
            # JSON construction below runs on CPU, including every flag/item.
            metric_batch = {name: value[terminal_ids].detach().cpu().tolist()
                            for name, value in self._last_metrics.items()} if terminal_indices else {}
            fields = {"probe_overshoot": self._probe_overshoot, "unsafe": self._unsafe,
                      "support_qualified": self._support_qualified, "local_success": self._local_success,
                      "episode_id": self._episode_ids, "elapsed_steps": self.episode_length_buf,
                      "episode_return": self._episode_return, "max_clearance_m": self._max_clearance,
                      "maximum_support_samples": self._max_support_samples}
            field_batch = {name: value[terminal_ids].detach().cpu().tolist()
                           for name, value in fields.items()} if terminal_indices else {}
            measurement_unavailable = self._force_unavailable or self._table_contact_channel.snapshot()["unavailable_reason"]
            for offset, index in enumerate(terminal_indices):
                metrics = {name: _json_finite(values[offset]) for name, values in metric_batch.items()}
                episode = {name: values[offset] for name, values in field_batch.items()}
                gates = {name: value for name, value in metrics.items() if isinstance(value, bool)}
                failure = []
                if self._force_unavailable:
                    failure.append("contact_measurement_unavailable")
                if episode["probe_overshoot"]:
                    failure.append("local_probe_exceeded_6mm_before_qualification")
                if episode["unsafe"]:
                    failure.append("unsafe_physical_state")
                if not episode["support_qualified"]:
                    failure.append("continuous_support_not_qualified")
                if not episode["local_success"]:
                    failure.append("local_support_window_not_qualified")
                if not metrics.get("no_positive_gap_hand_load", False):
                    failure.append("positive_gap_hand_normal_load_or_measurement_unavailable")
                if not metrics.get("friction_partner_manifold_certified", False):
                    failure.append("friction_partner_normal_manifold_uncertified")
                self._outcomes.append({"env_id": index, "episode_id": int(episode["episode_id"]),
                                       "seed": int(self._evaluation_seed if self._eval_mode else train_seed),
                                       "rollout_domain": "physx_precontact_validation" if self._eval_mode else "physx_precontact_training",
                                       "layout_id": "contact_entry_seed_jitter",
                                       "validation_offset_index": index % len(config["rl_teacher"]["validation_offsets_m"]) if self._eval_mode else None,
                                       "validation_offset_m": config["rl_teacher"]["validation_offsets_m"][index % len(config["rl_teacher"]["validation_offsets_m"])] if self._eval_mode else None,
                                       "elapsed_steps": int(episode["elapsed_steps"]),
                                       "sim_time_s": float(episode["elapsed_steps"]*self.step_dt),
                                       "episode_return": float(episode["episode_return"]),
                                       "success": None if measurement_unavailable else bool(episode["local_success"]),
                                       "local_success": False if measurement_unavailable else bool(episode["local_success"]),
                                       "qualification_supported": self._qualification_supported,
                                       "unavailable_reason": measurement_unavailable, "gates": gates, "metrics": metrics,
                                       "max_clearance_m": float(episode["max_clearance_m"]),
                                       "maximum_support_samples": int(episode["maximum_support_samples"]),
                                       "failure_reasons": failure,
                                       **({"control_probe_profile": dict(self._control_probe_profile)} if self._control_probe_enabled else {})})
            self._interrupted += int(((~terminal) & (self.episode_length_buf[env_ids] > 0)).sum().item())
            super()._reset_idx(env_ids)
            n = len(env_ids)
            origins = self.scene.env_origins[env_ids]
            root = torch.tensor(initial["root_pose_xyzw"], device=self.device).repeat(n, 1)
            root[:, :3] += origins
            cube = torch.tensor(initial["cube_pose_xyzw"], device=self.device).repeat(n, 1)
            generator = self._eval_generator if self._eval_mode else self._train_generator
            jitter = settings["eval_xy_jitter_m"] if self._eval_mode else settings["train_xy_jitter_m"]
            yaw_jitter = settings["eval_yaw_jitter_rad"] if self._eval_mode else settings["train_yaw_jitter_rad"]
            if self._eval_mode:
                offsets = torch.tensor(config["rl_teacher"]["validation_offsets_m"], device=self.device)
                offset_ids = env_ids.long() % len(offsets)
                cube[:, :2] += offsets[offset_ids]
            else:
                cube[:, :2] += (2*torch.rand((n, 2), device=self.device, generator=generator)-1)*jitter
            yaw = (2*torch.rand((n,), device=self.device, generator=generator)-1)*yaw_jitter
            yaw_quat = torch.stack((torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw/2), torch.cos(yaw/2)), -1)
            cube[:, 3:] = quat_mul(yaw_quat, cube[:, 3:])
            cube[:, :3] += origins
            q = self._initial_q.repeat(n, 1)
            self.robot.write_root_pose_to_sim_index(root_pose=root, env_ids=env_ids)
            self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)
            self.robot.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
            self.robot.set_joint_position_target_index(target=q, env_ids=env_ids)
            self.cube.write_root_pose_to_sim_index(root_pose=cube, env_ids=env_ids)
            self.cube.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=env_ids)
            self._targets[env_ids] = q
            self._actions[env_ids] = 0
            self._previous_actions[env_ids] = 0
            self._episode_ids[env_ids] += 1
            self._episode_return[env_ids] = 0
            self._support_count[env_ids] = 0
            self._support_qualified[env_ids] = False
            self._local_success[env_ids] = False
            self._unsafe[env_ids] = False
            self._probe_overshoot[env_ids] = False
            self._grip_locked[env_ids] = False
            self._start_cube_z[env_ids] = cube[:, 2]
            self._expected_reset_cube[env_ids] = cube
            self._start_palm_z[env_ids] = float("nan")
            self._max_clearance[env_ids] = -float("inf")
            self._max_support_samples[env_ids] = 0
            self._resets += n
            self._pending_physics_sample = False
            self._write_evaluation_row({"event": "episode_reset", "env_ids": env_ids.detach().cpu().tolist(),
                                        "episode_ids": self._episode_ids[env_ids].detach().cpu().tolist(),
                                        "reset_count": self._resets,
                                        "native_physics_step_count": int(self.sim.get_physics_step_count()),
                                        "declared_root_pose_xyzw": root.detach().cpu().tolist(),
                                        "declared_cube_pose_xyzw": cube.detach().cpu().tolist(),
                                        "declared_joint_position_rad": q.detach().cpu().tolist(),
                                        "velocity_protocol": "all velocities zero only at episode reset"})
            if self._video_manifest and bool((env_ids == 0).any()):
                self._video_manifest.write(json.dumps({"event": "episode_reset", "env_id": 0,
                                                       "episode_id": int(self._episode_ids[0].item()),
                                                       "next_frame_index": self._video_status["frames"],
                                                       "native_physics_step_count": int(self.sim.get_physics_step_count()),
                                                       "sim_time_s": int(self.sim.get_physics_step_count())*float(self.sim.get_physics_dt()),
                                                       **({"profile_name": self._control_probe_profile["name"]} if self._control_probe_enabled else {})}) + "\n")
                self._video_manifest.flush()

        def pop_episode_outcomes(self):
            result, self._outcomes = self._outcomes, []
            return result

        def _write_control_probe(self, row):
            if not self._control_probe_enabled or self._control_probe_stream is None:
                raise RLEnvironmentError("diagnostic evidence stream is not enabled")
            row = {**row, "profile_name": self._control_probe_profile["name"],
                   "control_probe_profile": dict(self._control_probe_profile),
                   "env_ids": list(range(self.num_envs))}
            self._control_probe_stream.write(json.dumps(_json_finite(_probe_payload(torch, row)),
                                                         ensure_ascii=False, allow_nan=False)+"\n")
            self._control_probe_stream.flush()

        def configure_control_probe(self, profile):
            """Change diagnostic controller identity at a completed control boundary."""
            if not self._control_probe_enabled:
                raise RLEnvironmentError("controller diagnostics were not enabled at construction")
            if self._pending_physics_sample or self._control_transition_snapshot is not None:
                raise RLEnvironmentError("a profile may change only after its complete transition is recorded")
            selected = control_probe_profile(profile)
            if selected is None:
                raise RLEnvironmentError("diagnostic profiles cannot be silently disabled")
            self._control_probe_profile = selected
            row = {"event": "control_profile_boundary", "profile": dict(selected),
                   "native_physics_step_count": int(self.sim.get_physics_step_count()),
                   "next_frame_index": self._video_status["frames"],
                   "next_command_id": self._control_command_id+1,
                   "joint_names": list(self.robot.joint_names), "action_joint_names": list(ACTION_JOINT_NAMES),
                   "action_joint_ids": list(self._action_ids), "main_lift_authorized": False}
            self._write_control_probe(row)
            self._write_evaluation_row(row)
            if self._video_manifest:
                self._video_manifest.write(json.dumps({**row, "profile_name": selected["name"]})+"\n")
                self._video_manifest.flush()
            return dict(selected)

        def control_diagnostic_state(self):
            """Return measured tensors; calling this never advances or writes physics."""
            if not self._control_probe_enabled:
                raise RLEnvironmentError("controller diagnostics were not enabled at construction")
            body_rotation = tensor(self.robot.data.body_quat_w)[:, self._palm_id]
            offset_w = quat_apply(body_rotation, self._palm_local)
            jacobian = tensor(self.robot.data.body_link_jacobian_w)
            if jacobian.shape[1] != self.robot.num_bodies-1 or jacobian.shape[-1] != 43:
                raise RLEnvironmentError("diagnostic site Jacobian requires the verified fixed-root articulation")
            point_jacobian = offset_point_jacobian(torch,
                jacobian[:, self._palm_id-1][:, :, self._action_ids[:7]], offset_w)
            per_step = torch.minimum(self._velocity*self.step_dt,
                                     torch.full_like(self._velocity, float(config["rl_teacher"]["action_scale_rad"])))
            values = {
                "q_action_rad": tensor(self.robot.data.joint_pos)[:, self._action_ids],
                "qd_action_rad_s": tensor(self.robot.data.joint_vel)[:, self._action_ids],
                "target_action_rad": self._targets[:, self._action_ids],
                "per_step_action_rad": per_step,
                "joint_limits_action_rad": self._limits[:, self._action_ids],
                "palm_position_world_m": tensor(self.robot.data.body_pos_w)[:, self._palm_id]+offset_w,
                "palm_quaternion_xyzw": quat_mul(body_rotation, self._palm_local_rotation),
                "palm_point_jacobian_w": point_jacobian,
                "cube_position_world_m": tensor(self.cube.data.root_pos_w),
                "cube_quaternion_xyzw": tensor(self.cube.data.root_quat_w),
                "group_normal_force_n": torch.where((self.episode_length_buf > 0)[:, None],
                    self._last_metrics.get("group_normal_force_n", torch.zeros(self.num_envs, 3, device=self.device)),
                    torch.zeros(self.num_envs, 3, device=self.device)),
                "episode_control_steps": self.episode_length_buf,
                "episode_id": self._episode_ids,
                "q_all_rad": tensor(self.robot.data.joint_pos),
                "qd_all_rad_s": tensor(self.robot.data.joint_vel),
                "target_all_rad": self._targets,
                "tip_link_origin_world_m": tensor(self.robot.data.body_pos_w)[:, self._tip_ids],
                "unsafe_latched": self._unsafe,
                "probe_overshoot": self._probe_overshoot,
                "preload_contact_ready": (preload_contact_ready(torch, self._last_metrics, self.num_envs, self.device) &
                                          self._grip_locked & (self.episode_length_buf > 0)),
                "contact_metrics_current_episode": self.episode_length_buf > 0,
                "metrics": self._last_metrics,
            }
            def clone(value):
                if torch.is_tensor(value):
                    return value.detach().clone()
                return {name: clone(item) for name, item in value.items()}
            result = {name: clone(value) for name, value in values.items()}
            result.update(action_joint_ids=list(self._action_ids), action_joint_names=list(ACTION_JOINT_NAMES),
                          all_joint_names=list(self.robot.joint_names),
                          tip_link_order=["thumb_2", "middle_1", "index_1"],
                          control_probe_profile=dict(self._control_probe_profile),
                          native_physics_step_count=int(self.sim.get_physics_step_count()))
            return result

        def record_control_probe_transition(self):
            """Persist actual post-step, pre-reset states exactly once per command."""
            row = self._control_transition_snapshot
            if not self._control_probe_enabled or row is None or row["command_id"] <= self._control_transition_recorded:
                raise RLEnvironmentError("no new complete diagnostic control transition is available")
            if row["profile"] != self._control_probe_profile:
                raise RLEnvironmentError("diagnostic profile changed before transition recording")
            self._write_control_probe(row)
            self._control_transition_recorded = row["command_id"]
            self._control_transition_snapshot = None
            return {"command_id": row["command_id"], "profile_name": row["profile"]["name"],
                    "native_physics_step_count": row["native_physics_step_count"]}

        def state_hold_actions(self):
            """Request current measured 14-joint positions through the same limiter.

            A zero policy action instead holds the previous position target.
            This request may be clipped by velocity, anchor and probe guards.
            """
            current = tensor(self.robot.data.joint_pos)[:, self._action_ids]
            per_step = torch.minimum(self._velocity*self.step_dt,
                                     torch.full_like(self._velocity, float(config["rl_teacher"]["action_scale_rad"])))
            raw = (current-self._targets[:, self._action_ids])/per_step
            self._state_hold_request_clipped += int((torch.abs(raw) > 1).any(dim=-1).sum().item())
            self._next_action_origin = "measured_state_hold_request_through_shared_limits"
            return raw.clamp(-1.0, 1.0)

        def _capture_evaluation_frame(self):
            if self._video_status["failure"] or not self._video_manifest:
                return
            try:
                from PIL import Image
                native_step = int(self.sim.get_physics_step_count())
                if native_step == self._last_video_native_step:
                    raise RLEnvironmentError("duplicate native physics step in video frame capture")
                self._evaluation_camera.update(dt=self.step_dt, force_recompute=True)
                rgb = tensor(self._evaluation_camera.data.output["rgb"])
                if rgb.dtype != torch.uint8 or tuple(rgb.shape[:3]) != (1, 480, 640) or rgb.shape[-1] not in (3, 4):
                    raise RLEnvironmentError("camera did not provide actual 640x480 uint8 RGB(A) pixels")
                pixels = rgb[0, :, :, :3].detach().cpu().numpy()
                frame = int(self._video_status["frames"])
                path = self._video_frame_dir / "frame_{:06d}.png".format(frame)
                Image.fromarray(pixels, mode="RGB").save(path)
                payload = path.read_bytes()
                camera_frame = int(tensor(self._evaluation_camera.frame)[0].item())
                if self._last_camera_frame is not None and camera_frame <= self._last_camera_frame:
                    raise RLEnvironmentError("camera measurement counter did not advance")
                row = {"event": "camera_frame", "frame_index": frame, "path": path.name,
                       "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                       "native_physics_step_count": native_step, "physics_samples": self._physics_samples,
                       "actual_physics_dt_s": float(self.sim.get_physics_dt()),
                       "sim_time_s": native_step*float(self.sim.get_physics_dt()),
                       "episode_id": int(self._episode_ids[0].item()),
                       "episode_control_steps": int(self.episode_length_buf[0].item()),
                       "camera_frame_counter": camera_frame,
                       "cube_position_world_m": tensor(self.cube.data.root_pos_w)[0].detach().cpu().tolist(),
                       "cube_quaternion_xyzw": tensor(self.cube.data.root_quat_w)[0].detach().cpu().tolist(),
                       "pre_reset": True,
                       **({"profile_name": self._control_probe_profile["name"]} if self._control_probe_enabled else {})}
                self._video_manifest.write(json.dumps(row, allow_nan=False) + "\n")
                self._video_manifest.flush()
                self._video_status["frames"] = frame+1
                self._last_video_native_step = native_step
                self._last_camera_frame = camera_frame
            except Exception as exc:
                self._video_status["failure"] = type(exc).__name__ + ": " + str(exc)
                self._video_manifest.write(json.dumps({"event": "camera_failure", "failure": self._video_status["failure"]}) + "\n")
                self._video_manifest.flush()
                if output:
                    with (output / "evaluation_video_failure.json").open("x", encoding="utf-8") as stream:
                        json.dump(self._video_status, stream, indent=2)

        def set_evaluation_mode(self, enabled, seed=None):
            if self._record:
                self._record.close()
                self._record = None
            self._eval_mode = bool(enabled)
            if enabled:
                self._evaluation_seed = int(settings["eval_seed"] if seed is None else seed)
                self._eval_generator.manual_seed(self._evaluation_seed)
                self._open_evaluation_recording()

        def _open_evaluation_recording(self):
            if output is None:
                return
            self._record_index += 1
            path = output / "evaluation_physics_{:04d}.jsonl".format(self._record_index)
            self._record = path.open("x", encoding="utf-8")
            self._write_evaluation_row({"event": "recording_header", "environment_version": ENVIRONMENT_VERSION,
                                        "recorded_env_ids": list(range(min(8, self.num_envs))),
                                        "evaluation_seed": self._evaluation_seed,
                                        "joint_names": list(self.robot.joint_names),
                                        "action_joint_names": list(ACTION_JOINT_NAMES),
                                        "physics_dt_s": self.physics_dt, "control_period_s": self.step_dt,
                                        "contact_force_semantics": "raw PhysX point normal force and separated friction aggregate, both divided by actual physics dt",
                                        "main_lift_authorized": False, "policy_contract": self.runtime_audit()["policy_contract"]})

        def _write_evaluation_row(self, row):
            if self._record:
                if self._control_probe_enabled:
                    row = {**row, "control_probe_profile": dict(self._control_probe_profile)}
                self._record.write(json.dumps(_json_finite(row), ensure_ascii=False, allow_nan=False) + "\n")

        def runtime_audit(self):
            audit = {"environment_version": ENVIRONMENT_VERSION, "settings": dict(settings),
                    "policy_contract": {"environment_version": ENVIRONMENT_VERSION,
                                        "contact_extraction_backend": CONTACT_COLLECTION_VERSION,
                                        "support_gate_version": SUPPORT_GATE_VERSION,
                                        "table_body_protocol": TABLE_BODY_PROTOCOL,
                                        "table_rigid_schema": self._table_schema_audit,
                                        "table_channel_admission": "native rigid binding plus positive normal-force witness required; unsupported Table fields export null; kinematic body remains at original fixed pose without episode writes",
                                        "actual_side_contact_definition": "distal opposition uses abs(raw signed Fn)>20mN AND separation<=0; side-normal limit checks max(abs(world-z),abs(cube-local-z)) for every physical Fn>0,separation<=0 point including weaker points; every thumb/index/middle requires a physical loaded point; positive gaps cannot provide opposition or grip lock",
                                        "aggregate_force_scope": "full forces retained for shaping; qualification and first grip lock forbid every hand nonzero Fn at positive gap; every partner with any nonzero raw friction patch requires a nonempty complete normal manifold with all separations<=0 including zero-Fn points; failure is current qualification false, not sticky API failure; no patch-to-point separation equivalence or MuJoCo efc claim",
                                        "raw_contact_orientation": "reversed view preserves canonical point normals and negates signed scalar force; aggregate normal/friction vectors negate; Cube raw ledger unchanged",
                                        "cube_raw_contact_capacity_per_env": 256,
                                        "reference_hand_raw_contact_capacity_per_body_per_env": 32,
                                        "contact_measurement_capacity_change": "Cube raw buffer expanded from historical 32 to 256 points per env; measurement capacity only, not a physics setting",
                                        "contact_crosscheck_protocol": CONTACT_CROSSCHECK,
                                        "hand_contact_partner_indices": self._cube_hand_filter_indices,
                                        "observation_version": OBSERVATION_VERSION, "observation_dim": OBSERVATION_DIM,
                                        "action_version": ACTION_VERSION, "action_dim": 14,
                                        "action_joint_names": list(ACTION_JOINT_NAMES),
                                        "action_units": "normalized joint target velocity; rad/s after source limit scaling",
                                        "velocity_limits_rad_s": list(policy_velocity_limits(config)),
                                        "physics_dt_s": self.physics_dt, "control_period_s": self.step_dt,
                                        "asset_sha256": asset_sha, "quaternion_order": "xyzw",
                                        "resolved_settings": dict(settings),
                                        "task_protocol": config["rl_teacher"]["task_protocol"],
                                        "physics_protocol": config["rl_teacher"]["physics_protocol"],
                                        "reset_protocol": config["rl_teacher"]["reset_protocol"],
                                        "scope": config["rl_teacher"]["scope"],
                                        "training_position_jitter_m": config["rl_teacher"]["training_position_jitter_m"],
                                        "training_yaw_jitter_rad": config["rl_teacher"]["training_yaw_jitter_rad"],
                                        "validation_offsets_m": config["rl_teacher"]["validation_offsets_m"],
                                        "action_scale_rad": config["rl_teacher"]["action_scale_rad"],
                                        "joint_limit_tolerance_rad": config["eval"]["joint_limit_tolerance_rad"],
                                        "actual_pd_specifications": actuator_specs,
                                        "pd_backend": "implicit_force_drive_without_mujoco_qfrc_bias_v1",
                                        "seed_state_sha256": config["rl_teacher"].get("seed_state_sha256"),
                                        "asset_manifest_sha256": config["rl_teacher"].get("asset_manifest_sha256"),
                                        "asset_verification": asset_audit,
                                        "initial_state_velocity_protocol": initial.get("reset_velocity_protocol"),
                                        "reset_vertical_lift_m": initial.get("reset_vertical_lift_m"),
                                        "qualification_semantics": "raw per-point distal opposition and normals; source loaded-contact ANDs; 100 consecutive 2ms samples",
                                        "compiled_asset_files": self._compiled_files,
                                        "hand_material_bindings": self._material_bindings,
                                        "stage_only_instance_overrides": self._instance_overrides,
                                        "tensor_recursive_leaf_pattern_match": tensor_settings.get("/physics/tensors/recursiveLeafPatternMatch"),
                                        "cube_contact_partner_names": self._cube_filter_body_names,
                                        "other_robot_contact_partner_indices": self._cube_other_filter_indices,
                                        "fixed_root_audit": self._fixed_root_audit,
                                        "native_joint_axis_frame_unit_schema": self._joint_schema_audit,
                                        "palm_site_body_name": palm_site_body,
                                        "palm_site_position_m": initial["palm_site_position_m"],
                                        "palm_site_quaternion_xyzw": initial["palm_site_quaternion_xyzw"],
                                        "palm_contact_body_name": palm_contact_body,
                                        "official_native_fk_reference": initial.get("official_native_fk_reference"),
                                        "initial_state_sha256": hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest()},
                    "robot_joint_names": list(self.robot.joint_names), "action_joint_indices": list(self._action_ids),
                    "initial_state": initial, "physics_version": config["rl_teacher"]["physics_protocol"],
                    "asset_kind": "official_native_usd" if native_official_asset else "historical_mjcf_conversion_control",
                    "physics_differences": ["PhysX TGS, not MuJoCo Noslip10", "hard root fixation, not compliant base weld",
                                            "source static table is now a gravity-disabled kinematic rigid body at the same geometry, material and fixed pose for native GPU pair contacts",
                                            "implicit shared gains without MuJoCo qfrc_bias feedforward",
                                            "explicit max static/dynamic friction, no MuJoCo torsional/rolling friction mapping",
                                            "official USD collision geometry retained with 1 mm contact offset" if native_official_asset else
                                                "historical PhysX convex collision conversion with 1 mm contact offset"],
                    "source_joint_limit_differences_rad": self._source_joint_limit_differences,
                    "contact_body_binding_audit": self._contact_body_binding_audit,
                    "table_contact_channel": self._table_contact_channel.snapshot(),
                    "cube_raw_contact_capacity_per_env": 256,
                    "reference_hand_raw_contact_capacity_per_body_per_env": 32,
                    "contact_collection_crosscheck": dict(self._contact_crosscheck),
                    "success_scope": "local precontact-reset curriculum only; original MuJoCo 15/19 and 20 not certified",
                    "contact_data_unavailable_reason": self._force_unavailable,
                    "contact_extraction_backend": CONTACT_COLLECTION_VERSION,
                    "qualification_supported": self._qualification_supported,
                    "mechanical_migration_qualification": self._mechanical_migration,
                    "support_required_samples": consecutive_window_samples(settings["support_window_s"], self.physics_dt),
                    "original_main_lift_enabled": False,
                    "main_lift_authorized": False,
                    "observation_cameras": [], "in_episode_object_state_writes": 0, "object_attachments": 0}
            if self._control_probe_enabled:
                audit["policy_contract"]["control_guard_version"] = self._control_probe_profile["control_guard_version"]
                audit["control_probe_profile"] = dict(self._control_probe_profile)
                audit["diagnostic_legacy_postcheck"] = "exact residuals recorded; source-pinned roundoff-only allowance, meaningful box/height violations latch unsafe"
            return audit

        def step_audit(self):
            return {"physics_samples": self._physics_samples, "reset_count": self._resets,
                    "interrupted_episode_count": self._interrupted, "evaluation_mode": self._eval_mode,
                    "main_lift_authorized": False, "action_shield_count": self._shield_count,
                    "action_origin": self._action_origin,
                    "zero_action_semantics": "previous_target_hold",
                    "measured_state_hold_request_clipped_count": self._state_hold_request_clipped,
                    "sim_time_s": self._physics_samples*self.physics_dt,
                    "native_physics_step_count": int(self.sim.get_physics_step_count()),
                    "actual_physics_dt_s": float(self.sim.get_physics_dt()),
                    "control_period_s": float(self.step_dt),
                    "contact_data_unavailable_reason": self._force_unavailable,
                    "contact_extraction_backend": CONTACT_COLLECTION_VERSION,
                    "contact_collection_crosscheck": dict(self._contact_crosscheck),
                    "qualification_supported": self._qualification_supported,
                    "table_contact_channel": self._table_contact_channel.snapshot(),
                    "evaluation_video": dict(self._video_status),
                    "nonfinite_state": bool((~self._last_metrics.get("finite", torch.ones(self.num_envs, dtype=torch.bool, device=self.device))).any()),
                    "success_count_pending": sum(row["success"] is True for row in self._outcomes),
                    "reward_components": {name: float(value.mean().item()) for name, value in self._reward_terms.items()},
                    **({"control_probe_profile": dict(self._control_probe_profile),
                        "control_command_count": self._control_command_id,
                        "control_transition_count": self._control_transition_recorded,
                        "control_guard_correction_count": self._control_guard_correction_count,
                        "control_guard_unsafe_count": self._control_guard_unsafe_count,
                        "shield_count_semantics": "legacy applied scalar<1; projected actual target correction nonzero"}
                       if self._control_probe_enabled else {})}

        def close(self):
            if getattr(self, "_control_probe_stream", None):
                self._control_probe_stream.close()
                self._control_probe_stream = None
            if getattr(self, "_crosscheck_record", None):
                self._crosscheck_record.close()
                self._crosscheck_record = None
            if output and hasattr(self, "_contact_crosscheck"):
                path = output / "contact_collection_crosscheck_summary.json"
                if not path.exists():
                    with path.open("x", encoding="utf-8") as stream:
                        json.dump(self._contact_crosscheck, stream, indent=2, allow_nan=False)
            if getattr(self, "_record", None):
                self._record.close()
                self._record = None
            if getattr(self, "_video_manifest", None):
                self._video_manifest.close()
                self._video_manifest = None
            if output and hasattr(self, "_video_status"):
                self._video_status["expected_control_frames"] = self._physics_samples // 33
                self._video_status["all_control_frames_recorded"] = (
                    self._standalone_video and self._video_status["failure"] is None and
                    self._video_status["frames"] == self._physics_samples // 33 and self._video_status["frames"] > 0)
                path = output / "evaluation_video_status.json"
                if not path.exists():
                    with path.open("x", encoding="utf-8") as stream:
                        json.dump(self._video_status, stream, indent=2)
            self._evaluation_camera = None
            super().close()

    return Environment()


def _json_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    return value
