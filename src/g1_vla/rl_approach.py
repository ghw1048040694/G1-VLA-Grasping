"""Pure, episode-bound arm approach references; no simulator or admission writes.

The environment owns collision/force history and physical handoff. This module
only consumes measured state, advances declared joint waypoints, and reports
kinematic arrival. The grasp hand is held at its separately pinned source pose.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


APPROACH_VERSION = "g1_task_preshape_arm_home_approach_v1"
ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
HAND_JOINT_NAMES = (
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint", "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
)
ALL_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint", "left_hand_index_0_joint",
    "left_hand_index_1_joint", *ARM_JOINT_NAMES, *HAND_JOINT_NAMES,
)
APPROACH_PARAMETERS = {
    "control_period_s": .066,
    "joint_arrival_tolerance_rad": .015,
    "joint_velocity_arrival_tolerance_rad_s": .05,
    "palm_position_arrival_tolerance_m": .003,
    "palm_rotation_arrival_tolerance_rad": .05,
    "hand_seed_arrival_tolerance_rad": .015,
    "approach_timeout_s": 16.0,
    "handoff_required_consecutive_native_samples": 25,
    "grasp_budget_s": 16.0,
    "total_episode_budget_s": 32.0,
    "workspace_min_palm_z_m": .78,
    "workspace_max_palm_z_m": 1.10,
    "maximum_object_xy_drift_m": .002,
    "maximum_object_rotation_rad": .05,
}
RESOURCE_KEYS = {
    "version", "original_candidate_sha256", "root_pose_xyzw", "source_hand_rad",
    "home_joint_positions_rad", "home_palm_position_world_m", "home_palm_quaternion_xyzw",
    "waypoints", "pregrasp_joint_positions_rad", "pregrasp_palm_position_world_m",
    "pregrasp_palm_quaternion_xyzw",
}
WAYPOINT_KEYS = {"name", "arm_target_rad", "duration_s", "palm_position_world_m", "palm_quaternion_xyzw"}
FAILURE_CODES = {
    "nonfinite_measurement": 1,
    "home_reset_mismatch": 2,
    "approach_timeout": 4,
    "environment_unsafe": 8,
    "stale_or_invalid_episode_clock": 16,
    "declared_target_outside_native_limits": 32,
    "palm_workspace_violation": 64,
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value, name):
    _require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             name + " must be a lowercase SHA256")
    return value


def _number(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), name + " must be finite numeric")
    return float(value)


def _vector(value, length, name):
    _require(type(value) is list and len(value) == length, name + " has incorrect length/type")
    return [_number(item, name) for item in value]


def _quaternion(value, name):
    result = _vector(value, 4, name)
    _require(abs(sum(item*item for item in result)-1.0) <= 1e-6, name + " must be a unit xyzw quaternion")
    return result


def _same_vector(a, b, tolerance=1e-10):
    return max(abs(x-y) for x, y in zip(a, b)) <= tolerance


def _same_rotation(a, b):
    return _same_vector(a, b) or _same_vector(a, [-x for x in b])


def _parameters(value):
    _require(type(value) is dict and set(value) == set(APPROACH_PARAMETERS), "approach parameters require exact keys")
    for key, expected in APPROACH_PARAMETERS.items():
        item = value[key]
        _require(type(item) is type(expected) if type(expected) is int else type(item) in (int, float),
                 "approach parameter has incorrect numeric type: " + key)
        _require(not isinstance(item, bool) and math.isfinite(item) and item == expected,
                 "approach parameter differs from fixed protocol: " + key)
    return deepcopy(APPROACH_PARAMETERS)


def validate_approach_declaration(value):
    """Validate an optional explicit resource/parameter declaration, without I/O."""
    if value is None:
        return None
    _require(type(value) is dict and set(value) == {"version", "resource_path", "resource_sha256", "parameters"},
             "approach declaration requires exact keys")
    _require(value["version"] == APPROACH_VERSION, "unknown approach declaration version")
    _require(type(value["resource_path"]) is str and bool(value["resource_path"].strip()) and
             "\0" not in value["resource_path"], "approach resource path must be a nonempty string")
    _sha(value["resource_sha256"], "resource_sha256")
    result = deepcopy(value)
    result["parameters"] = _parameters(value["parameters"])
    return result


def validate_approach_resource(value, *, expected_candidate_sha256=None):
    """Require a complete arm-only reset/route/target with pinned source hand."""
    _require(type(value) is dict and set(value) == RESOURCE_KEYS, "approach resource requires exact keys")
    _require(value["version"] == APPROACH_VERSION, "unknown approach resource version")
    _sha(value["original_candidate_sha256"], "original_candidate_sha256")
    if expected_candidate_sha256 is not None:
        _require(value["original_candidate_sha256"] == _sha(expected_candidate_sha256, "expected_candidate_sha256"),
                 "approach source candidate identity mismatch")
    result = deepcopy(value)
    root = _vector(value["root_pose_xyzw"], 7, "root_pose_xyzw")
    _quaternion(root[3:], "root_pose_xyzw quaternion")
    result["root_pose_xyzw"] = root
    hand = _vector(value["source_hand_rad"], 7, "source_hand_rad")
    result["source_hand_rad"] = hand
    for prefix in ("home", "pregrasp"):
        key = prefix + "_joint_positions_rad"
        joints = value[key]
        _require(type(joints) is dict and set(joints) == set(ALL_JOINT_NAMES) and len(joints) == 43,
                 key + " must name all43 canonical joints exactly")
        result[key] = {name: _number(joints[name], key+"/"+name) for name in ALL_JOINT_NAMES}
        _require(_same_vector([joints[name] for name in HAND_JOINT_NAMES], hand),
                 prefix + " hand must equal the pinned source hand")
        poskey, quatkey = prefix+"_palm_position_world_m", prefix+"_palm_quaternion_xyzw"
        result[poskey] = _vector(value[poskey], 3, poskey)
        result[quatkey] = _quaternion(value[quatkey], quatkey)
        _require(APPROACH_PARAMETERS["workspace_min_palm_z_m"] <= result[poskey][2] <=
                 APPROACH_PARAMETERS["workspace_max_palm_z_m"], prefix + " palm outside fixed approach workspace")
    home, goal = result["home_joint_positions_rad"], result["pregrasp_joint_positions_rad"]
    _require(all(home[name] == goal[name] for name in ALL_JOINT_NAMES if name not in ARM_JOINT_NAMES),
             "approach may change only the seven right arm joints")
    _require(any(abs(home[name]-goal[name]) > 1e-10 for name in ARM_JOINT_NAMES),
             "home and pregrasp arm must differ for a home-to-grasp course")
    waypoints = value["waypoints"]
    _require(type(waypoints) is list and 1 <= len(waypoints) <= 16, "approach requires one to16 explicit waypoints")
    result["waypoints"] = []
    names = set()
    for waypoint in waypoints:
        _require(type(waypoint) is dict and set(waypoint) == WAYPOINT_KEYS, "waypoint requires exact keys")
        name = waypoint["name"]
        _require(type(name) is str and bool(name.strip()) and name not in names, "waypoint names must be unique nonempty strings")
        names.add(name)
        duration = _number(waypoint["duration_s"], "waypoint duration_s")
        _require(duration > 0, "waypoint duration must be positive")
        position = _vector(waypoint["palm_position_world_m"], 3, "waypoint palm position")
        _require(APPROACH_PARAMETERS["workspace_min_palm_z_m"] <= position[2] <=
                 APPROACH_PARAMETERS["workspace_max_palm_z_m"], "waypoint palm outside fixed approach workspace")
        result["waypoints"].append({"name": name, "arm_target_rad": _vector(waypoint["arm_target_rad"], 7, "waypoint arm"),
            "duration_s": duration, "palm_position_world_m": position,
            "palm_quaternion_xyzw": _quaternion(waypoint["palm_quaternion_xyzw"], "waypoint palm quaternion")})
    _require(sum(item["duration_s"] for item in result["waypoints"]) <= APPROACH_PARAMETERS["approach_timeout_s"],
             "nominal waypoint durations exceed the whole approach budget")
    last = result["waypoints"][-1]
    _require(_same_vector(last["arm_target_rad"], [goal[name] for name in ARM_JOINT_NAMES]) and
             _same_vector(last["palm_position_world_m"], result["pregrasp_palm_position_world_m"]) and
             _same_rotation(last["palm_quaternion_xyzw"], result["pregrasp_palm_quaternion_xyzw"]),
             "last waypoint must equal the pinned pregrasp arm and palm pose")
    return result


def load_approach_resource(path, expected_sha256, *, expected_candidate_sha256=None):
    """Verify exact file bytes before strict parsing; never rewrite a resource."""
    _sha(expected_sha256, "expected resource SHA256")
    _require(isinstance(path, (str, Path)) and bool(str(path)) and "\0" not in str(path), "invalid approach resource path")
    p = Path(path)
    _require(p.is_file() and not p.is_symlink(), "approach resource must be an existing regular nonsymlink file")
    before = p.stat()
    _require(0 < before.st_size <= 512*1024, "approach resource size outside bounded JSON limit")
    payload = p.read_bytes()
    after = p.stat()
    _require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "approach resource changed during read")
    _require(hashlib.sha256(payload).hexdigest() == expected_sha256, "approach resource bytes SHA256 mismatch")
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate approach JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("nonfinite approach JSON constant: " + value)
    value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=invalid_constant)
    return validate_approach_resource(value, expected_candidate_sha256=expected_candidate_sha256)


def approach_hand_posture_tracking(torch, *, original_tracking, admitted, approach_failed,
                                   fresh_valid, current_normal_available, physical_group_loaded,
                                   robot_cube_force_n, unsafe):
    """Permit only free-hand posture tracking during the unadmitted approach.

    The static grasp-palm reference is deliberately far from arm home. Its
    tracking gate must not remove the source hand's free-space posture spring.
    This separate hand-only gate is evaluated from the current native sample.
    Any current normal (including a positive-gap/speculative normal), physical
    loaded group, or nonzero robot/Cube force forbids the free-space override.
    After admission, return the original tracking value exactly. This function
    grants no arm tracking, physical handoff, or object-support authorization.
    """
    t = torch
    _require(t.is_tensor(original_tracking) and original_tracking.ndim == 1 and
             original_tracking.numel() > 0 and original_tracking.dtype == t.bool,
             "hand posture tracking requires per-environment original boolean tracking")
    n, device = original_tracking.shape[0], original_tracking.device
    for name, value in (("admitted", admitted), ("approach_failed", approach_failed),
                        ("fresh_valid", fresh_valid), ("unsafe", unsafe)):
        _require(t.is_tensor(value) and tuple(value.shape) == (n,) and value.dtype == t.bool and
                 value.device == device, "hand posture tracking boolean shape/device: " + name)
    for name, value in (("current_normal_available", current_normal_available),
                        ("physical_group_loaded", physical_group_loaded)):
        _require(t.is_tensor(value) and tuple(value.shape) == (n, 3) and value.dtype == t.bool and
                 value.device == device, "hand posture tracking group boolean shape/device: " + name)
    _require(t.is_tensor(robot_cube_force_n) and tuple(robot_cube_force_n.shape) == (n,) and
             robot_cube_force_n.dtype in (t.float32, t.float64) and robot_cube_force_n.device == device,
             "hand posture tracking needs current raw per-environment robot/Cube force")
    finite_nonnegative_force = t.isfinite(robot_cube_force_n) & (robot_cube_force_n >= 0)
    free_normal = ~current_normal_available.any(-1)
    free_physical = ~physical_group_loaded.any(-1)
    exactly_unloaded = finite_nonnegative_force & (robot_cube_force_n == 0)
    free_gate = (~admitted & fresh_valid & ~approach_failed & ~unsafe &
                 free_normal & free_physical & exactly_unloaded)
    mask = t.where(admitted, original_tracking, free_gate)
    return mask, {
        "version": "g1_approach_free_hand_posture_tracking_v1",
        "original_palm_tracking": original_tracking.clone(),
        "approach_admitted": admitted.clone(), "approach_failed": approach_failed.clone(),
        "native_sample_fresh_and_valid": fresh_valid.clone(), "unsafe": unsafe.clone(),
        "current_normal_available_by_group": current_normal_available.clone(),
        "physically_loaded_by_group": physical_group_loaded.clone(),
        "raw_robot_cube_force_n": robot_cube_force_n.clone(),
        "finite_nonnegative_raw_robot_cube_force": finite_nonnegative_force,
        "all_groups_without_current_normal": free_normal,
        "all_groups_without_physical_load": free_physical,
        "exactly_zero_robot_cube_load": exactly_unloaded,
        "free_approach_hand_posture_tracking_permitted": free_gate,
        "effective_hand_posture_tracking": mask.clone(),
        "post_admission_original_tracking_unchanged": True,
        "scope": "hand source-posture tracking only; not arm tracking or physical support",
        "object_support_torque_must_use_original_palm_tracking": True,
        "force_mode_position_springs_remain_owned_by_current_normal_controller": True,
    }


class HomeApproachPlanner:
    """Measured joint-waypoint reference generator; physical admission is external.

    State uses the existing diagnostic tensor names, plus env_origin_world_m,
    velocity_limits_arm_rad_s and approach_admitted. Actions are normalized arm7
    requests. The environment still applies its original anchor/mechanical/height
    guards and revalidates native2ms targets. These requests are not physical
    velocity guarantees or collision certificates.
    """

    def __init__(self, torch, resource, *, parameters=None):
        self.t = torch
        self.resource = validate_approach_resource(resource)
        self.parameters = _parameters(APPROACH_PARAMETERS if parameters is None else parameters)
        self._initialized = False

    def _tensor(self, state, name, shape, dtype=None):
        t = self.t
        value = state.get(name)
        _require(t.is_tensor(value) and tuple(value.shape) == shape and value.device == self.device,
                 "approach state tensor shape/device: " + name)
        if dtype is not None:
            _require(value.dtype == dtype, "approach state tensor dtype: " + name)
        return value

    def _initialize(self, q):
        t = self.t
        self.n, self.device, self.dtype = q.shape[0], q.device, q.dtype
        _require(self.dtype in (t.float32, t.float64), "approach requires float32/float64 measured state")
        def a(value):
            return t.as_tensor(value, dtype=self.dtype, device=self.device)
        self.home = a([self.resource["home_joint_positions_rad"][name] for name in ARM_JOINT_NAMES])
        self.hand = a(self.resource["source_hand_rad"])
        self.home_palm = a(self.resource["home_palm_position_world_m"])
        self.home_rotation = a(self.resource["home_palm_quaternion_xyzw"])
        self.goals = a([x["arm_target_rad"] for x in self.resource["waypoints"]])
        self.palm_goals = a([x["palm_position_world_m"] for x in self.resource["waypoints"]])
        self.rotation_goals = a([x["palm_quaternion_xyzw"] for x in self.resource["waypoints"]])
        self.durations = a([x["duration_s"] for x in self.resource["waypoints"]])
        self.ids = t.full((self.n,), -1, device=self.device, dtype=t.int64)
        self.last_controls = self.ids.clone()
        self.stage = t.zeros_like(self.ids)
        self.elapsed = t.zeros(self.n, device=self.device, dtype=self.dtype)
        self.reference = self.home[None].expand(self.n, -1).clone()
        self.start = self.reference.clone()
        self.failed = t.zeros(self.n, device=self.device, dtype=t.bool)
        self.failure_code = t.zeros_like(self.ids)
        self.first_failure_control = t.full_like(self.ids, -1)
        self.admitted = t.zeros_like(self.failed)
        self.admission_control = t.full_like(self.ids, -1)
        self.episode_start_native = t.full_like(self.ids, -1)
        self.home_joint_error = self.elapsed.clone()
        self.home_palm_error = self.elapsed.clone()
        self.home_rotation_error = self.elapsed.clone()
        self.home_hand_error = self.elapsed.clone()
        self.last_native = None
        self._initialized = True

    def _rotation_error(self, reference, actual):
        t = self.t
        dot = (reference*actual).sum(-1).abs().clamp(0, 1)
        return 2*t.acos(dot)

    def actions(self, state, dt):
        t, p = self.t, self.parameters
        _require(type(dt) in (int, float) and not isinstance(dt, bool) and math.isfinite(dt) and
                 dt == p["control_period_s"], "approach requires the declared66ms macro period")
        _require(type(state) is dict, "approach state must be a dictionary")
        q = state.get("q_action_rad")
        _require(t.is_tensor(q) and q.ndim == 2 and q.shape[1] == 14 and q.shape[0] > 0,
                 "approach requires measured arm7+hand7")
        if not self._initialized:
            self._initialize(q)
        q = self._tensor(state, "q_action_rad", (self.n, 14), self.dtype)
        qd = self._tensor(state, "qd_action_rad_s", (self.n, 14), self.dtype)
        target = self._tensor(state, "target_action_rad", (self.n, 14), self.dtype)
        per_step = self._tensor(state, "per_step_action_rad", (self.n, 14), self.dtype)
        limits = self._tensor(state, "joint_limits_action_rad", (self.n, 14, 2), self.dtype)
        origin = self._tensor(state, "env_origin_world_m", (self.n, 3), self.dtype)
        palm = self._tensor(state, "palm_position_world_m", (self.n, 3), self.dtype)
        rotation = self._tensor(state, "palm_quaternion_xyzw", (self.n, 4), self.dtype)
        unsafe = self._tensor(state, "unsafe_latched", (self.n,), t.bool)
        admitted = self._tensor(state, "approach_admitted", (self.n,), t.bool)
        ids = self._tensor(state, "episode_id", (self.n,))
        controls = self._tensor(state, "episode_control_steps", (self.n,))
        _require(ids.dtype in (t.int32, t.int64) and controls.dtype in (t.int32, t.int64), "approach episode clocks must be integers")
        native = state.get("native_physics_step_count")
        _require(type(native) is int and native >= 0, "approach requires current native integer clock")
        velocity = state.get("velocity_limits_arm_rad_s")
        _require(t.is_tensor(velocity) and velocity.device == self.device and velocity.dtype == self.dtype,
                 "approach requires native arm velocity limits")
        if tuple(velocity.shape) == (7,):
            velocity = velocity[None].expand(self.n, -1)
        _require(tuple(velocity.shape) == (self.n, 7), "approach native arm velocity-limit shape")
        ids, controls = ids.to(t.int64), controls.to(t.int64)
        changed = ids != self.ids
        clock_advanced = self.last_native is None or native == self.last_native+33
        new_epoch = changed & (ids > self.ids) & (controls == 0) & clock_advanced
        continuing = ~changed & (controls == self.last_controls+1) & clock_advanced
        fresh = (new_epoch | continuing) & (ids >= 0) & (controls >= 0)
        # Only an authoritative real reset epoch may clear local planner state.
        self.stage[new_epoch] = 0
        self.elapsed[new_epoch] = 0
        self.reference[new_epoch] = self.home
        self.start[new_epoch] = self.home
        self.failed[new_epoch] = False
        self.failure_code[new_epoch] = 0
        self.first_failure_control[new_epoch] = -1
        self.admitted[new_epoch] = False
        self.admission_control[new_epoch] = -1
        self.episode_start_native[new_epoch] = native
        home_pos = self.home_palm[None]+origin
        home_joint = (q[:, :7]-self.home).abs().amax(-1)
        home_hand = (q[:, 7:]-self.hand).abs().amax(-1)
        home_palm = t.linalg.vector_norm(palm-home_pos, dim=-1)
        home_rotation = self._rotation_error(self.home_rotation[None], rotation)
        for current, measured in ((self.home_joint_error, home_joint), (self.home_palm_error, home_palm),
                                  (self.home_rotation_error, home_rotation), (self.home_hand_error, home_hand)):
            current[new_epoch] = measured[new_epoch]
        tensors = (q, qd, target, per_step, limits, origin, palm, rotation, velocity)
        finite = t.stack([t.isfinite(value).reshape(self.n, -1).all(-1) for value in tensors]).all(0)
        valid = (finite & (per_step > 0).all(-1) & (velocity > 0).all(-1) &
                 (limits[..., 0] <= limits[..., 1]).all(-1) &
                 ((rotation.square().sum(-1)-1).abs() <= 1e-5))
        workspace = ((palm[:, 2]-origin[:, 2] >= p["workspace_min_palm_z_m"]) &
                     (palm[:, 2]-origin[:, 2] <= p["workspace_max_palm_z_m"]))
        home_matches = ((home_joint <= p["joint_arrival_tolerance_rad"]) &
                        (home_hand <= p["hand_seed_arrival_tolerance_rad"]) &
                        (home_palm <= p["palm_position_arrival_tolerance_m"]) &
                        (home_rotation <= p["palm_rotation_arrival_tolerance_rad"]))
        low, high = limits[:, :7, 0], limits[:, :7, 1]
        all_targets = t.cat((self.home[None], self.goals), 0)
        targets_in_limits = ((all_targets[None] >= low[:, None]) & (all_targets[None] <= high[:, None])).all((1, 2))
        targets_in_limits &= ((target[:, :7] >= low) & (target[:, :7] <= high)).all(-1)
        episode_elapsed = controls.to(self.dtype)*dt
        pre_admitted = self.admitted.clone()
        newly_admitted = admitted & ~self.admitted
        self.admission_control[newly_admitted] = controls[newly_admitted]
        self.admitted |= admitted
        timeout = (episode_elapsed >= p["approach_timeout_s"]) & ~self.admitted
        active_history = ~pre_admitted
        code = ((~valid).to(t.int64)*FAILURE_CODES["nonfinite_measurement"] |
                (new_epoch & ~home_matches).to(t.int64)*FAILURE_CODES["home_reset_mismatch"] |
                timeout.to(t.int64)*FAILURE_CODES["approach_timeout"] |
                unsafe.to(t.int64)*FAILURE_CODES["environment_unsafe"] |
                (~fresh).to(t.int64)*FAILURE_CODES["stale_or_invalid_episode_clock"] |
                (~targets_in_limits).to(t.int64)*FAILURE_CODES["declared_target_outside_native_limits"] |
                (~workspace).to(t.int64)*FAILURE_CODES["palm_workspace_violation"])
        code = t.where(active_history, code, t.zeros_like(code))
        first = (code != 0) & ~self.failed
        self.first_failure_control[first] = controls[first]
        self.failure_code |= code
        self.failed |= code != 0
        old_stage = self.stage.clone()
        goal = self.goals[old_stage]
        before_elapsed = self.elapsed.clone()
        before_reference = self.reference.clone()
        tracking_error = (q[:, :7]-before_reference).abs().amax(-1)
        hand_error = (q[:, 7:]-self.hand).abs().amax(-1)
        tracking = ((tracking_error <= p["joint_arrival_tolerance_rad"]) &
                    (hand_error <= p["hand_seed_arrival_tolerance_rad"]))
        progress = valid & fresh & workspace & tracking & ~self.failed & ~self.admitted
        self.elapsed += t.where(progress, t.full_like(self.elapsed, dt), t.zeros_like(self.elapsed))
        fraction = (self.elapsed/self.durations[old_stage]).clamp(0, 1)
        blend = fraction.pow(3)*(10-15*fraction+6*fraction.square())
        nominal = self.start+(goal-self.start)*blend[:, None]
        maximum_delta = t.minimum(velocity*dt, per_step[:, :7])
        limited_reference = before_reference+(nominal-before_reference).clamp(min=-maximum_delta, max=maximum_delta)
        self.reference = t.where(progress[:, None], limited_reference, before_reference)
        desired = t.minimum(t.maximum(self.reference, target[:, :7]-maximum_delta), target[:, :7]+maximum_delta)
        desired = t.minimum(t.maximum(desired, low), high)
        request = ((desired-target[:, :7])/per_step[:, :7]).clamp(-1, 1)
        request = t.where((self.failed | self.admitted)[:, None], t.zeros_like(request), request)
        joint_error = (q[:, :7]-goal).abs().amax(-1)
        joint_speed = qd[:, :7].abs().amax(-1)
        palm_goal = self.palm_goals[old_stage]+origin
        palm_error = t.linalg.vector_norm(palm-palm_goal, dim=-1)
        rotation_error = self._rotation_error(self.rotation_goals[old_stage], rotation)
        joint_arrived = ((joint_error <= p["joint_arrival_tolerance_rad"]) &
                         (joint_speed <= p["joint_velocity_arrival_tolerance_rad_s"]))
        final = old_stage == len(self.goals)-1
        final_pose_arrived = ((palm_error <= p["palm_position_arrival_tolerance_m"]) &
                              (rotation_error <= p["palm_rotation_arrival_tolerance_rad"]) &
                              (hand_error <= p["hand_seed_arrival_tolerance_rad"]))
        arrived = (fraction >= 1) & joint_arrived & progress
        kinematic_arrived = arrived & final & final_pose_arrived
        advance = arrived & ~final
        self.start[advance] = goal[advance]
        self.stage[advance] += 1
        self.elapsed[advance] = 0
        # A malformed/decreasing epoch cannot lower the last accepted epoch and
        # then masquerade as a later real reset to clear an existing failure.
        self.ids = t.where(new_epoch, ids, self.ids)
        self.last_controls = controls.clone()
        self.last_native = native
        approach_elapsed = t.where(self.admitted, self.admission_control.clamp_min(0).to(self.dtype)*dt, episode_elapsed)
        evidence = {
            "version": APPROACH_VERSION, "parameters": deepcopy(p), "episode_id": ids.clone(),
            "episode_control_steps": controls.clone(), "native_physics_step_count": native,
            "episode_reset_native_physics_step": self.episode_start_native.clone(),
            "new_real_episode": new_epoch, "measurement_clock_fresh": fresh,
            "home_arm_error_max_rad": self.home_joint_error.clone(),
            "home_hand_error_max_rad": self.home_hand_error.clone(),
            "home_palm_position_error_m": self.home_palm_error.clone(),
            "home_palm_rotation_error_rad": self.home_rotation_error.clone(),
            "canonical_world_origin_from_state_m": origin.clone(),
            "waypoint_index": old_stage, "waypoint_names": [x["name"] for x in self.resource["waypoints"]],
            "waypoint_elapsed_before_request_s": before_elapsed, "waypoint_fraction": fraction,
            "waypoint_elapsed_after_request_s": self.elapsed.clone(),
            "waypoint_duration_s": self.durations[old_stage],
            "waypoint_advance_after_this_request": advance, "waypoint_index_after_request": self.stage.clone(),
            "quintic_unlimited_reference_arm_rad": nominal, "previous_planner_reference_arm_rad": before_reference,
            "slewed_planner_reference_arm_rad": self.reference.clone(), "requested_arm_target_rad": desired,
            "previous_native_macro_target_arm_rad": target[:, :7].clone(), "normalized_arm_request": request.clone(),
            "native_velocity_limits_arm_rad_s": velocity.clone(), "maximum_reference_increment_rad": maximum_delta,
            "tracking_error_max_rad": tracking_error, "tracking_paused": ~tracking,
            "current_waypoint_joint_error_max_rad": joint_error, "current_arm_velocity_max_rad_s": joint_speed,
            "current_waypoint_palm_position_error_m": palm_error,
            "current_waypoint_palm_rotation_error_rad": rotation_error, "current_hand_seed_error_max_rad": hand_error,
            "kinematic_arrived": kinematic_arrived, "physical_handoff_decided_here": False,
            "environment_measurement_admitted": admitted.clone(), "admission_latched": self.admitted.clone(),
            "admission_episode_control_step": self.admission_control.clone(),
            "episode_elapsed_s": episode_elapsed, "approach_elapsed_s": approach_elapsed,
            "approach_timeout": timeout, "finite_and_structurally_valid_measurement": valid,
            "palm_inside_declared_workspace": workspace, "declared_targets_inside_native_limits": targets_in_limits,
            "failed": self.failed.clone(), "failure_code": self.failure_code.clone(),
            "failure_codes": dict(FAILURE_CODES), "first_failure_episode_control_step": self.first_failure_control.clone(),
            "failure_resets_only_with_real_new_episode": True,
            "object_motion_and_25_native_sample_admission_owned_by_environment": True,
            "native_pose_velocity_failures_and_quota_not_mutated": True,
            "normalized_request_still_requires_environment_native_limits_and_collision_checks": True,
        }
        return request, evidence
