"""Privileged-state G1 reinforcement-learning curriculum in Isaac Lab 3.

Isaac, torch and USD are imported only by ``build_environment`` after AppLauncher.
This PhysX curriculum has its own physics identity. It never certifies a MuJoCo
regression layout, exports demonstrations, or interprets reward as task success.
"""

from __future__ import annotations

import hashlib
import copy
from collections import deque
import json
import math
from pathlib import Path

from .control import LOWER_BODY_JOINT_NAMES
from .contracts import canonical_sha256, read_joint_order
from .rl_protocol import RL_PHYSICS, RL_PHYSICS_FIXED_PALM_ADJACENCY, RL_PHYSICS_EFFORT_GRASP


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


class ContactQualifiedFullLift:
    """Per-physics-sample full-task gates, separate from the local milestone.

    Admission uses the original fixed grip reference and 100-sample support
    window. No previously failed force/domain sample can be cleared by a later
    good window. Once lifting starts, loss of any original support gate is a
    terminal failure, rather than a reason to keep pushing upward.
    """

    def __init__(self, torch, num_envs, device, declaration):
        from .rl_effort_protocol import validate_full_lift_protocol
        self.torch = torch
        self.declaration = validate_full_lift_protocol(declaration)
        if self.declaration is None:
            raise ValueError("Full-lift state requires an explicit declaration")
        self.local_support_qualified = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.main_lift_started = torch.zeros_like(self.local_support_qualified)
        self.main_lift_authorized = torch.zeros_like(self.local_support_qualified)
        self.main_lift_support_lost = torch.zeros_like(self.local_support_qualified)
        self.failed = torch.zeros_like(self.local_support_qualified)
        self.full_success = torch.zeros_like(self.local_support_qualified)
        self.whole_episode_force_clean = torch.ones_like(self.local_support_qualified)
        self.whole_episode_original_domain_clean = torch.ones_like(self.local_support_qualified)
        self.hold_samples = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.maximum_hold_samples = torch.zeros_like(self.hold_samples)
        self.physics_samples = torch.zeros_like(self.hold_samples)
        self.first_failure_sample = torch.zeros_like(self.hold_samples)
        self.first_failure_code = torch.zeros_like(self.hold_samples)

    def reset(self, env_ids):
        for name in ("local_support_qualified", "main_lift_started", "main_lift_authorized", "main_lift_support_lost",
                     "failed", "full_success", "hold_samples", "maximum_hold_samples", "physics_samples",
                     "first_failure_sample", "first_failure_code"):
            getattr(self, name)[env_ids] = 0
        self.whole_episode_force_clean[env_ids] = True
        self.whole_episode_original_domain_clean[env_ids] = True

    def update(self, *, measurement_valid, group_force_n, original_hand_domain,
               local_support_qualified, support_current, clearance_m, object_lift_m, unsafe):
        t, declaration = self.torch, self.declaration
        self.physics_samples += 1
        was_failed = self.failed.clone()
        force_clean = (measurement_valid & t.isfinite(group_force_n).all(-1) &
                       (group_force_n >= 0).all(-1) & (group_force_n <= 1.5).all(-1))
        domain_clean = measurement_valid & original_hand_domain
        self.whole_episode_force_clean &= force_clean
        self.whole_episode_original_domain_clean &= domain_clean
        self.local_support_qualified |= local_support_qualified
        height_safe = (t.isfinite(clearance_m) & t.isfinite(object_lift_m) &
                       (clearance_m <= declaration["actual_height_limit_m"]) &
                       (object_lift_m <= declaration["actual_height_limit_m"]))
        physical_safe = (support_current & force_clean & domain_clean & height_safe & ~unsafe)
        self.main_lift_support_lost |= self.main_lift_started & ~physical_safe
        # These histories can never become clean again in this episode. End
        # an inadmissible full-task attempt rather than waiting out its budget.
        self.failed |= (self.main_lift_support_lost | ~self.whole_episode_force_clean |
                        ~self.whole_episode_original_domain_clean | unsafe | ~height_safe)
        first = self.failed & ~was_failed
        # Bitset: 1 missing raw sample, 2 force, 4 original hand domain,
        # 8 loss of an original main-lift support gate, 16 height, 32 unsafe.
        code = ((~measurement_valid).long() + (measurement_valid & ~force_clean).long()*2 +
                (~original_hand_domain).long()*4 + self.main_lift_support_lost.long()*8 +
                (~height_safe).long()*16 + unsafe.long()*32)
        self.first_failure_sample[first] = self.physics_samples[first]
        self.first_failure_code[first] = code[first]
        admit = (local_support_qualified & physical_safe & self.whole_episode_force_clean &
                 self.whole_episode_original_domain_clean & ~self.failed)
        self.main_lift_started |= admit
        self.main_lift_authorized = self.main_lift_started & physical_safe & ~self.failed
        hold = (self.main_lift_authorized & (clearance_m >= declaration["target_corner_clearance_m"]) &
                (object_lift_m >= declaration["target_corner_clearance_m"]))
        self.hold_samples = t.where(hold, self.hold_samples+1, t.zeros_like(self.hold_samples))
        self.maximum_hold_samples = t.maximum(self.maximum_hold_samples, self.hold_samples)
        self.full_success = ((self.hold_samples >= declaration["hold_consecutive_samples"]) &
                             ~self.failed & self.whole_episode_force_clean &
                             self.whole_episode_original_domain_clean)
        return self.metrics()

    def metrics(self):
        return {name: getattr(self, name).clone() for name in (
            "local_support_qualified", "main_lift_started", "main_lift_authorized", "main_lift_support_lost", "failed",
            "full_success", "whole_episode_force_clean", "whole_episode_original_domain_clean",
            "hold_samples", "maximum_hold_samples", "physics_samples", "first_failure_sample", "first_failure_code")}


class ContactEquivalenceError(AssertionError):
    """A same-step native contact-view crosscheck failed; never unsupported."""


# This is a new model identity, not a claim that the original mesh is disjoint.
FIXED_PALM_FILTER_GRAPH_SHA256 = "1bc9a5d33856280a489137634f3ddcbb84196ef231b6dfa902acbd271eac600c"
FIXED_PALM_FILTER_CONTRACT_SHA256 = "04dd990d50f4632012408b13747313a5504614e47a13d05d9540c31ef66fb770"
FIXED_PALM_FILTER_ASSET_SHA256 = "d23b9bbc8b59cf20fa1983f55874f855a76e038b9e22eae39542e62be5b4ba3d"


def _fixed_palm_canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


def fixed_palm_adjacency_filter_contract(joint_topology=None):
    """Declare the pinned allowlist, or verify a real full official joint graph.

    None is for pure protocol comparison only. The native authoring function
    always supplies every real USD joint, including enabled/collision flags.
    Full endpoints must share their joint's robot root; leaf-name hashing never
    excuses a cross-environment endpoint or an extra/missing/disabled joint.
    """
    component = ["right_hand_camera_base_link", "right_hand_palm_link", "right_wrist_yaw_link"]
    neighbors = ["right_hand_index_0_link", "right_hand_middle_0_link",
                 "right_hand_thumb_0_link", "right_wrist_pitch_link"]
    if joint_topology is not None:
        if not isinstance(joint_topology, (tuple, list)) or len(joint_topology) != 55:
            raise RLEnvironmentError("fixed-palm filtering requires the complete 55-joint official graph")
        normalized, roots, names = [], set(), set()
        for row in joint_topology:
            if not isinstance(row, dict):
                raise RLEnvironmentError("joint topology rows must be objects")
            name, kind, path = row.get("name"), row.get("type"), row.get("path")
            if not isinstance(name, str) or not name or "/" in name or name in names:
                raise RLEnvironmentError("invalid or duplicate fixed-palm topology joint name")
            if kind not in ("PhysicsFixedJoint", "PhysicsRevoluteJoint"):
                raise RLEnvironmentError("unsupported official topology joint type")
            endpoints_all = row.get("body0", []) + row.get("body1", []) if (
                isinstance(row.get("body0"), list) and isinstance(row.get("body1"), list)) else []
            if not endpoints_all or not isinstance(endpoints_all[0], str) or not endpoints_all[0].startswith("/"):
                raise RLEnvironmentError("official joint requires an absolute top-level body endpoint")
            root = endpoints_all[0].rsplit("/", 1)[0]
            # The world-fixing root_joint is directly under Robot while the
            # other joints are under Robot/joints. Reference-prefix relocation
            # is allowed; endpoint root and joint basename remain exact.
            if (not root or ".." in root.split("/") or not isinstance(path, str)
                    or not path.startswith(root+"/") or not path.endswith("/"+name)
                    or ".." in path.split("/")):
                raise RLEnvironmentError("official joint must belong to its exact body robot root")
            roots.add(root)
            names.add(name)
            for attribute, expected in (("physics:jointEnabled", True), ("physics:collisionEnabled", False)):
                value = row.get(attribute)
                if isinstance(value, dict):
                    value = value.get("value")
                if type(value) is not bool or value is not expected:
                    raise RLEnvironmentError("official joint enabled/collision flag changed: "+name)
            item = {"name": name, "type": kind}
            for key in ("body0", "body1"):
                endpoints = row.get(key)
                if not isinstance(endpoints, (list, tuple)) or len(endpoints) > 1:
                    raise RLEnvironmentError("official joints require single body endpoints")
                if not endpoints:
                    item[key] = None
                    continue
                endpoint = endpoints[0]
                if not isinstance(endpoint, str) or not endpoint.startswith(root+"/"):
                    raise RLEnvironmentError("cross-root or cross-environment joint endpoint")
                leaf = endpoint[len(root)+1:]
                if not leaf or "/" in leaf or leaf in (".", ".."):
                    raise RLEnvironmentError("official joint endpoint is not an exact top-level body")
                item[key] = leaf
            normalized.append(item)
        normalized.sort(key=lambda item: item["name"])
        if len(roots) != 1 or _fixed_palm_canonical_hash(normalized) != FIXED_PALM_FILTER_GRAPH_SHA256:
            raise RLEnvironmentError("official full joint graph differs from the pinned fixed-palm topology")
        reached = {"right_wrist_yaw_link"}
        while True:
            expanded = reached | {other for row in normalized if row["type"] == "PhysicsFixedJoint"
                                  for own, other in ((row["body0"], row["body1"]), (row["body1"], row["body0"]))
                                  if own in reached and other is not None}
            if expanded == reached:
                break
            reached = expanded
        adjacent = {other for row in normalized if row["type"] == "PhysicsRevoluteJoint"
                    for own, other in ((row["body0"], row["body1"]), (row["body1"], row["body0"]))
                    if own in reached and other is not None and other not in reached}
        if sorted(reached) != component or sorted(adjacent) != neighbors:
            raise RLEnvironmentError("fixed-palm component or direct revolute adjacency differs")
    pairs = sorted({tuple(sorted((a, b))) for a in component for b in component+neighbors if a != b})
    contract = {"contract_version": "g1_right_wrist_fixed_component_adjacent_pairs_v1",
                "asset_sha256": FIXED_PALM_FILTER_ASSET_SHA256,
                "fixed_component": component, "revolute_adjacent_bodies": neighbors,
                "excluded_body_pairs": [list(pair) for pair in pairs]}
    if len(pairs) != 15 or _fixed_palm_canonical_hash(contract) != FIXED_PALM_FILTER_CONTRACT_SHA256:
        raise RLEnvironmentError("fixed-palm allowlist differs from its audited declaration")
    return {"contract": contract, "source_graph_sha256": FIXED_PALM_FILTER_GRAPH_SHA256,
            "sha256": FIXED_PALM_FILTER_CONTRACT_SHA256}


def fixed_palm_filter_relationships(num_envs):
    """Exact one-sided USD relationships; USD filtering itself is symmetric."""
    if type(num_envs) is not int or not 1 <= num_envs <= 512:
        raise RLEnvironmentError("fixed-palm filtering requires 1..512 environments")
    pairs = fixed_palm_adjacency_filter_contract()["contract"]["excluded_body_pairs"]
    expected = {}
    for env_id in range(num_envs):
        prefix = "/World/envs/env_"+str(env_id)+"/Robot/"
        for left, right in pairs:
            expected.setdefault(prefix+left, []).append(prefix+right)
    return {path: sorted(targets) for path, targets in sorted(expected.items())}


def validate_fixed_palm_filter_relationships(actual, num_envs):
    """Reject missing, extra, cross-env, Cube/Table, or duplicate filter targets."""
    expected = fixed_palm_filter_relationships(num_envs)
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise RLEnvironmentError("fixed-palm filtered relationship prim allowlist mismatch")
    normalized = {}
    for path, targets in actual.items():
        if (not isinstance(targets, (list, tuple)) or any(not isinstance(p, str) for p in targets)
                or len(set(targets)) != len(targets) or sorted(targets) != expected[path]):
            raise RLEnvironmentError("fixed-palm filtered relationship target allowlist mismatch: "+str(path))
        normalized[path] = sorted(targets)
    return {"available": True, "num_envs": num_envs, "excluded_pair_count_per_env": 15,
            "excluded_pair_count_total": 15*num_envs, "relationships": normalized,
            "relationships_sha256": _fixed_palm_canonical_hash(normalized)}


def readback_fixed_palm_adjacency_filters(stage, usd_physics, num_envs):
    """Read all env filter APIs and the unchanged global self-collision flags."""
    actual, self_collisions = {}, {}
    roots = {"/World/envs/env_"+str(i)+"/Robot": i for i in range(num_envs)}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith("/World/envs/"):
            continue
        if prim.HasAPI(usd_physics.FilteredPairsAPI):
            actual[path] = [str(target) for target in usd_physics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()]
        if prim.HasAPI(usd_physics.ArticulationRootAPI):
            root = next((root for root in roots if path == root or path.startswith(root+"/")), None)
            if root is not None:
                value = prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get()
                if type(value) is not bool or value is not True or root in self_collisions:
                    raise RLEnvironmentError("fixed-palm override must preserve one globally self-colliding articulation per env")
                self_collisions[root] = {"prim_path": path, "enabled_self_collisions": value}
    if set(self_collisions) != set(roots):
        raise RLEnvironmentError("fixed-palm override could not read every environment's self-collision setting")
    result = validate_fixed_palm_filter_relationships(actual, num_envs)
    result["global_self_collision_readback"] = self_collisions
    return result


def author_fixed_palm_adjacency_filters(stage, usd_physics, num_envs, asset_sha256):
    """Author the new model on the session stage after clone, before physics.

    Asset bytes, drive gains/limits, material and every other collision pair are
    untouched. Native report zeros for these pairs mean deliberately filtered,
    never proof that the authored meshes are disjoint or of a solved grasp.
    """
    if asset_sha256 != FIXED_PALM_FILTER_ASSET_SHA256:
        raise RLEnvironmentError("fixed-palm model override requires the pinned official USD asset")
    expected = fixed_palm_filter_relationships(num_envs)
    roots = {"/World/envs/env_"+str(i)+"/Robot": i for i in range(num_envs)}
    topology = {root: [] for root in roots}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith("/World/envs/") and prim.HasAPI(usd_physics.FilteredPairsAPI):
            raise RLEnvironmentError("fixed-palm override refuses unexpected preexisting filtered-pair APIs")
        if not prim.IsA(usd_physics.Joint):
            continue
        root = next((root for root in roots if path.startswith(root+"/")), None)
        if root is None:
            continue
        joint = usd_physics.Joint(prim)
        topology[root].append({"name": prim.GetName(), "type": prim.GetTypeName(), "path": path,
                               "body0": [str(p) for p in joint.GetBody0Rel().GetTargets()],
                               "body1": [str(p) for p in joint.GetBody1Rel().GetTargets()],
                               "physics:jointEnabled": joint.GetJointEnabledAttr().Get(),
                               "physics:collisionEnabled": joint.GetCollisionEnabledAttr().Get()})
    # Check every graph and endpoint before authoring even a single exclusion.
    for rows in topology.values():
        fixed_palm_adjacency_filter_contract(rows)
    for path, targets in expected.items():
        for body in [path]+targets:
            prim = stage.GetPrimAtPath(body)
            if not prim.IsValid() or not prim.HasAPI(usd_physics.RigidBodyAPI) or prim.IsInstanceProxy():
                raise RLEnvironmentError("fixed-palm filtered endpoint is not a writable exact rigid body: "+body)
    for path, targets in expected.items():
        api = usd_physics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(path))
        if not api or not api.CreateFilteredPairsRel().SetTargets(targets):
            raise RLEnvironmentError("could not author exact fixed-palm filtered relationship: "+path)
    result = readback_fixed_palm_adjacency_filters(stage, usd_physics, num_envs)
    result.update({"model_override_version": "g1_right_wrist_fixed_component_adjacent_pairs_v1",
                   "authoring_phase": "after_clone_before_physics_initialization",
                   "source_asset_bytes_unchanged": True,
                   "source_graph_verified_env_count": num_envs,
                   "source_graph_sha256": FIXED_PALM_FILTER_GRAPH_SHA256,
                   "interpretation": "explicit fixed-component adjacency collision exclusions; zero pair force is expected filtering, not disjoint geometry"})
    return result



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


def current_relative_contact_motion(torch, *, link_jacobian_world, link_position_world,
                                    joint_velocity_full43, point_world, inward_normal_world,
                                    control_point_valid, physical_point_loaded, point_weight,
                                    cube_com_position_world, cube_com_linear_velocity_world,
                                    cube_angular_velocity_world, include_cube_point_jacobian=False):
    """Current point approach speed, including every robot DoF and Cube twist.

    Positive normal velocity closes the gap; negative velocity separates.
    Existing normal-control points and force weights are retained exactly.
    A group with any unproved control or physical point is unavailable. The
    caller supplies same-sample tensors and separately checks epoch freshness.
    No point, normal, or velocity from a previous sample is retained here.
    """
    t = torch
    n = joint_velocity_full43.shape[0]
    shapes = ((link_jacobian_world, (n,7,6,43)), (link_position_world,(n,7,3)),
        (joint_velocity_full43,(n,43)), (point_world,(n,7,32,3)),
        (inward_normal_world,(n,7,32,3)), (point_weight,(n,7,32)),
        (cube_com_position_world,(n,3)), (cube_com_linear_velocity_world,(n,3)),
        (cube_angular_velocity_world,(n,3)))
    if any(not t.is_tensor(value) or tuple(value.shape)!=shape or
           value.dtype!=joint_velocity_full43.dtype or value.device!=joint_velocity_full43.device
           for value,shape in shapes):
        raise ValueError("Relative contact motion requires matching native full43 floating tensors")
    if not joint_velocity_full43.is_floating_point():
        raise ValueError("Relative contact motion requires floating native velocities")
    for mask in (control_point_valid, physical_point_loaded):
        if (not t.is_tensor(mask) or tuple(mask.shape)!=(n,7,32) or mask.dtype!=t.bool or
                mask.device!=joint_velocity_full43.device):
            raise ValueError("Relative contact point masks require matching N by7 by32 booleans")
    common_finite = (t.isfinite(joint_velocity_full43).all(-1) &
        t.isfinite(cube_com_position_world).all(-1) & t.isfinite(cube_com_linear_velocity_world).all(-1) &
        t.isfinite(cube_angular_velocity_world).all(-1))
    link_finite = t.isfinite(link_jacobian_world).all(dim=(-1,-2)) & t.isfinite(link_position_world).all(-1)
    normal_norm = t.linalg.vector_norm(inward_normal_world,dim=-1)
    point_finite = (t.isfinite(point_world).all(-1) & t.isfinite(inward_normal_world).all(-1) &
        t.isfinite(point_weight) & (point_weight>0) & ((normal_norm-1).abs()<=1e-3))
    proved = (control_point_valid & point_finite & link_finite[:,:,None] & common_finite[:,None,None])
    # Replace unavailable values before matrix products. Unknown data must
    # disable this optional proof, rather than polluting the original law.
    finite_or_zero = lambda value:t.where(t.isfinite(value),value,t.zeros_like(value))
    link_j = finite_or_zero(link_jacobian_world)
    qd = finite_or_zero(joint_velocity_full43)
    point = finite_or_zero(point_world)
    normal = finite_or_zero(inward_normal_world)
    offset = point-finite_or_zero(link_position_world)[:,:,None]
    link_twist = t.einsum("nbij,nj->nbi",link_j,qd)
    finger_velocity = (link_twist[:,:,None,:3] + t.linalg.cross(
        link_twist[:,:,None,3:].expand_as(offset),offset,dim=-1))
    cube_offset = point-finite_or_zero(cube_com_position_world)[:,None,None]
    cube_velocity = (finite_or_zero(cube_com_linear_velocity_world)[:,None,None] + t.linalg.cross(
        finite_or_zero(cube_angular_velocity_world)[:,None,None].expand_as(cube_offset),cube_offset,dim=-1))
    finger_inward = (normal*finger_velocity).sum(-1)
    cube_inward = (normal*cube_velocity).sum(-1)
    relative = (normal*(finger_velocity-cube_velocity)).sum(-1)
    group_summaries = []
    weighted_full43 = []
    weighted_cube6 = []
    for first,last in ((0,3),(3,5),(5,7)):
        usable = proved[:,first:last]
        original = control_point_valid[:,first:last]
        physical = physical_point_loaded[:,first:last]
        unknown_control = (original & ~usable).sum(dim=(1,2))
        unknown_physical = (physical & ~usable).sum(dim=(1,2))
        weight = t.where(usable,point_weight[:,first:last],t.zeros_like(point_weight[:,first:last]))
        total = weight.sum(dim=(1,2))
        normalized = weight/total.clamp_min(1e-12)[:,None,None]
        available = (total>0) & (unknown_control==0) & (unknown_physical==0)
        def weighted(value):
            result=(normalized*t.where(usable,value[:,first:last],t.zeros_like(value[:,first:last]))).sum(dim=(1,2))
            return t.where(available,result,t.zeros_like(result))
        local=relative[:,first:last]
        maximum=t.where(usable,local,t.full_like(local,float("-inf"))).amax(dim=(1,2))
        minimum=t.where(usable,local,t.full_like(local,float("inf"))).amin(dim=(1,2))
        group_summaries.append({"relative_velocity_current":available,
            "relative_inward_velocity_m_s":weighted(relative),
            "weighted_finger_inward_velocity_m_s":weighted(finger_inward),
            "weighted_cube_inward_velocity_m_s":weighted(cube_inward),
            "maximum_current_point_relative_inward_velocity_m_s":t.where(available,maximum,t.zeros_like(maximum)),
            "minimum_current_point_relative_inward_velocity_m_s":t.where(available,minimum,t.zeros_like(minimum)),
            "original_control_point_count":original.sum(dim=(1,2)),
            "proved_control_point_count":usable.sum(dim=(1,2)),
            "unproved_control_point_count":unknown_control,"unproved_physical_loaded_point_count":unknown_physical,
            "physical_loaded_point_count":physical.sum(dim=(1,2)),"original_force_weight_sum":total})
        # Raw-window evidence permits an independent full43 dot-product audit;
        # its rows use the same original point weights, never a new selection.
        point_j = (link_j[:,first:last,None,:3] + t.linalg.cross(
            link_j[:,first:last,None,3:].transpose(-1,-2).expand(-1,-1,32,-1,-1),
            offset[:,first:last,:,None].expand(-1,-1,-1,43,-1),dim=-1).transpose(-1,-2))
        normal_j=(normal[:,first:last,:,:,None]*point_j).sum(-2)
        weighted_full43.append((normalized[...,None]*t.where(usable[...,None],normal_j,
            t.zeros_like(normal_j))).sum(dim=(1,2)))
        if include_cube_point_jacobian:
            cube_j=t.cat((normal[:,first:last],t.linalg.cross(
                cube_offset[:,first:last],normal[:,first:last],dim=-1)),dim=-1)
            weighted_cube6.append((normalized[...,None]*t.where(usable[...,None],cube_j,
                t.zeros_like(cube_j))).sum(dim=(1,2)))
    summary={key:t.stack([row[key] for row in group_summaries],dim=1) for key in group_summaries[0]}
    if include_cube_point_jacobian:
        summary["cube_normal_point_jacobian"]=t.stack(weighted_cube6,dim=1)
    raw={"link_origin_jacobian_world_full43_m":link_jacobian_world.detach().clone(),
        "link_origin_position_world_m":link_position_world.detach().clone(),
        "joint_velocity_full43_rad_s":joint_velocity_full43.detach().clone(),
        "link_origin_twist_world":link_twist.detach().clone(),
        "group_weighted_normal_jacobian_full43_m":t.stack(weighted_full43,dim=1).detach().clone(),
        "cube_com_position_world_m":cube_com_position_world.detach().clone(),
        "cube_com_linear_velocity_world_m_s":cube_com_linear_velocity_world.detach().clone(),
        "cube_angular_velocity_world_rad_s":cube_angular_velocity_world.detach().clone()}
    return summary,raw


def _native_rotation_xyzw(torch,q):
    x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),dim=-1).reshape(q.shape[:-1]+(3,3))


def reconstruct_native_hand_inertia(torch, *, link_jacobian_world_full43, link_quaternion_xyzw,
                                    body_mass_kg, body_inertia_prim_column_major, body_com_local_pose,
                                    native_mass_matrix_full43, native_armature_full43,
                                    hand_body_ids, hand_joint_ids):
    """Classify the SDK hand block against independently read native rigid inertia.

    Other body hand-J columns must currently vanish. The selected native body
    parameters and COM offsets reconstruct rigid-body Mhh; actual armature is
    added exactly once only after raw SDK Mhh matches either rigid or dynamic.
    This does not claim arm/hand constrained contact or drive effective inertia.
    """
    t=torch;n,b=link_quaternion_xyzw.shape[:2];dtype=link_jacobian_world_full43.dtype
    device=link_jacobian_world_full43.device
    shapes=((link_jacobian_world_full43,(n,b,6,43)),(link_quaternion_xyzw,(n,b,4)),
        (body_mass_kg,(n,b)),(body_inertia_prim_column_major,(n,b,9)),(body_com_local_pose,(n,b,7)),
        (native_mass_matrix_full43,(n,43,43)),(native_armature_full43,(n,43)))
    if any(not t.is_tensor(v) or tuple(v.shape)!=shape or v.dtype!=dtype or v.device!=device
            or not v.is_floating_point() for v,shape in shapes):
        raise ValueError("Native hand inertia reconstruction requires matching current fixed-root body tensors")
    if len(hand_body_ids)!=7 or len(set(hand_body_ids))!=7 or len(hand_joint_ids)!=7 or len(set(hand_joint_ids))!=7:
        raise ValueError("Native hand inertia requires the seven explicitly bound bodies and joints")
    j=link_jacobian_world_full43[:,:,:,hand_joint_ids]
    other=[i for i in range(b) if i not in hand_body_ids]
    other_max=j[:,other].abs().amax(dim=(1,2,3)) if other else t.zeros(n,dtype=dtype,device=device)
    q=link_quaternion_xyzw[:,hand_body_ids];m=body_mass_kg[:,hand_body_ids]
    ip=body_inertia_prim_column_major[:,hand_body_ids].reshape(n,7,3,3).transpose(-1,-2)
    com=body_com_local_pose[:,hand_body_ids,:3];jh=j[:,hand_body_ids]
    armature=native_armature_full43[:,hand_joint_ids]
    raw=native_mass_matrix_full43[:,hand_joint_ids][:,:,hand_joint_ids]
    finite=t.stack([t.isfinite(v).reshape(n,-1).all(-1) for v in (j,q,m,ip,com,armature,raw)]).all(0)
    quat_valid=(t.linalg.vector_norm(q,dim=-1)-1).abs().amax(-1)<=1e-4
    safe=lambda v:t.where(t.isfinite(v),v,t.zeros_like(v))
    r=_native_rotation_xyzw(t,safe(q));offset=(r@safe(com)[...,None]).squeeze(-1)
    jw=safe(jh[:,:,3:]);jcom=safe(jh[:,:,:3])+t.linalg.cross(
        jw.transpose(-1,-2),offset[:,:,None].expand(-1,-1,7,-1),dim=-1).transpose(-1,-2)
    iw=r@safe(ip)@r.transpose(-1,-2)
    rigid=(safe(m)[:,:,None,None]*(jcom.transpose(-1,-2)@jcom)+jw.transpose(-1,-2)@iw@jw).sum(1)
    dynamic=rigid+t.diag_embed(safe(armature))
    rigid_error=(safe(raw)-rigid).abs().amax((-1,-2))
    dynamic_error=(safe(raw)-dynamic).abs().amax((-1,-2))
    rigid_match=rigid_error<=1e-8+1e-3*rigid.abs().amax((-1,-2))
    dynamic_match=dynamic_error<=1e-8+1e-3*dynamic.abs().amax((-1,-2))
    symmetry=(ip-ip.transpose(-1,-2)).abs().amax((1,2,3))<=1e-10+1e-4*ip.abs().amax((1,2,3))
    eye=t.eye(3,dtype=dtype,device=device)[None,None].expand(n,7,-1,-1)
    _,info=t.linalg.cholesky_ex(t.where((finite&symmetry)[:,None,None,None],iw,eye))
    valid=(finite&quat_valid&symmetry&(info==0).all(-1)&(m>0).all(-1)&
        (armature>=0).all(-1)&(other_max<=1e-6)&(rigid_match|dynamic_match))
    classification=t.zeros(n,dtype=t.int64,device=device)
    classification=t.where(valid&rigid_match,t.ones_like(classification),classification)
    classification=t.where(valid&dynamic_match&~rigid_match,t.full_like(classification,2),classification)
    classification=t.where(valid&dynamic_match&rigid_match,t.full_like(classification,3),classification)
    return dynamic,valid,{"raw_sdk_hand_mass_matrix":raw,"reconstructed_rigid_hand_mass_matrix":rigid,
        "reconstructed_dynamic_hand_mass_matrix":dynamic,"native_hand_armature":armature,
        "raw_sdk_minus_rigid_max_abs_kg_m2":rigid_error,"raw_sdk_minus_dynamic_max_abs_kg_m2":dynamic_error,
        "native_mass_classification_code":classification,
        "classification_codes":{"0":"unproved","1":"sdk_matches_rigid_armature_added_once",
            "2":"sdk_matches_dynamic_armature_not_added_twice","3":"both_matches_within_declared_tolerance"},
        "other_body_hand_jacobian_max_abs":other_max,"other_body_hand_jacobian_zero_tolerance":1e-6,
        "comparison_absolute_tolerance_kg_m2":1e-8,"comparison_relative_tolerance":1e-3,
        "hand_mass_semantics_current":valid,
        "mass_model_scope":"native_rigid_bodies_plus_readback_joint_armature_hand7_other_DOF_held_fixed_no_drive_or_contact_inertia"}


def native_relative_damping_state(torch, *, hand_mass_matrix, hand_armature,
                                  cube_mass_kg, cube_inertia_prim_column_major,
                                  cube_link_quaternion_xyzw, cube_normal_point_jacobian,
                                  physical_point_count, relative_velocity_current, hand_mass_semantics_current=None):
    """Prepare optional native inertia evidence; unknown numeric rows stay unknown.

    Native rigid-body inertia is at COM but expressed in the body prim frame,
    with column-major nine entries. Rotate by the actor/link world quaternion,
    never by the differently oriented principal-inertia COM quaternion.
    Native M's nonzero-armature semantics are not assumed here.
    """
    t=torch;n=hand_mass_matrix.shape[0];dtype=hand_mass_matrix.dtype;device=hand_mass_matrix.device
    shapes=((hand_mass_matrix,(n,7,7)),(hand_armature,(n,7)),(cube_mass_kg,(n,1)),
        (cube_inertia_prim_column_major,(n,9)),(cube_link_quaternion_xyzw,(n,4)),
        (cube_normal_point_jacobian,(n,3,6)))
    if any(not t.is_tensor(v) or tuple(v.shape)!=shape or v.dtype!=dtype or v.device!=device
           or not v.is_floating_point() for v,shape in shapes):
        raise ValueError("Native damping inertia requires matching floating tensors")
    if (physical_point_count.shape!=(n,3) or physical_point_count.dtype not in (t.int32,t.int64) or
            physical_point_count.device!=device or relative_velocity_current.shape!=(n,3) or
            relative_velocity_current.dtype!=t.bool or relative_velocity_current.device!=device):
        raise ValueError("Native damping counts/current proof require matching group tensors")
    finite=t.stack([t.isfinite(v).reshape(n,-1).all(-1) for v,_ in shapes]).all(0)
    safe=lambda v:t.where(t.isfinite(v),v,t.zeros_like(v))
    mass=safe(hand_mass_matrix);inertia=safe(cube_inertia_prim_column_major).reshape(n,3,3).transpose(-1,-2)
    eye3=t.eye(3,dtype=dtype,device=device)[None].expand(n,-1,-1)
    eye7=t.eye(7,dtype=dtype,device=device)[None].expand(n,-1,-1)
    symmetry=(mass-mass.transpose(-1,-2)).abs().amax((-1,-2))<=1e-7+1e-4*mass.abs().amax((-1,-2))
    symmetry &= (inertia-inertia.transpose(-1,-2)).abs().amax((-1,-2))<=1e-10+1e-4*inertia.abs().amax((-1,-2))
    q=safe(cube_link_quaternion_xyzw);norm=t.linalg.vector_norm(q,dim=-1)
    quaternion_valid=(norm-1).abs()<=1e-4
    q=t.where(quaternion_valid[:,None],q,t.tensor([0.,0.,0.,1.],dtype=dtype,device=device)[None])
    rotation=_native_rotation_xyzw(t,q)
    inertia_world=rotation@inertia@rotation.transpose(-1,-2)
    _,mass_info=t.linalg.cholesky_ex(t.where((finite&symmetry)[:,None,None],mass,eye7))
    chol,inertia_info=t.linalg.cholesky_ex(t.where((finite&symmetry)[:,None,None],inertia_world,eye3))
    # A zero armature is unambiguous; otherwise require the independent native
    # rigid-body reconstruction/SDK comparison supplied by the caller.
    armature_known=(hand_armature==0).all(-1) if hand_mass_semantics_current is None else hand_mass_semantics_current
    if (armature_known.shape!=(n,) or armature_known.dtype!=t.bool or armature_known.device!=device):
        raise ValueError("Native hand mass semantics require matching per-environment proof")
    valid=(finite&symmetry&quaternion_valid&(mass_info==0)&(inertia_info==0)&
        (cube_mass_kg[:,0]>0)&armature_known)
    inverse=t.zeros(n,6,6,dtype=dtype,device=device)
    inverse[:,:3,:3]=eye3/cube_mass_kg[:,0].clamp_min(t.finfo(dtype).tiny)[:,None,None]
    inverse[:,3:,3:]=t.cholesky_inverse(t.where((inertia_info==0)[:,None,None],chol,eye3))
    inverse=t.where(valid[:,None,None],inverse,t.zeros_like(inverse))
    current=relative_velocity_current & valid[:,None] & (physical_point_count>=0)
    state={"hand_mass_matrix":t.where(valid[:,None,None],mass,t.zeros_like(mass)),
        "cube_inverse_spatial_inertia_world":inverse,
        "cube_normal_point_jacobian":safe(cube_normal_point_jacobian),
        "physical_point_count":physical_point_count,"current":current}
    isotropic=(inertia-(inertia.diagonal(dim1=-2,dim2=-1).mean(-1)[:,None,None]*eye3)).abs().amax((-1,-2))
    evidence={"hand_mass_matrix_from_readback_and_reconstruction":hand_mass_matrix,"native_hand_armature":hand_armature,
        "armature_semantics_current":armature_known,"cube_mass_readback_kg":cube_mass_kg,
        "cube_inertia_prim_column_major_readback":cube_inertia_prim_column_major,
        "cube_link_quaternion_xyzw":cube_link_quaternion_xyzw,"cube_inertia_world_kg_m2":inertia_world,
        "inertia_numeric_valid":valid,"mass_spd_info":mass_info,"cube_inertia_spd_info":inertia_info,
        "cube_inertia_isotropic_readback":isotropic<=1e-10+1e-4*inertia.abs().amax((-1,-2)),
        "cube_inertia_frame":"native_COM_inertia_expressed_in_body_prim_frame_rotated_by_link_quaternion",
        "raw_native_mass_armature_not_silently_added_or_assumed_included":True}
    return state,evidence


def revalidated_native_arm_substep(torch, *, segment_origin, macro_endpoint, previous_written,
                                  measured_arm, native_index, substeps, physics_dt_s,
                                  velocity_limits_rad_s, anchor_delta_rad, joint_lower, joint_upper,
                                  point_jz, palm_z, start_palm_z, height_cap_m, enabled, sample_fresh):
    """Linear native target schedule with fresh box/height revalidation.

    Only enabled main-lift environments use this schedule. Others retain the
    original macro hold. A failure returns the measured pose as an explicit
    unsafe override; it never labels that emergency hold velocity-certified.
    """
    from .rl_probe_control import apply_probe_guard
    t=torch; n=macro_endpoint.shape[0]
    matrices=(segment_origin,macro_endpoint,previous_written,measured_arm,joint_lower,joint_upper,point_jz)
    if (type(substeps) is not int or substeps!=33 or not math.isclose(physics_dt_s,.002,abs_tol=1e-12,rel_tol=0) or
            any(tuple(value.shape)!=(n,7) for value in matrices) or
            any(value.dtype!=macro_endpoint.dtype or value.device!=macro_endpoint.device for value in matrices)):
        raise ValueError("Native arm interpolation requires matching seven-joint tensors and33 actual2ms substeps")
    if isinstance(anchor_delta_rad,bool) or not math.isfinite(anchor_delta_rad) or anchor_delta_rad<=0:
        raise ValueError("Native arm interpolation requires the original finite positive anchor delta")
    for height in (palm_z,start_palm_z):
        if (not t.is_tensor(height) or tuple(height.shape)!=(n,) or height.dtype!=macro_endpoint.dtype or
                height.device!=macro_endpoint.device):
            raise ValueError("Native arm interpolation requires matching current palm-height tensors")
    for mask in (enabled,sample_fresh):
        if tuple(mask.shape)!=(n,) or mask.dtype!=t.bool or mask.device!=macro_endpoint.device:
            raise ValueError("Native arm interpolation requires matching episode/freshness booleans")
    if tuple(native_index.shape)!=(n,) or native_index.dtype!=t.int64 or native_index.device!=macro_endpoint.device:
        raise ValueError("Native arm interpolation requires per-environment native indices")
    limits=t.as_tensor(velocity_limits_rad_s,dtype=macro_endpoint.dtype,device=macro_endpoint.device)
    if limits.shape==(7,):limits=limits[None].expand(n,-1)
    if limits.shape!=(n,7) or not bool(t.isfinite(limits).all()) or bool((limits<=0).any()):
        raise ValueError("Native arm interpolation requires original positive finite velocity limits")
    index_valid=(native_index>=1)&(native_index<=substeps)
    finite=t.stack([t.isfinite(value).all(-1) for value in matrices]).all(0)
    cap=t.as_tensor(height_cap_m,dtype=macro_endpoint.dtype,device=macro_endpoint.device)
    if cap.ndim==0:cap=cap.expand(n)
    if cap.shape!=(n,):raise ValueError("Native arm interpolation requires scalar or N-element height caps")
    fresh=(sample_fresh & index_valid & finite & t.isfinite(palm_z) & t.isfinite(start_palm_z) &
           t.isfinite(cap) & (cap>=0))
    fraction=(native_index.to(macro_endpoint.dtype)/substeps).clamp(0,1)
    nominal=segment_origin+(macro_endpoint-segment_origin)*fraction[:,None]
    output=macro_endpoint.clone()
    low=t.maximum(t.maximum(previous_written-limits*physics_dt_s,measured_arm-anchor_delta_rad),joint_lower)
    high=t.minimum(t.minimum(previous_written+limits*physics_dt_s,measured_arm+anchor_delta_rad),joint_upper)
    requires_unsafe=enabled & ~fresh
    active=enabled & fresh
    selected=t.nonzero(active,as_tuple=False).flatten()
    guard_evidence=None
    if selected.numel():
        target,guard_evidence=apply_probe_guard(t,nominal[selected],previous_written[selected],measured_arm[selected],
            low[selected],high[selected],point_jz[selected],palm_z[selected],start_palm_z[selected],
            cap[selected],mode="box_halfspace_projected_v2")
        output[selected]=target
        requires_unsafe[selected] |= guard_evidence["requires_unsafe"]
    failure_hold=t.where(t.isfinite(measured_arm),measured_arm,previous_written)
    output=t.where(requires_unsafe[:,None],failure_hold,output)
    delta=output-previous_written
    nominal_delta=nominal-previous_written
    denominator=t.linalg.vector_norm(nominal_delta,dim=-1)
    ratio=t.where(denominator>1e-12,t.linalg.vector_norm(delta,dim=-1)/denominator.clamp_min(1e-12),
                  t.zeros_like(denominator))
    return output,{"schedule":"native_linear33_revalidated_v1","interpolation_enabled":enabled,
        "sample_fresh":sample_fresh,"native_index":native_index.clone(),"fraction":fraction,
        "segment_origin_arm_rad":segment_origin,"macro_endpoint_arm_rad":macro_endpoint,
        "previous_actually_written_arm_target_rad":previous_written,
        "current_measured_arm_rad":measured_arm,
        "current_point_jz_m_per_rad":point_jz,
        "current_palm_z_m":palm_z,"start_palm_z_m":start_palm_z,"height_cap_m":cap,
        "nominal_interpolated_arm_target_rad":nominal,"revalidated_arm_target_rad":output,
        "nominal_increment_rad":nominal_delta,"written_increment_rad":delta,
        "written_increment_rad_s":delta/physics_dt_s,
        "written_target_minus_measured_arm_rad":output-measured_arm,
        "interpolation_correction_rad":output-nominal,
        "written_delta_norm_over_nominal_delta_norm":ratio,
        "original_velocity_limits_rad_s":limits,"native_lower_bound_rad":low,"native_upper_bound_rad":high,
        "mechanical_lower_bound_rad":joint_lower,"mechanical_upper_bound_rad":joint_upper,
        "requires_unsafe":requires_unsafe,"failure_override_is_not_velocity_certified":requires_unsafe,
        "projected_environment_indices":selected,"native_height_guard":guard_evidence,
        "target_is_not_measured_joint_motion_or_object_velocity_guarantee":True}


class NativeJointLoadDiagnostics:
    """Read-only native load telemetry, separate from support or motor torque.

    Construct only for control_probe. No sensor, actuator or stage is changed.
    Getter failures retain None plus a reason; reset-stale rows are excluded.
    """

    PROTOCOL = "native_projected_incoming_joint_load_control_boundary_v1"

    def __init__(self, torch, as_tensor, view, joint_names, body_names, num_envs):
        self.torch, self.as_tensor, self.view = torch, as_tensor, view
        self.num_envs = num_envs
        self.joint_names, self.body_names = list(joint_names), list(body_names)
        self.binding_error = None
        try:
            native_joints = list(view.shared_metatype.dof_names)
            native_bodies = list(view.shared_metatype.link_names)
            if (view.count != num_envs or native_joints != self.joint_names or
                    native_bodies != self.body_names):
                raise ValueError("native articulation count/name order differs from robot binding")
            binding = {"verified": True, "joint_names": native_joints,
                       "body_names": native_bodies, "native_count": int(view.count)}
        except Exception as exc:
            self.binding_error = type(exc).__name__ + ": " + str(exc)
            binding = {"verified": False, "unavailable_reason": self.binding_error,
                       "expected_joint_names": self.joint_names, "expected_body_names": self.body_names}
        self.binding = binding
        self.readback = _probe_payload(torch, {
            "protocol": self.PROTOCOL, "binding": binding,
            "native_drive_max_force_nm": self._read("get_dof_max_forces", (num_envs, len(joint_names))),
            "native_friction_properties": self._read("get_dof_friction_properties", (num_envs, len(joint_names), 3)),
            "friction_last_dimension": ["static_friction_effort_nm", "dynamic_friction_effort_nm",
                                        "viscous_friction_coefficient_nm_s_rad"],
            "readback_count": 1,
            "semantics": "native property readback, not measured drive torque; all43 DOFs are revolute"})

    def _read(self, method, expected_shape, env_ids=None):
        try:
            if self.binding_error is not None:
                raise ValueError(self.binding_error)
            value = self.as_tensor(getattr(self.view, method)())
            if tuple(value.shape) != expected_shape:
                raise ValueError(method + " unexpected native shape: " + str(tuple(value.shape)))
            if env_ids is not None:
                value = value.index_select(0, env_ids.to(value.device))
            if not value.is_floating_point() or not bool(self.torch.isfinite(value).all()):
                raise ValueError(method + " nonfinite or nonfloating native values")
            return {"available": True, "values": value.detach().clone(), "unavailable_reason": None}
        except Exception as exc:
            return {"available": False, "values": None,
                    "unavailable_reason": type(exc).__name__ + ": " + str(exc)}

    def snapshot(self, native_step, episode_ids, episode_controls, *, phase):
        # A reset writes state without solving physics. Never pair the previous
        # episode's native load with the newly reset q at the same native step.
        eligible = self.torch.nonzero(episode_controls > 0, as_tuple=False).flatten()
        invalid = self.torch.nonzero(episode_controls <= 0, as_tuple=False).flatten()
        result = {"protocol": self.PROTOCOL, "native_physics_step_count": int(native_step),
                  "sample_phase": phase, "env_ids": eligible.detach().cpu().tolist(),
                  "episode_ids": episode_ids.index_select(0, eligible).detach().clone(),
                  "invalid_for_current_episode_env_ids": invalid.detach().cpu().tolist(),
                  "reset_stale_reason": "no completed physics/control transition since episode reset",
                  "binding_names_reference": "control_profile_boundary.native_joint_load_readback.binding",
                  "projected_joint_force_semantics": "native incoming wrench projected along DOF motion; Nm for revolute DOFs; not isolated drive torque or isolated constraint torque",
                  "incoming_joint_wrench_semantics": "native total active+passive incoming joint wrench, child joint frame; body_names order; first3 N, last3 Nm",
                  "measurement_cadence": "one sample per requested control boundary; does not certify every2ms transient"}
        if eligible.numel():
            projected = self._read("get_dof_projected_joint_forces",
                                   (self.num_envs, len(self.joint_names)), eligible)
            incoming = self._read("get_link_incoming_joint_force",
                                  (self.num_envs, len(self.body_names), 6), eligible)
        else:
            projected = {"available": False, "values": None,
                         "unavailable_reason": "no environment has post-reset physics in this boundary"}
            incoming = dict(projected)
        result.update(projected_joint_force_nm=projected, incoming_joint_wrench_child=incoming,
                      available=projected["available"] and incoming["available"])
        return result


class NativeSelfPairDiagnostics:
    """One read-only batched view for declared thumb0/1-camera pairs only.

    API/report-threshold preconditions are observed, never activated or set.
    Normal data cannot identify tangential friction or certify load support.
    """

    PROTOCOL = "native_thumb01_camera_self_pair_control_boundary_v1"

    def __init__(self, torch, as_tensor, simulation_view, stage, pairs, num_envs, physics_dt):
        self.torch, self.as_tensor = torch, as_tensor
        self.num_envs, self.dt = num_envs, float(physics_dt)
        self.view, self.binding_error = None, None
        self._cache_key, self._cache = None, None
        self._last_counted_step = None
        self._positive_post_control_samples = [0]*len(pairs)
        self.pairs = [dict(row, sensor_index=i) for i, row in enumerate(pairs)]
        self.readback = {"protocol": self.PROTOCOL, "available": False,
                         "pairs": self.pairs, "preexisting_contact_reports": [],
                         "raw_point_capacity_global": 32*len(pairs),
                         "friction_measured": False, "physics_or_reporting_writes": 0}
        try:
            if len(pairs) != 2*num_envs or not math.isclose(self.dt, .002, abs_tol=1e-12, rel_tol=0):
                raise ValueError("self-pair view requires2 thumb bodies per env and actual2ms dt")
            sensors = [row["sensor_path"] for row in pairs]
            filters = [[row["filter_path"]] for row in pairs]
            if len(set(sensors)) != len(sensors):
                raise ValueError("self-pair sensor paths must be unique")
            for env_id in range(num_envs):
                rows = [row for row in pairs if row["env_id"] == env_id]
                if {row["sensor_body"] for row in rows} != {"right_hand_thumb_0_link", "right_hand_thumb_1_link"} or len(rows) != 2:
                    raise ValueError("missing or duplicate declared thumb pair in environment")
                for row in rows:
                    if (row["sensor_path"].rsplit("/", 1)[-1] != row["sensor_body"] or
                            row["filter_path"].rsplit("/", 1)[-1] != "right_hand_camera_base_link" or
                            row["sensor_path"].rsplit("/", 1)[0] != row["filter_path"].rsplit("/", 1)[0] or
                            ("/env_"+str(env_id)+"/") not in row["sensor_path"]):
                        raise ValueError("self-pair paths must bind named bodies within the same environment")
            for path in dict.fromkeys(sensors+[row[0] for row in filters]):
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid() or "PhysicsRigidBodyAPI" not in prim.GetAppliedSchemas():
                    raise ValueError("missing declared rigid body: "+path)
                if "PhysxContactReportAPI" not in prim.GetAppliedSchemas():
                    raise ValueError("preexisting contact reporter absent: "+path)
                attr = prim.GetAttribute("physxContactReport:threshold")
                threshold = attr.Get() if attr.IsValid() else None
                finite_threshold = threshold is not None and math.isfinite(float(threshold))
                self.readback["preexisting_contact_reports"].append({"path": path,
                    "threshold_n": float(threshold) if finite_threshold else None,
                    "threshold_valid": finite_threshold})
                if not finite_threshold or float(threshold) != 0.0:
                    raise ValueError("preexisting report threshold cannot certify weak contact: "+path)
            view = simulation_view.create_rigid_contact_view(
                sensors, filter_patterns=filters, max_contact_data_count=32*len(pairs))
            actual_sensors, actual_filters = list(view.sensor_paths), list(view.filter_paths)
            self.readback.update(native_sensor_paths=actual_sensors, native_filter_paths=actual_filters,
                                 native_sensor_count=int(view.sensor_count), native_filter_count=int(view.filter_count))
            # Installed API exposes routing paths; unsupported layouts are
            # explicitly unavailable rather than guessing a cross-env pairing.
            normalized_filters = [[p] if isinstance(p, str) else list(p) for p in actual_filters]
            if (view.sensor_count != len(pairs) or view.filter_count != 1 or
                    actual_sensors != sensors or normalized_filters != filters):
                raise ValueError("native self-pair count/path/filter routing mismatch")
            self.view = view
            self.readback.update(available=True, unavailable_reason=None)
        except Exception as exc:
            self.binding_error = type(exc).__name__+": "+str(exc)
            self.readback.update(available=False, unavailable_reason=self.binding_error)

    def summary(self):
        seen = [i for i, count in enumerate(self._positive_post_control_samples) if count]
        return {**self.readback, "positive_witness_post_control_samples": list(self._positive_post_control_samples),
                "positive_witness_seen_sensor_indices": seen,
                "self_pair_reporting_status": "positive_witness_observed" if seen else "no_positive_witness_yet",
                "absence_of_selfcollision_certified": False,
                "cadence": "control boundaries only; not every2ms transient"}

    def snapshot(self, native_step, episode_ids, episode_controls, *, phase):
        torch = self.torch
        ids, controls = episode_ids.detach().cpu().tolist(), episode_controls.detach().cpu().tolist()
        key = (int(native_step), tuple(ids), tuple(controls))
        if key != self._cache_key:
            valid = [i for i, row in enumerate(self.pairs) if controls[row["env_id"]] > 0]
            result = {"protocol": self.PROTOCOL, "native_physics_step_count": int(native_step),
                      "sensor_indices": valid, "episode_ids": [ids[self.pairs[i]["env_id"]] for i in valid],
                      "reset_stale_excluded_sensor_indices": [i for i in range(len(self.pairs)) if i not in valid],
                      "binding_reference": "control_profile_boundary.native_self_pair_readback.pairs",
                      "available": False, "values": None, "positive_witness_sensor_indices": [],
                      "measurement_dt_s": self.dt, "friction_measured": False}
            try:
                if self.view is None:
                    raise ValueError(self.binding_error or "native pair view unavailable")
                if not valid:
                    raise ValueError("no post-reset physics for any declared pair")
                net = self.as_tensor(self.view.get_net_contact_forces(dt=self.dt)).detach().clone()
                matrix = self.as_tensor(self.view.get_contact_force_matrix(dt=self.dt)).detach().clone()
                raw = tuple(self.as_tensor(x).detach().clone() for x in self.view.get_contact_data(dt=self.dt))
                if len(raw) != 6:
                    raise ValueError("native contact data must return forces,points,normals,separation,COUNTS,STARTS")
                force, point, normal, separation, counts, starts = raw
                sensors, capacity = len(self.pairs), int(self.view.max_contact_data_count)
                force, separation = force.reshape(-1), separation.reshape(-1)
                if (tuple(net.shape) != (sensors, 3) or tuple(matrix.shape) != (sensors, 1, 3) or
                        tuple(counts.shape) != (sensors, 1) or tuple(starts.shape) != (sensors, 1) or
                        force.numel() != capacity or separation.numel() != capacity or
                        tuple(point.shape) != (capacity, 3) or tuple(normal.shape) != (capacity, 3)):
                    raise ValueError("unexpected native self-pair tensor shape")
                integer_types = (torch.int32, torch.int64, torch.uint32)
                if counts.dtype not in integer_types or starts.dtype not in integer_types:
                    raise ValueError("native counts/starts must be int32/int64/uint32 integer tensors")
                # Installed tensor API allocates both buffers as uint32.
                # Convert before reductions/index arithmetic; never accept floats.
                counts, starts = counts.to(dtype=torch.int64), starts.to(dtype=torch.int64)
                if int(counts.sum().item()) >= capacity:
                    raise ValueError("self-pair raw buffer saturated; completeness unavailable")
                counts, starts, bucket, indices, _ = _ragged_contact_indices(torch, counts, starts, capacity)
                selected = torch.tensor(valid, device=matrix.device, dtype=torch.int64)
                valid_mask = torch.zeros(sensors, dtype=torch.bool, device=bucket.device)
                valid_mask[selected] = True
                keep = valid_mask[bucket]
                active = (force[indices][keep], point[indices][keep], normal[indices][keep], separation[indices][keep])
                if not all(bool(torch.isfinite(x).all()) for x in (net[selected], matrix[selected], *active)):
                    raise ValueError("nonfinite active native self-pair data")
                reconstructed = torch.zeros(sensors, 3, device=matrix.device, dtype=matrix.dtype)
                reconstructed.index_add_(0, bucket, force[indices, None]*normal[indices])
                if not torch.allclose(reconstructed[selected], matrix[selected, 0], atol=1e-5, rtol=1e-4):
                    raise ValueError("raw signed Fn*normal does not reconstruct native selected-pair matrix")
                witness = (force[indices].abs() > 0) & (separation[indices] <= 0) & keep
                witness_sensors = sorted(set(bucket[witness].detach().cpu().tolist()))
                result.update(available=True, unavailable_reason=None, positive_witness_sensor_indices=witness_sensors,
                    values={"normal_net_world_n": net[selected], "pair_normal_force_world_n": matrix[selected],
                            "other_partner_normal_residual_world_n": net[selected]-matrix[selected, 0],
                            "counts": counts[selected], "starts_native_buffer": starts[selected],
                            "raw_sensor_indices": bucket[keep], "raw_native_indices": indices[keep],
                            "signed_normal_force_n": force[indices][keep], "point_world_m": point[indices][keep],
                            "normal_world": normal[indices][keep], "separation_m": separation[indices][keep]},
                    raw_ledger_note="compact active-point ledger: use raw_sensor_indices/raw_native_indices, not native starts to slice compact arrays",
                    normal_force_semantics="native force on thumb sensor, world frame; net includes all partners, selected matrix only same-env camera; signed Fn and canonical normal preserved",
                    raw_matrix_reconstruction={"passed": True, "atol_n": 1e-5, "rtol": 1e-4,
                                               "scope": "observation consistency only; no physical success gate changed"})
            except Exception as exc:
                result.update(available=False, values=None, unavailable_reason=type(exc).__name__+": "+str(exc))
            self._cache_key, self._cache = key, result
        row = {**self._cache, "sample_phase": phase}
        if phase == "post_physics_pre_reset" and int(native_step) != self._last_counted_step:
            if row["available"]:
                for i in row["positive_witness_sensor_indices"]:
                    self._positive_post_control_samples[i] += 1
            self._last_counted_step = int(native_step)
        row["positive_witness_seen_sensor_indices"] = [i for i, n in enumerate(self._positive_post_control_samples) if n]
        row["absence_of_selfcollision_certified"] = False
        return row


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
    declaration = config.get("rl_diagnostics", {}).get("full_lift")
    if declaration is not None:
        from .rl_effort_protocol import validate_full_lift_protocol
        declaration = validate_full_lift_protocol(declaration)
        # The local 6 mm milestone remains intact. The independently versioned
        # full-task declaration owns the later height bounds and episode time.
        settings["episode_length_s"] = declaration["episode_length_s"]
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
    from .rl_probe_protocol import validate_video_recording
    video_recording=validate_video_recording(config.get("rl_diagnostics",{}).get("video_recording"),num_envs)
    if video_recording is not None and (probe_profile is None or not eval_mode or output_dir is None):
        raise RLEnvironmentError("Declared multi-environment RGB is a standalone diagnostic recording only")
    if probe_profile is not None and (not eval_mode or output_dir is None or num_envs > 8):
        raise RLEnvironmentError("control diagnostics require standalone evaluation, output and at most eight environments")
    settings = resolve_settings(config)
    from .rl_effort_protocol import validate_full_lift_protocol
    full_lift = validate_full_lift_protocol(config.get("rl_diagnostics", {}).get("full_lift"))
    physics_protocol = config["rl_teacher"]["physics_protocol"]
    if physics_protocol not in (RL_PHYSICS, RL_PHYSICS_FIXED_PALM_ADJACENCY, RL_PHYSICS_EFFORT_GRASP):
        raise RLEnvironmentError("unsupported explicit RL physics protocol")
    effort_model = physics_protocol == RL_PHYSICS_EFFORT_GRASP
    fixed_palm_model = physics_protocol in (RL_PHYSICS_FIXED_PALM_ADJACENCY, RL_PHYSICS_EFFORT_GRASP)
    effort_contract, effort_candidate = None, None
    approach_course=config.get("rl_diagnostics",{}).get("approach_course")
    approach_resource, approach_home_fk = None, None
    control_preshape=config.get("rl_diagnostics",{}).get("control_preshape")
    preshape_resolution=None
    if control_preshape is not None and not effort_model:
        raise RLEnvironmentError("Control preshape requires the explicit effort diagnostic model")
    if effort_model:
        from .rl_effort_protocol import validate_effort_grasp_contract
        diagnostic = config.get("rl_diagnostics", {})
        if not probe_profile or diagnostic.get("profile_plan") != "balanced_effort_v6":
            raise RLEnvironmentError("new effort physics is only admitted by its standalone diagnostic plan")
        effort_contract = validate_effort_grasp_contract(diagnostic["effort_grasp_controller"])
        generation = effort_contract.get("contact_generation")
        if generation is not None:
            settings["contact_offset_m"] = generation["contact_offset_m"]
            settings["rest_offset_m"] = generation["rest_offset_m"]
        payload = Path(diagnostic["candidate_path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != diagnostic["candidate_sha256"]:
            raise RLEnvironmentError("effort geometry/reset resource changed")
        effort_candidate = json.loads(payload)
        if control_preshape is not None:
            from .rl_preshape import validate_control_preshape, resolve_preshape_frames
            control_preshape=validate_control_preshape(control_preshape)
            preshape_resolution=resolve_preshape_frames(control_preshape,effort_candidate,diagnostic["candidate_sha256"])
    initial = _seed_state(config)
    original_initial = copy.deepcopy(initial)
    if effort_model and "reset_joint_positions_rad" in effort_candidate:
        override = effort_candidate["reset_joint_positions_rad"]
        if (set(override) != set(initial["joint_positions_rad"]) or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in override.values())):
            raise RLEnvironmentError("new reset must explicitly name all43 finite joint positions")
        initial = copy.deepcopy(initial)
        initial["joint_positions_rad"] = dict(override)
        reference = effort_candidate.get("reset_native_fk_reference")
        if not isinstance(reference, dict):
            raise RLEnvironmentError("new joint reset requires independently computed native FK reference")
        initial["official_native_fk_reference"] = reference
        initial["root_pose_xyzw"] = list(effort_candidate.get("reset_root_pose_xyzw", initial["root_pose_xyzw"]))
    if approach_course is not None:
        from .rl_approach import validate_approach_declaration
        from .rl_approach_state import load_course_reference
        approach_course=validate_approach_declaration(approach_course)
        if (not effort_model or full_lift is None or
                full_lift["version"]!="g1_contact_qualified_arm_home_lift_v5_32s"):
            raise RLEnvironmentError("Arm-home approach requires its explicit diagnostic full-task version")
        approach_resource,approach_home_fk=load_course_reference(config,config["rl_diagnostics"],effort_candidate)
        initial=copy.deepcopy(initial)
        initial["joint_positions_rad"]=copy.deepcopy(approach_resource["home_joint_positions_rad"])
        initial["official_native_fk_reference"]=approach_home_fk
    actuator_specs = _actuator_specifications(initial)
    if effort_model:
        for name, cap in zip(RIGHT_HAND, effort_contract["actuation"]["total_torque_caps_nm"]):
            actuator_specs[name].update(kp=0., kd=0., effort_limit_nm=cap)
    asset_path = Path(asset_path).resolve()
    if not asset_path.is_file() or asset_path.suffix.lower() not in (".usd", ".usda", ".usdc", ".xml"):
        raise RLEnvironmentError("a prepared robot-only MJCF or USD asset is required")
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise RLEnvironmentError("num_envs must be a positive integer")
    asset_sha = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    asset_audit = audit_asset_manifest(config, asset_path)
    native_official_asset = asset_path.suffix.lower() in (".usd", ".usda", ".usdc")
    if fixed_palm_model and (not native_official_asset or asset_sha != FIXED_PALM_FILTER_ASSET_SHA256):
        raise RLEnvironmentError("physics v4 fixed-palm adjacency requires the pinned official native USD")
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
    from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab.sim.utils.prims import bind_physics_material
    from pxr import Usd, UsdPhysics, PhysxSchema
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

    geometry_observation = config.get("rl_diagnostics", {}).get("collision_geometry_audit") or {}
    scene_query_observation = (effort_model and geometry_observation.get("route") == "live_surface_queries_v2"
                               and geometry_observation.get("enable_scene_query_support") is True)

    @configclass
    class EnvironmentCfg(DirectRLEnvCfg):
        decimation = 33
        episode_length_s = float(settings["episode_length_s"])
        action_space = len(ACTION_JOINT_NAMES)
        observation_space = OBSERVATION_DIM
        state_space = 0
        seed = int(train_seed)
        sim = SimulationCfg(dt=0.002, device=device, render_interval=33,
                            enable_scene_query_support=scene_query_observation,
                            physics=PhysxCfg(solver_type=1, enable_stabilization=False,
                                             friction_offset_threshold=(effort_contract["contact_generation"]["friction_offset_threshold_m"]
                                                 if effort_model and "contact_generation" in effort_contract else .04),
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
            self._video_recording = video_recording
            if video_recording is not None:
                self._evaluation_cameras = {}
                self._declared_video_streams = {}
            self._video_status = {"requested": bool(eval_mode), "enabled": False,
                                  "frames": 0, "failure": None, "camera_scope": "fixed external env0 audit view"}
            if video_recording is not None:
                self._video_status.update(video_recording=video_recording,cameras=[],
                    camera_scope="fixed_external_views_of_declared_original_environment_ids")
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
            self._physics_pinch = None
            self._physics_pinch_reference = None
            self._physics_pinch_update_count = 0
            self._physics_pinch_last_sample = None
            self._effort_grasp = None
            self._effort_reference = None
            self._effort_geometry = None
            self._effort_update_count = 0
            self._effort_last_sample = None
            self._effort_material_readback = None
            self._effort_drive_readback = None
            self._effort_contact_offset_schema = None
            self._effort_contact_generation_readback = None
            self._posture_ff_update_count = 0
            self._posture_ff_last_sample = None
            self._effort_raw_samples_written = 0
            self._effort_raw_samples_omitted = 0
            self._effort_previous_physical = None
            self._effort_raw_post_remaining = 0
            self._effort_raw_triggers = 0
            self._effort_raw_omitted_triggers = 0
            self._effort_raw_history = deque(maxlen=effort_contract["logging"]["raw_pre_steps"] if effort_model else 0)
            self._effort_raw_last_written_native = -1
            self._effort_raw_episode_ids = None
            self._effort_raw_loss_seen = None
            self._effort_raw_max_force = 0.
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
                pinch_closeup = (self._control_probe_profile is not None and
                                 self._control_probe_profile["name"] in ("compliant_pinch_pose_hold", "physics_pinch_pose_hold", "balanced_effort_pose_hold"))
                eye_offset = [0.635, -0.262, 1.045] if pinch_closeup else [1.05, -0.65, 1.30]
                target_offset = [0.32, 0.055, 0.82] if pinch_closeup else [0.35, 0.055, 0.80]
                if approach_course is not None:
                    eye_offset,target_offset=[.90,-.55,1.30],[.35,.03,.89]
                eye = self.scene.env_origins[0] + torch.tensor(eye_offset, device=self.device)
                target = self.scene.env_origins[0] + torch.tensor(target_offset, device=self.device)
                self._evaluation_camera.set_world_poses_from_view(eye[None], target[None])
                self._video_status.update(enabled=True, eye_world_m=eye.detach().cpu().tolist(),
                                          target_world_m=target.detach().cpu().tolist(),
                                          camera_scope="fixed external env0 pinch closeup" if pinch_closeup else "fixed external env0 audit view",
                                          camera_view_protocol="g1_pinch_closeup_env0_v1" if pinch_closeup else "g1_overview_env0_v1")
            if video_recording is not None:
                pinch_closeup=self._control_probe_profile["name"] in (
                    "compliant_pinch_pose_hold","physics_pinch_pose_hold","balanced_effort_pose_hold")
                eye_offset=[.635,-.262,1.045] if pinch_closeup else [1.05,-.65,1.30]
                target_offset=[.32,.055,.82] if pinch_closeup else [.35,.055,.80]
                if approach_course is not None:
                    eye_offset,target_offset=[.90,-.55,1.30],[.35,.03,.89]
                for env_id in video_recording["env_ids"]:
                    eye=self.scene.env_origins[env_id]+torch.tensor(eye_offset,device=self.device)
                    target=self.scene.env_origins[env_id]+torch.tensor(target_offset,device=self.device)
                    self._evaluation_cameras[env_id].set_world_poses_from_view(eye[None],target[None])
                    self._video_status["cameras"].append({"camera_env_id":env_id,"enabled":True,"frames":0,"failure":None,
                        "frame_directory":"video/env_"+str(env_id)+"/rgb_frames",
                        "frame_ledger":"video/env_"+str(env_id)+"/video_frames.jsonl",
                        "eye_world_m":eye.detach().cpu().tolist(),"target_world_m":target.detach().cpu().tolist(),
                        "camera_view_protocol":("g1_high_wait_vertical_approach_overview_v1" if approach_course is not None else
                            "g1_pinch_closeup_declared_env_v1" if pinch_closeup else "g1_overview_declared_env_v1")})
                self._video_status["enabled"]=True
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
            if control_preshape is not None:
                self._control_preshape_contact_seen=torch.zeros_like(self._episode_ids,dtype=torch.bool)
                self._control_preshape_first_contact_native=torch.full_like(self._episode_ids,-1)
                self._control_preshape_contact_episode_ids=self._episode_ids.clone()
                self._control_preshape_contact_last_native=-1
            self._table_positive_per_env = torch.zeros_like(self._episode_ids)
            self._episode_return = torch.zeros(self.num_envs, device=self.device)
            self._support_count = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
            self._support_qualified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._local_success = torch.zeros_like(self._support_qualified)
            self._full_lift = (ContactQualifiedFullLift(torch, self.num_envs, self.device, full_lift)
                               if full_lift is not None else None)
            if self._full_lift is not None:
                self._full_lift_failure_arm_target = self._targets[:, self._action_ids[:7]].clone()
                frames = effort_candidate["geometry_evidence"]["source_detail"]["stage_keyframes"]
                if preshape_resolution is not None:
                    # Execution frames never define the original physical score domain.
                    frames=preshape_resolution["original_frames"]
                trajectory = torch.as_tensor([f.get("hand_target_rad", f.get("target_rad")) for f in frames],
                    device=self.device, dtype=self._targets.dtype)
                if tuple(trajectory.shape) != (6,7) or not bool(torch.isfinite(trajectory).all()):
                    raise RLEnvironmentError("full lift requires immutable original six-keyframe hand bounds")
                self._effort_actual_domain_lower = (trajectory.amin(0)-.025)[None].expand(self.num_envs,-1).clone()
                self._effort_actual_domain_upper = (trajectory.amax(0)+.025)[None].expand(self.num_envs,-1).clone()
            self._native_arm_interpolation = (full_lift is not None and
                full_lift.get("arm_target_schedule") in ("native_linear33_revalidated_v1",
                    "home_and_lift_native_linear33_revalidated_v1"))
            self._keep_current_hand_force_on_full_failure = (full_lift is not None and
                full_lift.get("full_lift_failure_response")=="stop_arm_keep_current_hand_force_feedback_v1")
            if self._native_arm_interpolation:
                self._last_written_arm_target = self._targets[:,self._action_ids[:7]].clone()
                self._last_written_arm_target_episode_ids = torch.full_like(self._episode_ids,-1)
                self._last_written_arm_target_next_native_step = torch.full_like(self._episode_ids,-1)
                self._arm_segment_origin = self._last_written_arm_target.clone()
                self._arm_segment_endpoint = self._last_written_arm_target.clone()
                self._arm_segment_native_index = torch.zeros_like(self._episode_ids)
                self._arm_segment_episode_ids = self._episode_ids.clone()
                self._arm_segment_start_native = int(self.sim.get_physics_step_count())
                self._arm_segment_enabled = torch.zeros_like(self._support_qualified)
                self._arm_segment_cancelled = torch.zeros_like(self._support_qualified)
                self._arm_segment_failure_hold = self._last_written_arm_target.clone()
            self._unsafe = torch.zeros_like(self._support_qualified)
            if approach_course is not None:
                from .rl_approach_state import NativeApproachAdmission
                self._approach=NativeApproachAdmission(torch,approach_resource,approach_course["parameters"],
                    self.num_envs,self.device)
                self._approach_plan_complete=torch.zeros_like(self._support_qualified)
                self._approach_planner_failed=torch.zeros_like(self._support_qualified)
                self._approach_intent_episode_ids=self._episode_ids.clone()
                self._approach_contact_last=None
                self._approach_last=None
                self._approach_measurement_unavailable=[None]*self.num_envs
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
            self._last_contact_episode_ids = torch.full_like(self._episode_ids, -1)
            self._last_contact_native_step = -1
            self._last_written_hand_target = self._targets[:, self._action_ids[7:]].clone()
            self._last_written_hand_effort = torch.zeros(self.num_envs, 7, device=self.device)
            self._last_written_full_effort = torch.zeros(self.num_envs, 43, device=self.device)
            self._effort_hand_body_ids = [self.robot.body_names.index(name) for name in HAND_BODIES]
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
            if self._control_probe_enabled:
                self._native_joint_load_diagnostic = NativeJointLoadDiagnostics(
                    torch, tensor, self.robot.root_view, self.robot.joint_names,
                    self.robot.body_names, self.num_envs)
                from isaaclab_physx.physics import PhysxManager
                self._native_self_pair_diagnostic = NativeSelfPairDiagnostics(
                    torch, tensor, PhysxManager.get_physics_sim_view(), get_current_stage(),
                    self._native_self_pair_paths, self.num_envs, self.physics_dt)
                if approach_course is not None:
                    from .rl_approach_contact import NativeApproachContactWitness
                    self._approach_contact=NativeApproachContactWitness(torch,tensor,
                        PhysxManager.get_physics_sim_view(),get_current_stage(),self.num_envs,self.physics_dt)
                    if self._approach_contact.summary().get("available") is not True:
                        raise RLEnvironmentError("Arm-home collision view cannot certify complete declared routing: "+
                            str(self._approach_contact.summary()))
            if output:
                output.mkdir(parents=True, exist_ok=True)
                if self._control_probe_enabled:
                    self._control_probe_stream = (output / "control_commands.jsonl").open("x", encoding="utf-8")
                if eval_mode:
                    self._open_evaluation_recording()
                    if video_recording is not None:
                        self._open_declared_video_recording()
                    else:
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
                if effort_model:
                    self._verify_effort_native_model()
                self.configure_control_probe(self._control_probe_profile)

        def _setup_scene(self):
            pd_specs = {name: value for name, value in actuator_specs.items()
                        if not effort_model or name not in RIGHT_HAND}
            actuators = {"shared_pd_without_mujoco_bias_v1": ImplicitActuatorCfg(
                joint_names_expr=list(pd_specs) if effort_model else [".*"],
                stiffness={name: value["kp"] for name, value in pd_specs.items()},
                damping={name: value["kd"] for name, value in pd_specs.items()},
                effort_limit_sim={name: value["effort_limit_nm"] for name, value in pd_specs.items()},
                armature={name: value["armature"] for name, value in pd_specs.items()},
                viscous_friction={name: value["dof_damping"] for name, value in pd_specs.items()})}
            if effort_model:
                caps = dict(zip(RIGHT_HAND, effort_contract["actuation"]["total_torque_caps_nm"]))
                actuators["right_hand_bounded_explicit_effort_v1"] = IdealPDActuatorCfg(
                    joint_names_expr=list(RIGHT_HAND), stiffness=0., damping=0.,
                    effort_limit=caps, effort_limit_sim=caps,
                    armature={name: actuator_specs[name]["armature"] for name in RIGHT_HAND},
                    viscous_friction={name: actuator_specs[name]["dof_damping"] for name in RIGHT_HAND})
            self.robot = Articulation(ArticulationCfg(
                prim_path="/World/envs/env_.*/Robot", spawn=robot_spawn,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=tuple(initial["root_pose_xyzw"][:3]), rot=tuple(initial["root_pose_xyzw"][3:]),
                    joint_pos=initial["joint_positions_rad"], joint_vel={".*": 0.0}),
                actuators=actuators))
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
            if effort_model:
                pad_material = material(settings["hand_friction"])
                pad_material.func("/World/RLFingerPadContactMaterial", pad_material)
                pad_api = PhysxSchema.PhysxMaterialAPI.Apply(stage.GetPrimAtPath("/World/RLFingerPadContactMaterial"))
                declared = effort_contract["material"]
                pad_api.CreateCompliantContactStiffnessAttr(declared["compliant_contact_stiffness_n_m"])
                pad_api.CreateCompliantContactDampingAttr(declared["compliant_contact_damping_ns_m"])
                pad_api.CreateCompliantContactAccelerationSpringAttr(False)
                pad_api.CreateRestitutionCombineModeAttr(declared["restitution_combine_mode"])
                pad_api.CreateDampingCombineModeAttr(declared["damping_combine_mode"])
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
                    if effort_model and ancestor.GetName() in effort_contract["material"]["compliance_candidate_body_names"]:
                        material_path = "/World/RLFingerPadContactMaterial"
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
            if self._control_probe_enabled:
                self._native_self_pair_paths = [
                    {"env_id": env_id, "sensor_body": body,
                     "sensor_path": all_body_paths[body].replace("/env_.*/", "/env_"+str(env_id)+"/"),
                     "filter_path": all_body_paths["right_hand_camera_base_link"].replace("/env_.*/", "/env_"+str(env_id)+"/")}
                    for env_id in range(self.num_envs)
                    for body in ("right_hand_thumb_0_link", "right_hand_thumb_1_link")]
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
                if video_recording is None:
                    self._evaluation_camera = Camera(CameraCfg(
                        prim_path="/World/RLEvaluationCamera", update_period=0.066,
                        width=640, height=480, data_types=["rgb"],
                        renderer_cfg=IsaacRtxRendererCfg(),
                        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=32.0,
                                                         clipping_range=(0.01, 20.0))))
                else:
                    for env_id in video_recording["env_ids"]:
                        self._evaluation_cameras[env_id]=Camera(CameraCfg(
                            prim_path="/World/RLEvaluationCamera_env_"+str(env_id),update_period=.066,
                            width=640,height=480,data_types=["rgb"],renderer_cfg=IsaacRtxRendererCfg(),
                            spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=32.,
                                clipping_range=(.01,20.))))
                # One global camera is updated explicitly. Registering it in
                # the N-env scene would pass partial reset IDs > 0 to a camera
                # with only one instance and corrupt its reset bookkeeping.
                light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
                light.func("/World/RLEvaluationLight", light)
            if effort_model and "contact_generation" in effort_contract:
                self._author_effort_contact_offsets(stage)
            self.scene.clone_environments(copy_from_source=False)
            if fixed_palm_model:
                if effort_model:
                    from .rl_collision_model import author_bilateral_fixed_cluster_filters
                    self._fixed_cluster_filter_readback = author_bilateral_fixed_cluster_filters(
                        stage, UsdPhysics, self.num_envs, asset_sha)
                else:
                    self._fixed_cluster_filter_readback = author_fixed_palm_adjacency_filters(
                        stage, UsdPhysics, self.num_envs, asset_sha)
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=[])

        def _author_effort_contact_offsets(self, stage):
            """Author every source shape, including instance-proxy robot shapes.

            Native readback after physics initialization remains authoritative;
            this declaration never filters a measured normal/manifold point.
            """
            declaration = effort_contract["contact_generation"]
            result = {}
            for name in ("Robot", "Cube", "Table"):
                root = stage.GetPrimAtPath("/World/envs/env_0/"+name)
                paths = [str(p.GetPath()) for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                         if p.HasAPI(UsdPhysics.CollisionAPI)]
                if not paths or len(paths) != len(set(paths)):
                    raise RLEnvironmentError("contact generation requires every unique collision shape: "+name)
                records = []
                for path in paths:
                    prim = stage.GetPrimAtPath(path)
                    instance = prim
                    while instance.IsValid() and not instance.IsInstance():
                        instance = instance.GetParent()
                    if instance.IsValid():
                        instance_path = str(instance.GetPath())
                        instance.SetInstanceable(False)
                        self._instance_overrides.append(instance_path.replace("/env_0/", "/env_*/"))
                        prim = stage.GetPrimAtPath(path)
                    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    # This order is also safe for a live native shape with a
                    # previous zero rest offset; 0 must remain > rest offset.
                    api.CreateRestOffsetAttr(declaration["rest_offset_m"])
                    api.CreateContactOffsetAttr(declaration["contact_offset_m"])
                    contact, rest = api.GetContactOffsetAttr().Get(), api.GetRestOffsetAttr().Get()
                    if (contact is None or rest is None or
                            not math.isclose(float(contact), declaration["contact_offset_m"], rel_tol=0, abs_tol=1e-12) or
                            not math.isclose(float(rest), declaration["rest_offset_m"], rel_tol=0, abs_tol=1e-12)):
                        raise RLEnvironmentError("contact-generation USD shape offsets did not apply: "+path)
                    records.append({"collider": path.replace("/env_0/", "/env_*/"),
                                    "contact_offset_m": contact, "rest_offset_m": rest})
                result[name] = records
            self._effort_contact_offset_schema = result

        def _verify_effort_contact_generation(self):
            """Apply/read native all-shape offsets before the first physics step."""
            declaration = effort_contract["contact_generation"]
            native_step = int(self.sim.get_physics_step_count())
            if native_step != 0:
                raise RLEnvironmentError("contact-generation model must be fixed before task physics")
            if self._effort_contact_offset_schema is None:
                raise RLEnvironmentError("contact-generation model is missing its complete source-shape declaration")
            native = {}
            for name, view in (("Robot", self.robot.root_view), ("Cube", self.cube.root_view), ("Table", self.table.root_view)):
                contact_before = tensor(view.get_contact_offsets()).clone()
                rest_before = tensor(view.get_rest_offsets()).clone()
                authored_count = len(self._effort_contact_offset_schema[name])
                # CollisionAPI prims and cooked native shapes are distinct
                # counts (for example compound/cooked geometry). The native
                # view's whole max_shapes array, never the USD count, sets the
                # parameter-buffer width. No native entry is discarded.
                shape_count = int(view.max_shapes)
                shape = (self.num_envs, shape_count)
                paths = list(view.prim_paths)
                binding_ok = (len(paths) == self.num_envs and all(
                    path == "/World/envs/env_{}/{}".format(i, name) or
                    (name == "Robot" and path.startswith("/World/envs/env_{}/Robot/".format(i)))
                    for i, path in enumerate(paths)))
                homogeneous = getattr(view, "is_homogeneous", None) if name == "Robot" else None
                diagnostic = {"body": name, "view_count": int(view.count), "max_shapes": shape_count,
                    "native_contact_offset_shape": list(contact_before.shape),
                    "native_rest_offset_shape": list(rest_before.shape),
                    "authored_collider_api_count": authored_count,
                    "contact_offsets_finite": bool(torch.isfinite(contact_before).all()),
                    "rest_offsets_finite": bool(torch.isfinite(rest_before).all()),
                    "native_prim_paths": paths, "exact_per_env_binding": binding_ok,
                    "native_articulation_is_homogeneous": homogeneous,
                    "contact_offsets_values": str(contact_before), "rest_offsets_values": str(rest_before)}
                if (int(view.count) != self.num_envs or shape_count < 1 or authored_count < 1 or not binding_ok or
                        (name == "Robot" and homogeneous is not True) or
                        tuple(contact_before.shape) != shape or tuple(rest_before.shape) != shape or
                        not bool(torch.isfinite(contact_before).all()) or not bool(torch.isfinite(rest_before).all())):
                    raise RLEnvironmentError("native all-shape offset binding/count/finite mismatch: "+json.dumps(diagnostic))
                desired_contact = torch.full(shape, declaration["contact_offset_m"], dtype=torch.float32, device="cpu")
                desired_rest = torch.full(shape, declaration["rest_offset_m"], dtype=torch.float32, device="cpu")
                already_matched = (bool(torch.isclose(contact_before.cpu(), desired_contact, atol=1e-12, rtol=0).all()) and
                                   bool(torch.isclose(rest_before.cpu(), desired_rest, atol=1e-12, rtol=0).all()))
                if not already_matched:
                    indices = wp.array(list(range(self.num_envs)), dtype=wp.uint32, device="cpu")
                    # Parameter setters require CPU arrays in this installed
                    # backend. Lower rest first so contact=0 remains legal.
                    view.set_rest_offsets(wp.from_torch(desired_rest), indices=indices)
                    view.set_contact_offsets(wp.from_torch(desired_contact), indices=indices)
                contact_after = tensor(view.get_contact_offsets()).clone()
                rest_after = tensor(view.get_rest_offsets()).clone()
                passed = (tuple(contact_after.shape) == shape and tuple(rest_after.shape) == shape and
                    bool(torch.isfinite(contact_after).all()) and bool(torch.isfinite(rest_after).all()) and
                    bool(torch.isclose(contact_after.cpu(), desired_contact, atol=1e-12, rtol=0).all()) and
                    bool(torch.isclose(rest_after.cpu(), desired_rest, atol=1e-12, rtol=0).all()))
                if not passed:
                    diagnostic.update(contact_offsets_after_shape=list(contact_after.shape),
                        rest_offsets_after_shape=list(rest_after.shape),
                        contact_offsets_after_values=str(contact_after), rest_offsets_after_values=str(rest_after),
                        wanted_contact_offset_m=declaration["contact_offset_m"], wanted_rest_offset_m=declaration["rest_offset_m"])
                    raise RLEnvironmentError("native contact/rest offsets differ from declared zero-margin model: "+json.dumps(diagnostic))
                native[name] = {"verified": True, "view_count": int(view.count), "native_shape_count_per_env": shape_count,
                    "authored_collision_api_prim_count_per_env": authored_count,
                    "native_minus_authored_count": shape_count-authored_count,
                    "source_to_native_one_to_one_shape_mapping_claimed": False,
                    "all_native_shape_entries_checked": self.num_envs*shape_count,
                    "native_prim_paths": paths, "exact_per_env_binding": binding_ok,
                    "native_articulation_is_homogeneous": homogeneous,
                    "source_collision_shapes": self._effort_contact_offset_schema[name],
                    "contact_offsets_before_m": contact_before, "rest_offsets_before_m": rest_before,
                    "prephysics_native_setters_applied": not already_matched,
                    "setter_order": "rest_then_contact_cpu_parameter_arrays" if not already_matched else "not_needed",
                    "contact_offsets_after_m": contact_after, "rest_offsets_after_m": rest_after}
            stage = get_current_stage()
            scene_path = self.sim.cfg.physics_prim_path
            scene = stage.GetPrimAtPath(scene_path)
            if not scene.IsValid() or not scene.HasAPI(PhysxSchema.PhysxSceneAPI):
                raise RLEnvironmentError("contact-generation friction threshold requires the actual physics-scene schema")
            actual = PhysxSchema.PhysxSceneAPI(scene).GetFrictionOffsetThresholdAttr().Get()
            if actual is None or not math.isfinite(float(actual)) or float(actual) != declaration["friction_offset_threshold_m"]:
                raise RLEnvironmentError("physics-scene USD friction generation threshold differs from declaration")
            self._effort_contact_generation_readback = _probe_payload(torch, {
                "verified": True, "contract": declaration, "native_physics_step_count_before_model_binding": native_step,
                "native_shape_offsets": native,
                "scene_friction_offset_threshold": {"physics_scene_path": scene_path, "usd_value_m": actual,
                    "readback_kind": "USD_PhysxSceneAPI_not_compiled_PxScene_getter", "compiled_scene_getter_available": False},
                "normal_manifold_filtering": False, "original_physical_acceptance_unchanged": True,
                "zero_margin_is_new_contact_generation_model_not_a_contact_or_force_guarantee": True})

        def _verify_effort_native_model(self):
            """Fail closed on actual drive/material bindings, not config labels."""
            ids = self._action_ids[7:]
            kp = tensor(self.robot.root_view.get_dof_stiffnesses()).clone()
            kd = tensor(self.robot.root_view.get_dof_dampings()).clone()
            maximum = tensor(self.robot.root_view.get_dof_max_forces()).clone()
            expected_kp = torch.tensor([actuator_specs[n]["kp"] for n in self.robot.joint_names], device=kp.device)
            expected_kd = torch.tensor([actuator_specs[n]["kd"] for n in self.robot.joint_names], device=kd.device)
            expected_caps = torch.tensor([actuator_specs[n]["effort_limit_nm"] for n in self.robot.joint_names], device=maximum.device)
            passed = (tuple(kp.shape) == (self.num_envs, 43) and
                bool(torch.isclose(kp, expected_kp[None], atol=1e-5, rtol=0).all()) and
                bool(torch.isclose(kd, expected_kd[None], atol=1e-5, rtol=0).all()) and
                bool(torch.isclose(maximum, expected_caps[None], atol=1e-5, rtol=0).all()))
            gravity = tensor(self.robot.data.gravity_compensation_forces)
            if gravity.shape != (self.num_envs, 43) or not bool(torch.isfinite(gravity).all()):
                raise RLEnvironmentError("effort control requires finite current native gravity compensation")
            self._effort_drive_readback = _probe_payload(torch, {
                "verified": passed, "joint_names": list(self.robot.joint_names),
                "hand_joint_indices": ids, "native_stiffness_nm_rad": kp,
                "native_damping_nm_s_rad": kd, "native_drive_force_caps_nm": maximum,
                "initial_gravity_compensation_hand_nm": gravity[:, ids],
                "explicit_actuator_type": type(self.robot.actuators["right_hand_bounded_explicit_effort_v1"]).__name__,
                "gravity_inside_total_cap": True, "other_36_pd_values_verified": passed,
                "drive_force_cap_is_not_external_contact_force_or_isolated_torque_measurement": True})
            if not passed:
                raise RLEnvironmentError("native hand7 drives must be zero and other36 original PD unchanged")
            stage = get_current_stage()
            from pxr import UsdShade
            mat = effort_contract["material"]
            expected_pad = (mat["compliant_contact_stiffness_n_m"], mat["compliant_contact_damping_ns_m"])
            records = []
            for i in range(self.num_envs):
                for binding in self._material_bindings:
                    path = binding["collider"].replace("/env_*/", "/env_{}/".format(i))
                    prim = stage.GetPrimAtPath(path)
                    bound = [str(p) for p in prim.GetRelationship("material:binding:physics").GetTargets()]
                    if bound != [binding["material"]]:
                        raise RLEnvironmentError("native finger material binding mismatch: " + path)
                    material_prim = stage.GetPrimAtPath(bound[0])
                    api = PhysxSchema.PhysxMaterialAPI(material_prim)
                    stiffness = api.GetCompliantContactStiffnessAttr().Get()
                    damping = api.GetCompliantContactDampingAttr().Get()
                    acceleration = api.GetCompliantContactAccelerationSpringAttr().Get()
                    pad = binding["body"] in mat["compliance_candidate_body_names"]
                    want_k, want_c = expected_pad if pad else (0., 0.)
                    if (not math.isclose(float(stiffness or 0), want_k, rel_tol=0., abs_tol=1e-6) or
                            not math.isclose(float(damping or 0), want_c, rel_tol=0., abs_tol=1e-6) or bool(acceleration)):
                        raise RLEnvironmentError("native hand contact spring differs from declared stiffness/damping: " + path)
                    if pad and (str(api.GetRestitutionCombineModeAttr().Get()) != mat["restitution_combine_mode"] or
                            str(api.GetDampingCombineModeAttr().Get()) != mat["damping_combine_mode"]):
                        raise RLEnvironmentError("native compliant contact combination mode mismatch")
                    physics_material = UsdPhysics.MaterialAPI(material_prim)
                    friction = settings["palm_friction"] if binding["body"] == palm_contact_body else settings["hand_friction"]
                    actual_friction = [float(physics_material.GetStaticFrictionAttr().Get()),
                                       float(physics_material.GetDynamicFrictionAttr().Get())]
                    if (any(not math.isclose(value, friction, rel_tol=0., abs_tol=1e-6) for value in actual_friction) or
                            float(physics_material.GetRestitutionAttr().Get()) != 0.):
                        raise RLEnvironmentError("effort model changed undeclared friction/restitution")
                    records.append({"collider": path, "body": binding["body"], "material": bound[0],
                        "stiffness_n_m": stiffness, "damping_ns_m": damping,
                        "acceleration_spring": acceleration, "soft_candidate": pad,
                        "static_dynamic_friction": actual_friction, "friction_readback_float_tolerance": 1e-6})
            external_records, rigid_body_records = [], []
            for i in range(self.num_envs):
                for body_name in mat["always_rigid_body_names"]:
                    prefix = "/World/envs/env_{}/".format(i)
                    path = prefix + body_name if body_name in ("Cube", "Table") else prefix + "Robot/" + body_name
                    body = stage.GetPrimAtPath(path)
                    if not body.IsValid() or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                        raise RLEnvironmentError("missing declared always-rigid body during material readback: " + path)
                    colliders = [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                                 if prim.HasAPI(UsdPhysics.CollisionAPI)]
                    if not colliders and body_name in ("Cube", "Table"):
                        raise RLEnvironmentError("Cube/Table must retain actual collision geometry: " + path)
                    body_record = {"env_id": i, "body": body_name, "body_path": path,
                                   "collider_count": len(colliders), "colliders": [],
                                   "status": "verified_rigid_materials" if colliders else "no_collision_geometry"}
                    for prim in colliders:
                        # Resolve inheritance, collection bindings, purpose
                        # fallback and instance proxies using USD's actual
                        # binding rules rather than the nearest raw relation.
                        bound_material, binding_rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
                        if bound_material and bound_material.GetPrim().IsValid():
                            material_prim = bound_material.GetPrim()
                            api = PhysxSchema.PhysxMaterialAPI(material_prim)
                            k = api.GetCompliantContactStiffnessAttr().Get()
                            c = api.GetCompliantContactDampingAttr().Get()
                            acceleration = api.GetCompliantContactAccelerationSpringAttr().Get()
                            binding_path = str(binding_rel.GetPath()) if binding_rel else None
                            material_path = str(material_prim.GetPath())
                            resolution = "resolved_usd_physics_purpose_binding"
                        else:
                            # A genuinely unbound collider uses the engine's
                            # rigid default. A dangling authored binding is
                            # not equivalent to being unbound and must fail.
                            ancestor, declared_binding = prim, bool(binding_rel and binding_rel.GetTargets())
                            while ancestor.IsValid():
                                for rel in ancestor.GetRelationships():
                                    if str(rel.GetName()).startswith("material:binding") and rel.GetTargets():
                                        declared_binding = True
                                ancestor = ancestor.GetParent()
                            if declared_binding:
                                raise RLEnvironmentError("unresolved authored material binding on always-rigid collider: " + str(prim.GetPath()))
                            k, c, acceleration = 0., 0., False
                            binding_path, material_path = None, None
                            resolution = "unbound_engine_rigid_default_no_usd_material_override"
                        if (not math.isfinite(float(k or 0)) or not math.isfinite(float(c or 0)) or
                                float(k or 0) != 0 or float(c or 0) != 0):
                            raise RLEnvironmentError("declared always-rigid collider has compliant contact material: " + str(prim.GetPath()))
                        record = {"collider": str(prim.GetPath()), "body": body_name,
                                  "instance_proxy": bool(prim.IsInstanceProxy()),
                                  "material": material_path, "binding_relationship": binding_path,
                                  "binding_resolution": resolution, "stiffness_n_m": k,
                                  "damping_ns_m": c, "acceleration_spring": acceleration}
                        body_record["colliders"].append(record)
                        if body_name in ("Cube", "Table"):
                            external_records.append(record)
                    rigid_body_records.append(body_record)
            self._effort_material_readback = {"verified": True, "variant": mat["variant"],
                "bindings": records, "cube_table_rigid_readback": external_records,
                "always_rigid_body_readback": rigid_body_records,
                "always_rigid_body_names_verified": list(mat["always_rigid_body_names"]),
                "material_readback_scope": "actual USD collision material bindings and schema including instance proxies; not a measured contact spring response",
                "compliance_is_model_change_not_force_cap_or_real_robot_certification": True}
            if "contact_generation" in effort_contract:
                self._verify_effort_contact_generation()

        def _pre_physics_step(self, actions):
            if tuple(actions.shape) != (self.num_envs, 14) or not bool(torch.isfinite(actions).all()):
                raise RLEnvironmentError("nonfinite or incorrectly shaped policy action")
            if self._control_probe_enabled and self._control_transition_snapshot is not None:
                raise RLEnvironmentError("previous diagnostic transition must be recorded before the next command")
            self._previous_actions[:] = self._actions
            self._action_origin, self._next_action_origin = self._next_action_origin, "policy_or_zero_target_hold"
            self._actions[:] = actions.clamp(-1.0, 1.0)
            if self._physics_pinch is not None:
                self._action_origin = "declared_macro_pose_and_geometry_reference_plus_native_2ms_force_admittance"
            if self._effort_grasp is not None:
                self._action_origin = "macro_arm_position_reference_and_native_2ms_explicit_hand_effort"
                # These seven slots are inactive bookkeeping in effort mode.
                # They are never handed to a native position drive in _apply_action.
                self._targets[:, self._action_ids[7:]] = tensor(self.robot.data.joint_pos)[:, self._action_ids[7:]]
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
            palm_height_limit = settings["probe_clearance_max_m"]
            if getattr(self, "_full_lift", None) is not None:
                palm_height_limit = torch.where(self._full_lift.main_lift_authorized,
                    torch.full_like(palm_z, full_lift["palm_lift_request_max_m"]),
                    torch.full_like(palm_z, settings["probe_clearance_max_m"]))
            if approach_course is not None:
                approach_cap=approach_course["parameters"]["workspace_max_palm_z_m"]-float(
                    approach_resource["pregrasp_palm_position_world_m"][2])
                palm_height_limit=torch.where(~self._approach.admitted,
                    torch.full_like(palm_z,approach_cap),palm_height_limit)
            allowance = (self._start_palm_z + palm_height_limit - previous_prediction).clamp_min(0)
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
                    point_jz, palm_z, self._start_palm_z, palm_height_limit, mode=version)
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
                macro_full_reference = desired.clone()
                macro_reference = desired[:, 7:].clone()
                if self._physics_pinch is not None:
                    if self._physics_pinch_reference is None:
                        raise RLEnvironmentError("2ms pinch requires an explicit macro geometry reference")
                    self._physics_pinch_macro_low = low[:, 7:].clone()
                    self._physics_pinch_macro_high = high[:, 7:].clone()
                    # The macro request has never been written to the drive.
                    # Preserve the last real write; only _apply_action may move
                    # the hand through the declared per-physics-step controller.
                    desired[:, 7:] = self._last_written_hand_target
                self._write_control_probe({
                    "event": "control_command", "command_id": self._control_command_id,
                    "native_physics_step_count": int(self.sim.get_physics_step_count()),
                    "episode_id": self._episode_ids, "episode_control_steps": self.episode_length_buf,
                    "native_joint_load_measurement": self._control_probe_joint_load("pre_command"),
                    "native_self_pair_measurement": self._control_probe_self_pair("pre_command"),
                    "raw_action": actions, "clipped_action": self._actions,
                    "previous_target_action_rad": previous, "measured_q_action_rad": current,
                    "measured_qd_action_rad_s": tensor(self.robot.data.joint_vel)[:, self._action_ids],
                    "per_step_action_rad": per_step, "lower_action_bound_rad": low, "upper_action_bound_rad": high,
                    "unbounded_candidate_action_rad": previous+self._actions*per_step,
                    "limited_candidate_action_rad": candidate, "post_guard_target_action_rad": desired,
                    "macro_hand_reference_before_physics_feedback_rad": macro_reference,
                    "unwritten_macro_reference_action_rad": macro_full_reference if self._physics_pinch is not None else None,
                    "guard_evidence_scope": ("macro_guarded_endpoint_before_native_arm_interpolation" if
                        getattr(self,"_native_arm_interpolation",False) else
                        "macro_arm_guard_and_unwritten_hand_reference_before_inner_feedback" if self._physics_pinch is not None else "macro_applied_target"),
                    "hand_target_execution": "2ms_feedback_after_completed_sample" if self._physics_pinch is not None else "macro_target_hold",
                    "arm_target_execution": ("home_and_lift_native_linear33_revalidated" if approach_course is not None else "main_lift_native_linear33_revalidated" if
                        getattr(self,"_native_arm_interpolation",False) else "macro_target_hold"),
                    "point_jz": point_jz, "palm_position_world_m": tensor(self.robot.data.body_pos_w)[:, self._palm_id]+offset_w,
                    "palm_point_jacobian_w": offset_point_jacobian(torch, palm_jac, offset_w),
                    "start_palm_z_m": self._start_palm_z, "guard_evidence": evidence,
                    "pre_guard_infeasible": infeasible, "post_lower_bound_violation_rad": lower_violation,
                    "post_upper_bound_violation_rad": upper_violation, "post_bound_tolerance_rad": bound_tolerance,
                    "post_bounds_passed": bounds_ok, "requires_unsafe": guard_unsafe,
                    "legacy_diagnostic_difference": "post-guard box/height violations are explicit unsafe; roundoff-only legacy residuals remain logged"})
            self._targets[:, self._action_ids] = desired
            if getattr(self,"_native_arm_interpolation",False):
                self._arm_segment_origin = self._last_written_arm_target.clone()
                self._arm_segment_endpoint = desired[:,:7].clone()
                self._arm_segment_native_index.zero_()
                self._arm_segment_episode_ids = self._episode_ids.clone()
                self._arm_segment_start_native = int(self.sim.get_physics_step_count())
                self._arm_segment_enabled = ((torch.ones_like(self._full_lift.main_lift_started) if
                    approach_course is not None else self._full_lift.main_lift_started) & ~self._unsafe &
                                             ~self._arm_segment_cancelled)

        def _native_arm_substep_target(self):
            """Prepare one original-limits-revalidated target; do not write it."""
            ids=self._action_ids[:7]
            measured=tensor(self.robot.data.joint_pos)[:,ids]
            self._arm_segment_native_index += 1
            native=int(self.sim.get_physics_step_count())
            fresh=((self._arm_segment_episode_ids==self._episode_ids) &
                (self._arm_segment_start_native+self._arm_segment_native_index-1==native))
            nonfinite_measured=~torch.isfinite(measured).all(-1)
            self._unsafe |= nonfinite_measured
            cancelled=self._unsafe | self._full_lift.failed
            first_cancel=cancelled & ~self._arm_segment_cancelled
            raw_hold=torch.where(self._full_lift.failed[:,None],self._full_lift_failure_arm_target,measured)
            hold_fallback=~torch.isfinite(raw_hold)
            hold=torch.where(hold_fallback,self._last_written_arm_target,raw_hold)
            self._unsafe |= hold_fallback.any(-1)
            self._arm_segment_failure_hold[first_cancel]=hold[first_cancel]
            self._arm_segment_cancelled |= cancelled
            body_rotation=tensor(self.robot.data.body_quat_w)[:,self._palm_id]
            offset=quat_apply(body_rotation,self._palm_local)
            palm=tensor(self.robot.data.body_pos_w)[:,self._palm_id]+offset
            jac=tensor(self.robot.data.body_link_jacobian_w)
            if jac.shape[1:]!=(self.robot.num_bodies-1,6,43):
                raise RLEnvironmentError("native arm interpolation requires fixed-root43 current Jacobians")
            point_j=offset_point_jacobian(torch,jac[:,self._palm_id-1][:,:,ids],offset)
            cap=torch.where(self._full_lift.main_lift_started,
                torch.full_like(palm[:,2],full_lift["palm_lift_request_max_m"]),
                torch.full_like(palm[:,2],settings["probe_clearance_max_m"]))
            if approach_course is not None:
                cap=torch.where(~self._approach.admitted,torch.full_like(cap,
                    approach_course["parameters"]["workspace_max_palm_z_m"]-
                    float(approach_resource["pregrasp_palm_position_world_m"][2])),cap)
            target,evidence=revalidated_native_arm_substep(torch,
                segment_origin=self._arm_segment_origin,macro_endpoint=self._arm_segment_endpoint,
                previous_written=self._last_written_arm_target,measured_arm=measured,
                native_index=self._arm_segment_native_index,
                substeps=full_lift["arm_target_interpolation_substeps"],physics_dt_s=self.physics_dt,
                velocity_limits_rad_s=self._velocity[:7],anchor_delta_rad=float(config["control"]["max_anchor_delta_rad"]),
                joint_lower=self._limits[:,ids,0],joint_upper=self._limits[:,ids,1],point_jz=point_j[:,2],
                palm_z=palm[:,2],start_palm_z=self._start_palm_z,height_cap_m=cap,
                enabled=self._arm_segment_enabled & ~self._arm_segment_cancelled,sample_fresh=fresh)
            guard_failed=evidence["requires_unsafe"] & ~self._arm_segment_cancelled
            self._arm_segment_failure_hold[guard_failed]=measured[guard_failed]
            self._arm_segment_cancelled |= guard_failed
            self._unsafe |= guard_failed
            target=torch.where(self._arm_segment_cancelled[:,None],self._arm_segment_failure_hold,target)
            actual_delta=target-self._last_written_arm_target
            nominal_norm=torch.linalg.vector_norm(evidence["nominal_increment_rad"],dim=-1)
            evidence.update(segment_episode_id=self._arm_segment_episode_ids.clone(),
                segment_start_native_step=self._arm_segment_start_native,
                arm_schedule_cancelled=self._arm_segment_cancelled.clone(),
                cancelled_measured_pose_hold_rad=self._arm_segment_failure_hold.clone(),
                cancelled_pose_nonfinite_axis_fallback_to_last_successful_write=hold_fallback,
                prepared_target_after_failure_override_rad=target.clone(),
                written_increment_rad=actual_delta,
                written_increment_rad_s=actual_delta/self.physics_dt,
                written_target_minus_measured_arm_rad=target-measured,
                written_delta_norm_over_nominal_delta_norm=torch.where(nominal_norm>1e-12,
                    torch.linalg.vector_norm(actual_delta,dim=-1)/nominal_norm.clamp_min(1e-12),
                    torch.zeros_like(nominal_norm)),
                failure_override_is_not_velocity_certified=self._arm_segment_cancelled.clone())
            return target,evidence

        def _reset_native_arm_schedule(self,env_ids,arm):
            """Reset actual-write history only with the real episode reset."""
            self._last_written_arm_target[env_ids]=arm
            self._last_written_arm_target_episode_ids[env_ids]=self._episode_ids[env_ids]
            self._last_written_arm_target_next_native_step[env_ids]=int(self.sim.get_physics_step_count())+1
            self._arm_segment_origin[env_ids]=arm
            self._arm_segment_endpoint[env_ids]=arm
            self._arm_segment_native_index[env_ids]=0
            self._arm_segment_episode_ids[env_ids]=self._episode_ids[env_ids]
            self._arm_segment_enabled[env_ids]=False
            self._arm_segment_cancelled[env_ids]=False
            self._arm_segment_failure_hold[env_ids]=arm

        def _apply_action(self):
            if self._pending_physics_sample:
                self._sample_physics()
            if getattr(self, "_full_lift", None) is not None:
                # Stop the pending macro ascent at the very next 2 ms command,
                # holding the measured arm pose captured on the failure sample.
                failed = self._full_lift.failed
                self._targets[:, self._action_ids[:7]] = torch.where(failed[:, None],
                    self._full_lift_failure_arm_target, self._targets[:, self._action_ids[:7]])
            if self._physics_pinch is not None:
                self._update_physics_pinch()
            if self._effort_grasp is not None:
                self._update_effort_grasp()
                position_ids = [i for i in range(43) if i not in self._action_ids[7:]]
                position_target=self._targets[:,position_ids]
                arm_evidence=None
                if getattr(self,"_native_arm_interpolation",False):
                    arm_target,arm_evidence=self._native_arm_substep_target()
                    position_target=position_target.clone()
                    columns=[position_ids.index(i) for i in self._action_ids[:7]]
                    position_target[:,columns]=arm_target
                self.robot.set_joint_position_target_index(target=position_target, joint_ids=position_ids)
                if arm_evidence is not None:
                    # Record only after the real SDK target-buffer write has
                    # returned. This is not a getter of an isolated drive torque
                    # or a guarantee of measured joint/object velocity.
                    self._last_written_arm_target=arm_target.detach().clone()
                    self._last_written_arm_target_episode_ids=self._episode_ids.clone()
                    self._last_written_arm_target_next_native_step=torch.full_like(
                        self._episode_ids,int(self.sim.get_physics_step_count())+1)
                    self._write_control_probe({"event":"physics_arm_position_target_written",
                        "command_id":self._control_command_id,"episode_id":self._episode_ids.clone(),
                        "native_physics_step_count":int(self.sim.get_physics_step_count()),
                        "target_applies_to_next_native_step":int(self.sim.get_physics_step_count())+1,
                        "actually_written_arm_target_rad":self._last_written_arm_target,
                        "measured_arm_rad":tensor(self.robot.data.joint_pos)[:,self._action_ids[:7]],
                        "measured_arm_velocity_rad_s":tensor(self.robot.data.joint_vel)[:,self._action_ids[:7]],
                        "position_write_joint_ids":position_ids,"arm_schedule":arm_evidence,
                        "write_source":"arguments_of_successful_set_joint_position_target_index",
                        "native_target_readback_claimed":False})
                self._write_effort_targets()
            else:
                self.robot.set_joint_position_target_index(target=self._targets)
            if self._physics_pinch is not None:
                self._last_written_hand_target = self._targets[:, self._action_ids[7:]].clone()
            self._pending_physics_sample = True

        def _write_effort_targets(self):
            """Write one fresh full-DOF effort vector; the implicit arm PD stays native.

            Posture caps constrain this added feedforward only. Native implicit PD
            torque is additive and is not certified by these feedforward caps.
            The other 26 user-effort entries are overwritten with zero every
            physics step, so a partial setter cannot retain an earlier effort.
            """
            declaration = effort_contract["actuation"]["posture_gravity_feedforward"]
            expected_names = list(RIGHT_ARM)+["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
            names = declaration["joint_order"]
            hand_ids = list(self._action_ids[7:])
            if (names != expected_names or len(set(self.robot.joint_names)) != 43 or
                    any(name not in self.robot.joint_names for name in names)):
                raise RLEnvironmentError("posture feedforward must bind exact arm7 plus waist3 joint names")
            posture_ids = [self.robot.joint_names.index(name) for name in names]
            if (len(set(posture_ids+hand_ids)) != 17 or
                    [self.robot.joint_names[i] for i in self._action_ids[:7]] != list(RIGHT_ARM) or
                    [self.robot.joint_names[i] for i in hand_ids] != list(RIGHT_HAND)):
                raise RLEnvironmentError("posture10 and hand7 effort mappings must be exact and disjoint")
            gravity = tensor(self.robot.data.gravity_compensation_forces).clone()
            hand = self._last_written_hand_effort.clone()
            full = torch.zeros(self.num_envs, 43, device=self.device, dtype=gravity.dtype)
            if (tuple(gravity.shape) != (self.num_envs, 43) or
                    tuple(hand.shape) != (self.num_envs, 7) or
                    not bool(torch.isfinite(gravity).all()) or not bool(torch.isfinite(hand).all())):
                # Clear all user effort before aborting; do not leave a prior
                # finite actuation target live after a nonfinite measurement.
                self.robot.set_joint_effort_target_index(target=full)
                self._last_written_full_effort = full.clone()
                self._last_written_hand_effort.zero_()
                raise RLEnvironmentError("fresh posture gravity and hand effort must be finite full43/hand7 tensors")
            caps = torch.tensor(declaration["feedforward_caps_nm"], device=gravity.device, dtype=gravity.dtype)
            original = torch.tensor([actuator_specs[name]["effort_limit_nm"] for name in names],
                                    device=gravity.device, dtype=gravity.dtype)
            hand_caps = torch.tensor(effort_contract["actuation"]["total_torque_caps_nm"],
                                     device=hand.device, dtype=hand.dtype)
            if (tuple(caps.shape) != (10,) or not bool(torch.isfinite(caps).all()) or
                    not bool(((caps > 0) & (caps <= original)).all()) or
                    not bool((hand.abs() <= hand_caps).all())):
                self.robot.set_joint_effort_target_index(target=full)
                self._last_written_full_effort = full.clone()
                self._last_written_hand_effort.zero_()
                raise RLEnvironmentError("posture FF caps must stay within original joint caps and hand effort within total hand caps")
            posture_ff = gravity[:, posture_ids].clamp(-caps, caps)
            full[:, posture_ids], full[:, hand_ids] = posture_ff, hand
            full[self._unsafe] = 0
            # This getter reports user actuation, not isolated implicit drive
            # torque. Read it before replacing targets for the next step.
            previous, previous_reason = None, None
            getter = getattr(self.robot.root_view, "get_dof_actuation_forces", None)
            if getter is None:
                previous_reason = "native_get_dof_actuation_forces_unavailable"
            else:
                try:
                    observed = tensor(getter()).clone()
                    if tuple(observed.shape) != (self.num_envs, 43) or not bool(torch.isfinite(observed).all()):
                        previous_reason = "native_user_actuation_measurement_invalid_shape_or_nonfinite"
                    else:
                        previous = observed
                except (AttributeError, RuntimeError, TypeError, NotImplementedError) as exc:
                    previous_reason = type(exc).__name__+": "+str(exc)
            native_step = int(self.sim.get_physics_step_count())
            self.robot.set_joint_effort_target_index(target=full)
            self._last_written_full_effort = full.clone()
            self._last_written_hand_effort = full[:, hand_ids].clone()
            self._posture_ff_update_count += 1
            self._posture_ff_last_sample = native_step
            self._write_control_probe({"event": "physics_full_effort_target", "command_id": self._control_command_id,
                "native_physics_step_count": native_step, "episode_id": self._episode_ids,
                "next_target_applies_after_completed_native_step": native_step,
                "native_gravity_compensation_all43_nm": gravity,
                "posture_joint_indices": posture_ids, "hand_joint_indices": hand_ids,
                "posture_gravity_feedforward_contract": declaration,
                "raw_posture_gravity_compensation_nm": gravity[:, posture_ids],
                "applied_posture_gravity_feedforward_nm": full[:, posture_ids],
                "next_hand_total_effort_target_nm": full[:, hand_ids],
                "next_full43_user_effort_target_nm": full,
                "native_previous_completed_step_user_actuation_nm": previous,
                "native_previous_user_actuation_unavailable_reason": previous_reason,
                "previous_measurement_in_current_episode": self._last_contact_episode_ids == self._episode_ids,
                "unsafe_latched": self._unsafe,
                "unused_other26_user_effort_targets_zero": True,
                "native_implicit_pd_values_unchanged": True,
                "combined_posture_pd_and_ff_total_not_certified_bounded": True,
                "native_user_actuation_is_not_isolated_implicit_drive_torque": True})

        def _clear_effort_targets(self, env_ids):
            """Clear all 43 user efforts at reset, including non-hand entries."""
            zero = torch.zeros(len(env_ids), 43, device=self.device,
                               dtype=self._last_written_full_effort.dtype)
            self.robot.set_joint_effort_target_index(target=zero, env_ids=env_ids)
            self._last_written_full_effort[env_ids] = 0
            self._last_written_hand_effort[env_ids] = 0

        def configure_effort_grasp(self, contract, *, domain_posture_guard=None, contact_transition=None):
            from .rl_effort_protocol import validate_effort_grasp_contract, validate_contact_domain_posture_guard
            from .rl_hand_control import BalancedEffortGraspController
            if not effort_model or validate_effort_grasp_contract(contract) != effort_contract:
                raise RLEnvironmentError("effort controller cannot silently change its physical contract")
            if self._effort_grasp is not None:
                raise RLEnvironmentError("effort controller cannot be replaced during a run")
            self._effort_domain_posture_guard = validate_contact_domain_posture_guard(domain_posture_guard)
            self._effort_contact_transition = contact_transition
            guard_arguments = {}
            if self._effort_domain_posture_guard is not None or contact_transition is not None or full_lift is not None:
                frames = effort_candidate["geometry_evidence"]["source_detail"]["stage_keyframes"]
                if preshape_resolution is not None:
                    frames=preshape_resolution["original_frames"]
                trajectory = torch.as_tensor([f.get("hand_target_rad", f.get("target_rad")) for f in frames],
                    device=self.device, dtype=tensor(self.robot.data.joint_pos).dtype)
                if tuple(trajectory.shape) != (6, 7) or not bool(torch.isfinite(trajectory).all()):
                    raise RLEnvironmentError("domain posture guard requires the pinned original six keyframes")
                tolerance = .025
                self._effort_actual_domain_lower = (trajectory.amin(0)-tolerance)[None].expand(self.num_envs, -1).clone()
                self._effort_actual_domain_upper = (trajectory.amax(0)+tolerance)[None].expand(self.num_envs, -1).clone()
                if self._effort_domain_posture_guard is not None:
                    guard_arguments["domain_posture_guard"] = self._effort_domain_posture_guard
            if contact_transition is not None:
                guard_arguments["contact_transition"] = contact_transition
            self._effort_grasp = BalancedEffortGraspController(torch,
                tensor(self.robot.data.joint_pos)[:, self._action_ids[7:]], self._episode_ids,
                parameters=contract["controller"]["parameters"], **guard_arguments)
            self._object_support_scale = torch.zeros(self.num_envs, device=self.device)
            self._object_support_episode_ids = self._episode_ids.clone()
            self._object_support_last_sample = None
            self._write_control_probe({"event": "effort_grasp_controller_installed", "contract": contract,
                "contact_domain_posture_guard": self._effort_domain_posture_guard,
                "contact_transition_control": contact_transition,
                "actual_domain_lower_hand_rad": getattr(self, "_effort_actual_domain_lower", None),
                "actual_domain_upper_hand_rad": getattr(self, "_effort_actual_domain_upper", None),
                "native_physics_step_count": int(self.sim.get_physics_step_count()),
                "episode_id": self._episode_ids, "drive_readback": self._effort_drive_readback,
                "material_readback": self._effort_material_readback, "optimizer_steps": 0})

        def set_effort_grasp_reference(self, **reference):
            required = {"nominal_hand_rad", "geometry_lower_hand_rad", "geometry_upper_hand_rad",
                        "closing_axis_hand7", "palm_reference_world_m", "palm_reference_quaternion_xyzw"}
            if self._effort_grasp is None or set(reference) != required:
                raise RLEnvironmentError("effort grasp needs the exact declared macro reference fields")
            self._effort_reference = {k: v.detach().clone() for k, v in reference.items()}
            self._effort_reference["episode_id"] = self._episode_ids.clone()

        def set_approach_plan_evidence(self, evidence):
            """Macro intent only; native measurements own the admission latch."""
            if approach_course is None or self._pending_physics_sample:
                raise RLEnvironmentError("Approach intent requires its complete control boundary")
            for key in ("kinematic_arrived","failed"):
                value=evidence.get(key)
                if (not torch.is_tensor(value) or value.dtype!=torch.bool or
                        tuple(value.shape)!=(self.num_envs,) or value.device!=self._episode_ids.device):
                    raise RLEnvironmentError("Approach intent must name per-env measured kinematics and planner status")
            if not torch.is_tensor(evidence.get("episode_id")) or not torch.equal(evidence["episode_id"],self._episode_ids):
                raise RLEnvironmentError("Approach intent has a stale episode")
            self._approach_plan_complete=evidence["kinematic_arrived"].detach().clone()
            self._approach_planner_failed=evidence["failed"].detach().clone()
            self._approach_intent_episode_ids=self._episode_ids.clone()

        def _effort_contact_geometry(self, hand):
            """Current native points -> link-origin shifted Jacobians, no old normals.

            Speculative points may steer control. They never become physical
            loaded points or alter the independent legacy acceptance formula.
            """
            jac = tensor(self.robot.data.body_link_jacobian_w)
            self._effort_relative_motion_raw = None
            if jac.shape[1:] != (self.robot.num_bodies-1, 6, 43):
                raise RLEnvironmentError("effort control requires fixed-base link-origin Jacobians")
            j = jac[:, [i-1 for i in self._effort_hand_body_ids]][:, :, :, self._action_ids[7:]]
            link = tensor(self.robot.data.body_pos_w)[:, self._effort_hand_body_ids]
            point = hand["point"][:, :7]
            offset = point-link[:, :, None]
            angular = j[:, :, None, 3:, :].transpose(-1, -2).expand(-1, -1, 32, -1, -1)
            correction = torch.cross(angular, offset[:, :, :, None].expand_as(angular), dim=-1).transpose(-1, -2)
            point_j = j[:, :, None, :3, :]+correction
            cube = tensor(self.cube.data.root_pos_w)
            n = hand["point_normal"][:, :7]
            toward_cube = cube[:, None, None]-point
            sign = torch.where((n*toward_cube).sum(-1) >= 0, 1., -1.)
            inward = n*sign[..., None]
            valid = hand["point_valid"][:, :7] & (hand["separation"][:, :7] <= 2*settings["contact_offset_m"])
            valid &= inward[:, :, :, 2].abs() <= settings["max_abs_normal_z"]
            norm = torch.linalg.vector_norm(inward, dim=-1)
            valid &= torch.isfinite(norm) & (norm > .5)
            inward = inward/norm.clamp_min(1e-12)[..., None]
            weights = (hand["point_force"][:, :7].abs()+.001)*valid
            a, centers, normals, availability, point_jacobians = [], [], [], [], []
            for first, last in ((0, 3), (3, 5), (5, 7)):
                w = weights[:, first:last]
                total = w.sum(dim=(1, 2))
                w = w/total.clamp_min(1e-12)[:, None, None]
                row = (inward[:, first:last, :, :, None]*point_j[:, first:last]).sum(-2)
                a.append((w[..., None]*row).sum(dim=(1, 2)))
                point_jacobians.append((w[..., None, None]*point_j[:, first:last]).sum(dim=(1, 2)))
                centers.append((w[..., None]*point[:, first:last]).sum(dim=(1, 2)))
                normals.append((w[..., None]*inward[:, first:last]).sum(dim=(1, 2)))
                availability.append(total > 0)
            self._effort_geometry = {"normal_jacobian": torch.stack(a, 1).detach().clone(),
                "point_jacobian_world_m": torch.stack(point_jacobians, 1).detach().clone(),
                "point_world_m": torch.stack(centers, 1).detach().clone(),
                "inward_normal_world": torch.stack(normals, 1).detach().clone(),
                "current_normal_available": torch.stack(availability, 1).detach().clone(),
                "episode_id": self._episode_ids.clone(),
                "native_step": int(self.sim.get_physics_step_count())}
            transition = getattr(self,"_effort_contact_transition",None)
            relative_viscous = (transition is not None and
                transition.get("version")=="g1_coordinated_contact_transition_v3_relative_viscous_preload")
            if transition is not None and transition.get("version") in (
                    "g1_coordinated_contact_transition_v2_separating_preload",
                    "g1_coordinated_contact_transition_v3_relative_viscous_preload"):
                try:
                    motion,raw = current_relative_contact_motion(torch,
                        link_jacobian_world=jac[:,[i-1 for i in self._effort_hand_body_ids]],
                        link_position_world=link,
                        joint_velocity_full43=tensor(self.robot.data.joint_vel),
                        point_world=point,inward_normal_world=inward,control_point_valid=valid,
                        physical_point_loaded=(hand["point_valid"][:,:7] &
                            (hand["point_force"][:,:7].abs()>0) & (hand["separation"][:,:7]<=0)),
                        point_weight=hand["point_force"][:,:7].abs()+.001,
                        cube_com_position_world=tensor(self.cube.data.root_com_pos_w),
                        cube_com_linear_velocity_world=tensor(self.cube.data.root_com_lin_vel_w),
                        cube_angular_velocity_world=tensor(self.cube.data.root_com_ang_vel_w),
                        **({"include_cube_point_jacobian":True} if relative_viscous else {}))
                    identity={"episode_id":self._episode_ids.clone(),
                        "native_step":int(self.sim.get_physics_step_count())}
                    self._effort_geometry["relative_contact_motion"]={**motion,**identity,
                        "available":True,"source":"current_full43_link_origin_Jqd_minus_Cube_COM_point_twist",
                        "positive_velocity_is_inward_closing":True,
                        "point_weight_semantics":"unchanged_abs_raw_Fn_plus_0p001_over_original_control_points",
                        "all_physical_loaded_points_must_have_current_motion_proof":True}
                    self._effort_relative_motion_raw={**raw,**identity}
                    if relative_viscous:
                        try:
                            mass43=tensor(self.robot.data.mass_matrix)
                            if tuple(mass43.shape)!=(point.shape[0],43,43):
                                raise ValueError("Native fixed-root mass matrix must be N by43 by43")
                            hand_ids=self._action_ids[7:]
                            def matching_native(value):
                                return tensor(value).to(device=mass43.device,dtype=mass43.dtype)
                            properties=getattr(self,"_effort_native_inertial_properties",None)
                            if properties is None:
                                properties={"robot_body_mass_kg":matching_native(self.robot.root_view.get_masses()).clone(),
                                    "robot_body_inertia_prim_column_major":matching_native(self.robot.root_view.get_inertias()).clone(),
                                    "robot_body_com_local_pose":matching_native(self.robot.root_view.get_coms()).clone(),
                                    "robot_armature_full43":matching_native(self.robot.root_view.get_dof_armatures()).clone(),
                                    "cube_mass_kg":matching_native(self.cube.root_view.get_masses()).clone(),
                                    "cube_inertia_prim_column_major":matching_native(self.cube.root_view.get_inertias()).clone()}
                                expected={"robot_body_mass_kg":(mass43.shape[0],self.robot.num_bodies),
                                    "robot_body_inertia_prim_column_major":(mass43.shape[0],self.robot.num_bodies,9),
                                    "robot_body_com_local_pose":(mass43.shape[0],self.robot.num_bodies,7),
                                    "robot_armature_full43":(mass43.shape[0],43),"cube_mass_kg":(mass43.shape[0],1),
                                    "cube_inertia_prim_column_major":(mass43.shape[0],9)}
                                if any(properties[key].shape!=shape for key,shape in expected.items()):
                                    raise ValueError("Native inertia/property buffers do not match the verified body/joint order")
                                self._effort_native_inertial_properties=properties
                                self._write_control_probe({"event":"native_relative_damping_inertial_readback",
                                    **identity,"read_only_static_native_properties":properties,
                                    "robot_body_names":self.robot.body_names,"robot_joint_names":self.robot.joint_names,
                                    "robot_hand_body_ids":self._effort_hand_body_ids,"robot_hand_joint_ids":hand_ids,
                                    "inertia_frame":"COM_inertia_expressed_in_body_prim_frame_column_major9",
                                    "properties_reused_only_with_unchanged_native_model":True,
                                    "physical_parameter_writes":0})
                            armature=properties["robot_armature_full43"]
                            full_j=torch.cat((torch.zeros_like(jac[:,:1]),jac),dim=1)
                            dynamic,mass_current,reconstruction=reconstruct_native_hand_inertia(torch,
                                link_jacobian_world_full43=full_j,
                                link_quaternion_xyzw=tensor(self.robot.data.body_quat_w),
                                body_mass_kg=properties["robot_body_mass_kg"],
                                body_inertia_prim_column_major=properties["robot_body_inertia_prim_column_major"],
                                body_com_local_pose=properties["robot_body_com_local_pose"],
                                native_mass_matrix_full43=mass43,native_armature_full43=armature,
                                hand_body_ids=self._effort_hand_body_ids,hand_joint_ids=hand_ids)
                            damping,damping_evidence=native_relative_damping_state(torch,
                                hand_mass_matrix=dynamic,
                                hand_armature=armature[:,hand_ids],
                                hand_mass_semantics_current=mass_current,
                                cube_mass_kg=properties["cube_mass_kg"],
                                cube_inertia_prim_column_major=properties["cube_inertia_prim_column_major"],
                                cube_link_quaternion_xyzw=matching_native(self.cube.data.root_link_quat_w),
                                cube_normal_point_jacobian=motion["cube_normal_point_jacobian"],
                                physical_point_count=motion["physical_loaded_point_count"],
                                relative_velocity_current=motion["relative_velocity_current"])
                            physical_points=(hand["point_valid"][:,:7] &
                                (hand["point_force"][:,:7].abs()>0) & (hand["separation"][:,:7]<=0))
                            rigid_point_counts=[]
                            for first,last,pad in ((0,3,2),(3,5,4),(5,7,6)):
                                rigid=[index for index in range(first,last) if index!=pad]
                                rigid_point_counts.append((valid[:,rigid] | physical_points[:,rigid]).sum((1,2)))
                            nonsoft=torch.stack(rigid_point_counts,dim=1)
                            damping["current"] &= nonsoft==0
                            damping_evidence.update(reconstruction)
                            damping_evidence.update(nonsoft_current_control_or_physical_point_count=nonsoft,
                                all_group_points_use_declared_soft_pad=nonsoft==0,
                                declared_soft_hand_body_indices=[2,4,6])
                            self._effort_geometry["relative_damping_state"]={**damping,**identity,
                                "available":True,"native_inertia_readback":damping_evidence}
                            self._effort_native_inertia_status={**identity,"available":True,
                                "classification_code":reconstruction["native_mass_classification_code"],
                                "hand_mass_semantics_current":mass_current,
                                "group_state_current":damping["current"],
                                "raw_sdk_minus_rigid_max_abs_kg_m2":reconstruction["raw_sdk_minus_rigid_max_abs_kg_m2"],
                                "raw_sdk_minus_dynamic_max_abs_kg_m2":reconstruction["raw_sdk_minus_dynamic_max_abs_kg_m2"]}
                            old_count_ids=getattr(self,"_effort_mass_counted_episode_ids",None)
                            unique=(torch.ones_like(self._episode_ids,dtype=torch.bool) if
                                getattr(self,"_effort_mass_counted_native_step",None)!=identity["native_step"] or old_count_ids is None
                                else old_count_ids!=self._episode_ids)
                            self._effort_valid_mass_environment_samples=getattr(self,
                                "_effort_valid_mass_environment_samples",torch.zeros_like(self._episode_ids))+(mass_current&unique).to(torch.int64)
                            self._effort_mass_counted_episode_ids=self._episode_ids.clone()
                            self._effort_mass_counted_native_step=identity["native_step"]
                            self._effort_relative_motion_raw["native_mass_matrix_full43"]=mass43.detach().clone()
                            self._effort_relative_motion_raw["native_armature_full43"]=armature.detach().clone()
                        except (AttributeError,TypeError,ValueError,RuntimeError) as exc:
                            self._effort_geometry["relative_damping_state"]={**identity,"available":False,
                                "unavailable_reason":type(exc).__name__+": "+str(exc),
                                "fallback":"original_unmodified_hand7_normal_feedback"}
                            self._effort_native_inertia_status={**identity,"available":False,
                                "unavailable_reason":type(exc).__name__+": "+str(exc)}
                except (AttributeError,TypeError,ValueError,RuntimeError) as exc:
                    # An optional motion proof cannot invalidate successful
                    # original force/geometry reads or fabricate separation.
                    self._effort_geometry["relative_contact_motion"]={"available":False,
                        "episode_id":self._episode_ids.clone(),
                        "native_step":int(self.sim.get_physics_step_count()),
                        "unavailable_reason":type(exc).__name__+": "+str(exc),
                        "fallback":"original_unmodified_hand7_normal_feedback"}

        def _object_weight_support_torque(self, *, valid, tracking, geometry, current):
            """Request object weight through fresh physical points, never an attachment.

            This is a desired static support wrench. The hand controller must
            still put it inside its common scale/caps and measured support gates.
            """
            declaration = effort_contract["actuation"]["object_weight_support"]
            changed = self._object_support_episode_ids != self._episode_ids
            self._object_support_scale[changed] = 0
            self._object_support_episode_ids = self._episode_ids.clone()
            native_step = int(self.sim.get_physics_step_count())
            gate = valid & tracking & ~self._unsafe & current.all(-1) & self._grip_locked
            gate &= preload_contact_ready(torch, self._last_metrics, self.num_envs, self.device)
            point_j = torch.zeros(self.num_envs, 3, 3, 7, device=self.device)
            if geometry is not None:
                point_j = geometry["point_jacobian_world_m"]
                if tuple(point_j.shape) != (self.num_envs, 3, 3, 7) or not bool(torch.isfinite(point_j).all()):
                    raise RLEnvironmentError("object support requires finite current native group point Jacobians")
                gate &= (geometry["episode_id"] == self._episode_ids) & (geometry["native_step"] == native_step)
            else:
                gate &= False
            self._object_support_scale = torch.where(gate, self._object_support_scale,
                                                      torch.zeros_like(self._object_support_scale))
            if self._object_support_last_sample != native_step:
                self._object_support_scale = torch.where(gate,
                    (self._object_support_scale+self.physics_dt/declaration["ramp_s"]).clamp(max=1),
                    self._object_support_scale)
            self._object_support_last_sample = native_step
            shares = torch.tensor(declaration["group_weight_shares"], device=self.device, dtype=point_j.dtype)
            up = torch.tensor(declaration["world_up_axis"], device=self.device, dtype=point_j.dtype)
            group_force = (self._object_support_scale[:, None, None]*declaration["object_mass_kg"]*
                           declaration["gravity_m_s2"]*shares[None, :, None]*up[None, None])
            torque = torch.einsum("ngci,ngc->ni", point_j, group_force)
            return torque, {"contract": declaration, "gate": gate,
                "ramp_scale": self._object_support_scale.clone(), "current_group_requested_force_world_n": group_force,
                "requested_hand_object_support_torque_before_common_cap_nm": torque,
                "request_is_not_measured_object_support_or_friction_certification": True}

        def _update_effort_grasp(self):
            ref = self._effort_reference
            if ref is None:
                raise RLEnvironmentError("missing explicit effort macro reference")
            native_step = int(self.sim.get_physics_step_count())
            ids = self._action_ids[7:]
            q = tensor(self.robot.data.joint_pos)[:, ids]
            geometry = self._effort_geometry
            fresh = (self._last_contact_episode_ids == self._episode_ids) & (ref["episode_id"] == self._episode_ids)
            fresh &= self._last_contact_native_step == native_step
            metrics = self._last_metrics
            valid = fresh & metrics.get("finite", torch.zeros_like(fresh)) & metrics.get("joint_safe", torch.zeros_like(fresh))
            valid &= metrics.get("table_pair_measurement_supported", torch.zeros_like(fresh))
            if geometry is None:
                a = torch.zeros(self.num_envs, 3, 7, device=self.device)
                current = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.device)
            else:
                geometry_fresh = (geometry["episode_id"] == self._episode_ids) & (geometry["native_step"] == native_step)
                current = geometry["current_normal_available"] & fresh[:, None] & geometry_fresh[:, None]
                a = geometry["normal_jacobian"]
            body_rotation = tensor(self.robot.data.body_quat_w)[:, self._palm_id]
            palm = tensor(self.robot.data.body_pos_w)[:, self._palm_id]+quat_apply(body_rotation, self._palm_local)
            rotation = quat_mul(body_rotation, self._palm_local_rotation)
            dot = (rotation*ref["palm_reference_quaternion_xyzw"]).sum(-1).abs().clamp(0, 1)
            tracking = ((torch.linalg.vector_norm(palm-ref["palm_reference_world_m"], dim=-1) <= .003) &
                        (2*torch.acos(dot) <= .05))
            physical = metrics.get("group_physical_loaded_contact_count", torch.zeros(self.num_envs, 3, device=self.device)) > 0
            hand_tracking=tracking
            approach_hand_tracking=None
            if approach_course is not None:
                from .rl_approach import approach_hand_posture_tracking
                hand_tracking,approach_hand_tracking=approach_hand_posture_tracking(torch,
                    original_tracking=tracking,admitted=self._approach.admitted,
                    approach_failed=self._approach.failed,fresh_valid=valid,
                    current_normal_available=current,physical_group_loaded=physical,
                    robot_cube_force_n=metrics.get("approach_robot_cube_contact_absolute_sum_n",
                        torch.full_like(q[:,0],float("inf"))),unsafe=self._unsafe|self._full_lift.failed)
            gravity = tensor(self.robot.data.gravity_compensation_forces)[:, ids]
            support_torque, support_evidence = self._object_weight_support_torque(
                valid=valid, tracking=tracking, geometry=geometry, current=current)
            guard_feedback = {}
            if (getattr(self, "_effort_domain_posture_guard", None) is not None or
                    getattr(self, "_effort_contact_transition", None) is not None):
                guard_feedback = {"contact_point_jacobian_world_m": geometry["point_jacobian_world_m"]
                    if geometry is not None else torch.zeros(self.num_envs, 3, 3, 7, device=self.device),
                    "actual_domain_lower_hand_rad": self._effort_actual_domain_lower,
                    "actual_domain_upper_hand_rad": self._effort_actual_domain_upper,
                    "physical_contact_safe": metrics.get("contact_safe", torch.zeros_like(valid))}
            relative_feedback = {}
            motion = geometry.get("relative_contact_motion") if geometry is not None else None
            if motion is not None and motion.get("available") is True:
                motion_fresh = (motion["episode_id"]==self._episode_ids) & (motion["native_step"]==native_step)
                motion_current = motion["relative_velocity_current"] & current & fresh[:,None] & motion_fresh[:,None]
                relative_feedback={"relative_velocity_current":motion_current,
                    "relative_inward_velocity_m_s":torch.where(motion_current,
                        motion["relative_inward_velocity_m_s"],torch.zeros_like(motion["relative_inward_velocity_m_s"])),
                    "maximum_current_point_relative_inward_velocity_m_s":torch.where(motion_current,
                        motion["maximum_current_point_relative_inward_velocity_m_s"],
                        torch.zeros_like(motion["maximum_current_point_relative_inward_velocity_m_s"]))}
            damping_feedback={}
            damping=geometry.get("relative_damping_state") if geometry is not None else None
            if (getattr(self,"_effort_contact_transition",None) is not None and
                    self._effort_contact_transition.get("version")=="g1_coordinated_contact_transition_v3_relative_viscous_preload" and
                    damping is not None and damping.get("available") is True):
                damping_fresh=(damping["episode_id"]==self._episode_ids)&(damping["native_step"]==native_step)
                damping_current=(damping["current"] & relative_feedback.get("relative_velocity_current",
                    torch.zeros_like(current)) & current & fresh[:,None] & damping_fresh[:,None])
                damping_feedback["relative_damping_state"]={key:damping[key] for key in (
                    "hand_mass_matrix","cube_inverse_spatial_inertia_world","cube_normal_point_jacobian",
                    "physical_point_count")}
                damping_feedback["relative_damping_state"]["current"]=damping_current
            failure_feedback={}
            if (getattr(self,"_keep_current_hand_force_on_full_failure",False) and
                    bool(self._full_lift.failed.any())):
                failure_feedback["freeze_reference_by_environment"]=self._full_lift.failed.clone()
            effort, evidence, unsafe = self._effort_grasp.update(sample_id=native_step,
                episode_id=self._episode_ids, dt=self.physics_dt, measured_hand_rad=q,
                measured_hand_velocity_rad_s=tensor(self.robot.data.joint_vel)[:, ids],
                nominal_hand_rad=ref["nominal_hand_rad"], gravity_hand_nm=gravity,
                inward_normal_jacobian=a, group_contact_valid=current,
                group_physical_loaded=physical & fresh[:, None],
                raw_group_force_n=metrics.get("group_normal_force_n", torch.zeros(self.num_envs, 3, device=self.device)),
                contact_measurement_valid=valid, tracking=hand_tracking, unsafe_latched=self._unsafe,
                joint_lower_hand_rad=self._limits[:, ids, 0], joint_upper_hand_rad=self._limits[:, ids, 1],
                geometry_lower_hand_rad=ref["geometry_lower_hand_rad"], geometry_upper_hand_rad=ref["geometry_upper_hand_rad"],
                nominal_velocity_limit_hand_rad_s=self._velocity[7:][None].expand_as(q),
                penetration_m=metrics.get("maximum_penetration_m", torch.zeros(self.num_envs, device=self.device)),
                closing_axis_hand7=ref["closing_axis_hand7"], object_support_torque_nm=support_torque,
                **guard_feedback, **relative_feedback, **damping_feedback, **failure_feedback)
            self._unsafe |= unsafe
            if (getattr(self,"_effort_contact_transition",None) is not None and
                    self._effort_contact_transition.get("version")=="g1_coordinated_contact_transition_v3_relative_viscous_preload"):
                damping_evidence=evidence.get("contact_transition",{}).get("relative_preload_damping",{})
                enabled=damping_evidence.get("actual_enabled_by_group")
                if torch.is_tensor(enabled) and enabled.shape==(self.num_envs,3) and enabled.dtype==torch.bool:
                    old_ids=getattr(self,"_effort_damping_counted_episode_ids",None)
                    unique=(torch.ones_like(self._episode_ids,dtype=torch.bool) if
                        getattr(self,"_effort_damping_counted_native_step",None)!=native_step or old_ids is None
                        else old_ids!=self._episode_ids)
                    self._effort_actual_relative_damping_group_samples=getattr(self,
                        "_effort_actual_relative_damping_group_samples",torch.zeros_like(enabled,dtype=torch.int64))+(enabled&unique[:,None]).to(torch.int64)
                    self._effort_damping_counted_episode_ids=self._episode_ids.clone()
                    self._effort_damping_counted_native_step=native_step
            self._last_written_hand_effort = effort.clone()
            self._effort_update_count += 1
            self._effort_last_sample = native_step
            predicted_wrench = None
            if geometry is not None:
                normal_force = geometry["inward_normal_world"]*evidence["group_force_command_n"][..., None]
                torque = torch.cross(geometry["point_world_m"]-tensor(self.cube.data.root_pos_w)[:, None], normal_force, dim=-1)
                predicted_wrench = torch.cat((normal_force.sum(1), torque.sum(1)), -1)
            self._write_control_probe({"event": "physics_hand_effort", "command_id": self._control_command_id,
                "native_physics_step_count": native_step, "force_measurement_native_step": self._last_contact_native_step,
                "episode_id": self._episode_ids, "force_measurement_episode_id": self._last_contact_episode_ids,
                "contact_measurement_current_episode": fresh, "measured_hand_rad": q,
                "next_hand_effort_target_nm": effort, "current_normal_geometry": geometry,
                "estimated_normal_only_object_wrench_before_torque_saturation": predicted_wrench,
                "object_weight_support_request": support_evidence,
                **({"approach_hand_posture_tracking":approach_hand_tracking} if approach_course is not None else {}),
                "wrench_estimate_excludes_friction_and_is_not_verified_balance": True,
                "evidence": evidence, "requires_unsafe": unsafe,
                "hand_position_drive_disabled": True, "torque_command_cap_is_not_physical_force_cap": True})

        def configure_physics_pinch(self, contract):
            from .rl_probe_protocol import physics_pinch_controller_contract
            from .rl_hand_control import PhysicsStepForceAdmittance
            expected = physics_pinch_controller_contract()
            if (contract != expected or self._control_probe_profile["name"] != "physics_pinch_pose_hold" or
                    not math.isclose(self.physics_dt, expected["parameters"]["physics_dt_s"], abs_tol=1e-12)):
                raise RLEnvironmentError("Physics pinch contract/profile/cadence mismatch")
            if self._physics_pinch is not None:
                raise RLEnvironmentError("Physics pinch cannot be replaced during a run")
            self._physics_pinch = PhysicsStepForceAdmittance(torch,
                self._last_written_hand_target, self._episode_ids,
                parameters=contract["parameters"])
            self._physics_pinch_contract = contract
            self._write_control_probe({"event": "physics_pinch_controller_installed",
                "contract": contract, "native_physics_step_count": int(self.sim.get_physics_step_count()),
                "episode_id": self._episode_ids, "physics_steps_executed": self._physics_samples})

        def set_physics_pinch_reference(self, *, nominal_hand_rad, geometry_lower_hand_rad,
                geometry_upper_hand_rad, closing_axis_hand7, palm_reference_world_m,
                palm_reference_quaternion_xyzw):
            if self._physics_pinch is None:
                raise RLEnvironmentError("Physics pinch is not installed")
            matrices = (nominal_hand_rad, geometry_lower_hand_rad, geometry_upper_hand_rad)
            if (any(tuple(x.shape) != (self.num_envs, 7) or not bool(torch.isfinite(x).all()) for x in matrices) or
                    bool((geometry_lower_hand_rad > geometry_upper_hand_rad).any()) or
                    tuple(closing_axis_hand7.shape) != (7,) or
                    tuple(palm_reference_world_m.shape) != (self.num_envs, 3) or
                    tuple(palm_reference_quaternion_xyzw.shape) != (self.num_envs, 4)):
                raise RLEnvironmentError("Malformed physics-pinch reference")
            self._physics_pinch_reference = {
                "nominal_hand_rad": nominal_hand_rad.detach().clone(),
                "geometry_lower_hand_rad": geometry_lower_hand_rad.detach().clone(),
                "geometry_upper_hand_rad": geometry_upper_hand_rad.detach().clone(),
                "closing_axis_hand7": closing_axis_hand7.detach().clone(),
                "palm_reference_world_m": palm_reference_world_m.detach().clone(),
                "palm_reference_quaternion_xyzw": palm_reference_quaternion_xyzw.detach().clone(),
                "episode_id": self._episode_ids.clone()}

        def _update_physics_pinch(self):
            reference = self._physics_pinch_reference
            if reference is None:
                raise RLEnvironmentError("Missing physics-pinch reference")
            native_step = int(self.sim.get_physics_step_count())
            ids = self._action_ids[7:]
            measured = tensor(self.robot.data.joint_pos)[:, ids]
            actual_applied = self._last_written_hand_target
            # A completed sample belongs to the reset epoch that generated it.
            # episode_length_buf increments only every33 steps and cannot test
            # first-macro freshness for the inner loop.
            fresh = ((self._last_contact_episode_ids == self._episode_ids) &
                     (reference["episode_id"] == self._episode_ids) &
                     (self._last_contact_native_step == native_step))
            body_rotation = tensor(self.robot.data.body_quat_w)[:, self._palm_id]
            palm_position = tensor(self.robot.data.body_pos_w)[:, self._palm_id] + quat_apply(body_rotation, self._palm_local)
            palm_rotation = quat_mul(body_rotation, self._palm_local_rotation)
            dot = (palm_rotation*reference["palm_reference_quaternion_xyzw"]).sum(-1).abs().clamp(0, 1)
            tracking = ((torch.linalg.vector_norm(palm_position-reference["palm_reference_world_m"], dim=-1) <= .003) &
                        (2*torch.acos(dot) <= .05))
            anchor = float(config["control"]["max_anchor_delta_rad"])
            per_step = self._velocity[7:]*self.physics_dt
            low = torch.maximum(torch.maximum(actual_applied-per_step, measured-anchor), self._limits[:, ids, 0])
            high = torch.minimum(torch.minimum(actual_applied+per_step, measured+anchor), self._limits[:, ids, 1])
            metrics = self._last_metrics
            valid = fresh & metrics.get("finite", torch.zeros_like(fresh)) & metrics.get("joint_safe", torch.zeros_like(fresh))
            valid &= metrics.get("table_pair_measurement_supported", torch.zeros_like(fresh))
            target, evidence, unsafe = self._physics_pinch.update(
                sample_id=native_step, episode_id=self._episode_ids, dt=self.physics_dt,
                measured_hand_rad=measured, applied_hand_rad=actual_applied,
                nominal_hand_rad=reference["nominal_hand_rad"],
                geometry_lower_hand_rad=reference["geometry_lower_hand_rad"],
                geometry_upper_hand_rad=reference["geometry_upper_hand_rad"],
                step_lower_hand_rad=low, step_upper_hand_rad=high,
                macro_lower_hand_rad=self._physics_pinch_macro_low,
                macro_upper_hand_rad=self._physics_pinch_macro_high,
                raw_group_force_n=metrics.get("group_normal_force_n", torch.zeros(self.num_envs, 3, device=self.device)),
                contact_measurement_valid=valid, tracking=tracking, unsafe_latched=self._unsafe,
                penetration_m=metrics.get("maximum_penetration_m", torch.zeros(self.num_envs, device=self.device)),
                closing_axis_hand7=reference["closing_axis_hand7"])
            self._targets[:, ids] = target
            self._unsafe |= unsafe
            self._physics_pinch_update_count += 1
            self._physics_pinch_last_sample = native_step
            self._write_control_probe({"event": "physics_hand_target", "command_id": self._control_command_id,
                "native_physics_step_count": native_step, "force_measurement_native_step": self._last_contact_native_step,
                "episode_id": self._episode_ids, "force_measurement_episode_id": self._last_contact_episode_ids,
                "contact_measurement_current_episode": fresh, "measured_hand_rad": measured,
                "previous_actually_written_hand_rad": actual_applied, "next_hand_target_rad": target,
                "macro_lower_hand_rad": self._physics_pinch_macro_low,
                "macro_upper_hand_rad": self._physics_pinch_macro_high,
                "native_2ms_lower_hand_rad": low, "native_2ms_upper_hand_rad": high,
                "force_normal_n": metrics.get("group_normal_force_n"),
                "tracking": tracking, "requires_unsafe": unsafe, "evidence": evidence,
                "target_bounds_are_not_actual_force_bounds": True})

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
            contact_result={"cube_total_force": cube_normal+friction.sum(1),
                    "table_normal": normal[:, 0], "table_fn": fn[:, 0], "table_loaded_count": loaded[:, 0],
                    "robot_normal": normal[:, 1:].sum(dim=1), "robot_fn": fn[:, 1:].sum(dim=1),
                    "maximum_penetration_m": depth.amax(dim=1),
                    "other_fn": fn[:, other].sum(dim=1), "other_loaded_count": loaded[:, other].sum(dim=1),
                    "unfiltered_normal_residual_n": result["unfiltered_normal_residual_n"],
                    "hand": result["hand"]}
            if approach_course is not None:
                bucket=friction_raw["bucket"]
                robot_patch=(bucket%width)!=0
                friction_abs=torch.zeros(self.num_envs,device=self.device,dtype=normal.dtype)
                friction_abs.index_add_(0,torch.div(bucket[robot_patch],width,rounding_mode="floor"),
                    torch.linalg.vector_norm(friction_raw["force"][robot_patch],dim=-1))
                contact_result["robot_raw_contact_absolute_sum_n"]=fn[:,1:].sum(1)+friction_abs
            return contact_result

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

        def _update_preshape_contact_latch(self,physical_point_count,measurement_valid):
            """Observe completed native samples; freeze only the next macro reference."""
            native=int(self.sim.get_physics_step_count())
            if native<=self._control_preshape_contact_last_native:
                return
            changed=self._control_preshape_contact_episode_ids!=self._episode_ids
            self._control_preshape_contact_seen[changed]=False
            self._control_preshape_first_contact_native[changed]=-1
            self._control_preshape_contact_episode_ids=self._episode_ids.clone()
            loaded=measurement_valid & (physical_point_count>0).any(-1)
            first=loaded & ~self._control_preshape_contact_seen
            self._control_preshape_first_contact_native=torch.where(first,
                torch.full_like(self._episode_ids,native),self._control_preshape_first_contact_native)
            self._control_preshape_contact_seen |= loaded
            self._control_preshape_contact_last_native=native

        def _sample_physics(self):
            active_effort = getattr(self, "_effort_grasp", None) is not None
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
            group_measurement_available = False
            try:
                x, y, z, w = rotation.unbind(-1)
                object_up_world = torch.stack((2*(x*z+w*y), 2*(y*z-w*x), 1-2*(x*x+y*y)), dim=-1)
                cube_contacts = self._cube_contact_values(object_up_world)
                if active_effort:
                    self._effort_contact_geometry(cube_contacts["hand"])
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
                group_measurement_available = True
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
            if approach_course is not None:
                native=int(self.sim.get_physics_step_count())
                self._approach_contact_last=self._approach_contact.sample_completed(native,self._episode_ids)
                path_available=torch.tensor(self._approach_contact_last["per_env_available"],device=self.device,dtype=torch.bool)
                path_clean=torch.tensor(self._approach_contact_last["per_env_clean"],device=self.device,dtype=torch.bool)
                if not bool(path_available.all()):
                    for i,available in enumerate(self._approach_contact_last["per_env_available"]):
                        if not available and self._approach_measurement_unavailable[i] is None:
                            self._approach_measurement_unavailable[i]=str(self._approach_contact_last["per_env_unavailable_reason"][i])
                body_quat=tensor(self.robot.data.body_quat_w)[:,self._palm_id]
                palm_offset=quat_apply(body_quat,self._palm_local)
                q_all=tensor(self.robot.data.joint_pos)
                hand=q_all[:,self._action_ids[7:]]
                original_domain=((hand>=self._effort_actual_domain_lower)&(hand<=self._effort_actual_domain_upper)).all(-1)
                whole_clean=(self._full_lift.whole_episode_force_clean & self._full_lift.whole_episode_original_domain_clean &
                    ~self._full_lift.failed & original_domain & torch.isfinite(groups).all(-1) &
                    (groups>=0).all(-1) & (groups<=1.5).all(-1))
                from .rl_approach_state import current_object_height_unsafe
                current_overshoot=current_object_height_unsafe(torch,clearance,position[:,2]-self._start_cube_z,
                    self._full_lift.main_lift_started,local_cap_m=settings["probe_clearance_max_m"],
                    local_tolerance_m=settings["probe_overshoot_tolerance_m"],full_cap_m=full_lift["actual_height_limit_m"])
                stale_intent=self._approach_intent_episode_ids!=self._episode_ids
                self._approach_last=self._approach.update(native_step=native,episode_ids=self._episode_ids,
                    q_action=q_all[:,self._action_ids],qd_action=tensor(self.robot.data.joint_vel)[:,self._action_ids],
                    palm_position=tensor(self.robot.data.body_pos_w)[:,self._palm_id]+palm_offset,
                    palm_quaternion=quat_mul(body_quat,self._palm_local_rotation),cube_position=position,cube_quaternion=rotation,
                    env_origins=self.scene.env_origins,measurement_valid=finite & bool(group_measurement_available) & path_available,
                    path_collision_clean=path_clean,
                    robot_cube_force_n=(cube_contacts["robot_raw_contact_absolute_sum_n"] if group_measurement_available else torch.full_like(depth,float("inf"))),
                    mechanical_safe=joint_safe,unsafe=self._unsafe | self._probe_overshoot | current_overshoot |
                        ~finite | (depth>settings["max_penetration_m"]),
                    whole_task_clean=whole_clean,plan_complete_requested=self._approach_plan_complete & ~stale_intent,
                    planner_failed=self._approach_planner_failed | stale_intent,full_success=self._full_lift.full_success)
                if bool(self._approach_last["new_admission"].any()):
                    self._write_control_probe({"event":"arm_home_measured_admission",**self._approach_last,
                        "measured_joint_position_all43_rad":q_all,
                        "measured_joint_velocity_all43_rad_s":tensor(self.robot.data.joint_vel),
                        "palm_position_world_m":tensor(self.robot.data.body_pos_w)[:,self._palm_id]+palm_offset,
                        "palm_quaternion_xyzw":quat_mul(body_quat,self._palm_local_rotation),
                        "cube_position_world_m":position,"cube_quaternion_xyzw":rotation,
                        "full_lift_state_before_current_sample_update":self._full_lift.metrics(),
                        "no_physical_state_write_no_reset_no_history_clear":True})
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
            if approach_course is not None:
                support &= self._approach.admitted & ~self._approach.failed
            self._support_count = torch.where(support, self._support_count + 1, 0)
            self._max_support_samples = torch.maximum(self._max_support_samples, self._support_count)
            qualified_now = self._support_count >= consecutive_window_samples(settings["support_window_s"], self.physics_dt)
            self._support_qualified = qualified_now
            lift_height = position[:, 2] - self._start_cube_z
            probe_limit = settings["probe_clearance_max_m"] + settings["probe_overshoot_tolerance_m"]
            if getattr(self, "_full_lift", None) is not None:
                probe_limit = torch.where(self._full_lift.main_lift_started,
                    torch.full_like(clearance, full_lift["actual_height_limit_m"]),
                    torch.full_like(clearance, probe_limit))
            self._probe_overshoot |= (clearance > probe_limit) | (lift_height > probe_limit)
            self._unsafe |= ~finite | ~joint_safe | self._probe_overshoot | (depth > settings["max_penetration_m"])
            # This version terminates at qualified 2--6 mm support. It never
            # executes or claims the original 0.1 m lift / 0.5 s carry gate.
            self._local_success = qualified_now & ~self._unsafe
            if hasattr(self,"_control_preshape_contact_seen"):
                self._update_preshape_contact_latch(group_physical,finite & bool(group_measurement_available))
            full_lift_metrics = {}
            if getattr(self, "_full_lift", None) is not None:
                hand = tensor(self.robot.data.joint_pos)[:, self._action_ids[7:]]
                original_domain = (torch.isfinite(hand).all(-1) &
                    (hand >= self._effort_actual_domain_lower).all(-1) &
                    (hand <= self._effort_actual_domain_upper).all(-1))
                was_failed = self._full_lift.failed.clone()
                full_lift_metrics = self._full_lift.update(
                    # Successful raw hand/Cube data can witness zero hand
                    # force while the separate Table channel is still waiting
                    # for its first positive-load capability witness. Table
                    # capability remains mandatory in support/admission.
                    measurement_valid=finite & bool(group_measurement_available),
                    group_force_n=groups, original_hand_domain=original_domain,
                    local_support_qualified=self._local_success, support_current=support,
                    clearance_m=clearance, object_lift_m=lift_height, unsafe=(self._unsafe |
                        self._approach.failed if approach_course is not None else self._unsafe))
                newly_failed = self._full_lift.failed & ~was_failed
                self._full_lift_failure_arm_target[newly_failed] = tensor(
                    self.robot.data.joint_pos)[newly_failed][:, self._action_ids[:7]]
                if not getattr(self,"_keep_current_hand_force_on_full_failure",False):
                    self._unsafe |= self._full_lift.failed
                full_lift_metrics["original_hand_domain_current"] = original_domain
                full_lift_metrics["raw_group_measurement_valid_current"] = torch.full_like(
                    finite, bool(group_measurement_available)) & finite
                full_lift_metrics["full_lift_failure_arm_target_held"] = self._full_lift.failed.clone()
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
            if getattr(self, "_full_lift", None) is not None:
                self._last_metrics.update(full_lift_metrics)
                self._last_metrics["success"] = self._full_lift.full_success
            if approach_course is not None:
                self._last_metrics.update(approach_admitted=self._approach.admitted.clone(),
                    approach_failed=self._approach.failed.clone(),
                    approach_admission_native=self._approach.admission_native.clone(),
                    approach_first_failure_native=self._approach.first_failure_native.clone(),
                    approach_first_failure_code=self._approach.first_failure_code.clone(),
                    approach_robot_cube_contact_absolute_sum_n=(cube_contacts["robot_raw_contact_absolute_sum_n"]
                        if group_measurement_available else torch.full_like(depth,float("inf"))),
                    whole_path_collision_clean=self._approach.collision_clean.clone(),
                    whole_approach_object_clean=self._approach.object_clean.clone())
            self._last_contact_episode_ids = self._episode_ids.clone()
            self._last_contact_native_step = int(self.sim.get_physics_step_count())
            if self._physics_pinch is not None or active_effort:
                self._write_control_probe({"event": "physics_hand_measurement",
                    "command_id": self._control_command_id,
                    "native_physics_step_count": self._last_contact_native_step,
                    "episode_id": self._episode_ids, "episode_control_steps": self.episode_length_buf,
                    "measured_hand_rad": tensor(self.robot.data.joint_pos)[:, self._action_ids[7:]],
                    "measured_hand_velocity_rad_s": tensor(self.robot.data.joint_vel)[:, self._action_ids[7:]],
                    **({"measured_arm_rad":tensor(self.robot.data.joint_pos)[:,self._action_ids[:7]],
                        "measured_arm_velocity_rad_s":tensor(self.robot.data.joint_vel)[:,self._action_ids[:7]],
                        "actually_written_arm_position_target_rad":self._last_written_arm_target,
                        "arm_target_write_episode_id":self._last_written_arm_target_episode_ids,
                        "arm_target_write_applies_to_native_step":self._last_written_arm_target_next_native_step,
                        "arm_target_measurement_scope":"completed_native_sample_and_last_successful_position_target_buffer_write_not_sdk_target_readback"}
                       if getattr(self,"_native_arm_interpolation",False) else {}),
                    "actually_written_hand_target_rad": self._last_written_hand_target,
                    "actually_written_hand_effort_target_nm": self._last_written_hand_effort if active_effort else None,
                    "hand_position_target_active": not active_effort,
                    "cube_position_world_m": position if active_effort else None,
                    "cube_linear_velocity_world_m_s": velocity if active_effort else None,
                    "cube_contact_normal_plus_declared_partner_friction_world_n": cube_contacts.get("cube_total_force") if active_effort and contact_available else None,
                    "metrics": self._last_metrics, "unsafe_latched": self._unsafe,
                    **({"approach_native":self._approach_last,"approach_contact_witness":self._approach_contact_last,
                        "measured_joint_position_all43_rad":tensor(self.robot.data.joint_pos),
                        "measured_joint_velocity_all43_rad_s":tensor(self.robot.data.joint_vel),
                        "palm_position_world_m":tensor(self.robot.data.body_pos_w)[:,self._palm_id]+quat_apply(
                            tensor(self.robot.data.body_quat_w)[:,self._palm_id],self._palm_local),
                        "palm_quaternion_xyzw":quat_mul(tensor(self.robot.data.body_quat_w)[:,self._palm_id],self._palm_local_rotation),
                        "cube_quaternion_xyzw":rotation,"environment_origins_world_m":self.scene.env_origins}
                       if approach_course is not None else {}),
                    "sample_scope": "every_completed_2ms_sample_including_terminal_pre_reset_no_force_decimation"})
            self._physics_samples += 1
            if self._record and active_effort:
                self._record_bounded_effort_raw(state)
            if self._record and not active_effort:
                count = min(8, self.num_envs)
                row = {"event": "post_physics", "physics_sample": self._physics_samples,
                       "native_physics_step_count": int(self.sim.get_physics_step_count()),
                       "actual_physics_dt_s": float(self.sim.get_physics_dt()),
                       "episode_id": self._episode_ids[:count].detach().cpu().tolist(),
                       "env_ids": list(range(count)), "force_unavailable": self._force_unavailable,
                       "qualification_supported": self._qualification_supported,
                       "table_contact_channel": self._table_contact_channel.snapshot(),
                       "main_lift_authorized": (self._full_lift.main_lift_authorized if getattr(self, "_full_lift", None) is not None else False),
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

        def _record_bounded_effort_raw(self, state):
            """Retain bounded raw windows; compact physical evidence stays full-rate."""
            specification = effort_contract["logging"]
            physical = self._last_metrics["group_physical_loaded_contact_count"] > 0
            new_epoch = (torch.ones_like(self._episode_ids, dtype=torch.bool) if self._effort_raw_episode_ids is None
                         else self._effort_raw_episode_ids != self._episode_ids)
            if self._effort_raw_loss_seen is None:
                self._effort_raw_loss_seen = torch.zeros_like(new_epoch)
            self._effort_raw_loss_seen[new_epoch] = False
            loss = torch.zeros_like(new_epoch)
            if self._effort_previous_physical is not None:
                loss = (self._effort_previous_physical & ~physical).any(-1) & ~new_epoch & ~self._effort_raw_loss_seen
            peak = float(self._last_metrics["group_normal_force_n"].amax().item())
            peak_trigger = peak > max(1.5, self._effort_raw_max_force+.05)
            trigger = bool(new_epoch.any() or loss.any() or peak_trigger)
            reasons = (["reset_initialization"] if bool(new_epoch.any()) else [])
            if bool(loss.any()):
                reasons.append("physical_contact_loss_first_per_episode")
            if peak_trigger:
                reasons.append("force_above_acceptance_cap_new_run_peak")
            if approach_course is not None:
                admission_event=bool(self._approach_last["new_admission"].any())
                failure_event=bool((self._approach.first_failure_native==int(self.sim.get_physics_step_count())).any())
                if admission_event:reasons.append("arm_home_measured_admission")
                if failure_event:reasons.append("arm_home_first_failure")
                trigger |= admission_event or failure_event
            self._effort_raw_loss_seen |= loss
            self._effort_raw_episode_ids = self._episode_ids.clone()
            self._effort_previous_physical = physical.clone()
            self._effort_raw_max_force = max(peak, self._effort_raw_max_force)
            position, rotation, velocity, angular, relative, relative_rotation, _ = state
            row = {"event": "post_physics", "physics_sample": self._physics_samples,
                "native_physics_step_count": int(self.sim.get_physics_step_count()),
                "actual_physics_dt_s": float(self.sim.get_physics_dt()),
                "episode_id": self._episode_ids, "env_ids": list(range(self.num_envs)),
                "force_unavailable": self._force_unavailable, "qualification_supported": self._qualification_supported,
                "metrics": self._last_metrics, "joint_position_rad": tensor(self.robot.data.joint_pos),
                "joint_velocity_rad_s": tensor(self.robot.data.joint_vel), "action_origin": self._action_origin,
                "hand_effort_target_nm": self._last_written_hand_effort,
                "hand_position_targets_inactive": True, "cube_position_world_m": position,
                "cube_quaternion_xyzw": rotation, "cube_linear_velocity_world_m_s": velocity,
                "cube_angular_velocity_world_rad_s": angular, "cube_position_in_palm_m": relative,
                "cube_rotation_in_palm_xyzw": relative_rotation, "raw_contacts": self._raw_contact_tensors,
                "main_lift_authorized": (self._full_lift.main_lift_authorized if getattr(self, "_full_lift", None) is not None else False), "raw_capture_reasons": reasons}
            relative_raw=getattr(self,"_effort_relative_motion_raw",None)
            if (relative_raw is not None and relative_raw["native_step"]==int(self.sim.get_physics_step_count()) and
                    torch.equal(relative_raw["episode_id"],self._episode_ids)):
                row["relative_contact_motion_raw_reconstruction"]=relative_raw
            if getattr(self,"_native_arm_interpolation",False):
                row["actually_written_arm_position_target_rad"]=self._last_written_arm_target
                row["arm_target_schedule"]=full_lift["arm_target_schedule"]
            if approach_course is not None:
                row.update(approach_native=self._approach_last,approach_contact_witness=self._approach_contact_last)
            def frozen(value):
                if torch.is_tensor(value):
                    return value.detach().clone()
                if isinstance(value, dict):
                    return {k: frozen(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [frozen(v) for v in value]
                return value
            snapshot = frozen(row)
            def write(sample):
                native = sample["native_physics_step_count"]
                if native <= self._effort_raw_last_written_native or self._effort_raw_samples_written >= specification["raw_max_samples"]:
                    return
                self._write_evaluation_row(_probe_payload(torch, sample))
                self._effort_raw_samples_written += 1
                self._effort_raw_last_written_native = native
            if trigger:
                if (self._effort_raw_triggers < specification["raw_max_trigger_windows"] and
                        self._effort_raw_samples_written < specification["raw_max_samples"]):
                    self._effort_raw_triggers += 1
                    for previous in self._effort_raw_history:
                        write(previous)
                    self._effort_raw_post_remaining = max(self._effort_raw_post_remaining, specification["raw_post_steps"]+1)
                else:
                    self._effort_raw_omitted_triggers += 1
            if self._effort_raw_post_remaining > 0:
                write(snapshot)
                self._effort_raw_post_remaining -= 1
            self._effort_raw_history.append(snapshot)
            self._effort_raw_samples_omitted = self._physics_samples-self._effort_raw_samples_written

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
                    "native_joint_load_measurement": self._control_probe_joint_load("post_physics_pre_reset"),
                    "native_self_pair_measurement": self._control_probe_self_pair("post_physics_pre_reset"),
                    "main_lift_authorized": (self._full_lift.main_lift_authorized.clone() if getattr(self, "_full_lift", None) is not None else False), "pre_reset": True}
                if getattr(self,"_native_arm_interpolation",False):
                    self._control_transition_snapshot["macro_endpoint_action_rad"]=self._targets.detach().clone()
                    actual_target=self._targets.detach().clone()
                    actual_target[:,self._action_ids[:7]]=self._last_written_arm_target
                    self._control_transition_snapshot["applied_target_rad"]=actual_target
                    self._control_transition_snapshot["arm_target_source"]="last_successful_native_position_target_write"
            if self._standalone_video:
                self._capture_evaluation_frame()
            dropped = tensor(self.cube.data.root_pos_w)[:, 2] < 0.70
            success = self._local_success if getattr(self, "_full_lift", None) is None else self._full_lift.full_success
            failed = self._full_lift.failed if getattr(self,"_full_lift",None) is not None else torch.zeros_like(self._unsafe)
            terminated = self._unsafe | success | failed | dropped
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
            if self._full_lift is not None:
                fields.update(self._full_lift.metrics())
            field_batch = {name: value[terminal_ids].detach().cpu().tolist()
                           for name, value in fields.items()} if terminal_indices else {}
            base_measurement_unavailable = self._force_unavailable or self._table_contact_channel.snapshot()["unavailable_reason"]
            for offset, index in enumerate(terminal_indices):
                measurement_unavailable=base_measurement_unavailable
                if approach_course is not None:
                    measurement_unavailable=measurement_unavailable or self._approach_measurement_unavailable[index]
                metrics = {name: _json_finite(values[offset]) for name, values in metric_batch.items()}
                episode = {name: values[offset] for name, values in field_batch.items()}
                gates = {name: value for name, value in metrics.items() if isinstance(value, bool)}
                failure = []
                if self._force_unavailable:
                    failure.append("contact_measurement_unavailable")
                if episode["probe_overshoot"]:
                    failure.append("full_lift_or_local_height_bound_exceeded" if self._full_lift is not None else
                                   "local_probe_exceeded_6mm_before_qualification")
                if episode["unsafe"]:
                    failure.append("unsafe_physical_state")
                if not episode["support_qualified"]:
                    failure.append("continuous_support_not_qualified")
                if not episode["local_success"]:
                    failure.append("local_support_window_not_qualified")
                if self._full_lift is not None:
                    if episode["local_support_qualified"]:
                        failure = [reason for reason in failure if reason not in (
                            "local_support_window_not_qualified", "continuous_support_not_qualified")]
                    for key, reason in (("whole_episode_force_clean", "whole_episode_1p5n_force_limit_failed"),
                                        ("whole_episode_original_domain_clean", "whole_episode_original_hand_domain_failed"),
                                        ("full_success", "full_10cm_500ms_hold_not_qualified")):
                        if not episode[key]:
                            failure.append(reason)
                    if episode["main_lift_support_lost"]:
                        failure.append("main_lift_original_support_gate_lost")
                if approach_course is not None:
                    if not metrics.get("approach_admitted",False):failure.append("arm_home_measured_entry_not_admitted")
                    if metrics.get("approach_failed",False):failure.append("arm_home_path_or_entry_failed")
                    if self._approach_measurement_unavailable[index]:failure.append("arm_home_path_measurement_unavailable")
                if not metrics.get("no_positive_gap_hand_load", False):
                    failure.append("positive_gap_hand_normal_load_or_measurement_unavailable")
                if not metrics.get("friction_partner_manifold_certified", False):
                    failure.append("friction_partner_normal_manifold_uncertified")
                self._outcomes.append({"env_id": index, "episode_id": int(episode["episode_id"]),
                                       "seed": int(self._evaluation_seed if self._eval_mode else train_seed),
                                       "rollout_domain": ("physx_high_wait_vertical_approach_validation" if approach_course is not None else
                                           "physx_precontact_validation" if self._eval_mode else "physx_precontact_training"),
                                       "layout_id": ("high_wait_green15_settled_seed_jitter" if approach_course is not None else "contact_entry_seed_jitter"),
                                       "validation_offset_index": index % len(config["rl_teacher"]["validation_offsets_m"]) if self._eval_mode else None,
                                       "validation_offset_m": config["rl_teacher"]["validation_offsets_m"][index % len(config["rl_teacher"]["validation_offsets_m"])] if self._eval_mode else None,
                                       "elapsed_steps": int(episode["elapsed_steps"]),
                                       "sim_time_s": float(episode["elapsed_steps"]*self.step_dt),
                                       "episode_return": float(episode["episode_return"]),
                                       "success": None if measurement_unavailable else bool(episode[
                                           "local_success" if self._full_lift is None else "full_success"]),
                                       "local_success": False if measurement_unavailable else bool(episode["local_success"]),
                                       "qualification_supported": self._qualification_supported and (approach_course is None or self._approach_measurement_unavailable[index] is None),
                                       "unavailable_reason": measurement_unavailable, "gates": gates, "metrics": metrics,
                                       "max_clearance_m": float(episode["max_clearance_m"]),
                                       "maximum_support_samples": int(episode["maximum_support_samples"]),
                                       "failure_reasons": failure,
                                       **({key: episode[key] for key in self._full_lift.metrics()}
                                          if self._full_lift is not None else {}),
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
            self._last_written_hand_target[env_ids] = q[:, self._action_ids[7:]]
            if effort_model:
                self._clear_effort_targets(env_ids)
            self._last_contact_episode_ids[env_ids] = -1
            self._actions[env_ids] = 0
            self._previous_actions[env_ids] = 0
            self._episode_ids[env_ids] += 1
            if hasattr(self,"_control_preshape_contact_seen"):
                self._control_preshape_contact_seen[env_ids]=False
                self._control_preshape_first_contact_native[env_ids]=-1
                self._control_preshape_contact_episode_ids[env_ids]=self._episode_ids[env_ids]
            self._episode_return[env_ids] = 0
            self._support_count[env_ids] = 0
            self._support_qualified[env_ids] = False
            self._local_success[env_ids] = False
            if self._full_lift is not None:
                self._full_lift.reset(env_ids)
                self._full_lift_failure_arm_target[env_ids] = q[:, self._action_ids[:7]]
            if getattr(self,"_native_arm_interpolation",False):
                self._reset_native_arm_schedule(env_ids,q[:,self._action_ids[:7]])
            self._unsafe[env_ids] = False
            self._probe_overshoot[env_ids] = False
            self._grip_locked[env_ids] = False
            self._start_cube_z[env_ids] = cube[:, 2]
            self._expected_reset_cube[env_ids] = cube
            self._start_palm_z[env_ids] = float("nan")
            if approach_course is not None:
                # Predeclared grasp target datum, never the high home palm and
                # never recaptured to conceal an actual entry tracking error.
                self._start_palm_z[env_ids]=float(approach_resource["pregrasp_palm_position_world_m"][2])+origins[:,2]
                self._approach.reset(env_ids,self._episode_ids[env_ids],int(self.sim.get_physics_step_count()),cube)
                self._approach_contact.reset(env_ids,self._episode_ids[env_ids],int(self.sim.get_physics_step_count()))
                self._approach_plan_complete[env_ids]=False
                self._approach_planner_failed[env_ids]=False
                self._approach_intent_episode_ids[env_ids]=self._episode_ids[env_ids]
                for i in env_ids.detach().cpu().tolist():self._approach_measurement_unavailable[i]=None
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
            if video_recording is not None and self._declared_video_streams:
                reset_ids=set(env_ids.detach().cpu().tolist())
                for camera_env_id in video_recording["env_ids"]:
                    if camera_env_id in reset_ids:
                        self._write_declared_video_event({"event":"episode_reset","env_id":camera_env_id,
                            "episode_id":int(self._episode_ids[camera_env_id].item()),
                            "next_frame_index":self._video_status["frames"],
                            "native_physics_step_count":int(self.sim.get_physics_step_count()),
                            "sim_time_s":int(self.sim.get_physics_step_count())*float(self.sim.get_physics_dt()),
                            "profile_name":self._control_probe_profile["name"]},env_ids=[camera_env_id])

        def pop_episode_outcomes(self):
            result, self._outcomes = self._outcomes, []
            return result

        def export_collision_geometry_audit(self, declaration):
            """Export this stage's requested collision representation before task steps.

            The cooking interface is not a getter for the attached GPU shape's
            cached vertices. Keep that limitation with the exported identity.
            """
            if not effort_model or output is None or self._physics_samples or self._pending_physics_sample:
                raise RLEnvironmentError("collision geometry export requires the effort model before task physics")
            route = declaration.get("route", "same_stage_convex_request_v1")
            if route == "live_surface_queries_v2":
                from .rl_collision_geometry import export_collision_surface_audit as export_collision_geometry
            else:
                from .rl_collision_geometry import export_collision_geometry
            stage = get_current_stage()
            def properties():
                values = []
                for name, view, methods in (
                    ("Robot", self.robot.root_view, ("get_dof_stiffnesses", "get_dof_dampings",
                        "get_dof_max_forces", "get_dof_friction_properties", "get_contact_offsets", "get_rest_offsets")),
                    ("Cube", self.cube.root_view, ("get_contact_offsets", "get_rest_offsets")),
                    ("Table", self.table.root_view, ("get_contact_offsets", "get_rest_offsets"))):
                    for method in methods:
                        v = tensor(getattr(view, method)()).detach().clone().cpu()
                        if not bool(torch.isfinite(v).all()):
                            raise RLEnvironmentError("nonfinite native geometry-observation property: "+name+"/"+method)
                        values.append({"body": name, "getter": method, "shape": list(v.shape),
                            "values": v.tolist()})
                layer_records = [{"identifier": layer.identifier,
                    "sha256": hashlib.sha256(layer.ExportToString().encode()).hexdigest()}
                    for layer in stage.GetLayerStack()]
                encoded = json.dumps(values, sort_keys=True, allow_nan=False).encode()
                return {"native_drive_friction_shape_offsets_sha256": hashlib.sha256(encoded).hexdigest(),
                    "native_getter_shapes": [{k: v for k, v in row.items() if k != "values"} for row in values],
                    "usd_local_layer_stack": layer_records}
            before_properties = properties()
            before_step = int(self.sim.get_physics_step_count())
            before_q = tensor(self.robot.data.joint_pos).detach().clone()
            before_qd = tensor(self.robot.data.joint_vel).detach().clone()
            def cube_state():
                return torch.cat([tensor(getattr(self.cube.data, name)) for name in
                    ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w")], dim=-1)
            before_cube = cube_state().detach().clone()
            before_positions = tensor(self.robot.data.body_pos_w).detach().clone()
            before_rotations = tensor(self.robot.data.body_quat_w).detach().clone()
            roots = list(self.robot.root_view.prim_paths)
            bodies = []
            for env_id in declaration["env_ids"]:
                root_path = roots[env_id]
                if root_path != f"/World/envs/env_{env_id}/Robot/root_joint":
                    raise RLEnvironmentError("collision export native articulation path is not the declared Robot root")
                for name in declaration["body_names"]:
                    index = self.robot.body_names.index(name)
                    bodies.append({"env_id": env_id, "body_name": name,
                        "body_path": root_path.rsplit("/", 1)[0]+"/"+name,
                        "native_articulation_prim_path": root_path, "native_link_index": index,
                        "current_position_world_m": before_positions[env_id, index].cpu().tolist(),
                        "current_quaternion_xyzw": before_rotations[env_id, index].cpu().tolist()})
            extra = {}
            if route == "live_surface_queries_v2":
                extra["scene_query_support"] = {
                    "enabled": bool(self.sim.cfg.enable_scene_query_support),
                    "source": "resolved SimulationCfg readback; usable query coverage checked by actual hits, not a compiled manager getter",
                    "protocol": "g1_live_scene_query_instrumentation_v2"}
            result = export_collision_geometry(stage, bodies=bodies,
                output_path=output/"collision_geometry_representation.json", **extra)
            after_properties = properties()
            properties_unchanged = before_properties == after_properties
            unchanged = (int(self.sim.get_physics_step_count()) == before_step and
                torch.equal(before_q, tensor(self.robot.data.joint_pos)) and
                torch.equal(before_qd, tensor(self.robot.data.joint_vel)) and
                torch.equal(before_cube, cube_state()) and
                torch.equal(before_positions, tensor(self.robot.data.body_pos_w)) and
                torch.equal(before_rotations, tensor(self.robot.data.body_quat_w)))
            result.update(native_step_before=before_step,
                          geometry_observation_route=route,
                          native_step_after=int(self.sim.get_physics_step_count()),
                          current_state_unchanged=bool(unchanged), task_physics_steps=self._physics_samples,
                          property_snapshot_before=before_properties, property_snapshot_after=after_properties,
                          observed_native_properties_and_usd_layers_unchanged=properties_unchanged,
                          property_scope="fresh native drives/friction/shape offsets and USD local layer contents; not a getter of attached cooked vertices or all compiled material/filter internals")
            if (not unchanged or not properties_unchanged or
                    (declaration["require_complete"] and result.get("complete") is not True)):
                raise RLEnvironmentError("collision geometry export unavailable or changed current state: "+json.dumps(result))
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
                   "action_joint_ids": list(self._action_ids), "main_lift_authorized": (self._full_lift.main_lift_authorized if getattr(self, "_full_lift", None) is not None else False),
                   "native_joint_load_readback": self._native_joint_load_diagnostic.readback,
                   "native_self_pair_readback": self._native_self_pair_diagnostic.summary()}
            self._write_control_probe(row)
            self._write_evaluation_row(row)
            if video_recording is not None:
                self._write_declared_video_event({**row,"profile_name":selected["name"]})
            if self._video_manifest:
                self._video_manifest.write(json.dumps(_json_finite(_probe_payload(torch,
                    {**row, "profile_name": selected["name"]})), allow_nan=False)+"\n")
                self._video_manifest.flush()
            return dict(selected)

        def _control_probe_joint_load(self, phase):
            if not self._control_probe_enabled:
                raise RLEnvironmentError("joint-load diagnostics require control_probe")
            return self._native_joint_load_diagnostic.snapshot(
                self.sim.get_physics_step_count(), self._episode_ids, self.episode_length_buf, phase=phase)

        def _control_probe_self_pair(self, phase):
            if not self._control_probe_enabled:
                raise RLEnvironmentError("self-pair diagnostics require control_probe")
            return self._native_self_pair_diagnostic.snapshot(
                self.sim.get_physics_step_count(), self._episode_ids, self.episode_length_buf, phase=phase)

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
                          native_physics_step_count=int(self.sim.get_physics_step_count()),
                          native_joint_load_measurement=self._control_probe_joint_load("current_episode_control_state"),
                          native_self_pair_measurement=self._control_probe_self_pair("current_episode_control_state"))
            if hasattr(self,"_control_preshape_contact_seen"):
                result.update(control_preshape_contact_seen=self._control_preshape_contact_seen.clone(),
                    control_preshape_first_contact_native=self._control_preshape_first_contact_native.clone(),
                    control_preshape_contact_episode_ids=self._control_preshape_contact_episode_ids.clone())
            if approach_course is not None:
                result.update(approach_admitted=self._approach.admitted.clone(),
                    per_step_action_rad=per_step[None].expand(self.num_envs,-1).clone(),
                    env_origin_world_m=self.scene.env_origins.clone(),velocity_limits_arm_rad_s=self._velocity[:7].clone(),
                    approach_pregrasp_palm_position_world_m=torch.as_tensor(approach_resource["pregrasp_palm_position_world_m"],
                        device=self.device,dtype=self._targets.dtype)[None]+self.scene.env_origins,
                    approach_pregrasp_palm_quaternion_xyzw=torch.as_tensor(approach_resource["pregrasp_palm_quaternion_xyzw"],
                        device=self.device,dtype=self._targets.dtype)[None].expand(self.num_envs,-1).clone())
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

        def _open_declared_video_recording(self):
            """Create separate original-image ledgers; no physics identities change."""
            for camera in self._video_status["cameras"]:
                env_id=camera["camera_env_id"]
                frame_dir=output/camera["frame_directory"]
                frame_dir.mkdir(parents=True,exist_ok=False)
                manifest=(output/camera["frame_ledger"]).open("x",encoding="utf-8")
                self._declared_video_streams[env_id]={"frame_dir":frame_dir,"manifest":manifest,"last_camera_frame":None}
                self._write_declared_video_event({"event":"camera_header","width":640,"height":480,
                    "fps_numerator":500,"fps_denominator":33,"video_recording":video_recording,
                    "scope":"actual_original_env_pre_reset_post_physics_external_RGB_no_policy_input",
                    "eye_world_m":camera["eye_world_m"],"target_world_m":camera["target_world_m"],
                    "control_probe":dict(self._control_probe_profile)},env_ids=[env_id])

        def _write_declared_video_event(self,row,env_ids=None):
            selected=self._video_recording["env_ids"] if env_ids is None else env_ids
            for env_id in selected:
                stream=self._declared_video_streams[env_id]["manifest"]
                stream.write(json.dumps(_json_finite(_probe_payload(torch,{**row,"camera_env_id":env_id})),
                    ensure_ascii=False,allow_nan=False)+"\n")
                stream.flush()

        def _capture_declared_evaluation_frames(self):
            """Capture one synchronized frame set at a completed native boundary."""
            if self._video_status["failure"]:
                return
            failed_env=None
            try:
                from PIL import Image
                native_step=int(self.sim.get_physics_step_count())
                if native_step==self._last_video_native_step:
                    raise RLEnvironmentError("duplicate native physics step in declared camera frame set")
                frame=int(self._video_status["frames"])
                captured=[]
                for camera_status in self._video_status["cameras"]:
                    env_id=camera_status["camera_env_id"];failed_env=env_id
                    if camera_status["frames"]!=frame:
                        raise RLEnvironmentError("declared camera frame counters diverged")
                    camera=self._evaluation_cameras[env_id]
                    camera.update(dt=self.step_dt,force_recompute=True)
                    rgb=tensor(camera.data.output["rgb"])
                    if rgb.dtype!=torch.uint8 or tuple(rgb.shape[:3])!=(1,480,640) or rgb.shape[-1] not in (3,4):
                        raise RLEnvironmentError("declared camera requires real640x480uint8RGB(A)")
                    camera_frame=int(tensor(camera.frame)[0].item())
                    previous=self._declared_video_streams[env_id]["last_camera_frame"]
                    if previous is not None and camera_frame<=previous:
                        raise RLEnvironmentError("declared camera measurement counter did not advance")
                    captured.append((env_id,camera_status,camera_frame,rgb[0,:,:,:3].detach().cpu().numpy()))
                if int(self.sim.get_physics_step_count())!=native_step:
                    raise RLEnvironmentError("native physics advanced between cameras in a frame set")
                if frame==0:
                    memory={"scope":"after_all_declared_first_camera_RGB_readbacks_not_Torch_only_RTX_memory",
                        "camera_count":len(captured),"camera_env_ids":list(self._video_recording["env_ids"]),
                        "native_physics_step_count":native_step,"actual_memory_admission_not_inferred_from_pixel_size":True}
                    if str(self.device).startswith("cuda") and torch.cuda.is_available():
                        free,total=torch.cuda.mem_get_info(self.device)
                        memory.update(device_free_bytes=int(free),device_total_bytes=int(total),
                            torch_allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
                            torch_reserved_bytes=int(torch.cuda.memory_reserved(self.device)),
                            torch_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)),
                            torch_peak_reserved_bytes=int(torch.cuda.max_memory_reserved(self.device)))
                    else:
                        memory["device_memory_unavailable_reason"]="CUDA_unavailable"
                    self._video_status["first_frame_memory"]=memory
                for env_id,camera_status,camera_frame,pixels in captured:
                    failed_env=env_id;resources=self._declared_video_streams[env_id]
                    path=resources["frame_dir"]/("frame_{:06d}.png".format(frame))
                    Image.fromarray(pixels,mode="RGB").save(path)
                    payload=path.read_bytes()
                    self._write_declared_video_event({"event":"camera_frame","frame_index":frame,"path":path.name,
                        "bytes":len(payload),"sha256":hashlib.sha256(payload).hexdigest(),
                        "native_physics_step_count":native_step,"physics_samples":self._physics_samples,
                        "actual_physics_dt_s":float(self.sim.get_physics_dt()),
                        "sim_time_s":native_step*float(self.sim.get_physics_dt()),
                        "episode_id":int(self._episode_ids[env_id].item()),
                        "episode_control_steps":int(self.episode_length_buf[env_id].item()),
                        "camera_frame_counter":camera_frame,"pre_reset":True,
                        "cube_position_world_m":tensor(self.cube.data.root_pos_w)[env_id],
                        "cube_quaternion_xyzw":tensor(self.cube.data.root_quat_w)[env_id],
                        "profile_name":self._control_probe_profile["name"]},env_ids=[env_id])
                    resources["last_camera_frame"]=camera_frame
                    camera_status["frames"]=frame+1
                self._video_status["frames"]=frame+1
                self._last_video_native_step=native_step
            except Exception as exc:
                failure=type(exc).__name__+": "+str(exc)
                self._video_status["failure"]=failure
                for camera in self._video_status["cameras"]:
                    if camera["camera_env_id"]==failed_env:camera["failure"]=failure
                self._write_declared_video_event({"event":"camera_failure","failure":failure,
                    "failed_camera_env_id":failed_env,"next_frame_index":self._video_status["frames"]})
                if output:
                    with (output/"evaluation_video_failure.json").open("x",encoding="utf-8") as stream:
                        json.dump(self._video_status,stream,ensure_ascii=False,allow_nan=False)

        def _capture_evaluation_frame(self):
            if getattr(self,"_video_recording",None) is not None:
                self._capture_declared_evaluation_frames()
                return
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
                                        "main_lift_authorized": (self._full_lift.main_lift_authorized if getattr(self, "_full_lift", None) is not None else False), "policy_contract": self.runtime_audit()["policy_contract"]})

        def _write_evaluation_row(self, row):
            if self._record:
                if self._control_probe_enabled:
                    row = {**row, "control_probe_profile": dict(self._control_probe_profile)}
                self._record.write(json.dumps(_json_finite(_probe_payload(torch, row)),
                                              ensure_ascii=False, allow_nan=False) + "\n")

        def _relative_damping_small_status(self):
            transition=getattr(self,"_effort_contact_transition",None)
            if transition is None or transition.get("version")!="g1_coordinated_contact_transition_v3_relative_viscous_preload":
                return None
            return {**getattr(self,"_effort_native_inertia_status",{"available":False,"state":"not_measured_yet"}),
                "valid_mass_environment_samples":getattr(self,"_effort_valid_mass_environment_samples",0),
                "actual_relative_damping_group_environment_samples":getattr(self,"_effort_actual_relative_damping_group_samples",0),
                "scope":"small_read_only_summary_no_additional_native_queries_or_large_matrices"}

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
                    "original_main_lift_enabled": full_lift is not None,
                    "main_lift_authorized": (self._full_lift.main_lift_authorized if getattr(self, "_full_lift", None) is not None else False),
                    "observation_cameras": [], "in_episode_object_state_writes": 0, "object_attachments": 0}
            if fixed_palm_model:
                if effort_model:
                    from .rl_collision_model import bilateral_fixed_cluster_contract, readback_bilateral_fixed_cluster_filters
                    audit["policy_contract"]["fixed_cluster_collision_filter"] = bilateral_fixed_cluster_contract()
                    current_filter = readback_bilateral_fixed_cluster_filters(get_current_stage(), UsdPhysics, self.num_envs)
                else:
                    audit["policy_contract"]["fixed_cluster_collision_filter"] = fixed_palm_adjacency_filter_contract()
                    current_filter = readback_fixed_palm_adjacency_filters(get_current_stage(), UsdPhysics, self.num_envs)
                audit["fixed_cluster_collision_filter_override"] = {
                    **self._fixed_cluster_filter_readback, "current_stage_readback": current_filter}
                audit["physics_differences"].append(
                    "explicit bilateral fixed-component and direct revolute-adjacency30 exclusions" if effort_model else
                    "explicit right wrist fixed-component internal and direct revolute-adjacency 15-pair exclusions; all other self-collisions remain enabled")
            if self._control_probe_enabled:
                audit["policy_contract"]["control_guard_version"] = self._control_probe_profile["control_guard_version"]
                audit["control_probe_profile"] = dict(self._control_probe_profile)
                audit["diagnostic_native_joint_load_readback"] = self._native_joint_load_diagnostic.readback
                audit["diagnostic_native_self_pair_readback"] = self._native_self_pair_diagnostic.summary()
                audit["diagnostic_legacy_postcheck"] = "exact residuals recorded; source-pinned roundoff-only allowance, meaningful box/height violations latch unsafe"
            if effort_model:
                audit["policy_contract"].update(effort_grasp_controller=copy.deepcopy(effort_contract),
                    pd_backend="original_other36_implicit_and_right_hand7_explicit_zero_native_pd_v1",
                    action_units="arm velocity-limited position reference; hand action slots inactive, independent2ms explicit effort in Nm",
                    explicit_reset_resource_sha256=config["rl_diagnostics"]["candidate_sha256"],
                    reset_protocol=effort_candidate["reset_protocol"],
                    historical_reset_protocol=config["rl_teacher"]["reset_protocol"],
                    original_seed_state_retained_sha256=hashlib.sha256(json.dumps(original_initial, sort_keys=True).encode()).hexdigest())
                if full_lift is not None:
                    audit["policy_contract"]["full_lift"] = copy.deepcopy(full_lift)
                    audit["success_scope"] = "contact-qualified 10cm corner clearance and continuous 0.5s hold; original20 not certified"
                transition = config.get("rl_diagnostics", {}).get("contact_transition_control")
                if transition is not None:
                    audit["policy_contract"]["contact_transition_control"] = copy.deepcopy(transition)
                if "contact_generation" in effort_contract:
                    audit["policy_contract"]["contact_generation"] = copy.deepcopy(effort_contract["contact_generation"])
                    audit["diagnostic_contact_generation_readback"] = self._effort_contact_generation_readback
                    audit["physics_differences"].append("new zero-margin Robot/Cube/Table contact generation with negative one-micrometer per-shape rest offset and zero scene friction threshold; raw normal manifold and original acceptance formulas unchanged")
                audit["diagnostic_effort_drive_readback"] = self._effort_drive_readback
                audit["diagnostic_posture_gravity_feedforward"] = {
                    "contract": copy.deepcopy(effort_contract["actuation"]["posture_gravity_feedforward"]),
                    "source": "fresh native gravity_compensation_forces at every 2ms action application",
                    "routing": "single full43 user effort write: arm7 plus waist3 bounded gravity FF, hand7 bounded total effort, other26 zero",
                    "native_implicit_pd_values_unchanged": True,
                    "combined_posture_pd_and_ff_total_not_certified_bounded": True,
                    "native_user_actuation_measurement_is_not_isolated_implicit_drive_torque": True}
                audit["diagnostic_material_readback"] = self._effort_material_readback
                audit["physics_differences"].extend([
                    "right hand7 implicit PD removed; explicit current-contact-Jacobian effort with gravity included inside commanded torque cap",
                    "right arm7 and waist3 receive separately capped fresh native gravity feedforward; original implicit PD unchanged, combined PD plus FF torque not certified bounded by FF cap",
                    "declared finger-pad compliant-contact material variant; original rigid model and strict qualification retained as separate control",
                    "explicit new geometry/reset resource and measured arm pose reference; no old trained actor loaded"])
            damping_status=self._relative_damping_small_status()
            if damping_status is not None:
                audit["relative_damping_native_status"]=damping_status
            if video_recording is not None:
                audit["video_recording"]=video_recording
            if control_preshape is not None:
                audit["policy_contract"]["control_preshape"]=control_preshape
                audit["control_preshape_resolution"]=preshape_resolution
            if approach_course is not None:
                audit["policy_contract"].update(approach_course=copy.deepcopy(approach_course),
                    explicit_reset_resource_sha256=approach_course["resource_sha256"],
                    reset_protocol=approach_course["version"],
                    pregrasp_target_resource_sha256=config["rl_diagnostics"]["candidate_sha256"],
                    arm_lift_origin="pinned_pregrasp_FK_world_Z_not_home_or_recaptured_entry",
                    whole_episode_original_hand_domain_preserved=True)
                audit["approach_home_fk_reference"]=copy.deepcopy(approach_home_fk)
                audit["approach_collision_readback"]=self._approach_contact.summary()
                audit["approach_resource"]=copy.deepcopy(approach_resource)
                audit["success_scope"]="high_wait_vertical_approach_with_task_preshaped_hand_then_original_grasp_lift; not side_home, original20 or pick_place"
            return _json_finite(_probe_payload(torch, audit))

        def step_audit(self):
            audit = {"physics_samples": self._physics_samples, "reset_count": self._resets,
                    "interrupted_episode_count": self._interrupted, "evaluation_mode": self._eval_mode,
                    "main_lift_authorized": (self._full_lift.main_lift_authorized if getattr(self, "_full_lift", None) is not None else False), "action_shield_count": self._shield_count,
                    "action_origin": self._action_origin,
                    "zero_action_semantics": "previous_target_hold",
                    "measured_state_hold_request_clipped_count": self._state_hold_request_clipped,
                    "sim_time_s": self._physics_samples*self.physics_dt,
                    "native_physics_step_count": int(self.sim.get_physics_step_count()),
                    "actual_physics_dt_s": float(self.sim.get_physics_dt()),
                    "control_period_s": float(self.step_dt),
                    "physics_pinch_feedback": {"enabled": self._physics_pinch is not None,
                        "update_calls": self._physics_pinch_update_count,
                        "last_feedback_sample": self._physics_pinch_last_sample,
                        "declared_feedback_period_s": self.physics_dt if self._physics_pinch is not None else None},
                    "effort_grasp_feedback": {"enabled": self._effort_grasp is not None,
                        "contact_domain_posture_guard": getattr(self, "_effort_domain_posture_guard", None),
                        "contact_transition_control": getattr(self, "_effort_contact_transition", None),
                        "full_lift_protocol": full_lift,
                        "native_arm_target_schedule": (full_lift.get("arm_target_schedule") if full_lift is not None else None),
                        "last_actually_written_arm_target_rad": (self._last_written_arm_target if
                            getattr(self,"_native_arm_interpolation",False) else None),
                        "full_lift_current_state": self._full_lift.metrics() if self._full_lift is not None else None,
                        "controller_composition_version": getattr(self._effort_grasp, "contract", {}).get("version"),
                        "update_calls": self._effort_update_count, "last_feedback_sample": self._effort_last_sample,
                        "declared_feedback_period_s": self.physics_dt if self._effort_grasp is not None else None,
                        "hand_position_drive_disabled": effort_model,
                        "posture_gravity_feedforward_update_calls": self._posture_ff_update_count,
                        "posture_gravity_feedforward_last_completed_sample": self._posture_ff_last_sample,
                        "posture_gravity_feedforward_period_s": self.physics_dt if self._effort_grasp is not None else None,
                        "posture_pd_plus_ff_total_torque_certified_bounded": False,
                        "user_effort_route": "single_full43_arm7waist3FF_hand7total_other26zero" if effort_model else None,
                        "raw_samples_written": self._effort_raw_samples_written,
                        "raw_samples_omitted": self._effort_raw_samples_omitted,
                        "raw_trigger_windows_written": self._effort_raw_triggers,
                        "raw_triggers_omitted": self._effort_raw_omitted_triggers,
                        "compact_2ms_complete": True if self._effort_grasp is not None else None},
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
                        "native_self_pair_diagnostic": self._native_self_pair_diagnostic.summary(),
                        "control_guard_correction_count": self._control_guard_correction_count,
                        "control_guard_unsafe_count": self._control_guard_unsafe_count,
                        "shield_count_semantics": "legacy applied scalar<1; projected actual target correction nonzero"}
                       if self._control_probe_enabled else {})}
            damping_status=self._relative_damping_small_status()
            if damping_status is not None:
                audit["relative_damping_native_status"]=damping_status
            return _json_finite(_probe_payload(torch, audit))

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
            if video_recording is not None:
                for resources in self._declared_video_streams.values():
                    resources["manifest"].close()
                expected=self._physics_samples//33
                for camera in self._video_status["cameras"]:
                    camera["expected_control_frames"]=expected
                    camera["all_control_frames_recorded"]=(camera["enabled"] and camera["failure"] is None and
                        camera["frames"]==expected and expected>0)
            if output and hasattr(self, "_video_status"):
                self._video_status["expected_control_frames"] = self._physics_samples // 33
                self._video_status["all_control_frames_recorded"] = (
                    self._standalone_video and self._video_status["failure"] is None and
                    self._video_status["frames"] == self._physics_samples // 33 and self._video_status["frames"] > 0)
                if video_recording is not None:
                    self._video_status["all_control_frames_recorded"] &= all(
                        camera["all_control_frames_recorded"] for camera in self._video_status["cameras"])
                path = output / "evaluation_video_status.json"
                if not path.exists():
                    with path.open("x", encoding="utf-8") as stream:
                        json.dump(self._video_status, stream, indent=2)
            self._evaluation_camera = None
            if video_recording is not None:
                self._evaluation_cameras.clear()
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
