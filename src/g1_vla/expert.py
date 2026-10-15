"""Offline, privileged green/right teacher using the shared physical executor.

IK modifies a separate kinematic workspace. Only the declared initial home reset
sets the live qpos; the episode thereafter advances by PDController exclusively.
No policy/checkpoint/optimizer is imported, and no object state is teleported.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .contracts import ActionContract, canonical_sha256
from .control import ControlError, PDController, _api, controlled_joints
from .sim.scene import HOME_POSITIONS_M, OBJECT_ASSIST_OFFSET_M, PALM_NAMES, build_scene


class ExpertError(ValueError):
    """Teacher task, source geometry or temporal pairing is unsupported."""


# This is a robot hand convention, not an experiment configuration.
HAND_CLOSED_TARGETS = {
    "left_hand_thumb_0_joint": .40, "left_hand_thumb_1_joint": .60,
    "left_hand_thumb_2_joint": 1.20, "left_hand_index_0_joint": -1.00,
    "left_hand_index_1_joint": -1.20, "left_hand_middle_0_joint": -1.00,
    "left_hand_middle_1_joint": -1.20,
    "right_hand_thumb_0_joint": -.40, "right_hand_thumb_1_joint": -.60,
    "right_hand_thumb_2_joint": -1.20, "right_hand_index_0_joint": 1.00,
    "right_hand_index_1_joint": 1.20, "right_hand_middle_0_joint": 1.00,
    "right_hand_middle_1_joint": 1.20,
}

PROFILE_KEYS = frozenset({
    "profile", "grasp_yaw_deg", "closure_multiplier", "finger_scale",
    "contact_compensation_m", "pregrasp_height_m", "pregrasp_lateral_m",
    "high_pregrasp_height_m", "home_settle_s", "high_approach_s",
    "pregrasp_s", "contact_approach_s", "close_duration_s", "grasp_hold_s",
    "lift_height_m", "lift_duration_s", "final_hold_s", "adaptive_regrasp",
    "ik_max_error_m", "record_rgb", "palm_position_tolerance_m",
    "joint_target_tolerance_rad", "phase_timeout_scale",
    "ik_collision_margin_m", "ik_collision_weight",
    "grasp_center_in_palm_m", "ik_orientation_weight",
})


def smoothstep(alpha):
    alpha = min(1., max(0., float(alpha)))
    return alpha * alpha * (3. - 2. * alpha)


def interpolate(start, end, alpha):
    alpha = smoothstep(alpha)
    return tuple((1. - alpha) * a + alpha * b for a, b in zip(start, end))


def bounded_planning_seed(actual, lower, upper):
    """Make a legal interpolation reference without changing measured state.

    Soft mechanical limits can admit a small, still acceptance-safe overrun.
    That measured value remains the controller anchor and safety evidence; it
    must not become an out-of-range requested target at interpolation alpha=0.
    """
    import numpy as np
    values, low, high = (np.asarray(item, dtype=float) for item in (actual, lower, upper))
    if (values.ndim != 1 or values.shape != low.shape or values.shape != high.shape
            or not np.isfinite(values).all() or not np.isfinite(low).all()
            or not np.isfinite(high).all() or np.any(low > high)):
        raise ExpertError("planning seed and bounds require matching finite vectors")
    return np.clip(values, low, high).copy()


def effective_compensation(profile, hand="right"):
    """Legacy profile stores left-world y; right flips it exactly once."""
    result = [float(value) for value in profile["contact_compensation_m"]]
    if hand == "right":
        result[1] *= -1.
    elif hand != "left":
        raise ExpertError("a teacher profile requires one explicit hand")
    return tuple(result)


def green_target_from_position(object_position, profile, *, height=0., lateral=0., compensate=False):
    """Build a privileged teacher goal from the measured, current object center."""
    if len(object_position) != 3 or any(not math.isfinite(float(value)) for value in object_position):
        raise ExpertError("current green object position must contain three finite coordinates")
    if profile["profile"] in ("green_right_cavity_feedback_v3", "green_right_cartesian_feedback_v4"):
        # The site is at local x=.14, close to the finger roots.  The grasp
        # cavity is on their shared +local-y side toward the opposing thumb.
        # Every stage uses this same frame relation, including its approach.
        x, y, z = profile["grasp_center_in_palm_m"]
        angle = math.radians(profile["grasp_yaw_deg"])
        center_world = (math.cos(angle) * x - math.sin(angle) * y,
                        math.sin(angle) * x + math.cos(angle) * y, z)
        return tuple(float(object_position[index]) - center_world[index]
                     + (0., lateral, height)[index] for index in range(3))
    offset = OBJECT_ASSIST_OFFSET_M["green_cube"]
    compensation = effective_compensation(profile) if compensate else (0., 0., 0.)
    return tuple(float(object_position[index]) + offset[index] + compensation[index]
                 + (0., lateral - .03, height)[index] for index in range(3))


def hand_target(joint_name, profile):
    if joint_name not in HAND_CLOSED_TARGETS:
        raise ExpertError("unknown canonical hand joint {}".format(joint_name))
    for index, finger in enumerate(("thumb", "index", "middle")):
        if "hand_{}_".format(finger) in joint_name:
            return .5 * HAND_CLOSED_TARGETS[joint_name] * profile["closure_multiplier"] * profile["finger_scale"][index]
    raise ExpertError("unknown finger {}".format(joint_name))


def validate_teacher(config):
    if config["train"]["enabled"] or config["wm"]["enabled"]:
        raise ExpertError("teacher acceptance requires train.enabled=false and wm.enabled=false")
    if (config["task"]["mode"] != "grasp_lift" or config["task"]["hand"] != "right"
            or config["task"]["object_ids"] != ["green_cube"]):
        raise ExpertError("the implemented acceptance teacher is green_cube/right/grasp_lift only")
    if config["scene"]["assisted_grasp"] or config["eval"]["assisted_grasp"]:
        raise ExpertError("strict teacher acceptance forbids assisted grasp")
    profile = config["expert"].get("teacher")
    if not isinstance(profile, dict) or set(profile) != PROFILE_KEYS:
        raise ExpertError("expert.teacher must contain the complete explicit physical profile")
    if profile["profile"] not in ("green_right_recovery_v1", "green_right_geometry_feedback_v2",
                                  "green_right_cavity_feedback_v3", "green_right_cartesian_feedback_v4"):
        raise ExpertError("teacher requires a declared green/right physical profile")
    for key in PROFILE_KEYS - {"profile", "finger_scale", "contact_compensation_m", "grasp_center_in_palm_m", "adaptive_regrasp", "record_rgb", "grasp_yaw_deg"}:
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ExpertError("expert.teacher.{} must be finite and positive".format(key))
    if isinstance(profile["grasp_yaw_deg"], bool) or not math.isfinite(float(profile["grasp_yaw_deg"])):
        raise ExpertError("grasp_yaw_deg must be finite")
    for key in ("finger_scale", "contact_compensation_m", "grasp_center_in_palm_m"):
        values = profile[key]
        if not isinstance(values, list) or len(values) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ExpertError("{} requires three finite values".format(key))
    if any(v <= 0 for v in profile["finger_scale"]):
        raise ExpertError("finger_scale must be positive")
    if (profile["profile"] in ("green_right_cavity_feedback_v3", "green_right_cartesian_feedback_v4")
            and any(value != 0 for value in profile["contact_compensation_m"])):
        raise ExpertError("cavity profiles require zero legacy contact compensation")
    if profile["profile"] == "green_right_cartesian_feedback_v4":
        if config["expert"].get("trajectory", {}).get("backend") != "actual_grasp_center_cartesian_v1":
            raise ExpertError("Cartesian v4 requires its explicit trajectory configuration")
    if not all(isinstance(profile[key], bool) for key in ("adaptive_regrasp", "record_rgb")):
        raise ExpertError("adaptive_regrasp and record_rgb must be booleans")
    if profile["phase_timeout_scale"] < 1:
        raise ExpertError("phase_timeout_scale must be at least one")
    if profile["final_hold_s"] < config["eval"]["hold_duration_s"]:
        raise ExpertError("final_hold_s must cover the configured acceptance hold")
    return profile


def pose_tracking_reached(tracking, profile):
    """Check the measured palm and requested arm target using the fixed gates."""
    return bool(tracking
                and tracking["palm_position_error_m"] <= profile["palm_position_tolerance_m"]
                and tracking["requested_joint_error_rad"] <= profile["joint_target_tolerance_rad"])


def contact_closure_ready(phase, elapsed_s, tracking, profile):
    """Keep the hand open until the physical approach, including its time, ends.

    Contact is deliberately not a prerequisite: requiring an open hand to have
    already established sustained contact would deadlock the closing action.
    """
    return bool(phase == "contact" and elapsed_s >= profile["contact_approach_s"]
                and pose_tracking_reached(tracking, profile))


def closure_complete(now_s, close_start_s, profile):
    """A first touch cannot skip the complete declared closing interval."""
    return bool(close_start_s is not None
                and now_s - close_start_s + 1e-9 >= profile["close_duration_s"])


def cartesian_path_executed(progress, actual_endpoint_reached, requested_arm, applied_arm):
    """A proposed endpoint is insufficient without execution and true tracking."""
    return bool(progress and progress.get("end_reference_reached") and actual_endpoint_reached
                and len(requested_arm) == len(applied_arm)
                and all(abs(float(first) - float(second)) <= 1e-9
                        for first, second in zip(requested_arm, applied_arm)))


def phase_tracking_reached(phase, tracking, profile, *, contact, minimum_lift_m,
                           measured_hold_s, lift_height_m, hold_duration_s):
    """A phase advances by reached physical state, never by elapsed time alone."""
    if phase not in ("home", "high_approach", "pregrasp", "contact", "grasp", "lift", "hold"):
        raise ExpertError("unknown teacher phase")
    reached = pose_tracking_reached(tracking, profile)
    if phase in ("contact", "grasp", "lift", "hold"):
        reached = reached and contact
    if phase in ("lift", "hold"):
        reached = reached and minimum_lift_m >= lift_height_m
    if phase == "hold":
        reached = reached and measured_hold_s >= hold_duration_s
    return bool(reached)


def physical_gate(samples, *, lift_height_m, hold_duration_s):
    """A launch/bounce or one old contact cannot satisfy sustained grasp."""
    duration, maximum, last_end, contact_steps = 0., 0., None, 0
    for row in samples:
        start, end = float(row["start_s"]), float(row["end_s"])
        if not math.isfinite(start + end) or end <= start:
            raise ExpertError("physical gate samples require finite increasing interval boundaries")
        if last_end is not None and not math.isclose(start, last_end, abs_tol=1e-7):
            duration = 0.
        valid = (row["contact"] and not row["assisted"] and row["lift_height_m"] >= lift_height_m
                 and row.get("finite", True) and row.get("joint_safe", True))
        contact_steps += int(bool(row["contact"]))
        duration = duration + end - start if valid else 0.
        maximum = max(maximum, duration)
        last_end = end
    return {"contact_steps": contact_steps, "sustained_lift_hold_s": maximum,
            "physical_grasp_passed": bool(maximum + 1e-9 >= hold_duration_s)}


def arm_ik_contract(model, *, hand=None, mujoco_api=None):
    """Canonical initialization is bilateral; rollout is right arm only."""
    import numpy as np
    api = _api(mujoco_api)
    names, ids = [], []
    for jid in range(model.njnt):
        name = api.mj_id2name(model, api.mjtObj.mjOBJ_JOINT, jid) or ""
        arm = any(part in name for part in ("shoulder_", "elbow_", "wrist_"))
        selected = (name.startswith("waist_") or arm and name.startswith(("left_", "right_"))) if hand is None else arm and name.startswith(hand + "_")
        if selected:
            names.append(name)
            ids.append(jid)
    qpos = np.asarray([model.jnt_qposadr[jid] for jid in ids], dtype=int)
    lower = np.asarray([model.jnt_range[jid, 0] for jid in ids], dtype=float)
    upper = np.asarray([model.jnt_range[jid, 1] for jid in ids], dtype=float)
    for index, name in enumerate(names):
        if name == "waist_pitch_joint":
            lower[index], upper[index] = -.1, .1
        elif name in ("waist_yaw_joint", "waist_roll_joint"):
            lower[index], upper[index] = -.05, .05
    if len(names) != (17 if hand is None else 7):
        raise ExpertError("unexpected canonical arm IK dimensions")
    return names, qpos, lower, upper


def arm_torso_collision_pairs(model, *, include_elbow=False, mujoco_api=None):
    """Declare active torso/shoulder pairs, adding observed elbow pairs for v3.

    These are the self contacts measured in the failed teacher, not replacement
    collision shapes. Mask, welded-body, parent-body and explicit-exclusion
    filters retain the original model's collision eligibility.
    """
    api = _api(mujoco_api)
    torso = api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, "torso_link")
    body_names = ["right_shoulder_roll_link", "right_shoulder_yaw_link"]
    if include_elbow:
        body_names.append("right_elbow_link")
    arm_bodies = {api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, name) for name in body_names}
    if torso < 0 or min(arm_bodies) < 0:
        raise ExpertError("declared torso/right-arm collision bodies are missing")
    excluded = set(int(value) for value in model.exclude_signature)
    pairs = []
    for first in range(model.ngeom):
        body_a = int(model.geom_bodyid[first])
        if body_a != torso:
            continue
        for second in range(model.ngeom):
            body_b = int(model.geom_bodyid[second])
            if body_b not in arm_bodies:
                continue
            compatible = (int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
                          or int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
            weld_a, weld_b = int(model.body_weldid[body_a]), int(model.body_weldid[body_b])
            adjacent = (int(model.body_parentid[weld_a]) == weld_b
                        or int(model.body_parentid[weld_b]) == weld_a)
            signature = (min(body_a, body_b) << 16) + max(body_a, body_b)
            if compatible and weld_a != weld_b and not adjacent and signature not in excluded:
                pairs.append((first, second))
    expected = 9 if include_elbow else 6
    if len(pairs) != expected:
        raise ExpertError("source-audited torso/arm constraint requires exactly {} active geom pairs".format(expected))
    return tuple(pairs)


def collision_distance_residual(distances_m, margin_m, weight):
    """Fixed-length signed-distance hinge; all geometry stays in the model."""
    return tuple(weight * min(0., float(distance) - margin_m) for distance in distances_m)


def solve_pose(model, workspace, palm_ids, qpos_ids, lower, upper, targets, seed,
               orientations, *, orientation_weight=.04, max_error_m=.08,
               collision_pairs=(), collision_margin_m=.002, collision_weight=5.,
               mujoco_api=None):
    """Pure kinematic workspace update; never called on live rollout data."""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    api = _api(mujoco_api)
    bounded = np.clip(seed, lower + 1e-6, upper - 1e-6)

    def distances():
        return tuple(float(api.mj_geomDistance(model, workspace, first, second,
                                               collision_margin_m + .05, None))
                     for first, second in collision_pairs)

    def residual(values):
        workspace.qpos[qpos_ids] = values
        api.mj_forward(model, workspace)
        position = np.concatenate([workspace.site_xpos[site] - target for site, target in zip(palm_ids, targets)])
        orientation = np.concatenate([Rotation.from_matrix(expected.T @ workspace.site_xmat[site].reshape(3, 3)).as_rotvec()
                                      for site, expected in zip(palm_ids, orientations)])
        collision = collision_distance_residual(distances(), collision_margin_m, collision_weight)
        return np.concatenate((position, orientation_weight * orientation, .003 * values, collision))

    solution = least_squares(residual, bounded, bounds=(lower, upper), max_nfev=1200,
                             xtol=1e-10, ftol=1e-10, gtol=1e-10)
    residual(solution.x)
    errors = [float(np.linalg.norm(workspace.site_xpos[site] - target)) for site, target in zip(palm_ids, targets)]
    orientation_errors = [float(np.linalg.norm(Rotation.from_matrix(
        expected.T @ workspace.site_xmat[site].reshape(3, 3)).as_rotvec()))
        for site, expected in zip(palm_ids, orientations)]
    final_distances = distances()
    object_body = api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, "green_cube")
    generated_self_contacts = _robot_self_contacts(model, workspace, object_body, api) if collision_pairs else {}
    pairs_evidence = [{"geom_ids": [int(first), int(second)],
                       "body_names": [api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
                                      for geom in (first, second)],
                       "distance_m": distance}
                      for (first, second), distance in zip(collision_pairs, final_distances)]
    collision_passed = all(distance >= 0. for distance in final_distances) and not generated_self_contacts
    return solution.x.copy(), {"solver_success": bool(max(errors) <= max_error_m and collision_passed),
                               "optimizer_converged": bool(solution.success),
                               "seed_was_clipped": bool(np.any(seed != bounded)),
                               "maximum_position_error_m": max(errors),
                               "maximum_orientation_error_rad": max(orientation_errors),
                               "position_acceptance_limit_m": float(max_error_m),
                               "collision_avoidance_applied": bool(collision_pairs),
                               "collision_margin_m": float(collision_margin_m),
                               "collision_weight": float(collision_weight),
                               "collision_pairs": pairs_evidence,
                               "collision_pair_signature_sha256": canonical_sha256([
                                   {"geom_ids": row["geom_ids"], "body_names": row["body_names"]}
                                   for row in pairs_evidence]),
                               "minimum_declared_self_pair_distance_m": min(final_distances) if final_distances else None,
                               "generated_robot_self_contacts": generated_self_contacts,
                               "collision_acceptance_passed": bool(collision_passed),
                               "cost": float(solution.cost)}


def _robot_self_contacts(model, data, object_body, api):
    """Read actual generated robot self contacts without changing their masks."""
    pairs = {}
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > 0:
            continue
        bodies = tuple(int(model.geom_bodyid[int(g)]) for g in (contact.geom1, contact.geom2))
        if 0 in bodies or object_body in bodies:
            continue
        names = [api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, body) or str(body) for body in bodies]
        key = " :: ".join(sorted(names))
        pairs[key] = max(pairs.get(key, 0.), max(0., -float(contact.dist)))
    return pairs


def _geometry_evidence(model, data, object_body, palm_id, api):
    """Snapshot measured geometry; this is privileged teacher diagnostics only."""
    import numpy as np
    palm = np.asarray(data.site_xpos[palm_id], dtype=float)
    rotation = np.asarray(data.site_xmat[palm_id], dtype=float).reshape(3, 3)
    position = np.asarray(data.xpos[object_body], dtype=float)
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > 0:
            continue
        geoms = (int(contact.geom1), int(contact.geom2))
        contacts.append({
            "geom_names": [api.mj_id2name(model, api.mjtObj.mjOBJ_GEOM, geom) or str(geom) for geom in geoms],
            "body_names": [api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom]))
                           or str(int(model.geom_bodyid[geom])) for geom in geoms],
            "distance_m": float(contact.dist)})
    return {"timestamp_s": float(data.time), "palm_position_m": palm.tolist(),
            "object_position_m": position.tolist(),
            "object_offset_in_palm_frame_m": (rotation.T @ (position - palm)).tolist(),
            "palm_rotation_world_from_local": rotation.tolist(),
            "generated_contacts": contacts,
            "robot_self_contact_penetration_m": _robot_self_contacts(model, data, object_body, api)}


def _contact_evidence(model, data, object_body, api):
    import numpy as np
    count, maximum, pairs = 0, 0., []
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > 0:
            continue
        geoms = (int(contact.geom1), int(contact.geom2))
        bodies = tuple(int(model.geom_bodyid[g]) for g in geoms)
        if object_body not in bodies:
            continue
        other = 1 - bodies.index(object_body)
        bname = api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, bodies[other]) or ""
        gname = api.mj_id2name(model, api.mjtObj.mjOBJ_GEOM, geoms[other]) or ""
        if not (bname.startswith("right_hand_") or gname == "right_palm_contact_pad"):
            continue
        force = np.zeros(6)
        api.mj_contactForce(model, data, index, force)
        count += 1
        maximum = max(maximum, abs(float(force[0])))
        pairs.append(gname or bname)
    return count, maximum, pairs


def green_contact_penetration_evidence(model, data, object_body, api):
    """Read hand/table overlap depths separately from unchanged success gates.

    The inherited source mesh and contact capsules can both be active. Positive
    depth is a numerical/model-fidelity diagnostic, never hidden by a PASS.
    """
    maximum = {kind: {"penetration_m": 0., "pair": None} for kind in ("hand", "table")}
    for index in range(data.ncon):
        contact = data.contact[index]
        if contact.dist > 0:
            continue
        geoms = (int(contact.geom1), int(contact.geom2))
        bodies = tuple(int(model.geom_bodyid[geom]) for geom in geoms)
        if object_body not in bodies:
            continue
        other = 1 - bodies.index(object_body)
        body_name = api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, bodies[other]) or ""
        geom_name = api.mj_id2name(model, api.mjtObj.mjOBJ_GEOM, geoms[other]) or ""
        kind = ("hand" if body_name.startswith("right_hand_") or geom_name == "right_palm_contact_pad"
                else "table" if geom_name == "calibration_table" else None)
        depth = max(0., -float(contact.dist))
        if kind is None or depth <= maximum[kind]["penetration_m"]:
            continue
        maximum[kind] = {
            "penetration_m": depth,
            "pair": {"geom_ids": list(geoms),
                     "geom_names": [api.mj_id2name(model, api.mjtObj.mjOBJ_GEOM, geom)
                                    or "geom#{}".format(geom) for geom in geoms],
                     "body_names": [api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, body)
                                    or "body#{}".format(body) for body in bodies]}}
    return maximum


def state_is_finite(data):
    """Include forces/time, even when a controller aborts before its callback."""
    fields = ("qpos", "qvel", "qacc", "ctrl", "actuator_force", "qfrc_bias")
    return bool(math.isfinite(float(data.time)) and all(
        math.isfinite(float(value)) for key in fields for value in getattr(data, key)))


def joint_limit_excesses(model, data, controlled):
    """Raw mechanical overrun for every commanded joint, including frozen ones."""
    result = {}
    for name, item in controlled.items():
        jid = item["joint_id"]
        if bool(model.jnt_limited[jid]):
            value = float(data.qpos[item["qpos_id"]])
            low, high = (float(item) for item in model.jnt_range[jid])
            result[name] = max(0., low - value, value - high) if math.isfinite(value) else None
    return result


def run_teacher_episode(config, spec, run_dir):
    """Run exactly one predeclared layout and persist true control-step pairs.

    Failure returns a report and partial evidence instead of discarding a layout.
    The caller owns layout generation and the unchanged acceptance threshold.
    """
    import numpy as np
    from scipy.spatial.transform import Rotation
    from .data_pairing import build_paired_episode
    from .observation import ObservationBuilder, ObservationSpec

    profile = validate_teacher(config)
    contract = ActionContract.from_config(config)
    api = _api()
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path, raw_path = run_dir / "teacher_report.json", run_dir / "expert_pick_place_episode.npz"
    if report_path.exists() or raw_path.exists():
        raise ExpertError("teacher episode must have a new run directory")
    if set(spec.get("positions_m", {})) != {"green_cube"}:
        raise ExpertError("teacher layout must contain only green_cube")
    layout_fingerprint = spec.get("layout_fingerprint") or canonical_sha256(spec["positions_m"])
    cartesian_profile = profile["profile"] == "green_right_cartesian_feedback_v4"
    report = {"schema_version": 2,
              "teacher_protocol": ("green_right_strict_shared_pd_cartesian_v4" if cartesian_profile else "green_right_strict_shared_pd_cavity_v3"
                                   if profile["profile"] == "green_right_cavity_feedback_v3"
                                   else "green_right_strict_shared_pd_feedback_v2"),
              "task": "green_cube_right_grasp_lift", "layout": spec,
              "layout_fingerprint": layout_fingerprint, "train_started": False,
              "assisted_grasp": False, "profile": dict(profile),
              "effective_contact_compensation_m": list(effective_compensation(profile)),
              "source_evidence": ["G1LANG-585/recovery_0000_green_cube", "G1LANG-586/recovery_0001_green_cube", "G1LANG-589/recovery_0001_green_cube"],
              "source_evidence_scope": "legacy recovery reference only; current geometry/closure profile requires full physical revalidation",
              "initialization": "declared canonical bilateral home before physical time zero; no in-episode state reset",
              "rollout_joint_scope": "right seven arm and seven hand joints; waist and left upper body held",
              "changes_from_legacy": ["robot-only damping", "explicit actuator force limits", "command cadence and anchor/velocity limiter", "waist and left arm held", "close only after measured contact-pose approach", "IK position limit cannot exceed physical tracking gate"],
              "passed": False, "failure": None, "record_rgb": profile["record_rgb"],
              "joint_limit_tolerance_rad": config["eval"]["joint_limit_tolerance_rad"],
              "future_phase_poses_precomputed": False,
              "latest_phase_planning_attempt": None,
              "raw_episode": str(raw_path), "report": str(report_path)}
    observations, commands, timing, evidence, requested_rows, after_rows, qacc_rows, target_rows, tracking_rows = [], [], [], [], [], [], [], [], []
    substeps, renderer, controller, model, data = [], None, None, None, None
    diagnostic_video = None
    phase_records, saturation_counts, maximum_actuator_force, physics_warning_counts = [], {}, {}, []
    replan_ledger = []
    phase_reference_ledger = []
    self_collision_counts, self_collision_penetrations, phase_self_collision_counts = {}, {}, {}
    maximum_green_penetration = {kind: {"penetration_m": 0., "pair": None,
                                      "timestamp_s": None, "phase": None} for kind in ("hand", "table")}
    closure_start_event = None
    current_phase, planned_next_phase = None, None
    maximum_joint_overrun, pending = {}, {}
    planner, active_plan, path_progress, path_status, grip_lock = None, None, None, {}, None
    trajectory = {"backend": config["expert"].get("trajectory", {}).get("backend") if cartesian_profile else None,
                  "plans": [], "execution_ledger": [], "grip_lock": None, "progress_blocked_count": 0,
                  "workspace_snapshots": [],
                  "limiter_mismatch_count": 0, "planning_failure": None,
                  "settings": dict(config["expert"].get("trajectory", {})) if cartesian_profile else None,
                  "maximum_closed_rotation_error_rad": 0., "maximum_taskpoint_tracking_error_m": 0.,
                  "rotation_measurement_scope": "actual lift/hold post-control states relative to the post-grasp grip lock at the original control cadence; not every physics substep"}
    try:
        source = Path(config["scene"]["asset_path"])
        if not source.is_absolute():
            source = Path(config["paths"]["root"]) / source
        report["scene"] = build_scene(source, run_dir / "scene.xml", spec["positions_m"], run_dir=run_dir,
                                      preserve_source_camera=False, include_blue_box=False)
        model = api.MjModel.from_xml_path(report["scene"]["scene"])
        data, workspace = api.MjData(model), api.MjData(model)
        api.mj_forward(model, data)
        workspace.qpos[:] = data.qpos
        workspace.qvel[:] = data.qvel
        canonical_names, canonical_ids, canonical_low, canonical_high = arm_ik_contract(model, mujoco_api=api)
        palms = [api.mj_name2id(model, api.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES]
        if min(palms) < 0:
            raise ExpertError("canonical palm sites are missing")
        home, home_report = solve_pose(model, workspace, palms, canonical_ids, canonical_low, canonical_high,
                                       np.asarray(HOME_POSITIONS_M), np.zeros(len(canonical_names)),
                                       np.repeat(np.eye(3)[None], 2, axis=0), orientation_weight=.06,
                                       max_error_m=profile["ik_max_error_m"], mujoco_api=api)
        report["ik"] = {"home": home_report}
        if not home_report["solver_success"]:
            report["failed_ik_label"] = "home"
            raise ExpertError("canonical home IK failed")
        # This is the single declared initialization before all recorded control.
        data.qpos[canonical_ids] = home
        controlled = controlled_joints(model, mujoco_api=api)
        for name, binding in controlled.items():
            if "hand_" in name:
                low, high = model.jnt_range[binding["joint_id"]]
                value = .01 if abs(low) < 1e-9 else -.01 if abs(high) < 1e-9 else float(data.qpos[binding["qpos_id"]])
                data.qpos[binding["qpos_id"]] = value
        api.mj_forward(model, data)
        controller = PDController(model, data, config, contract=contract, mujoco_api=api)
        names = contract.joint_names
        bindings = controller.bindings
        baseline = np.asarray([data.qpos[item["qpos_id"]] for item in bindings], dtype=float)
        right_names, ids, low, high = arm_ik_contract(model, hand="right", mujoco_api=api)
        arm_indices = [names.index(name) for name in right_names]
        object_body = api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, "green_cube")
        if object_body < 0:
            raise ExpertError("green object is missing")
        initial_position = data.xpos[object_body].copy()
        report["initial_object_position_m"] = initial_position.tolist()
        report["control"] = controller.metadata()
        report["action_contract"] = contract.metadata()
        orientations = np.asarray([Rotation.from_euler("z", profile["grasp_yaw_deg"], degrees=True).as_matrix()])
        cavity_profile = profile["profile"] in ("green_right_cavity_feedback_v3", "green_right_cartesian_feedback_v4")
        collision_pairs = arm_torso_collision_pairs(model, include_elbow=cavity_profile, mujoco_api=api)
        report["ik_collision_constraint_scope"] = (
            "nine source-active torso/right-shoulder-and-elbow geometry pairs; reject any generated robot self contact at the solution"
            if cavity_profile else "six source-active torso/right-shoulder geometry pairs; reject any generated robot self contact at the solution")
        report["ik_collision_pair_count"] = len(collision_pairs)
        report["grasp_target_geometry"] = {
            "mapping": ("actual taskpoint C(q)=palm(q)+R(q)@grasp_center_in_palm_m; fixed planned endpoint palm retains original tracking gate" if cartesian_profile else "object_world - Rz(configured_yaw) @ grasp_center_in_palm_m + world_phase_lateral_and_height"
                        if cavity_profile else "legacy object assist offset, world lateral - .03m, optional mirrored legacy compensation"),
            "grasp_center_in_palm_m": list(profile["grasp_center_in_palm_m"]) if cavity_profile else None,
            "grasp_yaw_deg": float(profile["grasp_yaw_deg"]),
            "contact_palm_offset_world_m": list(green_target_from_position((0., 0., 0.), profile, compensate=True)),
            "legacy_assist_offset_applied": not cavity_profile,
            "ik_orientation_weight": float(profile["ik_orientation_weight"])}
        if cartesian_profile:
            report["grasp_target_geometry"]["nominal_offset_scope"] = "ideal-yaw diagnostic only; planner and execution use actual FK palm rotation"
            trajectory["scene_reference"] = {key: report["scene"][key]
                for key in ("source", "source_sha256", "scene", "scene_sha256")}
        poses = {"home": baseline[arm_indices].copy()}
        cartesian_targets = {"home": data.site_xpos[palms[1]].copy()}
        reports = {"home": home_report}
        report["ik"] = reports
        report["initial_joint_position_rad"] = baseline.tolist()

        def pose(label, target, seed):
            workspace.qpos[:] = data.qpos
            workspace.qvel[:] = data.qvel
            values, detail = solve_pose(model, workspace, [palms[1]], ids, low, high, np.asarray([target]), seed,
                                        orientations, orientation_weight=profile["ik_orientation_weight"],
                                        max_error_m=min(profile["ik_max_error_m"],
                                            profile["palm_position_tolerance_m"]),
                                        collision_pairs=collision_pairs,
                                        collision_margin_m=profile["ik_collision_margin_m"],
                                        collision_weight=profile["ik_collision_weight"], mujoco_api=api)
            reports[label] = detail
            cartesian_targets[label] = np.asarray(target).copy()
            if not detail["solver_success"]:
                raise ExpertError("{} IK failed: position={:.4f}m, self_collision_gate={}".format(
                    label, detail["maximum_position_error_m"], detail["collision_acceptance_passed"]))
            return values

        def target_from_object(height=0., lateral=0., compensate=False):
            return np.asarray(green_target_from_position(data.xpos[object_body], profile,
                              height=height, lateral=lateral, compensate=compensate))

        def refresh_phase_pose(phase, key, target, *, target_relation="current_object_center"):
            nonlocal planned_next_phase, active_plan, path_progress, path_status, grip_lock
            planned_next_phase = phase
            visited_seed = data.qpos[ids].copy()
            label = "{}_live".format(phase)
            attempt = {
                "phase": phase, "previous_phase": current_phase,
                "timestamp_s": float(data.time), "status": "solving",
                "current_object_position_m": data.xpos[object_body].copy().tolist(),
                "visited_right_arm_qpos_rad": visited_seed.tolist(),
                "target_palm_position_m": np.asarray(target).tolist(),
                "target_relation": target_relation, "live_state_modified": False}
            report["latest_phase_planning_attempt"] = attempt
            if cartesian_profile:
                attempt["entry_full_state"] = planner.state_snapshot(data)
            try:
                if cartesian_profile:
                    from .teacher_path_progress import JointPathProgress
                    if phase == "lift":
                        if grip_lock is not None:
                            raise ExpertError("actual post-grasp grip may only be locked once")
                        grip_lock = planner.latch_grip(data)
                        trajectory["grip_lock"] = grip_lock
                    next_plan = planner.plan_phase(phase, data,
                        np.asarray(controller.previous_target)[arm_indices],
                        grip_lock=grip_lock, previous_plan=active_plan)
                    active_plan, path_status = next_plan, {}
                    trajectory["plans"].append(next_plan)
                    values = np.asarray(next_plan["q_waypoints"][-1])
                    target = np.asarray(next_plan["endpoint_palm"])
                    attempt["target_palm_position_m"] = target.tolist()
                    attempt["target_relation"] = target_relation = "fixed_FK_endpoint_of_audited_actual_taskpoint_path"
                    reports[label] = {"solver_success": True, "reference_sha256": next_plan["reference_sha256"],
                        "maximum_position_error_m": max((item["taskpoint_error_m"] for item in next_plan["waypoint_details"]), default=0.),
                        "collision_acceptance_passed": True, "budget": next_plan["budget"],
                        "target_C": next_plan["target_C"], "target_R": next_plan["target_R"],
                        "fixed_endpoint_palm_m": next_plan["endpoint_palm"]}

                    def online_chord(first, second, context):
                        # The object remains free: refresh its actual quaternion,
                        # translation and every hand joint before each query.
                        planner.sync_actual(data)
                        snapshots = trajectory["workspace_snapshots"]
                        if not snapshots or snapshots[-1]["time_s"] != float(data.time):
                            snapshots.append(planner.state_snapshot(data))
                        closed_geometry = active_plan["closed"] or close_start_s is not None
                        result = planner.validate_chord(first, second, context, closed=closed_geometry,
                            lock_rotation=active_plan["target_R"] if active_plan["closed"] else None)
                        result["workspace_snapshot_index"] = len(snapshots) - 1
                        return result

                    path_progress = JointPathProgress(next_plan["q_waypoints"], planner.velocity,
                        controller.cadence.period_s, low, high,
                        max_anchor_delta=float(np.min(planner.anchor)), chord_validator=online_chord)
                else:
                    values = pose(label, target, visited_seed)
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                if cartesian_profile:
                    trajectory["planning_failure"] = {"phase": phase, "error": str(exc),
                                                       "detail": getattr(exc, "detail", {})}
                attempt.update({"status": "rejected", "ik": reports.get(label),
                                "error": "{}: {}".format(type(exc).__name__, exc)})
                report["failure_phase"] = phase
                report["failure_kind"] = "phase_entry_ik_rejection"
                report["failed_ik_label"] = label
                raise
            attempt.update({"status": "accepted", "ik": dict(reports[label]),
                            "solution_right_arm_qpos_rad": values.tolist()})
            poses[key] = values
            cartesian_targets[key] = np.asarray(target).copy()
            replan_ledger.append({
                "phase": phase, "timestamp_s": float(data.time),
                "current_object_position_m": data.xpos[object_body].copy().tolist(),
                "visited_right_arm_qpos_rad": visited_seed.tolist(),
                "target_palm_position_m": np.asarray(target).tolist(),
                "solution_right_arm_qpos_rad": values.tolist(),
                "target_relation": target_relation,
                "ik": dict(reports[label]),
                "live_state_modified": False})

        report["ik"] = reports
        report["initial_joint_position_rad"] = baseline.tolist()
        report["initial_geometry"] = _geometry_evidence(model, data, object_body, palms[1], api)
        from .teacher_video import TeacherDiagnosticVideo
        try:
            diagnostic_video = TeacherDiagnosticVideo(model, config, run_dir, mujoco_api=api)
        except (ValueError, RuntimeError, OSError, TypeError) as exc:
            report["diagnostic_video"] = {"complete": False, "error": "{}: {}".format(type(exc).__name__, exc)}
        builder = None
        if profile["record_rgb"]:
            renderer = api.Renderer(model, height=config["observation"]["image_height"], width=config["observation"]["image_width"])
            builder = ObservationBuilder(model, renderer, names, ObservationSpec.from_config(config), mujoco_api=api)
        if cartesian_profile:
            from .teacher_trajectory import CartesianTeacherPlanner
            planner = CartesianTeacherPlanner(model, config, contract, mujoco_api=api)
            trajectory["collision_pairs"] = planner.pair_evidence
            trajectory["collision_pair_signature_sha256"] = planner.pair_signature
        language = "grasp the green cube with the right hand and lift it"
        current_phase, phase_start = "home", 0.
        actual_phase_seed = data.qpos[ids].copy()
        phase_seed = bounded_planning_seed(actual_phase_seed, low, high)
        phase_reference_ledger.append({
            "phase": current_phase, "timestamp_s": float(data.time),
            "joint_names": list(right_names),
            "raw_actual_arm_qpos_rad": actual_phase_seed.tolist(),
            "bounded_plan_seed_rad": phase_seed.tolist(),
            "clip_delta_rad": (phase_seed - actual_phase_seed).tolist(),
            "maximum_abs_clip_rad": float(np.max(np.abs(phase_seed - actual_phase_seed))),
            "live_state_modified": False, "controller_anchor_modified": False})
        active_cartesian_target = cartesian_targets["home"].copy()
        interval_contacts, interval_lift, interval_safe, interval_finite = [], [], [], []
        pending = {}
        warnings_initial = np.asarray([item.number for item in data.warning], dtype=int)
        hold_seconds = 0.
        latest_tracking = {}

        def finite_and_safe():
            finite = state_is_finite(data)
            excesses = joint_limit_excesses(model, data, controlled)
            for name, excess in excesses.items():
                if excess is not None:
                    maximum_joint_overrun[name] = max(maximum_joint_overrun.get(name, 0.), excess)
            tolerance = config["eval"]["joint_limit_tolerance_rad"]
            safe = all(value is not None and value <= tolerance for value in excesses.values())
            return finite, bool(safe)

        def substep_observer(**event):
            finite, safe = finite_and_safe()
            interval_safe.append(safe)
            interval_finite.append(finite)
            if not finite:
                substeps.append({"timestamp_s": float(data.time) if math.isfinite(float(data.time)) else None,
                                 "contact_count": 0, "normal_force_n": 0., "lift_height_m": 0.,
                                 "joint_safe": False, "finite": False, "max_abs_qacc": None,
                                 "pairs": [], "saturated_actuators": []})
                raise ControlError("teacher physics became non-finite")
            count, force, pairs = _contact_evidence(model, data, object_body, api)
            green_penetration = green_contact_penetration_evidence(model, data, object_body, api)
            for kind, detail in green_penetration.items():
                if detail["penetration_m"] > maximum_green_penetration[kind]["penetration_m"]:
                    maximum_green_penetration[kind] = dict(detail, timestamp_s=float(data.time), phase=current_phase)
            self_contacts = _robot_self_contacts(model, data, object_body, api)
            for pair, penetration in self_contacts.items():
                self_collision_counts[pair] = self_collision_counts.get(pair, 0) + 1
                self_collision_penetrations[pair] = max(self_collision_penetrations.get(pair, 0.), penetration)
                phase_counts = phase_self_collision_counts.setdefault(current_phase, {})
                phase_counts[pair] = phase_counts.get(pair, 0) + 1
            lift = float(data.xpos[object_body, 2] - initial_position[2])
            if not math.isfinite(lift):
                lift = 0.
            saturated = []
            for name, binding in controlled.items():
                aid = binding["actuator_id"]
                actuator_force = abs(float(data.actuator_force[aid]))
                maximum_actuator_force[name] = max(maximum_actuator_force.get(name, 0.), actuator_force)
                if model.actuator_forcelimited[aid]:
                    force_low, force_high = model.actuator_forcerange[aid]
                    force_bound = max(abs(float(force_low)), abs(float(force_high)))
                    if force_bound > 0 and actuator_force >= force_bound - 1e-7:
                        saturation_counts[name] = saturation_counts.get(name, 0) + 1
                        saturated.append(name)
            interval_contacts.append(bool(count))
            interval_lift.append(lift)
            substeps.append({"timestamp_s": float(data.time), "contact_count": count, "normal_force_n": force,
                             "lift_height_m": lift, "joint_safe": safe, "finite": finite,
                             "max_abs_qacc": float(np.max(np.abs(data.qacc))) if finite else None,
                             "pairs": pairs, "saturated_actuators": saturated,
                             "green_contact_penetration": green_penetration,
                             "robot_self_contact_pairs": sorted(self_contacts)})
            if not finite:
                raise ControlError("teacher physics became non-finite")
            warnings = np.asarray([item.number for item in data.warning], dtype=int)
            physics_warning_counts[:] = warnings.tolist()
            if np.any(warnings > warnings_initial):
                raise ControlError("MuJoCo warning detected during teacher execution")

        def observer(**event):
            nonlocal hold_seconds
            if event["event"] == "before":
                if diagnostic_video is not None:
                    diagnostic_video.capture(data, current_phase)
                interval_contacts.clear()
                interval_lift.clear()
                interval_safe.clear()
                interval_finite.clear()
                frame = builder.observe(data, language) if builder is not None else {
                    "joint_position": [float(data.qpos[item["qpos_id"]]) for item in bindings],
                    "joint_velocity": [float(data.qvel[item["qvel_id"]]) for item in bindings],
                    "timestamp_s": float(data.time), "language": language}
                pending.clear()
                pending.update({"frame": frame, "start_s": float(data.time),
                                "command": list(event["executed_target"]), "requested": list(event["requested_target"])})
            elif event["event"] == "after":
                final_finite, final_safe = finite_and_safe()
                finite = final_finite and bool(interval_finite) and all(interval_finite)
                safe = final_safe and bool(interval_safe) and all(interval_safe)
                end_s = float(data.time)
                all_contact = bool(interval_contacts) and all(interval_contacts)
                min_lift = min(interval_lift) if interval_lift else 0.
                hold_seconds = hold_seconds + end_s - pending["start_s"] if all_contact and min_lift >= config["eval"]["lift_height_m"] and finite and safe else 0.
                observations.append(pending["frame"])
                commands.append(pending["command"])
                requested_rows.append(pending["requested"])
                after = [float(data.qpos[item["qpos_id"]]) for item in bindings]
                after_rows.append(after)
                qacc_rows.append([float(data.qacc[item["qvel_id"]]) for item in bindings])
                timing.append({"start_s": pending["start_s"], "end_s": end_s,
                               "sequence_index": len(timing), "state_segment_id": 0,
                               "restart_wait": current_phase == "home",
                               "post_joint_position": after,
                               "post_joint_velocity": [float(data.qvel[item["qvel_id"]]) for item in bindings], "post_time_s": end_s})
                evidence.append({"start_s": pending["start_s"], "end_s": end_s,
                                 "contact": all_contact, "any_contact": any(interval_contacts),
                                 "lift_height_m": min_lift, "hold_seconds": hold_seconds,
                                 "assisted": False, "finite": finite, "joint_safe": safe,
                                 "phase": current_phase})
                target_rows.append(data.xpos[object_body].copy().tolist())
                latest_tracking.clear()
                latest_tracking.update({
                    "timestamp_s": end_s, "phase": current_phase,
                    "palm_position_error_m": float(np.linalg.norm(data.site_xpos[palms[1]] - active_cartesian_target)),
                    "requested_joint_error_rad": float(np.max(np.abs(np.asarray(pending["requested"])[arm_indices] - data.qpos[ids]))),
                    "executed_joint_error_rad": float(np.max(np.abs(np.asarray(pending["command"])[arm_indices] - data.qpos[ids]))),
                    "right_hand_requested_error_rad": max(abs(pending["requested"][i] - after[i]) for i, name in enumerate(names) if name.startswith("right_hand_")),
                    "command_limiter_delta_rad": float(np.max(np.abs(np.asarray(pending["requested"]) - pending["command"]))),
                })
                if cartesian_profile and active_plan is not None:
                    from .teacher_trajectory import rotation_error
                    actual_pose = planner.actual_pose(data)
                    planner.sync_actual(data)
                    reference = planner.reference_pose(np.asarray(pending["requested"])[arm_indices])
                    tracking_error = float(np.linalg.norm(actual_pose["C"] - reference["C"]))
                    closed_error = (rotation_error(grip_lock["R"], actual_pose["R"])
                                    if grip_lock is not None and active_plan["closed"] else None)
                    mismatch = float(np.max(np.abs(np.asarray(pending["requested"])[arm_indices]
                                                   - np.asarray(pending["command"])[arm_indices])))
                    path_executed = cartesian_path_executed(path_status, planner.end_reached(active_plan, data),
                        np.asarray(pending["requested"])[arm_indices], np.asarray(pending["command"])[arm_indices])
                    latest_tracking.update({"cartesian_path_executed": path_executed,
                        "actual_grasp_center_m": actual_pose["C"].tolist(),
                        "reference_grasp_center_m": reference["C"].tolist(),
                        "endpoint_grasp_center_error_m": float(np.linalg.norm(actual_pose["C"] - active_plan["target_C"])),
                        "declared_taskpoint_goal_error_m": float(np.linalg.norm(actual_pose["C"] - active_plan["target_C"])),
                        "fixed_endpoint_reference_C_error_m": float(np.linalg.norm(actual_pose["C"] - active_plan["endpoint_C"])),
                        "taskpoint_reference_tracking_error_m": tracking_error,
                        "closed_grip_rotation_error_rad": closed_error, "arm_limiter_delta_rad": mismatch})
                    trajectory["maximum_taskpoint_tracking_error_m"] = max(trajectory["maximum_taskpoint_tracking_error_m"], tracking_error)
                    if closed_error is not None:
                        trajectory["maximum_closed_rotation_error_rad"] = max(trajectory["maximum_closed_rotation_error_rad"], closed_error)
                    trajectory["limiter_mismatch_count"] += int(mismatch > 1e-9)
                    trajectory["execution_ledger"].append({"timestamp_s": end_s, "phase": current_phase,
                        "reference_sha256": active_plan["reference_sha256"], "progress": dict(path_status),
                        "raw_actual_arm_qpos_rad": actual_pose["arm"].tolist(),
                        "actual_C": actual_pose["C"].tolist(), "actual_R": actual_pose["R"].tolist(),
                        "requested_arm_rad": np.asarray(pending["requested"])[arm_indices].tolist(),
                        "applied_arm_rad": np.asarray(pending["command"])[arm_indices].tolist(),
                        "fixed_endpoint_palm_m": active_plan["endpoint_palm"], "tracking": dict(latest_tracking)})
                tracking_rows.append(dict(latest_tracking))
                pending.clear()
            else:
                raise ExpertError("unknown controller observer event")

        phases = ("home", "high_approach", "pregrasp", "contact", "grasp", "lift", "hold")
        duration = {"home": profile["home_settle_s"], "high_approach": profile["high_approach_s"],
                    "pregrasp": profile["pregrasp_s"], "contact": profile["contact_approach_s"],
                    "grasp": profile["grasp_hold_s"], "lift": profile["lift_duration_s"],
                    "hold": profile["final_hold_s"]}
        destination = {"home": "home", "high_approach": "high", "pregrasp": "pregrasp",
                       "contact": "grasp", "grasp": "grasp", "lift": "lift", "hold": "lift"}
        close_start_s = None
        completed = False
        while not completed:
            elapsed = float(data.time) - phase_start
            if (close_start_s is None and contact_closure_ready(current_phase, elapsed, latest_tracking, profile)
                    and (not cartesian_profile or latest_tracking.get("cartesian_path_executed", False))):
                close_start_s = float(data.time)
                closure_start_event = {
                    "timestamp_s": close_start_s, "phase": current_phase,
                    "elapsed_contact_approach_s": elapsed,
                    "trigger": "completed_nominal_open_approach_and_measured_palm_and_arm_gates",
                    "tracking": dict(latest_tracking),
                    "geometry": _geometry_evidence(model, data, object_body, palms[1], api),
                    "right_hand_joint_position_rad": {
                        name: float(data.qpos[bindings[index]["qpos_id"]])
                        for index, name in enumerate(names) if name.startswith("right_hand_")}}
            desired = poses[destination[current_phase]]
            active_cartesian_target = cartesian_targets[destination[current_phase]]
            target = baseline.copy()
            if cartesian_profile and current_phase != "home":
                from .teacher_trajectory import rotation_error
                planner.sync_actual(data)
                actual_pose = planner.actual_pose(data)
                reference_pose = planner.reference_pose(path_progress.reference)
                settings = config["expert"]["trajectory"]
                tracking_ready = bool(
                    np.max(np.abs(actual_pose["arm"] - path_progress.reference)) <= settings["reference_tracking_rad"]
                    and np.linalg.norm(actual_pose["C"] - reference_pose["C"]) <= settings["reference_tracking_m"]
                    and (not active_plan["closed"] or rotation_error(active_plan["target_R"], actual_pose["R"])
                         <= settings["closed_orientation_tolerance_rad"]))
                path_status = path_progress.step(actual_pose["arm"], np.asarray(controller.previous_target)[arm_indices], tracking_ready)
                trajectory["progress_blocked_count"] += int(path_status["blocked"])
                if path_status["reference"] is None:
                    raise ExpertError("Cartesian follower has no legal acknowledged reference: {}".format(path_status["reason"]))
                target[arm_indices] = path_status["reference"]
            else:
                target[arm_indices] = interpolate(phase_seed, desired, elapsed / duration[current_phase])
            close = smoothstep((float(data.time) - close_start_s) / profile["close_duration_s"]) if close_start_s is not None else 0.
            for index, name in enumerate(names):
                if name.startswith("right_hand_"):
                    target[index] = (1 - close) * baseline[index] + close * hand_target(name, profile)
            # Teacher plans at every actual observation. The eventual data chunk
            # is built from eight next real control commands by the shared codec.
            controller.execute_target(target.tolist(), joint_names=names,
                                      observer=observer, substep_observer=substep_observer)
            if cartesian_profile and latest_tracking.get("arm_limiter_delta_rad", 0.) > 1e-9:
                raise ExpertError("shared limiter changed an audited Cartesian arm reference")
            elapsed = float(data.time) - phase_start
            contact_now = bool(interval_contacts) and all(interval_contacts)
            reached = phase_tracking_reached(current_phase, latest_tracking, profile,
                        contact=contact_now, minimum_lift_m=min(interval_lift), measured_hold_s=hold_seconds,
                        lift_height_m=config["eval"]["lift_height_m"], hold_duration_s=config["eval"]["hold_duration_s"])
            if current_phase == "contact":
                reached = reached and closure_complete(float(data.time), close_start_s, profile)
            if cartesian_profile and current_phase != "home":
                reached = reached and latest_tracking.get("cartesian_path_executed", False)
            if elapsed >= duration[current_phase] and reached:
                phase_records.append({"phase": current_phase, "start_s": phase_start,
                                      "end_s": float(data.time), "tracking": dict(latest_tracking),
                                      "actual_contact": contact_now, "completed": True,
                                      "geometry": _geometry_evidence(model, data, object_body, palms[1], api),
                                      "closure_started": close_start_s is not None,
                                      "closure_complete": closure_complete(float(data.time), close_start_s, profile)})
                if current_phase == "hold":
                    completed = True
                    continue
                next_phase = phases[phases.index(current_phase) + 1]
                if next_phase == "high_approach":
                    refresh_phase_pose(next_phase, "high", target_from_object(profile["high_pregrasp_height_m"]))
                elif next_phase == "pregrasp":
                    refresh_phase_pose(next_phase, "pregrasp", target_from_object(profile["pregrasp_height_m"], -profile["pregrasp_lateral_m"]))
                elif next_phase == "contact":
                    # The cube has physically settled by now. Spawn coordinates
                    # are not the current contact goal or the visited IK seed.
                    refresh_phase_pose(next_phase, "grasp", target_from_object(compensate=True))
                elif next_phase == "grasp":
                    refresh_phase_pose(next_phase, "grasp", target_from_object(compensate=True))
                elif next_phase == "lift":
                    refresh_phase_pose(next_phase, "lift", target_from_object(profile["lift_height_m"], compensate=True))
                elif next_phase == "hold":
                    # Preserve the attained lift reference. Adding lift_height
                    # to the now-lifted object again would cause a second lift.
                    refresh_phase_pose(next_phase, "lift", cartesian_targets["lift"].copy(),
                                       target_relation="preserve_completed_lift_reference")
                actual_phase_seed = data.qpos[ids].copy()
                phase_seed = bounded_planning_seed(actual_phase_seed, low, high)
                phase_reference_ledger.append({
                    "phase": next_phase, "timestamp_s": float(data.time),
                    "joint_names": list(right_names),
                    "raw_actual_arm_qpos_rad": actual_phase_seed.tolist(),
                    "bounded_plan_seed_rad": phase_seed.tolist(),
                    "clip_delta_rad": (phase_seed - actual_phase_seed).tolist(),
                    "maximum_abs_clip_rad": float(np.max(np.abs(phase_seed - actual_phase_seed))),
                    "live_state_modified": False, "controller_anchor_modified": False})
                current_phase, phase_start = next_phase, float(data.time)
                planned_next_phase = None
            elif elapsed >= duration[current_phase] * profile["phase_timeout_scale"]:
                phase_records.append({"phase": current_phase, "start_s": phase_start,
                                      "end_s": float(data.time), "tracking": dict(latest_tracking),
                                      "actual_contact": contact_now, "completed": False,
                                      "geometry": _geometry_evidence(model, data, object_body, palms[1], api),
                                      "closure_started": close_start_s is not None,
                                      "closure_complete": closure_complete(float(data.time), close_start_s, profile)})
                raise ExpertError("{} timeout: actual palm/joint/contact/lift gate did not pass".format(current_phase))
        gate = physical_gate(evidence, lift_height_m=config["eval"]["lift_height_m"], hold_duration_s=config["eval"]["hold_duration_s"])
        report.update(gate)
        report["passed"] = bool(gate["physical_grasp_passed"] and all(row["joint_safe"] for row in substeps))
        if not report["passed"]:
            report["failure"] = "strict sustained physical lift/hold or joint-safety gate failed"
    except (ValueError, RuntimeError, OSError, TypeError) as exc:
        report["failure"] = "{}: {}".format(type(exc).__name__, exc)
        report["passed"] = False
    finally:
        if diagnostic_video is not None:
            try:
                report["diagnostic_video"] = diagnostic_video.finish()
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                report["diagnostic_video"] = {"complete": False, "error": "{}: {}".format(type(exc).__name__, exc)}
        if renderer is not None:
            try:
                renderer.close()
            except (ValueError, RuntimeError, OSError) as exc:
                report["renderer_close_error"] = str(exc)
                report["passed"] = False
        if data is not None:
            report["final_live_state_finite"] = state_is_finite(data)
            physics_warning_counts[:] = [int(item.number) for item in data.warning]
            if controller is not None:
                for name, excess in joint_limit_excesses(model, data, controller.controlled).items():
                    if excess is not None:
                        maximum_joint_overrun[name] = max(maximum_joint_overrun.get(name, 0.), excess)
            if not report["final_live_state_finite"]:
                report["passed"] = False
                report["failure"] = "non-finite final live physics state; interrupted command is diagnostic only"
        else:
            report["final_live_state_finite"] = None
    if pending:
        # An aborted interval is not appended as a full-duration paired label.
        interrupted = {
            "start_time_s": pending["start_s"],
            "last_time_s": float(data.time) if math.isfinite(float(data.time)) else None,
            "visited_joint_position_rad": pending["frame"]["joint_position"],
            "visited_joint_velocity_rad_s": pending["frame"]["joint_velocity"],
            "requested_joint_target_rad": pending["requested"],
            "limited_joint_target_rad": pending["command"],
            "pairing_eligible": False, "training_eligible": False,
            "reason": "command interval interrupted before complete post-state callback"}
        report["interrupted_control_interval"] = interrupted
        interrupted_path = run_dir / "interrupted_control.npz"
        diagnostic = {
            "metadata_json": np.asarray(json.dumps(interrupted, allow_nan=False)),
            "physics_evidence_json": np.asarray(json.dumps(substeps, allow_nan=False))}
        if "images" in pending["frame"]:
            for index, key in enumerate(sorted(pending["frame"]["images"])):
                diagnostic["images_camera_{}".format(index)] = np.asarray(pending["frame"]["images"][key])
            diagnostic["camera_keys"] = np.asarray(sorted(pending["frame"]["images"]))
        np.savez_compressed(interrupted_path, **diagnostic)
        report["interrupted_control_artifact"] = str(interrupted_path)
    report["phase_replan_ledger"] = replan_ledger
    if cartesian_profile:
        report["trajectory"] = trajectory
        report["trajectory"]["live_state_modified"] = False
        report["trajectory"]["acceptance_scope"] = "additional path/taskpoint/orientation conditions AND original palm/joint/contact/lift/hold gates; no phase duration or timeout extension"
    report["phase_reference_ledger"] = phase_reference_ledger
    report["planning_seed_reference"] = "mechanically bounded copy of raw measured right-arm state for interpolation only; live qpos, observation, controller anchor and original joint-safety tolerance remain unchanged"
    if cartesian_profile:
        report["planning_seed_reference"] = "v4 path starts at the prior actually applied legal command; raw actual state is retained for tracking, all FK snapshots and the unchanged safety gate; legacy bounded-plan-seed ledger is diagnostic only"
    report["active_execution_phase"] = current_phase
    report["planned_next_phase"] = planned_next_phase
    report.setdefault("failed_ik_label", None)
    report["closure_start_event"] = closure_start_event
    report["closure_planning"] = "open through completed contact approach and measured palm/arm tracking gates; full declared close interval required before contact phase passes"
    if cartesian_profile:
        report["closure_planning"] += "; additionally requires executed full Cartesian path and true actual taskpoint within configured {}m; no early close while descending".format(config["expert"]["trajectory"]["taskpoint_tolerance_m"])
    report["robot_self_collision_substeps_by_pair"] = self_collision_counts
    report["maximum_self_collision_penetration_m_by_pair"] = self_collision_penetrations
    report["phase_robot_self_collision_substeps_by_pair"] = phase_self_collision_counts
    report["maximum_green_contact_penetration_evidence"] = maximum_green_penetration
    report["maximum_green_hand_contact_penetration_m"] = maximum_green_penetration["hand"]["penetration_m"]
    report["maximum_green_table_contact_penetration_m"] = maximum_green_penetration["table"]["penetration_m"]
    report["contact_penetration_diagnostic_scope"] = "actual every-substep green/right-hand and green/calibration-table contacts; inherited mesh/capsule overlap retained; depth does not modify the original acceptance gate"
    report["phase_planning"] = "home initialization only before control/video; future poses solved once their phase is entered from actual object state and visited right-arm seed; rejected entry records latest_phase_planning_attempt; hold preserves completed lift reference"
    report["phases"] = phase_records
    report["actuation_saturation_substeps_by_joint"] = saturation_counts
    report["maximum_actual_actuator_force_by_joint"] = maximum_actuator_force
    report["physics_warning_counts"] = physics_warning_counts
    report["actual_tracking"] = tracking_rows[-1] if tracking_rows else None
    report["teacher_action_planning"] = "one actual sensor state and next shared-limited control command per step"
    report["maximum_limiter_adjustment_rad"] = max((row["command_limiter_delta_rad"] for row in tracking_rows), default=0.)
    report["control_steps"] = len(commands)
    report["physics_steps"] = len(substeps)
    report["maximum_lift_m"] = max((row["lift_height_m"] for row in substeps), default=0.)
    report["physical_contact_substeps"] = sum(row["contact_count"] > 0 for row in substeps)
    report["maximum_contact_normal_force_n"] = max((row["normal_force_n"] for row in substeps), default=0.)
    report["maximum_abs_qacc"] = max((row["max_abs_qacc"] or 0. for row in substeps), default=0.)
    report["joint_limit_violation_substeps"] = sum(not row["joint_safe"] for row in substeps)
    report["maximum_mechanical_joint_overrun_rad_by_joint"] = maximum_joint_overrun
    report["maximum_mechanical_joint_overrun_rad"] = max(maximum_joint_overrun.values(), default=0.)
    report["all_numeric_finite"] = (bool(all(row["finite"] for row in substeps)
                                          and report["final_live_state_finite"]) if substeps else report["final_live_state_finite"])
    report["training_eligible"] = False
    report["pairing_eligible"] = False
    report["command_limit_audit"] = {"checked": False, "passed": False}
    if commands:
        actual = np.asarray(commands, dtype=float)
        visited = np.asarray([row["joint_position"] for row in observations], dtype=float)
        previous = np.vstack((baseline[None], actual[:-1]))
        velocity = np.abs(actual - previous) / controller.cadence.period_s
        anchor_delta = np.abs(actual - visited)
        frozen = np.asarray(contract.max_velocity_rad_s) == 0
        velocity_fail = velocity > np.asarray(contract.max_velocity_rad_s)[None] + 1e-9
        anchor_fail = anchor_delta > np.asarray(contract.max_anchor_delta_rad)[None] + 1e-9
        frozen_fail = np.any(np.abs(actual[:, frozen] - baseline[frozen][None]) > 1e-9)
        limit_ok = not bool(velocity_fail.any() or anchor_fail.any() or frozen_fail)
        report["command_limit_audit"] = {
            "checked": True, "passed": limit_ok,
            "maximum_applied_velocity_rad_s_by_joint": dict(zip(names, velocity.max(axis=0).tolist())),
            "maximum_anchor_delta_rad_by_joint": dict(zip(names, anchor_delta.max(axis=0).tolist())),
            "velocity_violation_samples": int(velocity_fail.sum()),
            "anchor_violation_samples": int(anchor_fail.sum()),
            "frozen_command_changed": bool(frozen_fail),
            "frozen_anchor_scope": "hold initial command; all 31 visited-state anchor bounds checked, never state clamped"}
        if not limit_ok:
            report["passed"] = False
            report["failure"] = "actual applied command violated a shared execution limit"

    if commands:
        source = {"parent_root": str(config["paths"]["root"]), "source_rollout": str(run_dir),
                  "parent_rollout_id": str(spec["layout_id"]), "layout_fingerprint": layout_fingerprint,
                  "regime": "expert", "takeover_frame": None, "language_instruction": observations[0]["language"],
                  "object_id": "green_cube", "hand": "right", "episode_id": str(spec["layout_id"])}
        try:
            raw = build_paired_episode(observations, commands, timing, names, source,
                                       physical_evidence=evidence, expected_interval_s=controller.cadence.period_s)
            raw.update(requested_joint_target_rad=np.asarray(requested_rows, dtype=np.float64),
                       post_joint_position_rad=np.asarray(after_rows, dtype=np.float64),
                       post_joint_acceleration_rad_s2=np.asarray(qacc_rows, dtype=np.float64),
                       post_target_position_m=np.asarray(target_rows, dtype=np.float64),
                       tracking_json=np.asarray(json.dumps(tracking_rows, allow_nan=False)),
                       physics_evidence_json=np.asarray(json.dumps(substeps, allow_nan=False)),
                       scene_manifest_json=np.asarray(json.dumps(report["scene"], allow_nan=False)))
            np.savez_compressed(raw_path, **raw)
            # Raw evidence stays diagnostic until a separate pairing/source audit.
            report["pairing_eligible"] = bool(json.loads(str(raw["metadata_json"].item()))["pairing_eligible"])
        except (ValueError, TypeError, OSError) as exc:
            report["pairing_error"] = "{}: {}".format(type(exc).__name__, exc)
            report["passed"] = False
    report["raw_episode_exists"] = raw_path.is_file()
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report
