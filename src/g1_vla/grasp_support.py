"""Read-only measured hand support and an additional continuous load gate.

Contact forces are from post-step, post-mj_forward state under held ctrl. They
are not the forces used by the preceding integration solver. No function here
advances physics, writes state, commands a joint, or substitutes for the original
phase/safety/physical acceptance gates.
"""
import copy
import itertools
import math

import numpy as np

from .contracts import canonical_sha256
from .control import _api
from .hand_contact import finger_group


FORCE_SCOPE = "post_mj_step_existing_mj_forward_under_held_ctrl; not preceding integration solver force"


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(name + " must be a matching finite array")
    return result.copy()


def _rotation(value, name):
    matrix = _array(value, (3, 3), name)
    if not np.allclose(matrix.T @ matrix, np.eye(3), rtol=0, atol=1e-8) or np.linalg.det(matrix) < 0:
        raise ValueError(name + " must be a proper rotation")
    return matrix


def box_corners_world(center, rotation, half_sizes):
    center = _array(center, (3,), "box center")
    rotation = _rotation(rotation, "box rotation")
    sizes = _array(half_sizes, (3,), "box half sizes")
    if np.any(sizes <= 0):
        raise ValueError("box half sizes must be positive")
    local = np.asarray(list(itertools.product((-1., 1.), repeat=3))) * sizes
    return center + local @ rotation.T


def contact_wrench_on_object(force_contact_frame, frame_rows_world, position_world, com_world, object_geom_index):
    """Return force and torque about the object's COM, including local torque."""
    if isinstance(object_geom_index, bool) or object_geom_index not in (0, 1):
        raise ValueError("object geom index must be zero or one")
    local = _array(force_contact_frame, (6,), "contact force/torque")
    frame = _rotation(frame_rows_world, "contact frame rows")
    position, com = _array(position_world, (3,), "contact position"), _array(com_world, (3,), "object COM")
    # Contact frame normal points geom1 -> geom2; its wrench acts positively
    # on geom2 and with the opposite sign on geom1.
    sign = 1. if object_geom_index == 1 else -1.
    force = sign * frame.T @ local[:3]
    torque = np.cross(position - com, force) + sign * frame.T @ local[3:]
    return force, torque


def native_grasp_support(model, data, object_body, palm_site, grip_reference=None, *, joint_safe, mujoco_api=None):
    """Measure a free box above the fixed horizontal calibration table.

    ``grip_reference`` is captured once from the real load-transfer start. The
    caller must not re-lock it each tick; SupportWindow binds its hash. None
    permits pre-lock side-contact diagnostics, but cannot pass the load gate. Right
    finger meshes/capsules and the palm pad are all included, without altering
    masks or contacts. Table and other-object support remain separate.
    """
    api = _api(mujoco_api)
    if not isinstance(joint_safe, bool):
        raise ValueError("joint safety must be an explicit measured boolean")
    reference, origin_relative = None, None
    now = float(data.time)
    if not math.isfinite(now):
        raise ValueError("support time must be finite")
    if grip_reference is not None:
        if not isinstance(grip_reference, dict) or "timestamp_s" not in grip_reference:
            raise ValueError("load transfer requires a fixed measured grip reference")
        origin_time = float(grip_reference["timestamp_s"])
        origin_relative = _array(grip_reference["object_in_palm_frame_m"], (3,), "fixed grip reference")
        if not math.isfinite(origin_time) or now < origin_time - 1e-9:
            raise ValueError("support time must follow the fixed grip reference")
        reference = {"timestamp_s": origin_time, "object_in_palm_frame_m": origin_relative.tolist()}
    gravity = _array(model.opt.gravity, (3,), "gravity")
    if np.any(np.abs(gravity[:2]) > 1e-12) or gravity[2] >= 0:
        raise ValueError("support measurement requires downward vertical gravity")
    mass = float(model.body_mass[object_body])
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("object must have finite positive mass")
    first, count = int(model.body_geomadr[object_body]), int(model.body_geomnum[object_body])
    geoms = [g for g in range(first, first + count)
             if int(model.geom_contype[g]) or int(model.geom_conaffinity[g])]
    if len(geoms) != 1 or int(model.geom_type[geoms[0]]) != int(api.mjtGeom.mjGEOM_BOX):
        raise ValueError("support measurement requires one active box collision geom")
    joint_start = int(model.body_jntadr[object_body])
    if (int(model.body_jntnum[object_body]) != 1 or joint_start < 0
            or int(model.jnt_type[joint_start]) != int(api.mjtJoint.mjJNT_FREE)):
        raise ValueError("support measurement requires the independently free object")
    cube_geom = geoms[0]
    table_geom = api.mj_name2id(model, api.mjtObj.mjOBJ_GEOM, "calibration_table")
    if (table_geom < 0 or int(model.geom_type[table_geom]) != int(api.mjtGeom.mjGEOM_BOX)
            or int(model.body_weldid[int(model.geom_bodyid[table_geom])]) != 0):
        raise ValueError("support measurement requires the fixed box calibration table")
    table_R = _rotation(np.asarray(data.geom_xmat[table_geom]).reshape(3, 3), "table rotation")
    if not np.allclose(table_R[:, 2], [0., 0., 1.], rtol=0, atol=1e-9):
        raise ValueError("support measurement requires a horizontal upward table top")
    table_sizes = _array(np.asarray(model.geom_size[table_geom])[:3], (3,), "table half sizes")
    if np.any(table_sizes <= 0):
        raise ValueError("table half sizes must be positive")
    plane_point = _array(data.geom_xpos[table_geom], (3,), "table position") + table_R[:, 2] * table_sizes[2]
    corners = box_corners_world(data.geom_xpos[cube_geom], np.asarray(data.geom_xmat[cube_geom]).reshape(3, 3),
                                np.asarray(model.geom_size[cube_geom])[:3])
    clearance = float(np.min((corners - plane_point) @ table_R[:, 2]))
    com = _array(data.xipos[object_body], (3,), "actual object COM")
    object_position = _array(data.xpos[object_body], (3,), "actual object position")
    palm_position = _array(data.site_xpos[palm_site], (3,), "actual palm position")
    palm_R = _rotation(np.asarray(data.site_xmat[palm_site]).reshape(3, 3), "actual palm rotation")
    relative = palm_R.T @ (object_position - palm_position)
    velocity = np.zeros(6)
    # mjOBJ_BODY asks for COM-centred velocity, in world orientation (flag=0).
    api.mj_objectVelocity(model, data, api.mjtObj.mjOBJ_BODY, object_body, velocity, 0)
    if not np.isfinite(velocity).all():
        raise ValueError("object COM velocity must be finite")
    totals = {key: np.zeros(3) for key in ("hand_force", "hand_torque", "table_force", "table_torque", "other_force", "other_torque")}
    groups = {name: {"loaded_contact_count": 0, "normal_force_n": 0., "normal_force_weighted_direction": np.zeros(3)}
              for name in ("thumb", "index", "middle", "palm")}
    counts = {"table": 0, "other": 0}
    normal_totals = {"table": 0., "other": 0.}
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        gids = (int(contact.geom1), int(contact.geom2))
        bodies = tuple(int(model.geom_bodyid[g]) for g in gids)
        if object_body not in bodies:
            continue
        object_index = bodies.index(object_body)
        other = 1 - object_index
        body_name = api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, bodies[other]) or ""
        geom_name = api.mj_id2name(model, api.mjtObj.mjOBJ_GEOM, gids[other]) or "geom#{}".format(gids[other])
        group = finger_group(body_name, geom_name)
        kind = "hand" if group is not None else "table" if gids[other] == table_geom else "other"
        local = np.zeros(6)
        api.mj_contactForce(model, data, index, local)
        frame = np.asarray(contact.frame).reshape(3, 3)
        force, torque = contact_wrench_on_object(local, frame, contact.pos, com, object_index)
        distance = float(contact.dist)
        if not math.isfinite(distance):
            raise ValueError("native contact distance must be finite")
        loaded = bool(distance <= 0. and int(contact.efc_address) >= 0 and local[0] > 0.)
        normal_on_object = (1. if object_index == 1 else -1.) * frame[0]
        if loaded:
            totals[kind + "_force"] += force
            totals[kind + "_torque"] += torque
            if kind == "hand":
                groups[group]["loaded_contact_count"] += 1
                groups[group]["normal_force_n"] += float(local[0])
                groups[group]["normal_force_weighted_direction"] += float(local[0]) * normal_on_object
            else:
                counts[kind] += 1
                normal_totals[kind] += float(local[0])
        contacts.append({"contact_index": index, "geom_ids": list(gids), "object_geom_index": object_index,
                         "kind": kind, "group": group, "other_geom": geom_name, "distance_m": distance,
                         "efc_address": int(contact.efc_address), "loaded": loaded,
                         "position_world_m": np.asarray(contact.pos).tolist(), "frame_rows_world": frame.tolist(),
                         "force_contact_frame": local.tolist(), "force_on_object_world_n": force.tolist(),
                         "torque_on_object_about_com_world_nm": torque.tolist(),
                         "normal_on_object_world": normal_on_object.tolist()})
    directions = {}
    for name, group in groups.items():
        weighted = group.pop("normal_force_weighted_direction")
        norm = float(np.linalg.norm(weighted))
        directions[name] = weighted / norm if norm > 1e-12 else None
        group["force_weighted_normal_direction_world"] = directions[name].tolist() if directions[name] is not None else None
    opposition = {name: float(directions["thumb"] @ directions[name]) if directions["thumb"] is not None and directions[name] is not None else None
                  for name in ("index", "middle")}
    finite = all(np.isfinite(np.asarray(getattr(data, field))).all()
                 for field in ("qpos", "qvel", "qacc", "ctrl", "actuator_force"))
    result = {"timestamp_s": now, "finite": bool(finite), "joint_safe": joint_safe,
              "force_sample_scope": FORCE_SCOPE, "object_mass_kg": mass, "gravity_world_m_s2": gravity.tolist(),
              "object_weight_n": float(-mass * gravity[2]), "object_com_world_m": com.tolist(),
              "object_position_world_m": object_position.tolist(), "object_corners_world_m": corners.tolist(),
              "table_top_plane_point_world_m": plane_point.tolist(), "table_top_plane_normal_world": table_R[:, 2].tolist(),
              "minimum_bottom_clearance_m": clearance,
              "object_com_linear_velocity_world_m_s": velocity[3:].tolist(), "object_angular_velocity_world_rad_s": velocity[:3].tolist(),
              "object_linear_speed_m_s": float(np.linalg.norm(velocity[3:])), "object_angular_speed_rad_s": float(np.linalg.norm(velocity[:3])),
              "actual_object_in_palm_frame_m": relative.tolist(), "fixed_grip_reference": reference,
              "grip_reference_sha256": canonical_sha256(reference) if reference is not None else None,
              "grip_relative_drift_m": float(np.linalg.norm(relative - origin_relative)) if origin_relative is not None else None,
              "groups": groups, "loaded_table_contact_count": counts["table"], "loaded_other_contact_count": counts["other"],
              "table_normal_force_n": normal_totals["table"], "other_normal_force_n": normal_totals["other"],
              "opposition_dot_diagnostic": opposition, "opposition_scope": "diagnostic only; not force closure or measured load transfer", "contacts": contacts}
    result.update({key + ("_world_n" if key.endswith("force") else "_about_com_world_nm"): value.tolist()
                   for key, value in totals.items()})
    return result


class SupportWindow:
    """Require a current contiguous measured window; readiness is never latched."""
    KEYS = {"minimum_bottom_clearance_m", "maximum_table_support_fraction", "hand_weight_ratio_min", "hand_weight_ratio_max",
            "max_linear_speed_m_s", "max_angular_speed_rad_s", "max_grip_drift_m", "stable_window_s", "required_groups"}

    def __init__(self, settings, sample_period_s):
        if not isinstance(settings, dict) or set(settings) != self.KEYS:
            raise ValueError("support settings must contain the complete declared threshold set")
        if isinstance(sample_period_s, bool) or not isinstance(sample_period_s, (int, float)) or not math.isfinite(sample_period_s) or sample_period_s <= 0:
            raise ValueError("support sample period must be positive and finite")
        self.settings = copy.deepcopy(settings)
        for key in self.KEYS - {"required_groups"}:
            value = settings[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("support thresholds must be positive and finite")
        groups = settings["required_groups"]
        if (not isinstance(groups, list) or not groups or len(set(groups)) != len(groups)
                or set(groups) - {"thumb", "index", "middle", "palm"}):
            raise ValueError("required support groups must be unique named fingers/palm")
        if settings["hand_weight_ratio_min"] > settings["hand_weight_ratio_max"]:
            raise ValueError("hand weight ratio bounds are reversed")
        self.period = float(sample_period_s)
        self.previous_time = None
        self.window_start = None
        self.reference_hash = None
        self.maximum_contiguous_valid_s = 0.
        self._rows = []
        self._current_status = None

    @property
    def rows(self):
        return copy.deepcopy(self._rows)

    @property
    def ready(self):
        return bool(self._current_status and self._current_status["ready"])

    @property
    def last_status(self):
        return copy.deepcopy(self._current_status)

    def update(self, sample):
        # Rejected/malformed evidence must not leave a previous ready latch.
        self._current_status = None
        try:
            result = self._update(sample)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            self.window_start = None
            raise
        self._current_status = copy.deepcopy(result)
        return result

    def _update(self, sample):
        now = float(sample["timestamp_s"])
        if not math.isfinite(now) or self.previous_time is not None and now <= self.previous_time:
            raise ValueError("support samples must have strictly increasing finite timestamps")
        signature = sample.get("grip_reference_sha256")
        reference = sample.get("fixed_grip_reference")
        reference_present = signature is not None and reference is not None
        if reference_present:
            if (not isinstance(signature, str) or len(signature) != 64
                    or not isinstance(reference, dict) or canonical_sha256(reference) != signature):
                raise ValueError("support sample must bind its actual fixed grip reference")
            reference_time = float(reference["timestamp_s"])
            _array(reference["object_in_palm_frame_m"], (3,), "fixed grip reference")
            if not math.isfinite(reference_time) or reference_time > now + 1e-9:
                raise ValueError("fixed grip reference cannot come from a future state")
            if self.reference_hash is not None and signature != self.reference_hash:
                raise ValueError("load transfer grip reference cannot be reset inside the window")
        elif signature is not None or reference is not None:
            raise ValueError("support sample has incomplete fixed grip provenance")
        if sample.get("force_sample_scope") != FORCE_SCOPE:
            raise ValueError("support window requires declared post-refresh measured force scope")
        if not isinstance(sample.get("finite"), bool) or not isinstance(sample.get("joint_safe"), bool):
            raise ValueError("support samples require explicit measured finite/joint safety flags")
        fields = ("object_weight_n", "minimum_bottom_clearance_m", "table_normal_force_n", "object_linear_speed_m_s",
                  "object_angular_speed_rad_s")
        numeric = [float(sample[key]) for key in fields]
        force = _array(sample["hand_force_world_n"], (3,), "hand force")
        if not np.isfinite(numeric).all() or numeric[0] <= 0 or any(value < 0 for value in numeric[2:]):
            raise ValueError("support measurements must be finite with positive weight and nonnegative norms/load")
        weight, clearance, table_force, speed, omega = numeric
        drift = sample.get("grip_relative_drift_m")
        if drift is not None:
            drift = float(drift)
            if not math.isfinite(drift) or drift < 0:
                raise ValueError("fixed grip drift must be finite and nonnegative")
        ratio = float(force[2] / weight)
        groups = sample["groups"]
        for group in self.settings["required_groups"]:
            count = groups[group]["loaded_contact_count"]
            normal = float(groups[group]["normal_force_n"])
            if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                    or not math.isfinite(normal) or normal < 0):
                raise ValueError("support groups require measured nonnegative loaded count and normal force")
        for key in ("loaded_table_contact_count", "loaded_other_contact_count"):
            if isinstance(sample[key], bool) or not isinstance(sample[key], int) or sample[key] < 0:
                raise ValueError("support counts must be explicit nonnegative integers")
        opts = self.settings
        checks = {"finite": sample["finite"], "joint_safe": sample["joint_safe"],
                  "fixed_grip_reference_present": bool(reference_present),
                  "required_groups_loaded": all(groups[name]["loaded_contact_count"] > 0 and groups[name]["normal_force_n"] > 0 for name in opts["required_groups"]),
                  "no_loaded_table_contact": sample["loaded_table_contact_count"] == 0,
                  "table_support_unloaded": table_force <= opts["maximum_table_support_fraction"] * weight,
                  "no_other_loaded_external_contact": sample["loaded_other_contact_count"] == 0,
                  "hand_upward_supports_weight": opts["hand_weight_ratio_min"] <= ratio <= opts["hand_weight_ratio_max"],
                  "actual_box_bottom_clear": clearance >= opts["minimum_bottom_clearance_m"],
                  "linear_speed_stable": speed <= opts["max_linear_speed_m_s"],
                  "angular_speed_stable": omega <= opts["max_angular_speed_rad_s"],
                  "fixed_grip_drift_bounded": drift is not None and drift <= opts["max_grip_drift_m"]}
        additional = sample.get("additional_required_checks", {})
        if not isinstance(additional, dict):
            raise ValueError("additional required support checks must be a named boolean dictionary")
        for name, passed in additional.items():
            if not isinstance(name, str) or not name.strip() or name != name.strip() or not isinstance(passed, bool):
                raise ValueError("additional support checks require unique nonempty names and explicit booleans")
            checks["additional:" + name] = passed
        cadence = self.previous_time is None or math.isclose(now - self.previous_time, self.period, rel_tol=0, abs_tol=1e-9)
        current_valid = all(checks.values())
        if not current_valid:
            self.window_start = None
        elif self.window_start is None or not cadence:
            self.window_start = now
        duration = 0. if self.window_start is None else now - self.window_start
        self.maximum_contiguous_valid_s = max(self.maximum_contiguous_valid_s, duration)
        ready = bool(current_valid and cadence and duration + 1e-12 >= opts["stable_window_s"])
        result = {"timestamp_s": now, "sample_period_s": self.period, "cadence_contiguous": bool(cadence),
                  "checks": checks, "current_sample_valid": bool(current_valid), "contiguous_valid_s": duration,
                  "window_start_s": self.window_start, "maximum_contiguous_valid_s": self.maximum_contiguous_valid_s,
                  "hand_upward_weight_ratio": ratio, "ready": ready, "grip_reference_sha256": signature,
                  "failure_reasons": [key for key, passed in checks.items() if not passed] + ([] if cadence else ["sample_cadence_gap"])}
        if reference_present:
            self.reference_hash = signature
        self.previous_time = now
        self._rows.append(copy.deepcopy(result))
        return copy.deepcopy(result)
