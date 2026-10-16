"""Contact-normal position feedback through the unchanged shared PD.

Privileged teacher only. Force goals are candidate setpoints, not measured
forces or a force-closure certificate. All requests still pass the common
mechanical, anchor and adjacent-command velocity limits.
"""

import numpy as np
from .hand_contact import native_hand_contacts
from .control import unitree_gains


def equivalent_position(actual, nominal, normal_jacobian, kp, group_force,
                        posture_gain, maximum_posture_delta):
    actual, nominal = np.asarray(actual), np.asarray(nominal)
    matrix = np.asarray(normal_jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(actual) or not len(matrix):
        raise ValueError("contact Jacobian must contain one row per actual surface contact")
    if (not np.isfinite(matrix).all() or not np.isfinite(actual).all() or not np.isfinite(nominal).all()
            or not np.isfinite([group_force, kp, posture_gain, maximum_posture_delta]).all()
            or group_force < 0 or kp <= 0 or posture_gain < 0 or maximum_posture_delta < 0):
        raise ValueError("normal target requires finite state/Jacobian and nonnegative force")
    torque = matrix.sum(axis=0) * group_force / len(matrix)
    nullspace = np.eye(len(actual)) - np.linalg.pinv(matrix, rcond=1e-7) @ matrix
    posture = nullspace @ (nominal - actual) * posture_gain
    norm = float(np.max(np.abs(posture)))
    if norm > maximum_posture_delta:
        posture *= maximum_posture_delta / norm
    return actual + torque / kp + posture, torque, posture


# Numerical cancellation threshold for a Jacobian sum, in metres/radian.
# This is not a force, contact, penetration or task acceptance threshold.
NORMAL_DIRECTION_EPS_M_PER_RAD = 1e-10


def pressure_position_step(previous_applied, nominal, normal_jacobian, measured_force,
                           force_goal, adjustment_rate, period_s, maximum_force_step,
                           posture_gain, maximum_posture_delta):
    """Integrate pressure error from the last *executed* position reference.

    The normalized net normal determines joint-space direction. Its magnitude
    is not a force-to-torque gain: rate has units rad/(N*s). Full contact rows,
    including opposed normals, define the posture nullspace independently.
    No integrator state or unapplied request is retained by this pure helper.
    """
    previous = np.asarray(previous_applied, dtype=float).copy()
    nominal = np.asarray(nominal, dtype=float).copy()
    matrix = np.asarray(normal_jacobian, dtype=float).copy()
    if (previous.ndim != 1 or not len(previous) or nominal.shape != previous.shape
            or matrix.ndim != 2 or not len(matrix) or matrix.shape[1] != len(previous)):
        raise ValueError("pressure feedback requires aligned joint vectors and nonempty normal rows")
    scalars = np.asarray([measured_force, force_goal, adjustment_rate, period_s,
                          maximum_force_step, posture_gain, maximum_posture_delta], dtype=float)
    if (not np.isfinite(previous).all() or not np.isfinite(nominal).all() or not np.isfinite(matrix).all()
            or not np.isfinite(scalars).all() or np.any(scalars < 0) or period_s <= 0):
        raise ValueError("pressure feedback state, contacts and nonnegative settings must be finite")
    net = matrix.sum(axis=0)
    net_norm = float(np.linalg.norm(net))
    usable = net_norm > NORMAL_DIRECTION_EPS_M_PER_RAD
    direction = net / net_norm if usable else np.zeros_like(previous)
    error = float(force_goal - measured_force)
    uncapped = direction * adjustment_rate * error * period_s
    pressure = np.clip(uncapped, -maximum_force_step, maximum_force_step)
    nullspace = np.eye(len(previous)) - np.linalg.pinv(matrix, rcond=1e-7) @ matrix
    posture = nullspace @ (nominal - previous) * posture_gain
    norm = float(np.max(np.abs(posture)))
    if norm > maximum_posture_delta:
        posture *= maximum_posture_delta / norm
    requested = previous + pressure + posture
    if not all(np.isfinite(value).all() for value in (requested, pressure, posture)):
        raise ValueError("pressure feedback produced a nonfinite request")
    return {"requested": requested, "pressure_increment": pressure,
            "uncapped_pressure_increment": uncapped, "posture_increment": posture,
            "normal_direction": direction, "net_normal": net, "net_normal_norm": net_norm,
            "normal_direction_usable": bool(usable), "force_error_n": error}


class ContactHandControl:
    def __init__(self, model, data, controller, object_body, settings, nominal, baseline, api):
        if settings.get("backend") != "contact_normal_position_v2":
            raise ValueError("ContactHandControl v2 requires explicit contact_normal_position_v2 backend")
        self.model, self.data, self.api, self.object_body = model, data, api, object_body
        self.controller, self.settings = controller, settings
        self.nominal, self.baseline = np.asarray(nominal, dtype=float).copy(), np.asarray(baseline, dtype=float).copy()
        size = len(controller.contract.joint_names)
        if (self.nominal.shape != (size,) or self.baseline.shape != (size,)
                or not np.isfinite(self.nominal).all() or not np.isfinite(self.baseline).all()):
            raise ValueError("hand targets must match the finite shared joint order")
        self.groups = {finger: [i for i, name in enumerate(controller.contract.joint_names)
            if name.startswith("right_hand_" + finger + "_")] for finger in ("thumb", "index", "middle")}
        self.touched = {finger: False for finger in self.groups}
        self.rows = []

    def target(self, close, phase, grip_lock):
        model, data, api, opts = self.model, self.data, self.api, self.settings
        result = self.baseline.copy()
        if not np.isfinite(close) or close < 0 or close > 1:
            raise ValueError("hand closure fraction must be finite in [0,1]")
        if close <= 0.:
            return result
        previous_all = np.asarray(self.controller.previous_target, dtype=float).copy()
        if previous_all.shape != result.shape or not np.isfinite(previous_all).all():
            raise ValueError("hand feedback requires the actual previous applied command")
        sample = native_hand_contacts(model, data, self.object_body, api)
        drift = None
        if grip_lock is not None:
            palm = api.mj_name2id(model, api.mjtObj.mjOBJ_SITE, "right_palm_center")
            rotation = np.asarray(data.site_xmat[palm]).reshape(3, 3)
            relative = rotation.T @ (data.xpos[self.object_body] - data.site_xpos[palm])
            drift = float(np.linalg.norm(relative - np.asarray(grip_lock["object_in_palm_frame_m"])))
            if not np.isfinite(drift):
                raise ValueError("fixed grip drift must be finite")
        details = {}
        for finger, indices in self.groups.items():
            if not indices:
                raise ValueError("contact controller requires all three right finger groups")
            bindings = [self.controller.bindings[i] for i in indices]
            actual = np.array([data.qpos[b["qpos_id"]] for b in bindings])
            previous = previous_all[indices].copy()
            nominal = self.baseline[indices] + close * (self.nominal[indices] - self.baseline[indices])
            direction = np.sign(self.nominal[indices] - self.baseline[indices])
            group = sample["groups"][finger]
            force = float(group["normal_force_n"])
            depth = float(group["maximum_penetration_m"])
            if not np.isfinite(actual).all() or not np.isfinite([force, depth]).all() or force < 0 or depth < 0:
                raise ValueError("native hand feedback must be finite with nonnegative load/depth")
            loaded = [c for c in sample["contacts"] if c["group"] == finger
                and c["distance_m"] <= 0 and c["efc_address"] >= 0 and c["force_contact_frame"][0] > 0.]
            self.touched[finger] |= force >= opts["minimum_loaded_force_n"]
            pressure = np.zeros(len(indices)); uncapped = pressure.copy(); posture = pressure.copy()
            normal_direction = pressure.copy(); net_normal = pressure.copy(); net_norm = 0.; normal_usable = False
            torque = pressure.copy(); jacobians = []; feedback_error = opts["normal_force_target_n"] - force
            goal = float(opts["normal_force_target_n"])
            # Any native overlap in this group, including an unloaded mesh,
            # takes precedence over pressure integration and lost-finger search.
            if depth > opts["maximum_penetration_m"] or force > opts["maximum_group_force_n"]:
                wanted = actual - direction * opts["relief_step_rad"]
                mode = "bounded_unload"
            elif loaded:
                for contact in loaded:
                    object_index = contact["object_geom_index"]
                    hand_geom = contact["geom_ids"][1 - object_index]
                    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    api.mj_jac(model, data, jacp, jacr, np.asarray(contact["position_world_m"]), int(model.geom_bodyid[hand_geom]))
                    into = np.asarray(contact["frame_rows_world"])[0] * (1. if object_index == 1 else -1.)
                    jacobians.append(into @ jacp[:, [b["qvel_id"] for b in bindings]])
                update = pressure_position_step(previous, nominal, jacobians, force, goal,
                    opts["normal_adjustment_rate_rad_n_s"], self.controller.cadence.period_s,
                    opts["maximum_force_step_rad"], opts["posture_gain"], opts["maximum_posture_delta_rad"])
                wanted, pressure, posture = update["requested"], update["pressure_increment"], update["posture_increment"]
                uncapped, normal_direction = update["uncapped_pressure_increment"], update["normal_direction"]
                net_normal, net_norm, normal_usable = update["net_normal"], update["net_normal_norm"], update["normal_direction_usable"]
                # The old J^T F equivalent is retained only for interpretation.
                # It is not added to the request and is never actuator torque.
                torque = np.asarray(jacobians).sum(axis=0) * goal / len(jacobians)
                mode = "applied_reference_pressure_feedback" if normal_usable else "cancelled_net_normal_no_pressure_increment"
            elif drift is not None and drift > opts["maximum_recovery_drift_m"]:
                wanted, mode = actual.copy(), "lost_object_recovery_inhibited"
            elif not self.touched[finger]:
                wanted, mode = nominal, "original_closure_search"
            elif drift is None:
                wanted, mode = previous.copy(), "lost_finger_takeup_requires_fixed_grip"
            else:
                wanted = previous + direction * opts["takeup_velocity_rad_s"] * self.controller.cadence.period_s
                mode = "bounded_lost_finger_takeup"
            lower = np.minimum(self.baseline[indices], self.baseline[indices] + opts["maximum_closing_scale"] * (self.nominal[indices] - self.baseline[indices]))
            upper = np.maximum(self.baseline[indices], self.baseline[indices] + opts["maximum_closing_scale"] * (self.nominal[indices] - self.baseline[indices]))
            for j, binding in enumerate(bindings):
                mechanical = model.jnt_range[binding["joint_id"]]
                lower[j], upper[j] = max(lower[j], mechanical[0]), min(upper[j], mechanical[1])
            if np.any(lower > upper) or not np.isfinite(wanted).all():
                raise ValueError("hand request or mechanical/closing bound intersection is invalid")
            bounded = np.clip(wanted, lower, upper)
            result[indices] = bounded
            kp = unitree_gains(self.controller.contract.joint_names[indices[0]])[0] * self.controller.hand_gain_scale
            details[finger] = {"mode": mode, "effective_force_goal_n": goal, "measured_force_n": force,
                "force_error_n": float(feedback_error), "maximum_native_penetration_m": depth,
                "previous_applied_q": previous.tolist(), "actual_q": actual.tolist(), "nominal_q": nominal.tolist(),
                "raw_requested_q": wanted.tolist(), "bounded_requested_q": bounded.tolist(),
                "closing_direction": direction.tolist(), "normal_jacobian_rows_m_per_rad": np.asarray(jacobians).reshape(-1, len(indices)).tolist(),
                "net_normal_jacobian_m_per_rad": net_normal.tolist(), "net_normal_jacobian_norm_m_per_rad": net_norm,
                "normalized_net_normal_direction": normal_direction.tolist(), "normal_direction_usable": normal_usable,
                "normal_direction_numeric_epsilon_m_per_rad": NORMAL_DIRECTION_EPS_M_PER_RAD,
                "pressure_increment_rad": pressure.tolist(), "uncapped_pressure_increment_rad": uncapped.tolist(),
                "nullspace_posture_delta_rad": posture.tolist(),
                "theoretical_equivalent_normal_torque_nm": torque.tolist(),
                "theoretical_equivalent_position_delta_rad": (torque / kp).tolist(),
                "equivalent_torque_scope": "J_transpose_F diagnostic only, not applied torque or request increment",
                "recovery_drift_guard_passed": drift is not None and drift <= opts["maximum_recovery_drift_m"]}
        self.rows.append({"time_s": float(data.time), "phase": phase, "close_fraction": float(close),
                          "backend": opts["backend"], "reference_origin": "controller.previous_target_last_actually_applied",
                          "control_period_s": float(self.controller.cadence.period_s),
                          "fixed_grip_relative_drift_m": drift, "groups": details})
        return result
