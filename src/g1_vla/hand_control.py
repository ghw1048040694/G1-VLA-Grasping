"""Contact-normal equivalent position targets through the unchanged shared PD.

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


class ContactHandControl:
    def __init__(self, model, data, controller, object_body, settings, nominal, baseline, api):
        self.model, self.data, self.api, self.object_body = model, data, api, object_body
        self.controller, self.settings = controller, settings
        self.nominal, self.baseline = np.asarray(nominal), np.asarray(baseline)
        self.groups = {finger: [i for i, name in enumerate(controller.contract.joint_names)
            if name.startswith("right_hand_" + finger + "_")] for finger in ("thumb", "index", "middle")}
        self.touched = {finger: False for finger in self.groups}
        self.rows = []

    def target(self, close, phase, grip_lock):
        model, data, api, opts = self.model, self.data, self.api, self.settings
        result = self.baseline.copy()
        if close <= 0.:
            return result
        sample = native_hand_contacts(model, data, self.object_body, api)
        drift = None
        if grip_lock is not None:
            palm = api.mj_name2id(model, api.mjtObj.mjOBJ_SITE, "right_palm_center")
            rotation = np.asarray(data.site_xmat[palm]).reshape(3, 3)
            relative = rotation.T @ (data.xpos[self.object_body] - data.site_xpos[palm])
            drift = float(np.linalg.norm(relative - np.asarray(grip_lock["object_in_palm_frame_m"])))
        details = {}
        for finger, indices in self.groups.items():
            bindings = [self.controller.bindings[i] for i in indices]
            actual = np.array([data.qpos[b["qpos_id"]] for b in bindings])
            nominal = self.baseline[indices] + close * (self.nominal[indices] - self.baseline[indices])
            direction = np.sign(self.nominal[indices] - self.baseline[indices])
            group = sample["groups"][finger]
            force = group["normal_force_n"]
            loaded = [c for c in sample["contacts"] if c["group"] == finger
                and c["distance_m"] <= 0 and c["efc_address"] >= 0 and c["force_contact_frame"][0] > 0.]
            self.touched[finger] |= force >= opts["minimum_loaded_force_n"]
            torque = np.zeros(len(indices))
            posture = np.zeros(len(indices))
            if group["maximum_penetration_m"] > opts["maximum_penetration_m"] or force > opts["maximum_group_force_n"]:
                wanted = actual - direction * opts["relief_step_rad"]
                mode = "bounded_unload"
            elif loaded:
                jacobians = []
                for contact in loaded:
                    object_index = contact["object_geom_index"]
                    hand_geom = contact["geom_ids"][1 - object_index]
                    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                    api.mj_jac(model, data, jacp, jacr, np.asarray(contact["position_world_m"]), int(model.geom_bodyid[hand_geom]))
                    into = np.asarray(contact["frame_rows_world"])[0] * (1. if object_index == 1 else -1.)
                    jacobians.append(into @ jacp[:, [b["qvel_id"] for b in bindings]])
                goal = max(0., min(2 * opts["normal_force_target_n"],
                    opts["normal_force_target_n"] + opts["force_feedback_gain"] * (opts["normal_force_target_n"] - force)))
                kp = unitree_gains(self.controller.contract.joint_names[indices[0]])[0] * self.controller.hand_gain_scale
                wanted, torque, posture = equivalent_position(actual, nominal, jacobians, kp, goal,
                    opts["posture_gain"], opts["maximum_posture_delta_rad"])
                mode = "normal_force_equivalent_position"
            elif drift is not None and drift > opts["maximum_recovery_drift_m"]:
                wanted, mode = actual.copy(), "lost_object_recovery_inhibited"
            elif not self.touched[finger]:
                wanted, mode = nominal, "original_closure_search"
            else:
                wanted = actual + direction * opts["takeup_velocity_rad_s"] * self.controller.cadence.period_s
                mode = "bounded_lost_finger_takeup"
            lower = np.minimum(self.baseline[indices], self.baseline[indices] + opts["maximum_closing_scale"] * (self.nominal[indices] - self.baseline[indices]))
            upper = np.maximum(self.baseline[indices], self.baseline[indices] + opts["maximum_closing_scale"] * (self.nominal[indices] - self.baseline[indices]))
            for j, binding in enumerate(bindings):
                mechanical = model.jnt_range[binding["joint_id"]]
                lower[j], upper[j] = max(lower[j], mechanical[0]), min(upper[j], mechanical[1])
            bounded = np.clip(wanted, lower, upper)
            result[indices] = bounded
            details[finger] = {"mode": mode, "measured_force_n": force,
                "maximum_native_penetration_m": group["maximum_penetration_m"],
                "actual_q": actual.tolist(), "raw_requested_q": wanted.tolist(), "bounded_requested_q": bounded.tolist(),
                "equivalent_normal_torque_nm": torque.tolist(), "nullspace_posture_delta_rad": posture.tolist()}
        self.rows.append({"time_s": float(data.time), "phase": phase, "close_fraction": float(close),
                          "fixed_grip_relative_drift_m": drift, "groups": details})
        return result
