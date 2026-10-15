"""Privileged grasp-centre Cartesian references, checked without stepping physics.

All kinematics/contact queries use a separate MjData copied from the visited
state. The free object is neither attached nor repositioned for planning.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .contracts import canonical_sha256
from .control import _api, controlled_joints


class TrajectoryError(ValueError):
    def __init__(self, message, detail=None):
        super().__init__(message)
        self.detail = detail or {}


def rotation_error(first, second):
    from scipy.spatial.transform import Rotation
    return float(np.linalg.norm(Rotation.from_matrix(np.asarray(first).T @ np.asarray(second)).as_rotvec()))


def grasp_center(position, rotation, offset):
    return np.asarray(position, dtype=float) + np.asarray(rotation, dtype=float) @ np.asarray(offset, dtype=float)


def bounded_absolute_jacobian(function, values, lower, upper, step_rad):
    """Absolute central differences, or one-sided differences at joint bounds.

    The denominator is the actual sampled interval, including narrow bounds.
    Re-evaluating the unperturbed vector restores the kinematic workspace.
    """
    q, low, high = (np.asarray(vector, dtype=float).copy() for vector in (values, lower, upper))
    if (q.ndim != 1 or low.shape != q.shape or high.shape != q.shape
            or not all(np.isfinite(vector).all() for vector in (q, low, high))
            or np.any(low > high) or np.any(q < low) or np.any(q > high)
            or not math.isfinite(step_rad) or step_rad <= 0):
        raise TrajectoryError("absolute Jacobian requires finite bounded joint vectors and positive step")
    base = np.asarray(function(q.copy()), dtype=float).copy()
    if not np.isfinite(base).all():
        raise TrajectoryError("absolute Jacobian requires finite residuals")
    jacobian = np.zeros(base.shape + (len(q),), dtype=float)
    try:
        for index in range(len(q)):
            if q[index] - step_rad >= low[index] and q[index] + step_rad <= high[index]:
                first, second = q[index] - step_rad, q[index] + step_rad
            elif q[index] + step_rad <= high[index]:
                first, second = q[index], q[index] + step_rad
            elif q[index] - step_rad >= low[index]:
                first, second = q[index] - step_rad, q[index]
            else:
                first, second = low[index], high[index]
            if second == first:
                continue
            sampled = []
            for value in (first, second):
                if value == q[index]:
                    result = base
                else:
                    probe = q.copy()
                    probe[index] = value
                    result = np.asarray(function(probe), dtype=float)
                if result.shape != base.shape or not np.isfinite(result).all():
                    raise TrajectoryError("absolute Jacobian residual shape/finiteness changed")
                sampled.append(result.copy())
            jacobian[..., index] = (sampled[1] - sampled[0]) / (second - first)
    finally:
        function(q.copy())
    if not np.isfinite(jacobian).all():
        raise TrajectoryError("absolute Jacobian produced non-finite derivatives")
    return jacobian


def project_pose_feasibility(initial, lower, upper, evaluate, *, position_tolerance_m,
                             orientation_tolerance_rad, position_weight, orientation_weight,
                             distance_scale_m, max_iterations, finite_difference_rad,
                             continuity_reference, continuity_weight):
    """One bounded projection from the existing soft solution, with unit-scale constraints.

    ``evaluate`` supplies C_error_m, rotation_error_vector_rad and the fixed
    source-active self_pair_distances_m. Native contacts remain the caller's
    separate hard check. A solver status alone never establishes feasibility.
    This helper is not a seed search and does not change any acceptance bound.
    """
    from scipy.optimize import minimize
    q0, low, high = (np.asarray(values, dtype=float).copy() for values in (initial, lower, upper))
    reference = np.asarray(continuity_reference, dtype=float).copy()
    values = (position_tolerance_m, position_weight, orientation_weight, distance_scale_m,
              finite_difference_rad, continuity_weight)
    if (q0.ndim != 1 or q0.shape != low.shape or q0.shape != high.shape
            or not all(np.isfinite(vector).all() for vector in (q0, low, high))
            or np.any(low > high) or np.any(q0 < low) or np.any(q0 > high)
            or reference.shape != q0.shape or not np.isfinite(reference).all()
            or np.any(reference < low) or np.any(reference > high)
            or isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations <= 0
            or any(not math.isfinite(value) or value <= 0 for value in values)
            or orientation_tolerance_rad is not None and (
                not math.isfinite(orientation_tolerance_rad) or orientation_tolerance_rad <= 0)):
        raise TrajectoryError("invalid bounded feasibility projection inputs")
    interior = 1. - 1e-6
    previous, cached = None, None

    def evaluated(q):
        nonlocal previous, cached
        if previous is None or not np.array_equal(q, previous):
            row = evaluate(np.asarray(q).copy())
            cached = {key: np.asarray(row[key], dtype=float).copy() for key in
                      ("C_error_m", "rotation_error_vector_rad", "self_pair_distances_m")}
            if (cached["C_error_m"].shape != (3,) or cached["rotation_error_vector_rad"].shape != (3,)
                    or cached["self_pair_distances_m"].shape != (9,)
                    or not all(np.isfinite(vector).all() for vector in cached.values())):
                raise TrajectoryError("projection evaluator returned non-finite or malformed residuals")
            previous = np.asarray(q).copy()
        return cached

    def objective(q):
        row = evaluated(q)
        scale = position_weight * position_tolerance_m
        return float(np.sum((row["C_error_m"] / position_tolerance_m) ** 2)
                     + np.sum((orientation_weight * row["rotation_error_vector_rad"] / scale) ** 2)
                     + np.sum((continuity_weight * (np.asarray(q) - reference) / scale) ** 2))

    def inequalities(q):
        row = evaluated(q)
        result = [1. - float(np.sum((row["C_error_m"] / (position_tolerance_m * interior)) ** 2))]
        result.extend((row["self_pair_distances_m"] / distance_scale_m - 1e-6).tolist())
        if orientation_tolerance_rad is not None:
            result.append(1. - float(np.sum((row["rotation_error_vector_rad"]
                                            / (orientation_tolerance_rad * interior)) ** 2)))
        return np.asarray(result)

    result = minimize(objective, q0, method="SLSQP", bounds=list(zip(low, high)),
                      constraints={"type": "ineq", "fun": inequalities},
                      options={"maxiter": int(max_iterations), "ftol": 1e-10, "eps": finite_difference_rad})
    final, scaled = evaluated(result.x), inequalities(result.x)
    C_error = float(np.linalg.norm(final["C_error_m"]))
    R_error = float(np.linalg.norm(final["rotation_error_vector_rad"]))
    hard = bool(np.all(result.x >= low) and np.all(result.x <= high)
                and C_error <= position_tolerance_m and np.min(final["self_pair_distances_m"]) >= 0.
                and (orientation_tolerance_rad is None or R_error <= orientation_tolerance_rad))
    return {"q": result.x.tolist(), "optimizer_converged": bool(result.success), "status": int(result.status),
            "message": str(result.message), "iterations": int(result.nit), "function_evaluations": int(result.nfev),
            "mathematical_constraints_passed": hard, "native_contacts_checked": False,
            "taskpoint_error_m": C_error, "orientation_error_rad": R_error,
            "self_pair_distances_m": final["self_pair_distances_m"].tolist(),
            "scaled_inequalities": scaled.tolist(), "minimum_scaled_inequality": float(np.min(scaled)),
            "normalization": {"position_m": position_tolerance_m, "self_distance_m": distance_scale_m,
                "closed_orientation_rad": orientation_tolerance_rad, "internal_interior_factor": interior,
                "self_distance_internal_inset_m": distance_scale_m * 1e-6},
            "finite_difference_rad": float(finite_difference_rad),
            "finite_difference_scope": "SLSQP absolute difference step; soft IK uses declared bound-aware absolute residual Jacobian",
            "continuity_reference_q": reference.tolist(), "joint_continuity_weight": float(continuity_weight),
            "scope": "single projection of the current soft IK solution; native contacts still require final rejection checks"}


def path_velocity_budget(waypoints, velocity, period_s, anchor_delta, nominal_s, timeout_s,
                         *, closure_s=0., tracking_reserve_rad=0.):
    """Continuous polyline timing; waypoints need not each consume a full tick.

    This is a reference lower bound, never a claim about physical settling.
    A tracking reserve makes the declared anchor bound conservative.
    """
    points, vmax = np.asarray(waypoints, dtype=float), np.asarray(velocity, dtype=float)
    reserve = np.asarray(anchor_delta, dtype=float) - tracking_reserve_rad
    if (points.ndim != 2 or points.shape[1:] != vmax.shape or len(points) < 1
            or not np.isfinite(points).all() or not np.isfinite(vmax).all() or not np.isfinite(reserve).all()
            or not math.isfinite(period_s) or period_s <= 0 or np.any(vmax <= 0) or np.any(reserve <= 0)):
        raise TrajectoryError("invalid trajectory timing dimensions or non-positive bound")
    effective = np.minimum(vmax, reserve / period_s)
    delta = np.abs(np.diff(points, axis=0))
    times = np.max(delta / effective[None], axis=1) if len(delta) else np.asarray([])
    continuous = float(times.sum())
    rounded_once = math.ceil(max(0., continuous / period_s - 1e-10)) * period_s
    required = max(float(nominal_s), rounded_once) + closure_s
    return {"segment_reference_time_s": times.tolist(), "continuous_reference_time_s": continuous,
            "rounded_reference_time_s": rounded_once, "minimum_nominal_time_s": float(nominal_s),
            "closing_time_s": float(closure_s), "original_timeout_s": float(timeout_s),
            "required_reference_time_s": required, "within_original_timeout": bool(required <= timeout_s + 1e-9),
            "tracking_reserve_rad": float(tracking_reserve_rad),
            "scope": "reference feasibility only; actual tracking, contact and settling may still timeout"}


class CartesianTeacherPlanner:
    """Plan one original phase from its true visited state and prior command."""

    def __init__(self, model, config, contract, *, mujoco_api=None):
        from .expert import arm_ik_contract, arm_torso_collision_pairs
        self.api, self.model, self.config, self.contract = _api(mujoco_api), model, config, contract
        self.profile, self.settings = config["expert"]["teacher"], config["expert"]["trajectory"]
        self.names, self.ids, self.lower, self.upper = arm_ik_contract(model, hand="right", mujoco_api=self.api)
        self.action_indices = [contract.joint_names.index(name) for name in self.names]
        self.velocity = np.asarray(contract.max_velocity_rad_s)[self.action_indices]
        self.anchor = np.asarray(contract.max_anchor_delta_rad)[self.action_indices]
        self.period = 1. / contract.control_hz
        self.center = np.asarray(self.profile["grasp_center_in_palm_m"], dtype=float)
        self.palm = self.api.mj_name2id(model, self.api.mjtObj.mjOBJ_SITE, "right_palm_center")
        self.object_body = self.api.mj_name2id(model, self.api.mjtObj.mjOBJ_BODY, "green_cube")
        self.pairs = arm_torso_collision_pairs(model, include_elbow=True, mujoco_api=self.api)
        self.pair_evidence = [{"geom_ids": [first, second],
            "geom_names": [self.api.mj_id2name(model, self.api.mjtObj.mjOBJ_GEOM, gid) for gid in (first, second)],
            "body_names": [self.api.mj_id2name(model, self.api.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid]))
                           for gid in (first, second)]} for first, second in self.pairs]
        self.pair_signature = canonical_sha256(self.pair_evidence)
        self.workspace = self.api.MjData(model)
        self.controlled = controlled_joints(model, mujoco_api=self.api)
        self.hand_geoms = set()
        for gid in range(model.ngeom):
            body = self.api.mj_id2name(model, self.api.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
            geom = self.api.mj_id2name(model, self.api.mjtObj.mjOBJ_GEOM, gid) or ""
            if body.startswith("right_hand_") or geom == "right_palm_contact_pad":
                self.hand_geoms.add(gid)
        self.contact_rotation = None
        self.last_failure, self.attempts = None, []
        if (not isinstance(self.settings["interpolation_samples"], int)
                or self.settings["interpolation_samples"] < 2):
            raise TrajectoryError("chord checks require at least two samples including both endpoints")

    def state_snapshot(self, data):
        return {"time_s": float(data.time), "qpos": np.asarray(data.qpos).tolist(),
                "qvel": np.asarray(data.qvel).tolist(), "nq": int(self.model.nq), "nv": int(self.model.nv)}

    def actual_pose(self, data):
        position = np.asarray(data.site_xpos[self.palm], dtype=float).copy()
        rotation = np.asarray(data.site_xmat[self.palm], dtype=float).reshape(3, 3).copy()
        return {"palm": position, "R": rotation, "C": grasp_center(position, rotation, self.center),
                "arm": np.asarray(data.qpos[self.ids], dtype=float).copy()}

    def _copy_visited(self, data):
        self.workspace.qpos[:] = data.qpos
        self.workspace.qvel[:] = data.qvel
        self.workspace.time = data.time
        self.api.mj_forward(self.model, self.workspace)

    def sync_actual(self, data):
        """Refresh every free joint and hand joint before online chord queries."""
        self._copy_visited(data)

    def reference_pose(self, q):
        self.workspace.qpos[self.ids] = q
        self.api.mj_forward(self.model, self.workspace)
        return self.actual_pose(self.workspace)

    def _contacts(self):
        from .expert import _robot_self_contacts
        self_contacts = _robot_self_contacts(self.model, self.workspace, self.object_body, self.api)
        hand = []
        for index in range(self.workspace.ncon):
            contact = self.workspace.contact[index]
            if contact.dist > 0:
                continue
            geoms = (int(contact.geom1), int(contact.geom2))
            bodies = tuple(int(self.model.geom_bodyid[geom]) for geom in geoms)
            if self.object_body in bodies and any(geom in self.hand_geoms for geom in geoms):
                hand.append({"geom_ids": list(geoms), "distance_m": float(contact.dist),
                             "body_names": [self.api.mj_id2name(self.model, self.api.mjtObj.mjOBJ_BODY, body)
                                            or str(body) for body in bodies]})
        distances = [float(self.api.mj_geomDistance(self.model, self.workspace, first, second,
                                                   self.profile["ik_collision_margin_m"] + .05, None))
                     for first, second in self.pairs]
        return {"self_pair_distances_m": distances, "minimum_self_pair_distance_m": min(distances),
                "generated_robot_self_contacts": self_contacts, "generated_hand_object_contacts": hand}

    def _solve(self, target, orientation, seed, *, closed=False):
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
        target, orientation = np.asarray(target), np.asarray(orientation)
        orientation_weight = (self.settings["closed_orientation_weight"] if closed
                              else self.profile["ik_orientation_weight"])
        collision_weight = self.profile["ik_collision_weight"] * self.settings["collision_cost_scale"]
        continuity_reference = np.clip(np.asarray(seed, dtype=float), self.lower, self.upper).copy()
        continuity_weight = self.settings["joint_continuity_weight"]

        def residual(values):
            pose = self.reference_pose(values)
            contacts = self._contacts()
            collision = collision_weight * np.minimum(
                0., np.asarray(contacts["self_pair_distances_m"]) - self.profile["ik_collision_margin_m"])
            return np.concatenate((self.settings["position_weight"] * (pose["C"] - target),
                                   orientation_weight * Rotation.from_matrix(orientation.T @ pose["R"]).as_rotvec(),
                                   continuity_weight * (values - continuity_reference), collision))

        def jacobian(values):
            return bounded_absolute_jacobian(residual, values, self.lower, self.upper,
                                             self.settings["solver_finite_difference_rad"])

        result = least_squares(residual, np.clip(continuity_reference, self.lower + 1e-6, self.upper - 1e-6),
                               bounds=(self.lower, self.upper), max_nfev=self.settings["solver_max_nfev"],
                               jac=jacobian, xtol=1e-10, ftol=1e-10, gtol=1e-10)

        def describe(values, optimizer_converged):
            # Always recompute FK and *all* native contacts at this candidate.
            pose, contacts = self.reference_pose(values), self._contacts()
            position_error = float(np.linalg.norm(pose["C"] - target))
            angle_error = rotation_error(orientation, pose["R"])
            mathematical = bool(np.all(values >= self.lower) and np.all(values <= self.upper)
                and position_error <= self.settings["taskpoint_tolerance_m"]
                and min(contacts["self_pair_distances_m"]) >= 0.
                and (not closed or angle_error <= self.settings["closed_orientation_tolerance_rad"]))
            native = bool(not contacts["generated_robot_self_contacts"]
                          and (closed or not contacts["generated_hand_object_contacts"]))
            return dict(contacts, taskpoint_error_m=position_error, orientation_error_rad=angle_error,
                q=values.tolist(), C=pose["C"].tolist(), R=pose["R"].tolist(), palm=pose["palm"].tolist(),
                target_C=target.tolist(), target_R=orientation.tolist(), optimizer_converged=bool(optimizer_converged),
                closed=closed, position_weight=self.settings["position_weight"], collision_weight=collision_weight,
                orientation_weight=orientation_weight, mathematical_constraints_passed=mathematical,
                native_contact_checks_passed=native, passed=bool(mathematical and native),
                continuity_reference_q=continuity_reference.tolist(), joint_continuity_weight=float(continuity_weight),
                soft_ik_jacobian={"method": "bound_aware_absolute_central_or_one_sided",
                                  "step_rad": self.settings["solver_finite_difference_rad"],
                                  "workspace_restored_to_unperturbed_q": True})

        candidate = result.x.copy()
        detail = describe(candidate, result.success)
        if not detail["mathematical_constraints_passed"]:
            soft_detail = detail

            def projection_residual(values):
                current = self.reference_pose(values)
                contacts = self._contacts()
                return {"C_error_m": current["C"] - target,
                    "rotation_error_vector_rad": Rotation.from_matrix(orientation.T @ current["R"]).as_rotvec(),
                    "self_pair_distances_m": contacts["self_pair_distances_m"]}

            projection = project_pose_feasibility(candidate, self.lower, self.upper, projection_residual,
                position_tolerance_m=self.settings["taskpoint_tolerance_m"],
                orientation_tolerance_rad=self.settings["closed_orientation_tolerance_rad"] if closed else None,
                position_weight=self.settings["position_weight"], orientation_weight=orientation_weight,
                distance_scale_m=self.profile["ik_collision_margin_m"],
                max_iterations=self.settings["solver_max_nfev"],
                finite_difference_rad=self.settings["solver_finite_difference_rad"],
                continuity_reference=continuity_reference, continuity_weight=continuity_weight)
            candidate = np.asarray(projection["q"], dtype=float)
            detail = describe(candidate, projection["optimizer_converged"])
            detail["soft_ik_before_projection"] = soft_detail
            detail["feasibility_projection"] = projection
            detail["fresh_FK_and_native_checks_after_projection"] = True
        self.attempts.append(detail)
        if not detail["passed"]:
            self.last_failure = detail
            raise TrajectoryError("Cartesian waypoint rejected by strict geometry/taskpoint checks", detail)
        return candidate.copy(), detail

    def validate_chord(self, first, second, context=None, *, closed=False, lock_rotation=None):
        samples = []
        spacing = self.settings["maximum_joint_sample_spacing_rad"]
        count = max(self.settings["interpolation_samples"],
                    int(math.ceil(float(np.max(np.abs(np.asarray(second) - first))) / spacing)) + 1)
        for alpha in np.linspace(0., 1., count):
            q = np.asarray(first) + alpha * (np.asarray(second) - first)
            pose, contacts = self.reference_pose(q), self._contacts()
            angle = rotation_error(lock_rotation, pose["R"]) if closed and lock_rotation is not None else None
            passed = (np.all(q >= self.lower) and np.all(q <= self.upper)
                      and contacts["minimum_self_pair_distance_m"] >= 0.
                      and not contacts["generated_robot_self_contacts"]
                      and (closed or not contacts["generated_hand_object_contacts"])
                      and (angle is None or angle <= self.settings["closed_orientation_tolerance_rad"]))
            samples.append(dict(contacts, alpha=float(alpha), q=q.tolist(), C=pose["C"].tolist(),
                                R=pose["R"].tolist(), closed_rotation_error_rad=angle, passed=bool(passed)))
        return {"passed": all(sample["passed"] for sample in samples), "samples": samples,
                "closed": closed, "maximum_joint_sample_spacing_rad": spacing,
                "sample_count": count, "context": context,
                "scope": "workspace FK/contact sampling; actual PD path remains separately measured"}

    def latch_grip(self, data):
        pose = self.actual_pose(data)
        obj_position = np.asarray(data.xpos[self.object_body], dtype=float)
        obj_rotation = np.asarray(data.xmat[self.object_body], dtype=float).reshape(3, 3)
        return {"timestamp_s": float(data.time), "palm_position_m": pose["palm"].tolist(),
                "R": pose["R"].tolist(), "C": pose["C"].tolist(), "raw_arm_qpos_rad": pose["arm"].tolist(),
                "object_position_m": obj_position.tolist(), "object_R": obj_rotation.tolist(),
                "object_in_palm_frame_m": (pose["R"].T @ (obj_position - pose["palm"])).tolist(),
                "object_rotation_in_palm_frame": (pose["R"].T @ obj_rotation).tolist(),
                "object_state_modified": False, "source": "actual attained post-closure grasp; free object remains dynamic"}

    def plan_phase(self, phase, data, previous_arm_command, *, grip_lock=None, previous_plan=None):
        from scipy.spatial.transform import Rotation
        started = time.monotonic()
        self._copy_visited(data)
        entry_snapshot = self.state_snapshot(data)
        actual = self.actual_pose(data)
        q0 = np.asarray(previous_arm_command, dtype=float).copy()
        if (q0.shape != self.lower.shape or np.any(q0 < self.lower) or np.any(q0 > self.upper)
                or np.any(np.abs(q0 - actual["arm"]) > self.anchor + 1e-9)):
            raise TrajectoryError("last applied arm reference is outside mechanical/current-anchor bounds")
        start = self.reference_pose(q0)
        object_position = np.asarray(data.xpos[self.object_body], dtype=float).copy()
        closed = phase in ("grasp", "lift", "hold")
        high_solution, lookahead = None, []
        waypoints, details = [q0], []
        target, goal_R = None, None

        def failure_context(kind, *, checks=None, budget=None):
            return {"phase": phase, "failure_kind": kind, "entry_full_state": entry_snapshot,
                    "raw_actual_start_q": actual["arm"].tolist(), "last_applied_start_q": q0.tolist(),
                    "object_position_m": object_position.tolist(),
                    "phase_target_C": None if target is None else np.asarray(target).tolist(),
                    "phase_target_R": None if goal_R is None else np.asarray(goal_R).tolist(),
                    "q_waypoints": [q.tolist() for q in waypoints], "waypoint_details": list(details),
                    "high_orientation_lookahead": list(lookahead), "chord_checks": checks, "budget": budget,
                    "collision_pairs": self.pair_evidence, "collision_pair_signature_sha256": self.pair_signature,
                    "closed": closed, "live_state_modified": False}

        def checked_solve(C, R, seed, *, closed=False):
            try:
                return self._solve(C, R, seed, closed=closed)
            except TrajectoryError as exc:
                exc.detail = dict(exc.detail, partial_plan=failure_context("waypoint_rejection"))
                self.last_failure = exc.detail
                raise

        if phase == "high_approach":
            # High transit may use a shorter audited joint route, away from
            # contact. Descending pre/contact phases retain Cartesian paths.
            nominal_R = Rotation.from_euler("z", self.profile["grasp_yaw_deg"], degrees=True).as_matrix()
            target = object_position + (0., 0., self.profile["high_pregrasp_height_m"])
            high_seed, high_initial = checked_solve(target, nominal_R, actual["arm"])
            lookahead.append(high_initial)
            _, final = checked_solve(object_position, nominal_R, high_seed)
            lookahead.append(final)
            self.contact_rotation = np.asarray(final["R"])
            goal_R = self.contact_rotation
            high_solution, high_final = checked_solve(target, goal_R, high_seed)
            lookahead.append(high_final)
        elif phase == "pregrasp":
            if self.contact_rotation is None:
                raise TrajectoryError("contact orientation was not admitted at the high approach")
            target, goal_R = object_position + (0., -self.profile["pregrasp_lateral_m"], self.profile["pregrasp_height_m"]), self.contact_rotation
        elif phase == "contact":
            target, goal_R = object_position, self.contact_rotation
        elif phase == "grasp":
            target, goal_R = actual["C"], actual["R"]
        elif phase == "lift":
            if grip_lock is None:
                raise TrajectoryError("lift requires a recorded actual post-grasp grip lock")
            target = np.asarray(grip_lock["C"]) + (0., 0., self.profile["lift_height_m"])
            goal_R = np.asarray(grip_lock["R"])
        elif phase == "hold":
            if previous_plan is None or grip_lock is None:
                raise TrajectoryError("hold requires the completed lift reference and grip lock")
            target, goal_R = np.asarray(previous_plan["target_C"]), np.asarray(grip_lock["R"])
        else:
            raise TrajectoryError("unsupported Cartesian teacher phase {}".format(phase))
        goal_R = np.asarray(goal_R)
        if phase == "high_approach":
            count = max(1, int(math.ceil(float(np.max(np.abs(high_solution - q0)))
                                         / self.settings["reference_tracking_rad"])))
            waypoints = [q0 + (index / count) * (high_solution - q0) for index in range(count + 1)]
            details = [lookahead[-1]]
        elif phase == "hold":
            waypoints = [q0, np.asarray(previous_plan["q_waypoints"][-1])]
            details = []
        else:
            count = max(1, int(math.ceil(float(np.linalg.norm(target - start["C"])) / self.settings["waypoint_spacing_m"])))
            delta_R = Rotation.from_matrix(start["R"].T @ goal_R).as_rotvec()
            waypoints, details = [q0], []
            for index in range(1, count + 1):
                progress = index / count
                C = start["C"] + progress * (target - start["C"])
                R = start["R"] @ Rotation.from_rotvec(progress * delta_R).as_matrix() if phase == "high_approach" else goal_R
                q, detail = checked_solve(C, R, waypoints[-1], closed=closed)
                waypoints.append(q)
                details.append(detail)
        chord_checks = [self.validate_chord(first, second, closed=closed, lock_rotation=goal_R)
                        for first, second in zip(waypoints[:-1], waypoints[1:])]
        if not all(check["passed"] for check in chord_checks):
            self.last_failure = {"phase": phase, "chord_checks": chord_checks,
                                 "partial_plan": failure_context("chord_geometry_rejection", checks=chord_checks)}
            raise TrajectoryError("Cartesian path interpolation crosses an unsafe geometry", self.last_failure)
        key = {"high_approach": "high_approach_s", "pregrasp": "pregrasp_s", "contact": "contact_approach_s",
               "grasp": "grasp_hold_s", "lift": "lift_duration_s", "hold": "final_hold_s"}[phase]
        nominal = self.profile[key]
        budget = path_velocity_budget(waypoints, self.velocity, self.period, self.anchor, nominal,
                                      nominal * self.profile["phase_timeout_scale"],
                                      closure_s=self.profile["close_duration_s"] if phase == "contact" else 0.,
                                      tracking_reserve_rad=self.settings["reference_tracking_rad"])
        if not budget["within_original_timeout"]:
            self.last_failure = {"phase": phase, "budget": budget,
                                 "partial_plan": failure_context("original_budget_rejection", checks=chord_checks, budget=budget)}
            raise TrajectoryError("Cartesian reference cannot fit the original phase time budget", self.last_failure)
        end = self.reference_pose(waypoints[-1])
        plan = {"phase": phase, "q_waypoints": [q.tolist() for q in waypoints],
                "joint_names": list(self.names), "target_C": np.asarray(target).tolist(), "target_R": goal_R.tolist(),
                "endpoint_C": end["C"].tolist(), "endpoint_R": end["R"].tolist(), "endpoint_palm": end["palm"].tolist(),
                "endpoint_orientation_error_rad": rotation_error(goal_R, end["R"]),
                "raw_actual_start_q": actual["arm"].tolist(), "last_applied_start_q": q0.tolist(),
                "waypoint_details": details, "chord_checks": chord_checks, "budget": budget, "closed": closed,
                "grip_lock": grip_lock, "planning_wall_s": time.monotonic() - started,
                "object_qpos_copied_from_actual": True, "live_state_modified": False,
                "pair_count": len(self.pairs), "center_local_m": self.center.tolist(),
                "collision_pairs": self.pair_evidence, "collision_pair_signature_sha256": self.pair_signature,
                "entry_full_state": entry_snapshot, "high_orientation_lookahead": lookahead,
                "route": "dense_audited_joint_transit" if phase == "high_approach" else "actual_grasp_center_cartesian",
                "model_joint_mapping": [{"name": self.api.mj_id2name(self.model, self.api.mjtObj.mjOBJ_JOINT, index),
                    "qpos_address": int(self.model.jnt_qposadr[index]), "dof_address": int(self.model.jnt_dofadr[index])}
                    for index in range(self.model.njnt)]}
        plan["reference_sha256"] = canonical_sha256({key: plan[key] for key in ("phase", "q_waypoints", "target_C", "target_R")})
        return plan

    def end_reached(self, plan, data):
        pose = self.actual_pose(data)
        return bool(np.max(np.abs(pose["arm"] - np.asarray(plan["q_waypoints"][-1]))) <= self.settings["reference_tracking_rad"]
                    and np.linalg.norm(pose["C"] - np.asarray(plan["target_C"])) <= self.settings["taskpoint_tolerance_m"]
                    and np.linalg.norm(pose["C"] - np.asarray(plan["endpoint_C"])) <= self.settings["reference_tracking_m"]
                    and (not plan["closed"] or rotation_error(plan["target_R"], pose["R"])
                         <= self.settings["closed_orientation_tolerance_rad"]))
