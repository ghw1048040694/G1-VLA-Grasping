"""Cartesian references preserve real states and reject unsafe soft IK results."""
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from g1_vla.teacher_trajectory import (CartesianTeacherPlanner, TrajectoryError,
                                       grasp_center, path_velocity_budget, rotation_error,
                                       project_pose_feasibility, bounded_absolute_jacobian)


class TrajectoryTests(unittest.TestCase):
    def test_absolute_jacobian_respects_bounds_actual_intervals_and_restores_workspace(self):
        q = np.array([.2, 0., 1., .000005])
        low, high = np.array([-1., 0., 0., 0.]), np.array([1., 1., 1., .00001])
        seen, workspace = [], {}
        def residual(values):
            self.assertTrue(np.all(values >= low) and np.all(values <= high))
            seen.append(values.copy())
            workspace["q"] = values.copy()
            return np.array([values[0] ** 2, values[1] ** 2, values[2] ** 2, 3 * values[3]])
        jac = bounded_absolute_jacobian(residual, q, low, high, .0001)
        np.testing.assert_allclose(jac, np.diag([.4, .0001, 1.9999, 3.]), atol=1e-10)
        np.testing.assert_array_equal(workspace["q"], q)
        np.testing.assert_array_equal(seen[-1], q)
        np.testing.assert_array_equal(q, [.2, 0., 1., .000005])

    def test_absolute_jacobian_uses_declared_step_even_at_zero_joint(self):
        seen = []
        def residual(q):
            seen.append(float(q[0]))
            return np.array([q[0]])
        np.testing.assert_allclose(bounded_absolute_jacobian(residual, [0.], [-1.], [1.], .0001), [[1.]])
        self.assertIn(-.0001, seen)
        self.assertIn(.0001, seen)
        self.assertEqual(seen[-1], 0.)

    def test_absolute_jacobian_restores_workspace_when_probe_raises_or_changes_shape(self):
        for mode in ("raise", "shape"):
            workspace = {}
            def residual(q):
                workspace["q"] = q.copy()
                if q[0] > 0.:
                    if mode == "raise":
                        raise RuntimeError("probe failed")
                    return np.zeros(2)
                return np.zeros(1)
            with self.assertRaises((RuntimeError, TrajectoryError)):
                bounded_absolute_jacobian(residual, [0.], [-1.], [1.], .0001)
            np.testing.assert_array_equal(workspace["q"], [0.])

    def test_soft_ik_regularization_is_relative_to_copied_legal_seed(self):
        planner = self.planner()
        seed, observed = np.array([.4]), {}
        def solve(residual, initial, **kwargs):
            observed["at_seed"] = residual(seed.copy())
            observed["at_zero"] = residual(np.zeros(1))
            observed["jac"] = kwargs["jac"](seed.copy())
            return SimpleNamespace(x=seed.copy(), success=True)
        with patch("scipy.optimize.least_squares", side_effect=solve):
            _, detail = planner._solve([.4, 0., 0.], np.eye(3), seed)
        self.assertAlmostEqual(observed["at_seed"][6], 0.)
        self.assertAlmostEqual(observed["at_zero"][6], -.003 * .4)
        self.assertEqual(detail["continuity_reference_q"], [.4])
        self.assertAlmostEqual(observed["jac"][6, 0], .003)
        np.testing.assert_array_equal(seed, [.4])

    def test_projection_uses_same_continuity_seed_instead_of_pulling_toward_zero(self):
        def evaluate(q):
            return {"C_error_m": [0., 0., 0.], "rotation_error_vector_rad": [0., 0., 0.],
                    "self_pair_distances_m": [.01] * 9}
        reference = np.array([.4])
        result = project_pose_feasibility([.1], [-1.], [1.], evaluate,
            position_tolerance_m=.003, orientation_tolerance_rad=None,
            position_weight=10., orientation_weight=.04, distance_scale_m=.002,
            finite_difference_rad=.0001, continuity_reference=reference, continuity_weight=.003, max_iterations=100)
        self.assertGreater(result["q"][0], .39)
        self.assertEqual(result["continuity_reference_q"], [.4])
        np.testing.assert_array_equal(reference, [.4])

    def test_projection_respects_millimetre_and_angle_constraints_together(self):
        def evaluate(q):
            return {"C_error_m": [q[0] - .01, 0., 0.],
                    "rotation_error_vector_rad": [0., 0., 10 * q[0] - .1],
                    "self_pair_distances_m": [q[0]] * 9}
        result = project_pose_feasibility([.02], [-1.], [1.], evaluate,
            position_tolerance_m=.003, orientation_tolerance_rad=.05,
            position_weight=10., orientation_weight=1., distance_scale_m=.002, finite_difference_rad=.0001, continuity_reference=[0.], continuity_weight=.003, max_iterations=100)
        self.assertTrue(result["mathematical_constraints_passed"])
        self.assertLessEqual(result["taskpoint_error_m"], .003)
        self.assertLessEqual(result["orientation_error_rad"], .05)
        self.assertFalse(result["native_contacts_checked"])
        self.assertAlmostEqual(result["normalization"]["self_distance_internal_inset_m"], 2e-9)

    def test_projection_cannot_label_incompatible_closed_constraints_feasible(self):
        def evaluate(q):
            return {"C_error_m": [q[0] - .01, 0., 0.],
                    "rotation_error_vector_rad": [0., 0., 10 * q[0] - .2],
                    "self_pair_distances_m": [.01] * 9}
        result = project_pose_feasibility([.02], [-1.], [1.], evaluate,
            position_tolerance_m=.003, orientation_tolerance_rad=.05,
            position_weight=10., orientation_weight=1., distance_scale_m=.002, finite_difference_rad=.0001, continuity_reference=[0.], continuity_weight=.003, max_iterations=30)
        self.assertFalse(result["mathematical_constraints_passed"])
        self.assertTrue(result["taskpoint_error_m"] > .003 or result["orientation_error_rad"] > .05)

    def test_projection_preserves_seed_and_fixed_collision_constraint_dimension(self):
        seed = np.array([.02])
        def evaluate(q):
            q[0] = .01  # The callback cannot mutate the optimizer's vector.
            return {"C_error_m": [0., 0., 0.], "rotation_error_vector_rad": [0., 0., 0.],
                    "self_pair_distances_m": [.01] * 9}
        result = project_pose_feasibility(seed, [-1.], [1.], evaluate,
            position_tolerance_m=.003, orientation_tolerance_rad=None,
            position_weight=10., orientation_weight=.04, distance_scale_m=.002, finite_difference_rad=.0001, continuity_reference=[0.], continuity_weight=.003, max_iterations=5)
        np.testing.assert_array_equal(seed, [.02])
        self.assertEqual(len(result["scaled_inequalities"]), 10)

    def test_projection_rejects_changing_pair_count_and_invalid_iteration_budget(self):
        calls = []
        def evaluate(q):
            calls.append(q.copy())
            return {"C_error_m": [0., 0., 0.], "rotation_error_vector_rad": [0., 0., 0.],
                    "self_pair_distances_m": [.01] * (9 if len(calls) == 1 else 8)}
        options = dict(position_tolerance_m=.003, orientation_tolerance_rad=None,
                       position_weight=10., orientation_weight=.04, distance_scale_m=.002, finite_difference_rad=.0001, continuity_reference=[0.], continuity_weight=.003)
        with self.assertRaisesRegex(TrajectoryError, "malformed residuals"):
            project_pose_feasibility([.02], [-1.], [1.], evaluate, max_iterations=5, **options)
        for invalid in (True, 0, -1, float("nan")):
            with self.assertRaisesRegex(TrajectoryError, "invalid bounded"):
                project_pose_feasibility([.02], [-1.], [1.], evaluate, max_iterations=invalid, **options)

    def test_projection_feasibility_is_independent_of_optimizer_status(self):
        def evaluate(q):
            return {"C_error_m": [q[0] - .01, 0., 0.],
                    "rotation_error_vector_rad": [0., 0., 0.], "self_pair_distances_m": [.01] * 9}
        for converged, final_q, expected in ((True, .02, False), (False, .01, True)):
            answer = SimpleNamespace(x=np.array([final_q]), success=converged, status=0 if converged else 9,
                                     message="mock completed", nit=1, nfev=1)
            with patch("scipy.optimize.minimize", return_value=answer):
                result = project_pose_feasibility([.02], [-1.], [1.], evaluate,
                    position_tolerance_m=.003, orientation_tolerance_rad=None,
                    position_weight=10., orientation_weight=.04, distance_scale_m=.002, finite_difference_rad=.0001, continuity_reference=[0.], continuity_weight=.003, max_iterations=5)
            self.assertEqual(result["optimizer_converged"], converged)
            self.assertEqual(result["mathematical_constraints_passed"], expected)
            self.assertFalse(result["native_contacts_checked"])

    def planner(self):
        planner = CartesianTeacherPlanner.__new__(CartesianTeacherPlanner)
        planner.settings = {"position_weight": 10., "collision_cost_scale": 10.,
            "closed_orientation_weight": 1., "joint_continuity_weight": .003, "solver_max_nfev": 12, "solver_finite_difference_rad": .0001,
            "taskpoint_tolerance_m": .003, "closed_orientation_tolerance_rad": .05,
            "reference_tracking_rad": .02, "reference_tracking_m": .003, "interpolation_samples": 3,
            "maximum_joint_sample_spacing_rad": .02}
        planner.profile = {"ik_orientation_weight": .04, "ik_collision_weight": 5.,
                           "ik_collision_margin_m": .002}
        planner.lower, planner.upper = np.array([-1.]), np.array([1.])
        planner.attempts, planner.last_failure = [], None
        planner.reference_pose = lambda q: {"C": np.array([q[0], 0., 0.]),
            "R": np.eye(3), "palm": np.array([q[0], -.05, 0.])}
        planner._contacts = lambda: {"self_pair_distances_m": [.004] * 9,
            "minimum_self_pair_distance_m": .004,
            "generated_robot_self_contacts": {}, "generated_hand_object_contacts": []}
        return planner

    def test_taskpoint_uses_actual_rotation_and_leaves_inputs_unchanged(self):
        palm = np.array([1., 2., 3.])
        R = Rotation.from_euler("z", 90., degrees=True).as_matrix()
        offset = np.array([0., .05, 0.])
        before = [value.copy() for value in (palm, R, offset)]
        np.testing.assert_allclose(grasp_center(palm, R, offset), [.95, 2., 3.])
        for value, original in zip((palm, R, offset), before):
            np.testing.assert_array_equal(value, original)
        self.assertAlmostEqual(rotation_error(np.eye(3), R), np.pi / 2)

    def test_dense_knots_share_one_continuous_reference_budget(self):
        points = np.linspace(0., .1, 101)[:, None]
        result = path_velocity_budget(points, [.5], .066, .08, .1, .5, closure_s=.2)
        self.assertAlmostEqual(result["continuous_reference_time_s"], .2)
        self.assertAlmostEqual(result["rounded_reference_time_s"], .264)
        self.assertTrue(result["within_original_timeout"])
        self.assertLess(result["required_reference_time_s"], 101 * .066)

    def test_closure_time_and_anchor_reserve_cannot_be_hidden(self):
        result = path_velocity_budget([[0.], [1.]], [10.], .1, .08, .9, 2.7,
                                      closure_s=.6, tracking_reserve_rad=.02)
        self.assertAlmostEqual(result["continuous_reference_time_s"], 1. / .6)
        self.assertAlmostEqual(result["required_reference_time_s"], 2.3)
        result = path_velocity_budget([[0.], [2.]], [10.], .1, .08, .9, 2.7,
                                      closure_s=.6, tracking_reserve_rad=.02)
        self.assertFalse(result["within_original_timeout"])
        with self.assertRaises(TrajectoryError):
            path_velocity_budget([[0.], [1.]], [1.], .1, .02, .9, 2.7,
                                 tracking_reserve_rad=.02)

    def test_optimizer_success_cannot_override_taskpoint_or_collision(self):
        planner = self.planner()
        with patch("scipy.optimize.least_squares", return_value=SimpleNamespace(x=np.array([.01]), success=True)), \
                patch("g1_vla.teacher_trajectory.project_pose_feasibility", return_value={
                    "q": [.01], "optimizer_converged": True, "mathematical_constraints_passed": True}) as projection:
            with self.assertRaisesRegex(TrajectoryError, "strict geometry/taskpoint"):
                planner._solve([0., 0., 0.], np.eye(3), [0.])
            self.assertEqual(projection.call_count, 1)
        self.assertTrue(planner.last_failure["fresh_FK_and_native_checks_after_projection"])
        contacts = planner._contacts()
        contacts["generated_robot_self_contacts"] = {"torso:elbow": .00001}
        planner._contacts = lambda: copy.deepcopy(contacts)
        with patch("scipy.optimize.least_squares", return_value=SimpleNamespace(x=np.array([0.]), success=True)):
            with self.assertRaises(TrajectoryError):
                planner._solve([0., 0., 0.], np.eye(3), [0.])
        self.assertTrue(planner.last_failure["optimizer_converged"])
        self.assertFalse(planner.last_failure["passed"])

    def test_projection_mathematical_success_still_rejects_unlisted_native_contacts(self):
        for category in ("generated_robot_self_contacts", "generated_hand_object_contacts"):
            planner = self.planner()
            contacts = planner._contacts()
            contacts[category] = {"unlisted_pair": .001} if "self" in category else [{"distance_m": -.001}]
            planner._contacts = lambda: copy.deepcopy(contacts)
            with patch("scipy.optimize.least_squares", return_value=SimpleNamespace(x=np.array([.01]), success=True)), \
                    patch("g1_vla.teacher_trajectory.project_pose_feasibility", return_value={
                        "q": [0.], "optimizer_converged": True, "mathematical_constraints_passed": True}):
                with self.assertRaises(TrajectoryError):
                    planner._solve([0., 0., 0.], np.eye(3), [0.])
            self.assertTrue(planner.last_failure["mathematical_constraints_passed"])
            self.assertFalse(planner.last_failure["native_contact_checks_passed"])

    def test_projection_repairs_once_then_rechecks_actual_candidate_without_status_shortcut(self):
        planner = self.planner()
        with patch("scipy.optimize.least_squares", return_value=SimpleNamespace(x=np.array([.01]), success=True)), \
                patch("g1_vla.teacher_trajectory.project_pose_feasibility", return_value={
                    "q": [0.], "optimizer_converged": False, "mathematical_constraints_passed": True}) as projection:
            q, detail = planner._solve([0., 0., 0.], np.eye(3), [0.])
        self.assertEqual(projection.call_count, 1)
        self.assertTrue(detail["passed"])
        self.assertFalse(detail["optimizer_converged"])
        self.assertEqual(q.tolist(), [0.])
        self.assertAlmostEqual(detail["soft_ik_before_projection"]["taskpoint_error_m"], .01)

    def test_native_hand_contact_is_forbidden_open_but_permitted_after_grip(self):
        planner = self.planner()
        contacts = planner._contacts()
        contacts["generated_hand_object_contacts"] = [{"distance_m": -.0001}]
        planner._contacts = lambda: copy.deepcopy(contacts)
        with patch("scipy.optimize.least_squares", return_value=SimpleNamespace(x=np.array([0.]), success=True)):
            with self.assertRaises(TrajectoryError):
                planner._solve([0., 0., 0.], np.eye(3), [0.])
            _, closed = planner._solve([0., 0., 0.], np.eye(3), [0.], closed=True)
        self.assertTrue(closed["passed"])

    def test_closed_soft_orientation_is_also_a_hard_constraint(self):
        planner = self.planner()
        requested = Rotation.from_euler("z", .2).as_matrix()
        with patch("scipy.optimize.least_squares", return_value=SimpleNamespace(x=np.array([0.]), success=True)):
            with self.assertRaises(TrajectoryError):
                planner._solve([0., 0., 0.], requested, [0.], closed=True)
            _, diagnostic = planner._solve([0., 0., 0.], requested, [0.], closed=False)
        self.assertTrue(diagnostic["passed"])
        self.assertGreater(diagnostic["orientation_error_rad"], .05)

    def test_chord_checks_all_native_self_contacts_and_both_endpoints(self):
        planner = self.planner()
        original = planner.reference_pose
        sampled = []
        def pose(q):
            sampled.append(float(q[0]))
            return original(q)
        planner.reference_pose = pose
        contacts = planner._contacts()
        def queried():
            result = copy.deepcopy(contacts)
            if sampled[-1] > .9:
                result["generated_robot_self_contacts"] = {"unlisted_arm:leg": .001}
            return result
        planner._contacts = queried
        result = planner.validate_chord([0.], [1.])
        self.assertEqual((sampled[0], sampled[-1]), (0., 1.))
        self.assertEqual(result["sample_count"], 51)
        self.assertLessEqual(float(np.max(np.diff(sampled))), .0200000001)
        self.assertFalse(result["passed"])
        self.assertEqual(result["samples"][-1]["generated_robot_self_contacts"], {"unlisted_arm:leg": .001})

    def raw_recovery_planner(self, actual):
        planner = self.planner()
        planner.ids = [0]
        planner.workspace = SimpleNamespace(qpos=np.asarray(actual, dtype=float).copy())
        planner.config = {"eval": {"joint_limit_tolerance_rad": .003}}
        return planner

    def test_raw_recovery_accepts_observed_32microradian_soft_overrun_without_clipping(self):
        first, second = np.array([1.000032216]), np.array([1.])
        context = {"kind": "actual_state_to_candidate", "actual": first.tolist()}
        original = (first.copy(), second.copy(), copy.deepcopy(context))
        planner = self.raw_recovery_planner(first)
        result = planner.validate_chord(first, second, context)
        self.assertTrue(result["passed"])
        self.assertTrue(result["raw_recovery"]["actual_start_verified"])
        self.assertTrue(result["raw_recovery"]["exception_used"])
        self.assertFalse(result["samples"][0]["nominal_mechanical_bounds_passed"])
        self.assertTrue(result["samples"][-1]["nominal_mechanical_bounds_passed"])
        self.assertAlmostEqual(result["raw_recovery"]["maximum_initial_joint_limit_excess_rad"], .000032216)
        excess = [row["joint_limit_excess_rad"][0] for row in result["samples"]]
        self.assertTrue(all(second <= first for first, second in zip(excess, excess[1:])))
        np.testing.assert_array_equal(first, original[0])
        np.testing.assert_array_equal(second, original[1])
        self.assertEqual(context, original[2])
        np.testing.assert_array_equal(planner.workspace.qpos, first)

    def test_raw_recovery_rejects_unsafe_start_or_illegal_candidate(self):
        for first, second in (([1.0031], [1.]), ([1.000032216], [1.000001]),
                              ([1.000032216], [1.00004])):
            planner = self.raw_recovery_planner(first)
            result = planner.validate_chord(first, second, {"kind": "actual_state_to_candidate", "actual": list(first)})
            self.assertFalse(result["passed"])
            self.assertFalse(result["raw_recovery"]["eligible"])

    def test_raw_recovery_requires_exact_real_start_and_only_actual_context(self):
        first, second = [1.000032216], [1.]
        for context in (None, {"kind": "polyline_piece"}, {"kind": "applied_command_shortcut"},
                        {"kind": "actual_state_to_candidate", "actual": [1.00003]}):
            result = self.raw_recovery_planner(first).validate_chord(first, second, context)
            self.assertFalse(result["passed"])
        # Merely echoing a fabricated first vector in context is insufficient:
        # it must also match the untouched workspace copied from the live state.
        planner = self.raw_recovery_planner([1.00003])
        result = planner.validate_chord(first, second, {"kind": "actual_state_to_candidate", "actual": first})
        self.assertFalse(result["passed"])
        self.assertFalse(result["raw_recovery"]["actual_start_verified"])

    def test_raw_recovery_preserves_all_native_and_closed_orientation_rejections(self):
        first, second = [1.000032216], [1.]
        context = {"kind": "actual_state_to_candidate", "actual": first}
        planner = self.raw_recovery_planner(first)
        contacts = planner._contacts()
        contacts["generated_robot_self_contacts"] = {"torso:arm": .0001}
        planner._contacts = lambda: contacts
        self.assertFalse(planner.validate_chord(first, second, context)["passed"])
        planner = self.raw_recovery_planner(first)
        self.assertFalse(planner.validate_chord(first, second, context, closed=True,
            lock_rotation=Rotation.from_euler("z", .051).as_matrix())["passed"])

    def test_closed_endpoint_requires_actual_orientation_and_taskpoint(self):
        planner = self.planner()
        pose = {"arm": np.array([.1]), "C": np.zeros(3), "R": np.eye(3)}
        planner.actual_pose = lambda data: pose
        current_reference_C = np.zeros(3)
        planner.sync_actual = lambda data: None
        planner.reference_pose = lambda q: {"C": current_reference_C.copy()}
        plan = {"q_waypoints": [[0.], [.1]], "target_C": [0., 0., 0.],
                "endpoint_C": [0., 0., 0.],
                "target_R": np.eye(3).tolist(), "closed": True}
        self.assertTrue(planner.end_reached(plan, None))
        pose["R"] = Rotation.from_euler("x", .051).as_matrix()
        self.assertFalse(planner.end_reached(plan, None))
        planner.settings["taskpoint_tolerance_m"] = .015
        pose["C"] = np.array([.004, 0., 0.])
        self.assertFalse(planner.end_reached(plan, None))  # 15mm goal does not erase 3mm reference tracking.
        pose["R"], pose["C"] = np.eye(3), np.array([.0031, 0., 0.])
        self.assertFalse(planner.end_reached(plan, None))
        plan["endpoint_C"] = [.014, 0., 0.]
        current_reference_C[:] = [.014, 0., 0.]
        pose["C"] = np.array([.010, 0., 0.])
        self.assertFalse(planner.end_reached(plan, None))
        pose["C"] = np.array([.014, 0., 0.])
        self.assertTrue(planner.end_reached(plan, None))

    def test_endpoint_tracking_accounts_for_base_frame_without_moving_world_goal_or_arm_reference(self):
        planner = self.planner()
        planner.settings["taskpoint_tolerance_m"] = .015
        base = np.array([.006, 0., 0.])
        actual_arm = np.array([.1])
        plan = {"q_waypoints": [[0.], [.1]], "target_C": [0., 0., 0.],
                "endpoint_C": [0., 0., 0.], "target_R": np.eye(3).tolist(), "closed": False}
        unchanged = copy.deepcopy(plan)
        planner.actual_pose = lambda data: {"arm": actual_arm.copy(), "C": base.copy(), "R": np.eye(3)}
        planner.sync_actual = lambda data: None
        queried = []
        def reference(q):
            queried.append(np.asarray(q).copy())
            return {"C": base + np.array([q[0] - .1, 0., 0.])}
        planner.reference_pose = reference
        self.assertTrue(planner.end_reached(plan, None))
        np.testing.assert_array_equal(queried[-1], [.1])
        self.assertEqual(plan, unchanged)
        # Accurate joint tracking cannot excuse violating the fixed world goal.
        base[:] = [.016, 0., 0.]
        self.assertFalse(planner.end_reached(plan, None))
        # Reference FK is independent of the measured right-arm q: a .01rad
        # discrepancy passes the .02 joint gate, but fails 3mm task tracking.
        base[:] = [.006, 0., 0.]
        planner.actual_pose = lambda data: {"arm": np.array([.11]), "C": base + [.01, 0., 0.], "R": np.eye(3)}
        plan["target_C"] = [.006, 0., 0.]
        self.assertFalse(planner.end_reached(plan, None))
        np.testing.assert_array_equal(queried[-1], [.1])

    def test_workspace_refresh_copies_entire_free_object_state_without_mutating_live(self):
        planner = self.planner()
        planner.model = SimpleNamespace(nq=12, nv=11)
        planner.workspace = SimpleNamespace(qpos=np.zeros(12), qvel=np.zeros(11), time=0.)
        data = SimpleNamespace(qpos=np.arange(12, dtype=float), qvel=np.arange(11, dtype=float), time=3.)
        before = planner.state_snapshot(data)
        planner.api = SimpleNamespace(mj_forward=lambda model, workspace: None)
        planner.sync_actual(data)
        np.testing.assert_array_equal(planner.workspace.qpos, data.qpos)
        np.testing.assert_array_equal(planner.workspace.qvel, data.qvel)
        planner.workspace.qpos[9:] = 0.
        self.assertEqual(planner.state_snapshot(data), before)

    def test_grip_lock_records_actual_relative_transform_without_moving_object(self):
        planner = self.planner()
        R = Rotation.from_euler("z", .3).as_matrix()
        palm = np.array([.3, -.03, .8])
        planner.actual_pose = lambda data: {"palm": palm, "R": R,
            "C": grasp_center(palm, R, [0., .05, 0.]), "arm": np.array([.1])}
        planner.object_body = 0
        data = SimpleNamespace(time=5., xpos=np.array([[.31, .01, .81]]), xmat=np.eye(3)[None])
        old = data.xpos.copy()
        lock = planner.latch_grip(data)
        np.testing.assert_allclose(lock["object_in_palm_frame_m"], R.T @ (old[0] - palm))
        self.assertFalse(lock["object_state_modified"])
        np.testing.assert_array_equal(data.xpos, old)

    def test_lift_and_hold_preserve_attained_rotation_and_apply_height_once(self):
        planner = self.planner()
        R = Rotation.from_euler("y", .47).as_matrix()
        locked_C = np.array([.3, .01, .8])
        planner.center = np.array([0., .05, 0.])
        planner.lower, planner.upper = np.full(3, -2.), np.full(3, 2.)
        planner.anchor, planner.velocity, planner.period = np.full(3, .08), np.ones(3), .066
        planner.names, planner.pairs = ["a", "b", "c"], [(0, 1)] * 9
        planner.pair_evidence, planner.pair_signature = [], "fake-geometry-test"
        planner.object_body = 0
        planner.model = SimpleNamespace(njnt=0)
        planner._copy_visited = lambda data: None
        planner.state_snapshot = lambda data: {"time_s": data.time, "qpos": data.qpos.tolist()}
        planner.profile.update(lift_height_m=.22, lift_duration_s=2., final_hold_s=1., phase_timeout_scale=3.)
        planner.settings["waypoint_spacing_m"] = .01
        planner.actual_pose = lambda data: {"C": data.qpos.copy(), "R": R.copy(),
            "palm": data.qpos - R @ planner.center, "arm": data.qpos.copy()}
        planner.reference_pose = lambda q: {"C": np.asarray(q).copy(), "R": R.copy(),
            "palm": np.asarray(q) - R @ planner.center, "arm": np.asarray(q).copy()}
        def solve(C, orientation, seed, closed=False):
            self.assertTrue(closed)
            np.testing.assert_allclose(orientation, R)
            return np.asarray(C).copy(), {"taskpoint_error_m": 0., "R": R.tolist()}
        planner._solve = solve
        planner.validate_chord = lambda *args, **kwargs: {"passed": True}
        data = SimpleNamespace(time=6., xpos=np.array([[99., 99., 99.]]), qpos=locked_C.copy())
        lock = {"C": locked_C.tolist(), "R": R.tolist()}
        lifted = planner.plan_phase("lift", data, data.qpos, grip_lock=lock)
        np.testing.assert_allclose(lifted["target_C"], locked_C + [0., 0., .22])
        np.testing.assert_allclose(lifted["target_R"], R)
        data.qpos = np.asarray(lifted["q_waypoints"][-1])
        held = planner.plan_phase("hold", data, data.qpos, grip_lock=lock, previous_plan=lifted)
        np.testing.assert_allclose(held["target_C"], lifted["target_C"])
        np.testing.assert_allclose(held["target_R"], R)
        np.testing.assert_array_equal(data.xpos, [[99., 99., 99.]])
        # Budget rejections retain the exact generated path, rather than only
        # the number of seconds that made it fail.
        data.qpos = locked_C.copy()
        planner.velocity = np.full(3, .001)
        with self.assertRaises(TrajectoryError) as rejected:
            planner.plan_phase("lift", data, data.qpos, grip_lock=lock)
        partial = rejected.exception.detail["partial_plan"]
        self.assertEqual(partial["failure_kind"], "original_budget_rejection")
        self.assertGreater(len(partial["q_waypoints"]), 2)
        self.assertEqual(partial["entry_full_state"]["qpos"], locked_C.tolist())
        self.assertFalse(partial["live_state_modified"])
        np.testing.assert_array_equal(data.qpos, locked_C)

    def test_open_orientation_changes_geodesically_from_applied_reference_but_closed_is_fixed(self):
        for phase in ("pregrasp", "contact", "lift"):
            planner = self.planner()
            start_R = Rotation.from_euler("z", .1).as_matrix()
            goal_R = Rotation.from_euler("z", .12 if phase == "lift" else .8).as_matrix()
            planner.contact_rotation = goal_R.copy()
            planner.center = np.array([0., .05, 0.])
            planner.lower, planner.upper = np.full(3, -2.), np.full(3, 2.)
            planner.anchor, planner.velocity, planner.period = np.full(3, .08), np.ones(3), .066
            planner.names, planner.pairs = ["a", "b", "c"], [(0, 1)] * 9
            planner.pair_evidence, planner.pair_signature = [], "fake-geometry-test"
            planner.object_body, planner.model = 0, SimpleNamespace(njnt=0)
            planner._copy_visited = lambda data: None
            planner.state_snapshot = lambda data: {"time_s": data.time, "qpos": data.qpos.tolist()}
            planner.profile.update(pregrasp_lateral_m=.015, pregrasp_height_m=.12,
                pregrasp_s=.8, contact_approach_s=.9, close_duration_s=.6,
                lift_height_m=.22, lift_duration_s=2., phase_timeout_scale=3.)
            planner.settings["waypoint_spacing_m"] = .01
            # Measured R differs slightly from the applied-reference FK. The
            # orientation and C path must use the same independent q0 frame.
            planner.actual_pose = lambda data: {"C": data.qpos.copy(),
                "R": Rotation.from_euler("z", .104).as_matrix(),
                "palm": data.qpos - start_R @ planner.center, "arm": data.qpos.copy()}
            R_workspace, targets = [start_R.copy()], []
            planner.reference_pose = lambda q: {"C": np.asarray(q).copy(), "R": R_workspace[0].copy(),
                "palm": np.asarray(q) - R_workspace[0] @ planner.center, "arm": np.asarray(q).copy()}
            def solve(C, R, seed, *, closed=False):
                self.assertEqual(closed, phase == "lift")
                targets.append(np.asarray(R).copy())
                R_workspace[0] = np.asarray(R).copy()
                return np.asarray(C).copy(), {"taskpoint_error_m": 0., "R": np.asarray(R).tolist()}
            planner._solve = solve
            planner.validate_chord = lambda *args, **kwargs: {"passed": True}
            q0 = np.array([.3, -.005, 1.08 if phase == "pregrasp" else .92])
            data = SimpleNamespace(time=3., xpos=np.array([[.3, .01, .8]]), qpos=q0.copy())
            lock = {"C": q0.tolist(), "R": goal_R.tolist()} if phase == "lift" else None
            plan = planner.plan_phase(phase, data, q0, grip_lock=lock)
            self.assertGreater(len(targets), 1)
            np.testing.assert_allclose(targets[-1], goal_R, atol=1e-12)
            if phase == "lift":
                for R in targets:
                    np.testing.assert_allclose(R, goal_R, atol=1e-12)
            else:
                total = rotation_error(start_R, goal_R)
                self.assertAlmostEqual(rotation_error(start_R, targets[0]), total / len(targets))
                self.assertLess(rotation_error(start_R, targets[0]), total)
                self.assertEqual(plan["orientation_schedule"], "entry_reference_FK_geodesic_to_contact_goal")
            np.testing.assert_array_equal(data.qpos, q0)


if __name__ == "__main__":
    unittest.main()
