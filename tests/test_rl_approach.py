"""Pure resource and measured-waypoint tests; no Isaac, simulator or optimizer."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from g1_vla.rl_approach import (
    ALL_JOINT_NAMES, APPROACH_PARAMETERS, APPROACH_VERSION, ARM_JOINT_NAMES,
    FAILURE_CODES, HAND_JOINT_NAMES, HomeApproachPlanner, load_approach_resource,
    approach_hand_posture_tracking, validate_approach_declaration, validate_approach_resource,
)

try:
    import torch
except ImportError:
    torch = None


CANDIDATE_SHA = "a"*64
SOURCE_HAND = [.3, .1, -1.1, .8, .3, .14, 1.3]


def resource(*, two_waypoints=False, duration=.198):
    home = dict.fromkeys(ALL_JOINT_NAMES, 0.0)
    home.update(zip(HAND_JOINT_NAMES, SOURCE_HAND))
    goal = dict(home)
    goal[ARM_JOINT_NAMES[0]] = .12
    waypoints = [{"name": "pregrasp", "arm_target_rad": [.12, 0., 0., 0., 0., 0., 0.],
                  "duration_s": duration, "palm_position_world_m": [.12, 0., 1.],
                  "palm_quaternion_xyzw": [0., 0., 0., 1.]}]
    if two_waypoints:
        waypoints.insert(0, {"name": "clearance", "arm_target_rad": [.06, 0., 0., 0., 0., 0., 0.],
                             "duration_s": duration, "palm_position_world_m": [.06, 0., 1.],
                             "palm_quaternion_xyzw": [0., 0., 0., 1.]})
    return {"version": APPROACH_VERSION, "original_candidate_sha256": CANDIDATE_SHA,
            "root_pose_xyzw": [0., 0., 0., 0., 0., 0., 1.], "source_hand_rad": list(SOURCE_HAND),
            "home_joint_positions_rad": home, "home_palm_position_world_m": [0., 0., 1.],
            "home_palm_quaternion_xyzw": [0., 0., 0., 1.], "waypoints": waypoints,
            "pregrasp_joint_positions_rad": goal, "pregrasp_palm_position_world_m": [.12, 0., 1.],
            "pregrasp_palm_quaternion_xyzw": [0., 0., 0., 1.]}


def declaration():
    return {"version": APPROACH_VERSION, "resource_path": "resources/approach/course_v1.json",
            "resource_sha256": "b"*64, "parameters": deepcopy(APPROACH_PARAMETERS)}


class ApproachContractTests(unittest.TestCase):
    def test_none_keeps_optional_course_disabled(self):
        self.assertIsNone(validate_approach_declaration(None))

    def test_exact_declaration_returns_independent_copy_and_fixed_parameters(self):
        value = declaration()
        checked = validate_approach_declaration(value)
        self.assertEqual(checked, value)
        checked["parameters"]["approach_timeout_s"] = 999
        self.assertEqual(value["parameters"]["approach_timeout_s"], 16.)
        self.assertEqual(APPROACH_PARAMETERS["handoff_required_consecutive_native_samples"], 25)
        self.assertEqual(APPROACH_PARAMETERS["total_episode_budget_s"], 32.)

    def test_unknown_missing_parameters_bool_and_changed_gate_rejected(self):
        cases = []
        v = declaration();v["extra"] = True;cases.append(v)
        v = declaration();del v["resource_sha256"];cases.append(v)
        v = declaration();v["resource_sha256"] = "B"*64;cases.append(v)
        v = declaration();v["resource_path"] = "\0bad";cases.append(v)
        v = declaration();v["parameters"]["maximum_object_xy_drift_m"] = .003;cases.append(v)
        v = declaration();v["parameters"]["handoff_required_consecutive_native_samples"] = 25.;cases.append(v)
        v = declaration();v["parameters"]["approach_timeout_s"] = True;cases.append(v)
        v = declaration();v["parameters"]["approach_timeout_s"] = float("nan");cases.append(v)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_approach_declaration(value)

    def test_arm_only_resource_and_candidate_identity(self):
        checked = validate_approach_resource(resource(), expected_candidate_sha256=CANDIDATE_SHA)
        self.assertEqual(len(checked["home_joint_positions_rad"]), 43)
        self.assertEqual(len(set(ALL_JOINT_NAMES)), 43)
        self.assertEqual([checked["home_joint_positions_rad"][n] for n in HAND_JOINT_NAMES], SOURCE_HAND)
        with self.assertRaises(ValueError):
            validate_approach_resource(resource(), expected_candidate_sha256="c"*64)

    def test_full_joint_identity_and_unchanged_nonarm_hand_are_required(self):
        cases = []
        v = resource();del v["home_joint_positions_rad"]["left_elbow_joint"];cases.append(v)
        v = resource();v["pregrasp_joint_positions_rad"]["left_elbow_joint"] = .01;cases.append(v)
        v = resource();v["home_joint_positions_rad"][HAND_JOINT_NAMES[0]] += .001;cases.append(v)
        v = resource();v["home_joint_positions_rad"]["unknown_joint"] = 0.;cases.append(v)
        v = resource();v["pregrasp_joint_positions_rad"]["waist_pitch_joint"] = False;cases.append(v)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_approach_resource(value)

    def test_endpoint_fk_workspace_and_resource_schema_are_strict(self):
        cases = []
        v = resource();v["waypoints"][-1]["arm_target_rad"][0] += .01;cases.append(v)
        v = resource();v["waypoints"][-1]["palm_position_world_m"][0] += .01;cases.append(v)
        v = resource();v["home_palm_position_world_m"][2] = 1.11;cases.append(v)
        v = resource();v["home_palm_quaternion_xyzw"][3] = 2.;cases.append(v)
        v = resource();v["waypoints"][-1]["duration_s"] = 17.;cases.append(v)
        v = resource();v["waypoints"][-1]["duration_s"] = 0.;cases.append(v)
        v = resource();v["waypoints"][-1]["extra"] = 0;cases.append(v)
        v = resource();v["unknown"] = 1;cases.append(v)
        v = resource();v["home_joint_positions_rad"] = dict(v["pregrasp_joint_positions_rad"]);cases.append(v)
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_approach_resource(value)

    def test_loader_verifies_original_file_hash_and_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"course.json"
            payload = json.dumps(resource()).encode()
            p.write_bytes(payload)
            expected = hashlib.sha256(payload).hexdigest()
            self.assertEqual(load_approach_resource(p, expected)["version"], APPROACH_VERSION)
            with self.assertRaises(ValueError):
                load_approach_resource(p, "0"*64)
            duplicate = b'{"version":"one","version":"two"}'
            p.write_bytes(duplicate)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_approach_resource(p, hashlib.sha256(duplicate).hexdigest())
            nonfinite = b'{"x":NaN}'
            p.write_bytes(nonfinite)
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                load_approach_resource(p, hashlib.sha256(nonfinite).hexdigest())

    def test_loader_rejects_symlink(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/"course.json";p.write_text(json.dumps(resource()))
            link = Path(td)/"link.json"
            try:
                link.symlink_to(p)
            except OSError:
                self.skipTest("host does not permit symlink creation")
            with self.assertRaises(ValueError):
                load_approach_resource(link, hashlib.sha256(p.read_bytes()).hexdigest())


@unittest.skipIf(torch is None, "Torch is optional in the ordinary source-only environment")
class ApproachHandPostureTrackingTests(unittest.TestCase):
    def arguments(self, n=1):
        return dict(original_tracking=torch.zeros(n, dtype=torch.bool),
            admitted=torch.zeros(n, dtype=torch.bool), approach_failed=torch.zeros(n, dtype=torch.bool),
            fresh_valid=torch.ones(n, dtype=torch.bool), current_normal_available=torch.zeros(n, 3, dtype=torch.bool),
            physical_group_loaded=torch.zeros(n, 3, dtype=torch.bool),
            robot_cube_force_n=torch.zeros(n, dtype=torch.float64), unsafe=torch.zeros(n, dtype=torch.bool))

    def test_only_fresh_unloaded_unfailed_approach_can_override_original_tracking(self):
        kw = self.arguments(8)
        kw["fresh_valid"][1] = False
        kw["approach_failed"][2] = True
        kw["unsafe"][3] = True
        kw["current_normal_available"][4, 0] = True  # Speculative/positive-gap normal is sufficient to reject.
        kw["physical_group_loaded"][5, 2] = True
        kw["robot_cube_force_n"][6] = 1e-14
        kw["robot_cube_force_n"][7] = float("nan")
        mask, ev = approach_hand_posture_tracking(torch, **kw)
        self.assertEqual(mask.tolist(), [True, False, False, False, False, False, False, False])
        self.assertTrue(ev["object_support_torque_must_use_original_palm_tracking"])
        self.assertIn("not arm tracking", ev["scope"])

    def test_before_admission_original_true_does_not_bypass_free_hand_requirements(self):
        kw = self.arguments();kw["original_tracking"][:] = True;kw["fresh_valid"][:] = False
        mask, _ = approach_hand_posture_tracking(torch, **kw)
        self.assertFalse(bool(mask[0]))
        kw["fresh_valid"][:] = True;kw["robot_cube_force_n"][:] = -.01
        mask, _ = approach_hand_posture_tracking(torch, **kw)
        self.assertFalse(bool(mask[0]))

    def test_after_admission_exact_original_tracking_is_returned(self):
        kw = self.arguments(2);kw["admitted"][:] = True
        kw["original_tracking"][:] = torch.tensor([False, True])
        kw["fresh_valid"][:] = False;kw["unsafe"][:] = True
        kw["current_normal_available"][:] = True;kw["robot_cube_force_n"][:] = 1.
        mask, _ = approach_hand_posture_tracking(torch, **kw)
        self.assertTrue(torch.equal(mask, kw["original_tracking"]))

    def test_wrong_shapes_types_and_missing_force_cannot_be_silently_zero(self):
        for key, value in (("fresh_valid", torch.ones(1)),
                           ("current_normal_available", torch.zeros(1, 7, dtype=torch.bool)),
                           ("robot_cube_force_n", None), ("robot_cube_force_n", torch.zeros(1, dtype=torch.int64))):
            with self.subTest(key=key):
                kw = self.arguments();kw[key] = value
                with self.assertRaises(ValueError):
                    approach_hand_posture_tracking(torch, **kw)

    def effort_arguments(self):
        seed = torch.full((1, 7), .3, dtype=torch.float64)
        return seed, dict(sample_id=0, episode_id=torch.tensor([2]), dt=.002,
            measured_hand_rad=seed.clone(), measured_hand_velocity_rad_s=torch.zeros_like(seed),
            nominal_hand_rad=seed.clone(), gravity_hand_nm=torch.zeros_like(seed),
            inward_normal_jacobian=torch.zeros(1, 3, 7, dtype=seed.dtype),
            group_contact_valid=torch.zeros(1, 3, dtype=torch.bool),
            group_physical_loaded=torch.zeros(1, 3, dtype=torch.bool),
            raw_group_force_n=torch.zeros(1, 3, dtype=seed.dtype),
            contact_measurement_valid=torch.ones(1, dtype=torch.bool),
            tracking=torch.zeros(1, dtype=torch.bool), unsafe_latched=torch.zeros(1, dtype=torch.bool),
            joint_lower_hand_rad=torch.full_like(seed, -2.), joint_upper_hand_rad=torch.full_like(seed, 2.),
            geometry_lower_hand_rad=torch.full_like(seed, -2.), geometry_upper_hand_rad=torch.full_like(seed, 2.),
            nominal_velocity_limit_hand_rad_s=torch.ones_like(seed), penetration_m=torch.zeros(1, dtype=seed.dtype),
            closing_axis_hand7=torch.tensor([-1., -1., -1., 1., 1., 1., 1.], dtype=seed.dtype))

    def test_real_balanced_controller_restores_free_source_posture_without_arm_tracking(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        seed, kw = self.effort_arguments()
        original = BalancedEffortGraspController(torch, seed, kw["episode_id"])
        approach = BalancedEffortGraspController(torch, seed, kw["episode_id"])
        original.update(**kw);approach.update(**kw)  # Original fresh-feedback bootstrap, no physics execution.
        kw["sample_id"] = 1;kw["measured_hand_rad"][:, 0] -= .01
        old_effort, old_ev, old_unsafe = original.update(**kw)
        mask, ev = approach_hand_posture_tracking(torch, **self.arguments())
        self.assertFalse(bool(ev["original_palm_tracking"][0]))
        kw["tracking"] = mask
        new_effort, new_ev, new_unsafe = approach.update(**kw)
        self.assertFalse(bool(old_unsafe.any() or new_unsafe.any()))
        self.assertTrue(torch.equal(old_ev["position_spring_after_contact_mode_mask_nm"], torch.zeros_like(seed)))
        self.assertAlmostEqual(float(new_ev["position_spring_after_contact_mode_mask_nm"][0, 0]), .15*.01)
        self.assertGreater(float(new_effort[0, 0]), float(old_effort[0, 0]))
        self.assertTrue(bool((new_effort.abs() <= torch.tensor([.06, .04, .04, .04, .04, .04, .04])).all()))

    def test_real_force_mode_keeps_loaded_finger_position_springs_strictly_zero(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        for admitted in (False, True):
            with self.subTest(admitted=admitted):
                seed, kw = self.effort_arguments()
                controller = BalancedEffortGraspController(torch, seed, kw["episode_id"])
                controller.update(**kw)
                kw["sample_id"] = 1;kw["measured_hand_rad"] -= .01
                kw["inward_normal_jacobian"][0, 0, :3] = torch.tensor([.02, .01, .01])
                kw["group_contact_valid"][0, 0] = True;kw["group_physical_loaded"][0, 0] = True
                kw["raw_group_force_n"][0, 0] = .2
                gate = self.arguments();gate["admitted"][:] = admitted;gate["original_tracking"][:] = True
                gate["current_normal_available"][0, 0] = True;gate["physical_group_loaded"][0, 0] = True
                gate["robot_cube_force_n"][:] = .2
                kw["tracking"], _ = approach_hand_posture_tracking(torch, **gate)
                _, ev, unsafe = controller.update(**kw)
                self.assertFalse(bool(unsafe.any()))
                spring = ev["position_spring_after_contact_mode_mask_nm"]
                self.assertTrue(torch.equal(spring[:, :3], torch.zeros_like(spring[:, :3])))
                self.assertTrue(bool(ev["position_spring_disabled_by_current_normal_joint"][0, :3].all()))
                if not admitted:
                    self.assertTrue(torch.equal(spring, torch.zeros_like(spring)))


@unittest.skipIf(torch is None, "Torch is optional in the ordinary source-only environment")
class HomeApproachPlannerTests(unittest.TestCase):
    def state(self, n=1):
        dtype = torch.float64
        q = torch.zeros(n, 14, dtype=dtype)
        q[:, 7:] = torch.tensor(SOURCE_HAND, dtype=dtype)
        origins = torch.zeros(n, 3, dtype=dtype)
        if n > 1:
            origins[1:, 0] = torch.arange(1, n, dtype=dtype)*3
        return {"q_action_rad": q, "qd_action_rad_s": torch.zeros_like(q), "target_action_rad": q.clone(),
                "per_step_action_rad": torch.full_like(q, .08),
                "joint_limits_action_rad": torch.tensor([-2., 2.], dtype=dtype).repeat(n, 14, 1),
                "env_origin_world_m": origins, "palm_position_world_m": origins+torch.tensor([0., 0., 1.]),
                "palm_quaternion_xyzw": torch.tensor([0., 0., 0., 1.], dtype=dtype).repeat(n, 1),
                "unsafe_latched": torch.zeros(n, dtype=torch.bool),
                "approach_admitted": torch.zeros(n, dtype=torch.bool),
                "episode_id": torch.full((n,), 2, dtype=torch.int64),
                "episode_control_steps": torch.zeros(n, dtype=torch.int64),
                "native_physics_step_count": 0, "velocity_limits_arm_rad_s": torch.ones(7, dtype=dtype)}

    def advance(self, state, request=None, *, track=True):
        if request is not None:
            state["target_action_rad"][:, :7] += request*state["per_step_action_rad"][:, :7]
        if track:
            state["q_action_rad"][:, :7] = state["target_action_rad"][:, :7]
            state["palm_position_world_m"] = state["env_origin_world_m"].clone()
            state["palm_position_world_m"][:, 0] += state["q_action_rad"][:, 0]
            state["palm_position_world_m"][:, 2] += 1.
        state["episode_control_steps"] += 1
        state["native_physics_step_count"] += 33

    def ready(self, planner, state):
        for _ in range(100):
            request, ev = planner.actions(state, .066)
            if bool(ev["kinematic_arrived"].all()):
                return request, ev
            self.advance(state, request)
        self.fail("synthetic waypoint did not arrive")

    def test_measured_quintic_course_arrives_without_deciding_physical_admission(self):
        state = self.state(2)
        planner = HomeApproachPlanner(torch, resource(two_waypoints=True))
        request, ev = self.ready(planner, state)
        self.assertEqual(tuple(request.shape), (2, 7))
        self.assertTrue(bool(ev["kinematic_arrived"].all()))
        self.assertFalse(ev["physical_handoff_decided_here"])
        self.assertFalse(bool(ev["admission_latched"].any()))
        self.assertTrue(bool((ev["home_palm_position_error_m"] == 0).all()))
        self.assertTrue(bool((ev["waypoint_index"] == 1).all()))

    def test_reference_and_published_request_obey_native_velocity_and_macro_bound(self):
        state = self.state();state["velocity_limits_arm_rad_s"].fill_(.02)
        planner = HomeApproachPlanner(torch, resource(duration=.066))
        for _ in range(25):
            request, ev = planner.actions(state, .066)
            self.assertTrue(bool((request.abs() <= 1).all()))
            self.assertTrue(bool((request*state["per_step_action_rad"][:, :7]).abs().le(.02*.066+1e-12).all()))
            self.assertTrue(bool((ev["slewed_planner_reference_arm_rad"]-ev["previous_planner_reference_arm_rad"]).abs().le(.02*.066+1e-12).all()))
            self.advance(state, request)

    def test_tracking_lag_pauses_reference_clock(self):
        state = self.state();planner = HomeApproachPlanner(torch, resource(duration=.066))
        request, before = planner.actions(state, .066)
        self.advance(state, request, track=False)
        _, after = planner.actions(state, .066)
        self.assertTrue(bool(after["tracking_paused"][0]))
        self.assertEqual(float(after["waypoint_elapsed_before_request_s"][0]), float(after["waypoint_elapsed_after_request_s"][0]))
        self.assertTrue(torch.equal(after["slewed_planner_reference_arm_rad"], after["previous_planner_reference_arm_rad"]))

    def test_time_alone_and_high_actual_velocity_cannot_advance_waypoint(self):
        state = self.state();planner = HomeApproachPlanner(torch, resource(two_waypoints=True, duration=.066))
        request, ev = planner.actions(state, .066)
        self.assertEqual(int(ev["waypoint_index_after_request"][0]), 0)
        self.advance(state, request)
        state["qd_action_rad_s"][:, 0] = .051
        request, ev = planner.actions(state, .066)
        self.assertEqual(int(ev["waypoint_index_after_request"][0]), 0)
        self.advance(state, request)
        state["qd_action_rad_s"][:, 0] = .049
        _, ev = planner.actions(state, .066)
        self.assertEqual(int(ev["waypoint_index_after_request"][0]), 1)

    def test_final_arrival_also_requires_actual_palm_rotation_and_source_hand(self):
        for kind in ("position", "rotation", "hand"):
            with self.subTest(kind=kind):
                state = self.state();planner = HomeApproachPlanner(torch, resource())
                self.ready(planner, state)
                self.advance(state)
                if kind == "position":
                    state["palm_position_world_m"][:, 0] += .004
                elif kind == "rotation":
                    state["palm_quaternion_xyzw"][:] = torch.tensor([0., 0., math.sin(.1), math.cos(.1)])
                else:
                    state["q_action_rad"][:, 7] += .016
                _, ev = planner.actions(state, .066)
                self.assertFalse(bool(ev["kinematic_arrived"][0]))

    def test_environment_admission_freezes_progress_and_returns_no_arm_intervention(self):
        state = self.state();planner = HomeApproachPlanner(torch, resource())
        self.ready(planner, state)
        elapsed = planner.elapsed.clone()
        self.advance(state);state["approach_admitted"][:] = True
        request, ev = planner.actions(state, .066)
        self.assertTrue(bool(ev["admission_latched"][0]))
        self.assertTrue(torch.equal(request, torch.zeros_like(request)))
        self.assertTrue(torch.equal(planner.elapsed, elapsed))
        self.assertEqual(float(ev["approach_elapsed_s"][0]), float(ev["episode_elapsed_s"][0]))
        self.advance(state)
        _, later = planner.actions(state, .066)
        self.assertEqual(float(later["approach_elapsed_s"][0]), float(ev["approach_elapsed_s"][0]))
        self.assertGreater(float(later["episode_elapsed_s"][0]), float(ev["episode_elapsed_s"][0]))

    def test_failure_latches_and_only_changed_real_reset_epoch_clears_that_row(self):
        state = self.state(2);state["unsafe_latched"][1] = True
        planner = HomeApproachPlanner(torch, resource())
        request, ev = planner.actions(state, .066)
        self.assertEqual(ev["failed"].tolist(), [False, True])
        self.advance(state, request);state["unsafe_latched"].zero_()
        _, ev = planner.actions(state, .066)
        self.assertEqual(ev["failed"].tolist(), [False, True])
        self.advance(state)
        state["episode_id"][1] = 3;state["episode_control_steps"][1] = 0
        state["target_action_rad"][1, :7] = 0;state["q_action_rad"][1, :7] = 0
        state["palm_position_world_m"][1] = state["env_origin_world_m"][1]+torch.tensor([0., 0., 1.])
        _, ev = planner.actions(state, .066)
        self.assertFalse(bool(ev["failed"][1]))
        self.assertEqual(ev["new_real_episode"].tolist(), [False, True])

    def test_duplicate_or_decreasing_epoch_cannot_erase_failure(self):
        state = self.state();planner = HomeApproachPlanner(torch, resource())
        planner.actions(state, .066)
        request, ev = planner.actions(state, .066)
        self.assertTrue(bool(ev["failed"][0]))
        self.assertTrue(torch.equal(request, torch.zeros_like(request)))
        self.advance(state);state["episode_id"][0] = 1;state["episode_control_steps"][0] = 0
        planner.actions(state, .066)
        self.advance(state);state["episode_id"][0] = 2;state["episode_control_steps"][0] = 0
        _, ev = planner.actions(state, .066)
        self.assertTrue(bool(ev["failed"][0]))
        self.assertFalse(bool(ev["new_real_episode"][0]))

    def test_home_fk_error_is_not_hidden_by_estimating_clone_origin(self):
        state = self.state(2);state["palm_position_world_m"][1, 0] += .004
        _, ev = HomeApproachPlanner(torch, resource()).actions(state, .066)
        self.assertEqual(ev["failed"].tolist(), [False, True])
        self.assertAlmostEqual(float(ev["home_palm_position_error_m"][1]), .004)

    def test_timeout_uses_actual_episode_clock_even_while_tracking_paused(self):
        state = self.state();planner = HomeApproachPlanner(torch, resource(duration=.066))
        for _ in range(244):
            request, ev = planner.actions(state, .066)
            self.advance(state, request, track=False)
        self.assertTrue(bool(ev["failed"][0]))
        self.assertTrue(int(ev["failure_code"][0]) & FAILURE_CODES["approach_timeout"])
        self.assertGreaterEqual(float(ev["episode_elapsed_s"][0]), 16.)
        self.assertTrue(torch.equal(request, torch.zeros_like(request)))

    def test_nonfinite_workspace_and_native_target_limit_fail_closed(self):
        for kind, code in (("nonfinite", "nonfinite_measurement"), ("workspace", "palm_workspace_violation"),
                           ("limits", "declared_target_outside_native_limits")):
            with self.subTest(kind=kind):
                state = self.state()
                if kind == "nonfinite":state["q_action_rad"][0, 0] = float("nan")
                elif kind == "workspace":state["palm_position_world_m"][0, 2] = 1.11
                else:state["joint_limits_action_rad"][0, 0, 1] = .1
                request, ev = HomeApproachPlanner(torch, resource()).actions(state, .066)
                self.assertTrue(bool(ev["failed"][0]))
                self.assertTrue(int(ev["failure_code"][0]) & FAILURE_CODES[code])
                self.assertTrue(bool(torch.isfinite(request).all()))
                self.assertTrue(torch.equal(request, torch.zeros_like(request)))

    def test_actions_never_mutate_input_state_and_reject_missing_admission_field(self):
        state = self.state();before = {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
        HomeApproachPlanner(torch, resource()).actions(state, .066)
        for key in state:
            self.assertTrue(torch.equal(state[key], before[key]) if torch.is_tensor(state[key]) else state[key] == before[key])
        del state["approach_admitted"]
        with self.assertRaisesRegex(ValueError, "approach_admitted"):
            HomeApproachPlanner(torch, resource()).actions(state, .066)


if __name__ == "__main__":
    unittest.main()
