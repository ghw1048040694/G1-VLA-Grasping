"""Behavioral disturbance tests for the explicit zero-training pinch controller."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest


TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH, "real torch is required for measured hand-control math")
class CompliantPinchControllerTests(unittest.TestCase):
    def setUp(self):
        import torch
        from g1_vla.rl_hand_control import CompliantPinchController
        self.t = torch
        self.constructor = CompliantPinchController
        root = Path(__file__).resolve().parents[1]
        self.candidate = json.loads((root / "resources" / "rl_geometry_candidates" /
            "balanced_thumb_force_limited_v3.json").read_text())
        self.frames = self.candidate["geometry_evidence"]["source_detail"]["stage_keyframes"]
        self.seed = self.frames[0]["target_rad"]
        self.goals = torch.tensor([f["target_rad"] for f in self.frames[1:]])
        self.state = {"q_action_rad": torch.zeros(2, 14),
            "target_action_rad": torch.zeros(2, 14),
            "joint_limits_action_rad": torch.tensor([[[-2., 2.]] * 14] * 2),
            "episode_id": torch.zeros(2, dtype=torch.long),
            "group_normal_force_n": torch.zeros(2, 3),
            "contact_metrics_current_episode": torch.ones(2, dtype=torch.bool),
            "unsafe_latched": torch.zeros(2, dtype=torch.bool),
            "metrics": {"group_physical_loaded_contact_count": torch.zeros(2, 3, dtype=torch.long),
                "no_positive_gap_hand_load": torch.ones(2, dtype=torch.bool),
                "friction_partner_manifold_certified": torch.ones(2, dtype=torch.bool),
                "maximum_penetration_m": torch.zeros(2), "finite": torch.ones(2, dtype=torch.bool),
                "joint_safe": torch.ones(2, dtype=torch.bool),
                "table_pair_measurement_supported": torch.ones(2, dtype=torch.bool)}}
        self.state["q_action_rad"][:, 7:] = torch.tensor(self.seed)
        self.state["target_action_rad"][:, 7:] = torch.tensor(self.seed)
        self.controller = self.constructor(torch, self.seed, self.state, self.candidate)
        self.no_reset = torch.zeros(2, dtype=torch.bool)
        self.tracking = torch.ones(2, dtype=torch.bool)

    def step(self, *, changed=None, tracking=None):
        return self.controller.target(self.state, self.no_reset if changed is None else changed,
                                      .066, 1.5, self.tracking if tracking is None else tracking)

    def final_stage(self, error=0.):
        self.controller.stage_index[:] = 4
        self.controller.stage_elapsed[:] = self.controller.durations[4]
        self.controller.stage_start[:] = self.goals[-2]
        q = self.goals[-1] - self.controller.axis * error
        self.state["q_action_rad"][:, 7:] = q
        self.state["target_action_rad"][:, 7:] = q

    def test_free_space_target_lead_is_visible_but_does_not_certify_arrival(self):
        for _ in range(30):
            target, _, e = self.step()
        # A measured joint that never moves cannot pass the stage merely because
        # its nominal smoothstep finished. The larger lead is an explicit request.
        self.assertEqual(int(e["closure_stage_index"][0]), 0)
        self.assertFalse(e["stage_advance_after_this_request"].any())
        self.assertAlmostEqual(float(target[0, 3]-self.state["q_action_rad"][0, 10]), .04, places=6)
        self.assertFalse(e["force_feedback_contact_mode_latched"].any())
        self.assertTrue(e["force_feedback_does_not_hard_bound_real_force"])

    def test_stage0_near_arrival_allows_thumb_and_checks_active_measured_joints(self):
        self.controller.stage_elapsed[:] = self.controller.durations[0]
        self.state["q_action_rad"][:, 7:] = self.goals[0]
        self.state["target_action_rad"][:, 7:] = self.goals[0]
        self.state["q_action_rad"][0, 10] -= .024
        self.state["q_action_rad"][1, 10] -= .026
        _, _, e = self.step()
        self.assertTrue(bool(e["stage_advance_after_this_request"][0]))
        self.assertFalse(bool(e["stage_advance_after_this_request"][1]))
        self.assertFalse(e["force_feedback_physical_contact_ready"].any())
        self.assertTrue(bool(e["stage0_kinematic_near_arrival"][0]))
        self.assertFalse(e["closure_path_complete"].any())

    def test_thumb_reorientation_is_not_blocked_by_old_index_equilibrium_error(self):
        self.controller.stage_index[:] = 1
        self.controller.stage_start[:] = self.goals[0]
        self.controller.stage_elapsed[:] = self.controller.durations[1]
        self.state["q_action_rad"][:, 7:] = self.goals[1]
        self.state["target_action_rad"][:, 7:] = self.goals[1]
        self.state["q_action_rad"][:, 10] -= .03
        _, _, e = self.step()
        self.assertTrue(e["stage_advance_after_this_request"].all())
        self.assertGreater(float(e["stage_measured_error_max_rad"].min()), .025)
        self.assertLess(float(e["stage_active_measured_error_max_rad"].max()), .001)
        self.assertFalse(e["closure_path_complete"].any())

    def test_positive_gap_stage0_precontact_can_release_thumb_but_is_not_final_support(self):
        self.controller.stage_elapsed[:] = self.controller.durations[0]
        self.state["q_action_rad"][:, 7:] = self.goals[0]
        self.state["q_action_rad"][:, 10] -= .04
        self.state["target_action_rad"][:, 7:] = self.state["q_action_rad"][:, 7:]
        self.state["group_normal_force_n"][:, 1:] = .1
        self.state["metrics"]["no_positive_gap_hand_load"][:] = False
        _, _, first = self.step()
        self.assertFalse(first["stage_kinematic_arrival"].any())
        self.assertFalse(first["stage_advance_after_this_request"].any())
        _, _, second = self.step()
        self.assertTrue(second["stage0_raw_force_precontact_arrival"].all())
        self.assertTrue(second["stage_advance_after_this_request"].all())
        self.assertFalse(second["closure_path_complete"].any())
        self.assertFalse(second["force_feedback_physical_contact_ready"].any())
        self.final_stage(error=.018)
        self.state["group_normal_force_n"][:] = .4
        _, _, final = self.step()
        self.assertFalse(final["closure_path_complete"].any())
        self.assertFalse(final["stage_final_physical_contact_equilibrium"].any())

    def test_force_disturbance_opens_only_closing_axes_and_stays_within_retreat_budget(self):
        self.final_stage()
        self.state["group_normal_force_n"][:] = .9
        original = self.state["target_action_rad"][:, 7:].clone()
        target, _, e = self.step()
        axis = self.controller.axis
        self.assertTrue(((target-original)*axis <= 0).all())
        self.assertGreater(float(target[0, 0]), float(original[0, 0]))
        self.assertLess(float(target[0, 3]), float(original[0, 3]))
        self.assertTrue(self.t.equal(target[:, 1:3], original[:, 1:3]))
        self.assertLessEqual(float((target-original).abs().max()), .08*.066+1e-6)
        for _ in range(30):
            self.state["target_action_rad"][:, 7:] = target
            self.state["q_action_rad"][:, 7:] = target
            target, _, e = self.step()
        self.assertLessEqual(float((target-original).abs().max()), .040001)
        self.assertFalse(e["closure_path_complete"].any())

    def test_load_removal_recloses_slowly_without_resetting_contact_mode(self):
        self.final_stage()
        self.state["group_normal_force_n"][:] = 1.
        for _ in range(8):
            target, _, _ = self.step()
            self.state["target_action_rad"][:, 7:] = target
            self.state["q_action_rad"][:, 7:] = target
        withdrawn = target.clone()
        self.state["group_normal_force_n"][:] = 0.
        target, _, e = self.step()
        self.assertTrue(e["force_feedback_contact_mode_latched"].all())
        self.assertTrue(((target-withdrawn)*self.controller.axis >= -1e-7).all())
        self.assertGreater(float((target-withdrawn).abs().max()), 1e-4)
        self.assertLessEqual(float((target-withdrawn).abs().max()), .04*.066+1e-6)
        self.assertLessEqual(float((target-self.state["q_action_rad"][:, 7:]).abs().max()), .012001)

    def test_force_spike_overrides_palm_pause_only_for_bounded_retreat(self):
        self.final_stage()
        self.state["group_normal_force_n"][:, 1] = 10.
        previous = self.state["target_action_rad"][:, 7:].clone()
        target, _, e = self.step(tracking=~self.tracking)
        self.assertTrue((target[:, 3:5] < previous[:, 3:5]).all())
        self.assertTrue(self.t.equal(target[:, :3], previous[:, :3]))
        self.assertTrue(self.t.equal(target[:, 5:], previous[:, 5:]))
        self.assertLessEqual(float((target-previous).abs().max()), .08*.066+1e-6)
        self.assertTrue(e["raw_force_over_limit_by_group"][:, 1].all())
        self.assertTrue(e["force_relief_by_group"][:, 1].all())
        self.assertFalse(e["stage_advance_after_this_request"].any())
        self.assertFalse(e["closure_path_complete"].any())
        self.state["unsafe_latched"][:] = True
        unsafe_target, _, e = self.step(tracking=~self.tracking)
        self.assertTrue(self.t.equal(unsafe_target, previous))
        self.assertFalse(e["force_relief_by_group"].any())

    def test_force_retreat_uses_applied_reference_when_nominal_has_run_ahead(self):
        self.final_stage()
        # Native target lags the time-based waypoint by much more than 40mrad.
        # Clamping to nominal-40mrad here would wrongly close against a spike.
        self.state["q_action_rad"][:, 10] = .65
        self.state["target_action_rad"][:, 10] = .65
        self.state["group_normal_force_n"][:, 1] = 2.
        previous = self.state["target_action_rad"][:, 7:].clone()
        target, _, e = self.step()
        self.assertTrue((target[:, 3:5] < previous[:, 3:5]).all())
        self.assertTrue(e["hand_request_constraints_feasible"].all())
        self.assertTrue(self.t.allclose(
            e["force_admittance_most_closed_actual_applied_target_hand_rad"][:, 3:5],
            previous[:, 3:5]))
        self.assertFalse(e["closure_path_complete"].any())

    def test_incompatible_lead_and_force_step_hold_instead_of_reversing_force_feedback(self):
        self.final_stage()
        self.state["q_action_rad"][:, 10:12] -= .1
        self.state["group_normal_force_n"][:, 1] = 2.
        previous = self.state["target_action_rad"][:, 7:].clone()
        target, _, e = self.step()
        self.assertTrue(self.t.equal(target[:, 3:5], previous[:, 3:5]))
        self.assertFalse(e["hand_request_constraints_feasible"].any())
        self.assertFalse(e["force_relief_by_group"][:, 1].any())
        self.assertFalse(e["closure_path_complete"].any())

    def test_positive_gap_force_latches_feedback_but_cannot_finish_contact_equilibrium(self):
        self.final_stage(error=.018)
        self.state["group_normal_force_n"][:] = .4
        self.state["metrics"]["group_physical_loaded_contact_count"][:] = 0
        self.state["metrics"]["no_positive_gap_hand_load"][:] = False
        _, _, e = self.step()
        self.assertTrue(e["force_feedback_contact_mode_latched"].all())
        self.assertFalse(e["stage_final_physical_contact_equilibrium"].any())
        self.assertFalse(e["closure_path_complete"].any())

    def test_final_unloaded_thumb_search_is_slow_and_capped_without_extra_index_closure(self):
        self.final_stage()
        original = self.state["target_action_rad"][:, 7:].clone()
        previous = original.clone()
        for _ in range(65):
            target, _, e = self.step()
            self.assertLessEqual(float((target-previous).abs().max()), .04*.066+1e-6)
            self.assertTrue(self.t.equal(target[:, 1:], original[:, 1:]))
            self.assertTrue(e["adaptive_thumb_search_active"].all())
            self.assertFalse(e["force_feedback_contact_mode_latched"].any())
            self.state["target_action_rad"][:, 7:] = target
            self.state["q_action_rad"][:, 7:] = target
            previous = target.clone()
        floor = float(self.goals[-1, 0])-self.controller.parameters["adaptive_thumb0_closing_budget_rad"]
        self.assertAlmostEqual(float(target[0, 0]), floor, places=6)
        self.assertFalse(e["closure_path_complete"].any())
        self.assertTrue(e["adaptive_thumb_search_uses_unchanged_force_error_law"])

    def test_positive_gap_force_above_setpoint_suppresses_search_instead_of_forcing_through(self):
        self.final_stage()
        self.state["group_normal_force_n"][:, 0] = .8
        self.state["metrics"]["no_positive_gap_hand_load"][:] = False
        previous = self.state["target_action_rad"][:, 7:].clone()
        target, _, e = self.step()
        self.assertTrue((target[:, 0] > previous[:, 0]).all())
        self.assertTrue(e["adaptive_thumb_search_active"].all())
        self.assertTrue(e["adaptive_thumb_search_suppressed_by_force_error"].all())
        self.assertTrue(e["adaptive_thumb_raw_force_without_certified_physical_contact"].all())
        self.assertFalse(e["stage_final_physical_contact_equilibrium"].any())

    def test_extended_thumb_equilibrium_uses_actual_target_tracking_and_still_needs_physical_contacts(self):
        self.final_stage()
        self.state["q_action_rad"][:, 7] = self.goals[-1, 0]-.05
        self.state["target_action_rad"][:, 7] = self.goals[-1, 0]-.05
        self.state["group_normal_force_n"][:] = .4
        self.state["metrics"]["group_physical_loaded_contact_count"][:] = 1
        _, _, e = self.step()
        self.assertFalse(e["stage_final_joint_arrival"].any())
        self.assertTrue(e["stage_final_adaptive_equilibrium_geometry"].all())
        self.assertTrue(e["stage_final_physical_contact_equilibrium"].all())
        self.assertTrue(e["closure_path_complete"].all())
        # An articulation lag cannot be hidden by selecting an adaptive target.
        self.state["q_action_rad"][0, 7] += .03
        _, _, e = self.step()
        self.assertFalse(bool(e["stage_final_adaptive_equilibrium_geometry"][0]))
        self.assertFalse(bool(e["closure_path_complete"][0]))

    def test_adaptive_extension_is_unavailable_before_final_nominal_segment_finishes(self):
        self.controller.stage_index[:] = 3
        self.controller.stage_elapsed[:] = self.controller.durations[3]
        self.controller.stage_start[:] = self.goals[2]
        self.state["q_action_rad"][:, 7:] = self.goals[3]
        self.state["target_action_rad"][:, 7:] = self.goals[3]
        _, _, e = self.step()
        self.assertFalse(e["adaptive_thumb_search_started"].any())
        self.assertTrue(e["stage_advance_after_this_request"].all())
        target, _, e = self.step()
        self.assertFalse(e["adaptive_thumb_search_started"].any())
        self.assertTrue((target[:, 0] >= self.goals[-1, 0]).all())

    def test_final_contact_equilibrium_requires_each_physical_input_and_is_not_support(self):
        self.final_stage(error=.018)
        self.state["group_normal_force_n"][:] = .4
        self.state["metrics"]["group_physical_loaded_contact_count"][:] = 1
        _, _, e = self.step()
        self.assertTrue(e["stage_final_physical_contact_equilibrium"].all())
        self.assertTrue(e["closure_path_complete"].all())
        self.assertTrue(e["stage_final_physical_contact_equilibrium_is_not_support"])
        cases = [
            ("group_physical_loaded_contact_count", (0, 1), 0),
            ("no_positive_gap_hand_load", 0, False),
            ("friction_partner_manifold_certified", 0, False),
            ("maximum_penetration_m", 0, .00301),
            ("table_pair_measurement_supported", 0, False),
        ]
        for key, idx, value in cases:
            with self.subTest(key=key):
                old = self.state["metrics"][key].clone()
                self.state["metrics"][key][idx] = value
                _, _, evidence = self.step()
                self.assertFalse(bool(evidence["closure_path_complete"][0]))
                self.state["metrics"][key] = old
        self.state["group_normal_force_n"][0, 2] = 1.51
        _, _, e = self.step()
        self.assertFalse(bool(e["closure_path_complete"][0]))

    def test_missing_stale_nan_unsafe_or_untracked_feedback_holds_and_cannot_advance(self):
        self.controller.stage_elapsed[:] = self.controller.durations[0]
        self.state["q_action_rad"][:, 7:] = self.goals[0]
        self.state["target_action_rad"][:, 7:] = self.goals[0]
        for case in ("missing", "stale", "nan", "unsafe", "palm"):
            with self.subTest(case=case):
                original_state = copy.deepcopy(self.state)
                original_elapsed = self.controller.stage_elapsed.clone()
                if case == "missing":
                    del self.state["metrics"]["group_physical_loaded_contact_count"]
                elif case == "stale":
                    self.state["contact_metrics_current_episode"][:] = False
                elif case == "nan":
                    self.state["group_normal_force_n"][:] = float("nan")
                elif case == "unsafe":
                    self.state["unsafe_latched"][:] = True
                tracking = self.tracking if case != "palm" else ~self.tracking
                target, _, e = self.step(tracking=tracking)
                self.assertTrue(self.t.equal(target, original_state["target_action_rad"][:, 7:]))
                self.assertFalse(e["stage_advance_after_this_request"].any())
                self.assertFalse(e["closure_path_complete"].any())
                self.assertTrue(self.t.equal(original_elapsed, self.controller.stage_elapsed))
                self.state = original_state

    def test_first_reset_sample_holds_once_then_can_use_current_contact_measurements(self):
        self.state["contact_metrics_current_episode"][:] = False
        self.state["group_normal_force_n"][:] = 99.  # Deliberately stale old load.
        target, _, e = self.step()
        self.assertTrue(self.t.equal(target, self.state["target_action_rad"][:, 7:]))
        self.assertFalse(e["force_feedback_contact_mode_latched"].any())
        self.state["contact_metrics_current_episode"][:] = True
        self.state["group_normal_force_n"][:] = 0.
        for _ in range(8):
            target, _, e = self.step()
        self.assertGreater(float((target-self.state["q_action_rad"][:, 7:]).abs().max()), .006)
        self.assertFalse(e["hand_request_held"].any())

    def test_episode_reset_clears_only_that_environments_phase_and_contact_memory(self):
        self.final_stage(error=.018)
        self.state["group_normal_force_n"][:] = .8
        self.step()
        self.state["episode_id"][0] += 1
        self.state["q_action_rad"][0, 7:] = self.t.tensor(self.seed)
        self.state["target_action_rad"][0, 7:] = self.t.tensor(self.seed)
        self.state["contact_metrics_current_episode"][0] = False
        target, _, e = self.step()
        self.assertEqual(int(e["closure_stage_index"][0]), 0)
        self.assertEqual(int(e["closure_stage_index"][1]), 4)
        self.assertFalse(e["force_feedback_contact_mode_latched"][0].any())
        self.assertTrue(e["force_feedback_contact_mode_latched"][1].all())
        self.assertTrue(self.t.equal(target[0], self.t.tensor(self.seed)))
        self.assertFalse(bool(e["closure_path_complete"][0]))

    def test_parameters_reset_source_and_waypoint_topology_are_fail_closed(self):
        from g1_vla.rl_hand_control import COMPLIANT_PINCH_PARAMETERS
        for mutation in ("unknown", "value", "type"):
            params = copy.deepcopy(COMPLIANT_PINCH_PARAMETERS)
            if mutation == "unknown":
                params["unexpected"] = 1.
            elif mutation == "value":
                params["free_target_lead_max_rad"] += .001
            else:
                params["force_target_n"] = True
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "contract"):
                self.constructor(self.t, self.seed, self.state, self.candidate, parameters=params)
        candidate = copy.deepcopy(self.candidate)
        candidate["geometry_evidence"]["source_detail"]["stage_keyframes"][2]["target_rad"][0] -= .01
        with self.assertRaisesRegex(ValueError, "topology"):
            self.constructor(self.t, self.seed, self.state, candidate)
        self.state["joint_limits_action_rad"][:, 7, 0] = .95
        with self.assertRaisesRegex(ValueError, "adaptive thumb budget"):
            self.constructor(self.t, self.seed, self.state, self.candidate)
        self.state["joint_limits_action_rad"][:, 7, 0] = -2.
        self.state["episode_id"][0] += 1
        self.state["q_action_rad"][0, 7] -= .01
        with self.assertRaisesRegex(ValueError, "measured source"):
            self.step()


@unittest.skipUnless(TORCH, "real torch is required for 2ms controller sequence tests")
class PhysicsStepPinchTests(unittest.TestCase):
    def setUp(self):
        CompliantPinchControllerTests.setUp(self)
        from g1_vla.rl_hand_control import PhysicsStepForceAdmittance, PhysicsPinchTrajectory
        frames = self.candidate["geometry_evidence"]["source_detail"]["stage_keyframes"]
        for frame in frames[2:]:
            frame["target_rad"][1] = -.55
        for frame in frames[3:]:
            frame["target_rad"][2] = -.02
        frames[4]["target_rad"][0] = 1.044
        frames[5]["target_rad"][0] = .97
        self.goals = self.t.tensor([f["target_rad"] for f in frames[1:]])
        self.trajectory = PhysicsPinchTrajectory(self.t, self.seed, self.state, self.candidate)
        self.applied = self.t.tensor(self.seed)[None].repeat(2, 1)
        self.measured = self.applied.clone()
        self.nominal = self.applied.clone()
        self.axis = self.t.tensor([-1., 0., 0., 1., 1., 1., 1.])
        self.inner = PhysicsStepForceAdmittance(self.t, self.applied, self.state["episode_id"])
        self.state["metrics"].update(contact_safe=self.t.ones(2, dtype=self.t.bool),
            opposition_dot=self.t.full((2,), -.7), maximum_abs_normal_z=self.t.full((2,), .2))

    def inputs(self, sample):
        return {"sample_id": sample, "episode_id": self.state["episode_id"].clone(), "dt": .002,
            "measured_hand_rad": self.measured.clone(), "applied_hand_rad": self.applied.clone(),
            "nominal_hand_rad": self.nominal.clone(),
            "geometry_lower_hand_rad": self.t.tensor(self.seed)[None].repeat(2, 1)-.5,
            "geometry_upper_hand_rad": self.t.tensor(self.seed)[None].repeat(2, 1)+.5,
            "step_lower_hand_rad": self.applied-.001, "step_upper_hand_rad": self.applied+.001,
            "macro_lower_hand_rad": self.t.tensor(self.seed)[None].repeat(2, 1)-.2,
            "macro_upper_hand_rad": self.t.tensor(self.seed)[None].repeat(2, 1)+.2,
            "raw_group_force_n": self.t.zeros(2, 3),
            "contact_measurement_valid": self.t.ones(2, dtype=self.t.bool),
            "tracking": self.t.ones(2, dtype=self.t.bool),
            "unsafe_latched": self.t.zeros(2, dtype=self.t.bool),
            "penetration_m": self.t.zeros(2), "closing_axis_hand7": self.axis}

    def prime(self):
        inputs = self.inputs(0)
        inputs["contact_measurement_valid"][:] = False
        target, evidence, unsafe = self.inner.update(**inputs)
        self.assertTrue(self.t.equal(target, self.applied))
        self.assertFalse(unsafe.any())
        self.assertTrue(evidence["first_sample_hold"])

    def commit(self, target):
        self.applied = target.clone()
        self.measured = target.clone()

    def test_free_reference_follows_original_speed_box_without_force_close_cap(self):
        self.prime()
        self.nominal[:, 3] += .2
        target, evidence, unsafe = self.inner.update(**self.inputs(1))
        self.assertAlmostEqual(float(target[0, 3]-self.applied[0, 3]), .001, places=6)
        self.assertGreater(float(target[0, 3]-self.applied[0, 3]), .04*.002)
        self.assertFalse(evidence["force_feedback_contact_mode_latched"].any())
        self.assertFalse(unsafe.any())

    def test_force_spike_changes_next_two_ms_target_and_not_the_measured_sample(self):
        self.prime()
        inputs = self.inputs(1)
        inputs["raw_group_force_n"][:] = 2.
        inputs["tracking"][:] = False
        target, evidence, unsafe = self.inner.update(**inputs)
        self.assertGreater(float(target[0, 0]), float(self.applied[0, 0]))
        self.assertLess(float(target[0, 3]), float(self.applied[0, 3]))
        self.assertLessEqual(float((target-self.applied).abs().max()), .08*.002+1e-7)
        self.assertEqual(evidence["force_sample_id"], 1)
        self.assertEqual(evidence["target_for_native_sample_id"], 2)
        self.assertTrue(evidence["raw_force_over_limit_by_group"].all())
        self.assertTrue(evidence["force_request_bounds_are_not_measured_force_bounds"])
        self.assertFalse(unsafe.any())

    def test_contact_loss_alternation_keeps_mode_and_never_switches_to_fast_free_motion(self):
        self.prime()
        self.nominal[:, 3] += .1
        for sample in range(1, 61):
            inputs = self.inputs(sample)
            inputs["raw_group_force_n"][:] = .7 if sample % 2 else 0.
            target, evidence, unsafe = self.inner.update(**inputs)
            self.assertTrue(evidence["force_feedback_contact_mode_latched"].all())
            self.assertFalse(unsafe.any())
            self.assertLessEqual(float((target[:, [0, 3, 4, 5, 6]]-
                self.applied[:, [0, 3, 4, 5, 6]]).abs().max()), .04*.002+1e-7)
            self.commit(target)

    def test_force_deadband_holds_targets_but_thumb_reorientation_remains_independent(self):
        self.prime()
        self.nominal[:, 1] += .02
        inputs = self.inputs(1)
        inputs["raw_group_force_n"][:] = .39
        target, evidence, unsafe = self.inner.update(**inputs)
        self.assertTrue(evidence["force_feedback_deadband"].all())
        self.assertTrue(self.t.equal(target[:, [0, 3, 4, 5, 6]], self.applied[:, [0, 3, 4, 5, 6]]))
        self.assertGreater(float(target[0, 1]), float(self.applied[0, 1]))
        self.assertFalse(unsafe.any())

    def test_duplicate_sample_does_not_integrate_a_second_force_response(self):
        self.prime()
        inputs = self.inputs(1)
        inputs["raw_group_force_n"][:] = .8
        target, _, _ = self.inner.update(**inputs)
        self.commit(target)
        duplicate = self.inputs(1)
        duplicate["raw_group_force_n"][:] = .8
        same, evidence, unsafe = self.inner.update(**duplicate)
        self.assertTrue(self.t.equal(same, self.applied))
        self.assertTrue(evidence["duplicate_sample_hold"])
        self.assertFalse(unsafe.any())
        skipped, evidence, unsafe = self.inner.update(**self.inputs(3))
        self.assertTrue(self.t.equal(skipped, self.applied))
        self.assertTrue(evidence["sample_discontinuity"])
        self.assertTrue(unsafe.all())

    def test_first_contact_recovers_tighter_lead_continuously_without_false_infeasibility(self):
        self.prime()
        self.applied[:, 3] += .015
        initial_applied = self.applied.clone()
        for sample in range(1, 101):
            inputs = self.inputs(sample)
            inputs["raw_group_force_n"][:, 1] = .1  # Force law alone requests closing.
            target, evidence, unsafe = self.inner.update(**inputs)
            self.assertFalse(unsafe.any())
            self.assertLessEqual(float((target-self.applied).abs().max()), .08*.002+1e-7)
            if sample == 1:
                self.assertTrue(evidence["contact_lead_recovery"].all())
                self.assertTrue(evidence["actual_requested_lead_not_yet_met"].all())
                self.assertLess(float(target[0, 3]), float(self.applied[0, 3]))
            self.applied = target.clone()
            # Measured q deliberately does not instantaneously follow target.
        self.assertLess(float(self.applied[0, 3]-self.measured[0, 3]), .003001)
        self.assertLess(float(self.applied[0, 3]), float(initial_applied[0, 3]))

    def test_high_force_retreat_is_not_reversed_to_meet_tight_lead(self):
        self.prime()
        self.applied[:, 3] -= .01  # Applied target already open relative to q.
        inputs = self.inputs(1)
        inputs["raw_group_force_n"][:, 1] = 2.
        target, evidence, unsafe = self.inner.update(**inputs)
        self.assertLess(float(target[0, 3]), float(self.applied[0, 3]))
        self.assertTrue(evidence["contact_lead_high_force_retreat_priority_by_joint"][:, 3].all())
        self.assertTrue(evidence["actual_requested_lead_not_yet_met"].all())
        self.assertFalse(unsafe.any())

    def test_recorded_single_ulp_lead_breaches_enter_recovery_without_relaxing_native_bounds(self):
        self.prime()
        # Real run05 first-episode index0 states: both are one FP32 ULP beyond
        # q+3mrad but below the former independent recovery detection tolerance.
        self.applied[:, 3] = self.t.tensor([.8173501491546631, .8181098699569702])
        self.measured[:, 3] = self.t.tensor([.8143500685691833, .8151097893714905])
        inputs = self.inputs(1)
        inputs["raw_group_force_n"][:, 1] = self.t.tensor([.04888264834880829, .3249495327472687])
        inputs["macro_lower_hand_rad"] = self.applied-.2
        inputs["macro_upper_hand_rad"] = self.applied+.2
        target, e, unsafe = self.inner.update(**inputs)
        exact_upper = self.measured[:, 3]+.003
        self.assertTrue((self.applied[:, 3] > exact_upper).all())
        self.assertTrue(e["contact_lead_recovery_by_joint"][:, 3].all())
        self.assertTrue(e["contact_lead_recovery_too_closed_by_joint"][:, 3].all())
        self.assertTrue(self.t.equal(target[:, 3], exact_upper))
        self.assertFalse(e["actual_requested_lead_not_yet_met_by_joint"][:, 3].any())
        self.assertFalse(e["request_constraints_infeasible"].any())
        self.assertFalse(unsafe.any())
        self.assertTrue((target >= inputs["step_lower_hand_rad"]).all())
        self.assertTrue((target <= inputs["step_upper_hand_rad"]).all())
        self.assertTrue((target >= inputs["macro_lower_hand_rad"]).all())
        self.assertTrue((target <= inputs["macro_upper_hand_rad"]).all())

    def test_native_box_conflict_still_fails_closed_during_lead_recovery(self):
        self.prime()
        self.applied[:, 3] += .015
        inputs = self.inputs(1)
        inputs["raw_group_force_n"][:, 1] = 2.
        inputs["macro_lower_hand_rad"][:, 3] = self.applied[:, 3]+.01
        target, evidence, unsafe = self.inner.update(**inputs)
        self.assertTrue(self.t.equal(target, self.applied))
        self.assertTrue(evidence["request_constraints_infeasible"].all())
        self.assertTrue(unsafe.all())

    def test_missing_nan_depth_or_unsafe_feedback_does_not_write_a_new_target(self):
        for case in ("missing", "nan", "depth", "unsafe"):
            with self.subTest(case=case):
                self.setUp()
                self.prime()
                self.inner.update(**self.inputs(1))  # Establish actual valid feedback first.
                inputs = self.inputs(2)
                inputs["raw_group_force_n"][:] = 2.
                if case == "missing":
                    inputs["contact_measurement_valid"][0] = False
                elif case == "nan":
                    inputs["raw_group_force_n"][0, 0] = float("nan")
                elif case == "depth":
                    inputs["penetration_m"][0] = .00301
                else:
                    inputs["unsafe_latched"][0] = True
                target, _, unsafe = self.inner.update(**inputs)
                self.assertTrue(self.t.equal(target[0], self.applied[0]))
                self.assertTrue(bool(unsafe[0]))
                self.assertFalse(bool(unsafe[1]))

    def test_initial_measurement_handshake_can_wait_but_loss_after_valid_feedback_is_unsafe(self):
        self.prime()
        self.nominal[:, 3] += .1
        for sample in range(1, 8):
            inputs = self.inputs(sample)
            inputs["contact_measurement_valid"][:] = False
            target, e, unsafe = self.inner.update(**inputs)
            self.assertTrue(self.t.equal(target, self.applied))
            self.assertTrue(e["waiting_for_first_valid_measurement"].all())
            self.assertFalse(unsafe.any())
        target, e, unsafe = self.inner.update(**self.inputs(8))
        self.assertFalse(e["waiting_for_first_valid_measurement"].any())
        self.assertTrue(e["seen_valid_feedback_for_current_episode"].all())
        self.assertGreater(float(target[0, 3]), float(self.applied[0, 3]))
        self.assertFalse(unsafe.any())
        self.commit(target)
        inputs = self.inputs(9)
        inputs["contact_measurement_valid"][:] = False
        target, e, unsafe = self.inner.update(**inputs)
        self.assertTrue(self.t.equal(target, self.applied))
        self.assertTrue(e["previously_valid_measurement_lost"].all())
        self.assertTrue(unsafe.all())

    def test_physics_retreat_budget_has_no_windup_and_load_withdrawal_recloses_slowly(self):
        self.prime()
        initial = self.applied.clone()
        for sample in range(1, 501):
            inputs = self.inputs(sample)
            inputs["raw_group_force_n"][:] = .8
            target, _, unsafe = self.inner.update(**inputs)
            self.assertFalse(unsafe.any())
            self.assertLessEqual(float((target-initial).abs().max()), .040001)
            self.commit(target)
        prior = self.applied.clone()
        target, e, unsafe = self.inner.update(**self.inputs(501))
        self.assertTrue(e["force_feedback_contact_mode_latched"].all())
        self.assertGreater(float(((target-prior)*self.axis).max()), 0.)
        self.assertLessEqual(float((target-prior).abs().max()), .04*.002+1e-7)
        self.assertFalse(unsafe.any())

    def test_partial_episode_reset_clears_contact_and_stale_force_without_resetting_other_env(self):
        self.prime()
        inputs = self.inputs(1)
        inputs["raw_group_force_n"][:] = .8
        target, _, _ = self.inner.update(**inputs)
        self.commit(target)
        self.state["episode_id"][0] += 1
        self.applied[0] = self.t.tensor(self.seed)
        self.measured[0] = self.applied[0]
        inputs = self.inputs(2)
        inputs["raw_group_force_n"][:] = .8
        inputs["raw_group_force_n"][0] = 100.
        inputs["contact_measurement_valid"][0] = False
        target, evidence, unsafe = self.inner.update(**inputs)
        self.assertTrue(self.t.equal(target[0], self.t.tensor(self.seed)))
        self.assertFalse(evidence["force_feedback_contact_mode_latched"][0].any())
        self.assertTrue(evidence["force_feedback_contact_mode_latched"][1].all())
        self.assertFalse(unsafe.any())

    def test_macro_reference_advances_near_contact_without_force_integrating_or_mutating_targets(self):
        self.trajectory.stage_elapsed[:] = self.trajectory.durations[0]
        self.state["q_action_rad"][:, 7:] = self.goals[0]
        self.state["q_action_rad"][:, 10] -= .04
        self.state["target_action_rad"][:, 7:] = self.state["q_action_rad"][:, 7:]
        self.state["group_normal_force_n"][:, 1:] = .1
        self.state["metrics"]["no_positive_gap_hand_load"][:] = False
        before = self.state["target_action_rad"].clone()
        for _ in range(2):
            nominal, _, evidence = self.trajectory.reference(self.state, self.no_reset, .066, 1.5, self.tracking)
        self.assertTrue(self.t.equal(before, self.state["target_action_rad"]))
        self.assertTrue(self.t.equal(nominal, self.goals[0][None].repeat(2, 1)))
        self.assertTrue(evidence["stage_advance_after_this_request"].all())
        self.assertTrue(evidence["macro_reference_only_no_force_target_update"])
        self.assertFalse(evidence["closure_path_complete"].any())
        self.assertFalse(hasattr(self.trajectory, "contact_latched"))

    def test_v5_final_equilibrium_accepts_real_contact_before_nominal_thumb_but_requires_side_gates(self):
        self.trajectory.stage_index[:] = 4
        self.trajectory.stage_elapsed[:] = self.trajectory.durations[4]
        self.trajectory.stage_start[:] = self.goals[-2]
        self.state["q_action_rad"][:, 7:] = self.goals[-1]
        self.state["q_action_rad"][:, 7] = 1.024
        self.state["target_action_rad"][:, 7:] = self.state["q_action_rad"][:, 7:]
        self.state["group_normal_force_n"][:] = .4
        self.state["metrics"]["group_physical_loaded_contact_count"][:] = 1
        _, _, evidence = self.trajectory.reference(self.state, self.no_reset, .066, 1.5, self.tracking)
        self.assertGreater(float(evidence["stage_measured_error_max_rad"].min()), .05)
        self.assertTrue(evidence["stage_final_physical_contact_equilibrium"].all())
        self.assertTrue(evidence["closure_path_complete"].all())
        for key, value in (("opposition_dot", -.3), ("maximum_abs_normal_z", .7),
                           ("no_positive_gap_hand_load", False), ("contact_safe", False)):
            with self.subTest(key=key):
                old = self.state["metrics"][key].clone()
                self.state["metrics"][key][:] = value
                _, _, e = self.trajectory.reference(self.state, self.no_reset, .066, 1.5, self.tracking)
                self.assertFalse(e["closure_path_complete"].any())
                self.state["metrics"][key] = old
        del self.state["metrics"]["opposition_dot"]
        _, _, e = self.trajectory.reference(self.state, self.no_reset, .066, 1.5, self.tracking)
        self.assertFalse(e["closure_path_complete"].any())

    def test_v5_final_equilibrium_cannot_hide_other_joint_or_actual_applied_tracking_error(self):
        self.trajectory.stage_index[:] = 4
        self.trajectory.stage_elapsed[:] = self.trajectory.durations[4]
        self.trajectory.stage_start[:] = self.goals[-2]
        self.state["q_action_rad"][:, 7:] = self.goals[-1]
        self.state["q_action_rad"][:, 7] = 1.024
        self.state["target_action_rad"][:, 7:] = self.state["q_action_rad"][:, 7:]
        self.state["target_action_rad"][0, 7] -= .03
        self.state["q_action_rad"][1, 8] += .03  # Thumb1 orientation must be attained.
        self.state["target_action_rad"][1, 8] = self.state["q_action_rad"][1, 8]
        self.state["group_normal_force_n"][:] = .4
        self.state["metrics"]["group_physical_loaded_contact_count"][:] = 1
        _, _, e = self.trajectory.reference(self.state, self.no_reset, .066, 1.5, self.tracking)
        self.assertFalse(e["closure_path_complete"].any())

    def test_middle_contact_may_stop_before_free_goal_only_with_real_opposed_contact_and_orientation(self):
        from g1_vla.rl_hand_control import PhysicsPinchTrajectory
        candidate = copy.deepcopy(self.candidate)
        frames = candidate["geometry_evidence"]["source_detail"]["stage_keyframes"]
        frames[-1]["target_rad"][5] += .04
        frames[-1]["target_rad"][6] += .04
        trajectory = PhysicsPinchTrajectory(self.t, self.seed, self.state, candidate)
        trajectory.stage_index[:] = 4
        trajectory.stage_elapsed[:] = trajectory.durations[4]
        trajectory.stage_start[:] = trajectory.goals[-2]
        self.state["q_action_rad"][:, 7:] = trajectory.goals[-1]
        self.state["q_action_rad"][:, 7] = 1.024
        self.state["q_action_rad"][:, 12:14] -= .04
        self.state["target_action_rad"][:, 7:] = self.state["q_action_rad"][:, 7:]
        self.state["group_normal_force_n"][:] = .4
        self.state["metrics"]["group_physical_loaded_contact_count"][:] = 1
        _, _, e = trajectory.reference(self.state, self.no_reset, .066, 1.5, self.tracking)
        self.assertGreater(float(e["stage_final_equilibrium_other_six_goal_error_rad"].min()), .025)
        self.assertTrue(e["stage_final_physical_contact_equilibrium"].all())
        self.assertTrue(e["closure_path_complete"].all())
        self.state["metrics"]["group_physical_loaded_contact_count"][0, 2] = 0
        self.state["q_action_rad"][1, 8] += .03
        self.state["target_action_rad"][1, 8] = self.state["q_action_rad"][1, 8]
        _, _, e = trajectory.reference(self.state, self.no_reset, .066, 1.5, self.tracking)
        self.assertFalse(e["closure_path_complete"].any())

    def test_current_v5_geometry_resource_produces_declared_caps_without_elapsed_time_acceptance(self):
        from g1_vla.rl_hand_control import PhysicsPinchTrajectory
        root = Path(__file__).resolve().parents[1]
        candidate = json.loads((root/"resources/rl_geometry_candidates/opposed_contact_physics_pinch_v4.json").read_text())
        trajectory = PhysicsPinchTrajectory(self.t, self.seed, self.state, candidate)
        trajectory.stage_index[:] = 4
        trajectory.stage_elapsed[:] = trajectory.durations[4]
        trajectory.stage_start[:] = trajectory.goals[-2]
        nominal, _, e = trajectory.reference(self.state, self.no_reset, .066, 1.5, self.tracking)
        self.assertAlmostEqual(float(nominal[0, 0]), .94, places=6)
        self.assertAlmostEqual(float(e["geometry_lower_hand_rad"][0, 0]), .92, places=6)
        self.assertAlmostEqual(float(nominal[0, 1]), -.52, places=6)
        self.assertAlmostEqual(float(nominal[0, 2]), -.04, places=6)
        self.assertFalse(e["closure_path_complete"].any())


@unittest.skipUnless(TORCH, "real torch is required for explicit effort-control math")
class BalancedEffortGraspTests(unittest.TestCase):
    def setUp(self):
        import torch
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        from g1_vla.rl_effort_protocol import balanced_effort_controller_contract
        self.t = torch
        self.parameters = balanced_effort_controller_contract()["parameters"]
        self.seed = torch.zeros(2, 7)
        self.ids = torch.zeros(2, dtype=torch.long)
        self.controller = BalancedEffortGraspController(torch, self.seed, self.ids)
        self.sample = 0
        self.a = torch.zeros(2, 3, 7)
        self.a[:, 0, 0] = -.04
        self.a[:, 1, 3:5] = torch.tensor([.03, .02])
        self.a[:, 2, 5:7] = torch.tensor([.03, .02])
        self.kw = {
            "episode_id": self.ids, "dt": .002, "measured_hand_rad": self.seed.clone(),
            "measured_hand_velocity_rad_s": torch.zeros(2, 7), "nominal_hand_rad": self.seed.clone(),
            "gravity_hand_nm": torch.zeros(2, 7), "inward_normal_jacobian": self.a,
            "group_contact_valid": torch.ones(2, 3, dtype=torch.bool),
            "group_physical_loaded": torch.ones(2, 3, dtype=torch.bool),
            "raw_group_force_n": torch.tensor([[.8, .4, .4]]*2),
            "contact_measurement_valid": torch.ones(2, dtype=torch.bool),
            "tracking": torch.ones(2, dtype=torch.bool), "unsafe_latched": torch.zeros(2, dtype=torch.bool),
            "joint_lower_hand_rad": torch.full((2, 7), -2.), "joint_upper_hand_rad": torch.full((2, 7), 2.),
            "geometry_lower_hand_rad": torch.full((2, 7), -1.),
            "geometry_upper_hand_rad": torch.full((2, 7), 1.),
            "nominal_velocity_limit_hand_rad_s": torch.full((2, 7), .9),
            "penetration_m": torch.zeros(2), "closing_axis_hand7": torch.tensor([-1., 0., 0., 1., 1., 1., 1.]),
        }
        self.step()  # No first-sample force may leak into the new controller.

    def step(self):
        result = self.controller.update(sample_id=self.sample, **self.kw)
        self.sample += 1
        return result

    def test_current_normal_sign_and_declared_three_group_balance(self):
        effort, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertAlmostEqual(float(effort[0, 0]), -.032, places=6)
        self.assertAlmostEqual(float(effort[0, 3]), .012, places=6)
        self.assertAlmostEqual(float(effort[0, 4]), .008, places=6)
        self.assertTrue(self.t.allclose(e["group_force_command_n"], self.kw["raw_group_force_n"]))
        # Opposing x normals (-1,+1,+1) yield zero reference net x force.
        normals = self.t.tensor([[-1., 0., 0.], [1., 0., 0.], [1., 0., 0.]])
        reference_net = e["group_force_command_n"] @ normals
        self.assertTrue(self.t.allclose(reference_net, self.t.zeros_like(reference_net)))
        self.assertTrue(e["force_target_distribution_is_not_verified_wrench_balance"])
        self.kw["inward_normal_jacobian"] = -self.a
        reversed_effort, _, _ = self.step()
        self.assertTrue(self.t.allclose(reversed_effort, -effort))

    def test_no_hidden_position_spring_along_current_contact_normal(self):
        # Deliberately huge position error along the three force rows must not
        # add squeezing torque. This detects retaining the old 9 Nm/rad PD law.
        self.kw["nominal_hand_rad"] = self.a.sum(1)*5
        self.kw["nominal_velocity_limit_hand_rad_s"][:] = 1000.
        effort, e, _ = self.step()
        expected = self.t.tensor([[-.032, 0., 0., .012, .008, .012, .008]]*2)
        self.assertTrue(self.t.allclose(effort, expected, atol=1e-7, rtol=0))
        self.assertGreater(float(e["position_spring_before_contact_mode_mask_nm"].abs().max()), .01)
        self.assertTrue(self.t.equal(e["position_spring_after_contact_mode_mask_nm"], self.t.zeros_like(effort)))
        self.assertLess(float(e["normal_posture_residual_m_nm"].abs().max()), 1e-8)
        self.assertTrue(e["requires_zero_native_position_velocity_drives"])

    def test_current_force_mode_does_not_chase_tangential_thumb_reorientation_nominal(self):
        self.kw["nominal_hand_rad"][:, 1] = .2
        self.kw["nominal_velocity_limit_hand_rad_s"][:] = 1000.
        effort, e, _ = self.step()
        self.assertEqual(float(effort[0, 1]), 0.)
        self.assertAlmostEqual(float(effort[0, 0]), -.032, places=6)
        self.assertGreater(float(e["position_spring_before_contact_mode_mask_nm"][0, 1]), .02)
        self.assertTrue(e["position_spring_disabled_by_current_normal_joint"].all())
        self.assertLess(float(e["normal_posture_residual_m_nm"].abs().max()), 1e-8)

    def test_large_tangential_index_error_cannot_add_contact_downward_drive(self):
        # [2,-3] is orthogonal to this index normal row [.03,.02]. The old
        # normal-only position projection kept this error and chased the cube
        # tangentially despite a perfectly balanced normal-force reading.
        nominal = self.t.tensor([0., .5, -.4, .4, -.6, .4, -.6])
        self.assertAlmostEqual(float((self.a[0, 1]*nominal).sum()), 0., places=7)
        self.kw["nominal_hand_rad"][:] = nominal
        self.kw["nominal_velocity_limit_hand_rad_s"][:] = 1000.
        effort, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        expected = self.t.tensor([[-.032, 0., 0., .012, .008, .012, .008]]*2)
        self.assertTrue(self.t.allclose(effort, expected, atol=1e-7, rtol=0))
        self.assertGreater(float(e["position_spring_before_contact_mode_mask_nm"].abs().max()), .05)
        self.assertTrue(self.t.equal(e["posture_torque_after_normal_projection_nm"], self.t.zeros_like(effort)))

    def test_contact_mode_retains_damping_for_tangential_motion_without_position_spring(self):
        self.kw["nominal_hand_rad"][:, 1] = .8
        self.kw["nominal_velocity_limit_hand_rad_s"][:] = 1000.
        self.kw["measured_hand_velocity_rad_s"][:, 1] = .5
        effort, e, _ = self.step()
        self.assertAlmostEqual(float(effort[0, 1]), -.015*.5, places=7)
        self.assertEqual(float(e["position_spring_after_contact_mode_mask_nm"][0, 1]), 0.)
        self.assertLess(float(effort[0, 1]*self.kw["measured_hand_velocity_rad_s"][0, 1]), 0.)

    def test_gravity_is_inside_total_cap_and_shared_scaling_preserves_force_ratios(self):
        self.kw["gravity_hand_nm"][:, 0] = -.05
        effort, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertAlmostEqual(float(effort[0, 0]), -.06, places=6)
        alpha = float(e["non_gravity_common_scale"][0])
        self.assertGreater(alpha, 0)
        self.assertLess(alpha, 1)
        self.assertAlmostEqual(float(effort[0, 3])/.012, alpha, places=6)
        self.assertAlmostEqual(float(effort[0, 5])/.012, alpha, places=6)
        self.kw["gravity_hand_nm"][:] = .2
        effort, e, _ = self.step()
        caps = self.t.tensor(self.parameters["joint_torque_limits_nm"])
        self.assertTrue((effort.abs() <= caps).all())
        self.assertTrue(e["gravity_saturated_by_joint"].all())

    def test_overforce_unclipped_negative_command_and_actual_opening_with_palm_lag(self):
        self.kw["raw_group_force_n"][:] = self.t.tensor([54., 2., 3.])
        self.kw["tracking"][:] = False
        self.kw["gravity_hand_nm"][:] = self.kw["closing_axis_hand7"][None]*.03
        self.kw["nominal_hand_rad"][:] = self.kw["closing_axis_hand7"][None]
        effort, e, unsafe = self.step()
        self.assertFalse(unsafe.any())  # Force-cap failure is recorded, never hidden as safe force.
        self.assertEqual(float(e["raw_group_normal_force_n"][0, 0]), 54.)
        self.assertTrue(e["raw_force_over_limit_by_group"].all())
        self.assertTrue((e["group_force_command_n"] < 0).all())
        self.assertTrue(e["gravity_removed_for_overforce_by_joint"].all())
        closing_work = effort*self.kw["closing_axis_hand7"][None]
        self.assertTrue((closing_work <= 0).all())
        self.assertTrue((closing_work[:, [0, 3, 4, 5, 6]] < 0).all())
        self.assertTrue(e["requested_torque_bounds_are_not_measured_force_bounds"])

    def test_no_current_normal_uses_axis_opening_for_overforce(self):
        self.kw["group_contact_valid"][:] = False
        self.kw["inward_normal_jacobian"][:] = 0
        self.kw["raw_group_force_n"][:] = 2.
        effort, e, _ = self.step()
        expected = -.02*self.kw["closing_axis_hand7"]
        self.assertTrue(self.t.allclose(effort, expected[None].expand_as(effort)))
        self.assertFalse(e["current_normal_available_by_group"].any())
        self.assertTrue(self.t.equal(e["normal_force_torque_nm"], self.t.zeros_like(effort)))

    def test_candidate_specific_closing_axes_control_only_missing_direction_emergency(self):
        # The new side-contact reset closes thumb1/2 and index1 in directions
        # different from the old geometry. No legacy signs may be substituted.
        axis = self.t.tensor([-1., -1., -1., 1., -1., 1., 1.])
        self.kw["closing_axis_hand7"] = axis
        self.kw["group_contact_valid"][:] = False
        self.kw["raw_group_force_n"][:] = 2.
        effort, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(self.t.allclose(effort, (-.02*axis)[None].expand_as(effort)))
        self.assertGreater(float(effort[0, 1]), 0)
        self.assertGreater(float(effort[0, 4]), 0)
        self.assertTrue((effort*axis[None] < 0).all())
        for invalid in ([-1., -1., -.5, 1., -1., 1., 1.],
                        [0., 0., 0., 1., -1., 1., 1.],
                        [-1., -1., -1., 0., 0., 1., 1.],
                        [-1., -1., -1., 1., -1., float("nan"), 1.]):
            self.kw["closing_axis_hand7"] = self.t.tensor(invalid)
            with self.assertRaises(ValueError):
                self.step()

    def test_speculative_current_point_can_control_but_never_becomes_physical_evidence(self):
        self.kw["group_physical_loaded"][:] = False
        self.kw["raw_group_force_n"][:] = 0
        effort, e, _ = self.step()
        self.assertTrue(e["current_normal_available_by_group"].all())
        self.assertFalse(e["physical_loaded_by_group_passthrough"].any())
        self.assertAlmostEqual(float(e["group_force_command_n"][0, 0]), 1.2, places=6)
        self.assertLess(float(effort[0, 0]), 0)
        self.assertTrue(e["control_normal_does_not_qualify_physical_support"])
        self.assertNotIn("support_qualified", e)

    def test_loss_of_current_normal_discards_old_direction_and_reacquires_slowly(self):
        self.step()
        self.kw["group_contact_valid"][:] = False
        self.kw["inward_normal_jacobian"][:] = 0
        self.kw["raw_group_force_n"][:] = 0
        self.kw["group_physical_loaded"][:] = False
        self.kw["nominal_hand_rad"][:] = self.kw["closing_axis_hand7"][None]*.5
        first, e, _ = self.step()
        self.assertTrue(e["lost_current_normal_by_group"].all())
        self.assertTrue(e["contact_mode_latched_by_group"].all())
        self.assertTrue(e["reacquisition_active_by_group"].all())
        self.assertTrue(self.t.equal(e["normal_force_torque_nm"], self.t.zeros_like(first)))
        self.assertLess(float(first.abs().max()), .0001)
        previous_ref = e["reference_hand_rad"].clone()
        for _ in range(30):
            _, e, _ = self.step()
            delta = (e["reference_hand_rad"]-previous_ref).abs()
            self.assertLessEqual(float(delta.max()), .04*.002+1e-8)
            previous_ref = e["reference_hand_rad"].clone()
        self.assertLess(float(previous_ref.abs().max()), .003)
        self.assertFalse(e["current_normal_available_by_group"].any())

    def test_zero_force_with_current_normal_keeps_feedback_and_velocity_damping(self):
        self.step()
        self.kw["raw_group_force_n"][:] = 0
        self.kw["measured_hand_velocity_rad_s"][:, 0] = -2.
        effort, e, _ = self.step()
        self.assertTrue(e["contact_mode_latched_by_group"].all())
        self.assertTrue(e["current_normal_available_by_group"].all())
        self.assertGreater(float(e["group_force_command_n"][0, 0]), 0)
        self.assertLess(float(e["group_force_command_n"][0, 0]), 1.2)
        self.assertGreater(float(e["inward_velocity_m_s"][0, 0]), 0)
        self.assertLess(float(effort[0, 0]), 0)

    def test_lost_thumb_normal_resets_all_group_references_and_reacquires_orientation_slowly(self):
        self.kw["nominal_hand_rad"][:, 1:3] = self.t.tensor([.8, -.8])
        self.kw["nominal_velocity_limit_hand_rad_s"][:] = 1000.
        for _ in range(3):
            _, e, _ = self.step()
        self.assertTrue((e["reference_hand_rad"][:, 1:3].abs() > .5).all())
        self.kw["measured_hand_rad"][:, 1:3] = self.t.tensor([.08, -.08])
        self.kw["group_contact_valid"][:, 0] = False
        self.kw["raw_group_force_n"][:, 0] = 0
        effort, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(e["reacquisition_active_by_group"][:, 0].all())
        delta = (e["reference_hand_rad"][:, 1:3]-self.kw["measured_hand_rad"][:, 1:3]).abs()
        self.assertTrue((delta <= .04*.002+1e-8).all())
        self.assertTrue((effort[:, 1:3].abs() <= .15*.04*.002+1e-8).all())
        self.assertTrue(self.t.equal(e["normal_force_torque_nm"][:, :3], self.t.zeros(2, 3)))

    def test_zero_force_current_point_then_point_loss_never_returns_to_fast_free_chasing(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        self.controller = BalancedEffortGraspController(self.t, self.seed, self.ids)
        self.sample = 0
        self.kw["raw_group_force_n"][:] = 0
        self.kw["group_physical_loaded"][:] = False
        self.step()
        _, e, _ = self.step()
        self.assertTrue(e["current_normal_available_by_group"].all())
        self.assertTrue(e["contact_mode_latched_by_group"].all())
        self.kw["group_contact_valid"][:] = False
        self.kw["inward_normal_jacobian"][:] = 0
        self.kw["nominal_hand_rad"][:] = self.kw["closing_axis_hand7"][None]*.5
        _, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(e["reacquisition_active_by_group"].all())
        self.assertLessEqual(float(e["reference_hand_rad"].abs().max()), .04*.002+1e-8)

    def test_free_approach_has_bounded_reference_and_no_position_target_output(self):
        self.kw["group_contact_valid"][:] = False
        self.kw["raw_group_force_n"][:] = 0
        self.kw["nominal_hand_rad"][:] = self.kw["closing_axis_hand7"][None]*.5
        self.kw["geometry_lower_hand_rad"][:, 0] = -.01
        self.kw["geometry_upper_hand_rad"][:, 3:] = .01
        previous_ref = self.controller.reference.clone()
        for _ in range(20):
            effort, e, unsafe = self.step()
            self.assertFalse(unsafe.any())
            delta = (e["reference_hand_rad"]-previous_ref).abs()
            self.assertTrue((delta <= .9*.002+1e-7).all())
            previous_ref = e["reference_hand_rad"].clone()
        self.assertAlmostEqual(float(previous_ref[0, 0]), -.01, places=6)
        self.assertTrue((previous_ref[:, 3:] <= .01).all())
        self.assertGreater(float(effort[0, 3]), 0)
        self.assertNotIn("next_hand_target_rad", e)

    def test_actual_geometry_excursion_is_reported_separately_from_bounded_reference(self):
        self.kw["geometry_lower_hand_rad"][:, 0] = -.01
        self.kw["geometry_upper_hand_rad"][:, 1] = 0.
        self.kw["measured_hand_rad"][:, 0] = -.012
        self.kw["measured_hand_rad"][:, 1] = .004
        _, e, unsafe = self.step()
        # Geometry is a reference envelope. Native mechanical tolerance and
        # actual contact/height gates are independent environment checks.
        self.assertFalse(unsafe.any())
        self.assertAlmostEqual(float(e["actual_geometry_lower_violation_rad"][0, 0]), .002, places=7)
        self.assertAlmostEqual(float(e["actual_geometry_upper_violation_rad"][0, 1]), .004, places=7)
        self.assertTrue(e["geometry_and_velocity_boxes_constrain_reference_not_actual_motion"])
        self.assertTrue((e["reference_hand_rad"] >= self.kw["geometry_lower_hand_rad"]).all())
        self.assertTrue((e["reference_hand_rad"] <= self.kw["geometry_upper_hand_rad"]).all())

    def test_first_valid_delay_then_lost_measurement_and_nonfinite_fail_closed(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        self.controller = BalancedEffortGraspController(self.t, self.seed, self.ids)
        self.sample = 0
        self.kw["contact_measurement_valid"][:] = False
        self.kw["gravity_hand_nm"][:] = .005
        self.kw["nominal_hand_rad"][:] = .5
        for _ in range(5):
            effort, e, unsafe = self.step()
            self.assertFalse(unsafe.any())
            self.assertTrue(e["waiting_for_first_valid_measurement"].all())
            self.assertTrue(self.t.equal(e["reference_hand_rad"], self.seed))
            self.assertTrue(self.t.allclose(effort, self.t.full_like(effort, .005)))
        self.kw["contact_measurement_valid"][:] = True
        _, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(e["measurement_fresh_for_effort"].all())
        self.kw["contact_measurement_valid"][0] = False
        self.kw["measured_hand_velocity_rad_s"][1, 2] = float("nan")
        effort, e, unsafe = self.step()
        self.assertTrue(unsafe.all())
        self.assertTrue(self.t.equal(effort, self.t.zeros_like(effort)))
        self.assertTrue(bool(e["measurement_lost_after_valid"][0]))

    def test_duplicate_gap_and_reset_do_not_reuse_old_force_or_integrate_twice(self):
        effort, e, _ = self.step()
        previous_ref = self.controller.reference.clone()
        duplicate, e, unsafe = self.controller.update(sample_id=self.sample-1, **self.kw)
        self.assertTrue(e["sample_duplicate_hold"])
        self.assertTrue(self.t.equal(duplicate, effort))
        self.assertTrue(self.t.equal(self.controller.reference, previous_ref))
        self.assertFalse(unsafe.any())
        self.kw["episode_id"] = self.t.ones_like(self.ids)
        self.kw["contact_measurement_valid"][:] = False
        self.kw["raw_group_force_n"][:] = 54.
        reset, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(e["episode_reset_hold"].all())
        self.assertFalse(e["contact_mode_latched_by_group"].any())
        self.assertTrue(self.t.equal(reset, self.t.zeros_like(reset)))
        self.sample += 1
        effort, e, unsafe = self.step()
        self.assertTrue(unsafe.all())
        self.assertTrue(e["sample_discontinuity"])
        self.assertTrue(self.t.equal(effort, self.t.zeros_like(effort)))

    def test_invalid_jacobian_depth_or_bound_cannot_emit_effort(self):
        self.kw["inward_normal_jacobian"][0, 0, 3] = .01
        self.kw["penetration_m"][1] = .0031
        effort, e, unsafe = self.step()
        self.assertTrue(unsafe.all())
        self.assertTrue(self.t.equal(effort, self.t.zeros_like(effort)))
        self.assertTrue(bool(e["cross_group_jacobian_rejected"][0]))
        self.kw["inward_normal_jacobian"][0, 0, 3] = 0
        self.kw["penetration_m"][:] = 0
        self.kw["geometry_lower_hand_rad"][:] = 3.
        effort, e, unsafe = self.step()
        self.assertTrue(unsafe.all())
        self.assertTrue(e["invalid_reference_bounds"].all())
        self.assertTrue(self.t.equal(effort, self.t.zeros_like(effort)))

    def test_duplicate_cannot_sneak_a_new_episode_into_cached_force_state(self):
        self.step()
        self.kw["episode_id"] = self.t.tensor([1, 0])
        result, e, unsafe = self.controller.update(sample_id=self.sample-1, **self.kw)
        self.assertTrue(bool(unsafe[0]))
        self.assertFalse(bool(unsafe[1]))
        self.assertTrue(bool(e["duplicate_sample_episode_conflict"][0]))
        self.assertEqual(int(self.controller.ids[0]), 0)
        self.assertTrue(self.t.equal(result[0], self.t.zeros(7)))
        self.kw["contact_measurement_valid"][0] = False
        result, e, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(bool(e["episode_reset_hold"][0]))
        self.assertFalse(bool(e["contact_mode_latched_by_group"][0].any()))
        self.assertTrue(self.t.equal(result[0], self.t.zeros(7)))

    def test_parameters_are_explicitly_declared_and_native_gate_cannot_be_relaxed(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        parameters = copy.deepcopy(self.parameters)
        parameters["joint_stiffness_nm_rad"] = .2
        changed = BalancedEffortGraspController(self.t, self.seed, self.ids, parameters=parameters)
        self.assertNotEqual(changed.contract_sha256, self.controller.contract_sha256)
        parameters["group_force_limit_n"] = 2.
        with self.assertRaises(ValueError):
            BalancedEffortGraspController(self.t, self.seed, self.ids, parameters=parameters)
        parameters = copy.deepcopy(self.parameters)
        parameters["normal_force_feedback_gain"] = True
        with self.assertRaises(ValueError):
            BalancedEffortGraspController(self.t, self.seed, self.ids, parameters=parameters)


if __name__ == "__main__":
    unittest.main()
