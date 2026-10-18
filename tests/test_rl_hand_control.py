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

    def test_explicit_zero_support_is_bitwise_identical_to_omitted_input(self):
        omitted = self.controller.__class__(self.t, self.seed, self.ids)
        explicit = self.controller.__class__(self.t, self.seed, self.ids)
        omitted.update(sample_id=0, **self.kw)
        explicit.update(sample_id=0, **self.kw)
        for sample in range(1, 6):
            inputs = {k: v.clone() if self.t.is_tensor(v) else v for k, v in self.kw.items()}
            inputs["gravity_hand_nm"][:] = .003*sample
            inputs["measured_hand_velocity_rad_s"][:] = .01*sample
            if sample == 2:
                inputs["group_contact_valid"][:] = False
            if sample == 3:
                inputs["raw_group_force_n"][:, 1] = 3.
            if sample == 4:
                inputs["tracking"][:] = False
            if sample == 5:
                inputs["unsafe_latched"][0] = True
            left = omitted.update(sample_id=sample, **inputs)
            right = explicit.update(sample_id=sample, object_support_torque_nm=self.t.zeros(2, 7), **inputs)
            self.assertTrue(self.t.equal(left[0].view(self.t.int32), right[0].view(self.t.int32)))
            self.assertTrue(self.t.equal(left[2], right[2]))
            for key in left[1]:
                if self.t.is_tensor(left[1][key]):
                    self.assertTrue(self.t.equal(left[1][key], right[1][key]), key)

    def test_object_weight_support_reaches_normal_nullspace_without_position_spring(self):
        # Thumb joint1 has no normal leverage in this fixture. Object support
        # is an explicit world-load torque, so normal projection must not erase it.
        baseline, _, _ = self.step()
        support = self.t.zeros(2, 7)
        support[:, 1] = .01
        original = support.clone()
        self.kw["object_support_torque_nm"] = support
        effort, evidence, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(self.t.allclose(effort-baseline, support, atol=1e-8, rtol=0))
        self.assertTrue(self.t.equal(evidence["position_spring_after_contact_mode_mask_nm"], self.t.zeros_like(support)))
        for key in ("object_support_torque_input_nm", "object_support_torque_requested_nm",
                    "object_support_torque_after_safety_mask_nm", "object_support_torque_after_common_scale_nm"):
            self.assertTrue(self.t.equal(evidence[key], support), key)
        self.assertTrue(self.t.equal(original, support))

    def test_support_and_normal_share_remaining_torque_budget_after_gravity(self):
        support = self.t.zeros(2, 7)
        support[:, 0] = -.01
        support[:, 1] = .2
        self.kw["object_support_torque_nm"] = support
        self.kw["gravity_hand_nm"][:, 0] = -.05
        effort, evidence, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(self.t.allclose(evidence["non_gravity_common_scale"], self.t.full((2,), .2), atol=1e-7, rtol=0))
        self.assertAlmostEqual(float(effort[0, 0]), -.05+.2*(-.032-.01), places=7)
        self.assertAlmostEqual(float(effort[0, 1]), .04, places=7)
        self.assertAlmostEqual(float(effort[0, 3]), .2*.012, places=7)
        self.assertTrue(self.t.allclose(evidence["object_support_torque_after_common_scale_nm"], .2*support, atol=1e-8, rtol=0))
        caps = self.t.tensor(self.parameters["joint_torque_limits_nm"])
        self.assertTrue((effort.abs() <= caps).all())
        self.assertTrue(evidence["object_support_shares_non_gravity_alpha_and_total_caps"])

    def test_group_overforce_removes_entire_finger_support_before_common_scale(self):
        self.kw["raw_group_force_n"][0, 0] = 2.
        self.kw["raw_group_force_n"][1, 1] = 2.
        baseline = self.controller.__class__(self.t, self.seed, self.ids)
        baseline.update(sample_id=0, **self.kw)
        old_effort, _, _ = baseline.update(sample_id=self.sample, **self.kw)
        self.kw["object_support_torque_nm"] = self.t.full((2, 7), .005)
        effort, evidence, unsafe = self.step()
        self.assertFalse(unsafe.any())
        masked = evidence["object_support_torque_after_safety_mask_nm"]
        self.assertTrue(self.t.equal(masked[0, :3], self.t.zeros(3)))
        self.assertTrue(self.t.equal(masked[1, 3:5], self.t.zeros(2)))
        self.assertTrue(self.t.equal(effort[0, :3], old_effort[0, :3]))
        self.assertTrue(self.t.equal(effort[1, 3:5], old_effort[1, 3:5]))
        self.assertTrue((masked[0, 3:] > 0).all())
        self.assertTrue((masked[1, [0, 1, 2, 5, 6]] > 0).all())

    def test_nonfinite_support_fails_closed_per_environment(self):
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(invalid=invalid):
                support = self.t.full((2, 7), .005)
                support[0, 2] = invalid
                self.kw["object_support_torque_nm"] = support
                effort, evidence, unsafe = self.step()
                self.assertTrue(bool(unsafe[0]))
                self.assertFalse(bool(unsafe[1]))
                self.assertTrue(self.t.isfinite(effort).all())
                self.assertTrue(self.t.equal(effort[0], self.t.zeros(7)))
                self.assertTrue(self.t.equal(evidence["object_support_torque_after_common_scale_nm"][0], self.t.zeros(7)))

    def test_support_shape_type_and_dtype_are_explicit(self):
        for invalid in ([0.]*7, self.t.zeros(7), self.t.zeros(2, 6),
                        self.t.zeros(2, 7, dtype=self.t.bool),
                        self.t.zeros(2, 7, dtype=self.t.float64)):
            with self.subTest(type=type(invalid), shape=getattr(invalid, "shape", None)):
                self.kw["object_support_torque_nm"] = invalid
                with self.assertRaises(ValueError):
                    self.step()

    def test_support_cannot_bypass_bootstrap_duplicate_or_unsafe_hold(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        self.controller = BalancedEffortGraspController(self.t, self.seed, self.ids)
        self.sample = 0
        self.kw["object_support_torque_nm"] = self.t.full((2, 7), .01)
        first, evidence, unsafe = self.step()
        self.assertFalse(unsafe.any())
        self.assertTrue(self.t.equal(first, self.t.zeros_like(first)))
        self.assertFalse(evidence["object_support_torque_after_safety_mask_nm"].any())
        active, _, _ = self.step()
        duplicate, evidence, _ = self.controller.update(sample_id=1, **self.kw)
        self.assertTrue(self.t.equal(active, duplicate))
        self.assertTrue(evidence["sample_duplicate_hold"])
        self.assertFalse(evidence["object_support_torque_after_safety_mask_nm"].any())
        self.kw["unsafe_latched"][:] = True
        effort, evidence, unsafe = self.step()
        self.assertTrue(unsafe.all())
        self.assertTrue(self.t.equal(effort, self.t.zeros_like(effort)))
        self.assertFalse(evidence["object_support_torque_after_common_scale_nm"].any())

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



@unittest.skipUnless(TORCH, "real torch is required for current-contact domain guard math")
class ContactDomainPostureGuardTests(unittest.TestCase):
    DECLARATION = {
        "version": "g1_contact_domain_nullspace_posture_guard_v1",
        "hand_joint_index": 2, "activation_margin_rad": .006,
        "restoring_stiffness_nm_rad": 2., "maximum_guard_joint_torque_nm": .004,
        "minimum_nullspace_joint_component": .05, "svd_tolerance_m": .0001,
        "original_actual_geometry_tolerance_rad": .025,
        "protect_current_normal_and_world_z": True,
        "requires_all_groups_physically_loaded": True,
        "requires_original_contact_safe": True,
        "all_original_acceptance_gates_unchanged": True,
    }

    def setUp(self):
        BalancedEffortGraspTests.setUp(self)
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        self.kw["inward_normal_jacobian"][:, 0, :3] = self.t.tensor(
            [-.022998522967100143, -.056815437972545624, -.02566390484571457])
        self.point_jacobian = self.t.zeros(2, 3, 3, 7)
        self.point_jacobian[:, 0, 2, :3] = self.t.tensor(
            [-.01418279018253088, .0018831974593922496, -.027741195634007454])
        self.lower = self.t.full((2, 7), -.2)
        self.upper = self.t.full((2, 7), .2)
        self.kw.update(contact_point_jacobian_world_m=self.point_jacobian,
            actual_domain_lower_hand_rad=self.lower, actual_domain_upper_hand_rad=self.upper,
            physical_contact_safe=self.t.ones(2, dtype=self.t.bool))
        self.controller = BalancedEffortGraspController(self.t, self.seed, self.ids,
                                                       domain_posture_guard=self.DECLARATION)
        self.sample = 0
        self.step()
        self.kw["measured_hand_rad"][:, 2] = -.198

    step = BalancedEffortGraspTests.step

    def guard(self):
        effort, evidence, unsafe = self.step()
        return effort, evidence["contact_domain_posture_guard"], evidence, unsafe

    def test_real_crossing_geometry_retains_thumb2_authority_and_two_constraints(self):
        _, guard, evidence, unsafe = self.guard()
        self.assertFalse(unsafe.any())
        self.assertTrue(guard["gate"].all())
        self.assertTrue((guard["retained_rank"] == 2).all())
        self.assertTrue(self.t.allclose(guard["nullspace_thumb2_component"],
                                       self.t.full((2,), .209899066), atol=1e-6, rtol=0))
        self.assertTrue((guard["guard_torque_before_total_alpha_nm"][:, 2] > 0).all())
        self.assertTrue(self.t.equal(guard["guard_torque_before_total_alpha_nm"][:, 3:], self.t.zeros(2, 4)))
        self.assertLess(float(guard["normal_projection_residual_m_nm"].abs().max()), 1e-9)
        self.assertLess(float(guard["world_z_projection_residual_m_nm"].abs().max()), 1e-9)
        self.assertTrue(self.t.equal(evidence["position_spring_after_contact_mode_mask_nm"], self.t.zeros(2, 7)))
        self.assertTrue(guard["geometric_projection_is_not_dynamic_force_isolation"])

    def test_upper_boundary_is_restored_inward_and_interior_has_no_spring(self):
        self.kw["measured_hand_rad"][0, 2] = .198
        self.kw["measured_hand_rad"][1, 2] = 0.
        _, guard, _, _ = self.guard()
        self.assertLess(float(guard["guard_torque_before_total_alpha_nm"][0, 2]), 0.)
        self.assertTrue(self.t.equal(guard["guard_torque_before_total_alpha_nm"][1], self.t.zeros(7)))
        self.assertAlmostEqual(float(guard["interior_lower_thumb2_rad"][0]), -.194, places=6)
        self.assertAlmostEqual(float(guard["interior_upper_thumb2_rad"][0]), .194, places=6)

    def test_pre_cap_total_telemetry_includes_the_optional_guard_component(self):
        _, guard, evidence, _ = self.guard()
        expected = (evidence["gravity_input_nm"]+evidence["joint_impedance_before_normal_projection_nm"]+
            evidence["normal_force_torque_nm"]+evidence["opening_fallback_torque_nm"]+
            guard["guard_torque_before_total_alpha_nm"])
        self.assertTrue(self.t.allclose(evidence["unprojected_total_torque_nm"], expected, atol=1e-8, rtol=0))
        self.assertTrue(guard["unprojected_total_diagnostic_includes_projected_guard"])
        self.assertTrue(self.t.equal(self.lower, self.t.full((2, 7), -.2)))
        self.assertTrue(self.t.equal(self.upper, self.t.full((2, 7), .2)))

    def test_guard_common_cap_preserves_normal_and_z_before_existing_total_caps(self):
        self.kw["measured_hand_rad"][:, 2] = -.8
        self.kw["gravity_hand_nm"][:, 1] = -.039
        effort, guard, evidence, _ = self.guard()
        self.assertTrue((guard["guard_common_scale"] < 1).all())
        self.assertTrue((guard["guard_torque_before_total_alpha_nm"].abs().amax(-1) <= .004000001).all())
        self.assertLess(float(guard["normal_projection_residual_m_nm"].abs().max()), 1e-9)
        self.assertLess(float(guard["world_z_projection_residual_m_nm"].abs().max()), 1e-9)
        self.assertTrue((evidence["non_gravity_common_scale"] < 1).all())
        self.assertTrue(self.t.allclose(guard["guard_torque_after_total_common_scale_nm"],
            evidence["non_gravity_common_scale"][:, None]*guard["guard_torque_before_total_alpha_nm"]))
        caps = self.t.tensor(self.parameters["joint_torque_limits_nm"])
        self.assertTrue((effort.abs() <= caps+1e-7).all())

    def test_every_current_physical_gate_is_required_and_overforce_disables_whole_guard(self):
        cases = (("physical_contact_safe", (0,), False), ("tracking", (0,), False),
                 ("group_contact_valid", (0, 2), False), ("group_physical_loaded", (0, 1), False),
                 ("raw_group_force_n", (0, 1), 1.6), ("unsafe_latched", (0,), True))
        for key, index, value in cases:
            old = self.kw[key].clone()
            self.kw[key][index] = value
            _, guard, _, _ = self.guard()
            with self.subTest(key=key):
                self.assertFalse(bool(guard["gate"][0]))
                self.assertTrue(self.t.equal(guard["guard_torque_before_total_alpha_nm"][0], self.t.zeros(7)))
            self.kw[key] = old

    def test_rank_loss_and_eliminated_thumb2_direction_disable_guard(self):
        self.kw["contact_point_jacobian_world_m"][:, 0, 2, :3] = self.kw["inward_normal_jacobian"][:, 0, :3]
        _, guard, _, _ = self.guard()
        self.assertFalse(guard["rank_supported"].any())
        self.assertFalse(guard["gate"].any())
        self.kw["inward_normal_jacobian"][:, 0, :3] = self.t.tensor([0., 0., .03])
        self.kw["contact_point_jacobian_world_m"][:, 0, 2, :3] = self.t.tensor([.03, 0., 0.])
        _, guard, _, _ = self.guard()
        self.assertTrue(guard["rank_supported"].all())
        self.assertFalse(guard["thumb2_component_supported"].any())
        self.assertTrue(self.t.equal(guard["guard_torque_before_total_alpha_nm"], self.t.zeros(2, 7)))

    def test_missing_or_nonfinite_current_geometry_never_reuses_previous_guard(self):
        _, guard, _, _ = self.guard()
        self.assertTrue(guard["gate"].all())
        self.kw["contact_point_jacobian_world_m"] = None
        _, guard, _, _ = self.guard()
        self.assertFalse(guard["inputs_supplied"])
        self.assertIn("contact_point_jacobian_world_m", guard["missing_inputs"])
        self.assertTrue(self.t.equal(guard["guard_torque_before_total_alpha_nm"], self.t.zeros(2, 7)))
        self.kw["contact_point_jacobian_world_m"] = self.point_jacobian.clone()
        self.kw["contact_point_jacobian_world_m"][0, 0, 2, 0] = float("nan")
        _, guard, _, _ = self.guard()
        self.assertFalse(bool(guard["current_input_finite"][0]))
        self.assertTrue(self.t.isfinite(guard["guard_torque_before_total_alpha_nm"]).all())
        self.assertFalse(bool(guard["gate"][0]))

    def test_reset_and_duplicate_never_apply_a_new_guard_from_history(self):
        self.guard()
        # A duplicate holds the previous complete effort, not a new addition.
        effort, evidence, _ = self.controller.update(sample_id=self.sample-1, **self.kw)
        self.assertTrue(evidence["contact_domain_posture_guard"]["duplicate_holds_prior_complete_effort_without_new_guard_evaluation"])
        self.assertTrue(self.t.equal(evidence["contact_domain_posture_guard"]["guard_component_for_current_new_command_nm"], self.t.zeros(2, 7)))
        self.ids[0] += 1
        self.kw["measured_hand_rad"][0] = self.seed[0]
        _, guard, _, _ = self.guard()
        self.assertFalse(bool(guard["safe_fresh_sample"][0]))
        self.assertFalse(bool(guard["gate"][0]))
        self.kw["contact_point_jacobian_world_m"] = None
        _, guard, _, _ = self.guard()
        self.assertFalse(guard["gate"].any())

    def test_explicit_none_preserves_original_identity_and_all_effort_bits(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        from g1_vla.rl_effort_protocol import balanced_effort_controller_contract
        omitted = BalancedEffortGraspController(self.t, self.seed, self.ids)
        explicit = BalancedEffortGraspController(self.t, self.seed, self.ids, domain_posture_guard=None)
        self.assertEqual(omitted.contract, balanced_effort_controller_contract())
        self.assertEqual(omitted.contract_sha256, explicit.contract_sha256)
        for sample in range(4):
            left = omitted.update(sample_id=sample, **self.kw)
            right = explicit.update(sample_id=sample, **self.kw)
            self.assertTrue(self.t.equal(left[0].view(self.t.int32), right[0].view(self.t.int32)))
            self.assertNotIn("contact_domain_posture_guard", left[1])
            self.assertNotIn("contact_domain_posture_guard", right[1])
        self.assertIn("+g1_contact_domain_nullspace_posture_guard_v1", self.controller.contract["version"])
        self.assertNotEqual(self.controller.contract_sha256, omitted.contract_sha256)

    def test_invalid_new_tensor_shapes_types_and_declarations_are_rejected(self):
        self.kw["contact_point_jacobian_world_m"] = self.t.zeros(2, 3, 3, 6)
        with self.assertRaises(ValueError):
            self.guard()
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        invalid = dict(self.DECLARATION, activation_margin_rad=.007)
        with self.assertRaises(ValueError):
            BalancedEffortGraspController(self.t, self.seed, self.ids, domain_posture_guard=invalid)


@unittest.skipUnless(TORCH, "real torch is required for coordinated native-contact control math")
class ContactTransitionControlTests(unittest.TestCase):
    step = BalancedEffortGraspTests.step

    def setUp(self):
        BalancedEffortGraspTests.setUp(self)
        from g1_vla.rl_effort_protocol import contact_transition_control_contract
        self.declaration = contact_transition_control_contract()
        self.kw["physical_contact_safe"] = self.t.ones(2, dtype=self.t.bool)
        self.kw["actual_domain_lower_hand_rad"] = self.t.full((2, 7), -.2)
        self.kw["actual_domain_upper_hand_rad"] = self.t.full((2, 7), .2)
        self.kw["contact_point_jacobian_world_m"] = self.t.zeros(2, 3, 3, 7)
        self.reset_controller()

    def reset_controller(self, *, guard=None):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        self.controller = BalancedEffortGraspController(self.t, self.seed, self.ids,
            domain_posture_guard=guard, contact_transition=self.declaration)
        self.sample = 0
        self.step()

    def test_stage20_current_index_unload_retains_both_negative_axes(self):
        # Exact Stage20 failure geometry: both J entries are positive, but the
        # candidate's index1 closing axis is negative. The old guard erased it.
        axis = self.t.tensor([-1., -1., -1., 1., -1., 1., 1.])
        self.kw["closing_axis_hand7"] = axis
        self.kw["inward_normal_jacobian"][:, 1, 3:5] = self.t.tensor([.060865, .01621])
        self.kw["raw_group_force_n"][:, 1] = 2.096934557
        self.kw["gravity_hand_nm"][:] = .03
        self.kw["measured_hand_velocity_rad_s"][:] = .2
        self.kw["nominal_hand_rad"][:] = .5
        self.kw["object_support_torque_nm"] = self.t.full((2, 7), .02)
        old = self.controller.__class__(self.t, self.seed, self.ids)
        old.update(sample_id=0, **self.kw)
        old_effort, _, _ = old.update(sample_id=1, **self.kw)
        effort, evidence, unsafe = self.step()
        self.assertFalse(unsafe.any())
        transition = evidence["contact_transition"]
        alpha = evidence["non_gravity_common_scale"]
        expected = -.4*self.kw["inward_normal_jacobian"][:, 1, 3:5]*alpha[:, None]
        self.assertTrue(self.t.allclose(effort[:, 3:5], expected, atol=1e-8, rtol=0))
        self.assertEqual(float(old_effort[0, 4]), 0.)
        self.assertLess(float(effort[0, 4]), 0.)
        self.assertTrue(transition["fixed_closing_axis_mask_bypassed_by_current_normal_joint"][:, 4].all())
        for key in ("gravity_command_nm", "posture_torque_after_normal_projection_nm",
                    "object_support_torque_after_safety_mask_nm"):
            self.assertTrue(self.t.equal(evidence[key][:, 3:5], self.t.zeros(2, 2)), key)

    def test_mixed_current_jacobian_signs_never_command_inward_overforce_component(self):
        generator = self.t.Generator().manual_seed(739)
        self.kw["raw_group_force_n"][:] = 2.
        self.kw["closing_axis_hand7"] = self.t.tensor([-1., -1., -1., 1., -1., 1., 1.])
        self.kw["object_support_torque_nm"] = self.t.full((2, 7), .03)
        self.kw["gravity_hand_nm"][:] = -.03
        for _ in range(25):
            a = self.t.zeros(2, 3, 7)
            a[:, 0, :3] = self.t.randn(2, 3, generator=generator)*.15
            a[:, 1, 3:5] = self.t.randn(2, 2, generator=generator)*.15
            a[:, 2, 5:] = self.t.randn(2, 2, generator=generator)*.15
            self.kw["inward_normal_jacobian"] = a
            effort, evidence, unsafe = self.step()
            self.assertFalse(unsafe.any())
            alpha = evidence["non_gravity_common_scale"]
            self.assertTrue(self.t.allclose(effort, alpha[:, None]*(-.4*a.sum(1)), atol=1e-8, rtol=0))
            pairing = a*effort[:, None, :]
            self.assertTrue((pairing <= 0).all())
            self.assertTrue((pairing.sum(-1) < 0).all())
            self.assertTrue(evidence["force_relief_by_group"].all())
            self.assertTrue((effort.abs() <= self.t.tensor(self.parameters["joint_torque_limits_nm"])+1e-8).all())

    def test_missing_and_degenerate_current_normal_keep_original_axis_fallback(self):
        self.kw["closing_axis_hand7"] = self.t.tensor([-1., -1., -1., 1., -1., 1., 1.])
        self.kw["raw_group_force_n"][:] = 2.
        self.step()  # A formerly known row must not leak into missing-current control.
        self.kw["group_contact_valid"][:] = False
        self.kw["inward_normal_jacobian"] *= -1
        effort, evidence, _ = self.step()
        expected = (-.02*self.kw["closing_axis_hand7"])[None].expand_as(effort)
        self.assertTrue(self.t.equal(effort, expected))
        self.assertFalse(evidence["contact_transition"]["current_normal_unload_by_group"].any())
        self.assertTrue(evidence["contact_transition"]["missing_current_normal_axis_fallback_by_group"].all())
        self.kw["group_contact_valid"][:] = True
        self.kw["inward_normal_jacobian"] *= 0
        degenerate, evidence, _ = self.step()
        self.assertTrue(self.t.equal(degenerate, expected))
        self.assertTrue(evidence["current_normal_degenerate_by_group"].all())

    def test_first_partial_contact_discards_missing_finger_error_and_caps_native_reference(self):
        self.kw["group_contact_valid"][:] = False
        self.kw["group_physical_loaded"][:] = False
        self.kw["raw_group_force_n"][:] = 0
        self.kw["nominal_hand_rad"][:] = .5
        self.kw["nominal_velocity_limit_hand_rad_s"][:] = 2.
        for _ in range(10):
            _, evidence, _ = self.step()
        self.assertGreater(float(evidence["reference_hand_rad"][0, 3]), .03)
        self.kw["group_contact_valid"][:, 0] = True
        self.kw["group_physical_loaded"][:, 0] = True
        self.kw["raw_group_force_n"][:, 0] = .2
        self.kw["measured_hand_rad"][:, 3:] = .01
        _, evidence, _ = self.step()
        transition = evidence["contact_transition"]
        self.assertTrue(transition["first_any_contact_resets_missing_reference"].all())
        self.assertTrue(transition["reference_reset_to_current_measured_joint"][:, 3:].all())
        self.assertTrue(self.t.allclose(evidence["reference_hand_rad"][:, 3:], self.t.full((2, 4), .01008), atol=1e-8, rtol=0))
        for _ in range(15):
            previous = evidence["reference_hand_rad"].clone()
            _, evidence, _ = self.step()
            delta = (evidence["reference_hand_rad"]-previous).abs()
            self.assertTrue((delta[:, 3:] <= .04*.002+1e-8).all())
        self.assertTrue((evidence["position_spring_after_contact_mode_mask_nm"][:, :3] == 0).all())

    def test_preload_waits_25_complete_native_samples_then_ramps_100ms(self):
        preload = self.t.tensor([.2, .1, .1])[None].expand(2, -1)
        full = self.t.tensor([.8, .4, .4])[None].expand(2, -1)
        self.kw["raw_group_force_n"][:] = preload
        for count in range(1, 26):
            _, evidence, _ = self.step()
            transition = evidence["contact_transition"]
            self.assertTrue((transition["full_contact_settle_count"] == count).all())
            self.assertTrue(self.t.equal(evidence["group_force_target_n"], preload))
        for ramp_step in range(1, 51):
            _, evidence, _ = self.step()
            expected = preload+(ramp_step/50)*(full-preload)
            self.assertTrue(self.t.allclose(evidence["group_force_target_n"], expected, atol=3e-7, rtol=0))
        self.assertTrue(self.t.allclose(evidence["group_force_target_n"], full, atol=3e-7, rtol=0))

    def test_partial_contact_reset_outside_reference_box_returns_by_native_slew_not_projection(self):
        # Both index joints are within the original actual domain but outside
        # the narrower reference box. A single-sample clamp would jump 10mrad.
        self.kw["geometry_lower_hand_rad"][:] = -.2
        self.kw["geometry_upper_hand_rad"][:] = .2
        self.kw["actual_domain_lower_hand_rad"][:] = -.225
        self.kw["actual_domain_upper_hand_rad"][:] = .225
        self.kw["group_contact_valid"][:, 1:] = False
        self.kw["group_physical_loaded"][:, 1:] = False
        self.kw["raw_group_force_n"][:] = self.t.tensor([.2, 0., 0.])
        self.kw["measured_hand_rad"][:, 3:5] = self.t.tensor([.21, -.21])
        self.kw["nominal_hand_rad"][:, 3:5] = self.t.tensor([-.5, .5])
        _, evidence, unsafe = self.step()
        self.assertFalse(unsafe.any())
        transition = evidence["contact_transition"]
        self.assertTrue(transition["first_any_contact_resets_missing_reference"].all())
        self.assertTrue(self.t.equal(transition["reference_integration_origin_hand_rad"][:, 3:5],
                                     self.kw["measured_hand_rad"][:, 3:5]))
        self.assertTrue(self.t.allclose(evidence["reference_hand_rad"][:, 3:5],
            self.t.tensor([[.20992, -.20992]]*2), atol=2e-8, rtol=0))
        self.assertTrue((transition["reference_integration_delta_hand_rad"][:, 3:5].abs() <= .00008002).all())
        self.assertTrue(transition["reference_box_excursion_did_not_increase_joint"].all())
        self.assertGreater(float(transition["temporary_reference_box_upper_excursion_rad"][0, 3]), .009)
        self.assertGreater(float(transition["temporary_reference_box_lower_excursion_rad"][0, 4]), .009)
        previous = evidence["reference_hand_rad"].clone()
        _, evidence, _ = self.step()
        self.assertTrue(((evidence["reference_hand_rad"]-previous)[:, 3:5].abs() <= .00008002).all())
        self.assertTrue(evidence["contact_transition"]["reference_box_excursion_did_not_increase_joint"].all())
        self.assertTrue(self.t.equal(self.kw["actual_domain_lower_hand_rad"], self.t.full((2, 7), -.225)))
        self.assertTrue(self.t.equal(self.kw["actual_domain_upper_hand_rad"], self.t.full((2, 7), .225)))

    def test_every_settle_condition_loss_immediately_resets_preload_ramp(self):
        preload = self.t.tensor([.2, .1, .1])[None].expand(2, -1)
        cases = (("group_physical_loaded", (0, 1), False), ("group_contact_valid", (0, 2), False),
                 ("physical_contact_safe", (0,), False), ("tracking", (0,), False),
                 ("raw_group_force_n", (0, 1), 1.6))
        for key, index, value in cases:
            self.reset_controller()
            for _ in range(35):
                self.step()
            original = self.kw[key].clone()
            self.kw[key][index] = value
            _, evidence, _ = self.step()
            transition = evidence["contact_transition"]
            with self.subTest(key=key):
                self.assertFalse(bool(transition["full_contact_settling_gate"][0]))
                self.assertEqual(int(transition["full_contact_settle_count"][0]), 0)
                self.assertEqual(float(transition["full_force_ramp_elapsed_s"][0]), 0.)
                self.assertTrue(self.t.equal(evidence["group_force_target_n"][0], preload[0]))
                self.assertGreater(float(transition["full_force_ramp_elapsed_s"][1]), 0.)
            self.kw[key] = original

    def test_free_thumb2_domain_recovery_has_no_projection_claim_and_shares_cap(self):
        self.kw["group_contact_valid"][:, 0] = False
        self.kw["group_physical_loaded"][:, 0] = False
        self.kw["raw_group_force_n"][:, 0] = 0
        self.kw["measured_hand_rad"][0, 2] = -.198
        self.kw["measured_hand_rad"][1, 2] = .198
        self.kw["gravity_hand_nm"][:, 2] = .0399
        effort, evidence, _ = self.step()
        recovery = evidence["contact_transition"]["thumb_domain_recovery"]
        torque = recovery["guard_torque_before_total_alpha_nm"]
        self.assertTrue(recovery["free_thumb2_gate"].all())
        self.assertFalse(recovery["partial_contact_projection_gate"].any())
        self.assertGreater(float(torque[0, 2]), 0)
        self.assertLess(float(torque[1, 2]), 0)
        self.assertTrue(self.t.equal(torque[:, [0, 1, 3, 4, 5, 6]], self.t.zeros(2, 6)))
        self.assertTrue((torque.abs() <= .004000001).all())
        self.assertLess(float(evidence["non_gravity_common_scale"][0]), 1.)
        self.assertTrue((effort.abs() <= self.t.tensor(self.parameters["joint_torque_limits_nm"])+1e-8).all())
        self.assertTrue(recovery["free_thumb2_has_no_nullspace_claim"])

    def partial_fixture(self):
        self.reset_controller(guard=ContactDomainPostureGuardTests.DECLARATION)
        self.kw["inward_normal_jacobian"][:, 0, :3] = self.t.tensor(
            [-.022998522967100143, -.056815437972545624, -.02566390484571457])
        self.kw["contact_point_jacobian_world_m"][:, 0, 2, :3] = self.t.tensor(
            [-.01418279018253088, .0018831974593922496, -.027741195634007454])
        self.kw["group_physical_loaded"][:, 1:] = False
        self.kw["group_contact_valid"][:, 1:] = False
        self.kw["raw_group_force_n"][:] = self.t.tensor([.2, 0., 0.])
        self.kw["physical_contact_safe"][:] = False
        self.kw["measured_hand_rad"][:, 2] = -.198

    def test_physically_loaded_thumb_without_current_normal_never_gets_free_boundary_torque(self):
        self.kw["group_contact_valid"][:, 0] = False
        self.kw["group_physical_loaded"][:, 0] = self.t.tensor([True, False])
        self.kw["raw_group_force_n"][:, 0] = self.t.tensor([.2, 0.])
        self.kw["measured_hand_rad"][:, 2] = -.198
        _, evidence, unsafe = self.step()
        recovery = evidence["contact_transition"]["thumb_domain_recovery"]
        self.assertFalse(unsafe.any())
        self.assertFalse(bool(recovery["thumb_current_normal"][0]))
        self.assertTrue(bool(recovery["thumb_physically_loaded"][0]))
        self.assertFalse(bool(recovery["free_thumb2_gate"][0]))
        self.assertFalse(bool(recovery["partial_contact_projection_gate"][0]))
        self.assertTrue(self.t.equal(recovery["guard_torque_before_total_alpha_nm"][0], self.t.zeros(7)))
        self.assertTrue(bool(recovery["free_thumb2_gate"][1]))
        self.assertGreater(float(recovery["guard_torque_before_total_alpha_nm"][1, 2]), 0.)

    def test_partial_loaded_thumb_uses_current_normal_and_z_nullspace_without_acceptance_claim(self):
        self.partial_fixture()
        _, evidence, unsafe = self.step()
        recovery = evidence["contact_transition"]["thumb_domain_recovery"]
        self.assertFalse(unsafe.any())
        self.assertTrue(recovery["partial_contact_projection_gate"].all())
        self.assertFalse(recovery["free_thumb2_gate"].any())
        self.assertFalse(evidence["contact_domain_posture_guard"]["gate"].any())
        self.assertLess(float(recovery["partial_normal_projection_residual_m_nm"].abs().max()), 1e-9)
        self.assertLess(float(recovery["partial_world_z_projection_residual_m_nm"].abs().max()), 1e-9)
        self.assertGreater(float(recovery["guard_torque_before_total_alpha_nm"][0, 2]), 0.)
        self.assertTrue(self.t.equal(evidence["position_spring_after_contact_mode_mask_nm"][:, :3], self.t.zeros(2, 3)))
        self.assertTrue(recovery["partial_contact_guard_is_not_physical_acceptance"])

    def test_partial_recovery_requires_current_point_rank_loaded_and_tracking(self):
        self.partial_fixture()
        cases = (("group_physical_loaded", (0, 0), False), ("tracking", (0,), False),
                 ("raw_group_force_n", (0, 0), 1.6), ("unsafe_latched", (0,), True))
        for key, index, value in cases:
            original = self.kw[key].clone()
            self.kw[key][index] = value
            _, evidence, _ = self.step()
            recovery = evidence["contact_transition"]["thumb_domain_recovery"]
            with self.subTest(key=key):
                self.assertFalse(bool(recovery["partial_contact_projection_gate"][0]))
                self.assertTrue(self.t.equal(recovery["guard_torque_before_total_alpha_nm"][0], self.t.zeros(7)))
            self.kw[key] = original
        self.kw["contact_point_jacobian_world_m"] = None
        _, evidence, _ = self.step()
        self.assertFalse(evidence["contact_transition"]["thumb_domain_recovery"]["partial_contact_projection_gate"].any())
        self.kw["contact_point_jacobian_world_m"] = self.t.zeros(2, 3, 3, 7)
        self.kw["contact_point_jacobian_world_m"][:, 0, 2, :3] = self.kw["inward_normal_jacobian"][:, 0, :3]
        _, evidence, _ = self.step()
        self.assertFalse(evidence["contact_transition"]["thumb_domain_recovery"]["partial_contact_projection_gate"].any())

    def test_full_original_guard_does_not_receive_a_second_thumb_recovery(self):
        self.partial_fixture()
        self.kw["group_contact_valid"][:] = True
        self.kw["group_physical_loaded"][:] = True
        self.kw["physical_contact_safe"][:] = True
        _, evidence, _ = self.step()
        self.assertTrue(evidence["contact_domain_posture_guard"]["gate"].all())
        recovery = evidence["contact_transition"]["thumb_domain_recovery"]
        self.assertFalse(recovery["free_thumb2_gate"].any())
        self.assertFalse(recovery["partial_contact_projection_gate"].any())
        self.assertTrue(self.t.equal(recovery["guard_torque_before_total_alpha_nm"], self.t.zeros(2, 7)))

    def test_duplicate_reset_and_unsafe_do_not_advance_settle_or_reuse_recovery(self):
        for _ in range(32):
            previous, evidence, _ = self.step()
        count = self.controller.preload_settle_count.clone()
        ramp = self.controller.preload_ramp_elapsed_s.clone()
        self.kw["inward_normal_jacobian"] *= -1
        duplicate, evidence, _ = self.controller.update(sample_id=self.sample-1, **self.kw)
        self.assertTrue(self.t.equal(duplicate, previous))
        self.assertTrue(self.t.equal(count, self.controller.preload_settle_count))
        self.assertTrue(self.t.equal(ramp, self.controller.preload_ramp_elapsed_s))
        self.assertTrue(evidence["contact_transition"]["duplicate_holds_prior_complete_effort_without_new_transition"])
        self.kw["episode_id"] = self.ids+1
        _, evidence, _ = self.step()
        self.assertTrue((self.controller.preload_settle_count == 0).all())
        self.assertTrue((self.controller.preload_ramp_elapsed_s == 0).all())
        self.assertFalse(self.controller.previous_any_contact.any())
        self.kw["unsafe_latched"][:] = True
        effort, evidence, unsafe = self.step()
        self.assertTrue(unsafe.all())
        self.assertTrue(self.t.equal(effort, self.t.zeros_like(effort)))
        self.assertTrue((evidence["contact_transition"]["full_contact_settle_count"] == 0).all())

    def test_none_preserves_original_contract_every_tensor_bit_and_reference_state(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        from g1_vla.rl_effort_protocol import balanced_effort_controller_contract
        omitted = BalancedEffortGraspController(self.t, self.seed, self.ids)
        explicit = BalancedEffortGraspController(self.t, self.seed, self.ids, contact_transition=None)
        self.assertEqual(omitted.contract, balanced_effort_controller_contract())
        self.assertEqual(omitted.contract_sha256, explicit.contract_sha256)
        for sample in range(9):
            if sample == 2:
                self.kw["raw_group_force_n"][:] = 2.
                self.kw["closing_axis_hand7"] = self.t.tensor([-1., -1., -1., 1., -1., 1., 1.])
            if sample == 3:
                self.kw["group_contact_valid"][:] = False
            if sample == 4:
                self.kw["measured_hand_rad"][:, 2] = -.198
            if sample == 5:
                self.kw["tracking"][:] = False
            if sample == 6:
                self.kw["unsafe_latched"][0] = True
            inputs = dict(self.kw)
            left = omitted.update(sample_id=sample, **inputs)
            right = explicit.update(sample_id=sample, **inputs)
            self.assertTrue(self.t.equal(left[0].view(self.t.int32), right[0].view(self.t.int32)))
            self.assertTrue(self.t.equal(left[2], right[2]))
            self.assertNotIn("contact_transition", left[1])
            self.assertNotIn("contact_transition", right[1])
            for key in left[1]:
                if self.t.is_tensor(left[1][key]):
                    a, b = left[1][key], right[1][key]
                    self.assertTrue(self.t.equal(a.view(self.t.int32), b.view(self.t.int32)) if a.dtype == self.t.float32 else self.t.equal(a, b), key)

    def test_changed_transition_declaration_is_rejected_and_identity_is_versioned(self):
        from g1_vla.rl_hand_control import BalancedEffortGraspController
        for key, value in (("full_contact_settle_samples", 24), ("full_force_ramp_s", .05),
                           ("partial_thumb2_normal_world_z_guard", False),
                           ("missing_group_reference_velocity_limit_rad_s", .5)):
            invalid = dict(self.declaration, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                BalancedEffortGraspController(self.t, self.seed, self.ids, contact_transition=invalid)
        old = BalancedEffortGraspController(self.t, self.seed, self.ids)
        self.assertNotEqual(old.contract_sha256, self.controller.contract_sha256)
        self.assertIn("+g1_coordinated_contact_transition_v1", self.controller.contract["version"])


if __name__ == "__main__":
    unittest.main()
