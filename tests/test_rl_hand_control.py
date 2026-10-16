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


if __name__ == "__main__":
    unittest.main()
