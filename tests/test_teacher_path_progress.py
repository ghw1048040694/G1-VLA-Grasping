"""A geometry-audited reference path never substitutes for observed execution."""
import copy
import unittest

from g1_vla.teacher_path_progress import JointPathProgress


def clear(first, last, context):
    return {"passed": True, "checked_kind": context["kind"]}


class PathProgressTests(unittest.TestCase):
    def make(self, waypoints=None, vmax=None, dt=.066, **kwargs):
        return JointPathProgress(waypoints or [[0., 0.], [.2, .1]], vmax or [1., .5],
                                 dt, [-1., -1.], [1., 1.], **kwargs)

    def test_shared_arc_time_obeys_each_velocity_and_anchor(self):
        follower = self.make()
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertAlmostEqual(result["u_s"], .066)
        self.assertEqual(result["reference"], [.066, .033])
        self.assertTrue(result["limit_checks"]["passed"])
        # Actual lag restricts progress; it does not get clipped to the path.
        result = follower.step([0., 0.], [.066, .033], True)
        self.assertAlmostEqual(result["reference"][0], .08)
        self.assertAlmostEqual(result["reference"][1], .04)
        self.assertLessEqual(result["advanced_u_s"], .066)

    def test_cross_many_collinear_knots_without_one_period_per_knot(self):
        follower = self.make([[0., 0.], [.01, .005], [.02, .01], [.03, .015], [.2, .1]])
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertAlmostEqual(result["reference"][0], .066)
        self.assertEqual(result["segment_index"], 3)

    def test_corner_shortcut_and_each_crossed_piece_are_validated(self):
        calls = []

        def validate(first, last, context):
            calls.append((list(first), list(last), copy.deepcopy(context)))
            return {"passed": True}

        follower = self.make([[0., 0.], [.02, 0.], [.02, .02]], [1., 1.],
                             chord_validator=validate)
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertTrue(result["end_reference_reached"])
        self.assertEqual([x[2]["kind"] for x in calls],
                         ["polyline_piece", "polyline_piece", "applied_command_shortcut",
                          "actual_state_to_candidate"])
        self.assertEqual(calls[-1][:2], ([0., 0.], [.02, .02]))
        self.assertFalse(result["complete"])

    def test_actual_lag_chord_is_checked_in_addition_to_applied_command_chord(self):
        calls = []

        def validate(first, last, context):
            calls.append((first, last, context["kind"]))
            return context["kind"] != "actual_state_to_candidate"

        follower = self.make(chord_validator=validate)
        result = follower.step([-.01, .01], [0., 0.], True)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["u_s"], 0.)
        self.assertEqual(calls[-1][0], [-.01, .01])
        self.assertEqual(calls[-1][2], "actual_state_to_candidate")

    def test_no_callback_does_not_skip_an_unaudited_corner(self):
        follower = self.make([[0., 0.], [.02, 0.], [.02, .02]], [1., 1.])
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertEqual(result["reference"], [.02, 0.])
        self.assertFalse(result["end_reference_reached"])
        self.assertTrue(result["limit_checks"]["geometry_limited_at_corner_without_callback"])

    def test_reversal_consumes_arc_time_even_if_net_displacement_is_small(self):
        follower = self.make([[0., 0.], [.03, 0.], [-.03, 0.]], [1., 1.],
                             chord_validator=clear)
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertAlmostEqual(follower.total_duration_s, .09)
        self.assertAlmostEqual(result["u_s"], .066)
        self.assertAlmostEqual(result["reference"][0], -.006)
        self.assertFalse(result["end_reference_reached"])

    def test_never_skip_box_illegal_middle_to_legal_returning_endpoint(self):
        follower = self.make([[0., 0.], [.15, 0.], [0., 0.]], [10., 1.],
                             dt=.2, chord_validator=clear)
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertAlmostEqual(result["reference"][0], .08)
        self.assertFalse(result["end_reference_reached"])
        self.assertAlmostEqual(result["u_s"], .008)

    def test_tracking_false_holds_without_nominal_clock_advancement(self):
        follower = self.make()
        for _ in range(100):
            result = follower.step([0., 0.], [0., 0.], False)
            self.assertEqual(result["u_s"], 0.)
            self.assertEqual(result["reference"], [0., 0.])
            self.assertFalse(result["complete"])
            self.assertTrue(result["blocked"])

    def test_completion_needs_executed_endpoint_and_new_actual_tracking(self):
        follower = self.make([[0., 0.], [.01, .005]])
        request = follower.step([0., 0.], [0., 0.], True)
        self.assertTrue(request["end_reference_reached"])
        self.assertFalse(request["complete"])
        held = follower.step([.005, .0025], request["reference"], False)
        self.assertFalse(held["complete"])
        confirmed = follower.step(request["reference"], request["reference"], True)
        self.assertTrue(confirmed["complete"])

    def test_failed_or_modified_execution_cannot_acknowledge_future_progress(self):
        follower = self.make([[0., 0.], [.01, .005]])
        request = follower.step([0., 0.], [0., 0.], True)
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["reason"], "previous_applied_does_not_match_current_reference")
        self.assertFalse(result["complete"])
        self.assertEqual(result["executed_u_s"], 0.)
        self.assertEqual(result["u_s"], request["u_s"])
        self.assertIsNone(result["reference"])

    def test_validator_rejection_or_exception_is_atomic(self):
        for bad in (lambda a, b, c: False,
                    lambda a, b, c: {"passed": False, "distance_m": -.001},
                    lambda a, b, c: 1):
            follower = self.make(chord_validator=bad)
            result = follower.step([0., 0.], [0., 0.], True)
            self.assertTrue(result["blocked"])
            self.assertEqual(result["reason"], "geometry_validator_rejected")
            self.assertEqual(result["u_s"], 0.)
            self.assertEqual(follower.reference, [0., 0.])
            self.assertFalse(result["validation"]["passed"])

    def test_later_piece_rejection_cannot_commit_an_earlier_partial_piece(self):
        def reject_second(a, b, context):
            return not (context["kind"] == "polyline_piece" and context["segment_index"] == 1)

        follower = self.make([[0., 0.], [.02, 0.], [.02, .02]], [1., 1.],
                             chord_validator=reject_second)
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertEqual(result["u_s"], 0.)
        self.assertEqual(result["reference"], [0., 0.])

    def test_direct_shortcut_rejection_cannot_use_piecewise_audit_as_a_substitute(self):
        follower = self.make([[0., 0.], [.02, 0.], [.02, .02]], [1., 1.],
                             chord_validator=lambda a, b, c: c["kind"] != "applied_command_shortcut")
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertEqual(result["u_s"], 0.)
        self.assertTrue(result["blocked"])

    def test_zero_velocity_joint_stays_at_previous_target_and_rejects_moving_path(self):
        follower = self.make([[0., .2], [.2, .2]], [1., 0.])
        result = follower.step([0., .21], [0., .2], True)
        self.assertEqual(result["reference"][1], .2)
        with self.assertRaises(ValueError):
            self.make([[0., 0.], [.2, .01]], [1., 0.])

    def test_no_feasible_intersection_or_illegal_current_reference_is_explicit(self):
        follower = self.make()
        result = follower.step([.5, 0.], [0., 0.], True)
        self.assertTrue(result["blocked"])
        self.assertIsNone(result["reference"])
        self.assertFalse(result["limit_checks"]["passed"])
        follower = self.make(dt=.2)
        result = follower.step([.1, 0.], [0., 0.], True)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["u_s"], 0.)
        self.assertIsNone(result["reference"])

    def test_inputs_outputs_callback_and_ledger_are_copies(self):
        points, vmax, actual, previous = [[0., 0.], [.2, .1]], [1., .5], [0., 0.], [0., 0.]

        def mutate(first, last, context):
            first[0] = 999
            last[0] = 999
            context["kind"] = "mutated"
            return True

        follower = self.make(points, vmax, chord_validator=mutate)
        points[1][0], vmax[0] = 999, 999
        result = follower.step(actual, previous, True)
        self.assertEqual(actual, [0., 0.])
        self.assertEqual(previous, [0., 0.])
        self.assertEqual(result["reference"], [.066, .033])
        result["reference"][0] = 999
        reference = follower.reference
        reference[0] = 999
        ledger = follower.records
        ledger[0]["reference"][0] = 999
        self.assertEqual(follower.reference, [.066, .033])
        self.assertEqual(follower.records[0]["reference"], [.066, .033])

    def test_repeated_zero_length_and_subpicosecond_segments_are_not_nominal_completion(self):
        follower = self.make([[0., 0.], [0., 0.], [1e-14, 0.]], [1., 1.])
        result = follower.step([0., 0.], [0., 0.], True)
        self.assertEqual(result["reference"], [1e-14, 0.])
        self.assertFalse(result["complete"])
        self.assertTrue(follower.step([1e-14, 0.], [1e-14, 0.], True)["complete"])

    def test_stationary_path_requires_actual_tracking_boolean(self):
        follower = self.make([[.1, .2], [.1, .2]], [0., 0.])
        self.assertFalse(follower.step([.1, .2], [.1, .2], False)["complete"])
        self.assertTrue(follower.step([.1, .2], [.1, .2], True)["complete"])

    def test_explicit_none_is_identical_to_original_records_and_references(self):
        original, optional = self.make(chord_validator=clear), self.make(chord_validator=clear)
        previous = [0., 0.]
        for _ in range(5):
            left = original.step(previous, previous, True)
            right = optional.step(previous, previous, True, maximum_advance_s=None)
            self.assertEqual(left, right)
            self.assertNotIn("maximum_advance_s", right)
            previous = left["reference"]
        self.assertEqual(original.records, optional.records)

    def test_arc_cap_shortens_progress_across_polyline_without_changing_physical_limits(self):
        follower = self.make([[0., 0.], [.01, 0.], [.01, .01], [.1, .01]], [1., 1.], chord_validator=clear)
        result = follower.step([0., 0.], [0., 0.], True, maximum_advance_s=.025)
        self.assertAlmostEqual(result["u_s"], .025)
        self.assertEqual(result["segment_index"], 2)
        self.assertAlmostEqual(result["reference"][0], .015)
        self.assertAlmostEqual(result["reference"][1], .01)
        self.assertEqual(result["maximum_advance_s"], .025)
        self.assertEqual(result["period_s"], .066)
        self.assertTrue(result["limit_checks"]["passed"])
        # Bounds still use the original 66ms period, not a redefined cadence.
        self.assertEqual(result["limit_checks"]["intersection_upper"], [.066, .066])
        self.assertEqual([x["context"]["kind"] for x in result["validation"]["checks"]],
                         ["polyline_piece"] * 3 + ["applied_command_shortcut", "actual_state_to_candidate"])

    def test_zero_cap_holds_acknowledges_and_can_confirm_executed_endpoint(self):
        follower = self.make([[0., 0.], [.01, .005]])
        hold = follower.step([0., 0.], [0., 0.], True, maximum_advance_s=0.)
        self.assertEqual(hold["reference"], [0., 0.])
        self.assertEqual(hold["u_s"], 0.)
        self.assertEqual(hold["advanced_u_s"], 0.)
        self.assertFalse(hold["blocked"])
        self.assertFalse(hold["complete"])
        request = follower.step([0., 0.], [0., 0.], True, maximum_advance_s=.01)
        self.assertTrue(request["end_reference_reached"])
        self.assertFalse(request["complete"])
        ack = follower.step(request["reference"], request["reference"], True, maximum_advance_s=0.)
        self.assertTrue(ack["complete"])
        self.assertEqual(ack["executed_u_s"], request["u_s"])
        self.assertEqual(ack["advanced_u_s"], 0.)

    def test_cap_cannot_bypass_ack_tracking_anchor_or_geometry_rejection(self):
        follower = self.make(chord_validator=clear)
        proposed = follower.step([0., 0.], [0., 0.], True, maximum_advance_s=.01)
        unacked = follower.step([0., 0.], [0., 0.], True, maximum_advance_s=0.)
        self.assertIsNone(unacked["reference"])
        self.assertTrue(unacked["blocked"])
        self.assertEqual(unacked["executed_u_s"], 0.)
        waiting = follower.step(proposed["reference"], proposed["reference"], False, maximum_advance_s=.01)
        self.assertEqual(waiting["advanced_u_s"], 0.)
        self.assertTrue(waiting["blocked"])
        illegal = self.make().step([.5, 0.], [0., 0.], True, maximum_advance_s=0.)
        self.assertIsNone(illegal["reference"])
        self.assertFalse(illegal["limit_checks"]["passed"])
        rejected = self.make(chord_validator=lambda *args: False).step([0., 0.], [0., 0.], True, maximum_advance_s=.01)
        self.assertEqual(rejected["u_s"], 0.)
        self.assertTrue(rejected["blocked"])
        self.assertEqual(rejected["reason"], "geometry_validator_rejected")

    def test_invalid_caps_do_not_change_progress_or_acknowledge_unexecuted_references(self):
        follower = self.make()
        proposed = follower.step([0., 0.], [0., 0.], True)
        before = follower.records
        for value in (-.001, .066001, float("nan"), float("inf"), True, "0.01"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                follower.step(proposed["reference"], proposed["reference"], True, maximum_advance_s=value)
            self.assertEqual(follower.records, before)
            self.assertEqual(follower.reference, proposed["reference"])

    def test_nonfinite_invalid_dimensions_bounds_and_period_are_rejected(self):
        for dt in (0., -1., float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                self.make(dt=dt)
        for actual in ([float("nan"), 0.], [0.], [True, 0.]):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                self.make().step(actual, [0., 0.], True)
        with self.assertRaises(ValueError):
            self.make([[0., 0.], [1.01, 0.]])
        with self.assertRaises(ValueError):
            self.make().step([0., 0.], [0., 0.], "ready")


if __name__ == "__main__":
    unittest.main()
