import unittest

from g1_vla.evaluation import evaluate_trace, summarize_results


def trace(contact=True, assisted=False):
    return [{"time_s": i / 10, "target_lift_m": .12, "target_contact": contact,
             "assisted": assisted, "numerically_stable": True, "joint_safe": True,
             "target_in_box": True, "released": True, "placement_stable": True} for i in range(8)]


class EvaluationTests(unittest.TestCase):
    def test_real_sustained_lift(self):
        self.assertTrue(evaluate_trace(trace())["strict_success"])

    def test_assist_or_in_box_alone_never_passes(self):
        self.assertFalse(evaluate_trace(trace(assisted=True))["strict_success"])
        self.assertFalse(evaluate_trace(trace(contact=False), task="pick_place")["strict_success"])

    def test_contact_interruption_restarts_hold(self):
        rows = trace()
        rows[3]["target_contact"] = False
        self.assertFalse(evaluate_trace(rows)["strict_success"])

    def test_unsafe_or_unstable_anywhere_fails(self):
        for field in ("joint_safe", "numerically_stable"):
            rows = trace()
            rows[2][field] = False
            self.assertFalse(evaluate_trace(rows)["strict_success"])

    def test_missing_boolean_or_duplicate_timestamps_rejected(self):
        rows = trace()
        del rows[1]["assisted"]
        with self.assertRaises(ValueError):
            evaluate_trace(rows)
        rows = trace()
        rows[1]["time_s"] = rows[0]["time_s"]
        with self.assertRaises(ValueError):
            evaluate_trace(rows)

    def test_empty_suite_is_not_zero_success(self):
        with self.assertRaises(ValueError):
            summarize_results([])


if __name__ == "__main__":
    unittest.main()
