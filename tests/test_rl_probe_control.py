"""CPU Torch checks for probe projection mathematics; no robot simulation."""

import importlib.util
import itertools
import math
import unittest

from g1_vla.rl_probe_control import apply_probe_guard


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "Torch is required for the control-math regression")
class ProbeControlTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def inputs(self, desired, *, previous=None, current=None, low=None, high=None,
               jz=None, palm_z=None, start_z=None):
        torch = self.torch
        desired = torch.tensor(desired, dtype=torch.float64)
        count = len(desired)
        tensor = lambda value: torch.tensor(value, dtype=torch.float64)
        return [desired,
                torch.zeros_like(desired) if previous is None else tensor(previous),
                torch.zeros_like(desired) if current is None else tensor(current),
                torch.full_like(desired, -0.1) if low is None else tensor(low),
                torch.full_like(desired, 0.1) if high is None else tensor(high),
                tensor([[1, 0, 0, 0, 0, 0, 0]]*count) if jz is None else tensor(jz),
                torch.zeros(count, dtype=torch.float64) if palm_z is None else tensor(palm_z),
                torch.zeros(count, dtype=torch.float64) if start_z is None else tensor(start_z)]

    def guard(self, values, mode="box_halfspace_projected_v2", cap=0.006):
        return apply_probe_guard(self.torch, *values, cap, mode=mode)

    def test_axis_projection_preserves_horizontal_and_downward_components(self):
        torch = self.torch
        values = self.inputs([[0.02, 0.04, -0.03, 0, 0, 0, 0], [-0.02, 0.04, -0.03, 0, 0, 0, 0]])
        target, evidence = self.guard(values)
        torch.testing.assert_close(target[0], torch.tensor([0.006, 0.04, -0.03, 0, 0, 0, 0], dtype=torch.float64))
        torch.testing.assert_close(target[1], values[0][1])
        self.assertTrue(bool(evidence["postcondition_passed"].all()))
        self.assertFalse(bool(evidence["scalar_factor_applied"].any()))
        legacy, _ = self.guard(values, "legacy_scalar_v1")
        self.assertAlmostEqual(float(legacy[0, 1]), 0.012)

    def test_projection_preserves_nullspace_and_matches_closed_form_optimum(self):
        torch = self.torch
        values = self.inputs([[0.04, 0.02, 0.03, 0, 0, 0, 0]], jz=[[1, 1, 0, 0, 0, 0, 0]])
        target, evidence = self.guard(values)
        torch.testing.assert_close(target[0], torch.tensor([0.013, -0.007, 0.03, 0, 0, 0, 0], dtype=torch.float64))
        self.assertAlmostEqual(float(target[0, 0]-target[0, 1]), 0.02)
        self.assertTrue(bool(evidence["feasible"][0]))
        self.assertLessEqual(float(evidence["predicted_post"][0]), 0.006)

    def test_box_limits_and_previous_outside_new_anchor_intersection(self):
        torch = self.torch
        values = self.inputs([[0.06, 0.01, 0, 0, 0, 0, 0]],
            current=[[0.13, 0, 0, 0, 0, 0, 0]],
            low=[[0.05, -0.08, 0, 0, 0, 0, 0]], high=[[0.08, 0.08, 0, 0, 0, 0, 0]],
            jz=[[1, -1, 0, 0, 0, 0, 0]], palm_z=[0.135])
        legacy, old = self.guard(values, "legacy_scalar_v1")
        self.assertLess(float(legacy[0, 0]), 0.05)
        self.assertGreater(float(old["lower_limit_violations"][0, 0]), 0)
        self.assertTrue(bool(old["requires_unsafe"][0]))
        target, evidence = self.guard(values)
        self.assertTrue(bool(evidence["feasible"][0]))
        self.assertFalse(bool(evidence["requires_unsafe"][0]))
        torch.testing.assert_close(target[0, :2], torch.tensor([0.05, 0.049], dtype=torch.float64))

    def test_infeasible_halfspace_or_box_holds_previous_and_requires_unsafe(self):
        torch = self.torch
        values = self.inputs([[0.02, 0, 0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0, 0, 0]],
            previous=[[0.003, 0, 0, 0, 0, 0, 0], [0.004, 0, 0, 0, 0, 0, 0]],
            low=[[0.01, 0, 0, 0, 0, 0, 0], [0.02, 0, 0, 0, 0, 0, 0]],
            high=[[0.03, 0, 0, 0, 0, 0, 0], [0.01, 0, 0, 0, 0, 0, 0]])
        target, evidence = self.guard(values)
        torch.testing.assert_close(target, values[1])
        self.assertFalse(bool(evidence["feasible"].any()))
        self.assertTrue(bool(evidence["infeasible_previous_hold"].all()))
        self.assertTrue(bool(evidence["requires_unsafe"].all()))

    def test_legacy_formula_is_exact_and_exposes_existing_height_violations(self):
        torch = self.torch
        torch.manual_seed(43)
        values = self.inputs(torch.randn(8, 7).mul(0.02).tolist(),
            previous=torch.randn(8, 7).mul(0.01).tolist(), current=torch.randn(8, 7).mul(0.01).tolist(),
            jz=torch.randn(8, 7).mul(0.2).tolist(), palm_z=torch.randn(8).mul(0.002).tolist())
        desired, previous, current, _, _, jz, palm, start = values
        prediction = palm+(jz*(previous-current)).sum(-1)
        delta = (jz*(desired-previous)).sum(-1)
        allowance = (start+0.006-prediction).clamp_min(0)
        factor = torch.where(delta > allowance, allowance/delta.clamp_min(1e-12), torch.ones_like(allowance)).clamp(0, 1)
        target, evidence = self.guard(values, "legacy_scalar_v1")
        self.assertTrue(torch.equal(target, previous+(desired-previous)*factor[:, None]))
        self.assertTrue(torch.equal(evidence["factor"], factor))
        descent = self.inputs([[-0.001, 0, 0, 0, 0, 0, 0]], palm_z=[0.01])
        _, old = self.guard(descent, "legacy_scalar_v1")
        self.assertEqual(float(old["factor"][0]), 1.0)
        self.assertTrue(bool(old["requires_unsafe"][0]))
        _, new = self.guard(descent)
        self.assertTrue(bool(new["postcondition_passed"][0]))

    @staticmethod
    def active_set_oracle(desired, low, high, jz, bound):
        """Independent three-dimensional exhaustive active-set QP reference."""
        candidates = []
        for status in itertools.product((-1, 0, 1), repeat=3):
            free = [i for i, value in enumerate(status) if value == 0]
            fixed = [i for i in range(3) if i not in free]
            for height_active in (False, True):
                x = [desired[i] if i in free else low[i] if status[i] == -1 else high[i] for i in range(3)]
                if height_active:
                    norm = sum(jz[i]**2 for i in free)
                    if norm == 0:
                        continue
                    dual = (sum(jz[i]*x[i] for i in range(3))-bound)/norm
                    if dual < 0:
                        continue
                    for i in free:
                        x[i] = desired[i]-dual*jz[i]
                if (all(low[i]-1e-12 <= x[i] <= high[i]+1e-12 for i in range(3)) and
                        sum(jz[i]*x[i] for i in range(3)) <= bound+1e-12):
                    candidates.append((sum((x[i]-desired[i])**2 for i in range(3)), x))
        return min(candidates)

    def test_projected_targets_match_independent_active_set_optimum(self):
        torch = self.torch
        torch.manual_seed(16)
        desired = torch.randn(12, 3).mul(0.04).tolist()
        directions = torch.randn(12, 3).tolist()
        low, high = [-0.02, -0.015, -0.005], [0.03, 0.025, 0.02]
        values = self.inputs([row+[0]*4 for row in desired], low=[low+[0]*4]*12,
                             high=[high+[0]*4]*12, jz=[row+[0]*4 for row in directions])
        target, evidence = self.guard(values)
        for index in range(12):
            optimum, reference = self.active_set_oracle(desired[index], low, high, directions[index], 0.006)
            self.assertTrue(bool(evidence["postcondition_passed"][index]))
            torch.testing.assert_close(target[index, :3], torch.tensor(reference, dtype=torch.float64), rtol=1e-8, atol=1e-10)
            actual = sum((float(target[index, i])-desired[index][i])**2 for i in range(3))
            self.assertAlmostEqual(actual, optimum, places=11)

    def test_float32_batch_postconditions_zero_jacobian_and_inputs_unchanged(self):
        torch = self.torch
        values = [value.float() for value in self.inputs([[0.02, 0.01, 0, 0, 0, 0, 0],
            [-0.02, 0.01, 0, 0, 0, 0, 0], [0.02, 0.01, 0, 0, 0, 0, 0]],
            jz=[[1, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0], [0]*7])]
        snapshots = [value.clone() for value in values]
        target, evidence = self.guard(values)
        self.assertTrue(bool(evidence["postcondition_passed"].all()))
        self.assertTrue(bool(evidence["feasible"].all()))
        torch.testing.assert_close(target[1:], values[0][1:])
        for before, after in zip(snapshots, values):
            self.assertTrue(torch.equal(before, after))

    def test_nonfinite_and_malformed_inputs_fail_before_returning_targets(self):
        torch = self.torch
        for index in range(8):
            values = self.inputs([[0]*7])
            values[index].flatten()[0] = math.nan
            with self.subTest(input=index), self.assertRaises(FloatingPointError):
                self.guard(values)
        with self.assertRaises(FloatingPointError):
            self.guard(self.inputs([[0]*7]), cap=math.inf)
        with self.assertRaises(ValueError):
            self.guard(self.inputs([[0]*7]), mode="disable_shield")
        with self.assertRaises(ValueError):
            self.guard(self.inputs([[0]*7]), cap=-0.006)
        values = self.inputs([[0]*7])
        values[1] = values[1][:, :6]
        with self.assertRaises(ValueError):
            self.guard(values)

    def test_legacy_roundoff_is_explicit_without_relaxing_projected_constraints(self):
        values = self.inputs([[0]*7], jz=[[0]*7], palm_z=[0.006+5e-8])
        _, old = self.guard(values, "legacy_scalar_v1")
        self.assertFalse(bool(old["postcondition_passed"][0]))
        self.assertTrue(bool(old["numerical_postcondition_passed"][0]))
        self.assertTrue(bool(old["legacy_roundoff_only"][0]))
        self.assertFalse(bool(old["requires_unsafe"][0]))
        self.assertGreater(float(old["height_violation_m"][0]), 0)
        _, projected = self.guard(values)
        self.assertFalse(bool(projected["feasible"][0]))
        self.assertTrue(bool(projected["requires_unsafe"][0]))
        larger = self.inputs([[0]*7], jz=[[0]*7], palm_z=[0.006+2e-7])
        _, old = self.guard(larger, "legacy_scalar_v1")
        self.assertFalse(bool(old["numerical_postcondition_passed"][0]))
        self.assertTrue(bool(old["requires_unsafe"][0]))


if __name__ == "__main__":
    unittest.main()
