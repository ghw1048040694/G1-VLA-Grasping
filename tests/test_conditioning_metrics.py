"""Frozen-output collapse cannot fit labels or change paired denominators."""

import hashlib
import importlib.util
import inspect
import unittest
from unittest.mock import patch

from g1_vla import conditioning_metrics as metrics
from g1_vla.constant_residual import decode_constant_residual
from g1_vla.expanded_residual import TRAIN

HAS_NUMPY = importlib.util.find_spec("numpy") is not None


class ConditioningHelperContractTests(unittest.TestCase):
    def test_model_collapse_has_no_labels_validation_or_checkpoint_arguments(self):
        parameters = inspect.signature(metrics.fit_model_output_constant).parameters
        self.assertEqual(list(parameters), ["train_scaled_residual", "train_ids", "action_std",
            "allowed_train_episode_ids", "np", "expected_train_identities"])
        self.assertEqual(parameters["expected_train_identities"].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameters["expected_train_identities"].default, inspect.Parameter.empty)


@unittest.skipUnless(HAS_NUMPY, "NumPy is required for real frozen-output and raw paired calculations")
class ConditioningArrayTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np = np

    def train_fixture(self, frames=(0, 30)):
        np = self.np
        ids = np.asarray([(episode, frame) for episode in TRAIN for frame in frames], dtype=np.int64)
        values = np.zeros((len(ids), 16, 28), dtype=np.float32)
        values[::2] = np.float32(.1)
        values[1::2] = np.float32(.3)
        std = np.full(28, .3, dtype=np.float32)
        return values, ids, std

    def fit(self, values, ids, std, **kwargs):
        return metrics.fit_model_output_constant(values, ids, std, TRAIN, self.np,
            expected_train_identities=kwargs.get("expected", ids))

    def test_model_raw_multiply_matches_fp32_decoder_before_float64_median(self):
        np = self.np
        values, ids, std = self.train_fixture()
        with patch.object(np, "median", wraps=np.median) as median:
            constant, report = self.fit(values, ids, std)
        multiplied = np.multiply(values, std, dtype=np.float32)
        self.assertEqual(median.call_args.args[0].dtype, np.float64)
        self.assertTrue(np.array_equal(median.call_args.args[0], multiplied.astype(np.float64)))
        self.assertNotEqual(float(multiplied[0, 0, 0]), float(values[0, 0, 0]) * float(std[0]))
        expected = np.median(multiplied.astype(np.float64), axis=0).astype(np.float32)
        self.assertEqual(constant.tobytes(), expected.tobytes())
        self.assertEqual(report["pooled_model_output_count"], len(TRAIN) * 2)
        self.assertEqual(report["raw_scale_multiply_dtype"], "float32")
        self.assertEqual(report["median_dtype"], "float64")
        self.assertFalse(report["labels_used_for_fit"])
        self.assertFalse(report["validation_or_test_used_for_fit"])
        self.assertFalse(report["train_label_residual_l1_optimum_claimed"])
        self.assertEqual(report["optimizer_steps"], 0)
        self.assertFalse(report["checkpoint_parameters_modified"])

    def test_collapse_pools_anchors_instead_of_episode_medians(self):
        np = self.np
        # Thirteen single-anchor episodes at zero and one longer episode at 8:
        # the pooled median is 8 although the median of episode medians is 0.
        identities = [(episode, 0) for episode in TRAIN[:-1]] + [(TRAIN[-1], index * 30) for index in range(15)]
        ids = np.asarray(identities, dtype=np.int64)
        values = np.zeros((len(ids), 16, 28), dtype=np.float32)
        values[-15:] = 8
        constant, report = self.fit(values, ids, np.ones(28, dtype=np.float32))
        self.assertTrue(np.array_equal(constant, np.full((16, 28), 8, dtype=np.float32)))
        self.assertEqual(report["model_output_count_by_episode"][str(TRAIN[-1])], 15)
        self.assertEqual(report["parameter_bytes_sha256"], hashlib.sha256(constant.tobytes()).hexdigest())

    def test_even_median_has_one_parameter_for_every_horizon_and_joint(self):
        np = self.np
        values, ids, std = self.train_fixture()
        offsets = np.arange(16 * 28, dtype=np.float32).reshape(16, 28) / np.float32(1000)
        values[::2] = offsets
        values[1::2] = offsets + np.float32(.2)
        parameter, report = self.fit(values, ids, np.ones(28, dtype=np.float32))
        expected = ((values[0].astype(np.float64) + values[1].astype(np.float64)) / 2).astype(np.float32)
        self.assertTrue(np.array_equal(parameter, expected))
        self.assertEqual(report["even_population_rule"], "midpoint_of_two_central_values")
        self.assertEqual(parameter.dtype, np.float32)
        self.assertTrue(parameter.flags.c_contiguous)

    def test_full_fourteen_train_membership_and_exact_identity_order_are_required(self):
        np = self.np
        values, ids, std = self.train_fixture()
        changed = ids.copy()
        changed[0, 0] = 180
        with self.assertRaises(metrics.ConditioningMetricsError):
            self.fit(values, changed, std)
        for identity in (280, 281, 282):
            changed = ids.copy()
            changed[0, 0] = identity
            with self.subTest(identity=identity), self.assertRaises(metrics.ConditioningMetricsError):
                self.fit(values, changed, std)
        with self.assertRaises(metrics.ConditioningMetricsError):
            self.fit(values, ids, std, expected=ids[::-1].copy())
        with self.assertRaises(metrics.ConditioningMetricsError):
            metrics.fit_model_output_constant(values, ids, std, list(TRAIN[:-1]) + [180], np,
                                             expected_train_identities=ids)
        with self.assertRaises(metrics.ConditioningMetricsError):
            metrics.fit_model_output_constant(values, ids, std, [float(episode) for episode in TRAIN], np,
                                             expected_train_identities=ids)
        with self.assertRaises(metrics.ConditioningMetricsError):
            self.fit(values[2:], ids[2:], std)

    def test_frame_zero_stride_30_sequence_cannot_drop_internal_anchors(self):
        values, ids, std = self.train_fixture((0, 60))
        with self.assertRaises(metrics.ConditioningMetricsError):
            self.fit(values, ids, std)
        values, ids, std = self.train_fixture((30, 60))
        with self.assertRaises(metrics.ConditioningMetricsError):
            self.fit(values, ids, std)

    def test_invalid_dtype_nonfinite_scale_duplicate_or_fractional_identity_rejected(self):
        np = self.np
        values, ids, std = self.train_fixture()
        invalid_values = [values.astype(np.float64), values[:, :15], np.full_like(values, np.nan)]
        for changed in invalid_values:
            with self.subTest(shape=changed.shape, dtype=changed.dtype), self.assertRaises(metrics.ConditioningMetricsError):
                self.fit(changed, ids, std)
        for changed in (std.astype(np.float64), np.zeros_like(std), np.full_like(std, np.inf)):
            with self.assertRaises(metrics.ConditioningMetricsError):
                self.fit(values, ids, changed)
        duplicate = ids.copy()
        duplicate[1] = duplicate[0]
        for changed in (duplicate, ids.astype(np.float32), np.full(ids.shape, True), ids - 1000):
            with self.assertRaises(metrics.ConditioningMetricsError):
                self.fit(values, changed, std)
        overflow = np.full_like(values, np.finfo(np.float32).max)
        with self.assertRaises(metrics.ConditioningMetricsError):
            self.fit(overflow, ids, np.full_like(std, 2))

    def test_zero_collapsed_output_decodes_without_altering_signed_raw_anchor_bits(self):
        np = self.np
        values, ids, std = self.train_fixture()
        values[:] = np.float32(-0.)
        constant, _ = self.fit(values, ids, std)
        state = np.zeros((2, 28), dtype=np.float32)
        state[::2, ::2] = np.float32(-0.)
        prediction = decode_constant_residual(state, constant, np)
        self.assertEqual(prediction.tobytes(), np.broadcast_to(state[:, None, :], (2, 16, 28)).copy().tobytes())

    def evaluation_fixture(self):
        np = self.np
        ids = np.asarray([[180, 0], [180, 30], [181, 0]], dtype=np.int64)
        target = np.zeros((3, 16, 28), dtype=np.float32)
        full = target.copy()
        other = target.copy()
        full[1] = 2
        full[2] = 1
        other[:] = 1
        state = np.zeros((3, 28), dtype=np.float32)
        return full, other, target, state, ids

    def test_paired_sign_exact_ties_and_episode_counts_are_explicit(self):
        np = self.np
        full, other, target, _, ids = self.evaluation_fixture()
        result = metrics.paired_conditioning_metrics(full, other, target, ids, np)
        self.assertEqual(result["anchor_comparison_counts"], {"improved": 1, "worsened": 1, "tied": 1})
        self.assertEqual([item["comparator_minus_full_mae_rad"] for item in result["per_anchor"]], [1, -1, 0])
        self.assertEqual(result["comparator_minus_full_mae_rad"], 0)
        self.assertEqual(result["episode_comparison_counts"], {"improved": 0, "worsened": 0, "tied": 2})
        self.assertEqual(result["valid_scalar_count"], 3 * 16 * 28)
        self.assertFalse(result["anchor_independence_assumed"])
        self.assertFalse(result["significance_test_performed"])
        self.assertFalse(result["used_for_parameter_selection"])

    def test_global_mae_weights_anchors_instead_of_episode_means(self):
        np = self.np
        full, other, target, _, ids = self.evaluation_fixture()
        full[:] = 0
        other[:2] = 1
        other[2] = 4
        result = metrics.paired_conditioning_metrics(full, other, target, ids, np)
        self.assertEqual(result["comparator_minus_full_mae_rad"], 2)
        self.assertEqual(result["per_episode"]["180"]["comparator_minus_full_mae_rad"], 1)
        self.assertEqual(result["per_episode"]["181"]["comparator_minus_full_mae_rad"], 4)
        self.assertNotEqual(result["comparator_minus_full_mae_rad"], 2.5)

    def test_raw_metrics_keep_joint_horizon_arms_hands_and_hold_denominators(self):
        np = self.np
        full, _, target, state, ids = self.evaluation_fixture()
        full[:, :, :14] = 1
        full[:, :, 14:] = 3
        result = metrics.raw_conditioning_metrics(full, target, state, ids, np)
        self.assertEqual(result["policy"]["all_28"]["mae"], 2)
        self.assertEqual(result["policy"]["arms_14"]["mae"], 1)
        self.assertEqual(result["policy"]["hands_14"]["mae"], 3)
        self.assertEqual(result["policy"]["mae_by_joint"], [1] * 14 + [3] * 14)
        self.assertEqual(result["policy"]["mae_by_horizon"], [2] * 16)
        self.assertEqual(result["policy"]["valid_scalar_count"], result["state_hold"]["valid_scalar_count"])
        self.assertIsNone(result["policy_to_state_hold_mae_ratio"])
        self.assertEqual(set(result["per_episode"]), {"180", "181"})

    def test_motion_strata_share_raw_and_paired_denominators_without_selecting(self):
        np = self.np
        ids = np.asarray([[180, 0], [180, 30], [180, 60], [180, 90]], dtype=np.int64)
        speed = np.asarray([np.nan, .02, .05, .1], dtype=np.float64)
        target = np.zeros((4, 16, 28), dtype=np.float32)
        full = np.ones_like(target)
        other = np.full_like(target, 2)
        state = np.zeros((4, 28), dtype=np.float32)
        raw = metrics.raw_conditioning_metrics(full, target, state, ids, np, speeds=speed)
        paired = metrics.paired_conditioning_metrics(full, other, target, ids, np, speeds=speed)
        for result in (raw, paired):
            motion = result["motion_strata"]
            self.assertEqual(motion["partition_anchor_count"], 4)
            self.assertEqual([value["anchor_count"] for value in motion["strata"].values()], [1, 1, 1, 1])
            self.assertFalse(motion["used_as_model_input"])
        bad = speed.copy()
        bad[0] = 0
        with self.assertRaises(metrics.ConditioningMetricsError):
            metrics.raw_conditioning_metrics(full, target, state, ids, np, speeds=bad)
        bad = speed.copy()
        bad[1] = np.nan
        with self.assertRaises(metrics.ConditioningMetricsError):
            metrics.paired_conditioning_metrics(full, other, target, ids, np, speeds=bad)

    def test_empty_motion_groups_are_none_instead_of_nan_metrics(self):
        np = self.np
        full, other, target, state, ids = self.evaluation_fixture()
        speed = np.asarray([np.nan, .2, np.nan], dtype=np.float64)
        result = metrics.raw_conditioning_metrics(full, target, state, ids, np, speeds=speed)
        self.assertIsNone(result["motion_strata"]["strata"]["stationary"]["metrics"])
        self.assertIsNone(result["motion_strata"]["strata"]["intermediate"]["metrics"])

    def test_constant_output_variation_is_zero_but_variable_output_is_not_usefulness_proof(self):
        np = self.np
        full, _, _, _, ids = self.evaluation_fixture()
        constant = np.ones((16, 28), dtype=np.float32)
        full[:] = 1
        result = metrics.model_residual_variation(full, constant, ids, np)
        self.assertEqual(result["mean_coordinate_std_rad"], 0)
        self.assertEqual(result["coordinates_with_exactly_nonzero_range"], 0)
        self.assertEqual(result["distance_to_train_output_constant_mae_rad"], 0)
        full[0] = -1
        full[1] = 3
        result = metrics.model_residual_variation(full, constant, ids, np)
        self.assertEqual(result["coordinates_with_exactly_nonzero_range"], 16 * 28)
        self.assertEqual(result["max_coordinate_range_rad"], 4)
        self.assertEqual(result["distance_to_train_output_constant_mae_rad"], 4 / 3)
        self.assertFalse(result["input_output_variation_proves_useful_state_matching"])
        self.assertEqual(result["population_std_ddof"], 0)

    def test_paired_shape_and_identity_errors_and_tunable_ties_are_rejected(self):
        np = self.np
        full, other, target, state, ids = self.evaluation_fixture()
        for changed in (other[:2], other[:, :15], other.astype(np.float64), np.full_like(other, np.inf)):
            with self.assertRaises(metrics.ConditioningMetricsError):
                metrics.paired_conditioning_metrics(full, changed, target, ids, np)
        duplicate = ids.copy()
        duplicate[1] = duplicate[0]
        with self.assertRaises(metrics.ConditioningMetricsError):
            metrics.paired_conditioning_metrics(full, other, target, duplicate, np)
        for tolerance in (True, .00001, -1, np.nan):
            with self.assertRaises(metrics.ConditioningMetricsError):
                metrics.paired_conditioning_metrics(full, other, target, ids, np, tie_tolerance_rad=tolerance)
        with self.assertRaises(metrics.ConditioningMetricsError):
            metrics.raw_conditioning_metrics(full, target, state.astype(np.float64), ids, np)
        with self.assertRaises(metrics.ConditioningMetricsError):
            metrics.model_residual_variation(full, np.zeros((16, 28)), ids, np)


if __name__ == "__main__":
    unittest.main()
