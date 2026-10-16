import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from g1_vla.contracts import ActionContract
from g1_vla.data_pairing import audit_causal_action_labels, encode_paired_chunks, PairingError
from g1_vla.native_action_labels import (LABEL_PROTOCOL, LabelProtocolError, encode_causal_prefixes,
                                         validate_masked_labels, masked_action_stats,
                                         masked_action_errors, masked_action_diagnostics, masked_smooth_l1_loss)
from test_data_pairing import NAMES, build


def contract(amplitude=0.04, velocity=1.0):
    return ActionContract(tuple(NAMES), 1 / 0.066, 0.002, (amplitude,) * 31, (velocity,) * 31)


class CausalPrefixTests(unittest.TestCase):
    def test_real_prefix_is_lossless_common_anchor_and_first_four_execution_identical(self):
        episode = build()
        original = {key: value.copy() for key, value in episode.items()}
        result = encode_causal_prefixes(episode, contract())
        labels = result['arrays']
        self.assertEqual(labels['anchor_record_index'].tolist(), list(range(1, 7)))
        self.assertEqual(labels['valid_prefix_length'].tolist(), [4] * 6)
        self.assertEqual(result['summary']['candidate_anchor_count'], 10)
        self.assertEqual(result['summary']['excluded_anchor_count'], 4)
        for row, start in enumerate(labels['anchor_record_index']):
            np.testing.assert_array_equal(labels['action_absolute_target_rad'][row, :4], episode['action_joint_position_rad'][start:start + 4])
            self.assertTrue(labels['action_is_pad'][row, 4:].all())
            self.assertEqual(labels['source_record_index'][row].tolist(), list(range(start, start + 4)) + [-1] * 4)
            # First command is A[start], never A[start+1] or post-state q.
            np.testing.assert_array_equal(labels['action_absolute_target_rad'][row, 0], episode['action_joint_position_rad'][start])
            self.assertFalse(np.array_equal(labels['action_absolute_target_rad'][row, 0], episode['post_joint_position_rad'][start]))
            decoded = contract().decode_chunk(episode['observation_joint_position_rad'][start].tolist(), labels['action_delta_chunk_rad'][row].tolist(), joint_names=NAMES)
            executable = contract().executable_targets(episode['observation_joint_position_rad'][start].tolist(), labels['action_delta_chunk_rad'][row].tolist(), joint_names=NAMES,
                                                       previous_target=episode['action_joint_position_rad'][start - 1].tolist())
            np.testing.assert_allclose(executable, episode['action_joint_position_rad'][start:start + 4], atol=1e-12, rtol=0)
        validate_masked_labels(labels, contract())
        for key in episode:
            np.testing.assert_array_equal(episode[key], original[key])
        self.assertFalse(json.loads(str(labels['metadata_json']))['training_eligible'])
        self.assertEqual(len(encode_paired_chunks(episode, contract())['anchor_record_index']), 3)
        self.assertFalse(encode_paired_chunks(episode, contract())['action_contract_consistent'].any())

    def test_no_previous_command_no_rescue_of_invalid_first_or_future_anchor_substitution(self):
        episode = build()
        result = encode_causal_prefixes(episode, contract())
        self.assertEqual(result['excluded'][0]['stop_reason'], 'previous_applied_command_unproven')
        episode['action_joint_position_rad'][1, 0] = 0.9
        result = encode_causal_prefixes(episode, contract())
        excluded = {r['anchor_record_index']: r for r in result['excluded']}
        self.assertEqual(excluded[1]['valid_prefix_length'], 0)
        self.assertEqual(excluded[2]['stop_reason'], 'adjacent_command_velocity')
        # Future states can have their own valid anchors. They do not rescue a
        # command outside the current anchor's amplitude budget.
        episode = build()
        labels = encode_causal_prefixes(episode, contract())['arrays']
        self.assertEqual(labels['valid_prefix_length'][0], 4)
        self.assertLess(abs(episode['action_joint_position_rad'][5, 0] - episode['observation_joint_position_rad'][5, 0]), .04)

    def test_segment_restart_and_source_tail_never_cross_or_skip(self):
        episode = build(14)
        episode['state_segment_id'][6:] = 1
        episode['continuity_from_previous'][6] = False
        episode['pairing_valid'][6] = False
        episode['restart_wait'][7] = True
        episode['pairing_valid'][7] = False
        labels = encode_causal_prefixes(episode, contract(.08))['arrays']
        for row, start in enumerate(labels['anchor_record_index']):
            indices = labels['source_record_index'][row][~labels['action_is_pad'][row]]
            self.assertTrue(episode['pairing_valid'][indices].all())
            self.assertEqual(len(set(episode['state_segment_id'][indices].tolist())), 1)
            self.assertTrue((indices < len(episode['pairing_valid'])).all())
            self.assertTrue((labels['observation_history_record_index'][row] <= start).all())
        self.assertNotIn(6, labels['anchor_record_index'])
        self.assertNotIn(7, labels['anchor_record_index'])

    def test_history_has_no_future_state_and_target_changes_do_not_change_past_input(self):
        episode = build()
        first = encode_causal_prefixes(episode, contract())['arrays']
        mutated = copy.deepcopy(episode)
        mutated['post_joint_position_rad'][4:] += 100
        mutated['observation_joint_position_rad'][4:] += 100
        second = encode_causal_prefixes(mutated, contract())['arrays']
        first_row = np.flatnonzero(first['anchor_record_index'] == 1)[0]
        second_row = np.flatnonzero(second['anchor_record_index'] == 1)[0]
        np.testing.assert_array_equal(first['observation_history_record_index'][first_row], [0, 0, 0, 1])
        np.testing.assert_array_equal(first['anchor_joint_position_rad'][first_row], second['anchor_joint_position_rad'][second_row])
        np.testing.assert_array_equal(first['action_delta_chunk_rad'][first_row], second['action_delta_chunk_rad'][second_row])

    def test_artifact_validator_rejects_suffix_truth_fake_source_and_future_input(self):
        original = encode_causal_prefixes(build(), contract())['arrays']
        for key in ('action_is_pad', 'source_record_index', 'observation_history_record_index', 'action_absolute_target_rad'):
            labels = copy.deepcopy(original)
            if key == 'action_is_pad':
                labels[key][0, 6] = False
            elif key == 'source_record_index':
                labels[key][0, 4] = 5
            elif key == 'observation_history_record_index':
                labels[key][0, -1] = 2
            else:
                labels[key][0, -1, 0] += .001
            with self.assertRaises(LabelProtocolError):
                validate_masked_labels(labels, contract())

    def test_invalid_timestamp_indices_and_contract_identity_are_rejected(self):
        for field, value in (("command_start_time_s", float("nan")), ("command_end_time_s", float("inf"))):
            episode = build()
            episode[field][3] = value
            with self.assertRaises(LabelProtocolError):
                encode_causal_prefixes(episode, contract())
        episode = build()
        episode["command_end_time_s"][3] -= .001
        with self.assertRaises(LabelProtocolError):
            encode_causal_prefixes(episode, contract())
        episode = build()
        episode["state_segment_id"] = episode["state_segment_id"].astype(float)
        with self.assertRaises(LabelProtocolError):
            encode_causal_prefixes(episode, contract())
        labels = encode_causal_prefixes(build(), contract())["arrays"]
        with self.assertRaisesRegex(LabelProtocolError, "execution contract"):
            validate_masked_labels(labels, contract(.08))

    def test_concrete_export_preserves_raw_legacy_and_is_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = build()
            metadata = json.loads(str(episode['metadata_json']))
            metadata['source']['parent_root'] = str(root.resolve())
            episode['metadata_json'] = np.asarray(json.dumps(metadata))
            raw = root / 'raw.npz'
            episode['tracking_json'] = np.asarray(json.dumps([{'phase': 'home' if i < 2 else 'contact'} for i in range(10)]))
            np.savez_compressed(raw, **episode)
            (root / 'teacher_report.json').write_text(json.dumps({'passed': False, 'raw_episode': str(raw.resolve()),
                                                                  'layout_fingerprint': 'a' * 64}))
            before = raw.read_bytes()
            config = {'expert': {'labels': {'backend': 'native_causal_prefix_v1'}}}
            with patch('g1_vla.contracts.ActionContract.from_config', return_value=contract()):
                report = audit_causal_action_labels(root, config, run_dir=root / 'derived', records=[{'raw_path': str(raw)}])
                self.assertEqual(report['errors'], [])
                self.assertEqual(report['summary']['admitted_anchor_count'], 6)
                self.assertEqual(report['summary']['legacy_full8_count'], 3)
                self.assertEqual(report['summary']['legacy_full8_inconsistent_count'], 3)
                row = report['records'][0]
                self.assertTrue(row['raw_unchanged'])
                self.assertTrue(row['physical_provenance']['inherited_any_frame_physical_positive'])
                self.assertFalse(row['physical_provenance']['reported_expert_episode_passed'])
                self.assertEqual(row['phase_coverage']['home']['candidate_anchors'], 2)
                self.assertEqual(row['phase_coverage']['contact']['admitted_anchors'], 5)
                with np.load(row['label_artifact_path'], allow_pickle=False) as labels:
                    self.assertEqual(masked_action_diagnostics(labels, contract())['sample_count'], 6)
                with self.assertRaisesRegex(PairingError, 'immutable'):
                    audit_causal_action_labels(root, config, run_dir=root / 'derived', records=[{'raw_path': str(raw)}])
            self.assertEqual(before, raw.read_bytes())


class MaskConsumerTests(unittest.TestCase):
    def test_padding_invariance_of_stats_metrics_and_last_real_endpoint(self):
        labels = encode_causal_prefixes(build(), contract())['arrays']
        target, mask = labels['action_delta_chunk_rad'], labels['action_is_pad']
        predicted = target + .02
        stats = masked_action_stats(target, mask)
        errors = masked_action_errors(predicted, target, mask)
        garbage_target, garbage_predicted = target.copy(), predicted.copy()
        garbage_target[mask] = np.nan
        garbage_predicted[mask] = -1e30
        altered_stats = masked_action_stats(garbage_target, mask)
        np.testing.assert_array_equal(stats['mean'], altered_stats['mean'])
        np.testing.assert_array_equal(stats['std'], altered_stats['std'])
        self.assertEqual(errors, masked_action_errors(garbage_predicted, garbage_target, mask))
        self.assertAlmostEqual(errors['last_real_endpoint_mae_rad'], .02)
        self.assertEqual(stats['real_position_count'], 24)

    def test_torch_masked_loss_padding_value_and_gradient_invariance(self):
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        target = torch.zeros(2, 8, 31)
        mask = torch.tensor([[False] * 4 + [True] * 4, [False] * 6 + [True] * 2])
        prediction = torch.full_like(target, .2, requires_grad=True)
        first = masked_smooth_l1_loss(prediction, target, mask)
        first.backward()
        gradient = prediction.grad.clone()
        changed_target = target.clone()
        changed_target[mask] = float('nan')
        changed_prediction = prediction.detach().clone()
        changed_prediction[mask] = 1e20
        changed_prediction.requires_grad_(True)
        second = masked_smooth_l1_loss(changed_prediction, changed_target, mask)
        second.backward()
        self.assertEqual(float(first), float(second))
        self.assertTrue(torch.equal(gradient, changed_prediction.grad))
        self.assertTrue((gradient[mask] == 0).all())
        self.assertTrue((gradient[~mask] > 0).all())

    def test_existing_native_consumer_explicitly_rejects_new_protocol(self):
        from g1_vla.training import ContinuousDataset, TrainingNotReady, action_label, get_stats
        cache = {'label_protocol': LABEL_PROTOCOL, 'action_is_pad': np.zeros((1, 8), bool)}
        for operation in (lambda: ContinuousDataset(cache, contract()), lambda: action_label(cache, 0, 0, contract()),
                          lambda: get_stats(cache, contract())):
            with self.assertRaisesRegex(TrainingNotReady, 'no approved causal-prefix cache bridge'):
                operation()
