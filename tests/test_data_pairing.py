import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from g1_vla.contracts import ActionContract, canonical_sha256
from g1_vla.data_pairing import (PairingError, audit_raw_pairing, build_paired_episode,
                                 encode_paired_chunks, select_action31)


NAMES = ["joint_{}".format(index) for index in range(31)]


def source(root="/project", split="diagnostic"):
    return {"parent_root": root, "source_rollout": "outputs/episode/expert_pick_place_episode.npz",
            "parent_rollout_id": "original_episode", "layout_fingerprint": "a" * 64,
            "regime": "expert", "takeover_frame": None, "split": split}


def episode_args(count=10, sensors=False):
    observations, targets, timing, physical = [], [], [], []
    for index in range(count):
        q = np.full(31, index * 0.01)
        observation = {"joint_position": q, "joint_velocity": np.full(31, 0.1),
                       "timestamp_s": index * 0.066}
        if sensors:
            observation.update(images={key: np.full((2, 3, 3), index, np.uint8)
                                       for key in ("head", "left", "right")},
                               finger_sites=np.zeros(108), tactile=[0.0], language="lift the green cube")
        observations.append(observation)
        targets.append(q + 0.005)
        timing.append({"start_s": index * 0.066, "end_s": (index + 1) * 0.066,
                       "sequence_index": index, "state_segment_id": 0, "restart_wait": False,
                       "post_joint_position": np.full(31, (index + 1) * 0.01),
                       "post_joint_velocity": np.full(31, 0.1), "post_time_s": (index + 1) * 0.066})
        physical.append({"contact": index >= 3, "lift_height_m": 0.2 if index >= 3 else 0.0,
                         "hold_seconds": 0.6 if index >= 8 else 0.0, "assisted": False,
                         "finite": True, "joint_safe": True})
    return observations, targets, timing, physical


def build(count=10, sensors=False):
    observations, targets, timing, physical = episode_args(count, sensors)
    return build_paired_episode(observations, targets, timing, NAMES, source(),
                                physical_evidence=physical, expected_interval_s=0.066)


def config(root):
    metadata = root / "joint_info.json"
    metadata.write_text(json.dumps({"features": {"action": {"shape": [31], "names": NAMES}}}))
    return {"action": {"joint_metadata_path": str(metadata), "joint_order_sha256": canonical_sha256(NAMES)},
            "control": {"control_hz": 1.0 / 0.066}}


def legacy(root, name="episode"):
    path = root / "outputs" / name / "expert_pick_place_episode.npz"
    path.parent.mkdir(parents=True)
    # Lower-body names and a reversed upper-body order require real name lookup.
    names = ["leg_{}".format(i) for i in range(12)] + list(reversed(NAMES))
    arrays = {"joint_names": names, "joint_position_rad": np.arange(4 * 43, dtype=np.float32).reshape(4, 43),
              "joint_velocity_rad_s": np.ones((4, 43), np.float32),
              "action_joint_position_rad": np.arange(4 * 43, dtype=np.float32).reshape(4, 43) + 0.02,
              "time_s": np.asarray([0, 0.066, 0.132, 0.198], np.float32)}
    np.savez_compressed(path, **arrays)
    return path, arrays


class PairingSchemaTests(unittest.TestCase):
    def test_full_sensor_actual_interval_is_persisted_without_modification(self):
        observations, targets, timing, physical = episode_args(sensors=True)
        arrays = build_paired_episode(observations, targets, timing, NAMES, source(),
                                      physical_evidence=physical, expected_interval_s=0.066)
        np.testing.assert_array_equal(arrays["observation_joint_position_rad"], [row["joint_position"] for row in observations])
        np.testing.assert_array_equal(arrays["action_joint_position_rad"], targets)
        np.testing.assert_array_equal(arrays["images_camera_0"][4], observations[4]["images"]["head"])
        self.assertTrue(arrays["pairing_valid"].all())
        metadata = json.loads(str(arrays["metadata_json"]))
        self.assertTrue(metadata["pairing_eligible"])
        self.assertTrue(metadata["sensor_complete"])
        self.assertFalse(metadata["training_eligible"])
        self.assertFalse(metadata["time_resampled"])
        self.assertFalse(metadata["labels_modified"])

    def test_episode_success_never_broadcasts_positive_frame_labels(self):
        arrays = build()
        self.assertEqual(arrays["frame_physical_positive"].tolist(), [False] * 8 + [True, True])
        self.assertTrue(json.loads(str(arrays["metadata_json"]))["episode_strict_physical_success"])

    def test_unsafe_or_nonfinite_intervals_cannot_be_physical_positive(self):
        observations, targets, timing, physical = episode_args()
        physical[8]["finite"] = False
        physical[9]["joint_safe"] = False
        arrays = build_paired_episode(observations, targets, timing, NAMES, source(), physical_evidence=physical)
        self.assertFalse(arrays["frame_physical_positive"].any())
        self.assertFalse(arrays["physical_numeric_finite"][8])
        self.assertFalse(arrays["physical_joint_safe"][9])
        self.assertTrue(arrays["pairing_valid"].all())

    def test_missing_safety_evidence_keeps_diagnostic_pairs_but_no_positive(self):
        observations, targets, timing, physical = episode_args()
        for row in physical:
            del row["finite"]
            del row["joint_safe"]
        arrays = build_paired_episode(observations, targets, timing, NAMES, source(), physical_evidence=physical)
        self.assertTrue(arrays["pairing_valid"].all())
        self.assertFalse(arrays["frame_physical_positive"].any())
        self.assertFalse(arrays["physical_numeric_finite"].any())
        self.assertFalse(arrays["physical_joint_safe"].any())
        self.assertIn("physical_safety_unproven", json.loads(str(arrays["metadata_json"]))["review_issues"])

    def test_safety_flags_must_be_explicit_booleans(self):
        for field in ("finite", "joint_safe"):
            observations, targets, timing, physical = episode_args()
            physical[8][field] = 1
            with self.assertRaisesRegex(PairingError, "explicit measured boolean"):
                build_paired_episode(observations, targets, timing, NAMES, source(), physical_evidence=physical)

    def test_nonfinite_positions_and_wrong_joint_identity_fail(self):
        observations, targets, timing, physical = episode_args()
        observations[0]["joint_position"][0] = np.nan
        with self.assertRaises(PairingError):
            build_paired_episode(observations, targets, timing, NAMES, source())
        with self.assertRaises(PairingError):
            select_action31(np.zeros((3, 31)), ["duplicate"] * 31, NAMES)
        with self.assertRaises(PairingError):
            select_action31(np.zeros((3, 31)), NAMES, NAMES[:-1] + ["missing"])

    def test_hidden_state_jump_and_velocity_reset_fail(self):
        for key in ("joint_position", "joint_velocity"):
            observations, targets, timing, physical = episode_args()
            observations[3][key][0] += 1e-6
            with self.assertRaisesRegex(PairingError, "hidden state jump"):
                build_paired_episode(observations, targets, timing, NAMES, source())

    def test_time_normalization_gaps_and_skipped_command_cannot_pass(self):
        for mutation in ("time", "interval", "gap", "sequence"):
            observations, targets, timing, physical = episode_args()
            if mutation == "time":
                observations[2]["timestamp_s"] += 0.001
            elif mutation == "interval":
                timing[2]["end_s"] -= 0.001
            elif mutation == "gap":
                timing[2]["end_s"] -= 0.001
                timing[2]["post_time_s"] -= 0.001
            else:
                timing[2]["sequence_index"] += 1
            with self.assertRaises(PairingError):
                build_paired_episode(observations, targets, timing, NAMES, source(), expected_interval_s=0.066)

    def test_no_post_state_cannot_claim_continuity(self):
        observations, targets, timing, physical = episode_args()
        for row in timing:
            for key in ("post_joint_position", "post_joint_velocity", "post_time_s"):
                del row[key]
        arrays = build_paired_episode(observations, targets, timing, NAMES, source())
        self.assertFalse(arrays["pairing_valid"].any())
        self.assertFalse(json.loads(str(arrays["metadata_json"]))["pairing_eligible"])

    def test_restart_hold_is_never_replaced_by_future_motion(self):
        observations, targets, timing, physical = episode_args()
        timing[0]["restart_wait"] = True
        targets[0] = observations[0]["joint_position"].copy()
        arrays = build_paired_episode(observations, targets, timing, NAMES, source())
        self.assertFalse(arrays["pairing_valid"][0])
        self.assertFalse(arrays["correction_nonzero"][0])
        np.testing.assert_array_equal(arrays["action_joint_position_rad"][0], observations[0]["joint_position"])

    def test_shared_chunk_encoder_keeps_one_anchor_and_never_crosses_segments(self):
        contract = ActionContract(tuple(NAMES), 1 / 0.066, 0.002, (0.08,) * 31, (1.0,) * 31)
        arrays = build()
        result = encode_paired_chunks(arrays, contract)
        self.assertEqual(result["anchor_record_index"].tolist(), [0, 1, 2])
        self.assertTrue(result["action_contract_consistent"].all())
        np.testing.assert_allclose(result["action_delta_chunk_rad"][1, :, 0], np.arange(8) * 0.01 + 0.005)
        observations, targets, timing, physical = episode_args()
        for row in timing[5:]:
            row["state_segment_id"] = 1
        # A declared reset is still a boundary; eight-step chunks cannot span it.
        arrays = build_paired_episode(observations, targets, timing, NAMES, source())
        self.assertEqual(len(encode_paired_chunks(arrays, contract)["anchor_record_index"]), 0)

    def test_future_teacher_targets_outside_anchor_contract_are_flagged_not_clipped(self):
        contract = ActionContract(tuple(NAMES), 1 / 0.066, 0.002, (0.04,) * 31, (1.0,) * 31)
        arrays = build()
        before = arrays["action_joint_position_rad"].copy()
        result = encode_paired_chunks(arrays, contract)
        self.assertFalse(result["action_contract_consistent"].any())
        np.testing.assert_array_equal(arrays["action_joint_position_rad"], before)
        self.assertAlmostEqual(result["action_delta_chunk_rad"][0, 7, 0], 0.075)

    def test_sensorless_missing_lineage_and_missing_physics_stay_diagnostic(self):
        observations, targets, timing, physical = episode_args()
        arrays = build_paired_episode(observations, targets, timing, NAMES, {})
        metadata = json.loads(str(arrays["metadata_json"]))
        self.assertFalse(metadata["pairing_eligible"])
        self.assertFalse(metadata["sensor_complete"])
        self.assertFalse(arrays["frame_physical_positive"].any())


class RawAuditTests(unittest.TestCase):
    def test_historical_sparse_recording_is_diagnostic_and_raw_hash_stays_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, arrays = legacy(root)
            before = path.read_bytes()
            report = audit_raw_pairing(root, config(root), run_dir=root / "runs" / "audit", records=[{"raw_path": str(path)}])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(report["summary"]["training_eligible_count"], 0)
            row = report["records"][0]
            self.assertEqual(row["pairing_status"], "historical_sampled_only")
            self.assertFalse(row["pairing_eligible"])
            with np.load(row["derived_evidence_path"], allow_pickle=False) as z:
                expected = select_action31(arrays["action_joint_position_rad"], arrays["joint_names"], NAMES)
                np.testing.assert_array_equal(z["next_sampled_command_target_rad"], expected[1:])
                self.assertEqual(z["recorded_joint_position_rad"].shape, (4, 31))
            with self.assertRaisesRegex(PairingError, "immutable"):
                audit_raw_pairing(root, config(root), run_dir=root / "runs" / "audit", records=[])

    def test_takeover_checks_parent_and_real_teacher_start_not_overwritten_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, arrays = legacy(root, "G1LANG-659_direct")
            q = select_action31(arrays["joint_position_rad"], arrays["joint_names"], NAMES)
            visited_q = q[0].copy()
            visited_q[0] -= 0.01
            np.savez_compressed(path.parent / "policy_visited_state.npz", joint_names=NAMES,
                                observation_joint_position_rad=visited_q[None], observation_joint_velocity_rad_s=np.ones((1, 31)))
            parent = root / "outputs" / "parent" / "trajectory.npz"
            parent.parent.mkdir()
            np.savez_compressed(parent, joint_names=NAMES, observation_joint_position_rad=visited_q[None],
                                observation_joint_velocity_rad_s=np.ones((1, 31)))
            row = {"raw_path": str(path), "source_rollout": str(parent), "takeover_frame": 0}
            report = audit_raw_pairing(root, config(root), records=[row])
            evidence = report["records"][0]["takeover_audit"]
            self.assertTrue(evidence["parent_exact"])
            self.assertFalse(evidence["visited_start_preserved"])
            self.assertAlmostEqual(evidence["teacher_observed_start_position"]["max_abs_error"], 0.01, places=4)
            self.assertEqual(report["summary"]["takeover_start_mismatch_count"], 1)

    def test_new_schema_revalidated_and_forged_frame_label_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "outputs" / "episode" / "expert_pick_place_episode.npz"
            path.parent.mkdir(parents=True)
            arrays = build(sensors=True)
            np.savez_compressed(path, **arrays)
            cfg = config(root)
            report = audit_raw_pairing(root, cfg, records=[{"raw_path": str(path)}])
            self.assertEqual(report["summary"]["error_count"], 0)
            self.assertEqual(report["records"][0]["pairing_status"], "exact_control_interval")
            self.assertEqual(report["records"][0]["frame_positive_count"], 2)
            arrays["frame_physical_positive"][0] = True
            np.savez_compressed(path, **arrays)
            report = audit_raw_pairing(root, cfg, records=[{"raw_path": str(path)}])
            self.assertEqual(report["summary"]["error_count"], 1)
            self.assertFalse(report["records"][0]["pairing_eligible"])

    def test_new_schema_missing_safety_stays_diagnostic_without_raw_rewrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "outputs" / "episode" / "expert_pick_place_episode.npz"
            path.parent.mkdir(parents=True)
            arrays = build()
            del arrays["physical_numeric_finite"]
            del arrays["physical_joint_safe"]
            arrays["frame_physical_positive"][:] = False
            np.savez_compressed(path, **arrays)
            before = path.read_bytes()
            report = audit_raw_pairing(root, config(root), records=[{"raw_path": str(path)}])
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(report["summary"]["error_count"], 0)
            self.assertEqual(report["records"][0]["frame_positive_count"], 0)
            self.assertIn("physical_safety_unproven", report["records"][0]["review_issues"])

    def test_new_schema_safety_arrays_reject_truthy_integer_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "outputs" / "episode" / "expert_pick_place_episode.npz"
            path.parent.mkdir(parents=True)
            arrays = build()
            arrays["physical_joint_safe"] = np.ones(10, dtype=np.int64)
            np.savez_compressed(path, **arrays)
            report = audit_raw_pairing(root, config(root), records=[{"raw_path": str(path)}])
            self.assertEqual(report["summary"]["error_count"], 1)
            self.assertEqual(report["records"][0]["pairing_status"], "unproven")

    def test_audit_preserves_missing_safety_provenance_after_false_masks_are_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "outputs" / "episode" / "expert_pick_place_episode.npz"
            path.parent.mkdir(parents=True)
            observations, targets, timing, physical = episode_args()
            for row in physical:
                del row["finite"]
            arrays = build_paired_episode(observations, targets, timing, NAMES, source(), physical_evidence=physical)
            np.savez_compressed(path, **arrays)
            report = audit_raw_pairing(root, config(root), records=[{"raw_path": str(path)}])
            self.assertEqual(report["summary"]["error_count"], 0)
            self.assertEqual(report["records"][0]["frame_positive_count"], 0)
            self.assertIn("physical_safety_unproven", report["records"][0]["review_issues"])

    def test_active_split_leakage_blocks_new_schema_pairing_eligibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = config(root)
            records = []
            for name, split in (("first", "train"), ("second", "test")):
                path = root / "outputs" / name / "expert_pick_place_episode.npz"
                path.parent.mkdir(parents=True)
                arrays = build()
                metadata = json.loads(str(arrays["metadata_json"]))
                metadata["source"] = source(str(root), split)
                arrays["metadata_json"] = np.asarray(json.dumps(metadata))
                np.savez_compressed(path, **arrays)
                records.append({"raw_path": str(path)})
            report = audit_raw_pairing(root, cfg, records=records)
            self.assertTrue(report["split_leakage"])
            self.assertEqual(report["summary"]["pairing_eligible_count"], 0)
            self.assertEqual(report["summary"]["training_eligible_count"], 0)

    def test_unsafe_npz_pickle_and_missing_timestamp_remain_unproven(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path, arrays = legacy(root)
            arrays["object_payload"] = np.asarray([{"unsafe": True}], dtype=object)
            np.savez_compressed(path, **arrays)
            report = audit_raw_pairing(root, config(root), records=[{"raw_path": str(path)}])
            self.assertEqual(report["summary"]["error_count"], 1)
            self.assertEqual(report["records"][0]["pairing_status"], "unproven")


if __name__ == "__main__":
    unittest.main()
