import copy
import tempfile
import unittest
from pathlib import Path

from g1_vla.data_registry import (RegistryValidationError, assign_group_splits,
                                  detect_split_leakage, group_key, read_registry,
                                  strict_physical_success, validate_registry, write_registry)


def record(source="rollout-1.npz", layout="layout-1", **extra):
    row = {"parent_root": "/evidence", "source_rollout": source,
           "layout_fingerprint": layout, "takeover_frame": 12,
           "regime": "direct_takeover", "physical_gate": {
               "contact": True, "lift_height_m": 0.1, "hold_seconds": 0.5, "assisted": False},
           "is_positive": True}
    row.update(extra)
    return row


class RegistryTests(unittest.TestCase):
    def test_in_box_alone_and_expert_failures_cannot_be_positive(self):
        rows = [record(physical_gate={"in_box": True}), record(expert_success=False)]
        report = validate_registry(rows)
        self.assertFalse(report["ok"])
        self.assertEqual(sum(error["code"] == "invalid_positive" for error in report["errors"]), 2)

    def test_strict_gate_requires_real_contact_lift_hold_and_no_assistance(self):
        self.assertTrue(strict_physical_success(record()))
        for field, value in (("contact", False), ("lift_height_m", 0.09), ("hold_seconds", 0.49),
                             ("assisted", True), ("lift_height_m", float("nan")), ("lift_height_m", True)):
            row = record()
            row["physical_gate"][field] = value
            self.assertFalse(strict_physical_success(row))
        placement = record(task="pick_place")
        self.assertFalse(strict_physical_success(placement))
        placement["physical_gate"].update({"in_box": True, "released": True})
        self.assertTrue(strict_physical_success(placement))

    def test_failed_raw_record_is_valid_when_negative(self):
        row = record(is_positive=False, expert_success=False, physical_gate={"contact": False, "in_box": True})
        self.assertTrue(validate_registry([row])["ok"])

    def test_unknown_provenance_and_missing_takeover_frame_rejected(self):
        self.assertFalse(validate_registry([record(parent_root="")])["ok"])
        self.assertFalse(validate_registry([record(takeover_frame=None)])["ok"])
        with self.assertRaises(RegistryValidationError):
            assign_group_splits([record(layout_fingerprint="")])

    def test_language_and_multiple_takeover_variants_share_group_and_split(self):
        rows = [record(language="pick green", takeover_frame=12),
                record(language="lift the green block", takeover_frame=40),
                record(regime="waypoint_recovery", takeover_frame=44)]
        old = copy.deepcopy(rows)
        self.assertEqual(group_key(rows[0]), group_key(rows[1]))
        assigned = assign_group_splits(rows)
        self.assertEqual(len(set(row["split"] for row in assigned)), 1)
        self.assertEqual(len(set(row["group_id"] for row in assigned)), 1)
        self.assertEqual(rows, old)
        self.assertEqual(assigned[2]["regime"], "waypoint_recovery")

    def test_global_shared_source_or_layout_joins_transitively(self):
        rows = [record(source="same.npz", layout="a"), record(source="same.npz", layout="b"),
                record(source="other.npz", layout="b")]
        assigned = assign_group_splits(rows, train_fraction=0.4, val_fraction=0.3)
        self.assertEqual(len(set(row["split"] for row in assigned)), 1)
        self.assertEqual(len(set(row["group_id"] for row in assigned)), 1)
        leaking = [dict(rows[0], split="train"), dict(rows[1], split="test")]
        self.assertFalse(validate_registry(leaking)["ok"])
        self.assertEqual(detect_split_leakage(leaking)[0]["identity_kind"], "source")

    def test_absolute_parent_source_identity_survives_shard_roots(self):
        rows = [record(source="/original/run.npz", parent_root="/shard-a", layout="a", split="train"),
                record(source="/original/run.npz", parent_root="/shard-b", layout="b", split="test")]
        self.assertTrue(detect_split_leakage(rows))

    def test_split_assignment_is_deterministic_and_balanced_by_component(self):
        rows = [record(source="{}.npz".format(index), layout="layout-{}".format(index)) for index in range(10)]
        first = assign_group_splits(rows, seed=42)
        second = assign_group_splits(reversed(rows), seed=42)
        self.assertEqual({row["source_rollout"]: row["split"] for row in first},
                         {row["source_rollout"]: row["split"] for row in second})
        self.assertEqual(sum(row["split"] == "train" for row in first), 8)
        self.assertEqual(sum(row["split"] == "val" for row in first), 1)
        self.assertEqual(sum(row["split"] == "test" for row in first), 1)

    def test_training_excluded_sources_do_not_enter_any_model_split(self):
        row = record(training_eligible=False)
        assigned = assign_group_splits([row])[0]
        self.assertEqual(assigned["split"], "excluded")
        self.assertFalse(validate_registry([dict(row, split="train")])["ok"])

    def test_invalid_fractions_and_malformed_rows_rejected(self):
        with self.assertRaises(ValueError):
            assign_group_splits([record()], train_fraction=0.9, val_fraction=0.2)
        with self.assertRaises(ValueError):
            assign_group_splits([record()], train_fraction=float("nan"))
        self.assertFalse(validate_registry(["not a row"])["ok"])
        self.assertFalse(validate_registry([record(split=["train"])])["ok"])

    def test_registry_snapshot_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".json", ".jsonl"):
                path = Path(directory) / ("raw_registry" + suffix)
                rows = assign_group_splits([record()])
                write_registry(rows, path)
                self.assertEqual(read_registry(path), rows)
                prior = path.read_bytes()
                with self.assertRaises(FileExistsError):
                    write_registry(rows, path)
                self.assertEqual(path.read_bytes(), prior)


if __name__ == "__main__":
    unittest.main()
