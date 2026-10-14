import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from g1_vla.artifacts import classify_artifact, inventory_artifacts, write_manifest


class ArtifactTests(unittest.TestCase):
    def test_raw_failures_preserved_even_when_training_excluded(self):
        row = classify_artifact("outputs/G1LANG-632/failed/raw.npz")
        self.assertTrue(row["protected"])
        self.assertEqual(row["action"], "keep")
        self.assertIs(row["training_eligible"], False)
        self.assertIs(row["cleanup_eligible"], False)

    def test_cache_candidate_never_grants_deletion_eligibility(self):
        for number in ("619", "624", "632", "636", "642", "648", "650"):
            for extension in ("pt", "npy"):
                row = classify_artifact("outputs/g1lang_{}/dataset.{}".format(number, extension))
                self.assertEqual(row["action"], "cleanup_candidate")
                self.assertIs(row["cleanup_eligible"], False)
                self.assertTrue(row["cleanup_blockers"])

    def test_historical_archive_and_source_rollouts(self):
        for number in ("625", "665", "686b"):
            row = classify_artifact("outputs/G1LANG-{}/dataset.pt".format(number))
            self.assertEqual(row["action"], "archive")
            self.assertIs(row["training_eligible"], False)
        for number in ("639", "662", "673", "679", "682"):
            self.assertTrue(classify_artifact("outputs/G1LANG-{}/video.mp4".format(number))["protected"])
        self.assertEqual(classify_artifact("outputs/2026-10-14/625/cache.npy")["action"], "review")

    def test_checkpoint_and_recovery_specs_protected(self):
        self.assertTrue(classify_artifact("outputs/G1LANG-624/checkpoint/model.pt", 12)["protected"])
        self.assertTrue(classify_artifact("outputs/G1LANG-684/recovery_spec.json")["protected"])
        self.assertTrue(classify_artifact("outputs/G1LANG-690/summary.json")["protected"])

    def test_inventory_reads_allowlisted_metadata_and_hashes_large_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "outputs" / "G1LANG-624"
            output.mkdir(parents=True)
            raw = output / "failed.npz"
            raw.write_bytes(b"raw failure" * 100)
            cache = output / "dataset.pt"
            cache.write_bytes(b"derived cache" * 100)
            summary = output / "summary.json"
            summary.write_text(json.dumps({"source_rollout": "G1LANG-639/raw.npz",
                                           "rebuild_recipe": "recipes/cache.json", "token": "DO-NOT-ECHO",
                                           "rebuild_verified": True}), encoding="utf-8")
            report = inventory_artifacts(root, ["outputs"], large_file_threshold=16)
            rows = {Path(row["path"]).name: row for row in report["records"]}
            self.assertEqual(rows["failed.npz"]["sha256"], hashlib.sha256(raw.read_bytes()).hexdigest())
            self.assertEqual(rows["dataset.pt"]["hash_status"], "skipped_large")
            self.assertEqual(rows["dataset.pt"]["parent_sources"], ["G1LANG-639/raw.npz"])
            self.assertEqual(rows["dataset.pt"]["recipe_paths"], ["recipes/cache.json"])
            self.assertIs(rows["dataset.pt"]["cleanup_eligible"], False)
            self.assertNotIn("DO-NOT-ECHO", json.dumps(report))
            self.assertEqual(cache.read_bytes(), b"derived cache" * 100)

    def test_explicit_external_journal_and_malformed_summary_do_not_abort(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            output = root / "outputs"
            output.mkdir()
            (output / "summary.json").write_text("{broken", encoding="utf-8")
            journal = Path(outside) / "Lerobot.md"
            journal.write_text("Protected journal", encoding="utf-8")
            report = inventory_artifacts(root, ["outputs", journal, "missing"])
            self.assertEqual(report["summary"]["artifact_count"], 2)
            self.assertEqual(report["summary"]["error_count"], 2)
            journal_row = next(row for row in report["records"] if row["path"] == str(journal))
            self.assertTrue(journal_row["external"])
            self.assertTrue(journal_row["protected"])
            self.assertEqual(journal_row["action"], "keep")
            self.assertEqual(journal_row["sha256"], hashlib.sha256(journal.read_bytes()).hexdigest())

    def test_symlinks_are_not_followed(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            output = root / "outputs"
            output.mkdir()
            target = Path(outside) / "secret.npz"
            target.write_bytes(b"outside")
            os.symlink(str(target), str(output / "linked.npz"))
            os.symlink(outside, str(output / "linked_dir"))
            report = inventory_artifacts(root, ["outputs"])
            self.assertEqual(report["summary"]["artifact_count"], 2)
            self.assertTrue(all(row["hash_status"] == "skipped_symlink" for row in report["records"]))
            self.assertTrue(all(row["protected"] for row in report["records"]))

    def test_relative_external_journal_is_normalized_without_losing_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "projects" / "G1-VLA-Grasping"
            root.mkdir(parents=True)
            notes = base / "notes"
            notes.mkdir()
            journal = notes / "Lerobot.md"
            journal.write_text("shared log", encoding="utf-8")
            report = inventory_artifacts(root, ["../../notes/Lerobot.md"])
            row = report["records"][0]
            self.assertEqual(row["path"], str(journal))
            self.assertTrue(row["external"])
            self.assertTrue(row["protected"])

    def test_rebuild_reports_are_excluded_from_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "outputs" / "rebuild" / "runs"
            generated.mkdir(parents=True)
            (generated / "summary.json").write_text("{}", encoding="utf-8")
            report = inventory_artifacts(root, ["outputs"])
            self.assertEqual(report["summary"]["artifact_count"], 0)
            explicit = inventory_artifacts(root, ["outputs"], exclude_paths=[])
            self.assertEqual(explicit["summary"]["artifact_count"], 1)

    def test_manifest_never_overwrites_and_jsonl_preserves_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inventory.jsonl"
            manifest = {"read_only": True, "records": [{"path": "raw.npz"}], "errors": []}
            write_manifest(manifest, path)
            old = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_manifest({"records": []}, path)
            self.assertEqual(path.read_bytes(), old)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["record_type"], "inventory_metadata")
            self.assertTrue(rows[0]["read_only"])
            self.assertEqual(rows[1]["record_type"], "artifact")


if __name__ == "__main__":
    unittest.main()
