import tempfile
import unittest
from pathlib import Path

from g1_vla.runtime.experiment import ExperimentRun, canonical_hash, write_new_json


class RuntimeTests(unittest.TestCase):
    def test_evidence_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = ExperimentRun(root, root / "runtime", "inventory", {"train": {"enabled": False}}, ["g1-vla"], "protected")
            run.begin()
            result = run.finish({"ok": True}, True)
            self.assertFalse(result["training_started"])
            with self.assertRaises(FileExistsError):
                write_new_json(run.path / "summary.json", {"changed": True})
            with self.assertRaises(FileExistsError):
                ExperimentRun(root, root / "runtime", "inventory", {}, [], "protected").begin()

    def test_run_id_cannot_escape_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                ExperimentRun(Path(directory), Path(directory), "plan", {}, [], "../escape")

    def test_real_training_flag_is_preserved_in_summary_and_finish(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = ExperimentRun(root, root / "runtime", "train", {}, [], "trained")
            run.begin()
            report = run.finish({"training_started": True, "optimizer_steps": 3}, True)
            self.assertTrue(report["training_started"])
            self.assertEqual(report["optimizer_steps"], 3)
            self.assertTrue(json.loads((run.path / "finished.json").read_text())["training_started"])

    def test_config_hash_is_canonical(self):
        self.assertEqual(canonical_hash({"x": 1, "y": 2}), canonical_hash({"y": 2, "x": 1}))

    def test_crashed_native_process_does_not_turn_unknown_training_into_false(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = ExperimentRun(root, root / "runtime", "rl-train", {}, [], "unknown")
            run.begin()
            report = run.finish({"training_started": None, "optimizer_steps": None}, False)
            self.assertIsNone(report["training_started"])
            self.assertIsNone(json.loads((run.path / "finished.json").read_text())["training_started"])


if __name__ == "__main__":
    unittest.main()
