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

    def test_config_hash_is_canonical(self):
        self.assertEqual(canonical_hash({"x": 1, "y": 2}), canonical_hash({"y": 2, "x": 1}))


if __name__ == "__main__":
    unittest.main()
