import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from g1_vla.cli import main

PROJECT = Path(__file__).resolve().parents[1]


class PreparationCLITests(unittest.TestCase):
    def make_project(self, directory):
        root = Path(directory)
        (root / "configs").mkdir()
        (root / "resources").mkdir()
        config = json.loads((PROJECT / "configs/experiment.json").read_text())
        metadata = PROJECT / "resources/g1_action31_train_root_stub/meta/info.json"
        destination = root / "resources/info.json"
        destination.write_bytes(metadata.read_bytes())
        config["action"]["joint_metadata_path"] = "resources/info.json"
        (root / "resources/scene.xml").write_text("<mujoco><worldbody/></mujoco>")
        config["scene"]["asset_path"] = "resources/scene.xml"
        config["control"]["max_velocity_rad_s"] = None
        path = root / "configs/experiment.json"
        path.write_text(json.dumps(config))
        return root, path

    def run_cli(self, args):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            status = main(args)
        return status, json.loads(output.getvalue())

    def test_plan_and_dry_run_do_not_write_or_load_simulation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.make_project(directory)
            with patch("g1_vla.cli.inventory_artifacts", side_effect=AssertionError("scan must not start")):
                status, report = self.run_cli(["experiment", "--config", str(path), "--stage", "inventory", "--dry-run"])
            self.assertEqual(status, 0)
            self.assertFalse(report["training_started"])
            self.assertFalse((root / "outputs").exists())
            self.assertFalse(report["ready_for_execution"])

    def test_training_stage_is_a_recorded_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.make_project(directory)
            status, report = self.run_cli(["experiment", "--config", str(path), "--stage", "train", "--run-id", "no-training"])
            self.assertEqual(status, 2)
            self.assertFalse(report["training_started"])
            self.assertEqual(report["training_gate"]["status"], "disabled")
            self.assertFalse(any(root.rglob("*.pt")))

    def test_first_inventory_creates_protection_parent_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.make_project(directory)
            inventory = {"records": [], "errors": [], "summary": {"artifact_count": 0}}
            with patch("g1_vla.cli.inventory_artifacts", return_value=inventory):
                status, report = self.run_cli(["experiment", "--config", str(path), "--stage", "inventory", "--run-id", "first"])
            self.assertEqual(status, 0)
            self.assertTrue((root / ".g1-rebuild/inventory.json").is_file())
            original = (root / ".g1-rebuild/inventory.json").read_bytes()
            with patch("g1_vla.cli.inventory_artifacts", return_value={**inventory, "summary": {"artifact_count": 2}}):
                status, report = self.run_cli(["experiment", "--config", str(path), "--stage", "inventory", "--run-id", "second"])
            self.assertEqual(status, 0)
            self.assertEqual((root / ".g1-rebuild/inventory.json").read_bytes(), original)

    def test_reference_checkpoint_promotion_updates_inventory_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.make_project(directory)
            reference = root / "resources/checkpoints/reference.pt"
            reference.parent.mkdir()
            reference.write_bytes(b"historical reference")
            inventory = {"records": [{"path": str(reference), "relative_path": "resources/checkpoints/reference.pt",
                         "experiment_ids": [], "kind": "file", "sha256": None, "action": "review",
                         "protected": False, "reasons": []}], "errors": [], "summary": {"artifact_count": 1, "actions": {"review": 1}}}
            with patch("g1_vla.cli.inventory_artifacts", return_value=inventory):
                status, report = self.run_cli(["experiment", "--config", str(path), "--stage", "inventory", "--run-id", "reference"])
            self.assertEqual(status, 0)
            self.assertEqual(report["inventory"]["actions"], {"keep": 1})
            protection = json.loads((root / ".g1-rebuild/protected-artifacts.json").read_text())
            self.assertIsNotNone(protection["records"][0]["sha256"])

    def test_old_training_alias_and_forwarded_args_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                main(["train-vla", "--epochs", "20"])
            self.assertEqual(error.exception.code, 2)
            with self.assertRaises(SystemExit):
                main(["experiment", "--config", "fake.json", "--epochs", "20"])

    def test_conditional_training_still_requires_dataset_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.make_project(directory)
            config = json.loads(path.read_text())
            config["train"]["enabled"] = True
            path.write_text(json.dumps(config))
            status, report = self.run_cli(["experiment", "--config", str(path), "--stage", "train"])
            self.assertEqual(status, 2)
            self.assertEqual(report["training_gate"]["status"], "blocked")
            self.assertFalse(report["training_gate"]["optimizer_created"])
            self.assertFalse(any(root.rglob("*.pt")))

    def test_native_compute_success_uses_inference_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.make_project(directory)
            with patch("g1_vla.offline_inference.run_native_act_inference", return_value={
                    "status": "passed_native_compute_probe", "inference_passed": True}):
                status, report = self.run_cli(["experiment", "--config", str(path), "--stage", "offline-infer"])
            self.assertEqual(status, 0)
            self.assertFalse(report["training_started"])
            self.assertEqual(json.loads((Path(report["run_dir"]) / "finished.json").read_text())["status"], "success")

    def test_structural_only_external_audit_requires_camera_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path = self.make_project(directory)
            with patch("g1_vla.external_data.audit_official_sample", return_value={
                    "status": "passed", "bounded_video_probes_passed": 0}):
                status, report = self.run_cli(["experiment", "--config", str(path), "--stage", "external-audit"])
            self.assertEqual(status, 2)
            self.assertFalse(report["training_started"])


if __name__ == "__main__":
    unittest.main()
