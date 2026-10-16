import copy
import json
from pathlib import Path
import tempfile
import unittest

from g1_vla.config import validate_config
from g1_vla.contracts import ActionContract
from g1_vla.rl_protocol import RL_TEACHER_DEFAULTS, file_hash, validate_rl_settings, verify_rl_assets
from g1_vla.rl_teacher import windows_to_wsl, _smoke_admission


PROJECT = Path(__file__).resolve().parents[1]


class RLProtocolTests(unittest.TestCase):
    def test_legacy_config_is_readable_without_enabling_RL_or_mutating_its_caller(self):
        config = json.loads((PROJECT / "configs/experiment.json").read_text())
        config.pop("rl_teacher")
        result = validate_config(config)
        self.assertNotIn("rl_teacher", config)
        self.assertIs(result["rl_teacher"]["enabled"], False)
        metadata = json.loads((PROJECT / "resources/g1_action31_train_root_stub/meta/info.json").read_text())
        self.assertEqual(ActionContract.from_config(config, metadata=metadata).contract_sha256,
                         ActionContract.from_config(result, metadata=metadata).contract_sha256)

    def test_RL_cannot_silently_change_physics_action_bounds_or_original_scoring_scope(self):
        for key, value in (("physics_dt_s", .004), ("decimation", 16), ("cube_mass_kg", .01),
                           ("qualification_required", False), ("scope", "formal_original20"),
                           ("max_iterations", 401), ("num_envs", 513), ("max_wall_time_s", 3601),
                           ("seed", True), ("smoke_steps", 32), ("action_scale_rad", .2)):
            config = copy.deepcopy(RL_TEACHER_DEFAULTS)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_rl_settings(config)

    def test_missing_asset_pin_and_unpaired_checkpoint_are_rejected(self):
        config = copy.deepcopy(RL_TEACHER_DEFAULTS)
        config["enabled"] = True
        with self.assertRaises(ValueError):
            validate_rl_settings(config)
        config["enabled"] = False
        config["checkpoint_path"] = "random.pt"
        with self.assertRaises(ValueError):
            validate_rl_settings(config)

    def test_native_paths_cannot_be_unc_relative_or_parent_traversal(self):
        self.assertEqual(windows_to_wsl("D:/G1-RL-Teacher/runs/x"), Path("/mnt/d/G1-RL-Teacher/runs/x"))
        for path in ("relative/path", "D:relative", "D:/../Windows", "//server/share"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                windows_to_wsl(path)

    def test_train_requires_matching_smoke_not_a_successful_script_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {"artifacts": {"runs_dir": directory}}
            report = Path(directory) / "runs/smoke/reports/rl_teacher.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({"stage": "rl-smoke", "passed": True, "release_sha256": "old"}))
            with self.assertRaises(ValueError):
                _smoke_admission(config, "new")
            self.assertEqual(_smoke_admission(config, "old")["report_path"], str(report))

    def test_loading_an_unlisted_robot_is_rejected_even_when_other_assets_match(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            robot = assets / "robot.xml"
            robot.write_text("<mujoco/>")
            seed = assets / "seed.json"
            seed.write_text("{}")
            rows = [{"path": file.name, "bytes": file.stat().st_size, "sha256": file_hash(file)}
                    for file in (robot, seed)]
            manifest = assets / "manifest.json"
            manifest.write_text(json.dumps({"files": rows}))
            config = {"paths": {"root": directory}, "rl_teacher": copy.deepcopy(RL_TEACHER_DEFAULTS)}
            config["rl_teacher"].update(enabled=True, asset_path=str(assets / "different.xml"),
                seed_state_path=str(seed), asset_manifest_path=str(manifest),
                seed_state_sha256=file_hash(seed), asset_manifest_sha256=file_hash(manifest))
            with self.assertRaisesRegex(ValueError, "exact official USD"):
                verify_rl_assets(config)


if __name__ == "__main__":
    unittest.main()
