"""Raw timing/identity diagnostics and the shared speed-limit application."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from g1_vla.artifacts import sha256_file
from g1_vla.config import load_config
from g1_vla.contracts import ActionContract, read_joint_order
from g1_vla.control import LOWER_BODY_JOINT_NAMES
from g1_vla.control_limits import (
    ControlLimitError, audit_control_limits, command_motion_arrays,
    summarize_command_motion,
)


PROJECT = Path(__file__).resolve().parents[1]
META = PROJECT / "resources/g1_action31_train_root_stub/meta/info.json"
NAMES = read_joint_order(json.loads(META.read_text()))


def arrays(count=13):
    # Deliberately put lower body in the middle and reverse the identity order.
    names = list(reversed(NAMES[:10] + LOWER_BODY_JOINT_NAMES + NAMES[10:]))
    command = np.zeros((count, len(names)), dtype=np.float64)
    for index in range(2, count):
        command[index, names.index("right_elbow_joint")] = (index - 1) * 0.02
    return {
        "joint_names": np.asarray(names), "time_s": np.arange(count) * 0.066,
        "joint_position_rad": command.copy(), "joint_velocity_rad_s": np.zeros_like(command),
        "action_joint_position_rad": command, "task_phase": np.asarray([0, 0] + [2] * (count - 2)),
        "assist_active": np.zeros(count, dtype=np.int64),
        "active_arm_index": np.ones(count, dtype=np.int64),
    }


class ControlLimitTests(unittest.TestCase):
    def test_identity_mapping_and_measured_timestamps(self):
        source = arrays(4)
        source["time_s"] = np.asarray([0.0, 0.1, 0.3, 0.4])
        motion = command_motion_arrays(source, NAMES)
        elbow = NAMES.index("right_elbow_joint")
        self.assertAlmostEqual(motion["values"]["sampled_command_velocity_rad_s"][1, elbow], 0.1)
        self.assertAlmostEqual(motion["values"]["sampled_command_velocity_rad_s"][2, elbow], 0.2)
        self.assertEqual(motion["command"].shape, (4, 31))

    def test_startup_wait_and_recovery_motion_stay_separate(self):
        source = arrays(5)
        source["task_phase"] = np.asarray([0, 0, 0, 2, 2])
        result = summarize_command_motion(source, NAMES)
        self.assertEqual(result["groups"]["startup_wait"]["interval_count"], 1)
        self.assertEqual(result["groups"]["startup_recovery_motion"]["interval_count"], 1)
        self.assertEqual(result["groups"]["effective_grasp_lift"]["interval_count"], 2)
        self.assertEqual(result["groups"]["effective_grasp_lift"]["arm_hand"]["sampled_command_velocity_rad_s"]["hand"]["max"], 0.0)

    def test_missing_duplicate_or_inconsistent_joints_are_rejected(self):
        source = arrays()
        source["joint_names"][0] = source["joint_names"][1]
        with self.assertRaisesRegex(ControlLimitError, "identities"):
            command_motion_arrays(source, NAMES)
        source = arrays()
        source["joint_position_rad"] = source["joint_position_rad"][:-1]
        with self.assertRaisesRegex(ControlLimitError, "timestamps"):
            command_motion_arrays(source, NAMES)

    def test_nonmonotonic_nonfinite_and_assisted_sources_are_rejected(self):
        for mutation, message in (
                (lambda value: value["time_s"].__setitem__(3, value["time_s"][2]), "increasing"),
                (lambda value: value["action_joint_position_rad"].__setitem__((3, 0), np.nan), "finite"),
                (lambda value: value["assist_active"].__setitem__(3, 1), "assisted")):
            source = arrays()
            mutation(source)
            with self.subTest(message=message), self.assertRaisesRegex(ControlLimitError, message):
                command_motion_arrays(source, NAMES)

    def _project(self, root, source=None, passed=True):
        config = load_config(PROJECT / "configs/experiment.json", project_root=root)
        metadata = root / config["action"]["joint_metadata_path"]
        metadata.parent.mkdir(parents=True)
        metadata.write_text(META.read_text())
        config["paths"]["root"] = str(root)
        path = root / "outputs/fixture/expert_pick_place_episode.npz"
        path.parent.mkdir(parents=True)
        summary = {
            "active_arm": "right", "target_object": "green_cube", "grasp_mode": "contact",
            "passed": passed, "controller_gain_scale": {"body": 5, "hand": 6},
            "grasp_physics_audit": {"physical_grasp_passed": passed},
        }
        source = arrays() if source is None else source
        source["metadata_json"] = np.asarray(json.dumps(summary))
        np.savez(path, **source)
        (path.parent / "language_pick_place_summary.json").write_text(json.dumps(summary))
        row = {
            "raw_role": "expert_trajectory", "target_object": "green_cube",
            "relative_raw_path": str(path.relative_to(root)), "raw_sha256": sha256_file(path),
            "historical_outcome": "success" if passed else "failure", "regime": "unknown_recovery",
        }
        registry = root / config["data"]["registry_path"]
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps(row) + "\n")
        return config, path

    def test_recommendation_ceil_is_source_derived_and_unused_joints_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._project(root)
            original = copy.deepcopy(config)
            report = audit_control_limits(root, config)
            self.assertEqual(report["status"], "passed")
            recommendation = report["recommendation"]
            elbow = NAMES.index("right_elbow_joint")
            self.assertEqual(recommendation["max_velocity_rad_s"][elbow], 0.31)
            self.assertEqual(recommendation["max_velocity_rad_s"][0], 0.0)
            self.assertFalse(recommendation["hardware_calibration"])
            self.assertFalse(report["sources"][0]["new_strict_teacher_gate_certified"])
            self.assertFalse(report["sources"][0]["training_eligible"])
            self.assertEqual(config, original)

    def test_failed_sources_cannot_establish_a_successful_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = self._project(root, passed=False)
            report = audit_control_limits(root, config)
            self.assertEqual(report["status"], "incomplete")
            self.assertIsNone(report["recommendation"])
            self.assertEqual(report["audited_right_sources"], 1)

    def test_protected_hash_mismatch_is_not_silently_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, path = self._project(root)
            np.savez(path, **arrays(14))
            report = audit_control_limits(root, config)
            self.assertEqual(report["status"], "incomplete")
            self.assertIn("hash", report["rejected_sources"][0]["reason"])

    def test_shared_contract_speed_zero_holds_previous_command(self):
        config = load_config(PROJECT / "configs/experiment.json")
        config["control"]["max_velocity_rad_s"] = [0.0] * 31
        moving = NAMES.index("right_elbow_joint")
        config["control"]["max_velocity_rad_s"][moving] = 0.31
        contract = ActionContract.from_config(config)
        targets = [[0.05] * 31 for _ in range(8)]
        limited = contract.executable_targets([0.0] * 31, targets, joint_names=NAMES)
        self.assertEqual(len(limited), 4)
        for index, target in enumerate(limited):
            self.assertAlmostEqual(target[0], 0.0)
            self.assertLessEqual(target[moving], (index + 1) * 0.31 * 0.066 + 1e-12)


if __name__ == "__main__":
    unittest.main()
