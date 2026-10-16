"""Math and immutable input checks for the separate PhysX RL curriculum."""

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from g1_vla.rl_environment import (
    ACTION_JOINT_NAMES,
    DEFAULT_SETTINGS,
    RLEnvironmentError,
    _actuator_specifications,
    _seed_state,
    audit_asset_manifest,
    audit_fixed_root_joints,
    consecutive_window_samples,
    minimum_box_clearance,
    migration_pose_errors,
    policy_velocity_limits,
    resolve_settings,
    sensor_body_path_map,
    wxyz_to_xyzw,
)
from g1_vla.control import LOWER_BODY_JOINT_NAMES


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    value = json.loads((ROOT / "configs/experiment.json").read_text(encoding="utf-8"))
    value["paths"]["root"] = str(ROOT)
    # Tests do not depend on a currently selected training run or its asset tree.
    value["rl_teacher"] = {
        "action_joint_names": list(ACTION_JOINT_NAMES),
        "target_lift_height_m": 0.006,
        "training_position_jitter_m": 0.0002,
        "training_yaw_jitter_rad": 0.001,
        "probe_height_cap_m": 0.006,
        "probe_overshoot_tolerance_m": 0.0005,
        "evaluation_seed": 2026101602,
    }
    return value


class PhysicalGeometryTest(unittest.TestCase):
    def test_quaternion_conversion_preserves_rotation_and_rejects_invalid_norm(self):
        self.assertEqual(wxyz_to_xyzw((1, 0, 0, 0)), (0, 0, 0, 1))
        self.assertEqual(wxyz_to_xyzw((0.5, 0.5, 0.5, 0.5)), (0.5, 0.5, 0.5, 0.5))
        with self.assertRaises(RLEnvironmentError):
            wxyz_to_xyzw((1, 1, 0, 0))
        with self.assertRaises(RLEnvironmentError):
            wxyz_to_xyzw((float("nan"), 0, 0, 0))

    def test_rotated_cube_com_height_is_not_bottom_clearance(self):
        q = (0, math.sin(math.pi/8), 0, math.cos(math.pi/8))
        upright = minimum_box_clearance((0, 0, 0.78), (0, 0, 0, 1))
        tilted = minimum_box_clearance((0, 0, 0.78), q)
        self.assertAlmostEqual(upright, 0.005)
        self.assertAlmostEqual(tilted, 0.03 - 0.025*math.sqrt(2))
        self.assertLess(tilted, 0)

    def test_analytic_clearance_matches_all_eight_transformed_corners(self):
        q = (0.2, -0.3, 0.1, math.sqrt(0.86))
        x, y, z, w = q
        row_z = (2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y))
        corners = [0.793 + sum(a*b for a, b in zip(row_z, (sx*.025, sy*.025, sz*.025))) - .75
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        self.assertAlmostEqual(minimum_box_clearance((1, 2, .793), q), min(corners))

    def test_window_uses_elapsed_span_between_first_and_last_sample(self):
        self.assertEqual(consecutive_window_samples(.198, .002), 100)
        self.assertEqual(consecutive_window_samples(.198, .001), 199)
        self.assertGreaterEqual((consecutive_window_samples(.198, .002)-1)*.002, .198)
        for value in (0, -1, float("inf")):
            with self.assertRaises(RLEnvironmentError):
                consecutive_window_samples(value, .002)

    def test_migration_fk_detects_joint_axis_or_quaternion_order_errors(self):
        identity = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        same = migration_pose_errors((.3, .1, .8), identity, (.3, .1, .8), (0, 0, 0, 1))
        self.assertEqual(same, {"position_error_m": 0, "rotation_error_rad": 0})
        wrong = migration_pose_errors((.3, .1, .8), identity, (.303, .104, .8),
                                      (0, 0, math.sin(math.pi/4), math.cos(math.pi/4)))
        self.assertAlmostEqual(wrong["position_error_m"], .005)
        self.assertAlmostEqual(wrong["rotation_error_rad"], math.pi/2)


class ProtocolInputsTest(unittest.TestCase):
    def test_same_named_collision_mesh_cannot_replace_rigid_sensor_body(self):
        name = "right_hand_thumb_0_link"
        body = "/Robot/"+name
        mesh = body+"/collisions/"+name
        self.assertEqual(sensor_body_path_map([(name, body, True), (name, mesh, False)], [name]), {name: body})
        with self.assertRaises(RLEnvironmentError):
            sensor_body_path_map([(name, mesh, False)], [name])
        with self.assertRaises(RLEnvironmentError):
            sensor_body_path_map([(name, body, True), (name, body+"_duplicate", True)], [name])

    def test_imported_world_fix_is_preserved_and_missing_disabled_or_duplicate_fix_rejected(self):
        fixed = {"path": "/Robot/fix_base_joint", "body0": [], "body1": ["/Robot/Geometry/pelvis"], "enabled": True}
        self.assertEqual(audit_fixed_root_joints([fixed])["world_fixed_joint_count"], 1)
        for invalid in ([], [fixed, fixed], [dict(fixed, enabled=False)],
                        [dict(fixed, body1=["/Robot/Geometry/green_cube"]) ]):
            with self.assertRaises(RLEnvironmentError):
                audit_fixed_root_joints(invalid)

    def test_action_limits_follow_metadata_names_in_explicit_policy_order(self):
        config = configuration()
        limits = policy_velocity_limits(config)
        self.assertEqual(len(limits), 14)
        self.assertEqual(limits[:7], (.87, .29, .66, 2.01, .56, 1.71, .51))
        self.assertEqual(limits[7:], (.6, .9, 1.76, 1.47, 1.76, 1.47, 1.76))
        config["action"]["joint_order_sha256"] = "0"*64
        with self.assertRaises(RLEnvironmentError):
            policy_velocity_limits(config)

    def test_local_curriculum_does_not_silently_enable_formal_main_lift(self):
        config = configuration()
        original = copy.deepcopy(config)
        resolved = resolve_settings(config)
        self.assertEqual(config, original)
        self.assertEqual(resolved["lift_height_m"], .006)
        self.assertEqual(resolved["train_xy_jitter_m"], .0002)
        self.assertNotIn("hold_window_s", DEFAULT_SETTINGS)
        config["rl_teacher"]["target_lift_height_m"] = .1
        with self.assertRaises(RLEnvironmentError):
            resolve_settings(config)

    def test_wrong_policy_joint_order_and_physics_cadence_fail(self):
        config = configuration()
        config["rl_teacher"]["action_joint_names"].reverse()
        with self.assertRaises(RLEnvironmentError):
            resolve_settings(config)
        config = configuration()
        config["control"]["physics_substeps"] = 32
        with self.assertRaises(RLEnvironmentError):
            resolve_settings(config)

    def test_seed_requires_actual_object_rotation_and_all_43_joint_identities(self):
        config = configuration()
        names = json.loads((ROOT / config["action"]["joint_metadata_path"]).read_text())["features"]["action"]["names"]
        seed = {"joint_positions_rad": {name: .0 for name in set(names) | set(LOWER_BODY_JOINT_NAMES)},
                "root_pose_xyzw": [0, 0, .793, 0, 0, 0, 1],
                "object_position_world_m": [.35, .055, .77505],
                "object_quaternion_xyzw": [0, 0, 0, 1]}
        config["rl_teacher"]["initial_state"] = seed
        normalized = _seed_state(config)
        self.assertEqual(len(normalized["joint_positions_rad"]), 43)
        self.assertEqual(normalized["cube_pose_xyzw"], [.35, .055, .77505, 0, 0, 0, 1])
        del seed["object_quaternion_xyzw"]
        with self.assertRaises(RLEnvironmentError):
            _seed_state(config)
        seed["object_quaternion_xyzw"] = [0, 0, 0, 1]
        del seed["joint_positions_rad"][names[0]]
        with self.assertRaises(RLEnvironmentError):
            _seed_state(config)

    def test_pd_parameters_are_read_from_source_not_recomputed_generic_gains(self):
        initial = {"joint_positions_rad": {"right_elbow_joint": .3}, "actuators": {
            "right_elbow_joint": {"kp": 123, "kd": 4.5, "force_range_nm": [-25, 25],
                                  "armature_kg_m2": .01, "damping_nm_s_rad": .2}}}
        result = _actuator_specifications(initial)
        self.assertEqual(result["right_elbow_joint"], {"kp": 123, "kd": 4.5, "effort_limit_nm": 25,
                                                      "armature": .01, "dof_damping": .2})
        initial["actuators"]["right_elbow_joint"]["force_range_nm"] = [-20, 25]
        with self.assertRaises(RLEnvironmentError):
            _actuator_specifications(initial)

    def test_asset_manifest_verifies_content_and_rejects_escaping_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "robot.xml"
            asset.write_bytes(b"<mujoco/>")
            entry = {"path": "robot.xml", "bytes": len(asset.read_bytes()),
                     "sha256": hashlib.sha256(asset.read_bytes()).hexdigest()}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"files": [entry]}))
            config = configuration()
            config["rl_teacher"].update(asset_manifest_path=str(manifest),
                                         asset_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual(audit_asset_manifest(config, asset)["verified_file_count"], 1)
            asset.write_bytes(b"<modified/>")
            with self.assertRaises(RLEnvironmentError):
                audit_asset_manifest(config, asset)
            entry["path"] = "../escaped.xml"
            manifest.write_text(json.dumps({"files": [entry]}))
            config["rl_teacher"]["asset_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaises(RLEnvironmentError):
                audit_asset_manifest(config, asset)

    def test_module_import_does_not_import_torch_isaac_or_warp(self):
        code = "import sys; import g1_vla.rl_environment; assert not any(n in sys.modules for n in ('torch', 'warp', 'isaaclab', 'isaacsim'))"
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
