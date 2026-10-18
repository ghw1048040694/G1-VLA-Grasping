"""Geometry observation cannot silently select new controls or physics."""

import copy
import json
from pathlib import Path
import unittest

from g1_vla.rl_probe_protocol import validate_probe_settings


class CollisionGeometrySettingsTests(unittest.TestCase):
    def settings(self):
        p = Path(__file__).resolve().parents[1]/"configs/experiment.json"
        v = json.loads(p.read_text())["rl_diagnostics"]
        v["collision_geometry_audit"] = {"enabled": True, "env_ids": [0],
            "body_names": ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"],
            "require_complete": True}
        return v

    def test_export_is_optional_for_historical_runs_and_does_not_change_effort_contract(self):
        value = self.settings()
        result = validate_probe_settings(value)
        self.assertEqual(result["effort_grasp_controller"], value["effort_grasp_controller"])
        self.assertEqual(result["guided_probe_height_m"], .0025)
        self.assertFalse(result["training_allowed"])
        self.assertEqual(result["optimizer_steps"], 0)
        value.pop("collision_geometry_audit")
        self.assertIsNone(validate_probe_settings(value)["collision_geometry_audit"])
        with self.assertRaises(ValueError):
            validate_probe_settings({"collision_geometry_audit": result["collision_geometry_audit"]})

    def test_wrong_body_scope_incomplete_or_mutating_options_rejected(self):
        for key, change in [("env_ids", [0, 1]), ("env_ids", [False]),
                ("body_names", ["Cube"]), ("enabled", False), ("enabled", 1),
                ("require_complete", False), ("require_complete", 1),
                ("set_hull_vertex_limit", 256)]:
            value = copy.deepcopy(self.settings())
            value["collision_geometry_audit"][key] = change
            with self.subTest(key=key, change=change), self.assertRaises(ValueError):
                validate_probe_settings(value)
