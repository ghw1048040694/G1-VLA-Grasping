import json
import tempfile
import unittest
from pathlib import Path

from g1_vla.data_ingest import ingest_raw_registry
from g1_vla.data_registry import assign_group_splits, validate_registry, write_registry


def scene(camera="head", robot="pelvis", x="0.40", quat="1 0 0 0"):
    return ('<mujoco><worldbody><body name="{}" pos="0 0 0.793"/>'
            '<camera name="{}" pos="1 2 3"/><body name="green_cube" pos="{} 0.1 0.795" quat="{}"/>'
            '<body name="blue_receptacle" pos="0.59 0 0.76"/></worldbody></mujoco>').format(robot, camera, x, quat)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def make_policy(root, name, x="0.4"):
    directory = root / "outputs" / name / "episode_0001"
    directory.mkdir(parents=True)
    (directory / "trajectory.npz").write_bytes(b"raw-policy-not-a-real-numpy-file")
    (directory / "scene.xml").write_text(scene(x=x), encoding="utf-8")
    write_json(directory / "summary.json", {"target_object": "green_cube", "passed": False})
    return directory


def make_recovery(root, name, source, safe=False, measured=False, outcome=True, episode="episode_0001"):
    directory = root / "outputs" / name / "recovery_0001_green_cube"
    directory.mkdir(parents=True)
    (directory / "expert_pick_place_episode.npz").write_bytes(b"raw-expert-not-a-real-numpy-file")
    (directory / "policy_visited_state.npz").write_bytes(b"visited-not-a-real-numpy-file")
    (directory / "g1_language_pick_place.xml").write_text(scene(), encoding="utf-8")
    write_json(directory / "recovery_spec.json", {"source_rollout": source, "source_episode": episode,
                                                "source_frame": 74, "safe_waypoint_recovery": safe})
    payload = {"target_object": "green_cube", "language_instruction": "lift the green block", "passed": outcome,
               "initial_direct_continuation": {"enabled": not safe},
               "lift_height_m": 0.22, "lift_hold_duration_s": 10.0,
               "grasp_physics_audit": {"physical_contact_steps": 2744, "target_lift_m": 0.249,
                                       "assisted_constraint_active_steps": 0, "physical_grasp_passed": outcome}}
    if measured:
        payload["grasp_physics_audit"]["measured_hold_seconds"] = 0.6
    write_json(directory / "language_pick_place_summary.json", payload)
    return directory


class RawIngestTests(unittest.TestCase):
    def test_actual_parent_and_regimes_registered_without_loading_npz(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = make_policy(root, "G1LANG-682_parent")
            direct = make_recovery(root, "G1LANG-683_direct", str(parent))
            waypoint = make_recovery(root, "G1LANG-684_waypoint", str(parent), safe=True)
            report = ingest_raw_registry(root)
            self.assertEqual(report["summary"]["record_count"], 5)
            self.assertEqual(report["summary"]["error_count"], 0)
            direct_row = next(row for row in report["records"] if row["raw_path"] == str(direct / "expert_pick_place_episode.npz"))
            waypoint_row = next(row for row in report["records"] if row["raw_path"] == str(waypoint / "expert_pick_place_episode.npz"))
            self.assertEqual(direct_row["source_rollout"], str(parent / "trajectory.npz"))
            self.assertEqual(direct_row["regime"], "direct_continuation")
            self.assertEqual(waypoint_row["regime"], "safe_waypoint")
            self.assertEqual(direct_row["group_id"], waypoint_row["group_id"])
            self.assertTrue(all(row["training_eligible"] is False and row["is_positive"] is False for row in report["records"]))
            self.assertTrue(validate_registry(report["records"])["ok"])

    def test_commanded_hold_does_not_become_measured_hold_or_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = make_policy(root, "G1LANG-682_parent")
            child = make_recovery(root, "G1LANG-684_waypoint", str(parent), safe=True)
            row = next(row for row in ingest_raw_registry(root)["records"] if row["raw_path"] == str(child / "expert_pick_place_episode.npz"))
            self.assertIsNone(row["physical_gate"]["hold_seconds"])
            self.assertIn("measured_hold_missing", row["review_issues"])
            self.assertFalse(row["strict_physical_success"])
            self.assertEqual(row["split"], "diagnostic")
            self.assertEqual(row["historical_outcome"], "success")

    def test_complete_historical_gate_still_cannot_become_new_training_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = make_policy(root, "G1LANG-682_parent")
            child = make_recovery(root, "G1LANG-684_waypoint", str(parent), safe=True, measured=True)
            row = next(row for row in ingest_raw_registry(root)["records"] if row["raw_path"] == str(child / "expert_pick_place_episode.npz"))
            self.assertTrue(row["strict_physical_success"])
            self.assertEqual(row["split"], "historical_regression")
            self.assertFalse(row["is_positive"])
            self.assertFalse(row["training_eligible"])

    def test_failure_is_retained_with_successful_contact_and_missing_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = make_recovery(root, "G1LANG-663_failed", "outputs/missing_parent/episode_0001", outcome=False)
            report = ingest_raw_registry(root)
            self.assertEqual(len(report["records"]), 2)
            row = next(row for row in report["records"] if row["raw_path"] == str(child / "expert_pick_place_episode.npz"))
            self.assertEqual(row["parent_resolution"], "parent_unresolved")
            self.assertIsNone(row["source_rollout"])
            self.assertIsNone(row["group_id"])
            self.assertFalse(row["expert_success"])
            self.assertEqual(row["historical_outcome"], "failure")
            self.assertTrue(row["physical_gate"]["contact"])
            self.assertTrue(validate_registry(report["records"])["ok"])
            write_registry(report["records"], root / "raw_registry.jsonl")
            assigned = assign_group_splits(report["records"])
            self.assertTrue(all(row["split"] == "diagnostic" and row["group_id"] is None for row in assigned))

    def test_retired_project_path_maps_only_to_an_existing_current_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "G1-VLA-Grasping"
            parent = make_policy(root, "G1LANG-682_parent")
            old = "/home/ubuntu/VLA-Learning/projects/G1-UpperBody/" + str(parent.relative_to(root))
            child = make_recovery(root, "G1LANG-683_direct", old)
            row = next(row for row in ingest_raw_registry(root)["records"] if row["raw_path"] == str(child / "expert_pick_place_episode.npz"))
            self.assertEqual(row["source_rollout"], str(parent / "trajectory.npz"))
            broken = json.loads((child / "recovery_spec.json").read_text(encoding="utf-8"))
            broken["source_rollout"] = old.replace("G1LANG-682_parent", "G1LANG-999_missing")
            write_json(child / "recovery_spec.json", broken)
            row = next(row for row in ingest_raw_registry(root)["records"] if row["raw_path"] == str(child / "expert_pick_place_episode.npz"))
            self.assertIsNone(row["source_rollout"])

    def test_episode_basename_cannot_collide_between_source_rollouts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = make_policy(root, "G1LANG-679_first")
            second = make_policy(root, "G1LANG-682_second")
            child_one = make_recovery(root, "G1LANG-680_first_child", str(first))
            child_two = make_recovery(root, "G1LANG-683_second_child", str(second))
            rows = ingest_raw_registry(root)["records"]
            one = next(row for row in rows if row["raw_path"] == str(child_one / "expert_pick_place_episode.npz"))
            two = next(row for row in rows if row["raw_path"] == str(child_two / "expert_pick_place_episode.npz"))
            self.assertNotEqual(one["source_rollout"], two["source_rollout"])
            self.assertNotEqual(one["group_id"], two["group_id"])

    def test_layout_identity_uses_object_poses_and_ignores_camera_and_robot(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = make_policy(root, "G1LANG-679_first")
            second = make_policy(root, "G1LANG-682_second")
            (second / "scene.xml").write_text(scene(camera="new_camera", robot="robot_new", quat="-1 0 0 0"), encoding="utf-8")
            rows = ingest_raw_registry(root)["records"]
            self.assertEqual(rows[0]["layout_fingerprint"], rows[1]["layout_fingerprint"])
            (second / "scene.xml").write_text(scene(x="0.41"), encoding="utf-8")
            rows = ingest_raw_registry(root)["records"]
            self.assertNotEqual(rows[0]["layout_fingerprint"], rows[1]["layout_fingerprint"])

    def test_no_layout_fingerprint_is_fabricated_from_root_or_bare_positions(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = make_policy(root, "G1LANG-682_parent")
            # Remove only the test fixture to represent a missing historical XML.
            (parent / "scene.xml").unlink()
            write_json(parent / "summary.json", {"object_initial_position_m": {"green_cube": [0.4, 0.1, 0.795]}, "passed": False})
            row = ingest_raw_registry(root)["records"][0]
            self.assertIsNone(row["layout_fingerprint"])
            self.assertIsNone(row["group_id"])
            self.assertIn("layout_unresolved", row["review_issues"])

    def test_explicit_layout_json_can_supply_complete_poses(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = make_policy(root, "G1LANG-682_parent")
            (parent / "scene.xml").unlink()
            write_json(parent / "layout.json", {"objects": {"green_cube": {"pos": [0.4, 0.1, 0.795], "quat": [1, 0, 0, 0]}}})
            row = ingest_raw_registry(root)["records"][0]
            self.assertIsNotNone(row["layout_fingerprint"])
            self.assertEqual(row["layout_source"], str(parent / "layout.json"))

    def test_malformed_summary_and_scene_do_not_discard_the_raw_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = make_policy(root, "G1LANG-682_parent")
            (parent / "summary.json").write_text("{broken", encoding="utf-8")
            (parent / "scene.xml").write_text("<broken", encoding="utf-8")
            report = ingest_raw_registry(root)
            self.assertEqual(len(report["records"]), 1)
            self.assertEqual(report["summary"]["error_count"], 2)
            self.assertEqual(report["records"][0]["split"], "diagnostic")

    def test_diagnostic_incomplete_provenance_is_forbidden_in_active_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            child = make_recovery(root, "G1LANG-684_waypoint", "outputs/missing_parent/episode_0001", safe=True)
            row = next(row for row in ingest_raw_registry(root)["records"] if row["raw_path"] == str(child / "expert_pick_place_episode.npz"))
            self.assertTrue(validate_registry([row])["ok"])
            row["split"] = "train"
            self.assertFalse(validate_registry([row])["ok"])

    def test_generated_rebuild_outputs_not_imported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_policy(root, "rebuild")
            self.assertEqual(ingest_raw_registry(root)["summary"]["record_count"], 0)


if __name__ == "__main__":
    unittest.main()
