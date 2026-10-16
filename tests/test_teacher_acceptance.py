"""Teacher suite accounting and frozen external identity without any physics."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from g1_vla.artifacts import sha256_file
from g1_vla.config import load_config
from g1_vla.contracts import canonical_sha256, read_joint_order
from g1_vla.data_pairing import build_paired_episode
from g1_vla.teacher_acceptance import TeacherAcceptanceError, read_layout_manifest, run_teacher_acceptance


PROJECT = Path(__file__).resolve().parents[1]
META = PROJECT / "resources/g1_action31_train_root_stub/meta/info.json"
NAMES = read_joint_order(json.loads(META.read_text()))


def manifest():
    return {"schema_version": 1, "scope": "predeclared independent green/right simulation layouts",
            "smoke": {"layout_id": "smoke", "positions_m": {"green_cube": [.34, -.15, .795]}},
            "layouts": [{"layout_id": "layout_{:02d}".format(index),
                         "positions_m": {"green_cube": [round(.33 + index * .001, 8), -.15, .795]}}
                        for index in range(20)]}


def configuration(root):
    config = load_config(PROJECT / "configs/experiment.json", project_root=root)
    config["train"]["enabled"] = False
    config["train"]["backend"] = "current_action31"
    metadata = Path(config["action"]["joint_metadata_path"])
    metadata.parent.mkdir(parents=True)
    metadata.write_text(META.read_text())
    source = Path(config["scene"]["asset_path"])
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("<mujoco><worldbody><body name='pelvis'/></worldbody><equality><weld name='warehouse_fixed_base' body1='pelvis'/></equality></mujoco>")
    config["control"]["max_velocity_rad_s"] = [1.] * 31
    layouts = root / "layouts.json"
    layouts.write_text(json.dumps(manifest()))
    config["eval"]["layouts_path"] = str(layouts)
    return config


def fake_teacher(config, spec, episode_dir, *, sensors=True, safe=True, finite=True, limit=True,
                 scene_matches=True, report_passed=True, short=False):
    """Persist exact synthetic interval evidence; this is not a physical solver."""
    episode_dir = Path(episode_dir)
    frozen = episode_dir.parents[1] / "teacher_layout_manifest.json"
    if not frozen.is_file() or not (episode_dir.parents[1] / "teacher_layout_manifest_provenance.json").is_file():
        raise AssertionError("layout manifest and its hash must be frozen before the teacher starts")
    episode_dir.mkdir(parents=True)
    observations, timing, targets, physical = [], [], [], []
    count = 6 if short else 13
    for index in range(count):
        row = {"joint_position": np.zeros(31), "joint_velocity": np.zeros(31), "timestamp_s": index * .066}
        if sensors:
            row.update(images={key: np.zeros((2, 3, 3), dtype=np.uint8) for key in
                               ("observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist")},
                       finger_sites=np.zeros(108), tactile=[0.], language="lift the green cube")
        observations.append(row)
        targets.append(np.zeros(31))
        timing.append({"start_s": index * .066, "end_s": (index + 1) * .066,
                       "sequence_index": index, "state_segment_id": 0, "restart_wait": index == 0,
                       "post_joint_position": np.zeros(31), "post_joint_velocity": np.zeros(31),
                       "post_time_s": (index + 1) * .066})
        physical.append({"contact": True, "lift_height_m": .2, "hold_seconds": .6, "assisted": False,
                         "finite": True, "joint_safe": True})
    source = {"parent_root": config["paths"]["root"], "source_rollout": str(episode_dir),
              "parent_rollout_id": spec["layout_id"], "layout_fingerprint": spec["layout_fingerprint"],
              "regime": "expert", "takeover_frame": None}
    raw = build_paired_episode(observations, targets, timing, NAMES, source,
                               physical_evidence=physical, expected_interval_s=.066)
    raw_path = episode_dir / "expert_pick_place_episode.npz"
    np.savez_compressed(raw_path, **raw)
    positions = {"green_cube": list(spec["positions_m"]["green_cube"])}
    if not scene_matches:
        positions["green_cube"][0] += .001
    position = " ".join(str(value) for value in positions["green_cube"])
    scene_path = episode_dir / "scene.xml"
    scene_path.write_text("<mujoco><worldbody><body name='pelvis'/><body name='green_cube' pos='{}'/></worldbody><equality><weld name='warehouse_fixed_base' body1='pelvis'/></equality></mujoco>".format(position))
    asset = Path(config["scene"]["asset_path"])
    scene = {"scene": str(scene_path), "scene_sha256": sha256_file(scene_path),
             "source": str(asset), "source_sha256": sha256_file(asset),
             "positions_m": spec["positions_m"], "object_names": ["green_cube"],
             "assisted_grasp": False, "fixed_base_equality": "warehouse_fixed_base"}
    return {"passed": report_passed, "failure": None, "raw_episode": str(raw_path),
            "control_steps": count, "physics_steps": count * 33,
            "joint_limit_violation_substeps": 0 if safe else 1, "all_numeric_finite": finite,
            "maximum_limiter_adjustment_rad": .02,
            "command_limit_audit": {"checked": True, "passed": limit},
            "scene": scene, "layout_fingerprint": spec["layout_fingerprint"]}


class TeacherManifestTests(unittest.TestCase):
    def test_invalid_count_duplicate_id_duplicate_pose_and_nonfinite_fail(self):
        for mode in ("count", "id", "position", "nan", "extra_object", "unsafe_id"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                value = manifest()
                if mode == "count":
                    value["layouts"].pop()
                elif mode == "id":
                    value["layouts"][1]["layout_id"] = value["layouts"][0]["layout_id"]
                elif mode == "position":
                    value["layouts"][1]["positions_m"] = value["layouts"][0]["positions_m"]
                elif mode == "nan":
                    value["layouts"][1]["positions_m"]["green_cube"][0] = float("nan")
                elif mode == "extra_object":
                    value["layouts"][1]["positions_m"]["yellow_rod"] = [0, 0, 0]
                else:
                    value["layouts"][1]["layout_id"] = "../escape"
                path = root / "layouts.json"
                path.write_text(json.dumps(value))
                with self.assertRaises(TeacherAcceptanceError):
                    read_layout_manifest(path, expected_count=20)

    def test_duplicate_json_keys_fail_without_teacher_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            path.write_text('{"schema_version": 1, "schema_version": 1}')
            with self.assertRaisesRegex(TeacherAcceptanceError, "duplicate"):
                read_layout_manifest(path, expected_count=20)


class TeacherAcceptanceTests(unittest.TestCase):
    def test_focus_uses_original_ids_and_never_claims_formal_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration(root)
            config["eval"]["diagnostic_layout_ids"] = ["layout_15", "layout_19"]
            def teacher(configuration, spec, episode_dir):
                return fake_teacher(configuration, spec, episode_dir,
                                    report_passed=spec["layout_id"] != "layout_19")
            with patch("g1_vla.teacher_acceptance.run_teacher_episode", side_effect=teacher):
                report = run_teacher_acceptance(config, root / "runs/focus", focus=True)
            self.assertEqual([row["layout_id"] for row in report["layouts"]], ["layout_15", "layout_19"])
            self.assertEqual(report["summary"]["denominator"], 2)
            self.assertEqual(report["summary"]["required_successes"], 2)
            self.assertEqual(report["summary"]["formal_required_successes"], 18)
            self.assertFalse(report["passed"])
            self.assertFalse(report["formal_gate_evaluated"])
            self.assertIsNone(report["formal_acceptance_passed"])
            self.assertFalse(report["training_started"])

    def test_focus_rejects_unknown_or_duplicate_ids(self):
        for ids in (["unknown"], ["layout_15", "layout_15"]):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = configuration(root)
                config["eval"]["diagnostic_layout_ids"] = ids
                with patch("g1_vla.teacher_acceptance.run_teacher_episode") as teacher:
                    with self.assertRaises(ValueError):
                        run_teacher_acceptance(config, root / "runs/focus", focus=True)
                    teacher.assert_not_called()

    def setUp(self):
        # Suite accounting tests use synthetic pixels and no video encoder.
        patcher = patch("g1_vla.teacher_acceptance.export_teacher_evidence",
                        return_value={"complete": True, "videos": []})
        self.addCleanup(patcher.stop)
        patcher.start()

    def test_full_18_of_20_preserves_exception_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration(root)

            def teacher(configuration, spec, episode_dir):
                if spec["layout_id"] in ("layout_18", "layout_19"):
                    raise RuntimeError("synthetic teacher failure")
                return fake_teacher(configuration, spec, episode_dir)

            with patch("g1_vla.teacher_acceptance.run_teacher_episode", side_effect=teacher):
                report = run_teacher_acceptance(config, root / "runs/full")
            self.assertTrue(report["formal_acceptance_passed"])
            self.assertEqual(report["summary"]["successes"], 18)
            self.assertEqual(report["summary"]["denominator"], 20)
            self.assertEqual(report["summary"]["exception_layout_count"], 2)
            self.assertEqual(report["summary"]["training_eligible_count"], 0)
            self.assertFalse(report["training_started"])
            self.assertEqual(report["summary"]["real_future_chunk_count"], 18 * 5)
            self.assertTrue((root / "runs/full/teacher_acceptance.json").is_file())
            self.assertTrue(all(not row["training_eligible"] and row["split"] == "diagnostic" for row in report["layouts"]))

    def test_smoke_cannot_claim_formal_20_layout_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration(root)
            with patch("g1_vla.teacher_acceptance.run_teacher_episode", side_effect=fake_teacher):
                report = run_teacher_acceptance(config, root / "runs/smoke", smoke=True)
            self.assertTrue(report["passed"])
            self.assertFalse(report["formal_gate_evaluated"])
            self.assertIsNone(report["formal_acceptance_passed"])
            self.assertFalse(report["smoke_counts_toward_formal_acceptance"])
            self.assertEqual(report["summary"]["denominator"], 1)
            self.assertEqual(report["summary"]["formal_required_successes"], 18)
            self.assertEqual(report["summary"]["real_future_chunk_count"], 5)
            self.assertEqual(report["layouts"][0]["chunks"]["restart_wait_interval_count"], 1)

    def test_sensor_missing_limit_safety_finite_and_actual_scene_each_fail(self):
        for option, reason in (("sensors", "sensor_recording_incomplete"),
                               ("limit", "command_limit_audit_failed"),
                               ("safe", "joint_safety_gate_failed"),
                               ("finite", "numeric_finiteness_gate_failed"),
                               ("scene_matches", "scene_layout_audit_failed")):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = configuration(root)
                def teacher(configuration, spec, episode_dir):
                    return fake_teacher(configuration, spec, episode_dir, **{option: False})
                with patch("g1_vla.teacher_acceptance.run_teacher_episode", side_effect=teacher):
                    report = run_teacher_acceptance(config, root / "runs/smoke", smoke=True)
                self.assertFalse(report["passed"])
                self.assertIn(reason, report["layouts"][0]["failure_reasons"])
                self.assertEqual(report["summary"]["denominator"], 1)

    def test_no_end_padding_or_training_promotion_for_short_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration(root)
            def teacher(configuration, spec, episode_dir):
                return fake_teacher(configuration, spec, episode_dir, short=True)
            with patch("g1_vla.teacher_acceptance.run_teacher_episode", side_effect=teacher):
                report = run_teacher_acceptance(config, root / "runs/smoke", smoke=True)
            self.assertEqual(report["summary"]["real_future_chunk_count"], 0)
            chunks = report["layouts"][0]["chunks"]
            self.assertFalse(chunks["end_padding"])
            self.assertFalse(chunks["label_clipping"])
            self.assertFalse(chunks["training_eligible"])

    def test_existing_evidence_never_overwrites_manifest_or_runs_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = configuration(root)
            run_dir = root / "runs/smoke"
            with patch("g1_vla.teacher_acceptance.run_teacher_episode", side_effect=fake_teacher):
                run_teacher_acceptance(config, run_dir, smoke=True)
            before = (run_dir / "teacher_layout_manifest.json").read_bytes()
            with patch("g1_vla.teacher_acceptance.run_teacher_episode") as teacher:
                with self.assertRaisesRegex(TeacherAcceptanceError, "immutable"):
                    run_teacher_acceptance(config, run_dir, smoke=True)
                teacher.assert_not_called()
            self.assertEqual((run_dir / "teacher_layout_manifest.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
