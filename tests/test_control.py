"""Pure control and XML migration tests; no MuJoCo physics is started."""

import copy
import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

from g1_vla.config import load_config
from g1_vla.contracts import ActionContract, ContractError, read_joint_order
from g1_vla.control import (
    LOWER_BODY_JOINT_NAMES, ControlCadence, ControlError, PDController,
    action_joint_bindings, apply_regularized_dynamics, controlled_joints,
    map_action_targets, pd_torque, unitree_gains,
)
from g1_vla.sim.scene import SceneError, build_scene, migrate_actuator_force_ranges, verify_contact_solver
from g1_vla.config import NOSLIP10_SOLVER


PROJECT = Path(__file__).resolve().parents[1]


def configuration():
    return load_config(PROJECT / "configs" / "experiment.json")


def source_joint_names():
    return read_joint_order(json.loads((PROJECT / "resources" / "g1_action31_train_root_stub" / "meta" / "info.json").read_text()))


def fake_control_model():
    names = source_joint_names()
    robot = tuple(LOWER_BODY_JOINT_NAMES) + names
    # Deliberately reorder actuators so a positional slice cannot work.
    actuators = tuple(reversed(robot))
    joints = ("floating_base_joint",) + robot + ("green_cube_freejoint",)
    model = SimpleNamespace(
        nu=43, njnt=len(joints), neq=1,
        opt=SimpleNamespace(timestep=0.002),
        jnt_qposadr=[0] + list(range(7, 50)) + [50],
        jnt_dofadr=[0] + list(range(6, 49)) + [49],
        actuator_trnid=[[joints.index(name), -1] for name in actuators],
        dof_damping=[0.0] * 55, dof_armature=[0.0] * 55,
        jnt_range=[(-3.0, 3.0)] * len(joints), jnt_limited=[False] + [True] * 43 + [False],
        eq_type=[1], eq_obj1id=[1], eq_obj2id=[0], eq_active0=[True],
    )
    data = SimpleNamespace(qpos=[0.0] * 57, qvel=[0.0] * 55, qfrc_bias=[0.0] * 55,
                           ctrl=[0.0] * 43, eq_active=[True])
    kinds = SimpleNamespace(mjOBJ_ACTUATOR="actuator", mjOBJ_JOINT="joint", mjOBJ_EQUALITY="equality", mjOBJ_BODY="body")
    lookup = {"actuator": actuators, "joint": joints,
              "equality": ("warehouse_fixed_base",), "body": ("world", "pelvis")}
    api = SimpleNamespace(mjtObj=kinds, mjtEq=SimpleNamespace(mjEQ_WELD=1), traces=[])
    api.mj_id2name = lambda _model, kind, index: lookup[kind][index]
    api.mj_name2id = lambda _model, kind, name: lookup[kind].index(name) if name in lookup[kind] else -1
    upper_joint_id = joints.index(names[0])
    upper_actuator_id = actuators.index(names[0])

    def fake_step(_model, state):
        # Fixture changes are deterministic inputs, not a physics integrator.
        api.traces.append(state.ctrl[upper_actuator_id])
        state.qpos[model.jnt_qposadr[upper_joint_id]] += 0.001
        state.qfrc_bias[model.jnt_dofadr[upper_joint_id]] += 0.01
    api.mj_step = fake_step
    api.forward_calls = 0
    def fake_forward(_model, state):
        api.forward_calls += 1
    api.mj_forward = fake_forward
    return model, data, api


class ControlTests(unittest.TestCase):
    def test_metadata_mapping_survives_actuator_reordering(self):
        model, data, api = fake_control_model()
        names = source_joint_names()
        controlled = controlled_joints(model, mujoco_api=api)
        bindings = action_joint_bindings(controlled, names)
        self.assertEqual(bindings[0]["actuator_id"], 30)
        holds = {name: 0.0 for name in controlled}
        targets = map_action_targets(list(range(31)), names, controlled, holds)
        self.assertEqual(targets[names[-1]], 30)
        self.assertEqual([targets[name] for name in LOWER_BODY_JOINT_NAMES], [0.0] * 12)
        del controlled[names[0]]
        with self.assertRaisesRegex(ControlError, "mapping mismatch"):
            action_joint_bindings(controlled, names)

    def test_transmission_identity_mismatch_is_rejected(self):
        model, _, api = fake_control_model()
        model.actuator_trnid[0][0] = 1
        with self.assertRaisesRegex(ControlError, "transmission"):
            controlled_joints(model, mujoco_api=api)

    def test_gain_profile_and_scaled_pd(self):
        self.assertEqual(unitree_gains("right_hand_thumb_2_joint"), (1.5, 0.2))
        self.assertEqual(unitree_gains("right_wrist_pitch_joint"), (40.0, 1.5))
        self.assertEqual(unitree_gains("right_elbow_joint"), (80.0, 3.0))
        self.assertEqual(unitree_gains("waist_yaw_joint"), (300.0, 3.0))
        self.assertAlmostEqual(pd_torque("waist_yaw_joint", 0.1, 0.0, 0.2, 1.0,
                                       body_gain_scale=4, hand_gain_scale=6), 119.8)

    def test_regularization_excludes_object_freejoint(self):
        model, _, api = fake_control_model()
        apply_regularized_dynamics(model, mujoco_api=api)
        self.assertEqual(model.dof_damping[49], 0.0)
        self.assertEqual(model.dof_armature[49], 0.0)
        self.assertGreater(model.dof_damping[6], 0.0)

    def test_explicit_actual_cadence(self):
        cadence = ControlCadence.from_config(configuration())
        self.assertEqual(cadence.physics_substeps, 33)
        self.assertEqual(cadence.nominal_control_hz, 15)
        self.assertAlmostEqual(cadence.period_s, 0.066)
        self.assertAlmostEqual(cadence.control_hz, 500.0 / 33.0)
        config = configuration()
        config["control"]["control_hz"] = 15.0
        with self.assertRaisesRegex(ControlError, "actual"):
            ControlCadence.from_config(config)

    def test_unresolved_speed_blocks_before_mujoco_import(self):
        config = configuration()
        config["control"]["max_velocity_rad_s"] = None
        with self.assertRaisesRegex(ContractError, "unresolved"):
            PDController(None, None, config)

    def test_pd_feedback_is_recomputed_each_substep(self):
        model, data, api = fake_control_model()
        config = configuration()
        config["control"]["max_velocity_rad_s"] = 1.0
        controller = PDController(model, data, config, mujoco_api=api)
        targets = controller.execute_chunk([0.0] * 31, [[0.08] * 31] * 8, joint_names=source_joint_names())
        self.assertEqual(len(targets), 4)
        self.assertEqual(len(api.traces), 4 * 33)
        self.assertEqual(api.forward_calls, 4 * 33)
        self.assertNotEqual(api.traces[0], api.traces[1])
        self.assertTrue(all(abs(value) <= 0.08 for row in targets for value in row))

    def test_expert_one_interval_uses_shared_limits_and_observers(self):
        model, data, api = fake_control_model()
        config = configuration()
        config["control"]["max_velocity_rad_s"] = [1.0] * 30 + [0.0]
        controller = PDController(model, data, config, mujoco_api=api)
        events, substeps = [], []
        target = controller.execute_target([1.0] * 31, joint_names=source_joint_names(),
            observer=lambda **event: events.append(event),
            substep_observer=lambda **event: substeps.append(event))
        self.assertEqual(len(api.traces), 33)
        self.assertEqual([event["event"] for event in events], ["before", "after"])
        self.assertEqual(len(substeps), 33)
        self.assertAlmostEqual(target[0], .066)
        self.assertEqual(target[-1], 0.)
        self.assertEqual(events[0]["requested_target"], (1.0,) * 31)
        self.assertEqual(events[0]["executed_target"], target)

    def test_anchor_and_fixed_weld_are_required(self):
        model, data, api = fake_control_model()
        config = configuration()
        config["control"]["max_velocity_rad_s"] = 1.0
        controller = PDController(model, data, config, mujoco_api=api)
        with self.assertRaisesRegex(ControlError, "anchor"):
            controller.execute_chunk([1.0] * 31, [[0.01] * 31] * 8, joint_names=source_joint_names())
        data.eq_active[0] = False
        with self.assertRaisesRegex(ControlError, "weld must be active"):
            controller.execute_chunk([0.0] * 31, [[0.01] * 31] * 8, joint_names=source_joint_names())
        self.assertEqual(api.traces, [])


class SceneTests(unittest.TestCase):
    def test_force_transfer_follows_joint_reference_not_actuator_name(self):
        root = ET.fromstring('<mujoco><worldbody><body><joint name="j" actuatorfrcrange="-2 2"/></body></worldbody>'
                             '<actuator><motor name="different_actuator_name" joint="j"/></actuator></mujoco>')
        report = migrate_actuator_force_ranges(root)
        self.assertEqual(root.find("actuator/motor").get("forcerange"), "-2 2")
        self.assertEqual(report[0]["joint"], "j")
        self.assertIsNone(root.find("worldbody/body/joint").get("actuatorfrcrange"))
        root.find("actuator/motor").set("forcerange", "-1 1")
        root.find("worldbody/body/joint").set("actuatorfrcrange", "-2 2")
        report = migrate_actuator_force_ranges(root)
        self.assertFalse(report[0]["changed"])
        self.assertEqual(report[0]["new_actuator_range"], "-1 1")

    def test_scene_is_exclusive_run_local_and_records_physics_change(self):
        source = PROJECT / "resources" / "g1_warehouse_pregrasp.xml"
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            run_dir.mkdir()
            output = run_dir / "scene.xml"
            manifest = build_scene(source, output, {"green_cube": [0.36, -0.1, 0.795]}, run_dir=run_dir)
            root = ET.parse(output).getroot()
            self.assertEqual(manifest["source_sha256"], before)
            self.assertEqual(len(manifest["actuator_force_transfers"]), 43)
            self.assertFalse(manifest["legacy_equivalence"])
            self.assertTrue(manifest["requires_teacher_revalidation"])
            self.assertEqual([element.get("name") for element in root.find("equality")], ["warehouse_fixed_base"])
            self.assertIsNone(root.find(".//body[@name='red_triangle']"))
            self.assertIsNone(root.find(".//body[@name='yellow_rod']"))
            self.assertIsNotNone(root.find(".//body[@name='green_cube']"))
            self.assertEqual(manifest["historical_unresolved_approximations"]["yellow_rod"]["collision"], "box_approximation")
            with self.assertRaisesRegex(SceneError, "new file"):
                build_scene(source, output, {"green_cube": [0.36, -0.1, 0.795]}, run_dir=run_dir)
            with self.assertRaises(SceneError):
                build_scene(source, source.parent / "new_scene.xml", {"green_cube": [0, 0, 0]}, run_dir=source.parent)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_default_contact_backend_keeps_exact_xml_and_original_scene_identity(self):
        source = PROJECT / "resources" / "g1_warehouse_pregrasp.xml"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = build_scene(source, root / "implicit.xml", {"green_cube": [0.36, -.1, .795]}, run_dir=root)
            second = build_scene(source, root / "explicit.xml", {"green_cube": [0.36, -.1, .795]}, run_dir=root,
                                 contact_model={"backend": "source_default_v1", "time_constant_s": .02, "damping_ratio": 1.})
            self.assertEqual((root / "implicit.xml").read_bytes(), (root / "explicit.xml").read_bytes())
            self.assertEqual(first["scene_protocol"], "g1_strict_scene_force_transfer_v1")
            self.assertEqual(second["scene_protocol"], first["scene_protocol"])
            self.assertNotIn("contact_model", first)
            self.assertEqual(second["contact_model"]["explicit_pairs"], [])

    def test_noslip_candidate_xml_diff_is_one_global_option_only(self):
        source = PROJECT / "resources" / "g1_warehouse_pregrasp.xml"
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            settings = {"backend": "source_default_noslip10_v1", "time_constant_s": .02, "damping_ratio": 1.}
            baseline = build_scene(source, root / "baseline.xml", {"green_cube": [.36, -.1, .795]}, run_dir=root)
            candidate = build_scene(source, root / "candidate.xml", {"green_cube": [.36, -.1, .795]}, run_dir=root, contact_model=settings)
            tree = ET.parse(root / "candidate.xml").getroot()
            self.assertEqual(tree.find("option").attrib.pop("noslip_iterations"), "10")
            if ET.parse(root / "baseline.xml").getroot().find("option") is None:
                self.assertEqual(len(tree.find("option")), 0)
                self.assertEqual(tree.find("option").attrib, {})
                tree.remove(tree.find("option"))
            self.assertEqual(ET.tostring(tree), (root / "baseline.xml").read_bytes())
            self.assertEqual(candidate["contact_model"]["explicit_pairs"], [])
            self.assertIn("global", candidate["contact_model"]["scope"])
            self.assertNotEqual(candidate["scene_protocol"], baseline["scene_protocol"])
            self.assertEqual(candidate["actuator_force_transfers"], baseline["actuator_force_transfers"])
        self.assertEqual(source.read_bytes(), before)

    def test_versioned_solver_rejects_wrong_compiled_parameters_and_engine(self):
        settings = {"backend": "source_default_noslip10_v1", "time_constant_s": .02, "damping_ratio": 1.}
        values = {k: v for k, v in NOSLIP10_SOLVER.items() if k != "mujoco_version"}
        values["timestep"] = .002
        api = SimpleNamespace(__version__="3.2.5")
        self.assertEqual(verify_contact_solver(SimpleNamespace(opt=SimpleNamespace(**values)), settings, api), NOSLIP10_SOLVER)
        for key, value in (("noslip_iterations", 0), ("impratio", 2.), ("timestep", .001)):
            changed = dict(values, **{key: value})
            with self.subTest(key=key), self.assertRaises(SceneError):
                verify_contact_solver(SimpleNamespace(opt=SimpleNamespace(**changed)), settings, api)
        with self.assertRaises(SceneError):
            verify_contact_solver(SimpleNamespace(opt=SimpleNamespace(**values)), settings, SimpleNamespace(__version__="3.3.0"))

    def test_candidate_changes_only_fifteen_original_hand_green_pair_solrefs(self):
        source = PROJECT / "resources" / "g1_warehouse_pregrasp.xml"
        before = source.read_bytes()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = build_scene(source, root / "baseline.xml", {"green_cube": [0.36, -.1, .795]}, run_dir=root)
            candidate = build_scene(source, root / "candidate.xml", {"green_cube": [0.36, -.1, .795]}, run_dir=root,
                                    contact_model={"backend": "right_hand_green_compliance_v2", "time_constant_s": .006, "damping_ratio": 1.})
            xml = ET.parse(root / "candidate.xml").getroot()
            detail = candidate["contact_model"]
            pairs = xml.find("contact").findall("pair")
            self.assertEqual(len(pairs), 15)
            self.assertEqual(len(detail["new_names_for_original_unnamed_active_geoms"]), 7)
            self.assertTrue(candidate["scene_protocol"].startswith("g1_strict_scene_right_hand_green_compliance_v2_"))
            self.assertNotEqual(candidate["scene_protocol"], baseline["scene_protocol"])
            self.assertTrue(detail["uncalibrated_simulation_hypothesis"])
            self.assertEqual(detail["time_constant_in_physics_steps"], 3.)
            for pair, record in zip(pairs, detail["explicit_pairs"]):
                self.assertEqual(pair.get("geom2"), "green_cube_geom")
                self.assertNotIn("calibration_table", pair.attrib.values())
                self.assertEqual(pair.get("condim"), "3")
                self.assertEqual([float(v) for v in pair.get("solref").split()], [.006, 1.])
                self.assertEqual([float(v) for v in pair.get("solimp").split()], [.9, .95, .001, .5, 2.])
                self.assertEqual(float(pair.get("margin")), 0.)
                self.assertEqual(float(pair.get("gap")), 0.)
                mu = 2. if pair.get("geom1") == "right_palm_contact_pad" else 2.2
                self.assertEqual([float(v) for v in pair.get("friction").split()], [mu, mu, .05, .005, .005])
                expected = dict(record["original_dynamic_contact_parameters"], solref=[.006, 1.])
                self.assertEqual(record["compiled_expected_parameters"], expected)
                self.assertTrue(record["original_dynamic_mask_compatible"])
                geom = xml.find(".//geom[@name='{}']".format(pair.get("geom1")))
                self.assertEqual((geom.get("contype"), geom.get("conaffinity")), ("2", "4"))
            # Names have no dynamics; remove just the aliases and new pairs to
            # recover the original XML, including all geometry/forces/masks.
            for name in detail["new_names_for_original_unnamed_active_geoms"]:
                xml.find(".//geom[@name='{}']".format(name)).attrib.pop("name")
            xml.remove(xml.find("contact"))
            self.assertEqual(ET.tostring(xml), (root / "baseline.xml").read_bytes())
            self.assertEqual(candidate["actuator_force_transfers"], baseline["actuator_force_transfers"])
        self.assertEqual(before, source.read_bytes())

    def test_contact_candidate_rejects_unknown_inheritance_exclusion_and_unsafe_timeconstant(self):
        source = PROJECT / "resources" / "g1_warehouse_pregrasp.xml"
        settings = {"backend": "right_hand_green_compliance_v2", "time_constant_s": .006, "damping_ratio": 1.}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resources = root / "resources"
            resources.mkdir()
            for index, mutation in enumerate(("defaults", "exclusion", "predefined")):
                xml = ET.fromstring(source.read_bytes())
                if mutation == "defaults":
                    ET.SubElement(xml, "default")
                else:
                    contact = ET.SubElement(xml, "contact")
                    if mutation == "exclusion":
                        ET.SubElement(contact, "exclude", body1="right_hand_index_0_link", body2="green_cube")
                    else:
                        ET.SubElement(contact, "pair", geom1="right_palm_contact_pad", geom2="green_cube_geom")
                modified = resources / (mutation + ".xml")
                modified.write_bytes(ET.tostring(xml))
                with self.assertRaises(SceneError):
                    build_scene(modified, root / ("bad_{}.xml".format(index)), {"green_cube": [0.36, -.1, .795]}, run_dir=root, contact_model=settings)
            for index, invalid in enumerate((dict(settings, time_constant_s=.004), dict(settings, damping_ratio=.5),
                                               dict(settings, backend="unknown"))):
                with self.assertRaises(SceneError):
                    build_scene(source, root / ("invalid_{}.xml".format(index)), {"green_cube": [0.36, -.1, .795]}, run_dir=root, contact_model=invalid)

    def test_camera_preservation_is_explicit(self):
        source = PROJECT / "resources" / "g1_warehouse_pregrasp.xml"
        original_camera = ET.parse(source).getroot().find(".//camera[@name='head_camera']")
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            manifest = build_scene(source, run_dir / "scene.xml", {"green_cube": [0.36, -0.1, 0.795]},
                                   run_dir=run_dir, preserve_source_camera=True)
            self.assertEqual(manifest["head_camera"]["xyaxes"], original_camera.get("xyaxes"))
            self.assertEqual(manifest["head_camera"]["fovy"], original_camera.get("fovy"))


if __name__ == "__main__":
    unittest.main()
