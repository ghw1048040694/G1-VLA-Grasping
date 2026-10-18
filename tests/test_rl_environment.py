"""Math and immutable input checks for the separate PhysX RL curriculum."""

import ast
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

from g1_vla.rl_environment import (
    ACTION_JOINT_NAMES,
    ContactQualifiedFullLift,
    DEFAULT_SETTINGS,
    ENVIRONMENT_VERSION,
    FIXED_PALM_FILTER_ASSET_SHA256,
    author_fixed_palm_adjacency_filters,
    fixed_palm_adjacency_filter_contract,
    fixed_palm_filter_relationships,
    readback_fixed_palm_adjacency_filters,
    validate_fixed_palm_filter_relationships,
    RLEnvironmentError,
    TABLE_BODY_PROTOCOL,
    TableContactChannel,
    _actuator_specifications,
    _seed_state,
    _compare_hand_contacts,
    _reduce_contact_pairs,
    _reduce_friction_pairs,
    _summarize_hand_contacts,
    _probe_payload,
    _json_finite,
    audit_asset_manifest,
    audit_fixed_root_joints,
    consecutive_window_samples,
    control_probe_profile,
    current_relative_contact_motion,
    native_relative_damping_state,
    reconstruct_native_hand_inertia,
    revalidated_native_arm_substep,
    RIGHT_ARM,
    RIGHT_HAND,
    cube_contact_partner_layout,
    minimum_box_clearance,
    migration_pose_errors,
    offset_point_jacobian,
    preload_contact_ready,
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
    def test_controller_diagnostics_require_explicit_versioned_identity(self):
        self.assertIsNone(control_probe_profile(None))
        profile = {"name": "policy_legacy", "action_source": "frozen_actor",
                   "control_guard_version": "legacy_scalar_v1"}
        result = control_probe_profile(profile)
        self.assertEqual(result, profile)
        self.assertIsNot(result, profile)
        for invalid in ({}, dict(profile, name=""), dict(profile, control_guard_version="unbounded"), []):
            with self.assertRaises(RLEnvironmentError):
                control_probe_profile(invalid)

    def test_table_is_declared_as_gravity_disabled_kinematic_at_original_geometry(self):
        self.assertTrue(TABLE_BODY_PROTOCOL["kinematic_enabled"])
        self.assertTrue(TABLE_BODY_PROTOCOL["disable_gravity"])
        self.assertEqual(TABLE_BODY_PROTOCOL["size_m"], (.84, 1.1, .06))
        self.assertEqual(TABLE_BODY_PROTOCOL["position_m"], (.52, 0, .72))

    def test_table_channel_requires_native_binding_and_positive_load_witness(self):
        channel = TableContactChannel()
        self.assertFalse(channel.snapshot()["supported"])
        channel.bind(["/World/envs/env_0/Table", "/World/envs/env_1/Table"], 2,
                     kinematic=True, disable_gravity=True)
        channel.observe(0, 0, 0)
        self.assertFalse(channel.snapshot()["supported"])
        channel.observe(1, .785, 0)
        self.assertFalse(channel.snapshot()["supported"])
        channel.observe(2, .3, 1)
        state = channel.snapshot()
        self.assertTrue(state["supported"])
        self.assertEqual(state["valid_physics_samples"], 3)
        self.assertEqual(state["positive_force_physics_samples"], 1)
        self.assertEqual(state["positive_force_any_env_physics_samples"], 2)
        self.assertEqual(state["bound_env_count"], 2)
        self.assertEqual(state["positive_force_env_samples"], 3)
        self.assertEqual(state["max_force_n"], .785)
        self.assertIsNone(state["unavailable_reason"])
        channel.unavailable("native pair read failed")
        channel.observe(2, .9, 2)
        self.assertFalse(channel.snapshot()["supported"])

    def test_table_channel_cannot_treat_missing_binding_or_nonfinite_force_as_zero(self):
        channel = TableContactChannel()
        with self.assertRaises(RLEnvironmentError):
            channel.bind(["/Robot/floor"], 1, kinematic=True, disable_gravity=True)
        self.assertFalse(channel.snapshot()["supported"])
        with self.assertRaises(FloatingPointError):
            channel.observe(0, float("nan"), 0)
        with self.assertRaises(FloatingPointError):
            channel.observe(-1, .5, 0)

    def test_contact_collection_backend_cannot_silently_change(self):
        config = configuration()
        config["rl_teacher"]["contact_extraction_backend"] = "old-nine-views"
        with self.assertRaises(RLEnvironmentError):
            resolve_settings(config)

    def test_cube_partners_preserve_camera_and_wrist_as_other_robot_contacts(self):
        hand = ["right_hand_thumb_2_link", "right_hand_palm_link"]
        paths = {name: "/Robot/"+name for name in hand+["right_hand_camera_base_link", "right_wrist_yaw_link", "pelvis"]}
        names, other = cube_contact_partner_layout(paths, hand)
        self.assertEqual(names[0], "Table")
        self.assertEqual({names[index] for index in other}, {"right_hand_camera_base_link", "right_wrist_yaw_link", "pelvis"})
        self.assertEqual(len(names), 6)
        with self.assertRaises(RLEnvironmentError):
            cube_contact_partner_layout(paths, hand+["missing_body"])

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


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch-free host; run controller tensor tests on CPU Torch")
class ControllerDiagnosticTensorTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def method(self, name, **extra):
        # Exercise the real nested environment method without importing Isaac
        # or constructing a simulator. Only measured tensor providers are fake.
        tree = ast.parse((ROOT / "src/g1_vla/rl_environment.py").read_text())
        environment = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "Environment")
        method = next(node for node in environment.body if isinstance(node, ast.FunctionDef) and node.name == name)
        scope = {"torch": self.torch, "tensor": lambda value: value, "RLEnvironmentError": RLEnvironmentError,
                 "__package__": "g1_vla", **extra}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "<actual-environment-method>", "exec"), scope)
        return scope[name]

    def test_site_jacobian_corrects_angular_velocity_at_offset_without_mutation(self):
        t = self.torch
        jacobian = t.zeros(1, 6, 2)
        jacobian[0, 5, 0], jacobian[0, 4, 1] = 1, 2
        saved = jacobian.clone()
        shifted = offset_point_jacobian(t, jacobian, t.tensor([[.14, 0, 0]]))
        self.assertAlmostEqual(shifted[0, 1, 0].item(), .14, places=6)
        self.assertAlmostEqual(shifted[0, 2, 1].item(), -.28, places=6)
        self.assertTrue(t.equal(jacobian, saved))
        self.assertTrue(t.equal(shifted[:, 3:], jacobian[:, 3:]))

    def test_real_recording_header_serializes_full_lift_tensor_state(self):
        t = self.torch
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            fake = SimpleNamespace(_record_index=0, _evaluation_seed=17, _record=None, num_envs=2,
                _control_probe_enabled=True, _control_probe_profile={"name":"full_lift"},
                robot=SimpleNamespace(joint_names=["joint"]), physics_dt=.002, step_dt=.066,
                _full_lift=SimpleNamespace(main_lift_authorized=t.tensor([False,True])),
                runtime_audit=lambda:{"policy_contract":{"full_lift":{"test_tensor":t.tensor([1,2])}},
                                      "nested_state":{"force":t.tensor([float("nan"),1.5])}})
            writer = self.method("_write_evaluation_row", json=json,
                _probe_payload=_probe_payload, _json_finite=_json_finite)
            fake._write_evaluation_row=lambda row:writer(fake,row)
            opener=self.method("_open_evaluation_recording",output=output,ACTION_JOINT_NAMES=ACTION_JOINT_NAMES,
                               ENVIRONMENT_VERSION=ENVIRONMENT_VERSION)
            try:
                opener(fake)
                fake._record.flush()
                header=json.loads((output/"evaluation_physics_0001.jsonl").read_text())
                self.assertEqual(header["event"],"recording_header")
                self.assertEqual(header["main_lift_authorized"],[False,True])
                self.assertEqual(header["policy_contract"]["full_lift"]["test_tensor"],[1,2])
                writer(fake,{"event":"physics_sample","nested":{"force":t.tensor([float("nan"),1.5])}})
                fake._record.flush()
                sample=json.loads((output/"evaluation_physics_0001.jsonl").read_text().splitlines()[1])
                self.assertEqual(sample["nested"]["force"],[None,1.5])
            finally:
                if fake._record is not None:
                    fake._record.close()

    def test_preload_readiness_requires_real_contact_but_does_not_claim_support(self):
        t = self.torch
        keys = ("contact_safe", "grip_reference_locked", "no_positive_gap_hand_load",
                "friction_partner_manifold_certified", "finite", "joint_safe")
        metrics = {key: t.ones(2, dtype=t.bool) for key in keys}
        metrics.update(no_loaded_table_contact=t.zeros(2, dtype=t.bool),
                       support_current=t.zeros(2, dtype=t.bool))
        self.assertTrue(preload_contact_ready(t, metrics, 2, "cpu").all())
        metrics["grip_reference_locked"][0] = False
        self.assertEqual(preload_contact_ready(t, metrics, 2, "cpu").tolist(), [False, True])
        del metrics["friction_partner_manifold_certified"]
        self.assertFalse(preload_contact_ready(t, metrics, 2, "cpu").any())

    def test_no_diagnostic_preserves_historical_scalar_target_formula(self):
        t = self.torch
        jac = t.zeros(1, 1, 6, 43)
        jac[0, 0, 2, 0], jac[0, 0, 4, 1] = 1, -2
        data = SimpleNamespace(joint_pos=t.zeros(1, 43), body_quat_w=t.tensor([[[0., 0, 0, 1], [0., 0, 0, 1]]]),
                               body_pos_w=t.zeros(1, 2, 3), body_link_jacobian_w=jac)
        fake = SimpleNamespace(num_envs=1, _control_probe_enabled=False, _physics_pinch=None, _effort_grasp=None, _previous_actions=t.zeros(1, 14),
            _actions=t.zeros(1, 14), _next_action_origin="actor", _targets=t.zeros(1, 43),
            _action_ids=list(range(14)), _velocity=t.full((14,), .5), step_dt=.066,
            _limits=t.tensor([[[-1., 1.]]*43]), _unsafe=t.zeros(1, dtype=t.bool),
            robot=SimpleNamespace(data=data, num_bodies=2), _palm_id=1, _palm_local=t.tensor([[.14, 0, 0]]),
            _start_palm_z=t.full((1,), float("nan")), _shield_count=0)
        action = t.ones(1, 14)
        function = self.method("_pre_physics_step", quat_apply=lambda q, offset: offset,
            config={"rl_teacher": {"action_scale_rad": .08}, "control": {"max_anchor_delta_rad": .08}},
            settings={"probe_clearance_max_m": .006})
        function(fake, action)
        previous = t.zeros(1, 7)
        increment = t.full((1, 7), .5*.066)
        jz = t.tensor([[1., .28, 0, 0, 0, 0, 0]])
        predicted = (jz*increment).sum(-1)
        factor = (.006/predicted.clamp_min(1e-12)).clamp(0, 1)
        expected = previous+increment*factor[:, None]
        self.assertTrue(t.equal(fake._targets[:, :7], expected))
        self.assertTrue(t.equal(fake._targets[:, 7:14], increment))
        self.assertEqual(fake._shield_count, 1)
        self.assertFalse(fake._unsafe.any())

    def test_full_lift_palm_guard_uses_declared_height_only_after_admission(self):
        t = self.torch
        jac = t.zeros(1, 1, 6, 43); jac[0, 0, 2, 0] = 1
        data = SimpleNamespace(joint_pos=t.zeros(1, 43), body_quat_w=t.tensor([[[0.,0,0,1],[0.,0,0,1]]]),
            body_pos_w=t.zeros(1, 2, 3), body_link_jacobian_w=jac)
        base = dict(num_envs=1, _control_probe_enabled=False, _physics_pinch=None, _effort_grasp=None,
            _previous_actions=t.zeros(1,14), _actions=t.zeros(1,14), _next_action_origin="guide",
            _targets=t.zeros(1,43), _action_ids=list(range(14)), _velocity=t.full((14,),.5), step_dt=.066,
            _limits=t.tensor([[[-1.,1.]]*43]), _unsafe=t.zeros(1,dtype=t.bool),
            robot=SimpleNamespace(data=data,num_bodies=2), _palm_id=1, _palm_local=t.zeros(1,3),
            _start_palm_z=t.full((1,),float("nan")), _shield_count=0)
        function = self.method("_pre_physics_step", quat_apply=lambda q, offset: offset,
            config={"rl_teacher":{"action_scale_rad":.08},"control":{"max_anchor_delta_rad":.08}},
            settings={"probe_clearance_max_m":.006}, full_lift={"palm_lift_request_max_m":.11})
        for admitted, expected in ((False,.006),(True,.033)):
            fake = SimpleNamespace(**copy.deepcopy(base))
            fake._full_lift = SimpleNamespace(main_lift_authorized=t.tensor([admitted]))
            function(fake,t.ones(1,14))
            self.assertAlmostEqual(fake._targets[0,0].item(),expected,places=6)


    def test_terminal_transition_keeps_actual_pre_reset_state_and_records_once(self):
        t = self.torch
        profile = {"name": "guided", "action_source": "guidance", "control_guard_version": "box_halfspace_projected_v2"}
        q = t.full((1, 43), .123)
        metric = t.tensor([.2])
        fake = SimpleNamespace(_pending_physics_sample=False, _control_probe_enabled=True, _control_command_id=1,
            _control_probe_profile=profile, _episode_ids=t.tensor([7]), episode_length_buf=t.tensor([91]),
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=q, joint_vel=t.zeros_like(q))),
            cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=t.tensor([[0., 0, .775]]), root_quat_w=t.tensor([[0., 0, 0, 1]]))),
            sim=SimpleNamespace(get_physics_step_count=lambda: 3003), _targets=q.clone(),
            _last_metrics={"example": metric}, _unsafe=t.tensor([False]), _local_success=t.tensor([False]),
            _standalone_video=False, max_episode_length=91, _control_transition_recorded=0)
        fake._control_probe_joint_load = lambda phase: {"available": False, "sample_phase": phase}
        fake._control_probe_self_pair = lambda phase: {"available": False, "sample_phase": phase}
        terminated, truncated = self.method("_get_dones")(fake)
        self.assertFalse(terminated.any())
        self.assertTrue(truncated.all())
        q.zero_(); metric.zero_(); fake._episode_ids += 1
        recorded = []
        fake._write_control_probe = recorded.append
        method = self.method("record_control_probe_transition")
        method(fake)
        self.assertAlmostEqual(recorded[0]["measured_q_rad"][0, 0].item(), .123, places=6)
        self.assertAlmostEqual(recorded[0]["metrics"]["example"][0].item(), .2, places=6)
        self.assertEqual(recorded[0]["episode_id"].item(), 7)
        self.assertTrue(recorded[0]["pre_reset"])
        with self.assertRaises(RLEnvironmentError):
            method(fake)

    def test_diagnostic_state_has_measured_site_tensors_and_masks_previous_episode_contact(self):
        t = self.torch
        keys = ("contact_safe", "grip_reference_locked", "no_positive_gap_hand_load",
                "friction_partner_manifold_certified", "finite", "joint_safe")
        metrics = {key: t.tensor([True]) for key in keys}
        metrics["group_normal_force_n"] = t.full((1, 3), .4)
        q = t.full((1, 43), .1)
        data = SimpleNamespace(joint_pos=q, joint_vel=t.zeros_like(q), body_link_jacobian_w=t.zeros(1, 1, 6, 43),
                               body_quat_w=t.tensor([[[0., 0, 0, 1], [0., 0, 0, 1]]]), body_pos_w=t.zeros(1, 2, 3))
        fake = SimpleNamespace(_control_probe_enabled=True, num_envs=1, device="cpu", _palm_id=1,
            _palm_local=t.tensor([[.14, 0, 0]]), _palm_local_rotation=t.tensor([[0., 0, 0, 1]]),
            robot=SimpleNamespace(data=data, num_bodies=2, joint_names=["joint"+str(i) for i in range(43)]),
            cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=t.zeros(1, 3), root_quat_w=t.tensor([[0., 0, 0, 1]]))),
            _action_ids=list(range(14)), _tip_ids=[0, 1, 0], _velocity=t.full((14,), .5), step_dt=.066,
            _targets=t.full((1, 43), .2), _limits=t.tensor([[[-1., 1.]]*43]), _last_metrics=metrics,
            episode_length_buf=t.tensor([0]), _episode_ids=t.tensor([8]), _grip_locked=t.tensor([False]),
            _unsafe=t.tensor([False]), _probe_overshoot=t.tensor([False]), _control_probe_profile={"name": "test"},
            sim=SimpleNamespace(get_physics_step_count=lambda: 3003))
        fake._control_probe_joint_load = lambda phase: {"available": False, "sample_phase": phase}
        fake._control_probe_self_pair = lambda phase: {"available": False, "sample_phase": phase}
        method = self.method("control_diagnostic_state", quat_apply=lambda q, offset: offset,
            quat_mul=lambda first, second: first, offset_point_jacobian=offset_point_jacobian,
            preload_contact_ready=preload_contact_ready, ACTION_JOINT_NAMES=ACTION_JOINT_NAMES,
            config={"rl_teacher": {"action_scale_rad": .08}})
        result = method(fake)
        self.assertEqual(result["palm_point_jacobian_w"].shape, (1, 6, 7))
        self.assertEqual(result["joint_limits_action_rad"].shape, (1, 14, 2))
        self.assertEqual(result["episode_id"].item(), 8)
        self.assertFalse(result["preload_contact_ready"].any())
        self.assertFalse(result["group_normal_force_n"].any())
        result["q_action_rad"].zero_()
        self.assertTrue((q == .1).all())
        fake.episode_length_buf[:] = 1
        fake._grip_locked[:] = True
        self.assertTrue(method(fake)["preload_contact_ready"].all())

    def test_profile_cannot_change_with_an_unrecorded_terminal_snapshot(self):
        fake = SimpleNamespace(_control_probe_enabled=True, _pending_physics_sample=False,
                               _control_transition_snapshot={"command_id": 1})
        with self.assertRaises(RLEnvironmentError):
            self.method("configure_control_probe")(fake, {"name": "next"})


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch is required")
class ContactQualifiedFullLiftTests(unittest.TestCase):
    def setUp(self):
        import torch
        from g1_vla.rl_effort_protocol import full_lift_protocol_contract
        self.t = torch
        self.state = ContactQualifiedFullLift(torch,2,"cpu",full_lift_protocol_contract())
        self.inputs = dict(measurement_valid=torch.ones(2,dtype=torch.bool),
            group_force_n=torch.tensor([[.8,.4,.4]]*2), original_hand_domain=torch.ones(2,dtype=torch.bool),
            local_support_qualified=torch.zeros(2,dtype=torch.bool), support_current=torch.ones(2,dtype=torch.bool),
            clearance_m=torch.full((2,),.0023), object_lift_m=torch.full((2,),.0023),
            unsafe=torch.zeros(2,dtype=torch.bool))

    def test_local_milestone_does_not_complete_full_task_and_hold_is_251_samples(self):
        for _ in range(99):
            self.state.update(**self.inputs)
        self.assertFalse(self.state.main_lift_started.any())
        self.inputs["local_support_qualified"][:] = True
        self.state.update(**self.inputs)
        self.assertTrue(self.state.main_lift_authorized.all())
        self.assertFalse(self.state.full_success.any())
        self.inputs["clearance_m"][:] = .1
        self.inputs["object_lift_m"][:] = .1
        for _ in range(250):
            self.state.update(**self.inputs)
        self.assertFalse(self.state.full_success.any())
        self.state.update(**self.inputs)
        self.assertTrue(self.state.full_success.all())
        self.assertEqual(self.state.hold_samples.tolist(),[251,251])

    def test_earlier_force_or_original_domain_failure_prevents_later_admission(self):
        self.inputs["group_force_n"][0,1] = 1.5001
        self.inputs["original_hand_domain"][1] = False
        self.state.update(**self.inputs)
        self.inputs["group_force_n"][:] = .4
        self.inputs["original_hand_domain"][:] = True
        self.inputs["local_support_qualified"][:] = True
        for _ in range(300):
            self.state.update(**self.inputs)
        self.assertTrue(self.state.local_support_qualified.all())
        self.assertFalse(self.state.main_lift_started.any())
        self.state.reset(self.t.tensor([0]))
        self.state.update(**self.inputs)
        self.assertEqual(self.state.main_lift_started.tolist(),[True,False])

    def test_corner_clearance_alone_cannot_replace_original_10cm_object_rise(self):
        self.inputs["local_support_qualified"][:] = True
        self.state.update(**self.inputs)
        self.inputs["clearance_m"][:] = .1
        self.inputs["object_lift_m"][:] = .09999
        for _ in range(251):
            self.state.update(**self.inputs)
        self.assertTrue(self.state.main_lift_authorized.all())
        self.assertFalse(self.state.full_success.any())
        self.assertEqual(self.state.hold_samples.tolist(),[0,0])
        self.inputs["object_lift_m"][:] = .1
        for _ in range(251):
            self.state.update(**self.inputs)
        self.assertTrue(self.state.full_success.all())

    def test_main_support_loss_is_sticky_failure_and_cannot_resume_upward(self):
        self.inputs["local_support_qualified"][:] = True
        self.state.update(**self.inputs)
        self.inputs["support_current"][0] = False
        self.inputs["clearance_m"][1] = .116
        self.state.update(**self.inputs)
        self.assertTrue(self.state.failed.all())
        self.assertFalse(self.state.main_lift_authorized.any())
        self.inputs["support_current"][:] = True
        self.inputs["clearance_m"][:] = .1
        self.inputs["object_lift_m"][:] = .1
        for _ in range(300):
            self.state.update(**self.inputs)
        self.assertFalse(self.state.full_success.any())
        self.assertEqual(self.state.maximum_hold_samples.tolist(),[0,0])



@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch is required")
class NativeArmTargetScheduleTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch
        z=torch.zeros(2,7,dtype=torch.float64)
        j=z.clone();j[:,0]=1
        self.values=dict(segment_origin=z.clone(),macro_endpoint=torch.full_like(z,.033),
            previous_written=z.clone(),measured_arm=z.clone(),native_index=torch.ones(2,dtype=torch.int64),
            substeps=33,physics_dt_s=.002,velocity_limits_rad_s=torch.ones(7,dtype=torch.float64),
            anchor_delta_rad=.08,joint_lower=torch.full_like(z,-2),joint_upper=torch.full_like(z,2),
            point_jz=j,palm_z=torch.zeros(2,dtype=torch.float64),start_palm_z=torch.zeros(2,dtype=torch.float64),
            height_cap_m=.11,enabled=torch.ones(2,dtype=torch.bool),sample_fresh=torch.ones(2,dtype=torch.bool))

    def method(self,name,**extra):
        tree=ast.parse((ROOT/"src/g1_vla/rl_environment.py").read_text())
        owner=next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=="Environment")
        node=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name==name)
        scope={"torch":self.t,"tensor":lambda v:v,"RLEnvironmentError":RLEnvironmentError,
            "__package__":"g1_vla","math":math,"json":json,
            "revalidated_native_arm_substep":revalidated_native_arm_substep,
            "offset_point_jacobian":offset_point_jacobian,"quat_apply":lambda q,v:v,**extra}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"<actual-native-arm-schedule>","exec"),scope)
        return scope[name]

    def test_33_real_substep_targets_reach_endpoint_without_macro_jump(self):
        t=self.t;v=self.values
        for index in range(1,34):
            v["native_index"][:]=index
            target,evidence=revalidated_native_arm_substep(t,**v)
            self.assertTrue(t.allclose(target,t.full_like(target,.001*index),atol=1e-12,rtol=0))
            self.assertFalse(evidence["requires_unsafe"].any())
            self.assertTrue((evidence["written_increment_rad_s"].abs()<=1+1e-12).all())
            v["previous_written"]=target.clone();v["measured_arm"]=target.clone()
            v["palm_z"]=target[:,0].clone()
        self.assertTrue(t.equal(target,v["macro_endpoint"]))

    def test_each_substep_rechecks_rate_mechanical_and_current_height_not_only_endpoint(self):
        t=self.t;v=self.values
        v["velocity_limits_rad_s"][:]=.05
        for index in range(1,34):
            v["native_index"][:]=index
            target,e=revalidated_native_arm_substep(t,**v)
            self.assertTrue((e["written_increment_rad_s"].abs()<=.05+1e-12).all())
            v["previous_written"]=target.clone();v["measured_arm"]=target.clone();v["palm_z"]=target[:,0]
        self.assertTrue(t.allclose(target,t.full_like(target,.0033),atol=1e-12,rtol=0))
        self.assertFalse(t.equal(target,v["macro_endpoint"]),"Never snap to an unreached endpoint at substep33")
        v=copy.deepcopy(self.values);v["native_index"][:]=1
        v["previous_written"].zero_();v["measured_arm"].zero_();v["palm_z"].zero_()
        v["velocity_limits_rad_s"][:]=1.;v["joint_upper"][:,1]=.0003;v["height_cap_m"]=.0005
        target,e=revalidated_native_arm_substep(t,**v)
        self.assertLessEqual(target[:,0].max().item(),.0005)
        self.assertLessEqual(target[:,1].max().item(),.0003)
        self.assertFalse(e["requires_unsafe"].any())
        for key,source in (("current_measured_arm_rad","measured_arm"),
                ("current_point_jz_m_per_rad","point_jz"),("current_palm_z_m","palm_z"),
                ("start_palm_z_m","start_palm_z")):
            self.assertTrue(t.equal(e[key],v[source]))
        self.assertTrue(t.equal(e["height_cap_m"],t.full((2,),.0005,dtype=t.float64)))

    def test_stale_native_epoch_and_nonfinite_height_abort_while_inactive_path_stays_macro_hold(self):
        t=self.t;v=self.values
        v["enabled"][1]=False;v["sample_fresh"][0]=False;v["measured_arm"][0]=.001
        target,e=revalidated_native_arm_substep(t,**v)
        self.assertEqual(e["requires_unsafe"].tolist(),[True,False])
        self.assertTrue(t.equal(target[0],v["measured_arm"][0]))
        self.assertTrue(t.equal(target[1],v["macro_endpoint"][1]))
        v["sample_fresh"][:]=True;v["palm_z"][0]=float("nan")
        _,e=revalidated_native_arm_substep(t,**v)
        self.assertTrue(e["requires_unsafe"][0])
        for anchor in (0.,float("nan"),True):
            v["anchor_delta_rad"]=anchor
            with self.assertRaises(ValueError):revalidated_native_arm_substep(t,**v)

    def test_real_sdk_write_precedes_actual_target_evidence_and_preserves_other36_mapping(self):
        t=self.t;events=[];writes=[]
        q=t.zeros(2,43);arm=t.full((2,7),.0002)
        def write(**values):
            events.append("sdk_write");writes.append(values["target"].clone())
        def log(row):
            self.assertEqual(events[-1],"sdk_write")
            events.append("logged")
            self.assertTrue(t.equal(row["actually_written_arm_target_rad"],arm))
            self.assertFalse(row["native_target_readback_claimed"])
        fake=SimpleNamespace(_pending_physics_sample=False,_full_lift=None,_physics_pinch=None,
            _effort_grasp=object(),_native_arm_interpolation=True,_action_ids=list(range(14)),
            _targets=t.ones(2,43),_episode_ids=t.tensor([2,3]),_control_command_id=4,
            _update_effort_grasp=lambda:None,_native_arm_substep_target=lambda:(arm,{"native_index":t.tensor([1,1])}),
            _write_control_probe=log,_write_effort_targets=lambda:None,
            sim=SimpleNamespace(get_physics_step_count=lambda:0),
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=q,joint_vel=q.clone()),set_joint_position_target_index=write))
        self.method("_apply_action")(fake)
        self.assertEqual(events,["sdk_write","logged"])
        self.assertEqual(writes[0].shape,(2,36))
        self.assertTrue(t.equal(writes[0][:,:7],arm))
        self.assertTrue((writes[0][:,7:]==1).all())
        self.assertTrue(t.equal(fake._last_written_arm_target,arm))
        self.assertTrue(t.equal(fake._last_written_arm_target_episode_ids,fake._episode_ids))
        self.assertEqual(fake._last_written_arm_target_next_native_step.tolist(),[1,1])

    def test_completed_terminal_measurement_records_arm_state_and_successful_target_write_only_for_new_schedule(self):
        t=self.t;rows=[];q=t.arange(86,dtype=t.float64).reshape(2,43)
        tree=ast.parse((ROOT/"src/g1_vla/rl_environment.py").read_text())
        sample=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=="_sample_physics")
        call=next(node for node in ast.walk(sample) if isinstance(node,ast.Call) and
            isinstance(node.func,ast.Attribute) and node.func.attr=="_write_control_probe" and
            node.args and isinstance(node.args[0],ast.Dict) and any(
                isinstance(value,ast.Constant) and value.value=="physics_hand_measurement"
                for value in node.args[0].values))
        code=compile(ast.Expression(body=call),"<actual-completed-native-measurement>","eval")
        fake=SimpleNamespace(_native_arm_interpolation=True,_control_command_id=4,_last_contact_native_step=101,
            _episode_ids=t.tensor([2,3]),episode_length_buf=t.tensor([20,30]),_action_ids=list(range(14)),
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=q,joint_vel=q+.5)),
            _last_written_arm_target=t.full((2,7),.001),_last_written_arm_target_episode_ids=t.tensor([2,3]),
            _last_written_arm_target_next_native_step=t.tensor([101,101]),
            _last_written_hand_target=t.zeros(2,7),_last_written_hand_effort=t.zeros(2,7),
            _last_metrics={},_unsafe=t.tensor([False,False]),_write_control_probe=rows.append)
        scope={"self":fake,"tensor":lambda value:value,"active_effort":False}
        eval(code,scope)
        row=rows[-1]
        self.assertTrue(t.equal(row["measured_arm_rad"],q[:,:7]))
        self.assertTrue(t.equal(row["measured_arm_velocity_rad_s"],(q+.5)[:,:7]))
        self.assertTrue(t.equal(row["actually_written_arm_position_target_rad"],fake._last_written_arm_target))
        self.assertEqual(row["arm_target_write_episode_id"].tolist(),[2,3])
        self.assertEqual(row["arm_target_write_applies_to_native_step"].tolist(),[101,101])
        self.assertIn("not_sdk_target_readback",row["arm_target_measurement_scope"])
        fake._native_arm_interpolation=False
        eval(code,scope)
        for key in ("measured_arm_rad","measured_arm_velocity_rad_s","actually_written_arm_position_target_rad",
                "arm_target_write_episode_id","arm_target_write_applies_to_native_step","arm_target_measurement_scope"):
            self.assertNotIn(key,rows[-1])

    def test_actual_main_cancel_holds_first_failure_pose_and_only_real_reset_clears_history(self):
        t=self.t;z=t.zeros(1,7);native=[100]
        jac=t.zeros(1,1,6,43);jac[0,0,2,0]=1
        data=SimpleNamespace(joint_pos=t.zeros(1,43),body_quat_w=t.tensor([[[0.,0,0,1],[0.,0,0,1]]]),
            body_pos_w=t.zeros(1,2,3),body_link_jacobian_w=jac)
        fake=SimpleNamespace(_action_ids=list(range(14)),_episode_ids=t.tensor([2]),
            _arm_segment_origin=z.clone(),_arm_segment_endpoint=t.full_like(z,.033),
            _last_written_arm_target=z.clone(),_arm_segment_native_index=t.zeros(1,dtype=t.int64),
            _last_written_arm_target_episode_ids=t.tensor([2]),_last_written_arm_target_next_native_step=t.tensor([100]),
            _arm_segment_episode_ids=t.tensor([2]),_arm_segment_start_native=100,
            _arm_segment_enabled=t.tensor([True]),_arm_segment_cancelled=t.tensor([False]),
            _arm_segment_failure_hold=z.clone(),_unsafe=t.tensor([False]),
            _full_lift=SimpleNamespace(failed=t.tensor([False]),main_lift_started=t.tensor([True])),
            _full_lift_failure_arm_target=t.full_like(z,.0007),_palm_id=1,_palm_local=t.zeros(1,3),
            _start_palm_z=t.zeros(1),_velocity=t.ones(14),_limits=t.tensor([[[-2.,2.]]*43]),physics_dt=.002,
            sim=SimpleNamespace(get_physics_step_count=lambda:native[0]),
            robot=SimpleNamespace(data=data,num_bodies=2))
        method=self.method("_native_arm_substep_target",full_lift={"palm_lift_request_max_m":.11,
            "arm_target_interpolation_substeps":33},settings={"probe_clearance_max_m":.006},
            config={"control":{"max_anchor_delta_rad":.08}})
        first,e=method(fake);self.assertTrue(t.allclose(first,t.full_like(z,.001)))
        fake._last_written_arm_target=first.clone();fake._full_lift.failed[:]=True
        data.joint_pos[:,:7]=.0007;native[0]=101
        target,e=method(fake)
        self.assertTrue(t.allclose(target,t.full_like(z,.0007)))
        self.assertTrue(e["arm_schedule_cancelled"].all())
        self.assertTrue(e["failure_override_is_not_velocity_certified"].all())
        self.assertFalse(fake._unsafe.any(),"Task failure alone is separate from hard physical unsafe")
        fake._last_written_arm_target=target.clone();data.joint_pos[:,:7]=.0006;native[0]=102
        target,_=method(fake)
        self.assertTrue(t.allclose(target,t.full_like(z,.0007)),"Keep the first failure pose; no resumed ascent")
        fake._episode_ids+=1
        self.method("_reset_native_arm_schedule")(fake,t.tensor([0]),z)
        self.assertEqual(fake._arm_segment_episode_ids.tolist(),[3])
        self.assertFalse(fake._arm_segment_cancelled.any())
        self.assertFalse(fake._arm_segment_enabled.any())
        self.assertFalse(fake._last_written_arm_target.any())
        fake._last_written_arm_target[0,0]=.0009
        fake._full_lift.failed[:]=True
        fake._full_lift_failure_arm_target[0,0]=float("nan")
        data.joint_pos[0,0]=float("nan")
        target,e=method(fake)
        self.assertTrue(t.isfinite(target).all(),"Failure override must never reintroduce NaN into the SDK write")
        self.assertAlmostEqual(target[0,0].item(),.0009,places=7)
        self.assertTrue(fake._unsafe.all())
        self.assertTrue(e["cancelled_pose_nonfinite_axis_fallback_to_last_successful_write"][0,0])
        self.assertTrue(e["failure_override_is_not_velocity_certified"].all())

    def test_full_task_failure_keeps_current_hand_effort_but_hard_unsafe_still_clears_full43(self):
        from g1_vla.rl_effort_protocol import effort_grasp_contract
        t=self.t;contract=effort_grasp_contract();decl=contract["actuation"]["posture_gravity_feedforward"]
        names=list(RIGHT_ARM)+list(RIGHT_HAND)+["waist_yaw_joint","waist_roll_joint","waist_pitch_joint"]
        names += ["other_"+str(i) for i in range(26)]
        specs={name:{"effort_limit_nm":cap} for name,cap in zip(decl["joint_order"],decl["feedforward_caps_nm"])}
        calls=[];ids=t.tensor([2,2])
        fake=SimpleNamespace(num_envs=2,device="cpu",_action_ids=list(range(14)),
            _last_written_hand_effort=t.full((2,7),.01),_last_written_full_effort=t.zeros(2,43),
            _unsafe=t.tensor([False,True]),_full_lift=SimpleNamespace(failed=t.tensor([True,True])),
            _episode_ids=ids,_last_contact_episode_ids=ids.clone(),_posture_ff_update_count=0,
            _posture_ff_last_sample=None,_control_command_id=1,_write_control_probe=lambda row:None,
            sim=SimpleNamespace(get_physics_step_count=lambda:10),robot=SimpleNamespace(joint_names=names,
                root_view=SimpleNamespace(),data=SimpleNamespace(gravity_compensation_forces=t.full((2,43),.002)),
                set_joint_effort_target_index=lambda **values:calls.append(values["target"].clone())))
        self.method("_write_effort_targets",RIGHT_ARM=RIGHT_ARM,RIGHT_HAND=RIGHT_HAND,
            effort_contract=contract,actuator_specs=specs)(fake)
        self.assertTrue(t.equal(calls[0][0,7:14],t.full((7,),.01)))
        self.assertTrue((calls[0][1]==0).all())
        self.assertTrue(fake._full_lift.failed.all(),"Retaining force feedback never restores task success")

    def test_full_failure_terminates_without_hard_unsafe_and_local_milestone_does_not_terminate(self):
        t=self.t
        fake=SimpleNamespace(_pending_physics_sample=False,_control_probe_enabled=False,_standalone_video=False,
            _unsafe=t.tensor([False]),_local_success=t.tensor([True]),
            _full_lift=SimpleNamespace(failed=t.tensor([False]),full_success=t.tensor([False])),
            episode_length_buf=t.tensor([20]),max_episode_length=243,
            cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=t.tensor([[0.,0.,.779]]))))
        method=self.method("_get_dones")
        terminated,truncated=method(fake)
        self.assertFalse(terminated.any());self.assertFalse(truncated.any())
        fake._full_lift.failed[:]=True
        terminated,truncated=method(fake)
        self.assertTrue(terminated.all());self.assertFalse(truncated.any())
        self.assertFalse(fake._unsafe.any())


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"), "Original camera adapters require Torch and Pillow")
class DeclaredMultiCameraTests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        import torch
        from types import MethodType
        t=torch
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);declaration={"version":"g1_declared_multienv_native_rgb_v1","env_ids":list(range(8))}
            tree=ast.parse((ROOT/"src/g1_vla/rl_environment.py").read_text())
            owner=next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=="Environment")
            scope={"torch":t,"tensor":lambda value:value,"json":json,"hashlib":hashlib,"_probe_payload":_probe_payload,
                "_json_finite":_json_finite,"output":output,"video_recording":declaration,"RLEnvironmentError":RLEnvironmentError}
            for name in ("_open_declared_video_recording","_write_declared_video_event","_capture_declared_evaluation_frames","_capture_evaluation_frame"):
                node=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name==name)
                exec(compile(ast.Module(body=[node],type_ignores=[]),"<actual-declared-camera>","exec"),scope)
            native=[33];cameras={};statuses=[]
            for env_id in range(8):
                frame=t.zeros(1,dtype=t.int64)
                cameras[env_id]=SimpleNamespace(frame=frame,data=SimpleNamespace(output={"rgb":t.full((1,480,640,4),20+env_id,dtype=t.uint8)}),
                    update=lambda dt,force_recompute,f=frame:f.add_(1))
                statuses.append({"camera_env_id":env_id,"enabled":True,"frames":0,"failure":None,
                    "frame_directory":f"video/env_{env_id}/rgb_frames","frame_ledger":f"video/env_{env_id}/video_frames.jsonl",
                    "eye_world_m":[env_id+.635,-.262,1.045],"target_world_m":[env_id+.32,.055,.82]})
            cube_pos=t.zeros(8,3);cube_pos[:,0]=t.arange(8);quat=t.zeros(8,4);quat[:,3]=1
            fake=SimpleNamespace(device="cpu",_video_recording=declaration,_evaluation_cameras=cameras,_declared_video_streams={},
                _video_status={"requested":True,"enabled":True,"failure":None,"frames":0,"video_recording":declaration,"cameras":statuses},
                _last_video_native_step=None,_physics_samples=33,_standalone_video=True,_episode_ids=t.full((8,),2,dtype=t.int64),
                episode_length_buf=t.ones(8,dtype=t.int64),step_dt=.066,_control_probe_enabled=True,
                _control_probe_profile={"name":"balanced_effort_pose_hold","action_source":"guided_hand_pose_hold","control_guard_version":"box_halfspace_projected_v2"},
                cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=cube_pos,root_quat_w=quat)),
                sim=SimpleNamespace(get_physics_step_count=lambda:native[0],get_physics_dt=lambda:.002))
            for name,value in scope.items():
                if name.startswith(("_open_declared","_write_declared","_capture_declared","_capture_evaluation")):
                    setattr(fake,name,MethodType(value,fake))
            try:
                fake._open_declared_video_recording()
                yield fake,output,native,owner,scope
            finally:
                for resources in fake._declared_video_streams.values():resources["manifest"].close()

    def test_eight_original_cameras_map_identity_and_async_episode_resets_to_distinct_ledgers(self):
        import torch
        from PIL import Image
        with self.fixture() as (fake,output,native,owner,scope):
            fake._capture_evaluation_frame()
            self.assertEqual(fake._video_status["frames"],1)
            self.assertEqual([camera["frames"] for camera in fake._video_status["cameras"]],[1]*8)
            self.assertEqual(fake._video_status["first_frame_memory"]["camera_count"],8)
            reset=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name=="_reset_idx")
            branch=next(node for node in reset.body if isinstance(node,ast.If) and
                isinstance(node.test,ast.BoolOp) and isinstance(node.test.values[0],ast.Compare) and
                isinstance(node.test.values[0].left,ast.Name) and node.test.values[0].left.id=="video_recording")
            fake._episode_ids[4]=3;fake.episode_length_buf[4]=0
            reset_scope={**scope,"self":fake,"env_ids":torch.tensor([4])}
            exec(compile(ast.Module(body=branch.body,type_ignores=[]),"<actual-declared-camera-reset-ledger>","exec"),reset_scope)
            native[0]=66;fake._physics_samples=66;fake.episode_length_buf+=1
            fake._capture_evaluation_frame()
            self.assertEqual(fake._video_status["frames"],2,"A macro records one set rather than eight frames")
            for env_id in range(8):
                ledger=output/f"video/env_{env_id}/video_frames.jsonl"
                rows=[json.loads(line) for line in ledger.read_text().splitlines()]
                self.assertEqual(rows[0]["video_recording"],fake._video_recording)
                frames=[row for row in rows if row["event"]=="camera_frame"]
                self.assertEqual([r["camera_env_id"] for r in frames],[env_id,env_id])
                self.assertEqual([r["native_physics_step_count"] for r in frames],[33,66])
                self.assertEqual([r["episode_id"] for r in frames],[2,3] if env_id==4 else [2,2])
                self.assertEqual(len([r for r in rows if r["event"]=="episode_reset"]),int(env_id==4))
                self.assertEqual(frames[-1]["cube_position_world_m"][0],env_id)
                for row in frames:
                    path=output/f"video/env_{env_id}/rgb_frames"/row["path"];data=path.read_bytes()
                    self.assertEqual(hashlib.sha256(data).hexdigest(),row["sha256"])
                    self.assertEqual(len(data),row["bytes"])
                    with Image.open(path) as image:self.assertEqual(image.getpixel((0,0)),(20+env_id,)*3)

    def test_camera_stall_or_duplicate_native_sample_is_explicit_incomplete_evidence(self):
        with self.fixture() as (fake,output,native,_,__):
            fake._capture_evaluation_frame();fake._capture_evaluation_frame()
            self.assertIn("duplicate native",fake._video_status["failure"])
            self.assertEqual(fake._video_status["frames"],1)
            self.assertTrue((output/"evaluation_video_failure.json").is_file())
        with self.fixture() as (fake,output,native,_,__):
            fake._capture_evaluation_frame();native[0]=66
            fake._evaluation_cameras[4].update=lambda **kwargs:None
            fake._capture_evaluation_frame()
            self.assertIn("counter did not advance",fake._video_status["failure"])
            self.assertIsNotNone(fake._video_status["cameras"][4]["failure"])
            self.assertEqual(fake._video_status["frames"],1)
            self.assertEqual(len(list((output/"video/env_0/rgb_frames").glob("*.png"))),1)

    def test_close_requires_all_original_camera_counts_and_closes_every_ledger(self):
        for missing in (False,True):
            with self.subTest(missing_camera_frame=missing),self.fixture() as (fake,output,native,owner,scope):
                fake._capture_evaluation_frame()
                if missing:fake._video_status["cameras"][7]["frames"]=0
                method=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name=="close")
                class Base:
                    def close(self):self.base_closed=True
                namespace={**scope,"Base":Base}
                wrapper=ast.ClassDef(name="ActualClose",bases=[ast.Name(id="Base",ctx=ast.Load())],
                    keywords=[],body=[method],decorator_list=[])
                exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),
                    "<actual-declared-camera-close>","exec"),namespace)
                closer=namespace["ActualClose"]();closer.__dict__.update(vars(fake))
                streams=[resources["manifest"] for resources in fake._declared_video_streams.values()]
                closer.close()
                self.assertTrue(closer.base_closed)
                self.assertTrue(all(stream.closed for stream in streams))
                status=json.loads((output/"evaluation_video_status.json").read_text())
                self.assertEqual(status["expected_control_frames"],1)
                self.assertEqual(status["all_control_frames_recorded"],not missing)
                self.assertEqual(status["cameras"][7]["all_control_frames_recorded"],not missing)
                self.assertEqual(status["frames"],1)

    def test_render_advancing_physics_invalidates_the_entire_frame_set(self):
        with self.fixture() as (fake,output,native,_,__):
            camera=fake._evaluation_cameras[4]
            def advancing_update(**kwargs):
                camera.frame.add_(1);native[0]+=1
            camera.update=advancing_update
            fake._capture_evaluation_frame()
            self.assertIn("physics advanced between cameras",fake._video_status["failure"])
            self.assertEqual(fake._video_status["frames"],0)
            self.assertTrue(all(c["frames"]==0 for c in fake._video_status["cameras"]))
            self.assertEqual(len(list(output.glob("video/env_*/rgb_frames/*.png"))),0)

    def test_legacy_capture_body_is_unchanged_when_new_declaration_is_none(self):
        baseline=ROOT.parent/"g1-stage15/src/g1_vla/rl_environment.py"
        if not baseline.is_file():self.skipTest("Full04 baseline is available only in the local isolated review")
        def capture(path):
            tree=ast.parse(path.read_text());return next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=="_capture_evaluation_frame")
        before=capture(baseline);after=capture(ROOT/"src/g1_vla/rl_environment.py")
        self.assertEqual(ast.dump(ast.Module(body=after.body[1:],type_ignores=[]),include_attributes=False),
            ast.dump(ast.Module(body=before.body,type_ignores=[]),include_attributes=False))


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU Torch is required")
class RelativeContactMotionTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch
        self.values=dict(link_jacobian_world=torch.zeros(1,7,6,43),
            link_position_world=torch.zeros(1,7,3),joint_velocity_full43=torch.zeros(1,43),
            point_world=torch.zeros(1,7,32,3),inward_normal_world=torch.zeros(1,7,32,3),
            control_point_valid=torch.zeros(1,7,32,dtype=torch.bool),
            physical_point_loaded=torch.zeros(1,7,32,dtype=torch.bool),point_weight=torch.ones(1,7,32)*.101,
            cube_com_position_world=torch.tensor([[.05,0,0.]]),
            cube_com_linear_velocity_world=torch.zeros(1,3),cube_angular_velocity_world=torch.zeros(1,3))
        for body in (2,4,6):
            self.values["point_world"][0,body,0]=torch.tensor([.02,.01,0])
            self.values["inward_normal_world"][0,body,0,0]=1
            self.values["control_point_valid"][0,body,0]=True
            self.values["physical_point_loaded"][0,body,0]=True
        # This native DoF is deliberately outside the seven hand columns.
        self.values["joint_velocity_full43"][0,42]=1

    def motion(self):
        return current_relative_contact_motion(self.t,**self.values)

    def geometry_adapter(self):
        t=self.t;v=self.values
        tree=ast.parse((ROOT/"src/g1_vla/rl_environment.py").read_text())
        owner=next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=="Environment")
        node=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name=="_effort_contact_geometry")
        scope={"torch":t,"tensor":lambda value:value,"RLEnvironmentError":RLEnvironmentError,
            "settings":{"contact_offset_m":0.,"max_abs_normal_z":.6},
            "current_relative_contact_motion":current_relative_contact_motion,
            "native_relative_damping_state":native_relative_damping_state,
            "reconstruct_native_hand_inertia":reconstruct_native_hand_inertia}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"<actual-relative-motion-adapter>","exec"),scope)
        jac=t.zeros(1,8,6,43);jac[:,:7]=v["link_jacobian_world"]
        positions=t.zeros(1,9,3);positions[:,1:8]=v["link_position_world"]
        cube=SimpleNamespace(root_pos_w=v["cube_com_position_world"],root_com_pos_w=v["cube_com_position_world"],
            root_com_lin_vel_w=v["cube_com_linear_velocity_world"],root_com_ang_vel_w=v["cube_angular_velocity_world"])
        env=SimpleNamespace(_effort_contact_transition={"version":"g1_coordinated_contact_transition_v2_separating_preload"},
            robot=SimpleNamespace(num_bodies=9,data=SimpleNamespace(body_link_jacobian_w=jac,body_pos_w=positions,
                joint_vel=v["joint_velocity_full43"])),cube=SimpleNamespace(data=cube),
            _effort_hand_body_ids=list(range(1,8)),_action_ids=list(range(14)),_episode_ids=t.tensor([2]),
            sim=SimpleNamespace(get_physics_step_count=lambda:9))
        hand={"point":t.zeros(1,8,32,3),"point_normal":t.zeros(1,8,32,3),
            "point_valid":t.zeros(1,8,32,dtype=t.bool),"point_force":t.zeros(1,8,32),
            "separation":t.zeros(1,8,32)}
        for target,source in (("point","point_world"),("point_normal","inward_normal_world"),
                              ("point_valid","control_point_valid")):
            hand[target][:,:7]=v[source]
        hand["point_force"][:,:7]=t.where(v["physical_point_loaded"],v["point_weight"]-.001,0.)
        return env,hand,scope["_effort_contact_geometry"]

    def test_native_adapter_new_motion_is_current_and_unavailable_sdk_keeps_original_geometry(self):
        env,hand,method=self.geometry_adapter()
        method(env,hand)
        original=env._effort_geometry["normal_jacobian"].clone()
        motion=env._effort_geometry["relative_contact_motion"]
        self.assertTrue(motion["available"])
        self.assertTrue(motion["relative_velocity_current"].all())
        self.assertEqual(motion["native_step"],9)
        self.assertEqual(motion["episode_id"].tolist(),[2])
        self.assertEqual(env._effort_relative_motion_raw["link_origin_jacobian_world_full43_m"].shape,(1,7,6,43))
        del env.cube.data.root_com_pos_w
        method(env,hand)
        self.assertFalse(env._effort_geometry["relative_contact_motion"]["available"])
        self.assertTrue(self.t.equal(original,env._effort_geometry["normal_jacobian"]))
        self.assertTrue(env._effort_geometry["current_normal_available"].all())
        self.assertIsNone(env._effort_relative_motion_raw)

    def test_old_transition_never_requests_new_sdk_motion_fields(self):
        env,hand,method=self.geometry_adapter()
        env._effort_contact_transition={"version":"g1_coordinated_contact_transition_v1"}
        del env.cube.data.root_com_pos_w
        method(env,hand)
        original=env._effort_geometry["normal_jacobian"].clone()
        self.assertNotIn("relative_contact_motion",env._effort_geometry)
        env._effort_contact_transition=None
        method(env,hand)
        self.assertTrue(self.t.equal(original,env._effort_geometry["normal_jacobian"]))
        self.assertNotIn("relative_contact_motion",env._effort_geometry)

    def test_native_forwarding_requires_both_motion_and_geometry_epoch_and_sample_freshness(self):
        t=self.t;calls=[]
        tree=ast.parse((ROOT/"src/g1_vla/rl_environment.py").read_text())
        owner=next(node for node in ast.walk(tree) if isinstance(node,ast.ClassDef) and node.name=="Environment")
        node=next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name=="_update_effort_grasp")
        scope={"torch":t,"tensor":lambda value:value,"RLEnvironmentError":RLEnvironmentError,
            "quat_apply":lambda q,v:v,"quat_mul":lambda q,r:q}
        exec(compile(ast.Module(body=[node],type_ignores=[]),"<actual-relative-motion-forward>","exec"),scope)
        def update(**values):
            calls.append(values)
            evidence={"group_force_command_n":t.zeros(1,3)}
            if "relative_damping_state" in values:
                evidence["contact_transition"]={"relative_preload_damping":{
                    "actual_enabled_by_group":values["relative_damping_state"]["current"].clone()}}
            return t.zeros(1,7),evidence,t.zeros(1,dtype=t.bool)
        ids=t.tensor([2]);quat=t.zeros(1,9,4);quat[...,3]=1
        motion={"available":True,"episode_id":ids.clone(),"native_step":9,
            "relative_velocity_current":t.tensor([[True,False,True]]),
            "relative_inward_velocity_m_s":t.tensor([[-.01,-.02,-.03]]),
            "maximum_current_point_relative_inward_velocity_m_s":t.tensor([[-.009,-.019,-.029]])}
        env=SimpleNamespace(num_envs=1,device="cpu",_episode_ids=ids,_action_ids=list(range(14)),
            _palm_id=8,_palm_local=t.zeros(1,3),_palm_local_rotation=t.tensor([[0.,0,0,1]]),
            _effort_reference={"episode_id":ids.clone(),"nominal_hand_rad":t.zeros(1,7),
                "geometry_lower_hand_rad":t.full((1,7),-1.),"geometry_upper_hand_rad":t.full((1,7),1.),
                "closing_axis_hand7":t.tensor([-1.,0,0,1,1,1,1]),"palm_reference_world_m":t.zeros(1,3),
                "palm_reference_quaternion_xyzw":t.tensor([[0.,0,0,1]])},
            _last_contact_episode_ids=ids.clone(),_last_contact_native_step=9,
            _effort_geometry={"relative_contact_motion":motion,"episode_id":ids.clone(),"native_step":9,
                "current_normal_available":t.ones(1,3,dtype=t.bool),"normal_jacobian":t.zeros(1,3,7),
                "point_jacobian_world_m":t.zeros(1,3,3,7),"inward_normal_world":t.zeros(1,3,3),
                "point_world_m":t.zeros(1,3,3)},
            _effort_contact_transition={"version":"g1_coordinated_contact_transition_v2_separating_preload"},
            _effort_actual_domain_lower=t.full((1,7),-1.),_effort_actual_domain_upper=t.full((1,7),1.),
            _last_metrics={key:t.ones(1,dtype=t.bool) for key in
                ("finite","joint_safe","table_pair_measurement_supported","contact_safe")},
            _unsafe=t.zeros(1,dtype=t.bool),_limits=t.tensor([[[-2.,2.]]*43]),_velocity=t.ones(14),
            _effort_update_count=0,_control_command_id=1,_effort_grasp=SimpleNamespace(update=update),
            _write_control_probe=lambda row:None,_object_weight_support_torque=lambda **kw:(t.zeros(1,7),{}),
            sim=SimpleNamespace(get_physics_step_count=lambda:9),physics_dt=.002,
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=t.zeros(1,43),joint_vel=t.zeros(1,43),
                body_pos_w=t.zeros(1,9,3),body_quat_w=quat,gravity_compensation_forces=t.zeros(1,43))),
            cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=t.zeros(1,3))))
        env._last_metrics.update(group_physical_loaded_contact_count=t.ones(1,3,dtype=t.long),
            group_normal_force_n=t.full((1,3),.1),maximum_penetration_m=t.zeros(1))
        method=scope["_update_effort_grasp"]
        method(env)
        self.assertEqual(calls[-1]["relative_velocity_current"].tolist(),[[True,False,True]])
        self.assertTrue(t.equal(calls[-1]["relative_inward_velocity_m_s"],t.tensor([[-.01,0.,-.03]])))
        for stale in ("motion_epoch","motion_sample","geometry_epoch","geometry_sample"):
            motion["episode_id"]=ids.clone();motion["native_step"]=9
            env._effort_geometry["episode_id"]=ids.clone();env._effort_geometry["native_step"]=9
            if stale=="motion_epoch":motion["episode_id"]=ids+1
            if stale=="motion_sample":motion["native_step"]=8
            if stale=="geometry_epoch":env._effort_geometry["episode_id"]=ids+1
            if stale=="geometry_sample":env._effort_geometry["native_step"]=8
            method(env)
            self.assertFalse(calls[-1]["relative_velocity_current"].any(),stale)
            self.assertFalse(calls[-1]["relative_inward_velocity_m_s"].any(),stale)
        motion["available"]=False
        method(env)
        self.assertNotIn("relative_velocity_current",calls[-1])
        motion.update(available=True,episode_id=ids.clone(),native_step=9)
        env._effort_geometry.update(episode_id=ids.clone(),native_step=9)
        damping={"available":True,"episode_id":ids.clone(),"native_step":9,
            "hand_mass_matrix":t.eye(7)[None],"cube_inverse_spatial_inertia_world":t.eye(6)[None],
            "cube_normal_point_jacobian":t.zeros(1,3,6),"physical_point_count":t.ones(1,3,dtype=t.int64),
            "current":t.ones(1,3,dtype=t.bool)}
        env._effort_geometry["relative_damping_state"]=damping
        method(env)
        self.assertNotIn("relative_damping_state",calls[-1],"v2 cannot request a new inertia input")
        env._effort_contact_transition["version"]="g1_coordinated_contact_transition_v3_relative_viscous_preload"
        method(env)
        self.assertEqual(calls[-1]["relative_damping_state"]["current"].tolist(),[[True,False,True]])
        self.assertEqual(env._effort_actual_relative_damping_group_samples.tolist(),[[1,0,1]])
        for key in ("episode_id","native_step"):
            damping[key]=ids+1 if key=="episode_id" else 8
            method(env)
            self.assertFalse(calls[-1]["relative_damping_state"]["current"].any())
            damping[key]=ids.clone() if key=="episode_id" else 9
        self.assertEqual(env._effort_actual_relative_damping_group_samples.tolist(),[[1,0,1]],"Duplicate/native-stale inputs never inflate the cumulative enabled count")
        motion["available"]=False
        method(env)
        self.assertFalse(calls[-1]["relative_damping_state"]["current"].any())
        env._keep_current_hand_force_on_full_failure=True
        env._full_lift=SimpleNamespace(failed=t.tensor([False]))
        method(env)
        self.assertNotIn("freeze_reference_by_environment",calls[-1])
        env._full_lift.failed[:]=True
        method(env)
        self.assertTrue(calls[-1]["freeze_reference_by_environment"].all())
        self.assertFalse(env._unsafe.any())

    def test_common_rigid_motion_is_zero_relative_with_rotating_cube_com_offset(self):
        t=self.t;v=self.values
        v["link_position_world"][0,:,1]=t.arange(7)*.01
        com=v["cube_com_position_world"]
        velocity=t.tensor([[.03,-.01,.02]]);omega=t.tensor([[0.,0.,.2]])
        v["cube_com_linear_velocity_world"]=velocity
        v["cube_angular_velocity_world"]=omega
        link_velocity=velocity[:,None]+t.linalg.cross(omega[:,None].expand(1,7,3),
            v["link_position_world"]-com[:,None],dim=-1)
        v["link_jacobian_world"][0,:,:3,42]=link_velocity[0]
        v["link_jacobian_world"][0,:,3:,42]=omega.expand(7,3)
        motion,raw=self.motion()
        self.assertTrue(motion["relative_velocity_current"].all())
        self.assertTrue(t.allclose(motion["relative_inward_velocity_m_s"],t.zeros(1,3),atol=1e-8,rtol=0))
        self.assertTrue(t.allclose(motion["maximum_current_point_relative_inward_velocity_m_s"],t.zeros(1,3),atol=1e-8,rtol=0))
        full43_inward=t.einsum("ngi,ni->ng",raw["group_weighted_normal_jacobian_full43_m"],v["joint_velocity_full43"])
        self.assertTrue(t.allclose(full43_inward,motion["weighted_finger_inward_velocity_m_s"],atol=1e-8,rtol=0))
        self.assertTrue((full43_inward.abs()>0).all())

    def test_opening_finger_can_still_have_cube_approaching_it(self):
        t=self.t;v=self.values
        v["link_jacobian_world"][0,:,0,42]=-.01
        v["cube_com_linear_velocity_world"][0,0]=-.03
        motion,_=self.motion()
        self.assertTrue(motion["relative_velocity_current"].all())
        self.assertTrue(t.allclose(motion["weighted_finger_inward_velocity_m_s"],t.full((1,3),-.01)))
        self.assertTrue(t.allclose(motion["relative_inward_velocity_m_s"],t.full((1,3),.02)))
        self.assertTrue((motion["maximum_current_point_relative_inward_velocity_m_s"]>0).all())

    def test_one_closing_even_zero_force_point_cannot_be_hidden_by_negative_group_mean(self):
        t=self.t;v=self.values
        v["link_jacobian_world"][0,:,5,42]=.1
        v["point_world"][0,2,0]=t.tensor([.02,.02,0])
        v["point_world"][0,2,1]=t.tensor([.02,-.02,0])
        v["inward_normal_world"][0,2,1,0]=1
        v["control_point_valid"][0,2,1]=True
        v["physical_point_loaded"][0,2,1]=False
        v["point_weight"][0,2,1]=.001
        motion,_=self.motion()
        self.assertTrue(motion["relative_velocity_current"][0,0])
        self.assertLess(motion["relative_inward_velocity_m_s"][0,0].item(),0)
        self.assertAlmostEqual(motion["maximum_current_point_relative_inward_velocity_m_s"][0,0].item(),.002,places=7)
        self.assertEqual(motion["original_control_point_count"][0,0].item(),2)

    def test_any_physical_point_excluded_by_original_control_filter_disables_proof(self):
        v=self.values
        v["control_point_valid"][0,2,0]=False
        motion,_=self.motion()
        self.assertEqual(motion["relative_velocity_current"].tolist(),[[False,True,True]])
        self.assertEqual(motion["unproved_physical_loaded_point_count"].tolist(),[[1,0,0]])
        self.assertEqual(motion["relative_inward_velocity_m_s"][0,0].item(),0)

    def test_nan_motion_is_unknown_and_never_becomes_a_separation_certificate(self):
        v=self.values;v["cube_angular_velocity_world"][0,2]=float("nan")
        motion,_=self.motion()
        self.assertFalse(motion["relative_velocity_current"].any())
        for value in motion.values():
            if value.is_floating_point():
                self.assertTrue(self.t.isfinite(value).all())
        self.assertFalse(motion["maximum_current_point_relative_inward_velocity_m_s"].any())

    def test_different_point_normals_are_dotted_before_original_force_weighting(self):
        t=self.t;v=self.values
        v["point_world"][0,2,1]=t.tensor([.02,-.02,0])
        v["inward_normal_world"][0,2,1,1]=1
        v["control_point_valid"][0,2,1]=True
        v["physical_point_loaded"][0,2,1]=True
        v["point_weight"][0,2,1]=.201
        v["link_jacobian_world"][0,:,0,42]=.01
        v["link_jacobian_world"][0,:,1,42]=-.02
        v["link_jacobian_world"][0,:,5,42]=.1
        motion,raw=self.motion()
        expected=(.101*(.01-.1*.01)+.201*(-.02+.1*.02))/(.101+.201)
        self.assertAlmostEqual(motion["relative_inward_velocity_m_s"][0,0].item(),expected,places=7)
        recomputed=t.einsum("ngi,ni->ng",raw["group_weighted_normal_jacobian_full43_m"],v["joint_velocity_full43"])
        self.assertTrue(t.allclose(recomputed,motion["weighted_finger_inward_velocity_m_s"],atol=1e-8,rtol=0))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Native inertia adapters require real Torch")
class NativeRelativeDampingStateTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t=torch;t=torch
        j=t.zeros(1,9,6,43)
        for i in range(7):j[0,i+1,5,i+7]=1
        quat=t.zeros(1,9,4);quat[...,3]=1
        com=t.zeros(1,9,7);com[...,0]=.1;com[...,6]=1
        mass=t.eye(43)[None]*.01
        for i in range(7,14):mass[0,i,i]=.006
        armature=t.zeros(1,43);armature[:,7:14]=.001
        self.inertia=dict(link_jacobian_world_full43=j,link_quaternion_xyzw=quat,
            body_mass_kg=t.full((1,9),.2),body_inertia_prim_column_major=(t.eye(3)*.004).reshape(1,1,9).repeat(1,9,1),
            body_com_local_pose=com,native_mass_matrix_full43=mass,native_armature_full43=armature,
            hand_body_ids=list(range(1,8)),hand_joint_ids=list(range(7,14)))
        self.state=dict(hand_mass_matrix=t.eye(7)[None]*.007,hand_armature=t.zeros(1,7),
            cube_mass_kg=t.tensor([[.08]]),cube_inertia_prim_column_major=t.diag(t.tensor([.002,.003,.004])).reshape(1,9),
            cube_link_quaternion_xyzw=t.tensor([[0.,0.,math.sqrt(.5),math.sqrt(.5)]]),
            cube_normal_point_jacobian=t.zeros(1,3,6),physical_point_count=t.ones(1,3,dtype=t.int64),
            relative_velocity_current=t.ones(1,3,dtype=t.bool))

    def test_native_rigid_com_offset_reconstruction_classifies_excluded_and_included_armature_once(self):
        t=self.t;v=self.inertia
        dynamic,current,e=reconstruct_native_hand_inertia(t,**v)
        self.assertTrue(current.all());self.assertTrue(t.allclose(dynamic,t.eye(7)[None]*.007,atol=1e-8,rtol=0))
        self.assertEqual(e["native_mass_classification_code"].tolist(),[1])
        v["native_mass_matrix_full43"][:,7:14,7:14]=dynamic
        dynamic2,current,e=reconstruct_native_hand_inertia(t,**v)
        self.assertTrue(current.all());self.assertTrue(t.equal(dynamic2,dynamic))
        self.assertEqual(e["native_mass_classification_code"].tolist(),[2])
        v["native_mass_matrix_full43"][:,7:14,7:14]=t.eye(7)[None]*.05
        _,current,e=reconstruct_native_hand_inertia(t,**v)
        self.assertFalse(current.any());self.assertEqual(e["native_mass_classification_code"].tolist(),[0])

    def test_unmodelled_body_hand_jacobian_or_nonfinite_readback_is_unknown(self):
        t=self.t;v=self.inertia
        v["link_jacobian_world_full43"][0,8,0,7]=.01
        _,current,e=reconstruct_native_hand_inertia(t,**v)
        self.assertFalse(current.any());self.assertAlmostEqual(e["other_body_hand_jacobian_max_abs"].item(),.01,places=7)
        v["link_jacobian_world_full43"][0,8,0,7]=0
        v["body_mass_kg"][0,3]=float("nan")
        _,current,_=reconstruct_native_hand_inertia(t,**v)
        self.assertFalse(current.any())

    def test_cube_native_prim_inertia_rotates_to_world_without_assuming_isotropic_box(self):
        t=self.t;state,e=native_relative_damping_state(t,**self.state)
        self.assertTrue(state["current"].all());inverse=state["cube_inverse_spatial_inertia_world"]
        self.assertTrue(t.allclose(inverse[0,:3,:3],t.eye(3)*12.5))
        self.assertTrue(t.allclose(inverse[0,3:,3:],t.diag(t.tensor([1/.003,1/.002,1/.004])),atol=.0001,rtol=1e-6))
        self.assertFalse(e["cube_inertia_isotropic_readback"].any())
        self.state["cube_inertia_prim_column_major"]=t.eye(3).reshape(1,9)*.002
        _,e=native_relative_damping_state(t,**self.state)
        self.assertTrue(e["cube_inertia_isotropic_readback"].all())

    def test_unknown_or_nonspd_numeric_readback_disables_proof_without_nan_state(self):
        t=self.t
        for key in ("hand_mass_matrix","cube_mass_kg","cube_inertia_prim_column_major","cube_link_quaternion_xyzw"):
            with self.subTest(key=key):
                v=copy.deepcopy(self.state);v[key].reshape(-1)[0]=float("nan")
                state,_=native_relative_damping_state(t,**v)
                self.assertFalse(state["current"].any())
                for value in state.values():self.assertTrue(t.isfinite(value).all())
        v=copy.deepcopy(self.state);v["hand_mass_matrix"].zero_()
        state,_=native_relative_damping_state(t,**v);self.assertFalse(state["current"].any())
        v=copy.deepcopy(self.state);v["hand_armature"][:]=.001
        state,_=native_relative_damping_state(t,**v);self.assertFalse(state["current"].any())
        v["hand_mass_semantics_current"]=t.ones(1,dtype=t.bool)
        state,_=native_relative_damping_state(t,**v);self.assertTrue(state["current"].all())

    def test_cube_spatial_normal_is_weighted_per_point_before_aggregation(self):
        helper=RelativeContactMotionTests();helper.setUp();v=helper.values;t=self.t
        v["cube_com_position_world"].zero_();v["point_world"][0,2,0]=t.tensor([0.,.01,0.])
        v["point_world"][0,2,1]=t.tensor([.02,0.,0.]);v["inward_normal_world"][0,2,1]=t.tensor([0.,1.,0.])
        v["control_point_valid"][0,2,1]=True;v["physical_point_loaded"][0,2,1]=True
        v["point_weight"][0,2,0]=.25;v["point_weight"][0,2,1]=.75
        motion,_=current_relative_contact_motion(t,**v,include_cube_point_jacobian=True)
        expected=t.tensor([.25,.75,0.,0.,0.,.0125])
        self.assertTrue(t.allclose(motion["cube_normal_point_jacobian"][0,0],expected))
        original,_=current_relative_contact_motion(t,**v)
        self.assertNotIn("cube_normal_point_jacobian",original,"Original v2 output schema is unchanged")

    def test_native_v3_reads_properties_once_and_rigid_contact_group_falls_back_without_filtering_points(self):
        helper=RelativeContactMotionTests();helper.setUp();env,hand,method=helper.geometry_adapter();t=self.t
        env._effort_contact_transition["version"]="g1_coordinated_contact_transition_v3_relative_viscous_preload"
        v=self.inertia;data=env.robot.data
        data.body_link_jacobian_w=v["link_jacobian_world_full43"][:,1:]
        data.body_quat_w=v["link_quaternion_xyzw"];data.mass_matrix=v["native_mass_matrix_full43"]
        env.robot.body_names=["body"+str(i) for i in range(9)]
        env.robot.joint_names=["joint"+str(i) for i in range(43)]
        calls=[];rows=[]
        def getter(key):
            def get():calls.append(key);return v[key]
            return get
        env.robot.root_view=SimpleNamespace(get_masses=getter("body_mass_kg"),get_inertias=getter("body_inertia_prim_column_major"),
            get_coms=getter("body_com_local_pose"),get_dof_armatures=getter("native_armature_full43"))
        env.cube.root_view=SimpleNamespace(get_masses=lambda:t.tensor([[.08]]),get_inertias=lambda:t.eye(3).reshape(1,9)*.002)
        env.cube.data.root_link_quat_w=t.tensor([[0.,0.,0.,1.]])
        env._write_control_probe=rows.append
        method(env,hand)
        damping=env._effort_geometry["relative_damping_state"]
        self.assertTrue(damping["available"]);self.assertTrue(damping["current"].all())
        self.assertEqual(len(calls),4);self.assertEqual(len(rows),1)
        self.assertEqual(env._effort_valid_mass_environment_samples.tolist(),[1])
        self.assertEqual(env._effort_relative_motion_raw["native_mass_matrix_full43"].shape,(1,43,43))
        hand["point"][0,0,0]=hand["point"][0,2,0]
        hand["point_normal"][0,0,0]=hand["point_normal"][0,2,0]
        hand["point_valid"][0,0,0]=True;hand["point_force"][0,0,0]=.1
        method(env,hand)
        self.assertEqual(len(calls),4);self.assertEqual(len(rows),1)
        self.assertEqual(env._effort_valid_mass_environment_samples.tolist(),[1],"Same native/episode is counted once")
        damping=env._effort_geometry["relative_damping_state"]
        self.assertEqual(damping["current"].tolist(),[[False,True,True]])
        self.assertEqual(damping["native_inertia_readback"]["nonsoft_current_control_or_physical_point_count"].tolist(),[[1,0,0]])
        self.assertEqual(env._effort_geometry["relative_contact_motion"]["original_control_point_count"].tolist(),[[2,1,1]])
        original=env._effort_geometry["normal_jacobian"].clone()
        del data.mass_matrix
        method(env,hand)
        self.assertFalse(env._effort_geometry["relative_damping_state"]["available"])
        self.assertTrue(env._effort_geometry["relative_contact_motion"]["available"])
        self.assertTrue(t.equal(env._effort_geometry["normal_jacobian"],original),"Missing native inertia must retain the old normal-control geometry")
        self.assertFalse(env._effort_native_inertia_status["available"])

    def test_small_audit_exposes_only_versioned_scalar_status_without_native_getters(self):
        helper=NativeArmTargetScheduleTests();helper.setUp();method=helper.method("_relative_damping_small_status")
        fake=SimpleNamespace(_effort_contact_transition=None)
        self.assertIsNone(method(fake))
        for version in ("g1_coordinated_contact_transition_v1","g1_coordinated_contact_transition_v2_separating_preload"):
            fake._effort_contact_transition={"version":version};self.assertIsNone(method(fake))
        fake._effort_contact_transition={"version":"g1_coordinated_contact_transition_v3_relative_viscous_preload"}
        self.assertEqual(method(fake)["state"],"not_measured_yet")
        fake._effort_native_inertia_status={"available":True,"classification_code":[1],
            "hand_mass_semantics_current":[True],"raw_sdk_minus_rigid_max_abs_kg_m2":[1e-9]}
        fake._effort_valid_mass_environment_samples=[99]
        fake._effort_actual_relative_damping_group_samples=[[2,3,4]]
        result=method(fake)
        self.assertEqual(result["valid_mass_environment_samples"],[99])
        self.assertEqual(result["actual_relative_damping_group_environment_samples"],[[2,3,4]])
        self.assertNotIn("hand_mass_matrix",result)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch-free host; run sparse contact tests in the native runtime")
class SparseContactReductionTest(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch

    def make_contacts(self):
        torch = self.torch
        width, hand = 11, [5, 1, 9, 2, 10, 4, 7, 8]
        counts = torch.zeros(2, width, dtype=torch.int64)
        starts = torch.zeros_like(counts)
        force = torch.full((32,), float("nan"))
        points, normals = torch.zeros(32, 3), torch.zeros(32, 3)
        separation = torch.zeros(32)
        normal = torch.zeros(2, width, 3)
        for env, partner, slot, value, direction in (
                (0, 0, 19, .015, (0, 0, 1)),
                (0, hand[2], 7, .1, (1, 0, 0)),
                (0, hand[4], 2, .2, (-1, 0, 0)),
                (0, hand[6], 27, .3, (-1, 0, 0)),
                (1, hand[0], 12, .01, (0, 0, 1))):
            counts[env, partner], starts[env, partner] = 1, slot
            force[slot], normals[slot] = value, torch.tensor(direction)
            separation[slot], points[slot] = -.0001, torch.tensor((slot*.001, env, .8))
            normal[env, partner] = value*normals[slot]
        return dict(normal=normal, friction=torch.zeros_like(normal), net_normal=normal.sum(1),
                    force=force, points=points, normals=normals, separation=separation,
                    counts=counts, starts=starts, hand_indices=hand, force_threshold=.02)

    def test_named_noncontiguous_partners_and_raw_contact_semantics(self):
        torch = self.torch
        values = self.make_contacts()
        result = _reduce_contact_pairs(torch, **values)
        hand = result["hand"]
        self.assertAlmostEqual(result["fn"][0, 0].item(), .015)
        self.assertEqual(result["physical_loaded_count"][0, 0].item(), 1)
        self.assertEqual(hand["physical_loaded_count"][1, 0].item(), 1)
        self.assertFalse(hand["point_loaded"][1, 0].any())
        self.assertTrue(torch.equal(hand["normal"], -values["normal"][:, values["hand_indices"]]))
        self.assertEqual(hand["point_normal"][0, 2, 0].tolist(), [1, 0, 0])
        self.assertAlmostEqual(hand["point_force"][0, 2, 0].item(), -values["force"][7].item())
        self.assertEqual(hand["point"][0, 2, 0].tolist(), values["points"][7].tolist())
        summary = _summarize_hand_contacts(torch, hand)
        self.assertTrue(torch.allclose(summary["groups"][0], torch.tensor([.1, .2, .3])))
        self.assertEqual(summary["opposition"][0].item(), -1)
        self.assertEqual(summary["opposition"][1].item(), 1)

    def test_reversed_view_keeps_canonical_normals_and_reverses_signed_scalar(self):
        torch = self.torch
        values = self.make_contacts()
        # Recorded native evidence: the Cube and hand share the point normal,
        # but their scalar forces have opposite signs, including signed zero.
        values["force"][7] = -.7766782
        values["normals"][7] = torch.tensor((-.702233, -.711947, -.000703))
        values["normal"][0, values["hand_indices"][2]] = values["force"][7]*values["normals"][7]
        values["net_normal"] = values["normal"].sum(1)
        result = _reduce_contact_pairs(torch, **values)
        hand = result["hand"]
        self.assertTrue(torch.equal(hand["point_normal"][0, 2, 0], values["normals"][7]))
        self.assertAlmostEqual(hand["point_force"][0, 2, 0].item(), .7766782, places=6)
        reconstructed = (hand["point_force"][..., None]*hand["point_normal"]).sum(2)
        self.assertTrue(torch.allclose(reconstructed, hand["normal"]))
        cube_point = result["raw"]["bucket"] == values["hand_indices"][2]
        self.assertEqual(result["raw"]["force"][cube_point].item(), values["force"][7].item())

    def test_friction_counts_and_starts_keep_distinct_semantics_and_copy_alias(self):
        torch = self.torch
        forces = torch.arange(24, dtype=torch.float32).reshape(8, 3)
        counts, starts = torch.tensor([[2, 1]]), torch.tensor([[4, 0]])
        result, raw = _reduce_friction_pairs(torch, forces, torch.zeros_like(forces), counts, starts)
        self.assertTrue(torch.equal(result[0, 0], forces[4]+forces[5]))
        self.assertTrue(torch.equal(result[0, 1], forces[0]))
        counts.fill_(0)
        starts.fill_(7)
        self.assertEqual(raw["counts"].tolist(), [[2, 1]])
        self.assertEqual(raw["starts"].tolist(), [[4, 0]])

    def test_invalid_indices_and_selected_nan_fail_closed(self):
        torch = self.torch
        values = self.make_contacts()
        values["force"][7] = float("nan")
        with self.assertRaises(FloatingPointError):
            _reduce_contact_pairs(torch, **values)
        values = self.make_contacts()
        values["starts"][0, 0] = 32
        with self.assertRaises(RLEnvironmentError):
            _reduce_contact_pairs(torch, **values)

    def test_positive_gap_with_large_solver_force_cannot_qualify_as_side_contact(self):
        torch = self.torch
        values = self.make_contacts()
        for slot in (7, 2, 27):
            values["force"][slot] = .85
            values["separation"][slot] = .000058
        hand = _reduce_contact_pairs(torch, **values)["hand"]
        summary = _summarize_hand_contacts(torch, hand)
        self.assertTrue((summary["groups"][0] > .02).all())
        self.assertEqual(summary["group_physical_loaded_count"][0].tolist(), [0, 0, 0])
        self.assertFalse(hand["point_loaded"][0].any())
        self.assertEqual(summary["opposition"][0].item(), 1)
        # Touching exactly at zero gap is a physical contact; a positive gap
        # is excluded even when PhysX reports a nonzero solver impulse.
        values["separation"][[7, 2, 27]] = 0
        summary = _summarize_hand_contacts(torch, _reduce_contact_pairs(torch, **values)["hand"])
        self.assertEqual(summary["group_physical_loaded_count"][0].tolist(), [1, 1, 1])
        self.assertEqual(summary["opposition"][0].item(), -1)

    def test_side_normal_limit_includes_physical_points_below_distal_threshold(self):
        torch = self.torch
        result = _reduce_contact_pairs(torch, **self.make_contacts())
        # Env 1 has only 10mN on a near-vertical actual thumb contact.
        self.assertFalse(result["hand"]["point_loaded"][1].any())
        self.assertEqual(_summarize_hand_contacts(torch, result["hand"])["normal_vertical"][1].item(), 1)

    def test_side_normal_checks_current_cube_frame_as_well_as_world_frame(self):
        torch = self.torch
        values = self.make_contacts()
        # A horizontal world normal can point through the top of a tilted cube.
        values["object_up_world"] = torch.tensor([[1., 0, 0], [0, 0, 1.]])
        summary = _summarize_hand_contacts(torch, _reduce_contact_pairs(torch, **values)["hand"])
        self.assertEqual(summary["normal_vertical"][0].item(), 1)

    def test_actual_low_force_plus_large_positive_gap_load_is_not_certified(self):
        torch = self.torch
        values = self.make_contacts()
        partner = values["hand_indices"][2]
        values["counts"][0, partner] = 2
        values["force"][7], values["force"][8] = .03, .85
        values["separation"][8], values["normals"][8] = .000058, values["normals"][7]
        hand = _reduce_contact_pairs(torch, **values)["hand"]
        summary = _summarize_hand_contacts(torch, hand)
        self.assertTrue((summary["group_physical_loaded_count"][0] > 0).all())
        self.assertEqual(summary["opposition"][0].item(), -1)
        self.assertFalse(summary["no_positive_gap_hand_load"][0])

    def test_friction_partner_needs_nonempty_fully_nonpositive_normal_manifold(self):
        torch = self.torch
        values = self.make_contacts()
        partner = values["hand_indices"][2]
        patches = torch.zeros_like(values["counts"])
        patches[0, partner] = 1
        values["friction_nonzero_patch_count"] = patches
        summary = _summarize_hand_contacts(torch, _reduce_contact_pairs(torch, **values)["hand"])
        self.assertTrue(summary["friction_manifold_certified"][0])
        # Even a zero-Fn positive-gap point prevents certifying a partner
        # carrying friction; the API provides no separate patch separation.
        values["counts"][0, partner] = 2
        values["force"][8], values["separation"][8] = 0, .000058
        summary = _summarize_hand_contacts(torch, _reduce_contact_pairs(torch, **values)["hand"])
        self.assertTrue(summary["no_positive_gap_hand_load"][0])
        self.assertFalse(summary["friction_manifold_certified"][0])
        values["counts"][0, partner] = 0
        summary = _summarize_hand_contacts(torch, _reduce_contact_pairs(torch, **values)["hand"])
        self.assertFalse(summary["friction_manifold_certified"][0])

    def test_cancelling_friction_patches_remain_nonzero_for_certification(self):
        torch = self.torch
        forces = torch.tensor([[1., 0, 0], [-1., 0, 0], [0, 0, 0]])
        aggregate, raw = _reduce_friction_pairs(torch, forces, torch.zeros_like(forces),
                                               torch.tensor([[2]]), torch.tensor([[0]]))
        self.assertFalse(aggregate.ne(0).any())
        self.assertEqual(raw["nonzero_patch_count"].item(), 2)

    def test_saturated_buffers_cannot_certify_complete_manifolds(self):
        torch = self.torch
        forces = torch.ones(2, 3)
        with self.assertRaises(RLEnvironmentError):
            _reduce_friction_pairs(torch, forces, forces, torch.tensor([[2]]), torch.tensor([[0]]))
        values = self.make_contacts()
        values["force"] = values["force"][:28]
        values["counts"][0, 0] = 28
        values["starts"][0, 0] = 0
        values["force"].nan_to_num_(0)
        with self.assertRaises(RLEnvironmentError):
            _reduce_contact_pairs(torch, **values)

    def test_crosscheck_matches_world_points_without_raw_array_order(self):
        torch = self.torch
        hand = _reduce_contact_pairs(torch, **self.make_contacts())["hand"]
        reference = {name: value.clone() for name, value in hand.items()}
        for name in ("point", "point_normal", "point_force", "point_valid", "point_loaded", "separation"):
            reference[name] = reference[name].flip(2)
        self.assertTrue(_compare_hand_contacts(torch, hand, reference)["passed"])
        reference["point"][0, 2, -1, 0] += .001
        self.assertFalse(_compare_hand_contacts(torch, hand, reference)["passed"])

    def test_opposition_uses_worst_loaded_point_not_mean_normal(self):
        torch = self.torch
        hand = _reduce_contact_pairs(torch, **self.make_contacts())["hand"]
        hand["point_normal"][0, 4, 1] = hand["point_normal"][0, 2, 0]
        hand["point_loaded"][0, 4, 1] = True
        self.assertEqual(_summarize_hand_contacts(torch, hand)["opposition"][0].item(), 1)


@unittest.skipUnless(importlib.util.find_spec("torch"), "native load telemetry math requires real Torch")
class NativeJointLoadDiagnosticTests(unittest.TestCase):
    def setUp(self):
        import torch
        from g1_vla.rl_environment import NativeJointLoadDiagnostics
        self.t = torch
        self.cls = NativeJointLoadDiagnostics
        self.joints = ["joint"+str(i) for i in range(43)]
        self.bodies = ["body"+str(i) for i in range(55)]
        class View:
            count = 2
            shared_metatype = SimpleNamespace(dof_names=self.joints, link_names=self.bodies)
            def __init__(inner):
                inner.calls = {}
                inner.projected = torch.arange(86, dtype=torch.float32).reshape(2, 43)
                inner.incoming = torch.arange(660, dtype=torch.float32).reshape(2, 55, 6)
            def call(inner, name, value):
                inner.calls[name] = inner.calls.get(name, 0)+1
                return value
            def get_dof_max_forces(inner):
                return inner.call("max", torch.full((2, 43), 1.4))
            def get_dof_friction_properties(inner):
                return inner.call("friction", torch.tensor([0., 0., .02]).expand(2, 43, 3))
            def get_dof_projected_joint_forces(inner):
                return inner.call("projected", inner.projected)
            def get_link_incoming_joint_force(inner):
                return inner.call("incoming", inner.incoming)
        self.view = View()
        self.diag = self.cls(torch, lambda x: x, self.view, self.joints, self.bodies, 2)

    def snapshot(self, controls=(1, 1)):
        return self.diag.snapshot(33, self.t.tensor([7, 8]), self.t.tensor(controls), phase="post_physics_pre_reset")

    def test_native_property_readback_once_with_exact_binding_and_friction_order(self):
        self.snapshot(); self.snapshot()
        self.assertEqual(self.view.calls["max"], 1)
        self.assertEqual(self.view.calls["friction"], 1)
        self.assertEqual(self.diag.readback["binding"]["joint_names"], self.joints)
        self.assertEqual(self.diag.readback["friction_last_dimension"],
            ["static_friction_effort_nm", "dynamic_friction_effort_nm", "viscous_friction_coefficient_nm_s_rad"])

    def test_reset_stale_rows_are_excluded_not_zeroed(self):
        row = self.snapshot((0, 4))
        self.assertEqual(row["env_ids"], [1])
        self.assertEqual(row["invalid_for_current_episode_env_ids"], [0])
        self.assertTrue(self.t.equal(row["projected_joint_force_nm"]["values"], self.view.projected[1:]))
        self.assertEqual(row["incoming_joint_wrench_child"]["values"].shape, (1, 55, 6))

    def test_all_reset_boundary_is_unavailable_without_native_reads(self):
        row = self.snapshot((0, 0))
        self.assertFalse(row["available"])
        self.assertIsNone(row["projected_joint_force_nm"]["values"])
        self.assertNotIn("projected", self.view.calls)
        self.assertNotIn("incoming", self.view.calls)

    def test_getter_failure_and_nonfinite_channels_export_none(self):
        def missing(): raise RuntimeError("native channel unsupported")
        self.view.get_dof_projected_joint_forces = missing
        self.view.incoming[0, 1, 2] = float("nan")
        row = self.snapshot()
        self.assertFalse(row["available"])
        self.assertIsNone(row["projected_joint_force_nm"]["values"])
        self.assertIn("unsupported", row["projected_joint_force_nm"]["unavailable_reason"])
        self.assertIsNone(row["incoming_joint_wrench_child"]["values"])
        self.assertIn("nonfinite", row["incoming_joint_wrench_child"]["unavailable_reason"])

    def test_shape_failure_and_native_buffer_reuse_are_not_silenced(self):
        row = self.snapshot()
        copied = row["projected_joint_force_nm"]["values"].clone()
        self.view.projected.fill_(-123.)
        self.assertTrue(self.t.equal(row["projected_joint_force_nm"]["values"], copied))
        self.view.incoming = self.t.zeros(2, 54, 6)
        bad = self.snapshot()["incoming_joint_wrench_child"]
        self.assertFalse(bad["available"])
        self.assertIsNone(bad["values"])
        self.assertIn("shape", bad["unavailable_reason"])

    def test_binding_mismatch_cannot_assign_loads_to_declared_names(self):
        self.view.shared_metatype = SimpleNamespace(dof_names=self.joints[::-1], link_names=self.bodies)
        diag = self.cls(self.t, lambda x: x, self.view, self.joints, self.bodies, 2)
        row = diag.snapshot(33, self.t.tensor([7, 8]), self.t.tensor([1, 1]), phase="test")
        self.assertFalse(diag.readback["binding"]["verified"])
        self.assertFalse(row["available"])
        self.assertIsNone(row["projected_joint_force_nm"]["values"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "native self-pair helper requires real Torch")
class NativeSelfPairDiagnosticTests(unittest.TestCase):
    def setUp(self):
        import torch
        from g1_vla.rl_environment import NativeSelfPairDiagnostics
        self.t = torch
        self.cls = NativeSelfPairDiagnostics
        self.pairs = [{"env_id": e, "sensor_body": b,
            "sensor_path": "/World/envs/env_"+str(e)+"/Robot/"+b,
            "filter_path": "/World/envs/env_"+str(e)+"/Robot/right_hand_camera_base_link"}
            for e in range(2) for b in ("right_hand_thumb_0_link", "right_hand_thumb_1_link")]
        class Attr:
            def __init__(inner, value): inner.value = value
            def IsValid(inner): return inner.value is not None
            def Get(inner): return inner.value
        class Prim:
            def IsValid(inner): return True
            def GetAppliedSchemas(inner): return ["PhysicsRigidBodyAPI", "PhysxContactReportAPI"]
            def GetAttribute(inner, name): return Attr(0.)
        self.stage = SimpleNamespace(GetPrimAtPath=lambda path: Prim())
        class View:
            sensor_count, filter_count, max_contact_data_count = 4, 1, 128
            sensor_paths = [r["sensor_path"] for r in self.pairs]
            filter_paths = [[r["filter_path"]] for r in self.pairs]
            def __init__(inner):
                inner.calls = 0
                inner.net = torch.zeros(4, 3)
                inner.matrix = torch.zeros(4, 1, 3)
                inner.force, inner.sep = torch.zeros(128), torch.zeros(128)
                inner.point, inner.normal = torch.zeros(128, 3), torch.zeros(128, 3)
                inner.counts, inner.starts = torch.zeros(4, 1, dtype=torch.int32), torch.zeros(4, 1, dtype=torch.int32)
            def get_net_contact_forces(inner, dt): self.assertEqual(dt, .002); return inner.net
            def get_contact_force_matrix(inner, dt): return inner.matrix
            def get_contact_data(inner, dt):
                inner.calls += 1
                return inner.force, inner.point, inner.normal, inner.sep, inner.counts, inner.starts
        self.view = View()
        self.create_calls = []
        def create(sensors, *, filter_patterns, max_contact_data_count):
            self.create_calls.append((sensors, filter_patterns, max_contact_data_count))
            return self.view
        self.sim = SimpleNamespace(create_rigid_contact_view=create)
        self.diag = self.cls(torch, lambda x: x, self.sim, self.stage, self.pairs, 2, .002)

    def snapshot(self, controls=(1, 1), phase="post_physics_pre_reset", step=33):
        return self.diag.snapshot(step, self.t.tensor([7, 8]), self.t.tensor(controls), phase=phase)

    def add_contact(self, sensor=0):
        self.view.counts[sensor, 0] = 1
        self.view.force[0], self.view.normal[0, 0], self.view.sep[0] = 2., 1., -.0001
        self.view.matrix[sensor, 0, 0], self.view.net[sensor, 0] = 2., 3.

    def test_exact_batched_routing_and_no_positive_witness_is_not_absence(self):
        self.assertEqual(len(self.create_calls), 1)
        self.assertEqual(self.create_calls[0][0], [r["sensor_path"] for r in self.pairs])
        self.assertEqual(self.create_calls[0][1], [[r["filter_path"]] for r in self.pairs])
        row = self.snapshot()
        self.assertTrue(row["available"])
        self.assertEqual(row["positive_witness_sensor_indices"], [])
        self.assertFalse(row["absence_of_selfcollision_certified"])
        self.assertEqual(self.diag.summary()["self_pair_reporting_status"], "no_positive_witness_yet")

    def test_raw_mapping_positive_witness_and_other_partner_residual(self):
        self.add_contact()
        row = self.snapshot()
        self.assertEqual(row["positive_witness_sensor_indices"], [0])
        self.assertEqual(row["values"]["raw_sensor_indices"].tolist(), [0])
        self.assertAlmostEqual(float(row["values"]["other_partner_normal_residual_world_n"][0, 0]), 1.)
        self.assertEqual(self.diag.summary()["positive_witness_post_control_samples"], [1, 0, 0, 0])
        self.snapshot(phase="pre_command")
        self.snapshot()
        self.assertEqual(self.view.calls, 1)
        self.assertEqual(self.diag.summary()["positive_witness_post_control_samples"], [1, 0, 0, 0])
        self.view.force.fill_(-9.)
        self.assertEqual(row["values"]["signed_normal_force_n"].tolist(), [2.])

    def test_reset_stale_pairs_are_excluded_and_allreset_does_not_read(self):
        self.add_contact(sensor=0)
        row = self.snapshot(controls=(0, 1))
        self.assertEqual(row["sensor_indices"], [2, 3])
        self.assertEqual(row["positive_witness_sensor_indices"], [])
        self.assertEqual(row["values"]["raw_sensor_indices"].numel(), 0)
        calls = self.view.calls
        cold = self.snapshot(controls=(0, 0))
        self.assertFalse(cold["available"])
        self.assertIsNone(cold["values"])
        self.assertEqual(self.view.calls, calls)

    def test_missing_report_or_censored_threshold_does_not_create_or_set(self):
        for apis, threshold in ((["PhysicsRigidBodyAPI"], 0.),
                                (["PhysicsRigidBodyAPI", "PhysxContactReportAPI"], 1.),
                                (["PhysicsRigidBodyAPI", "PhysxContactReportAPI"], float("nan"))):
            attr = SimpleNamespace(IsValid=lambda: True, Get=lambda: threshold)
            prim = SimpleNamespace(IsValid=lambda: True, GetAppliedSchemas=lambda: apis, GetAttribute=lambda _: attr)
            stage = SimpleNamespace(GetPrimAtPath=lambda _: prim)
            calls = len(self.create_calls)
            diag = self.cls(self.t, lambda x: x, self.sim, stage, self.pairs, 2, .002)
            self.assertFalse(diag.readback["available"])
            json.dumps(diag.summary(), allow_nan=False)
            self.assertEqual(len(self.create_calls), calls)
            self.assertIsNone(diag.snapshot(33, self.t.tensor([7,8]), self.t.tensor([1,1]), phase="test")["values"])

    def test_binding_mismatch_and_raw_overflow_are_missing_not_zero(self):
        self.view.filter_paths = self.view.filter_paths[::-1]
        bad = self.cls(self.t, lambda x: x, self.sim, self.stage, self.pairs, 2, .002)
        self.assertFalse(bad.readback["available"])
        self.view.counts[0, 0] = 128
        row = self.snapshot()
        self.assertFalse(row["available"])
        self.assertIsNone(row["values"])
        self.assertIn("saturated", row["unavailable_reason"])

    def test_nonfinite_active_or_bad_reconstruction_never_creates_witness(self):
        self.add_contact()
        self.view.matrix[0, 0, 0] = 4.
        row = self.snapshot()
        self.assertFalse(row["available"])
        self.assertEqual(row["positive_witness_sensor_indices"], [])
        self.view.matrix[0, 0, 0] = 2.
        self.view.force[0] = float("nan")
        row = self.snapshot(step=66)
        self.assertFalse(row["available"])
        self.assertIn("nonfinite", row["unavailable_reason"])


    def test_exact8env_16_unique_sensors_share_one_batched_view(self):
        pairs = [{"env_id": e, "sensor_body": b,
            "sensor_path": "/World/envs/env_"+str(e)+"/Robot/"+b,
            "filter_path": "/World/envs/env_"+str(e)+"/Robot/right_hand_camera_base_link"}
            for e in range(8) for b in ("right_hand_thumb_0_link", "right_hand_thumb_1_link")]
        self.view.sensor_count, self.view.max_contact_data_count = 16, 512
        self.view.sensor_paths = [r["sensor_path"] for r in pairs]
        self.view.filter_paths = [[r["filter_path"]] for r in pairs]
        calls = len(self.create_calls)
        diag = self.cls(self.t, lambda x: x, self.sim, self.stage, pairs, 8, .002)
        self.assertTrue(diag.readback["available"])
        self.assertEqual(len(self.create_calls)-calls, 1)
        self.assertEqual(len(set(self.create_calls[-1][0])), 16)
        self.assertEqual(self.create_calls[-1][2], 512)

    def test_nonfinite_reset_stale_rows_do_not_poison_valid_episode_pairs(self):
        self.add_contact(sensor=0)
        self.view.force[0] = float("nan")
        self.view.matrix[0, 0, 0] = float("nan")
        self.view.net[0, 0] = float("nan")
        row = self.snapshot(controls=(0, 1))
        self.assertTrue(row["available"])
        self.assertEqual(row["sensor_indices"], [2, 3])
        self.assertEqual(row["values"]["raw_native_indices"].numel(), 0)


    def test_native_uint32_indices_and_column_force_layout(self):
        self.add_contact()
        self.view.counts = self.t.tensor([[1], [0], [0], [0]], dtype=self.t.uint32)
        self.view.starts = self.t.tensor([[0], [4294967295], [4294967295], [4294967295]], dtype=self.t.uint32)
        self.view.force = self.view.force[:, None]
        self.view.sep = self.view.sep[:, None]
        row = self.snapshot()
        self.assertTrue(row["available"], row.get("unavailable_reason"))
        self.assertEqual(row["positive_witness_sensor_indices"], [0])
        self.assertEqual(row["values"]["counts"].dtype, self.t.int64)
        self.assertEqual(row["values"]["starts_native_buffer"].dtype, self.t.int64)
        self.assertEqual(row["values"]["signed_normal_force_n"].tolist(), [2.])

    def test_float_indices_are_rejected_even_with_integer_values(self):
        self.add_contact()
        self.view.counts = self.view.counts.float()
        row = self.snapshot()
        self.assertFalse(row["available"])
        self.assertIsNone(row["values"])
        self.assertIn("integer", row["unavailable_reason"])



class FixedPalmAdjacencyFilterTest(unittest.TestCase):
    # The entire independently audited official graph, including the 12 fixed
    # joints. This fixture is not computed from the incoming filter contract.
    GRAPH = [('d435_joint', 'PhysicsFixedJoint', 'torso_link', 'd435_link'), ('head_joint', 'PhysicsFixedJoint', 'torso_link', 'head_link'), ('imu_in_pelvis_joint', 'PhysicsFixedJoint', 'pelvis', 'imu_in_pelvis'), ('imu_in_torso_joint', 'PhysicsFixedJoint', 'torso_link', 'imu_in_torso'), ('left_ankle_pitch_joint', 'PhysicsRevoluteJoint', 'left_knee_link', 'left_ankle_pitch_link'), ('left_ankle_roll_joint', 'PhysicsRevoluteJoint', 'left_ankle_pitch_link', 'left_ankle_roll_link'), ('left_elbow_joint', 'PhysicsRevoluteJoint', 'left_shoulder_yaw_link', 'left_elbow_link'), ('left_hand_camera_base_joint', 'PhysicsFixedJoint', 'left_wrist_yaw_link', 'left_hand_camera_base_link'), ('left_hand_index_0_joint', 'PhysicsRevoluteJoint', 'left_hand_palm_link', 'left_hand_index_0_link'), ('left_hand_index_1_joint', 'PhysicsRevoluteJoint', 'left_hand_index_0_link', 'left_hand_index_1_link'), ('left_hand_middle_0_joint', 'PhysicsRevoluteJoint', 'left_hand_palm_link', 'left_hand_middle_0_link'), ('left_hand_middle_1_joint', 'PhysicsRevoluteJoint', 'left_hand_middle_0_link', 'left_hand_middle_1_link'), ('left_hand_palm_joint', 'PhysicsFixedJoint', 'left_wrist_yaw_link', 'left_hand_palm_link'), ('left_hand_thumb_0_joint', 'PhysicsRevoluteJoint', 'left_hand_palm_link', 'left_hand_thumb_0_link'), ('left_hand_thumb_1_joint', 'PhysicsRevoluteJoint', 'left_hand_thumb_0_link', 'left_hand_thumb_1_link'), ('left_hand_thumb_2_joint', 'PhysicsRevoluteJoint', 'left_hand_thumb_1_link', 'left_hand_thumb_2_link'), ('left_hip_pitch_joint', 'PhysicsRevoluteJoint', 'pelvis', 'left_hip_pitch_link'), ('left_hip_roll_joint', 'PhysicsRevoluteJoint', 'left_hip_pitch_link', 'left_hip_roll_link'), ('left_hip_yaw_joint', 'PhysicsRevoluteJoint', 'left_hip_roll_link', 'left_hip_yaw_link'), ('left_knee_joint', 'PhysicsRevoluteJoint', 'left_hip_yaw_link', 'left_knee_link'), ('left_shoulder_pitch_joint', 'PhysicsRevoluteJoint', 'torso_link', 'left_shoulder_pitch_link'), ('left_shoulder_roll_joint', 'PhysicsRevoluteJoint', 'left_shoulder_pitch_link', 'left_shoulder_roll_link'), ('left_shoulder_yaw_joint', 'PhysicsRevoluteJoint', 'left_shoulder_roll_link', 'left_shoulder_yaw_link'), ('left_wrist_pitch_joint', 'PhysicsRevoluteJoint', 'left_wrist_roll_link', 'left_wrist_pitch_link'), ('left_wrist_roll_joint', 'PhysicsRevoluteJoint', 'left_elbow_link', 'left_wrist_roll_link'), ('left_wrist_yaw_joint', 'PhysicsRevoluteJoint', 'left_wrist_pitch_link', 'left_wrist_yaw_link'), ('logo_joint', 'PhysicsFixedJoint', 'torso_link', 'logo_link'), ('mid360_joint', 'PhysicsFixedJoint', 'torso_link', 'mid360_link'), ('pelvis_contour_joint', 'PhysicsFixedJoint', 'pelvis', 'pelvis_contour_link'), ('right_ankle_pitch_joint', 'PhysicsRevoluteJoint', 'right_knee_link', 'right_ankle_pitch_link'), ('right_ankle_roll_joint', 'PhysicsRevoluteJoint', 'right_ankle_pitch_link', 'right_ankle_roll_link'), ('right_elbow_joint', 'PhysicsRevoluteJoint', 'right_shoulder_yaw_link', 'right_elbow_link'), ('right_hand_camera_base_joint', 'PhysicsFixedJoint', 'right_wrist_yaw_link', 'right_hand_camera_base_link'), ('right_hand_index_0_joint', 'PhysicsRevoluteJoint', 'right_hand_palm_link', 'right_hand_index_0_link'), ('right_hand_index_1_joint', 'PhysicsRevoluteJoint', 'right_hand_index_0_link', 'right_hand_index_1_link'), ('right_hand_middle_0_joint', 'PhysicsRevoluteJoint', 'right_hand_palm_link', 'right_hand_middle_0_link'), ('right_hand_middle_1_joint', 'PhysicsRevoluteJoint', 'right_hand_middle_0_link', 'right_hand_middle_1_link'), ('right_hand_palm_joint', 'PhysicsFixedJoint', 'right_wrist_yaw_link', 'right_hand_palm_link'), ('right_hand_thumb_0_joint', 'PhysicsRevoluteJoint', 'right_hand_palm_link', 'right_hand_thumb_0_link'), ('right_hand_thumb_1_joint', 'PhysicsRevoluteJoint', 'right_hand_thumb_0_link', 'right_hand_thumb_1_link'), ('right_hand_thumb_2_joint', 'PhysicsRevoluteJoint', 'right_hand_thumb_1_link', 'right_hand_thumb_2_link'), ('right_hip_pitch_joint', 'PhysicsRevoluteJoint', 'pelvis', 'right_hip_pitch_link'), ('right_hip_roll_joint', 'PhysicsRevoluteJoint', 'right_hip_pitch_link', 'right_hip_roll_link'), ('right_hip_yaw_joint', 'PhysicsRevoluteJoint', 'right_hip_roll_link', 'right_hip_yaw_link'), ('right_knee_joint', 'PhysicsRevoluteJoint', 'right_hip_yaw_link', 'right_knee_link'), ('right_shoulder_pitch_joint', 'PhysicsRevoluteJoint', 'torso_link', 'right_shoulder_pitch_link'), ('right_shoulder_roll_joint', 'PhysicsRevoluteJoint', 'right_shoulder_pitch_link', 'right_shoulder_roll_link'), ('right_shoulder_yaw_joint', 'PhysicsRevoluteJoint', 'right_shoulder_roll_link', 'right_shoulder_yaw_link'), ('right_wrist_pitch_joint', 'PhysicsRevoluteJoint', 'right_wrist_roll_link', 'right_wrist_pitch_link'), ('right_wrist_roll_joint', 'PhysicsRevoluteJoint', 'right_elbow_link', 'right_wrist_roll_link'), ('right_wrist_yaw_joint', 'PhysicsRevoluteJoint', 'right_wrist_pitch_link', 'right_wrist_yaw_link'), ('waist_pitch_joint', 'PhysicsRevoluteJoint', 'waist_roll_link', 'torso_link'), ('waist_roll_joint', 'PhysicsRevoluteJoint', 'waist_yaw_link', 'waist_roll_link'), ('waist_yaw_joint', 'PhysicsRevoluteJoint', 'pelvis', 'waist_yaw_link'), ('root_joint', 'PhysicsFixedJoint', None, 'pelvis')]

    @classmethod
    def topology(cls, env_id=0):
        root = "/World/envs/env_"+str(env_id)+"/Robot"
        return [{"name": n, "type": t, "path": root+("/" if n == "root_joint" else "/joints/")+n,
                 "body0": [root+"/"+a] if a else [], "body1": [root+"/"+b] if b else [],
                 "physics:jointEnabled": True, "physics:collisionEnabled": False}
                for n, t, a, b in cls.GRAPH]

    def test_exact_full_graph_yields_independently_pinned_contract_and_fresh_copy(self):
        actual = fixed_palm_adjacency_filter_contract(self.topology())
        self.assertEqual(actual, fixed_palm_adjacency_filter_contract())
        self.assertEqual(actual["source_graph_sha256"], "1bc9a5d33856280a489137634f3ddcbb84196ef231b6dfa902acbd271eac600c")
        self.assertEqual(actual["sha256"], "04dd990d50f4632012408b13747313a5504614e47a13d05d9540c31ef66fb770")
        encoded = json.dumps(actual["contract"], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self.assertEqual(hashlib.sha256(encoded.encode()).hexdigest(), actual["sha256"])
        self.assertEqual(len(actual["contract"]["excluded_body_pairs"]), 15)
        actual["contract"]["excluded_body_pairs"].append(["Cube", "Table"])
        self.assertEqual(len(fixed_palm_adjacency_filter_contract()["contract"]["excluded_body_pairs"]), 15)

    def test_incomplete_duplicate_wrong_graph_and_joint_flags_are_rejected(self):
        variants = [self.topology()[:-1], self.topology()+[self.topology()[0]]]
        for key, value in (("type", "PhysicsFixedJoint"), ("physics:jointEnabled", False),
                           ("physics:collisionEnabled", True), ("physics:collisionEnabled", 0)):
            rows = self.topology()
            rows[4][key] = value
            variants.append(rows)
        rows = self.topology()
        rows[-1] = rows[0]
        variants.append(rows)
        rows = self.topology()
        rows[4]["body0"] = ["/World/envs/env_0/Robot/right_hand_thumb_1_link"]
        variants.append(rows)
        for rows in variants:
            with self.subTest(rows=rows[-1]), self.assertRaises(RLEnvironmentError):
                fixed_palm_adjacency_filter_contract(rows)

    def test_cross_env_endpoint_and_entire_mixed_root_rows_cannot_hide_in_leaf_hash(self):
        rows = self.topology()
        rows[0]["body1"] = ["/World/envs/env_1/Robot/d435_link"]
        with self.assertRaisesRegex(RLEnvironmentError, "cross"):
            fixed_palm_adjacency_filter_contract(rows)
        rows = self.topology()
        rows[0] = self.topology(1)[0]
        with self.assertRaises(RLEnvironmentError):
            fixed_palm_adjacency_filter_contract(rows)

    def test_exact15_per_env_excludes_no_object_or_nonadjacent_thumb(self):
        relationships = fixed_palm_filter_relationships(8)
        result = validate_fixed_palm_filter_relationships(relationships, 8)
        self.assertEqual(result["excluded_pair_count_total"], 120)
        for path, targets in relationships.items():
            prefix = path.rsplit("/", 1)[0]+"/"
            self.assertTrue(all(p.startswith(prefix) for p in targets))
            self.assertFalse(any(name in path+" "+" ".join(targets)
                                 for name in ("/Cube", "/Table", "thumb_1", "thumb_2")))
        for n in (0, True, 513, 8.0):
            with self.assertRaises(RLEnvironmentError):
                fixed_palm_filter_relationships(n)

    def test_extra_pair_cross_env_missing_duplicate_and_wrong_readback_are_rejected(self):
        original = fixed_palm_filter_relationships(2)
        left = next(iter(original))
        variants = []
        for target in ("/World/envs/env_0/Cube", "/World/envs/env_0/Table",
                       "/World/envs/env_0/Robot/right_hand_thumb_1_link",
                       "/World/envs/env_1/Robot/right_hand_index_0_link"):
            changed = copy.deepcopy(original)
            changed[left].append(target)
            variants.append(changed)
        changed = copy.deepcopy(original)
        del changed[left]
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed[left].append(changed[left][0])
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["/World/envs/env_0/Cube"] = ["/World/envs/env_0/Table"]
        variants.append(changed)
        for value in variants:
            with self.assertRaises(RLEnvironmentError):
                validate_fixed_palm_filter_relationships(value, 2)

    @classmethod
    def fake_native_stage(cls, num_envs):
        class RigidBodyAPI:
            pass
        class ArticulationRootAPI:
            pass
        class Relationship:
            def __init__(self, prim, name):
                self.prim, self.name = prim, name
            def GetTargets(self):
                return self.prim.rels.get(self.name, [])
            def SetTargets(self, targets):
                self.prim.rels[self.name] = list(targets)
                if self.prim.corrupt_target is not None:
                    self.prim.rels[self.name].append(self.prim.corrupt_target)
                return True
        class Prim:
            def __init__(self, path, kind="", row=None):
                self.path, self.kind, self.row = path, kind, row
                self.apis, self.rels = set(), {}
                self.corrupt_target = None
                self.self_collisions = True
            def GetPath(self):
                return self.path
            def GetName(self):
                return self.path.rsplit("/", 1)[-1]
            def GetTypeName(self):
                return self.kind
            def IsA(self, schema):
                return schema is Joint and self.row is not None
            def HasAPI(self, schema):
                return schema in self.apis
            def GetAttribute(self, name):
                value = self.self_collisions if name == "physxArticulation:enabledSelfCollisions" else self.row[name]
                return SimpleNamespace(Get=lambda: value)
            def IsValid(self):
                return True
            def IsInstanceProxy(self):
                return False
        class Joint:
            def __init__(self, prim):
                self.prim = prim
            def GetBody0Rel(self):
                return SimpleNamespace(GetTargets=lambda: self.prim.row["body0"])
            def GetBody1Rel(self):
                return SimpleNamespace(GetTargets=lambda: self.prim.row["body1"])
            def GetJointEnabledAttr(self):
                return self.prim.GetAttribute("physics:jointEnabled")
            def GetCollisionEnabledAttr(self):
                return self.prim.GetAttribute("physics:collisionEnabled")
        class FilteredPairsAPI:
            def __init__(self, prim):
                self.prim = prim
            @classmethod
            def Apply(api, prim):
                prim.apis.add(api)
                return api(prim)
            def GetFilteredPairsRel(self):
                return Relationship(self.prim, "filteredPairs")
            CreateFilteredPairsRel = GetFilteredPairsRel
        prims = {}
        for env_id in range(num_envs):
            root = "/World/envs/env_"+str(env_id)+"/Robot"
            for row in cls.topology(env_id):
                prims[row["path"]] = Prim(row["path"], row["type"], row)
                for key in ("body0", "body1"):
                    for path in row[key]:
                        if path not in prims:
                            prims[path] = Prim(path)
                            prims[path].apis.add(RigidBodyAPI)
            prims[root+"/pelvis"].apis.add(ArticulationRootAPI)
            for body in ("Cube", "Table"):
                path = "/World/envs/env_"+str(env_id)+"/"+body
                prims[path] = Prim(path)
                prims[path].apis.add(RigidBodyAPI)
        stage = SimpleNamespace(Traverse=lambda: iter(prims.values()), GetPrimAtPath=lambda path: prims[path])
        physics = SimpleNamespace(RigidBodyAPI=RigidBodyAPI, ArticulationRootAPI=ArticulationRootAPI,
                                  Joint=Joint, FilteredPairsAPI=FilteredPairsAPI)
        return stage, physics, prims

    def test_native_authoring_validates_all8_graphs_and_exact_relationships(self):
        stage, physics, prims = self.fake_native_stage(8)
        result = author_fixed_palm_adjacency_filters(stage, physics, 8, FIXED_PALM_FILTER_ASSET_SHA256)
        self.assertEqual(result["source_graph_verified_env_count"], 8)
        self.assertEqual(result["excluded_pair_count_total"], 120)
        self.assertEqual(result["authoring_phase"], "after_clone_before_physics_initialization")
        self.assertEqual(len(result["global_self_collision_readback"]), 8)
        for path, prim in prims.items():
            if physics.FilteredPairsAPI in prim.apis:
                self.assertIn(path, fixed_palm_filter_relationships(8))
            if path.endswith(("/Cube", "/Table")):
                self.assertFalse(prim.HasAPI(physics.FilteredPairsAPI))
        prims[next(iter(result["relationships"]))].rels["filteredPairs"].append("/World/envs/env_0/Cube")
        with self.assertRaises(RLEnvironmentError):
            readback_fixed_palm_adjacency_filters(stage, physics, 8)

    def test_last_env_wrong_graph_rejects_before_authoring_any_relationship(self):
        stage, physics, prims = self.fake_native_stage(8)
        prims["/World/envs/env_7/Robot/joints/right_hand_palm_joint"].row["physics:jointEnabled"] = False
        with self.assertRaises(RLEnvironmentError):
            author_fixed_palm_adjacency_filters(stage, physics, 8, FIXED_PALM_FILTER_ASSET_SHA256)
        self.assertFalse(any(prim.HasAPI(physics.FilteredPairsAPI) for prim in prims.values()))

    def test_native_wrong_asset_preexisting_filter_and_readback_corruption_reject(self):
        for failure in ("asset", "preexisting", "readback", "global_self_collisions"):
            stage, physics, prims = self.fake_native_stage(2)
            asset_sha = FIXED_PALM_FILTER_ASSET_SHA256
            if failure == "asset":
                asset_sha = "0"*64
            elif failure == "preexisting":
                physics.FilteredPairsAPI.Apply(prims["/World/envs/env_0/Cube"])
            elif failure == "readback":
                prims[next(iter(fixed_palm_filter_relationships(2)))].corrupt_target = "/World/envs/env_1/Table"
            else:
                prims["/World/envs/env_1/Robot/pelvis"].self_collisions = False
            with self.subTest(failure=failure), self.assertRaises(RLEnvironmentError):
                author_fixed_palm_adjacency_filters(stage, physics, 2, asset_sha)


if __name__ == "__main__":
    unittest.main()
