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
from types import SimpleNamespace

from g1_vla.rl_environment import (
    ACTION_JOINT_NAMES,
    DEFAULT_SETTINGS,
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
    audit_asset_manifest,
    audit_fixed_root_joints,
    consecutive_window_samples,
    control_probe_profile,
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
        fake = SimpleNamespace(num_envs=1, _control_probe_enabled=False, _physics_pinch=None, _previous_actions=t.zeros(1, 14),
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
