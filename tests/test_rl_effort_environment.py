"""Run the actual native effort adapter methods with tensor-only fake views.

No Isaac application, geometry solver, training loop or alternative controller is
started. The contact-point and routing fixtures have independently known answers.
"""

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
TORCH = importlib.util.find_spec("torch") is not None


def actual_method(name, scope, *, before_assignment=None):
    tree = ast.parse((ROOT/"src/g1_vla/rl_environment.py").read_text())
    environment = next(v for v in ast.walk(tree) if isinstance(v, ast.ClassDef) and v.name == "Environment")
    method = copy.deepcopy(next(v for v in environment.body if isinstance(v, ast.FunctionDef) and v.name == name))
    if before_assignment:
        selected = []
        for statement in method.body:
            if isinstance(statement, ast.Assign) and any(
                    (isinstance(v, ast.Name) and v.id == before_assignment) or
                    (isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name) and
                     v.value.id == "self" and "self."+v.attr == before_assignment)
                    for v in statement.targets):
                break
            selected.append(statement)
        method.body = selected
    namespace = {"__name__": "g1_vla.rl_environment", "__package__": "g1_vla", **scope}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<actual-native-effort-adapter>", "exec"), namespace)
    return namespace[name]


@unittest.skipUnless(TORCH, "real torch is required for native effort adapter math")
class NativeEffortAdapterTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.t = torch
        self.scope = {"torch": torch, "tensor": lambda v: v, "RLEnvironmentError": RuntimeError,
                      "math": math, "json": json,
                      "settings": {"contact_offset_m": .001, "max_abs_normal_z": .6}}

    def guide_fixture(self, *, acquisition=None, axis=None):
        t = self.t
        tree = ast.parse((ROOT/"src/g1_vla/rl_diagnostics.py").read_text())
        nodes = [node for node in tree.body if
                 (isinstance(node, ast.ClassDef) and node.name == "EffortGraspGuide") or
                 (isinstance(node, ast.FunctionDef) and node.name in ("_bounded_vector", "_orientation_error")) or
                 (isinstance(node, ast.Assign) and any(isinstance(v, ast.Name) and v.id == "POSE_SERVO"
                                                      for v in node.targets))]
        scope = {}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual-bootstrap-effort-guide>", "exec"), scope)
        candidate = json.loads((ROOT/"resources/rl_geometry_candidates/opposed_face_effort_pinch_v1.json").read_text())
        if axis is not None:
            candidate["closing_axis_hand7"] = axis
        seed = candidate["source_hand_rad"]
        q = t.zeros(2, 14)
        q[:, 7:] = t.tensor(seed)
        state = {"q_action_rad": q, "target_action_rad": q.clone(),
            "joint_limits_action_rad": t.tensor([[[-2., 2.]]*14]*2),
            "episode_id": t.zeros(2, dtype=t.long), "palm_position_world_m": t.zeros(2, 3),
            "palm_quaternion_xyzw": t.tensor([[0., 0., 0., 1.]]*2),
            "group_normal_force_n": t.zeros(2, 3), "unsafe_latched": t.zeros(2, dtype=t.bool),
            "preload_contact_ready": t.zeros(2, dtype=t.bool),
            "palm_point_jacobian_w": t.eye(6, 7)[None].repeat(2, 1, 1),
            "per_step_action_rad": t.full((2, 14), .0198), "metrics": {}}
        settings = {"guided_preload_controls": 3, "guided_probe_speed_m_s": .001,
                    "guided_probe_height_m": .0025}
        settings["contact_acquisition"] = acquisition
        guide = scope["EffortGraspGuide"](t, settings, seed, state, candidate)
        return guide, state

    def test_actual_guide_first_request_accepts_empty_metrics_without_inventing_contact(self):
        guide, state = self.guide_fixture()
        requested, e = guide.actions(self.t.zeros(2, 14), state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue(self.t.isfinite(requested).all())
        self.assertGreater(float((e["guided_target_hand_rad"]-state["q_action_rad"][:, 7:]).abs().max()), 0)
        self.assertFalse(e["stage_contact_geometry_is_control_signal_not_acceptance"].any())
        self.assertFalse(e["closure_path_complete"].any())
        self.assertFalse(e["probe_advance_permitted"].any())
        self.assertTrue((e["preload_control_boundary_count"] == 0).all())
        # Missing telemetry does not block a measured free-space waypoint.
        state["q_action_rad"][:, 7:] = guide.goals[0]
        guide.elapsed[:] = guide.durations[0]
        _, e = guide.actions(self.t.zeros(2, 14), state, "guided_hand_pose_hold", .066, 1.5)
        self.assertTrue(e["stage_advance_after_this_request"].all())
        self.assertTrue((e["stage0_precontact_raw_force_controls"] == 0).all())

    def test_actual_guide_empty_metrics_cannot_qualify_final_or_stale_preload(self):
        guide, state = self.guide_fixture()
        guide.stage[:] = len(guide.names)-1
        guide.elapsed[:] = guide.durations[-1]
        guide.start[:] = guide.goals[-2]
        guide.ready_count[:] = 4
        state["q_action_rad"][:, 7:] = guide.goals[-1]
        state["preload_contact_ready"][:] = True  # Even an inconsistent stale flag cannot promote missing feedback.
        for _ in range(4):
            _, e = guide.actions(self.t.zeros(2, 14), state, "guided_hand_pose_hold", .066, 1.5)
            self.assertFalse(e["closure_path_complete"].any())
            self.assertFalse(e["preload_contact_ready"].any())
            self.assertFalse(e["probe_advance_permitted"].any())
            self.assertTrue((e["probe_offset_m"] == 0).all())

    def test_actual_guide_partial_contact_metrics_default_to_unavailable(self):
        complete = {"contact_safe": self.t.ones(2, dtype=self.t.bool),
                    "group_physical_loaded_contact_count": self.t.ones(2, 3, dtype=self.t.long),
                    "opposition_dot": self.t.full((2,), -1.), "maximum_abs_normal_z": self.t.zeros(2),
                    "maximum_penetration_m": self.t.zeros(2)}
        for missing in complete:
            with self.subTest(missing=missing):
                guide, state = self.guide_fixture()
                state["metrics"] = {key: value for key, value in complete.items() if key != missing}
                state["group_normal_force_n"][:] = .4
                guide.elapsed[:] = guide.durations[0]
                guide.precontact_count[:] = 1
                _, e = guide.actions(self.t.zeros(2, 14), state, "guided_hand_pose_hold", .066, 1.5)
                self.assertFalse(e["stage_advance_after_this_request"].any())
                self.assertFalse(e["stage_contact_geometry_is_control_signal_not_acceptance"].any())
                self.assertTrue((e["stage0_precontact_raw_force_controls"] == 0).all())

    def test_touch_search_constructor_rejects_invalid_closing_axis(self):
        for axis in ([1.]*6, [float("nan")]*7, [2.]*7, [.5]*7, [0.]*7,
                     [0.,0.,0.,1.,-1.,1.,1.]):
            with self.subTest(axis=axis), self.assertRaises(ValueError):
                self.guide_fixture(acquisition={"version": "test"}, axis=axis)

    def touch_search_fixture(self):
        guide, state = self.guide_fixture()
        guide.acquisition = {"reference_closing_rate_rad_s": .02,
            "maximum_extra_closure_rad": .02, "raw_force_stop_n": .02}
        guide.stage[:] = len(guide.names)-1
        guide.elapsed[:] = guide.durations[-1]
        guide.start[:] = guide.goals[-2]
        state["q_action_rad"][:, 7:] = guide.goals[-1]
        t = self.t
        state["contact_metrics_current_episode"] = t.ones(2, dtype=t.bool)
        state["metrics"] = {"contact_safe": t.zeros(2, dtype=t.bool),
            "group_physical_loaded_contact_count": t.zeros(2, 3, dtype=t.long),
            "opposition_dot": t.ones(2), "maximum_abs_normal_z": t.zeros(2),
            "maximum_penetration_m": t.zeros(2), "finite": t.ones(2, dtype=t.bool),
            "joint_safe": t.ones(2, dtype=t.bool),
            "table_pair_measurement_supported": t.ones(2, dtype=t.bool)}
        return guide, state

    def touch_step(self, guide, state):
        return guide.actions(self.t.zeros(2, 14), state, "guided_hand_pose_hold", .066, 1.5)[1]

    def test_touch_search_only_missing_unloaded_group_moves_bounded_reference(self):
        g, s = self.touch_search_fixture()
        s["metrics"]["group_physical_loaded_contact_count"][:, 1] = 1
        s["group_normal_force_n"][:, 2] = .02
        e = self.touch_step(g, s)
        expected = self.t.tensor([[.00132, 0, 0]]*2)
        self.assertTrue(self.t.allclose(e["contact_search_offset_by_group_rad"], expected))
        self.assertTrue(self.t.equal(e["contact_search_active_by_group"], expected > 0))
        self.assertTrue(self.t.allclose(e["guided_target_hand_rad"], g.goals[-1]+expected[:, [0,0,0,1,1,2,2]]*g.axis))
        self.assertFalse(e["closure_path_complete"].any())
        self.assertFalse(e["probe_advance_permitted"].any())
        self.assertTrue((e["probe_offset_m"] == 0).all())

    def test_touch_search_stops_accumulation_on_physical_touch_or_raw_force(self):
        g, s = self.touch_search_fixture()
        self.touch_step(g, s)
        previous = g.search_offset.clone()
        s["metrics"]["group_physical_loaded_contact_count"][:, 0] = 1
        s["group_normal_force_n"][:, 1:] = .02
        e = self.touch_step(g, s)
        self.assertTrue(self.t.equal(g.search_offset, previous))
        self.assertFalse(e["contact_search_active_by_group"].any())

    def test_touch_search_saturates_inside_original_actual_domain_and_mechanical_limits(self):
        g, s = self.touch_search_fixture()
        # One mechanical upper limit is closer than the search reference budget.
        g.mechanical_high[:, 3] = g.high[:, 3]+.005
        for _ in range(30):
            e = self.touch_step(g, s)
        self.assertTrue(self.t.equal(g.search_offset, self.t.full((2, 3), .02)))
        self.assertTrue((e["geometry_lower_hand_rad"] >= g.low-.025).all())
        self.assertTrue((e["geometry_upper_hand_rad"] <= g.high+.025).all())
        self.assertTrue((e["guided_target_hand_rad"] <= g.mechanical_high).all())
        self.assertTrue((e["guided_target_hand_rad"] >= g.mechanical_low).all())
        self.assertTrue(self.t.allclose(e["geometry_upper_hand_rad"][:, 3], g.high[:, 3]+.005))

    def test_touch_search_no_early_boundary_expansion_and_episode_reset_is_individual(self):
        g, s = self.touch_search_fixture()
        g.stage[:] = 0
        e = self.touch_step(g, s)
        self.assertTrue(self.t.equal(e["geometry_lower_hand_rad"], g.low))
        self.assertTrue(self.t.equal(e["geometry_upper_hand_rad"], g.high))
        self.assertTrue((g.search_offset == 0).all())
        g.stage[:] = len(g.names)-1
        g.elapsed[:] = g.durations[-1]
        self.touch_step(g, s)
        previous = g.search_offset[1].clone()
        s["episode_id"][0] += 1
        e = self.touch_step(g, s)
        self.assertTrue((g.search_offset[0] == 0).all())
        self.assertTrue(self.t.all(g.search_offset[1] > previous))
        self.assertTrue(self.t.equal(e["geometry_lower_hand_rad"][0], g.low[0]))

    def test_touch_search_requires_safe_current_complete_measurements_and_final_duration(self):
        changes = ["missing_metrics", "stale", "invalid", "joint", "table", "tracking", "unsafe",
                   "overforce", "nan_force", "negative_force", "nan_count", "negative_count",
                   "nan_depth", "negative_depth", "depth_limit", "duration"]
        for change in changes:
            with self.subTest(change=change):
                g, s = self.touch_search_fixture()
                m = s["metrics"]
                if change == "missing_metrics": m.pop("maximum_abs_normal_z")
                elif change == "stale": s["contact_metrics_current_episode"][:] = False
                elif change == "invalid": m["finite"][:] = False
                elif change == "joint": m["joint_safe"][:] = False
                elif change == "table": m["table_pair_measurement_supported"][:] = False
                elif change == "tracking": s["palm_position_world_m"][:, 0] = .004
                elif change == "unsafe": s["unsafe_latched"][:] = True
                elif change == "overforce": s["group_normal_force_n"][:, 2] = 1.5001
                elif change == "nan_force": s["group_normal_force_n"][:, 2] = float("nan")
                elif change == "negative_force": s["group_normal_force_n"][:, 2] = -.01
                elif change == "nan_count": m["group_physical_loaded_contact_count"] = self.t.full((2,3), float("nan"))
                elif change == "negative_count": m["group_physical_loaded_contact_count"][:, 1] = -1
                elif change == "nan_depth": m["maximum_penetration_m"][:] = float("nan")
                elif change == "negative_depth": m["maximum_penetration_m"][:] = -.001
                elif change == "depth_limit": m["maximum_penetration_m"][:] = .00301
                elif change == "duration": g.elapsed[:] = 0
                e = self.touch_step(g, s)
                self.assertFalse(e["contact_search_active_by_group"].any())
                self.assertTrue((g.search_offset == 0).all())
                self.assertFalse(e["probe_advance_permitted"].any())

    def test_touch_search_none_retains_original_reference_domain(self):
        g, s = self.touch_search_fixture()
        g.acquisition = None
        e = self.touch_step(g, s)
        self.assertTrue(self.t.equal(e["guided_target_hand_rad"], g.goals[-1][None].expand(2, -1)))
        self.assertTrue(self.t.equal(e["geometry_lower_hand_rad"], g.low))
        self.assertTrue(self.t.equal(e["geometry_upper_hand_rad"], g.high))
        self.assertTrue((g.search_offset == 0).all())

    def geometry_fixture(self):
        t = self.t
        jac = t.zeros(2, 8, 6, 43)
        # Revolute Z axes. The point's 1/2 cm lever, not the link-origin
        # translation (which is zero), produces the inward linear Jacobian.
        jac[:, 2, 5, 7] = 1.
        jac[:, 4, 5, 10] = -1.
        jac[:, 6, 5, 12] = -1.
        positions = t.zeros(2, 9, 3)
        positions[:, 3] = t.tensor([.025, .01, 0.])
        positions[:, 5] = t.tensor([-.025, -.01, 0.])
        positions[:, 7] = t.tensor([-.025, -.02, 0.])
        env = SimpleNamespace(num_envs=2, _action_ids=list(range(14)),
            _effort_hand_body_ids=list(range(1, 8)), _episode_ids=t.tensor([2, 3]),
            robot=SimpleNamespace(num_bodies=9, data=SimpleNamespace(body_link_jacobian_w=jac, body_pos_w=positions)),
            cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=t.zeros(2, 3))),
            sim=SimpleNamespace(get_physics_step_count=lambda: 9))
        hand = {"point": t.zeros(2, 8, 32, 3), "point_normal": t.zeros(2, 8, 32, 3),
                "point_valid": t.zeros(2, 8, 32, dtype=t.bool), "point_force": t.zeros(2, 8, 32),
                "separation": t.full((2, 8, 32), .0005)}
        hand["point"][:, 2, 0, 0] = .025
        hand["point"][:, [4, 6], 0, 0] = -.025
        hand["point_normal"][:, 2, 0, 0] = 1.
        hand["point_normal"][:, [4, 6], 0, 0] = -1.
        hand["point_valid"][:, [2, 4, 6], 0] = True
        # A palm point must not be broadcast against the seven finger Jacobians.
        hand["point_valid"][:, 7, 0] = True
        hand["point"][:, 7, 0] = 100.
        hand["point_normal"][:, 7, 0] = t.tensor([1., 0., 0.])
        return env, hand

    def test_current_native_point_shift_has_correct_sign_and_ignores_eighth_palm(self):
        env, hand = self.geometry_fixture()
        actual_method("_effort_contact_geometry", self.scope)(env, hand)
        result = env._effort_geometry
        expected = self.t.zeros(2, 3, 7)
        expected[:, 0, 0] = -.01
        expected[:, 1, 3] = .01
        expected[:, 2, 5] = .02
        self.assertTrue(self.t.allclose(result["normal_jacobian"], expected, atol=1e-8, rtol=0))
        self.assertTrue(result["current_normal_available"].all())
        self.assertEqual(result["native_step"], 9)
        self.assertTrue(self.t.equal(result["episode_id"], self.t.tensor([2, 3])))
        # A positive closing velocity gives positive inward speed in all groups.
        velocity = self.t.tensor([[-1., 0., 0., 1., 0., 1., 0.]]*2)
        inward_speed = self.t.einsum("ngi,ni->ng", result["normal_jacobian"], velocity)
        self.assertTrue((inward_speed > 0).all())

    def test_current_zero_force_speculative_direction_is_not_erased_or_reused_when_gone(self):
        env, hand = self.geometry_fixture()
        method = actual_method("_effort_contact_geometry", self.scope)
        method(env, hand)
        self.assertTrue(env._effort_geometry["current_normal_available"].all())
        hand["point_valid"][:, 2] = False
        method(env, hand)
        self.assertFalse(env._effort_geometry["current_normal_available"][:, 0].any())
        self.assertTrue(self.t.equal(env._effort_geometry["normal_jacobian"][:, 0], self.t.zeros(2, 7)))
        self.assertTrue(env._effort_geometry["current_normal_available"][:, 1:].all())

    def test_reversing_canonical_point_normals_does_not_reverse_inward_actuation(self):
        env, hand = self.geometry_fixture()
        method = actual_method("_effort_contact_geometry", self.scope)
        method(env, hand)
        original = env._effort_geometry["normal_jacobian"].clone()
        hand["point_normal"] *= -1
        method(env, hand)
        self.assertTrue(self.t.equal(env._effort_geometry["normal_jacobian"], original))

    def test_vertical_or_far_point_is_rejected_for_control_without_rewriting_raw_force(self):
        env, hand = self.geometry_fixture()
        hand["point_normal"][:, 2, 0] = self.t.tensor([0., 0., 1.])
        hand["separation"][:, 4, 0] = .0021
        hand["point_force"][:, 4, 0] = 54.
        actual_method("_effort_contact_geometry", self.scope)(env, hand)
        self.assertFalse(env._effort_geometry["current_normal_available"][:, :2].any())
        self.assertTrue(env._effort_geometry["current_normal_available"][:, 2].all())
        self.assertTrue((hand["point_force"][:, 4, 0] == 54.).all())

    def test_physics_routing_samples_before_effort_and_never_writes_hand_position_target(self):
        t, events = self.t, []
        position_calls, effort_calls = [], []
        def position(**kw):
            events.append("position36")
            position_calls.append(kw)
        def effort(**kw):
            events.append("effort43")
            effort_calls.append(kw)
        def full_write():
            effort(target=t.zeros(2, 43))
        env = SimpleNamespace(_pending_physics_sample=True, _physics_pinch=None, _effort_grasp=object(),
            _sample_physics=lambda: events.append("sample-k"),
            _update_effort_grasp=lambda: events.append("effort-from-k"),
            _action_ids=list(range(14)), _targets=t.arange(86, dtype=t.float32).reshape(2, 43),
            _last_written_hand_effort=t.full((2, 7), .01),
            _write_effort_targets=full_write,
            robot=SimpleNamespace(set_joint_position_target_index=position, set_joint_effort_target_index=effort))
        actual_method("_apply_action", self.scope)(env)
        self.assertEqual(events, ["sample-k", "effort-from-k", "position36", "effort43"])
        self.assertEqual(len(position_calls), 1)
        self.assertEqual(len(position_calls[0]["joint_ids"]), 36)
        self.assertFalse(set(position_calls[0]["joint_ids"]) & set(range(7, 14)))
        self.assertNotIn("joint_ids", effort_calls[0])
        self.assertEqual(tuple(effort_calls[0]["target"].shape), (2, 43))

    def full_effort_fixture(self, *, native_getter=True):
        from g1_vla.rl_effort_protocol import RIGHT_ARM_JOINT_NAMES, RIGHT_HAND_JOINT_NAMES, effort_grasp_contract
        t, calls, rows = self.t, [], []
        contract = effort_grasp_contract()
        declaration = contract["actuation"]["posture_gravity_feedforward"]
        # A deliberately non-contiguous native order proves that names, not
        # assumed columns in the 43-DOF tensors, determine effort routing.
        names = ["other_joint_{}".format(i) for i in range(26)] + list(RIGHT_HAND_JOINT_NAMES) + declaration["joint_order"]
        names = names[::2]+names[1::2]
        arm_ids = [names.index(n) for n in RIGHT_ARM_JOINT_NAMES]
        hand_ids = [names.index(n) for n in RIGHT_HAND_JOINT_NAMES]
        posture_ids = [names.index(n) for n in declaration["joint_order"]]
        caps = t.tensor(declaration["feedforward_caps_nm"])
        gravity = t.arange(86, dtype=t.float32).reshape(2, 43)-30
        # Independent native hard04 readback: waist pitch is50, not yaw's88.
        original_caps = [25., 25., 25., 25., 25., 5., 5., 88., 50., 50.]
        specs = {n: {"effort_limit_nm": original_caps[i]} for i, n in enumerate(declaration["joint_order"])}
        view = SimpleNamespace()
        previous = t.full((2, 43), .123)
        if native_getter:
            view.get_dof_actuation_forces = lambda: previous
        env = SimpleNamespace(num_envs=2, device="cpu", _action_ids=arm_ids+hand_ids,
            _last_written_hand_effort=t.full((2, 7), .01),
            _last_written_full_effort=t.full((2, 43), 999.),
            _unsafe=t.tensor([False, True]), _episode_ids=t.tensor([4, 5]),
            _last_contact_episode_ids=t.tensor([4, -1]),
            _posture_ff_update_count=0, _posture_ff_last_sample=None, _control_command_id=3,
            _write_control_probe=rows.append,
            sim=SimpleNamespace(get_physics_step_count=lambda: 72),
            robot=SimpleNamespace(joint_names=names, root_view=view,
                data=SimpleNamespace(gravity_compensation_forces=gravity),
                set_joint_effort_target_index=lambda **kw: calls.append(kw)))
        scope = dict(self.scope, RIGHT_ARM=tuple(RIGHT_ARM_JOINT_NAMES), RIGHT_HAND=tuple(RIGHT_HAND_JOINT_NAMES),
                     effort_contract=contract, actuator_specs=specs)
        return env, calls, rows, gravity, posture_ids, hand_ids, contract, actual_method("_write_effort_targets", scope)

    def test_full43_fresh_posture_ff_clips_without_stale_unused_efforts_and_masks_unsafe(self):
        env, calls, rows, gravity, posture_ids, hand_ids, contract, method = self.full_effort_fixture()
        t = self.t
        method(env)
        self.assertEqual(len(calls), 1)
        self.assertNotIn("joint_ids", calls[0])
        applied = calls[0]["target"]
        caps = t.tensor(contract["actuation"]["posture_gravity_feedforward"]["feedforward_caps_nm"])
        self.assertTrue(t.equal(applied[0, posture_ids], gravity[0, posture_ids].clamp(-caps, caps)))
        self.assertTrue(t.equal(applied[0, hand_ids], t.full((7,), .01)))
        unused = [i for i in range(43) if i not in posture_ids+hand_ids]
        self.assertEqual(len(unused), 26)
        self.assertTrue((applied[:, unused] == 0).all())
        self.assertTrue((applied[1] == 0).all(), "unsafe zeros both posture FF and hand effort")
        self.assertTrue(t.equal(env._last_written_hand_effort, applied[:, hand_ids]))
        self.assertEqual(rows[0]["native_physics_step_count"], 72)
        self.assertTrue(t.equal(rows[0]["native_previous_completed_step_user_actuation_nm"], t.full((2, 43), .123)))
        self.assertTrue(rows[0]["combined_posture_pd_and_ff_total_not_certified_bounded"])
        self.assertEqual(env._posture_ff_update_count, 1)
        gravity[:] = -.002
        method(env)
        self.assertTrue(t.equal(calls[-1]["target"][0, posture_ids], t.full((10,), -.002)), "fresh g is reread every call")
        self.assertEqual(env._posture_ff_update_count, 2)

    def test_full43_optional_native_user_actuation_readback_is_explicitly_unavailable(self):
        env, _, rows, _, _, _, _, method = self.full_effort_fixture(native_getter=False)
        method(env)
        self.assertIsNone(rows[0]["native_previous_completed_step_user_actuation_nm"])
        self.assertEqual(rows[0]["native_previous_user_actuation_unavailable_reason"], "native_get_dof_actuation_forces_unavailable")

    def test_full43_nonfinite_gravity_or_hand_effort_clears_all_user_effort_before_abort(self):
        for source in ("gravity", "hand"):
            env, calls, _, gravity, _, _, _, method = self.full_effort_fixture()
            if source == "gravity":
                gravity[0, 3] = float("nan")
            else:
                env._last_written_hand_effort[1, 1] = float("inf")
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "finite full43/hand7"):
                method(env)
            self.assertEqual(len(calls), 1)
            self.assertTrue((calls[0]["target"] == 0).all())

    def test_full43_declared_posture_caps_cannot_exceed_original_and_hand_cannot_bypass_total_cap(self):
        for source in ("posture", "waist_pitch", "hand"):
            env, calls, _, _, _, _, contract, method = self.full_effort_fixture()
            if source == "posture":
                contract["actuation"]["posture_gravity_feedforward"]["feedforward_caps_nm"][0] = 26.
            elif source == "waist_pitch":
                contract["actuation"]["posture_gravity_feedforward"]["feedforward_caps_nm"][-1] = 88.
            else:
                env._last_written_hand_effort[0, 0] = .061
            with self.subTest(source=source), self.assertRaisesRegex(RuntimeError, "caps"):
                method(env)
            self.assertTrue((calls[0]["target"] == 0).all())

    def test_full43_reset_clears_only_selected_environments_without_partial_joint_residue(self):
        env, calls, _, _, _, _, _, _ = self.full_effort_fixture()
        env._last_written_hand_effort[:] = .02
        actual_method("_clear_effort_targets", self.scope)(env, self.t.tensor([1]))
        self.assertEqual(len(calls), 1)
        self.assertNotIn("joint_ids", calls[0])
        self.assertEqual(tuple(calls[0]["target"].shape), (1, 43))
        self.assertTrue((calls[0]["target"] == 0).all())
        self.assertTrue((env._last_written_full_effort[1] == 0).all())
        self.assertTrue((env._last_written_full_effort[0] == 999.).all())
        self.assertTrue((env._last_written_hand_effort[1] == 0).all())

    def generation_fixture(self, *, already_matched=False):
        from g1_vla.rl_effort_protocol import effort_grasp_contract
        t, calls = self.t, []
        contract = effort_grasp_contract()
        contract["contact_generation"] = {
            "version": "g1_zero_margin_physical_contact_v1", "contact_offset_m": 0.,
            "rest_offset_m": -1e-6, "friction_offset_threshold_m": 0.,
            "normal_manifold_filtering": False, "original_physical_acceptance_unchanged": True}
        class View:
            def __init__(self, name, count):
                self.name, self.count, self.max_shapes = name, 2, count
                self.prim_paths = ["/World/envs/env_{}/{}".format(i, name)+( "/root_joint" if name == "Robot" else "") for i in range(2)]
                self.is_homogeneous = True
                self.contact = t.full((2, count), 0. if already_matched else .001)
                self.rest = t.full((2, count), -1e-6 if already_matched else 0.)
                self.ignore_contact_write = False
            def get_contact_offsets(self): return self.contact
            def get_rest_offsets(self): return self.rest
            def set_rest_offsets(self, data, indices):
                self.assert_cpu(data, indices)
                calls.append((self.name, "rest"))
                self.rest = data.clone()
            def set_contact_offsets(self, data, indices):
                self.assert_cpu(data, indices)
                # Reproduce the actual SDK's strict setter-order condition.
                if not bool((data > self.rest).all()):
                    raise RuntimeError("contact must be strictly greater than rest")
                calls.append((self.name, "contact"))
                if not self.ignore_contact_write: self.contact = data.clone()
            @staticmethod
            def assert_cpu(data, indices):
                if data.device.type != "cpu" or indices.device.type != "cpu":
                    raise RuntimeError("native model parameter setters must use CPU")
                if indices.tolist() != [0, 1]:
                    raise RuntimeError("model writes must name every environment once")
        class Attr:
            def __init__(self): self.value = 0.
            def Get(self): return self.value
        attr = Attr()
        api = SimpleNamespace(GetFrictionOffsetThresholdAttr=lambda: attr)
        scene = SimpleNamespace(IsValid=lambda: True, HasAPI=lambda cls: True)
        stage = SimpleNamespace(GetPrimAtPath=lambda path: scene)
        views = {name: View(name, count) for name, count in (("Robot", 3), ("Cube", 1), ("Table", 1))}
        env = SimpleNamespace(num_envs=2, sim=SimpleNamespace(get_physics_step_count=lambda: 0,
                cfg=SimpleNamespace(physics_prim_path="/physicsScene")),
            robot=SimpleNamespace(root_view=views["Robot"]), cube=SimpleNamespace(root_view=views["Cube"]),
            table=SimpleNamespace(root_view=views["Table"]),
            _effort_contact_offset_schema={name: [{"collider": "source_shape_"+str(i)} for i in range(view.max_shapes)]
                                           for name, view in views.items()})
        warp = SimpleNamespace(from_torch=lambda v: v,
            array=lambda values, **kw: t.tensor(values, dtype=t.int64, device=kw["device"]), uint32=object())
        scope = dict(self.scope, effort_contract=contract, wp=warp,
                     PhysxSchema=SimpleNamespace(PhysxSceneAPI=lambda prim: api),
                     get_current_stage=lambda: stage, _probe_payload=lambda torch, value: value)
        return env, calls, views, attr, actual_method("_verify_effort_contact_generation", scope)

    def test_zero_margin_native_every_shape_rest_before_contact_and_actual_readback(self):
        env, calls, views, _, method = self.generation_fixture()
        method(env)
        self.assertEqual(calls, [(name, kind) for name in ("Robot", "Cube", "Table") for kind in ("rest", "contact")])
        audit = env._effort_contact_generation_readback
        self.assertTrue(audit["verified"])
        self.assertFalse(audit["normal_manifold_filtering"])
        self.assertTrue(audit["original_physical_acceptance_unchanged"])
        for name, view in views.items():
            self.assertTrue((view.contact == 0).all())
            self.assertTrue(self.t.equal(view.rest, self.t.full_like(view.rest, -1e-6)))
            row = audit["native_shape_offsets"][name]
            self.assertEqual(row["native_shape_count_per_env"], view.max_shapes)
            self.assertTrue(row["prephysics_native_setters_applied"])
        self.assertFalse(audit["scene_friction_offset_threshold"]["compiled_scene_getter_available"])

    def test_zero_margin_authors_all_robot_shapes_including_non_hand_instance_children(self):
        env0, _, _, _, method = self.generation_fixture()
        contract = method.__globals__["effort_contract"]
        prims, ranges, uninstanced, writes = {}, {}, [], []
        invalid = SimpleNamespace(IsValid=lambda: False)
        def prim(path, *, collision=False, instance=False, parent=invalid):
            p = SimpleNamespace(path=path, instance=instance, attributes={}, IsValid=lambda: True,
                GetPath=lambda: path, HasAPI=lambda cls: collision, GetParent=lambda: parent)
            p.IsInstance = lambda: p.instance
            def uninstance(value):
                p.instance = value
                uninstanced.append(path)
            p.SetInstanceable = uninstance
            prims[path] = p
            return p
        for name in ("Robot", "Cube", "Table"):
            root = prim("/World/envs/env_0/"+name)
            if name == "Robot":
                left = prim(root.path+"/left_foot_collision_instance", instance=True, parent=root)
                ranges[root.path] = [root, left,
                    prim(left.path+"/mesh", collision=True, parent=left),
                    prim(root.path+"/right_hand/mesh", collision=True, parent=root)]
            else:
                ranges[root.path] = [root, prim(root.path+"/geometry/mesh", collision=True, parent=root)]
        class API:
            def __init__(self, p): self.p = p
            def CreateRestOffsetAttr(self, v): self.p.attributes["rest"] = v; writes.append((self.p.path, "rest"))
            def CreateContactOffsetAttr(self, v): self.p.attributes["contact"] = v; writes.append((self.p.path, "contact"))
            def GetRestOffsetAttr(self): return SimpleNamespace(Get=lambda: self.p.attributes["rest"])
            def GetContactOffsetAttr(self): return SimpleNamespace(Get=lambda: self.p.attributes["contact"])
        stage = SimpleNamespace(GetPrimAtPath=lambda path: prims[path])
        env = SimpleNamespace(_instance_overrides=[])
        scope = dict(self.scope, effort_contract=contract,
            Usd=SimpleNamespace(PrimRange=lambda root, flag: ranges[root.path], TraverseInstanceProxies=lambda: True),
            UsdPhysics=SimpleNamespace(CollisionAPI=object()),
            PhysxSchema=SimpleNamespace(PhysxCollisionAPI=SimpleNamespace(Apply=API)))
        actual_method("_author_effort_contact_offsets", scope)(env, stage)
        self.assertEqual({k: len(v) for k, v in env._effort_contact_offset_schema.items()}, {"Robot": 2, "Cube": 1, "Table": 1})
        self.assertEqual(uninstanced, ["/World/envs/env_0/Robot/left_foot_collision_instance"])
        self.assertEqual([kind for _, kind in writes], ["rest", "contact"]*4)
        self.assertTrue(all(p.attributes == {"rest": -1e-6, "contact": 0.} for p in prims.values() if p.attributes))

    def test_zero_margin_already_imported_values_need_no_native_setters(self):
        env, calls, _, _, method = self.generation_fixture(already_matched=True)
        method(env)
        self.assertEqual(calls, [])
        self.assertTrue(env._effort_contact_generation_readback["verified"])

    def test_zero_margin_native_cooked_shape_count_can_differ_but_every_entry_is_set_and_checked(self):
        env, _, views, _, method = self.generation_fixture()
        v = views["Robot"]
        v.max_shapes = 5  # Three authored collider APIs need not cook into three native shapes.
        v.contact = self.t.full((2, 5), .001)
        v.rest = self.t.zeros(2, 5)
        method(env)
        row = env._effort_contact_generation_readback["native_shape_offsets"]["Robot"]
        self.assertEqual(row["native_shape_count_per_env"], 5)
        self.assertEqual(row["authored_collision_api_prim_count_per_env"], 3)
        self.assertEqual(row["native_minus_authored_count"], 2)
        self.assertEqual(row["all_native_shape_entries_checked"], 10)
        self.assertFalse(row["source_to_native_one_to_one_shape_mapping_claimed"])
        self.assertTrue((v.contact == 0).all())
        self.assertTrue((v.rest == self.t.full_like(v.rest, -1e-6)).all())

    def test_zero_margin_rejects_cross_env_binding_and_heterogeneous_native_articulations(self):
        for mode in ("cross_env", "heterogeneous"):
            env, _, views, _, method = self.generation_fixture()
            if mode == "cross_env": views["Robot"].prim_paths[1] = "/World/envs/env_0/Robot/root_joint"
            else: views["Robot"].is_homogeneous = False
            with self.subTest(mode=mode), self.assertRaisesRegex(RuntimeError, "authored_collider_api_count"):
                method(env)

    def test_zero_margin_native_shape_count_nonfinite_or_ignored_write_fail_closed(self):
        for failure in ("shape", "nonfinite", "ignored"):
            env, _, views, _, method = self.generation_fixture()
            if failure == "shape": views["Robot"].max_shapes = 4
            elif failure == "nonfinite": views["Robot"].rest[0, 0] = float("nan")
            else: views["Robot"].ignore_contact_write = True
            with self.subTest(failure=failure), self.assertRaisesRegex(RuntimeError, "offset"):
                method(env)
            self.assertFalse(hasattr(env, "_effort_contact_generation_readback"))

    def test_zero_margin_scene_usd_mismatch_is_not_excused_by_native_shape_match(self):
        env, _, _, attr, method = self.generation_fixture(already_matched=True)
        attr.value = .04
        with self.assertRaisesRegex(RuntimeError, "USD friction"):
            method(env)

    def test_zero_margin_cannot_change_model_after_first_task_physics_step(self):
        env, calls, _, _, method = self.generation_fixture()
        env.sim.get_physics_step_count = lambda: 1
        with self.assertRaisesRegex(RuntimeError, "before task physics"):
            method(env)
        self.assertEqual(calls, [])

    def test_actual_actuator_construction_splits_hand7_from_the_original_pd36(self):
        from g1_vla.rl_effort_protocol import RIGHT_HAND_JOINT_NAMES, effort_grasp_contract
        hand = tuple(RIGHT_HAND_JOINT_NAMES)
        names = ["other_joint_{}".format(i) for i in range(36)]+list(hand)
        specs = {name: {"kp": 0. if name in hand else 400., "kd": 0. if name in hand else 6.,
                         "effort_limit_nm": .04 if name in hand else 25.,
                         "armature": .0005, "dof_damping": .02} for name in names}
        class Cfg(SimpleNamespace):
            pass
        Cfg.InitialStateCfg = Cfg
        class Implicit(Cfg):
            pass
        class Explicit(Cfg):
            pass
        scope = dict(self.scope, actuator_specs=specs, effort_model=True, RIGHT_HAND=hand,
            effort_contract=effort_grasp_contract(), ImplicitActuatorCfg=Implicit, IdealPDActuatorCfg=Explicit,
            ArticulationCfg=Cfg, Articulation=lambda cfg: SimpleNamespace(cfg=cfg), robot_spawn=object(),
            initial={"root_pose_xyzw": [0., 0., 0., 0., 0., 0., 1.],
                     "joint_positions_rad": {name: 0. for name in names}})
        env = SimpleNamespace()
        actual_method("_setup_scene", scope, before_assignment="self.cube")(env)
        actuators = env.robot.cfg.actuators
        implicit = actuators["shared_pd_without_mujoco_bias_v1"]
        explicit = actuators["right_hand_bounded_explicit_effort_v1"]
        self.assertIsInstance(implicit, Implicit)
        self.assertIsInstance(explicit, Explicit)
        self.assertEqual(len(implicit.joint_names_expr), 36)
        self.assertFalse(set(implicit.joint_names_expr) & set(hand))
        self.assertEqual(explicit.joint_names_expr, list(hand))
        self.assertEqual(explicit.stiffness, 0.)
        self.assertEqual(explicit.damping, 0.)
        self.assertEqual(explicit.effort_limit, explicit.effort_limit_sim)
        self.assertEqual(list(explicit.effort_limit.values()), [.06]+[.04]*6)

    def test_native_drive_readback_rejects_a_remaining_pd_drive_or_changed_other_joint(self):
        t = self.t
        from g1_vla.rl_effort_protocol import RIGHT_HAND_JOINT_NAMES
        names = ["other_joint_{}".format(i) for i in range(36)]+list(RIGHT_HAND_JOINT_NAMES)
        hand_ids = list(range(36, 43))
        kp = t.tensor([[400.]*36+[0.]*7]*2)
        kd = t.tensor([[6.]*36+[0.]*7]*2)
        maximum = t.tensor([[25.]*36+[.06]+[.04]*6]*2)
        specs = {name: {"kp": float(kp[0, i]), "kd": float(kd[0, i]),
                        "effort_limit_nm": float(maximum[0, i])} for i, name in enumerate(names)}
        scope = dict(self.scope, actuator_specs=specs, _probe_payload=lambda torch, value: value)
        env = SimpleNamespace(num_envs=2, _action_ids=list(range(7))+hand_ids,
            robot=SimpleNamespace(joint_names=names, root_view=SimpleNamespace(
                get_dof_stiffnesses=lambda: kp, get_dof_dampings=lambda: kd,
                get_dof_max_forces=lambda: maximum),
                data=SimpleNamespace(gravity_compensation_forces=t.zeros(2, 43)),
                actuators={"right_hand_bounded_explicit_effort_v1": type("IdealPDActuator", (), {})()}))
        method = actual_method("_verify_effort_native_model", scope, before_assignment="stage")
        method(env)
        self.assertTrue(env._effort_drive_readback["verified"])
        kp[1, hand_ids[0]] = 9.
        with self.assertRaisesRegex(RuntimeError, "native hand7 drives"):
            method(env)
        kp[1, hand_ids[0]] = 0.
        kd[1, 0] = 5.
        with self.assertRaisesRegex(RuntimeError, "native hand7 drives"):
            method(env)

    def update_fixture(self):
        t, calls, rows = self.t, [], []
        def effort_update(**kw):
            calls.append(kw)
            return t.full((2, 7), .01), {"group_force_command_n": t.tensor([[.8, .4, .4]]*2)}, t.zeros(2, dtype=t.bool)
        ids = t.tensor([2, 3])
        quat = t.zeros(2, 9, 4)
        quat[..., 3] = 1.
        env = SimpleNamespace(num_envs=2, device="cpu", _episode_ids=ids,
            _action_ids=list(range(14)), _palm_id=8, _palm_local=t.zeros(2, 3),
            _palm_local_rotation=t.tensor([[0., 0., 0., 1.]]*2),
            _effort_reference={"episode_id": ids.clone(), "nominal_hand_rad": t.zeros(2, 7),
                "geometry_lower_hand_rad": t.full((2, 7), -1.), "geometry_upper_hand_rad": t.full((2, 7), 1.),
                "closing_axis_hand7": t.tensor([-1., 0., 0., 1., 1., 1., 1.]),
                "palm_reference_world_m": t.zeros(2, 3),
                "palm_reference_quaternion_xyzw": t.tensor([[0., 0., 0., 1.]]*2)},
            _last_contact_episode_ids=ids.clone(), _last_contact_native_step=9,
            _effort_geometry={"current_normal_available": t.ones(2, 3, dtype=t.bool),
                "normal_jacobian": t.zeros(2, 3, 7), "inward_normal_world": t.zeros(2, 3, 3),
                "point_world_m": t.zeros(2, 3, 3), "native_step": 9, "episode_id": ids.clone()},
            _last_metrics={"finite": t.ones(2, dtype=t.bool), "joint_safe": t.ones(2, dtype=t.bool),
                "table_pair_measurement_supported": t.ones(2, dtype=t.bool),
                "group_physical_loaded_contact_count": t.ones(2, 3, dtype=t.long),
                "group_normal_force_n": t.tensor([[.8, .4, .4], [54., .4, .4]]),
                "maximum_penetration_m": t.zeros(2)},
            _unsafe=t.zeros(2, dtype=t.bool), _limits=t.tensor([[[-2., 2.]]*43]*2),
            _velocity=t.full((14,), .9), _effort_update_count=0, _control_command_id=1,
            _effort_grasp=SimpleNamespace(update=effort_update), _write_control_probe=rows.append,
            _object_weight_support_torque=lambda **kw: (t.zeros(2, 7), {"gate": t.zeros(2, dtype=t.bool)}),
            sim=SimpleNamespace(get_physics_step_count=lambda: 9), physics_dt=.002,
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=t.zeros(2, 43), joint_vel=t.zeros(2, 43),
                body_pos_w=t.zeros(2, 9, 3), body_quat_w=quat, gravity_compensation_forces=t.full((2, 43), .002))),
            cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=t.zeros(2, 3))))
        # This fixture deliberately has identity rotations and zero offset.
        scope = dict(self.scope, quat_apply=lambda q, v: v, quat_mul=lambda q, r: q)
        return env, calls, rows, actual_method("_update_effort_grasp", scope)

    def test_domain_guard_native_install_uses_original_candidate_domain_not_search_box(self):
        from g1_vla.rl_effort_protocol import effort_grasp_contract
        t = self.t
        contract = effort_grasp_contract()
        candidate = json.loads((ROOT/"resources/rl_geometry_candidates/opposed_face_effort_pinch_v1.json").read_text())
        declaration = json.loads((ROOT/"configs/experiment.json").read_text())["rl_diagnostics"]["contact_domain_posture_guard"]
        q = t.zeros(2, 43)
        q[:, 7:14] = t.tensor(candidate["source_hand_rad"])
        events = []
        env = SimpleNamespace(num_envs=2, device="cpu", _effort_grasp=None,
            _action_ids=list(range(14)), _episode_ids=t.tensor([2,2]),
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=q)),
            sim=SimpleNamespace(get_physics_step_count=lambda: 0),
            _write_control_probe=events.append, _effort_drive_readback={}, _effort_material_readback={})
        method = actual_method("configure_effort_grasp", dict(self.scope, effort_model=True,
            effort_contract=contract, effort_candidate=candidate))
        method(env, contract, domain_posture_guard=declaration)
        frames = candidate["geometry_evidence"]["source_detail"]["stage_keyframes"]
        raw = t.tensor([f.get("hand_target_rad", f.get("target_rad")) for f in frames])
        self.assertTrue(t.equal(env._effort_actual_domain_lower, (raw.amin(0)-.025)[None].expand(2,-1)))
        self.assertTrue(t.equal(env._effort_actual_domain_upper, (raw.amax(0)+.025)[None].expand(2,-1)))
        self.assertEqual(events[-1]["contact_domain_posture_guard"], declaration)
        self.assertFalse(events[-1]["optimizer_steps"])
        with self.assertRaisesRegex(RuntimeError, "cannot be replaced"):
            method(env, contract, domain_posture_guard=declaration)

    def test_domain_guard_adapter_forwards_current_point_geometry_and_original_bounds(self):
        env, calls, rows, method = self.update_fixture()
        t = self.t
        env._effort_domain_posture_guard = {"version": "fixture"}
        env._effort_actual_domain_lower = t.full((2, 7), -.9)
        env._effort_actual_domain_upper = t.full((2, 7), .9)
        jac = t.arange(126).reshape(2,3,3,7).float()
        env._effort_geometry["point_jacobian_world_m"] = jac
        env._last_metrics["contact_safe"] = t.tensor([True, False])
        method(env)
        self.assertTrue(t.equal(calls[-1]["contact_point_jacobian_world_m"], jac))
        self.assertTrue(t.equal(calls[-1]["actual_domain_lower_hand_rad"], env._effort_actual_domain_lower))
        self.assertTrue(t.equal(calls[-1]["actual_domain_upper_hand_rad"], env._effort_actual_domain_upper))
        self.assertTrue(t.equal(calls[-1]["physical_contact_safe"], t.tensor([True, False])))
        # A stale geometry cannot be promoted by a physically-safe cached flag.
        env._effort_geometry["native_step"] = 8
        method(env)
        self.assertFalse(calls[-1]["group_contact_valid"].any())
        del env._effort_domain_posture_guard
        method(env)
        self.assertNotIn("contact_point_jacobian_world_m", calls[-1])
        self.assertNotIn("actual_domain_lower_hand_rad", calls[-1])

    def weight_support_fixture(self):
        from g1_vla.rl_effort_protocol import effort_grasp_contract
        from g1_vla.rl_environment import preload_contact_ready
        t = self.t
        ids = t.tensor([2, 3])
        jac = t.zeros(2, 3, 3, 7)
        jac[:, 0, 2, 0] = .02
        jac[:, 1, 2, 3] = -.03
        jac[:, 2, 2, 5] = .04
        geometry = {"point_jacobian_world_m": jac, "episode_id": ids.clone(), "native_step": 9}
        fields = ("contact_safe", "grip_reference_locked", "no_positive_gap_hand_load",
                  "friction_partner_manifold_certified", "finite", "joint_safe")
        env = SimpleNamespace(num_envs=2, device="cpu", physics_dt=.002,
            _episode_ids=ids.clone(), _object_support_episode_ids=ids.clone(),
            _object_support_scale=t.zeros(2), _object_support_last_sample=None,
            _last_metrics={name: t.ones(2, dtype=t.bool) for name in fields},
            _unsafe=t.zeros(2, dtype=t.bool), _grip_locked=t.ones(2, dtype=t.bool),
            sim=SimpleNamespace(get_physics_step_count=lambda: geometry["native_step"]))
        scope = dict(self.scope, effort_contract=effort_grasp_contract(), preload_contact_ready=preload_contact_ready)
        args = {"valid": t.ones(2, dtype=t.bool), "tracking": t.ones(2, dtype=t.bool),
                "geometry": geometry, "current": t.ones(2, 3, dtype=t.bool)}
        return env, args, actual_method("_object_weight_support_torque", scope)

    def test_object_weight_world_z_uses_current_point_jacobian_and_ramps_inside_requested_budget(self):
        env, args, method = self.weight_support_fixture()
        for sample in range(100):
            args["geometry"]["native_step"] = 9+sample
            tau, evidence = method(env, **args)
        expected = self.t.tensor([[.007848, 0., 0., -.005886, 0., .007848, 0.]]*2)
        self.assertTrue(self.t.allclose(tau, expected, atol=1e-8, rtol=0))
        self.assertTrue(self.t.allclose(evidence["current_group_requested_force_world_n"].sum(1)[:, 2],
                                        self.t.full((2,), .7848), atol=1e-7, rtol=0))
        self.assertTrue(evidence["request_is_not_measured_object_support_or_friction_certification"])

    def test_object_support_requires_all_original_contact_fields_current_epoch_and_tracking(self):
        for failure in ("certificate", "missing_field", "stale", "normal_missing", "tracking", "unsafe"):
            env, args, method = self.weight_support_fixture()
            env._object_support_scale[:] = 1
            if failure == "certificate": env._last_metrics["friction_partner_manifold_certified"][1] = False
            elif failure == "missing_field": del env._last_metrics["contact_safe"]
            elif failure == "stale": args["geometry"]["episode_id"][1] = 2
            elif failure == "normal_missing": args["current"][1, 2] = False
            elif failure == "tracking": args["tracking"][1] = False
            else: env._unsafe[1] = True
            with self.subTest(failure=failure):
                tau, evidence = method(env, **args)
                self.assertFalse(bool(evidence["gate"][1]))
                self.assertTrue((tau[1] == 0).all())
                self.assertEqual(float(env._object_support_scale[1]), 0.)

    def test_object_support_duplicate_samples_do_not_speed_ramp_and_episode_change_clears_it(self):
        env, args, method = self.weight_support_fixture()
        first, _ = method(env, **args)
        second, _ = method(env, **args)
        self.assertTrue(self.t.equal(first, second))
        env._object_support_scale[:] = 1
        env._episode_ids[1] = 4  # Geometry from episode3 must never support episode4.
        _, evidence = method(env, **args)
        self.assertEqual(float(evidence["ramp_scale"][1]), 0.)

    def test_object_support_rejects_nonfinite_point_jacobian(self):
        env, args, method = self.weight_support_fixture()
        args["geometry"]["point_jacobian_world_m"][0, 1, 2, 3] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite current"):
            method(env, **args)

    def test_causal_adapter_preserves_unclipped_force_and_marks_old_episode_feedback_unavailable(self):
        env, calls, rows, method = self.update_fixture()
        env._last_contact_episode_ids[1] = 2
        method(env)
        self.assertEqual(calls[0]["sample_id"], 9)
        self.assertEqual(calls[0]["dt"], .002)
        self.assertTrue(self.t.equal(calls[0]["contact_measurement_valid"], self.t.tensor([True, False])))
        self.assertFalse(calls[0]["group_contact_valid"][1].any())
        self.assertEqual(float(calls[0]["raw_group_force_n"][1, 0]), 54.)
        self.assertTrue(self.t.equal(calls[0]["gravity_hand_nm"], self.t.full((2, 7), .002)))
        self.assertEqual(rows[0]["event"], "physics_hand_effort")
        self.assertEqual(rows[0]["force_measurement_native_step"], 9)

    def test_geometry_own_sample_and_episode_identity_must_match_force_sample(self):
        env, calls, _, method = self.update_fixture()
        env._effort_geometry["native_step"] = 8
        method(env)
        self.assertFalse(calls[0]["group_contact_valid"].any(), "stale A cannot be paired with fresh Fn")
        env, calls, _, method = self.update_fixture()
        env._effort_geometry["episode_id"][1] = 2
        method(env)
        self.assertTrue(calls[0]["group_contact_valid"][0].all())
        self.assertFalse(calls[0]["group_contact_valid"][1].any())


@unittest.skipUnless(TORCH, "real torch is required for state invariance checks")
class CollisionGeometryBoundaryTests(unittest.TestCase):
    def fixture(self):
        import hashlib
        import torch
        t = torch
        names = ["right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"]
        data = SimpleNamespace(joint_pos=t.zeros(8, 43), joint_vel=t.zeros(8, 43),
            body_pos_w=t.zeros(8, 3, 3), body_quat_w=t.tensor([0., 0., 0., 1.]).repeat(8, 3, 1))
        cube = SimpleNamespace(root_pos_w=t.zeros(8, 3),
            root_quat_w=t.tensor([0., 0., 0., 1.]).repeat(8, 1),
            root_lin_vel_w=t.zeros(8, 3), root_ang_vel_w=t.zeros(8, 3))
        env = SimpleNamespace(_physics_samples=0, _pending_physics_sample=False, step=0,
            robot=SimpleNamespace(data=data, body_names=names,
                root_view=SimpleNamespace(prim_paths=[f"/World/envs/env_{i}/Robot/root_joint" for i in range(8)])),
            cube=SimpleNamespace(data=cube))
        env.sim = SimpleNamespace(get_physics_step_count=lambda: env.step,
                                  cfg=SimpleNamespace(enable_scene_query_support=True))
        for name in ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_max_forces", "get_dof_friction_properties",
                     "get_contact_offsets", "get_rest_offsets"):
            setattr(env.robot.root_view, name, lambda: t.ones(8, 43))
        for body in (env.cube,):
            body.root_view = SimpleNamespace(get_contact_offsets=lambda: t.zeros(8, 1),
                                             get_rest_offsets=lambda: -t.ones(8, 1)*1e-6)
        env.table = SimpleNamespace(root_view=env.cube.root_view)
        stage = SimpleNamespace(layer_text="unchanged stage")
        layer = SimpleNamespace(identifier="active-layer", ExportToString=lambda: stage.layer_text)
        stage.GetLayerStack = lambda: [layer]
        method = actual_method("export_collision_geometry_audit", {
            "torch": t, "tensor": lambda v: v, "RLEnvironmentError": RuntimeError,
            "effort_model": True, "output": Path("/unused-audit-output"),
            "get_current_stage": lambda: stage, "json": json, "hashlib": hashlib,
            "__package__": "g1_vla", "__name__": "g1_vla.rl_environment"})
        return env, method, stage, {"env_ids": [0], "body_names": names, "require_complete": True}

    def call(self, env, method, declaration, callback):
        from unittest.mock import patch
        import sys
        with patch.dict(sys.modules, {"g1_vla.rl_collision_geometry":
                SimpleNamespace(export_collision_geometry=callback, export_collision_surface_audit=callback)}):
            return method(env, declaration)

    def test_exports_same_stage_and_native_link_identity_without_task_step(self):
        env, method, stage, declaration = self.fixture()
        def export(actual, *, bodies, output_path):
            self.assertIs(actual, stage)
            self.assertEqual([x["native_link_index"] for x in bodies], [0, 1, 2])
            self.assertTrue(all(x["body_path"].startswith("/World/envs/env_0/Robot/") for x in bodies))
            return {"complete": True, "attached_shape_cache_verified": False}
        result = self.call(env, method, declaration, export)
        self.assertTrue(result["current_state_unchanged"])
        self.assertEqual(result["native_step_before"], result["native_step_after"])
        self.assertEqual(result["task_physics_steps"], 0)
        self.assertFalse(result["attached_shape_cache_verified"])
        self.assertTrue(result["observed_native_properties_and_usd_layers_unchanged"])

    def test_unavailable_representation_cannot_start_the_physical_profile(self):
        env, method, _, declaration = self.fixture()
        with self.assertRaises(RuntimeError):
            self.call(env, method, declaration, lambda *a, **k: {"complete": False})

    def test_surface_route_is_explicit_and_does_not_claim_full_hull(self):
        env, method, _, declaration = self.fixture()
        declaration["route"] = "live_surface_queries_v2"
        result = self.call(env, method, declaration,
            lambda *a, **k: {"complete": True, "full_convex_hull_complete": False})
        self.assertEqual(result["geometry_observation_route"], "live_surface_queries_v2")
        self.assertFalse(result["full_convex_hull_complete"])

    def test_any_step_or_measured_state_mutation_rejects_before_profile(self):
        for change in ("step", "q", "cube_velocity", "body_pose", "native_drive", "usd_layer"):
            env, method, stage, declaration = self.fixture()
            def export(*a, **k):
                if change == "step": env.step += 1
                elif change == "q": env.robot.data.joint_pos[0, 0] += .00001
                elif change == "cube_velocity": env.cube.data.root_lin_vel_w[0, 0] += .00001
                elif change == "body_pose": env.robot.data.body_pos_w[0, 0, 0] += .00001
                elif change == "usd_layer": stage.layer_text = "unexpected changed layer"
                else:
                    old = env.robot.root_view.get_dof_max_forces
                    env.robot.root_view.get_dof_max_forces = lambda: old()*2
                return {"complete": True}
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.call(env, method, declaration, export)

    def test_after_physics_or_wrong_native_root_cannot_export(self):
        for change in ("physics", "pending", "root"):
            env, method, _, declaration = self.fixture()
            if change == "physics": env._physics_samples = 1
            elif change == "pending": env._pending_physics_sample = True
            else: env.robot.root_view.prim_paths[0] = "/World/unrelated/Robot/root_joint"
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                self.call(env, method, declaration, lambda *a, **k: {"complete": True})


if __name__ == "__main__":
    unittest.main()
