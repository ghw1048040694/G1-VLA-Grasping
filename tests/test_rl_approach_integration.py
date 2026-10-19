"""Unified arm-home contract and actual guide/reset regressions, without Sim."""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from g1_vla.rl_approach import APPROACH_PARAMETERS
from g1_vla.rl_approach_state import load_course_reference
from g1_vla.rl_effort_protocol import full_lift_effort_grasp_contract, full_lift_protocol_contract
from g1_vla.rl_preshape import (
    ARM_HOME_CONTROL_PRESHAPE_VERSION, control_preshape_contract,
    resolve_preshape_frames, validate_control_preshape,
)
from g1_vla.rl_probe_protocol import validate_probe_settings


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_PATH = ROOT/"resources/rl_geometry_candidates/opposed_face_effort_pinch_v2.json"
COURSE_PATH = ROOT/"resources/rl_approach/high_wait_vertical_entry_v1.json"
CANDIDATE_SHA = "2daa38dafba243eae42cd1450d9a0030f7d0503e6073647309109f3cbc885a60"
HOME_FULL_VERSION = "g1_contact_qualified_arm_home_lift_v5_32s"
OLD_FULL_VERSION = "g1_contact_qualified_full_lift_v4_native_arm_16s"
HAS_TORCH = importlib.util.find_spec("torch") is not None


def candidate():
    payload = CANDIDATE_PATH.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CANDIDATE_SHA:
        raise AssertionError("test requires the actual pinned original candidate")
    return json.loads(payload)


def configuration():
    value = json.loads((ROOT/"configs/experiment.json").read_text())
    value["paths"]["root"] = str(ROOT)
    return value


def diagnostic_settings(*, home=True, num_envs=1):
    """Select a complete named task, independent of the currently active quota."""
    value = configuration()["rl_diagnostics"]
    protocol = ("home_lift1_v1" if num_envs == 1 else "home_lift8_v1") if home else "full_lift8_v2"
    declared = full_lift_effort_grasp_contract(protocol)
    # Retain the selected D3 compliant material/controller, which must match its
    # transition declaration. Only task acceptance/version/quota are selected.
    for key in ("version", "evaluation", "acceptance"):
        value["effort_grasp_controller"][key] = copy.deepcopy(declared[key])
    quota = declared["evaluation"]
    value.update(num_envs=quota["num_envs"], episodes_per_env=quota["episodes_per_env"],
        max_control_steps=quota["max_control_steps"], max_wall_time_s=quota["max_wall_time_s"],
        full_lift=full_lift_protocol_contract(HOME_FULL_VERSION if home else OLD_FULL_VERSION),
        control_preshape=control_preshape_contract(ARM_HOME_CONTROL_PRESHAPE_VERSION if home else None),
        video_recording=None)
    value["approach_course"] = ({"version": "g1_task_preshape_arm_home_approach_v1",
        "resource_path": "resources/rl_approach/high_wait_vertical_entry_v1.json",
        "resource_sha256": hashlib.sha256(COURSE_PATH.read_bytes()).hexdigest(),
        "parameters": copy.deepcopy(APPROACH_PARAMETERS)}
        if home else None)
    return value


class UnifiedApproachContractTests(unittest.TestCase):
    def test_complete_named_home1_home8_and_old_precontact_contracts_are_admitted(self):
        for home, population in ((True, 1), (True, 8), (False, 8)):
            with self.subTest(home=home, population=population):
                value = diagnostic_settings(home=home, num_envs=population)
                before = copy.deepcopy(value)
                checked = validate_probe_settings(value)
                self.assertEqual(value, before)
                self.assertEqual(checked["num_envs"], population)
                self.assertIs(checked["training_allowed"], False)
                self.assertEqual(checked["optimizer_steps"], 0)
                self.assertEqual(checked["full_lift"]["hold_consecutive_samples"], 251)
                self.assertEqual(checked["full_lift"]["episode_length_s"], 32. if home else 16.)

    def test_home_v5_cannot_use_no_course_even_with_legacy_preshape(self):
        value = diagnostic_settings()
        value.update(approach_course=None, control_preshape=control_preshape_contract())
        with self.assertRaisesRegex(ValueError, "arm-home full task"):
            validate_probe_settings(value)

    def test_v2_preshape_cannot_claim_an_ordinary_precontact_reset(self):
        value = diagnostic_settings(home=False, num_envs=8)
        value["control_preshape"] = control_preshape_contract(ARM_HOME_CONTROL_PRESHAPE_VERSION)
        with self.assertRaisesRegex(ValueError, "Arm-home preshape requires"):
            validate_probe_settings(value)

    def test_legacy_v1_preshape_cannot_be_attached_to_new_home_course(self):
        value = diagnostic_settings()
        value["control_preshape"] = control_preshape_contract()
        with self.assertRaisesRegex(ValueError, "Arm-home course requires"):
            validate_probe_settings(value)

    def test_population_and_quota_mismatches_cannot_change_which_episodes_are_scored(self):
        for population, mutate in (
                (1, lambda v:v.update(num_envs=8)),
                (8, lambda v:v.update(num_envs=1)),
                (1, lambda v:v["effort_grasp_controller"]["evaluation"].update(num_envs=8)),
                (8, lambda v:v["effort_grasp_controller"]["evaluation"].update(num_envs=1)),
                (1, lambda v:v.update(episodes_per_env=4)),
                (1, lambda v:v.update(max_control_steps=243))):
            value = diagnostic_settings(num_envs=population); mutate(value)
            with self.subTest(population=population, mutation=mutate), self.assertRaises(ValueError):
                validate_probe_settings(value)

    def test_v1_v2_have_identical_numeric_frames_domains_and_distinct_truthful_reset_scope(self):
        old, new = control_preshape_contract(), control_preshape_contract(ARM_HOME_CONTROL_PRESHAPE_VERSION)
        differences = {key for key in old if old[key] != new[key]}
        self.assertEqual(differences, {"version", "palm_offset_schedule", "original_reset_and_episode_budget_unchanged"})
        self.assertIs(old["original_reset_and_episode_budget_unchanged"], True)
        self.assertIs(new["original_reset_and_episode_budget_unchanged"], False)
        self.assertIs(new["original_hand_frames_and_physical_grasp_gates_unchanged"], True)
        self.assertEqual(resolve_preshape_frames(old, candidate(), CANDIDATE_SHA),
                         resolve_preshape_frames(new, candidate(), CANDIDATE_SHA))
        new["original_reset_and_episode_budget_unchanged"] = True
        with self.assertRaises(ValueError): validate_control_preshape(new)


class CourseReferenceBindingTests(unittest.TestCase):
    def test_actual_home_fk_reference_uses_home_pose_and_retains_original_asset_identity(self):
        original = candidate()
        before = copy.deepcopy(original)
        resource, fk = load_course_reference(configuration(), diagnostic_settings(), original)
        self.assertEqual(original, before)
        self.assertEqual(fk["palm_world_position_m"], resource["home_palm_position_world_m"])
        self.assertEqual(fk["palm_quaternion_xyzw"], resource["home_palm_quaternion_xyzw"])
        self.assertNotEqual(fk["palm_world_position_m"], original["reset_native_fk_reference"]["palm_world_position_m"])
        self.assertGreater(fk["palm_world_position_m"][2]-resource["pregrasp_palm_position_world_m"][2], .17)
        for key in ("usd_entry_sha256", "usd_structure_sha256", "joint_position_units", "root_body", "palm_site_body", "body_count"):
            self.assertEqual(fk[key], original["reset_native_fk_reference"][key])
        self.assertEqual(fk["original_pregrasp_candidate_sha256"], CANDIDATE_SHA)
        self.assertEqual(resource["pregrasp_joint_positions_rad"], original["reset_joint_positions_rad"])
        self.assertEqual(resource["source_hand_rad"], original["source_hand_rad"])
        resource["home_palm_position_world_m"][2] = 77.
        self.assertNotEqual(fk["palm_world_position_m"][2], 77., "FK and resource must not share mutable arrays")

    def test_rehashed_but_changed_pregrasp_or_root_cannot_rebind_the_pinned_candidate(self):
        from g1_vla.rl_approach import ARM_JOINT_NAMES, HAND_JOINT_NAMES
        changes = ("arm", "root", "hand", "palm", "quaternion")
        for change in changes:
            resource = json.loads(COURSE_PATH.read_text())
            if change == "arm":
                resource["pregrasp_joint_positions_rad"][ARM_JOINT_NAMES[0]] += 1e-8
                resource["waypoints"][-1]["arm_target_rad"][0] += 1e-8
            elif change == "root": resource["root_pose_xyzw"][0] += 1e-8
            elif change == "hand":
                resource["source_hand_rad"][0] += 1e-8
                for key in ("home_joint_positions_rad", "pregrasp_joint_positions_rad"):
                    resource[key][HAND_JOINT_NAMES[0]] += 1e-8
            elif change == "palm":
                resource["pregrasp_palm_position_world_m"][0] += 1e-8
                resource["waypoints"][-1]["palm_position_world_m"][0] += 1e-8
            else:
                for owner, key in ((resource, "pregrasp_palm_quaternion_xyzw"),
                                   (resource["waypoints"][-1], "palm_quaternion_xyzw")):
                    owner[key] = [-x for x in owner[key]]
            with self.subTest(change=change), TemporaryDirectory() as directory:
                path = Path(directory)/"course.json"; path.write_text(json.dumps(resource))
                settings = diagnostic_settings()
                settings["approach_course"].update(resource_path=str(path),
                    resource_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                with self.assertRaisesRegex(ValueError, "exact original pregrasp target and root"):
                    load_course_reference(configuration(), settings, candidate())


@unittest.skipUnless(HAS_TORCH, "real Torch CPU is required for actual guide/reset math")
class ActualGuideAndResetIntegrationTests(unittest.TestCase):
    def fixture(self):
        import torch as t
        self.t = t
        source = ROOT/"src/g1_vla/rl_diagnostics.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if (
            isinstance(node, ast.ClassDef) and node.name == "EffortGraspGuide") or (
            isinstance(node, ast.FunctionDef) and node.name in ("_bounded_vector", "_orientation_error")) or (
            isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "POSE_SERVO" for x in node.targets))]
        namespace = {"__name__": "g1_vla.rl_diagnostics", "__package__": "g1_vla", "copy": copy}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual-EffortGraspGuide>", "exec"), namespace)
        cfg, settings, original = configuration(), diagnostic_settings(), candidate()
        resource, _ = load_course_reference(cfg, settings, original)
        origins = t.tensor([[0., 0., 0.], [2., 0., 1.]])
        q = t.zeros(2, 14); q[:, 7:] = t.tensor(original["source_hand_rad"])
        limits = t.tensor([[[-2., 2.]]*7+original["geometry_evidence"]["hand_joint_limits_rad"]]*2)
        pre = t.tensor(resource["pregrasp_palm_position_world_m"])[None]+origins
        rotation = t.tensor(resource["pregrasp_palm_quaternion_xyzw"])[None].repeat(2,1)
        state = {"q_action_rad": q, "target_action_rad": q.clone(), "joint_limits_action_rad": limits,
            "episode_id": t.tensor([2, 3]), "palm_position_world_m": t.tensor(resource["home_palm_position_world_m"])[None]+origins,
            "palm_quaternion_xyzw": t.tensor(resource["home_palm_quaternion_xyzw"])[None].repeat(2,1),
            "group_normal_force_n": t.zeros(2,3), "unsafe_latched": t.zeros(2,dtype=t.bool),
            "preload_contact_ready": t.zeros(2,dtype=t.bool), "metrics": {},
            "palm_point_jacobian_w": t.eye(6,7)[None].repeat(2,1,1), "per_step_action_rad": t.full((2,14),.0198),
            "control_preshape_contact_seen": t.zeros(2,dtype=t.bool),
            "control_preshape_first_contact_native": t.full((2,),-1,dtype=t.int64),
            "control_preshape_contact_episode_ids": t.tensor([2,3]), "native_physics_step_count": 0,
            "approach_admitted": t.zeros(2,dtype=t.bool),
            "approach_pregrasp_palm_position_world_m": pre.clone(),
            "approach_pregrasp_palm_quaternion_xyzw": rotation.clone()}
        guide = namespace["EffortGraspGuide"](t, settings, original["source_hand_rad"], state, original)
        return guide, state, resource, origins

    def action(self, guide, state):
        state["native_physics_step_count"] += 33
        return guide.actions(self.t.zeros(2,14), state, "guided_hand_pose_hold", .066, 1.5)

    def test_actual_near_pregrasp_without_admission_never_closes_or_spends_grasp_stage_budget(self):
        guide, state, _, _ = self.fixture()
        state["palm_position_world_m"] = state["approach_pregrasp_palm_position_world_m"].clone()
        state["palm_quaternion_xyzw"] = state["approach_pregrasp_palm_quaternion_xyzw"].clone()
        seed = state["q_action_rad"][:,7:].clone()
        for _ in range(30):
            _, evidence = self.action(guide, state)
            self.assertTrue(self.t.equal(evidence["guided_target_hand_rad"], seed))
            self.assertFalse(bool(evidence["probe_advance_permitted"].any()))
        self.assertTrue(self.t.equal(guide.elapsed, self.t.zeros(2)))
        self.assertEqual(guide.stage.tolist(), [0,0])
        self.assertTrue(self.t.equal(guide.preshape_fraction, self.t.zeros(2)))

    def test_actual_measured_admission_releases_only_its_own_hand_without_inventing_support(self):
        guide, state, _, _ = self.fixture()
        state["palm_position_world_m"] = state["approach_pregrasp_palm_position_world_m"].clone()
        state["palm_quaternion_xyzw"] = state["approach_pregrasp_palm_quaternion_xyzw"].clone()
        state["approach_admitted"][0] = True
        before = state["q_action_rad"].clone()
        _, evidence = self.action(guide, state)
        self.assertGreater(float((evidence["guided_target_hand_rad"][0]-before[0,7:]).abs().max()), 0.)
        self.assertTrue(self.t.equal(evidence["guided_target_hand_rad"][1], before[1,7:]))
        self.assertGreater(float(guide.elapsed[0]), 0.)
        self.assertEqual(float(guide.elapsed[1]), 0.)
        self.assertTrue(self.t.equal(state["q_action_rad"], before), "Guide cannot write measured joints")
        self.assertFalse(bool(evidence["closure_path_complete"].any()))
        self.assertFalse(bool(evidence["probe_advance_permitted"].any()))

    def test_actual_guide_keeps_static_pregrasp_z_through_handoff_and_new_home_reset(self):
        guide, state, resource, origins = self.fixture()
        expected = state["approach_pregrasp_palm_position_world_m"].clone()
        self.assertTrue(self.t.equal(guide.palm, expected))
        self.assertGreater(float((state["palm_position_world_m"]-guide.palm)[:,2].min()), .17)
        state["palm_position_world_m"] = expected.clone();state["palm_position_world_m"][:,2] += .002
        state["palm_quaternion_xyzw"] = state["approach_pregrasp_palm_quaternion_xyzw"].clone()
        state["approach_admitted"][:] = True
        _, evidence = self.action(guide, state)
        self.assertTrue(self.t.equal(evidence["palm_pose_reference_world_m"][:,2], expected[:,2]))
        state["episode_id"] += 2;state["control_preshape_contact_episode_ids"] = state["episode_id"].clone()
        state["approach_admitted"][:] = False
        state["palm_position_world_m"] = self.t.tensor(resource["home_palm_position_world_m"])[None]+origins
        _, evidence = self.action(guide, state)
        self.assertTrue(self.t.equal(evidence["palm_pose_reference_world_m"][:,2], expected[:,2]))
        self.assertTrue(self.t.equal(guide.elapsed, self.t.zeros(2)))

    def test_actual_reset_branch_anchors_pregrasp_not_home_and_clears_only_selected_episode(self):
        t = __import__("torch")
        resource = json.loads(COURSE_PATH.read_text())
        tree = ast.parse((ROOT/"src/g1_vla/rl_environment.py").read_text())
        cls = next(x for x in ast.walk(tree) if isinstance(x,ast.ClassDef) and x.name=="Environment")
        reset = next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="_reset_idx")
        block = next(x for x in reset.body if isinstance(x,ast.If) and
            "approach_course" in ast.unparse(x.test) and any(
                isinstance(y,ast.Assign) and any("self._start_palm_z" in ast.unparse(z)
                    for z in y.targets) for y in x.body))
        calls = []
        spy = lambda label: SimpleNamespace(reset=lambda *args:calls.append((label,args)))
        env = SimpleNamespace(_start_palm_z=t.tensor([77.,88.]), _approach=spy("admission"), _approach_contact=spy("contact"),
            _episode_ids=t.tensor([4,8]), _approach_plan_complete=t.ones(2,dtype=t.bool),
            _approach_planner_failed=t.ones(2,dtype=t.bool), _approach_intent_episode_ids=t.tensor([3,7]),
            _approach_measurement_unavailable=["old0","old1"], sim=SimpleNamespace(get_physics_step_count=lambda:99))
        env_ids, origins = t.tensor([1]), t.tensor([[2.,0.,3.]])
        cube = t.tensor([[2.35,.055,3.775,0.,0.,0.,1.]])
        scope = {"self":env,"env_ids":env_ids,"origins":origins,"cube":cube,
                 "approach_course":diagnostic_settings()["approach_course"],"approach_resource":resource}
        exec(compile(ast.Module(body=[block],type_ignores=[]),"<actual-arm-home-reset-branch>","exec"),scope)
        self.assertEqual(float(env._start_palm_z[0]),77.)
        self.assertAlmostEqual(float(env._start_palm_z[1]),resource["pregrasp_palm_position_world_m"][2]+3.,places=6)
        self.assertNotAlmostEqual(float(env._start_palm_z[1]),resource["home_palm_position_world_m"][2]+3.,places=3)
        self.assertEqual(env._approach_measurement_unavailable,["old0",None])
        self.assertEqual(env._approach_plan_complete.tolist(),[True,False])
        self.assertEqual([label for label,_ in calls],["admission","contact"])
        self.assertTrue(t.equal(calls[0][1][3],cube))


@unittest.skipUnless(HAS_TORCH, "real Torch CPU is required for producer/consumer interface math")
class ActualDiagnosticStatePlannerInterfaceTests(unittest.TestCase):
    def producer(self, population, dtype, *, course_selected=True):
        """Fake native buffers only; the complete state producer is real AST."""
        import torch as t
        from g1_vla.rl_environment import ACTION_JOINT_NAMES, offset_point_jacobian, preload_contact_ready
        resource = json.loads(COURSE_PATH.read_text())
        # A non-action-first native DOF order also tests the real index mapping.
        names = sorted(resource["home_joint_positions_rad"], reverse=True)
        action_ids = [names.index(name) for name in ACTION_JOINT_NAMES]
        q = t.tensor([[resource["home_joint_positions_rad"][name] for name in names]],
                     dtype=dtype).repeat(population,1)
        origins = t.tensor([[2.*i,0.,.25*i] for i in range(population)],dtype=dtype)
        home = t.tensor(resource["home_palm_position_world_m"],dtype=dtype)[None]+origins
        home_quat = t.tensor(resource["home_palm_quaternion_xyzw"],dtype=dtype)[None].repeat(population,1)
        body_positions = t.zeros(population,2,3,dtype=dtype);body_positions[:,1] = home
        body_rotations = home_quat[:,None].repeat(1,2,1)
        data = SimpleNamespace(joint_pos=q, joint_vel=t.zeros_like(q),
            body_link_jacobian_w=t.zeros(population,1,6,43,dtype=dtype),
            body_quat_w=body_rotations,body_pos_w=body_positions)
        clock = SimpleNamespace(native=0)
        epochs = t.arange(7,7+population,dtype=t.int64)
        env = SimpleNamespace(_control_probe_enabled=True,num_envs=population,device="cpu",_palm_id=1,
            _palm_local=t.zeros(population,3,dtype=dtype),
            _palm_local_rotation=t.tensor([[0.,0.,0.,1.]],dtype=dtype).repeat(population,1),
            robot=SimpleNamespace(data=data,num_bodies=2,joint_names=names),
            cube=SimpleNamespace(data=SimpleNamespace(root_pos_w=origins+t.tensor([.35,.055,.775],dtype=dtype),
                root_quat_w=t.tensor([[0.,0.,0.,1.]],dtype=dtype).repeat(population,1))),
            _action_ids=action_ids,_tip_ids=[0,1,0],_velocity=t.linspace(.3,1.2,14,dtype=dtype),step_dt=.066,
            _targets=q.clone(),_limits=t.tensor([[[-2.,2.]]*43]*population,dtype=dtype),_last_metrics={},
            episode_length_buf=t.zeros(population,dtype=t.int64),_episode_ids=epochs,
            _grip_locked=t.zeros(population,dtype=t.bool),_unsafe=t.zeros(population,dtype=t.bool),
            _probe_overshoot=t.zeros(population,dtype=t.bool),_control_probe_profile={"name":"interface_fixture"},
            _control_preshape_contact_seen=t.zeros(population,dtype=t.bool),
            _control_preshape_first_contact_native=t.full((population,),-1,dtype=t.int64),
            _control_preshape_contact_episode_ids=epochs.clone(),
            _approach=SimpleNamespace(admitted=t.zeros(population,dtype=t.bool)),
            scene=SimpleNamespace(env_origins=origins),sim=SimpleNamespace(get_physics_step_count=lambda:clock.native))
        env._control_probe_joint_load = lambda phase:{"available":False,"sample_phase":phase}
        env._control_probe_self_pair = lambda phase:{"available":False,"sample_phase":phase}
        tree = ast.parse((ROOT/"src/g1_vla/rl_environment.py").read_text())
        cls = next(x for x in ast.walk(tree) if isinstance(x,ast.ClassDef) and x.name=="Environment")
        method = next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="control_diagnostic_state")
        scope = {"__package__":"g1_vla","torch":t,"tensor":lambda value:value,"RLEnvironmentError":RuntimeError,
            # Local point offset is exactly zero and local rotation is identity;
            # these providers cannot alter shape, dtype or device of any state.
            "quat_apply":lambda quat,offset:offset,"quat_mul":lambda quat,local:quat,
            "offset_point_jacobian":offset_point_jacobian,"preload_contact_ready":preload_contact_ready,
            "ACTION_JOINT_NAMES":ACTION_JOINT_NAMES,"config":configuration(),
            "approach_course":diagnostic_settings()["approach_course"] if course_selected else None,
            "approach_resource":resource}
        exec(compile(ast.Module(body=[method],type_ignores=[]),"<actual-control_diagnostic_state>","exec"),scope)
        return env, scope["control_diagnostic_state"], resource, clock

    def test_actual_course_state_feeds_real_planner_for_one_and_two_envs_without_shape_repair(self):
        import torch as t
        from g1_vla.rl_approach import HomeApproachPlanner
        for population in (1,2):
            for dtype in (t.float32,t.float64):
                with self.subTest(population=population,dtype=dtype):
                    env, producer, resource, clock = self.producer(population,dtype)
                    planner = HomeApproachPlanner(t,resource)
                    native_velocity = env._velocity.clone()
                    for boundary in (0,1):
                        clock.native = boundary*33;env.episode_length_buf[:] = boundary
                        state = producer(env)
                        # All values below are produced by the real method; no
                        # test-side N x14 expansion or consumer field patching.
                        for key, shape in (("q_action_rad",(population,14)),("qd_action_rad_s",(population,14)),
                                ("target_action_rad",(population,14)),("per_step_action_rad",(population,14)),
                                ("joint_limits_action_rad",(population,14,2)),("env_origin_world_m",(population,3)),
                                ("palm_position_world_m",(population,3)),("palm_quaternion_xyzw",(population,4)),
                                ("velocity_limits_arm_rad_s",(7,))):
                            self.assertEqual(tuple(state[key].shape),shape,key)
                            self.assertEqual(state[key].dtype,dtype,key)
                            self.assertEqual(state[key].device,env.robot.data.joint_pos.device,key)
                        for key, kind in (("unsafe_latched",t.bool),("approach_admitted",t.bool),
                                          ("episode_id",t.int64),("episode_control_steps",t.int64)):
                            self.assertEqual(tuple(state[key].shape),(population,),key)
                            self.assertEqual(state[key].dtype,kind,key)
                            self.assertEqual(state[key].device,env.robot.data.joint_pos.device,key)
                        self.assertIs(type(state["native_physics_step_count"]),int)
                        expected_step = t.minimum(native_velocity*.066,t.full_like(native_velocity,
                            configuration()["rl_teacher"]["action_scale_rad"]))
                        self.assertTrue(t.equal(state["per_step_action_rad"],expected_step[None].expand(population,-1)))
                        before = state["q_all_rad"].clone()
                        request,evidence = planner.actions(state,.066)
                        self.assertEqual(tuple(request.shape),(population,7))
                        self.assertEqual(request.dtype,dtype)
                        self.assertTrue(bool(t.isfinite(request).all()))
                        self.assertTrue(bool((request.abs()<=1).all()))
                        self.assertFalse(bool(evidence["failed"].any()))
                        self.assertTrue(t.equal(state["q_all_rad"],before))
                        state["per_step_action_rad"][0,0] = 99.
                        self.assertTrue(t.equal(env._velocity,native_velocity))
                        if population==2:
                            self.assertEqual(state["per_step_action_rad"][1,0],expected_step[0],
                                             "expanded rows need independent cloned storage")

    def test_actual_none_course_state_preserves_legacy_shared_limits_shape(self):
        import torch as t
        for population in (1,2):
            with self.subTest(population=population):
                env, producer, _, _ = self.producer(population,t.float32,course_selected=False)
                state = producer(env)
                self.assertEqual(tuple(state["per_step_action_rad"].shape),(14,))
                self.assertEqual(tuple(state["q_action_rad"].shape),(population,14))
                self.assertNotIn("approach_admitted",state)
                self.assertNotIn("env_origin_world_m",state)
                self.assertNotIn("velocity_limits_arm_rad_s",state)


if __name__ == "__main__":
    unittest.main()
