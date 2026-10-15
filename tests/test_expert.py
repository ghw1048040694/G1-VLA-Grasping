"""Physical-success accounting rejects launch, contact loss and unsafe holds."""
import copy
import ast
import inspect
import unittest
from types import SimpleNamespace

from g1_vla.expert import (ExpertError, effective_compensation, hand_target,
                           physical_gate, phase_tracking_reached, validate_teacher,
                           joint_limit_excesses, state_is_finite, green_target_from_position,
                           pose_tracking_reached, contact_closure_ready, closure_complete,
                           _robot_self_contacts, collision_distance_residual,
                           arm_torso_collision_pairs, run_teacher_episode,
                           green_contact_penetration_evidence, bounded_planning_seed,
                           interpolate)


PROFILE = {
    "profile": "green_right_recovery_v1", "grasp_yaw_deg": 45.,
    "closure_multiplier": 1.2, "finger_scale": [1., 1., 1.],
    "contact_compensation_m": [-.015, -.03, -.055],
    "pregrasp_height_m": .12, "pregrasp_lateral_m": .015,
    "high_pregrasp_height_m": .28, "home_settle_s": .5,
    "high_approach_s": 1.8, "pregrasp_s": .8,
    "contact_approach_s": .9, "close_duration_s": .6,
    "grasp_hold_s": .8, "lift_height_m": .22,
    "lift_duration_s": 2., "final_hold_s": 1.,
    "adaptive_regrasp": True, "ik_max_error_m": .08, "record_rgb": False,
    "palm_position_tolerance_m": .03, "joint_target_tolerance_rad": .06, "phase_timeout_scale": 3.,
    "ik_collision_margin_m": .002, "ik_collision_weight": 5.,
    "grasp_center_in_palm_m": [0., .05, 0.], "ik_orientation_weight": .04,
}


def sample(start, duration=.1, *, contact=True, height=.12, assisted=False, safe=True):
    return {"start_s": start, "end_s": start + duration, "contact": contact,
            "lift_height_m": height, "assisted": assisted, "finite": True, "joint_safe": safe}


class TeacherTests(unittest.TestCase):
    def gate(self, rows):
        return physical_gate(rows, lift_height_m=.1, hold_duration_s=.5)

    def test_future_phase_ik_is_lazy_and_video_exists_before_first_control(self):
        source = inspect.getsource(run_teacher_episode)
        function = ast.parse(source).body[0]
        parent = {}
        for node in ast.walk(function):
            for child in ast.iter_child_nodes(node):
                parent[child] = node
        future_calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == "pose"]
        self.assertEqual(len(future_calls), 1)
        owner = parent[future_calls[0]]
        while not isinstance(owner, ast.FunctionDef):
            owner = parent[owner]
        self.assertEqual(owner.name, "refresh_phase_pose")
        control_loop = next(node for node in ast.walk(function) if isinstance(node, ast.While))
        video_constructor = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                                 and isinstance(node.func, ast.Name)
                                 and node.func.id == "TeacherDiagnosticVideo")
        self.assertLess(video_constructor.lineno, control_loop.lineno)
        first_control = next(node for node in ast.walk(control_loop) if isinstance(node, ast.Call)
                             and isinstance(node.func, ast.Attribute)
                             and node.func.attr == "execute_target")
        entries = [node for node in ast.walk(function) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == "refresh_phase_pose"]
        self.assertTrue(entries)
        self.assertTrue(all(node.lineno > first_control.lineno for node in entries))

    def test_launch_without_contact_cannot_pass(self):
        rows = [sample(i * .1, contact=False, height=.4) for i in range(12)]
        self.assertFalse(self.gate(rows)["physical_grasp_passed"])

    def test_contact_loss_resets_hold(self):
        rows = [sample(i * .1) for i in range(4)]
        rows += [sample(.4, contact=False)]
        rows += [sample(.5 + i * .1) for i in range(4)]
        self.assertAlmostEqual(self.gate(rows)["sustained_lift_hold_s"], .4)
        self.assertFalse(self.gate(rows)["physical_grasp_passed"])

    def test_assisted_or_unsafe_samples_never_extend_hold(self):
        for change in ({"assisted": True}, {"safe": False}):
            rows = [sample(i * .1, **change) for i in range(8)]
            self.assertFalse(self.gate(rows)["physical_grasp_passed"])

    def test_missing_time_cannot_bridge_hold(self):
        rows = [sample(0), sample(.1), sample(.2), sample(.7), sample(.8), sample(.9)]
        self.assertFalse(self.gate(rows)["physical_grasp_passed"])

    def test_contiguous_true_physical_hold_passes(self):
        self.assertTrue(self.gate([sample(i * .1) for i in range(5)])["physical_grasp_passed"])

    def test_tracking_and_contact_required_before_phase_advance(self):
        tracking = {"palm_position_error_m": .01, "requested_joint_error_rad": .02}
        values = dict(contact=True, minimum_lift_m=.12, measured_hold_s=.6,
                      lift_height_m=.1, hold_duration_s=.5)
        self.assertTrue(phase_tracking_reached("hold", tracking, PROFILE, **values))
        values["contact"] = False
        self.assertFalse(phase_tracking_reached("contact", tracking, PROFILE, **values))
        values["contact"] = True
        tracking["palm_position_error_m"] = .08
        self.assertFalse(phase_tracking_reached("high_approach", tracking, PROFILE, **values))

    def test_closure_waits_for_completed_approach_and_both_measured_pose_gates(self):
        tracking = {"palm_position_error_m": .01, "requested_joint_error_rad": .02}
        self.assertFalse(contact_closure_ready("contact", .899, tracking, PROFILE))
        self.assertFalse(contact_closure_ready("pregrasp", 3., tracking, PROFILE))
        self.assertFalse(contact_closure_ready("contact", 1., {}, PROFILE))
        self.assertTrue(contact_closure_ready("contact", .9, tracking, PROFILE))
        for key, bad in (("palm_position_error_m", .03001),
                         ("requested_joint_error_rad", .06001)):
            changed = dict(tracking, **{key: bad})
            self.assertFalse(pose_tracking_reached(changed, PROFILE))
            self.assertFalse(contact_closure_ready("contact", 2., changed, PROFILE))

    def test_first_contact_cannot_skip_the_full_closing_interval(self):
        self.assertFalse(closure_complete(10., None, PROFILE))
        self.assertFalse(closure_complete(.599, 0., PROFILE))
        self.assertTrue(closure_complete(.6, 0., PROFILE))
        self.assertTrue(closure_complete(1.6, 1., PROFILE))

    def test_robot_collision_evidence_keeps_penetration_and_excludes_object_and_world(self):
        names = {1: "torso_link", 2: "right_shoulder_yaw_link", 3: "green_cube"}
        api = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1),
                              mj_id2name=lambda _model, _kind, body: names.get(body))
        model = SimpleNamespace(geom_bodyid=[0, 1, 2, 3])
        contacts = [SimpleNamespace(geom1=1, geom2=2, dist=-.002),
                    SimpleNamespace(geom1=2, geom2=1, dist=-.004),
                    SimpleNamespace(geom1=2, geom2=3, dist=-.02),
                    SimpleNamespace(geom1=2, geom2=0, dist=-.03),
                    SimpleNamespace(geom1=1, geom2=2, dist=.001)]
        data = SimpleNamespace(ncon=len(contacts), contact=contacts)
        self.assertEqual(_robot_self_contacts(model, data, 3, api),
                         {"right_shoulder_yaw_link :: torso_link": .004})

    def test_green_overlap_diagnostic_keeps_mesh_and_capsule_and_table_separate(self):
        body_names = {0: "world", 1: "green_cube", 2: "right_hand_index_1_link", 3: "torso_link"}
        geom_names = {0: "green_cube_geom", 1: "source_finger_mesh", 2: "finger_capsule",
                      3: "calibration_table", 4: "torso_geom"}
        api = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1, mjOBJ_GEOM=2),
                              mj_id2name=lambda _model, kind, item:
                              (body_names if kind == 1 else geom_names).get(item))
        model = SimpleNamespace(geom_bodyid=[1, 2, 2, 0, 3])
        contacts = [SimpleNamespace(geom1=0, geom2=1, dist=-.035),
                    SimpleNamespace(geom1=2, geom2=0, dist=-.014),
                    SimpleNamespace(geom1=0, geom2=3, dist=-.0097),
                    SimpleNamespace(geom1=4, geom2=0, dist=-.04)]
        result = green_contact_penetration_evidence(model,
                    SimpleNamespace(ncon=len(contacts), contact=contacts), 1, api)
        self.assertAlmostEqual(result["hand"]["penetration_m"], .035)
        self.assertEqual(result["hand"]["pair"]["geom_names"], ["green_cube_geom", "source_finger_mesh"])
        self.assertAlmostEqual(result["table"]["penetration_m"], .0097)
        self.assertEqual(result["table"]["pair"]["geom_ids"], [0, 3])

    def test_collision_distance_residual_keeps_length_and_repels_approaching_geometry(self):
        residual = collision_distance_residual([.05, .002, .001, -.003], .002, 5.)
        self.assertEqual(len(residual), 4)
        self.assertEqual(residual[:2], (0., 0.))
        self.assertAlmostEqual(residual[2], -.005)
        self.assertAlmostEqual(residual[3], -.025)

    def test_collision_pair_contract_preserves_source_masks_and_exclusions(self):
        body_ids = {"torso_link": 1, "right_shoulder_roll_link": 3,
                    "right_shoulder_yaw_link": 4}
        api = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1),
                              mj_name2id=lambda _model, _kind, name: body_ids[name])
        model = SimpleNamespace(ngeom=5, geom_bodyid=[1, 1, 1, 3, 4],
                                geom_contype=[1] * 5, geom_conaffinity=[1] * 5,
                                body_weldid=list(range(5)), body_parentid=[0, 0, 1, 2, 3],
                                exclude_signature=[])
        pairs = arm_torso_collision_pairs(model, mujoco_api=api)
        self.assertEqual(pairs, ((0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4)))
        model.geom_contype[0] = model.geom_conaffinity[0] = 0
        with self.assertRaisesRegex(ExpertError, "6 active"):
            arm_torso_collision_pairs(model, mujoco_api=api)
        model.geom_contype[0] = model.geom_conaffinity[0] = 1
        model.exclude_signature = [(1 << 16) + 4]
        with self.assertRaisesRegex(ExpertError, "6 active"):
            arm_torso_collision_pairs(model, mujoco_api=api)

    def test_cavity_pair_contract_adds_only_the_observed_source_elbow_pairs(self):
        body_ids = {"torso_link": 1, "right_shoulder_roll_link": 3,
                    "right_shoulder_yaw_link": 4, "right_elbow_link": 5}
        api = SimpleNamespace(mjtObj=SimpleNamespace(mjOBJ_BODY=1),
                              mj_name2id=lambda _model, _kind, name: body_ids[name])
        model = SimpleNamespace(ngeom=6, geom_bodyid=[1, 1, 1, 3, 4, 5],
                                geom_contype=[1] * 6, geom_conaffinity=[1] * 6,
                                body_weldid=list(range(6)), body_parentid=[0, 0, 1, 2, 3, 4],
                                exclude_signature=[])
        original = arm_torso_collision_pairs(model, mujoco_api=api)
        extended = arm_torso_collision_pairs(model, include_elbow=True, mujoco_api=api)
        self.assertEqual(len(original), 6)
        self.assertEqual(len(extended), 9)
        self.assertEqual(set(extended) - set(original), {(0, 5), (1, 5), (2, 5)})

    def test_mechanical_overrun_remains_visible_inside_declared_tolerance(self):
        model = SimpleNamespace(jnt_limited=[True, True], jnt_range=[[-1., 1.], [-1., 1.]])
        data = SimpleNamespace(qpos=[1.002, -1.004])
        bindings = {"right_hand": {"joint_id": 0, "qpos_id": 0},
                    "left_frozen_hand": {"joint_id": 1, "qpos_id": 1}}
        excess = joint_limit_excesses(model, data, bindings)
        self.assertAlmostEqual(excess["right_hand"], .002)
        self.assertAlmostEqual(excess["left_frozen_hand"], .004)
        self.assertFalse(all(value <= .003 for value in excess.values()))

    def test_planning_seed_bounds_requests_without_clamping_actual_or_safety(self):
        import numpy as np
        actual = np.asarray([2.0948735, -.004, .3])
        original = actual.copy()
        low, high = np.asarray([-1.0472, 0., -1.]), np.asarray([2.0944, 1.5708, 1.])
        seed = bounded_planning_seed(actual, low, high)
        self.assertTrue(np.array_equal(actual, original))
        self.assertFalse(np.shares_memory(seed, actual))
        self.assertEqual(seed.tolist(), [2.0944, 0., .3])
        for alpha in (0., .25, .5, 1.):
            requested = np.asarray(interpolate(seed, [1., .6, 0.], alpha))
            self.assertTrue(np.all(requested >= low) and np.all(requested <= high))
        model = SimpleNamespace(jnt_limited=[True] * 3, jnt_range=list(zip(low, high)))
        data = SimpleNamespace(qpos=actual)
        bindings = {str(i): {"joint_id": i, "qpos_id": i} for i in range(3)}
        excess = joint_limit_excesses(model, data, bindings)
        self.assertAlmostEqual(excess["0"], .0004735)
        self.assertGreater(excess["1"], .003)
        self.assertFalse(all(value <= .003 for value in excess.values()))

    def test_final_numeric_gate_checks_force_and_time(self):
        data = SimpleNamespace(time=0., qpos=[0.], qvel=[0.], qacc=[0.],
                               ctrl=[0.], actuator_force=[0.], qfrc_bias=[0.])
        self.assertTrue(state_is_finite(data))
        data.actuator_force[0] = float("inf")
        self.assertFalse(state_is_finite(data))
        data.actuator_force[0] = 0.
        data.time = float("nan")
        self.assertFalse(state_is_finite(data))

    def test_contact_goal_follows_settled_and_displaced_object(self):
        spawn = [.31239584, .01742009, .795]
        settled = [.31239584, .01742009, .7744281478732685]
        shifted = [.29524443138555484, .036276717469801975, .7831437958332935]
        first = green_target_from_position(spawn, PROFILE, compensate=True)
        live = green_target_from_position(settled, PROFILE, compensate=True)
        moved = green_target_from_position(shifted, PROFILE, compensate=True)
        self.assertAlmostEqual(live[2], .7194281478732685)
        self.assertAlmostEqual(first[2] - live[2], .0205718521267315)
        self.assertAlmostEqual(moved[0], shifted[0] + .005)
        self.assertAlmostEqual(moved[1], shifted[1])
        self.assertEqual(spawn, [.31239584, .01742009, .795])

    def test_source_right_compensation_flips_once(self):
        self.assertEqual(effective_compensation(PROFILE), (-.015, .03, -.055))
        self.assertEqual(effective_compensation(PROFILE, "left"), (-.015, -.03, -.055))

    def test_geometry_feedback_target_aligns_finger_center_without_rewriting_legacy_profile(self):
        profile = copy.deepcopy(PROFILE)
        profile["profile"] = "green_right_geometry_feedback_v2"
        profile["contact_compensation_m"][2] = 0.
        current_object = [.35, .03, .7744281478732685]
        target = green_target_from_position(current_object, profile, compensate=True)
        self.assertAlmostEqual(target[0], .355)
        self.assertAlmostEqual(target[1], .03)
        self.assertEqual(target[2], current_object[2])
        self.assertEqual(PROFILE["contact_compensation_m"][2], -.055)

    def test_cavity_target_uses_one_hand_local_offset_for_all_phase_heights(self):
        profile = copy.deepcopy(PROFILE)
        profile.update(profile="green_right_cavity_feedback_v3", grasp_yaw_deg=0.,
                       contact_compensation_m=[0., 0., 0.])
        position = [.35, .03, .7744281478732685]
        for height, lateral in ((.28, 0.), (.12, -.015), (0., 0.), (.22, 0.)):
            target = green_target_from_position(position, profile, height=height, lateral=lateral)
            self.assertAlmostEqual(target[0], position[0])
            self.assertAlmostEqual(target[1], position[1] - .05 + lateral)
            self.assertAlmostEqual(target[2], position[2] + height)
            self.assertEqual(target, green_target_from_position(position, profile, height=height,
                                                               lateral=lateral, compensate=True))
        profile["grasp_yaw_deg"] = 90.
        turned = green_target_from_position(position, profile)
        self.assertAlmostEqual(turned[0], position[0] + .05)
        self.assertAlmostEqual(turned[1], position[1])
        self.assertAlmostEqual(turned[2], position[2])

    def test_cavity_profile_cannot_silently_stack_legacy_compensation(self):
        profile = copy.deepcopy(PROFILE)
        profile.update(profile="green_right_cavity_feedback_v3", grasp_yaw_deg=0.,
                       contact_compensation_m=[0., 0., 0.])
        config = {"train": {"enabled": False}, "wm": {"enabled": False},
                  "task": {"mode": "grasp_lift", "hand": "right", "object_ids": ["green_cube"]},
                  "scene": {"assisted_grasp": False}, "eval": {"assisted_grasp": False, "hold_duration_s": .5},
                  "expert": {"teacher": profile}}
        self.assertEqual(validate_teacher(config)["grasp_center_in_palm_m"], [0., .05, 0.])
        profile["contact_compensation_m"][1] = .001
        with self.assertRaisesRegex(ExpertError, "zero legacy"):
            validate_teacher(config)

    def test_canonical_hand_profile_is_named_and_signed(self):
        self.assertAlmostEqual(hand_target("right_hand_thumb_2_joint", PROFILE), -.72)
        self.assertAlmostEqual(hand_target("right_hand_index_1_joint", PROFILE), .72)
        with self.assertRaises(ExpertError):
            hand_target("right_elbow_joint", PROFILE)

    def test_teacher_rejects_training_and_wrong_task(self):
        config = {"train": {"enabled": False}, "wm": {"enabled": False},
                  "task": {"mode": "grasp_lift", "hand": "right", "object_ids": ["green_cube"]},
                  "scene": {"assisted_grasp": False}, "eval": {"assisted_grasp": False, "hold_duration_s": .5},
                  "expert": {"teacher": copy.deepcopy(PROFILE)}}
        self.assertEqual(validate_teacher(config)["profile"], PROFILE["profile"])
        config["train"]["enabled"] = True
        with self.assertRaisesRegex(ExpertError, "train.enabled=false"):
            validate_teacher(config)
        config["train"]["enabled"] = False
        config["task"]["hand"] = "left"
        with self.assertRaisesRegex(ExpertError, "green_cube/right"):
            validate_teacher(config)


if __name__ == "__main__":
    unittest.main()
