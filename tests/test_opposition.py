"""Independent pure/mock guards for measured side closure and fixed grip locks."""

import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from g1_vla.expert import ExpertError, closure_complete, opposition_contact_status
from g1_vla.hand_control import OppositionHandControl, keyframe_hand_target
from g1_vla.teacher_trajectory import CartesianTeacherPlanner, TrajectoryError, rotation_error


HAND_NAMES = (
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
)


def side_settings():
    return dict(minimum_group_force_n=.02, maximum_abs_normal_up=.6,
                maximum_opposition_dot=-.5, maximum_penetration_m=.003)


def side_sample():
    return dict(groups={name: {"normal_force_n": .3} for name in ("thumb", "index", "middle")},
                contacts=[dict(kind="hand", group=name, loaded=True, distance_m=-.001,
                               contact_index=index, normal_on_object_world=normal)
                          for index, (name, normal) in enumerate((
                              ("thumb", [1., 0., 0.]), ("index", [-1., 0., 0.]),
                              ("middle", [-1., 0., 0.])))],
                opposition_dot_diagnostic={"index": -.8, "middle": -.7})


def hand_fixture():
    settings = dict(backend="side_opposition_preload_v1", hand_joint_names=list(HAND_NAMES),
                    preshape_rad=[0.] * 7,
                    closure_keyframes=[dict(fraction=0., joint_target_rad=[0.] * 7),
                                       dict(fraction=.5, joint_target_rad=[.2] * 7),
                                       dict(fraction=1., joint_target_rad=[.5] * 7)],
                    maximum_penetration_m=.003, maximum_group_force_n=1.5,
                    relief_step_rad=.01)
    model = SimpleNamespace(jnt_range=np.tile([-2., 2.], (7, 1)),
                            geom_bodyid=np.array([0, 1, 2, 3]), nv=7)
    data = SimpleNamespace(qpos=np.full(7, .2), time=1.)
    controller = SimpleNamespace(contract=SimpleNamespace(joint_names=HAND_NAMES),
                                 bindings=[dict(joint_id=i, qpos_id=i, qvel_id=i) for i in range(7)],
                                 previous_target=np.full(7, .4))
    class API:
        @staticmethod
        def mj_jac(model, data, jp, jr, point, body):
            jp[0, :] = 1.
    sample = dict(groups={name: dict(normal_force_n=.02, maximum_penetration_m=.001)
                          for name in ("thumb", "index", "middle")}, contacts=[])
    control = OppositionHandControl(model, data, controller, 0, settings, np.zeros(7), API())
    return control, model, data, controller, settings, sample


def mock_contact(group="thumb", *, object_index=1, normal_sign=1., distance=-.001):
    # The second proper frame reverses its normal. Reversing object index
    # simultaneously keeps the actual hand -> object normal unchanged.
    frame = np.diag([normal_sign, 1., normal_sign])
    gid = {"thumb": 1, "index": 2, "middle": 3}[group]
    return dict(group=group, distance_m=distance, object_geom_index=object_index,
                geom_ids=[gid, 0] if object_index == 1 else [0, gid],
                frame_rows_world=frame.tolist(), position_world_m=[0., 0., 0.])


class MeasuredSideContactTests(unittest.TestCase):
    def test_actual_three_sides_pass_but_each_missing_group_or_force_fails(self):
        self.assertTrue(opposition_contact_status(side_sample(), np.eye(3), side_settings())["ready"])
        for name in ("thumb", "index", "middle"):
            for mode in ("missing", "weak"):
                sample = side_sample()
                if mode == "missing":
                    sample["contacts"] = [row for row in sample["contacts"] if row["group"] != name]
                else:
                    sample["groups"][name]["normal_force_n"] = .019999
                status = opposition_contact_status(sample, np.eye(3), side_settings())
                self.assertFalse(status["checks"]["actual_side_normals"], (name, mode))

    def test_current_cube_frame_and_world_up_are_both_checked(self):
        quarter_turn = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
        # World-horizontal x is the rotated cube's top/bottom normal.
        status = opposition_contact_status(side_sample(), quarter_turn, side_settings())
        self.assertFalse(status["checks"]["actual_side_normals"])
        sample = side_sample()
        sample["contacts"][0]["normal_on_object_world"] = [0., 0., 1.]
        # World vertical is locally horizontal here, and must still fail.
        status = opposition_contact_status(sample, quarter_turn, side_settings())
        self.assertFalse(status["checks"]["actual_side_normals"])

    def test_one_loaded_top_contact_cannot_hide_in_a_group_average(self):
        sample = side_sample()
        extra = copy.deepcopy(sample["contacts"][0])
        extra.update(contact_index=4, normal_on_object_world=[0., 0., 1.])
        sample["contacts"].append(extra)
        self.assertFalse(opposition_contact_status(sample, np.eye(3), side_settings())["ready"])

    def test_unloaded_deep_mesh_overlap_still_fails_penetration(self):
        sample = side_sample()
        extra = copy.deepcopy(sample["contacts"][0])
        extra.update(contact_index=4, loaded=False, distance_m=-.003001)
        sample["contacts"].append(extra)
        status = opposition_contact_status(sample, np.eye(3), side_settings())
        self.assertTrue(status["checks"]["actual_side_normals"])
        self.assertFalse(status["checks"]["penetration_bounded"])

    def test_both_opposition_pairs_are_required_and_each_can_reject(self):
        for name in ("index", "middle"):
            sample = side_sample()
            sample["opposition_dot_diagnostic"][name] = -.49999
            self.assertFalse(opposition_contact_status(sample, np.eye(3), side_settings())["ready"])
        for mapping in ({}, {"index": -.8}, {"middle": -.8}, {"index": -.8, "middle": -.8, "extra": -1.}):
            sample = side_sample()
            sample["opposition_dot_diagnostic"] = mapping
            with self.subTest(mapping=mapping), self.assertRaises((ExpertError, ValueError)):
                opposition_contact_status(sample, np.eye(3), side_settings())

    def test_invalid_rotation_or_normal_cannot_be_physical_evidence(self):
        for matrix in (np.zeros((3, 3)), np.diag([1., 1., -1.]), np.diag([1., 1., 1.01])):
            with self.subTest(matrix=matrix), self.assertRaises((ExpertError, ValueError)):
                opposition_contact_status(side_sample(), matrix, side_settings())
        sample = side_sample()
        sample["contacts"][0]["normal_on_object_world"] = [2., 0., 0.]
        with self.assertRaises((ExpertError, ValueError)):
            opposition_contact_status(sample, np.eye(3), side_settings())


class GeometricClosureTests(unittest.TestCase):
    def test_first_touch_does_not_stop_the_declared_geometric_closure(self):
        control, model, data, controller, settings, sample = hand_fixture()
        before = copy.deepcopy((data, controller.previous_target, model.jnt_range))
        with patch("g1_vla.hand_control.native_hand_contacts", return_value=sample):
            first = control.target(.5, "contact", None)
            later = control.target(1., "contact", None)
        np.testing.assert_allclose(first, .2)
        np.testing.assert_allclose(later, .5)
        self.assertTrue(all(row["mode"] == "declared_geometric_closure" for row in control.rows[-1]["groups"].values()))
        np.testing.assert_array_equal(data.qpos, before[0].qpos)
        np.testing.assert_array_equal(controller.previous_target, before[1])
        np.testing.assert_array_equal(model.jnt_range, before[2])
        self.assertEqual(data.time, before[0].time)

    def test_unload_normal_sign_is_correct_for_either_object_geom_index(self):
        for index, sign in ((1, 1.), (0, -1.)):
            control, _, data, _, _, sample = hand_fixture()
            sample["groups"]["thumb"]["normal_force_n"] = 1.50001
            sample["contacts"] = [mock_contact(object_index=index, normal_sign=sign)]
            with patch("g1_vla.hand_control.native_hand_contacts", return_value=sample):
                requested = control.target(1., "grasp", None)
            np.testing.assert_allclose(requested[:3], data.qpos[:3] - .01 / math.sqrt(3))
            self.assertEqual(control.rows[-1]["groups"]["thumb"]["mode"], "native_normal_bounded_unload")

    def test_deep_unloaded_overlap_can_unload_and_cancelled_normals_hold_applied(self):
        control, _, data, controller, _, sample = hand_fixture()
        sample["groups"]["thumb"].update(normal_force_n=0., maximum_penetration_m=.003001)
        sample["contacts"] = [mock_contact(distance=-.003001)]
        with patch("g1_vla.hand_control.native_hand_contacts", return_value=sample):
            requested = control.target(1., "grasp", None)
        self.assertTrue(np.all(requested[:3] < data.qpos[:3]))
        sample["contacts"].append(mock_contact(normal_sign=-1., distance=-.003001))
        with patch("g1_vla.hand_control.native_hand_contacts", return_value=sample):
            cancelled = control.target(1., "grasp", None)
        np.testing.assert_array_equal(cancelled[:3], controller.previous_target[:3])
        self.assertEqual(control.rows[-1]["groups"]["thumb"]["mode"], "unload_direction_cancelled_hold_applied")

    def test_mechanical_bounding_changes_only_the_request_copy(self):
        control, _, data, controller, _, sample = hand_fixture()
        data.qpos[:3] = 1.999
        sample["groups"]["thumb"]["normal_force_n"] = 1.6
        sample["contacts"] = [mock_contact(normal_sign=-1.)]
        before_actual, before_previous = data.qpos.copy(), controller.previous_target.copy()
        with patch("g1_vla.hand_control.native_hand_contacts", return_value=sample):
            requested = control.target(1., "grasp", None)
        np.testing.assert_array_equal(requested[:3], 2.)
        np.testing.assert_array_equal(data.qpos, before_actual)
        np.testing.assert_array_equal(controller.previous_target, before_previous)

    def test_keyframe_interval_and_complete_closing_time_are_not_skipped(self):
        control, _, _, _, settings, _ = hand_fixture()
        np.testing.assert_array_equal(keyframe_hand_target(settings["closure_keyframes"], 0.), 0.)
        np.testing.assert_array_equal(keyframe_hand_target(settings["closure_keyframes"], 1.), .5)
        for fraction in (-.01, 1.01, float("nan")):
            with self.assertRaises(ValueError):
                keyframe_hand_target(settings["closure_keyframes"], fraction)
        self.assertFalse(closure_complete(1.599, 1., {"close_duration_s": .6}))
        self.assertTrue(closure_complete(1.6, 1., {"close_duration_s": .6}))


class FixedPreprobeReferenceTests(unittest.TestCase):
    def test_latch_uses_actual_relative_pose_and_is_not_a_live_alias(self):
        planner = CartesianTeacherPlanner.__new__(CartesianTeacherPlanner)
        planner.object_body = 0
        palm_R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        pose = dict(palm=np.array([.3, .1, .8]), C=np.array([.3, .15, .8]),
                    R=palm_R, arm=np.arange(7.) / 10)
        planner.actual_pose = lambda data: pose
        data = SimpleNamespace(time=2., xpos=np.array([[.31, .14, .8]]), xmat=np.eye(3).reshape(1, 9))
        lock = planner.latch_grip(data)
        np.testing.assert_allclose(lock["object_in_palm_frame_m"], [.04, -.01, 0.], atol=1e-15)
        np.testing.assert_array_equal(lock["object_rotation_in_palm_frame"], palm_R.T)
        old = copy.deepcopy(lock)
        data.xpos[0, :] += 1.
        pose["C"][:] += 1.
        self.assertEqual(lock, old)
        # Main lift must use the original lock origin: the 6mm probe is
        # included in, rather than added to, the original 22cm displacement.
        main_goal = np.asarray(lock["C"]) + [0., 0., .22]
        probe_goal = np.asarray(lock["C"]) + [0., 0., .006]
        self.assertAlmostEqual(main_goal[2] - probe_goal[2], .214)

    def test_raw_relative_rotation_does_not_use_cube_symmetry(self):
        quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        self.assertAlmostEqual(rotation_error(np.eye(3), quarter_turn), math.pi / 2)
        self.assertGreater(rotation_error(np.eye(3), quarter_turn), .15)


if __name__ == "__main__":
    unittest.main()
