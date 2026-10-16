import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from g1_vla.hand_control import equivalent_position, ContactHandControl


class HandControlMathTests(unittest.TestCase):
    def test_force_torque_units_and_all_normal_nullspace(self):
        q, torque, posture = equivalent_position([0., 0.], [1., 1.], [[.05, 0.]], 9., .3, .05, .003)
        np.testing.assert_allclose(torque, [.015, 0.])
        np.testing.assert_allclose(q, [.015 / 9, .003])
        np.testing.assert_allclose(np.array([[.05, 0.]]) @ posture, 0., atol=1e-15)
        _, _, opposed = equivalent_position([0., 0.], [1., 1.], [[.05, 0.], [-.05, 0.]], 9., .3, .05, .003)
        self.assertAlmostEqual(opposed[0], 0.)  # Do not project on the cancelled net normal.

    def test_nonfinite_and_invalid_dimensions_fail_closed(self):
        for matrix in ([], [[np.nan]], [[1., 2.]]):
            with self.assertRaises(ValueError):
                equivalent_position([0.], [0.], matrix, 9., .3, .05, .003)

    def test_live_requests_preserve_state_and_unrelated_joints(self):
        names = ("right_hand_thumb_1_joint", "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_elbow_joint")
        model = SimpleNamespace(nv=4, geom_bodyid=np.array([1, 2]), jnt_range=np.array([[-2., 2.]] * 4))
        data = SimpleNamespace(qpos=np.zeros(4), qvel=np.zeros(4), time=1.)
        controller = SimpleNamespace(contract=SimpleNamespace(joint_names=names), hand_gain_scale=6.,
            bindings=[dict(qpos_id=i, qvel_id=i, joint_id=i) for i in range(4)],
            cadence=SimpleNamespace(period_s=.066))
        class API:
            @staticmethod
            def mj_jac(model, data, jp, jr, position, body):
                jp[0, 1] = .05
        opts = dict(minimum_loaded_force_n=.02, maximum_penetration_m=.003, maximum_group_force_n=1.5,
            normal_force_target_n=.3, force_feedback_gain=.25, posture_gain=.05,
            maximum_posture_delta_rad=.003, relief_step_rad=.01, maximum_recovery_drift_m=.015,
            takeup_velocity_rad_s=.1, maximum_closing_scale=1.5)
        ctl = ContactHandControl(model, data, controller, 2, opts, [-.6, .6, .6, 1.], np.zeros(4), API())
        def sample(depth=.001, force=.3):
            groups = {f: dict(normal_force_n=0., maximum_penetration_m=0.) for f in ctl.groups}
            groups["index"] = dict(normal_force_n=force, maximum_penetration_m=depth)
            return dict(groups=groups, contacts=[dict(group="index", distance_m=-depth, efc_address=0,
                force_contact_frame=[force, 0, 0, 0, 0, 0], object_geom_index=1, geom_ids=[0, 1],
                frame_rows_world=np.eye(3), position_world_m=[0, 0, 0])])
        with patch("g1_vla.hand_control.native_hand_contacts", return_value=sample()):
            target = ctl.target(1., "grasp", None)
        self.assertGreater(target[1], 0.)
        self.assertAlmostEqual(target[1], .015 / 9.)
        self.assertEqual(target[3], 0.)
        np.testing.assert_array_equal(data.qpos, np.zeros(4))
        data.qpos[1] = .2
        with patch("g1_vla.hand_control.native_hand_contacts", return_value=sample(.014, 0.)):
            target = ctl.target(1., "grasp", None)
        self.assertLess(target[1], .2)  # Deep unladen native overlap still unloads.
        self.assertEqual(ctl.rows[-1]["groups"]["index"]["mode"], "bounded_unload")
        self.assertEqual(data.qpos[1], .2)
        data.site_xpos, data.site_xmat = np.zeros((1, 3)), np.eye(3).reshape(1, 9)
        data.xpos = np.array([[0., 0, 0], [0., 0, 0], [.2, 0, 0]])
        ctl.api.mjtObj = SimpleNamespace(mjOBJ_SITE=0)
        ctl.api.mj_name2id = lambda *args: 0
        with patch("g1_vla.hand_control.native_hand_contacts", return_value=sample(0., 0.)):
            target = ctl.target(1., "hold", {"object_in_palm_frame_m": [0., 0, 0]})
        self.assertEqual(target[1], .2)
        self.assertEqual(ctl.rows[-1]["groups"]["index"]["mode"], "lost_object_recovery_inhibited")
        self.assertEqual(ctl.rows[-1]["groups"]["thumb"]["mode"], "lost_object_recovery_inhibited")
