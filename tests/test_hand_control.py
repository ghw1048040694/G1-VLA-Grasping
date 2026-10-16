import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from g1_vla.hand_control import equivalent_position, pressure_position_step, ContactHandControl


class HandControlMathTests(unittest.TestCase):
    def test_force_torque_units_and_all_normal_nullspace(self):
        q, torque, posture = equivalent_position([0., 0.], [1., 1.], [[.05, 0.]], 9., .3, .05, .003)
        np.testing.assert_allclose(torque, [.015, 0.])
        np.testing.assert_allclose(q, [.015 / 9, .003])
        np.testing.assert_allclose(np.array([[.05, 0.]]) @ posture, 0., atol=1e-15)
        _, _, opposed = equivalent_position([0., 0.], [1., 1.], [[.05, 0.], [-.05, 0.]], 9., .3, .05, .003)
        self.assertAlmostEqual(opposed[0], 0.)

    def test_nonfinite_and_invalid_dimensions_fail_closed(self):
        for matrix in ([], [[np.nan]], [[1., 2.]]):
            with self.assertRaises(ValueError):
                equivalent_position([0.], [0.], matrix, 9., .3, .05, .003)
            with self.assertRaises(ValueError):
                pressure_position_step([0.], [0.], matrix, .2, .6, .1, .066, .005, .05, .003)
        with self.assertRaises(ValueError):
            pressure_position_step([0.], [0.], [[.05]], .2, .6, .1, 0., .005, .05, .003)

    def test_pressure_error_sign_units_and_per_joint_capping(self):
        for measured in (.2, .6, 1.0):
            step = pressure_position_step([.3, -.2], [.3, -.2], [[.03, -.04]], measured, .6, .1, .066, .005, .05, .003)
            expected = np.asarray([.6, -.8]) * .1 * (.6 - measured) * .066
            np.testing.assert_allclose(step['pressure_increment'], expected)
            np.testing.assert_allclose(step['requested'], [.3, -.2] + expected)
        capped = pressure_position_step([.3, -.2], [.3, -.2], [[.03, -.04]], 0., 10., 1., .066, .005, .05, .003)
        np.testing.assert_array_equal(capped['pressure_increment'], [.005, -.005])
        self.assertTrue((np.abs(capped['uncapped_pressure_increment']) > .005).all())

    def test_cancelled_net_normal_has_no_pressure_and_full_matrix_nullspace(self):
        step = pressure_position_step([.2, .1], [1., 1.], [[.05, 0.], [-.05, 0.]], .2, .6, .1, .066, .005, .05, .003)
        np.testing.assert_array_equal(step['pressure_increment'], [0., 0.])
        self.assertFalse(step['normal_direction_usable'])
        self.assertAlmostEqual(step['posture_increment'][0], 0.)
        self.assertAlmostEqual(step['posture_increment'][1], .003)
        np.testing.assert_allclose(np.asarray([[.05, 0.], [-.05, 0.]]) @ step['posture_increment'], 0., atol=1e-15)
        fullrank = pressure_position_step([.2, .1], [1., 1.], [[.05, 0.], [0., .02]], .2, .6, .1, .066, .005, .05, .003)
        np.testing.assert_allclose(fullrank['posture_increment'], 0., atol=1e-15)

    def test_pure_pressure_inputs_are_not_modified(self):
        previous, nominal, matrix = np.array([.2, .1]), np.array([.6, .7]), np.array([[.05, .02]])
        copies = [x.copy() for x in (previous, nominal, matrix)]
        pressure_position_step(previous, nominal, matrix, .2, .6, .1, .066, .005, .05, .003)
        for value, before in zip((previous, nominal, matrix), copies):
            np.testing.assert_array_equal(value, before)


def fixture():
    names = ('right_hand_thumb_1_joint', 'right_hand_index_0_joint', 'right_hand_middle_0_joint', 'right_elbow_joint')
    model = SimpleNamespace(nv=4, geom_bodyid=np.array([1, 2]), jnt_range=np.array([[-2., 2.]] * 4))
    data = SimpleNamespace(qpos=np.zeros(4), qvel=np.zeros(4), time=1., site_xpos=np.zeros((1, 3)),
                           site_xmat=np.eye(3).reshape(1, 9), xpos=np.zeros((3, 3)))
    controller = SimpleNamespace(contract=SimpleNamespace(joint_names=names), hand_gain_scale=6.,
        previous_target=(-.1, .2, .3, .4), bindings=[dict(qpos_id=i, qvel_id=i, joint_id=i) for i in range(4)],
        cadence=SimpleNamespace(period_s=.066))
    class API:
        mjtObj = SimpleNamespace(mjOBJ_SITE=0)
        @staticmethod
        def mj_name2id(*args): return 0
        @staticmethod
        def mj_jac(model, data, jp, jr, position, body): jp[0, 1] = .05
    opts = dict(backend='contact_normal_position_v2', minimum_loaded_force_n=.02, maximum_penetration_m=.003,
        maximum_group_force_n=1.5, normal_force_target_n=.6, normal_adjustment_rate_rad_n_s=.1,
        maximum_force_step_rad=.005, posture_gain=.05, maximum_posture_delta_rad=.003, relief_step_rad=.01,
        maximum_recovery_drift_m=.015, takeup_velocity_rad_s=.1, maximum_closing_scale=1.5)
    ctl = ContactHandControl(model, data, controller, 2, opts, [-.6, .6, .6, 1.], np.zeros(4), API())
    return ctl, model, data, controller


def sample(depth=.001, force=.3):
    groups = {finger: dict(normal_force_n=0., maximum_penetration_m=0.) for finger in ('thumb', 'index', 'middle', 'palm')}
    groups['index'] = dict(normal_force_n=force, maximum_penetration_m=depth)
    return dict(groups=groups, contacts=[dict(group='index', distance_m=-depth, efc_address=0,
        force_contact_frame=[force, 0, 0, 0, 0, 0], object_geom_index=1, geom_ids=[0, 1],
        frame_rows_world=np.eye(3), position_world_m=[0, 0, 0])])


class HandControlFeedbackTests(unittest.TestCase):
    def test_loaded_reference_does_not_follow_yielding_actual_and_does_not_wind_up_unapplied_requests(self):
        ctl, model, data, controller = fixture()
        before_q = data.qpos.copy(); before_prev = controller.previous_target
        with patch('g1_vla.hand_control.native_hand_contacts', return_value=sample()):
            first = ctl.target(1., 'grasp', None)
            self.assertAlmostEqual(first[1], .2 + .1 * (.6 - .3) * .066)
            data.qpos[1] = -.01  # Actual yielding cannot reset the accumulated applied reference.
            repeat = ctl.target(1., 'grasp', None)
            self.assertAlmostEqual(repeat[1], first[1])  # No acknowledgement, no integration.
            controller.previous_target = (-.1, .2005, .3, .4)  # Shared limiter applied only part of the last request.
            acknowledged = ctl.target(1., 'grasp', None)
        self.assertAlmostEqual(acknowledged[1], .2005 + .1 * (.6 - .3) * .066)
        self.assertEqual(first[3], 0.)
        self.assertEqual(before_prev, (-.1, .2, .3, .4))
        self.assertEqual(data.qpos[1], -.01)
        detail = ctl.rows[-1]['groups']['index']
        self.assertEqual(detail['previous_applied_q'], [.2005])
        self.assertEqual(detail['normal_jacobian_rows_m_per_rad'], [[.05]])
        self.assertEqual(detail['effective_force_goal_n'], .6)
        np.testing.assert_allclose(detail['theoretical_equivalent_normal_torque_nm'], [.03])
        self.assertIn('diagnostic only', detail['equivalent_torque_scope'])
        self.assertEqual(ctl.rows[-1]['reference_origin'], 'controller.previous_target_last_actually_applied')
        np.testing.assert_array_equal(data.qpos[[0, 2, 3]], before_q[[0, 2, 3]])

    def test_actual_previous_request_is_required_and_legacy_backend_not_silently_reinterpreted(self):
        ctl, model, data, controller = fixture()
        controller.previous_target = (0., np.nan, 0., 0.)
        with self.assertRaises(ValueError): ctl.target(1., 'grasp', None)
        opts = dict(ctl.settings, backend='contact_normal_position_v1')
        with self.assertRaises(ValueError):
            ContactHandControl(model, data, controller, 2, opts, ctl.nominal, ctl.baseline, ctl.api)

    def test_depth_unload_without_force_and_overload_unload_precede_integration(self):
        ctl, model, data, controller = fixture()
        data.qpos[1] = .2
        for contact in (sample(.014, 0.), sample(.001, 1.6)):
            with patch('g1_vla.hand_control.native_hand_contacts', return_value=contact):
                requested = ctl.target(1., 'grasp', None)
            self.assertAlmostEqual(requested[1], .19)
            self.assertEqual(ctl.rows[-1]['groups']['index']['mode'], 'bounded_unload')
            self.assertEqual(ctl.rows[-1]['groups']['index']['pressure_increment_rad'], [0.])
        self.assertEqual(data.qpos[1], .2)
        self.assertEqual(controller.previous_target, (-.1, .2, .3, .4))

    def test_lost_takeup_uses_applied_reference_only_with_fixed_near_grip(self):
        ctl, model, data, controller = fixture()
        ctl.touched['index'] = True
        data.qpos[1] = .05
        with patch('g1_vla.hand_control.native_hand_contacts', return_value=sample(0., 0.)):
            waiting = ctl.target(1., 'contact', None)
            self.assertAlmostEqual(waiting[1], .2)
            self.assertEqual(ctl.rows[-1]['groups']['index']['mode'], 'lost_finger_takeup_requires_fixed_grip')
            requested = ctl.target(1., 'hold', {'object_in_palm_frame_m': [0., 0., 0.]})
        self.assertAlmostEqual(requested[1], .2 + .1 * .066)
        self.assertEqual(ctl.rows[-1]['groups']['index']['mode'], 'bounded_lost_finger_takeup')
        self.assertEqual(data.qpos[1], .05)

    def test_far_drift_guard_precedes_even_never_touched_search(self):
        ctl, model, data, controller = fixture()
        data.qpos[:3] = [-.03, .1, .2]
        data.xpos[2] = [.015001, 0., 0.]
        before = data.qpos.copy()
        with patch('g1_vla.hand_control.native_hand_contacts', return_value=sample(0., 0.)):
            requested = ctl.target(1., 'hold', {'object_in_palm_frame_m': [0., 0., 0.]})
        np.testing.assert_array_equal(requested[:3], before[:3])
        self.assertTrue(all(group['mode'] == 'lost_object_recovery_inhibited' for group in ctl.rows[-1]['groups'].values()))
        np.testing.assert_array_equal(data.qpos, before)

    def test_raw_request_and_bounded_request_are_separate_without_state_mutation(self):
        ctl, model, data, controller = fixture()
        controller.previous_target = (-.1, .8999, .3, .4)
        with patch('g1_vla.hand_control.native_hand_contacts', return_value=sample(0.001, .1)):
            requested = ctl.target(1., 'lift', None)
        detail = ctl.rows[-1]['groups']['index']
        self.assertGreater(detail['raw_requested_q'][0], .9)
        self.assertAlmostEqual(detail['bounded_requested_q'][0], .9)
        self.assertAlmostEqual(requested[1], .9)
        self.assertEqual(controller.previous_target[1], .8999)
        np.testing.assert_array_equal(data.qpos, np.zeros(4))
