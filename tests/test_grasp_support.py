import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np

from g1_vla.contracts import canonical_sha256
from g1_vla.grasp_support import (FORCE_SCOPE, SupportWindow, box_corners_world,
                                  contact_wrench_on_object, native_grasp_support)


def settings():
    return dict(minimum_bottom_clearance_m=.002, maximum_table_support_fraction=.05,
                hand_weight_ratio_min=.8, hand_weight_ratio_max=1.2, max_linear_speed_m_s=.02,
                max_angular_speed_rad_s=.2, max_grip_drift_m=.003, stable_window_s=.198,
                required_groups=['thumb', 'index', 'middle'])


def fixture():
    body_names = ['world', 'green_cube', 'right_hand_thumb_1_link', 'right_hand_index_1_link',
                  'right_hand_middle_1_link', 'right_wrist_yaw_link', 'unrelated_body']
    geom_names = ['green_cube_geom', 'calibration_table', 'thumb_mesh', 'thumb_capsule', 'index_mesh',
                  'index_capsule', 'middle_mesh', 'middle_capsule', 'right_palm_contact_pad', 'unrelated_geom']
    model = SimpleNamespace(opt=SimpleNamespace(gravity=np.array([0., 0., -9.81])),
                            body_mass=np.array([0., .08, .1, .1, .1, .1, .1]),
                            body_geomadr=np.array([1, 0, 2, 4, 6, 8, 9]), body_geomnum=np.array([1, 1, 2, 2, 2, 1, 1]),
                            body_jntadr=np.array([-1, 0, 1, 2, 3, 4, 5]), body_jntnum=np.array([0, 1, 1, 1, 1, 1, 1]),
                            jnt_type=np.array([0, 3, 3, 3, 3, 3]), geom_type=np.array([6] * 10),
                            geom_bodyid=np.array([1, 0, 2, 2, 3, 3, 4, 4, 5, 6]),
                            geom_size=np.asarray([[.025, .025, .025], [.42, .55, .03]] + [[.01] * 3] * 8),
                            geom_contype=np.array([5] + [2] * 9), geom_conaffinity=np.array([3] + [4] * 9),
                            body_weldid=np.array([0, 1, 2, 3, 4, 5, 6]))
    data = SimpleNamespace(time=.002, qpos=np.zeros(7), qvel=np.zeros(6), qacc=np.zeros(6),
                           ctrl=np.zeros(3), actuator_force=np.zeros(3),
                           geom_xpos=np.asarray([[0., 0., .780], [0., 0., .72]] + [[0., 0., 0.]] * 8),
                           geom_xmat=np.tile(np.eye(3).reshape(1, 9), (10, 1)),
                           xpos=np.asarray([[0., 0., 0.], [0., 0., .780]] + [[0., 0., 0.]] * 5),
                           xipos=np.asarray([[0., 0., 0.], [0., 0., .780]] + [[0., 0., 0.]] * 5),
                           site_xpos=np.array([[0., -.05, .780]]), site_xmat=np.eye(3).reshape(1, 9),
                           contact=[], ncon=0, velocity=np.zeros(6))
    class API:
        mjtObj = SimpleNamespace(mjOBJ_BODY=1, mjOBJ_GEOM=5)
        mjtGeom = SimpleNamespace(mjGEOM_BOX=6)
        mjtJoint = SimpleNamespace(mjJNT_FREE=0)
        @staticmethod
        def mj_name2id(model, kind, name):
            names = body_names if kind == 1 else geom_names
            return names.index(name) if name in names else -1
        @staticmethod
        def mj_id2name(model, kind, index):
            return (body_names if kind == 1 else geom_names)[index]
        @staticmethod
        def mj_contactForce(model, state, index, output): output[:] = state.contact[index].force
        @staticmethod
        def mj_objectVelocity(model, state, kind, index, output, local):
            if local != 0 or kind != 1 or index != 1:
                raise AssertionError('velocity must be actual object COM in world orientation')
            output[:] = state.velocity
    reference = {'timestamp_s': 0., 'object_in_palm_frame_m': [0., .05, 0.]}
    return model, data, API(), reference


def contact(hand_geom, fn, fz, *, object_first=False, frame=None, point=None, dist=-.001, efc=0):
    if frame is None:
        frame = np.eye(3) if hand_geom in (2, 3, 8) else np.diag([-1., -1., 1.])
    force = np.array([fn, 0., fz, 0., 0., 0.])
    if object_first:
        # Reverse the complete contact frame so the opposite object index
        # describes the same force on the object; use a proper rotation.
        frame = np.diag([-1., 1., -1.]) @ frame
        force[1] *= -1.
    return SimpleNamespace(geom1=0 if object_first else hand_geom, geom2=hand_geom if object_first else 0,
                           force=force, frame=np.asarray(frame).reshape(9), pos=np.asarray(point if point is not None else [0., 0., .780]),
                           dist=dist, efc_address=efc)


def load(data):
    weight = .08 * 9.81
    data.contact = [contact(2, .6, weight / 2, point=[-.025, 0., .780]),
                    contact(4, .3, weight / 4, point=[.025, 0., .800]),
                    contact(6, .3, weight / 4, point=[.025, 0., .760])]
    data.ncon = len(data.contact)


def good_sample(time=0.):
    reference = {'timestamp_s': 0., 'object_in_palm_frame_m': [0., .05, 0.]}
    return dict(timestamp_s=time, finite=True, joint_safe=True, force_sample_scope=FORCE_SCOPE,
                grip_reference_sha256=canonical_sha256(reference), fixed_grip_reference=reference,
                object_weight_n=.7848, minimum_bottom_clearance_m=.005, table_normal_force_n=0.,
                object_linear_speed_m_s=0., object_angular_speed_rad_s=0., grip_relative_drift_m=0.,
                hand_force_world_n=[0., 0., .7848], loaded_table_contact_count=0, loaded_other_contact_count=0,
                groups={name: dict(loaded_contact_count=1, normal_force_n=.3) for name in ('thumb', 'index', 'middle')})


class SupportMeasurementTests(unittest.TestCase):
    def test_native_force_sign_frame_com_torque_and_all_hand_shapes(self):
        model, data, api, reference = fixture()
        load(data)
        before = copy.deepcopy(data)
        result = native_grasp_support(model, data, 1, 0, reference, joint_safe=True, mujoco_api=api)
        np.testing.assert_allclose(result['hand_force_world_n'], [0., 0., .7848], atol=1e-15)
        np.testing.assert_allclose(result['hand_torque_about_com_world_nm'], 0., atol=1e-15)
        self.assertEqual(result['groups']['thumb']['loaded_contact_count'], 1)
        self.assertEqual(result['opposition_dot_diagnostic'], {'index': -1., 'middle': -1.})
        self.assertAlmostEqual(result['minimum_bottom_clearance_m'], .005)
        self.assertEqual(result['force_sample_scope'], FORCE_SCOPE)
        np.testing.assert_array_equal(data.qpos, before.qpos)
        np.testing.assert_array_equal(data.qvel, before.qvel)
        self.assertEqual(data.time, before.time)
        # A capsule and the pad are included as well as original active meshes.
        data.contact += [contact(3, .1, 0.), contact(8, .1, 0.)]
        data.ncon = len(data.contact)
        extra = native_grasp_support(model, data, 1, 0, reference, joint_safe=True, mujoco_api=api)
        self.assertEqual(extra['groups']['thumb']['loaded_contact_count'], 2)
        self.assertEqual(extra['groups']['palm']['loaded_contact_count'], 1)

    def test_native_none_reference_is_physical_diagnostic_but_cannot_admit_load(self):
        model, data, api, reference = fixture()
        load(data)
        monitor = SupportWindow(settings(), .002)
        for index in range(110):
            data.time = index * .002
            sample = native_grasp_support(model, data, 1, 0, None, joint_safe=True, mujoco_api=api)
            result = monitor.update(sample)
        self.assertTrue(sample["finite"])
        self.assertIsNone(sample["grip_relative_drift_m"])
        self.assertIsNone(sample["fixed_grip_reference"])
        self.assertFalse(result["ready"])
        self.assertFalse(result["checks"]["fixed_grip_reference_present"])
        # Pre-lock contacts never contribute time after the one true lock.
        for index in range(110, 210):
            data.time = index * .002
            sample = native_grasp_support(model, data, 1, 0, reference, joint_safe=True, mujoco_api=api)
            result = monitor.update(sample)
        self.assertTrue(result["ready"])
        self.assertAlmostEqual(result["contiguous_valid_s"], .198)

    def test_wrench_local_torque_lever_arm_and_object_index_reversal(self):
        # Rows map local normal -> world y; local tangent1 -> world -x.
        frame = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        local = [2., 3., 4., 5., 6., 7.]
        force, torque = contact_wrench_on_object(local, frame, [1., 0., 0.], [0., 0., 0.], 1)
        np.testing.assert_array_equal(force, [-3., 2., 4.])
        np.testing.assert_array_equal(torque, [-6., 1., 9.])
        reverse_force, reverse_torque = contact_wrench_on_object(local, frame, [1., 0., 0.], [0., 0., 0.], 0)
        np.testing.assert_array_equal(reverse_force, -force)
        np.testing.assert_array_equal(reverse_torque, -torque)
        with self.assertRaises(ValueError):
            contact_wrench_on_object(local, frame, [1., 0., 0.], [0., 0., 0.], True)

    def test_loaded_evidence_requires_positive_fn_nonpositive_distance_and_efc(self):
        model, data, api, reference = fixture()
        for kwargs in ({'fn': 0.}, {'dist': .00001}, {'efc': -1}):
            options = dict(fn=.3, dist=-.001, efc=0)
            options.update(kwargs)
            data.contact = [contact(2, options['fn'], .7848, dist=options['dist'], efc=options['efc'])]
            data.ncon = 1
            result = native_grasp_support(model, data, 1, 0, reference, joint_safe=True, mujoco_api=api)
            self.assertEqual(result['groups']['thumb']['loaded_contact_count'], 0)
            np.testing.assert_array_equal(result['hand_force_world_n'], [0., 0., 0.])

    def test_table_load_separate_and_tilted_cube_corner_not_center_is_clearance(self):
        model, data, api, reference = fixture()
        # Contact normal on the cube is world +z, table is geom1/object geom2.
        table_frame = [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]
        data.contact = [contact(1, .7848, 0., frame=table_frame)]
        data.ncon = 1
        angle = math.pi / 4
        rotation = np.array([[math.cos(angle), 0., math.sin(angle)], [0., 1., 0.], [-math.sin(angle), 0., math.cos(angle)]])
        data.geom_xmat[0] = rotation.reshape(9)
        result = native_grasp_support(model, data, 1, 0, reference, joint_safe=True, mujoco_api=api)
        self.assertGreater(data.geom_xpos[0, 2] - .75 - .025, .002)  # Centre-only test would incorrectly pass.
        self.assertLess(result['minimum_bottom_clearance_m'], 0.)
        self.assertEqual(result['loaded_table_contact_count'], 1)
        self.assertAlmostEqual(result['table_normal_force_n'], .7848)
        np.testing.assert_allclose(result['table_force_world_n'], [0., 0., .7848])
        np.testing.assert_array_equal(result['hand_force_world_n'], 0.)

    def test_velocity_relative_drift_and_model_scope_fail_closed(self):
        model, data, api, reference = fixture()
        data.velocity[:] = [.1, .2, .3, .01, .02, .03]
        data.xpos[1, 0] = .004
        result = native_grasp_support(model, data, 1, 0, reference, joint_safe=True, mujoco_api=api)
        self.assertAlmostEqual(result['grip_relative_drift_m'], .004)
        np.testing.assert_array_equal(result['object_com_linear_velocity_world_m_s'], [.01, .02, .03])
        for mutation in ('gravity', 'sphere', 'movingtable', 'tilttable', 'nonfree'):
            m, d, a, r = fixture()
            if mutation == 'gravity': m.opt.gravity[0] = .1
            elif mutation == 'sphere': m.geom_type[0] = 2
            elif mutation == 'movingtable': m.body_weldid[0] = 2
            elif mutation == 'tilttable': d.geom_xmat[1] = np.diag([1., -1., -1.]).reshape(9)
            else: m.jnt_type[0] = 3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                native_grasp_support(m, d, 1, 0, r, joint_safe=True, mujoco_api=a)
        with self.assertRaises(ValueError):
            native_grasp_support(model, data, 1, 0, reference, joint_safe=1, mujoco_api=api)


class SupportWindowTests(unittest.TestCase):
    def test_measured_stable_support_requires_full_window_not_single_contact(self):
        monitor = SupportWindow(settings(), .002)
        for index in range(100):
            row = monitor.update(good_sample(index * .002))
            self.assertEqual(row['ready'], index == 99)
        self.assertAlmostEqual(row['contiguous_valid_s'], .198)
        self.assertTrue(row['checks']['hand_upward_supports_weight'])

    def test_one_side_table_carried_fn_only_slip_motion_and_unsafe_do_not_admit(self):
        for failure in ('one_side', 'table', 'fn_only', 'clearance', 'slip', 'speed', 'omega', 'joint', 'finite', 'external'):
            monitor = SupportWindow(settings(), .002)
            for index in range(110):
                sample = good_sample(index * .002)
                if failure == 'one_side': sample['groups']['index']['loaded_contact_count'] = 0
                elif failure == 'table': sample['loaded_table_contact_count'] = 1; sample['table_normal_force_n'] = .7848
                elif failure == 'fn_only': sample['hand_force_world_n'] = [.7848, 0., 0.]
                elif failure == 'clearance': sample['minimum_bottom_clearance_m'] = -.0001
                elif failure == 'slip': sample['grip_relative_drift_m'] = .00301
                elif failure == 'speed': sample['object_linear_speed_m_s'] = .02001
                elif failure == 'omega': sample['object_angular_speed_rad_s'] = .2001
                elif failure == 'joint': sample['joint_safe'] = False
                elif failure == 'finite': sample['finite'] = False
                else: sample['loaded_other_contact_count'] = 1
                result = monitor.update(sample)
                self.assertFalse(result['ready'], failure)
                self.assertEqual(result['contiguous_valid_s'], 0.)

    def test_lost_load_and_missing_samples_reset_window_without_borrowing_old_contact(self):
        monitor = SupportWindow(settings(), .002)
        for index in range(100): ready = monitor.update(good_sample(index * .002))
        self.assertTrue(ready['ready'])
        lost = good_sample(.200); lost['hand_force_world_n'][2] = 0.
        self.assertFalse(monitor.update(lost)['ready'])
        self.assertEqual(monitor.update(good_sample(.202))['contiguous_valid_s'], 0.)
        gap = monitor.update(good_sample(.208))
        self.assertFalse(gap['ready']); self.assertFalse(gap['cadence_contiguous'])
        self.assertEqual(gap['contiguous_valid_s'], 0.)
        self.assertIn('sample_cadence_gap', gap['failure_reasons'])
        for index in range(1, 100): result = monitor.update(good_sample(.208 + index * .002))
        self.assertTrue(result['ready'])

    def test_sixty_six_ms_samples_use_declared_cadence_and_four_actual_states(self):
        monitor = SupportWindow(settings(), .066)
        for index in range(4): result = monitor.update(good_sample(index * .066))
        self.assertTrue(result['ready'])
        wrong = SupportWindow(settings(), .002)
        for index in range(4): result = wrong.update(good_sample(index * .066))
        self.assertFalse(result['ready'])
        self.assertEqual(result['contiguous_valid_s'], 0.)

    def test_reference_cannot_reset_and_outputs_are_copies(self):
        monitor = SupportWindow(settings(), .002)
        sample = good_sample(0.)
        result = monitor.update(sample)
        result['checks']['finite'] = False
        self.assertTrue(monitor.rows[0]['checks']['finite'])
        changed = good_sample(.002)
        changed['fixed_grip_reference']['object_in_palm_frame_m'][0] = .01
        changed['grip_reference_sha256'] = canonical_sha256(changed['fixed_grip_reference'])
        with self.assertRaises(ValueError): monitor.update(changed)
        with self.assertRaises(ValueError): monitor.update(good_sample(0.))
        self.assertEqual(len(monitor.rows), 1)

    def test_additional_runtime_checks_are_and_gates_and_false_resets_window(self):
        monitor = SupportWindow(settings(), .002)
        for index in range(100):
            sample = good_sample(index * .002)
            sample["additional_required_checks"] = {"actual_side_normals": True, "opposing_thumb_fingers": True, "penetration_bounded": True}
            result = monitor.update(sample)
        self.assertTrue(result["ready"])
        self.assertTrue(result["checks"]["additional:actual_side_normals"])
        failed = good_sample(.200)
        failed["additional_required_checks"] = {"actual_side_normals": False}
        result = monitor.update(failed)
        self.assertFalse(result["ready"])
        self.assertEqual(result["contiguous_valid_s"], 0.)
        self.assertIn("additional:actual_side_normals", result["failure_reasons"])
        # Optional absence preserves old sample behavior, without inheriting
        # any time from the previously invalid side-contact observation.
        result = monitor.update(good_sample(.202))
        self.assertFalse(result["ready"])
        self.assertEqual(result["contiguous_valid_s"], 0.)
        self.assertFalse(any(key.startswith("additional:") for key in result["checks"]))

    def test_additional_runtime_checks_reject_implicit_truth_and_ambiguous_names(self):
        for additional in ({"side": 1}, {"side": None}, {"": True}, {" side": True}, {1: True}, [True], None):
            sample = good_sample()
            sample["additional_required_checks"] = additional
            with self.subTest(additional=additional), self.assertRaises(ValueError):
                SupportWindow(settings(), .002).update(sample)

    def test_invalid_measurement_cannot_leave_previous_ready_status(self):
        monitor = SupportWindow(settings(), .002)
        for index in range(100): monitor.update(good_sample(index * .002))
        self.assertTrue(monitor.ready)
        invalid = good_sample(.200); invalid["force_sample_scope"] = "unproved integration force"
        with self.assertRaises(ValueError): monitor.update(invalid)
        self.assertFalse(monitor.ready)
        self.assertIsNone(monitor.last_status)
        restarted = monitor.update(good_sample(.200))
        self.assertEqual(restarted["contiguous_valid_s"], 0.)
        self.assertFalse(monitor.ready)

    def test_invalid_thresholds_and_measurements_fail_closed(self):
        for period in (0., True, float('nan')):
            with self.assertRaises(ValueError): SupportWindow(settings(), period)
        for key, value in (('stable_window_s', 0.), ('hand_weight_ratio_min', 1.3), ('required_groups', ['thumb', 'thumb'])):
            opts = settings(); opts[key] = value
            with self.assertRaises(ValueError): SupportWindow(opts, .002)
        for field, value in (('finite', 1), ('object_linear_speed_m_s', float('nan')), ('table_normal_force_n', -.1)):
            sample = good_sample(); sample[field] = value
            with self.assertRaises(ValueError): SupportWindow(settings(), .002).update(sample)
