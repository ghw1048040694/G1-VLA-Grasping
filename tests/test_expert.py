"""Physical-success accounting rejects launch, contact loss and unsafe holds."""
import copy
import unittest
from types import SimpleNamespace

from g1_vla.expert import (ExpertError, effective_compensation, hand_target,
                           physical_gate, phase_tracking_reached, validate_teacher,
                           joint_limit_excesses, state_is_finite, green_target_from_position)


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
}


def sample(start, duration=.1, *, contact=True, height=.12, assisted=False, safe=True):
    return {"start_s": start, "end_s": start + duration, "contact": contact,
            "lift_height_m": height, "assisted": assisted, "finite": True, "joint_safe": safe}


class TeacherTests(unittest.TestCase):
    def gate(self, rows):
        return physical_gate(rows, lift_height_m=.1, hold_duration_s=.5)

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

    def test_mechanical_overrun_remains_visible_inside_declared_tolerance(self):
        model = SimpleNamespace(jnt_limited=[True, True], jnt_range=[[-1., 1.], [-1., 1.]])
        data = SimpleNamespace(qpos=[1.002, -1.004])
        bindings = {"right_hand": {"joint_id": 0, "qpos_id": 0},
                    "left_frozen_hand": {"joint_id": 1, "qpos_id": 1}}
        excess = joint_limit_excesses(model, data, bindings)
        self.assertAlmostEqual(excess["right_hand"], .002)
        self.assertAlmostEqual(excess["left_frozen_hand"], .004)
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
