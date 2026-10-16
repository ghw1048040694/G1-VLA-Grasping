import unittest
from g1_vla.hand_contact import finger_group, summarize_contacts


class ContactTelemetryTests(unittest.TestCase):
    def test_loaded_contact_requires_overlap_constraint_and_positive_normal_force(self):
        def row(distance, address, force):
            return {"group": "index", "distance_m": distance, "efc_address": address,
                    "force_contact_frame": [force, 0, 0, 0, 0, 0]}
        rows = [row(-.014, -1, 20), row(-.012, 0, 0), row(.001, 1, 4),
                row(-.003, 2, -3), row(-.0004, 3, .8)]
        groups = summarize_contacts(rows)
        self.assertEqual(groups["index"]["geometric_contacts"], 4)
        self.assertEqual(groups["index"]["loaded_contacts"], 1)
        self.assertEqual(groups["index"]["normal_force_n"], .8)
        self.assertEqual(groups["index"]["maximum_penetration_m"], .014)
        self.assertEqual(groups["index"]["maximum_loaded_penetration_m"], .0004)
        self.assertEqual(rows[0]["force_contact_frame"][0], 20)
        self.assertEqual(groups["thumb"]["loaded_contacts"], 0)

    def test_grouping_never_calls_left_or_table_right_finger(self):
        self.assertEqual(finger_group("right_hand_thumb_2_link", "x"), "thumb")
        self.assertEqual(finger_group("x", "right_palm_contact_pad"), "palm")
        self.assertIsNone(finger_group("left_hand_index_0_link", "left_palm_contact_pad"))
        self.assertIsNone(finger_group("calibration_table", "calibration_table"))
