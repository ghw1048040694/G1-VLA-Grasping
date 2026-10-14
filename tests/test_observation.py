"""Pure observed-geometry/contact/history tests; no rendering or physics."""

import copy
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

from g1_vla.contracts import ContractError, validate_observation
from g1_vla.observation import (
    ObservationError, ObservationSpec, SensorHistory, capsule_samples,
    external_hand_contact, finger_contact_sites,
)


PROJECT = Path(__file__).resolve().parents[1]


def spec_fixture():
    config = json.loads((PROJECT / "configs" / "experiment.json").read_text())
    config["observation"]["image_width"] = 2
    config["observation"]["image_height"] = 1
    return ObservationSpec.from_config(config)


def observation_fixture(time_s=0.0):
    spec = spec_fixture().metadata()
    return {
        "images": {key: [[[0, 128, 255], [0, 0, 0]]] for key in spec["camera_keys"]},
        "joint_position": [0.0] * 31, "joint_velocity": [0.0] * 31,
        "finger_sites": [0.0] * 108, "tactile": [0.0],
        "language": "grasp the green cube", "timestamp_s": time_s,
    }


class ObservationTests(unittest.TestCase):
    def test_fk_uses_local_z_endpoints_in_stable_108d_order(self):
        rotation = [0, 0, 1, 0, 1, 0, -1, 0, 0]
        samples = capsule_samples([1, 2, 3], rotation, 0.5)
        self.assertEqual(samples, (0.5, 2.0, 3.0, 1.0, 2.0, 3.0, 1.5, 2.0, 3.0))
        model = SimpleNamespace(geom_size=[[0.014, 0.5, 0]] * 12)
        data = SimpleNamespace(geom_xmat=[rotation] * 12, geom_xpos=[[index, 2, 3] for index in range(12)])
        sites = finger_contact_sites(model, data, tuple(range(12)))
        self.assertEqual(len(sites), 108)
        self.assertEqual(sites[:9], [-0.5, 2.0, 3.0, 0.0, 2.0, 3.0, 0.5, 2.0, 3.0])
        self.assertEqual(sites[54:63], [5.5, 2.0, 3.0, 6.0, 2.0, 3.0, 6.5, 2.0, 3.0])

    def test_tactile_uses_any_external_contact_and_excludes_robot_self_contact(self):
        # hand0, hand1, pelvis, world/table, unrelated external body
        geom_bodies, hands, robot = [3, 4, 1, 0, 99], {0, 1}, {1, 3, 4}
        self.assertEqual(external_hand_contact([(0, 4, -0.001)], hands, robot, geom_bodies), 1)
        self.assertEqual(external_hand_contact([(3, 0, 0)], hands, robot, geom_bodies), 1)
        self.assertEqual(external_hand_contact([(0, 1, -0.001)], hands, robot, geom_bodies), 0)
        self.assertEqual(external_hand_contact([(0, 2, -0.001)], hands, robot, geom_bodies), 0)
        self.assertEqual(external_hand_contact([(0, 4, 0.001)], hands, robot, geom_bodies), 0)

    def test_observation_schema_explicitly_versions_camera_geometry_and_tactile(self):
        spec = spec_fixture()
        metadata = spec.metadata()
        self.assertEqual(metadata["tactile_protocol"], "g1_hand_external_contact_binary_v1")
        self.assertEqual(metadata["finger_site_geometry_order"][0], "left_hand_thumb_1_link_contact")
        self.assertEqual(metadata["finger_site_geometry_order"][6], "right_hand_thumb_1_link_contact")
        changed = copy.deepcopy(metadata)
        changed["camera_keys"].reverse()
        changed["camera_ids"].reverse()
        self.assertNotEqual(spec.sha256, ObservationSpec(changed).sha256)

    def test_all_sensor_numeric_fields_and_images_reject_nonfinite_or_bad_dtype(self):
        spec = spec_fixture().metadata()
        for field, value in (("timestamp_s", math.nan), ("tactile", [math.nan]),
                             ("finger_sites", [math.inf] + [0.0] * 107),
                             ("joint_velocity", [math.nan] + [0.0] * 30)):
            observation = observation_fixture()
            observation[field] = value
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_observation(observation, observation_spec=spec)
        for value in (math.nan, 0.5, -1, 256, True):
            observation = observation_fixture()
            observation["images"][spec["camera_keys"][0]][0][0][0] = value
            with self.subTest(pixel=value), self.assertRaisesRegex(ContractError, "uint8"):
                validate_observation(observation, observation_spec=spec)

    def test_nested_history_allows_only_complete_causal_sensor_frames(self):
        spec = spec_fixture().metadata()
        for leaked in ("object_positions", "object_body_ids", "model", "data"):
            observation = observation_fixture(0.066)
            prior = observation_fixture(0.0)
            prior[leaked] = [0, 0, 0]
            observation["history"] = [prior]
            with self.subTest(leaked=leaked), self.assertRaisesRegex(ContractError, "unsupported"):
                validate_observation(observation, observation_spec=spec)
        observation = observation_fixture(0.066)
        observation["history"] = [observation_fixture(0.132)]
        with self.assertRaisesRegex(ContractError, "future"):
            validate_observation(observation, observation_spec=spec)

    def test_history_is_causal_padded_and_snapshot_copied(self):
        history = SensorHistory(spec_fixture())
        first = observation_fixture(0.0)
        history.append(first)
        first["joint_position"][0] = 99
        self.assertEqual([frame["timestamp_s"] for frame in history.frames()], [0] * 4)
        self.assertEqual(history.frames()[0]["joint_position"][0], 0)
        for index in range(1, 6):
            history.append(observation_fixture(index * 0.066))
        self.assertEqual([frame["timestamp_s"] for frame in history.frames()], [index * 0.066 for index in range(2, 6)])
        with self.assertRaisesRegex(ObservationError, "increasing"):
            history.append(observation_fixture(5 * 0.066))
        changed = observation_fixture(6 * 0.066)
        changed["language"] = "put the yellow rod away"
        with self.assertRaisesRegex(ObservationError, "reset history"):
            history.append(changed)


if __name__ == "__main__":
    unittest.main()
