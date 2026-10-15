"""Contract tests exercise observed failure modes without simulation or torch."""

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from g1_vla.config import ConfigError, load_config, validate_config
from g1_vla.contracts import (
    ActionContract, ContractError, canonical_sha256, make_checkpoint_metadata,
    observation_spec_from_config, read_joint_order, validate_checkpoint_metadata, validate_observation,
)


PROJECT = Path(__file__).resolve().parents[1]


def config_fixture():
    with (PROJECT / "configs" / "experiment.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def source_metadata():
    with (PROJECT / "resources" / "g1_action31_train_root_stub" / "meta" / "info.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def contract_fixture():
    config = config_fixture()
    config["control"]["max_anchor_delta_rad"] = 0.5
    config["control"]["max_velocity_rad_s"] = 1.0
    return ActionContract.from_config(config, metadata=source_metadata())


class ConfigTests(unittest.TestCase):
    def test_default_native_training_is_conditionally_authorized(self):
        config = load_config(PROJECT / "configs" / "experiment.json")
        self.assertTrue(config["train"]["enabled"])
        self.assertEqual(config["train"]["backend"], "grasp_square_act")
        self.assertEqual(config["train"]["authorization"], "conditional")
        self.assertFalse(config["wm"]["enabled"])
        self.assertEqual(config["paths"]["root"], str(PROJECT))
        self.assertEqual(config["action"]["joint_metadata_path"], str(
            PROJECT / "resources" / "g1_action31_train_root_stub" / "meta" / "info.json"))
        self.assertEqual(len(ActionContract.from_config(config).max_velocity_rad_s), 31)
        config["control"]["max_velocity_rad_s"] = None
        with self.assertRaisesRegex(ContractError, "unresolved"):
            ActionContract.from_config(config)

    def test_unknown_keys_rejected_recursively(self):
        for path in ("root", "action", "train", "observation"):
            config = config_fixture()
            owner = config if path == "root" else config[path]
            owner["quiet_typo"] = 1
            with self.subTest(path=path), self.assertRaisesRegex(ConfigError, "unknown keys"):
                validate_config(config)

    def test_native_split_and_model_cadence_are_not_silent_overrides(self):
        for change in ("split_overlap", "discovery_episode", "chunk_mismatch", "eval_overlap"):
            config = config_fixture()
            if change == "split_overlap":
                config["source_native"]["split_episode_ids"]["validation"][0] = config["source_native"]["split_episode_ids"]["train"][0]
            elif change == "discovery_episode":
                prior = config["source_native"]["episode_ids"][0]
                config["source_native"]["episode_ids"][0] = 0
                config["source_native"]["split_episode_ids"]["train"][0] = 0
            elif change == "chunk_mismatch":
                config["grasp_act"]["chunk_size"] += 1
            else:
                config["grasp_act"]["eval_stride"] = 1
            with self.subTest(change=change), self.assertRaises(ConfigError):
                validate_config(config)

    def test_diagnostic_controls_reject_silent_scope_or_seed_changes(self):
        for change in ("missing_visual_control", "unknown_variant", "changed_seed", "lag_without_zero", "unordered_motion_thresholds"):
            config = config_fixture()
            if change == "missing_visual_control":
                config["act_diagnostics"]["visual_modes"].pop()
            elif change == "unknown_variant":
                config["act_diagnostics"]["variants"] = ["use_test_to_select"]
            elif change == "changed_seed":
                config["act_diagnostics"]["seed"] += 1
            elif change == "lag_without_zero":
                config["native_diagnostics"]["lag_frames"] = [-1, 1]
            else:
                config["native_diagnostics"]["stationary_speed_rad_s"] = 1
            with self.subTest(change=change), self.assertRaises(ConfigError):
                validate_config(config)

    def test_visual_pretraining_preserves_protocol_and_download_identity(self):
        for change in ("vae", "missing_control", "changed_budget", "changed_preprocess", "changed_source", "changed_hash", "excess_download"):
            config = config_fixture()
            visual = config["visual_pretraining"]
            if change == "vae":
                visual["use_vae"] = True
            elif change == "missing_control":
                visual["variants"].pop()
            elif change == "changed_budget":
                visual["max_steps"] += 50
            elif change == "changed_preprocess":
                visual["image_preprocessing"] = "pretrained_only_imagenet_crop"
            elif change == "changed_source":
                visual["weights_url"] = "https://unverified.invalid/resnet18.pth"
            elif change == "changed_hash":
                visual["weights_sha256"] = "0" * 64
            else:
                visual["weights_max_bytes"] = 65 * 1024**2
            with self.subTest(change=change), self.assertRaises(ConfigError):
                validate_config(config)

    def test_velocity_zero_is_hold_and_bad_limits_are_rejected(self):
        config = config_fixture()
        config["control"]["max_velocity_rad_s"] = [0.] * 31
        self.assertEqual(ActionContract.from_config(config, metadata=source_metadata()).max_velocity_rad_s, (0.,) * 31)
        for values in ([0.] * 30, [-1.] * 31, [float("nan")] * 31):
            config["control"]["max_velocity_rad_s"] = values
            with self.subTest(values=values), self.assertRaises(ConfigError):
                validate_config(config)

    def test_train_and_wm_enabled_rejected(self):
        for section in ("wm",):
            config = config_fixture()
            config[section]["enabled"] = True
            with self.subTest(section=section), self.assertRaises(ConfigError):
                validate_config(config)
        config = config_fixture()
        config["train"]["enabled"] = True
        self.assertEqual(validate_config(config)["train"]["authorization"], "conditional")

    def test_boolean_numeric_and_invalid_timing_rejected(self):
        for value in (True, 0, -15, float("nan"), float("inf")):
            config = config_fixture()
            config["control"]["control_hz"] = value
            with self.subTest(value=value), self.assertRaises(ConfigError):
                validate_config(config)
        config = config_fixture()
        config["control"]["control_hz"] = 15.0
        with self.assertRaisesRegex(ConfigError, "integer number"):
            validate_config(config)

    def test_checkpoint_resume_and_warm_start_are_distinct(self):
        config = config_fixture()
        config["train"]["resume_checkpoint"] = "resume.pt"
        config["train"]["warm_start_checkpoint"] = "warm.pt"
        with self.assertRaisesRegex(ConfigError, "mutually exclusive"):
            validate_config(config)

    def test_split_must_group_layout_and_parent_rollout(self):
        config = config_fixture()
        config["data"]["split_group_fields"] = ["episode_id"]
        with self.assertRaisesRegex(ConfigError, "parent_rollout_id"):
            validate_config(config)

    def test_explicit_root_and_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "input.json"
            config_path.write_text(json.dumps(config_fixture()), encoding="utf-8")
            config = load_config(config_path, project_root=root / "project")
            self.assertEqual(config["data"]["build_dir"], str((root / "project" / "outputs" / "rebuild" / "data").resolve()))
            config_path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "duplicate JSON key"):
                load_config(config_path)


class ActionTests(unittest.TestCase):
    def test_frozen_command_cannot_hide_physical_drift_beyond_anchor(self):
        config = config_fixture()
        config["control"]["max_velocity_rad_s"] = [0.] * 31
        contract = ActionContract.from_config(config, metadata=source_metadata())
        targets = contract.limit_targets([0.] * 31, [[1.] * 31] * 8, joint_names=contract.joint_names)
        self.assertEqual(targets, ((0.,) * 31,) * 8)
        with self.assertRaisesRegex(ContractError, "bounds conflict"):
            contract.limit_targets([.1] * 31, [[.1] * 31] * 8,
                                  joint_names=contract.joint_names, previous_target=[0.] * 31)

    def test_joint_order_comes_from_source_and_mismatch_fails(self):
        metadata = source_metadata()
        names = read_joint_order(metadata)
        contract = contract_fixture()
        self.assertEqual(contract.joint_names, names)
        with self.assertRaisesRegex(ContractError, "joint order differs"):
            contract.decode_chunk([0.0] * 31, [[0.0] * 31] * 8,
                                  joint_names=tuple(reversed(names)))
        config = config_fixture()
        config["control"]["max_anchor_delta_rad"] = 0.5
        config["control"]["max_velocity_rad_s"] = 1
        config["action"]["joint_order_sha256"] = canonical_sha256(list(reversed(names)))
        with self.assertRaisesRegex(ContractError, "SHA-256"):
            ActionContract.from_config(config, metadata=metadata)

    def test_inconsistent_state_and_action_order_rejected(self):
        metadata = source_metadata()
        state = copy.deepcopy(metadata["features"]["action"])
        state["names"] = list(reversed(state["names"]))
        metadata["features"]["observation.state"] = state
        with self.assertRaisesRegex(ContractError, "joint order differ"):
            read_joint_order(metadata)

    def test_chunk_uses_one_anchor_instead_of_cumulative_deltas(self):
        contract = contract_fixture()
        anchor = [2.0] * 31
        targets = [[2.0 + (step + 1) * 0.1] * 31 for step in range(8)]
        encoded = contract.encode_chunk(anchor, targets, joint_names=contract.joint_names)
        decoded = contract.decode_chunk(anchor, encoded, joint_names=contract.joint_names)
        for step in range(8):
            self.assertAlmostEqual(encoded[step][0], (step + 1) * 0.1)
            self.assertAlmostEqual(decoded[step][0], targets[step][0])
        self.assertAlmostEqual(decoded[-1][0], 2.8)
        with self.assertRaisesRegex(ContractError, "exactly 8 steps"):
            contract.decode_chunk(anchor, encoded[:4], joint_names=contract.joint_names)

    def test_amplitude_and_adjacent_velocity_are_separate_limits(self):
        contract = contract_fixture()
        targets = [[10.0] * 31] * 8
        limited = contract.limit_targets([0.0] * 31, targets, joint_names=contract.joint_names)
        previous = [0.0] * 31
        for command in limited:
            self.assertLessEqual(abs(command[0]), 0.5)
            self.assertLessEqual(abs(command[0] - previous[0]), 1.0 / contract.control_hz + 1e-12)
            previous = command
        self.assertAlmostEqual(limited[-1][0], 0.5)
        self.assertEqual(len(contract.executable_targets([0.0] * 31, targets,
                                                        joint_names=contract.joint_names)), 4)
        with self.assertRaisesRegex(ContractError, "bounds conflict"):
            contract.limit_targets([0.0] * 31, targets, joint_names=contract.joint_names,
                                   previous_target=[10.0] * 31)

    def test_non_finite_action_and_wrong_shapes_rejected(self):
        contract = contract_fixture()
        bad = [[0.0] * 31 for _ in range(8)]
        bad[0][0] = math.nan
        with self.assertRaisesRegex(ContractError, "finite numbers"):
            contract.decode_chunk([0.0] * 31, bad, joint_names=contract.joint_names)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.contract = contract_fixture()
        self.vocabulary = ["<pad>", "grasp", "green", "cube"]
        self.stats = {"state_mean": [0.0] * 31, "state_std": [1.0] * 31}
        self.spec = observation_spec_from_config(config_fixture())
        self.metadata = make_checkpoint_metadata(self.contract, self.vocabulary, self.stats, observation_spec=self.spec)

    def test_valid_metadata_and_contract_hash(self):
        validate_checkpoint_metadata(self.metadata, self.contract, self.vocabulary, self.stats, observation_spec=self.spec)
        self.assertEqual(self.metadata["contract_sha256"], canonical_sha256(self.metadata["contract"]))

    def test_contract_vocabulary_statistics_mismatch_rejected(self):
        cases = []
        wrong_hash = copy.deepcopy(self.metadata)
        wrong_hash["contract_sha256"] = "0" * 64
        cases.append((wrong_hash, self.vocabulary, self.stats))
        cases.append((self.metadata, list(reversed(self.vocabulary)), self.stats))
        wrong_stats = copy.deepcopy(self.stats)
        wrong_stats["state_std"][0] = 2.0
        cases.append((self.metadata, self.vocabulary, wrong_stats))
        wrong_names = copy.deepcopy(self.metadata)
        wrong_names["contract"]["joint_names"] = list(reversed(self.contract.joint_names))
        cases.append((wrong_names, self.vocabulary, self.stats))
        for index, (metadata, vocabulary, stats) in enumerate(cases):
            with self.subTest(case=index), self.assertRaisesRegex(ContractError, "mismatch"):
                validate_checkpoint_metadata(metadata, self.contract, vocabulary, stats, observation_spec=self.spec)

    def test_missing_metadata_and_boolean_schema_rejected(self):
        with self.assertRaisesRegex(ContractError, "envelope"):
            validate_checkpoint_metadata({}, self.contract, self.vocabulary, self.stats, observation_spec=self.spec)
        metadata = copy.deepcopy(self.metadata)
        metadata["schema_version"] = True
        with self.assertRaisesRegex(ContractError, "schema_version mismatch"):
            validate_checkpoint_metadata(metadata, self.contract, self.vocabulary, self.stats, observation_spec=self.spec)

    def test_observation_whitelist_blocks_simulator_truth_and_history_leak(self):
        observation = {"images": {key: [[[0, 0, 0]] * self.spec["image_width"] for _ in range(self.spec["image_height"])]
                                  for key in self.spec["camera_keys"]},
                       "joint_position": [0.0] * 31, "joint_velocity": [0.0] * 31,
                       "finger_sites": [0.0] * 108, "tactile": [0.0], "timestamp_s": 0.0,
                       "language": "grasp green cube"}
        validate_observation(observation, observation_spec=self.spec)
        for field in ("sim", "model", "data", "object_truth", "object_positions"):
            invalid = dict(observation, **{field: object()})
            with self.subTest(field=field), self.assertRaises(ContractError):
                validate_observation(invalid, observation_spec=self.spec)
        observation["history"] = [{"object_truth": [0, 0, 0]}]
        with self.assertRaisesRegex(ContractError, "unsupported"):
            validate_observation(observation, observation_spec=self.spec)

    def test_camera_order_size_and_tactile_spec_must_match(self):
        reordered = copy.deepcopy(self.spec)
        reordered["camera_keys"] = list(reversed(reordered["camera_keys"]))
        reordered["camera_ids"] = list(reversed(reordered["camera_ids"]))
        with self.assertRaisesRegex(ContractError, "observation_spec mismatch"):
            validate_checkpoint_metadata(self.metadata, self.contract, self.vocabulary, self.stats, observation_spec=reordered)
        resized = copy.deepcopy(self.spec)
        resized["image_width"] += 1
        with self.assertRaisesRegex(ContractError, "observation_spec mismatch"):
            validate_checkpoint_metadata(self.metadata, self.contract, self.vocabulary, self.stats, observation_spec=resized)
        legacy = copy.deepcopy(self.spec)
        legacy["tactile_protocol"] = "object_filtered_legacy"
        with self.assertRaisesRegex(ContractError, "tactile_protocol"):
            make_checkpoint_metadata(self.contract, self.vocabulary, self.stats, observation_spec=legacy)


if __name__ == "__main__":
    unittest.main()
