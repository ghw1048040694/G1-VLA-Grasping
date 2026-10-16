"""Contract tests exercise observed failure modes without simulation or torch."""

import copy
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from g1_vla.config import ConfigError, load_config, validate_config
from g1_vla.contracts import (
    ActionContract, ContractError, canonical_sha256, make_checkpoint_metadata, scene_protocol_for_contact_model,
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
    def test_noslip_version_changes_only_scene_identity_and_preserves_older_config(self):
        config = config_fixture()
        config["scene"]["contact_model"]["backend"] = "source_default_v1"
        original = ActionContract.from_config(config, metadata=source_metadata())
        legacy = copy.deepcopy(config)
        del legacy["contact_calibration"]
        self.assertEqual(original.contract_sha256, ActionContract.from_config(legacy, metadata=source_metadata()).contract_sha256)
        self.assertNotIn("contact_calibration", legacy)  # validation has no caller mutation
        config["scene"]["contact_model"]["backend"] = "source_default_noslip10_v1"
        candidate = ActionContract.from_config(config, metadata=source_metadata())
        self.assertNotEqual(candidate.contract_sha256, original.contract_sha256)
        a, b = original.metadata(), candidate.metadata()
        self.assertNotEqual(a.pop("scene_protocol"), b.pop("scene_protocol"))
        self.assertEqual(a, b)
        with self.assertRaises(ContractError):
            replace(candidate, scene_protocol="g1_strict_scene_source_default_noslip10_v1_" + "0" * 64)
        config["scene"]["contact_model"]["time_constant_s"] = .006
        with self.assertRaises(ConfigError):
            validate_config(config)

    def test_calibration_protocol_is_predeclared_without_nested_boolean_or_parameter_overrides(self):
        for mutation in ("friction", "nested_bool", "threshold", "unknown"):
            config = config_fixture()
            block = config["contact_calibration"]
            if mutation == "friction":
                block["friction"][0] += .1
            elif mutation == "nested_bool":
                block["solref"][1] = True
            elif mutation == "threshold":
                block["acceptance"]["hold_every_sample_in_final_half_second"]["global_initial_COM_displacement_max_m"] = .01
            else:
                block["silent_best_of_trials"] = True
            with self.subTest(mutation=mutation), self.assertRaises(ConfigError):
                validate_config(config)

    def test_opposition_settings_cannot_silently_change_source_physics_or_closure_clock(self):
        config = config_fixture()
        config["expert"]["opposition"]["backend"] = "side_opposition_preload_v1"
        validate_config(config)
        for owner, key, value in (("opposition", "transfer_height_m", .011),
                                  ("opposition", "transfer_reference_duration_s", 3.),
                                  ("opposition", "maximum_opposition_dot", 0.),
                                  ("opposition", "planning_taskpoint_tolerance_m", .016),
                                  ("opposition", "planning_orientation_tolerance_rad", .051),
                                  ("hand_control", "backend", "contact_normal_position_v2"),
                                  ("lift_start", "backend", "measured_cube_slow_start_v1")):
            changed = copy.deepcopy(config)
            changed["expert"][owner][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ConfigError):
                validate_config(changed)
        changed = copy.deepcopy(config)
        changed["expert"]["opposition"]["closure_keyframes"][0]["joint_target_rad"][0] += .1
        with self.assertRaises(ConfigError):
            validate_config(changed)
        changed = copy.deepcopy(config)
        changed["scene"]["contact_model"] = {"backend": "right_hand_green_compliance_v2", "time_constant_s": .006, "damping_ratio": 1.}
        with self.assertRaises(ConfigError):
            validate_config(changed)

    def test_contact_model_identity_cannot_silently_reuse_the_original_contract(self):
        config = config_fixture()
        config["expert"]["opposition"]["backend"] = "disabled"  # Isolate the separate compliance-contract hypothesis.
        original = {"backend": "source_default_v1", "time_constant_s": .02, "damping_ratio": 1.}
        self.assertEqual(scene_protocol_for_contact_model(None), scene_protocol_for_contact_model(original))
        candidate = {"backend": "right_hand_green_compliance_v2", "time_constant_s": .006, "damping_ratio": 1.}
        config["scene"]["contact_model"] = original
        first = ActionContract.from_config(config, metadata=source_metadata())
        config["scene"]["contact_model"] = candidate
        second = ActionContract.from_config(config, metadata=source_metadata())
        self.assertNotEqual(first.contract_sha256, second.contract_sha256)
        self.assertEqual(first.joint_names, second.joint_names)
        self.assertEqual(first.max_velocity_rad_s, second.max_velocity_rad_s)
        self.assertEqual(second.scene_protocol, scene_protocol_for_contact_model(candidate))
        config["scene"]["contact_model"]["time_constant_s"] = .008
        self.assertNotEqual(second.contract_sha256, ActionContract.from_config(config, metadata=source_metadata()).contract_sha256)
        config["scene"]["contact_model"]["time_constant_s"] = .002
        with self.assertRaisesRegex(ConfigError, "three physics"):
            validate_config(config)
        with self.assertRaises(ContractError):
            scene_protocol_for_contact_model({**original, "time_constant_s": .006})

    def test_current_teacher_configuration_disables_training(self):
        config = load_config(PROJECT / "configs" / "experiment.json")
        self.assertFalse(config["train"]["enabled"])
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

    def test_cartesian_planning_cannot_bypass_original_tracking_gates(self):
        for key, value in (("taskpoint_tolerance_m", .04), ("reference_tracking_m", .04),
                           ("reference_tracking_rad", .07), ("interpolation_samples", 1),
                           ("maximum_joint_sample_spacing_rad", .07),
                           ("solver_finite_difference_rad", .03)):
            config = config_fixture()
            config["expert"]["trajectory"][key] = value
            with self.subTest(key=key), self.assertRaises(ConfigError):
                validate_config(config)
        config = config_fixture()
        config["expert"]["teacher"]["contact_compensation_m"][0] = .01
        with self.assertRaisesRegex(ConfigError, "forbids legacy"):
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

    def test_state_residual_and_coverage_preserve_zero_identity_and_budget(self):
        for change in ("nonzero_head", "drop_step_zero", "vae", "center_target", "inverse_anchor", "missing_control", "budget", "motion", "bins", "unbounded_candidates"):
            config = config_fixture()
            residual, coverage = config["state_residual"], config["trajectory_coverage"]
            if change == "nonzero_head":
                residual["zero_head_init"] = False
            elif change == "drop_step_zero":
                residual["include_step_zero_in_selection"] = False
            elif change == "vae":
                residual["use_vae"] = True
            elif change == "center_target":
                residual["target"] = "mean_centered_residual"
            elif change == "inverse_anchor":
                residual["decode"] = "inverse_normalized_state_anchor"
            elif change == "missing_control":
                residual["variants"] = ["rgb_residual_scratch"]
            elif change == "budget":
                residual["max_steps"] += 50
            elif change == "motion":
                coverage["moving_speed_rad_s"] = 0.2
            elif change == "bins":
                coverage["quarter_bins"] = 3
            else:
                coverage["expansion_group_limit"] = 5
            with self.subTest(change=change), self.assertRaises(ConfigError):
                validate_config(config)

    def test_expansion_keeps_candidates_parent_split_and_residual_budget_fixed(self):
        for change in ("candidate", "independent_claim", "coverage_disabled", "rows", "bytes",
                       "parent_val", "missing_rgb", "steps", "vae", "origin", "manifest", "reference"):
            config = config_fixture()
            expansion, model = config["train_expansion"], config["expanded_residual"]
            if change == "candidate":
                expansion["candidate_episode_ids"][0] = 13
            elif change == "independent_claim":
                expansion["require_independent_sessions"] = True
            elif change == "coverage_disabled":
                expansion["allow_same_source_coverage"] = False
            elif change == "rows":
                expansion["max_total_rows"] = 20001
            elif change == "bytes":
                expansion["max_derived_bytes"] = 6 * 1024**3
            elif change == "parent_val":
                config["source_native"]["split_episode_ids"]["validation"][0] = 183
                config["source_native"]["episode_ids"][6] = 183
            elif change == "missing_rgb":
                model["variants"].pop()
            elif change == "steps":
                model["max_steps"] += 50
            elif change == "vae":
                model["use_vae"] = True
            elif change == "origin":
                model["zero_head_init"] = False
            elif change == "manifest":
                model["data_manifest_path"] = "old/manifest.json"
            else:
                model["reference_residual_report_sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ConfigError):
                validate_config(config)

    def test_normalization_and_constant_controls_keep_the_declared_treatment(self):
        for change in ("budget", "statistics_population", "statistics_path", "missing_rgb", "manifest",
                       "fit_validation", "fit_order", "mean_fit", "decode", "stride", "reference"):
            config = config_fixture()
            model, constant = config["normalization_attribution"], config["constant_residual"]
            if change == "budget":
                model["max_steps"] += 50
            elif change == "statistics_population":
                model["model_statistics_source"] = "expanded_fourteen_train_only_v1"
            elif change == "statistics_path":
                model["reference_statistics_manifest_path"] = model["data_manifest_path"]
            elif change == "missing_rgb":
                model["variants"].pop()
            elif change == "manifest":
                model["data_manifest_sha256"] = "0" * 64
            elif change == "fit_validation":
                constant["fit_cohorts"].append("validation")
            elif change == "fit_order":
                constant["fit_cohorts"].reverse()
            elif change == "mean_fit":
                constant["method"] = "mean"
            elif change == "decode":
                constant["decode"] = "inverse_normalized_anchor_plus_bias"
            elif change == "stride":
                constant["eval_stride"] = 1
            else:
                constant["reference_expanded_report_sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ConfigError):
                validate_config(config)

    def test_state_conditioning_cannot_add_training_test_or_tuned_interventions(self):
        changes = {"optimizer_steps": 1, "modes": ["real", "train_mean"],
                   "eval_splits": ["train", "validation", "test"],
                   "variants": ["state_only_residual"], "eval_stride": 1, "eval_batch_size": 16,
                   "within_episode_rule": "label_nearest", "cross_episode_rule": "best_validation_donor",
                   "real_reproduction_atol": 1e-2, "tie_tolerance_rad": 1e-3,
                   "max_eval_wall_s": 601, "max_gpu_reserved_bytes": 8 * 1024**3,
                   "model_output_constant_method": "validation_fitted_constant",
                   "reference_expanded_report_sha256": "0" * 64}
        for key, value in changes.items():
            config = config_fixture()
            config["state_conditioning"][key] = value
            with self.subTest(key=key), self.assertRaises(ConfigError):
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

    def test_original_checkpoint_cannot_cross_the_versioned_global_friction_solver(self):
        original = replace(self.contract, scene_protocol=scene_protocol_for_contact_model({
            "backend": "source_default_v1", "time_constant_s": .02, "damping_ratio": 1.}))
        metadata = make_checkpoint_metadata(original, self.vocabulary, self.stats, observation_spec=self.spec)
        candidate = replace(original, scene_protocol=scene_protocol_for_contact_model({
            "backend": "source_default_noslip10_v1", "time_constant_s": .02, "damping_ratio": 1.}))
        with self.assertRaises(ContractError):
            validate_checkpoint_metadata(metadata, candidate, self.vocabulary, self.stats, observation_spec=self.spec)

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
