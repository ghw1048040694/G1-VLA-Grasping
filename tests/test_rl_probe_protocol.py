"""Tamper/ABI/admission tests, without Torch, Isaac or physical stepping."""

import copy
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from g1_vla import rl_probe_protocol as probe


@contextmanager
def fixture():
    """Synthetic byte-pinned provenance; private pins are patched only here."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = root / "joint_metadata.json"
        metadata.write_text('{"joint_order":"fixture"}\n')
        config = {"paths": {"root": str(root)}, "rl_teacher": {"physics_protocol": "fixture-fixed"},
                  "action": {"joint_metadata_path": str(metadata)},
                  "control": {"physics_dt_s": .002, "physics_substeps": 33, "max_anchor_delta_rad": .08},
                  "expert": {"pd": {"effort_limit_nm": 1.4}}}
        policy = {"observation_dim": 81, "action_dim": 14, "quaternion_order": "xyzw",
                  "physics_dt_s": .002, "control_period_s": .066,
                  "action_joint_names": ["right_joint_" + str(i) for i in range(14)],
                  "action_units": "normalized joint target velocity; rad/s after source limit scaling",
                  "asset_sha256": "a" * 64, "seed_state_sha256": "b" * 64,
                  "asset_manifest_sha256": "c" * 64,
                  "actual_pd_specifications": {"joint": {"kp": 9., "effort_limit_nm": 1.4}},
                  "resolved_settings": {"train_xy_jitter_m": .0002, "support_window_s": .198},
                  "native_joint_axis_frame_unit_schema": {"joint": {"axis": "X", "limit_rad": [-1., 1.]}},
                  "validation_offsets_m": [[-.0004, -.0002], [.0004, -.0002]]}
        records = [{"path": "src/g1_vla/" + ("nested/m.py" if i == 50 else "module_%02d.py" % i),
                    "bytes": i + 1, "sha256": "%064x" % (i + 100)} for i in range(51)]
        records += [{"path": name, "bytes": 1, "sha256": value} for name, value in
                    (("assets/robot.usd", "a" * 64), ("assets/seed.json", "b" * 64),
                     ("assets/manifest.json", "c" * 64))]
        declaration = {"schema_version": 1, "files": records, "source_config_sha256": "1" * 64}
        manifest = {**declaration, "sha256": probe._canonical(declaration),
                    "source_asset_manifest_sha256": "c" * 64, "source_seed_sha256": "b" * 64}
        manifest_path = root / "reference_manifest.json"
        audit_path = root / "reference_audit.json"
        checkpoint = root / "final78.pt"
        manifest_path.write_text(json.dumps(manifest))
        audit_path.write_text(json.dumps({"policy_contract": policy, "policy_contract_sha256": probe._canonical(policy)}))
        checkpoint.write_bytes(b"synthetic checkpoint bytes; this test never unpickles")
        pins = {"reference_manifest_sha256": probe._file_hash(manifest_path),
                "reference_environment_audit_sha256": probe._file_hash(audit_path),
                "reference_checkpoint_sha256": probe._file_hash(checkpoint),
                "reference_release_sha256": manifest["sha256"],
                "reference_source_tree_sha256": probe._reference_source_identity(manifest),
                "reference_training_config_sha256": probe._semantic_training_identity(config),
                "reference_policy_contract_sha256": probe._canonical(policy)}
        config["rl_diagnostics"] = {"enabled": True, "guided_hand_target_rad": [0.] * 7,
                                   "reference_manifest_path": str(manifest_path),
                                   "reference_environment_audit_path": str(audit_path),
                                   "reference_checkpoint_path": str(checkpoint), **pins}
        infos = {"runner_version": probe._REFERENCE_RUNNER_VERSION,
                 "source_tree_sha256": pins["reference_source_tree_sha256"],
                 "training_config_sha256": pins["reference_training_config_sha256"],
                 "policy_contract_sha256": pins["reference_policy_contract_sha256"],
                 "completed_updates": 78, "optimizer_steps": 1248, "policy_contract": copy.deepcopy(policy)}
        actual = copy.deepcopy(policy)
        actual["control_guard_version"] = "box_halfspace_projected_v2"
        current = {"policy_contract": actual, "policy_contract_sha256": probe._canonical(actual),
                   "source_tree_sha256": probe._current_source_identity()}
        with mock.patch.object(probe, "_REFERENCE_PINS", pins):
            yield config, infos, current, manifest, pins


class ProbeSettingsTests(unittest.TestCase):
    def test_import_does_not_import_tensor_or_simulator_packages(self):
        code = ("import sys; import g1_vla.rl_probe_protocol; "
                "assert not any(x.split('.')[0] in ('torch','isaaclab','isaacsim') for x in sys.modules)")
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)

    def test_defaults_are_disabled_and_validation_is_a_copy(self):
        value = {"profiles": copy.deepcopy(probe.RL_PROBE_DEFAULTS["profiles"])}
        original = copy.deepcopy(value)
        result = probe.validate_probe_settings(value)
        result["profiles"][0]["name"] = "mutated"
        self.assertEqual(value, original)
        self.assertFalse(probe.validate_probe_settings({})["enabled"])
        self.assertEqual(len(original["profiles"]), 4)

    def test_unknown_training_randomization_budget_and_guard_keys_are_rejected(self):
        for key, value in (("allow_source_mismatch", True), ("training_allowed", True),
                           ("training_position_jitter_m", .01), ("disable_shield", True),
                           ("optimizer_steps", 1), ("optimizer_steps", False),
                           ("num_envs", 16), ("episodes_per_env", 5), ("max_control_steps", 365),
                           ("guided_preload_controls", 2), ("guided_probe_height_m", .01),
                           ("guided_probe_speed_m_s", .02), ("guided_force_relief_step_rad", .05),
                           ("max_wall_time_s", 3601), ("max_wall_time_s", float("nan"))):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                probe.validate_probe_settings({key: value})
        profiles = copy.deepcopy(probe.RL_PROBE_DEFAULTS["profiles"])
        profiles[0]["control_guard_version"] = "none"
        with self.assertRaises(ValueError):
            probe.validate_probe_settings({"profiles": profiles})

    def test_guided_target_is_explicit_finite_and_never_a_hidden_training_control(self):
        with self.assertRaises(ValueError):
            probe.validate_probe_settings({"enabled": True})
        for value in ([0.] * 6, [True] * 7, [float("inf")] * 7):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.validate_probe_settings({"guided_hand_target_rad": value})
        self.assertEqual(probe.validate_probe_settings({"guided_hand_target_rad": [0] * 7})["guided_hand_target_rad"], [0.] * 7)

    def test_source_policy_pins_and_parent_traversal_cannot_be_overridden(self):
        with self.assertRaises(ValueError):
            probe.validate_probe_settings({"reference_source_tree_sha256": "0" * 64})
        with self.assertRaises(ValueError):
            probe.validate_probe_settings({"reference_checkpoint_path": "../other.pt"})


class ProbeReferenceTests(unittest.TestCase):
    def test_preflight_never_grants_physics_and_snapshot_contains_only_three_references(self):
        with fixture() as (config, _, _, _, _):
            result = probe.verify_probe_reference(config)
            self.assertTrue(result["reference_verified"])
            self.assertFalse(result["checkpoint_metadata_verified"])
            self.assertFalse(result["current_contract_verified"])
            self.assertFalse(result["ready_for_physical_probe"])
            paths = probe.reference_snapshot_files(config)
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(isinstance(path, Path) and path.is_absolute() for path in paths))

    def test_explicit_counterfactual_bridge_does_not_forge_source_identical_eval(self):
        with fixture() as (config, infos, current, _, _):
            before = copy.deepcopy((config, infos, current))
            result = probe.verify_probe_reference(config, infos, current)
            self.assertTrue(result["ready_for_physical_probe"])
            self.assertFalse(result["source_identical_trained_eval"])
            self.assertEqual(result["optimizer_steps"], 0)
            self.assertEqual((config, infos, current), before)
            self.assertNotEqual(result["reference_source_tree_sha256"], result["current_source_tree_sha256"])

    def test_reference_checkpoint_bytes_tamper_is_rejected(self):
        with fixture() as (config, infos, current, _, _):
            Path(config["rl_diagnostics"]["reference_checkpoint_path"]).write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                probe.verify_probe_reference(config, infos, current)

    def test_manifest_digest_checked_even_if_file_pin_is_updated(self):
        with fixture() as (config, _, _, manifest, pins):
            manifest["files"][0]["bytes"] += 1
            path = Path(config["rl_diagnostics"]["reference_manifest_path"])
            path.write_text(json.dumps(manifest))
            pins["reference_manifest_sha256"] = config["rl_diagnostics"]["reference_manifest_sha256"] = probe._file_hash(path)
            with self.assertRaisesRegex(ValueError, "canonical declaration"):
                probe.verify_probe_reference(config)

    def test_source_tree_uses_old_windows_path_identity(self):
        with fixture() as (_, _, _, manifest, _):
            posix = {x["path"][len("src/g1_vla/"):]: x["sha256"] for x in manifest["files"]
                     if x["path"].startswith("src/g1_vla/")}
            self.assertNotEqual(probe._reference_source_identity(manifest), probe._canonical(posix))
            self.assertEqual(probe._reference_source_identity(manifest), probe._canonical({k.replace("/", "\\"): v for k, v in posix.items()}))

    def test_config_control_PD_metadata_or_randomization_change_is_rejected(self):
        for change in ("anchor", "pd", "metadata", "randomization"):
            with self.subTest(change=change), fixture() as (config, _, _, _, _):
                if change == "anchor": config["control"]["max_anchor_delta_rad"] = .2
                elif change == "pd": config["expert"]["pd"]["effort_limit_nm"] = 300.
                elif change == "randomization": config["rl_teacher"]["training_position_jitter_m"] = .01
                else: Path(config["action"]["joint_metadata_path"]).write_text("new units/order")
                with self.assertRaisesRegex(ValueError, "semantics changed"):
                    probe.verify_probe_reference(config)

    def test_checkpoint_metadata_four_identities_and_actual_update_counts_are_strict(self):
        for key, value in (("source_tree_sha256", "0" * 64), ("training_config_sha256", "0" * 64),
                           ("policy_contract_sha256", "0" * 64), ("runner_version", "different"),
                           ("completed_updates", 79), ("optimizer_steps", 0), ("completed_updates", 78.0)):
            with self.subTest(key=key), fixture() as (config, infos, current, _, _):
                infos[key] = value
                with self.assertRaisesRegex(ValueError, "metadata"):
                    probe.verify_probe_reference(config, infos, current)

    def test_only_the_explicit_control_guard_field_may_differ(self):
        for change in ("pd", "units", "reset", "joint_order", "unknown"):
            with self.subTest(change=change), fixture() as (config, infos, current, _, _):
                policy = current["policy_contract"]
                if change == "pd": policy["actual_pd_specifications"]["joint"]["kp"] = 300.
                elif change == "units": policy["action_units"] = "degrees"
                elif change == "reset": policy["resolved_settings"]["train_xy_jitter_m"] = .01
                elif change == "joint_order": policy["action_joint_names"].reverse()
                else: policy["new_unverified_setting"] = True
                current["policy_contract_sha256"] = probe._canonical(policy)
                with self.assertRaisesRegex(ValueError, "Unexpected environment"):
                    probe.verify_probe_reference(config, infos, current)

    def test_both_declared_guard_versions_are_valid_and_missing_guard_is_not(self):
        with fixture() as (config, infos, current, _, _):
            for guard in ("legacy_scalar_v1", "box_halfspace_projected_v2"):
                current["policy_contract"]["control_guard_version"] = guard
                current["policy_contract_sha256"] = probe._canonical(current["policy_contract"])
                self.assertTrue(probe.verify_probe_reference(config, infos, current)["ready_for_physical_probe"])
            current["policy_contract"].pop("control_guard_version")
            with self.assertRaisesRegex(ValueError, "control_guard_version"):
                probe.verify_probe_reference(config, infos, current)

    def test_current_hash_cannot_be_fabricated_and_contract_hash_cannot_be_stale(self):
        with fixture() as (config, infos, current, _, _):
            current["source_tree_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "truthfully"):
                probe.verify_probe_reference(config, infos, current)
            current["source_tree_sha256"] = probe._current_source_identity()
            current["policy_contract_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "digest"):
                probe.verify_probe_reference(config, infos, current)


if __name__ == "__main__":
    unittest.main()
