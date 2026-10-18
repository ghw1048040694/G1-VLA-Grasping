"""Tamper/ABI/admission tests, without Torch, Isaac or physical stepping."""

import copy
from contextlib import contextmanager
import json
import math
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
        config = {"paths": {"root": str(root)}, "rl_teacher": {"physics_protocol": probe._OLD_PHYSICS},
                  "action": {"joint_metadata_path": str(metadata)},
                  "control": {"physics_dt_s": .002, "physics_substeps": 33, "max_anchor_delta_rad": .08},
                  "expert": {"pd": {"effort_limit_nm": 1.4}}}
        policy = {"physics_protocol": probe._OLD_PHYSICS, "observation_dim": 81, "action_dim": 14, "quaternion_order": "xyzw",
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


class FullLiftProbeSettingsTests(unittest.TestCase):
    @staticmethod
    def settings(full=False, protocol="full_lift8_v1"):
        from g1_vla.rl_effort_protocol import (effort_grasp_contract,
            contact_transition_control_contract, full_lift_protocol_contract,
            full_lift_effort_grasp_contract)
        value = copy.deepcopy(probe.RL_PROBE_DEFAULTS)
        value.update(enabled=True, profile_plan="balanced_effort_v6",
            backend="g1_control_physics_diagnostics_v6", candidate_path="candidate.json",
            candidate_sha256="a"*64, candidate_id="unit_candidate", guided_hand_target_rad=[0.]*7,
            profiles=copy.deepcopy(probe._PLANS["balanced_effort_v6"][1]),
            effort_grasp_controller=effort_grasp_contract(), episodes_per_env=1, max_control_steps=91,
            contact_acquisition={"version": "g1_bounded_measured_touch_search_v2",
                "reference_closing_rate_rad_s": .04, "maximum_extra_closure_rad": .02,
                "raw_force_stop_n": .02, "start_phase": "final_stage_duration_complete",
                "original_actual_geometry_tolerance_rad": .025,
                "requires_no_physical_loaded_point": True, "all_original_acceptance_gates_unchanged": True},
            contact_domain_posture_guard={"version": "g1_contact_domain_nullspace_posture_guard_v1",
                "hand_joint_index": 2, "activation_margin_rad": .006,
                "restoring_stiffness_nm_rad": 2., "maximum_guard_joint_torque_nm": .004,
                "minimum_nullspace_joint_component": .05, "svd_tolerance_m": .0001,
                "original_actual_geometry_tolerance_rad": .025,
                "protect_current_normal_and_world_z": True,
                "requires_all_groups_physically_loaded": True,
                "requires_original_contact_safe": True, "all_original_acceptance_gates_unchanged": True},
            contact_transition_control=contact_transition_control_contract())
        if full:
            value["full_lift"] = full_lift_protocol_contract()
            value["effort_grasp_controller"] = full_lift_effort_grasp_contract(protocol)
            value.update({key: value["effort_grasp_controller"]["evaluation"][key]
                          for key in ("episodes_per_env", "max_control_steps")})
        return value

    def test_optional_declarations_preserve_legacy_default(self):
        settings = probe.validate_probe_settings({})
        self.assertIsNone(settings["full_lift"])
        self.assertIsNone(settings["contact_transition_control"])
        self.assertEqual(settings["max_control_steps"], 364)
        old = self.settings(); old.pop("full_lift"); old.pop("contact_transition_control")
        settings = probe.validate_probe_settings(old)
        self.assertIsNone(settings["full_lift"])
        self.assertIsNone(settings["contact_transition_control"])
        self.assertEqual(settings["max_control_steps"], 91)

    def test_relative_damping_requires_original_compliant_pads_and_material_values(self):
        from g1_vla.rl_effort_protocol import contact_transition_control_contract
        value = self.settings()
        value["contact_transition_control"] = contact_transition_control_contract(
            "g1_coordinated_contact_transition_v3_relative_viscous_preload")
        with self.assertRaises(ValueError): probe.validate_probe_settings(value)
        material = value["effort_grasp_controller"]["material"]
        material.update(variant="finger_compliant_v5", compliant_contact_stiffness_n_m=1000.,
            compliant_contact_damping_ns_m=.8)
        self.assertEqual(probe.validate_probe_settings(value)["contact_transition_control"], value["contact_transition_control"])
        for key, changed in (("compliant_contact_stiffness_n_m", 999.), ("compliant_contact_damping_ns_m", .9)):
            original = material[key]; material[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError): probe.validate_probe_settings(value)
            material[key] = original

    def test_full_lift_quotas_declarations_and_original_probe_are_independent(self):
        for protocol, repeats, steps in [("full_lift8_v1", 1, 213), ("full_lift32_v1", 4, 852)]:
            value = self.settings(True, protocol)
            before = copy.deepcopy(value)
            result = probe.validate_probe_settings(value)
            self.assertEqual(result["episodes_per_env"], repeats)
            self.assertEqual(result["max_control_steps"], steps)
            self.assertEqual(result["guided_probe_height_m"], .0025)
            self.assertEqual(result["guided_preload_controls"], 3)
            self.assertFalse(result["training_allowed"])
            self.assertEqual(result["optimizer_steps"], 0)
            result["full_lift"]["hold_duration_s"] = .001
            self.assertEqual(value, before)

    def test_named_16s_evaluation_requires_matching_task_deadline(self):
        from g1_vla.rl_effort_protocol import full_lift_protocol_contract
        for name in ("full_lift8_v2", "full_lift32_v2"):
            value = self.settings(True, name)
            with self.subTest(name=name), self.assertRaises(ValueError):
                probe.validate_probe_settings(value)
            value["full_lift"] = full_lift_protocol_contract(
                "g1_contact_qualified_full_lift_v4_native_arm_16s")
            result = probe.validate_probe_settings(value)
            self.assertEqual(result["full_lift"]["episode_length_s"], 16.)
            self.assertEqual(result["effort_grasp_controller"]["evaluation"]["nominal_episode_budget_s"], 16.)
        value = self.settings(True)
        value["full_lift"] = full_lift_protocol_contract("g1_contact_qualified_full_lift_v4_native_arm_16s")
        with self.assertRaises(ValueError):
            probe.validate_probe_settings(value)

    def test_new_task_cannot_use_local_effort_or_omit_task_declaration(self):
        for full in (True, False):
            value = self.settings(full)
            value["full_lift"] = self.settings(not full)["full_lift"]
            with self.subTest(full=full), self.assertRaises(ValueError):
                probe.validate_probe_settings(value)

    def test_transition_and_main_lift_require_original_guard_and_acquisition(self):
        for full in (False, True):
            for missing in ("contact_acquisition", "contact_domain_posture_guard"):
                value = self.settings(full); value[missing] = None
                with self.subTest(full=full, missing=missing), self.assertRaises(ValueError):
                    probe.validate_probe_settings(value)
        value = self.settings(True); value["contact_transition_control"] = None
        with self.assertRaises(ValueError):
            probe.validate_probe_settings(value)
        for key in ("contact_transition_control", "full_lift"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.validate_probe_settings({key: self.settings(True)[key]})

    def test_full_task_cannot_change_original_layouts_force_or_optimizer_budget(self):
        for key, bad in [("max_control_steps", 91), ("episodes_per_env", 4),
                         ("num_envs", 32), ("optimizer_steps", 1), ("training_allowed", True),
                         ("guided_probe_height_m", .1)]:
            value = self.settings(True); value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.validate_probe_settings(value)

        for key in ("full_lift", "contact_transition_control"):
            value = self.settings(True); value[key]["hidden_force_override"] = True
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.validate_probe_settings(value)

    @contextmanager
    def native_contract_fixture(self, full=True):
        """Synthetic native readbacks; no simulator or physical admission."""
        from g1_vla.rl_protocol import RL_PHYSICS_EFFORT_GRASP
        settings = probe.validate_probe_settings(self.settings(full))
        contract = settings["effort_grasp_controller"]
        required = {"group_force_max_n": 1.5, "max_penetration_m": .003,
                    "opposition_dot_max": -.6, "max_abs_normal_z": .6,
                    "support_window_s": .198, "support_clearance_min_m": .002,
                    "episode_length_s": 6., "lift_height_m": .006,
                    "probe_clearance_max_m": .006, "contact_offset_m": .001, "rest_offset_m": 0.}
        original = {"asset_sha256": "b"*64, "seed_state_sha256": "c"*64,
                    "resolved_settings": required}
        settings["reference_policy_contract_sha256"] = probe._canonical(original)
        old = {"policy_contract": original, "policy_contract_sha256": probe._canonical(original)}
        reset = {"joint_"+str(i): 0. for i in range(43)}
        candidate = {"asset_sha256": original["asset_sha256"],
            "oldseed_sha256": original["seed_state_sha256"], "candidate_id": settings["candidate_id"],
            "geometry_evidence": {"source_detail": {"stage_keyframes": [
                {"target_rad": [0.]*7}, {"target_rad": [0.]*7}]}},
            "reset_joint_positions_rad": reset, "closing_axis_hand7": [-1.,-1.,-1.,1.,-1.,1.,1.]}
        actual_settings = {**required, "contact_offset_m": 0., "rest_offset_m": -1e-6}
        if full:
            actual_settings["episode_length_s"] = 14.
        actual = {"effort_grasp_controller": copy.deepcopy(contract),
            "contact_generation": copy.deepcopy(contract["contact_generation"]),
            "explicit_reset_resource_sha256": settings["candidate_sha256"],
            "resolved_settings": actual_settings,
            "asset_sha256": original["asset_sha256"], "seed_state_sha256": original["seed_state_sha256"],
            "full_lift": copy.deepcopy(settings["full_lift"]),
            "contact_transition_control": copy.deepcopy(settings["contact_transition_control"])}
        audit = {"policy_contract": actual, "source_tree_sha256": "d"*64,
            "physics_version": RL_PHYSICS_EFFORT_GRASP,
            "policy_contract_sha256": probe._canonical(actual),
            "diagnostic_effort_drive_readback": {"verified": True},
            "diagnostic_material_readback": {"verified": True},
            "diagnostic_contact_generation_readback": {"verified": True},
            "initial_state": {"joint_positions_rad": reset},
            "fixed_cluster_collision_filter_override": {"current_stage_readback": {
                "available": True, "num_envs": 8, "excluded_pair_count_per_env": 30,
                "excluded_pair_count_total": 240, "source_graph_verified_env_count": 8,
                "global_self_collision_readback": {str(i): {"enabled_self_collisions": True} for i in range(8)}}}}
        config = {"rl_teacher": {"physics_protocol": RL_PHYSICS_EFFORT_GRASP,
                                  "contact_offset_m": 0., "rest_offset_m": -1e-6}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"candidate.json"; path.write_text(json.dumps(candidate))
            with mock.patch.object(probe, "_read_pinned", side_effect=[(path, {}), (path, old)]), \
                 mock.patch.object(probe, "_path", return_value=path), \
                 mock.patch.object(probe, "_file_hash", return_value=settings["candidate_sha256"]), \
                 mock.patch.object(probe, "_reference_source_identity", return_value=settings["reference_source_tree_sha256"]), \
                 mock.patch.object(probe, "_current_source_identity", return_value="d"*64):
                yield config, settings, audit

    def test_native_full_readback_admits_only_explicit_duration_change(self):
        for full in (False, True):
            with self.subTest(full=full), self.native_contract_fixture(full) as (config, settings, audit):
                value = probe._verify_effort_probe_reference(config, settings, audit)
                self.assertTrue(value["ready_for_physical_probe"])
                self.assertEqual(value["full_lift"], settings["full_lift"])
                self.assertFalse(value["actor_loaded"])
                self.assertEqual(value["optimizer_steps"], 0)

    def test_native_full_and_transition_readbacks_cannot_be_missing_or_altered(self):
        for key in ("full_lift", "contact_transition_control"):
            with self.subTest(key=key), self.native_contract_fixture() as (config, settings, audit):
                actual = audit["policy_contract"]; actual.pop(key)
                audit["policy_contract_sha256"] = probe._canonical(actual)
                with self.assertRaisesRegex(ValueError, "control/task declaration"):
                    probe._verify_effort_probe_reference(config, settings, audit)

    def test_native_full_task_cannot_relabel_local_gate_settings(self):
        for key, bad in [("episode_length_s", 6.), ("lift_height_m", .1),
                         ("probe_clearance_max_m", .1145), ("group_force_max_n", 2.),
                         ("support_window_s", .1)]:
            with self.subTest(key=key), self.native_contract_fixture() as (config, settings, audit):
                actual = audit["policy_contract"]; actual["resolved_settings"][key] = bad
                audit["policy_contract_sha256"] = probe._canonical(actual)
                with self.assertRaisesRegex(ValueError, "original local qualification"):
                    probe._verify_effort_probe_reference(config, settings, audit)


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



def geometry_fixture(config, infos, current, pins):
    """Synthetic offline audit evidence, never real geometry qualification."""
    policy = infos["policy_contract"]
    for name in probe._HAND_JOINT_ORDER:
        policy["native_joint_axis_frame_unit_schema"][name] = {
            "lower_limit_degrees": -90., "upper_limit_degrees": 90.}
    policy_sha = probe._canonical(policy)
    audit = Path(config["rl_diagnostics"]["reference_environment_audit_path"])
    audit.write_text(json.dumps({"policy_contract": policy, "policy_contract_sha256": policy_sha,
        "initial_state": {"joint_positions_rad": {name: 0. for name in probe._HAND_JOINT_ORDER}}}))
    pins["reference_policy_contract_sha256"] = policy_sha
    pins["reference_environment_audit_sha256"] = probe._file_hash(audit)
    config["rl_diagnostics"].update(pins)
    infos["policy_contract_sha256"] = policy_sha
    actual = copy.deepcopy(policy)
    actual["control_guard_version"] = "box_halfspace_projected_v2"
    current.update(policy_contract=actual, policy_contract_sha256=probe._canonical(actual))
    evidence = {"protocol": "offline_native_usd_grasp_geometry_v1",
        "qualification": "candidate_only_not_physics_qualified", "units": "m_rad",
        "hand_joint_limits_rad": [[-math.pi/2, math.pi/2] for _ in range(7)],
        "source_detail": {"method": "synthetic unit fixture only"},
        "finger_surface_evidence": {}}
    for finger, distal in (("thumb", "thumb_2"), ("index", "index_1"), ("middle", "middle_1")):
        evidence["finger_surface_evidence"][finger] = {
            "contact_body": "right_hand_" + distal + "_link",
            "minimum_distance_to_cube_m": .0001, "closest_point_world_m3": [0., 0., 0.],
            "cube_closest_point_world_m3": [.0001, 0., 0.], "closing_direction_world3": [0., 1., 0.],
            "method": "synthetic separated closest surfaces"}
    evidence["source_detail"].update(no_object_state_change=True, palm_offset_xyz_world_m=[0.]*3,
        palm_offset_rpy_world_rad=[0.]*3, stage_keyframes=[{"name": "source", "target_rad": [0.]*7}] +
        [{"name": "stage"+str(i), "target_rad": [0.]*7, "duration_s": .4} for i in range(5)])
    candidate = {"schema_version": 1, "candidate_id": "synthetic_candidate_001",
        "asset_sha256": policy["asset_sha256"], "oldseed_sha256": policy["seed_state_sha256"],
        "hand_joint_order": list(probe._HAND_JOINT_ORDER), "target_rad": [0.] * 7,
        "geometry_evidence": evidence,
        "known_geometric_overlaps": [{"pair": ["camera", "thumb0"], "scope": "authored hull overlap only"}],
        "admission_limitations": ["Synthetic candidate only; no real contact qualification"]}
    path = audit.parent / "candidate.json"
    settings = config["rl_diagnostics"]
    settings.update(backend="g1_control_physics_diagnostics_v2", profile_plan="geometry_pose_only_v2",
                    candidate_id=candidate["candidate_id"], candidate_path=str(path), guided_target_lead_max_rad=.006)
    def save():
        path.write_text(json.dumps(candidate))
        settings["candidate_sha256"] = probe._file_hash(path)
    save()
    return candidate, path, save


class GeometryPlanTests(unittest.TestCase):
    def test_v1_implicit_plan_remains_four_way_and_v2_is_an_explicit_single_plan(self):
        first = probe.validate_probe_settings({})
        self.assertEqual(first["profile_plan"], "four_way_v1")
        self.assertEqual(len(first["profiles"]), 4)
        with self.assertRaisesRegex(ValueError, "backend"):
            probe.validate_probe_settings({"profile_plan": "geometry_pose_only_v2"})
        with self.assertRaises(ValueError):
            probe.validate_probe_settings({"profile_plan": "unknown_subset"})
        with fixture() as (config, infos, current, _, pins):
            geometry_fixture(config, infos, current, pins)
            cfg = probe.validate_probe_settings(config["rl_diagnostics"])
            self.assertEqual(cfg["profiles"], [first["profiles"][3]])
            for replacement in ([], first["profiles"], [first["profiles"][2]]):
                with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                    probe.validate_probe_settings({**config["rl_diagnostics"], "profiles": replacement})

    def test_v2_snapshot_adds_only_verified_candidate_and_never_forges_four_way_or_physics(self):
        with fixture() as (config, infos, current, _, pins):
            _, path, _ = geometry_fixture(config, infos, current, pins)
            pre = probe.verify_probe_reference(config)
            self.assertFalse(pre["ready_for_physical_probe"])
            result = probe.verify_probe_reference(config, infos, current)
            self.assertTrue(result["ready_for_physical_probe"])
            self.assertFalse(result["four_way_comparison_declared"])
            self.assertFalse(result["four_way_comparison_complete"])
            self.assertFalse(result["source_identical_trained_eval"])
            self.assertEqual(result["declared_profile_names"], ["guided_hand_pose_hold"])
            self.assertFalse(result["candidate_reference"]["geometry_recomputed_by_protocol"])
            self.assertFalse(result["candidate_reference"]["physical_qualification"])
            self.assertFalse(result["candidate_reference"]["self_collision_free_claimed"])
            self.assertTrue(result["candidate_reference"]["known_geometric_overlaps"])
            files = probe.reference_snapshot_files(config)
            self.assertEqual(len(files), 4)
            self.assertEqual(files[-1], path.resolve())

    def test_candidate_pin_is_rechecked_after_preflight_for_every_native_bridge(self):
        with fixture() as (config, infos, current, _, pins):
            _, path, _ = geometry_fixture(config, infos, current, pins)
            probe.verify_probe_reference(config)
            path.write_text(path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "candidate_sha256"):
                probe.verify_probe_reference(config, infos, current)

    def test_candidate_identity_and_exact_seven_joint_target_are_checked_even_with_new_hash(self):
        for key in ("candidate_id", "asset_sha256", "oldseed_sha256", "hand_joint_order", "target_rad", "schema_version"):
            with self.subTest(key=key), fixture() as (config, infos, current, _, pins):
                candidate, _, save = geometry_fixture(config, infos, current, pins)
                if key == "hand_joint_order": candidate[key].reverse()
                elif key == "target_rad": candidate[key][0] = .1
                elif key == "schema_version": candidate[key] = True
                else: candidate[key] = "other"
                save()
                with self.assertRaises(ValueError): probe.verify_probe_reference(config, infos, current)

    def test_offline_geometry_evidence_is_not_an_unchecked_boolean(self):
        for change in ("missing", "qualification", "finite", "method", "points", "bounds", "body"):
            with self.subTest(change=change), fixture() as (config, infos, current, _, pins):
                candidate, _, save = geometry_fixture(config, infos, current, pins)
                e = candidate["geometry_evidence"]
                if change == "missing": candidate["geometry_evidence"] = {"verified": True}
                elif change == "qualification": e["qualification"] = "physically_qualified"
                elif change == "finite": e["source_detail"]["unknown"] = float("nan")
                elif change == "method": e["finger_surface_evidence"]["thumb"]["method"] = ""
                elif change == "points": e["finger_surface_evidence"]["thumb"]["cube_closest_point_world_m3"][0] = 1.
                elif change == "bounds": e["hand_joint_limits_rad"][0] = [-100., 100.]
                else: e["finger_surface_evidence"]["thumb"]["contact_body"] = "palm"
                save()
                with self.assertRaises(ValueError): probe.verify_probe_reference(config, infos, current)

    def test_candidate_stages_pin_seed_targets_bounds_horizon_and_no_object_or_palm_changes(self):
        for change in ("source", "last", "bound", "count", "duration", "total", "object", "palm", "duplicate"):
            with self.subTest(change=change), fixture() as (config, infos, current, _, pins):
                candidate, _, save = geometry_fixture(config, infos, current, pins)
                d = candidate["geometry_evidence"]["source_detail"]
                stages = d["stage_keyframes"]
                if change == "source": stages[0]["target_rad"][0] += .0002
                elif change == "last": stages[-1]["target_rad"][0] += .0002
                elif change == "bound": stages[2]["target_rad"][0] = 9.
                elif change == "count": stages.pop()
                elif change == "duration": stages[2]["duration_s"] = 0.
                elif change == "total": stages[2]["duration_s"] = 9.
                elif change == "object": d["no_object_state_change"] = False
                elif change == "palm": d["palm_offset_xyz_world_m"][0] = .0001
                else: stages[2]["name"] = stages[1]["name"]
                save()
                with self.assertRaises(ValueError): probe.verify_probe_reference(config)

    def test_v2_cannot_select_legacy_guard_or_change_source_config_contract(self):
        with fixture() as (config, infos, current, _, pins):
            geometry_fixture(config, infos, current, pins)
            current["policy_contract"]["control_guard_version"] = "legacy_scalar_v1"
            current["policy_contract_sha256"] = probe._canonical(current["policy_contract"])
            with self.assertRaisesRegex(ValueError, "control_guard_version"):
                probe.verify_probe_reference(config, infos, current)
            config["control"]["max_anchor_delta_rad"] = 1.
            with self.assertRaisesRegex(ValueError, "semantics changed"):
                probe.verify_probe_reference(config)

    def test_candidate_symlink_and_unknown_json_fields_are_rejected(self):
        with fixture() as (config, infos, current, _, pins):
            candidate, path, save = geometry_fixture(config, infos, current, pins)
            candidate["unrecognized"] = "not approved"
            save()
            with self.assertRaises(ValueError): probe.verify_probe_reference(config)
            candidate.pop("unrecognized"); save()
            link = path.with_name("candidate_link.json"); link.symlink_to(path)
            config["rl_diagnostics"]["candidate_path"] = str(link)
            with self.assertRaisesRegex(ValueError, "regular"):
                probe.verify_probe_reference(config)

    def test_lead_is_explicit_bounded_and_v1_unchanged(self):
        with self.assertRaises(ValueError):
            probe.validate_probe_settings({"guided_target_lead_max_rad": .003})
        with fixture() as (config, infos, current, _, pins):
            geometry_fixture(config, infos, current, pins)
            for lead in (None, .003, .007, float("nan"), True):
                with self.subTest(lead=lead), self.assertRaises(ValueError):
                    probe.validate_probe_settings({**config["rl_diagnostics"], "guided_target_lead_max_rad": lead})

    def test_v1_load_config_path_resolution_roundtrips_to_reference_verification(self):
        from g1_vla import config as project_config
        template = Path(__file__).resolve().parents[1] / "configs/experiment.json"
        raw = json.loads(template.read_text())
        raw.pop("rl_diagnostics", None)
        with fixture() as (config, infos, current, _, pins):
            root = Path(config["paths"]["root"])
            raw["paths"]["root"] = str(root)
            config_path = root / "full_config.json"
            config_path.write_text(json.dumps(raw))
            # Include root's forthcoming pathresolver field without editing it.
            path_fields = copy.deepcopy(project_config._PATH_FIELDS)
            path_fields["rl_diagnostics"] += ("candidate_path",)
            with mock.patch.object(project_config, "_PATH_FIELDS", path_fields), \
                    mock.patch.object(project_config, "RL_PROBE_DEFAULTS", {**probe.RL_PROBE_DEFAULTS, **pins}):
                loaded = project_config.load_config(config_path)
            self.assertIsNone(loaded["rl_diagnostics"]["candidate_path"])
            probe.validate_probe_settings(loaded["rl_diagnostics"])
            # Preserve resolved defaults, then restore this fixture's pinned
            # reference identities and explicit guided target for admission.
            resolved = loaded["rl_diagnostics"]
            resolved.update(config["rl_diagnostics"])
            config["rl_diagnostics"] = resolved
            self.assertTrue(probe.verify_probe_reference(config, infos, current)["ready_for_physical_probe"])

    def test_v1_never_ignores_a_new_geometry_candidate(self):
        for key in ("candidate_path", "candidate_sha256", "candidate_id"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                probe.validate_probe_settings({key: "unexpected"})



@contextmanager
def fixed_cluster_fixture():
    """Synthetic asset pin; real allowlist topology retained, no simulator."""
    with fixture() as (config, infos, current, manifest, pins):
        candidate, _, _ = geometry_fixture(config, infos, current, pins)
        expected = probe._declared_fixed_filter_contract()
        expected["contract"]["asset_sha256"] = infos["policy_contract"]["asset_sha256"]
        expected["sha256"] = probe._canonical(expected["contract"])
        config["rl_diagnostics"].update(profile_plan="geometry_pose_fixed_cluster_v3",
            backend="g1_control_physics_diagnostics_v3")
        candidate_sha = config["rl_diagnostics"]["candidate_sha256"]
        config["rl_teacher"]["physics_protocol"] = probe._FIXED_PHYSICS
        current["physics_version"] = probe._FIXED_PHYSICS
        current["policy_contract"].update(physics_protocol=probe._FIXED_PHYSICS,
            fixed_cluster_collision_filter=copy.deepcopy(expected))
        current["policy_contract_sha256"] = probe._canonical(current["policy_contract"])
        with mock.patch.object(probe, "_V3_CANDIDATE_SHA256", candidate_sha), \
                mock.patch.object(probe, "_declared_fixed_filter_contract", return_value=expected):
            yield config, infos, current, expected


class FixedClusterModelPlanTests(unittest.TestCase):
    def test_official_expected_contract_is_independent_fifteen_pair_graph_pin(self):
        expected = probe._declared_fixed_filter_contract()
        self.assertEqual(expected["sha256"], "04dd990d50f4632012408b13747313a5504614e47a13d05d9540c31ef66fb770")
        self.assertEqual(probe._canonical(expected["contract"]), expected["sha256"])
        self.assertEqual(len(expected["contract"]["excluded_body_pairs"]), 15)
        self.assertEqual(expected["source_graph_sha256"], "1bc9a5d33856280a489137634f3ddcbb84196ef231b6dfa902acbd271eac600c")
        for pair in expected["contract"]["excluded_body_pairs"]:
            self.assertEqual(pair, sorted(pair))
            self.assertFalse(any(x in ("Cube", "Table") or x.startswith("left_") for x in pair))
        expected["contract"]["excluded_body_pairs"].clear()
        self.assertEqual(len(probe._declared_fixed_filter_contract()["contract"]["excluded_body_pairs"]), 15)

    def test_v3_preserves_candidate_and_exact_declared_plan(self):
        v = {"backend": "g1_control_physics_diagnostics_v3", "profile_plan": "geometry_pose_fixed_cluster_v3",
             "candidate_path": "candidate.json", "candidate_id": "declared", "candidate_sha256": probe._V3_CANDIDATE_SHA256,
             "guided_target_lead_max_rad": .006, "guided_hand_target_rad": [0.] * 7}
        result = probe.validate_probe_settings(v)
        self.assertEqual(result["profiles"], [probe.RL_PROBE_DEFAULTS["profiles"][3]])
        for field, value in (("candidate_sha256", "0"*64), ("backend", "g1_control_physics_diagnostics_v2"),
                             ("guided_target_lead_max_rad", .007), ("profiles", probe.RL_PROBE_DEFAULTS["profiles"])):
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_probe_settings({**v, field: value})

    def test_exact_new_model_bridge_preserves_true_current_and_old_rest_semantics(self):
        with fixed_cluster_fixture() as (config, infos, current, expected):
            before = copy.deepcopy((config, infos, current))
            result = probe.verify_probe_reference(config, infos, current)
            self.assertTrue(result["ready_for_physical_probe"])
            self.assertFalse(result["source_identical_trained_eval"])
            self.assertTrue(result["physics_model_comparison"]["model_changed"])
            self.assertEqual(result["physics_model_comparison"]["fixed_cluster_collision_filter"], expected)
            self.assertEqual(result["current_config_semantic_sha256"], probe._semantic_training_identity(config))
            self.assertNotEqual(result["current_config_semantic_sha256"], result["reference_training_config_sha256"])
            self.assertEqual(result["reference_rest_config_comparison_sha256"], result["reference_training_config_sha256"])
            self.assertEqual(infos["policy_contract"]["physics_protocol"], probe._OLD_PHYSICS)
            self.assertEqual((config, infos, current), before)

    def test_v3_rejects_unapproved_pairs_hash_graph_asset_or_extra_model_keys(self):
        for change in ("cube", "graph", "hash", "asset", "extra", "missing"):
            with self.subTest(change=change), fixed_cluster_fixture() as (config, infos, current, _):
                declaration = current["policy_contract"]["fixed_cluster_collision_filter"]
                if change == "cube":
                    declaration["contract"]["excluded_body_pairs"].append(["Cube", "right_hand_palm_link"])
                    declaration["sha256"] = probe._canonical(declaration["contract"])
                elif change == "graph": declaration["source_graph_sha256"] = "0"*64
                elif change == "hash": declaration["sha256"] = "0"*64
                elif change == "asset": declaration["contract"]["asset_sha256"] = "0"*64
                elif change == "extra": declaration["new_unverified_model_override"] = True
                else: del current["policy_contract"]["fixed_cluster_collision_filter"]
                current["policy_contract_sha256"] = probe._canonical(current["policy_contract"])
                with self.assertRaisesRegex(ValueError, "fixed-cluster"):
                    probe.verify_probe_reference(config, infos, current)

    def test_declared_model_delta_does_not_allow_PD_obs_reward_or_other_config_changes(self):
        for change in ("pd", "obs", "reward", "unknown", "config_anchor", "config_pd"):
            with self.subTest(change=change), fixed_cluster_fixture() as (config, infos, current, _):
                if change == "pd": current["policy_contract"]["actual_pd_specifications"]["joint"]["kp"] += 1
                elif change == "obs": current["policy_contract"]["observation_dim"] = 82
                elif change == "reward": current["policy_contract"]["resolved_settings"]["reward_dropped"] = True
                elif change == "unknown": current["policy_contract"]["allow_extra_collisions"] = True
                elif change == "config_anchor": config["control"]["max_anchor_delta_rad"] = 1.
                else: config["expert"]["pd"]["effort_limit_nm"] = 9.
                current["policy_contract_sha256"] = probe._canonical(current["policy_contract"])
                with self.assertRaises(ValueError): probe.verify_probe_reference(config, infos, current)

    def test_old_plans_cannot_admit_new_model_and_v3_cannot_claim_old_actual_physics(self):
        for change in ("v3_oldconfig", "v3_oldpolicy", "v3_oldtop", "v2_newphysics", "v2_extra_filter"):
            with self.subTest(change=change), fixed_cluster_fixture() as (config, infos, current, _):
                if change == "v3_oldconfig": config["rl_teacher"]["physics_protocol"] = probe._OLD_PHYSICS
                elif change == "v3_oldpolicy": current["policy_contract"]["physics_protocol"] = probe._OLD_PHYSICS
                elif change == "v3_oldtop": current["physics_version"] = probe._OLD_PHYSICS
                else:
                    config["rl_diagnostics"].update(profile_plan="geometry_pose_only_v2", backend="g1_control_physics_diagnostics_v2")
                    if change == "v2_extra_filter":
                        config["rl_teacher"]["physics_protocol"] = probe._OLD_PHYSICS
                        current["policy_contract"]["physics_protocol"] = probe._OLD_PHYSICS
                current["policy_contract_sha256"] = probe._canonical(current["policy_contract"])
                with self.assertRaises(ValueError): probe.verify_probe_reference(config, infos, current)

    def test_old_checkpoint_metadata_cannot_be_rewritten_as_trained_on_new_physics(self):
        with fixed_cluster_fixture() as (config, infos, current, _):
            infos["policy_contract"]["physics_protocol"] = probe._FIXED_PHYSICS
            with self.assertRaisesRegex(ValueError, "metadata"):
                probe.verify_probe_reference(config, infos, current)


@contextmanager
def compliant_fixture():
    with fixed_cluster_fixture() as (config, infos, current, model):
        controller = probe.compliant_pinch_controller_contract()
        config["rl_diagnostics"].update(
            profile_plan="compliant_pinch_v4", backend="g1_control_physics_diagnostics_v4",
            guided_target_lead_max_rad=None, compliant_pinch_controller=copy.deepcopy(controller))
        current["diagnostic_controller_protocol"] = copy.deepcopy(controller)
        current["control_probe_profile"] = copy.deepcopy(
            probe.validate_probe_settings(config["rl_diagnostics"])["profiles"][0])
        yield config, infos, current, model, controller


class CompliantPinchPlanTests(unittest.TestCase):
    def test_controller_is_a_pinned_copy_of_declared_outer_control_not_native_PD(self):
        controller = probe.compliant_pinch_controller_contract()
        self.assertEqual(controller["version"], "g1_joint_target_force_admittance_pinch_v1")
        self.assertEqual(controller["kind"], "outer_force_admittance_joint_targets")
        self.assertEqual(probe._canonical(controller),
                         "ce31ed56f0435dbc60270a40dde00f55170135806dd03547e0f2583ecb8c99ae")
        self.assertEqual(controller["parameters"]["force_target_n"], .35)
        self.assertEqual(controller["parameters"]["contact_feedback_onset_n"], .02)
        self.assertEqual(controller["parameters"]["adaptive_thumb0_closing_budget_rad"], .07)
        self.assertEqual(controller["parameters"]["stage0_duration_s"], .9)
        self.assertEqual(controller["parameters"]["stage0_contact_ready_controls"], 2)
        self.assertEqual(controller["reference_semantics"]["maximum_retreat_rad"],
                         "actual_most_closed_applied_target")
        self.assertIn("not_physically_loaded_or_support",
                      controller["reference_semantics"]["stage0_contact_force_min_n"])
        controller["parameters"]["force_target_n"] = 5.
        self.assertEqual(probe.compliant_pinch_controller_contract()["parameters"]["force_target_n"], .35)
        with mock.patch.dict(probe._COMPLIANT_PINCH_CONTROLLER["parameters"], force_target_n=.36):
            with self.assertRaisesRegex(ValueError, "pinned identity"):
                probe.compliant_pinch_controller_contract()

    def test_new_plan_declares_exact_single_profile_no_duplicate_lead(self):
        with compliant_fixture() as (config, _, _, _, _):
            result = probe.validate_probe_settings(config["rl_diagnostics"])
            self.assertEqual(result["profiles"], [{"name": "compliant_pinch_pose_hold",
                "action_source": "guided_hand_pose_hold", "control_guard_version": "box_halfspace_projected_v2"}])
            self.assertIsNone(result["guided_target_lead_max_rad"])
            for key, value in (("guided_target_lead_max_rad", .006), ("backend", "g1_control_physics_diagnostics_v3"),
                               ("compliant_pinch_controller", None), ("candidate_sha256", "0"*64),
                               ("profiles", probe.RL_PROBE_DEFAULTS["profiles"]), ("episodes_per_env", 5)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    probe.validate_probe_settings({**config["rl_diagnostics"], key: value})

    def test_every_parameter_is_exact_finite_and_not_a_support_override(self):
        for key in probe.compliant_pinch_controller_contract()["parameters"]:
            with self.subTest(key=key), compliant_fixture() as (config, _, _, _, _):
                changed = config["rl_diagnostics"]["compliant_pinch_controller"]["parameters"]
                changed[key] += .00001
                with self.assertRaises(ValueError): probe.validate_probe_settings(config["rl_diagnostics"])
        for change in ("version", "kind", "extra_top", "missing", "extra_parameter", "boolean", "nan", "inf",
                       "reference", "missing_reference", "float_counter"):
            with self.subTest(change=change), compliant_fixture() as (config, _, _, _, _):
                value = config["rl_diagnostics"]["compliant_pinch_controller"]
                if change in ("version", "kind"): value[change] = "unverified"
                elif change == "extra_top": value["allow_unsafe"] = True
                elif change == "missing": del value["parameters"]["maximum_penetration_m"]
                elif change == "extra_parameter": value["parameters"]["support_window_s"] = .01
                elif change == "boolean": value["parameters"]["force_target_n"] = True
                elif change == "nan": value["parameters"]["force_target_n"] = float("nan")
                elif change == "inf": value["parameters"]["force_target_n"] = float("inf")
                elif change == "reference": value["reference_semantics"]["maximum_retreat_rad"] = "nominal_candidate"
                elif change == "missing_reference": del value["reference_semantics"]["force_feedback"]
                else: value["parameters"]["stage0_contact_ready_controls"] = 2.0
                with self.assertRaises(ValueError): probe.validate_probe_settings(config["rl_diagnostics"])

    def test_adaptive_interval_and_schedule_are_explicit_and_mechanically_bounded(self):
        with compliant_fixture() as (config, infos, current, _, _):
            result = probe.verify_probe_reference(config, infos, current)
            intervention = result["candidate_reference"]["declared_control_interventions"]
            self.assertTrue(intervention["nominal_candidate_file_unchanged"])
            self.assertEqual(intervention["nominal_targets_rad"], [0.] * 7)
            self.assertEqual(intervention["thumb0_declared_target_interval_rad"], [-.07, 0.])
            self.assertEqual(intervention["other_six_extra_closing_budget_rad"], 0.)
            self.assertFalse(intervention["physical_qualification"])
            self.assertFalse(intervention["geometry_recomputed_by_protocol"])
            settings = probe.validate_probe_settings(config["rl_diagnostics"])
            target = [.99] + [0.] * 6
            limits = [[-1.5, 1.5] for _ in range(7)]
            stages = [{"name": "source"}] + [{"duration_s": x} for x in [1.5, .35, .2, .4, .4]]
            checked = probe._compliant_candidate_interventions(settings, target, limits, stages, 2.85)
            self.assertAlmostEqual(checked["thumb0_declared_target_interval_rad"][0], .92)
            self.assertAlmostEqual(checked["controller_nominal_stage_duration_total_s"], 2.25)
            self.assertEqual(checked["candidate_stage0_duration_s"], 1.5)
            self.assertEqual(checked["controller_stage0_duration_s"], .9)
            limits[0][0] = .95
            with self.assertRaisesRegex(ValueError, "native joint bounds"):
                probe._compliant_candidate_interventions(settings, target, limits, stages, 2.85)
            limits[0][0] = -1.5
            stages[1]["duration_s"] = .1
            with self.assertRaisesRegex(ValueError, "episode budget"):
                probe._compliant_candidate_interventions(settings, target, limits, stages, 6.)

    def test_old_plans_keep_their_original_control_and_reject_new_controller(self):
        self.assertIsNone(probe.validate_probe_settings({})["compliant_pinch_controller"])
        with self.assertRaisesRegex(ValueError, "Legacy"):
            probe.validate_probe_settings({"compliant_pinch_controller": probe.compliant_pinch_controller_contract()})
        for plan, backend in (("geometry_pose_only_v2", "g1_control_physics_diagnostics_v2"),
                              ("geometry_pose_fixed_cluster_v3", "g1_control_physics_diagnostics_v3")):
            with self.subTest(plan=plan), compliant_fixture() as (config, _, _, _, _):
                config["rl_diagnostics"].update(profile_plan=plan, backend=backend, guided_target_lead_max_rad=.006)
                with self.assertRaisesRegex(ValueError, "Legacy"):
                    probe.validate_probe_settings(config["rl_diagnostics"])
        with fixture() as (config, infos, current, _, _):
            current["diagnostic_controller_protocol"] = probe.compliant_pinch_controller_contract()
            with self.assertRaisesRegex(ValueError, "Legacy"):
                probe.verify_probe_reference(config, infos, current)

    def test_bridge_verifies_declared_control_separately_and_preserves_support_policy(self):
        with compliant_fixture() as (config, infos, current, model, controller):
            before = copy.deepcopy((config, infos, current))
            pre = probe.verify_probe_reference(config)
            self.assertFalse(pre["ready_for_physical_probe"])
            result = probe.verify_probe_reference(config, infos, current)
            self.assertTrue(result["ready_for_physical_probe"])
            self.assertFalse(result["source_identical_trained_eval"])
            self.assertFalse(result["four_way_comparison_declared"])
            self.assertFalse(result["four_way_comparison_complete"])
            self.assertEqual(result["declared_profile_names"], ["compliant_pinch_pose_hold"])
            self.assertEqual(result["diagnostic_controller_protocol"], controller)
            self.assertEqual(result["diagnostic_controller_protocol_sha256"], probe._canonical(controller))
            self.assertIn("not an environment-measured force or support certificate",
                          result["diagnostic_controller_protocol_source"])
            self.assertEqual(result["physics_model_comparison"]["fixed_cluster_collision_filter"], model)
            self.assertNotEqual(result["current_config_semantic_sha256"], result["reference_training_config_sha256"])
            self.assertEqual(result["reference_rest_config_comparison_sha256"], result["reference_training_config_sha256"])
            self.assertEqual((config, infos, current), before)

    def test_native_declaration_cannot_be_missing_changed_or_hidden_in_policy(self):
        for change in ("missing", "null", "parameter", "extra", "in_policy", "old_profile", "missing_profile"):
            with self.subTest(change=change), compliant_fixture() as (config, infos, current, _, _):
                if change == "missing": del current["diagnostic_controller_protocol"]
                elif change == "null": current["diagnostic_controller_protocol"] = None
                elif change == "parameter": current["diagnostic_controller_protocol"]["parameters"]["force_target_n"] = .36
                elif change == "extra": current["diagnostic_controller_protocol"]["parameters"]["skip_manifold"] = True
                elif change == "in_policy":
                    current["policy_contract"]["diagnostic_controller_protocol"] = current["diagnostic_controller_protocol"]
                    current["policy_contract_sha256"] = probe._canonical(current["policy_contract"])
                elif change == "old_profile": current["control_probe_profile"] = copy.deepcopy(probe._PROFILES[3])
                else: del current["control_probe_profile"]
                with self.assertRaises(ValueError): probe.verify_probe_reference(config, infos, current)

    def test_compliance_cannot_admit_old_physics_or_force_support_changes(self):
        for change in ("old_physics", "force_gate", "support_window", "pd", "reward", "new_checkpoint_policy"):
            with self.subTest(change=change), compliant_fixture() as (config, infos, current, _, _):
                if change == "old_physics": config["rl_teacher"]["physics_protocol"] = probe._OLD_PHYSICS
                elif change == "force_gate": current["policy_contract"]["resolved_settings"]["group_force_max_n"] = 10.
                elif change == "support_window": current["policy_contract"]["resolved_settings"]["support_window_s"] = .01
                elif change == "pd": current["policy_contract"]["actual_pd_specifications"]["joint"]["kp"] = 1.
                elif change == "reward": current["policy_contract"]["resolved_settings"]["reward_support"] = 0.
                else: infos["policy_contract"]["physics_protocol"] = probe._FIXED_PHYSICS
                current["policy_contract_sha256"] = probe._canonical(current["policy_contract"])
                with self.assertRaises(ValueError): probe.verify_probe_reference(config, infos, current)

    def test_ordinary_old_checkpoint_evaluation_still_rejects_new_physics(self):
        from types import SimpleNamespace
        from g1_vla import rl_ppo
        with compliant_fixture() as (config, infos, current, _, _):
            path = config["rl_diagnostics"]["reference_checkpoint_path"]
            settings = {"evaluation": {"checkpoint_path": path, "checkpoint_sha256": probe._file_hash(path)},
                        "device": "cpu", "training_config_sha256": infos["training_config_sha256"],
                        "source_tree_sha256": infos["source_tree_sha256"]}
            runner = mock.Mock()
            tensor_runtime = SimpleNamespace(load=lambda *args, **kwargs: {"infos": infos})
            with mock.patch.object(rl_ppo, "_make_runner", return_value=(runner, None)), \
                    mock.patch.object(rl_ppo, "_collect_evaluation") as collect:
                with self.assertRaisesRegex(ValueError, "policy contract"):
                    rl_ppo._evaluation(None, settings, Path(config["paths"]["root"]), {}, None, tensor_runtime, current)
                runner.alg.load.assert_not_called()
                collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class PhysicsStepPinchPlanTests(unittest.TestCase):
    def physics_fixture(self, config, current):
        c = probe.physics_pinch_controller_contract()
        config["rl_diagnostics"].update(profile_plan="physics_pinch_v5",
            backend="g1_control_physics_diagnostics_v5", compliant_pinch_controller=c,
            guided_target_lead_max_rad=None)
        current["diagnostic_controller_protocol"] = copy.deepcopy(c)
        current["control_probe_profile"] = copy.deepcopy(
            probe.validate_probe_settings(config["rl_diagnostics"])["profiles"][0])
        return c

    def test_new_inner_cadence_and_old_outer_contract_are_separate(self):
        c = probe.physics_pinch_controller_contract()
        self.assertEqual(c["parameters"]["physics_dt_s"], .002)
        self.assertEqual(c["parameters"]["contact_target_lead_max_rad"], .003)
        self.assertEqual(c["parameters"]["stage0_contact_ready_controls"], 2)
        self.assertEqual(c["parameters"]["force_target_n"], .35)
        self.assertEqual(c["parameters"]["force_deadband_n"], .05)
        self.assertNotEqual(c["version"], probe.compliant_pinch_controller_contract()["version"])
        self.assertEqual(probe._canonical(probe.compliant_pinch_controller_contract()),
            "ce31ed56f0435dbc60270a40dde00f55170135806dd03547e0f2583ecb8c99ae")
        c["parameters"]["physics_dt_s"] = .066
        self.assertEqual(probe.physics_pinch_controller_contract()["parameters"]["physics_dt_s"], .002)

    def test_inner_plan_bridge_preserves_original_policy_and_budget(self):
        with compliant_fixture() as (config, infos, current, _, _):
            self.physics_fixture(config, current)
            before = copy.deepcopy((config, infos, current))
            r = probe.verify_probe_reference(config, infos, current)
            self.assertTrue(r["ready_for_physical_probe"])
            self.assertFalse(r["source_identical_trained_eval"])
            self.assertEqual(r["optimizer_steps"], 0)
            self.assertEqual((config, infos, current), before)
            self.assertEqual(probe.validate_probe_settings(config["rl_diagnostics"])["guided_preload_controls"], 3)
            self.assertEqual(infos["policy_contract"]["physics_dt_s"], .002)
            self.assertEqual(infos["policy_contract"]["control_period_s"], .066)

    def test_misdeclared_inner_cadence_and_controller_are_rejected(self):
        for change in ("dt", "lead", "version", "profile", "extra_parameter", "old_contract", "budget"):
            with self.subTest(change=change), compliant_fixture() as (config, infos, current, _, _):
                self.physics_fixture(config, current)
                v = config["rl_diagnostics"]
                if change == "dt": v["compliant_pinch_controller"]["parameters"]["physics_dt_s"] = .066
                elif change == "lead": v["compliant_pinch_controller"]["parameters"]["contact_target_lead_max_rad"] = .012
                elif change == "version": v["compliant_pinch_controller"]["version"] = "unversioned"
                elif change == "profile": current["control_probe_profile"]["name"] = "compliant_pinch_pose_hold"
                elif change == "extra_parameter": v["compliant_pinch_controller"]["parameters"]["group_force_max_n"] = 100.
                elif change == "old_contract": v["compliant_pinch_controller"] = probe.compliant_pinch_controller_contract()
                else: v["max_control_steps"] = 365
                with self.assertRaises(ValueError): probe.verify_probe_reference(config, infos, current)
