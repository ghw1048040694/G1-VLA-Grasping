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


if __name__ == "__main__":
    unittest.main()
