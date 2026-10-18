"""Backend-free safety/provenance checks for effort calibration declarations."""

from copy import deepcopy
import json
import math
import unittest

from g1_vla.rl_effort_protocol import (
    balanced_effort_controller_contract, effort_grasp_contract,
    validate_balanced_effort_controller_contract, validate_effort_grasp_contract,
    validate_contact_domain_posture_guard,
    full_lift_protocol_contract, validate_full_lift_protocol,
    full_lift_effort_grasp_contract, FULL_LIFT_EFFORT_GRASP_VERSION,
    contact_transition_control_contract, validate_contact_transition_control,
    ALWAYS_RIGID_BODY_NAMES, FINGER_BODY_NAMES,
)


class EffortGraspProtocolTests(unittest.TestCase):
    def setUp(self):
        self.value = effort_grasp_contract()

    def reject(self, group, key, value):
        candidate = deepcopy(self.value)
        (candidate if group is None else candidate[group])[key] = value
        with self.assertRaises(ValueError):
            validate_effort_grasp_contract(candidate)

    @staticmethod
    def local_probe_settings():
        """Exercise the preserved local ABI independently of the active task."""
        from pathlib import Path
        config = json.loads((Path(__file__).resolve().parents[1]/"configs/experiment.json").read_text())
        value = config["rl_diagnostics"]
        contract = effort_grasp_contract()
        contract["evaluation"].update(protocol="acceptance32_v1", episodes_per_env=4, max_control_steps=364)
        value.update(full_lift=None, contact_transition_control=None,
            effort_grasp_controller=contract, num_envs=8, episodes_per_env=4, max_control_steps=364)
        return value

    def test_touch_acquisition_requires_exact_versioned_declaration(self):
        from g1_vla.rl_probe_protocol import validate_probe_settings
        value = self.local_probe_settings()
        declaration = value["contact_acquisition"]
        self.assertEqual(validate_probe_settings(value)["contact_acquisition"], declaration)
        previous = deepcopy(value)
        previous.pop("contact_acquisition")
        previous.pop("contact_domain_posture_guard", None)
        self.assertIsNone(validate_probe_settings(previous)["contact_acquisition"])
        for key in declaration:
            changed = deepcopy(value)
            changed["contact_acquisition"].pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_probe_settings(changed)
        for key, bad in [("maximum_extra_closure_rad", .025), ("reference_closing_rate_rad_s", .03),
                         ("raw_force_stop_n", 1.5), ("requires_no_physical_loaded_point", 1),
                         ("version", "unversioned"), ("start_phase", "bootstrap")]:
            changed = deepcopy(value)
            changed["contact_acquisition"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_probe_settings(changed)

    def test_touch_acquisition_v1_and_v2_require_matched_exact_rates(self):
        from g1_vla.rl_probe_protocol import validate_probe_settings
        original = self.local_probe_settings()
        for version, rate in [("g1_bounded_measured_touch_search_v1", .02),
                              ("g1_bounded_measured_touch_search_v2", .04)]:
            value = deepcopy(original)
            value["contact_acquisition"].update(version=version, reference_closing_rate_rad_s=rate)
            declaration = value["contact_acquisition"]
            self.assertEqual(validate_probe_settings(value)["contact_acquisition"], declaration)
            for wrong in [.03, .04 if rate == .02 else .02, True, math.nan]:
                changed = deepcopy(value)
                changed["contact_acquisition"]["reference_closing_rate_rad_s"] = wrong
                with self.subTest(version=version, wrong_rate=wrong), self.assertRaises(ValueError):
                    validate_probe_settings(changed)
            for key, wrong in [("maximum_extra_closure_rad", .021),
                               ("raw_force_stop_n", .04),
                               ("original_actual_geometry_tolerance_rad", .03),
                               ("all_original_acceptance_gates_unchanged", False)]:
                changed = deepcopy(value); changed["contact_acquisition"][key] = wrong
                with self.subTest(version=version, key=key), self.assertRaises(ValueError):
                    validate_probe_settings(changed)

    @staticmethod
    def domain_guard():
        return {"version": "g1_contact_domain_nullspace_posture_guard_v1",
            "hand_joint_index": 2, "activation_margin_rad": .006,
            "restoring_stiffness_nm_rad": 2.0, "maximum_guard_joint_torque_nm": .004,
            "minimum_nullspace_joint_component": .05, "svd_tolerance_m": .0001,
            "original_actual_geometry_tolerance_rad": .025,
            "protect_current_normal_and_world_z": True,
            "requires_all_groups_physically_loaded": True,
            "requires_original_contact_safe": True,
            "all_original_acceptance_gates_unchanged": True}

    def test_domain_guard_none_preserves_legacy_and_core_effort_contract(self):
        from g1_vla.rl_probe_protocol import validate_probe_settings
        self.assertIsNone(validate_contact_domain_posture_guard(None))
        self.assertIsNone(validate_probe_settings({})["contact_domain_posture_guard"])
        self.assertEqual(effort_grasp_contract()["version"], "g1_explicit_effort_grasp_protocol_v4_object_support")
        self.assertEqual(validate_effort_grasp_contract(self.value), self.value)
        self.assertNotIn("contact_domain_posture_guard", self.value)
        changed = deepcopy(self.value)
        changed["contact_domain_posture_guard"] = self.domain_guard()
        with self.assertRaises(ValueError):
            validate_effort_grasp_contract(changed)

    def test_domain_guard_exact_json_roundtrip_and_independent_copy(self):
        value = self.domain_guard()
        result = validate_contact_domain_posture_guard(json.loads(json.dumps(value)))
        self.assertEqual(result, value)
        result["activation_margin_rad"] = .1
        self.assertEqual(value, self.domain_guard())

    def test_domain_guard_requires_all_and_only_declared_fields(self):
        for key in self.domain_guard():
            value = self.domain_guard(); value.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_contact_domain_posture_guard(value)
        value = self.domain_guard(); value["hidden_override"] = True
        for bad in [value, [], False, "guard", 1]:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                validate_contact_domain_posture_guard(bad)

    def test_domain_guard_numerics_are_exact_typed_finite_and_not_booleans(self):
        for key, original in self.domain_guard().items():
            if type(original) not in (int, float):
                continue
            for bad in [True, False, math.nan, math.inf, str(original), original*1.01, -original]:
                value = self.domain_guard(); value[key] = bad
                with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                    validate_contact_domain_posture_guard(value)
        value = self.domain_guard(); value["hand_joint_index"] = 2.0
        with self.assertRaises(ValueError):
            validate_contact_domain_posture_guard(value)

    def test_domain_guard_declarations_cannot_disable_original_contact_or_expand_domain(self):
        for key, original in self.domain_guard().items():
            if type(original) is not bool:
                continue
            for bad in [False, 1, "true"]:
                value = self.domain_guard(); value[key] = bad
                with self.subTest(key=key, value=bad), self.assertRaises(ValueError):
                    validate_contact_domain_posture_guard(value)
        for key, bad in [("version", "unversioned"), ("original_actual_geometry_tolerance_rad", .03),
                         ("maximum_guard_joint_torque_nm", .04)]:
            value = self.domain_guard(); value[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contact_domain_posture_guard(value)

    def test_domain_guard_probe_requires_balanced_effort_and_contact_acquisition(self):
        from g1_vla.rl_probe_protocol import validate_probe_settings
        value = self.local_probe_settings()
        value["contact_domain_posture_guard"] = self.domain_guard()
        self.assertEqual(validate_probe_settings(value)["contact_domain_posture_guard"], self.domain_guard())
        for acquisition in (None,):
            changed = deepcopy(value); changed["contact_acquisition"] = acquisition
            with self.assertRaises(ValueError):
                validate_probe_settings(changed)
        with self.assertRaises(ValueError):
            validate_probe_settings({"contact_domain_posture_guard": self.domain_guard()})
        previous = deepcopy(value); previous.pop("contact_domain_posture_guard")
        self.assertIsNone(validate_probe_settings(previous)["contact_domain_posture_guard"])
        value["contact_domain_posture_guard"]["requires_original_contact_safe"] = 1
        with self.assertRaises(ValueError):
            validate_probe_settings(value)

    def test_default_is_fresh_and_json_roundtrips(self):
        original = effort_grasp_contract()
        result = validate_effort_grasp_contract(json.loads(json.dumps(original)))
        result["actuation"]["total_torque_caps_nm"][0] = 0.001
        self.assertEqual(effort_grasp_contract(), original)
        self.assertEqual(self.value, original)

    def test_compliant_material_is_explicit_positive_force_spring(self):
        self.value["material"].update(variant="finger_compliant_v5",
                                     compliant_contact_stiffness_n_m=1000.0,
                                     compliant_contact_damping_ns_m=0.8)
        self.assertEqual(validate_effort_grasp_contract(self.value), self.value)

    def test_explicit_parameters_vary_without_changing_code(self):
        self.value["material"].update(variant="finger_compliant_v5",
                                     compliant_contact_stiffness_n_m=800.0,
                                     compliant_contact_damping_ns_m=1.25)
        self.value["actuation"]["total_torque_caps_nm"] = [0.05] * 7
        self.value["controller"]["parameters"]["joint_torque_limits_nm"] = [0.05] * 7
        self.value["evaluation"]["max_wall_time_s"] = 900
        self.value["logging"].update(raw_pre_steps=2, raw_post_steps=4, raw_max_samples=256)
        self.assertEqual(validate_effort_grasp_contract(self.value), self.value)

    def test_acceptance32_requires_full_fixed_quota(self):
        self.value["evaluation"].update(protocol="acceptance32_v1", episodes_per_env=4,
                                       max_control_steps=364)
        validate_effort_grasp_contract(self.value)
        self.value["evaluation"]["max_control_steps"] = 91
        with self.assertRaises(ValueError):
            validate_effort_grasp_contract(self.value)

    def test_calibration_cannot_claim_32_episode_budget(self):
        self.reject("evaluation", "episodes_per_env", 4)
        self.reject("evaluation", "max_control_steps", 364)

    def test_unknown_fields_at_every_layer_rejected(self):
        for group in (None, "actuation", "material", "evaluation", "acceptance", "logging"):
            with self.subTest(group=group):
                self.reject(group, "hidden_override", True)
        value = deepcopy(self.value)
        value["material"]["friction"]["hidden_override"] = 0
        with self.assertRaises(ValueError):
            validate_effort_grasp_contract(value)

    def test_missing_fields_rejected(self):
        for group in self.value:
            value = deepcopy(self.value)
            del value[group]
            with self.subTest(group=group), self.assertRaises(ValueError):
                validate_effort_grasp_contract(value)

    def test_joint_order_and_caps_cannot_silently_reorder(self):
        self.reject(None, "joint_order", list(reversed(self.value["joint_order"])))
        self.reject("actuation", "total_torque_caps_nm", [0.04] * 6)
        self.reject("actuation", "total_torque_caps_nm", tuple([0.04] * 7))

    def test_total_caps_positive_finite_not_bool_and_within_native_effort(self):
        for bad in (0.0, -0.01, True, math.nan, math.inf, "0.06", 2.451):
            with self.subTest(bad=bad):
                caps = list(self.value["actuation"]["total_torque_caps_nm"])
                caps[0] = bad
                self.reject("actuation", "total_torque_caps_nm", caps)
        caps = [0.04] * 7
        caps[1] = 1.401
        self.reject("actuation", "total_torque_caps_nm", caps)

    def test_implicit_drive_and_model_gains_must_all_be_zero(self):
        for name in ("model_stiffness_nm_rad", "model_damping_nm_s_rad",
                     "native_drive_stiffness_nm_rad", "native_drive_damping_nm_s_rad"):
            self.reject("actuation", name, 0.01)
            self.reject("actuation", name, False)
        self.reject("actuation", "model", "ImplicitActuator")

    def test_gravity_and_feedforward_cannot_escape_total_cap(self):
        self.reject("actuation", "gravity_compensation_within_total_cap", False)
        self.reject("actuation", "gravity_compensation_within_total_cap", 1)

    def test_arm_feedforward_cannot_silently_change_joints_caps_or_claim_total_bound(self):
        for key, bad in (("feedforward_caps_nm", [100.] * 7),
                         ("joint_order", list(reversed(self.value["actuation"]["posture_gravity_feedforward"]["joint_order"]))),
                         ("combined_pd_and_ff_total_not_certified_bounded", False),
                         ("unused_other26_effort_targets_zero", False)):
            candidate = deepcopy(self.value)
            candidate["actuation"]["posture_gravity_feedforward"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_effort_grasp_contract(candidate)

    def test_hard_material_cannot_hide_compliance(self):
        self.reject("material", "compliant_contact_stiffness_n_m", 1000)
        self.reject("material", "compliant_contact_damping_ns_m", 0.8)

    def test_contact_generation_cannot_hide_filtering_or_change_model_margins(self):
        for key, bad in (("normal_manifold_filtering", True), ("contact_offset_m", .001),
                         ("rest_offset_m", 0.), ("friction_offset_threshold_m", .04),
                         ("original_physical_acceptance_unchanged", False)):
            candidate = deepcopy(self.value)
            candidate["contact_generation"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_effort_grasp_contract(candidate)

    def test_compliant_parameters_require_range_and_finite(self):
        self.value["material"]["variant"] = "finger_compliant_v5"
        for bad in (0, -1, True, math.nan, math.inf, 1000001):
            self.reject("material", "compliant_contact_stiffness_n_m", bad)
        self.value["material"]["compliant_contact_stiffness_n_m"] = 1000
        for bad in (-1, False, math.nan, math.inf, 1001):
            self.reject("material", "compliant_contact_damping_ns_m", bad)

    def test_no_acceleration_spring_negative_restitution_or_undeclared_offsets(self):
        self.reject("material", "compliant_contact_acceleration_spring", True)
        self.reject("material", "restitution", -1000.0)
        self.reject("material", "rest_offset_m", -0.0001)
        self.reject("material", "contact_offset_m", .001)

    def test_compliance_cannot_extend_to_cube_table_palm_or_change_friction(self):
        bodies = list(self.value["material"]["compliance_candidate_body_names"]) + ["Cube"]
        self.reject("material", "compliance_candidate_body_names", bodies)
        self.reject("material", "always_rigid_body_names", ["Table"])
        self.reject("material", "friction", {**self.value["material"]["friction"], "finger_static": 5.0})
        self.reject("material", "native_binding_readback_required", False)

    def test_default_compliance_is_distal_pads_and_other_fingers_stay_rigid(self):
        material = self.value["material"]
        self.assertEqual(material["compliance_candidate_body_names"], [
            "right_hand_thumb_2_link", "right_hand_index_1_link", "right_hand_middle_1_link"])
        self.assertIn("right_hand_thumb_0_link", material["always_rigid_body_names"])
        self.assertIn("right_hand_palm_link", material["always_rigid_body_names"])

    def test_explicit_finger_subset_requires_exact_hard_complement(self):
        material = self.value["material"]
        soft = ["right_hand_thumb_2_link", "right_hand_index_1_link"]
        material["compliance_candidate_body_names"] = soft
        material["always_rigid_body_names"] = list(ALWAYS_RIGID_BODY_NAMES) + [
            name for name in FINGER_BODY_NAMES if name not in soft]
        validate_effort_grasp_contract(self.value)
        material["always_rigid_body_names"].remove("right_hand_middle_1_link")
        with self.assertRaises(ValueError):
            validate_effort_grasp_contract(self.value)

    def test_no_empty_duplicate_unknown_or_nonlist_soft_subset(self):
        for names in ([], ["right_hand_thumb_2_link"] * 2, ["right_hand_palm_link"],
                      ["Cube"], [True], tuple(["right_hand_thumb_2_link"])):
            self.reject("material", "compliance_candidate_body_names", names)

    def test_acceptance_cannot_be_relaxed_even_if_named_calibration(self):
        for key, bad in (("group_force_max_n", 2.0), ("maximum_penetration_m", 0.004),
                         ("opposition_dot_max", -0.5), ("maximum_abs_normal_z", 0.7),
                         ("support_consecutive_samples", 99), ("minimum_corner_clearance_m", 0.001),
                         ("physical_loaded_separation_max_m", 1e-7),
                         ("legacy_complete_partner_friction_manifold_required", False),
                         ("all_nonzero_positive_gap_hand_load_forbidden", False),
                         ("main_lift_authorized", True), ("reward_used_as_success", True)):
            self.reject("acceptance", key, bad)

    def test_training_randomization_timing_or_population_cannot_drift(self):
        for key, bad in (("num_envs", 16), ("num_envs", True), ("training_allowed", True),
                         ("optimizer_steps", 1), ("physics_dt_s", 0.001),
                         ("control_period_s", 0.033), ("nominal_episode_budget_s", 7),
                         ("validation_offsets", "new_offsets"), ("randomization_expanded", True)):
            self.reject("evaluation", key, bad)

    def test_typed_quota_and_wall_budget_rejected(self):
        for group, key, value in (("evaluation", "episodes_per_env", True),
                                  ("evaluation", "max_control_steps", 91.0),
                                  ("evaluation", "max_wall_time_s", 3601),
                                  ("evaluation", "max_wall_time_s", math.nan)):
            self.reject(group, key, value)

    def test_logging_is_complete_compact_bounded_raw_no_hidden_omissions(self):
        self.reject("logging", "compact_2ms", "macro_only")
        self.reject("logging", "raw_mode", "all")
        self.reject("logging", "raw_omissions_and_truncation_reported", False)
        self.reject("logging", "raw_pre_steps", True)
        self.reject("logging", "raw_max_samples", 4097)
        self.reject("logging", "raw_max_samples", 3)

    def test_balanced_controller_roundtrip_and_fresh_copy(self):
        value = balanced_effort_controller_contract()
        result = validate_balanced_effort_controller_contract(value)
        result["parameters"]["group_force_targets_n"][0] = 0.5
        self.assertEqual(value, balanced_effort_controller_contract())

    def test_balanced_controller_all_parameters_typed_and_finite(self):
        default = balanced_effort_controller_contract()
        for key, original in default["parameters"].items():
            for bad in (True, math.nan, math.inf):
                value = deepcopy(default)
                value["parameters"][key] = [bad] * len(original) if isinstance(original, list) else bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    validate_balanced_effort_controller_contract(value)

    def test_balanced_controller_rejects_bad_shapes_and_unknown_keys(self):
        for key, bad in (("group_force_targets_n", [0.4] * 2),
                         ("joint_torque_limits_nm", [0.04] * 6),
                         ("group_force_targets_n", tuple([0.4] * 3))):
            value = balanced_effort_controller_contract()
            value["parameters"][key] = bad
            with self.assertRaises(ValueError):
                validate_balanced_effort_controller_contract(value)
        value = balanced_effort_controller_contract()
        value["parameters"]["hidden_gain"] = 1
        with self.assertRaises(ValueError):
            validate_balanced_effort_controller_contract(value)

    def test_balanced_controller_explicit_changes_get_their_own_values(self):
        value = balanced_effort_controller_contract()
        value["parameters"].update(group_force_targets_n=[0.6, 0.3, 0.3], joint_stiffness_nm_rad=0.12,
                                    normal_force_feedback_gain=0.8, reacquisition_velocity_limit_rad_s=0.02)
        self.assertEqual(validate_balanced_effort_controller_contract(value), value)

    def test_balanced_controller_force_caps_and_overforce_cannot_disappear(self):
        for key, bad in (("group_force_limit_n", 1.6), ("maximum_penetration_m", 0.004),
                         ("group_force_command_lower_n", 0.0), ("overforce_opening_torque_nm", 0.0),
                         ("group_force_command_upper_n", 1.6), ("contact_feedback_onset_n", 0.5),
                         ("reacquisition_velocity_limit_rad_s", 0.041)):
            value = balanced_effort_controller_contract()
            value["parameters"][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_balanced_effort_controller_contract(value)

    def test_controller_and_actuator_total_caps_must_be_identical(self):
        value = deepcopy(self.value)
        value["actuation"]["total_torque_caps_nm"][0] = 0.05
        with self.assertRaises(ValueError):
            validate_effort_grasp_contract(value)


class FullLiftProtocolTests(unittest.TestCase):
    def test_local_factory_stays_six_seconds_and_cannot_authorize_a_main_lift(self):
        local = effort_grasp_contract()
        self.assertEqual(local["version"], "g1_explicit_effort_grasp_protocol_v4_object_support")
        self.assertEqual(local["evaluation"]["nominal_episode_budget_s"], 6.)
        self.assertEqual(local["evaluation"]["max_control_steps"], 91)
        self.assertFalse(local["acceptance"]["main_lift_authorized"])
        self.assertEqual(local["acceptance"]["actual_height_limit_m"], .0065)
        self.assertIsNone(validate_full_lift_protocol(None))
        self.assertIsNone(validate_contact_transition_control(None))

    def test_full_task_has_real_10cm_clearance_and_half_second_sample_span(self):
        value = full_lift_protocol_contract()
        self.assertEqual(value["version"], "g1_contact_qualified_full_lift_v2_servo_headroom")
        self.assertEqual(value["palm_servo_speed_limit_m_s"], .018)
        self.assertEqual(value["target_corner_clearance_m"], .1)
        self.assertEqual(value["palm_total_lift_request_m"], .105)
        self.assertEqual(value["palm_lift_request_max_m"], .11)
        self.assertEqual(value["lift_speed_m_s"], .015)
        self.assertEqual(value["lift_acceleration_m_s2"], .03)
        self.assertEqual(value["palm_position_gain_per_s"], 8.)
        self.assertEqual(value["actual_height_limit_m"], .115)
        self.assertEqual(value["episode_length_s"], 14.)
        self.assertEqual((value["hold_consecutive_samples"]-1)*value["physics_dt_s"], .5)
        self.assertEqual((value["local_support_consecutive_samples"]-1)*value["physics_dt_s"], .198)
        self.assertEqual(validate_full_lift_protocol(json.loads(json.dumps(value))), value)
        copy = full_lift_protocol_contract(); copy["target_corner_clearance_m"] = .001
        self.assertEqual(value, full_lift_protocol_contract())

    def test_full_lift_servo_headroom_does_not_change_physical_speed_gate(self):
        old = full_lift_protocol_contract("g1_contact_qualified_full_lift_v1")
        new = full_lift_protocol_contract()
        self.assertNotIn("palm_servo_speed_limit_m_s", old)
        self.assertEqual(validate_full_lift_protocol(old), old)
        for key in old:
            if key != "version":
                self.assertEqual(new[key], old[key])
        mixed = dict(new, version=old["version"])
        with self.assertRaises(ValueError):
            validate_full_lift_protocol(mixed)
        with self.assertRaises(ValueError):
            full_lift_protocol_contract("unversioned")

    def test_full_declaration_is_exact_typed_and_has_no_optional_gate_bypass(self):
        original = full_lift_protocol_contract()
        for key, wanted in original.items():
            changed = deepcopy(original); changed.pop(key)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                validate_full_lift_protocol(changed)
            bads = ([1, False, "true"] if type(wanted) is bool and wanted else
                    [0, True, "false"] if type(wanted) is bool else
                    [True, math.nan, math.inf, str(wanted), wanted+1] if type(wanted) in (int, float) else
                    ["unversioned", None])
            for bad in bads:
                changed = deepcopy(original); changed[key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    validate_full_lift_protocol(changed)
        for bad in [False, [], "full", {**original, "skip_local_proof": True}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_full_lift_protocol(bad)

    def test_native_arm_schedule_is_separate_exact_task_version(self):
        old = full_lift_protocol_contract()
        new = full_lift_protocol_contract(
            "g1_contact_qualified_full_lift_v3_native_arm_interpolation")
        self.assertEqual(new["arm_target_schedule"], "native_linear33_revalidated_v1")
        self.assertEqual(new["arm_target_interpolation_substeps"], 33)
        self.assertEqual(new["full_lift_failure_response"],
                         "stop_arm_keep_current_hand_force_feedback_v1")
        self.assertEqual(validate_full_lift_protocol(new), new)
        self.assertEqual(full_lift_protocol_contract(), old)
        self.assertNotIn("arm_target_schedule", old)
        for key in old:
            if key != "version":
                self.assertEqual(new[key], old[key])
        for changed in [dict(new, arm_target_interpolation_substeps=32),
                        dict(new, arm_target_interpolation_substeps=True),
                        dict(new, arm_target_schedule="macro_step"),
                        dict(new, full_lift_failure_response="zero_all_user_effort"),
                        dict(new, version=old["version"]),
                        dict(new, version=[]), dict(new, version={}), dict(new, version=True),
                        {**old, "arm_target_schedule": new["arm_target_schedule"]}]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_full_lift_protocol(changed)

    def test_full_effort_preserves_all_physical_gates_except_declared_task_height(self):
        local = effort_grasp_contract()
        for name, repeats, steps, duration in [("full_lift8_v1", 1, 213, 14.),
                ("full_lift32_v1", 4, 852, 14.), ("full_lift8_v2", 1, 243, 16.),
                ("full_lift32_v2", 4, 972, 16.)]:
            value = full_lift_effort_grasp_contract(name)
            self.assertEqual(validate_effort_grasp_contract(value), value)
            self.assertEqual(value["version"], FULL_LIFT_EFFORT_GRASP_VERSION)
            self.assertEqual(value["evaluation"]["episodes_per_env"], repeats)
            self.assertEqual(value["evaluation"]["max_control_steps"], steps)
            self.assertEqual(steps, math.ceil(duration/.066)*repeats)
            self.assertFalse(value["evaluation"]["training_allowed"])
            self.assertEqual(value["evaluation"]["optimizer_steps"], 0)
            expected = deepcopy(local)
            expected["version"] = value["version"]
            expected["acceptance"].update(main_lift_authorized=True, actual_height_limit_m=.115)
            expected["evaluation"].update(protocol=name, nominal_episode_budget_s=duration,
                episodes_per_env=repeats, max_control_steps=steps)
            self.assertEqual(value, expected)

    def test_native_16s_task_only_adds_declared_physical_deadline(self):
        old = full_lift_protocol_contract("g1_contact_qualified_full_lift_v3_native_arm_interpolation")
        new = full_lift_protocol_contract("g1_contact_qualified_full_lift_v4_native_arm_16s")
        self.assertEqual(new["episode_length_s"], 16.)
        self.assertEqual(validate_full_lift_protocol(new), new)
        for key in old:
            if key not in ("version", "episode_length_s"):
                self.assertEqual(new[key], old[key])
        for task, duration in [(old, 16.), (new, 14.)]:
            with self.assertRaises(ValueError):
                validate_full_lift_protocol(dict(task, episode_length_s=duration))
        for name in ("full_lift8_v1", "full_lift8_v2"):
            value = full_lift_effort_grasp_contract(name)
            value["evaluation"]["nominal_episode_budget_s"] = 30.-value["evaluation"]["nominal_episode_budget_s"]
            with self.assertRaises(ValueError):
                validate_effort_grasp_contract(value)

    def test_task_version_deadline_height_and_quotas_cannot_be_mixed(self):
        full = full_lift_effort_grasp_contract()
        for group, key, bad in [(None, "version", effort_grasp_contract()["version"]),
            ("evaluation", "protocol", "calibration8_v1"),
            ("evaluation", "nominal_episode_budget_s", 6.),
            ("evaluation", "episodes_per_env", 4),
            ("evaluation", "max_control_steps", 91),
            ("evaluation", "max_control_steps", 213.),
            ("evaluation", "randomization_expanded", True),
            ("evaluation", "training_allowed", True),
            ("evaluation", "original20_acceptance", True),
            ("acceptance", "actual_height_limit_m", .0065),
            ("acceptance", "main_lift_authorized", False),
            ("acceptance", "group_force_max_n", 2.),
            ("acceptance", "support_consecutive_samples", 1)]:
            value = deepcopy(full)
            (value if group is None else value[group])[key] = bad
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                validate_effort_grasp_contract(value)
        for bad in ["acceptance32_v1", "full", None, True]:
            with self.subTest(protocol=bad), self.assertRaises(ValueError):
                full_lift_effort_grasp_contract(bad)

    def test_transition_contract_cannot_silently_drop_fields_or_enable_training(self):
        original = contact_transition_control_contract()
        self.assertEqual(validate_contact_transition_control(original), original)
        for key in original:
            value = deepcopy(original); value.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_contact_transition_control(value)
        value = deepcopy(original); value["training_allowed"] = True
        with self.assertRaises(ValueError):
            validate_contact_transition_control(value)


if __name__ == "__main__":
    unittest.main()
