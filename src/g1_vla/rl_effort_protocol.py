"""Pure declarations for explicit hand effort and contact-material calibration.

Validation is not a measurement of native torque, contact force, or support.
The simulator must independently bind/read back the declared drive/material
properties.  This module deliberately imports neither config nor a backend.
"""

from __future__ import annotations

from copy import deepcopy
import math


EFFORT_GRASP_VERSION = "g1_explicit_effort_grasp_protocol_v4_object_support"
RIGHT_ARM_JOINT_NAMES = tuple("right_" + name + "_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"))
RIGHT_HAND_JOINT_NAMES = tuple(
    "right_hand_" + name + "_joint"
    for name in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
)
FINGER_BODY_NAMES = tuple(name.replace("_joint", "_link") for name in RIGHT_HAND_JOINT_NAMES)
DISTAL_PAD_BODY_NAMES = tuple(FINGER_BODY_NAMES[i] for i in (2, 4, 6))
ALWAYS_RIGID_BODY_NAMES = (
    "Cube", "Table", "right_hand_palm_link", "right_wrist_yaw_link", "right_hand_camera_base_link",
)
_NATIVE_EFFORT_UPPER_NM = (2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4)
_KINDS = "explicit_hand_effort_and_material_calibration"
_CONTACT_TRANSITION_CONTROL = {
    "version": "g1_coordinated_contact_transition_v1",
    "unload_direction": "current_normal_jacobian_else_declared_axis",
    "missing_group_reference_velocity_limit_rad_s": .04,
    "reset_missing_reference_on_first_any_contact": True,
    "preload_group_force_targets_n": [.2, .1, .1],
    "full_contact_settle_samples": 25,
    "full_force_ramp_s": .1,
    "free_thumb2_activation_margin_rad": .006,
    "free_thumb2_restoring_stiffness_nm_rad": 2.0,
    "free_thumb2_maximum_guard_torque_nm": .004,
    "partial_thumb2_normal_world_z_guard": True,
    "all_original_acceptance_gates_unchanged": True,
}
_CONTACT_TRANSITION_CONTROL_V2 = deepcopy(_CONTACT_TRANSITION_CONTROL)
_CONTACT_TRANSITION_CONTROL_V2.update({
    "version": "g1_coordinated_contact_transition_v2_separating_preload",
    "separating_preload_command_floor_n": 0.0,
    "all_current_points_separating_threshold_m_s": -1e-5,
    "separating_floor_requires_physically_loaded_current_group": True,
})
_CONTACT_TRANSITION_CONTROL_V3 = deepcopy(_CONTACT_TRANSITION_CONTROL)
_CONTACT_TRANSITION_CONTROL_V3.update({
    "version": "g1_coordinated_contact_transition_v3_relative_viscous_preload",
    "relative_damping_mass_model": "current_hand7_principal_block_other36_fixed_diagonal_groups_v1",
    "relative_damping_velocity_source": "current_full43_minus_cube_com_point_velocity",
    "relative_damping_ratio": 1.0,
    "relative_damping_force_feedback_gain": .5,
    "relative_damping_coefficient_minimum_ns_m": .5,
    "relative_damping_coefficient_maximum_ns_m": 64.0,
    "relative_damping_explicit_gain_limit": 1.2,
    "relative_damping_material_point_stiffness_n_m": 1000.0,
    "relative_damping_material_point_damping_ns_m": .8,
    "relative_damping_partial_blend": "one_minus_full_force_ramp_fraction",
})
_CONTACT_DOMAIN_POSTURE_GUARD = {
    "version": "g1_contact_domain_nullspace_posture_guard_v1",
    "hand_joint_index": 2,
    "activation_margin_rad": .006,
    "restoring_stiffness_nm_rad": 2.0,
    "maximum_guard_joint_torque_nm": .004,
    "minimum_nullspace_joint_component": .05,
    "svd_tolerance_m": .0001,
    "original_actual_geometry_tolerance_rad": .025,
    "protect_current_normal_and_world_z": True,
    "requires_all_groups_physically_loaded": True,
    "requires_original_contact_safe": True,
    "all_original_acceptance_gates_unchanged": True,
}
_CONTACT_DOMAIN_POSTURE_GUARD_V2 = deepcopy(_CONTACT_DOMAIN_POSTURE_GUARD)
_CONTACT_DOMAIN_POSTURE_GUARD_V2.pop("hand_joint_index")
_CONTACT_DOMAIN_POSTURE_GUARD_V2.update({
    "version": "g1_contact_domain_nullspace_posture_guard_v2_common_thumb_interval",
    "hand_joint_indices": [0, 1, 2],
    "spring_coordinate": "unit_current_thumb_normal_world_z_nullspace_displacement_rad",
    "control_target": "maximum_common_interior_margin_up_to_activation_margin",
    "interval_solver": "analytic_pairwise_margin_float64_v1",
    "minimum_component_gate_scope": "unchanged_legacy_partial_thumb2_recovery_only",
})
_CONTACT_DOMAIN_POSTURE_GUARD_V3 = deepcopy(_CONTACT_DOMAIN_POSTURE_GUARD_V2)
_CONTACT_DOMAIN_POSTURE_GUARD_V3.update({
    "version": "g1_contact_domain_nullspace_posture_guard_v3_all_loaded_interior_preservation",
    "control_target": "captured_first_safe_all_loaded_margin_capped_by_current_common_feasibility",
    "interior_reserve_capture": "first_fresh_safe_all_groups_physically_loaded_original_thumb3_minimum_margin",
    "interior_reserve_floor_rad": .006,
    "interior_reserve_reset": "episode_or_current_contact_safety_loss_or_unsafe_discontinuity",
    "guard_position_difference_telemetry": "same_epoch_consecutive_native_2ms_not_control_feedback",
    "interior_reference_is_not_acceptance_or_cached_geometry": True,
})
_BALANCED_DEFAULTS = {
    "version": "g1_balanced_normal_and_object_support_effort_grasp_v2",
    "kind": "explicit_joint_effort_from_current_contact_normal_jacobian",
    "parameters": {
        "physics_dt_s": 0.002,
        "group_force_targets_n": [0.8, 0.4, 0.4],
        "normal_force_feedback_gain": 0.5,
        "normal_velocity_damping_n_s_m": 0.5,
        "group_force_command_lower_n": -0.4,
        "group_force_command_upper_n": 1.2,
        "joint_stiffness_nm_rad": 0.15,
        "joint_damping_nm_s_rad": 0.015,
        "joint_torque_limits_nm": [0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        "group_force_limit_n": 1.5,
        "contact_feedback_onset_n": 0.02,
        "normal_jacobian_rank_tolerance_m": 0.0001,
        "maximum_penetration_m": 0.003,
        "overforce_opening_torque_nm": 0.02,
        "reacquisition_velocity_limit_rad_s": 0.04,
    },
    "reference_semantics": {
        "contact_direction": "current_sample_normal_jacobian_only_positive_gap_permitted_for_control_not_physical_acceptance",
        "missing_current_direction": "never_reuse_old_normal_for_sustained_force_reset_all_group_joint_references_to_measured_then_bounded_joint_impedance_reacquisition_only",
        "contact_position_spring": "current_normal_force_mode_zero_position_spring_for_every_joint_of_affected_group_keep_only_normal_nullspace_damping_gravity_and_normal_force_control",
        "force_feedback": "unclipped_native_group_force_including_speculative_points_not_a_force_bound",
        "joint_effort_units": "Nm_for_seven_named_revolute_joints",
        "total_torque_clipping": "gravity_clipped_inside_total_cap_then_remaining_contributions_share_one_alpha_in_0_1",
        "overforce_priority": "raw_force_above_original_1p5N_cancels_positive_inward_force_and_group_gravity_posture_contributions_requests_opening",
        "overforce_without_current_direction": "known_joint_closing_axes_opening_fallback_bounded_by_declared_opening_torque_and_final_total_cap",
        "gravity_compensation": "included_in_the_same_total_per_joint_cap_never_added_after_clipping",
        "object_weight_support": "fresh_current_point_world_z_jacobian_torque_inside_same_alpha_and_total_cap_removed_on_group_overforce_not_measured_support",
        "reacquisition_velocity": "bounded_reference_motion_not_a_real_force_or_motion_guarantee",
        "physical_qualification": "unchanged_legacy_contact_gap_friction_force_geometry_and_100_sample_support_gates_independent_of_controller",
        "telemetry": "commanded_total_effort_is_not_claimed_as_isolated_measured_drive_torque",
    },
}
_FRICTION = {
    "finger_static": 2.2,
    "finger_dynamic": 2.2,
    "cube_static": 1.4,
    "cube_dynamic": 1.4,
    "palm_static": 2.0,
    "palm_dynamic": 2.0,
    "table_static": 1.0,
    "table_dynamic": 1.0,
    "combine_mode": "max",
}
_ACCEPTANCE = {
    "original_gate_version": "physx_complete_partner_manifold_actual_contact_v4",
    "group_force_min_n": 0.02,
    "group_force_max_n": 1.5,
    "maximum_penetration_m": 0.003,
    "opposition_dot_max": -0.6,
    "maximum_abs_normal_z": 0.6,
    "physical_loaded_separation_max_m": 0.0,
    "all_nonzero_positive_gap_hand_load_forbidden": True,
    "legacy_complete_partner_friction_manifold_required": True,
    "legacy_certificate_reported_separately": True,
    "support_consecutive_samples": 100,
    "support_first_to_last_s": 0.198,
    "minimum_corner_clearance_m": 0.002,
    "maximum_linear_speed_m_s": 0.02,
    "maximum_angular_speed_rad_s": 0.2,
    "maximum_grip_drift_m": 0.003,
    "table_force_weight_fraction_max": 0.05,
    "table_loaded_points_max": 0,
    "other_external_force_max_n": 0.02,
    "other_external_loaded_points_max": 0,
    "hand_upward_force_weight_fraction_min": 0.8,
    "hand_upward_force_weight_fraction_max": 1.2,
    "joint_limit_tolerance_rad": 0.003,
    "actual_height_limit_m": 0.0065,
    "main_lift_authorized": False,
    "reward_used_as_success": False,
}
_DEFAULTS = {
    "version": EFFORT_GRASP_VERSION,
    "kind": _KINDS,
    "joint_order": list(RIGHT_HAND_JOINT_NAMES),
    "controller": deepcopy(_BALANCED_DEFAULTS),
    "contact_generation": {
        "version": "g1_zero_margin_physical_contact_v1",
        "contact_offset_m": 0.0,
        "rest_offset_m": -1e-6,
        "friction_offset_threshold_m": 0.0,
        "normal_manifold_filtering": False,
        "original_physical_acceptance_unchanged": True,
    },
    "actuation": {
        "model": "IdealPDActuator",
        "model_stiffness_nm_rad": 0.0,
        "model_damping_nm_s_rad": 0.0,
        "native_drive_stiffness_nm_rad": 0.0,
        "native_drive_damping_nm_s_rad": 0.0,
        "total_torque_caps_nm": [0.06, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        "gravity_compensation_within_total_cap": True,
        "unclipped_and_clipped_total_torque_logged": True,
        "position_target_is_not_force_limit": True,
        "object_weight_support": {
            "version": "g1_current_contact_object_weight_feedforward_v1",
            "object_mass_kg": 0.08,
            "gravity_m_s2": 9.81,
            "world_up_axis": [0., 0., 1.],
            "group_weight_shares": [0.5, 0.25, 0.25],
            "ramp_s": 0.198,
            "requires_original_contact_safe_and_grip_lock": True,
            "fresh_point_jacobian_required": True,
            "inside_total_hand_torque_budget": True,
            "object_state_writes_or_attachments": False,
        },
        "posture_gravity_feedforward": {
            "version": "g1_native_arm7_waist3_gravity_feedforward_v1",
            "joint_order": list(RIGHT_ARM_JOINT_NAMES) + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            "feedforward_caps_nm": [25., 25., 25., 25., 25., 5., 5., 88., 50., 50.],
            "native_pd_values_unchanged": True,
            "combined_pd_and_ff_total_not_certified_bounded": True,
            "unused_other26_effort_targets_zero": True,
        },
    },
    "material": {
        "variant": "hard_v4_control",
        "compliant_contact_stiffness_n_m": 0.0,
        "compliant_contact_damping_ns_m": 0.0,
        "compliant_contact_acceleration_spring": False,
        "rest_offset_m": -1e-6,
        "contact_offset_m": 0.0,
        "restitution": 0.0,
        "restitution_combine_mode": "average",
        "damping_combine_mode": "average",
        "compliance_candidate_body_names": list(DISTAL_PAD_BODY_NAMES),
        "always_rigid_body_names": list(ALWAYS_RIGID_BODY_NAMES) + [
            name for name in FINGER_BODY_NAMES if name not in DISTAL_PAD_BODY_NAMES
        ],
        "friction": deepcopy(_FRICTION),
        "native_binding_readback_required": True,
    },
    "evaluation": {
        "protocol": "calibration8_v1",
        "num_envs": 8,
        "episodes_per_env": 1,
        "max_control_steps": 91,
        "nominal_episode_budget_s": 6.0,
        "physics_dt_s": 0.002,
        "control_period_s": 0.066,
        "max_wall_time_s": 3600.0,
        "validation_offsets": "original_eight_offsets_unchanged",
        "randomization_expanded": False,
        "training_allowed": False,
        "optimizer_steps": 0,
        "original20_acceptance": False,
    },
    "acceptance": deepcopy(_ACCEPTANCE),
    "logging": {
        "compact_2ms": "all_samples_including_terminal_and_reset_epoch",
        "raw_mode": "bounded_trigger_windows",
        "raw_pre_steps": 4,
        "raw_post_steps": 8,
        "raw_max_trigger_windows": 32,
        "raw_max_samples": 512,
        "raw_triggers": ["force_above_acceptance_cap", "physical_contact_loss", "reset_initialization"],
        "raw_omissions_and_truncation_reported": True,
        "native_q_velocity_written_target_and_total_effort_required": True,
    },
}


def effort_grasp_contract():
    """Return a fresh hard-v4, eight-episode calibration declaration.

    A compliant variant is declared by changing ``material.variant`` to
    ``finger_compliant_v5`` and its K/C to positive/nonnegative values; the
    suggested initial calibration is K=1000 N/m and C=0.8 N*s/m.  Change all
    three quota fields together for ``acceptance32_v1`` (8,4,364).
    """
    return deepcopy(_DEFAULTS)


def balanced_effort_controller_contract():
    """Return the initial declared controller, independent of its parameters."""
    return deepcopy(_BALANCED_DEFAULTS)


FULL_LIFT_EFFORT_GRASP_VERSION = "g1_explicit_effort_grasp_protocol_v5_full_lift"
_FULL_LIFT_PROTOCOL_V1 = {
    "version": "g1_contact_qualified_full_lift_v1",
    "target_corner_clearance_m": .1,
    "palm_total_lift_request_m": .105,
    "palm_lift_request_max_m": .11,
    "lift_speed_m_s": .015,
    "lift_acceleration_m_s2": .03,
    "palm_position_gain_per_s": 8.0,
    "hold_duration_s": .5,
    "hold_consecutive_samples": 251,
    "actual_height_limit_m": .115,
    "episode_length_s": 14.0,
    "physics_dt_s": .002,
    "local_support_consecutive_samples": 100,
    "local_support_first_to_last_s": .198,
    "requires_local_support_before_lift": True,
    "requires_whole_episode_force_clean_before_lift": True,
    "requires_whole_episode_original_hand_domain_clean_before_lift": True,
    "original_hand_domain_tolerance_rad": .025,
    "fixed_pre_transfer_grip_reference_required": True,
    "all_other_original_physical_gates_unchanged": True,
    "training_allowed": False,
    "optimizer_steps": 0,
}
_FULL_LIFT_PROTOCOL = {
    **_FULL_LIFT_PROTOCOL_V1,
    "version": "g1_contact_qualified_full_lift_v2_servo_headroom",
    "palm_servo_speed_limit_m_s": .018,
}
_FULL_LIFT_PROTOCOL_V3 = {
    **_FULL_LIFT_PROTOCOL,
    "version": "g1_contact_qualified_full_lift_v3_native_arm_interpolation",
    "arm_target_schedule": "native_linear33_revalidated_v1",
    "arm_target_interpolation_substeps": 33,
    "full_lift_failure_response": "stop_arm_keep_current_hand_force_feedback_v1",
}
_FULL_LIFT_PROTOCOL_V4 = {
    **_FULL_LIFT_PROTOCOL_V3,
    "version": "g1_contact_qualified_full_lift_v4_native_arm_16s",
    "episode_length_s": 16.0,
}
_FULL_LIFT_PROTOCOL_V5_HOME = {
    **_FULL_LIFT_PROTOCOL_V4,
    "version": "g1_contact_qualified_arm_home_lift_v5_32s",
    "episode_length_s": 32.0,
    "arm_target_schedule": "home_and_lift_native_linear33_revalidated_v1",
}
_FULL_LIFT_EVALUATION_QUOTAS = {
    "full_lift8_v1": (1, 213, 14.0), "full_lift32_v1": (4, 852, 14.0),
    "full_lift8_v2": (1, 243, 16.0), "full_lift32_v2": (4, 972, 16.0),
    "full_lift32_v3_wall7200": (4, 972, 16.0),
    "home_lift1_v1": (1, 486, 32.0),
    "home_lift8_v1": (1, 486, 32.0),
}
FULL_LIFT32_WALL7200_PROTOCOL = "full_lift32_v3_wall7200"


def effort_evaluation_wall_time_limit_s(value):
    """Resource ceiling only; callers still validate the whole declaration.

    This one named full32 protocol opts into7200 wall seconds. The physical
    episode, original8-offset quota and training limits remain independent.
    All historical effort declarations retain their3600-second ceiling.
    """
    if (type(value) is dict and value.get("version") == FULL_LIFT_EFFORT_GRASP_VERSION and
            type(value.get("evaluation")) is dict and
            value["evaluation"].get("protocol") == FULL_LIFT32_WALL7200_PROTOCOL):
        return 7200.0
    return 3600.0


def full_lift_protocol_contract(version=None):
    """Return the independent 10cm task declaration, never an admission proof.

    The local 2mm/100-sample proof remains mandatory before any main lift.
    A full task has a separately declared physical deadline and height limit;
    neither the local protocol nor any training budget changes implicitly.
    """
    if version is None or version == _FULL_LIFT_PROTOCOL["version"]:
        return deepcopy(_FULL_LIFT_PROTOCOL)
    if version == _FULL_LIFT_PROTOCOL_V1["version"]:
        return deepcopy(_FULL_LIFT_PROTOCOL_V1)
    if version == _FULL_LIFT_PROTOCOL_V3["version"]:
        return deepcopy(_FULL_LIFT_PROTOCOL_V3)
    if version == _FULL_LIFT_PROTOCOL_V4["version"]:
        return deepcopy(_FULL_LIFT_PROTOCOL_V4)
    if version == _FULL_LIFT_PROTOCOL_V5_HOME["version"]:
        return deepcopy(_FULL_LIFT_PROTOCOL_V5_HOME)
    raise ValueError("Unknown full-lift task version")


def full_lift_effort_grasp_contract(protocol="full_lift8_v1"):
    """Declare the new physical task while preserving the v4 local factory."""
    quotas = _FULL_LIFT_EVALUATION_QUOTAS
    if type(protocol) is not str or protocol not in quotas:
        raise ValueError("Unknown full-lift evaluation protocol")
    episodes, steps, duration = quotas[protocol]
    value = effort_grasp_contract()
    value["version"] = FULL_LIFT_EFFORT_GRASP_VERSION
    value["acceptance"].update(main_lift_authorized=True, actual_height_limit_m=.115)
    value["evaluation"].update(protocol=protocol, episodes_per_env=episodes,
        max_control_steps=steps, nominal_episode_budget_s=duration)
    if protocol == "home_lift1_v1":
        value["evaluation"]["num_envs"] = 1
    if protocol == FULL_LIFT32_WALL7200_PROTOCOL:
        value["evaluation"]["max_wall_time_s"] = effort_evaluation_wall_time_limit_s(value)
    return value


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(name + " requires exactly the declared fields")


def _exact(value, expected, name):
    """Reject bool-as-number and equal-but-differently-typed nested values."""
    if type(value) is not type(expected):
        # JSON numeric spelling 0 and 0.0 is immaterial, but booleans are not.
        if type(value) is int and type(expected) is float:
            if value == expected:
                return
        raise ValueError(name + " has an incompatible value/type")
    if isinstance(expected, dict):
        _keys(value, expected, name)
        for key in expected:
            _exact(value[key], expected[key], name + "." + key)
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(name + " has incompatible length")
        for i, (actual, wanted) in enumerate(zip(value, expected)):
            _exact(actual, wanted, name + "[" + str(i) + "]")
    elif value != expected:
        raise ValueError(name + " differs from the unchanged contract")


def _number(value, low, high, name, *, positive=False):
    if type(value) not in (int, float) or (type(value) is float and not math.isfinite(value)):
        raise ValueError(name + " must be a finite number")
    if value < low or value > high or (positive and value <= 0):
        raise ValueError(name + " is outside its permitted range")


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(name + " must be an integer in range")


def validate_contact_domain_posture_guard(value):
    """Validate the optional exact controller guard; never change acceptance.

    ``None`` preserves the prior controller path. A declared guard only adds
    bounded motor effort through the existing common alpha and total caps;
    its actual nullspace/current-contact conditions require runtime evidence.
    """
    if value is None:
        return None
    versions = {v["version"]: v for v in (_CONTACT_DOMAIN_POSTURE_GUARD,
        _CONTACT_DOMAIN_POSTURE_GUARD_V2, _CONTACT_DOMAIN_POSTURE_GUARD_V3)}
    version = value.get("version") if isinstance(value, dict) else None
    expected = versions.get(version, _CONTACT_DOMAIN_POSTURE_GUARD) if type(version) is str else _CONTACT_DOMAIN_POSTURE_GUARD
    _exact(value, expected, "contact_domain_posture_guard")
    return deepcopy(value)


def contact_domain_posture_guard_contract(version=None):
    """Select a guard explicitly; the historical thumb2 guard is the default."""
    if version is None or version == _CONTACT_DOMAIN_POSTURE_GUARD["version"]:
        return deepcopy(_CONTACT_DOMAIN_POSTURE_GUARD)
    if version == _CONTACT_DOMAIN_POSTURE_GUARD_V2["version"]:
        return deepcopy(_CONTACT_DOMAIN_POSTURE_GUARD_V2)
    if version == _CONTACT_DOMAIN_POSTURE_GUARD_V3["version"]:
        return deepcopy(_CONTACT_DOMAIN_POSTURE_GUARD_V3)
    raise ValueError("Unknown explicitly selected contact-domain guard version")


def validate_full_lift_protocol(value):
    """Validate the exact new task; ``None`` preserves the local protocol."""
    if value is None:
        return None
    versions = {v["version"]: v for v in
                (_FULL_LIFT_PROTOCOL_V1, _FULL_LIFT_PROTOCOL,
                 _FULL_LIFT_PROTOCOL_V3, _FULL_LIFT_PROTOCOL_V4, _FULL_LIFT_PROTOCOL_V5_HOME)}
    version = value.get("version") if isinstance(value, dict) else None
    expected = versions.get(version, _FULL_LIFT_PROTOCOL) if type(version) is str else _FULL_LIFT_PROTOCOL
    _exact(value, expected, "full_lift")
    return deepcopy(value)


def contact_transition_control_contract(version=None):
    """Coordinated contact acquisition and unloading; not a force guarantee."""
    if version is None or version == _CONTACT_TRANSITION_CONTROL["version"]:
        return deepcopy(_CONTACT_TRANSITION_CONTROL)
    if version == _CONTACT_TRANSITION_CONTROL_V2["version"]:
        return deepcopy(_CONTACT_TRANSITION_CONTROL_V2)
    if version == _CONTACT_TRANSITION_CONTROL_V3["version"]:
        return deepcopy(_CONTACT_TRANSITION_CONTROL_V3)
    raise ValueError("Unknown explicitly selected contact-transition version")


def validate_contact_transition_control(value):
    if value is None:
        return None
    versions = {v["version"]: v for v in
                (_CONTACT_TRANSITION_CONTROL, _CONTACT_TRANSITION_CONTROL_V2, _CONTACT_TRANSITION_CONTROL_V3)}
    version = value.get("version") if isinstance(value, dict) else None
    expected = versions.get(version, _CONTACT_TRANSITION_CONTROL) if type(version) is str else _CONTACT_TRANSITION_CONTROL
    _exact(value, expected, "contact_transition_control")
    return deepcopy(value)


def validate_balanced_effort_controller_contract(value):
    """Validate explicitly declared controller knobs without importing torch."""
    _keys(value, _BALANCED_DEFAULTS, "balanced_controller")
    for key in ("version", "kind", "reference_semantics"):
        _exact(value[key], _BALANCED_DEFAULTS[key], "balanced_controller." + key)
    parameters = value["parameters"]
    defaults = _BALANCED_DEFAULTS["parameters"]
    _keys(parameters, defaults, "balanced_controller.parameters")
    for key in ("physics_dt_s", "group_force_limit_n", "maximum_penetration_m"):
        _exact(parameters[key], defaults[key], "balanced_controller.parameters." + key)
    targets = parameters["group_force_targets_n"]
    if type(targets) is not list or len(targets) != 3:
        raise ValueError("group_force_targets_n requires three ordered values")
    for target in targets:
        _number(target, 0.0, 1.5, "group force target", positive=True)
    limits = parameters["joint_torque_limits_nm"]
    if type(limits) is not list or len(limits) != 7:
        raise ValueError("joint_torque_limits_nm requires seven ordered values")
    for cap, maximum in zip(limits, _NATIVE_EFFORT_UPPER_NM):
        _number(cap, 0.0, maximum, "controller torque limit", positive=True)
    ranges = {
        "normal_force_feedback_gain": (0.0, 20.0),
        "normal_velocity_damping_n_s_m": (0.0, 100.0),
        "group_force_command_lower_n": (-1.5, 0.0),
        "group_force_command_upper_n": (0.0, 1.5),
        "joint_stiffness_nm_rad": (0.0, 9.0),
        "joint_damping_nm_s_rad": (0.0, 1.0),
        "contact_feedback_onset_n": (0.0, 1.5),
        "normal_jacobian_rank_tolerance_m": (1e-8, 0.01),
        "overforce_opening_torque_nm": (0.0, min(limits)),
        "reacquisition_velocity_limit_rad_s": (0.0, 0.04),
    }
    for key, (low, high) in ranges.items():
        _number(parameters[key], low, high, "balanced_controller.parameters." + key)
    if parameters["contact_feedback_onset_n"] >= min(targets):
        raise ValueError("contact onset must be below every positive force target")
    if parameters["group_force_command_upper_n"] < max(targets):
        raise ValueError("force command range must contain the declared force targets")
    if parameters["group_force_command_lower_n"] >= 0 or parameters["overforce_opening_torque_nm"] <= 0:
        raise ValueError("overforce control must retain a bounded opening command")
    return deepcopy(value)


def validate_effort_grasp_contract(value):
    """Validate a full JSON declaration and return an independent copy.

    Torque caps, compliant K/C, wall budget, and bounded raw windows may be
    explicitly chosen within documented ranges.  Population, timing, physical
    acceptance, friction and drive-disable semantics cannot drift here.
    """
    _keys(value, _DEFAULTS, "effort_grasp")
    full_lift = value["version"] == FULL_LIFT_EFFORT_GRASP_VERSION
    expected = full_lift_effort_grasp_contract() if full_lift else _DEFAULTS
    for key in ("version", "kind", "joint_order", "acceptance", "contact_generation"):
        _exact(value[key], expected[key], "effort_grasp." + key)

    actuation = value["actuation"]
    _keys(actuation, _DEFAULTS["actuation"], "actuation")
    for key in actuation:
        if key != "total_torque_caps_nm":
            _exact(actuation[key], _DEFAULTS["actuation"][key], "actuation." + key)
    caps = actuation["total_torque_caps_nm"]
    if type(caps) is not list or len(caps) != 7:
        raise ValueError("total_torque_caps_nm requires seven ordered values")
    for i, (cap, maximum) in enumerate(zip(caps, _NATIVE_EFFORT_UPPER_NM)):
        _number(cap, 0.0, maximum, "total_torque_caps_nm[" + str(i) + "]", positive=True)
    controller = validate_balanced_effort_controller_contract(value["controller"])
    if controller["parameters"]["joint_torque_limits_nm"] != caps:
        raise ValueError("controller and actuation total torque caps must agree")

    material = value["material"]
    _keys(material, _DEFAULTS["material"], "material")
    variable = {"variant", "compliant_contact_stiffness_n_m", "compliant_contact_damping_ns_m",
                "compliance_candidate_body_names", "always_rigid_body_names"}
    for key in material:
        if key not in variable:
            _exact(material[key], _DEFAULTS["material"][key], "material." + key)
    variant = material["variant"]
    if type(variant) is not str or variant not in ("hard_v4_control", "finger_compliant_v5"):
        raise ValueError("unknown material variant")
    k, c = material["compliant_contact_stiffness_n_m"], material["compliant_contact_damping_ns_m"]
    _number(k, 0.0, 1000000.0, "material stiffness")
    _number(c, 0.0, 1000.0, "material damping")
    if variant == "hard_v4_control" and (k != 0 or c != 0):
        raise ValueError("hard-v4 requires zero compliant stiffness and damping")
    if variant == "finger_compliant_v5" and k <= 0:
        raise ValueError("compliant finger material requires positive stiffness")
    soft_names = material["compliance_candidate_body_names"]
    if (type(soft_names) is not list or not soft_names or
            any(type(name) is not str or name not in FINGER_BODY_NAMES for name in soft_names) or
            len(set(soft_names)) != len(soft_names)):
        raise ValueError("compliance candidates require a unique nonempty subset of the seven finger bodies")
    hard_names = list(ALWAYS_RIGID_BODY_NAMES) + [name for name in FINGER_BODY_NAMES if name not in soft_names]
    _exact(material["always_rigid_body_names"], hard_names, "material.always_rigid_body_names")

    evaluation = value["evaluation"]
    _keys(evaluation, _DEFAULTS["evaluation"], "evaluation")
    quota = (_FULL_LIFT_EVALUATION_QUOTAS if full_lift else
             {"calibration8_v1": (1, 91, 6.0), "acceptance32_v1": (4, 364, 6.0)})
    name = evaluation["protocol"]
    if type(name) is not str or name not in quota:
        raise ValueError("unknown effort evaluation protocol")
    for key in evaluation:
        if key not in ("protocol", "num_envs", "episodes_per_env", "max_control_steps", "max_wall_time_s",
                       "nominal_episode_budget_s"):
            _exact(evaluation[key], expected["evaluation"][key], "evaluation." + key)
    _exact(evaluation["num_envs"], 1 if name == "home_lift1_v1" else expected["evaluation"]["num_envs"], "evaluation.num_envs")
    episodes, steps, duration = quota[name]
    _exact(evaluation["nominal_episode_budget_s"], duration, "evaluation.nominal_episode_budget_s")
    if type(evaluation["episodes_per_env"]) is not int or evaluation["episodes_per_env"] != episodes:
        raise ValueError("evaluation quota does not match its named protocol")
    if type(evaluation["max_control_steps"]) is not int or evaluation["max_control_steps"] != steps:
        raise ValueError("control-step budget does not match named evaluation protocol")
    _number(evaluation["max_wall_time_s"], 1.0, effort_evaluation_wall_time_limit_s(value), "max_wall_time_s")

    logging = value["logging"]
    _keys(logging, _DEFAULTS["logging"], "logging")
    windows = {"raw_pre_steps", "raw_post_steps", "raw_max_trigger_windows", "raw_max_samples"}
    for key in logging:
        if key not in windows:
            _exact(logging[key], _DEFAULTS["logging"][key], "logging." + key)
    _integer(logging["raw_pre_steps"], 0, 128, "raw_pre_steps")
    _integer(logging["raw_post_steps"], 0, 128, "raw_post_steps")
    _integer(logging["raw_max_trigger_windows"], 1, 128, "raw_max_trigger_windows")
    _integer(logging["raw_max_samples"], 1, 4096, "raw_max_samples")
    if 1 + logging["raw_pre_steps"] + logging["raw_post_steps"] > logging["raw_max_samples"]:
        raise ValueError("raw sample budget cannot contain one declared trigger window")
    return deepcopy(value)
