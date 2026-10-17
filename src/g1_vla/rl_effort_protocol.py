"""Pure declarations for explicit hand effort and contact-material calibration.

Validation is not a measurement of native torque, contact force, or support.
The simulator must independently bind/read back the declared drive/material
properties.  This module deliberately imports neither config nor a backend.
"""

from __future__ import annotations

from copy import deepcopy
import math


EFFORT_GRASP_VERSION = "g1_explicit_effort_grasp_protocol_v3_physical_contact"
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
_BALANCED_DEFAULTS = {
    "version": "g1_balanced_normal_jacobian_effort_grasp_v1",
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
    for key in ("version", "kind", "joint_order", "acceptance", "contact_generation"):
        _exact(value[key], _DEFAULTS[key], "effort_grasp." + key)

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
    quota = {"calibration8_v1": (1, 91), "acceptance32_v1": (4, 364)}
    name = evaluation["protocol"]
    if type(name) is not str or name not in quota:
        raise ValueError("unknown effort evaluation protocol")
    for key in evaluation:
        if key not in ("protocol", "episodes_per_env", "max_control_steps", "max_wall_time_s"):
            _exact(evaluation[key], _DEFAULTS["evaluation"][key], "evaluation." + key)
    episodes, steps = quota[name]
    if type(evaluation["episodes_per_env"]) is not int or evaluation["episodes_per_env"] != episodes:
        raise ValueError("evaluation quota does not match its named protocol")
    if type(evaluation["max_control_steps"]) is not int or evaluation["max_control_steps"] != steps:
        raise ValueError("control-step budget does not match named evaluation protocol")
    _number(evaluation["max_wall_time_s"], 1.0, 3600.0, "max_wall_time_s")

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
