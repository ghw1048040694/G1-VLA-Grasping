"""Measured joint target control for the declared, untrained pinch intervention.

This is an outer force admittance around the unchanged native joint PD drives.
It bounds requests, not real contact force. Native contact/support qualification,
joint/anchor/speed bounds, episode deadlines and the arm controller stay outside
this module. In particular, a contact-mode latch never certifies physical support.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

from .rl_probe_protocol import compliant_pinch_controller_contract


COMPLIANT_PINCH_CONTRACT = compliant_pinch_controller_contract()
COMPLIANT_PINCH_PARAMETERS = copy.deepcopy(COMPLIANT_PINCH_CONTRACT["parameters"])
COMPLIANT_PINCH_VERSION = COMPLIANT_PINCH_CONTRACT["version"]


def _exact_parameters(parameters):
    expected = compliant_pinch_controller_contract()["parameters"]
    if parameters is None:
        return expected
    if (not isinstance(parameters, dict) or set(parameters) != set(expected) or
            any(type(parameters[k]) is not type(v) or parameters[k] != v
                for k, v in expected.items())):
        raise ValueError("Compliant pinch parameters differ from the declared controller contract")
    return copy.deepcopy(parameters)


class CompliantPinchController:
    """Traverse the five candidate waypoints with explicit force feedback.

    The first waypoint admits a named 25 mrad kinematic near-arrival. Later
    reorientation waypoints check only the joints they move, so a finger already
    pressing the Cube cannot indefinitely prevent thumb participation. The final
    waypoint still requires measured arrival or a separately evidenced physical
    contact equilibrium; neither route substitutes for the original preload gate.

    Contact-mode targets are recomputed from the last actually applied target,
    avoiding a hidden force-integral windup. After a load disappears the mode
    remains latched and reclosure remains slow until the episode resets.
    """

    def __init__(self, torch, initial_hand, state, candidate, *, parameters=None):
        self.torch = torch
        self.parameters = _exact_parameters(parameters)
        q = state["q_action_rad"]
        if q.ndim != 2 or q.shape[1] != 14 or not bool(torch.isfinite(q).all()):
            raise ValueError("Compliant pinch needs finite measured 14-joint reset positions")
        self.num_envs = q.shape[0]
        self.seed = torch.as_tensor(initial_hand, dtype=q.dtype, device=q.device)[None].expand(
            self.num_envs, -1).clone()
        if self.seed.shape != (self.num_envs, 7):
            raise ValueError("Compliant pinch requires seven source hand joints")
        try:
            frames = candidate["geometry_evidence"]["source_detail"]["stage_keyframes"]
            source = torch.as_tensor(frames[0]["target_rad"], dtype=q.dtype, device=q.device)
            goals = torch.as_tensor([x["target_rad"] for x in frames[1:]], dtype=q.dtype, device=q.device)
            durations = torch.as_tensor([x["duration_s"] for x in frames[1:]], dtype=q.dtype, device=q.device)
            names = [x["name"] for x in frames[1:]]
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError("Compliant pinch needs the verified five-stage geometry candidate") from exc
        if (source.shape != (7,) or goals.shape != (5, 7) or durations.shape != (5,) or
                not bool(torch.isfinite(goals).all()) or not bool(torch.isfinite(durations).all()) or
                not bool((durations > 0).all()) or
                not bool(torch.allclose(self.seed, source[None].expand_as(self.seed), atol=1e-4, rtol=0)) or
                not bool(torch.allclose(q[:, 7:], self.seed, atol=1e-4, rtol=0))):
            raise ValueError("Compliant pinch geometry or actual reset seed is invalid")
        limits = state["joint_limits_action_rad"][:, 7:]
        if limits.shape != (self.num_envs, 7, 2) or not bool(torch.isfinite(limits).all()):
            raise ValueError("Compliant pinch requires finite native hand limits")
        if bool(((goals[None] < limits[:, None, :, 0]) |
                 (goals[None] > limits[:, None, :, 1])).any()):
            raise ValueError("Compliant pinch geometry violates native joint limits")
        adaptive_floor = goals[-1, 0]-self.parameters["adaptive_thumb0_closing_budget_rad"]
        if bool((adaptive_floor < limits[:, 0, 0]).any()):
            raise ValueError("Compliant pinch adaptive thumb budget violates native joint limits")
        previous = torch.cat((source[None], goals[:-1]), dim=0)
        self.active = (goals - previous).abs() > 1e-7
        # This candidate first places index/middle, then thumb1, thumb2,
        # thumb0, then makes the bounded closing overtravel. Do not silently
        # apply a force-axis interpretation to a different waypoint topology.
        expected_active = torch.tensor([
            [0, 0, 0, 1, 1, 1, 1], [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 1]], dtype=torch.bool, device=q.device)
        if not bool(torch.equal(self.active, expected_active)):
            raise ValueError("Compliant pinch waypoint topology differs from its force-control axes")
        self.axis = torch.sign(goals[-1] - goals[-2])
        expected_axis = torch.tensor([-1., 0., 0., 1., 1., 1., 1.], dtype=q.dtype, device=q.device)
        if not bool(torch.equal(self.axis, expected_axis)):
            raise ValueError("Compliant pinch closing directions differ from the verified geometry")
        self.candidate_durations = durations.clone()
        durations[0] = self.parameters["stage0_duration_s"]
        self.goals, self.durations, self.names = goals, durations, names
        self.group_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2], device=q.device)
        self.ids = state["episode_id"].clone()
        self.stage_index = torch.zeros_like(self.ids)
        self.stage_elapsed = torch.zeros_like(q[:, 0])
        self.stage_start = self.seed.clone()
        self.contact_latched = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=q.device)
        self.most_closed_applied = self.seed.clone()
        self.adaptive_search_started = torch.zeros(self.num_envs, dtype=torch.bool, device=q.device)
        self.stage0_contact_ready_count = torch.zeros_like(self.ids)

    def _measurements(self, state):
        """Keep unavailable/stale telemetry distinct from a measured zero force."""
        t, n = self.torch, self.num_envs
        exemplar = self.seed[:, 0]
        false = t.zeros(n, dtype=t.bool, device=exemplar.device)
        zeros = t.zeros(n, 3, dtype=exemplar.dtype, device=exemplar.device)
        metrics = state.get("metrics")
        required = {
            "group_physical_loaded_contact_count": (n, 3),
            "no_positive_gap_hand_load": (n,),
            "friction_partner_manifold_certified": (n,),
            "maximum_penetration_m": (n,), "finite": (n,), "joint_safe": (n,),
            "table_pair_measurement_supported": (n,),
        }
        values = {"group_normal_force_n": (n, 3), "contact_metrics_current_episode": (n,),
                  "unsafe_latched": (n,)}
        if (not isinstance(metrics, dict) or
                any(k not in metrics or not t.is_tensor(metrics[k]) or tuple(metrics[k].shape) != shape
                    for k, shape in required.items()) or
                any(k not in state or not t.is_tensor(state[k]) or tuple(state[k].shape) != shape
                    for k, shape in values.items())):
            return {"available": false, "force": zeros, "physical": zeros,
                    "no_gap": false, "manifold": false,
                    "depth": t.full_like(exemplar, float("inf")), "unsafe": ~false,
                    "reason": "missing_or_malformed_contact_telemetry"}
        bool_metric_keys = ("no_positive_gap_hand_load", "friction_partner_manifold_certified",
                            "finite", "joint_safe", "table_pair_measurement_supported")
        if (any(metrics[k].dtype != t.bool for k in bool_metric_keys) or
                state["contact_metrics_current_episode"].dtype != t.bool or
                state["unsafe_latched"].dtype != t.bool):
            return {"available": false, "force": zeros, "physical": zeros,
                    "no_gap": false, "manifold": false,
                    "depth": t.full_like(exemplar, float("inf")), "unsafe": ~false,
                    "reason": "invalid_boolean_contact_telemetry"}
        force = state["group_normal_force_n"]
        physical = metrics["group_physical_loaded_contact_count"]
        depth = metrics["maximum_penetration_m"]
        valid = (t.isfinite(force).all(-1) & (force >= 0).all(-1) &
                 t.isfinite(physical).all(-1) & (physical >= 0).all(-1) &
                 t.isfinite(depth) & (depth >= 0) &
                 state["contact_metrics_current_episode"].bool() &
                 metrics["table_pair_measurement_supported"].bool() &
                 metrics["finite"].bool() & metrics["joint_safe"].bool())
        # Raw Fn may be positive at a solver contact offset. It can trigger
        # conservative force feedback, but cannot meet physical contact readiness.
        return {"available": valid, "force": force, "physical": physical,
                "no_gap": metrics["no_positive_gap_hand_load"].bool(),
                "manifold": metrics["friction_partner_manifold_certified"].bool(),
                "depth": depth, "unsafe": state["unsafe_latched"].bool(),
                "reason": "current_episode_native_contact_metrics"}

    def target(self, state, changed, dt, force_limit, tracking):
        """Return a hand request; the shared pipeline owns applied-target bounds."""
        t, p, n = self.torch, self.parameters, self.num_envs
        if (not isinstance(dt, (int, float)) or not math.isfinite(dt) or not 0 < dt <= .1 or
                not isinstance(force_limit, (int, float)) or not math.isfinite(force_limit) or
                force_limit <= p["force_target_n"]):
            raise ValueError("Compliant pinch needs finite control dt and the original positive force limit")
        if (not t.is_tensor(changed) or tuple(changed.shape) != (n,) or
                not t.is_tensor(tracking) or tuple(tracking.shape) != (n,)):
            raise ValueError("Compliant pinch needs one reset/tracking flag per environment")
        actual = state["q_action_rad"][:, 7:]
        applied = state["target_action_rad"][:, 7:]
        if actual.shape != self.seed.shape or applied.shape != self.seed.shape:
            raise ValueError("Compliant pinch hand feedback dimensions changed")
        changed = changed.bool() | (state["episode_id"] != self.ids)
        if bool(changed.any()) and not bool(t.allclose(
                actual[changed], self.seed[changed], atol=1e-4, rtol=0)):
            raise ValueError("Compliant pinch episode reset differs from the measured source hand")
        self.ids = state["episode_id"].clone()
        self.stage_index[changed] = 0
        self.stage_elapsed[changed] = 0
        self.stage_start[changed] = self.seed[changed]
        self.contact_latched[changed] = False
        self.most_closed_applied[changed] = self.seed[changed]
        self.adaptive_search_started[changed] = False
        self.stage0_contact_ready_count[changed] = 0

        m = self._measurements(state)
        measured_finite = t.isfinite(actual).all(-1) & t.isfinite(applied).all(-1)
        available = m["available"] & measured_finite
        safe_state = available & ~m["unsafe"] & (m["depth"] <= p["maximum_penetration_m"])
        force = t.where(available[:, None], m["force"], t.zeros_like(m["force"]))
        force_safe = (force <= force_limit).all(-1)
        operate = safe_state & tracking.bool()
        timer_run = operate & force_safe
        index = self.stage_index.clone()
        goal = self.goals[index]
        self.stage_elapsed += t.where(timer_run, dt, 0.)
        fraction = (self.stage_elapsed / self.durations[index]).clamp(0, 1)
        smooth = fraction.square() * (3 - 2 * fraction)
        nominal = self.stage_start + (goal-self.stage_start) * smooth[:, None]
        final = index == len(self.names)-1
        self.adaptive_search_started |= final & (fraction >= 1) & operate
        self.contact_latched |= (available[:, None] & (force > p["contact_feedback_onset_n"]))
        feedback_mode = self.contact_latched.clone()
        feedback_mode[:, 0] |= self.adaptive_search_started
        feedback_joints = feedback_mode[:, self.group_ids] & (self.axis[None] != 0)
        more_closed = (applied-self.most_closed_applied)*self.axis[None] > 0
        self.most_closed_applied = t.where(feedback_joints & available[:, None] & more_closed,
                                           applied, self.most_closed_applied)

        velocity = ((p["force_target_n"]-force) * p["admittance_gain_rad_s_per_n"]).clamp(
            min=-p["retreat_speed_max_rad_s"], max=p["close_speed_max_rad_s"])
        raw_admittance = applied + self.axis[None] * velocity[:, self.group_ids] * dt
        # The current nominal waypoint bounds all closing motion. Retreat stays
        # within 40 mrad of the most closed *actually applied* target after
        # contact, not an unreachable nominal that would force a stalled finger
        # to close when force feedback says to open. Never open beyond the seed.
        geometry_low, geometry_high = t.minimum(self.seed, nominal), t.maximum(self.seed, nominal)
        # This is a declared geometry delta: only the final thumb0 may search
        # below its original nominal, only after that nominal segment finishes.
        # It uses exactly the same force-error law, including opening at raw
        # positive-gap force above the setpoint. It never forces through that
        # equilibrium merely to manufacture a physically loaded contact.
        adaptive_thumb_floor = self.goals[-1, 0]-p["adaptive_thumb0_closing_budget_rad"]
        geometry_low[:, 0] = t.where(self.adaptive_search_started,
                                    adaptive_thumb_floor, geometry_low[:, 0])
        retreat_edge = self.most_closed_applied-self.axis[None]*p["maximum_retreat_rad"]
        compliance_low = t.where(self.axis[None] > 0, t.maximum(geometry_low, retreat_edge), geometry_low)
        compliance_high = t.where(self.axis[None] < 0, t.minimum(geometry_high, retreat_edge), geometry_high)
        # A projection must not reverse or accelerate a requested admittance
        # step. Intersect its one-control segment with all other request bounds.
        compliance_low = t.maximum(compliance_low, t.minimum(applied, raw_admittance))
        compliance_high = t.minimum(compliance_high, t.maximum(applied, raw_admittance))
        bounded_admittance = t.minimum(t.maximum(raw_admittance, compliance_low), compliance_high)
        before_lead = t.where(feedback_joints, bounded_admittance, nominal)
        lead = t.where(feedback_mode[:, self.group_ids],
            p["contact_target_lead_max_rad"], p["free_target_lead_max_rad"])
        request_low = t.maximum(geometry_low, actual-lead)
        request_high = t.minimum(geometry_high, actual+lead)
        request_low = t.where(feedback_joints, t.maximum(request_low, compliance_low), request_low)
        request_high = t.where(feedback_joints, t.minimum(request_high, compliance_high), request_high)
        feasible_joints = request_low <= request_high
        feasible = feasible_joints.all(-1)
        desired = t.minimum(t.maximum(before_lead, request_low), request_high)
        # On unavailable/unsafe/infeasible feedback retain the last target. A
        # nonfinite target itself has no safe write; use the verified seed as a
        # finite fail-closed request while the environment's unsafe latch aborts.
        hold = t.where(t.isfinite(applied), applied, self.seed)
        force_excess_joints = (force > force_limit)[:, self.group_ids] & (self.axis[None] != 0)
        # Palm lag pauses closing, but available non-unsafe force spikes still
        # get bounded retreat on the affected group's closing axes. Unknown or
        # unsafe measurements take priority and hold instead.
        movable_joints = safe_state[:, None] & feasible_joints & (
            tracking.bool()[:, None] | force_excess_joints)
        desired = t.where(movable_joints, desired, hold)
        actual_relief_joints = movable_joints & ((desired-applied)*self.axis[None] < -1e-9)
        actual_relief = t.stack([actual_relief_joints[:, self.group_ids == group].any(-1)
                                for group in range(3)], dim=-1)

        active = self.active[index]
        errors = (actual-goal).abs()
        active_error = t.where(active, errors, 0.).amax(-1)
        all_error = errors.amax(-1)
        threshold = t.where(index == 0, p["stage0_near_arrival_rad"], p["stage_active_arrival_rad"])
        kinematic_arrival = active_error <= threshold
        stage0_force_near = ((index == 0) & (fraction >= 1) & timer_run & feasible &
            (active_error <= p["stage0_contact_near_arrival_rad"]) &
            (force[:, 1:] >= p["stage0_contact_force_min_n"]).all(-1))
        self.stage0_contact_ready_count = t.where(stage0_force_near,
            (self.stage0_contact_ready_count+1).clamp(max=p["stage0_contact_ready_controls"]),
            t.zeros_like(self.stage0_contact_ready_count))
        stage0_precontact_arrival = self.stage0_contact_ready_count >= p["stage0_contact_ready_controls"]
        physical_contact = ((force > p["contact_feedback_onset_n"]).all(-1) & force_safe &
                            (m["physical"] > 0).all(-1) & m["no_gap"] & m["manifold"] &
                            (m["depth"] <= p["maximum_penetration_m"]) & available)
        final_joint_arrival = final & (all_error <= p["final_joint_arrival_rad"])
        thumb_in_adaptive_interval = ((actual[:, 0] >= adaptive_thumb_floor) &
                                      (actual[:, 0] <= self.goals[-1, 0]))
        thumb_applied_error = (actual[:, 0]-applied[:, 0]).abs()
        adaptive_equilibrium_geometry = (
            self.adaptive_search_started & thumb_in_adaptive_interval &
            (thumb_applied_error <= p["final_contact_equilibrium_arrival_rad"]) &
            (errors[:, 1:].amax(-1) <= p["final_contact_equilibrium_arrival_rad"]))
        nominal_equilibrium_geometry = all_error <= p["final_contact_equilibrium_arrival_rad"]
        final_contact_equilibrium = (final &
                                     (nominal_equilibrium_geometry | adaptive_equilibrium_geometry) &
                                     physical_contact)
        arrival_geometry = t.where(final, final_joint_arrival | final_contact_equilibrium, kinematic_arrival)
        arrival_geometry |= (index == 0) & stage0_precontact_arrival
        arrived = (fraction >= 1) & arrival_geometry & timer_run & feasible
        advance = arrived & ~final
        self.stage_start[advance] = goal[advance]
        self.stage_index[advance] += 1
        self.stage_elapsed[advance] = 0
        evidence = {
            "compliant_pinch_controller": compliant_pinch_controller_contract(),
            "closure_stage_index": index, "closure_stage_fraction": fraction,
            "closure_stage_names": self.names, "stage_goal_hand_rad": goal,
            "closure_candidate_stage_durations_s": self.candidate_durations,
            "closure_executed_stage_durations_s": self.durations,
            "stage_active_joints": active, "stage_measured_error_max_rad": all_error,
            "stage_active_measured_error_max_rad": active_error,
            "stage_arrival_tolerance_rad": threshold,
            "stage_kinematic_arrival": kinematic_arrival,
            "stage0_kinematic_near_arrival": (index == 0) & kinematic_arrival,
            "stage0_near_arrival_is_not_physical_contact_or_support": True,
            "stage0_raw_force_precontact_signal": stage0_force_near,
            "stage0_raw_force_precontact_control_count": self.stage0_contact_ready_count.clone(),
            "stage0_raw_force_precontact_arrival": stage0_precontact_arrival,
            "stage0_raw_force_precontact_is_not_physical_contact_or_support": True,
            "stage_final_joint_arrival": final_joint_arrival,
            "stage_final_physical_contact_equilibrium": final_contact_equilibrium,
            "stage_final_physical_contact_equilibrium_is_not_support": True,
            "stage_final_adaptive_equilibrium_geometry": adaptive_equilibrium_geometry,
            "stage_final_thumb_actual_applied_tracking_error_rad": thumb_applied_error,
            "stage_advance_after_this_request": advance,
            "closure_path_complete": arrived & final,
            "closure_paused_for_palm_tracking": ~tracking.bool(),
            "closure_paused_for_invalid_or_unsafe_measurements": ~safe_state,
            "closure_paused_for_force_excess": ~force_safe,
            "contact_measurement_available": available,
            "contact_measurement_source": m["reason"],
            "force_feedback_contact_mode_latched": self.contact_latched.clone(),
            "force_feedback_mode_including_final_thumb_search": feedback_mode,
            "force_feedback_mode_is_not_physical_contact_or_support": True,
            "force_feedback_raw_group_normal_force_n": m["force"],
            "force_feedback_group_physical_loaded_count": m["physical"],
            "force_feedback_no_positive_gap_hand_load": m["no_gap"],
            "force_feedback_friction_manifold_certified": m["manifold"],
            "force_feedback_physical_contact_ready": physical_contact,
            "force_admittance_group_velocity_rad_s": velocity,
            "force_admittance_closing_axis_hand7": self.axis,
            "force_admittance_most_closed_actual_applied_target_hand_rad": self.most_closed_applied.clone(),
            "force_admittance_maximum_retreat_reference": "actual_most_closed_applied_target_after_contact",
            "force_admittance_unbounded_target_hand_rad": raw_admittance,
            "force_admittance_bounded_target_hand_rad": bounded_admittance,
            "force_admittance_feedback_joint_mask": feedback_joints,
            "planned_hand_before_force_admittance_rad": nominal,
            "planned_hand_before_tracking_lead_rad": before_lead,
            "guided_target_hand_rad": desired,
            "guided_target_lead_max_rad": lead,
            "hand_request_constraints_feasible": feasible,
            "hand_request_constraints_feasible_by_joint": feasible_joints,
            "hand_request_held": ~movable_joints.any(-1),
            "hand_request_held_by_joint": ~movable_joints,
            "guided_request_lead_is_not_applied_target_bound": True,
            "tracking_lead_is_not_a_force_bound": True,
            "force_feedback_does_not_hard_bound_real_force": True,
            "force_admittance_reference": "previous_actual_applied_target_no_hidden_integral",
            "raw_force_over_limit_by_group": force > force_limit,
            "force_relief_by_group": actual_relief,
            "adaptive_thumb_search_started": self.adaptive_search_started.clone(),
            "adaptive_thumb_search_active": self.adaptive_search_started & ~(
                (m["physical"][:, 0] > 0) & m["no_gap"] & m["manifold"] & available),
            "adaptive_thumb_search_suppressed_by_force_error": self.adaptive_search_started &
                (velocity[:, 0] <= 0),
            "adaptive_thumb_search_request_enabled": self.adaptive_search_started & movable_joints[:, 0],
            "adaptive_thumb_search_paused_for_palm_tracking": self.adaptive_search_started & ~tracking.bool(),
            "adaptive_thumb_search_at_closing_budget": self.adaptive_search_started &
                (applied[:, 0] <= adaptive_thumb_floor+1e-7),
            "adaptive_thumb_raw_force_without_certified_physical_contact":
                (force[:, 0] > p["contact_feedback_onset_n"]) & ~(
                    (m["physical"][:, 0] > 0) & m["no_gap"] & m["manifold"] & available),
            "adaptive_thumb_minimum_joint_rad": adaptive_thumb_floor,
            "adaptive_thumb_nominal_final_joint_rad": self.goals[-1, 0],
            "adaptive_thumb_search_uses_unchanged_force_error_law": True,
        }
        return desired, fraction, evidence


def _physics_contract_and_parameters(parameters):
    from .rl_probe_protocol import physics_pinch_controller_contract
    contract = physics_pinch_controller_contract()
    expected = contract["parameters"]
    if parameters is not None and (
            not isinstance(parameters, dict) or set(parameters) != set(expected) or
            any(type(parameters[k]) is not type(v) or parameters[k] != v
                for k, v in expected.items())):
        raise ValueError("Physics pinch parameters differ from the declared controller contract")
    return contract, copy.deepcopy(expected)


class PhysicsPinchTrajectory:
    """66ms candidate reference/phase machine; it never integrates force targets.

    Its independent physics-step follower owns all actual hand writes. This
    machine only advances declared geometry phases and checks the original
    path/preload prerequisites. A raw-force near-contact signal may release the
    thumb phases; it never becomes a physical-contact or support certificate.
    """

    def __init__(self, torch, initial_hand, state, candidate, *, parameters=None):
        self.contract, self.parameters = _physics_contract_and_parameters(parameters)
        # Reuse the unchanged candidate parser and native-limit validation only.
        # Never call the v1 force target method or keep its force latch/integral.
        geometry = CompliantPinchController(torch, initial_hand, state, candidate)
        for name in ("torch", "num_envs", "seed", "goals", "candidate_durations",
                     "names", "active", "axis", "group_ids"):
            setattr(self, name, getattr(geometry, name))
        self.durations = self.candidate_durations.clone()
        self.durations[0] = self.parameters["stage0_duration_s"]
        if bool((self.goals[-1, 0]-self.parameters["adaptive_thumb0_closing_budget_rad"] <
                 state["joint_limits_action_rad"][:, 7, 0]).any()):
            raise ValueError("Physics pinch adaptive thumb budget violates native joint limits")
        self.ids = state["episode_id"].clone()
        self.stage_index = torch.zeros_like(self.ids)
        self.stage_elapsed = torch.zeros_like(self.seed[:, 0])
        self.stage_start = self.seed.clone()
        self.stage0_contact_ready_count = torch.zeros_like(self.ids)

    _measurements = CompliantPinchController._measurements

    def reference(self, state, changed, dt, force_limit, tracking):
        t, p, n = self.torch, self.parameters, self.num_envs
        if (type(dt) not in (int, float) or not math.isfinite(dt) or not 0 < dt <= .1 or
                type(force_limit) not in (int, float) or not math.isfinite(force_limit) or
                force_limit <= p["force_target_n"]):
            raise ValueError("Physics pinch reference needs finite control dt and force limit")
        if (tuple(changed.shape) != (n,) or tuple(tracking.shape) != (n,) or
                tracking.dtype != t.bool):
            raise ValueError("Physics pinch reference needs per-environment reset/tracking flags")
        actual, applied = state["q_action_rad"][:, 7:], state["target_action_rad"][:, 7:]
        changed = changed.bool() | (state["episode_id"] != self.ids)
        if bool(changed.any()) and not bool(t.allclose(
                actual[changed], self.seed[changed], atol=1e-4, rtol=0)):
            raise ValueError("Physics pinch reference reset differs from measured source hand")
        self.ids = state["episode_id"].clone()
        self.stage_index[changed] = 0
        self.stage_elapsed[changed] = 0
        self.stage_start[changed] = self.seed[changed]
        self.stage0_contact_ready_count[changed] = 0
        m = self._measurements(state)
        available = m["available"] & t.isfinite(actual).all(-1) & t.isfinite(applied).all(-1)
        force = t.where(available[:, None], m["force"], t.zeros_like(m["force"]))
        force_safe = (force <= force_limit).all(-1)
        safe = available & ~m["unsafe"] & (m["depth"] <= p["maximum_penetration_m"])
        progress = safe & tracking & force_safe
        index = self.stage_index.clone()
        goal = self.goals[index]
        self.stage_elapsed += t.where(progress, dt, 0.)
        fraction = (self.stage_elapsed/self.durations[index]).clamp(0, 1)
        smooth = fraction.square()*(3-2*fraction)
        nominal = self.stage_start+(goal-self.stage_start)*smooth[:, None]
        final = index == len(self.names)-1
        search = final & (fraction >= 1)
        floor = self.goals[-1, 0]-p["adaptive_thumb0_closing_budget_rad"]
        low, high = t.minimum(self.seed, nominal), t.maximum(self.seed, nominal)
        low[:, 0] = t.where(search, floor, low[:, 0])
        errors = (actual-goal).abs()
        active_error = t.where(self.active[index], errors, 0.).amax(-1)
        all_error = errors.amax(-1)
        tolerance = t.where(index == 0, p["stage0_near_arrival_rad"], p["stage_active_arrival_rad"])
        kinematic = active_error <= tolerance
        near_signal = ((index == 0) & (fraction >= 1) & progress &
            (active_error <= p["stage0_contact_near_arrival_rad"]) &
            (force[:, 1:] >= p["stage0_contact_force_min_n"]).all(-1))
        self.stage0_contact_ready_count = t.where(near_signal,
            (self.stage0_contact_ready_count+1).clamp(max=p["stage0_contact_ready_controls"]),
            t.zeros_like(self.stage0_contact_ready_count))
        near_arrival = self.stage0_contact_ready_count >= p["stage0_contact_ready_controls"]
        physical = ((force > p["contact_feedback_onset_n"]).all(-1) & force_safe &
            (m["physical"] > 0).all(-1) & m["no_gap"] & m["manifold"] & safe)
        nominal_final_arrival = final & (all_error <= p["final_joint_arrival_rad"])
        equilibrium_low = t.minimum(self.seed, self.goals[-1][None]).clone()
        equilibrium_high = t.maximum(self.seed, self.goals[-1][None]).clone()
        equilibrium_low[:, 0] = floor
        equilibrium_tolerance = p["final_contact_equilibrium_arrival_rad"]
        force_axes = self.axis != 0
        actual_force_axes_in_domain = (
            (actual[:, force_axes] >= equilibrium_low[:, force_axes]-equilibrium_tolerance) &
            (actual[:, force_axes] <= equilibrium_high[:, force_axes]+equilibrium_tolerance)).all(-1)
        orientation_error = errors[:, ~force_axes].amax(-1)
        adaptive_geometry = (search & actual_force_axes_in_domain &
            ((actual-applied).abs().amax(-1) <= equilibrium_tolerance) &
            (orientation_error <= equilibrium_tolerance))
        opposed_side = t.zeros(n, dtype=t.bool, device=actual.device)
        metrics = state.get("metrics", {})
        if (all(t.is_tensor(metrics.get(k)) and tuple(metrics[k].shape) == (n,)
                for k in ("contact_safe", "opposition_dot", "maximum_abs_normal_z")) and
                metrics["contact_safe"].dtype == t.bool):
            opposed_side = (metrics["contact_safe"] & t.isfinite(metrics["opposition_dot"]) &
                t.isfinite(metrics["maximum_abs_normal_z"]) &
                (metrics["opposition_dot"] <= -.6) & (metrics["opposition_dot"] >= -1) &
                (metrics["maximum_abs_normal_z"] >= 0) & (metrics["maximum_abs_normal_z"] <= .6))
        equilibrium = final & physical & opposed_side & adaptive_geometry
        arrival = t.where(final, nominal_final_arrival | equilibrium, kinematic)
        arrival |= (index == 0) & near_arrival
        arrived = (fraction >= 1) & progress & arrival
        advance = arrived & ~final
        self.stage_start[advance] = goal[advance]
        self.stage_index[advance] += 1
        self.stage_elapsed[advance] = 0
        evidence = {
            "physics_pinch_controller": copy.deepcopy(self.contract),
            "macro_reference_only_no_force_target_update": True,
            "guided_target_hand_rad": nominal,
            "nominal_hand_rad": nominal,
            "geometry_lower_hand_rad": low, "geometry_upper_hand_rad": high,
            "force_admittance_closing_axis_hand7": self.axis,
            "closure_stage_index": index, "closure_stage_fraction": fraction,
            "closure_stage_names": self.names, "stage_goal_hand_rad": goal,
            "stage_active_joints": self.active[index],
            "stage_active_measured_error_max_rad": active_error,
            "stage_measured_error_max_rad": all_error,
            "stage_arrival_tolerance_rad": tolerance,
            "stage_kinematic_arrival": kinematic,
            "stage0_kinematic_near_arrival": (index == 0) & kinematic,
            "stage0_raw_force_precontact_signal": near_signal,
            "stage0_raw_force_precontact_control_count": self.stage0_contact_ready_count.clone(),
            "stage0_raw_force_precontact_arrival": near_arrival,
            "stage0_raw_force_precontact_is_not_physical_contact_or_support": True,
            "stage_advance_after_this_request": advance,
            "closure_path_complete": arrived & final,
            "stage_final_joint_arrival": nominal_final_arrival,
            "stage_final_physical_contact_equilibrium": equilibrium,
            "stage_final_adaptive_equilibrium_geometry": adaptive_geometry,
            "stage_final_original_opposition_and_side_gates": opposed_side,
            "stage_final_equilibrium_thumb_source_to_extended_goal_interval_rad":
                t.stack((floor.expand_as(self.seed[:, 0]), self.seed[:, 0]), dim=-1),
            "stage_final_equilibrium_force_axis_nominal_domain_lower_rad": equilibrium_low,
            "stage_final_equilibrium_force_axis_nominal_domain_upper_rad": equilibrium_high,
            "stage_final_equilibrium_force_axis_actual_domain_tolerance_rad": equilibrium_tolerance,
            "stage_final_equilibrium_actual_force_axes_in_domain": actual_force_axes_in_domain,
            "stage_final_equilibrium_orientation_goal_error_rad": orientation_error,
            "stage_final_equilibrium_all_joint_applied_tracking_error_rad":
                (actual-applied).abs().amax(-1),
            "stage_final_equilibrium_other_six_goal_error_rad": errors[:, 1:].amax(-1),
            "stage_final_physical_contact_equilibrium_is_not_support": True,
            "closure_paused_for_palm_tracking": ~tracking,
            "closure_paused_for_invalid_or_unsafe_measurements": ~safe,
            "closure_paused_for_force_excess": ~force_safe,
            "contact_measurement_available": available,
            "force_feedback_raw_group_normal_force_n": m["force"],
            "force_feedback_group_physical_loaded_count": m["physical"],
            "force_feedback_physical_contact_ready": physical,
            "adaptive_thumb_search_started": search,
            "adaptive_thumb_minimum_joint_rad": floor,
            "closure_candidate_stage_durations_s": self.candidate_durations,
            "closure_executed_stage_durations_s": self.durations,
        }
        return nominal, fraction, evidence


class PhysicsStepForceAdmittance:
    """Causal 2ms target follower around unchanged native joint PD drives.

    Force at sample k informs the requested target for k+1. Repeated samples
    never integrate twice. All reference, step, macro, lead and retreat bounds
    constrain targets only; measured force and motion keep their original gates.
    """

    def __init__(self, torch, initial_applied_hand_rad, episode_id, *, parameters=None):
        self.contract, self.parameters = _physics_contract_and_parameters(parameters)
        self.torch = torch
        if (initial_applied_hand_rad.ndim != 2 or initial_applied_hand_rad.shape[1] != 7 or
                not bool(torch.isfinite(initial_applied_hand_rad).all())):
            raise ValueError("Physics admittance needs finite actually applied seven-joint targets")
        self.seed = initial_applied_hand_rad.detach().clone()
        self.num_envs = self.seed.shape[0]
        if tuple(episode_id.shape) != (self.num_envs,):
            raise ValueError("Physics admittance needs one episode identity per environment")
        self.ids = episode_id.clone()
        self.most_closed_applied = self.seed.clone()
        self.contact_latched = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.seed.device)
        self.seen_valid_feedback = torch.zeros(self.num_envs, dtype=torch.bool, device=self.seed.device)
        self.group_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2], device=self.seed.device)
        self.sample_id = None
        payload = json.dumps(self.contract, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()
        self.contract_sha256 = hashlib.sha256(payload).hexdigest()

    def update(self, *, sample_id, episode_id, dt, measured_hand_rad, applied_hand_rad,
               nominal_hand_rad, geometry_lower_hand_rad, geometry_upper_hand_rad,
               step_lower_hand_rad, step_upper_hand_rad, macro_lower_hand_rad, macro_upper_hand_rad,
               raw_group_force_n, contact_measurement_valid, tracking, unsafe_latched,
               penetration_m, closing_axis_hand7):
        t, p, n = self.torch, self.parameters, self.num_envs
        if (type(sample_id) is not int or sample_id < 0 or type(dt) not in (int, float) or
                not math.isfinite(dt) or dt != p["physics_dt_s"]):
            raise ValueError("Physics admittance requires the declared 2ms native sample identity")
        hand_values = (measured_hand_rad, applied_hand_rad, nominal_hand_rad,
                       geometry_lower_hand_rad, geometry_upper_hand_rad, step_lower_hand_rad,
                       step_upper_hand_rad, macro_lower_hand_rad, macro_upper_hand_rad)
        if any(not t.is_tensor(x) or tuple(x.shape) != (n, 7) for x in hand_values):
            raise ValueError("Physics admittance requires native N by seven target/bound tensors")
        if (tuple(episode_id.shape) != (n,) or tuple(raw_group_force_n.shape) != (n, 3) or
                tuple(penetration_m.shape) != (n,) or tuple(closing_axis_hand7.shape) != (7,) or
                any(tuple(x.shape) != (n,) or x.dtype != t.bool
                    for x in (contact_measurement_valid, tracking, unsafe_latched))):
            raise ValueError("Physics admittance feedback dimensions or boolean types are invalid")
        axis = closing_axis_hand7.to(device=self.seed.device, dtype=self.seed.dtype)
        if (not bool(t.isfinite(axis).all()) or not bool(((axis == 0) | (axis.abs() == 1)).all()) or
                bool((axis[1:3] != 0).any()) or axis[0] == 0 or
                not bool((axis[3:5] != 0).any()) or not bool((axis[5:] != 0).any())):
            raise ValueError("Physics admittance force axes must exclude thumb1/2 reorientation")
        changed = episode_id != self.ids
        self.ids = episode_id.clone()
        self.contact_latched[changed] = False
        self.most_closed_applied[changed] = self.seed[changed]
        self.seen_valid_feedback[changed] = False
        initial = self.sample_id is None
        duplicate = not initial and sample_id == self.sample_id
        discontinuity = not initial and not duplicate and sample_id != self.sample_id+1
        if not duplicate:
            self.sample_id = sample_id
        finite = t.ones(n, dtype=t.bool, device=self.seed.device)
        for value in hand_values:
            finite &= t.isfinite(value).all(-1)
        finite &= t.isfinite(raw_group_force_n).all(-1) & (raw_group_force_n >= 0).all(-1)
        finite &= t.isfinite(penetration_m) & (penetration_m >= 0)
        reset_matches = ((applied_hand_rad-self.seed).abs().amax(-1) <= 1e-4) | ~changed
        depth_safe = penetration_m <= p["maximum_penetration_m"]
        unavailable_unexpected = (~contact_measurement_valid & self.seen_valid_feedback &
                                  ~changed & (not duplicate))
        unsafe = (unsafe_latched | ~finite | ~reset_matches | ~depth_safe |
                  unavailable_unexpected | discontinuity)
        fresh = t.full((n,), not (initial or duplicate or discontinuity), dtype=t.bool,
                       device=self.seed.device) & ~changed
        safe = fresh & finite & contact_measurement_valid & ~unsafe
        self.seen_valid_feedback |= (finite & contact_measurement_valid & ~unsafe & ~changed &
                                     (not duplicate) & (not discontinuity))
        force = t.where((finite & contact_measurement_valid)[:, None],
                        raw_group_force_n, t.zeros_like(raw_group_force_n))
        self.contact_latched |= safe[:, None] & (force > p["contact_feedback_onset_n"])
        # The macro final envelope, not a force observation, permits searching
        # past the final thumb nominal. This mode never certifies physical touch.
        adaptive_search = ((geometry_lower_hand_rad[:, 0] < nominal_hand_rad[:, 0]-1e-7) &
                           (axis[0] < 0))
        mode = self.contact_latched.clone()
        mode[:, 0] |= adaptive_search
        feedback = mode[:, self.group_ids] & (axis[None] != 0)
        more_closed = (applied_hand_rad-self.most_closed_applied)*axis[None] > 0
        self.most_closed_applied = t.where(safe[:, None] & feedback & more_closed,
                                           applied_hand_rad, self.most_closed_applied)
        error = p["force_target_n"]-force
        velocity = (error*p["admittance_gain_rad_s_per_n"]).clamp(
            min=-p["retreat_speed_max_rad_s"], max=p["close_speed_max_rad_s"])
        deadband = error.abs() <= p["force_deadband_n"]
        velocity = t.where(deadband, 0., velocity)
        lead = t.where(mode[:, self.group_ids], p["contact_target_lead_max_rad"],
                       p["free_target_lead_max_rad"])
        lead_lower, lead_upper = measured_hand_rad-lead, measured_hand_rad+lead
        outside_previous_lead = (applied_hand_rad < lead_lower) | (applied_hand_rad > lead_upper)
        lead_recovery = mode[:, self.group_ids] & outside_previous_lead
        effective_signed_velocity = velocity[:, self.group_ids].clone()
        # Contracting the lead on first touch cannot instantaneously move an
        # already written target through the native velocity box. Recover by
        # opening a too-closed target; other lagging targets hold until measured
        # joints catch up. Never reverse a high-force opening request to close.
        too_closed = lead_recovery & feedback & t.where(
            axis[None] > 0, applied_hand_rad > lead_upper, applied_hand_rad < lead_lower)
        recovery_distance = t.where(axis[None] > 0,
            applied_hand_rad-lead_upper, lead_lower-applied_hand_rad).clamp_min(0)
        recovery_velocity = -(recovery_distance/dt).clamp(
            max=p["retreat_speed_max_rad_s"])
        effective_signed_velocity = t.where(too_closed,
            t.minimum(effective_signed_velocity, recovery_velocity),
            t.where(lead_recovery & feedback, effective_signed_velocity.clamp(max=0),
                    effective_signed_velocity))
        force_trial = applied_hand_rad+axis[None]*effective_signed_velocity*dt
        desired = t.where(feedback, force_trial, nominal_hand_rad)
        desired = t.where(lead_recovery & ~feedback, applied_hand_rad, desired)
        force_excess = (force > 1.5)[:, self.group_ids] & (axis[None] != 0)
        lead_force_priority = force_excess & feedback & (effective_signed_velocity < 0)
        relaxed_lead = lead_recovery | lead_force_priority
        low = t.maximum(t.maximum(geometry_lower_hand_rad, step_lower_hand_rad), macro_lower_hand_rad)
        high = t.minimum(t.minimum(geometry_upper_hand_rad, step_upper_hand_rad), macro_upper_hand_rad)
        low = t.where(relaxed_lead, low, t.maximum(low, lead_lower))
        high = t.where(relaxed_lead, high, t.minimum(high, lead_upper))
        retreat_edge = self.most_closed_applied-axis[None]*p["maximum_retreat_rad"]
        force_low = t.maximum(t.minimum(applied_hand_rad, force_trial),
                             t.where(axis[None] > 0, retreat_edge, low))
        force_high = t.minimum(t.maximum(applied_hand_rad, force_trial),
                              t.where(axis[None] < 0, retreat_edge, high))
        low = t.where(feedback, t.maximum(low, force_low), low)
        high = t.where(feedback, t.minimum(high, force_high), high)
        feasible_joints = low <= high
        # An impossible intersection is an explicit physical protocol failure;
        # retaining the last target is a fail-closed response, not a safe proof.
        infeasible = safe & ~feasible_joints.all(-1)
        unsafe |= infeasible
        enabled = safe[:, None] & ~unsafe[:, None] & feasible_joints & (
            tracking[:, None] | force_excess)
        bounded = t.minimum(t.maximum(desired, low), high)
        hold = t.where(t.isfinite(applied_hand_rad), applied_hand_rad, self.seed)
        target = t.where(enabled, bounded, hold)
        relief_joints = enabled & ((target-applied_hand_rad)*axis[None] < -1e-9)
        relief = t.stack([relief_joints[:, self.group_ids == g].any(-1) for g in range(3)], -1)
        evidence = {
            "controller_version": self.contract["version"],
            "controller_contract_sha256": self.contract_sha256,
            "force_sample_id": sample_id, "target_for_native_sample_id": sample_id+1,
            "physics_dt_s": dt, "first_sample_hold": initial,
            "duplicate_sample_hold": duplicate, "sample_discontinuity": discontinuity,
            "episode_reset_hold": changed, "reset_applied_target_matches_source": reset_matches,
            "contact_measurement_available": contact_measurement_valid & finite,
            "seen_valid_feedback_for_current_episode": self.seen_valid_feedback.clone(),
            "waiting_for_first_valid_measurement": ~self.seen_valid_feedback,
            "previously_valid_measurement_lost": unavailable_unexpected,
            "fresh_sample_for_current_episode": fresh,
            "unsafe": unsafe, "request_constraints_infeasible": infeasible,
            "request_constraints_feasible_by_joint": feasible_joints,
            "request_held_by_joint": ~enabled,
            "force_feedback_contact_mode_latched": self.contact_latched.clone(),
            "force_feedback_mode_including_final_thumb_search": mode,
            "force_feedback_raw_group_normal_force_n": raw_group_force_n,
            "force_feedback_error_n": error, "force_feedback_deadband": deadband,
            "force_admittance_group_velocity_rad_s": velocity,
            "force_admittance_effective_signed_joint_velocity_rad_s": effective_signed_velocity,
            "contact_lead_recovery_by_joint": lead_recovery,
            "contact_lead_recovery": lead_recovery.any(-1),
            "contact_lead_recovery_too_closed_by_joint": too_closed,
            "contact_lead_high_force_retreat_priority_by_joint": lead_force_priority,
            "actual_requested_lead_not_yet_met_by_joint": (target < lead_lower) | (target > lead_upper),
            "actual_requested_lead_not_yet_met": ((target < lead_lower) | (target > lead_upper)).any(-1),
            "force_admittance_closing_axis_hand7": axis,
            "force_admittance_previous_actually_written_hand_target_rad": applied_hand_rad,
            "force_admittance_most_closed_actual_applied_target_hand_rad": self.most_closed_applied.clone(),
            "force_admittance_unbounded_hand_target_rad": force_trial,
            "nominal_hand_rad": nominal_hand_rad,
            "bounded_hand_target_rad": target,
            "geometry_lower_hand_rad": geometry_lower_hand_rad,
            "geometry_upper_hand_rad": geometry_upper_hand_rad,
            "intersected_lower_hand_rad": low, "intersected_upper_hand_rad": high,
            "measured_hand_rad": measured_hand_rad,
            "guided_target_lead_max_rad": lead,
            "raw_force_over_limit_by_group": force > 1.5,
            "force_relief_by_group": relief,
            "adaptive_thumb_search_active": adaptive_search,
            "force_request_bounds_are_not_measured_force_bounds": True,
            "contact_mode_is_not_support_qualification": True,
            "reference": "last_actually_written_target_one_update_per_native_sample",
        }
        return target, evidence, unsafe
