"""Declared untrained hand interventions with independent physical qualification.

The legacy controllers use force admittance around native joint PD drives. The
separate effort controller requires the seven hand PD drives to be disabled.
Both bound requests, not real contact force. Native contact/support qualification,
episode deadlines and the arm controller stay outside this module. In particular,
a contact-mode latch never certifies physical support.
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


def _balanced_effort_contract_and_parameters(parameters):
    """Read the separate effort declaration without changing legacy contracts."""
    from .rl_effort_protocol import (balanced_effort_controller_contract,
                                     validate_balanced_effort_controller_contract)
    contract = balanced_effort_controller_contract()
    if parameters is not None:
        contract["parameters"] = copy.deepcopy(parameters)
    validated = validate_balanced_effort_controller_contract(contract)
    # A validator may return the validated declaration or validate in place.
    if validated is not None:
        contract = validated
    return copy.deepcopy(contract), copy.deepcopy(contract["parameters"])


def _contact_domain_guard_torque(t, declaration, *, q, a, point_jacobian, lower, upper,
                                physical_safe, safe, tracking, normal_available,
                                physical_loaded, overforce):
    """Bounded thumb-only posture torque in the current normal-and-Z nullspace.

    This geometric projection does not prove dynamic force decoupling or keep
    every manifold point fixed. Independent 2ms physical gates remain decisive.
    No contact Jacobian or guard effort is retained across samples or resets.
    """
    n = q.shape[0]
    missing = [name for name, value in (("contact_point_jacobian_world_m", point_jacobian),
        ("actual_domain_lower_hand_rad", lower), ("actual_domain_upper_hand_rad", upper),
        ("physical_contact_safe", physical_safe)) if value is None]
    supplied = not missing
    if supplied:
        for name, value, shape in (("contact point Jacobian", point_jacobian, (n, 3, 3, 7)),
                                  ("actual domain lower", lower, (n, 7)),
                                  ("actual domain upper", upper, (n, 7))):
            if (not t.is_tensor(value) or tuple(value.shape) != shape or not value.is_floating_point()
                    or value.dtype != q.dtype or value.device != q.device):
                raise ValueError(name+" requires matching native floating tensor dimensions")
        if (not t.is_tensor(physical_safe) or tuple(physical_safe.shape) != (n,)
                or physical_safe.dtype != t.bool or physical_safe.device != q.device):
            raise ValueError("physical_contact_safe requires matching native N boolean tensor")
        finite = (t.isfinite(point_jacobian).all(dim=(-1, -2, -3)) &
                  t.isfinite(lower).all(-1) & t.isfinite(upper).all(-1))
        bounds_valid = (lower <= upper).all(-1)
        margin = declaration["activation_margin_rad"]
        low = lower[:, 2]+margin
        high = upper[:, 2]-margin
        bounds_valid &= low <= high
        matrix = t.stack((a[:, 0, :3], point_jacobian[:, 0, 2, :3]), dim=1)
    else:
        finite = t.zeros(n, dtype=t.bool, device=q.device)
        bounds_valid = finite.clone()
        physical_safe = finite.clone()
        low, high = t.zeros_like(q[:, 2]), t.zeros_like(q[:, 2])
        matrix = t.zeros(n, 2, 3, device=q.device, dtype=q.dtype)
    matrix_finite = t.isfinite(matrix).all(dim=(-1, -2))
    finite &= matrix_finite
    matrix = t.where(finite[:, None, None], matrix, t.zeros_like(matrix))
    # full Vh is required: the reduced 2x3 SVD omits the one-dimensional null.
    _, singular, vh = t.linalg.svd(matrix, full_matrices=True)
    rank = (singular >= declaration["svd_tolerance_m"]).sum(-1)
    null = vh[:, 2, :]
    projector = null[:, :, None]*null[:, None, :]
    projected_e2 = projector[:, :, 2]
    component = projector[:, 2, 2]
    rank_ok = rank == 2
    component_ok = component >= declaration["minimum_nullspace_joint_component"]
    usable = finite & bounds_valid & rank_ok & component_ok
    gate = (usable & safe & tracking & normal_available.all(-1) &
            physical_loaded.all(-1) & physical_safe & ~overforce.any(-1))
    # A six-mrad interior target changes control only; original domain and its
    # original .025-rad tolerance are supplied independently, never enlarged.
    target = t.minimum(t.maximum(q[:, 2], low), high)
    error = t.where(finite & bounds_valid, target-q[:, 2], t.zeros_like(q[:, 2]))
    scalar = declaration["restoring_stiffness_nm_rad"]*error
    before_cap = t.zeros_like(q)
    before_cap[:, :3] = projected_e2*scalar[:, None]
    before_cap = t.where(gate[:, None], before_cap, t.zeros_like(before_cap))
    peak = before_cap.abs().amax(-1)
    common_scale = (declaration["maximum_guard_joint_torque_nm"]/peak.clamp_min(1e-12)).clamp(max=1.)
    torque = before_cap*common_scale[:, None]
    residual = t.einsum("nri,ni->nr", matrix, torque[:, :3])
    evidence = {"contract": copy.deepcopy(declaration), "inputs_supplied": supplied,
        "missing_inputs": missing, "current_input_finite": finite,
        "actual_domain_bounds_valid": bounds_valid, "safe_fresh_sample": safe,
        "palm_tracking": tracking, "all_group_current_normals": normal_available.all(-1),
        "all_groups_physically_loaded": physical_loaded.all(-1),
        "original_physical_contact_safe": physical_safe,
        "any_group_overforce": overforce.any(-1), "gate": gate,
        "current_normal_world_z_jacobian_m": matrix,
        "singular_values_m": singular, "retained_rank": rank, "required_rank": 2,
        "thumb_nullspace_projector": projector, "projected_thumb2_axis": projected_e2,
        "nullspace_thumb2_component": component, "rank_supported": rank_ok,
        "thumb2_component_supported": component_ok,
        "interior_lower_thumb2_rad": low, "interior_upper_thumb2_rad": high,
        "actual_domain_lower_margin_all7_rad": q-lower if supplied else None,
        "actual_domain_upper_margin_all7_rad": upper-q if supplied else None,
        "measured_thumb2_rad": q[:, 2], "boundary_error_rad": error,
        "restoring_scalar_nm": scalar, "requested_guard_before_own_cap_nm": before_cap,
        "guard_common_scale": common_scale, "guard_torque_before_total_alpha_nm": torque,
        "normal_projection_residual_m_nm": residual[:, 0],
        "world_z_projection_residual_m_nm": residual[:, 1],
        "original_actual_domain_not_enlarged": True,
        "all_force_mode_position_springs_remain_zero": True,
        "other_four_hand_joints_guard_torque_zero": True,
        "fresh_current_geometry_only_no_history": True,
        "geometric_projection_is_not_dynamic_force_isolation": True,
        "group_centroid_projection_does_not_fix_every_contact_manifold_point": True,
        "all_original_acceptance_gates_unchanged": True}
    return torque, evidence


def _transition_thumb_domain_recovery(t, declaration, projection_declaration, *, q, a,
                                      point_jacobian, lower, upper, safe, tracking,
                                      normal_available, physical_loaded, overforce,
                                      full_guard_gate):
    """Restore the original thumb domain during free/partial contact only.

    A free thumb receives a thumb2-only torque. A loaded partial-contact thumb
    instead uses the separately declared normal/Z nullspace construction. This
    is a control intervention, never a physical-contact acceptance certificate.
    """
    n = q.shape[0]
    supplied = lower is not None and upper is not None
    if supplied:
        for name, value in (("actual domain lower", lower), ("actual domain upper", upper)):
            if (not t.is_tensor(value) or tuple(value.shape) != (n, 7) or
                    not value.is_floating_point() or value.dtype != q.dtype or value.device != q.device):
                raise ValueError(name+" requires matching native N by seven floating tensor")
        finite = t.isfinite(lower).all(-1) & t.isfinite(upper).all(-1)
        margin = declaration["free_thumb2_activation_margin_rad"]
        low, high = lower[:, 2]+margin, upper[:, 2]-margin
        bounds_valid = finite & (lower <= upper).all(-1) & (low <= high)
    else:
        finite = t.zeros(n, dtype=t.bool, device=q.device)
        bounds_valid = finite.clone()
        low, high = t.zeros_like(q[:, 2]), t.zeros_like(q[:, 2])
    target = t.minimum(t.maximum(q[:, 2], low), high)
    error = t.where(bounds_valid, target-q[:, 2], t.zeros_like(q[:, 2]))
    scalar = declaration["free_thumb2_restoring_stiffness_nm_rad"]*error
    base_gate = bounds_valid & safe & tracking & ~overforce.any(-1) & ~full_guard_gate
    # A physically loaded thumb with an unavailable/degenerate current normal
    # is not free space. Do not inject an unprojected boundary torque there.
    free_gate = base_gate & ~normal_available[:, 0] & ~physical_loaded[:, 0]
    before_cap = t.zeros_like(q)
    before_cap[:, 2] = t.where(free_gate, scalar, t.zeros_like(scalar))

    # Reuse the original guard's explicitly declared rank/component tolerances;
    # without that declaration or a current point there is no projected guard.
    point_supplied = point_jacobian is not None
    if point_supplied and (not t.is_tensor(point_jacobian) or
            tuple(point_jacobian.shape) != (n, 3, 3, 7) or
            not point_jacobian.is_floating_point() or point_jacobian.dtype != q.dtype or
            point_jacobian.device != q.device):
        raise ValueError("contact point Jacobian requires matching native N by three by three by seven tensor")
    projection_supplied = point_supplied and projection_declaration is not None
    if projection_supplied:
        matrix = t.stack((a[:, 0, :3], point_jacobian[:, 0, 2, :3]), dim=1)
        matrix_finite = t.isfinite(matrix).all(dim=(-1, -2))
        matrix = t.where(matrix_finite[:, None, None], matrix, t.zeros_like(matrix))
    else:
        matrix = t.zeros(n, 2, 3, device=q.device, dtype=q.dtype)
        matrix_finite = t.zeros(n, dtype=t.bool, device=q.device)
    _, singular, vh = t.linalg.svd(matrix, full_matrices=True)
    tolerance = projection_declaration["svd_tolerance_m"] if projection_declaration else .0001
    minimum_component = projection_declaration["minimum_nullspace_joint_component"] if projection_declaration else .05
    rank = (singular >= tolerance).sum(-1)
    null = vh[:, 2, :]
    projector = null[:, :, None]*null[:, None, :]
    component = projector[:, 2, 2]
    partial_gate = (base_gate & matrix_finite & normal_available[:, 0] &
                    physical_loaded[:, 0] & ~physical_loaded.all(-1) &
                    (rank == 2) & (component >= minimum_component))
    partial = projector[:, :, 2]*scalar[:, None]
    before_cap[:, :3] += t.where(partial_gate[:, None], partial, t.zeros_like(partial))
    peak = before_cap.abs().amax(-1)
    scale = (declaration["free_thumb2_maximum_guard_torque_nm"]/peak.clamp_min(1e-12)).clamp(max=1.)
    torque = before_cap*scale[:, None]
    partial_torque = t.where(partial_gate[:, None], torque[:, :3], t.zeros_like(torque[:, :3]))
    residual = t.einsum("nri,ni->nr", matrix, partial_torque)
    evidence = {"original_actual_domain_inputs_supplied": supplied,
        "original_actual_domain_bounds_valid": bounds_valid,
        "original_actual_domain_lower_margin_all7_rad": q-lower if supplied else None,
        "original_actual_domain_upper_margin_all7_rad": upper-q if supplied else None,
        "safe_fresh_sample": safe, "palm_tracking": tracking,
        "any_group_overforce": overforce.any(-1), "original_full_guard_active": full_guard_gate,
        "thumb_current_normal": normal_available[:, 0],
        "thumb_physically_loaded": physical_loaded[:, 0],
        "all_groups_physically_loaded": physical_loaded.all(-1),
        "free_thumb2_gate": free_gate, "partial_contact_projection_gate": partial_gate,
        "current_point_jacobian_supplied": point_supplied,
        "original_projection_declaration_supplied": projection_declaration is not None,
        "current_normal_world_z_jacobian_m": matrix,
        "projection_current_input_finite": matrix_finite,
        "projection_singular_values_m": singular, "projection_rank": rank,
        "projection_thumb2_component": component, "projection_thumb_nullspace_projector": projector,
        "interior_lower_thumb2_rad": low, "interior_upper_thumb2_rad": high,
        "boundary_error_rad": error, "restoring_scalar_nm": scalar,
        "requested_guard_before_own_cap_nm": before_cap, "guard_common_scale": scale,
        "guard_torque_before_total_alpha_nm": torque,
        "partial_normal_projection_residual_m_nm": residual[:, 0],
        "partial_world_z_projection_residual_m_nm": residual[:, 1],
        "free_thumb2_has_no_nullspace_claim": True,
        "partial_contact_guard_is_not_physical_acceptance": True,
        "partial_projection_is_not_dynamic_force_isolation": True,
        "all_current_force_mode_position_springs_remain_zero": True,
        "fresh_current_geometry_only_no_history": True,
        "original_actual_domain_not_enlarged": True}
    return torque, evidence


def _separating_preload_command_floor(t, declaration, *, command, force, force_limit,
                                      safe, tracking, normal_available, physical_loaded,
                                      overforce, ramp_fraction, relative_velocity,
                                      maximum_point_relative_velocity, velocity_current):
    """Cancel active retreat only when every current point is separating.

    This does not clamp measured force or inject extra positive preload. Missing,
    stale, inconsistent or nonfinite velocity proof leaves the original command
    unchanged. The existing hand7 damping law is deliberately unchanged.
    """
    n = command.shape[0]
    for name, value in (("relative_inward_velocity_m_s", relative_velocity),
                        ("maximum_current_point_relative_inward_velocity_m_s", maximum_point_relative_velocity)):
        if value is not None and (not t.is_tensor(value) or tuple(value.shape) != (n, 3) or
                not value.is_floating_point() or value.dtype != command.dtype or value.device != command.device):
            raise ValueError(name+" requires matching native N by three floating tensor")
    if velocity_current is not None and (not t.is_tensor(velocity_current) or
            tuple(velocity_current.shape) != (n, 3) or velocity_current.dtype != t.bool or
            velocity_current.device != command.device):
        raise ValueError("relative_velocity_current requires matching native N by three boolean tensor")
    supplied = all(x is not None for x in (relative_velocity, maximum_point_relative_velocity, velocity_current))
    if supplied:
        finite = t.isfinite(relative_velocity) & t.isfinite(maximum_point_relative_velocity)
        consistent = relative_velocity <= maximum_point_relative_velocity
        current = velocity_current & finite & consistent
        separating = current & (maximum_point_relative_velocity <=
                                 declaration["all_current_points_separating_threshold_m_s"])
    else:
        finite = t.zeros(n, 3, dtype=t.bool, device=command.device)
        consistent = finite.clone()
        current = finite.clone()
        separating = finite.clone()
    gate = ((safe & tracking & (ramp_fraction < 1) & ~overforce.any(-1))[:, None] &
            normal_available & physical_loaded & (force < force_limit) & separating & (command < 0))
    after = t.where(gate, t.full_like(command, declaration["separating_preload_command_floor_n"]), command)
    evidence = {"velocity_inputs_supplied": supplied,
        "relative_inward_velocity_m_s": relative_velocity,
        "maximum_current_point_relative_inward_velocity_m_s": maximum_point_relative_velocity,
        "relative_velocity_declared_current": velocity_current,
        "relative_velocity_finite_by_group": finite,
        "weighted_mean_not_above_point_maximum": consistent,
        "relative_velocity_current_proof": current,
        "all_current_points_proven_separating": separating,
        "safe_fresh_sample": safe, "palm_tracking": tracking,
        "preload_or_ramp_not_complete": ramp_fraction < 1,
        "current_normal_available_by_group": normal_available,
        "group_physically_loaded": physical_loaded,
        "raw_group_force_below_original_limit": force < force_limit,
        "any_group_overforce": overforce.any(-1),
        "force_command_before_floor_n": command,
        "force_command_after_floor_n": after,
        "separating_preload_floor_applied_by_group": gate,
        "removed_opening_force_request_n": after-command,
        "relative_velocity_used_only_for_floor_gate_not_hand7_damping": True,
        "force_targets_not_increased_and_measured_force_not_clipped": True,
        "original_overforce_negative_unloading_priority_unchanged": True,
        "missing_stale_nonfinite_or_inconsistent_proof_uses_original_law": True,
        "fresh_current_point_proof_required_no_history": True,
        "zero_force_request_is_not_zero_complete_joint_effort": True,
        "requested_force_floor_is_not_a_measured_force_bound": True}
    return after, evidence


class BalancedEffortGraspController:
    """Joint efforts for the declared hand-only grasp intervention.

    The native position/velocity drives on these seven joints MUST be disabled
    by the environment. This class cannot inspect the native drive properties.
    It bounds the complete requested joint torque, including gravity, rather
    than claiming a bound on measured contact force. All contacts, acceptance
    gates and other 36 native drives remain outside this controller.

    Each current normal-Jacobian row maps hand velocity to inward velocity at
    the actual native contact point. A speculative point may supply a current
    row for feedback; it never establishes physically loaded contact. No row
    from an earlier sample is stored or reused. The declared .8/.4/.4 N target
    distribution is an internal-force reference, not a measured wrench balance.
    """

    def __init__(self, torch, initial_hand_rad, episode_id, *, parameters=None, domain_posture_guard=None,
                 contact_transition=None):
        self.contract, self.parameters = _balanced_effort_contract_and_parameters(parameters)
        self.domain_posture_guard = None
        if domain_posture_guard is not None:
            from .rl_effort_protocol import validate_contact_domain_posture_guard
            self.domain_posture_guard = validate_contact_domain_posture_guard(domain_posture_guard)
            self.contract["version"] += "+"+self.domain_posture_guard["version"]
            self.contract["domain_posture_guard"] = copy.deepcopy(self.domain_posture_guard)
        self.contact_transition = None
        if contact_transition is not None:
            from .rl_effort_protocol import validate_contact_transition_control
            self.contact_transition = validate_contact_transition_control(contact_transition)
            self.contract["version"] += "+"+self.contact_transition["version"]
            self.contract["contact_transition"] = copy.deepcopy(self.contact_transition)
        self.torch = torch
        if (not torch.is_tensor(initial_hand_rad) or not initial_hand_rad.is_floating_point() or initial_hand_rad.ndim != 2 or
                initial_hand_rad.shape[1] != 7 or not bool(torch.isfinite(initial_hand_rad).all())):
            raise ValueError("Balanced effort requires finite measured N by seven reset joints")
        self.seed = initial_hand_rad.detach().clone()
        self.num_envs = self.seed.shape[0]
        if (not torch.is_tensor(episode_id) or tuple(episode_id.shape) != (self.num_envs,) or
                episode_id.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Balanced effort requires integer episode identities")
        self.ids = episode_id.clone()
        self.reference = self.seed.clone()
        self.contact_latched = torch.zeros(self.num_envs, 3, dtype=torch.bool, device=self.seed.device)
        self.previous_normal_available = torch.zeros_like(self.contact_latched)
        if self.contact_transition is not None:
            self.previous_any_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.seed.device)
            self.preload_settle_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.seed.device)
            self.preload_ramp_sample_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.seed.device)
            self.preload_ramp_elapsed_s = torch.zeros(self.num_envs, dtype=self.seed.dtype, device=self.seed.device)
        self.seen_valid_feedback = torch.zeros(self.num_envs, dtype=torch.bool, device=self.seed.device)
        self.last_effort = torch.zeros_like(self.seed)
        self.sample_id = None
        self.group_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2], device=self.seed.device)
        payload = json.dumps(self.contract, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()
        self.contract_sha256 = hashlib.sha256(payload).hexdigest()

    def update(self, *, sample_id, episode_id, dt, measured_hand_rad,
               measured_hand_velocity_rad_s, nominal_hand_rad, gravity_hand_nm,
               inward_normal_jacobian, group_contact_valid, group_physical_loaded,
               raw_group_force_n, contact_measurement_valid, tracking, unsafe_latched,
               joint_lower_hand_rad, joint_upper_hand_rad, geometry_lower_hand_rad,
               geometry_upper_hand_rad, nominal_velocity_limit_hand_rad_s,
               penetration_m, closing_axis_hand7, object_support_torque_nm=None,
               contact_point_jacobian_world_m=None, actual_domain_lower_hand_rad=None,
               actual_domain_upper_hand_rad=None, physical_contact_safe=None,
               relative_inward_velocity_m_s=None,
               maximum_current_point_relative_inward_velocity_m_s=None,
               relative_velocity_current=None, freeze_reference_by_environment=None):
        """Use completed native sample k to request efforts for k+1 exactly once."""
        t, p, n = self.torch, self.parameters, self.num_envs
        if (type(sample_id) is not int or sample_id < 0 or type(dt) not in (int, float) or
                not math.isfinite(dt) or dt != p["physics_dt_s"]):
            raise ValueError("Balanced effort requires its declared native timestep and integer sample")
        hands = (measured_hand_rad, measured_hand_velocity_rad_s, nominal_hand_rad,
                 gravity_hand_nm, joint_lower_hand_rad, joint_upper_hand_rad,
                 geometry_lower_hand_rad, geometry_upper_hand_rad,
                 nominal_velocity_limit_hand_rad_s)
        if any(not t.is_tensor(v) or tuple(v.shape) != (n, 7) for v in hands):
            raise ValueError("Balanced effort needs native N by seven state/reference/bound tensors")
        if object_support_torque_nm is None:
            support_input = t.zeros_like(self.seed)
        else:
            if (not t.is_tensor(object_support_torque_nm) or
                    tuple(object_support_torque_nm.shape) != (n, 7) or
                    not object_support_torque_nm.is_floating_point() or
                    object_support_torque_nm.dtype != self.seed.dtype or
                    object_support_torque_nm.device != self.seed.device):
                raise ValueError("Object support torque requires native N by seven matching floating tensors")
            support_input = object_support_torque_nm
        if (not t.is_tensor(episode_id) or tuple(episode_id.shape) != (n,) or
                episode_id.dtype not in (t.int32, t.int64) or
                tuple(inward_normal_jacobian.shape) != (n, 3, 7) or
                tuple(raw_group_force_n.shape) != (n, 3) or tuple(penetration_m.shape) != (n,) or
                tuple(closing_axis_hand7.shape) != (7,) or
                any(tuple(v.shape) != (n, 3) or v.dtype != t.bool
                    for v in (group_contact_valid, group_physical_loaded)) or
                any(tuple(v.shape) != (n,) or v.dtype != t.bool
                    for v in (contact_measurement_valid, tracking, unsafe_latched))):
            raise ValueError("Balanced effort feedback dimensions or boolean types are invalid")
        if freeze_reference_by_environment is not None:
            if (not t.is_tensor(freeze_reference_by_environment) or
                    tuple(freeze_reference_by_environment.shape) != (n,) or
                    freeze_reference_by_environment.dtype != t.bool or
                    freeze_reference_by_environment.device != self.seed.device):
                raise ValueError("Reference freeze requires matching native N boolean environment mask")
        axis = closing_axis_hand7.to(device=self.seed.device, dtype=self.seed.dtype)
        if (closing_axis_hand7.dtype == t.bool or not bool(t.isfinite(axis).all()) or
                not bool(((axis == 0) | (axis.abs() == 1)).all()) or
                not bool(axis[:3].ne(0).any() & axis[3:5].ne(0).any() & axis[5:].ne(0).any())):
            raise ValueError("Balanced effort requires finite candidate closing axes in {-1,0,1} for every group")
        # A contact on one finger may not silently control another finger. The
        # fixed-root hand7 Jacobian has this exact kinematic sparsity.
        structural = self.group_ids[None, :] == t.arange(3, device=self.seed.device)[:, None]
        finite = t.ones(n, dtype=t.bool, device=self.seed.device)
        for value in hands:
            finite &= t.isfinite(value).all(-1)
        finite &= t.isfinite(support_input).all(-1)
        finite &= t.isfinite(inward_normal_jacobian).all(dim=(-1, -2))
        finite &= t.isfinite(raw_group_force_n).all(-1) & (raw_group_force_n >= 0).all(-1)
        finite &= t.isfinite(penetration_m) & (penetration_m >= 0)
        cross_group_jacobian = ((inward_normal_jacobian * (~structural)[None]).abs() > 1e-8).any(dim=(-1, -2))
        changed = episode_id != self.ids
        initial = self.sample_id is None
        duplicate = not initial and sample_id == self.sample_id
        discontinuity = not initial and not duplicate and sample_id != self.sample_id+1
        duplicate_epoch_conflict = changed & duplicate
        lower = t.maximum(joint_lower_hand_rad, geometry_lower_hand_rad)
        upper = t.minimum(joint_upper_hand_rad, geometry_upper_hand_rad)
        invalid_bounds = (lower > upper).any(-1) | (nominal_velocity_limit_hand_rad_s <= 0).any(-1)
        reset_mismatch = changed & ((measured_hand_rad-self.seed).abs().amax(-1) > 1e-4)
        lost_measurement = ~contact_measurement_valid & self.seen_valid_feedback & ~changed & (not duplicate)
        unsafe = (unsafe_latched | ~finite | invalid_bounds | cross_group_jacobian | reset_mismatch |
                  (penetration_m > p["maximum_penetration_m"]) | lost_measurement | discontinuity |
                  duplicate_epoch_conflict)
        fresh = t.full((n,), not (initial or duplicate or discontinuity), dtype=t.bool,
                       device=self.seed.device) & ~changed
        safe = fresh & contact_measurement_valid & ~unsafe
        if not duplicate:
            self.ids = episode_id.clone()
            self.sample_id = sample_id
            self.reference[changed] = self.seed[changed]
            self.contact_latched[changed] = False
            self.previous_normal_available[changed] = False
            self.seen_valid_feedback[changed] = False
            if self.contact_transition is not None:
                self.previous_any_contact[changed] = False
                self.preload_settle_count[changed] = 0
                self.preload_ramp_sample_count[changed] = 0
                self.preload_ramp_elapsed_s[changed] = 0
        self.seen_valid_feedback |= contact_measurement_valid & finite & ~unsafe & ~changed & (not duplicate)
        # Replace invalid data before linear algebra. Its environment is still
        # marked unsafe and receives zero effort; NaNs never enter SVD.
        q = t.where(finite[:, None], measured_hand_rad, self.seed)
        qd = t.where(finite[:, None], measured_hand_velocity_rad_s, t.zeros_like(self.seed))
        g = t.where(finite[:, None], gravity_hand_nm, t.zeros_like(self.seed))
        support_requested = t.where(finite[:, None], support_input, t.zeros_like(self.seed))
        force = t.where((finite & contact_measurement_valid)[:, None], raw_group_force_n,
                        t.zeros_like(raw_group_force_n))
        a_input = t.where(finite[:, None, None], inward_normal_jacobian,
                          t.zeros_like(inward_normal_jacobian))
        row_norm = t.linalg.vector_norm(a_input, dim=-1)
        normal_available = (group_contact_valid & safe[:, None] &
                            (row_norm >= p["normal_jacobian_rank_tolerance_m"]))
        a = t.where(normal_available[:, :, None], a_input, t.zeros_like(a_input))
        latched_before = self.contact_latched.clone()
        # Having used a current (possibly speculative, even zero-force) point
        # for force control is enough to latch its CONTROL mode. Otherwise the
        # next absent point would jump back to the fast free-space trajectory.
        self.contact_latched |= safe[:, None] & ((force > p["contact_feedback_onset_n"]) | normal_available)
        reacquire = self.contact_latched & ~normal_available & safe[:, None]
        lost_current_normal = self.previous_normal_available & ~normal_available & safe[:, None]
        force_axis = axis != 0
        # Drop old free-space position error when a current normal disappears.
        # Subsequent reacquisition starts from measured q with a declared rate.
        reset_reference = (lost_current_normal | (reacquire & ~latched_before))[:, self.group_ids]
        transition_any_contact = None
        transition_first_contact = None
        if self.contact_transition is not None:
            transition_any_contact = safe & (normal_available.any(-1) | group_physical_loaded.any(-1))
            transition_first_contact = transition_any_contact & ~self.previous_any_contact
            reset_reference |= (transition_first_contact[:, None] &
                                ~normal_available[:, self.group_ids])
        reference = t.where(reset_reference, q, self.reference)
        velocity_limit = nominal_velocity_limit_hand_rad_s.clone()
        velocity_limit = t.where(reacquire[:, self.group_ids],
                                 t.minimum(velocity_limit, t.full_like(velocity_limit,
                                           p["reacquisition_velocity_limit_rad_s"])), velocity_limit)
        if self.contact_transition is not None:
            partial_missing = transition_any_contact[:, None] & ~normal_available[:, self.group_ids]
            velocity_limit = t.where(partial_missing,
                t.minimum(velocity_limit, t.full_like(velocity_limit,
                    self.contact_transition["missing_group_reference_velocity_limit_rad_s"])), velocity_limit)
        nominal = t.minimum(t.maximum(nominal_hand_rad, lower), upper)
        if freeze_reference_by_environment is not None:
            # A task failure can stop positional chasing without disabling
            # current normal force feedback or emergency opening. Only fresh,
            # otherwise-safe samples may replace the reference; duplicate,
            # reset and unsafe handling below retain their original priority.
            freeze_reference = safe & freeze_reference_by_environment
            reference = t.where(freeze_reference[:, None], q, reference)
        if self.contact_transition is not None:
            reference_integration_origin = reference.clone()
        increment = (nominal-reference).clamp(min=-velocity_limit*dt, max=velocity_limit*dt)
        if freeze_reference_by_environment is not None:
            increment = t.where(freeze_reference[:, None], t.zeros_like(increment), increment)
        reference = reference + t.where((safe & tracking)[:, None], increment, t.zeros_like(increment))
        if self.contact_transition is None:
            reference = t.minimum(t.maximum(reference, lower), upper)
            if freeze_reference_by_environment is not None:
                reference = t.where(freeze_reference[:, None], q, reference)
        # A measured-q reset can start just outside the reference box while
        # remaining within the separate actual-domain tolerance. Projecting it
        # instantly back into the box defeats the native-sample slew bound.
        # The unchanged in-box nominal instead draws that reference inward at
        # the declared velocity. The actual/physical gates stay independent.
        # Palm lag suppresses all positional chasing. Measured force can still
        # request opening, and current gravity remains within the total cap.
        reference_for_impedance = t.where(tracking[:, None], reference, q)
        spring_before_mask = p["joint_stiffness_nm_rad"]*(reference_for_impedance-q)
        position_spring = t.where(normal_available[:, self.group_ids], t.zeros_like(q), spring_before_mask)
        damping = -p["joint_damping_nm_s_rad"]*qd
        impedance = position_spring+damping
        # In current force mode the whole finger loses its free-space position
        # spring, including tangential directions. Project damping through the
        # exact truncated-SVD normal nullspace; do not retain a tangential pose
        # spring that can push the Cube down while the normal force is correct.
        _, singular, vh = t.linalg.svd(a, full_matrices=False)
        rank = singular >= p["normal_jacobian_rank_tolerance_m"]
        normal_projection = t.einsum("nri,nr,nrj->nij", vh, rank.to(q.dtype), vh)
        identity = t.eye(7, device=q.device, dtype=q.dtype)[None]
        posture = t.einsum("nij,nj->ni", identity-normal_projection, impedance)
        targets = t.tensor(p["group_force_targets_n"], device=q.device, dtype=q.dtype)[None].expand(n, -1)
        transition_evidence = None
        if self.contact_transition is not None:
            if physical_contact_safe is None:
                transition_physical_safe = t.zeros(n, dtype=t.bool, device=q.device)
            else:
                if (not t.is_tensor(physical_contact_safe) or tuple(physical_contact_safe.shape) != (n,) or
                        physical_contact_safe.dtype != t.bool or physical_contact_safe.device != q.device):
                    raise ValueError("physical_contact_safe requires matching native N boolean tensor")
                transition_physical_safe = physical_contact_safe
            settling = (safe & tracking & normal_available.all(-1) & group_physical_loaded.all(-1) &
                        transition_physical_safe & ~(force > p["group_force_limit_n"]).any(-1))
            count_before = self.preload_settle_count.clone()
            ramp_before = self.preload_ramp_elapsed_s.clone()
            ramp_sample_before = self.preload_ramp_sample_count.clone()
            if not duplicate:
                self.preload_settle_count = t.where(settling,
                    (count_before+1).clamp(max=self.contact_transition["full_contact_settle_samples"]),
                    t.zeros_like(count_before))
                ready_before = count_before >= self.contact_transition["full_contact_settle_samples"]
                # Count native samples rather than accumulating float seconds:
                # the declared 100ms ramp ends exactly after 50 new 2ms samples.
                ramp_samples = round(self.contact_transition["full_force_ramp_s"]/dt)
                self.preload_ramp_sample_count = t.where(settling & ready_before,
                    (ramp_sample_before+1).clamp(max=ramp_samples), t.zeros_like(ramp_sample_before))
                self.preload_ramp_elapsed_s = self.preload_ramp_sample_count.to(q.dtype)*dt
            ramp_fraction = (self.preload_ramp_elapsed_s/self.contact_transition["full_force_ramp_s"]).clamp(0, 1)
            preload = t.tensor(self.contact_transition["preload_group_force_targets_n"],
                               device=q.device, dtype=q.dtype)[None].expand(n, -1)
            full_targets = targets
            targets = preload+ramp_fraction[:, None]*(full_targets-preload)
            transition_evidence = {"contract": copy.deepcopy(self.contact_transition),
                "any_current_normal_or_physical_contact": transition_any_contact,
                "first_any_contact_resets_missing_reference": transition_first_contact,
                "partial_contact_missing_normal_reference_cap_joint": partial_missing,
                "missing_group_reference_cap_is_inner_native_sample_bound": True,
                "reference_reset_to_current_measured_joint": reset_reference,
                "reference_integration_origin_hand_rad": reference_integration_origin,
                "reference_integration_delta_hand_rad": reference-reference_integration_origin,
                "temporary_reference_box_lower_excursion_rad": (lower-reference).clamp_min(0),
                "temporary_reference_box_upper_excursion_rad": (reference-upper).clamp_min(0),
                "reference_box_excursion_did_not_increase_joint":
                    (((lower-reference).clamp_min(0) <= (lower-reference_integration_origin).clamp_min(0)) &
                     ((reference-upper).clamp_min(0) <= (reference_integration_origin-upper).clamp_min(0))),
                "reference_box_reentry_is_rate_limited_not_single_sample_projection": True,
                "nominal_reference_box_and_original_actual_domain_not_enlarged": True,
                "all_group_current_normals": normal_available.all(-1),
                "all_groups_physically_loaded": group_physical_loaded.all(-1),
                "original_physical_contact_safe": transition_physical_safe,
                "full_contact_settling_gate": settling,
                "full_contact_settle_count_before": count_before,
                "full_contact_settle_count": self.preload_settle_count.clone(),
                "full_force_ramp_elapsed_s_before": ramp_before,
                "full_force_ramp_elapsed_s": self.preload_ramp_elapsed_s.clone(),
                "full_force_ramp_sample_count_before": ramp_sample_before,
                "full_force_ramp_sample_count": self.preload_ramp_sample_count.clone(),
                "full_force_ramp_fraction": ramp_fraction,
                "preload_group_force_targets_n": preload,
                "full_group_force_targets_n": full_targets,
                "applied_group_force_targets_n": targets,
                "contact_loss_or_unsafe_clears_settle_and_ramp": True,
                "force_targets_are_not_measured_force_bounds": True}
        error = targets-force
        inward_velocity = t.einsum("ngi,ni->ng", a, qd)
        command = (targets+p["normal_force_feedback_gain"]*error-
                   p["normal_velocity_damping_n_s_m"]*inward_velocity).clamp(
                       p["group_force_command_lower_n"], p["group_force_command_upper_n"])
        overforce = safe[:, None] & (force > p["group_force_limit_n"])
        command = t.where(overforce, t.full_like(command, p["group_force_command_lower_n"]), command)
        command = t.where(tracking[:, None] | overforce, command.clamp(max=p["group_force_command_upper_n"]),
                          command.clamp(max=0))
        command = t.where(normal_available, command, t.zeros_like(command))
        separating_floor_evidence = None
        if (self.contact_transition is not None and
                "separating_preload_command_floor_n" in self.contact_transition):
            command, separating_floor_evidence = _separating_preload_command_floor(
                t, self.contact_transition, command=command, force=raw_group_force_n,
                force_limit=p["group_force_limit_n"], safe=safe, tracking=tracking,
                normal_available=normal_available, physical_loaded=group_physical_loaded,
                overforce=overforce, ramp_fraction=ramp_fraction,
                relative_velocity=relative_inward_velocity_m_s,
                maximum_point_relative_velocity=maximum_current_point_relative_inward_velocity_m_s,
                velocity_current=relative_velocity_current)
            transition_evidence["separating_preload_floor"] = separating_floor_evidence
        force_torque = t.einsum("ngi,ng->ni", a, command)
        caps = t.tensor(p["joint_torque_limits_nm"], device=q.device, dtype=q.dtype)[None].expand(n, -1)
        gravity_clipped = g.clamp(min=-caps, max=caps)
        overforce_joint = overforce[:, self.group_ids]
        # Emergency unloading outranks gravity/posture on the affected finger.
        # Known closing axes allow opening even if no current normal exists.
        gravity_command = t.where(overforce_joint, t.zeros_like(g), gravity_clipped)
        posture = t.where(overforce_joint, t.zeros_like(posture), posture)
        guard_evidence = None
        if self.domain_posture_guard is not None:
            guard_torque, guard_evidence = _contact_domain_guard_torque(t, self.domain_posture_guard,
                q=q, a=a, point_jacobian=contact_point_jacobian_world_m,
                lower=actual_domain_lower_hand_rad, upper=actual_domain_upper_hand_rad,
                physical_safe=physical_contact_safe, safe=safe, tracking=tracking,
                normal_available=normal_available, physical_loaded=group_physical_loaded,
                overforce=overforce)
            posture = posture+guard_torque
        transition_guard_torque = None
        if self.contact_transition is not None:
            full_guard_gate = guard_evidence["gate"] if guard_evidence is not None else t.zeros(n, dtype=t.bool, device=q.device)
            transition_guard_torque, transition_guard_evidence = _transition_thumb_domain_recovery(
                t, self.contact_transition, self.domain_posture_guard, q=q, a=a,
                point_jacobian=contact_point_jacobian_world_m,
                lower=actual_domain_lower_hand_rad, upper=actual_domain_upper_hand_rad,
                safe=safe, tracking=tracking, normal_available=normal_available,
                physical_loaded=group_physical_loaded, overforce=overforce,
                full_guard_gate=full_guard_gate)
            posture = posture+transition_guard_torque
            transition_evidence["thumb_domain_recovery"] = transition_guard_evidence
        opening_fallback = t.where(overforce_joint & ~normal_available[:, self.group_ids] & force_axis[None],
                                   -axis[None]*p["overforce_opening_torque_nm"], t.zeros_like(g))
        non_gravity = posture+force_torque+opening_fallback
        # The environment supplies current-point Jz.T times its declared share
        # of object weight. It must qualify/ramp that request independently.
        # Emergency unloading removes the whole affected finger's support,
        # and no support request bypasses the existing common alpha or caps.
        support_after_mask = t.where((safe[:, None] & ~overforce_joint),
                                     support_requested, t.zeros_like(support_requested))
        # Select the original value at exactly zero input, including signed
        # zero, so both the omitted and explicit-zero API retain its bits.
        non_gravity = t.where(support_after_mask != 0,
                              non_gravity+support_after_mask, non_gravity)
        denominator = t.where(non_gravity != 0, non_gravity, t.ones_like(non_gravity))
        room = t.where(non_gravity > 0, (caps-gravity_command)/denominator,
                       t.where(non_gravity < 0, (-caps-gravity_command)/denominator,
                               t.ones_like(non_gravity)))
        alpha = room.amin(-1).clamp(0, 1)
        unclipped_total = g+impedance+force_torque+opening_fallback
        unclipped_total = t.where(support_after_mask != 0,
                                 unclipped_total+support_after_mask, unclipped_total)
        if guard_evidence is not None:
            # Keep the historical None path bit-identical. This diagnostic is
            # before the total cap but includes the optional projected guard.
            unclipped_total = unclipped_total+guard_torque
        if transition_guard_torque is not None:
            unclipped_total = unclipped_total+transition_guard_torque
        requested_total = gravity_command+non_gravity
        effort_before_emergency_axis_guard = gravity_command+alpha[:, None]*non_gravity
        # A current normal may have an unexpected per-joint sign. An overload
        # never keeps a positive known-closing-axis torque on a squeeze joint.
        remove_closing = (overforce_joint & force_axis[None] &
                          (effort_before_emergency_axis_guard*axis[None] > 0))
        if self.contact_transition is not None:
            # The current inward row, not a nominal closing-axis sign, defines
            # unloading. Overforce already removed every other affected-finger
            # contribution: tau=alpha*J.T*f_negative, hence each J_j*tau_j<=0.
            # Keep the original fixed-axis mask only when no current row exists.
            axis_mask_bypassed = remove_closing & normal_available[:, self.group_ids]
            remove_closing = remove_closing & ~normal_available[:, self.group_ids]
        effort = t.where(remove_closing, t.zeros_like(g), effort_before_emergency_axis_guard)
        effort = effort.clamp(min=-caps, max=caps)
        # Before the first valid feedback witness, apply only bounded gravity
        # and damping, without moving the nominal trajectory or faking contact.
        waiting = ~self.seen_valid_feedback & ~unsafe
        waiting_effort = (gravity_clipped-p["joint_damping_nm_s_rad"]*qd).clamp(min=-caps, max=caps)
        effort = t.where((~safe & ~unsafe)[:, None], waiting_effort, effort)
        if duplicate:
            effort = self.last_effort.clone()
        effort = t.where(unsafe[:, None], t.zeros_like(effort), effort)
        if not duplicate:
            self.reference = t.where(safe[:, None], reference, self.reference)
            self.previous_normal_available = t.where(safe[:, None], normal_available,
                                                      self.previous_normal_available)
            self.last_effort = effort.clone()
            if self.contact_transition is not None:
                self.previous_any_contact = transition_any_contact.clone()
        projected_residual = t.einsum("ngi,ni->ng", a, posture)
        opening_joint = (effort*axis[None] < 0) & force_axis[None]
        relief = t.stack((opening_joint[:, :3].any(-1), opening_joint[:, 3:5].any(-1),
                          opening_joint[:, 5:].any(-1)), dim=-1) & overforce
        if self.contact_transition is not None:
            current_pairing = a*effort[:, None, :]
            current_relief = (current_pairing < 0).any(-1) & overforce & normal_available
            relief = t.where(normal_available, current_relief, relief)
            transition_evidence.update({
                "current_normal_unload_by_group": overforce & normal_available,
                "missing_current_normal_axis_fallback_by_group": overforce & ~normal_available,
                "fixed_closing_axis_mask_bypassed_by_current_normal_joint": axis_mask_bypassed,
                "effort_before_emergency_direction_guard_nm": effort_before_emergency_axis_guard,
                "current_normal_actual_joint_pairing_m_nm": current_pairing,
                "current_normal_actual_group_pairing_m_nm": current_pairing.sum(-1),
                "current_normal_command_is_not_a_measured_force_or_velocity_guarantee": True,
                "normal_unload_uses_current_sample_only_no_history": True,
                "all_increments_share_original_non_gravity_alpha_and_total_caps": True,
                "duplicate_holds_prior_complete_effort_without_new_transition": duplicate})
            transition_guard_evidence["guard_torque_after_total_common_scale_nm"] = alpha[:, None]*transition_guard_torque
            applied_transition_guard = alpha[:, None]*transition_guard_torque
            applied_transition_guard = t.where(remove_closing, t.zeros_like(applied_transition_guard), applied_transition_guard)
            applied_transition_guard = t.where(safe[:, None] & ~unsafe[:, None], applied_transition_guard, t.zeros_like(applied_transition_guard))
            if duplicate:
                applied_transition_guard = t.zeros_like(applied_transition_guard)
            transition_guard_evidence["guard_component_for_current_new_command_nm"] = applied_transition_guard
        evidence = {
            "controller_version": self.contract["version"], "controller_contract_sha256": self.contract_sha256,
            "controller_contract": self.contract, "native_sample_id": sample_id,
            "next_native_sample_id": sample_id+1, "sample_duplicate_hold": duplicate,
            "sample_discontinuity": discontinuity, "episode_reset_hold": changed,
            "duplicate_sample_episode_conflict": duplicate_epoch_conflict,
            "measurement_fresh_for_effort": safe, "waiting_for_first_valid_measurement": waiting,
            "measurement_lost_after_valid": lost_measurement,
            "current_normal_available_by_group": normal_available,
            "current_normal_declared_available_by_group": group_contact_valid,
            "current_normal_degenerate_by_group": group_contact_valid & (row_norm < p["normal_jacobian_rank_tolerance_m"]),
            "physical_loaded_by_group_passthrough": group_physical_loaded,
            "control_normal_does_not_qualify_physical_support": True,
            "raw_group_normal_force_n": raw_group_force_n, "group_force_target_n": targets,
            "group_force_error_n": error, "group_force_command_n": command,
            "raw_force_over_limit_by_group": raw_group_force_n > p["group_force_limit_n"],
            "force_relief_by_group": relief, "contact_mode_latched_by_group": self.contact_latched,
            "reacquisition_active_by_group": reacquire, "lost_current_normal_by_group": lost_current_normal,
            "inward_normal_jacobian_m": a, "inward_velocity_m_s": inward_velocity,
            "normal_jacobian_singular_values_m": singular, "normal_jacobian_retained_rank": rank.sum(-1),
            "reference_hand_rad": reference, "bounded_nominal_hand_rad": nominal,
            "reference_velocity_limit_rad_s": velocity_limit,
            "actual_geometry_lower_violation_rad": (geometry_lower_hand_rad-measured_hand_rad).clamp_min(0),
            "actual_geometry_upper_violation_rad": (measured_hand_rad-geometry_upper_hand_rad).clamp_min(0),
            "actual_native_joint_lower_violation_rad": (joint_lower_hand_rad-measured_hand_rad).clamp_min(0),
            "actual_native_joint_upper_violation_rad": (measured_hand_rad-joint_upper_hand_rad).clamp_min(0),
            "geometry_and_velocity_boxes_constrain_reference_not_actual_motion": True,
            "native_mechanical_tolerance_and_height_gates_remain_external": True,
            "joint_impedance_before_normal_projection_nm": impedance,
            "position_spring_before_contact_mode_mask_nm": spring_before_mask,
            "position_spring_after_contact_mode_mask_nm": position_spring,
            "position_spring_disabled_by_current_normal_joint": normal_available[:, self.group_ids],
            "joint_velocity_damping_before_normal_projection_nm": damping,
            "posture_torque_after_normal_projection_nm": posture,
            "normal_posture_residual_m_nm": projected_residual,
            "normal_force_torque_nm": force_torque, "gravity_input_nm": gravity_hand_nm,
            "gravity_command_nm": gravity_command, "gravity_saturated_by_joint": g.abs() > caps,
            "gravity_removed_for_overforce_by_joint": overforce_joint,
            "opening_fallback_torque_nm": opening_fallback,
            "object_support_torque_input_nm": support_input,
            "object_support_torque_requested_nm": support_requested,
            "object_support_torque_after_safety_mask_nm": support_after_mask,
            "object_support_torque_after_common_scale_nm": alpha[:, None]*support_after_mask,
            "object_support_removed_for_overforce_by_joint": overforce_joint,
            "object_support_scaled_component_precedes_final_hold_or_unsafe_overrides": True,
            "object_support_shares_non_gravity_alpha_and_total_caps": True,
            "unprojected_total_torque_nm": unclipped_total, "requested_total_torque_nm": requested_total,
            "non_gravity_common_scale": alpha, "actual_bounded_effort_target_nm": effort,
            "total_torque_caps_nm": caps, "overforce_closing_axis_torque_removed": remove_closing,
            "unsafe_requires_zero_effort": unsafe, "invalid_reference_bounds": invalid_bounds,
            "cross_group_jacobian_rejected": cross_group_jacobian,
            "requires_zero_native_position_velocity_drives": True,
            "other_36_native_drives_not_changed_by_this_controller": True,
            "requested_torque_bounds_are_not_measured_force_bounds": True,
            "force_target_distribution_is_not_verified_wrench_balance": True,
        }
        if guard_evidence is not None:
            guard_evidence["guard_torque_after_total_common_scale_nm"] = alpha[:, None]*guard_torque
            guard_evidence["unprojected_total_diagnostic_includes_projected_guard"] = True
            # This decomposition follows exactly the same final hold, emergency,
            # duplicate and unsafe selections as the complete effort command.
            applied_guard = alpha[:, None]*guard_torque
            applied_guard = t.where(remove_closing, t.zeros_like(applied_guard), applied_guard)
            applied_guard = t.where(safe[:, None] & ~unsafe[:, None], applied_guard, t.zeros_like(applied_guard))
            if duplicate:
                applied_guard = t.zeros_like(applied_guard)
            guard_evidence["guard_component_for_current_new_command_nm"] = applied_guard
            guard_evidence["duplicate_holds_prior_complete_effort_without_new_guard_evaluation"] = duplicate
            evidence["contact_domain_posture_guard"] = guard_evidence
        if transition_evidence is not None:
            evidence["contact_transition"] = transition_evidence
        if freeze_reference_by_environment is not None:
            evidence["positional_reference_freeze"] = {
                "requested_by_environment": freeze_reference_by_environment.clone(),
                "fresh_safe_sample_applied": freeze_reference,
                "fresh_reference_equals_measured_hand": freeze_reference[:, None] & (reference == q),
                "nominal_chasing_increment_disabled": freeze_reference,
                "current_normal_feedback_gravity_and_overforce_law_unchanged": True,
                "duplicate_keeps_prior_complete_effort_without_new_freeze": duplicate,
                "hard_unsafe_zero_effort_priority_unchanged": True,
                "freeze_is_not_a_force_cap_or_verified_safe_placement": True}
        return effort, evidence, unsafe
