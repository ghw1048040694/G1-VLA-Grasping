"""Frozen-policy control diagnosis through the project's native entry point.

The historical checkpoint keeps its original identity. Modified control is an
explicit counterfactual, never a source-identical trained-policy evaluation.
No runner, critic, optimizer, backward pass, reward ablation or training exists
in this module. Physical support is scored by the unchanged environment gates.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .rl_probe_protocol import validate_probe_settings, verify_probe_reference


DIAGNOSTIC_VERSION = "g1_frozen_actor_opposing_closure_control_v4"
STAGED_PLANS = ("geometry_pose_only_v2", "geometry_pose_fixed_cluster_v3", "compliant_pinch_v4", "physics_pinch_v5")
POSE_SERVO = {"position_gain_per_s": 4.0, "orientation_gain_per_s": 2.0,
              "dls_regularizer": 0.02, "position_speed_max_m_s": 0.01,
              "angular_speed_max_rad_s": 0.1,
              "geometry_path_tracking_position_m": .003,
              "geometry_path_tracking_orientation_rad": .05,
              "reference": "actual source reset palm pose; only a gated 2.5mm probe may ascend"}


@contextmanager
def inference_only(torch):
    """Fail immediately if this diagnostic process attempts training."""
    original_init = torch.optim.Optimizer.__init__
    original_tensor_backward = torch.Tensor.backward
    original_backward = torch.autograd.backward

    def forbidden(*args, **kwargs):
        raise RuntimeError("rl-diagnose prohibits optimizer construction and backward")

    torch.optim.Optimizer.__init__ = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    try:
        with torch.no_grad():
            yield
    finally:
        torch.optim.Optimizer.__init__ = original_init
        torch.Tensor.backward = original_tensor_backward
        torch.autograd.backward = original_backward


def make_frozen_actor(torch, observations, saved, rl_settings):
    """Use the installed RSL model directly, without constructing PPO."""
    from rsl_rl.models import MLPModel
    from .rl_ppo import build_rsl_config, normalize_settings

    cfg = build_rsl_config(normalize_settings(rl_settings))
    actor_cfg = copy.deepcopy(cfg["actor"])
    actor_cfg.pop("class_name")
    actor = MLPModel(observations, cfg["obs_groups"], "actor", 14, **actor_cfg)
    actor = actor.to(rl_settings["device"])
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    actor.eval()
    actor.requires_grad_(False)
    return actor


def _bounded_vector(torch, value, maximum):
    length = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value * (maximum / length.clamp_min(1e-12)).clamp(max=1)


def _orientation_error(torch, reference, current):
    """Shortest small-angle world error for scalar-last unit quaternions."""
    left, right = reference[:, :3], -current[:, :3]
    xyz = (reference[:, 3:] * right + current[:, 3:] * left +
           torch.cross(left, right, dim=-1))
    w = reference[:, 3] * current[:, 3] - (left * right).sum(-1)
    return 2 * xyz * torch.where(w >= 0, 1., -1.)[:, None]


class GuidedClosure:
    """A declared diagnostic intervention, not an RL policy or demonstration."""

    def __init__(self, torch, settings, initial_hand, state, candidate=None):
        self.torch = torch
        self.settings = settings
        q = state["q_action_rad"]
        self.seed_hand = torch.as_tensor(initial_hand, dtype=q.dtype, device=q.device)[None].expand(q.shape[0], -1).clone()
        self.goal_hand = torch.as_tensor(settings["guided_hand_target_rad"], dtype=q.dtype, device=q.device)[None].expand_as(self.seed_hand).clone()
        limits = state["joint_limits_action_rad"][:, 7:]
        if bool(((self.goal_hand < limits[..., 0]) | (self.goal_hand > limits[..., 1])).any()):
            raise ValueError("Guided hand target violates native joint limits")
        self.ids = state["episode_id"].clone()
        self.palm = state["palm_position_world_m"].clone()
        self.rotation = state["palm_quaternion_xyzw"].clone()
        self.ready_count = torch.zeros_like(self.ids)
        self.probe = torch.zeros_like(q[:, 0])
        self.physics_pinch = settings.get("profile_plan") == "physics_pinch_v5"
        self.compliant = settings.get("profile_plan") in ("compliant_pinch_v4", "physics_pinch_v5")
        self.staged = settings.get("profile_plan") in STAGED_PLANS
        if self.staged:
            if candidate is None:
                raise ValueError("Staged closure requires its verified geometry candidate")
            frames = candidate["geometry_evidence"]["source_detail"]["stage_keyframes"]
            source = torch.as_tensor(frames[0]["target_rad"], dtype=q.dtype, device=q.device)
            if (not bool(torch.allclose(self.seed_hand, source[None].expand_as(self.seed_hand), atol=1e-4, rtol=0)) or
                    not bool(torch.allclose(state["q_action_rad"][:, 7:], self.seed_hand, atol=1e-4, rtol=0))):
                raise ValueError("Staged geometry path source differs from actual reset seed")
            self.stage_goals = torch.as_tensor([x["target_rad"] for x in frames[1:]], dtype=q.dtype, device=q.device)
            self.stage_durations = torch.as_tensor([x["duration_s"] for x in frames[1:]], dtype=q.dtype, device=q.device)
            if (self.stage_goals.shape != (5, 7) or self.stage_durations.shape != (5,) or
                    not bool(torch.isfinite(self.stage_goals).all()) or
                    not bool(torch.isfinite(self.stage_durations).all()) or
                    not bool((self.stage_durations > 0).all()) or
                    not bool(torch.equal(self.stage_goals[-1], self.goal_hand[0]))):
                raise ValueError("Staged geometry path dimensions/durations/final target are invalid")
            if bool(((self.stage_goals[None] < limits[:, None, :, 0]) |
                     (self.stage_goals[None] > limits[:, None, :, 1])).any()):
                raise ValueError("Staged geometry path violates native hand limits")
            self.stage_names = [x["name"] for x in frames[1:]]
            self.stage_index = torch.zeros_like(self.ids)
            self.stage_elapsed = torch.zeros_like(q[:, 0])
            self.stage_start = self.seed_hand.clone()
        if self.compliant:
            from .rl_hand_control import CompliantPinchController, PhysicsPinchTrajectory
            controller = PhysicsPinchTrajectory if self.physics_pinch else CompliantPinchController
            self.pinch = controller(torch, initial_hand, state, candidate,
                parameters=settings["compliant_pinch_controller"]["parameters"])

    def _staged_target(self, state, changed, dt, force_limit, tracking):
        torch = self.torch
        if bool(changed.any()) and not bool(torch.allclose(
                state["q_action_rad"][changed, 7:], self.seed_hand[changed], atol=1e-4, rtol=0)):
            raise ValueError("Staged episode reset differs from the measured source hand")
        self.stage_index[changed] = 0
        self.stage_elapsed[changed] = 0
        self.stage_start[changed] = self.seed_hand[changed]
        index = self.stage_index.clone()
        goal = self.stage_goals[index]
        duration = self.stage_durations[index]
        self.stage_elapsed += torch.where(tracking, dt, 0.)
        fraction = (self.stage_elapsed / duration).clamp(0, 1)
        smooth = fraction.square() * (3 - 2 * fraction)
        desired = self.stage_start + (goal - self.stage_start) * smooth[:, None]
        error = (state["q_action_rad"][:, 7:] - goal).abs().amax(-1)
        arrived = ((fraction >= 1) & (error <= .01) & tracking &
                   ~(state["group_normal_force_n"] > force_limit).any(-1))
        final = index == len(self.stage_names)-1
        advance = arrived & ~final
        # Time alone never advances a stage: use measured joints before reset.
        self.stage_start[advance] = goal[advance]
        self.stage_index[advance] += 1
        self.stage_elapsed[advance] = 0
        evidence = {"closure_stage_index": index, "closure_stage_fraction": fraction,
            "closure_stage_names": self.stage_names, "stage_goal_hand_rad": goal,
            "stage_measured_error_max_rad": error, "stage_arrival_tolerance_rad": .01,
            "closure_paused_for_palm_tracking": ~tracking,
            "stage_advance_after_this_request": advance, "closure_path_complete": arrived & final}
        return desired, fraction, evidence

    def actions(self, actor_actions, state, action_source, dt, force_limit):
        torch, cfg = self.torch, self.settings
        changed = state["episode_id"] != self.ids
        self.ids = state["episode_id"].clone()
        self.palm[changed] = state["palm_position_world_m"][changed]
        self.rotation[changed] = state["palm_quaternion_xyzw"][changed]
        self.ready_count[changed] = 0
        self.probe[changed] = 0
        stage_evidence = {}
        if self.staged:
            position_error = torch.linalg.vector_norm(self.palm-state["palm_position_world_m"], dim=-1)
            orientation_error = torch.linalg.vector_norm(_orientation_error(
                torch, self.rotation, state["palm_quaternion_xyzw"]), dim=-1)
            tracking = ((position_error <= POSE_SERVO["geometry_path_tracking_position_m"]) &
                        (orientation_error <= POSE_SERVO["geometry_path_tracking_orientation_rad"]))
            if self.compliant:
                method = self.pinch.reference if self.physics_pinch else self.pinch.target
                desired_hand, fraction, stage_evidence = method(
                    state, changed, dt, force_limit, tracking)
            else:
                desired_hand, fraction, stage_evidence = self._staged_target(state, changed, dt, force_limit, tracking)
            stage_evidence.update(actual_palm_reference_position_error_m=position_error,
                                  actual_palm_reference_orientation_error_rad=orientation_error)
        else:
            fraction = ((state["episode_control_steps"].to(actor_actions.dtype) + 1) * dt /
                        cfg["guided_closure_ramp_s"]).clamp(0, 1)
            smooth = fraction.square() * (3 - 2 * fraction)
            desired_hand = self.seed_hand + (self.goal_hand - self.seed_hand) * smooth[:, None]
        targets = state["target_action_rad"]
        per_step = state["per_step_action_rad"]
        overloaded = state["group_normal_force_n"] > force_limit
        if self.compliant:
            # The new controller owns force admittance and its two measured-q
            # lead bounds. Applying the historical relief/lead again would erase
            # its coordinated targets and create a second hidden controller.
            planned_hand = None
            lead = None
        else:
            direction = torch.sign(self.goal_hand - self.seed_hand)
            group_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2], device=targets.device)
            relief = overloaded[:, group_ids]
            relieved = targets[:, 7:] - direction * cfg["guided_force_relief_step_rad"]
            # Relief retreats toward the seed; it never opens past that declared seed.
            relieved = torch.where(direction > 0, torch.maximum(relieved, self.seed_hand),
                        torch.where(direction < 0, torch.minimum(relieved, self.seed_hand), targets[:, 7:]))
            desired_hand = torch.where(relief, relieved, desired_hand)
            planned_hand = desired_hand.clone()
            lead = cfg.get("guided_target_lead_max_rad")
            if lead is not None:
                measured_hand = state["q_action_rad"][:, 7:]
                desired_hand = torch.maximum(torch.minimum(desired_hand, measured_hand + lead), measured_hand - lead)
        result = actor_actions.clone()
        result[:, 7:] = ((desired_hand - targets[:, 7:]) / per_step[..., 7:]).clamp(-1, 1)
        evidence = {"action_source": action_source, "guided_target_hand_rad": desired_hand,
                    "planned_hand_before_tracking_lead_rad": planned_hand,
                    "guided_target_lead_max_rad": lead,
                    "guided_request_lead_is_not_applied_target_bound": True,
                    "tracking_lead_is_not_a_force_bound": True,
                    "force_relief_by_group": overloaded, "closure_fraction": fraction, **stage_evidence}
        if action_source == "guided_hand_pose_hold":
            ready = state["preload_contact_ready"].bool()
            if self.staged:
                ready = ready & stage_evidence["closure_path_complete"]
            self.ready_count = torch.where(ready, self.ready_count + 1, torch.zeros_like(self.ready_count))
            permit = self.ready_count >= cfg["guided_preload_controls"]
            self.probe = torch.where(permit, (self.probe + cfg["guided_probe_speed_m_s"] * dt).clamp(
                max=cfg["guided_probe_height_m"]), self.probe)
            goal = self.palm.clone()
            goal[:, 2] += self.probe
            linear = _bounded_vector(torch, (goal - state["palm_position_world_m"]) *
                                     POSE_SERVO["position_gain_per_s"], POSE_SERVO["position_speed_max_m_s"])
            # Pausing the reference offset alone would still chase an unfinished
            # upward probe. After any probe has started, loss of contact also
            # suppresses its positive Cartesian request. Before probing, ordinary
            # restoration to the original reset pose remains a pose-hold action.
            pause_upward = (~permit) & (self.probe > 0)
            linear[:, 2] = torch.where(pause_upward, linear[:, 2].clamp(max=0), linear[:, 2])
            angular = _bounded_vector(torch, _orientation_error(torch, self.rotation,
                state["palm_quaternion_xyzw"]) * POSE_SERVO["orientation_gain_per_s"],
                POSE_SERVO["angular_speed_max_rad_s"])
            jac = state["palm_point_jacobian_w"]
            rhs = torch.cat((linear, angular), -1)
            regular = torch.eye(6, dtype=jac.dtype, device=jac.device)[None] * POSE_SERVO["dls_regularizer"]**2
            velocity = (jac.transpose(1, 2) @ torch.linalg.solve(
                jac @ jac.transpose(1, 2) + regular, rhs[:, :, None])).squeeze(-1)
            # The historical v1 resets the PD target about measured q each
            # control. v2 integrates the previous applied bounded target so a
            # static gravity deflection does not erase corrective position error.
            anchor = targets[:, :7] if self.staged else state["q_action_rad"][:, :7]
            desired_arm = anchor + velocity * dt
            result[:, :7] = ((desired_arm - targets[:, :7]) / per_step[..., :7]).clamp(-1, 1)
            evidence.update(palm_pose_reference_world_m=goal, palm_rotation_reference_xyzw=self.rotation,
                preload_contact_ready=ready, preload_control_boundary_count=self.ready_count,
                probe_offset_m=self.probe, probe_advance_permitted=permit,
                upward_probe_request_clamped=pause_upward, requested_linear_velocity_world_m_s=linear,
                arm_target_reference="previous_applied_bounded_target" if self.staged else "measured_q_legacy_v1",
                preload_is_not_support_qualification=True, servo=dict(POSE_SERVO))
        if self.compliant:
            evidence["declared_controller_protocol"] = copy.deepcopy(cfg["compliant_pinch_controller"])
        return result, evidence


def _profile_summary(environment, actor, torch, cfg, settings, streams, deadline, profile, saved_infos):
    from .rl_environment import RIGHT_HAND
    from .rl_ppo import (EvaluationQuota, _assert_finite, _environment_audit, _json_safe,
                         _record_step, module_state_sha256, source_tree_identity,
                         verify_contact_measurements)

    native = environment.unwrapped
    native.configure_control_probe(profile)
    native.set_evaluation_mode(True, seed=cfg["rl_teacher"]["evaluation_seed"])
    environment.reset()
    observations = environment.get_observations().to(cfg["rl_teacher"]["device"])
    audit = _environment_audit(environment)
    audit["source_tree_sha256"] = source_tree_identity()
    if settings["profile_plan"] in ("compliant_pinch_v4", "physics_pinch_v5"):
        # This is an explicit software-control declaration, not a measured
        # contact or a change to the environment's physical policy contract.
        audit["diagnostic_controller_protocol"] = copy.deepcopy(settings["compliant_pinch_controller"])
    reference = verify_probe_reference(cfg, checkpoint_infos=saved_infos, current_audit=audit)
    if not reference["ready_for_physical_probe"]:
        raise ValueError("Frozen actor diagnostic bridge did not admit physical execution")
    if settings.get("profile_plan") in STAGED_PLANS:
        load = audit.get("diagnostic_native_joint_load_readback", {})
        pair = audit.get("diagnostic_native_self_pair_readback", {})
        if (load.get("binding", {}).get("verified") is not True or
                load.get("native_drive_max_force_nm", {}).get("available") is not True or
                load.get("native_friction_properties", {}).get("available") is not True or
                pair.get("available") is not True):
            raise RuntimeError("Geometry retest requires available native load/readback and explicit self-pair routing before physics")
    before = module_state_sha256(actor, torch)
    initial_hand = [audit["initial_state"]["joint_positions_rad"][name] for name in RIGHT_HAND]
    candidate = None
    if settings.get("profile_plan") in STAGED_PLANS:
        payload = Path(settings["candidate_path"]).read_bytes()
        if hashlib.sha256(payload).hexdigest() != settings["candidate_sha256"]:
            raise ValueError("Geometry candidate changed after native reference verification")
        candidate = json.loads(payload)
    guided = GuidedClosure(torch, settings, initial_hand, native.control_diagnostic_state(), candidate)
    if settings["profile_plan"] == "physics_pinch_v5":
        native.configure_physics_pinch(settings["compliant_pinch_controller"])
    quota = EvaluationQuota(settings["num_envs"], settings["num_envs"] * settings["episodes_per_env"])
    steps = 0
    environment._g1_last_step_audit = _json_safe(native.step_audit())
    first_frame = native.step_audit()["evaluation_video"]["frames"]
    while not quota.complete and steps < settings["max_control_steps"]:
        deadline.check()
        state = native.control_diagnostic_state()
        requested = actor(observations)
        _assert_finite(requested, torch, "frozen actor request")
        evidence = {"action_source": profile["action_source"], "frozen_actor_request": requested,
                    "actor_observation_81": observations["policy"],
                    "actor_normalized_observation_81": actor.get_latent(observations),
                    "episode_id": state["episode_id"], "episode_control_steps": state["episode_control_steps"],
                    "native_physics_step_count": state["native_physics_step_count"],
                    "action_pipeline": "actor -> intervention -> wrapper[-1,1] -> shared bounds -> probe guard; control_commands.raw_action is wrapper output"}
        if profile["action_source"] != "frozen_actor":
            requested, extra = guided.actions(requested, state,
                profile["action_source"], .066, audit["policy_contract"]["resolved_settings"]["group_force_max_n"])
            evidence.update(extra)
            if settings["profile_plan"] == "physics_pinch_v5":
                native.set_physics_pinch_reference(
                    nominal_hand_rad=extra["guided_target_hand_rad"],
                    geometry_lower_hand_rad=extra["geometry_lower_hand_rad"],
                    geometry_upper_hand_rad=extra["geometry_upper_hand_rad"],
                    closing_axis_hand7=extra["force_admittance_closing_axis_hand7"],
                    palm_reference_world_m=extra["palm_pose_reference_world_m"],
                    palm_reference_quaternion_xyzw=extra["palm_rotation_reference_xyzw"])
        streams["intent"].write({"profile": profile["name"], "control_step": steps + 1, **_json_safe(evidence),
                                 "intervention_request": _json_safe(requested)})
        observations, rewards, dones, extras = environment.step(requested.to(environment.device))
        native.record_control_probe_transition()
        _assert_finite((observations, rewards), torch, "diagnostic physical transition")
        observations = observations.to(cfg["rl_teacher"]["device"])
        steps += 1
        quota.add(_record_step(environment, extras, streams, profile["name"], steps, 78, torch))
        video = native.step_audit()["evaluation_video"]
        if video["failure"] is not None or not video["enabled"] or video["frames"] - first_frame != steps:
            raise RuntimeError("Diagnostic camera evidence is incomplete; physical results remain partial")
    after = module_state_sha256(actor, torch)
    if before != after:
        raise RuntimeError("Diagnostic inference changed actor weights or normalization")
    final = _json_safe(native.step_audit())
    channel = verify_contact_measurements(final, require_table_witness=True)
    summary = {**quota.report(), "profile": profile, "control_steps": steps,
               "optimizer_steps": 0, "source_identical_trained_eval": False,
               "frozen_actor_state_sha256_before": before, "frozen_actor_state_sha256_after": after,
               "reference_bridge": reference, "environment_audit": audit,
               "table_contact_channel": channel, "final_step_audit": final,
               "first_video_frame": first_frame, "last_video_frame_exclusive": final["evaluation_video"]["frames"],
               "result_scope": "diagnostic intervention; unchanged strict local gates; not a trained teacher acceptance"}
    streams["validations"].write(summary)
    return summary


def run_native_diagnostics(config, run_dir):
    """Execute declared counterfactuals with one native world and zero training."""
    from .rl_ppo import (Deadline, EvidenceStream, _environment_audit, _gpu_memory,
                         _json_safe, _new_streams, _seed, _write_json_new, file_sha256)

    settings = validate_probe_settings(config["rl_diagnostics"])
    if not settings["enabled"]:
        raise ValueError("rl-diagnose requires rl_diagnostics.enabled=true")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    result = {"diagnostic_version": DIAGNOSTIC_VERSION, "stage": "rl-diagnose", "passed": False,
              "status": "starting", "training_started": False, "optimizer_steps": 0,
              "source_identical_trained_eval": False, "profiles": [],
              "declared_profile_plan": settings["profile_plan"],
              "planned_profiles": copy.deepcopy(settings["profiles"]),
              "four_way_comparison_complete": False,
              "reward_used_as_success": False, "original_focus_or_20_acceptance": False,
              "started_utc": datetime.now(timezone.utc).isoformat(), "settings": settings}
    streams = _new_streams(run_dir)
    streams["intent"] = EvidenceStream(run_dir / "control_intent.jsonl")
    app, environment, torch = None, None, None
    deadline = Deadline(settings["max_wall_time_s"])
    try:
        result["reference_preflight"] = verify_probe_reference(config)
        os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = config["rl_teacher"]["git_executable"]
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, device=config["rl_teacher"]["device"], enable_cameras=True).app
        import torch as torch_module
        torch = torch_module
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from .rl_environment import build_environment
        _seed(config["rl_teacher"]["seed"], torch)
        deadline.check()
        result["runtime_versions"] = {name: importlib.metadata.version(name) for name in
                                     ("torch", "isaaclab", "isaaclab_rl", "rsl-rl-lib")}
        if result["runtime_versions"]["rsl-rl-lib"] != "5.0.1":
            raise ValueError("Frozen diagnostic actor requires the audited RSL5.0.1 model")
        memory = _gpu_memory(torch, config["rl_teacher"]["device"])
        result["memory_before_build"] = memory
        streams["memory"].write({"num_envs": settings["num_envs"], "phase": "before_build", "memory": memory})
        checkpoint = Path(settings["reference_checkpoint_path"])
        if file_sha256(checkpoint) != settings["reference_checkpoint_sha256"]:
            raise ValueError("Diagnostic checkpoint hash changed")
        saved = torch.load(checkpoint, map_location=config["rl_teacher"]["device"], weights_only=False)
        result["reference_checkpoint_metadata"] = verify_probe_reference(config, checkpoint_infos=saved.get("infos", {}))
        with inference_only(torch):
            environment = build_environment(config, num_envs=settings["num_envs"],
                device=config["rl_teacher"]["device"], asset_path=config["rl_teacher"]["asset_path"],
                train_seed=config["rl_teacher"]["seed"], eval_mode=True,
                output_dir=str(run_dir / "environment_8"), control_probe=settings["profiles"][0])
            environment = RslRlVecEnvWrapper(environment, clip_actions=1.0)
            memory = _gpu_memory(torch, config["rl_teacher"]["device"])
            result["memory_after_build"] = memory
            streams["memory"].write({"num_envs": settings["num_envs"], "phase": "after_build", "memory": memory})
            if memory.get("host_available_ram_bytes", 0) < 2**30 or memory.get("free_bytes", 0) < 2**30:
                raise MemoryError("Diagnostic world lacks its declared 1GiB host/device reserve")
            actor = make_frozen_actor(torch, environment.get_observations(), saved, config["rl_teacher"])
            for profile in settings["profiles"]:
                deadline.check()
                summary = _profile_summary(environment, actor, torch, config, settings,
                    streams, deadline, profile, saved["infos"])
                result["profiles"].append(summary)
                print("G1_CONTROL_PROFILE=" + json.dumps({"name": profile["name"],
                    "completed": summary["completed_episodes"], "successes": summary["successes"],
                    "quota_complete": summary["target_completed"], "optimizer_steps": 0}), flush=True)
                if not summary["target_completed"]:
                    raise RuntimeError("Diagnostic fixed per-offset quota was incomplete")
        result.update(passed=True, status="completed", all_declared_profiles_complete=True,
                      four_way_comparison_complete=settings["profile_plan"] == "four_way_v1")
    except BaseException as exc:
        result.update(status="failed", failure={"type": type(exc).__name__, "message": str(exc),
                                                  "traceback": traceback.format_exc()})
    finally:
        if environment is not None:
            try:
                environment.close()
            except BaseException as exc:
                result["environment_close_error"] = str(exc)
                result["passed"] = False
        video_status = run_dir / "environment_8/evaluation_video_status.json"
        result["evaluation_video_status"] = json.loads(video_status.read_text()) if video_status.is_file() else None
        result["video_evidence_complete"] = bool(result["evaluation_video_status"] and
            result["evaluation_video_status"].get("all_control_frames_recorded") is True and
            result["evaluation_video_status"].get("failure") is None)
        if not result["video_evidence_complete"]:
            result["passed"] = False
            result["status"] = "failed"
        if torch is not None:
            result["memory_final"] = _gpu_memory(torch, config["rl_teacher"]["device"])
        for stream in streams.values():
            stream.close()
        result["evidence"] = [{"path": str(stream.path), "bytes": stream.path.stat().st_size,
                               "records": stream.count, "sha256": file_sha256(stream.path)}
                              for stream in streams.values()]
        result["wall_elapsed_seconds"] = deadline.elapsed
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json_new(run_dir / "native_report.json", _json_safe(result))
        print("G1_RL_NATIVE_REPORT=" + json.dumps({"path": str(run_dir / "native_report.json"),
              "passed": result["passed"], "stage": "rl-diagnose"}), flush=True)
        if app is not None:
            app.close(exit_code=0 if result["passed"] else 2)
    return result
