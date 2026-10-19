"""Native-sample admission for a task-preshaped hand approaching from arm home.

This module never writes simulation state. Original full-lift force/domain
histories remain owned by ContactQualifiedFullLift throughout the same episode.
"""

from copy import deepcopy
from pathlib import Path


ARM_NAMES = tuple("right_"+s+"_joint" for s in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
    "wrist_roll", "wrist_pitch", "wrist_yaw"))


def current_object_height_unsafe(torch, clearance, object_lift, main_lift_started,
                                 *, local_cap_m, local_tolerance_m, full_cap_m):
    """Use the original current-sample height gates before logging admission."""
    cap=torch.where(main_lift_started,torch.full_like(clearance,full_cap_m),
                    torch.full_like(clearance,local_cap_m+local_tolerance_m))
    return (~torch.isfinite(clearance)|~torch.isfinite(object_lift)|
            (clearance>cap)|(object_lift>cap))


def load_course_reference(config, settings, candidate):
    from .rl_approach import load_approach_resource
    from .rl_collision_geometry import pose_matrix_xyzw
    declaration = settings.get("approach_course")
    if declaration is None:
        return None, None
    path = Path(declaration["resource_path"])
    if not path.is_absolute():
        path = Path(config["paths"]["root"])/path
    resource = load_approach_resource(path, declaration["resource_sha256"],
        expected_candidate_sha256=settings["candidate_sha256"])
    if (resource["pregrasp_joint_positions_rad"] != candidate["reset_joint_positions_rad"] or
            resource["root_pose_xyzw"] != candidate["reset_root_pose_xyzw"] or
            resource["source_hand_rad"] != candidate["source_hand_rad"] or
            resource["pregrasp_palm_position_world_m"] != candidate["reset_palm_pose_world_m"] or
            resource["pregrasp_palm_quaternion_xyzw"] != candidate["reset_palm_quaternion_xyzw"]):
        raise ValueError("Arm-home reference must preserve the exact original pregrasp target and root")
    old = candidate["reset_native_fk_reference"]
    matrix = pose_matrix_xyzw(resource["home_palm_position_world_m"], resource["home_palm_quaternion_xyzw"])
    fk = {k: deepcopy(old[k]) for k in (
        "usd_entry_sha256", "usd_structure_sha256", "joint_position_units", "root_body", "palm_site_body", "body_count")}
    fk.update(method="independent_official_native_FK_task_preshape_arm_home_v1",
        palm_world_position_m=deepcopy(resource["home_palm_position_world_m"]),
        palm_quaternion_xyzw=deepcopy(resource["home_palm_quaternion_xyzw"]),
        palm_rotation_world=[row[:3] for row in matrix[:3]],
        approach_resource_sha256=declaration["resource_sha256"],
        original_pregrasp_candidate_sha256=settings["candidate_sha256"])
    return resource, fk


class NativeApproachAdmission:
    """One irreversible admission and safety history, updated at every 2 ms."""

    def __init__(self, torch, resource, parameters, num_envs, device):
        self.t, self.resource, self.parameters = torch, deepcopy(resource), deepcopy(parameters)
        self.n, self.device = num_envs, device
        self.admitted = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.failed = torch.zeros_like(self.admitted)
        self.collision_clean = torch.ones_like(self.admitted)
        self.object_clean = torch.ones_like(self.admitted)
        self.count = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self.admission_native = torch.full_like(self.count, -1)
        self.first_failure_native = torch.full_like(self.count, -1)
        self.first_failure_code = torch.zeros_like(self.count)
        self.episode_id = torch.full_like(self.count, -1)
        self.reset_native = torch.full_like(self.count, -1)
        self.last_native = torch.full_like(self.count, -1)
        self.cube_reference = torch.zeros(num_envs, 7, device=device)
        self.last = None

    def reset(self, env_ids, episode_ids, native_step, cube_pose_xyzw):
        t=self.t
        if (type(native_step) is not int or native_step < 0 or not t.is_tensor(env_ids) or
                env_ids.ndim!=1 or env_ids.dtype not in (t.int32,t.int64) or
                env_ids.device!=self.admitted.device or len(set(env_ids.tolist()))!=env_ids.numel() or
                bool(((env_ids<0)|(env_ids>=self.n)).any()) or not t.is_tensor(episode_ids) or
                tuple(episode_ids.shape)!=tuple(env_ids.shape) or episode_ids.dtype not in (t.int32,t.int64) or
                episode_ids.device!=self.admitted.device or
                bool((episode_ids<=self.episode_id[env_ids]).any()) or
                bool(((self.last_native[env_ids]>=0)&(self.last_native[env_ids]!=native_step)).any())):
            raise ValueError("Admission reset requires unique env IDs, a strictly new episode and current completed boundary")
        if (not t.is_tensor(cube_pose_xyzw) or tuple(cube_pose_xyzw.shape)!=(env_ids.numel(),7) or
                cube_pose_xyzw.device!=self.admitted.device or not bool(t.isfinite(cube_pose_xyzw).all()) or
                bool(((t.linalg.vector_norm(cube_pose_xyzw[:,3:],dim=-1)-1).abs()>1e-4).any())):
            raise ValueError("Admission reset requires the actual declared finite unit-quaternion Cube reset")
        self.episode_id[env_ids] = episode_ids
        self.reset_native[env_ids] = native_step
        self.last_native[env_ids] = native_step
        self.cube_reference[env_ids] = cube_pose_xyzw
        for name in ("admitted", "failed", "count", "first_failure_code"):
            getattr(self, name)[env_ids] = 0
        for name in ("admission_native", "first_failure_native"):
            getattr(self, name)[env_ids] = -1
        self.collision_clean[env_ids] = True
        self.object_clean[env_ids] = True

    def update(self, *, native_step, episode_ids, q_action, qd_action,
               palm_position, palm_quaternion, cube_position, cube_quaternion,
               env_origins, measurement_valid, path_collision_clean,
               robot_cube_force_n, mechanical_safe, unsafe, whole_task_clean,
               plan_complete_requested, planner_failed, full_success):
        t, p, n = self.t, self.parameters, self.n
        if (type(native_step) is not int or bool((episode_ids != self.episode_id).any()) or
                bool((self.last_native+1 != native_step).any())):
            raise ValueError("Admission requires every native sample in the same real-reset epoch")
        for name, value, shape in (
                ("q", q_action, (n,14)), ("qd", qd_action, (n,14)),
                ("palm", palm_position, (n,3)), ("palm quaternion", palm_quaternion, (n,4)),
                ("cube", cube_position, (n,3)), ("cube quaternion", cube_quaternion, (n,4)),
                ("origins", env_origins, (n,3)), ("robot Cube force", robot_cube_force_n, (n,))):
            if not t.is_tensor(value) or tuple(value.shape) != shape or value.device != self.admitted.device:
                raise ValueError("Invalid admission measurement: "+name)
        for value in (measurement_valid, path_collision_clean, mechanical_safe, unsafe,
                      whole_task_clean, plan_complete_requested, planner_failed, full_success):
            if not t.is_tensor(value) or tuple(value.shape)!=(n,) or value.dtype != t.bool or value.device!=self.admitted.device:
                raise ValueError("Admission flags require matching per-environment boolean measurements")
        finite = t.isfinite(t.cat((q_action,qd_action,palm_position,palm_quaternion,
            cube_position,cube_quaternion,env_origins,robot_cube_force_n[:,None]),-1)).all(-1)
        quaternion_valid = ((t.linalg.vector_norm(palm_quaternion,dim=-1)-1).abs()<=1e-4) & (
            (t.linalg.vector_norm(cube_quaternion,dim=-1)-1).abs()<=1e-4)
        before = ~self.admitted
        cube_xy_error = t.linalg.vector_norm(cube_position[:,:2]-self.cube_reference[:,:2],dim=-1)
        cube_dot = (cube_quaternion*self.cube_reference[:,3:]).sum(-1).abs().clamp(0,1)
        cube_angle = 2*t.acos(cube_dot)
        object_now = ((cube_xy_error<=p["maximum_object_xy_drift_m"]) &
            (cube_angle<=p["maximum_object_rotation_rad"]) & (cube_position[:,2]-env_origins[:,2]>=.772))
        premature = before & (robot_cube_force_n!=0)
        self.collision_clean &= path_collision_clean
        self.object_clean &= ~before | object_now
        local_palm_z = palm_position[:,2]-env_origins[:,2]
        workspace = (local_palm_z>=p["workspace_min_palm_z_m"]) & (local_palm_z<=p["workspace_max_palm_z_m"])
        elapsed = (native_step-self.reset_native)*.002
        elapsed_grasp = (native_step-self.admission_native)*.002
        timeout = (before & (elapsed>p["approach_timeout_s"])) | (
            self.admitted & ~full_success & (elapsed_grasp>p["grasp_budget_s"]))
        code = ((~measurement_valid | ~finite | ~quaternion_valid).long() +
            (~path_collision_clean).long()*2 + premature.long()*4 +
            (before & ~object_now).long()*8 + (~workspace).long()*16 +
            timeout.long()*32 + (~mechanical_safe | unsafe | ~whole_task_clean).long()*64 + planner_failed.long()*128)
        newly_failed = (code!=0) & ~self.failed
        self.first_failure_native[newly_failed] = native_step
        self.first_failure_code[newly_failed] = code[newly_failed]
        self.failed |= code!=0
        target_arm = t.as_tensor([self.resource["pregrasp_joint_positions_rad"][k] for k in ARM_NAMES],device=self.device,dtype=q_action.dtype)
        target_hand = t.as_tensor(self.resource["source_hand_rad"],device=self.device,dtype=q_action.dtype)
        target_palm = t.as_tensor(self.resource["pregrasp_palm_position_world_m"],device=self.device,dtype=q_action.dtype)[None]+env_origins
        target_quat = t.as_tensor(self.resource["pregrasp_palm_quaternion_xyzw"],device=self.device,dtype=q_action.dtype)
        arm_error = (q_action[:,:7]-target_arm).abs().amax(-1)
        arm_speed = qd_action[:,:7].abs().amax(-1)
        hand_error = (q_action[:,7:]-target_hand).abs().amax(-1)
        palm_error = t.linalg.vector_norm(palm_position-target_palm,dim=-1)
        angle = 2*t.acos((palm_quaternion*target_quat).sum(-1).abs().clamp(0,1))
        arrived = (plan_complete_requested & (arm_error<=p["joint_arrival_tolerance_rad"]) &
            (arm_speed<=p["joint_velocity_arrival_tolerance_rad_s"]) &
            (hand_error<=p["hand_seed_arrival_tolerance_rad"]) &
            (palm_error<=p["palm_position_arrival_tolerance_m"]) &
            (angle<=p["palm_rotation_arrival_tolerance_rad"]))
        eligible = before & arrived & ~self.failed
        self.count = t.where(eligible,self.count+1,t.where(self.admitted,self.count,t.zeros_like(self.count)))
        new_admission = before & ~self.failed & (self.count>=p["handoff_required_consecutive_native_samples"])
        self.admitted |= new_admission
        self.admission_native[new_admission] = native_step
        self.last_native[:] = native_step
        self.last = {"version":"g1_native_arm_home_admission_v1", "episode_id":self.episode_id.clone(),
            "native_physics_step_count":native_step, "reset_native":self.reset_native.clone(),
            "admitted":self.admitted.clone(), "new_admission":new_admission,
            "admission_native":self.admission_native.clone(), "consecutive_arrival_samples":self.count.clone(),
            "failed":self.failed.clone(), "first_failure_native":self.first_failure_native.clone(),
            "first_failure_code":self.first_failure_code.clone(), "whole_path_collision_clean":self.collision_clean.clone(),
            "whole_approach_object_clean":self.object_clean.clone(), "measurement_valid":measurement_valid,
            "path_collision_clean_current":path_collision_clean, "robot_cube_force_n":robot_cube_force_n,
            "premature_robot_cube_contact":premature, "object_xy_error_m":cube_xy_error,
            "object_rotation_error_rad":cube_angle, "palm_workspace_current":workspace,
            "plan_complete_requested":plan_complete_requested, "planner_failed":planner_failed,
            "entry_arm_error_max_rad":arm_error, "entry_arm_speed_max_rad_s":arm_speed,
            "entry_hand_error_max_rad":hand_error, "entry_palm_error_m":palm_error,
            "entry_rotation_error_rad":angle, "entry_measured_kinematics_current":arrived,
            "admission_does_not_reset_physics_or_score_history":True}
        return self.last
