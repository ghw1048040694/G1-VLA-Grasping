"""Independent, prespecified numerical contact benchmark; never robot training data.

Only the explicit experiment stage executes this fixture. The cube is free;
kinematic jaws and a known, finite COM test load are the declared bench inputs.
Integration solver forces and subsequent refreshed forces are kept separately.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import canonical_sha256, scene_protocol_for_contact_model
from .runtime.experiment import write_new_json


class CalibrationError(ValueError):
    pass


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phase_boundaries(settings):
    hold = settings["hold_s"]
    pulse = hold + settings["disturbance_duration_s"]
    recover = pulse + settings["recovery_s"]
    opened = recover + settings["opening_duration_s"]
    return {"pulse_start": hold, "pulse_end": pulse, "opening_start": recover,
            "opening_end": opened, "end": opened + settings["free_fall_s"]}


def fixture_command(settings, time_s, timestep_s):
    """Absolute-time jaw pose and midpoint-sampled, COM-only known test force."""
    if (not all(math.isfinite(v) for v in (time_s, timestep_s))
            or time_s < 0 or timestep_s <= 0):
        raise CalibrationError("fixture time and timestep must be finite and valid")
    times = phase_boundaries(settings)
    midpoint = time_s + timestep_s / 2.
    force = np.zeros(3)
    if times["pulse_start"] <= midpoint < times["pulse_end"]:
        force[1] = settings["disturbance_peak_weight_fraction"] * settings["cube_mass_kg"] * 9.81 * math.sin(
            math.pi * (midpoint - times["pulse_start"]) / settings["disturbance_duration_s"])
    alpha = min(1., max(0., (time_s - times["opening_start"]) / settings["opening_duration_s"]))
    smooth = alpha * alpha * (3. - 2. * alpha)
    absolute_x = settings["capsule_center_x_m"] + smooth * settings["opening_distance_m"]
    phase = ("hold" if midpoint < times["pulse_start"] else
             "perturbation" if midpoint < times["pulse_end"] else
             "recovery" if midpoint < times["opening_start"] else
             "opening" if midpoint < times["opening_end"] else "free_fall")
    return {"input_time_s": float(time_s), "force_sample_time_s": float(midpoint), "phase": phase,
            "jaw_positions_m": [[-absolute_x, 0., settings["cube_center_z_m"]],
                                [absolute_x, 0., settings["cube_center_z_m"]]],
            "jaw_quaternions_wxyz": [[1., 0., 0., 0.], [1., 0., 0., 0.]],
            "external_force_world_n": force.tolist(), "external_torque_world_nm": [0., 0., 0.]}


def build_fixture_xml(settings, backend, timestep_s, *, floor_contact):
    """New run-local bench, with the common scene helper's versioned sole delta."""
    from .sim.scene import _apply_contact_model
    if backend not in settings["solver_backends"] or timestep_s not in settings["physics_timesteps_s"]:
        raise CalibrationError("trial is outside the prespecified four-case matrix")
    root = ET.Element("mujoco", model=settings["backend"])
    ET.SubElement(root, "option", timestep=format(timestep_s, ".17g"), gravity="0 0 -9.81",
                  integrator="Euler", solver="Newton", iterations="100", tolerance="1e-8",
                  cone="pyramidal", impratio="1", noslip_iterations="0", noslip_tolerance="1e-6")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 1", dir="0 0 -1")
    # Names are recorder aliases, not G1 head or wrist viewpoints.
    ET.SubElement(world, "camera", name="head_camera", pos="0.34 -0.46 0.46",
                  xyaxes="0.804 0.594 0 -0.290 0.392 0.872", fovy="50")
    ET.SubElement(world, "camera", name="right_wrist_camera", pos="0.34 0.30 0.40",
                  xyaxes="-0.662 0.750 0 -0.373 -0.329 0.868", fovy="50")
    floor = {key: str(value) if not isinstance(value, (list, tuple)) else " ".join(map(str, value))
             for key, value in floor_contact.items()}
    ET.SubElement(world, "geom", name="catch_floor", type="plane", size="1 1 .01",
                  pos="0 0 0", rgba=".3 .3 .3 1", **floor)
    common = {"condim": str(settings["condim"]), "friction": " ".join(map(str, settings["friction"])),
              "solref": " ".join(map(str, settings["solref"])),
              "solimp": " ".join(map(str, settings["solimp"]))}
    for name, sign in (("left_jaw", -1), ("right_jaw", 1)):
        body = ET.SubElement(world, "body", name=name, mocap="true",
            pos="{} 0 {}".format(sign * settings["capsule_center_x_m"], settings["cube_center_z_m"]))
        half = settings["capsule_half_length_m"]
        ET.SubElement(body, "geom", name=name + "_capsule", type="capsule",
            fromto="0 {} 0 0 {} 0".format(-half, half), size=str(settings["capsule_radius_m"]),
            rgba=".25 .45 .8 1", **common)
    cube = ET.SubElement(world, "body", name="calibration_cube", pos="0 0 " + str(settings["cube_center_z_m"]))
    ET.SubElement(cube, "freejoint", name="cube_free_joint")
    cube_contact = dict(common, friction=" ".join(map(str, settings["cube_friction"])))
    ET.SubElement(cube, "geom", name="calibration_cube_geom", type="box",
        size=" ".join([str(settings["cube_side_m"] / 2)] * 3), mass=str(settings["cube_mass_kg"]),
        rgba=".15 .8 .25 1", **cube_contact)
    declaration = {"backend": backend, "time_constant_s": settings["solref"][0],
                   "damping_ratio": settings["solref"][1]}
    model_details = _apply_contact_model(root, declaration)
    xml = ET.tostring(root, encoding="utf-8")
    manifest = {"fixture_only": True, "training_eligible": False, "fixture_protocol": settings["backend"],
        "contact_solver_protocol": scene_protocol_for_contact_model(declaration), "contact_model": model_details,
        "fixture_scene_protocol": settings["backend"] + "_" + hashlib.sha256(xml).hexdigest(),
        "physics_dt_s": timestep_s, "fixture_sha256": hashlib.sha256(xml).hexdigest(),
        "initial_soft_precompression_m": settings["cube_side_m"] / 2 + settings["capsule_radius_m"] - settings["capsule_center_x_m"],
        "floor_contact_parameters_from_source_table": floor_contact,
        "original_jaw_friction3": settings["friction"], "original_cube_friction3": settings["cube_friction"],
        "kinematic_jaws_infinite_support_declared": True, "cube_free_no_attachment": True,
        "external_load_scope": "known finite world-y COM benchmark pulse only; never a G1 task load",
        "camera_aliases": {"head_camera": "bench front view", "right_wrist_camera": "bench oblique view"}}
    return xml, manifest


def rotation_distance(first, second):
    from scipy.spatial.transform import Rotation
    return float(np.linalg.norm(Rotation.from_matrix(np.asarray(first).T @ np.asarray(second)).as_rotvec()))


def contact_wrench(model, data, cube_body, com_world, api):
    """Read whichever solver state the caller supplies, without any forward/step."""
    rows, total = [], {kind: np.zeros(6) for kind in ("left_jaw", "right_jaw", "floor", "other")}
    forces = {name: 0. for name in ("left_jaw", "right_jaw")}
    depth, geometric_jaw_contacts, other_loaded, other_force = 0., 0, 0, 0.
    other_geometric, maximum_jaw_force = 0, 0.
    max_cone_excess = 0.
    for index in range(data.ncon):
        c = data.contact[index]
        gids = (int(c.geom1), int(c.geom2))
        bodies = [int(model.geom_bodyid[g]) for g in gids]
        if cube_body not in bodies:
            continue
        object_index = bodies.index(cube_body)
        other_geom = gids[1 - object_index]
        body_name = api.mj_id2name(model, api.mjtObj.mjOBJ_BODY, bodies[1 - object_index]) or "world"
        geom_name = api.mj_id2name(model, api.mjtObj.mjOBJ_GEOM, other_geom) or str(other_geom)
        kind = body_name if body_name in forces else "floor" if geom_name == "catch_floor" else "other"
        force = np.zeros(6)
        api.mj_contactForce(model, data, index, force)
        frame = np.asarray(c.frame).reshape(3, 3)
        sign = 1. if object_index == 1 else -1.
        world_force = sign * frame.T @ force[:3]
        torque = np.cross(np.asarray(c.pos) - com_world, world_force) + sign * frame.T @ force[3:]
        loaded = bool(c.dist <= 0 and c.efc_address >= 0 and force[0] > 0)
        total[kind] += np.r_[world_force, torque]
        if kind in forces:
            geometric_jaw_contacts += int(c.dist <= 0)
            depth = max(depth, max(0., -float(c.dist)))
            maximum_jaw_force = max(maximum_jaw_force, float(np.max(np.abs(force[:3]))))
            if loaded:
                forces[kind] += float(force[0])
        else:
            other_geometric += int(c.dist <= 0)
            if loaded:
                other_loaded += 1
                other_force += float(force[0])
        mu = np.asarray(c.friction, dtype=float)
        utilization = (abs(force[1]) / mu[0] + abs(force[2]) / mu[1]) / force[0] if force[0] > 0 else None
        excess = max(0., abs(force[1]) / mu[0] + abs(force[2]) / mu[1] - force[0])
        max_cone_excess = max(max_cone_excess, excess)
        rows.append({"contact_index": index, "geom_ids": list(gids), "object_geom_index": object_index,
            "kind": kind, "other_geom": geom_name, "distance_m": float(c.dist), "dim": int(c.dim),
            "efc_address": int(c.efc_address), "loaded": loaded, "position_world_m": np.asarray(c.pos).tolist(),
            "frame_rows_world": frame.tolist(), "friction5": mu.tolist(), "force_contact_frame": force.tolist(),
            "force_on_cube_world_n": world_force.tolist(), "torque_on_cube_about_COM_world_nm": torque.tolist(),
            "pyramidal_L1_utilization": utilization, "pyramidal_cone_excess_n": excess})
    jaw = total["left_jaw"] + total["right_jaw"]
    return {"contacts": rows, "jaw_normal_force_n": forces, "jaw_force_world_n": jaw[:3].tolist(),
        "jaw_torque_about_COM_world_nm": jaw[3:].tolist(), "floor_other_loaded_contacts": other_loaded,
        "floor_other_geometric_contacts": other_geometric,
        "floor_other_normal_force_n": other_force, "all_jaw_geometric_contacts": geometric_jaw_contacts,
        "maximum_absolute_jaw_contact_force_n": maximum_jaw_force,
        "maximum_jaw_penetration_m": depth, "maximum_pyramidal_cone_excess_n": max_cone_excess,
        "all_force_world_n": sum(total.values())[:3].tolist(),
        "all_torque_about_COM_world_nm": sum(total.values())[3:].tolist()}


def _state(model, data, cube, api):
    velocity = np.zeros(6)
    api.mj_objectVelocity(model, data, api.mjtObj.mjOBJ_BODY, cube, velocity, 0)
    mass = float(model.body_mass[cube])
    R = np.asarray(data.xmat[cube]).reshape(3, 3)
    inertia_world = R @ np.diag(model.body_inertia[cube]) @ R.T
    kinetic = .5 * mass * float(velocity[3:] @ velocity[3:]) + .5 * float(velocity[:3] @ inertia_world @ velocity[:3])
    com = np.asarray(data.xipos[cube]).copy()
    return {"time_s": float(data.time), "qpos": data.qpos.tolist(), "qvel": data.qvel.tolist(),
        "act": data.act.tolist(), "ctrl": data.ctrl.tolist(), "qacc_warmstart": data.qacc_warmstart.tolist(),
        "mocap_pos": data.mocap_pos.tolist(), "mocap_quat": data.mocap_quat.tolist(),
        "xfrc_applied": data.xfrc_applied.tolist(), "qfrc_applied": data.qfrc_applied.tolist(),
        "COM_world_m": com.tolist(), "R_world": R.tolist(), "COM_velocity_world_m_s": velocity[3:].tolist(),
        "omega_world_rad_s": velocity[:3].tolist(), "kinetic_energy_j": kinetic,
        "potential_energy_j": -mass * float(np.asarray(model.opt.gravity) @ com)}


def _solver(model, data, api):
    niter = np.asarray(data.solver_niter).tolist()
    count = int(max(niter, default=0))
    stats = []
    for index in range(min(count, len(data.solver))):
        stat = data.solver[index]
        stats.append({key: float(getattr(stat, key)) for key in
            ("improvement", "gradient", "lineslope", "nactive", "nchange", "neval", "nupdate")
            if hasattr(stat, key)})
    constraints = {key: np.asarray(getattr(data, key)).tolist() for key in
        ("efc_force", "efc_R", "efc_b", "efc_aref", "efc_vel", "efc_pos", "efc_state") if hasattr(data, key)}
    rho = np.zeros(data.nefc)
    api.mj_mulJacVec(model, data, rho, data.qacc)
    rho -= data.efc_aref
    # Only this exact engine/monolithic Newton trace has a verified split.
    split = split_solver_trace(stats, count, engine_version=getattr(api, "__version__", None),
        newton=int(model.opt.solver) == int(api.mjtSolver.mjSOL_NEWTON),
        monolithic=not any(niter[1:]), main_budget=int(model.opt.iterations),
        noslip_budget=int(model.opt.noslip_iterations), main_tolerance=float(model.opt.tolerance),
        noslip_tolerance=float(model.opt.noslip_tolerance))
    projections, main_residual = [], None
    if data.nefc:
        J = np.zeros((data.nefc, model.nv))
        for j in range(model.nv):
            axis = np.zeros(model.nv)
            axis[j] = 1.
            column = np.zeros(data.nefc)
            api.mj_mulJacVec(model, data, column, axis)
            J[:, j] = column
        M = np.zeros((model.nv, model.nv))
        api.mj_fullM(model, M, data.qM)
        A = J @ np.linalg.solve(M, J.T)
        # Fixture-specific independent MAIN stationarity check. All active
        # constraints are pyramidal unilateral contact edges, with finite R.
        if all(int(c.dim) == 3 for c in data.contact if c.efc_address >= 0) and np.all(data.efc_R > 0):
            a_main = np.asarray(data.qacc_warmstart)
            rho_main = J @ a_main - data.efc_aref
            lambda_main = np.maximum(0., -rho_main / data.efc_R)
            primal_gradient = M @ (a_main - data.qacc_smooth) - J.T @ lambda_main
            scale = 1. / (float(model.stat.meaninertia) * model.nv)
            normalized = float(np.linalg.norm(primal_gradient) * scale)
            main_residual = {"normalized_primal_gradient_norm": normalized,
                "normalization_scale": scale, "primal_gradient": primal_gradient.tolist(),
                "reconstructed_MAIN_edge_forces": lambda_main.tolist(),
                "source_tolerance": float(model.opt.tolerance),
                "confirmed": bool(math.isfinite(normalized) and normalized <= model.opt.tolerance),
                "scope": "independent primal MAIN stationarity reconstructed from saved MAIN qacc, not final Noslip qacc"}
        for i in range(data.ncon):
            contact = data.contact[i]
            address = int(contact.efc_address)
            if int(contact.dim) != 3 or address < 0:
                continue
            for offset in (0, 2):
                j, k = address + offset, address + offset + 1
                first, second = float(data.efc_force[j]), float(data.efc_force[k])
                gradient = float(rho[j] - rho[k])
                curvature = float(A[j, j] + A[k, k] - A[j, k] - A[k, j])
                supported = first + second > 1e-15 and curvature > 0
                correction = float(np.clip(-gradient / curvature, -first, second)) if supported else None
                projections.append({"contact_index": i, "edge_indices": [j, k],
                    "force_sum_n": first + second, "unregularized_pair_gradient": gradient,
                    "unregularized_pair_curvature": curvature, "projected_force_correction_n": correction,
                    "supported": supported,
                    "scope": "post-Noslip fixed-state pair residual; no added R*force term"})
    official_main = split.get("main_last_reported_stop_test") is True
    main_confirmed = official_main or bool(main_residual and main_residual["confirmed"]) or data.nefc == 0
    if split["split_valid"]:
        split["main_termination_confirmed"] = main_confirmed
        split["main_termination_source"] = ("not_applicable_no_constraints" if data.nefc == 0 else
            "official_trace" if official_main else "independent_primal_MAIN_residual" if main_confirmed else "not_confirmed")
        split["budget_or_tolerance_failure"] = not main_confirmed or split["main_budget_exhausted"] or not split["noslip_last_improvement_stop_test"]
    deltas = [abs(p["projected_force_correction_n"]) for p in projections if p["projected_force_correction_n"] is not None]
    cube = api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, "calibration_cube")
    weight = float(model.body_mass[cube]) * abs(float(model.opt.gravity[2]))
    return {"solver_niter": niter, "iteration_stats": stats,
        "nefc": int(data.nefc), "nv": int(model.nv),
        "qacc_postprocessed": np.asarray(data.qacc).tolist(),
        "qacc_smooth": np.asarray(data.qacc_smooth).tolist(),
        "qacc_warmstart_MAIN_before_noslip": np.asarray(data.qacc_warmstart).tolist(),
        "solver_fwdinv": np.asarray(data.solver_fwdinv).tolist(),
        "iterations_budget": int(model.opt.iterations), "main_tolerance": float(model.opt.tolerance),
        "noslip_iterations_declared": int(model.opt.noslip_iterations),
        "noslip_tolerance": float(model.opt.noslip_tolerance),
        "termination": split, "post_noslip_projected_pair_diagnostics": projections,
        "MAIN_primal_stationarity": main_residual,
        "projected_force_correction_diagnostic": {"maximum_n": max(deltas, default=0.),
            "RMS_n": float(np.sqrt(np.mean(np.square(deltas)))) if deltas else 0.,
            "maximum_fraction_of_weight": max(deltas, default=0.) / weight,
            "supported_pairs": len(deltas), "unsupported_pairs": len(projections) - len(deltas),
            "scope": "quantitative residual only; no new residual pass tolerance or exact KKT claim"},
        "unregularized_acceleration_residual": rho.tolist(), "constraints": constraints,
        "warnings": [int(item.number) for item in data.warning],
        "scope": "native solver records before refresh; main regularized KKT is not a post-Noslip admission equation"}


def split_solver_trace(stats, total_count, *, engine_version, newton, monolithic,
                       main_budget, noslip_budget, main_tolerance, noslip_tolerance):
    """Decode only the verified 3.2.5 Newton trace layout; never guess a split."""
    result = {"split_valid": False, "main_iterations": None, "noslip_iterations": None,
              "budget_or_tolerance_failure": True}
    if engine_version != "3.2.5" or not newton or not monolithic or len(stats) != total_count:
        return result
    main, noslip, suffix = [], [], False
    for stat in stats:
        if stat.get("neval", 0) >= 2 and not suffix:
            main.append(stat)
        elif all(stat.get(key) == 0 for key in ("gradient", "lineslope", "neval", "nupdate")):
            suffix = True
            noslip.append(stat)
        else:
            return result
    if len(noslip) > noslip_budget:
        return result
    main_stop = bool(main[-1]["improvement"] < main_tolerance or main[-1]["gradient"] < main_tolerance) if main else None
    noslip_stop = not noslip_budget or not total_count or bool(noslip and noslip[-1]["improvement"] < noslip_tolerance)
    main_exhausted = len(main) >= main_budget and not main_stop
    noslip_exhausted = len(noslip) >= noslip_budget > 0 and not noslip_stop
    return {"split_valid": True, "main_iterations": len(main), "noslip_iterations": len(noslip),
        "main_last_reported_stop_test": main_stop, "noslip_last_improvement_stop_test": noslip_stop,
        "main_budget_exhausted": main_exhausted, "noslip_budget_exhausted": noslip_exhausted,
        "budget_or_tolerance_failure": main_exhausted or not main_stop or not noslip_stop,
        "main_early_stop_limitation": "CG alpha-zero early exit may have no additional trace; recorded residuals remain separate evidence"}


def residual_evidence_complete(solver):
    """Completeness/finiteness only; absence of a recorded residual fails closed."""
    try:
        size, nv = solver["nefc"], solver["nv"]
        for key in ("qacc_postprocessed", "qacc_smooth", "qacc_warmstart_MAIN_before_noslip"):
            values = np.asarray(solver[key], dtype=float)
            if values.shape != (nv,) or not np.isfinite(values).all():
                return False
        for key in ("efc_force", "efc_R", "efc_b", "efc_aref", "efc_vel", "efc_pos", "efc_state"):
            values = np.asarray(solver["constraints"][key], dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                return False
        rho = np.asarray(solver["unregularized_acceleration_residual"], dtype=float)
        if rho.shape != (size,) or not np.isfinite(rho).all():
            return False
        for pair in solver["post_noslip_projected_pair_diagnostics"]:
            for key in ("force_sum_n", "unregularized_pair_gradient", "unregularized_pair_curvature"):
                if not math.isfinite(pair[key]):
                    return False
            if pair["supported"] and not math.isfinite(pair["projected_force_correction_n"]):
                return False
        return all(all(math.isfinite(v) for v in stat.values()) for stat in solver["iteration_stats"])
    except (KeyError, TypeError, ValueError):
        return False


def score_trial(records, initial, settings, timestep_s, *, complete, video_complete):
    """Predetermined engineering gates; references are fixed recorded states."""
    times, gates = phase_boundaries(settings), settings["acceptance"]
    initial_com, initial_R = np.asarray(initial["COM_world_m"]), initial["R_world"]
    pre = next((r["state"] for r in records if abs(r["state"]["time_s"] - times["pulse_start"]) < 1e-8), None)
    released = next((r["state"] for r in records if abs(r["state"]["time_s"] - times["opening_end"]) < 1e-8), None)
    checks_by_phase = {key: [] for key in ("hold", "perturb_recovery", "recovery", "free_fall")}
    gravity = -9.81
    for row in records:
        s, force = row["state"], row["integration_force"]
        t = s["time_s"]
        drift = float(np.linalg.norm(np.asarray(s["COM_world_m"]) - initial_com))
        angle = rotation_distance(initial_R, s["R_world"])
        speed, omega = (float(np.linalg.norm(s[key])) for key in ("COM_velocity_world_m_s", "omega_world_rad_s"))
        weight = settings["cube_mass_kg"] * 9.81
        ratio = force["jaw_force_world_n"][2] / weight
        Fn = list(force["jaw_normal_force_n"].values())
        if .5 - 1e-8 <= t <= times["pulse_start"] + 1e-8:
            g = gates["hold_every_sample_in_final_half_second"]
            checks_by_phase["hold"].append({"time_s": t, "checks": {
                "global_drift": drift <= g["global_initial_COM_displacement_max_m"],
                "raw_rotation": angle <= g["raw_SO3_rotation_max_rad"],
                "speed": speed <= g["COM_speed_max_m_s"], "omega": omega <= g["angular_speed_max_rad_s"],
                "both_jaws_loaded": min(Fn) > g["each_jaw_loaded_Fn_min_n"],
                "normal_force_bounded": max(Fn) <= g["each_jaw_Fn_max_n"],
                "upward_support": g["hand_world_Fz_weight_ratio"][0] <= ratio <= g["hand_world_Fz_weight_ratio"][1],
                "no_other_support": force["floor_other_loaded_contacts"] == 0 and force["floor_other_normal_force_n"] <= g["floor_other_load_force_n_max"],
                "depth": force["maximum_jaw_penetration_m"] <= g["all_jaw_penetration_max_m"]}})
        if times["pulse_start"] < t <= times["opening_start"] + 1e-8:
            g = gates["perturb_and_recovery"]
            checks_by_phase["perturb_recovery"].append({"time_s": t, "checks": {
                "global_drift": drift <= g["global_initial_COM_displacement_max_m"],
                "global_rotation": angle <= g["global_raw_SO3_max_rad"],
                "depth": force["maximum_jaw_penetration_m"] <= g["all_jaw_penetration_max_m"],
                "normal_force_bounded": max(Fn) <= g["each_jaw_Fn_max_n"],
                "no_other_support": force["floor_other_loaded_contacts"] == 0}})
            if t >= times["opening_start"] - g["recovery_final_window_s"] - 1e-8:
                checks_by_phase["recovery"].append({"time_s": t, "checks": {
                    "pre_pulse_reference_present": pre is not None,
                    "pre_pulse_pose": pre is not None and np.linalg.norm(np.asarray(s["COM_world_m"]) - pre["COM_world_m"]) <= g["recovery_pose_error_vs_fixed_pre_pulse_pose_m_max"],
                    "pre_pulse_rotation": pre is not None and rotation_distance(pre["R_world"], s["R_world"]) <= g["recovery_raw_rotation_error_rad_max"],
                    "speed": speed <= g["recovery_speed_max_m_s"], "omega": omega <= g["recovery_omega_max_rad_s"],
                    "upward_support": g["recovery_Fz_weight_ratio"][0] <= ratio <= g["recovery_Fz_weight_ratio"][1],
                    "both_jaws_loaded": min(Fn) > settings["acceptance"]["hold_every_sample_in_final_half_second"]["each_jaw_loaded_Fn_min_n"],
                    "no_other_support": force["floor_other_loaded_contacts"] == 0}})
        if t > times["opening_end"] + 1e-8:
            g = gates["release_and_free_fall"]
            n = int(round((t - times["opening_end"]) / timestep_s))
            predicted_z = predicted_v = None
            if released:
                predicted_v = released["COM_velocity_world_m_s"][2] + n * timestep_s * gravity
                predicted_z = released["COM_world_m"][2] + n * timestep_s * released["COM_velocity_world_m_s"][2] + gravity * timestep_s ** 2 * n * (n + 1) / 2
            checks_by_phase["free_fall"].append({"time_s": t, "predicted_z_m": predicted_z, "predicted_vz_m_s": predicted_v, "checks": {
                "release_reference_present": released is not None,
                "no_jaw_contacts": all(f["all_jaw_geometric_contacts"] == 0 for f in (force, row["post_refresh_force"])),
                "no_jaw_force": all(f["maximum_absolute_jaw_contact_force_n"] <= g["all_jaw_force_abs_max_n"] for f in (force, row["post_refresh_force"])),
                "no_other_contacts": all(f["floor_other_geometric_contacts"] == 0 for f in (force, row["post_refresh_force"])),
                "no_external_test_force": np.max(np.abs(row["command"]["external_force_world_n"])) == 0.,
                "no_external_test_torque": np.max(np.abs(row["command"]["external_torque_world_nm"])) == 0.,
                "actual_external_wrenches_zero": np.max(np.abs(s["xfrc_applied"])) == 0.,
                "actual_generalized_force_zero": np.max(np.abs(s["qfrc_applied"])) == 0.,
                "gravity_position": released is not None and abs(s["COM_world_m"][2] - predicted_z) <= g["COM_vertical_prediction_error_max_m"],
                "gravity_velocity": released is not None and abs(s["COM_velocity_world_m_s"][2] - predicted_v) <= g["vertical_velocity_prediction_error_max_m_s"]}})
    # NumPy comparisons can produce np.bool_. Keep the declared Boolean
    # result exactly, but make every persisted gate a native JSON Boolean.
    for rows in checks_by_phase.values():
        for row in rows:
            row["checks"] = {name: bool(value) for name, value in row["checks"].items()}
    phase_passed = {phase: bool(rows) and all(all(r["checks"].values()) for r in rows)
                    for phase, rows in checks_by_phase.items()}
    numeric = all(r["finite"] for r in records)
    cadence = len(records) == round(times["end"] / timestep_s) and all(
        abs(r["state"]["time_s"] - (i + 1) * timestep_s) <= 1e-8 for i, r in enumerate(records))
    no_warnings = all(not any(r["integration_solver"]["warnings"]) for r in records)
    solver_budget_ok = all(r["integration_solver"]["termination"]["split_valid"] and
                          not r["integration_solver"]["termination"]["budget_or_tolerance_failure"] for r in records)
    solver_residuals_logged = bool(records) and all(residual_evidence_complete(r["integration_solver"]) for r in records)
    common = dict(complete=complete, numeric_finite=numeric, cadence_complete=cadence,
                  no_solver_warnings=no_warnings, solver_budget_ok=solver_budget_ok,
                  solver_residuals_logged=solver_residuals_logged, video_complete=video_complete)
    return {"passed": all(phase_passed.values()) and all(common.values()), "fixture_only": True,
        "phase_passed": phase_passed, "common_checks": common, "checks_by_phase": checks_by_phase,
        "fixed_initial_reference": initial, "fixed_pre_pulse_reference": pre, "actual_release_reference": released,
        "no_G1_task_or_hardware_promotion": True}


def convergence_report(first, second, settings):
    """Compare only real common timestamps; never interpolate or splice trials."""
    fine = {round(r["state"]["time_s"], 9): r for r in second}
    matches = [(r, fine[round(r["state"]["time_s"], 9)]) for r in first
               if round(r["state"]["time_s"], 9) in fine]
    times = phase_boundaries(settings)
    stable = {"hold": [(a, b) for a, b in matches if .5 - 1e-8 <= a["state"]["time_s"] <= times["pulse_start"] + 1e-8],
        "recovery": [(a, b) for a, b in matches if times["opening_start"] - settings["acceptance"]["perturb_and_recovery"]["recovery_final_window_s"] - 1e-8
                     <= a["state"]["time_s"] <= times["opening_start"] + 1e-8]}
    def differences(rows):
        return {"maximum_COM_difference_m": max((float(np.linalg.norm(np.asarray(a["state"]["COM_world_m"]) - b["state"]["COM_world_m"])) for a, b in rows), default=None),
                "maximum_raw_SO3_difference_rad": max((rotation_distance(a["state"]["R_world"], b["state"]["R_world"]) for a, b in rows), default=None),
                "mean_hand_Fz_difference_n": abs(float(np.mean([a["integration_force"]["jaw_force_world_n"][2] - b["integration_force"]["jaw_force_world_n"][2] for a, b in rows]))) if rows else None}
    all_diff = differences(matches)
    stable_diff = {name: differences(rows) for name, rows in stable.items()}
    def first_transition(rows):
        return next((r["state"]["time_s"] for r in rows if min(r["integration_force"]["jaw_normal_force_n"].values()) <= settings["acceptance"]["hold_every_sample_in_final_half_second"]["each_jaw_loaded_Fn_min_n"]), times["end"])
    transition = abs(first_transition(first) - first_transition(second))
    gates = settings["convergence"]
    checks = {"all_coarse_times_matched": len(matches) == len(first) and bool(first),
        "entire_COM": all_diff["maximum_COM_difference_m"] is not None and all_diff["maximum_COM_difference_m"] <= gates["entire_trial"]["maximum_COM_difference_m"],
        "entire_R": all_diff["maximum_raw_SO3_difference_rad"] is not None and all_diff["maximum_raw_SO3_difference_rad"] <= gates["entire_trial"]["maximum_raw_SO3_difference_rad"],
        "contact_transition": transition <= gates["entire_trial"]["first_contact_load_transition_time_difference_max_s"]}
    for name, diff in stable_diff.items():
        checks[name + ":COM"] = diff["maximum_COM_difference_m"] is not None and diff["maximum_COM_difference_m"] <= gates["stable_score_windows"]["maximum_COM_difference_m"]
        checks[name + ":R"] = diff["maximum_raw_SO3_difference_rad"] is not None and diff["maximum_raw_SO3_difference_rad"] <= gates["stable_score_windows"]["maximum_raw_SO3_difference_rad"]
        checks[name + ":Fz"] = diff["mean_hand_Fz_difference_n"] is not None and diff["mean_hand_Fz_difference_n"] <= gates["stable_score_windows"]["mean_hand_Fz_difference_n_max"]
    return {"passed": all(checks.values()), "checks": checks, "common_timestamps": len(matches),
        "stable_timestamps": {name: len(rows) for name, rows in stable.items()}, "entire_trial": all_diff, "stable_windows": stable_diff,
        "first_contact_load_transition_difference_s": transition}


def _trial(config, trial_dir, backend, dt, api, floor_contact):
    from .teacher_video import TeacherDiagnosticVideo
    settings = config["contact_calibration"]
    trial_dir.mkdir(parents=True, exist_ok=False)
    xml, manifest = build_fixture_xml(settings, backend, dt, floor_contact=floor_contact)
    scene = trial_dir / "fixture.xml"
    scene.write_bytes(xml)
    model = api.MjModel.from_xml_path(str(scene))
    data = api.MjData(model)
    cube = api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, "calibration_cube")
    if model.nq != 7 or model.nv != 6 or model.neq != 0 or model.body_mocapid[cube] != -1:
        raise CalibrationError("benchmark cube must have only its independent free joint and no constraints")
    api.mj_forward(model, data)
    initial = _state(model, data, cube, api)
    manifest["initial_state"] = initial
    manifest["compiled_options"] = {key: float(getattr(model.opt, key)) for key in
        ("timestep", "impratio", "iterations", "tolerance", "noslip_iterations", "noslip_tolerance", "cone", "solver", "integrator")}
    manifest["fixture_identity_sha256"] = canonical_sha256(manifest)
    write_new_json(trial_dir / "manifest.json", manifest)
    video_config = copy.deepcopy(config)
    video_config["control"].update(physics_dt_s=dt, physics_substeps=int(round(settings["video_period_s"] / dt)))
    video = TeacherDiagnosticVideo(model, video_config, trial_dir, mujoco_api=api)
    records, error = [], None
    total_steps = int(round(phase_boundaries(settings)["end"] / dt))
    video_stride = int(round(settings["video_period_s"] / dt))
    jaw_ids = [api.mj_name2id(model, api.mjtObj.mjOBJ_BODY, n) for n in ("left_jaw", "right_jaw")]
    mocap_ids = [int(model.body_mocapid[i]) for i in jaw_ids]
    raw_path = trial_dir / "raw_steps.jsonl"
    try:
        with raw_path.open("x") as stream:
            for index in range(total_steps):
                command = fixture_command(settings, index * dt, dt)
                if index % video_stride == 0:
                    video.capture(data, command["phase"])
                data.mocap_pos[mocap_ids] = command["jaw_positions_m"]
                data.mocap_quat[mocap_ids] = command["jaw_quaternions_wxyz"]
                data.xfrc_applied[:] = 0.
                data.xfrc_applied[cube, :3] = command["external_force_world_n"]
                before = _state(model, data, cube, api)
                api.mj_step(model, data)  # Only an explicit ROOT-run experiment invokes this function.
                integration_force = contact_wrench(model, data, cube, np.asarray(before["COM_world_m"]), api)
                integration_solver = _solver(model, data, api)
                api.mj_forward(model, data)
                state = _state(model, data, cube, api)
                refreshed = contact_wrench(model, data, cube, np.asarray(state["COM_world_m"]), api)
                mass = settings["cube_mass_kg"]
                impulse_residual = mass * (np.asarray(state["COM_velocity_world_m_s"]) - before["COM_velocity_world_m_s"]) / dt - np.asarray(integration_force["all_force_world_n"]) - np.asarray(command["external_force_world_n"]) - mass * np.asarray(model.opt.gravity)
                average_v = (np.asarray(state["COM_velocity_world_m_s"]) + before["COM_velocity_world_m_s"]) / 2
                average_w = (np.asarray(state["omega_world_rad_s"]) + before["omega_world_rad_s"]) / 2
                contact_work = dt * (np.asarray(integration_force["all_force_world_n"]) @ average_v
                    + np.asarray(integration_force["all_torque_about_COM_world_nm"]) @ average_w)
                test_work = dt * float(np.asarray(command["external_force_world_n"]) @ average_v)
                energy_delta = state["kinetic_energy_j"] + state["potential_energy_j"] - before["kinetic_energy_j"] - before["potential_energy_j"]
                row = {"step_index": index, "command": command, "integration_input_state": before, "state": state,
                    "integration_force": integration_force, "integration_solver": integration_solver,
                    "post_refresh_force": refreshed,
                    "integration_force_scope": "mj_step solver output before any refresh; geometry/COM reference is step input",
                    "post_refresh_force_scope": "new mj_forward under held test command; not preceding integrated force",
                    "translational_impulse_residual_n": impulse_residual.tolist(),
                    "energy_diagnostic": {"mechanical_energy_change_j": energy_delta,
                        "contact_work_midvelocity_j": float(contact_work), "test_load_work_midvelocity_j": test_work,
                        "discrete_work_residual_j": float(energy_delta - contact_work - test_work),
                        "scope": "trapezoidal-velocity work diagnostic, not exact energy conservation of a soft semi-implicit contact step"},
                    "finite": bool(np.isfinite(np.asarray(state["qpos"] + state["qvel"] + integration_force["all_force_world_n"] + integration_force["all_torque_about_COM_world_nm"])).all())}
                stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                records.append(row)
                if not row["finite"]:
                    raise CalibrationError("nonfinite state/force; remaining fixed schedule not executed")
            if total_steps % video_stride == 0:
                video.capture(data, "finished")
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    finally:
        data.xfrc_applied[:] = 0.
        video_report = video.finish()
    score = score_trial(records, initial, settings, dt, complete=len(records) == total_steps and error is None,
                        video_complete=video_report.get("complete") is True)
    arrays_path = trial_dir / "raw_state_arrays.npz"
    with arrays_path.open("xb") as handle:
        np.savez_compressed(handle, schema_version=np.asarray("independent_contact_fixture_v1"),
            fixture_only=np.asarray(True), training_eligible=np.asarray(False),
            time_s=np.asarray([r["state"]["time_s"] for r in records]),
            qpos=np.asarray([r["state"]["qpos"] for r in records]),
            qvel=np.asarray([r["state"]["qvel"] for r in records]),
            COM_world_m=np.asarray([r["state"]["COM_world_m"] for r in records]),
            R_world=np.asarray([r["state"]["R_world"] for r in records]),
            external_force_world_n=np.asarray([r["command"]["external_force_world_n"] for r in records]),
            raw_steps_sha256=np.asarray(_hash_file(raw_path)))
    report = {"backend": backend, "physics_dt_s": dt, "fixture_only": True, "training_eligible": False,
        "manifest": manifest, "error": error, "score": score, "raw_steps": str(raw_path),
        "raw_steps_sha256": _hash_file(raw_path), "actual_steps": len(records), "expected_steps": total_steps,
        "raw_state_arrays": str(arrays_path), "raw_state_arrays_sha256": _hash_file(arrays_path),
        "diagnostic_video": video_report, "no_state_reset_or_attachment": True}
    write_new_json(trial_dir / "report.json", report)
    return report, records


def compiled_invariants(model):
    """Hash all exported numeric model arrays and every numeric solver option."""
    arrays, options = {}, {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            arrays[name] = {"shape": list(value.shape), "dtype": value.dtype.str,
                            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    for name in dir(model.opt):
        if name.startswith("_") or name == "noslip_iterations":
            continue
        value = getattr(model.opt, name)
        if isinstance(value, np.ndarray):
            options[name] = value.tolist()
        elif not callable(value) and not isinstance(value, str):
            try:
                options[name] = float(value)
            except (TypeError, ValueError):
                pass
    return {"numeric_arrays": arrays, "solver_options_except_noslip_iterations": options}


def run_contact_calibration(config, run_dir):
    """Four fixed, independently scored trials; never auto-promote a task model."""
    if config["train"]["enabled"] or config.get("wm", {}).get("enabled"):
        raise CalibrationError("contact calibration is diagnostic only and requires training disabled")
    import mujoco
    from .sim.scene import _contact_values
    source = Path(config["scene"]["asset_path"])
    if not source.is_absolute():
        source = Path(config["paths"]["root"]) / source
    table = ET.parse(source).getroot().find(".//geom[@name='calibration_table']")
    if table is None:
        raise CalibrationError("source table material is required for the catch floor")
    values = _contact_values(table)
    floor_contact = {key: values[key] for key in ("condim", "friction", "solref", "solimp", "margin", "gap")}
    run_dir = Path(run_dir)
    trials, raw = [], {}
    parity = []
    for dt in config["contact_calibration"]["physics_timesteps_s"]:
        invariants = []
        for backend in config["contact_calibration"]["solver_backends"]:
            xml, _ = build_fixture_xml(config["contact_calibration"], backend, dt, floor_contact=floor_contact)
            model = mujoco.MjModel.from_xml_string(xml.decode())
            expected = 0 if backend == "source_default_v1" else 10
            if model.opt.noslip_iterations != expected:
                raise CalibrationError("compiled candidate does not match its declared Noslip count")
            invariants.append(compiled_invariants(model))
        equal = canonical_sha256(invariants[0]) == canonical_sha256(invariants[1])
        parity.append({"physics_dt_s": dt, "only_delta_noslip0_to10": equal,
                       "compiled_invariant_sha256": [canonical_sha256(value) for value in invariants],
                       "numeric_array_count": len(invariants[0]["numeric_arrays"])})
        if not equal:
            raise CalibrationError("paired models differ beyond the declared Noslip count")
    for dt in config["contact_calibration"]["physics_timesteps_s"]:
        for backend in config["contact_calibration"]["solver_backends"]:
            name = backend + "-dt" + str(round(dt * 1e6)) + "us"
            report, records = _trial(config, run_dir / "calibration" / name, backend, dt, mujoco, floor_contact)
            trials.append(report)
            raw[(backend, dt)] = records
    settings = config["contact_calibration"]
    convergence = {backend: convergence_report(raw[(backend, .002)], raw[(backend, .001)], settings)
                   for backend in settings["solver_backends"]}
    candidate = "source_default_noslip10_v1"
    admitted = all(t["score"]["passed"] for t in trials if t["backend"] == candidate) and convergence[candidate]["passed"]
    complete = all(all(t["score"]["common_checks"].values()) for t in trials)
    original = [t for t in trials if t["backend"] == "source_default_v1"]
    baseline_classification_agrees = original[0]["score"]["phase_passed"] == original[1]["score"]["phase_passed"]
    result = {"passed": bool(complete and admitted and baseline_classification_agrees), "fixtureonly": True, "training_started": False,
        "training_eligible": False, "optimizer_steps": 0, "variants": trials, "convergence": convergence,
        "calibrated_candidate": candidate if admitted and complete and baseline_classification_agrees else None,
        "compiled_pair_parity": parity, "baseline_phase_classification_agrees_across_dt": baseline_classification_agrees,
        "candidate_scope": "predeclared independent numerical bench only; not hardware calibration or G1 validation",
        "original_baseline_may_fail_holding_but_must_be_complete": True,
        "no_15_19_or_original20_promotion": True, "source_material_sha256": _hash_file(source)}
    write_new_json(run_dir / "reports" / "contact_calibration.json", result)
    return result
