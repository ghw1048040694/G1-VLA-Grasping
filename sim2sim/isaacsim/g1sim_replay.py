"""Import G1 MJCF into Isaac Sim and replay one successful MPC trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    return parser.parse_known_args()[0]


ARGS = parse_args()

from isaacsim import SimulationApp


simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": 1280,
        "height": 720,
        "renderer": "RaytracedLighting",
    }
)

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics


def enable_required_extensions() -> None:
    manager = omni.kit.app.get_app().get_extension_manager()
    for extension in (
        "omni.scene.optimizer.core",
        "isaacsim.robot.schema",
        "isaacsim.core.experimental.prims",
    ):
        manager.set_extension_enabled_immediate(extension, True)


def find_articulation_root() -> str:
    stage = omni.usd.get_context().get_stage()
    roots = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if len(roots) != 1:
        raise RuntimeError(f"Expected one articulation root, found {roots}")
    return roots[0]


def select_physx_variant() -> str:
    stage = omni.usd.get_context().get_stage()
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError("Converted USD has no default prim")
    variants = default_prim.GetVariantSet("Physics")
    names = list(variants.GetVariantNames())
    match = next((name for name in names if name.lower() == "physx"), None)
    if match is None:
        raise RuntimeError(f"Converted USD has no PhysX variant: {names}")
    variants.SetVariantSelection(match)
    simulation_app.update()
    return default_prim.GetPath().pathString


def add_light() -> None:
    stage = omni.usd.get_context().get_stage()
    light = UsdLux.DistantLight.Define(stage, "/Sim2SimKeyLight")
    light.CreateIntensityAttr(2500.0)
    light.CreateAngleAttr(0.5)


def world_position(stage, prim_path: str) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing scene prim: {prim_path}")
    transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.asarray(transform.ExtractTranslation(), dtype=np.float64)


def world_z(stage, prim_path: str) -> float:
    return float(world_position(stage, prim_path)[2])


def show_tote_markers(stage, tote_path: str) -> list[str]:
    marker_names = (
        "tote_left_grasp_site",
        "tote_right_grasp_site",
        "tote_left_assist_site",
        "tote_right_assist_site",
    )
    visible = []
    for name in marker_names:
        prim = stage.GetPrimAtPath(f"{tote_path}/{name}")
        if not prim.IsValid():
            continue
        UsdGeom.Imageable(prim).GetPurposeAttr().Set(UsdGeom.Tokens.default_)
        visible.append(name)
    return visible


def assisted_grasp_joint_paths(
    stage, default_prim_path: str, joint_names: list[str] | None = None
) -> list[str]:
    names = joint_names or ["left_assisted_grasp", "right_assisted_grasp"]
    paths = [f"{default_prim_path}/Physics/{name}" for name in names]
    missing = [path for path in paths if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        raise RuntimeError(f"Missing imported assisted-grasp joints: {missing}")
    return paths


def set_assisted_grasp_enabled(stage, joint_paths: list[str], enabled: bool) -> None:
    for path in joint_paths:
        attribute = stage.GetPrimAtPath(path).GetAttribute("physics:jointEnabled")
        if not attribute.IsValid():
            raise RuntimeError(f"Joint has no physics:jointEnabled attribute: {path}")
        if not attribute.Set(enabled):
            raise RuntimeError(f"Failed to set physics:jointEnabled={enabled}: {path}")


def main() -> None:
    enable_required_extensions()
    manifest = json.loads((ARGS.root / "manifest.json").read_text(encoding="utf-8"))
    mjcf_path = ARGS.root / manifest.get(
        "mjcf_file", "assets/g1_mpc04_episode0000.xml"
    )
    usd_dir = ARGS.root / "usd"
    usd_dir.mkdir(parents=True, exist_ok=True)

    config = MJCFImporterConfig(
        mjcf_path=str(mjcf_path),
        usd_path=str(usd_dir),
        import_scene=True,
        merge_mesh=False,
        allow_self_collision=False,
        fix_base=True,
        override_gain_type="fixed",
        override_bias_type="affine",
        override_gain_prm=[60.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        override_bias_prm=[0.0, -60.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
    )
    output_path = MJCFImporter(config).import_mjcf()
    opened, stage = stage_utils.open_stage(output_path)
    if not opened or stage is None:
        raise RuntimeError(f"Failed to open converted USD: {output_path}")
    simulation_app.update()
    default_prim_path = select_physx_variant()
    root_path = find_articulation_root()
    tracked_object_name = manifest.get("tracked_object_name", "warehouse_tote")
    tracked_object_path = f"{default_prim_path}/Geometry/{tracked_object_name}"
    visible_tote_markers = (
        show_tote_markers(stage, tracked_object_path)
        if tracked_object_name == "warehouse_tote"
        else []
    )
    grasp_joint_paths = assisted_grasp_joint_paths(
        stage,
        default_prim_path,
        manifest.get("assisted_grasp_joint_names"),
    )
    set_assisted_grasp_enabled(stage, grasp_joint_paths, False)
    stage.GetRootLayer().Save()

    add_light()
    set_camera_view(
        eye=np.array([2.5, 2.2, 1.8]),
        target=np.array([0.35, 0.0, 1.0]),
        camera_prim_path="/OmniverseKit_Persp",
    )

    SimulationManager.set_physics_sim_device("cpu")
    SimulationManager.set_physics_dt(1.0 / manifest["physics_fps"])
    simulation_app.update()

    robot = Articulation(root_path)
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    simulation_app.update()

    dof_names = list(robot.dof_names)
    action_names = manifest["action_joint_names"]
    missing = [name for name in action_names if name not in dof_names]
    report = {
        "experiment": manifest["experiment"],
        "usd_path": str(output_path),
        "default_prim": default_prim_path,
        "articulation_root": root_path,
        "isaac_dof_count": len(dof_names),
        "action_dof_count": len(action_names),
        "missing_action_joints": missing,
        "joint_contract_passed": not missing,
        "visible_tote_markers": visible_tote_markers,
        "tracked_object_name": tracked_object_name,
        "imported_assisted_grasp_joints": grasp_joint_paths,
        "activate_assisted_grasp": bool(
            manifest.get("activate_assisted_grasp", False)
        ),
        "mode": "inspect_only" if ARGS.inspect_only else "offline_action_replay",
    }
    if missing:
        raise RuntimeError(f"Isaac Sim is missing action joints: {missing}")

    indices = [dof_names.index(name) for name in action_names]
    trajectory = np.load(ARGS.root / "data/trajectory.npz")
    commands = trajectory["action_joint_position_rad"].astype(np.float32)
    if commands.shape[1] != len(action_names):
        raise ValueError(
            f"Trajectory has {commands.shape[1]} actions, expected {len(action_names)}"
        )
    phases = (
        trajectory["scheduled_phase"].astype(np.int64)
        if "scheduled_phase" in trajectory.files
        else None
    )
    if phases is not None and phases.shape != (len(commands),):
        raise ValueError(f"Unexpected scheduled_phase shape: {phases.shape}")
    assist_active = (
        trajectory["assist_active"].astype(bool)
        if "assist_active" in trajectory.files
        else None
    )
    if assist_active is not None and assist_active.shape != (len(commands),):
        raise ValueError(f"Unexpected assist_active shape: {assist_active.shape}")

    robot.set_dof_positions(commands[0][None, :], dof_indices=indices)
    robot.set_dof_position_targets(commands[0][None, :], dof_indices=indices)
    simulation_app.update()

    if not ARGS.inspect_only:
        substeps = manifest["physics_fps"] // manifest["control_fps"]
        for _ in range(round(ARGS.warmup_seconds * manifest["physics_fps"])):
            simulation_app.update()
        initial_tote_z = world_z(stage, tracked_object_path)
        actual = []
        tote_lifts = []
        tracked_positions = []
        grasp_activation_frame = None
        grasp_activation_phase = None
        grasp_activation_joint = None
        grasp_release_frame = None
        activation_phase = int(manifest.get("assisted_grasp_activation_phase", 4))
        for frame, command in enumerate(commands):
            if manifest.get("activate_assisted_grasp", False):
                if assist_active is not None:
                    if assist_active[frame] and grasp_activation_frame is None:
                        object_index = int(trajectory["grabbed_object_index"][frame])
                        arm_index = int(trajectory["grabbed_arm_index"][frame])
                        object_name = str(trajectory["object_names"][object_index])
                        side = "left" if arm_index == 0 else "right"
                        joint_name = f"{object_name}_{side}_assisted_grasp"
                        joint_path = f"{default_prim_path}/Physics/{joint_name}"
                        if joint_path not in grasp_joint_paths:
                            raise RuntimeError(
                                f"Trajectory requested unavailable joint: {joint_path}"
                            )
                        set_assisted_grasp_enabled(stage, [joint_path], True)
                        grasp_activation_frame = frame
                        grasp_activation_joint = joint_path
                    elif (
                        not assist_active[frame]
                        and grasp_activation_frame is not None
                        and grasp_release_frame is None
                    ):
                        set_assisted_grasp_enabled(stage, grasp_joint_paths, False)
                        grasp_release_frame = frame
                elif grasp_activation_frame is None and phases[frame] >= activation_phase:
                    set_assisted_grasp_enabled(stage, grasp_joint_paths, True)
                    grasp_activation_frame = frame
                    grasp_activation_phase = int(phases[frame])
            robot.set_dof_position_targets(command[None, :], dof_indices=indices)
            for _ in range(substeps):
                simulation_app.update()
            actual.append(robot.get_dof_positions().numpy()[0, indices])
            position = world_position(stage, tracked_object_path)
            tracked_positions.append(position)
            tote_lifts.append(float(position[2] - initial_tote_z))
        endpoint_lift = tote_lifts[-1]
        for _ in range(round(ARGS.hold_seconds * manifest["physics_fps"])):
            simulation_app.update()
            position = world_position(stage, tracked_object_path)
            tracked_positions.append(position)
            tote_lifts.append(float(position[2] - initial_tote_z))
        actual_array = np.asarray(actual, dtype=np.float32)
        error = actual_array - commands
        final_lift = tote_lifts[-1]
        report.update(
            {
                "frames": int(len(commands)),
                "duration_s": float(len(commands) / manifest["control_fps"]),
                "warmup_s": ARGS.warmup_seconds,
                "post_replay_hold_s": ARGS.hold_seconds,
                "joint_tracking_rmse_rad": float(np.sqrt(np.mean(error**2))),
                "joint_tracking_max_abs_rad": float(np.max(np.abs(error))),
                "tote_endpoint_lift_height_m": endpoint_lift,
                "tote_final_lift_height_m": final_lift,
                "tote_maximum_lift_height_m": max(tote_lifts),
                "assisted_grasp_activation_frame": grasp_activation_frame,
                "assisted_grasp_activation_phase": grasp_activation_phase,
                "assisted_grasp_activation_joint": grasp_activation_joint,
                "assisted_grasp_release_frame": grasp_release_frame,
                "required_lift_height_m": 0.10,
                "task_transfer_passed": final_lift >= 0.10,
                "replay_completed": True,
            }
        )
        if manifest.get("task_type") == "language_object_to_box":
            goal = np.asarray(manifest["goal_position_m"], dtype=np.float64)

            def inside_box(position: np.ndarray) -> bool:
                relative = position - goal
                return bool(
                    abs(relative[0]) <= 0.105
                    and abs(relative[1]) <= 0.125
                    and 0.75 <= position[2] <= 0.93
                )

            final_positions = {
                name: world_position(stage, f"{default_prim_path}/Geometry/{name}")
                for name in manifest["object_names"]
            }
            target_inside = inside_box(final_positions[tracked_object_name])
            wrong_objects_inside = [
                name
                for name, position in final_positions.items()
                if name != tracked_object_name and inside_box(position)
            ]
            report.update(
                {
                    "goal_position_m": goal.tolist(),
                    "tracked_object_final_position_m": final_positions[
                        tracked_object_name
                    ].tolist(),
                    "final_object_position_m": {
                        name: position.tolist()
                        for name, position in final_positions.items()
                    },
                    "target_in_box": target_inside,
                    "wrong_objects_in_box": wrong_objects_inside,
                    "task_transfer_passed": bool(
                        target_inside and not wrong_objects_inside
                    ),
                }
            )

    timeline.stop()
    report_path = ARGS.root / "isaacsim_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("G1SIM_REPORT=" + json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
