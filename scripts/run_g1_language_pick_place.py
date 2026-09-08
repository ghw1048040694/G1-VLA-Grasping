#!/usr/bin/env python3
"""Generate one language-conditioned G1 tabletop pick-and-place episode."""

from __future__ import annotations

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import find_objects, label
from scipy.spatial.transform import Rotation

from validate_g1_bimanual_actuation import (
    HAND_TARGETS,
    apply_regularized_dynamics,
    unitree_gains,
)

OBJECT_NAMES = ("red_triangle", "yellow_rod", "green_cube")
EXPERT_CONTRACT_VERSION = "g1lang_v4_contact_grasp"
OBJECT_LABELS = {
    "red_triangle": "red triangular prism",
    "yellow_rod": "yellow rod",
    "green_cube": "green cube",
}
INSTRUCTIONS = {
    name: f"put the {label} into the blue box" for name, label in OBJECT_LABELS.items()
}
PALM_NAMES = ("left_palm_center", "right_palm_center")
LEFT_PALM_INDEX = 0
RIGHT_PALM_INDEX = 1
HOME_POSITIONS_M = np.array(((0.22, 0.32, 1.00), (0.22, -0.32, 1.00)))
WAIST_PITCH_LIMIT_RAD = 0.10
TABLE_TOP_Z_M = 0.75
BIN_POSITION_M = np.array((0.59, 0.00, TABLE_TOP_Z_M + 0.01))
OBJECT_SLOTS_XY_M = (
    np.array((0.36, -0.34)),
    np.array((0.34, -0.10)),
    np.array((0.36, 0.20)),
)
# The palm site is near the finger tips in the G1 MJCF.  The fingers extend
# forward along +x, so the palm target must sit behind the object.  The old
# positive-x offsets placed the object inside the wrist collision mesh and made
# the assisted trajectory look like a magnetic attachment.
OBJECT_ASSIST_OFFSET_M = {
    "red_triangle": np.array((0.020, 0.0, 0.025)),
    "yellow_rod": np.array((0.005, 0.0, 0.020)),
    "green_cube": np.array((0.020, 0.0, 0.000)),
}
# RGB centroids are biased toward the visible front/top faces.  These small,
# fixed shape-calibration offsets are measured in the head-camera table frame.
OBJECT_RGB_LOCALIZATION_BIAS_M = {
    "red_triangle": np.array((0.007, 0.000, 0.0)),
    # The raw head-RGB ray estimate for the rod is already close to its
    # collision center; the previous +13 mm x bias put the fingers too far
    # forward and reduced the transport grasp margin.
    "yellow_rod": np.array((0.004, 0.004, 0.0)),
    "green_cube": np.array((0.001, 0.000, 0.0)),
}
OBJECT_HAND_CLOSURE_SCALE = {
    "red_triangle": {"thumb": 0.46, "index": 0.42, "middle": 0.32},
    "yellow_rod": {"thumb": 0.56, "index": 0.58, "middle": 0.58},
    "green_cube": {"thumb": 0.50, "index": 0.50, "middle": 0.50},
}
OBJECT_GRASP_YAW_DEG = {
    "red_triangle": 0.0,
    "yellow_rod": 0.0,
    "green_cube": 45.0,
}
OBJECT_MASS_KG = {
    "red_triangle": 0.12,
    "yellow_rod": 0.10,
    "green_cube": 0.08,
}
OBJECT_COLLISION_HALF_SIZE_M = {
    "red_triangle": (0.045, 0.040, 0.030),
    "yellow_rod": (0.060, 0.025, 0.025),
    "green_cube": (0.025, 0.025, 0.025),
}
# The torque-controlled arm settles above the nominal IK target while moving
# under gravity.  This correction is used only for the physical contact grasp;
# it was measured from the no-constraint trajectory and keeps the hand on the
# object instead of compensating with an equality constraint.
CONTACT_GRASP_COMPENSATION_M = np.array((-0.015, 0.030, -0.055))
HEAD_CAMERA_XYAXES = "0 -1 0 0.643 0 0.766"
HEAD_CAMERA_FOVY = "85"
PALM_ORIENTATION_TARGETS = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
OBJECT_SPAWN_Z_M = {
    "red_triangle": TABLE_TOP_Z_M + 0.030,
    "yellow_rod": TABLE_TOP_Z_M + 0.025,
    "green_cube": TABLE_TOP_Z_M + 0.045,
}


def smoothstep(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def object_hand_target(target_object: str, joint_name: str, closed: float) -> float:
    for finger, scale in OBJECT_HAND_CLOSURE_SCALE[target_object].items():
        if f"hand_{finger}_" in joint_name:
            return scale * closed
    raise ValueError(f"Unknown hand joint: {joint_name}")


def hand_profile_scale(joint_name: str, values: tuple[float, float, float]) -> float:
    """Return offline-teacher thumb/index/middle scaling for one hand joint."""
    if "hand_thumb_" in joint_name:
        return values[0]
    if "hand_index_" in joint_name:
        return values[1]
    if "hand_middle_" in joint_name:
        return values[2]
    raise ValueError(f"Unknown hand joint: {joint_name}")


def object_name(model: mujoco.MjModel, kind: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, kind, index) or f"unnamed_{index}"


def remove_old_task(root: ET.Element) -> None:
    for worldbody in root.findall("worldbody"):
        for body in list(worldbody.findall("body")):
            if body.get("name") in {"warehouse_tote", "blue_receptacle", *OBJECT_NAMES}:
                worldbody.remove(body)
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    for constraint in list(equality):
        name = constraint.get("name", "")
        if "assisted_grasp" in name:
            equality.remove(constraint)


def ensure_triangle_mesh(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name") == "red_triangle_prism_mesh":
            asset.remove(mesh)
    vertices = (
        (-0.045, -0.040, -0.030),
        (-0.045, 0.040, -0.030),
        (-0.045, 0.000, 0.045),
        (0.045, -0.040, -0.030),
        (0.045, 0.040, -0.030),
        (0.045, 0.000, 0.045),
    )
    faces = (
        (0, 2, 1),
        (3, 4, 5),
        (0, 1, 4),
        (0, 4, 3),
        (1, 2, 5),
        (1, 5, 4),
        (2, 0, 3),
        (2, 3, 5),
    )
    ET.SubElement(
        asset,
        "mesh",
        name="red_triangle_prism_mesh",
        vertex=" ".join(str(value) for vertex in vertices for value in vertex),
        face=" ".join(str(value) for face in faces for value in face),
    )


def add_free_object(
    scene: ET.Element,
    equality: ET.Element,
    name: str,
    position: np.ndarray,
    red_triangle_mesh_collision: bool = False,
) -> None:
    assist_offset = OBJECT_ASSIST_OFFSET_M[name]
    body = ET.SubElement(
        scene,
        "body",
        name=name,
        pos=" ".join(f"{value:.8f}" for value in position),
    )
    ET.SubElement(body, "freejoint", name=f"{name}_freejoint")
    if name == "red_triangle":
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_visual",
            type="mesh",
            mesh="red_triangle_prism_mesh",
            contype="0",
            conaffinity="0",
            density="0",
            rgba="0.90 0.08 0.06 1",
        )
        collision_attributes = {
            "name": f"{name}_collision",
            "mass": str(OBJECT_MASS_KG[name]),
            "friction": "1.4 0.02 0.002",
            "contype": "5",
            "conaffinity": "3",
            "rgba": "0.90 0.08 0.06 0",
        }
        if red_triangle_mesh_collision:
            collision_attributes.update(type="mesh", mesh="red_triangle_prism_mesh")
        else:
            collision_attributes.update(
                type="box",
                size=" ".join(
                    str(value) for value in OBJECT_COLLISION_HALF_SIZE_M[name]
                ),
            )
        ET.SubElement(body, "geom", **collision_attributes)
    elif name == "yellow_rod":
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_visual",
            type="cylinder",
            fromto="-0.06 0 0 0.06 0 0",
            size="0.025",
            contype="0",
            conaffinity="0",
            density="0",
            rgba="0.95 0.78 0.05 1",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_collision",
            type="box",
            size=" ".join(str(value) for value in OBJECT_COLLISION_HALF_SIZE_M[name]),
            mass=str(OBJECT_MASS_KG[name]),
            friction="1.4 0.02 0.01",
            contype="5",
            conaffinity="3",
            rgba="0.95 0.78 0.05 0",
        )
    else:
        ET.SubElement(
            body,
            "geom",
            name=f"{name}_geom",
            type="box",
            size=" ".join(str(value) for value in OBJECT_COLLISION_HALF_SIZE_M[name]),
            mass=str(OBJECT_MASS_KG[name]),
            friction="1.4 0.03 0.003",
            contype="5",
            conaffinity="3",
            rgba="0.08 0.70 0.20 1",
        )
    ET.SubElement(
        body,
        "site",
        name=f"{name}_center",
        type="sphere",
        pos="0 0 0",
        size="0.006",
        rgba="1 1 1 0.15",
    )
    ET.SubElement(
        body,
        "site",
        name=f"{name}_assist_site",
        type="sphere",
        pos=" ".join(f"{value:.6f}" for value in assist_offset),
        size="0.009",
        rgba="1 1 0 0.35",
    )
    # Do not add site-to-site equality constraints.  The strict evaluator
    # detects grasp from physical contacts; legacy MuJoCo site1/site2
    # constraints are intentionally absent so an XML-level attachment cannot
    # masquerade as learned grasping.


def add_blue_box(scene: ET.Element) -> None:
    box = ET.SubElement(
        scene,
        "body",
        name="blue_receptacle",
        pos=" ".join(f"{value:.8f}" for value in BIN_POSITION_M),
    )
    specs = {
        "bottom": ("0 0 0", "0.13 0.15 0.01"),
        # Keep the collision cavity consistent with the strict center-point
        # placement gate below.  The extra 1 cm gives the long yellow rod a
        # physically reachable margin without changing the task semantics.
        "front": ("-0.13 0 0.065", "0.01 0.15 0.065"),
        "back": ("0.13 0 0.065", "0.01 0.15 0.065"),
        "left": ("0 0.17 0.065", "0.13 0.01 0.065"),
        "right": ("0 -0.17 0.065", "0.13 0.01 0.065"),
    }
    for part, (pos, size) in specs.items():
        ET.SubElement(
            box,
            "geom",
            name=f"blue_box_{part}",
            type="box",
            pos=pos,
            size=size,
            friction="0.9 0.02 0.002",
            rgba="0.05 0.25 0.90 1",
        )
    ET.SubElement(
        box,
        "site",
        name="blue_box_drop_site",
        type="sphere",
        pos="0 0 0.19",
        size="0.012",
        rgba="0.10 0.85 1 0.35",
    )


def replace_hand_collision_hulls(root: ET.Element) -> None:
    """Use stable primitive hand contacts instead of oversized mesh hulls."""
    for side, sign in (("left", -1.0), ("right", 1.0)):
        body = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        if body is None:
            raise RuntimeError(f"Source scene has no {side} wrist yaw body")
        hand_bodies = [body]
        hand_bodies.extend(
            item
            for item in root.findall(".//body")
            if item.get("name", "").startswith(f"{side}_hand_")
        )
        for hand_body in hand_bodies:
            body_geoms = hand_body.findall("geom")
            for geom in body_geoms:
                # Dataset assets may already contain the generated contact
                # primitives.  Remove them before rebuilding the canonical
                # set so repeated evaluation remains valid XML.
                if geom.get("name", "").endswith("_palm_contact_pad") or geom.get(
                    "name", ""
                ).endswith("_contact"):
                    hand_body.remove(geom)
            body_geoms = hand_body.findall("geom")
            for geom in body_geoms:
                if geom.get("contype", "1") != "0":
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")
            if hand_body is not body and body_geoms:
                # Keep one convex collision hull aligned with each visible
                # finger link. Duplicate visual geoms remain non-colliding.
                body_geoms[0].set("contype", "2")
                body_geoms[0].set("conaffinity", "4")
                body_geoms[0].set("friction", "2.2 0.05 0.005")
        ET.SubElement(
            body,
            "geom",
            name=f"{side}_palm_contact_pad",
            type="box",
            pos=f"0.070 {-sign * 0.003:.3f} 0",
            size="0.025 0.040 0.040",
            friction="2.0 0.05 0.005",
            contype="2",
            conaffinity="4",
            rgba="0.7 0.7 0.7 0",
        )
        finger_specs = {
            f"{side}_hand_index_0_link": "0 0 0 0.046 0 0",
            f"{side}_hand_index_1_link": "0 0 0 0.050 0 0",
            f"{side}_hand_middle_0_link": "0 0 0 0.046 0 0",
            f"{side}_hand_middle_1_link": "0 0 0 0.050 0 0",
            f"{side}_hand_thumb_0_link": f"0 0 0 -0.003 {sign * 0.019:.3f} 0",
            f"{side}_hand_thumb_1_link": f"0 0 0 0 {sign * 0.046:.3f} 0",
            f"{side}_hand_thumb_2_link": f"0 0 0 0 {sign * 0.040:.3f} 0",
        }
        for body_name, fromto in finger_specs.items():
            finger_body = root.find(f".//body[@name='{body_name}']")
            if finger_body is None:
                raise RuntimeError(f"Source scene has no {body_name}")
            ET.SubElement(
                finger_body,
                "geom",
                name=f"{body_name}_contact",
                type="capsule",
                fromto=fromto,
                # A wider fingertip pad gives the free object a stable
                # numerical contact patch without adding any constraint.
                size="0.014",
                friction="2.2 0.05 0.005",
                contype="2",
                conaffinity="4",
                rgba="0.7 0.7 0.7 0",
            )


def build_scene(
    source: Path,
    destination: Path,
    positions: dict[str, np.ndarray],
    red_triangle_mesh_collision: bool = False,
) -> None:
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("autolimits", "true")
    # MuJoCo 2.x accepted actuatorfrcrange on joint elements; MuJoCo 3.x
    # rejects this legacy attribute.  Force limits remain defined on the
    # actuator elements and are therefore preserved by removing only this
    # unsupported XML attribute.
    for joint in root.iter("joint"):
        joint.attrib.pop("actuatorfrcrange", None)
        joint.attrib.pop("actuatorfrclimited", None)
    remove_old_task(root)
    ensure_triangle_mesh(root)
    scene = root.findall("worldbody")[-1]
    equality = root.find("equality")
    if equality is None:
        raise RuntimeError("Failed to create equality section")
    table = root.find(".//geom[@name='calibration_table']")
    if table is None:
        raise RuntimeError("Source scene has no calibration table")
    table.set("pos", "0.52 0 0.72")
    table.set("size", "0.42 0.55 0.03")
    head_camera = root.find(".//camera[@name='head_camera']")
    if head_camera is None:
        raise RuntimeError("Source scene has no head_camera")
    if os.environ.get("G1_PRESERVE_SOURCE_CAMERA") != "1":
        head_camera.set("xyaxes", HEAD_CAMERA_XYAXES)
        head_camera.set("fovy", HEAD_CAMERA_FOVY)
    replace_hand_collision_hulls(root)
    add_blue_box(scene)
    for name in OBJECT_NAMES:
        add_free_object(
            scene,
            equality,
            name,
            positions[name],
            red_triangle_mesh_collision=red_triangle_mesh_collision,
        )
    tree.write(destination, encoding="unicode", xml_declaration=False)


def controlled_joints(model: mujoco.MjModel) -> dict[str, dict[str, int]]:
    controlled = {}
    for actuator_id in range(model.nu):
        name = object_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            controlled[name] = {
                "actuator_id": actuator_id,
                "joint_id": joint_id,
                "qpos_id": int(model.jnt_qposadr[joint_id]),
                "qvel_id": int(model.jnt_dofadr[joint_id]),
            }
    return controlled


def arm_ik_contract(
    model: mujoco.MjModel,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = []
    joint_ids = []
    for joint_id in range(model.njnt):
        name = object_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name.startswith("waist_") or (
            (name.startswith("left_") or name.startswith("right_"))
            and any(part in name for part in ("shoulder_", "elbow_", "wrist_"))
        ):
            names.append(name)
            joint_ids.append(joint_id)
    qpos_ids = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids])
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids])
    for index, name in enumerate(names):
        if name == "waist_pitch_joint":
            lower[index], upper[index] = -WAIST_PITCH_LIMIT_RAD, WAIST_PITCH_LIMIT_RAD
        elif name in ("waist_yaw_joint", "waist_roll_joint"):
            lower[index], upper[index] = -0.05, 0.05
    return names, qpos_ids, lower, upper


def solve_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    palm_ids: list[int],
    qpos_ids: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_positions: np.ndarray,
    seed: np.ndarray,
    orientation_weight: float = 0.04,
) -> tuple[np.ndarray, dict]:
    bounded_seed = np.clip(seed, lower + 1e-6, upper - 1e-6)
    seed_was_clipped = bool(np.any(bounded_seed != seed))

    def residual(values: np.ndarray) -> np.ndarray:
        data.qpos[qpos_ids] = values
        mujoco.mj_forward(model, data)
        position = np.concatenate(
            [
                data.site_xpos[site] - target
                for site, target in zip(palm_ids, target_positions)
            ]
        )
        orientation = np.concatenate(
            [
                Rotation.from_matrix(
                    PALM_ORIENTATION_TARGETS[index].T
                    @ data.site_xmat[site].reshape(3, 3)
                ).as_rotvec()
                for index, site in enumerate(palm_ids)
            ]
        )
        return np.concatenate(
            (position, orientation_weight * orientation, 0.003 * values)
        )

    solution = least_squares(
        residual,
        bounded_seed,
        bounds=(lower, upper),
        max_nfev=1200,
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
    )
    data.qpos[qpos_ids] = solution.x
    mujoco.mj_forward(model, data)
    errors = [
        float(np.linalg.norm(data.site_xpos[site] - target))
        for site, target in zip(palm_ids, target_positions)
    ]
    maximum_error = max(errors)
    return solution.x.copy(), {
        "solver_success": bool(maximum_error <= 0.08),
        "optimizer_converged": bool(solution.success),
        "seed_was_clipped": seed_was_clipped,
        "maximum_position_error_m": maximum_error,
        "cost": float(solution.cost),
    }


def interpolate(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = smoothstep(alpha)
    return (1.0 - alpha) * start + alpha * end


def transport_interpolate(
    lift: np.ndarray,
    high: np.ndarray | None,
    place: np.ndarray,
    alpha: float,
    high_fraction: float,
) -> np.ndarray:
    if high is None:
        return interpolate(lift, place, alpha)
    if alpha < high_fraction:
        return interpolate(lift, high, alpha / high_fraction)
    return interpolate(
        high,
        place,
        (alpha - high_fraction) / (1.0 - high_fraction),
    )


def _largest_color_component(mask: np.ndarray) -> dict:
    labels, count = label(mask)
    if count == 0:
        return {"pixel_count": 0, "centroid_xy": None, "bbox_xyxy": None}
    slices = find_objects(labels)
    best_label = 0
    best_count = 0
    for component_label, component_slice in enumerate(slices, start=1):
        if component_slice is None:
            continue
        component_count = int(
            np.count_nonzero(labels[component_slice] == component_label)
        )
        if component_count > best_count:
            best_label = component_label
            best_count = component_count
    if best_label == 0:
        return {"pixel_count": 0, "centroid_xy": None, "bbox_xyxy": None}
    ys, xs = np.where(labels == best_label)
    return {
        "pixel_count": int(best_count),
        "centroid_xy": [float(xs.mean()), float(ys.mean())],
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
    }


def audit_head_camera_visibility(
    renderer: mujoco.Renderer, data: mujoco.MjData
) -> dict:
    """Audit RGB visibility at the initial pose; this is never used for control."""
    renderer.update_scene(data, camera="head_camera")
    image = np.asarray(renderer.render()).copy().astype(np.float32)
    red = (
        (image[:, :, 0] > 100)
        & (image[:, :, 0] > 1.5 * image[:, :, 1])
        & (image[:, :, 0] > 1.3 * image[:, :, 2])
    )
    yellow = (
        (image[:, :, 0] > 120)
        & (image[:, :, 1] > 90)
        & (image[:, :, 2] < 130)
        & (image[:, :, 0] > 1.2 * image[:, :, 2])
    )
    green = (
        (image[:, :, 1] > 80)
        & (image[:, :, 1] > 1.3 * image[:, :, 0])
        & (image[:, :, 1] > 0.9 * image[:, :, 2])
    )
    detections = {
        "red_triangle": _largest_color_component(red),
        "yellow_rod": _largest_color_component(yellow),
        "green_cube": _largest_color_component(green),
    }
    minimum_pixels = {"red_triangle": 100, "yellow_rod": 300, "green_cube": 100}
    for name, detection in detections.items():
        detection["visible"] = detection["pixel_count"] >= minimum_pixels[name]
        detection["minimum_pixels"] = minimum_pixels[name]
    return {
        "camera": "head_camera",
        "image_height": int(image.shape[0]),
        "image_width": int(image.shape[1]),
        "detections": detections,
        "all_objects_visible": all(item["visible"] for item in detections.values()),
    }


def localize_objects_from_head_rgb(
    model: mujoco.MjModel, data: mujoco.MjData, visibility: dict
) -> dict[str, np.ndarray]:
    """Back-project RGB centroids onto each object's known resting-height plane."""
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, visibility["camera"]
    )
    if camera_id < 0:
        raise RuntimeError(f"Camera not found: {visibility['camera']}")
    height = visibility["image_height"]
    width = visibility["image_width"]
    focal_px = height / (2.0 * math.tan(math.radians(model.cam_fovy[camera_id]) / 2.0))
    camera_position = data.cam_xpos[camera_id].copy()
    camera_rotation = data.cam_xmat[camera_id].reshape(3, 3).copy()
    estimates = {}
    for name in OBJECT_NAMES:
        centroid = visibility["detections"][name]["centroid_xy"]
        if centroid is None:
            raise RuntimeError(f"Cannot localize invisible object: {name}")
        pixel_x, pixel_y = centroid
        ray_camera = np.array(
            (
                (pixel_x - width / 2.0) / focal_px,
                -(pixel_y - height / 2.0) / focal_px,
                -1.0,
            )
        )
        ray_world = camera_rotation @ ray_camera
        object_z = OBJECT_SPAWN_Z_M[name]
        if abs(float(ray_world[2])) < 1e-8:
            raise RuntimeError(f"Head-camera ray is parallel to the table for {name}")
        distance = (object_z - camera_position[2]) / ray_world[2]
        if distance <= 0:
            raise RuntimeError(f"Head-camera ray points away from {name}")
        estimates[name] = (
            camera_position
            + distance * ray_world
            + OBJECT_RGB_LOCALIZATION_BIAS_M[name]
        )
    return estimates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--initial-upper-state-npz",
        type=Path,
        help="Optional NPZ containing observation_joint_position_rad to initialize the expert from an on-policy state.",
    )
    parser.add_argument(
        "--initial-state-recovery-s",
        type=float,
        default=0.0,
        help=(
            "Before the strict expert begins, interpolate an on-policy arm state "
            "back to the canonical open-hand home pose over this many seconds."
        ),
    )
    parser.add_argument(
        "--initial-safe-lift-s",
        type=float,
        default=0.0,
        help="Raise both palms from an on-policy state before traversing toward home.",
    )
    parser.add_argument(
        "--initial-safe-traverse-s",
        type=float,
        default=0.0,
        help="Move both raised palms horizontally above home before descending.",
    )
    parser.add_argument(
        "--initial-safe-lift-height-m",
        type=float,
        default=0.18,
        help="Vertical clearance used by the on-policy recovery waypoint.",
    )
    parser.add_argument(
        "--initial-home-settle-s",
        type=float,
        default=0.0,
        help=(
            "Additional open-hand hold at the canonical home pose after an "
            "on-policy state recovery and before the expert task clock starts."
        ),
    )
    parser.add_argument(
        "--initial-direct-continuation",
        action="store_true",
        help=(
            "Offline recovery-teacher mode: continue directly from an observed "
            "policy state to the physical grasp pose instead of returning home."
        ),
    )
    parser.add_argument(
        "--initial-direct-grasp-s",
        type=float,
        default=0.9,
        help="Duration of the offline direct-continuation approach to the grasp pose.",
    )
    parser.add_argument(
        "--initial-velocity-mode",
        choices=("restore", "zero"),
        default="restore",
        help=(
            "Restore the velocity saved with an initial policy state, or zero it. "
            "Use zero when the deployment observation contains positions only."
        ),
    )
    parser.add_argument("--target-object", choices=OBJECT_NAMES, required=True)
    parser.add_argument(
        "--force-active-arm",
        choices=("auto", "left", "right"),
        default="auto",
        help=(
            "Offline expert-data option. Auto preserves the geometric arm "
            "selection; explicit values support physical reachability audits."
        ),
    )
    parser.add_argument(
        "--hand-profile-scale",
        default="1,1,1",
        help=(
            "Offline expert thumb,index,middle target multipliers. This only "
            "changes recorded physical teacher labels."
        ),
    )
    parser.add_argument("--slot-permutation", default="0,1,2")
    parser.add_argument("--slot-offsets", default="0,0,0,0,0,0")
    parser.add_argument("--language-instruction")
    parser.add_argument("--record-demonstration", action="store_true")
    parser.add_argument(
        "--disable-demonstration-videos",
        action="store_true",
        help="When recording NPZ demonstrations, skip per-camera mp4 files.",
    )
    parser.add_argument("--disable-overview-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=15)
    parser.add_argument("--render-width", type=int, default=320)
    parser.add_argument("--render-height", type=int, default=240)
    parser.add_argument(
        "--render-skip",
        type=int,
        default=1,
        help="Render every Nth video frame and duplicate the last rendered frame.",
    )
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--grasp-approach-duration-s", type=float, default=0.9)
    parser.add_argument("--pregrasp-height-offset-m", type=float, default=0.12)
    parser.add_argument("--pregrasp-lateral-offset-m", type=float, default=0.015)
    parser.add_argument("--lift-height-m", type=float, default=0.22)
    parser.add_argument("--lift-duration-s", type=float, default=2.0)
    parser.add_argument("--lift-hold-duration-s", type=float, default=0.0)
    parser.add_argument("--initial-lift-stage-height-m", type=float, default=0.0)
    parser.add_argument("--initial-lift-stage-duration-s", type=float, default=0.45)
    parser.add_argument("--initial-lift-stage-hold-s", type=float, default=0.20)
    parser.add_argument("--transport-duration-s", type=float, default=2.5)
    parser.add_argument(
        "--high-transport-waypoint",
        action="store_true",
        help="Traverse above the bin at lift height before descending to place.",
    )
    parser.add_argument("--high-transport-fraction", type=float, default=0.65)
    parser.add_argument(
        "--transport-relative-pose-feedback",
        action="store_true",
        help=(
            "Offline-teacher mode: during lift-to-place, preserve the measured target-to-palm "
            "SE(3) transform and advance only while physical target contact is present."
        ),
    )
    parser.add_argument("--transport-feedback-hz", type=float, default=20.0)
    parser.add_argument("--transport-feedback-gain", type=float, default=0.65)
    parser.add_argument("--transport-max-correction-m", type=float, default=0.04)
    parser.add_argument("--transport-max-relative-error-m", type=float, default=0.06)
    parser.add_argument("--transport-contact-grace-s", type=float, default=0.08)
    parser.add_argument("--transport-contact-closure-feedback", action="store_true")
    parser.add_argument("--transport-contact-closure-max-boost", type=float, default=0.08)
    parser.add_argument("--transport-contact-closure-rate-per-s", type=float, default=0.05)
    parser.add_argument("--transport-contact-closure-decay-per-s", type=float, default=0.025)
    parser.add_argument("--adaptive-regrasp", action="store_true")
    parser.add_argument(
        "--red-triangle-mesh-collision",
        action="store_true",
        help="Use the visual red triangular prism convex hull for physical contact.",
    )
    parser.add_argument("--body-gain-scale", type=float, default=4.0)
    parser.add_argument("--hand-gain-scale", type=float, default=5.0)
    parser.add_argument("--hand-closure-multiplier", type=float, default=1.0)
    parser.add_argument("--hand-closure-delay-s", type=float, default=0.0)
    parser.add_argument(
        "--grasp-yaw-deg",
        type=float,
        default=None,
        help="Override the object-specific physical grasp orientation.",
    )
    parser.add_argument(
        "--contact-grasp-compensation",
        default=",".join(str(value) for value in CONTACT_GRASP_COMPENSATION_M),
        help=(
            "World-frame x,y,z palm-target compensation in metres for contact "
            "grasping; y is mirrored automatically for the right hand."
        ),
    )
    parser.add_argument(
        "--localization-source",
        choices=("privileged", "head_rgb"),
        default="privileged",
    )
    parser.add_argument(
        "--allow-occluded-initial-head-view",
        action="store_true",
        help=(
            "Allow a policy-visited upper-body pose to occlude objects in the initial "
            "head image. This is only for offline privileged-teacher recovery data; "
            "the default requires all objects in view."
        ),
    )
    parser.add_argument(
        "--grasp-mode",
        choices=("contact", "assisted"),
        default="contact",
        help=(
            "Use collision/friction-only grasping (default), or the legacy "
            "connect constraint for controlled ablation."
        ),
    )
    return parser.parse_args()


def run_episode(args: argparse.Namespace) -> dict:
    permutation = tuple(int(value) for value in args.slot_permutation.split(","))
    if sorted(permutation) != [0, 1, 2]:
        raise ValueError("--slot-permutation must contain 0,1,2 exactly once")
    offset_values = tuple(float(value) for value in args.slot_offsets.split(","))
    if len(offset_values) != 6:
        raise ValueError("--slot-offsets must contain six comma-separated x,y values")
    offsets = tuple(
        np.asarray(offset_values[index : index + 2]) for index in range(0, 6, 2)
    )
    positions = {
        name: np.array(
            (
                *(OBJECT_SLOTS_XY_M[permutation[index]] + offsets[index]),
                OBJECT_SPAWN_Z_M[name],
            )
        )
        for index, name in enumerate(OBJECT_NAMES)
    }
    grasp_yaw_deg = (
        OBJECT_GRASP_YAW_DEG[args.target_object]
        if args.grasp_yaw_deg is None
        else args.grasp_yaw_deg
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.duration_s <= 0:
        raise ValueError("--duration-s must be positive")
    if (
        args.body_gain_scale <= 0
        or args.hand_gain_scale <= 0
        or args.hand_closure_multiplier <= 0
        or args.hand_closure_delay_s < 0
        or args.grasp_approach_duration_s <= 0
        or args.pregrasp_lateral_offset_m < 0
        or args.lift_height_m <= 0
        or args.lift_duration_s <= 0
        or args.lift_hold_duration_s < 0
        or args.initial_lift_stage_height_m < 0
        or args.initial_lift_stage_duration_s <= 0
        or args.initial_lift_stage_hold_s < 0
        or args.initial_lift_stage_height_m >= args.lift_height_m
        or args.transport_duration_s <= 0
        or not 0.2 <= args.high_transport_fraction <= 0.85
        or args.transport_feedback_hz <= 0
        or not 0.0 <= args.transport_feedback_gain <= 1.0
        or args.transport_max_correction_m <= 0
        or args.transport_max_relative_error_m <= 0
        or args.transport_contact_grace_s < 0
        or args.transport_contact_closure_max_boost < 0
        or args.transport_contact_closure_rate_per_s <= 0
        or args.transport_contact_closure_decay_per_s <= 0
    ):
        raise ValueError("Controller gains, hand closure, and approach must be positive")
    contact_compensation_values = tuple(
        float(value) for value in args.contact_grasp_compensation.split(",")
    )
    if len(contact_compensation_values) != 3:
        raise ValueError(
            "--contact-grasp-compensation must contain comma-separated x,y,z"
        )
    contact_grasp_compensation = np.asarray(contact_compensation_values)
    try:
        hand_profile_values = tuple(
            float(value) for value in args.hand_profile_scale.split(",")
        )
    except ValueError as error:
        raise ValueError("--hand-profile-scale must be three comma-separated floats") from error
    if len(hand_profile_values) != 3 or any(value <= 0.0 for value in hand_profile_values):
        raise ValueError("--hand-profile-scale must contain three positive values")
    scene_path = args.output_dir / "g1_language_pick_place.xml"
    build_scene(
        args.asset.resolve(),
        scene_path,
        positions,
        red_triangle_mesh_collision=args.red_triangle_mesh_collision,
    )

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    controlled_names = tuple(controlled)
    selected_names, selected_qpos_ids, lower, upper = arm_ik_contract(model)
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in PALM_NAMES
    ]
    object_body_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in OBJECT_NAMES
    }
    object_site_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_assist_site")
        for name in OBJECT_NAMES
    }
    equality_ids = {
        (arm_index, name): mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            f"{name}_{'left' if arm_index == LEFT_PALM_INDEX else 'right'}_assisted_grasp",
        )
        for arm_index in (LEFT_PALM_INDEX, RIGHT_PALM_INDEX)
        for name in OBJECT_NAMES
    }
    bin_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "blue_box_drop_site"
    )

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    canonical_home_pose, home_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        HOME_POSITIONS_M,
        np.zeros(len(selected_names)),
        orientation_weight=0.06,
    )
    home_pose = canonical_home_pose.copy()
    initial_arm_pose = None
    if args.initial_upper_state_npz is not None:
        initial_npz = np.load(args.initial_upper_state_npz, allow_pickle=True)
        if "observation_joint_position_rad" not in initial_npz:
            raise ValueError("--initial-upper-state-npz must contain observation_joint_position_rad")
        upper_state = np.asarray(initial_npz["observation_joint_position_rad"][0], dtype=float)
        if upper_state.shape != (31,):
            raise ValueError(f"Expected 31D upper state, found {upper_state.shape}")
        upper_names = controlled_names[12:]
        initial_arm_pose = np.asarray(
            [upper_state[upper_names.index(name)] for name in selected_names],
            dtype=float,
        )
        initial_upper_velocity = None
        if "observation_joint_velocity_rad_s" in initial_npz:
            initial_upper_velocity = np.asarray(
                initial_npz["observation_joint_velocity_rad_s"][0], dtype=float
            )
            if initial_upper_velocity.shape != (31,):
                raise ValueError(
                    f"Expected 31D upper velocity, found {initial_upper_velocity.shape}"
                )
        initial_npz.close()
    else:
        initial_upper_velocity = None
    if (
        args.initial_state_recovery_s < 0.0
        or args.initial_safe_lift_s < 0.0
        or args.initial_safe_traverse_s < 0.0
        or args.initial_safe_lift_height_m <= 0.0
    ):
        raise ValueError("Initial recovery durations must be non-negative and lift height positive")
    if args.initial_home_settle_s < 0.0:
        raise ValueError("--initial-home-settle-s must be non-negative")
    if args.initial_state_recovery_s > 0.0 and initial_arm_pose is None:
        raise ValueError(
            "--initial-state-recovery-s requires --initial-upper-state-npz"
        )
    if (args.initial_safe_lift_s > 0.0 or args.initial_safe_traverse_s > 0.0) and initial_arm_pose is None:
        raise ValueError("Safe recovery waypoints require --initial-upper-state-npz")
    if (args.initial_safe_lift_s > 0.0) != (args.initial_safe_traverse_s > 0.0):
        raise ValueError("Safe recovery requires both lift and traverse durations")
    if args.initial_home_settle_s > 0.0 and initial_arm_pose is None:
        raise ValueError(
            "--initial-home-settle-s requires --initial-upper-state-npz"
        )
    if args.initial_direct_continuation and initial_arm_pose is None:
        raise ValueError("--initial-direct-continuation requires --initial-upper-state-npz")
    if args.initial_direct_continuation and (
        args.initial_state_recovery_s > 0.0
        or args.initial_home_settle_s > 0.0
        or args.initial_safe_lift_s > 0.0
        or args.initial_safe_traverse_s > 0.0
    ):
        raise ValueError("Direct continuation cannot be combined with a home reset")
    if args.initial_direct_grasp_s <= 0.0:
        raise ValueError("--initial-direct-grasp-s must be positive")
    mujoco.mj_resetData(model, data)
    data.qpos[selected_qpos_ids] = (
        initial_arm_pose if initial_arm_pose is not None else canonical_home_pose
    )
    open_hand_targets = {}
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        low, high = model.jnt_range[item["joint_id"]]
        if abs(float(low)) < 1e-9:
            value = 0.01
        elif abs(float(high)) < 1e-9:
            value = -0.01
        else:
            value = float(data.qpos[item["qpos_id"]])
        open_hand_targets[name] = value
        data.qpos[item["qpos_id"]] = value
    mujoco.mj_forward(model, data)
    initial_palm_positions = np.stack(
        [data.site_xpos[palm_id].copy() for palm_id in palm_ids]
    )

    renderer = mujoco.Renderer(
        model, height=args.render_height, width=args.render_width
    )
    if args.render_skip < 1:
        raise ValueError("--render-skip must be at least one")
    camera_visibility = audit_head_camera_visibility(renderer, data)
    if (
        not camera_visibility["all_objects_visible"]
        and not args.allow_occluded_initial_head_view
    ):
        raise RuntimeError(
            "head_camera does not visibly contain all task objects: "
            f"{camera_visibility}"
        )
    localization_source = getattr(args, "localization_source", "privileged")
    if camera_visibility["all_objects_visible"]:
        visual_object_positions = localize_objects_from_head_rgb(
            model, data, camera_visibility
        )
    elif localization_source == "head_rgb":
        raise RuntimeError(
            "--localization-source=head_rgb requires every task object in the initial head view"
        )
    else:
        visual_object_positions = {}
    if localization_source == "head_rgb":
        target_center = visual_object_positions[args.target_object]
        object_assist = target_center + OBJECT_ASSIST_OFFSET_M[args.target_object]
    else:
        object_assist = data.site_xpos[object_site_ids[args.target_object]].copy()
    active_palm_index = (
        LEFT_PALM_INDEX
        if args.force_active_arm == "left"
        else RIGHT_PALM_INDEX
        if args.force_active_arm == "right"
        else LEFT_PALM_INDEX
        if object_assist[1] >= 0.05
        else RIGHT_PALM_INDEX
    )
    active_side = "left" if active_palm_index == LEFT_PALM_INDEX else "right"
    PALM_ORIENTATION_TARGETS[:] = np.eye(3)
    PALM_ORIENTATION_TARGETS[active_palm_index] = Rotation.from_euler(
        "z", grasp_yaw_deg, degrees=True
    ).as_matrix()
    grasp_lateral_offset = (
        0.030 if active_palm_index == LEFT_PALM_INDEX else -0.030
    )
    object_assist[1] += grasp_lateral_offset
    high_pregrasp_targets = HOME_POSITIONS_M.copy()
    high_pregrasp_targets[active_palm_index] = object_assist + np.array(
        (0.0, 0.0, 0.28)
    )
    pregrasp_targets = HOME_POSITIONS_M.copy()
    lateral_offset = (
        args.pregrasp_lateral_offset_m
        if active_palm_index == LEFT_PALM_INDEX
        else -args.pregrasp_lateral_offset_m
    )
    pregrasp_targets[active_palm_index] = object_assist + np.array(
        (0.0, lateral_offset, args.pregrasp_height_offset_m)
    )
    grasp_targets = HOME_POSITIONS_M.copy()
    grasp_targets[active_palm_index] = object_assist
    if args.grasp_mode == "contact":
        contact_compensation = contact_grasp_compensation.copy()
        if active_palm_index == RIGHT_PALM_INDEX:
            contact_compensation[1] *= -1.0
        grasp_targets[active_palm_index] += contact_compensation
    lift_targets = grasp_targets.copy()
    lift_targets[active_palm_index, 2] += args.lift_height_m
    place_targets = HOME_POSITIONS_M.copy()
    place_targets[active_palm_index] = data.site_xpos[bin_site_id].copy()
    retreat_targets = place_targets.copy()
    retreat_targets[active_palm_index, 2] += 0.18

    safe_lift_pose = None
    safe_traverse_pose = None
    safe_pose_reports = {}
    if args.initial_safe_lift_s > 0.0:
        safe_lift_targets = initial_palm_positions.copy()
        safe_lift_targets[:, 2] += args.initial_safe_lift_height_m
        safe_lift_pose, safe_lift_report = solve_pose(
            model,
            data,
            palm_ids,
            selected_qpos_ids,
            lower,
            upper,
            safe_lift_targets,
            initial_arm_pose,
            orientation_weight=0.0,
        )
        safe_traverse_targets = HOME_POSITIONS_M.copy()
        safe_traverse_targets[:, 2] = np.maximum(
            safe_traverse_targets[:, 2], safe_lift_targets[:, 2]
        )
        safe_traverse_pose, safe_traverse_report = solve_pose(
            model,
            data,
            palm_ids,
            selected_qpos_ids,
            lower,
            upper,
            safe_traverse_targets,
            safe_lift_pose,
            orientation_weight=0.0,
        )
        safe_pose_reports = {
            "initial_safe_lift": safe_lift_report,
            "initial_safe_traverse": safe_traverse_report,
        }

    high_pregrasp_pose, high_pregrasp_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        high_pregrasp_targets,
        home_pose,
    )
    pregrasp_pose, pregrasp_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        pregrasp_targets,
        high_pregrasp_pose,
    )
    grasp_pose, grasp_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        grasp_targets,
        pregrasp_pose,
    )
    lift_pose, lift_report = solve_pose(
        model, data, palm_ids, selected_qpos_ids, lower, upper, lift_targets, grasp_pose
    )
    initial_lift_stage_pose = None
    initial_lift_stage_report = None
    if args.initial_lift_stage_height_m > 0.0:
        initial_lift_stage_targets = grasp_targets.copy()
        initial_lift_stage_targets[
            active_palm_index, 2
        ] += args.initial_lift_stage_height_m
        initial_lift_stage_pose, initial_lift_stage_report = solve_pose(
            model,
            data,
            palm_ids,
            selected_qpos_ids,
            lower,
            upper,
            initial_lift_stage_targets,
            grasp_pose,
        )
    place_pose, place_report = solve_pose(
        model, data, palm_ids, selected_qpos_ids, lower, upper, place_targets, lift_pose
    )
    high_transport_pose = None
    high_transport_report = None
    if args.high_transport_waypoint:
        high_transport_targets = place_targets.copy()
        high_transport_targets[active_palm_index, 2] = lift_targets[
            active_palm_index, 2
        ]
        high_transport_pose, high_transport_report = solve_pose(
            model, data, palm_ids, selected_qpos_ids, lower, upper,
            high_transport_targets, lift_pose,
        )
    retreat_pose, retreat_report = solve_pose(
        model,
        data,
        palm_ids,
        selected_qpos_ids,
        lower,
        upper,
        retreat_targets,
        place_pose,
    )
    pose_reports = {
        "home": home_report,
        "high_pregrasp": high_pregrasp_report,
        "pregrasp": pregrasp_report,
        "grasp": grasp_report,
        "lift": lift_report,
        "place": place_report,
        "retreat": retreat_report,
        **safe_pose_reports,
    }
    if initial_lift_stage_report is not None:
        pose_reports["initial_lift_stage"] = initial_lift_stage_report
    if high_transport_report is not None:
        pose_reports["high_transport"] = high_transport_report
    if any(not report["solver_success"] for report in pose_reports.values()):
        raise RuntimeError(f"IK failed: {pose_reports}")
    if (
        max(report["maximum_position_error_m"] for report in pose_reports.values())
        > 0.08
    ):
        raise RuntimeError(f"IK position error is too large: {pose_reports}")

    mujoco.mj_resetData(model, data)
    data.qpos[selected_qpos_ids] = (
        initial_arm_pose if initial_arm_pose is not None else home_pose
    )
    for name, item in controlled.items():
        if "hand_" not in name:
            continue
        data.qpos[item["qpos_id"]] = open_hand_targets[name]
    if (
        initial_upper_velocity is not None
        and args.initial_velocity_mode == "restore"
    ):
        upper_names = controlled_names[12:]
        for index, name in enumerate(upper_names):
            data.qvel[controlled[name]["qvel_id"]] = initial_upper_velocity[index]
    mujoco.mj_forward(model, data)
    initial_targets = {
        name: float(data.qpos[item["qpos_id"]]) for name, item in controlled.items()
    }
    initial_object_positions = {
        name: data.xpos[body_id].copy() for name, body_id in object_body_ids.items()
    }
    visual_localization_error_m = (
        {
            name: float(
                np.linalg.norm(
                    visual_object_positions[name] - initial_object_positions[name]
                )
            )
            for name in OBJECT_NAMES
        }
        if visual_object_positions
        else {}
    )
    selected_index = {name: index for index, name in enumerate(selected_names)}

    overview_camera = mujoco.MjvCamera()
    overview_camera.lookat[:] = (0.45, -0.05, 0.92)
    overview_camera.distance = 2.15
    overview_camera.azimuth = 145
    overview_camera.elevation = -12
    video_path = args.output_dir / "language_pick_place.mp4"
    writer = (
        None
        if args.disable_overview_video
        else imageio.get_writer(
            video_path, fps=args.video_fps, codec="libx264", quality=8
        )
    )
    task_camera_names = ("head_camera", "left_wrist_camera", "right_wrist_camera")
    camera_writers = {}
    if args.record_demonstration and not args.disable_demonstration_videos:
        for camera_name in task_camera_names:
            camera_writers[camera_name] = imageio.get_writer(
                args.output_dir / f"{camera_name}.mp4",
                fps=args.video_fps,
                codec="libx264",
                quality=8,
            )

    records = defaultdict(list)
    dt = float(model.opt.timestep)
    total_time = args.duration_s
    render_interval = max(1, round(1.0 / (args.video_fps * dt)))
    video_frame_index = 0
    last_overview_frame = None
    last_camera_frames = {}
    total_steps = round(total_time / dt)
    initial_state_recovery_s = (
        args.initial_state_recovery_s if initial_arm_pose is not None else 0.0
    )
    initial_safe_lift_s = (
        args.initial_safe_lift_s if initial_arm_pose is not None else 0.0
    )
    initial_safe_traverse_s = (
        args.initial_safe_traverse_s if initial_arm_pose is not None else 0.0
    )
    initial_home_settle_s = (
        args.initial_home_settle_s if initial_arm_pose is not None else 0.0
    )
    direct_continuation = bool(
        args.initial_direct_continuation and initial_arm_pose is not None
    )
    grasp_start_s = 0.0 if direct_continuation else 3.1
    grasp_end_s = grasp_start_s + args.grasp_approach_duration_s
    if direct_continuation:
        grasp_end_s = args.initial_direct_grasp_s
    grasp_hold_end_s = grasp_end_s + 0.8
    staged_lift_extra_s = (
        args.initial_lift_stage_duration_s + args.initial_lift_stage_hold_s
        if initial_lift_stage_pose is not None
        else 0.0
    )
    lift_end_s = grasp_hold_end_s + staged_lift_extra_s + args.lift_duration_s
    place_start_s = lift_end_s + args.lift_hold_duration_s
    place_end_s = place_start_s + args.transport_duration_s
    place_hold_end_s = place_end_s + 0.5
    retreat_end_s = place_hold_end_s + 1.5
    joint_limit_violations = 0
    joint_samples = 0
    wrong_object_max_displacement = 0.0
    assisted_constraint_active_steps = 0
    physical_contact_steps = 0
    physical_contact_samples = 0
    maximum_contacts_per_step = 0
    maximum_contact_normal_force_n = 0.0
    first_contact_time_s = None
    last_contact_time_s = None
    target_contact_pairs = defaultdict(int)
    target_maximum_height_m = float(
        initial_object_positions[args.target_object][2]
    )
    adaptive_regrasp_planned = False
    adaptive_regrasp_start_pose = None
    adaptive_regrasp_object_displacement_m = None
    adaptive_lift_planned = False
    adaptive_lift_object_displacement_m = None
    adaptive_transport_object_to_palm_offset_m = None
    transport_reference_position_palm_m = None
    transport_reference_position_world_m = None
    transport_reference_rotation_palm = None
    transport_start_object_position_m = None
    transport_start_object_rotation = None
    transport_lift_progress = 0.0
    transport_progress = 0.0
    transport_feedback_pose = None
    transport_last_replan_s = -np.inf
    transport_release_start_s = None
    transport_replans = 0
    transport_contact_loss_steps = 0
    transport_max_relative_error_m = 0.0
    transport_max_applied_correction_m = 0.0
    transport_max_rotation_error_deg = 0.0
    previous_contacts_this_step = 0
    transport_last_contact_task_s = -np.inf
    transport_contact_closure_boost = 0.0
    transport_max_contact_closure_boost = 0.0
    transport_contact_closure_recovery_steps = 0
    ik_data = mujoco.MjData(model)

    for step in range(total_steps):
        time_s = step * dt
        task_time_s = (
            time_s
            - initial_safe_lift_s
            - initial_safe_traverse_s
            - initial_state_recovery_s
            - initial_home_settle_s
        )
        if (
            args.adaptive_regrasp
            and not adaptive_regrasp_planned
            and task_time_s >= grasp_end_s
        ):
            ik_data.qpos[:] = data.qpos
            ik_data.qvel[:] = data.qvel
            mujoco.mj_forward(model, ik_data)
            adaptive_regrasp_start_pose = data.qpos[selected_qpos_ids].copy()
            current_object_assist = data.site_xpos[
                object_site_ids[args.target_object]
            ].copy()
            current_object_assist[1] += grasp_lateral_offset
            adaptive_grasp_targets = HOME_POSITIONS_M.copy()
            adaptive_grasp_targets[active_palm_index] = current_object_assist
            if args.grasp_mode == "contact":
                adaptive_grasp_targets[active_palm_index] += contact_compensation
            grasp_pose, adaptive_grasp_report = solve_pose(
                model,
                ik_data,
                palm_ids,
                selected_qpos_ids,
                lower,
                upper,
                adaptive_grasp_targets,
                adaptive_regrasp_start_pose,
            )
            adaptive_lift_targets = adaptive_grasp_targets.copy()
            adaptive_lift_targets[active_palm_index, 2] += args.lift_height_m
            lift_pose, adaptive_lift_report = solve_pose(
                model,
                ik_data,
                palm_ids,
                selected_qpos_ids,
                lower,
                upper,
                adaptive_lift_targets,
                grasp_pose,
            )
            current_palm_position = data.site_xpos[
                palm_ids[active_palm_index]
            ].copy()
            current_object_position = data.xpos[
                object_body_ids[args.target_object]
            ].copy()
            adaptive_transport_object_to_palm_offset_m = (
                current_object_position - current_palm_position
            )
            adaptive_place_targets = place_targets.copy()
            adaptive_place_targets[active_palm_index] = (
                data.site_xpos[bin_site_id].copy()
                - adaptive_transport_object_to_palm_offset_m
            )
            adaptive_retreat_targets = adaptive_place_targets.copy()
            adaptive_retreat_targets[active_palm_index, 2] += 0.18
            place_pose, adaptive_place_report = solve_pose(
                model,
                ik_data,
                palm_ids,
                selected_qpos_ids,
                lower,
                upper,
                adaptive_place_targets,
                lift_pose,
            )
            retreat_pose, adaptive_retreat_report = solve_pose(
                model,
                ik_data,
                palm_ids,
                selected_qpos_ids,
                lower,
                upper,
                adaptive_retreat_targets,
                place_pose,
            )
            pose_reports.update(
                {
                    "adaptive_grasp": adaptive_grasp_report,
                    "adaptive_lift": adaptive_lift_report,
                    "adaptive_place": adaptive_place_report,
                    "adaptive_retreat": adaptive_retreat_report,
                }
            )
            adaptive_regrasp_object_displacement_m = (
                data.xpos[object_body_ids[args.target_object]]
                - initial_object_positions[args.target_object]
            ).copy()
            adaptive_regrasp_planned = True
        if (
            args.adaptive_regrasp
            and adaptive_regrasp_planned
            and not adaptive_lift_planned
            and task_time_s >= grasp_hold_end_s
        ):
            ik_data.qpos[:] = data.qpos
            ik_data.qvel[:] = data.qvel
            mujoco.mj_forward(model, ik_data)
            grasp_pose = data.qpos[selected_qpos_ids].copy()
            current_object_assist = data.site_xpos[
                object_site_ids[args.target_object]
            ].copy()
            current_object_assist[1] += grasp_lateral_offset
            adaptive_lift_targets = HOME_POSITIONS_M.copy()
            adaptive_lift_targets[active_palm_index] = current_object_assist
            if args.grasp_mode == "contact":
                adaptive_lift_targets[active_palm_index] += contact_compensation
            adaptive_lift_targets[active_palm_index, 2] += args.lift_height_m
            lift_pose, adaptive_final_lift_report = solve_pose(
                model,
                ik_data,
                palm_ids,
                selected_qpos_ids,
                lower,
                upper,
                adaptive_lift_targets,
                grasp_pose,
            )
            if args.initial_lift_stage_height_m > 0.0:
                adaptive_initial_lift_stage_targets = adaptive_lift_targets.copy()
                adaptive_initial_lift_stage_targets[active_palm_index, 2] -= (
                    args.lift_height_m - args.initial_lift_stage_height_m
                )
                (
                    initial_lift_stage_pose,
                    adaptive_initial_lift_stage_report,
                ) = solve_pose(
                    model,
                    ik_data,
                    palm_ids,
                    selected_qpos_ids,
                    lower,
                    upper,
                    adaptive_initial_lift_stage_targets,
                    grasp_pose,
                )
                pose_reports["adaptive_initial_lift_stage"] = (
                    adaptive_initial_lift_stage_report
                )
            current_palm_position = data.site_xpos[
                palm_ids[active_palm_index]
            ].copy()
            current_object_position = data.xpos[
                object_body_ids[args.target_object]
            ].copy()
            adaptive_transport_object_to_palm_offset_m = (
                current_object_position - current_palm_position
            )
            adaptive_place_targets = place_targets.copy()
            adaptive_place_targets[active_palm_index] = (
                data.site_xpos[bin_site_id].copy()
                - adaptive_transport_object_to_palm_offset_m
            )
            adaptive_retreat_targets = adaptive_place_targets.copy()
            adaptive_retreat_targets[active_palm_index, 2] += 0.18
            place_pose, adaptive_final_place_report = solve_pose(
                model,
                ik_data,
                palm_ids,
                selected_qpos_ids,
                lower,
                upper,
                adaptive_place_targets,
                lift_pose,
            )
            if args.high_transport_waypoint:
                adaptive_high_transport_targets = adaptive_place_targets.copy()
                adaptive_high_transport_targets[active_palm_index, 2] = (
                    adaptive_lift_targets[active_palm_index, 2]
                )
                high_transport_pose, adaptive_high_transport_report = solve_pose(
                    model,
                    ik_data,
                    palm_ids,
                    selected_qpos_ids,
                    lower,
                    upper,
                    adaptive_high_transport_targets,
                    lift_pose,
                )
                pose_reports["adaptive_high_transport"] = adaptive_high_transport_report
            retreat_pose, adaptive_final_retreat_report = solve_pose(
                model,
                ik_data,
                palm_ids,
                selected_qpos_ids,
                lower,
                upper,
                adaptive_retreat_targets,
                place_pose,
            )
            pose_reports.update(
                {
                    "adaptive_final_lift": adaptive_final_lift_report,
                    "adaptive_final_place": adaptive_final_place_report,
                    "adaptive_final_retreat": adaptive_final_retreat_report,
                }
            )
            adaptive_lift_object_displacement_m = (
                data.xpos[object_body_ids[args.target_object]]
                - initial_object_positions[args.target_object]
            ).copy()
            adaptive_lift_planned = True

        if (
            args.transport_relative_pose_feedback
            and task_time_s >= grasp_hold_end_s
        ):
            current_palm_position = data.site_xpos[palm_ids[active_palm_index]].copy()
            current_palm_rotation = data.site_xmat[palm_ids[active_palm_index]].reshape(3, 3).copy()
            current_object_position = data.xpos[object_body_ids[args.target_object]].copy()
            current_object_rotation = data.xmat[object_body_ids[args.target_object]].reshape(3, 3).copy()
            if transport_reference_position_palm_m is None:
                transport_reference_position_palm_m = (
                    current_palm_rotation.T @ (current_object_position - current_palm_position)
                )
                transport_reference_position_world_m = (
                    current_object_position - current_palm_position
                )
                transport_reference_rotation_palm = current_palm_rotation.T @ current_object_rotation
                transport_start_object_position_m = current_object_position.copy()
                transport_start_object_rotation = current_object_rotation.copy()
            # Preserve translation in the palm frame but do not torque a
            # friction grasp to enforce the initial object rotation.  Long
            # objects need limited passive rotational compliance in transport.
            desired_palm_rotation = current_palm_rotation
            rotation_error = (
                transport_start_object_rotation.T @ current_object_rotation
            )
            rotation_cos = np.clip((np.trace(rotation_error) - 1.0) * 0.5, -1.0, 1.0)
            transport_max_rotation_error_deg = max(
                transport_max_rotation_error_deg,
                float(np.degrees(np.arccos(rotation_cos))),
            )
            relative_error_world = (
                current_object_position
                - current_palm_position
                - transport_reference_position_world_m
            )
            relative_error_m = float(np.linalg.norm(relative_error_world))
            transport_max_relative_error_m = max(
                transport_max_relative_error_m, relative_error_m
            )
            if previous_contacts_this_step > 0:
                transport_last_contact_task_s = task_time_s
            contact_ok = (
                task_time_s - transport_last_contact_task_s
                <= args.transport_contact_grace_s
            )
            if args.transport_contact_closure_feedback:
                if contact_ok:
                    transport_contact_closure_boost = max(
                        0.0,
                        transport_contact_closure_boost
                        - args.transport_contact_closure_decay_per_s * dt,
                    )
                else:
                    transport_contact_closure_boost = min(
                        args.transport_contact_closure_max_boost,
                        transport_contact_closure_boost
                        + args.transport_contact_closure_rate_per_s * dt,
                    )
                    transport_contact_closure_recovery_steps += 1
                transport_max_contact_closure_boost = max(
                    transport_max_contact_closure_boost,
                    transport_contact_closure_boost,
                )
            pose_enclosed = relative_error_m <= 0.5 * args.transport_max_relative_error_m
            grasp_ok = contact_ok or pose_enclosed
            if grasp_ok and relative_error_m <= args.transport_max_relative_error_m:
                if transport_lift_progress < 1.0:
                    transport_lift_progress = min(
                        1.0,
                        transport_lift_progress + dt / args.lift_duration_s,
                    )
                else:
                    transport_progress = min(
                        1.0,
                        transport_progress + dt / args.transport_duration_s,
                    )
            else:
                transport_contact_loss_steps += 1
            if task_time_s - transport_last_replan_s >= 1.0 / args.transport_feedback_hz:
                lifted_object_position = transport_start_object_position_m.copy()
                lifted_object_position[2] += args.lift_height_m
                desired_object_position = (
                    interpolate(
                        transport_start_object_position_m,
                        lifted_object_position,
                        transport_lift_progress,
                    )
                    if transport_lift_progress < 1.0
                    else interpolate(
                        lifted_object_position,
                        data.site_xpos[bin_site_id].copy(),
                        transport_progress,
                    )
                )
                nominal_palm_position = (
                    desired_object_position
                    - transport_reference_position_world_m
                )
                if grasp_ok:
                    requested_palm_position = (
                        nominal_palm_position
                        + args.transport_feedback_gain * relative_error_world
                    )
                else:
                    # Freeze task progress and recover the reference transform.
                    requested_palm_position = (
                        current_object_position
                        - transport_reference_position_world_m
                    )
                correction = requested_palm_position - current_palm_position
                correction_norm = float(np.linalg.norm(correction))
                if correction_norm > args.transport_max_correction_m:
                    correction *= args.transport_max_correction_m / correction_norm
                transport_max_applied_correction_m = max(
                    transport_max_applied_correction_m, float(np.linalg.norm(correction))
                )
                feedback_targets = HOME_POSITIONS_M.copy()
                feedback_targets[active_palm_index] = current_palm_position + correction
                ik_data.qpos[:] = data.qpos
                ik_data.qvel[:] = data.qvel
                mujoco.mj_forward(model, ik_data)
                candidate_pose, feedback_report = solve_pose(
                    model,
                    ik_data,
                    palm_ids,
                    selected_qpos_ids,
                    lower,
                    upper,
                    feedback_targets,
                    data.qpos[selected_qpos_ids],
                    orientation_weight=0.0,
                )
                if (
                    feedback_report["solver_success"]
                    and feedback_report["maximum_position_error_m"] <= 0.06
                ):
                    transport_feedback_pose = candidate_pose
                    transport_replans += 1
                transport_last_replan_s = task_time_s
            if (
                transport_lift_progress >= 1.0
                and
                transport_progress >= 1.0
                and grasp_ok
                and transport_release_start_s is None
            ):
                transport_release_start_s = task_time_s
        if time_s < initial_safe_lift_s:
            selected_pose, phase = (
                interpolate(
                    initial_arm_pose,
                    safe_lift_pose,
                    time_s / max(initial_safe_lift_s, 1e-8),
                ),
                0,
            )
        elif time_s < initial_safe_lift_s + initial_safe_traverse_s:
            selected_pose, phase = (
                interpolate(
                    safe_lift_pose,
                    safe_traverse_pose,
                    (time_s - initial_safe_lift_s) / max(initial_safe_traverse_s, 1e-8),
                ),
                0,
            )
        elif time_s < initial_safe_lift_s + initial_safe_traverse_s + initial_state_recovery_s:
            selected_pose, phase = (
                interpolate(
                    safe_traverse_pose if safe_traverse_pose is not None else initial_arm_pose,
                    canonical_home_pose,
                    (time_s - initial_safe_lift_s - initial_safe_traverse_s)
                    / max(initial_state_recovery_s, 1e-8),
                ),
                0,
            )
        elif time_s < (
            initial_safe_lift_s
            + initial_safe_traverse_s
            + initial_state_recovery_s
            + initial_home_settle_s
        ):
            selected_pose, phase = canonical_home_pose, 0
        elif direct_continuation:
            if task_time_s < grasp_end_s:
                selected_pose, phase = (
                    interpolate(
                        initial_arm_pose,
                        grasp_pose,
                        task_time_s / max(grasp_end_s, 1e-8),
                    ),
                    2,
                )
            elif task_time_s < grasp_hold_end_s:
                selected_pose, phase = grasp_pose, 3
            elif task_time_s < lift_end_s:
                lift_elapsed_s = task_time_s - grasp_hold_end_s
                if (
                    initial_lift_stage_pose is not None
                    and lift_elapsed_s < args.initial_lift_stage_duration_s
                ):
                    selected_pose = interpolate(
                        grasp_pose,
                        initial_lift_stage_pose,
                        lift_elapsed_s / args.initial_lift_stage_duration_s,
                    )
                elif (
                    initial_lift_stage_pose is not None
                    and lift_elapsed_s
                    < args.initial_lift_stage_duration_s
                    + args.initial_lift_stage_hold_s
                ):
                    selected_pose = initial_lift_stage_pose
                else:
                    remaining_lift_elapsed_s = lift_elapsed_s - staged_lift_extra_s
                    selected_pose = interpolate(
                        initial_lift_stage_pose
                        if initial_lift_stage_pose is not None
                        else grasp_pose,
                        lift_pose,
                        remaining_lift_elapsed_s / args.lift_duration_s,
                    )
                phase = 4
            elif task_time_s < place_start_s:
                selected_pose, phase = lift_pose, 4
            elif task_time_s < place_end_s:
                selected_pose, phase = (
                    transport_interpolate(
                        lift_pose,
                        high_transport_pose,
                        place_pose,
                        (task_time_s - place_start_s) / args.transport_duration_s,
                        args.high_transport_fraction,
                    ),
                    5,
                )
            elif task_time_s < place_hold_end_s:
                selected_pose, phase = place_pose, 6
            elif task_time_s < retreat_end_s:
                selected_pose, phase = (
                    interpolate(
                        place_pose,
                        retreat_pose,
                        (task_time_s - place_hold_end_s) / 1.5,
                    ),
                    7,
                )
            else:
                selected_pose, phase = retreat_pose, 7
        elif task_time_s < 0.5:
            selected_pose, phase = home_pose, 0
        elif task_time_s < 2.3:
            selected_pose, phase = (
                interpolate(home_pose, high_pregrasp_pose, (task_time_s - 0.5) / 1.8),
                1,
            )
        elif task_time_s < grasp_start_s:
            selected_pose, phase = (
                interpolate(high_pregrasp_pose, pregrasp_pose, (task_time_s - 2.3) / 0.8),
                2,
            )
        elif task_time_s < grasp_end_s:
            selected_pose, phase = (
                interpolate(
                    pregrasp_pose,
                    grasp_pose,
                    (task_time_s - grasp_start_s) / args.grasp_approach_duration_s,
                ),
                2,
            )
        elif task_time_s < grasp_hold_end_s:
            selected_pose, phase = (
                interpolate(
                    adaptive_regrasp_start_pose,
                    grasp_pose,
                    (task_time_s - grasp_end_s) / 0.8,
                )
                if adaptive_regrasp_start_pose is not None
                else grasp_pose,
                3,
            )
        elif task_time_s < lift_end_s:
            lift_elapsed_s = task_time_s - grasp_hold_end_s
            if (
                initial_lift_stage_pose is not None
                and lift_elapsed_s < args.initial_lift_stage_duration_s
            ):
                selected_pose = interpolate(
                    grasp_pose,
                    initial_lift_stage_pose,
                    lift_elapsed_s / args.initial_lift_stage_duration_s,
                )
            elif (
                initial_lift_stage_pose is not None
                and lift_elapsed_s
                < args.initial_lift_stage_duration_s
                + args.initial_lift_stage_hold_s
            ):
                selected_pose = initial_lift_stage_pose
            else:
                remaining_lift_elapsed_s = lift_elapsed_s - staged_lift_extra_s
                selected_pose = interpolate(
                    initial_lift_stage_pose
                    if initial_lift_stage_pose is not None
                    else grasp_pose,
                    lift_pose,
                    remaining_lift_elapsed_s / args.lift_duration_s,
                )
            phase = 4
        elif task_time_s < place_start_s:
            selected_pose, phase = lift_pose, 4
        elif task_time_s < place_end_s:
            selected_pose, phase = (
                transport_interpolate(
                    lift_pose,
                    high_transport_pose,
                    place_pose,
                    (task_time_s - place_start_s) / args.transport_duration_s,
                    args.high_transport_fraction,
                ),
                5,
            )
        elif task_time_s < place_hold_end_s:
            selected_pose, phase = place_pose, 6
        elif task_time_s < retreat_end_s:
            selected_pose, phase = (
                interpolate(place_pose, retreat_pose, (task_time_s - place_hold_end_s) / 1.5),
                7,
            )
        else:
            selected_pose, phase = retreat_pose, 7

        if (
            args.transport_relative_pose_feedback
            and task_time_s >= grasp_hold_end_s
            and transport_feedback_pose is not None
            and transport_release_start_s is None
        ):
            selected_pose, phase = transport_feedback_pose, 5

        # The default path is contact-driven: object motion must come from
        # hand geometry, friction, and the closing trajectory.  The legacy
        # connect constraint remains available only for explicit ablations.
        grasp_active = (
            args.grasp_mode == "assisted"
            and grasp_end_s <= task_time_s < place_end_s
        )
        for (arm_index, name), equality_id in equality_ids.items():
            # Equality activation is mutable state in current MuJoCo Python
            # bindings.  Strict contact mode writes zero on every step.
            data.eq_active[equality_id] = int(
                grasp_active
                and arm_index == active_palm_index
                and name == args.target_object
            )
        assisted_constraint_active_steps += int(grasp_active)
        close_alpha = smoothstep(
            (task_time_s - (grasp_start_s + args.hand_closure_delay_s)) / 0.6
        )
        if args.transport_relative_pose_feedback:
            open_alpha = (
                0.0
                if transport_release_start_s is None
                else smoothstep((task_time_s - transport_release_start_s) / 0.45)
            )
        else:
            open_alpha = smoothstep((task_time_s - (place_end_s - 0.05)) / 0.45)
        hand_alpha = close_alpha * (1.0 - open_alpha)
        commanded = dict(initial_targets)
        for name, index in selected_index.items():
            commanded[name] = float(selected_pose[index])
        for name, closed in HAND_TARGETS.items():
            if name.startswith(f"{active_side}_hand_"):
                grasp_target = object_hand_target(args.target_object, name, closed)
                grasp_target *= (
                    args.hand_closure_multiplier
                    + transport_contact_closure_boost
                )
                grasp_target *= hand_profile_scale(name, hand_profile_values)
                commanded[name] = (1.0 - hand_alpha) * open_hand_targets[
                    name
                ] + hand_alpha * grasp_target
            else:
                commanded[name] = open_hand_targets[name]

        for name, item in controlled.items():
            kp, kd = unitree_gains(name)
            gain_scale = (
                args.hand_gain_scale if "hand_" in name else args.body_gain_scale
            )
            kp *= gain_scale
            kd *= math.sqrt(gain_scale)
            qpos = float(data.qpos[item["qpos_id"]])
            qvel = float(data.qvel[item["qvel_id"]])
            torque = kp * (commanded[name] - qpos) - kd * qvel
            torque += float(data.qfrc_bias[item["qvel_id"]])
            data.ctrl[item["actuator_id"]] = torque
        mujoco.mj_step(model, data)

        contacts_this_step = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom_ids = (int(contact.geom1), int(contact.geom2))
            body_ids = tuple(int(model.geom_bodyid[geom_id]) for geom_id in geom_ids)
            if object_body_ids[args.target_object] not in body_ids:
                continue
            object_side = body_ids.index(object_body_ids[args.target_object])
            hand_side = 1 - object_side
            hand_body_name = object_name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_ids[hand_side]
            )
            hand_geom_name = object_name(
                model, mujoco.mjtObj.mjOBJ_GEOM, geom_ids[hand_side]
            )
            is_active_hand = hand_body_name.startswith(f"{active_side}_hand_")
            is_palm_pad = hand_geom_name == f"{active_side}_palm_contact_pad"
            if not (is_active_hand or is_palm_pad):
                continue
            contacts_this_step += 1
            pair_name = hand_geom_name if is_palm_pad else hand_body_name
            target_contact_pairs[pair_name] += 1
            contact_force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, contact_index, contact_force)
            maximum_contact_normal_force_n = max(
                maximum_contact_normal_force_n, abs(float(contact_force[0]))
            )
        if contacts_this_step:
            physical_contact_steps += 1
            physical_contact_samples += contacts_this_step
            maximum_contacts_per_step = max(
                maximum_contacts_per_step, contacts_this_step
            )
            if first_contact_time_s is None:
                first_contact_time_s = time_s
            last_contact_time_s = time_s
        previous_contacts_this_step = contacts_this_step
        target_maximum_height_m = max(
            target_maximum_height_m,
            float(data.xpos[object_body_ids[args.target_object], 2]),
        )

        for name, item in controlled.items():
            low, high = model.jnt_range[item["joint_id"]]
            qpos = float(data.qpos[item["qpos_id"]])
            joint_limit_violations += int(qpos < low - 1e-6 or qpos > high + 1e-6)
            joint_samples += 1
        for name in OBJECT_NAMES:
            if name == args.target_object:
                continue
            displacement = np.linalg.norm(
                data.xpos[object_body_ids[name]] - initial_object_positions[name]
            )
            wrong_object_max_displacement = max(
                wrong_object_max_displacement, float(displacement)
            )

        if step % render_interval == 0:
            render_current = (
                video_frame_index == 0
                or video_frame_index % args.render_skip == 0
            )
            if render_current and writer is not None:
                renderer.update_scene(data, camera=overview_camera)
                last_overview_frame = renderer.render().copy()
            if writer is not None:
                writer.append_data(last_overview_frame)
            if args.record_demonstration:
                records["time_s"].append(time_s)
                records["joint_position_rad"].append(
                    [
                        float(data.qpos[controlled[name]["qpos_id"]])
                        for name in controlled_names
                    ]
                )
                records["joint_velocity_rad_s"].append(
                    [
                        float(data.qvel[controlled[name]["qvel_id"]])
                        for name in controlled_names
                    ]
                )
                records["action_joint_position_rad"].append(
                    [commanded[name] for name in controlled_names]
                )
                records["task_phase"].append(phase)
                records["assist_active"].append(int(grasp_active))
                records["physical_contact_active"].append(
                    int(contacts_this_step > 0)
                )
                records["active_arm_index"].append(active_palm_index)
                records["object_position_m"].append(
                    np.stack(
                        [
                            data.xpos[object_body_ids[name]].copy()
                            for name in OBJECT_NAMES
                        ]
                    )
                )
                records["object_quaternion_wxyz"].append(
                    np.stack(
                        [
                            data.xquat[object_body_ids[name]].copy()
                            for name in OBJECT_NAMES
                        ]
                    )
                )
                records["palm_position_m"].append(
                    np.stack([data.site_xpos[site].copy() for site in palm_ids])
                )
                records["target_object_index"].append(
                    OBJECT_NAMES.index(args.target_object)
                )
                for camera_name, camera_writer in camera_writers.items():
                    if render_current or camera_name not in last_camera_frames:
                        renderer.update_scene(data, camera=camera_name)
                        last_camera_frames[camera_name] = renderer.render().copy()
                    camera_writer.append_data(last_camera_frames[camera_name])
            video_frame_index += 1

    if writer is not None:
        writer.close()
    for camera_writer in camera_writers.values():
        camera_writer.close()
    close_renderer = getattr(renderer, "close", None)
    if close_renderer is not None:
        close_renderer()
    mujoco.mj_forward(model, data)

    final_positions = {
        name: data.xpos[body_id].copy() for name, body_id in object_body_ids.items()
    }
    target_position = final_positions[args.target_object]
    relative_to_bin = target_position - BIN_POSITION_M
    target_in_box = bool(
        abs(relative_to_bin[0]) <= 0.12
        and abs(relative_to_bin[1]) <= 0.16
        and TABLE_TOP_Z_M <= target_position[2] <= TABLE_TOP_Z_M + 0.18
    )
    wrong_objects_in_box = []
    for name in OBJECT_NAMES:
        if name == args.target_object:
            continue
        relative = final_positions[name] - BIN_POSITION_M
        if abs(relative[0]) <= 0.12 and abs(relative[1]) <= 0.16:
            wrong_objects_in_box.append(name)
    violation_fraction = joint_limit_violations / max(joint_samples, 1)
    initial_visual_localization_passed = bool(
        camera_visibility["all_objects_visible"]
        and max(visual_localization_error_m.values()) <= 0.04
    )
    localization_passed = bool(
        initial_visual_localization_passed
        or (
            args.allow_occluded_initial_head_view
            and localization_source == "privileged"
        )
    )
    target_lift_m = (
        target_maximum_height_m
        - float(initial_object_positions[args.target_object][2])
    )
    physical_grasp_passed = bool(
        assisted_constraint_active_steps == 0
        and physical_contact_steps > 0
        and target_lift_m >= 0.10
    )
    grasp_mechanism_passed = (
        physical_grasp_passed
        if args.grasp_mode == "contact"
        else assisted_constraint_active_steps > 0
    )
    passed = bool(
        target_in_box
        and not wrong_objects_in_box
        and wrong_object_max_displacement <= 0.05
        and violation_fraction <= 0.01
        and localization_passed
        and grasp_mechanism_passed
    )
    instruction = args.language_instruction or INSTRUCTIONS[args.target_object]
    summary = {
        "experiment": "G1-Language-Grounded-Manipulation",
        "expert_contract_version": EXPERT_CONTRACT_VERSION,
        "task": "language_conditioned_object_to_box",
        "language_instruction": instruction,
        "target_object": args.target_object,
        "active_arm": active_side,
        "object_order": list(OBJECT_NAMES),
        "object_mass_kg": OBJECT_MASS_KG,
        "object_collision_half_size_m": {
            name: list(size) for name, size in OBJECT_COLLISION_HALF_SIZE_M.items()
        },
        "red_triangle_collision_shape": (
            "convex_triangular_prism"
            if args.red_triangle_mesh_collision
            else "box_approximation"
        ),
        "slot_permutation": list(permutation),
        "slot_offsets_xy_m": [offset.tolist() for offset in offsets],
        "object_initial_position_m": {
            name: value.tolist() for name, value in initial_object_positions.items()
        },
        "object_final_position_m": {
            name: value.tolist() for name, value in final_positions.items()
        },
        "blue_box_position_m": BIN_POSITION_M.tolist(),
        "camera_visibility_audit": camera_visibility,
        "initial_head_visibility_required": not args.allow_occluded_initial_head_view,
        "initial_visual_localization_passed": initial_visual_localization_passed,
        "localization_source": localization_source,
        "grasp_mode": args.grasp_mode,
        "controller_gain_scale": {
            "body": args.body_gain_scale,
            "hand": args.hand_gain_scale,
        },
        "hand_closure_multiplier": args.hand_closure_multiplier,
        "hand_profile_scale": {
            "thumb": hand_profile_values[0],
            "index": hand_profile_values[1],
            "middle": hand_profile_values[2],
        },
        "hand_closure_delay_s": args.hand_closure_delay_s,
        "grasp_yaw_deg": grasp_yaw_deg,
        "grasp_approach_duration_s": args.grasp_approach_duration_s,
        "lift_height_m": args.lift_height_m,
        "lift_duration_s": args.lift_duration_s,
        "lift_hold_duration_s": args.lift_hold_duration_s,
        "initial_lift_stage": {
            "enabled": initial_lift_stage_pose is not None,
            "height_m": args.initial_lift_stage_height_m,
            "duration_s": args.initial_lift_stage_duration_s,
            "hold_s": args.initial_lift_stage_hold_s,
        },
        "transport_duration_s": args.transport_duration_s,
        "high_transport_waypoint": {
            "enabled": bool(args.high_transport_waypoint),
            "high_fraction": args.high_transport_fraction,
        },
        "transport_relative_pose_feedback": {
            "enabled": bool(args.transport_relative_pose_feedback),
            "feedback_hz": args.transport_feedback_hz,
            "feedback_gain": args.transport_feedback_gain,
            "max_correction_m": args.transport_max_correction_m,
            "max_relative_error_gate_m": args.transport_max_relative_error_m,
            "contact_grace_s": args.transport_contact_grace_s,
            "contact_closure_feedback": bool(args.transport_contact_closure_feedback),
            "contact_closure_max_boost": args.transport_contact_closure_max_boost,
            "contact_closure_rate_per_s": args.transport_contact_closure_rate_per_s,
            "contact_closure_decay_per_s": args.transport_contact_closure_decay_per_s,
            "max_applied_contact_closure_boost": transport_max_contact_closure_boost,
            "contact_closure_recovery_steps": transport_contact_closure_recovery_steps,
            "replans": transport_replans,
            "contact_loss_steps": transport_contact_loss_steps,
            "final_progress": transport_progress,
            "final_lift_progress": transport_lift_progress,
            "release_started": transport_release_start_s is not None,
            "max_relative_error_m": transport_max_relative_error_m,
            "max_applied_correction_m": transport_max_applied_correction_m,
            "max_rotation_error_deg": transport_max_rotation_error_deg,
        },
        "initial_state_recovery": {
            "enabled": initial_arm_pose is not None and (
                initial_state_recovery_s > 0.0 or initial_safe_lift_s > 0.0
            ),
            "duration_s": initial_state_recovery_s,
            "safe_lift_s": initial_safe_lift_s,
            "safe_traverse_s": initial_safe_traverse_s,
            "safe_lift_height_m": (
                args.initial_safe_lift_height_m if initial_safe_lift_s > 0.0 else None
            ),
            "home_settle_s": initial_home_settle_s,
            "velocity_mode": args.initial_velocity_mode,
            "source": (
                str(args.initial_upper_state_npz)
                if args.initial_upper_state_npz is not None
                else None
            ),
        },
        "initial_direct_continuation": {
            "enabled": direct_continuation,
            "grasp_duration_s": args.initial_direct_grasp_s if direct_continuation else None,
        },
        "pregrasp_offset_m": [
            0.0,
            lateral_offset,
            args.pregrasp_height_offset_m,
        ],
        "adaptive_regrasp": {
            "enabled": args.adaptive_regrasp,
            "planned": adaptive_regrasp_planned,
            "object_displacement_at_replan_m": (
                adaptive_regrasp_object_displacement_m.tolist()
                if adaptive_regrasp_object_displacement_m is not None
                else None
            ),
            "lift_planned": adaptive_lift_planned,
            "object_displacement_at_lift_replan_m": (
                adaptive_lift_object_displacement_m.tolist()
                if adaptive_lift_object_displacement_m is not None
                else None
            ),
            "transport_object_to_palm_offset_m": (
                adaptive_transport_object_to_palm_offset_m.tolist()
                if adaptive_transport_object_to_palm_offset_m is not None
                else None
            ),
        },
        "contact_grasp_compensation_m": contact_compensation.tolist(),
        "grasp_physics_audit": {
            "assisted_constraint_active_steps": assisted_constraint_active_steps,
            "physical_contact_steps": physical_contact_steps,
            "physical_contact_step_fraction": physical_contact_steps
            / max(total_steps, 1),
            "physical_contact_samples": physical_contact_samples,
            "maximum_contacts_per_step": maximum_contacts_per_step,
            "maximum_contact_normal_force_n": maximum_contact_normal_force_n,
            "first_contact_time_s": first_contact_time_s,
            "last_contact_time_s": last_contact_time_s,
            "contact_pairs": dict(sorted(target_contact_pairs.items())),
            "target_maximum_height_m": target_maximum_height_m,
            "target_lift_m": target_lift_m,
            "physical_grasp_passed": physical_grasp_passed,
            "grasp_mechanism_passed": grasp_mechanism_passed,
        },
        "visual_object_position_estimate_m": {
            name: value.tolist() for name, value in visual_object_positions.items()
        },
        "visual_localization_error_m": visual_localization_error_m,
        "visual_localization_passed": localization_passed,
        "grasp_object_center_offset_m": {
            name: (-OBJECT_ASSIST_OFFSET_M[name]).tolist() for name in OBJECT_NAMES
        },
        "hand_grasp_profile": OBJECT_HAND_CLOSURE_SCALE[args.target_object],
        "active_hand_closed_joint_targets_rad": {
            name: args.hand_closure_multiplier
            * object_hand_target(args.target_object, name, closed)
            for name, closed in HAND_TARGETS.items()
            if name.startswith(f"{active_side}_hand_")
        },
        "target_in_box": target_in_box,
        "wrong_objects_in_box": wrong_objects_in_box,
        "wrong_object_max_displacement_m": wrong_object_max_displacement,
        "joint_limit_violation_fraction": violation_fraction,
        "ik": pose_reports,
        "passed": passed,
        "scene": str(scene_path),
        "video": None if args.disable_overview_video else str(video_path),
    }
    summary_path = args.output_dir / "language_pick_place_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.record_demonstration:
        np.savez_compressed(
            args.output_dir / "expert_pick_place_episode.npz",
            joint_names=np.asarray(controlled_names),
            time_s=np.asarray(records["time_s"], dtype=np.float32),
            joint_position_rad=np.asarray(
                records["joint_position_rad"], dtype=np.float32
            ),
            joint_velocity_rad_s=np.asarray(
                records["joint_velocity_rad_s"], dtype=np.float32
            ),
            action_joint_position_rad=np.asarray(
                records["action_joint_position_rad"], dtype=np.float32
            ),
            task_phase=np.asarray(records["task_phase"], dtype=np.int64),
            assist_active=np.asarray(records["assist_active"], dtype=np.int64),
            physical_contact_active=np.asarray(
                records["physical_contact_active"], dtype=np.int64
            ),
            active_arm_index=np.asarray(records["active_arm_index"], dtype=np.int64),
            object_position_m=np.asarray(
                records["object_position_m"], dtype=np.float32
            ),
            object_quaternion_wxyz=np.asarray(
                records["object_quaternion_wxyz"], dtype=np.float32
            ),
            palm_position_m=np.asarray(records["palm_position_m"], dtype=np.float32),
            target_object_index=np.asarray(
                records["target_object_index"], dtype=np.int64
            ),
            metadata_json=np.asarray(json.dumps(summary)),
        )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved {summary_path}", flush=True)
    return summary


def main() -> None:
    summary = run_episode(parse_args())
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
