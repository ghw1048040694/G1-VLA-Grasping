"""Pure scene construction migrated from the physical G1 task implementation.

There are no CLI, optimizer, environment-variable or simulator side effects.
The red visual prism and yellow cylinder retain their historical collision
approximations unless explicitly stated by the generated scene manifest.
"""
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping, Sequence

from ..control import FIXED_BASE_EQUALITY


class SceneError(ValueError):
    """A source scene or requested output violates the strict scene contract."""


OBJECT_NAMES = ('red_triangle', 'yellow_rod', 'green_cube')


EXPERT_CONTRACT_VERSION = 'g1lang_v4_contact_grasp'


OBJECT_LABELS = {'red_triangle': 'red triangular prism', 'yellow_rod': 'yellow rod', 'green_cube': 'green cube'}


INSTRUCTIONS = {name: f'put the {label} into the blue box' for (name, label) in OBJECT_LABELS.items()}


PALM_NAMES = ('left_palm_center', 'right_palm_center')


LEFT_PALM_INDEX = 0


RIGHT_PALM_INDEX = 1


HOME_POSITIONS_M = ((0.22, 0.32, 1.0), (0.22, -0.32, 1.0))


WAIST_PITCH_LIMIT_RAD = 0.1


TABLE_TOP_Z_M = 0.75


BIN_POSITION_M = (0.59, 0.0, TABLE_TOP_Z_M + 0.01)


OBJECT_SLOTS_XY_M = ((0.36, -0.34), (0.34, -0.1), (0.36, 0.2))


OBJECT_ASSIST_OFFSET_M = {'red_triangle': (0.02, 0.0, 0.025), 'yellow_rod': (0.005, 0.0, 0.02), 'green_cube': (0.02, 0.0, 0.0)}


OBJECT_RGB_LOCALIZATION_BIAS_M = {'red_triangle': (0.007, 0.0, 0.0), 'yellow_rod': (0.004, 0.004, 0.0), 'green_cube': (0.001, 0.0, 0.0)}


OBJECT_HAND_CLOSURE_SCALE = {'red_triangle': {'thumb': 0.46, 'index': 0.42, 'middle': 0.32}, 'yellow_rod': {'thumb': 0.56, 'index': 0.58, 'middle': 0.58}, 'green_cube': {'thumb': 0.5, 'index': 0.5, 'middle': 0.5}}


OBJECT_GRASP_YAW_DEG = {'red_triangle': 0.0, 'yellow_rod': 0.0, 'green_cube': 45.0}


OBJECT_MASS_KG = {'red_triangle': 0.12, 'yellow_rod': 0.1, 'green_cube': 0.08}


OBJECT_COLLISION_HALF_SIZE_M = {'red_triangle': (0.045, 0.04, 0.03), 'yellow_rod': (0.06, 0.025, 0.025), 'green_cube': (0.025, 0.025, 0.025)}


CONTACT_GRASP_COMPENSATION_M = (-0.015, 0.03, -0.055)


HEAD_CAMERA_XYAXES = '0 -1 0 0.643 0 0.766'


HEAD_CAMERA_FOVY = '85'


OBJECT_SPAWN_Z_M = {'red_triangle': TABLE_TOP_Z_M + 0.03, 'yellow_rod': TABLE_TOP_Z_M + 0.025, 'green_cube': TABLE_TOP_Z_M + 0.045}


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
    position: Sequence[float],
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


def migrate_actuator_force_ranges(root: ET.Element):
    """Transfer legacy joint force bounds to their actual actuator elements.

    The old helper dropped these attributes while the source motors had no
    force ranges. This is an explicit physics change; each transfer is logged.
    """
    legacy_ranges = {}
    for joint in root.iter("joint"):
        if joint.get("actuatorfrcrange") is not None:
            name, bounds = joint.get("name"), joint.get("actuatorfrcrange")
            if not name or name in legacy_ranges:
                raise SceneError("legacy joint force ranges require unique named joints")
            values = [float(value) for value in bounds.split()]
            if len(values) != 2 or not all(math.isfinite(value) for value in values) or values[0] >= values[1]:
                raise SceneError("invalid legacy force range for {}".format(name))
            legacy_ranges[name] = bounds
    records, transferred = [], set()
    for actuator_parent in root.findall("actuator"):
        for actuator in actuator_parent:
            joint_name = actuator.get("joint")
            if joint_name not in legacy_ranges:
                continue
            old_range = actuator.get("forcerange")
            if old_range is None:
                actuator.set("forcerange", legacy_ranges[joint_name])
                actuator.set("forcelimited", "true")
            records.append({"actuator": actuator.get("name"), "joint": joint_name,
                            "legacy_joint_range": legacy_ranges[joint_name],
                            "old_actuator_range": old_range, "new_actuator_range": actuator.get("forcerange"),
                            "changed": old_range is None})
            transferred.add(joint_name)
    if set(legacy_ranges) != transferred:
        raise SceneError("legacy joint force bounds have no corresponding actuator: {}".format(
            ", ".join(sorted(set(legacy_ranges) - transferred))))
    for joint in root.iter("joint"):
        joint.attrib.pop("actuatorfrcrange", None)
        joint.attrib.pop("actuatorfrclimited", None)
    return records


def assert_fixed_scene(root: ET.Element):
    equality = root.find("equality")
    constraints = list(equality) if equality is not None else []
    if len(constraints) != 1:
        raise SceneError("strict fixed stance permits only its pelvis-to-world base weld")
    base = constraints[0]
    if (base.tag != "weld" or base.get("name") != FIXED_BASE_EQUALITY or base.get("body1") != "pelvis"
            or base.get("body2") not in (None, "world") or base.get("active", "true") not in ("true", "1")):
        raise SceneError("source must retain an active warehouse_fixed_base pelvis-to-world weld")
    if root.find(".//body[@name='pelvis']") is None:
        raise SceneError("source scene has no pelvis body")
    return base


def _within(path: Path, parent: Path):
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _contact_values(geom):
    """Resolve this strict source's explicit geom attributes + MuJoCo defaults.

    Default classes are rejected by the caller rather than approximated here.
    """
    def values(key, default, count):
        try:
            result = [float(item) for item in geom.get(key, default).split()]
        except ValueError as exc:
            raise SceneError("invalid original geom contact parameter " + key) from exc
        if len(result) != count or not all(math.isfinite(item) for item in result):
            raise SceneError("unsupported original geom contact parameter " + key)
        return result
    try:
        integers = {key: int(geom.get(key, default)) for key, default in
                    (("contype", "1"), ("conaffinity", "1"), ("condim", "3"), ("priority", "0"))}
    except ValueError as exc:
        raise SceneError("invalid original integer contact setting") from exc
    return dict(integers, friction=values("friction", "1 0.005 0.0001", 3),
                solref=values("solref", "0.02 1", 2), solimp=values("solimp", "0.9 0.95 0.001 0.5 2", 5),
                solmix=values("solmix", "1", 1)[0], margin=values("margin", "0", 1)[0], gap=values("gap", "0", 1)[0])


def _mixed_contact_values(first, second):
    """MuJoCo 3.2.5 positive-format dynamic mixing, before our sole change."""
    if first["solref"] != [0.02, 1.] or second["solref"] != [0.02, 1.]:
        raise SceneError("compliance v2 must branch from the declared default solref 0.02 1")
    if first["priority"] != second["priority"]:
        preferred = first if first["priority"] > second["priority"] else second
        condim, friction, solimp = preferred["condim"], preferred["friction"], preferred["solimp"]
    else:
        if first["solmix"] <= 0 or second["solmix"] <= 0:
            raise SceneError("unsupported zero/negative source solmix")
        weight = first["solmix"] / (first["solmix"] + second["solmix"])
        condim = max(first["condim"], second["condim"])
        friction = [max(a, b) for a, b in zip(first["friction"], second["friction"])]
        solimp = [weight * a + (1. - weight) * b for a, b in zip(first["solimp"], second["solimp"])]
    return {"condim": condim, "friction": [friction[0], friction[0], friction[1], friction[2], friction[2]],
            "solref": [0.02, 1.], "solreffriction": [0., 0.], "solimp": list(solimp),
            "margin": max(first["margin"], second["margin"]), "gap": max(first["gap"], second["gap"])}


def _apply_contact_model(root, settings):
    """Add only declared, already-enabled right-hand/green pair overrides.

    Explicit pairs bypass MuJoCo's ordinary mask/exclude filters. Accordingly
    this builder rejects unknown inheritance, exclusions, or source pairs and
    selects only the original mask-compatible separate free-object bodies.
    """
    if settings is None:
        return None
    if not isinstance(settings, dict) or set(settings) != {"backend", "time_constant_s", "damping_ratio"}:
        raise SceneError("contact_model requires backend, time_constant_s and damping_ratio")
    backend, time_constant, ratio = settings["backend"], settings["time_constant_s"], settings["damping_ratio"]
    if (backend not in ("source_default_v1", "right_hand_green_compliance_v2")
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                   for value in (time_constant, ratio)) or time_constant <= 0 or ratio != 1.):
        raise SceneError("unsupported contact model or noncritical/nonfinite solver reference")
    option = root.find("option")
    timestep = float(option.get("timestep", "0.002")) if option is not None else .002
    if backend == "source_default_v1":
        if time_constant != .02:
            raise SceneError("source_default_v1 preserves solref 0.02 1")
        return {"backend": backend, "solref": [.02, 1.], "explicit_pairs": [],
                "scope": "original dynamic geom mixing; XML unchanged"}
    if not math.isfinite(timestep) or timestep <= 0 or time_constant < 3. * timestep:
        raise SceneError("compliance candidate time constant must be at least three physics timesteps")
    if root.find("default") is not None:
        raise SceneError("compliance candidate cannot infer inherited default contact classes")
    contact = root.find("contact")
    if contact is not None and contact.findall("pair"):
        raise SceneError("compliance candidate requires no pre-existing explicit source pairs")
    green = root.find(".//body[@name='green_cube']")
    if green is None or green.find("freejoint") is None:
        raise SceneError("compliance candidate requires the original independent green free body")
    cube = green.find("geom[@name='green_cube_geom']")
    if cube is None:
        raise SceneError("green collision geom is absent")
    cube_values = _contact_values(cube)
    finger_bodies = ["right_hand_thumb_{}_link".format(i) for i in range(3)] + [
        "right_hand_{}_{}_link".format(finger, i) for finger in ("index", "middle") for i in range(2)]
    selected, named = [], []
    for body_name in finger_bodies + ["right_wrist_yaw_link"]:
        body = root.find(".//body[@name='{}']".format(body_name))
        if body is None:
            raise SceneError("compliance source finger/palm body is absent: " + body_name)
        if contact is not None:
            for exclusion in contact.findall("exclude"):
                if {exclusion.get("body1"), exclusion.get("body2")} == {body_name, "green_cube"}:
                    raise SceneError("explicit compliance pair cannot bypass an original body exclusion")
        for index, geom in enumerate(body.findall("geom")):
            if body_name == "right_wrist_yaw_link" and geom.get("name") != "right_palm_contact_pad":
                continue
            values = _contact_values(geom)
            compatible = bool(values["contype"] & cube_values["conaffinity"] or cube_values["contype"] & values["conaffinity"])
            if not compatible:
                continue
            if geom.get("name") is None:
                alias = body_name + "_compliance_source_geom_{}".format(index)
                if root.find(".//geom[@name='{}']".format(alias)) is not None:
                    raise SceneError("candidate geom alias conflicts with source name")
                geom.set("name", alias)
                named.append(alias)
            selected.append((body_name, geom, values))
    if len(selected) != 15 or len(named) != 7:
        raise SceneError("compliance scope must preserve the original seven mesh/capsule links plus palm pad")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    records = []
    for body_name, geom, values in selected:
        original = _mixed_contact_values(values, cube_values)
        candidate = dict(original, solref=[float(time_constant), float(ratio)])
        pair_name = "right_green_compliance_" + geom.get("name")
        attributes = {"name": pair_name, "geom1": geom.get("name"), "geom2": cube.get("name")}
        for key, value in candidate.items():
            attributes[key] = " ".join(format(item, ".17g") for item in value) if isinstance(value, list) else format(value, ".17g")
        ET.SubElement(contact, "pair", **attributes)
        records.append({"name": pair_name, "hand_body": body_name, "geom1": geom.get("name"), "geom2": cube.get("name"),
                        "original_hand_geom_parameters": values, "original_green_geom_parameters": cube_values,
                        "original_dynamic_contact_parameters": original, "compiled_expected_parameters": candidate,
                        "original_dynamic_mask_compatible": True,
                        "body_filter_scope": "different non-welded trees: free green body versus articulated right hand; no direct-body exclusion"})
    return {"backend": backend, "solref": [float(time_constant), float(ratio)], "physics_timestep_s": timestep,
            "time_constant_in_physics_steps": time_constant / timestep,
            "explicit_pairs": records, "pair_count": len(records), "new_names_for_original_unnamed_active_geoms": named,
            "scope": "right seven original active finger mesh/capsule geoms and palm pad versus original green geom only",
            "only_changed_contact_parameter": "solref time constant", "cube_table_contact_unchanged": True,
            "geometry_masks_friction_mass_actuators_pd_unchanged": True,
            "uncalibrated_simulation_hypothesis": True,
            "calibration_scope": "candidate numerical compliance; no hardware material/contact calibration",
            "default_solref": [.02, 1.],
            "reference_sources": ["https://github.com/google-deepmind/mujoco/blob/3.2.5/doc/modeling.rst#contact-parameters",
                                  "https://github.com/google-deepmind/mujoco/blob/3.2.5/src/engine/engine_collision_driver.c"]}


def build_scene(source: Path, destination: Path, positions: Mapping[str, Sequence[float]], *,
                run_dir: Path, preserve_source_camera: bool = False,
                red_triangle_mesh_collision: bool = False, include_blue_box: bool = True,
                contact_model=None):
    """Create a new run-local XML and return its explicit geometry manifest.

    The source is never rewritten. A green-only positions dictionary creates
    only the current green task; later language stages can pass all three.
    """
    source, destination, run_dir = Path(source).resolve(), Path(destination).resolve(), Path(run_dir).resolve()
    if not run_dir.is_dir() or not _within(destination, run_dir):
        raise SceneError("destination must be inside an existing run directory")
    if destination == source or _within(destination, source.parent):
        raise SceneError("generated XML cannot be written into the source resource directory")
    if destination.exists() or destination.is_symlink():
        raise SceneError("generated scene must be a new file")
    if destination.suffix.lower() != ".xml":
        raise SceneError("scene destination must have an .xml extension")
    if not all(isinstance(value, bool) for value in (preserve_source_camera, red_triangle_mesh_collision, include_blue_box)):
        raise SceneError("scene options must be explicit booleans")
    if not isinstance(positions, dict) or not positions or set(positions) - set(OBJECT_NAMES):
        raise SceneError("positions must name one or more known task objects")
    normalized = {}
    for name, coordinate in positions.items():
        coordinate = coordinate.tolist() if hasattr(coordinate, "tolist") else coordinate
        if not isinstance(coordinate, (list, tuple)) or len(coordinate) != 3:
            raise SceneError("{} position must contain three metric coordinates".format(name))
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in coordinate):
            raise SceneError("{} position must contain finite metric coordinates".format(name))
        normalized[name] = tuple(float(value) for value in coordinate)
    try:
        source_bytes = source.read_bytes()
        root = ET.fromstring(source_bytes)
    except (OSError, ET.ParseError) as exc:
        raise SceneError("cannot read source XML: {}".format(exc)) from exc
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("autolimits", "true")
    # Preserve source-relative resource directories when relocating the XML.
    for attribute in ("meshdir", "texturedir", "assetdir"):
        value = compiler.get(attribute)
        if value and not Path(value).is_absolute():
            compiler.set(attribute, str((source.parent / value).resolve()))
    force_transfers = migrate_actuator_force_ranges(root)
    remove_old_task(root)
    assert_fixed_scene(root)
    if "red_triangle" in normalized:
        ensure_triangle_mesh(root)
    worlds = root.findall("worldbody")
    if not worlds:
        raise SceneError("source scene has no worldbody")
    scene, equality = worlds[-1], root.find("equality")
    table = root.find(".//geom[@name='calibration_table']")
    if table is None:
        raise SceneError("source scene has no calibration table")
    table.set("pos", "0.52 0 0.72")
    table.set("size", "0.42 0.55 0.03")
    head_camera = root.find(".//camera[@name='head_camera']")
    if head_camera is None:
        raise SceneError("source scene has no head_camera")
    if not preserve_source_camera:
        head_camera.set("xyaxes", HEAD_CAMERA_XYAXES)
        head_camera.set("fovy", HEAD_CAMERA_FOVY)
    replace_hand_collision_hulls(root)
    if include_blue_box:
        add_blue_box(scene)
    for name in OBJECT_NAMES:
        if name in normalized:
            add_free_object(scene, equality, name, normalized[name], red_triangle_mesh_collision=red_triangle_mesh_collision)
    assert_fixed_scene(root)
    contact_manifest = _apply_contact_model(root, contact_model)
    xml_bytes = ET.tostring(root, encoding="utf-8", xml_declaration=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(xml_bytes)
    approximations = {
        "red_triangle": {"visual": "triangular_prism_mesh", "collision": "convex_mesh" if red_triangle_mesh_collision else "box_approximation"},
        "yellow_rod": {"visual": "cylinder", "collision": "box_approximation"},
        "green_cube": {"visual": "box", "collision": "same_box"},
    }
    manifest = {
        "schema_version": 1, "scene_protocol": "g1_strict_scene_force_transfer_v1",
        "source": str(source), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "scene": str(destination), "scene_sha256": hashlib.sha256(xml_bytes).hexdigest(),
        "fixed_base_equality": FIXED_BASE_EQUALITY, "assisted_grasp": False,
        "object_names": [name for name in OBJECT_NAMES if name in normalized],
        "positions_m": {name: list(value) for name, value in normalized.items()},
        "preserve_source_camera": preserve_source_camera,
        "head_camera": {"xyaxes": head_camera.get("xyaxes"), "fovy": head_camera.get("fovy")},
        "visual_collision": {name: approximations[name] for name in normalized},
        "historical_unresolved_approximations": approximations,
        "hand_collision": "palm_box_and_finger_capsules_plus_one_convex_hull_per_finger_link",
        "hand_contact_masks": {"contype": 2, "conaffinity": 4},
        "tactile_collision_scope": "all_generated_external_contacts; legacy masks restrict which pairs are generated",
        "actuator_force_transfers": force_transfers,
        "legacy_equivalence": False,
        "changes_from_legacy": ["legacy joint force bounds explicitly transferred to actuators", "single-object scene permitted"],
        "requires_teacher_revalidation": True,
    }
    if contact_manifest is not None:
        from ..contracts import scene_protocol_for_contact_model
        manifest["scene_protocol"] = scene_protocol_for_contact_model(contact_model)
        manifest["contact_model"] = contact_manifest
    return manifest
