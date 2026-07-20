#!/usr/bin/env python3
"""Create and validate a project-owned G1 asset with task cameras."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


CAMERAS = {
    "head_camera": {
        "body": "torso_link",
        "pos": "0.08 0 0.42",
        "xyaxes": "0 -1 0 0 0 1",
        "fovy": "70",
    },
    "left_wrist_camera": {
        "body": "left_wrist_yaw_link",
        "pos": "0.04 0 0.11",
        "xyaxes": "0 -1 0 0 0 1",
        "fovy": "85",
    },
    "right_wrist_camera": {
        "body": "right_wrist_yaw_link",
        "pos": "0.04 0 0.11",
        "xyaxes": "0 -1 0 0 0 1",
        "fovy": "85",
    },
}


def find_body(root: ET.Element, name: str) -> ET.Element:
    body = root.find(f".//body[@name='{name}']")
    if body is None:
        raise RuntimeError(f"Body not found: {name}")
    return body


def add_calibration_scene(root: ET.Element) -> None:
    worldbodies = root.findall("worldbody")
    scene = worldbodies[-1]
    ET.SubElement(
        scene,
        "geom",
        name="calibration_table",
        type="box",
        pos="1.10 0 0.72",
        size="0.35 0.55 0.03",
        rgba="0.35 0.37 0.40 1",
    )
    ET.SubElement(
        scene,
        "geom",
        name="blue_tote",
        type="box",
        pos="0.95 0 0.88",
        size="0.16 0.22 0.13",
        rgba="0.08 0.35 0.85 1",
    )
    ET.SubElement(
        scene,
        "geom",
        name="left_marker",
        type="sphere",
        pos="0.78 0.28 1.02",
        size="0.06",
        rgba="0.9 0.15 0.12 1",
    )
    ET.SubElement(
        scene,
        "geom",
        name="right_marker",
        type="sphere",
        pos="0.78 -0.28 1.02",
        size="0.06",
        rgba="0.12 0.8 0.25 1",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-asset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_asset.resolve()
    tree = ET.parse(source)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("MJCF compiler element is missing")
    compiler.set("meshdir", str(source.parent / "meshes"))

    for camera_name, config in CAMERAS.items():
        body = find_body(root, config["body"])
        ET.SubElement(
            body,
            "camera",
            name=camera_name,
            pos=config["pos"],
            xyaxes=config["xyaxes"],
            fovy=config["fovy"],
        )
    add_calibration_scene(root)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    asset_path = args.output_dir / "g1_warehouse_cameras.xml"
    tree.write(asset_path, encoding="unicode", xml_declaration=False)

    model = mujoco.MjModel.from_xml_path(str(asset_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)

    image_checks = {}
    for camera_name in CAMERAS:
        renderer.update_scene(data, camera=camera_name)
        frame = renderer.render()
        image_path = args.output_dir / f"{camera_name}.png"
        Image.fromarray(frame).save(image_path)
        pixel_std = float(np.std(frame))
        blue_pixels = int(
            np.count_nonzero(
                (frame[:, :, 2] > 100)
                & (frame[:, :, 2] > frame[:, :, 0] * 1.3)
                & (frame[:, :, 2] > frame[:, :, 1] * 1.15)
            )
        )
        image_checks[camera_name] = {
            "path": str(image_path),
            "pixel_standard_deviation": pixel_std,
            "nonblank": pixel_std > 5.0,
            "blue_tote_pixels": blue_pixels,
            "calibration_target_visible": blue_pixels > 500,
        }
    renderer.close()

    camera_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
        for index in range(model.ncam)
    ]
    report = {
        "experiment": "G1WH-02-three-camera-interface",
        "source_asset": str(source),
        "derived_asset": str(asset_path.resolve()),
        "camera_count": model.ncam,
        "camera_names": camera_names,
        "resolution": [640, 480],
        "image_checks": image_checks,
        "all_cameras_present": set(camera_names) == set(CAMERAS),
        "all_images_nonblank": all(item["nonblank"] for item in image_checks.values()),
        "target_visible_in_all_cameras": all(
            item["calibration_target_visible"] for item in image_checks.values()
        ),
    }
    report_path = args.output_dir / "camera_audit.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
