"""Causal forward-kinematic samples of G1's physical finger contact capsules."""

from __future__ import annotations

import mujoco
import numpy as np


# These are the collision bodies that dominate the strict-grasp contact audit.
# Each capsule contributes both endpoints and its center, rather than collapsing
# the hand to a palm center or a single pad.
FINGER_CONTACT_LINKS = (
    "hand_thumb_1_link_contact",
    "hand_thumb_2_link_contact",
    "hand_middle_0_link_contact",
    "hand_middle_1_link_contact",
    "hand_index_0_link_contact",
    "hand_index_1_link_contact",
)
SIDES = ("left", "right")
SAMPLES_PER_CAPSULE = 3


def finger_contact_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_{link}")
        for side in SIDES
        for link in FINGER_CONTACT_LINKS
    ]
    if min(ids) < 0:
        raise ValueError("Scene is missing one or more physical finger contact capsules")
    return ids


def finger_contact_sites(
    model: mujoco.MjModel, data: mujoco.MjData, geom_ids: list[int]
) -> np.ndarray:
    """Return world-space endpoint/center samples in stable left-then-right order."""
    result = np.empty((len(geom_ids), SAMPLES_PER_CAPSULE, 3), dtype=np.float32)
    for index, geom_id in enumerate(geom_ids):
        center = data.geom_xpos[geom_id]
        # MuJoCo aligns the local z axis of a capsule with its fromto segment.
        axis = data.geom_xmat[geom_id].reshape(3, 3)[:, 2]
        half_length = float(model.geom_size[geom_id, 1])
        result[index, 0] = center - half_length * axis
        result[index, 1] = center
        result[index, 2] = center + half_length * axis
    return result.reshape(-1)


def finger_contact_site_dim(model: mujoco.MjModel) -> int:
    return len(finger_contact_geom_ids(model)) * SAMPLES_PER_CAPSULE * 3
