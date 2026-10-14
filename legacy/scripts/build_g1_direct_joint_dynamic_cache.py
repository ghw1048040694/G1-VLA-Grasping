#!/usr/bin/env python3
"""Build temporal direct-joint VLA supervision from strict dynamic rollouts.

Each row is an actually executed MuJoCo joint target. The VLA input retains
time-aligned RGB colour geometry from the head and both wrist cameras, language,
and state31. MuJoCo object truth remains an offline audit label only.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from g1_geometric_contact_vla import calibrated_object_positions
from g1_finger_contact_geometry import finger_contact_geom_ids, finger_contact_sites
from run_g1_language_pick_place import OBJECT_NAMES


TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
CANDIDATE_OFFSETS = np.asarray((0, 2, 4, 6), dtype=np.int64)
CAMERA_VIDEO_NAMES = (
    "head_camera.mp4",
    "left_wrist_camera.mp4",
    "right_wrist_camera.mp4",
)
HEAD_VISIBILITY_AREA = np.asarray((0.0013020833, 0.00390625, 0.0013020833))


@dataclass
class Episode:
    name: str
    source_root: str
    task: str
    target: int
    state: np.ndarray
    velocity: np.ndarray
    palms: np.ndarray
    finger_contact_sites: np.ndarray
    action: np.ndarray
    contact: np.ndarray
    phase: np.ndarray
    active_arm: np.ndarray
    color_features: np.ndarray
    multiview_color_features: np.ndarray
    visual_objects: np.ndarray
    visual_latch_frame: int
    true_objects: np.ndarray
    valid_frames: int
    counterfactual_initial_no_contact: bool
    contact_prefix_recovery: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        required=True,
        help="Root containing strict expert episode directories; may be repeated.",
    )
    parser.add_argument(
        "--recovery-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Strict on-policy recovery root. All of its episodes are appended "
            "to the training split, never the validation split."
        ),
    )
    parser.add_argument(
        "--contact-prefix-recovery-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Failed full-recovery roots from which only a verified contiguous "
            "contact-and-lift prefix is used as corrective supervision."
        ),
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-frames", type=int, default=176)
    parser.add_argument(
        "--recovery-skip-prefix-frames",
        type=int,
        default=0,
        help=(
            "Discard this many reset/home-settle frames from each recovery before "
            "using it as VLA supervision. Base expert episodes are never trimmed."
        ),
    )
    parser.add_argument(
        "--recovery-supervision-frames",
        type=int,
        default=0,
        help=(
            "When positive, supervise only this many initial frames from each "
            "strict recovery. This concentrates DAgger weight on the observed "
            "policy state and its corrective commands; base experts are never trimmed."
        ),
    )
    parser.add_argument("--validation-stride", type=int, default=6)
    parser.add_argument(
        "--recovery-repeat",
        type=int,
        default=1,
        help="Training-only multiplicity for strict on-policy recovery episodes.",
    )
    parser.add_argument(
        "--allow-contact-prefix-recovery",
        action="store_true",
        help=(
            "Accept only the verified contact-and-lift prefix of a failed full "
            "recovery. The invalid transport tail is masked from supervision."
        ),
    )
    parser.add_argument(
        "--contact-prefix-repeat",
        type=int,
        default=1,
        help="Training-only multiplicity for each verified contact-recovery prefix.",
    )
    parser.add_argument(
        "--action-representation",
        choices=("absolute_joint_position", "relative_joint_delta"),
        default="absolute_joint_position",
        help=(
            "Store expert actions as absolute joint targets or as target-minus-"
            "current-state deltas. The selected representation is carried into "
            "both the VLA and world-model checkpoints."
        ),
    )
    return parser.parse_args()


def color_features(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32) / 255.0
    red = (value[..., 0] > 0.47) & (value[..., 0] > value[..., 1] * 1.25) & (value[..., 0] > value[..., 2] * 1.25)
    yellow = (value[..., 0] > 0.47) & (value[..., 1] > 0.35) & (value[..., 2] < 0.47)
    green = (value[..., 1] > 0.31) & (value[..., 1] > value[..., 0] * 1.25) & (value[..., 1] > value[..., 2] * 1.15)
    height, width = image.shape[:2]
    x_grid = np.arange(width, dtype=np.float32)[None, :]
    y_grid = np.arange(height, dtype=np.float32)[:, None]
    result = []
    for mask in (red, yellow, green):
        count = max(float(mask.sum()), 1.0)
        result.extend(
            (
                float((mask * x_grid).sum() / count / width),
                float((mask * y_grid).sum() / count / height),
                count / float(height * width),
            )
        )
    return np.asarray(result, dtype=np.float32)


def strict_summary(path: Path) -> dict | None:
    summary_path = path / "language_pick_place_summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    audit = summary.get("grasp_physics_audit", {})
    if not (
        summary.get("passed")
        and summary.get("grasp_mode") == "contact"
        and audit.get("assisted_constraint_active_steps") == 0
        and audit.get("physical_contact_steps", 0) > 0
    ):
        return None
    return summary


def recovery_summary(path: Path) -> dict | None:
    summary_path = path / "language_pick_place_summary.json"
    if not summary_path.is_file():
        return None
    return json.loads(summary_path.read_text())


def audited_contact_prefix_frames(
    directory: Path, summary: dict, sequence_frames: int, start_frame: int
) -> int | None:
    """Return safe recovery rows ending before a verified grasp is lost.

    This deliberately does not accept a partial recovery as a completed task.
    It supplies only the contact-recovery transition; completed strict experts
    remain the sole supervision source for transport and placement.
    """
    audit = summary.get("grasp_physics_audit", {})
    if not (
        summary.get("grasp_mode") == "contact"
        and audit.get("assisted_constraint_active_steps") == 0
        and int(audit.get("physical_contact_steps", 0)) > 0
        and float(audit.get("target_lift_m", 0.0)) >= 0.10
    ):
        return None
    target_name = summary.get("target_object")
    if target_name not in OBJECT_NAMES:
        return None
    with np.load(directory / "expert_pick_place_episode.npz") as arrays:
        contact = arrays["physical_contact_active"].astype(bool)
        object_positions = arrays["object_position_m"].astype(np.float32)
    if object_positions.ndim != 3 or object_positions.shape[1:] != (len(OBJECT_NAMES), 3):
        return None
    first = int(np.flatnonzero(contact[start_frame:])[0] + start_frame) if np.any(contact[start_frame:]) else -1
    # The recovery state is the first model-visited state. Later incidental
    # contact cannot provide a corrective action for that state.
    if first < start_frame or first > start_frame + 1:
        return None
    last = first
    while last + 1 < len(contact) and contact[last + 1]:
        last += 1
    target = OBJECT_NAMES.index(target_name)
    lift = object_positions[start_frame : last + 1, target, 2].max() - object_positions[start_frame, target, 2]
    if float(lift) < 0.10:
        return None
    # Every learned candidate needs its t+6 label to remain inside the valid,
    # physically verified contact prefix.
    rows = last - start_frame + 1 - int(CANDIDATE_OFFSETS.max())
    return int(min(rows, sequence_frames)) if rows >= 16 else None


def episode_paths(root: Path) -> list[Path]:
    result = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not (path / "expert_pick_place_episode.npz").is_file():
            continue
        result.append(path)
    return result


def read_color_sequence(path: Path, frames: int, start_frame: int = 0) -> np.ndarray:
    reader = imageio.get_reader(path)
    try:
        values = []
        for index, frame in enumerate(reader):
            if index < start_frame:
                continue
            if len(values) >= frames:
                break
            image = np.asarray(frame).copy()
            if image.ndim != 3 or image.shape[2] < 3:
                raise ValueError(f"Expected RGB frame in {path}, found {image.shape}")
            values.append(color_features(image[..., :3]))
    finally:
        reader.close()
    if len(values) < frames:
        raise ValueError(f"Expected {frames} RGB frames in {path}, found {len(values)}")
    return np.stack(values).astype(np.float32)


def head_objects_visible(feature: np.ndarray) -> bool:
    """Match the RGB area thresholds used by the head-camera task audit."""
    return bool(np.all(feature[[2, 5, 8]] >= HEAD_VISIBILITY_AREA))


def latched_head_geometry(
    head_features: np.ndarray, calibration: torch.Tensor
) -> tuple[np.ndarray, int]:
    visible = np.flatnonzero(
        [head_objects_visible(feature) for feature in head_features]
    )
    if not len(visible):
        raise ValueError("No head-camera frame contains all three task objects")
    # A deployed policy cannot inspect future frames to choose a nicer image.
    # Lock the first complete head view, then hold that observed geometry.
    latch_frame = int(visible[0])
    with torch.no_grad():
        geometry = calibrated_object_positions(
            torch.from_numpy(head_features[latch_frame : latch_frame + 1]), calibration
        )[0].cpu().numpy().astype(np.float32).reshape(9)
    # Before an occluded initial pose clears, geometry is intentionally marked
    # unavailable. It is not replaced with object truth or a hallucinated ray.
    result = np.zeros((len(head_features), 9), dtype=np.float32)
    result[latch_frame:] = geometry
    return result, latch_frame


def finger_contact_site_sequence(
    directory: Path, joint_names: np.ndarray, joint_positions: np.ndarray
) -> np.ndarray:
    """Forward recorded state to sampled physical finger collision geometry."""
    scene = directory / "g1_language_pick_place.xml"
    if not scene.is_file():
        raise FileNotFoundError(f"Missing MuJoCo scene for contact geometry: {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos_ids = []
    for name in joint_names.tolist():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"Unknown recorded joint {name} in {directory}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
    geom_ids = finger_contact_geom_ids(model)
    result = np.empty((len(joint_positions), len(geom_ids) * 9), dtype=np.float32)
    for frame, values in enumerate(joint_positions):
        data.qpos[qpos_ids] = values
        mujoco.mj_forward(model, data)
        result[frame] = finger_contact_sites(model, data, geom_ids)
    return result


def load_episode(
    directory: Path,
    source_root: Path,
    calibration: torch.Tensor,
    sequence_frames: int,
    start_frame: int = 0,
    summary: dict | None = None,
    valid_frames: int | None = None,
    counterfactual_initial_no_contact: bool = False,
    contact_prefix_recovery: bool = False,
) -> Episode | None:
    summary = strict_summary(directory) if summary is None else summary
    if summary is None:
        return None
    with np.load(directory / "expert_pick_place_episode.npz") as arrays:
        joint_names = arrays["joint_names"]
        joint_positions = arrays["joint_position_rad"].astype(np.float32)
        state = joint_positions[:, 12:]
        velocity = arrays["joint_velocity_rad_s"][:, 12:].astype(np.float32)
        palms = arrays["palm_position_m"].reshape(len(state), 6).astype(np.float32)
        action = arrays["action_joint_position_rad"][:, 12:].astype(np.float32)
        contact = arrays["physical_contact_active"].astype(np.float32)
        phase = arrays["task_phase"].astype(np.int64)
        active_arm = arrays["active_arm_index"].astype(np.int64)
        true_objects = arrays["object_position_m"].reshape(len(state), 9).astype(np.float32)
    if not (
        state.shape[1] == velocity.shape[1] == action.shape[1] == 31
        and palms.shape[1] == 6
    ):
        raise ValueError(f"Unexpected direct action shapes in {directory}")
    final_frame = start_frame + sequence_frames + int(CANDIDATE_OFFSETS.max())
    if len(state) < final_frame:
        raise ValueError(f"Episode {directory} is shorter than requested direct-action sequence")
    window = slice(start_frame, start_frame + sequence_frames)
    contact_sites = finger_contact_site_sequence(
        directory, joint_names, joint_positions[window]
    )
    view_features = []
    for name in CAMERA_VIDEO_NAMES:
        video = directory / name
        if not video.is_file():
            raise FileNotFoundError(f"Missing camera video: {video}")
        view_features.append(read_color_sequence(video, sequence_frames, start_frame))
    feature = view_features[0]
    multiview = np.concatenate(view_features, axis=-1)
    visual, latch_frame = latched_head_geometry(feature, calibration)
    target_name = summary["target_object"]
    target = OBJECT_NAMES.index(target_name)
    task = summary["language_instruction"]
    if valid_frames is None:
        valid_frames = sequence_frames
    if not 1 <= valid_frames <= sequence_frames:
        raise ValueError(f"Invalid supervised frame count for {directory}: {valid_frames}")
    contact_window = contact[window].copy()
    if counterfactual_initial_no_contact:
        # The source policy observation had no real hand-object contact. The
        # first teacher action is retained as its counterfactual correction.
        contact_window[0] = 0.0
    return Episode(
        name=directory.name,
        source_root=str(source_root),
        task=task,
        target=target,
        state=state[window],
        velocity=velocity[window],
        palms=palms[window],
        finger_contact_sites=contact_sites,
        action=action[start_frame:final_frame],
        contact=contact_window,
        phase=np.clip(phase[window], 0, 7),
        active_arm=active_arm[window],
        color_features=feature,
        multiview_color_features=multiview,
        visual_objects=visual,
        visual_latch_frame=latch_frame,
        true_objects=true_objects[window],
        valid_frames=valid_frames,
        counterfactual_initial_no_contact=counterfactual_initial_no_contact,
        contact_prefix_recovery=contact_prefix_recovery,
    )


def build_vocab(tasks: list[str]) -> dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for task in tasks:
        for token in TOKEN_RE.findall(task.lower()):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def pack(
    episodes: list[Episode],
    tasks: list[str],
    vocab: dict[str, int],
    action_representation: str,
) -> dict:
    state = np.stack([item.state for item in episodes])
    velocity = np.stack([item.velocity for item in episodes])
    palms = np.stack([item.palms for item in episodes])
    contact_sites = np.stack([item.finger_contact_sites for item in episodes])
    action = np.stack(
        [
            np.stack(
                [item.action[offset : offset + len(item.state)] for offset in CANDIDATE_OFFSETS],
                axis=1,
            )
            for item in episodes
        ]
    )
    if action_representation == "relative_joint_delta":
        action = action - state[:, :, None, :]
    elif action_representation != "absolute_joint_position":
        raise ValueError(f"Unsupported action representation: {action_representation}")
    true_objects = np.stack([item.true_objects for item in episodes])
    frames = state.shape[1]
    return {
        "state": torch.from_numpy(state),
        "velocity": torch.from_numpy(velocity),
        "palm_position": torch.from_numpy(palms),
        "finger_contact_site_position": torch.from_numpy(
            contact_sites.reshape(len(episodes), -1)
        ),
        "candidate_action": torch.from_numpy(action),
        "physical_contact_active": torch.from_numpy(
            np.stack([item.contact for item in episodes])
        ),
        "valid_frame": torch.from_numpy(
            np.stack(
                [np.arange(frames) < item.valid_frames for item in episodes]
            )
        ),
        "counterfactual_initial_no_contact": torch.tensor(
            [item.counterfactual_initial_no_contact for item in episodes], dtype=torch.bool
        ),
        "contact_prefix_recovery": torch.tensor(
            [item.contact_prefix_recovery for item in episodes], dtype=torch.bool
        ),
        "phase": torch.from_numpy(np.stack([item.phase for item in episodes])),
        "active_arm": torch.from_numpy(np.stack([item.active_arm for item in episodes])),
        "target": torch.tensor([item.target for item in episodes], dtype=torch.long),
        "task_index": torch.tensor([tasks.index(item.task) for item in episodes], dtype=torch.long),
        "color_features": torch.from_numpy(np.stack([item.color_features for item in episodes])),
        "multiview_color_features": torch.from_numpy(
            np.stack([item.multiview_color_features for item in episodes])
        ),
        "visual_object_position": torch.from_numpy(np.stack([item.visual_objects for item in episodes])),
        "visual_latch_frame": torch.tensor(
            [item.visual_latch_frame for item in episodes], dtype=torch.int64
        ),
        "audit_true_object_position": torch.from_numpy(true_objects),
        "episode_name": [item.name for item in episodes],
        "source_root": [item.source_root for item in episodes],
        "tasks": tasks,
        "vocab": vocab,
        "candidate_offsets_frames": torch.from_numpy(CANDIDATE_OFFSETS.copy()),
        "action_representation": action_representation,
        "action_contract": (
            "strict_dynamic_direct_joint_deltas"
            if action_representation == "relative_joint_delta"
            else "strict_dynamic_direct_joint_candidates"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.sequence_frames < 16:
        raise ValueError("--sequence-frames must be at least 16")
    if args.validation_stride < 2:
        raise ValueError("--validation-stride must be at least two")
    if args.recovery_repeat < 1:
        raise ValueError("--recovery-repeat must be positive")
    if args.contact_prefix_repeat < 1:
        raise ValueError("--contact-prefix-repeat must be positive")
    if args.recovery_skip_prefix_frames < 0:
        raise ValueError("--recovery-skip-prefix-frames must be non-negative")
    if args.recovery_supervision_frames < 0:
        raise ValueError("--recovery-supervision-frames must be non-negative")
    calibration = torch.from_numpy(
        np.load(args.calibration)["coefficients"].astype(np.float32)
    )
    base_episodes = []
    recovery_episodes = []
    contact_prefix_episodes = []
    rejected = []
    for root in args.dataset_root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for directory in episode_paths(root):
            item = load_episode(directory, root, calibration, args.sequence_frames)
            if item is None:
                rejected.append(str(directory))
            else:
                base_episodes.append(item)
                print(f"DIRECT_DYNAMIC_CACHE_EPISODE={directory}", flush=True)
    for root in args.recovery_root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for directory in episode_paths(root):
            recovery_valid_frames = (
                min(args.recovery_supervision_frames, args.sequence_frames)
                if args.recovery_supervision_frames
                else None
            )
            item = load_episode(
                directory,
                root,
                calibration,
                args.sequence_frames,
                args.recovery_skip_prefix_frames,
                valid_frames=recovery_valid_frames,
            )
            if item is None:
                summary = recovery_summary(directory)
                valid_frames = (
                    audited_contact_prefix_frames(
                        directory,
                        summary,
                        args.sequence_frames,
                        args.recovery_skip_prefix_frames,
                    )
                    if args.allow_contact_prefix_recovery and summary is not None
                    else None
                )
                if valid_frames is None:
                    rejected.append(str(directory))
                    continue
                item = load_episode(
                    directory,
                    root,
                    calibration,
                    args.sequence_frames,
                    args.recovery_skip_prefix_frames,
                    summary=summary,
                    valid_frames=valid_frames,
                    counterfactual_initial_no_contact=True,
                    contact_prefix_recovery=True,
                )
                if item is None:
                    rejected.append(str(directory))
                    continue
                contact_prefix_episodes.append(item)
                recovery_episodes.append(item)
                print(
                    f"DIRECT_DYNAMIC_CONTACT_PREFIX_RECOVERY={directory} rows={valid_frames}",
                    flush=True,
                )
            else:
                recovery_episodes.append(item)
                print(f"DIRECT_DYNAMIC_RECOVERY_EPISODE={directory}", flush=True)
    for root in args.contact_prefix_recovery_root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for directory in episode_paths(root):
            summary = recovery_summary(directory)
            valid_frames = (
                audited_contact_prefix_frames(directory, summary, args.sequence_frames, 0)
                if summary is not None
                else None
            )
            if valid_frames is None:
                rejected.append(str(directory))
                continue
            item = load_episode(
                directory,
                root,
                calibration,
                args.sequence_frames,
                summary=summary,
                valid_frames=valid_frames,
                counterfactual_initial_no_contact=True,
                contact_prefix_recovery=True,
            )
            if item is None:
                rejected.append(str(directory))
                continue
            contact_prefix_episodes.append(item)
            print(
                f"DIRECT_DYNAMIC_CONTACT_PREFIX_RECOVERY={directory} rows={valid_frames}",
                flush=True,
            )
    if len(base_episodes) < args.validation_stride:
        raise ValueError("Not enough strict episodes for an episode-level split")
    base_episodes.sort(key=lambda item: (item.source_root, item.name))
    # Source episodes are ordered red/yellow/green in fixed triplets.  Split
    # within each target, otherwise every sixth row would validate only red.
    seen_by_target = [0] * len(OBJECT_NAMES)
    train, val = [], []
    for item in base_episodes:
        index = seen_by_target[item.target]
        seen_by_target[item.target] += 1
        (val if index % args.validation_stride == 0 else train).append(item)
    train.extend(recovery_episodes * args.recovery_repeat)
    train.extend(contact_prefix_episodes * args.contact_prefix_repeat)
    episodes = [*base_episodes, *recovery_episodes, *contact_prefix_episodes]
    if not train or not val:
        raise ValueError("Episode split produced an empty partition")
    tasks = sorted({item.task for item in episodes})
    vocab = build_vocab(tasks)
    train_cache = pack(train, tasks, vocab, args.action_representation)
    val_cache = pack(val, tasks, vocab, args.action_representation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(train_cache, args.output_dir / "train.pt")
    torch.save(val_cache, args.output_dir / "val.pt")
    localization_error = torch.cat(
        [
            torch.linalg.vector_norm(
                cache["visual_object_position"][index, latch].reshape(3, 3)
                - cache["audit_true_object_position"][index, latch].reshape(3, 3),
                dim=-1,
            )
            for cache in (train_cache, val_cache)
            for index, latch in enumerate(cache["visual_latch_frame"].tolist())
        ]
    )
    latch_frames = torch.cat(
        [cache["visual_latch_frame"] for cache in (train_cache, val_cache)]
    )
    summary = {
        "experiment": "G1LANG-direct-joint-dynamic-cache",
        "dataset_roots": [str(item) for item in args.dataset_root],
        "contact_prefix_recovery_roots": [
            str(item) for item in args.contact_prefix_recovery_root
        ],
        "strict_dynamic_episodes": len(episodes),
        "base_expert_episodes": len(base_episodes),
        "onpolicy_recovery_episodes": len(recovery_episodes),
        "contact_prefix_recovery_episodes": len(contact_prefix_episodes),
        "contact_prefix_recovery_supervised_frames": int(
            sum(item.valid_frames for item in contact_prefix_episodes)
        ),
        "contact_prefix_contract": (
            "initial policy-visited state is counterfactually tactile=0; only "
            "the contiguous verified contact-and-lift prefix is supervised"
            if contact_prefix_episodes
            else None
        ),
        "recovery_repeat": args.recovery_repeat,
        "contact_prefix_repeat": args.contact_prefix_repeat,
        "recovery_skip_prefix_frames": args.recovery_skip_prefix_frames,
        "recovery_supervision_frames": (
            args.recovery_supervision_frames or args.sequence_frames
        ),
        "weighted_recovery_training_episodes": (
            len(recovery_episodes) * args.recovery_repeat
        ),
        "weighted_contact_prefix_training_episodes": (
            len(contact_prefix_episodes) * args.contact_prefix_repeat
        ),
        "train_episodes": len(train),
        "validation_episodes": len(val),
        "sequence_frames": args.sequence_frames,
        "candidate_offsets_frames": CANDIDATE_OFFSETS.tolist(),
        "state_input_dim": 31,
        "joint_action_dim": 31,
        "candidate_count": len(CANDIDATE_OFFSETS),
        "action_representation": args.action_representation,
        "camera_input": "dynamic synchronized head/left-wrist/right-wrist RGB colour features",
        "object_geometry_input": "head RGB geometry latched at first all-object-visible frame; zeros before latch",
        "robot_geometry_input": (
            "causal forward-kinematic samples of physical thumb/index/middle "
            "collision capsules from state31"
        ),
        "head_geometry_latch_strategy": "first causal all-object-visible head RGB frame",
        "language_input": "expert language instruction",
        "labels": (
            "strict MuJoCo dynamically executed joint-target deltas relative to "
            "the causal state31"
            if args.action_representation == "relative_joint_delta"
            else "strict MuJoCo dynamically executed joint-position commands"
        ),
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "magnet_enabled": False,
        "privileged_positions_at_inference": False,
        "latched_head_camera_object_error_m": {
            "mean": float(localization_error.mean()),
            "p95": float(torch.quantile(localization_error, 0.95)),
            "max": float(localization_error.max()),
        },
        "head_geometry_latch_frame": {
            "mean": float(latch_frames.float().mean()),
            "p95": float(torch.quantile(latch_frames.float(), 0.95)),
            "max": int(latch_frames.max()),
        },
        "rejected_non_strict_episodes": rejected,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "rejected_non_strict_episodes"}, indent=2))


if __name__ == "__main__":
    main()
