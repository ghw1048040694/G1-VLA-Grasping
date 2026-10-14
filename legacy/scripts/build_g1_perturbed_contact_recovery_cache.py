#!/usr/bin/env python3
"""Build visual recovery labels from reachable perturbations around expert states.

The sampled state is a real MuJoCo state with perturbed robot joints and the
corresponding freshly rendered three-camera observation. The label is the
subsequent command segment from that exact, already successful strict-contact
expert trajectory. Deployment never sees object ground truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mujoco
import numpy as np
import torch

from build_g1_contact_action_cache import (
    orientation6,
    palm_lookahead_residual_contact_actions,
    palm_residual_contact_actions,
)
from g1_compact_vla import SpatialFrozenResNet18
from run_g1_language_pick_place import (
    OBJECT_NAMES,
    PALM_NAMES,
    controlled_joints,
)
from validate_g1_bimanual_actuation import apply_regularized_dynamics


CAMERAS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
# Phases 2 and 3 are represented by the clean strict-expert cache. Their
# state-reset recovery labels are excluded until they pass the same physical
# audit as phases 0/1; a frame that only works with unobserved source velocity
# is not a valid VLA target for a position-only policy.
PRECONTACT_PHASES = (1, 2, 3)
DEFAULT_RECOVERY_PHASES = (1,)
# Contact approach has a much narrower viable basin than the high approach.
# A roughly one-degree joint disturbance still creates a visual recovery task
# in phase 2 without teaching labels that cannot hold the object physically.
PHASE_PERTURBATION_SCALE = {0: 0.50, 1: 0.50, 2: 0.10, 3: 0.05}


def parse_recovery_phases(value: str) -> tuple[int, ...]:
    try:
        phases = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("recovery phases must be comma-separated integers") from error
    if not phases or any(phase < 0 or phase > 7 for phase in phases):
        raise argparse.ArgumentTypeError("recovery phases must be in [0, 7]")
    return phases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, required=True)
    parser.add_argument("--expert-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-target", type=int, default=20)
    parser.add_argument("--samples-per-episode", type=int, default=20)
    parser.add_argument("--chunk-horizon", type=int, default=20)
    parser.add_argument(
        "--recovery-phases",
        type=parse_recovery_phases,
        default=DEFAULT_RECOVERY_PHASES,
        help="Comma-separated expert phases eligible for perturbed recovery labels.",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def color_features(image: torch.Tensor) -> torch.Tensor:
    """Match the policy's online head-camera color features."""
    image = image[:3].clamp(0, 1)
    red = (image[0] > 0.47) & (image[0] > image[1] * 1.25) & (image[0] > image[2] * 1.25)
    yellow = (image[0] > 0.47) & (image[1] > 0.35) & (image[2] < 0.47)
    green = (image[1] > 0.31) & (image[1] > image[0] * 1.25) & (image[1] > image[2] * 1.15)
    height, width = image.shape[-2:]
    x_grid = torch.arange(width, dtype=image.dtype).view(1, width)
    y_grid = torch.arange(height, dtype=image.dtype).view(height, 1)
    values = []
    for mask in (red, yellow, green):
        count = mask.sum().to(image.dtype).clamp_min(1.0)
        values.extend(
            (
                (mask.to(image.dtype) * x_grid).sum() / count / width,
                (mask.to(image.dtype) * y_grid).sum() / count / height,
                count / float(height * width),
            )
        )
    return torch.stack(values).float()


def selected_episodes(source_root: Path, limit: int) -> list[Path]:
    grouped = {name: [] for name in OBJECT_NAMES}
    for path in sorted(source_root.glob("episode_*/language_pick_place_summary.json")):
        summary = json.loads(path.read_text())
        target = summary["target_object"]
        if target in grouped:
            grouped[target].append(path.parent)
    result = []
    for target in OBJECT_NAMES:
        if len(grouped[target]) < limit:
            raise ValueError(f"Only {len(grouped[target])} source episodes for {target}")
        result.extend(grouped[target][:limit])
    return result


def restore_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    names: np.ndarray,
    joint_position: np.ndarray,
    object_position: np.ndarray,
    object_quaternion: np.ndarray,
    baseline_qpos: np.ndarray,
) -> None:
    data.qpos[:] = baseline_qpos
    data.qvel[:] = 0.0
    position_by_name = dict(zip(names.tolist(), joint_position.tolist()))
    for name in upper_names:
        data.qpos[controlled[name]["qpos_id"]] = position_by_name[name]
    for object_index, name in enumerate(OBJECT_NAMES):
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint = int(model.body_jntadr[body])
        address = int(model.jnt_qposadr[joint])
        data.qpos[address : address + 3] = object_position[object_index]
        data.qpos[address + 3 : address + 7] = object_quaternion[object_index]
    mujoco.mj_forward(model, data)


def perturb_active_arm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    target: str,
    rng: np.random.Generator,
    amplitude_scale: float = 1.0,
) -> None:
    if not 0.0 < amplitude_scale <= 1.0:
        raise ValueError("amplitude_scale must be in (0, 1]")
    target_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{target}_assist_site")
    active_prefix = "left_" if data.site_xpos[target_site][1] >= 0.05 else "right_"
    scales = {
        "shoulder_pitch": 0.12,
        "shoulder_roll": 0.10,
        "shoulder_yaw": 0.10,
        "elbow": 0.14,
        "wrist_roll": 0.09,
        "wrist_pitch": 0.09,
        "wrist_yaw": 0.09,
    }
    for name, item in controlled.items():
        scale = next((value for key, value in scales.items() if key in name), None)
        if scale is None or not name.startswith(active_prefix):
            continue
        low, high = model.jnt_range[item["joint_id"]]
        scale *= amplitude_scale
        value = float(data.qpos[item["qpos_id"]]) + float(rng.uniform(-scale, scale))
        data.qpos[item["qpos_id"]] = np.clip(value, low + 1e-4, high - 1e-4)
    mujoco.mj_forward(model, data)


def contact_action(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controlled: dict[str, dict[str, int]],
    upper_names: list[str],
    palm_ids: list[int],
    joint_chunk: np.ndarray,
) -> torch.Tensor:
    rollout = mujoco.MjData(model)
    rollout.qpos[:] = data.qpos
    rollout.qvel[:] = data.qvel
    rows = []
    for action in joint_chunk:
        for name, value in zip(upper_names, action):
            rollout.qpos[controlled[name]["qpos_id"]] = value
        mujoco.mj_forward(model, rollout)
        rows.append(
            np.concatenate(
                (
                    action[:3],
                    np.concatenate([rollout.site_xpos[site].copy() for site in palm_ids]),
                    orientation6(rollout, palm_ids),
                    action[10:17],
                    action[24:31],
                )
            ).astype(np.float32)
        )
    return torch.from_numpy(np.stack(rows))


@torch.no_grad()
def rendered_vision(
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    encoder: SpatialFrozenResNet18,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    for camera in CAMERAS:
        renderer.update_scene(data, camera=camera)
        image = torch.from_numpy(renderer.render().copy()).permute(2, 0, 1).float() / 255.0
        images.append(image)
    stacked = torch.stack(images)
    vision = encoder(stacked.to(device)).cpu().half()
    return vision, color_features(stacked[0])


def sample_episode(
    episode_dir: Path,
    base: dict,
    args: argparse.Namespace,
    encoder: SpatialFrozenResNet18,
    device: str,
    rng: np.random.Generator,
    episode_offset: int,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    summary = json.loads((episode_dir / "language_pick_place_summary.json").read_text())
    target = summary["target_object"]
    target_index = OBJECT_NAMES.index(target)
    with np.load(episode_dir / "expert_pick_place_episode.npz", allow_pickle=True) as source:
        names = source["joint_names"].copy()
        positions = source["joint_position_rad"].copy()
        actions = source["action_joint_position_rad"].copy()
        objects = source["object_position_m"].copy()
        quaternions = source["object_quaternion_wxyz"].copy()
        phases = source["task_phase"].copy()
        contacts = source["physical_contact_active"].copy()
        active_arms = source["active_arm_index"].copy()
    recovery_phases = getattr(args, "recovery_phases", DEFAULT_RECOVERY_PHASES)
    candidates = np.flatnonzero(
        np.isin(phases, recovery_phases)
        & (contacts == 0)
        & (np.arange(len(phases)) + args.chunk_horizon < len(actions))
    )
    if len(candidates) == 0:
        raise RuntimeError(f"No pre-contact expert frames in {episode_dir}")
    model = mujoco.MjModel.from_xml_path(str(episode_dir / "g1_language_pick_place.xml"))
    apply_regularized_dynamics(model)
    data = mujoco.MjData(model)
    controlled = controlled_joints(model)
    upper_names = list(controlled)[12:]
    name_to_source_index = {name: index for index, name in enumerate(names.tolist())}
    missing_names = [name for name in upper_names if name not in name_to_source_index]
    if missing_names:
        raise RuntimeError(f"Source action is missing controlled joints: {missing_names}")
    upper_source_indices = np.asarray(
        [name_to_source_index[name] for name in upper_names], dtype=np.int64
    )
    palm_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        for name in PALM_NAMES
    ]
    object_site_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_assist_site")
        for name in OBJECT_NAMES
    }
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    baseline_qpos = data.qpos.copy()
    renderer = mujoco.Renderer(model, height=120, width=160)
    fields = {key: [] for key in (
        "vision", "state", "action", "contact_action", "palm_position",
        "palm_orientation", "object_position", "target", "phase", "active_arm",
        "color_features", "task_index", "episode_index", "frame_index",
    )}
    diagnostics = []
    task = summary["language_instruction"]
    task_index = list(base["tasks"]).index(task)
    attempts = 0
    while len(fields["state"]) < args.samples_per_episode:
        attempts += 1
        if attempts > args.samples_per_episode * 30:
            raise RuntimeError(f"Could not sample enough reachable perturbations for {episode_dir}")
        frame = int(rng.choice(candidates))
        restore_state(
            model, data, controlled, upper_names, names, positions[frame],
            objects[frame], quaternions[frame], baseline_qpos,
        )
        phase = int(phases[frame])
        perturbation_scale = PHASE_PERTURBATION_SCALE[phase]
        perturb_active_arm(
            model, data, controlled, target, rng,
            amplitude_scale=perturbation_scale,
        )
        # This is a counterfactual recovery sample around a particular expert
        # state, so the recorded future action segment is valid provenance.
        # Re-solving IK from a perturbed pose can choose a different local
        # minimum and destroy the physically verified contact trajectory.
        source_indices = np.arange(frame + 1, frame + 1 + args.chunk_horizon)
        chunk = actions[source_indices][:, upper_source_indices].astype(np.float32)
        active_arm = int(active_arms[frame])
        target_site = object_site_ids[target]
        palm_distance = float(np.linalg.norm(data.site_xpos[palm_ids[active_arm]] - data.site_xpos[target_site]))
        if not 0.04 <= palm_distance <= 0.34:
            continue
        vision, color = rendered_vision(renderer, data, encoder, device)
        contact = contact_action(model, data, controlled, upper_names, palm_ids, chunk)
        sample_index = len(fields["state"])
        fields["vision"].append(vision)
        fields["state"].append(torch.tensor(
            [data.qpos[controlled[name]["qpos_id"]] for name in upper_names], dtype=torch.float32
        ))
        fields["action"].append(torch.from_numpy(chunk))
        fields["contact_action"].append(contact)
        fields["palm_position"].append(torch.from_numpy(
            np.concatenate([data.site_xpos[site].copy() for site in palm_ids]).astype(np.float32)
        ))
        fields["palm_orientation"].append(torch.from_numpy(orientation6(data, palm_ids)))
        fields["object_position"].append(torch.from_numpy(objects[frame].reshape(-1).astype(np.float32)))
        fields["target"].append(torch.tensor(target_index, dtype=torch.long))
        fields["phase"].append(torch.tensor(phase, dtype=torch.long))
        fields["active_arm"].append(torch.tensor(active_arm, dtype=torch.long))
        fields["color_features"].append(color)
        fields["task_index"].append(torch.tensor(task_index, dtype=torch.long))
        fields["episode_index"].append(torch.tensor(
            int(base["episode_index"].max().item()) + 1 + episode_offset * args.samples_per_episode + sample_index,
            dtype=torch.long,
        ))
        fields["frame_index"].append(torch.tensor(0, dtype=torch.long))
        diagnostics.append({
            "source_frame": frame,
            "source_action_final_frame": int(source_indices[-1]),
            "palm_distance_to_target_m": palm_distance,
            "phase": phase,
            "active_arm": active_arm,
            "perturbation_scale": perturbation_scale,
            "first_action_delta_max_rad": float(
                np.max(np.abs(chunk[0] - fields["state"][-1].numpy()))
            ),
            "final_action_delta_max_rad": float(
                np.max(np.abs(chunk[-1] - fields["state"][-1].numpy()))
            ),
        })
    close = getattr(renderer, "close", None)
    if close is not None:
        close()
    return {key: torch.stack(value) for key, value in fields.items()}, diagnostics


def main() -> None:
    args = parse_args()
    if args.episodes_per_target < 1 or args.samples_per_episode < 1:
        raise ValueError("episode and sample counts must be positive")
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    base = torch.load(args.base_cache / "train.pt", map_location="cpu", weights_only=False)
    required = {
        "vision", "state", "action", "contact_action", "palm_position",
        "palm_orientation", "object_position", "target", "phase", "active_arm",
        "color_features", "task_index", "episode_index", "frame_index", "tasks",
    }
    missing = sorted(required - set(base))
    if missing:
        raise ValueError(f"Base cache lacks required fields: {missing}")
    encoder = SpatialFrozenResNet18().to(device).eval()
    rng = np.random.default_rng(args.seed)
    parts = []
    diagnostics = {}
    for offset, episode_dir in enumerate(selected_episodes(args.source_root, args.episodes_per_target)):
        part, detail = sample_episode(episode_dir, base, args, encoder, device, rng, offset)
        parts.append(part)
        diagnostics[episode_dir.name] = detail
        print(f"PERTURB_EPISODE={episode_dir.name} samples={len(part['state'])}", flush=True)
    generated = {key: torch.cat([part[key] for part in parts]) for key in parts[0]}
    generated["contact_palm_residual_action"] = palm_residual_contact_actions(
        generated["contact_action"], generated["palm_position"]
    )
    generated["contact_palm_lookahead_residual_action"] = palm_lookahead_residual_contact_actions(
        generated["contact_palm_residual_action"]
    )
    merged = dict(base)
    tensor_keys = tuple(generated)
    for key in tensor_keys:
        merged[key] = torch.cat((base[key], generated[key]), dim=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output_dir / "train.pt")
    shutil.copy2(args.base_cache / "val.pt", args.output_dir / "val.pt")
    summary = {
        "experiment": "G1LANG_perturbed_reachable_contact_recovery_cache",
        "base_cache": str(args.base_cache),
        "source_root": str(args.source_root),
        "base_samples": len(base["state"]),
        "generated_samples": len(generated["state"]),
        "merged_samples": len(merged["state"]),
        "episodes_per_target": args.episodes_per_target,
        "samples_per_episode": args.samples_per_episode,
        "chunk_horizon": args.chunk_horizon,
        "recovery_phases": list(args.recovery_phases),
        "input_contract": "fresh MuJoCo-rendered three RGB views + language + 31D state",
        "label_contract": (
            "offline strict-contact expert continuation from the sampled "
            "successful trajectory; current state is independently perturbed"
        ),
        "privileged_positions_at_inference": False,
        "diagnostics": diagnostics,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
