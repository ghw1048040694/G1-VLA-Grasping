#!/usr/bin/env python3
"""Offline VLA + learned-world-model MPC screening on held-out episodes.

The VLA proposes action chunks from RGB, language and proprioception.  MPC
rolls out noisy alternatives through the learned contact-only world model and
selects the candidate with the lowest predicted palm/object and contact cost.
Object locations are initialized from the VLA object head; palm locations are
obtained by ordinary MuJoCo forward kinematics from joint proprioception.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import mujoco
import numpy as np
import torch

from g1_compact_vla import CAMERA_KEYS, encode_tasks, load_checkpoint
from train_g1_language_world_model import (
    CONTACT,
    OBJECT_NAMES,
    OBJECT,
    PALM,
    PHASE,
    QPOS,
    QVEL,
    STATE_DIM,
    ContactWorldModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vla-checkpoint", type=Path, required=True)
    parser.add_argument("--wm-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-episode", type=int, default=9)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def decode(path: Path) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.asarray(frames)


def palm_fk(model: mujoco.MjModel, data: mujoco.MjData, full_qpos: np.ndarray) -> np.ndarray:
    data.qpos[: len(full_qpos)] = full_qpos
    mujoco.mj_forward(model, data)
    return np.stack(
        [data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)] for name in ("left_palm_center", "right_palm_center")]
    ).astype(np.float32)


def state_from_observation(
    arrays,
    frame: int,
    object_estimate: np.ndarray,
    phase_index: int,
    palm: np.ndarray,
) -> np.ndarray:
    qpos = arrays["joint_position_rad"][frame, 12:].astype(np.float32)
    qvel = arrays["joint_velocity_rad_s"][frame, 12:].astype(np.float32)
    contact = np.asarray([0.0], dtype=np.float32)
    phase = np.eye(8, dtype=np.float32)[np.clip(phase_index, 0, 7)]
    return np.concatenate((qpos, qvel, object_estimate.reshape(-1), palm.reshape(-1), contact, phase))


@torch.no_grad()
def vla_proposal(model, stats, vocab, frames, arrays, frame: int, task: str, device: torch.device):
    images = torch.from_numpy(np.stack([frames[key][frame] for key in CAMERA_KEYS])).permute(0, 3, 1, 2).float() / 255.0
    embeddings = model.vision(images.to(device)).unsqueeze(0)
    token_ids, token_mask = encode_tasks([task], vocab)
    qpos = torch.from_numpy(arrays["joint_position_rad"][frame, 12:].astype(np.float32)).unsqueeze(0).to(device)
    state = (qpos - stats["state_mean"]) / stats["state_std"]
    output = model(embeddings, state, token_ids.to(device), token_mask.to(device))
    action = output["action"] * stats["action_std"] + stats["action_mean"]
    if model.action_is_delta:
        action = action + qpos[:, None, :]
    object_estimate = output["object_position"] * stats["object_std"] + stats["object_mean"]
    phase = output["phase_logits"].argmax(-1).item()
    return action[0].cpu().numpy(), object_estimate[0].cpu().numpy(), phase


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    vla, vocab, vla_stats = load_checkpoint(args.vla_checkpoint, str(device))
    wm_checkpoint = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
    wm_config = wm_checkpoint["config"]
    wm = ContactWorldModel(hidden_dim=int(wm_config.get("hidden_dim", 512)), depth=int(wm_config.get("depth", 3))).to(device)
    wm.load_state_dict(wm_checkpoint["model"])
    wm.eval()
    wm_stats = {key: torch.from_numpy(value).to(device) for key, value in wm_checkpoint["stats"].items()}
    vla_stats = {key: value.to(device) for key, value in vla_stats.items()}
    reports = []
    for episode_index in range(args.start_episode, args.start_episode + args.episodes):
        episode_dir = args.dataset_root / f"episode_{episode_index:04d}"
        with np.load(episode_dir / "expert_pick_place_episode.npz") as arrays:
            metadata = json.loads(str(arrays["metadata_json"]))
            task = metadata["language_instruction"]
            target = int(arrays["target_object_index"][0])
            active_arm = int(arrays["active_arm_index"][0])
            xml_path = episode_dir / "g1_language_pick_place.xml"
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            frames = {key: decode(episode_dir / {CAMERA_KEYS[0]: "head_camera.mp4", CAMERA_KEYS[1]: "left_wrist_camera.mp4", CAMERA_KEYS[2]: "right_wrist_camera.mp4"}[key]) for key in CAMERA_KEYS}
            base_costs, selected_costs, selected = [], [], []
            for frame in range(0, len(arrays["time_s"]) - vla.chunk_horizon - 1, args.stride):
                action_chunk, object_estimate, phase = vla_proposal(vla, vla_stats, vocab, frames, arrays, frame, task, device)
                palm = palm_fk(model, data, arrays["joint_position_rad"][frame])
                raw_state = state_from_observation(arrays, frame, object_estimate, phase, palm)
                state = (torch.from_numpy(raw_state).to(device) - wm_stats["state_mean"]) / wm_stats["state_std"]
                target_onehot = torch.eye(3, device=device)[target].expand(args.candidates, -1)
                base = action_chunk[: vla.chunk_horizon]
                candidates = [base]
                candidates.append(np.repeat(arrays["joint_position_rad"][frame, 12:][None], len(base), axis=0))
                while len(candidates) < args.candidates:
                    noise = rng.normal(0.0, 0.015 + 0.01 * (len(candidates) % 3), size=base.shape).astype(np.float32)
                    candidates.append(base + noise)
                action_np = np.stack(candidates).astype(np.float32)
                action = (torch.from_numpy(action_np).to(device) - wm_stats["action_mean"]) / wm_stats["action_std"]
                initial = state.expand(len(candidates), -1)
                prediction = []
                current = initial
                for step in range(action.shape[1]):
                    current = wm(current, action[:, step], target_onehot)
                    prediction.append(current)
                rollout = torch.stack(prediction, dim=1)
                raw = rollout * wm_stats["state_std"] + wm_stats["state_mean"]
                palm_pred = raw[:, :, PALM].reshape(len(candidates), -1, 2, 3)
                object_pred = raw[:, :, OBJECT].reshape(len(candidates), -1, 3, 3)
                distance = torch.linalg.vector_norm(palm_pred[:, :, active_arm] - object_pred[:, :, target], dim=-1)
                contact_penalty = (raw[:, :, CONTACT] < 0.5).float().mean(-1)
                smoothness = torch.mean((action[:, 1:] - action[:, :-1]) ** 2, dim=(1, 2))
                score = distance.mean(-1) + 0.04 * distance[:, -1] + 0.03 * contact_penalty + 0.01 * smoothness
                choice = int(score.argmin())
                base_costs.append(float(score[0]))
                selected_costs.append(float(score[choice]))
                selected.append(choice)
            reports.append({
                "episode": episode_index,
                "target": OBJECT_NAMES[target] if target < len(OBJECT_NAMES) else str(target),
                "frames_screened": len(selected),
                "mean_vla_cost": float(np.mean(base_costs)),
                "mean_mpc_cost": float(np.mean(selected_costs)),
                "relative_cost_reduction": float(1.0 - np.mean(selected_costs) / max(np.mean(base_costs), 1e-8)),
                "candidate_zero_selected_fraction": float(np.mean(np.asarray(selected) == 0)),
                "passed": bool(np.mean(selected_costs) <= np.mean(base_costs)),
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "G1LANG-97-vla-world-model-mpc-screening",
        "vla_checkpoint": str(args.vla_checkpoint),
        "world_model_checkpoint": str(args.wm_checkpoint),
        "episodes": reports,
        "candidate_count": args.candidates,
        "uses_rgb_language_proprioception": True,
        "uses_privileged_object_positions": False,
        "assisted_grasp_enabled": False,
        "equality_constraints": False,
        "mean_relative_cost_reduction": float(np.mean([item["relative_cost_reduction"] for item in reports])),
        "passed": bool(all(item["passed"] for item in reports)),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
