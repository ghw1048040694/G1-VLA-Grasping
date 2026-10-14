#!/usr/bin/env python3
"""Fit a contact world model without task-phase or time inputs.

The model sees only measurable robot state, camera-derived object geometry,
contact-pad kinematics derived from robot state, and the joint command.  Physical contact remains an output
label, so it cannot be inferred from a phase/timer shortcut.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import mujoco
from torch import nn
from torch.utils.data import DataLoader, Dataset

from build_g1_direct_joint_dynamic_cache import episode_paths, strict_summary
from g1_finger_contact_geometry import finger_contact_geom_ids, finger_contact_sites
from run_g1_language_pick_place import OBJECT_NAMES


ACTION_DIM = 31
TARGET_COUNT = 3
QPOS = slice(0, 31)
QVEL = slice(31, 62)
OBJECT = slice(62, 71)
DEFAULT_CONTACT_SITE_DIM = 6


def state_layout(contact_site_dim: int) -> tuple[slice, slice, slice]:
    if contact_site_dim <= 0 or contact_site_dim % 3:
        raise ValueError("contact_site_dim must be a positive multiple of three")
    contact_sites = slice(71, 71 + contact_site_dim)
    contact = slice(contact_sites.stop, contact_sites.stop + 1)
    return OBJECT, contact_sites, contact


# Preserve the legacy names for standalone non-cache datasets while new cache
# checkpoints carry their explicit contact-site dimension.
_, PALM, CONTACT = state_layout(DEFAULT_CONTACT_SITE_DIM)
STATE_DIM = CONTACT.stop


@dataclass
class Episode:
    name: str
    source: str
    state: np.ndarray
    action: np.ndarray
    target: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help=(
            "Direct-joint dynamic cache. Its causal RGB-derived object geometry "
            "is used instead of simulator object positions."
        ),
    )
    parser.add_argument(
        "--recovery-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Strict on-policy recovery episodes appended only to training after "
            "the base expert validation split is fixed."
        ),
    )
    parser.add_argument(
        "--failure-rollout-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "No-contact VLA rollout roots added only to the training transitions. "
            "Their RGB-derived geometry and measured zero-contact labels calibrate "
            "the world model off the expert manifold."
        ),
    )
    parser.add_argument("--recovery-repeat", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=56)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--validation-stride", type=int, default=6)
    parser.add_argument(
        "--action-representation",
        choices=("absolute_joint_position", "relative_joint_delta"),
        default="absolute_joint_position",
        help="Required only for non-cache datasets; cache-backed training reads its contract.",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_episode(directory: Path, source: Path) -> Episode | None:
    summary = strict_summary(directory)
    if summary is None:
        return None
    with np.load(directory / "expert_pick_place_episode.npz") as arrays:
        qpos = arrays["joint_position_rad"][:, 12:].astype(np.float32)
        qvel = arrays["joint_velocity_rad_s"][:, 12:].astype(np.float32)
        objects = arrays["object_position_m"].reshape(len(qpos), 9).astype(np.float32)
        palms = arrays["palm_position_m"].reshape(len(qpos), 6).astype(np.float32)
        contact = arrays["physical_contact_active"].astype(np.float32)[:, None]
        action = arrays["action_joint_position_rad"][:, 12:].astype(np.float32)
        target = int(arrays["target_object_index"][0])
    state = np.concatenate((qpos, qvel, objects, palms, contact), axis=1)
    if state.shape[1] != STATE_DIM or action.shape[1] != ACTION_DIM:
        raise ValueError(f"Unexpected phase-blind state/action in {directory}: {state.shape}/{action.shape}")
    return Episode(directory.name, str(source), state, action, target)


def load_episodes(roots: list[Path]) -> list[Episode]:
    episodes = []
    for root in roots:
        for directory in episode_paths(root):
            item = load_episode(directory, root)
            if item is not None:
                episodes.append(item)
    if len(episodes) < 8:
        raise ValueError("Need at least eight strict physical dynamic episodes")
    episodes.sort(key=lambda item: (item.source, item.name))
    return episodes


def load_cache_split(path: Path, split: str) -> list[Episode]:
    """Load a causally visual state transition dataset from the VLA cache."""
    value = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "state",
        "velocity",
        "visual_object_position",
        "palm_position",
        "physical_contact_active",
        "candidate_action",
        "target",
        "episode_name",
        "source_root",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Cache {path} lacks required causal fields: {missing}")
    state = value["state"].numpy()
    velocity = value["velocity"].numpy()
    objects = value["visual_object_position"].numpy()
    sites = value.get(
        "finger_contact_site_position",
        value.get("contact_pad_position", value["palm_position"]),
    ).numpy()
    if sites.ndim == 2:
        sites = sites.reshape(state.shape[0], state.shape[1], -1)
    contact = value["physical_contact_active"].numpy()[..., None]
    # Candidate offset zero is the action actually executed at this state.
    action = value["candidate_action"].numpy()[:, :, 0]
    target = value["target"].numpy()
    valid = value.get(
        "valid_frame", torch.ones(state.shape[:2], dtype=torch.bool)
    ).numpy().astype(bool)
    if not (
        state.ndim == 3
        and state.shape == velocity.shape
        and objects.shape[:2] == state.shape[:2]
        and sites.shape[:2] == state.shape[:2]
        and contact.shape[:2] == state.shape[:2]
        and action.shape[:2] == state.shape[:2]
        and valid.shape == state.shape[:2]
    ):
        raise ValueError(f"Unexpected dynamic cache shapes in {path}")
    episodes = []
    for index in range(len(state)):
        valid_count = int(valid[index].sum())
        if valid_count < 2 or not np.all(valid[index, :valid_count]) or np.any(valid[index, valid_count:]):
            raise ValueError(f"Cache valid-frame mask is not a contiguous prefix at {path} row {index}")
        raw = np.concatenate(
            (
                state[index, :valid_count],
                velocity[index, :valid_count],
                objects[index, :valid_count],
                sites[index, :valid_count],
                contact[index, :valid_count],
            ),
            axis=-1,
        ).astype(np.float32)
        expected_state_dim = 31 + 31 + 9 + sites.shape[-1] + 1
        if raw.shape[-1] != expected_state_dim or action[index, :valid_count].shape[-1] != ACTION_DIM:
            raise ValueError(f"Unexpected cached state/action at {path} row {index}")
        episodes.append(
            Episode(
                name=f"{split}:{value['episode_name'][index]}",
                source=str(value["source_root"][index]),
                state=raw,
                action=action[index, :valid_count].astype(np.float32),
                target=int(target[index]),
            )
        )
    return episodes


def load_cache_splits(cache_dir: Path) -> tuple[list[Episode], list[Episode]]:
    train = load_cache_split(cache_dir / "train.pt", "train")
    validation = load_cache_split(cache_dir / "val.pt", "validation")
    if not train or not validation:
        raise ValueError("Causal cache must contain non-empty training and validation splits")
    return train, validation


def failure_contact_sites(
    scene: Path, joint_names: np.ndarray, states: np.ndarray
) -> np.ndarray:
    """Derive real finger collision samples from recorded robot states."""
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    qpos_ids = []
    for name in joint_names.tolist():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
        if joint_id < 0:
            raise ValueError(f"Unknown rollout joint {name} in {scene}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
    geom_ids = finger_contact_geom_ids(model)
    result = np.empty((len(states), len(geom_ids) * 9), dtype=np.float32)
    for frame, state in enumerate(states):
        data.qpos[qpos_ids] = state
        mujoco.mj_forward(model, data)
        result[frame] = finger_contact_sites(model, data, geom_ids)
    return result


def load_failure_rollouts(
    roots: list[Path], action_representation: str
) -> list[Episode]:
    """Load measured no-contact VLA transitions without simulator object truth."""
    episodes = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for directory in sorted(root.glob("episode_*")):
            summary_path = directory / "summary.json"
            trajectory_path = directory / "trajectory.npz"
            scene = directory / "scene.xml"
            if not (summary_path.is_file() and trajectory_path.is_file() and scene.is_file()):
                continue
            summary = json.loads(summary_path.read_text())
            target_name = summary.get("target_object")
            if (
                target_name not in OBJECT_NAMES
                or summary.get("task_success")
                or int(summary.get("hand_object_contact_candidate_steps", 0)) != 0
            ):
                continue
            # This is the same first-causal-head-RGB geometry latched by the
            # deployed policy. Do not substitute trajectory object_position_m.
            camera_objects = np.asarray(
                summary.get("camera_object_position_m"), dtype=np.float32
            )
            if camera_objects.shape != (len(OBJECT_NAMES), 3):
                raise ValueError(f"Missing RGB geometry in {summary_path}")
            with np.load(trajectory_path) as values:
                states = values["observation_joint_position_rad"].astype(np.float32)
                velocity = values["observation_joint_velocity_rad_s"].astype(np.float32)
                action = values["action_joint_position_rad"].astype(np.float32)
                joint_names = values["joint_names"]
            if not (states.shape == velocity.shape == action.shape and states.shape[1] == ACTION_DIM):
                raise ValueError(f"Unexpected no-contact rollout shape in {trajectory_path}")
            sites = failure_contact_sites(scene, joint_names, states)
            objects = np.broadcast_to(camera_objects.reshape(1, -1), (len(states), 9))
            contact = np.zeros((len(states), 1), dtype=np.float32)
            state = np.concatenate((states, velocity, objects, sites, contact), axis=-1)
            episodes.append(
                Episode(
                    name=f"failure:{directory.name}",
                    source=str(root),
                    state=state.astype(np.float32),
                    action=(
                        action - states
                        if action_representation == "relative_joint_delta"
                        else action
                    ),
                    target=OBJECT_NAMES.index(target_name),
                )
            )
    return episodes


def split(episodes: list[Episode], stride: int) -> tuple[list[Episode], list[Episode]]:
    counters = [0] * TARGET_COUNT
    train, val = [], []
    for item in episodes:
        index = counters[item.target]; counters[item.target] += 1
        (val if index % stride == 0 else train).append(item)
    return train, val


def normalizers(episodes: list[Episode]) -> dict[str, np.ndarray]:
    return {
        "state_mean": np.concatenate([item.state for item in episodes]).mean(0).astype(np.float32),
        "state_std": np.maximum(np.concatenate([item.state for item in episodes]).std(0), 1e-4).astype(np.float32),
        "action_mean": np.concatenate([item.action for item in episodes]).mean(0).astype(np.float32),
        "action_std": np.maximum(np.concatenate([item.action for item in episodes]).std(0), 1e-4).astype(np.float32),
    }


class TransitionDataset(Dataset):
    def __init__(self, episodes: list[Episode], stats: dict[str, np.ndarray], horizon: int):
        self.samples = []
        for episode in episodes:
            state = (episode.state - stats["state_mean"]) / stats["state_std"]
            action = (episode.action - stats["action_mean"]) / stats["action_std"]
            target = np.eye(TARGET_COUNT, dtype=np.float32)[episode.target]
            for start in range(len(state) - horizon):
                self.samples.append((state[start], action[start:start + horizon], state[start + 1:start + horizon + 1], target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return tuple(torch.from_numpy(value) for value in self.samples[index])


class PhaseBlindContactWorldModel(nn.Module):
    def __init__(self, hidden_dim: int = 512, depth: int = 3, state_dim: int = STATE_DIM):
        super().__init__()
        self.state_dim = int(state_dim)
        layers: list[nn.Module] = [nn.Linear(self.state_dim + ACTION_DIM + TARGET_COUNT, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend((nn.Linear(hidden_dim, hidden_dim), nn.SiLU()))
        self.backbone = nn.Sequential(*layers)
        self.delta_head = nn.Linear(hidden_dim, self.state_dim)

    def forward(self, state: torch.Tensor, action: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return state + self.delta_head(self.backbone(torch.cat((state, action, target), dim=-1)))


def rollout(model: PhaseBlindContactWorldModel, state: torch.Tensor, actions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predictions = []
    for step in range(actions.shape[1]):
        state = model(state, actions[:, step], target)
        predictions.append(state)
    return torch.stack(predictions, dim=1)


def weighted_loss(
    prediction: torch.Tensor,
    expected: torch.Tensor,
    contact_sites: slice,
    contact: slice,
) -> torch.Tensor:
    weights = torch.ones(prediction.shape[-1], device=prediction.device, dtype=prediction.dtype)
    weights[QPOS] = 2.0; weights[QVEL] = 1.5; weights[OBJECT] = 5.0
    # Keep the aggregate site loss comparable to the former six-coordinate
    # palm term instead of allowing 108 geometry coordinates to dominate it.
    weights[contact_sites] = 0.5; weights[contact] = 3.0
    return ((prediction - expected).square() * weights).mean()


@torch.no_grad()
def evaluate(model, episodes, stats, horizon, device):
    model.eval(); result = {}
    mean = torch.from_numpy(stats["state_mean"]).to(device); std = torch.from_numpy(stats["state_std"]).to(device)
    contact_site_dim = len(episodes[0].state[0]) - 31 - 31 - 9 - 1
    _, contact_sites, contact = state_layout(contact_site_dim)
    for length in sorted(set((1, min(5, horizon), horizon))):
        errors = {"qpos": [], "object": [], "contact_sites": [], "contact": []}
        for episode in episodes:
            starts = np.linspace(0, len(episode.state) - length - 1, min(64, len(episode.state) - length), dtype=int)
            initial = torch.from_numpy((episode.state[starts] - stats["state_mean"]) / stats["state_std"]).to(device)
            actions = torch.from_numpy(np.stack([(episode.action[start:start + length] - stats["action_mean"]) / stats["action_std"] for start in starts])).to(device)
            expected = torch.from_numpy(np.stack([(episode.state[start + 1:start + length + 1] - stats["state_mean"]) / stats["state_std"] for start in starts])).to(device)
            target = torch.tensor(np.eye(TARGET_COUNT, dtype=np.float32)[episode.target], device=device).expand(len(starts), -1)
            raw_prediction = rollout(model, initial, actions, target)[:, -1] * std + mean
            raw_expected = expected[:, -1] * std + mean
            errors["qpos"].append(torch.sqrt(torch.mean((raw_prediction[:, QPOS] - raw_expected[:, QPOS]).square())).item())
            errors["object"].append(torch.sqrt(torch.mean((raw_prediction[:, OBJECT] - raw_expected[:, OBJECT]).square())).item())
            errors["contact_sites"].append(torch.sqrt(torch.mean((raw_prediction[:, contact_sites] - raw_expected[:, contact_sites]).square())).item())
            errors["contact"].append(torch.mean(((raw_prediction[:, contact] >= 0.5) == (raw_expected[:, contact] >= 0.5)).float()).item())
        result[f"h{length}"] = {key: float(np.mean(value)) for key, value in errors.items()}
    return result


def main() -> None:
    args = parse_args()
    if args.recovery_repeat < 1:
        raise ValueError("--recovery-repeat must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.cache_dir is not None:
        if args.dataset_root or args.recovery_root:
            raise ValueError("--cache-dir cannot be combined with --dataset-root or --recovery-root")
        train, val = load_cache_splits(args.cache_dir)
        base_episodes = [*train, *val]
        recovery_episodes = []
        cache_summary = json.loads((args.cache_dir / "summary.json").read_text())
        action_representation = cache_summary.get(
            "action_representation", "absolute_joint_position"
        )
        base_strict_episode_count = int(
            cache_summary.get("base_expert_episodes", len(base_episodes))
        )
        recovery_strict_episode_count = int(
            cache_summary.get("onpolicy_recovery_episodes", 0)
        )
        recovery_repeat = int(cache_summary.get("recovery_repeat", 1))
    else:
        if not args.dataset_root:
            raise ValueError("Provide --cache-dir or at least one --dataset-root")
        base_episodes = load_episodes(args.dataset_root)
        train, val = split(base_episodes, args.validation_stride)
        recovery_episodes = load_episodes(args.recovery_root) if args.recovery_root else []
        train.extend(recovery_episodes * args.recovery_repeat)
        base_strict_episode_count = len(base_episodes)
        recovery_strict_episode_count = len(recovery_episodes)
        recovery_repeat = args.recovery_repeat
        action_representation = args.action_representation
    if action_representation not in {
        "absolute_joint_position",
        "relative_joint_delta",
    }:
        raise ValueError(f"Unsupported action representation: {action_representation}")
    failure_episodes = load_failure_rollouts(
        args.failure_rollout_root, action_representation
    )
    train.extend(failure_episodes)
    stats = normalizers(train)
    loader = DataLoader(TransitionDataset(train, stats, args.horizon), batch_size=args.batch_size, shuffle=True)
    state_dim = train[0].state.shape[-1]
    if any(item.state.shape[-1] != state_dim for item in [*train, *val]):
        raise ValueError("World-model cache rows disagree on state dimension")
    contact_site_dim = state_dim - 31 - 31 - 9 - 1
    _, contact_sites, contact = state_layout(contact_site_dim)
    model = PhaseBlindContactWorldModel(args.hidden_dim, args.depth, state_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best = math.inf; history = []; args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train(); total = samples = 0
        for state, action, expected, target in loader:
            state, action, expected, target = [item.to(device) for item in (state, action, expected, target)]
            prediction = rollout(model, state, action + 0.02 * torch.randn_like(action), target)
            loss = weighted_loss(prediction, expected, contact_sites, contact)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(state); samples += len(state)
        validation = evaluate(model, val, stats, args.horizon, device)
        value = {"epoch": epoch + 1, "train_loss": total / max(samples, 1), "validation": validation}; history.append(value)
        print("PHASE_BLIND_WORLD_MODEL_EPOCH=" + json.dumps(value), flush=True)
        score = validation[f"h{args.horizon}"]["contact_sites"] + validation[f"h{args.horizon}"]["object"] + (1.0 - validation[f"h{args.horizon}"]["contact"])
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "stats": stats, "config": {"hidden_dim": args.hidden_dim, "depth": args.depth, "state_dim": state_dim}, "state_dim": state_dim, "contact_site_dim": contact_site_dim, "action_dim": ACTION_DIM, "action_representation": action_representation, "target_count": TARGET_COUNT}, args.output_dir / "best.pt")
    summary = {"experiment": "G1LANG-phase-blind-contact-world-model", "dataset_roots": [str(root) for root in args.dataset_root], "cache_dir": str(args.cache_dir) if args.cache_dir is not None else None, "recovery_roots": [str(root) for root in args.recovery_root], "failure_rollout_roots": [str(root) for root in args.failure_rollout_root], "base_strict_episodes": base_strict_episode_count, "recovery_strict_episodes": recovery_strict_episode_count, "failure_no_contact_episodes": len(failure_episodes), "recovery_repeat": recovery_repeat, "train_episodes": len(train), "validation_episodes": len(val), "state_contract": "qpos31 + qvel31 + causal-head-RGB-object9 + physical-finger-contact-sites + measured-contact1; no task-phase/timer and no simulator object truth", "contact_site_dim": contact_site_dim, "action_representation": action_representation, "best_validation_score": best, "final": history[-1], "history": history}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
