#!/usr/bin/env python3
"""Continue SmolVLA training on exact-scene, same-frame language triplets."""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
import safetensors.torch
from torch.amp import GradScaler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.smolvla.modeling_smolvla import standardise_state_dict
from lerobot.policies.utils import get_device_from_parameters
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import get_safe_torch_device, has_method, init_logging
from g1_language_action_adapter import (
    ADAPTER_NAME,
    attach_language_action_adapter,
    compute_context_scene_action_chunk,
    set_context_scene_action_decoder_enabled,
    set_scene_action_decoder_enabled,
)


TARGET_COUNT = 3
TARGET_NAMES = ("red_triangle", "yellow_rod", "green_cube")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=22000)
    parser.add_argument("--max-frame-exclusive", type=int, default=16)
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--contrastive-margin", type=float, default=0.0)
    parser.add_argument("--contrastive-weight", type=float, default=0.0)
    parser.add_argument("--target-classification-weight", type=float, default=0.0)
    parser.add_argument("--scene-action-weight", type=float, default=0.0)
    parser.add_argument("--unfreeze-vlm-text-layers", type=int, default=0)
    parser.add_argument("--freeze-action-path", action="store_true")
    parser.add_argument("--scene-decoder-only", action="store_true")
    parser.add_argument("--context-scene-decoder-only", action="store_true")
    parser.add_argument("--verify-sampler-only", action="store_true")
    parser.add_argument("--verify-update-only", action="store_true")
    return parser.parse_args()


class ExactSceneTripletBatchSampler:
    def __init__(self, episode_data_index: dict, max_frame_exclusive: int):
        starts = [int(value) for value in episode_data_index["from"]]
        ends = [int(value) for value in episode_data_index["to"]]
        if len(starts) % TARGET_COUNT != 0:
            raise ValueError("Dataset episodes do not form complete target triplets")
        lengths = [end - start for start, end in zip(starts, ends, strict=True)]
        if min(lengths) < max_frame_exclusive:
            raise ValueError("Curriculum frame range exceeds an episode boundary")
        self.starts = starts
        self.scene_count = len(starts) // TARGET_COUNT
        self.max_frame_exclusive = max_frame_exclusive
        self.scene_frames = [
            (scene_index, frame_index)
            for scene_index in range(self.scene_count)
            for frame_index in range(max_frame_exclusive)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        for shuffled_index in torch.randperm(len(self.scene_frames)).tolist():
            scene_index, frame_index = self.scene_frames[shuffled_index]
            episode_start = scene_index * TARGET_COUNT
            yield [
                self.starts[episode_start + target_index] + frame_index
                for target_index in range(TARGET_COUNT)
            ]

    def __len__(self) -> int:
        return len(self.scene_frames)


def validate_triplet_batch(batch: dict) -> None:
    episodes = [int(value) for value in batch["episode_index"].reshape(-1)]
    frames = [int(value) for value in batch["frame_index"].reshape(-1)]
    targets = [
        int(value)
        for value in batch["complementary_info.target_object_index"].reshape(-1)
    ]
    if len(episodes) != TARGET_COUNT:
        raise ValueError(f"Expected a three-sample batch, found {len(episodes)}")
    if [episode % TARGET_COUNT for episode in episodes] != list(range(TARGET_COUNT)):
        raise ValueError(f"Batch target episode order is invalid: {episodes}")
    if len({episode // TARGET_COUNT for episode in episodes}) != 1:
        raise ValueError(f"Batch contains multiple scenes: {episodes}")
    if len(set(frames)) != 1:
        raise ValueError(f"Batch contains multiple frame indices: {frames}")
    if targets != list(range(TARGET_COUNT)):
        raise ValueError(f"Batch target order is invalid: {targets}")


def shared_flow_inputs(policy, batch_size: int, device: torch.device):
    noise = policy.model.sample_noise(
        (1, policy.config.chunk_size, policy.config.max_action_dim), device
    ).expand(batch_size, -1, -1)
    time_values = policy.model.sample_time(1, device).expand(batch_size)
    return noise, time_values


def configure_trainable_parameters(
    policy,
    unfreeze_vlm_text_layers: int,
    freeze_action_path: bool,
    scene_decoder_only: bool,
    context_scene_decoder_only: bool,
) -> tuple[dict, list[torch.nn.Parameter]]:
    vlm = policy.model.vlm_with_expert
    text_layers = vlm.get_vlm_model().text_model.layers
    if unfreeze_vlm_text_layers < 0 or unfreeze_vlm_text_layers >= len(text_layers):
        raise ValueError(
            "--unfreeze-vlm-text-layers must be in [0, num_vlm_layers - 1]"
        )
    if context_scene_decoder_only:
        decoder_names = (
            f"{ADAPTER_NAME}.context_token_proj.",
            f"{ADAPTER_NAME}.target_query_proj.",
            f"{ADAPTER_NAME}.target_scene_action_decoder.",
        )
        for name, parameter in policy.named_parameters():
            parameter.requires_grad = any(
                decoder_name in name for decoder_name in decoder_names
            )
    elif scene_decoder_only:
        decoder_names = (
            f"{ADAPTER_NAME}.scene_input_proj.",
            f"{ADAPTER_NAME}.scene_action_decoder.",
        )
        for name, parameter in policy.named_parameters():
            parameter.requires_grad = any(
                decoder_name in name for decoder_name in decoder_names
            )
    elif freeze_action_path:
        for name, parameter in policy.named_parameters():
            parameter.requires_grad = ADAPTER_NAME in name

    # The final VLM layer is excluded because its prefix output is not consumed by
    # another action-attention layer. Select the deepest fully effective layers.
    first_layer = len(text_layers) - unfreeze_vlm_text_layers - 1
    layer_indices = list(range(first_layer, len(text_layers) - 1))
    monitored_parameters = []
    for layer_index in layer_indices:
        for parameter in text_layers[layer_index].parameters():
            parameter.requires_grad = True
            monitored_parameters.append(parameter)

    total_parameters = sum(parameter.numel() for parameter in policy.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    trainable_parameter_names = []
    if scene_decoder_only or context_scene_decoder_only:
        trainable_parameter_names = [
            name
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad
        ]
        expected_names = (
            {
                f"model.{ADAPTER_NAME}.context_token_proj.weight",
                f"model.{ADAPTER_NAME}.context_token_proj.bias",
                f"model.{ADAPTER_NAME}.target_query_proj.weight",
                f"model.{ADAPTER_NAME}.target_query_proj.bias",
                f"model.{ADAPTER_NAME}.target_scene_action_decoder.weight",
                f"model.{ADAPTER_NAME}.target_scene_action_decoder.bias",
            }
            if context_scene_decoder_only
            else {
                f"model.{ADAPTER_NAME}.scene_input_proj.weight",
                f"model.{ADAPTER_NAME}.scene_input_proj.bias",
                f"model.{ADAPTER_NAME}.scene_action_decoder.weight",
                f"model.{ADAPTER_NAME}.scene_action_decoder.bias",
            }
        )
        if set(trainable_parameter_names) != expected_names:
            raise RuntimeError(
                "Scene decoder freeze boundary mismatch: "
                f"expected={sorted(expected_names)}, found={trainable_parameter_names}"
            )
    monitored_parameter_count = sum(
        parameter.numel() for parameter in monitored_parameters
    )
    audit = {
        "freeze_action_path": freeze_action_path,
        "scene_decoder_only": scene_decoder_only,
        "context_scene_decoder_only": context_scene_decoder_only,
        "unfrozen_vlm_text_layer_indices": layer_indices,
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "monitored_vlm_parameter_count": monitored_parameter_count,
    }
    if scene_decoder_only or context_scene_decoder_only:
        audit["trainable_parameter_names"] = trainable_parameter_names
    return audit, monitored_parameters


def gradient_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().float().square().sum())
    return squared_norm**0.5


def make_policy_with_language_adapter(cfg, dataset_meta, checkpoint):
    """Load both pre-adapter and adapter checkpoints without weakening checks."""
    try:
        policy = make_policy(cfg=cfg.policy, ds_meta=dataset_meta)
    except RuntimeError as error:
        if "unexpected" not in str(error):
            raise
        cfg.policy.pretrained_path = None
        policy = make_policy(cfg=cfg.policy, ds_meta=dataset_meta)
        attach_language_action_adapter(policy)
        state = safetensors.torch.load_file(checkpoint / "model.safetensors", device="cpu")
        scene_decoder_in_checkpoint = any(
            key.startswith("model.language_action_adapter.scene_action_decoder.")
            for key in state
        )
        context_scene_decoder_in_checkpoint = any(
            key.startswith(
                "model.language_action_adapter.target_scene_action_decoder."
            )
            for key in state
        )
        state, _ = standardise_state_dict(
            state, set(policy.state_dict().keys()), verbose=False
        )
        state = {
            key: value for key, value in state.items()
            if not key.startswith(("normalize_inputs", "normalize_targets", "unnormalize_outputs"))
        }
        missing, unexpected = policy.load_state_dict(state, strict=False)
        allowed_missing = {
            key for key in missing
            if key.startswith(("normalize_inputs", "normalize_targets", "unnormalize_outputs"))
            or key.startswith("model.language_action_adapter.action_output_proj.")
            or key.startswith("model.language_action_adapter.action_film_proj.")
            or key.startswith("model.language_action_adapter.target_classifier.")
            or key.startswith("model.language_action_adapter.target_action_chunk_proj.")
            or key.startswith("model.language_action_adapter.target_action_hidden_proj.")
            or key.startswith("model.language_action_adapter.target_suffix_proj.")
            or key.startswith("model.language_action_adapter.scene_input_proj.")
            or key.startswith("model.language_action_adapter.scene_action_decoder.")
            or key.startswith("model.language_action_adapter.context_token_proj.")
            or key.startswith("model.language_action_adapter.target_query_proj.")
            or key.startswith(
                "model.language_action_adapter.target_scene_action_decoder."
            )
        }
        if unexpected or set(missing) != allowed_missing:
            raise RuntimeError(
                f"Adapter checkpoint load mismatch: missing={missing}, unexpected={unexpected}"
            )
        set_scene_action_decoder_enabled(policy, scene_decoder_in_checkpoint)
        set_context_scene_action_decoder_enabled(
            policy, context_scene_decoder_in_checkpoint
        )
        return policy
    attach_language_action_adapter(policy)
    return policy


def update_policy_with_shared_flow(
    tracker: MetricsTracker,
    policy,
    batch: dict,
    optimizer,
    grad_scaler: GradScaler,
    lr_scheduler,
    grad_clip_norm: float,
    contrastive_margin: float,
    contrastive_weight: float,
    target_classification_weight: float,
    scene_action_weight: float,
    monitored_parameters: list[torch.nn.Parameter],
    scene_decoder_only: bool,
    context_scene_decoder_only: bool,
) -> tuple[float, dict]:
    started = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    direct_decoder_only = scene_decoder_only or context_scene_decoder_only
    with (
        torch.autocast(device_type=device.type)
        if policy.config.use_amp
        else nullcontext()
    ):
        if direct_decoder_only:
            model_batch = policy.normalize_inputs(dict(batch))
            images, img_masks = policy.prepare_images(model_batch)
            state = policy.prepare_state(model_batch)
            lang_tokens, lang_masks = policy.prepare_language(model_batch)
            prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            if context_scene_decoder_only:
                compute_context_scene_action_chunk(
                    policy.model,
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                )
            loss = torch.zeros((), device=device)
            correct_per_sample = torch.zeros(TARGET_COUNT, device=device)
        else:
            noise, time_values = shared_flow_inputs(policy, TARGET_COUNT, device)
            loss, output = policy.forward(
                dict(batch), noise=noise, time=time_values
            )
            correct_per_sample = output["losses_after_rm_padding"].mean(
                dim=(1, 2)
            ).detach()
    target_loss = torch.zeros((), device=device)
    if target_classification_weight > 0.0:
        targets = batch["complementary_info.target_object_index"].reshape(-1).long()
        target_loss = F.cross_entropy(
            policy.model.language_action_adapter.classify_target(
                policy.model._language_action_condition
            ),
            targets,
        )
    scene_action_loss = torch.zeros((), device=device)
    if scene_action_weight > 0.0:
        scene_prediction = (
            policy.model._language_action_context_scene_chunk
            if context_scene_decoder_only
            else policy.model._language_action_scene_chunk
        )
        normalized_action = policy.normalize_targets(
            {"action": batch["action"]}
        )["action"]
        target_action = policy.prepare_action({"action": normalized_action})
        scene_losses = F.mse_loss(
            scene_prediction, target_action, reduction="none"
        )
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            valid = (~action_is_pad).unsqueeze(-1).expand_as(scene_losses)
            scene_action_loss = (scene_losses * valid).sum() / valid.sum().clamp_min(1)
        else:
            scene_action_loss = scene_losses.mean()
    total_loss = (
        loss
        + target_classification_weight * target_loss
        + scene_action_weight * scene_action_loss
    )
    grad_scaler.scale(total_loss).backward()
    ranking_losses = []
    negative_gaps = []
    margin_satisfaction = []
    if contrastive_weight > 0.0:
        tasks = list(batch["task"])
        for shift in (1, 2):
            wrong_batch = dict(batch)
            wrong_batch["task"] = tasks[shift:] + tasks[:shift]
            with (
                torch.autocast(device_type=device.type)
                if policy.config.use_amp
                else nullcontext()
            ):
                _, wrong_output = policy.forward(
                    wrong_batch, noise=noise, time=time_values
                )
            wrong_per_sample = wrong_output["losses_after_rm_padding"].mean(
                dim=(1, 2)
            )
            negative_gap = wrong_per_sample - correct_per_sample
            ranking_loss = torch.relu(contrastive_margin - negative_gap).mean()
            grad_scaler.scale(
                contrastive_weight * ranking_loss / (TARGET_COUNT - 1)
            ).backward()
            ranking_losses.append(float(ranking_loss.detach()))
            negative_gaps.append(float(negative_gap.detach().mean()))
            margin_satisfaction.append(
                float((negative_gap.detach() >= contrastive_margin).float().mean())
            )
    grad_scaler.unscale_(optimizer)
    gradient_audit = {}
    if direct_decoder_only:
        missing_gradients = []
        nonfinite_gradients = []
        gradient_norms = {}
        for name, parameter in policy.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.grad is None:
                missing_gradients.append(name)
                continue
            gradient_norm = float(parameter.grad.detach().float().norm())
            gradient_norms[name] = gradient_norm
            if not torch.isfinite(parameter.grad).all() or not math.isfinite(
                gradient_norm
            ):
                nonfinite_gradients.append(name)
        gradient_audit = {
            "gradient_norms": gradient_norms,
            "missing_gradients": missing_gradients,
            "nonfinite_gradients": nonfinite_gradients,
        }
        if missing_gradients or nonfinite_gradients:
            raise RuntimeError(f"Scene decoder gradient audit failed: {gradient_audit}")
    tracker.vlm_grad_norm = gradient_l2_norm(monitored_parameters)
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), grad_clip_norm, error_if_nonfinite=direct_decoder_only
    )
    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()
    if lr_scheduler is not None:
        lr_scheduler.step()
    if has_method(policy, "update"):
        policy.update()
    tracker.loss = total_loss.item()
    tracker.grad_norm = grad_norm.item()
    tracker.lr = optimizer.param_groups[0]["lr"]
    tracker.update_s = time.perf_counter() - started
    tracker.ranking_loss = (
        sum(ranking_losses) / len(ranking_losses) if ranking_losses else 0.0
    )
    tracker.negative_gap = (
        sum(negative_gaps) / len(negative_gaps) if negative_gaps else 0.0
    )
    tracker.margin_satisfaction = (
        sum(margin_satisfaction) / len(margin_satisfaction)
        if margin_satisfaction
        else 0.0
    )
    tracker.target_loss = float(target_loss.detach())
    tracker.scene_action_loss = float(scene_action_loss.detach())
    return tracker.vlm_grad_norm.val, gradient_audit


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    source_checkpoint_dir = checkpoint.parent
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing source model: {checkpoint}")
    if args.output_dir.exists() and not (
        args.verify_sampler_only or args.verify_update_only
    ):
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")
    if args.max_frame_exclusive < 1:
        raise ValueError("--max-frame-exclusive must be positive")
    if args.contrastive_margin < 0.0 or args.contrastive_weight < 0.0:
        raise ValueError("Contrastive margin and weight must be non-negative")
    if args.target_classification_weight < 0.0:
        raise ValueError("Target classification weight must be non-negative")
    if args.scene_action_weight < 0.0:
        raise ValueError("Scene action weight must be non-negative")
    direct_decoder_only = (
        args.scene_decoder_only or args.context_scene_decoder_only
    )
    if args.scene_decoder_only and args.context_scene_decoder_only:
        raise ValueError("Choose only one direct scene decoder mode")
    if direct_decoder_only and args.scene_action_weight <= 0.0:
        raise ValueError("Direct scene decoder training requires --scene-action-weight > 0")
    if direct_decoder_only and args.freeze_action_path:
        raise ValueError("Choose either a direct scene decoder or --freeze-action-path")
    if direct_decoder_only and (
        args.contrastive_weight > 0.0
        or args.target_classification_weight > 0.0
        or args.unfreeze_vlm_text_layers > 0
    ):
        raise ValueError(
            "Direct scene decoder training cannot update contrastive, classifier, or VLM paths"
        )
    if args.unfreeze_vlm_text_layers < 0:
        raise ValueError("--unfreeze-vlm-text-layers must be non-negative")

    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.policy.pretrained_path = checkpoint
    cfg.checkpoint_path = source_checkpoint_dir
    cfg.output_dir = args.output_dir.resolve()
    cfg.resume = True
    cfg.steps = args.total_steps
    cfg.batch_size = TARGET_COUNT
    cfg.save_freq = args.save_freq
    cfg.log_freq = args.log_freq
    cfg.eval_freq = 0
    cfg.num_workers = 0
    cfg.wandb.enable = False

    init_logging()
    if cfg.seed is not None:
        set_seed(cfg.seed)
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating paired curriculum dataset")
    dataset = make_dataset(cfg)
    sampler = ExactSceneTripletBatchSampler(
        dataset.episode_data_index, args.max_frame_exclusive
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,
        batch_sampler=sampler,
        pin_memory=device.type == "cuda",
    )
    first_batch = next(iter(loader))
    validate_triplet_batch(first_batch)
    first_state_delta = float(
        (
            first_batch["observation.state"]
            - first_batch["observation.state"][0:1]
        )
        .abs()
        .max()
    )
    contract = {
        "experiment": "G1LANG-exact-scene-initial-triplet-curriculum",
        "source_checkpoint": str(checkpoint),
        "target_order": list(TARGET_NAMES),
        "scene_groups": sampler.scene_count,
        "frame_range": [0, args.max_frame_exclusive - 1],
        "triplets_per_sampler_epoch": len(sampler),
        "batch_size": TARGET_COUNT,
        "shared_flow_noise_within_triplet": True,
        "shared_flow_time_within_triplet": True,
        "contrastive_margin": args.contrastive_margin,
        "contrastive_weight": args.contrastive_weight,
        "target_classification_weight": args.target_classification_weight,
        "scene_action_weight": args.scene_action_weight,
        "negative_task_shifts": [1, 2] if args.contrastive_weight > 0.0 else [],
        "requested_unfreeze_vlm_text_layers": args.unfreeze_vlm_text_layers,
        "requested_freeze_action_path": args.freeze_action_path,
        "requested_scene_decoder_only": args.scene_decoder_only,
        "requested_context_scene_decoder_only": args.context_scene_decoder_only,
        "first_verified_batch_episode_indices": [
            int(value) for value in first_batch["episode_index"].reshape(-1)
        ],
        "first_verified_batch_frame_indices": [
            int(value) for value in first_batch["frame_index"].reshape(-1)
        ],
        "first_verified_batch_max_state_delta_rad": first_state_delta,
        "total_steps": args.total_steps,
    }
    print(json.dumps(contract, indent=2), flush=True)
    if args.verify_sampler_only:
        return
    logging.info("Creating policy, optimizer, and scheduler")
    policy = make_policy_with_language_adapter(cfg, dataset.meta, checkpoint)
    language_adapter = policy.model.language_action_adapter
    if args.scene_decoder_only:
        set_scene_action_decoder_enabled(policy, True)
    if args.context_scene_decoder_only:
        set_context_scene_action_decoder_enabled(policy, True)
    trainable_audit, monitored_parameters = configure_trainable_parameters(
        policy,
        args.unfreeze_vlm_text_layers,
        args.freeze_action_path,
        args.scene_decoder_only,
        args.context_scene_decoder_only,
    )
    # Monitor the adapter in the same finite-gradient smoke gate used for
    # language-layer interventions.  This prevents a silently disconnected
    # residual path from being mistaken for a successful update.
    monitored_parameters.extend(language_adapter.parameters())
    trainable_audit["language_action_adapter"] = {
        "enabled": True,
        "parameter_names": [
            name for name, _ in language_adapter.named_parameters()
        ],
        "zero_initialized_output": True,
    }
    contract.update(trainable_audit)
    print(f"TRAINABLE_PARAMETER_AUDIT={json.dumps(trainable_audit)}", flush=True)
    if not args.verify_update_only:
        args.output_dir.mkdir(parents=True)
        (args.output_dir / "triplet_curriculum_contract.json").write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)
    try:
        step, optimizer, lr_scheduler = load_training_state(
            source_checkpoint_dir, optimizer, lr_scheduler
        )
    except (ValueError, RuntimeError) as error:
        # A pre-adapter checkpoint has a smaller optimizer parameter list.
        # Keep its model weights, reset optimizer moments, and preserve the
        # source step from the checkpoint directory for an auditable short
        # adapter intervention.
        if ADAPTER_NAME not in " ".join(dict(policy.named_parameters()).keys()):
            raise
        try:
            step = int(source_checkpoint_dir.name)
        except ValueError:
            raise error
        logging.warning(
            "Resetting optimizer state at source step %d after parameter-shape mismatch: %s",
            step,
            error,
        )
    if step >= cfg.steps:
        raise ValueError(f"Source step {step} must be below total steps {cfg.steps}")

    metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "ranking_loss": AverageMeter("rank", ":.3f"),
        "negative_gap": AverageMeter("ngap", ":.3f"),
        "margin_satisfaction": AverageMeter("msat", ":.2f"),
        "vlm_grad_norm": AverageMeter("vlmg", ":.3f"),
        "target_loss": AverageMeter("tgt", ":.3f"),
        "scene_action_loss": AverageMeter("scene", ":.3f"),
    }
    tracker = MetricsTracker(
        TARGET_COUNT,
        len(sampler) * TARGET_COUNT,
        dataset.num_episodes,
        metrics,
        initial_step=step,
    )
    data_iterator = cycle(loader)
    if args.verify_update_only:
        batch = next(data_iterator)
        validate_triplet_batch(batch)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=device.type == "cuda")
        before_update = (
            {
                name: parameter.detach().clone()
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            }
            if direct_decoder_only
            else {}
        )
        vlm_grad_norm, gradient_audit = update_policy_with_shared_flow(
            tracker,
            policy,
            batch,
            optimizer,
            grad_scaler,
            lr_scheduler,
            cfg.optimizer.grad_clip_norm,
            args.contrastive_margin,
            args.contrastive_weight,
            args.target_classification_weight,
            args.scene_action_weight,
            monitored_parameters,
            args.scene_decoder_only,
            args.context_scene_decoder_only,
        )
        if monitored_parameters and (
            not math.isfinite(vlm_grad_norm) or vlm_grad_norm <= 0.0
        ):
            raise RuntimeError("Monitored parameters received no finite gradient")
        parameter_deltas = (
            {
                name: float((parameter.detach() - before_update[name]).abs().max())
                for name, parameter in policy.named_parameters()
                if parameter.requires_grad
            }
            if before_update
            else {}
        )
        if before_update and not any(delta > 0.0 for delta in parameter_deltas.values()):
            raise RuntimeError("No trainable parameter changed during smoke update")
        print(
            "UPDATE_AUDIT="
            + json.dumps(
                {
                    "gradients": gradient_audit,
                    "parameter_max_abs_deltas": parameter_deltas,
                }
            ),
            flush=True,
        )
        print(f"CONTRASTIVE_UPDATE_SMOKE={tracker}", flush=True)
        return
    logging.info(
        "Start triplet curriculum at step %d; total=%d; triplets/epoch=%d",
        step,
        cfg.steps,
        len(sampler),
    )
    for _ in range(step, cfg.steps):
        started = time.perf_counter()
        batch = next(data_iterator)
        tracker.dataloading_s = time.perf_counter() - started
        validate_triplet_batch(batch)
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device, non_blocking=device.type == "cuda")
        update_policy_with_shared_flow(
            tracker,
            policy,
            batch,
            optimizer,
            grad_scaler,
            lr_scheduler,
            cfg.optimizer.grad_clip_norm,
            args.contrastive_margin,
            args.contrastive_weight,
            args.target_classification_weight,
            args.scene_action_weight,
            monitored_parameters,
            args.scene_decoder_only,
            args.context_scene_decoder_only,
        )
        step += 1
        tracker.step()
        if cfg.log_freq > 0 and step % cfg.log_freq == 0:
            logging.info("TRIPLET_CURRICULUM %s", tracker)
            tracker.reset_averages()
        if cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps):
            logging.info("Checkpoint triplet curriculum after step %d", step)
            destination = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(
                destination, step, cfg, policy, optimizer, lr_scheduler
            )
            (destination / "pretrained_model" / "triplet_curriculum_contract.json").write_text(
                json.dumps(contract, indent=2) + "\n", encoding="utf-8"
            )
            update_last_checkpoint(destination)
    logging.info("End of triplet curriculum training")


if __name__ == "__main__":
    main()
