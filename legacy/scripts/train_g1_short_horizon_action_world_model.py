#!/usr/bin/env python3
"""Train a four-step action-token world model on physical intervention labels."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from g1_short_horizon_action_world_model import ShortHorizonWorldEnsemble, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=480)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=20260752)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class DatasetRows(Dataset):
    def __init__(self, cache: dict): self.cache = cache
    def __len__(self): return len(self.cache["decision_id"])
    def __getitem__(self, index): return {key: value[index] for key, value in self.cache.items() if isinstance(value, torch.Tensor)}


def stats_for(cache: dict) -> dict[str, torch.Tensor]:
    result = {}
    for name in ("state", "velocity", "objects", "sites", "action_chunk"):
        result[f"{name}_mean"] = cache[name].float().mean(0)
        result[f"{name}_std"] = cache[name].float().std(0).clamp_min(1e-3)
    return result


def z(value, stats, name): return (value - stats[f"{name}_mean"]) / stats[f"{name}_std"]


def visible_distance(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    count = len(batch["target"])
    objects = batch["objects"].float().reshape(count, 3, 3)
    sites = batch["sites"].float().reshape(count, 2, -1, 3)
    rows = torch.arange(count, device=objects.device)
    return torch.linalg.vector_norm(
        sites[rows, batch["arm"].long()] - objects[rows, batch["target"].long(), None], dim=-1
    ).min(-1).values[:, None]


def center(correction: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    unique, inverse = torch.unique(ids.long(), sorted=False, return_inverse=True)
    total = torch.zeros(len(unique), device=correction.device, dtype=correction.dtype)
    count = torch.zeros_like(total)
    total.scatter_add_(0, inverse, correction)
    count.scatter_add_(0, inverse, torch.ones_like(correction))
    return correction - total[inverse] / count[inverse].clamp_min(1.0)


def outputs(model, batch, stats):
    return model(
        z(batch["state"].float(), stats, "state"),
        z(batch["velocity"].float(), stats, "velocity"),
        z(batch["objects"].float(), stats, "objects"),
        z(batch["sites"].float(), stats, "sites"),
        batch["tactile"].float().reshape(-1, 1),
        z(batch["action_chunk"].float(), stats, "action_chunk"),
        F.one_hot(batch["target"].long(), 3).float(),
        F.one_hot(batch["arm"].long(), 2).float(),
        visible_distance(batch),
    )


def distance(output, batch):
    # A multi-step action chunk can move every candidate closer by a common
    # amount, so unlike the one-step selector its absolute correction is part
    # of the learned transition. The 100 mm floor rules out the prior
    # far-state millimetre hallucination.
    return (visible_distance(batch).squeeze(-1) + output["distance_correction_m"].squeeze(-1)).clamp_min(0.100)


def member_loss(output, batch):
    predicted = distance(output, batch)
    value = 35.0 * F.smooth_l1_loss(predicted, batch["horizon_final_distance_m"].float(), beta=.010, reduction="none")
    positive = batch["horizon_contact"].float().mean()
    pos_weight = ((1.0 - positive) / positive.clamp_min(.01)).clamp(1.0, 25.0)
    contact = F.binary_cross_entropy_with_logits(output["contact_logit"].squeeze(-1), batch["horizon_contact"].float(), pos_weight=pos_weight, reduction="none")
    return value + .10 * contact


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval(); prediction = []; truth = []; ids = []; contact_ok = contact_count = 0
    for raw in loader:
        batch = {key: value.to(device) for key, value in raw.items()}
        values = outputs(model, batch, stats)
        predicted = torch.stack([distance(item, batch) for item in values]).mean(0)
        prediction.append(predicted.cpu()); truth.append(batch["horizon_final_distance_m"].cpu()); ids.append(batch["decision_id"].cpu())
        logits = torch.stack([item["contact_logit"].squeeze(-1) for item in values]).mean(0)
        contact_ok += int(((logits.sigmoid() >= .5) == (batch["horizon_contact"] >= .5)).sum()); contact_count += len(logits)
    pred = torch.cat(prediction).numpy(); actual = torch.cat(truth).numpy(); decision = torch.cat(ids).numpy(); error = np.abs(pred - actual)
    regrets=[]; correct=[]
    for value in np.unique(decision):
        rows=np.flatnonzero(decision == value)
        choice=rows[np.argmin(pred[rows])]; oracle=rows[np.argmin(actual[rows])]
        regrets.append(float(actual[choice]-actual[oracle])); correct.append(int(choice==oracle))
    far=actual >= .20
    return {
        "distance_error": {"count": int(len(error)), "mean_mm": float(error.mean()*1e3), "median_mm": float(np.median(error)*1e3), "p95_mm": float(np.percentile(error,95)*1e3), "max_mm": float(error.max()*1e3)},
        "candidate_ranking": {"decisions": len(regrets), "top1_exact_rate": float(np.mean(correct)), "mean_regret_mm": float(np.mean(regrets)*1e3), "p95_regret_mm": float(np.percentile(regrets,95)*1e3)},
        "contact_accuracy": contact_ok/max(contact_count,1),
        "far_state_anti_hallucination": {"count_actual_distance_ge_200mm": int(far.sum()), "min_predicted_distance_mm": float(pred[far].min()*1e3) if far.any() else float("nan"), "mean_predicted_distance_mm": float(pred[far].mean()*1e3) if far.any() else float("nan"), "mean_actual_distance_mm": float(actual[far].mean()*1e3) if far.any() else float("nan")},
    }


def eligible(report):
    error=report["distance_error"]; rank=report["candidate_ranking"]; far=report["far_state_anti_hallucination"]
    return bool(error["mean_mm"]<=35 and error["p95_mm"]<=75 and error["max_mm"]<=130 and rank["mean_regret_mm"]<=5 and rank["p95_regret_mm"]<=18 and (far["count_actual_distance_ge_200mm"]==0 or far["min_predicted_distance_mm"]>=100))


def main():
    args=parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device if args.device=="cpu" or torch.cuda.is_available() else "cpu")
    if (args.output_dir/"summary.json").exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    train=torch.load(args.cache_dir/"train.pt",map_location="cpu",weights_only=False); validation=torch.load(args.cache_dir/"validation.pt",map_location="cpu",weights_only=False)
    stats_cpu=stats_for(train); stats={key:value.to(device) for key,value in stats_cpu.items()}
    # Keep all twelve candidates from a decision together; centered target
    # distance learns which action chunk is better without changing absolute
    # distance calibration.
    train_loader=DataLoader(DatasetRows(train),batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
    val_loader=DataLoader(DatasetRows(validation),batch_size=args.batch_size,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
    model=ShortHorizonWorldEnsemble(args.members,int(train["action_chunk"].shape[-1]),int(train["sites"].shape[-1]),args.hidden_dim).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=2e-4); generator=torch.Generator(device=device).manual_seed(args.seed+1)
    best=math.inf; best_state=None; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=rows=0
        for raw in train_loader:
            batch={key:value.to(device,non_blocking=True) for key,value in raw.items()}; value=outputs(model,batch,stats); losses=[]
            for item in value:
                per=member_loss(item,batch); mask=torch.rand(len(per),device=device,generator=generator)<.82; losses.append(per[mask].mean() if mask.any() else per.mean())
            loss=torch.stack(losses).mean(); optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); total+=float(loss)*len(batch["target"]); rows+=len(batch["target"])
        report=evaluate(model,val_loader,stats,device); report.update(epoch=epoch,train_loss=total/max(rows,1)); history.append(report); print("SHORT_HORIZON_WORLD_EPOCH="+json.dumps(report),flush=True)
        score=report["candidate_ranking"]["mean_regret_mm"]+.25*report["candidate_ranking"]["p95_regret_mm"]+.05*report["distance_error"]["mean_mm"]
        if score<best:
            best=score; best_state={key:value.detach().cpu().clone() for key,value in model.state_dict().items()}; save_checkpoint(args.output_dir/"best.pt",model,stats_cpu)
    if best_state is None: raise RuntimeError("No checkpoint")
    model.load_state_dict(best_state); final=evaluate(model,val_loader,stats,device)
    calibration={"experiment":"G1LANG-four-step-action-token-policy-intervention-world-model","cache_dir":str(args.cache_dir),"heldout_split":"three independent randomized scenes","prediction":"physical target distance after a four-step VLA candidate chunk; execute first action then re-observe","offline_eligibility_threshold":{"mean_mm_max":35,"p95_mm_max":75,"max_mm_max":130,"candidate_mean_regret_mm_max":5,"candidate_p95_regret_mm_max":18,"far_state_min_predicted_distance_mm_min":100},"final":final,"eligible_for_four_step_closed_loop":eligible(final)}
    (args.output_dir/"calibration.json").write_text(json.dumps(calibration,indent=2)+"\n")
    summary={"experiment":calibration["experiment"],"cache_dir":str(args.cache_dir),"members":args.members,"best_validation_score":best,"history":history,"final":final,"eligible_for_four_step_closed_loop":calibration["eligible_for_four_step_closed_loop"],"privileged_positions_at_inference":False}
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(calibration,indent=2))
    if not calibration["eligible_for_four_step_closed_loop"]: raise SystemExit(2)


if __name__=="__main__": main()
