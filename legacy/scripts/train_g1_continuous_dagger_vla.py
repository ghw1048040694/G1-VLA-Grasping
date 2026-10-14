#!/usr/bin/env python3
"""Train continuous action chunks with explicit visited-state correction supervision."""
from __future__ import annotations

import argparse, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from g1_continuous_dagger_vla import ARM_INDICES, HAND_INDICES, ContinuousDaggerVLA, save_checkpoint
from g1_direct_joint_candidate_vla import encode_tasks

LEFT_HAND = tuple(range(10, 17))
RIGHT_HAND = tuple(range(24, 31))
OPEN_HAND = torch.tensor([-.01, -.01, -.01, .01, .01, .01, .01])


class Data(Dataset):
    def __init__(self, cache, history, chunk, weighted, dagger_boost):
        self.cache, self.history, self.chunk, self.images = cache, history, chunk, None
        self.rows, weights = [], []
        self.tasks = [cache["tasks"][int(i)] for i in cache["task_index"].tolist()]
        self.ids, self.mask = encode_tasks(self.tasks, cache["vocab"])
        valid = cache["valid_frame"].bool().numpy()
        for episode, row in enumerate(valid):
            length = int(row.sum()); source = str(cache["source_root"][episode]).lower()
            correction = "recover" in source or "dagger" in source or "policy" in source
            for frame in range(max(0, length - chunk + 1)):
                self.rows.append((episode, frame, correction))
                base = float(cache["sample_weight"][episode, frame]) if weighted else 1.0
                weights.append(base * (dagger_boost if correction else 1.0))
        self.weights = np.asarray(weights, dtype=np.float64)
        if not self.rows: raise ValueError("No complete continuous action chunks")

    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        episode, frame, correction = self.rows[index]
        if self.images is None: self.images = np.load(self.cache["image_path"], mmap_mode="r")
        hist = np.maximum(np.arange(frame - self.history + 1, frame + 1), 0)
        target = int(self.cache["target"][episode]); arm = int(self.cache["active_arm"][episode, frame])
        obj = self.cache["objects"][episode, frame].reshape(3, 3)[target]
        sites = self.cache["contact_sites"][episode, frame].reshape(2, -1, 3)[arm]
        distance = torch.linalg.vector_norm(sites - obj, dim=-1).min().clamp(0, .5)
        action_plan = (
            self.cache["state"][episode, frame:frame+self.chunk].float()
            + self.cache["action_delta"][episode, frame:frame+self.chunk].float()
            - self.cache["state"][episode, frame].float()
        )
        # Teacher geometry is allowed only while preparing offline labels.
        # Keep the active hand open outside the contact-ready region; otherwise
        # a direct-continuation teacher starts closing at t=0 and the learner
        # predictably grasps empty air before its visual approach has converged.
        if float(distance) > .055:
            hand = LEFT_HAND if arm == 0 else RIGHT_HAND
            action_plan[:, hand] = OPEN_HAND - self.cache["state"][episode, frame, list(hand)].float()
        return {
            "images": torch.from_numpy(np.asarray(self.images[episode, hist]).copy()).permute(0,1,4,2,3).float().div_(255),
            "state": self.cache["state"][episode, hist].float(), "velocity": self.cache["velocity"][episode, hist].float(),
            "sites": self.cache["contact_sites"][episode, hist].float(), "tactile": self.cache["contact"][episode, hist].float(),
            # A deployable chunk is a sequence of future commanded joint
            # targets expressed relative to the state at planning time.  The
            # old local action_delta[f+i] labels were relative to different
            # future states and therefore could not be executed as a chunk.
            "action": action_plan,
            "target": torch.tensor(target), "arm": torch.tensor(arm), "phase": self.cache["task_phase"][episode, frame].long(),
            "distance": distance.reshape(1), "contact": self.cache["contact"][episode, frame].float(),
            "ids": self.ids[episode], "mask": self.mask[episode], "correction": torch.tensor(float(correction)),
        }


def get_stats(cache, action_chunk):
    valid = cache["valid_frame"].bool(); result = {}
    for name in ("state", "velocity", "contact_sites"):
        value = cache[name][valid].float(); result[name+"_mean"] = value.mean(0); result[name+"_std"] = value.std(0).clamp_min(1e-3)
    plans = []
    for episode, mask in enumerate(valid):
        length = int(mask.sum())
        for frame in range(max(0, length-action_chunk+1)):
            plan = (
                cache["state"][episode, frame:frame+action_chunk].float()
                + cache["action_delta"][episode, frame:frame+action_chunk].float()
                - cache["state"][episode, frame].float()
            )
            target = int(cache["target"][episode]); arm = int(cache["active_arm"][episode, frame])
            obj = cache["objects"][episode, frame].reshape(3,3)[target]
            sites = cache["contact_sites"][episode, frame].reshape(2,-1,3)[arm]
            if float(torch.linalg.vector_norm(sites-obj,dim=-1).min()) > .055:
                hand = LEFT_HAND if arm == 0 else RIGHT_HAND
                plan[:,hand] = OPEN_HAND-cache["state"][episode,frame,list(hand)].float()
            plans.append(plan)
    plan = torch.cat(plans, 0)
    result["action_delta_mean"] = plan.mean(0)
    result["action_delta_std"] = plan.std(0).clamp_min(1e-3)
    return result


def forward(model, batch, stats):
    output = model(batch["images"], (batch["state"]-stats["state_mean"])/stats["state_std"],
                   (batch["velocity"]-stats["velocity_mean"])/stats["velocity_std"],
                   (batch["sites"]-stats["contact_sites_mean"])/stats["contact_sites_std"],
                   batch["tactile"], batch["ids"], batch["mask"])
    truth = (batch["action"]-stats["action_delta_mean"])/stats["action_delta_std"]
    per_step = F.smooth_l1_loss(output["action"], truth, reduction="none").mean(-1)
    horizon_weight = torch.linspace(1.5, .65, truth.shape[1], device=truth.device)
    arm_step = F.smooth_l1_loss(output["action"][:,:,ARM_INDICES], truth[:,:,ARM_INDICES], reduction="none").mean(-1)
    hand_step = F.smooth_l1_loss(output["action"][:,:,HAND_INDICES], truth[:,:,HAND_INDICES], reduction="none").mean(-1)
    action_loss = (arm_step*horizon_weight).mean() + .45*(hand_step*horizon_weight).mean()
    endpoint_loss = F.smooth_l1_loss(output["action"][:,:,ARM_INDICES].sum(1), truth[:,:,ARM_INDICES].sum(1))
    hand_endpoint_loss = F.smooth_l1_loss(output["action"][:,:,HAND_INDICES].sum(1), truth[:,:,HAND_INDICES].sum(1))
    phase_valid = batch["phase"] >= 0
    phase_loss = F.cross_entropy(output["phase_logits"][phase_valid], batch["phase"][phase_valid]) if phase_valid.any() else output["phase_logits"].sum()*0
    loss = action_loss + .25*endpoint_loss + .08*hand_endpoint_loss + .20*F.cross_entropy(output["target_logits"],batch["target"]) + .10*F.cross_entropy(output["active_arm_logits"],batch["arm"]) + .18*phase_loss + .20*F.smooth_l1_loss(output["distance"],batch["distance"]/.25) + .12*F.binary_cross_entropy_with_logits(output["contact_logits"],batch["contact"])
    predicted = output["action"]*stats["action_delta_std"]+stats["action_delta_mean"]
    return loss, output, predicted


@torch.no_grad()
def evaluate(model, loader, stats, device):
    model.eval(); sq=first_sq=0.; count=first_count=rows=target_ok=phase_ok=phase_n=0; correction_sq=0.; correction_n=0
    for raw in loader:
        b={k:v.to(device) for k,v in raw.items()}; _,o,p=forward(model,b,stats); n=len(b["target"]); rows+=n
        sq+=float((p-b["action"]).square().sum()); count+=p.numel(); first_sq+=float((p[:,0]-b["action"][:,0]).square().sum()); first_count+=p[:,0].numel()
        target_ok+=int((o["target_logits"].argmax(-1)==b["target"]).sum()); valid=b["phase"]>=0; phase_ok+=int((o["phase_logits"].argmax(-1)[valid]==b["phase"][valid]).sum()); phase_n+=int(valid.sum())
        corr=b["correction"]>.5
        if corr.any(): correction_sq+=float((p[corr,0]-b["action"][corr,0]).square().sum()); correction_n+=p[corr,0].numel()
    return {"chunk_action_rmse_rad":math.sqrt(sq/max(count,1)),"first_action_rmse_rad":math.sqrt(first_sq/max(first_count,1)),"correction_first_action_rmse_rad":math.sqrt(correction_sq/max(correction_n,1)),"target_accuracy":target_ok/max(rows,1),"phase_accuracy":phase_ok/max(phase_n,1)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--cache-dir",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int,default=32); p.add_argument("--history-frames",type=int,default=4); p.add_argument("--action-chunk",type=int,default=8); p.add_argument("--dagger-boost",type=float,default=4); p.add_argument("--samples-per-epoch",type=int,default=0); p.add_argument("--factorized-actions",action="store_true"); p.add_argument("--init-checkpoint",type=Path, help="Warm-start model weights while recomputing normalization statistics from the new cache."); p.add_argument("--learning-rate",type=float,default=3e-4); p.add_argument("--seed",type=int,default=20260812); p.add_argument("--device",default="cuda"); a=p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); device=torch.device(a.device if a.device=="cpu" or torch.cuda.is_available() else "cpu")
    train=torch.load(a.cache_dir/"train.pt",map_location="cpu",weights_only=False); val=torch.load(a.cache_dir/"validation.pt",map_location="cpu",weights_only=False)
    stats={k:v.to(device) for k,v in get_stats(train,a.action_chunk).items()}; ts=Data(train,a.history_frames,a.action_chunk,True,a.dagger_boost); vs=Data(val,a.history_frames,a.action_chunk,False,1)
    draws=a.samples_per_epoch or len(ts); tl=DataLoader(ts,batch_size=a.batch_size,sampler=WeightedRandomSampler(ts.weights,draws,replacement=True)); vl=DataLoader(vs,batch_size=a.batch_size)
    model=ContinuousDaggerVLA(len(train["vocab"]),int(train["contact_sites"].shape[-1]),a.history_frames,a.action_chunk,factorized_actions=a.factorized_actions).to(device)
    if a.init_checkpoint is not None:
        checkpoint = torch.load(a.init_checkpoint, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        expected = {"history_frames": a.history_frames, "action_chunk": a.action_chunk, "factorized_actions": a.factorized_actions}
        for key, value in expected.items():
            if key in config and config[key] != value:
                raise ValueError(f"Warm-start config mismatch for {key}: {config[key]} != {value}")
        model.load_state_dict(checkpoint["model"], strict=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=2e-4)
    a.output_dir.mkdir(parents=True,exist_ok=True); best=math.inf; history=[]; metadata={"cache_dir":str(a.cache_dir),"dagger_boost":a.dagger_boost,"learning_rate":a.learning_rate,"init_checkpoint":str(a.init_checkpoint) if a.init_checkpoint else None,"train_correction_rows":int(sum(r[2] for r in ts.rows)),"validation_correction_rows":int(sum(r[2] for r in vs.rows))}
    for epoch in range(a.epochs):
        model.train(); total=n=0
        for raw in tl:
            b={k:v.to(device) for k,v in raw.items()}; loss,_,_=forward(model,b,stats); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); total+=float(loss)*len(b["target"]); n+=len(b["target"])
        report=evaluate(model,vl,stats,device); report.update(epoch=epoch+1,train_loss=total/max(n,1)); history.append(report); print("CONTINUOUS_DAGGER_EPOCH="+json.dumps(report),flush=True)
        score=report["correction_first_action_rmse_rad"]*2+report["first_action_rmse_rad"]
        if score<best: best=score; save_checkpoint(a.output_dir/"best.pt",model,train["vocab"],stats,{**metadata,"best_epoch":epoch+1})
    summary={"experiment":"G1LANG-continuous-action-chunk-visited-state-dagger-v2","action_representation":"future commanded joint targets relative to current planning state","best_score":best,"metadata":metadata,"history":history,"final":history[-1],"privileged_positions_at_inference":False}; (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__": main()
