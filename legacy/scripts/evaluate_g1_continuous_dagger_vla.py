#!/usr/bin/env python3
"""Strict closed-loop evaluation of the continuous DAgger VLA."""
from __future__ import annotations
import argparse, json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
import evaluate_g1_language_pick_place as base
from g1_continuous_dagger_vla import load_checkpoint
from g1_direct_joint_candidate_vla import encode_tasks
from g1_finger_contact_geometry import finger_contact_sites


class Policy:
    def __init__(self, checkpoint, device, limit):
        self.device=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu")
        self.model,self.vocab,self.stats,self.checkpoint=load_checkpoint(checkpoint,str(self.device)); self.limit=limit
        self.config=SimpleNamespace(chunk_size=1,max_action_dim=31); self.needs_temporal_history=True; self.reset()
    def reset(self): self.ctx=None; self.history=deque(maxlen=self.model.history_frames); self.reports=[]
    def set_planner_context(self,ctx): self.ctx=ctx
    def _sites(self): return torch.from_numpy(finger_contact_sites(self.ctx["model"],self.ctx["data"],self.ctx["finger_contact_geom_ids"])).float().to(self.device).reshape(1,-1)
    def _tactile(self):
        geoms=set(self.ctx["finger_contact_geom_ids"])|set(self.ctx["contact_pad_ids"]); bodies=set(self.ctx["object_body_ids"].values()); m,d=self.ctx["model"],self.ctx["data"]
        active=any((c.geom1 in geoms and int(m.geom_bodyid[c.geom2]) in bodies) or (c.geom2 in geoms and int(m.geom_bodyid[c.geom1]) in bodies) for c in d.contact[:d.ncon])
        return torch.tensor([[float(active)]],device=self.device)
    def observe_history(self,batch):
        d,c=self.ctx["data"],self.ctx["controlled"]; velocity=torch.tensor([[d.qvel[c[n]["qvel_id"]] for n in self.ctx["upper_names"]]],device=self.device,dtype=torch.float32)
        self.history.append({"images":torch.stack([batch[k][0].to(self.device) for k in base.CAMERAS],0),"state":batch["observation.state"].to(self.device),"velocity":velocity,"sites":self._sites(),"tactile":self._tactile(),"task":batch["task"]})
    def _stack(self):
        rows=list(self.history); rows=[rows[0]]*(self.model.history_frames-len(rows))+rows
        return {k:(torch.stack([x[k] for x in rows],0).unsqueeze(0) if k=="images" else torch.stack([x[k] for x in rows],1)) for k in ("images","state","velocity","sites","tactile")}
    @torch.no_grad()
    def predict_action_chunk(self,batch,noise=None):
        obs=self._stack(); task=self.history[-1]["task"]; ids,mask=encode_tasks(task if isinstance(task,list) else [task],self.vocab)
        out=self.model(obs["images"],(obs["state"]-self.stats["state_mean"])/self.stats["state_std"],(obs["velocity"]-self.stats["velocity_mean"])/self.stats["velocity_std"],(obs["sites"]-self.stats["contact_sites_mean"])/self.stats["contact_sites_std"],obs["tactile"],ids.to(self.device),mask.to(self.device))
        delta=(out["action"][0]*self.stats["action_delta_std"]+self.stats["action_delta_mean"]).clamp(-self.limit,self.limit)
        self.reports.append({"target_index":int(out["target_logits"].argmax(-1)[0]),"phase_index":int(out["phase_logits"].argmax(-1)[0]),"active_arm_index":int(out["active_arm_logits"].argmax(-1)[0]),"predicted_hand_object_distance_m":float(out["distance"][0,0]*.25),"predicted_contact_probability":float(out["contact_logits"].sigmoid()[0,0]),"first_delta_l2_rad":float(torch.linalg.vector_norm(delta[0]))})
        # Execute the learned future target sequence as an actual chunk.  Each
        # row is relative to the state at this planning instant, not a local
        # delta tied to an unknown future observation.
        return (obs["state"][0,-1][None]+delta).unsqueeze(0)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--asset",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--train-root",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--episodes",type=int,default=6); p.add_argument("--target",choices=base.OBJECT_NAMES); p.add_argument("--seed",type=int,default=20260831); p.add_argument("--control-fps",type=int,default=15); p.add_argument("--duration-s",type=float,default=8); p.add_argument("--layout-mode",choices=("slot_jitter","random"),default="random"); p.add_argument("--position-jitter-m",type=float,default=.018); p.add_argument("--action-delta-limit-rad",type=float,default=.08); p.add_argument("--replan-steps",type=int,default=4); p.add_argument("--body-gain-scale",type=float,default=5); p.add_argument("--hand-gain-scale",type=float,default=6); p.add_argument("--observation-width",type=int,default=96); p.add_argument("--observation-height",type=int,default=64); p.add_argument("--video-width",type=int,default=320); p.add_argument("--video-height",type=int,default=240); p.add_argument("--disable-video",action="store_true"); p.add_argument("--save-observation-videos",action="store_true"); p.add_argument("--device",default="cuda"); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True); a.disable_assisted_grasp=True
    if not 1 <= a.replan_steps <= 8: raise ValueError("replan steps must fit the learned eight-step chunk")
    names=json.loads((a.train_root/"meta/info.json").read_text())["features"]["action"]["names"]; policy=Policy(a.checkpoint,a.device,a.action_delta_limit_rad)
    specs=base.evaluation_specs(a.episodes if not a.target else a.episodes*3,a.seed,a.position_jitter_m,a.layout_mode)
    if a.target: specs=[s for s in specs if s["target_object"]==a.target][:a.episodes]
    reports=[]
    for spec in specs:
        report=base.run_episode(policy,spec,a,names); report["continuous_policy_reports"]=policy.reports; reports.append(report); print(f"CONTINUOUS_EVAL={len(reports)}/{len(specs)} target={report['target_object']} contact={report['hand_object_contact_candidate_steps']} success={report['task_success']}",flush=True)
    summary={"experiment":"G1LANG-continuous-action-chunk-visited-state-dagger-v1-strict-eval","checkpoint":str(a.checkpoint),"episodes":len(reports),"target_filter":a.target,"assisted_grasp_enabled":False,"privileged_positions_at_inference":False,"real_contact_rate":float(np.mean([r["hand_object_contact_candidate_steps"]>0 for r in reports])),"full_pick_place_success_rate":float(np.mean([r["task_success"] for r in reports])),"reports":reports}; (a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__": main()
