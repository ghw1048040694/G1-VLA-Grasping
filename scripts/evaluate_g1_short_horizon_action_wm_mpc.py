#!/usr/bin/env python3
"""Strict closed-loop MuJoCo gate for four-step learned world-model MPC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

import evaluate_g1_language_pick_place as base
from evaluate_g1_policy_intervention_wm_mpc import (
    CalibratedInterventionMPCPolicy as OneStepUtilities,
    physical_distance_audit,
)
from g1_action_token_vla import load_checkpoint as load_vla
from g1_direct_joint_candidate_vla import encode_tasks
from g1_short_horizon_action_world_model import load_checkpoint as load_world


class ShortHorizonActionMPCPolicy(OneStepUtilities):
    def __init__(self, vla_checkpoint, world_checkpoint, device, proposal_count, limit, horizon, uncertainty_weight):
        self.device=torch.device(device if device=="cpu" or torch.cuda.is_available() else "cpu")
        self.vla,self.vocab,self.stats,self.codebook,_=load_vla(vla_checkpoint,str(self.device))
        self.world,self.world_stats=load_world(world_checkpoint,str(self.device))
        if self.vla.contact_site_dim != self.world.contact_site_dim: raise ValueError("VLA/world geometry mismatch")
        if horizon < 2 or horizon > self.vla.action_chunk or self.world.action_dim != horizon*31: raise ValueError("World action chunk horizon disagrees with checkpoint")
        self.proposal_count=min(int(proposal_count),self.vla.token_count); self.limit=float(limit); self.horizon=int(horizon); self.uncertainty_weight=float(uncertainty_weight)
        self.config=SimpleNamespace(chunk_size=1,max_action_dim=31); self.needs_temporal_history=True; self.reset()

    def _visible_distance(self,batch):
        count=len(batch["target"]); objects=batch["objects"].reshape(count,3,3); sites=batch["sites"].reshape(count,2,-1,3); rows=torch.arange(count,device=self.device)
        return torch.linalg.vector_norm(sites[rows,batch["arm"]]-objects[rows,batch["target"],None],dim=-1).min(-1).values

    @torch.no_grad()
    def predict_action_chunk(self,batch,noise=None):
        del batch,noise
        observation=self._stack(); ids,mask=encode_tasks(self.history[-1]["task"],self.vocab)
        output=self.vla(observation["images"],(observation["state"]-self.stats["state_mean"])/self.stats["state_std"],(observation["velocity"]-self.stats["velocity_mean"])/self.stats["velocity_std"],(observation["sites"]-self.stats["contact_sites_mean"])/self.stats["contact_sites_std"],observation["tactile"],ids.to(self.device),mask.to(self.device))
        target=int(output["target_logits"].argmax(-1)[0]); arm=int(output["active_arm_logits"].argmax(-1)[0]); objects=output["object_mean"]*self.stats["objects_std"]+self.stats["objects_mean"]
        token_ids=output["action_token_logits"][0].topk(self.proposal_count).indices
        chunks=(self.codebook[token_ids,:self.horizon]+output["candidate_action_residual"][0,token_ids,:self.horizon]).clamp(-self.limit,self.limit)
        state=observation["state"][0,-1]; previous=torch.as_tensor(self.ctx["previous_action"],dtype=torch.float32,device=self.device); lower=torch.as_tensor(self.ctx["action_lower"],dtype=torch.float32,device=self.device); upper=torch.as_tensor(self.ctx["action_upper"],dtype=torch.float32,device=self.device)
        requested=(state[None]+chunks[:,0]).clamp(lower,upper); command=previous[None]+(requested-previous[None]).clamp(-self.limit,self.limit); count=len(command)
        wm={"state":state[None].expand(count,-1),"velocity":observation["velocity"][0,-1].expand(count,-1),"objects":objects.expand(count,-1),"sites":observation["sites"][0,-1].expand(count,-1),"tactile":observation["tactile"][0,-1].expand(count,-1),"action_chunk":chunks.flatten(1),"target":torch.full((count,),target,dtype=torch.long,device=self.device),"arm":torch.full((count,),arm,dtype=torch.long,device=self.device)}
        visible=self._visible_distance(wm)[:,None]
        members=self.world(self._z(wm["state"],"state"),self._z(wm["velocity"],"velocity"),self._z(wm["objects"],"objects"),self._z(wm["sites"],"sites"),wm["tactile"],self._z(wm["action_chunk"],"action_chunk"),F.one_hot(wm["target"],3).float(),F.one_hot(wm["arm"],2).float(),visible)
        distance=torch.stack([(visible.squeeze(-1)+item["distance_correction_m"].squeeze(-1)).clamp_min(.100) for item in members])
        mean=distance.mean(0); std=distance.std(0,correction=0); selected=int((mean+self.uncertainty_weight*std).argmin())
        tactile=float(observation["tactile"][0,-1])
        mode="four_step_action_token_learned_world_model_mpc"
        if tactile>=.5: selected=0; mode="measured_contact_handoff_to_vla_top_proposal"
        self.last_target_probabilities=output["target_logits"].softmax(-1)[0].cpu().tolist(); self.last_active_arm_probabilities=output["active_arm_logits"].softmax(-1)[0].cpu().tolist(); self.last_object_position=objects[0].cpu().reshape(3,3).tolist(); self.last_tactile_contact=tactile
        self.candidate_reports.append({"target_index":target,"active_arm_index":arm,"selection_mode":mode,"selected_rank":selected,"selected_token":int(token_ids[selected]),"predicted_selected_distance_mm":float(mean[selected]*1e3),"predicted_selected_distance_std_mm":float(std[selected]*1e3),"candidate_tokens":token_ids.cpu().tolist(),"candidate_distance_mm":(mean*1e3).cpu().tolist(),"candidate_distance_std_mm":(std*1e3).cpu().tolist(),"horizon_steps":self.horizon,"candidate_mean_abs_delta_mrad":(chunks[:,0].abs().mean(-1)*1e3).cpu().tolist()})
        return command[selected].reshape(1,1,31)


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset",type=Path,required=True); parser.add_argument("--vla-checkpoint",type=Path,required=True); parser.add_argument("--world-model-checkpoint",type=Path,required=True); parser.add_argument("--train-root",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--episodes",type=int,default=6); parser.add_argument("--seed",type=int,default=20260753); parser.add_argument("--control-fps",type=int,default=15); parser.add_argument("--duration-s",type=float,default=14); parser.add_argument("--layout-mode",choices=("slot_jitter","random"),default="random"); parser.add_argument("--position-jitter-m",type=float,default=.018); parser.add_argument("--proposal-count",type=int,default=12); parser.add_argument("--horizon",type=int,default=4); parser.add_argument("--action-delta-limit-rad",type=float,default=.08); parser.add_argument("--body-gain-scale",type=float,default=5); parser.add_argument("--hand-gain-scale",type=float,default=6); parser.add_argument("--uncertainty-weight",type=float,default=.5); parser.add_argument("--observation-width",type=int,default=160); parser.add_argument("--observation-height",type=int,default=120); parser.add_argument("--video-width",type=int,default=320); parser.add_argument("--video-height",type=int,default=240); parser.add_argument("--disable-video",action="store_true"); parser.add_argument("--save-observation-videos",action="store_true"); parser.add_argument("--device",default="cuda")
    return parser.parse_args()


def main():
    args=parse_args()
    if args.episodes<3 or args.episodes%3: raise ValueError("--episodes must be divisible by three")
    if (args.output_dir/"summary.json").exists(): raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=True); args.disable_assisted_grasp=True; args.replan_steps=1
    names=json.loads((args.train_root/"meta/info.json").read_text())["features"]["action"]["names"]
    policy=ShortHorizonActionMPCPolicy(args.vla_checkpoint,args.world_model_checkpoint,args.device,args.proposal_count,args.action_delta_limit_rad,args.horizon,args.uncertainty_weight)
    reports=[]
    for spec in base.evaluation_specs(args.episodes,args.seed,args.position_jitter_m,args.layout_mode):
        report=base.run_episode(policy,spec,args,names); report.update(predicted_target_probabilities=policy.last_target_probabilities,predicted_active_arm_probabilities=policy.last_active_arm_probabilities,camera_object_position_m=policy.last_object_position,measured_tactile_contact=policy.last_tactile_contact,world_model_candidates=policy.candidate_reports); report["physical_world_model_distance_audit"]=physical_distance_audit(report); reports.append(report)
    contacts=[r["hand_object_contact_candidate_steps"]>0 for r in reports]; language=float(np.mean([r["world_model_candidates"][-1]["target_index"]==base.OBJECT_NAMES.index(r["target_object"]) for r in reports])); audit=[r["physical_world_model_distance_audit"]["selected_world_model_distance_error_mm"]["mean"] for r in reports]
    summary={"experiment":"G1LANG-four-step-action-token-learned-world-model-mpc-strict-closed-loop","vla_checkpoint":str(args.vla_checkpoint),"world_model_checkpoint":str(args.world_model_checkpoint),"episodes":len(reports),"input_contract":"raw RGB x3 + language + state31 + qvel31 + measured finger sites + tactile","action_source":"VLA action chunks selected by a learned four-step physical world model; only action one is executed before re-observation","world_model_rollout_steps":args.horizon,"privileged_positions_at_inference":False,"assisted_grasp_enabled":False,"equality_constraints":False,"magnet_enabled":False,"reports":reports,"language_target_accuracy":language,"real_contact_rate":float(np.mean(contacts)),"no_grasp_rate":float(np.mean([r["grabbed_object"] is None for r in reports])),"full_pick_place_success_rate":float(np.mean([r["task_success"] for r in reports])),"mean_deployed_world_model_distance_error_mm":float(np.mean(audit))}
    summary["passed"]=bool(summary["language_target_accuracy"]>=.8 and summary["real_contact_rate"]>=.7 and summary["full_pick_place_success_rate"]>=.7 and summary["mean_deployed_world_model_distance_error_mm"]<=75)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps({key:value for key,value in summary.items() if key!="reports"},indent=2))


if __name__=="__main__": main()
