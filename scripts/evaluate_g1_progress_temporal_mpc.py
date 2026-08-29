#!/usr/bin/env python3
"""Strict gate: progress-memory raw RGB VLA plus ensemble learned WM-MPC."""
from __future__ import annotations
import argparse,json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
from torch.nn import functional as F
import evaluate_g1_language_pick_place as base
from g1_direct_joint_candidate_vla import encode_tasks
from g1_finger_contact_geometry import finger_contact_sites
from g1_progress_temporal_vla import load_checkpoint as load_vla
from g1_random_color_posterior import load_checkpoint as load_color,color_features
from g1_temporal_chunk_world_model import load_checkpoint as load_world

class Policy:
 def __init__(self,vla,color,world,device,horizon,population,iterations,limit):
  self.device=torch.device(device if device=='cpu' or torch.cuda.is_available() else 'cpu');self.vla,self.vocab,self.stats=load_vla(vla,str(self.device));self.color,self.cstats=load_color(color,str(self.device));self.world,self.wstats=load_world(world,str(self.device));self.horizon=horizon;self.population=population;self.iterations=iterations;self.limit=limit;self.config=SimpleNamespace(chunk_size=1,max_action_dim=31);self.needs_temporal_history=True;self.reset()
 def reset(self): self.ctx=None;self.history=deque(maxlen=self.vla.history_frames);self.latched_pose=None;self.candidate_reports=[];self.last_target_probabilities=None;self.last_active_arm_probabilities=None;self.last_object_position=None;self.last_tactile_contact=None
 def set_planner_context(self,c): self.ctx=c
 def _z(self,x,n): return (x-self.wstats[n+'_mean'])/self.wstats[n+'_std']
 def _raw(self,x,n): return x*self.wstats[n+'_std']+self.wstats[n+'_mean']
 def _sites(self): return torch.from_numpy(finger_contact_sites(self.ctx['model'],self.ctx['data'],self.ctx['finger_contact_geom_ids'])).float().to(self.device).reshape(1,-1)
 def _tactile(self):
  geoms=set(self.ctx['finger_contact_geom_ids'])|set(self.ctx['contact_pad_ids']);bodies=set(self.ctx['object_body_ids'].values());m,d=self.ctx['model'],self.ctx['data'];active=any((c.geom1 in geoms and int(m.geom_bodyid[c.geom2]) in bodies) or (c.geom2 in geoms and int(m.geom_bodyid[c.geom1]) in bodies) for c in d.contact[:d.ncon]);return torch.tensor([[float(active)]],device=self.device)
 def observe_history(self,b):
  d,c=self.ctx['data'],self.ctx['controlled'];images=torch.stack([b[k][0].to(self.device) for k in base.CAMERAS],0)
  if self.latched_pose is None:
   with torch.no_grad(): self.latched_pose=self.color((color_features(images[0:1])-self.cstats['feature_mean'])/self.cstats['feature_std'])*self.cstats['pose_std']+self.cstats['pose_mean']
  vel=torch.tensor([[d.qvel[c[n]['qvel_id']] for n in self.ctx['upper_names']]],device=self.device,dtype=torch.float32);self.history.append({'images':images,'state':b['observation.state'].to(self.device),'velocity':vel,'sites':self._sites(),'tactile':self._tactile(),'progress':torch.tensor([[min(self.ctx['frame']/169.,1.)]],device=self.device),'task':b['task']})
 def _stack(self):
  rows=list(self.history);rows=[rows[0]]*(self.vla.history_frames-len(rows))+rows;value={k:torch.stack([x[k] for x in rows],1) if k!='images' else torch.stack([x[k] for x in rows],0).unsqueeze(0) for k in ('images','state','velocity','sites','tactile','progress')};value['pose']=self.latched_pose[:,None].expand(-1,self.vla.history_frames,-1);return value
 def _cost(self,obs,objects,actions,target,arm):
  count=len(actions);one=F.one_hot(torch.full((count,),target,device=self.device),3).float();costs=[];near=[]
  for member in self.world.models:
   cur={'state':self._z(obs['state'],'state').expand(count,-1),'velocity':self._z(obs['velocity'],'velocity').expand(count,-1),'objects':self._z(objects,'objects').expand(count,-1),'sites':self._z(obs['sites'],'contact_sites').expand(count,-1)};ds=[]
   for step in range(self.horizon):
    out=member(cur['state'],cur['velocity'],cur['objects'],cur['sites'],obs['tactile'].expand(count,-1),self._z(actions[:,step],'action'),one);cur={'state':out['state'],'velocity':out['velocity'],'objects':out['objects'],'sites':out['contact_sites']};obj=self._raw(cur['objects'],'objects').reshape(count,3,3);site=self._raw(cur['sites'],'contact_sites').reshape(count,2,-1,3)[:,arm];ds.append(torch.linalg.vector_norm(site-obj[:,target,None],dim=-1).min(-1).values)
   dist=torch.stack(ds,1);motion=actions.square().mean((1,2)).sqrt();near.append(dist.min(1).values);costs.append(dist.min(1).values+.35*dist[:,-1]+.02*motion if float(obs['tactile'].item())<.5 else torch.linalg.vector_norm(obj[:,target]-torch.tensor(base.BIN_POSITION_M,device=self.device),dim=-1)+.02*motion)
  stack=torch.stack(costs);return stack.mean(0)+.6*stack.std(0),{'nearest':torch.stack(near).mean(0),'uncertainty':stack.std(0)}
 @torch.no_grad()
 def predict_action_chunk(self,b,noise=None):
  o=self._stack();ids,mask=encode_tasks(self.history[-1]['task'],self.vocab);vp=(o['pose']-self.stats['objects_mean'])/self.stats['objects_std'];out=self.vla(o['images'],(o['state']-self.stats['state_mean'])/self.stats['state_std'],(o['velocity']-self.stats['velocity_mean'])/self.stats['velocity_std'],(o['sites']-self.stats['contact_sites_mean'])/self.stats['contact_sites_std'],o['tactile'],vp,o['progress'],ids.to(self.device),mask.to(self.device));target=int(out['target_logits'].argmax(-1)[0]);arm=int(out['active_arm_logits'].argmax(-1)[0]);mean=out['action_mean'][0]*self.stats['action_delta_std']+self.stats['action_delta_mean'];std=(out['action_log_std'][0].exp()*self.stats['action_delta_std']).clamp(.002,.02);g=torch.Generator(device=self.device);g.manual_seed(int(self.ctx['base_seed']));reports=[]
  for _ in range(self.iterations):
   cand=(mean[None]+std[None]*torch.randn((self.population,self.horizon,31),generator=g,device=self.device)).clamp(-self.limit,self.limit);cost,metric=self._cost({k:o[k][:,-1] for k in ('state','velocity','sites','tactile')},self.latched_pose,cand,target,arm);elite=cand[cost.topk(max(4,self.population//8),largest=False).indices];mean=elite.mean(0);std=elite.std(0).clamp(.002,.02);reports.append({'best_cost':float(cost.min()),'best_closest_distance_m':float(metric['nearest'][cost.argmin()]),'best_ensemble_std_m':float(metric['uncertainty'][cost.argmin()] )})
  self.last_target_probabilities=out['target_logits'].softmax(-1)[0].cpu().tolist();self.last_active_arm_probabilities=out['active_arm_logits'].softmax(-1)[0].cpu().tolist();self.last_object_position=self.latched_pose[0].reshape(3,3).cpu().tolist();self.last_tactile_contact=float(o['tactile'][0,-1]);self.candidate_reports.append({'target_index':target,'active_arm_index':arm,'selection_mode':'raw_rgb_progress_memory_vla_plus_random_rgb_posterior_plus_ensemble_wm_mpc','planner':reports});return (o['state'][0,-1]+mean[0]).reshape(1,1,31)
def main():
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--vla-checkpoint',type=Path,required=True);p.add_argument('--color-checkpoint',type=Path,required=True);p.add_argument('--world-model-checkpoint',type=Path,required=True);p.add_argument('--train-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--episodes',type=int,default=3);p.add_argument('--seed',type=int,default=20260742);p.add_argument('--control-fps',type=int,default=15);p.add_argument('--duration-s',type=float,default=14);p.add_argument('--layout-mode',choices=('slot_jitter','random'),default='random');p.add_argument('--position-jitter-m',type=float,default=.018);p.add_argument('--horizon',type=int,default=8);p.add_argument('--population',type=int,default=96);p.add_argument('--iterations',type=int,default=2);p.add_argument('--action-delta-limit-rad',type=float,default=.08);p.add_argument('--body-gain-scale',type=float,default=5);p.add_argument('--hand-gain-scale',type=float,default=6);p.add_argument('--observation-width',type=int,default=160);p.add_argument('--observation-height',type=int,default=120);p.add_argument('--video-width',type=int,default=320);p.add_argument('--video-height',type=int,default=240);p.add_argument('--disable-video',action='store_true');p.add_argument('--save-observation-videos',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
 if a.episodes<3 or a.episodes%3: raise ValueError('episodes must be divisible by three')
 a.output_dir.mkdir(parents=True,exist_ok=True);a.disable_assisted_grasp=True;a.replan_steps=1;names=json.loads((a.train_root/'meta/info.json').read_text())['features']['action']['names'];policy=Policy(a.vla_checkpoint,a.color_checkpoint,a.world_model_checkpoint,a.device,a.horizon,a.population,a.iterations,a.action_delta_limit_rad);reports=[]
 for spec in base.evaluation_specs(a.episodes,a.seed,a.position_jitter_m,a.layout_mode):
  r=base.run_episode(policy,spec,a,names);r.update(predicted_target_probabilities=policy.last_target_probabilities,predicted_active_arm_probabilities=policy.last_active_arm_probabilities,camera_object_position_m=policy.last_object_position,measured_tactile_contact=policy.last_tactile_contact,world_model_candidates=policy.candidate_reports);reports.append(r)
 contacts=[r['hand_object_contact_candidate_steps']>0 for r in reports];acc=float(np.mean([r['world_model_candidates'][-1]['target_index']==base.OBJECT_NAMES.index(r['target_object']) for r in reports]));s={'experiment':'G1LANG-progress-memory-raw-rgb-vla-ensemble-wm-mpc-strict-gate','episodes':len(reports),'input_contract':'raw RGB x3 + language + state31 + qvel31 + finger sites + tactile + causal execution progress','privileged_positions_at_inference':False,'assisted_grasp_enabled':False,'equality_constraints':False,'magnet_enabled':False,'reports':reports,'language_target_accuracy':acc,'real_contact_rate':float(np.mean(contacts)),'no_grasp_rate':float(np.mean([r['grabbed_object'] is None for r in reports])),'full_pick_place_success_rate':float(np.mean([r['task_success'] for r in reports]) )};s['passed']=bool(all(contacts) and acc>=.8 and s['full_pick_place_success_rate']==1);(a.output_dir/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({k:v for k,v in s.items() if k!='reports'},indent=2))
if __name__=='__main__':main()
