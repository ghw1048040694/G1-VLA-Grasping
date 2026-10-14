#!/usr/bin/env python3
"""Strict MuJoCo gate for temporal action-chunk VLA and ensemble WM-MPC."""
from __future__ import annotations
import argparse, json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from torch.nn import functional as F
import evaluate_g1_language_pick_place as base
from g1_direct_joint_candidate_vla import encode_tasks
from g1_finger_contact_geometry import finger_contact_sites
from g1_temporal_chunk_vla import load_checkpoint as load_vla
from g1_temporal_chunk_world_model import load_checkpoint as load_world

class Policy:
 def __init__(self,vla,world,device,horizon,population,iterations,limit):
  self.device=torch.device(device if device=='cpu' or torch.cuda.is_available() else 'cpu');self.vla,self.vocab,self.stats=load_vla(vla,str(self.device));self.world,self.wstats=load_world(world,str(self.device));self.horizon=horizon;self.population=population;self.iterations=iterations;self.limit=limit;self.config=SimpleNamespace(chunk_size=1,max_action_dim=31);self.needs_temporal_history=True;self.reset()
 def reset(self): self.ctx=None;self.history=deque(maxlen=self.vla.history_frames);self.candidate_reports=[];self.last_target_probabilities=None;self.last_active_arm_probabilities=None;self.last_object_position=None;self.last_contact_site_position=None;self.last_tactile_contact=None
 def set_planner_context(self,ctx): self.ctx=ctx
 def _z(self,x,n): return (x-self.wstats[n+'_mean'])/self.wstats[n+'_std']
 def _raw(self,x,n): return x*self.wstats[n+'_std']+self.wstats[n+'_mean']
 def _sites(self): return torch.from_numpy(finger_contact_sites(self.ctx['model'],self.ctx['data'],self.ctx['finger_contact_geom_ids'])).float().to(self.device).reshape(1,-1)
 def _tactile(self):
  geoms=set(self.ctx['finger_contact_geom_ids'])|set(self.ctx['contact_pad_ids']);bodies=set(self.ctx['object_body_ids'].values());m=self.ctx['model'];d=self.ctx['data'];active=any((c.geom1 in geoms and int(m.geom_bodyid[c.geom2]) in bodies) or (c.geom2 in geoms and int(m.geom_bodyid[c.geom1]) in bodies) for c in d.contact[:d.ncon]);return torch.tensor([[float(active)]],device=self.device)
 def observe_history(self,batch):
  d=self.ctx['data'];c=self.ctx['controlled'];v=torch.tensor([[d.qvel[c[n]['qvel_id']] for n in self.ctx['upper_names']]],device=self.device,dtype=torch.float32);images=torch.stack([batch[k][0].to(self.device) for k in base.CAMERAS],0);self.history.append({'images':images,'state':batch['observation.state'].to(self.device),'velocity':v,'sites':self._sites(),'tactile':self._tactile(),'task':batch['task']})
 def _stack(self):
  rows=list(self.history);rows=[rows[0]]*(self.vla.history_frames-len(rows))+rows;return {k:torch.stack([x[k] for x in rows],1) if k!='images' else torch.stack([x[k] for x in rows],0).unsqueeze(0) for k in ('images','state','velocity','sites','tactile')}
 def _cost(self,obs,objects,actions,target,arm):
  count=len(actions);one=F.one_hot(torch.full((count,),target,device=self.device),3).float();costs=[];closest=[]
  for member in self.world.models:
   cur={'state':self._z(obs['state'],'state').expand(count,-1),'velocity':self._z(obs['velocity'],'velocity').expand(count,-1),'objects':self._z(objects,'objects').expand(count,-1),'sites':self._z(obs['sites'],'contact_sites').expand(count,-1)};ds=[]
   for step in range(self.horizon):
    out=member(cur['state'],cur['velocity'],cur['objects'],cur['sites'],obs['tactile'].expand(count,-1),self._z(actions[:,step],'action'),one);cur={'state':out['state'],'velocity':out['velocity'],'objects':out['objects'],'sites':out['contact_sites']};obj=self._raw(cur['objects'],'objects').reshape(count,3,3);sites=self._raw(cur['sites'],'contact_sites').reshape(count,2,-1,3)[:,arm];ds.append(torch.linalg.vector_norm(sites-obj[:,target,None],dim=-1).min(-1).values)
   dist=torch.stack(ds,1);motion=actions.square().mean((1,2)).sqrt();closest.append(dist.min(1).values)
   if float(obs['tactile'].item())<.5: costs.append(dist.min(1).values+.35*dist[:,-1]+.02*motion)
   else: costs.append(torch.linalg.vector_norm(obj[:,target]-torch.tensor(base.BIN_POSITION_M,device=self.device),dim=-1)+.02*motion)
  stack=torch.stack(costs);return stack.mean(0)+.60*stack.std(0),{'closest':torch.stack(closest).mean(0),'uncertainty':stack.std(0)}
 @torch.no_grad()
 def predict_action_chunk(self,batch,noise=None):
  obs=self._stack();ids,mask=encode_tasks(self.history[-1]['task'],self.vocab);out=self.vla(obs['images'],(obs['state']-self.stats['state_mean'])/self.stats['state_std'],(obs['velocity']-self.stats['velocity_mean'])/self.stats['velocity_std'],(obs['sites']-self.stats['contact_sites_mean'])/self.stats['contact_sites_std'],obs['tactile'],ids.to(self.device),mask.to(self.device));pose=out['object_mean']*self.stats['objects_std']+self.stats['objects_mean'];pose=torch.stack((pose[:,0::3].clamp(.25,.70),pose[:,1::3].clamp(-.45,.35),pose[:,2::3].clamp(.72,1.15)),-1).reshape(1,9);target=int(out['target_logits'].argmax(-1)[0]);arm=int(out['active_arm_logits'].argmax(-1)[0]);mean=out['action_mean'][0]*self.stats['action_delta_std']+self.stats['action_delta_mean'];std=(out['action_log_std'][0].exp()*self.stats['action_delta_std']).clamp(.002,.020);g=torch.Generator(device=self.device);g.manual_seed(int(self.ctx['base_seed']));reports=[]
  for i in range(self.iterations):
   cand=(mean[None]+std[None]*torch.randn((self.population,self.horizon,31),generator=g,device=self.device)).clamp(-self.limit,self.limit);cost,metrics=self._cost({k:obs[k][:,-1] if k!='images' else obs[k] for k in ('state','velocity','sites','tactile','images')},pose,cand,target,arm);elite=cand[cost.topk(max(4,self.population//8),largest=False).indices];mean=elite.mean(0);std=elite.std(0).clamp(.002,.020);reports.append({'best_cost':float(cost.min()),'best_closest_distance_m':float(metrics['closest'][cost.argmin()]),'best_ensemble_std_m':float(metrics['uncertainty'][cost.argmin()])})
  self.last_target_probabilities=out['target_logits'].softmax(-1)[0].cpu().tolist();self.last_active_arm_probabilities=out['active_arm_logits'].softmax(-1)[0].cpu().tolist();self.last_object_position=pose[0].reshape(3,3).cpu().tolist();self.last_contact_site_position=obs['sites'][0,-1].reshape(-1,3).cpu().tolist();self.last_tactile_contact=float(obs['tactile'][0,-1].item());self.candidate_reports.append({'target_index':target,'active_arm_index':arm,'selection_mode':'temporal_vla_action_chunk_plus_uncertainty_ensemble_world_model_mpc_no_predicted_contact_reward','planner':reports});return (obs['state'][0,-1]+mean[0]).reshape(1,1,31)

def main():
 p=argparse.ArgumentParser();p.add_argument('--asset',type=Path,required=True);p.add_argument('--vla-checkpoint',type=Path,required=True);p.add_argument('--world-model-checkpoint',type=Path,required=True);p.add_argument('--train-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--episodes',type=int,default=3);p.add_argument('--seed',type=int,default=20260732);p.add_argument('--control-fps',type=int,default=15);p.add_argument('--duration-s',type=float,default=14);p.add_argument('--layout-mode',choices=('slot_jitter','random'),default='random');p.add_argument('--position-jitter-m',type=float,default=.018);p.add_argument('--horizon',type=int,default=8);p.add_argument('--population',type=int,default=96);p.add_argument('--iterations',type=int,default=2);p.add_argument('--action-delta-limit-rad',type=float,default=.08);p.add_argument('--body-gain-scale',type=float,default=5);p.add_argument('--hand-gain-scale',type=float,default=6);p.add_argument('--observation-width',type=int,default=160);p.add_argument('--observation-height',type=int,default=120);p.add_argument('--video-width',type=int,default=320);p.add_argument('--video-height',type=int,default=240);p.add_argument('--disable-video',action='store_true');p.add_argument('--save-observation-videos',action='store_true');p.add_argument('--device',default='cuda');a=p.parse_args()
 if a.episodes<3 or a.episodes%3:
  raise ValueError('episodes divisible by three')
 a.output_dir.mkdir(parents=True,exist_ok=True);a.disable_assisted_grasp=True;a.replan_steps=1;meta=json.loads((a.train_root/'meta/info.json').read_text());policy=Policy(a.vla_checkpoint,a.world_model_checkpoint,a.device,a.horizon,a.population,a.iterations,a.action_delta_limit_rad);reports=[]
 for spec in base.evaluation_specs(a.episodes,a.seed,a.position_jitter_m,a.layout_mode):
  r=base.run_episode(policy,spec,a,meta['features']['action']['names']);r.update(predicted_target_probabilities=policy.last_target_probabilities,predicted_active_arm_probabilities=policy.last_active_arm_probabilities,camera_object_position_m=policy.last_object_position,measured_tactile_contact=policy.last_tactile_contact,world_model_candidates=policy.candidate_reports);reports.append(r);print(f'EVAL_EPISODE={len(reports)}/{a.episodes} target={r["target_object"]} grabbed={r["grabbed_object"]} passed={r["passed"]}',flush=True)
 contacts=[r['hand_object_contact_candidate_steps']>0 for r in reports];language=float(np.mean([r['world_model_candidates'][-1]['target_index']==base.OBJECT_NAMES.index(r['target_object']) for r in reports]));s={'experiment':'G1LANG-temporal-chunk-vla-ensemble-wm-mpc-strict-gate','vla_checkpoint':str(a.vla_checkpoint),'world_model_checkpoint':str(a.world_model_checkpoint),'episodes':len(reports),'input_contract':'raw RGB history x3 + language + state31 + measured qvel/finger geometry/tactile','candidate_selection_mode':'temporal VLA action chunks + uncertainty ensemble WM CEM MPC; no predicted-contact reward','privileged_positions_at_inference':False,'assisted_grasp_enabled':False,'equality_constraints':False,'magnet_enabled':False,'reports':reports,'language_target_accuracy':language,'real_contact_rate':float(np.mean(contacts)),'no_grasp_rate':float(np.mean([r['grabbed_object'] is None for r in reports])),'full_pick_place_success_rate':float(np.mean([r['task_success'] for r in reports]))};s['passed']=bool(all(contacts) and s['language_target_accuracy']>=.8 and s['full_pick_place_success_rate']==1.0);(a.output_dir/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps({k:v for k,v in s.items() if k!='reports'},indent=2))
if __name__=='__main__': main()
