#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,math,random
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from g1_direct_joint_candidate_vla import encode_tasks
from g1_random_color_posterior import load_checkpoint as load_color,color_features
from g1_progress_temporal_vla import ProgressTemporalVLA,save_checkpoint
class Data(Dataset):
 def __init__(self,c,h,k,w):
  self.c=c;self.h=h;self.k=k;self.im=None;self.rows=[];valid=c['valid_frame'].bool().numpy()
  for e,m in enumerate(valid): self.rows.extend((e,f) for f in range(max(0,int(m.sum())-k+1)))
  self.weights=np.asarray([float(c['sample_weight'][e,f]) if w else 1 for e,f in self.rows]);self.ids,self.mask=encode_tasks([c['tasks'][int(i)] for i in c['task_index'].tolist()],c['vocab'])
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  e,f=self.rows[i];hist=np.maximum(np.arange(f-self.h+1,f+1),0)
  if self.im is None:self.im=np.load(self.c['image_path'],mmap_mode='r')
  image=torch.from_numpy(np.asarray(self.im[e,hist]).copy()).permute(0,1,4,2,3).float().div_(255)
  return {'images':image,'state':self.c['state'][e,hist].float(),'velocity':self.c['velocity'][e,hist].float(),'sites':self.c['contact_sites'][e,hist].float(),'tactile':self.c['contact'][e,hist].float(),'progress':torch.tensor(hist/(self.c['valid_frame'].shape[1]-1),dtype=torch.float32).unsqueeze(-1),'action':self.c['action_delta'][e,f:f+self.k].float(),'objects':self.c['objects'][e,f].float(),'target':self.c['target'][e].long(),'arm':self.c['active_arm'][e,f].long(),'ids':self.ids[e],'mask':self.mask[e]}
def stats(c):
 v=c['valid_frame'].bool();return {f'{n}_{s}':getattr(c[n][v].float(),s)(0).clamp_min(1e-3) if s=='std' else getattr(c[n][v].float(),s)(0) for n in ('state','velocity','action_delta','objects','contact_sites') for s in ('mean','std')}
def visual(batch,color,cs,st):
 b,t=batch['images'].shape[:2]
 head=batch['images'][:,:,0].reshape(b*t,3,*batch['images'].shape[-2:])
 with torch.no_grad():
  raw=color((color_features(head)-cs['feature_mean'])/cs['feature_std'])*cs['pose_std']+cs['pose_mean']
 return (raw.reshape(b,t,9)-st['objects_mean'])/st['objects_std']
def run(m,b,st,color,cs):
 vp=visual(b,color,cs,st);o=m(b['images'],(b['state']-st['state_mean'])/st['state_std'],(b['velocity']-st['velocity_mean'])/st['velocity_std'],(b['sites']-st['contact_sites_mean'])/st['contact_sites_std'],b['tactile'],vp,b['progress'],b['ids'],b['mask']);act=(b['action']-st['action_delta_mean'])/st['action_delta_std'];loss=(.5*((o['action_mean']-act)/o['action_log_std'].exp()).square()+o['action_log_std']).mean()+.45*F.mse_loss(o['object_mean'],(b['objects']-st['objects_mean'])/st['objects_std'])+.2*F.cross_entropy(o['target_logits'],b['target'])+.08*F.cross_entropy(o['active_arm_logits'],b['arm']);return loss,o
def main():
 p=argparse.ArgumentParser();p.add_argument('--cache-dir',type=Path,required=True);p.add_argument('--color-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=24);p.add_argument('--seed',type=int,default=20260741);p.add_argument('--device',default='cuda');a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);dev=torch.device(a.device if a.device=='cpu' or torch.cuda.is_available() else 'cpu');tr=torch.load(a.cache_dir/'train.pt',map_location='cpu',weights_only=False);va=torch.load(a.cache_dir/'validation.pt',map_location='cpu',weights_only=False);st={k:v.to(dev) for k,v in stats(tr).items()};color,cs=load_color(a.color_checkpoint,str(dev));[x.requires_grad_(False) for x in color.parameters()];ts,vs=Data(tr,4,8,True),Data(va,4,8,False);tl=DataLoader(ts,batch_size=a.batch_size,sampler=WeightedRandomSampler(ts.weights,len(ts),replacement=True));vl=DataLoader(vs,batch_size=a.batch_size);m=ProgressTemporalVLA(len(tr['vocab']),int(tr['contact_sites'].shape[-1])).to(dev);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=2e-4);a.output_dir.mkdir(parents=True,exist_ok=True);best=1e9;hist=[]
 for e in range(a.epochs):
  m.train();tot=n=0
  for r in tl:
   b={k:v.to(dev) for k,v in r.items()};loss,_=run(m,b,st,color,cs);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();tot+=float(loss)*len(b['target']);n+=len(b['target'])
  m.eval();sq=count=ok=0
  with torch.no_grad():
   for r in vl:
    b={k:v.to(dev) for k,v in r.items()};_,o=run(m,b,st,color,cs);pr=o['action_mean']*st['action_delta_std']+st['action_delta_mean'];sq+=float((pr-b['action']).square().sum());count+=pr.numel();ok+=int((o['target_logits'].argmax(-1)==b['target']).sum())
  z={'epoch':e+1,'train_loss':tot/max(n,1),'chunk_action_rmse_rad':math.sqrt(sq/count),'target_accuracy':ok/len(vs)};hist.append(z);print('PROGRESS_TEMPORAL_VLA='+json.dumps(z),flush=True);score=z['chunk_action_rmse_rad'];
  if score<best:best=score;save_checkpoint(a.output_dir/'best.pt',m,tr['vocab'],st)
 (a.output_dir/'summary.json').write_text(json.dumps({'experiment':'G1LANG-progress-memory-raw-multiview-action-chunk-vla','cache_dir':str(a.cache_dir),'color_checkpoint':str(a.color_checkpoint),'best_validation_chunk_action_rmse_rad':best,'input_contract':'raw RGB x3 + language + state31 + qvel31 + finger sites + tactile + causal progress','privileged_positions_at_inference':False,'history':hist},indent=2)+'\n')
if __name__=='__main__':main()
