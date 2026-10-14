#!/usr/bin/env python3
"""Fit a learned RGB colour-geometry pose posterior on random evaluator layouts."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from g1_random_color_posterior import RandomColorPosePosterior,color_features,save_checkpoint

def main() -> None:
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int,default=400);p.add_argument('--seed',type=int,default=20260741);p.add_argument('--device',default='cuda');a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
 features=[];poses=[]
 for path in sorted(a.data_root.glob('episode_*')):
  cap=cv2.VideoCapture(str(path/'head_camera.mp4'));ok,image=cap.read();cap.release()
  if not ok: raise RuntimeError(f'Cannot read {path}/head_camera.mp4')
  image=torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div_(255).unsqueeze(0);features.append(color_features(image)[0])
  with np.load(path/'trajectory.npz') as v: poses.append(torch.from_numpy(v['object_position_m'][0].astype(np.float32).reshape(-1)))
 f=torch.stack(features);y=torch.stack(poses);split=max(30,round(.2*len(f)));fm,fs=f[:-split].mean(0),f[:-split].std(0).clamp_min(.01);ym,ys=y[:-split].mean(0),y[:-split].std(0).clamp_min(.001);device=torch.device(a.device if a.device=='cpu' or torch.cuda.is_available() else 'cpu');f,y,fm,fs,ym,ys=[x.to(device) for x in (f,y,fm,fs,ym,ys)];m=RandomColorPosePosterior().to(device);opt=torch.optim.AdamW(m.parameters(),lr=2e-3,weight_decay=1e-4);a.output_dir.mkdir(parents=True,exist_ok=True);best=float('inf');history=[]
 for epoch in range(a.epochs):
  m.train();out=m((f[:-split]-fm)/fs);loss=F.mse_loss(out,(y[:-split]-ym)/ys);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
  if (epoch+1)%10==0 or epoch==a.epochs-1:
   m.eval()
   with torch.no_grad(): err=torch.linalg.vector_norm((m((f[-split:]-fm)/fs)*ys+ym-y[-split:]).reshape(-1,3),dim=-1)
   r={'epoch':epoch+1,'train_loss':float(loss),'mean_mm':float(err.mean()*1e3),'p95_mm':float(torch.quantile(err,.95)*1e3),'max_mm':float(err.max()*1e3)};history.append(r);print('RANDOM_COLOR_POSTERIOR='+json.dumps(r),flush=True)
   if r['mean_mm']<best: best=r['mean_mm'];save_checkpoint(a.output_dir/'best.pt',m,fm,fs,ym,ys)
 summary={'experiment':'G1LANG-random-head-RGB-colour-pose-posterior','data_root':str(a.data_root),'train_episodes':len(f)-split,'validation_episodes':split,'input_contract':'raw head RGB; learned colour geometry posterior','privileged_positions_at_inference':False,'best_validation_mean_mm':best,'history':history};(a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__': main()
