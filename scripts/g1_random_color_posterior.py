#!/usr/bin/env python3
"""Learned object-pose posterior from causal raw head-camera RGB colour geometry."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn

def color_features(images: torch.Tensor) -> torch.Tensor:
    images = images[:, :3].clamp(0, 1); h, w = images.shape[-2:]
    red = (images[:, 0] > .47) & (images[:, 0] > images[:, 1] * 1.25) & (images[:, 0] > images[:, 2] * 1.25)
    yellow = (images[:, 0] > .47) & (images[:, 1] > .35) & (images[:, 2] < .47)
    green = (images[:, 1] > .31) & (images[:, 1] > images[:, 0] * 1.25) & (images[:, 1] > images[:, 2] * 1.15)
    x = torch.arange(w, device=images.device, dtype=images.dtype).view(1, 1, w); y = torch.arange(h, device=images.device, dtype=images.dtype).view(1, h, 1); values=[]
    for mask in (red, yellow, green):
        count=mask.flatten(1).sum(1).float().clamp_min(1); values.extend(((mask.float()*x).sum((1,2))/count/w,(mask.float()*y).sum((1,2))/count/h,count/(h*w)))
    return torch.stack(values, 1)

class RandomColorPosePosterior(nn.Module):
    def __init__(self) -> None:
        super().__init__(); self.network=nn.Sequential(nn.Linear(9,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,9))
    def forward(self, features: torch.Tensor) -> torch.Tensor: return self.network(features)

def save_checkpoint(path: Path, model: RandomColorPosePosterior, feature_mean, feature_std, pose_mean, pose_std) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); torch.save({'model':model.state_dict(),'feature_mean':feature_mean.cpu(),'feature_std':feature_std.cpu(),'pose_mean':pose_mean.cpu(),'pose_std':pose_std.cpu(),'input_contract':'causal raw head RGB colour centroid/area features','privileged_positions_at_inference':False},path)

def load_checkpoint(path: Path, device: str):
    v=torch.load(path,map_location=device,weights_only=False);m=RandomColorPosePosterior().to(device);m.load_state_dict(v['model']);m.eval();return m,{k:v[k].to(device) for k in ('feature_mean','feature_std','pose_mean','pose_std')}
