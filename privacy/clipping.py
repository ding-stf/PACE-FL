from __future__ import annotations
from collections.abc import Mapping
import torch

def l2_norm(tensors: Mapping[str,torch.Tensor])->torch.Tensor:
    if not tensors: return torch.tensor(0.0)
    return torch.sqrt(sum(x.double().square().sum() for x in tensors.values())).float()

def clip_group(tensors: Mapping[str,torch.Tensor],clip_norm: float)->tuple[dict[str,torch.Tensor],float]:
    if clip_norm<=0: raise ValueError("clip_norm must be positive")
    norm=l2_norm(tensors); scale=min(1.0,clip_norm/(float(norm)+1e-12))
    return {k:v*scale for k,v in tensors.items()},float(norm)

