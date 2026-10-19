from __future__ import annotations
from collections.abc import Mapping
import torch
from .temporal_spectral import dct,idct

GROUPS=("graph_low","graph_high","temporal_low","temporal_high","other")
def transform_update(update: Mapping[str,torch.Tensor])->dict[str,dict[str,torch.Tensor]]:
    out={g:{} for g in GROUPS}
    for name,tensor in update.items():
        if name=="graph_conv.theta":
            split=max(1,(len(tensor)+1)//2); out["graph_low"][name]=tensor[:split].clone(); out["graph_high"][name]=tensor[split:].clone()
        elif name.startswith("temporal_") and name.endswith("weight"):
            freq=dct(tensor,dim=2); split=max(1,(freq.shape[2]+1)//2); out["temporal_low"][name]=freq[:,:,:split].clone(); out["temporal_high"][name]=freq[:,:,split:].clone()
        else: out["other"][name]=tensor.clone()
    return out

def inverse_groups(groups: Mapping[str,Mapping[str,torch.Tensor]],template: Mapping[str,torch.Tensor])->dict[str,torch.Tensor]:
    out={}
    for name,tensor in template.items():
        if name=="graph_conv.theta": out[name]=torch.cat([groups["graph_low"][name],groups["graph_high"][name]],0)
        elif name.startswith("temporal_") and name.endswith("weight"):
            out[name]=idct(torch.cat([groups["temporal_low"][name],groups["temporal_high"][name]],2),dim=2)
        else: out[name]=groups["other"][name]
    return out

