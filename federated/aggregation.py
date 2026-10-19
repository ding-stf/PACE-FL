from __future__ import annotations
from collections.abc import Mapping,Sequence
import torch

Update=dict[str,torch.Tensor]
def model_update(local: Mapping[str,torch.Tensor],global_state: Mapping[str,torch.Tensor])->Update:
    return {k:local[k].detach().cpu()-global_state[k].detach().cpu() for k in global_state}
def apply_update(state: Mapping[str,torch.Tensor],update: Mapping[str,torch.Tensor])->Update:
    return {k:state[k].detach().cpu()+update[k].detach().cpu() for k in state}
def weighted_average_updates(updates: Sequence[Mapping[str,torch.Tensor]],weights: Sequence[float]|None=None)->Update:
    if not updates: raise ValueError("cannot aggregate empty updates")
    w=torch.as_tensor(weights if weights is not None else [1.0]*len(updates),dtype=torch.float64); w=w/w.sum(); out={}
    for key in updates[0]: out[key]=sum(float(a)*u[key] for a,u in zip(w,updates))
    return out

