from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from privacy.dp_fedavg import DPUpdate
from spectral.spectral_groups import GROUPS,inverse_groups,transform_update

ALLOWED_TOPK_RATIOS=(1.0,0.25,0.10)

@dataclass(frozen=True)
class SparseTensor:
    shape: tuple[int,...]; indices: torch.Tensor; values: torch.Tensor
@dataclass(frozen=True)
class SparsePayload:
    tensors: dict[str,SparseTensor]; template_shapes: dict[str,tuple[int,...]]; dp_applied: bool=True; domain: str="spectral"

def spectral_coefficients(tensors: dict[str,torch.Tensor])->tuple[dict[str,torch.Tensor],dict[str,tuple[int,...]]]:
    groups=transform_update(tensors); coefficients={f"{group}::{name}":tensor for group,items in groups.items() for name,tensor in items.items()}; return coefficients,{k:tuple(v.shape) for k,v in tensors.items()}

def _topk_coefficients(coefficients: dict[str,torch.Tensor],template_shapes: dict[str,tuple[int,...]],ratio: float)->SparsePayload:
    out={}
    for name,t in coefficients.items():
        flat=t.flatten(); k=max(1,int(math.ceil(flat.numel()*ratio))); idx=torch.topk(flat.abs(),k,sorted=False).indices; out[name]=SparseTensor(tuple(t.shape),idx.cpu(),flat[idx].cpu())
    return SparsePayload(out,template_shapes)

def spectral_topk(update: DPUpdate,ratio: float)->SparsePayload:
    if not update.dp_applied: raise ValueError("sparsification is permitted only after DP")
    if not any(abs(float(ratio)-allowed)<1e-12 for allowed in ALLOWED_TOPK_RATIOS):
        raise ValueError(f"ratio must be one of {ALLOWED_TOPK_RATIOS}")
    coefficients,template=spectral_coefficients(update.tensors); return _topk_coefficients(coefficients,template,ratio)

def dense_coefficients(payload: SparsePayload)->dict[str,torch.Tensor]:
    out={}
    for name,item in payload.tensors.items():
        flat=torch.zeros(int(torch.tensor(item.shape).prod()),dtype=item.values.dtype); flat[item.indices.long()]=item.values; out[name]=flat.reshape(item.shape)
    return out

def decode(payload: SparsePayload)->dict[str,torch.Tensor]:
    coefficients=dense_coefficients(payload); groups={g:{} for g in GROUPS}
    for key,tensor in coefficients.items():
        group,name=key.split("::",1); groups[group][name]=tensor
    template={name:torch.zeros(shape) for name,shape in payload.template_shapes.items()}; return inverse_groups(groups,template)
