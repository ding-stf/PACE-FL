from __future__ import annotations
from collections.abc import Mapping
import torch
from spectral.spectral_groups import GROUPS,inverse_groups,transform_update
from .clipping import clip_group
from .dp_fedavg import DPUpdate

class SpectralDPMechanism:
    def __init__(self,clip_norms: Mapping[str,float],noise_multiplier: float,generator: torch.Generator|None=None)->None:
        missing=set(GROUPS)-set(clip_norms)
        if missing: raise ValueError(f"missing clip norms: {sorted(missing)}")
        if any(value<=0 for value in clip_norms.values()): raise ValueError("all clip norms must be positive")
        if noise_multiplier<=0: raise ValueError("noise_multiplier must be positive")
        self.clip_norms=dict(clip_norms); self.noise_multiplier=noise_multiplier; self.generator=generator
    @property
    def composed_group_count(self)->int: return len(GROUPS)
    def privatize(self,update: Mapping[str,torch.Tensor])->DPUpdate:
        groups=transform_update(update); private={}; diagnostics={}
        for group,tensors in groups.items():
            clipped,before=clip_group(tensors,self.clip_norms[group]); diagnostics[f"{group}_pre_clip_norm"]=before
            c=self.clip_norms[group]; private[group]={k:v+torch.randn(v.shape,dtype=v.dtype,device=v.device,generator=self.generator)*self.noise_multiplier*c for k,v in clipped.items()}
        return DPUpdate(inverse_groups(private,update),True,diagnostics)
