from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
import torch
from .clipping import clip_group

@dataclass(frozen=True)
class DPUpdate:
    tensors: dict[str,torch.Tensor]
    dp_applied: bool
    diagnostics: dict[str,float]

class DPFedAvgMechanism:
    # The formal local-DP baseline must randomize before the update crosses the
    # client/server API, matching the PJSDP observation boundary.
    client_side_release=True
    def __init__(self,clip_norm: float,noise_multiplier: float,generator: torch.Generator|None=None)->None:
        if clip_norm<=0: raise ValueError("clip_norm must be positive")
        if noise_multiplier<=0: raise ValueError("noise_multiplier must be positive")
        self.clip_norm=clip_norm; self.noise_multiplier=noise_multiplier; self.generator=generator
    def privatize(self,update: Mapping[str,torch.Tensor])->DPUpdate:
        clipped,before=clip_group(update,self.clip_norm); noisy={k:v+torch.randn(v.shape,dtype=v.dtype,device=v.device,generator=self.generator)*self.noise_multiplier*self.clip_norm for k,v in clipped.items()}
        return DPUpdate(noisy,True,{"pre_clip_norm":before,"clip_norm":self.clip_norm})
