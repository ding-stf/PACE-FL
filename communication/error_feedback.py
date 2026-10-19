from __future__ import annotations
import math
import torch
from privacy.dp_fedavg import DPUpdate
from .sparse_sync import SparsePayload,_topk_coefficients,dense_coefficients,spectral_coefficients

class ErrorFeedback:
    def __init__(self,max_residual_norm: float|None=None)->None:
        if max_residual_norm is not None and max_residual_norm<=0:
            raise ValueError("max_residual_norm must be positive")
        self.max_residual_norm=max_residual_norm; self.residual: dict[str,torch.Tensor]={}
    def _clip_residual(self,residual: dict[str,torch.Tensor])->dict[str,torch.Tensor]:
        if self.max_residual_norm is None: return residual
        norm=math.sqrt(sum(float(value.double().square().sum()) for value in residual.values()))
        if norm<=self.max_residual_norm: return residual
        scale=self.max_residual_norm/(norm+1e-12)
        return {key:value*scale for key,value in residual.items()}
    def compress(self,update: DPUpdate,ratio: float)->SparsePayload:
        if not update.dp_applied: raise ValueError("error feedback is permitted only after DP")
        from .sparse_sync import ALLOWED_TOPK_RATIOS
        if not any(abs(float(ratio)-allowed)<1e-12 for allowed in ALLOWED_TOPK_RATIOS):
            raise ValueError(f"ratio must be one of {ALLOWED_TOPK_RATIOS}")
        coefficients,template=spectral_coefficients(update.tensors); merged={k:v+self.residual.get(k,torch.zeros_like(v)) for k,v in coefficients.items()}; payload=_topk_coefficients(merged,template,ratio); sparse=dense_coefficients(payload); self.residual=self._clip_residual({k:merged[k]-sparse[k] for k in merged}); return payload
