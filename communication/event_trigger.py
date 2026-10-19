from __future__ import annotations
import torch
from .sparse_sync import SparsePayload

def payload_l2_norm(payload: SparsePayload)->float:
    if not payload.dp_applied: raise ValueError("event trigger must inspect DP output only")
    return float(torch.sqrt(sum(x.values.double().square().sum() for x in payload.tensors.values())))

def should_upload(payload: SparsePayload,tau: float)->bool:
    return payload_l2_norm(payload)>=tau
