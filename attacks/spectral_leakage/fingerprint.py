from __future__ import annotations
from collections.abc import Sequence
import torch
from spectral.graph_spectral import graph_order_energy
from spectral.temporal_spectral import dct
from .collector import UpdateRecord

def _temporal_bands(update: dict[str,torch.Tensor])->torch.Tensor:
    bands=[]
    for name,tensor in update.items():
        if name.startswith("temporal_") and name.endswith("weight"):
            frequency_energy=dct(tensor,dim=2).square().sum(dim=(0,1,3)); chunks=torch.tensor_split(frequency_energy,3,dim=0); bands.append(torch.stack([c.sum() for c in chunks]))
    return torch.stack(bands).sum(0) if bands else torch.zeros(3)

def spectral_fingerprint(records: Sequence[UpdateRecord],round_coefficients: int=4)->torch.Tensor:
    if not records: raise ValueError("records cannot be empty")
    graph=torch.stack([graph_order_energy(r.layers) for r in records]); temporal=torch.stack([_temporal_bands(r.layers) for r in records]); eps=1e-12
    graph=graph/(graph.sum(1,keepdim=True)+eps); temporal=temporal/(temporal.sum(1,keepdim=True)+eps)
    summary=torch.cat([graph.mean(0),graph.std(0,unbiased=False),temporal.mean(0),temporal.std(0,unbiased=False)])
    cross=dct(torch.cat([graph,temporal],1),dim=0)[:round_coefficients].flatten()
    return torch.cat([summary,cross]).float()
