from __future__ import annotations
import torch
from datasets.dynamic_graph import batch_dynamic_adjacency
from .metrics import ForecastingTotals

@torch.no_grad()
def evaluate(model: torch.nn.Module,loader,adjacency: torch.Tensor,device: torch.device,mean: float=0.0,std: float=1.0,dynamic_graph: bool=False,dynamic_alpha: float=.5,dynamic_top_k: int=8)->ForecastingTotals:
    model.eval(); totals=ForecastingTotals()
    physical=adjacency.to(device)
    for x,y,valid in loader:
        x=x.to(device); graph=batch_dynamic_adjacency(physical,x,dynamic_alpha,dynamic_top_k) if dynamic_graph else physical
        prediction=model(x,graph).cpu()*std+mean; target=y*std+mean
        totals.update(prediction,target,valid)
    return totals
