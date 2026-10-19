from __future__ import annotations
from collections.abc import Mapping
import torch
def graph_order_energy(update: Mapping[str,torch.Tensor])->torch.Tensor:
    theta=update["graph_conv.theta"]; return theta.flatten(1).norm(dim=1)

