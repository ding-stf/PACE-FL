from __future__ import annotations
import torch
from torch import nn

def normalized_laplacian(adjacency: torch.Tensor)->torch.Tensor:
    adjacency=.5*(adjacency+adjacency.transpose(-1,-2)); degree=adjacency.sum(-1).clamp_min(1e-8); inv=degree.rsqrt(); eye=torch.eye(adjacency.shape[-1],device=adjacency.device,dtype=adjacency.dtype)
    return eye-inv.unsqueeze(-1)*adjacency*inv.unsqueeze(-2)

class ChebGraphConv(nn.Module):
    def __init__(self,in_channels: int,out_channels: int,order: int=3)->None:
        super().__init__(); self.order=order; self.theta=nn.Parameter(torch.empty(order,in_channels,out_channels)); self.bias=nn.Parameter(torch.zeros(out_channels)); nn.init.xavier_uniform_(self.theta)
    def forward(self,x: torch.Tensor,adjacency: torch.Tensor)->torch.Tensor:
        lap=normalized_laplacian(adjacency); terms=[x]; apply=(lambda value: torch.einsum("nm,btmc->btnc",lap,value)) if lap.ndim==2 else (lambda value: torch.einsum("bnm,btmc->btnc",lap,value))
        if self.order>1: terms.append(apply(x))
        for _ in range(2,self.order): terms.append(2*apply(terms[-1])-terms[-2])
        return torch.einsum("kbtnc,kco->btno",torch.stack(terms),self.theta)+self.bias

class STGCN(nn.Module):
    """Lightweight STGCN exposing Chebyshev orders and temporal kernels."""
    def __init__(self,in_channels: int=1,hidden_channels: int=16,horizon: int=12,cheb_order: int=3)->None:
        super().__init__(); self.temporal_in=nn.Conv2d(in_channels,hidden_channels,(3,1),padding=(1,0)); self.graph_conv=ChebGraphConv(hidden_channels,hidden_channels,cheb_order); self.temporal_out=nn.Conv2d(hidden_channels,hidden_channels,(3,1),padding=(1,0)); self.readout=nn.Linear(hidden_channels,horizon)
    @property
    def spectral_parameter_names(self)->dict[str,tuple[str,...]]: return {"graph":("graph_conv.theta",),"temporal":("temporal_in.weight","temporal_out.weight")}
    def forward(self,x: torch.Tensor,adjacency: torch.Tensor)->torch.Tensor:
        h=torch.relu(self.temporal_in(x.permute(0,3,1,2))).permute(0,2,3,1); h=torch.relu(self.graph_conv(h,adjacency)); h=torch.relu(self.temporal_out(h.permute(0,3,1,2))).permute(0,2,3,1)
        return self.readout(h[:,-1]).permute(0,2,1)
