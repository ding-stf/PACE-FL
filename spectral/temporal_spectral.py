from __future__ import annotations
import torch

def dct(x: torch.Tensor,dim: int=-1)->torch.Tensor:
    """Orthonormal DCT-II in native PyTorch."""
    x=x.movedim(dim,-1); n=x.shape[-1]; j=torch.arange(n,device=x.device,dtype=x.dtype); k=torch.arange(n,device=x.device,dtype=x.dtype)[:,None]
    basis=torch.cos(torch.pi*(j+.5)*k/n); basis[0]*=torch.sqrt(torch.tensor(1/n,dtype=x.dtype,device=x.device)); basis[1:]*=torch.sqrt(torch.tensor(2/n,dtype=x.dtype,device=x.device)); return torch.einsum("...j,kj->...k",x,basis).movedim(-1,dim)

def idct(x: torch.Tensor,dim: int=-1)->torch.Tensor:
    """Inverse of the orthonormal DCT-II, implemented by an explicit orthogonal basis."""
    x=x.movedim(dim,-1); n=x.shape[-1]; j=torch.arange(n,device=x.device,dtype=x.dtype); k=torch.arange(n,device=x.device,dtype=x.dtype)[:,None]
    basis=torch.cos(torch.pi*(j+.5)*k/n); basis[0]*=torch.sqrt(torch.tensor(1/n,dtype=x.dtype,device=x.device)); basis[1:]*=torch.sqrt(torch.tensor(2/n,dtype=x.dtype,device=x.device)); return torch.einsum("...k,kj->...j",x,basis).movedim(-1,dim)
