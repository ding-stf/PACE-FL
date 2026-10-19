from __future__ import annotations
import numpy as np
import torch

def pearson_dynamic_adjacency(window: np.ndarray,top_k: int=8)->np.ndarray:
    if window.ndim==3: window=window[...,0]
    corr=np.maximum(np.nan_to_num(np.corrcoef(window.T)),0).astype(np.float32); n=len(corr); k=min(max(top_k,0),max(n-1,0)); mask=np.zeros_like(corr,bool)
    if k: mask[np.arange(n)[:,None],np.argpartition(corr,-(k+1),axis=1)[:,-(k+1):]]=True
    out=np.maximum(np.where(mask,corr,0),np.where(mask,corr,0).T); np.fill_diagonal(out,1); return out

def blend_adjacency(physical: np.ndarray,window: np.ndarray,alpha: float=.5,top_k: int=8)->np.ndarray:
    if not 0<=alpha<=1: raise ValueError("alpha must be in [0,1]")
    return (alpha*physical+(1-alpha)*pearson_dynamic_adjacency(window,top_k)).astype(np.float32)

def batch_dynamic_adjacency(physical: torch.Tensor,history: torch.Tensor,alpha: float=.5,top_k: int=8)->torch.Tensor:
    z=history[...,0].permute(0,2,1); z=z-z.mean(2,keepdim=True); z=z/(z.norm(dim=2,keepdim=True)+1e-8)
    corr=torch.clamp(z@z.transpose(-1,-2),min=0); n=corr.shape[-1]; threshold=torch.topk(corr,k=min(max(top_k+1,1),n),dim=2).values[:,:,-1:]
    dynamic=torch.where(corr>=threshold,corr,torch.zeros_like(corr)); dynamic=torch.maximum(dynamic,dynamic.transpose(-1,-2)); eye=torch.eye(n,device=history.device,dtype=history.dtype); dynamic=dynamic*(1-eye)+eye
    return alpha*physical.unsqueeze(0)+(1-alpha)*dynamic
