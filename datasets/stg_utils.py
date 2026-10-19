from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass(frozen=True)
class STGData:
    values: np.ndarray
    adjacency: np.ndarray
    coordinates: np.ndarray | None = None
    name: str = "unknown"
    synthetic: bool = False
    def __post_init__(self) -> None:
        if self.values.ndim != 3: raise ValueError("values must have shape [time, node, feature]")
        n = self.values.shape[1]
        if self.adjacency.shape != (n, n): raise ValueError("adjacency shape must match nodes")
        if not np.isfinite(self.values).all(): raise ValueError("values contain NaN or infinity")

@dataclass(frozen=True)
class TemporalSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

def temporal_split(values: np.ndarray, train_ratio: float = .7, val_ratio: float = .1) -> TemporalSplits:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1: raise ValueError("invalid ratios")
    a, b = int(len(values)*train_ratio), int(len(values)*(train_ratio+val_ratio))
    return TemporalSplits(values[:a], values[a:b], values[b:])

class WindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, values: np.ndarray, history: int = 12, horizon: int = 12, target_valid: np.ndarray | None = None) -> None:
        self.values = torch.as_tensor(values, dtype=torch.float32); self.history = history; self.horizon = horizon
        expected = values.shape[:2]
        if target_valid is None:
            target_valid = np.ones(expected, dtype=bool)
        if target_valid.shape != expected:
            raise ValueError(f"target_valid must have shape {expected}")
        self.target_valid = torch.as_tensor(target_valid, dtype=torch.bool)
        self.length = len(values)-history-horizon+1
        if self.length <= 0: raise ValueError("series too short")
    def __len__(self) -> int: return self.length
    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e=i+self.history
        return self.values[i:e], self.values[e:e+self.horizon,:,0], self.target_valid[e:e+self.horizon]

def make_synthetic_stg(num_nodes: int = 20, timesteps: int = 200, seed: int = 42) -> STGData:
    rng=np.random.default_rng(seed); coords=rng.uniform(0,1,(num_nodes,2)).astype(np.float32)
    dist=np.linalg.norm(coords[:,None]-coords[None,:],axis=-1); adj=np.exp(-5*dist).astype(np.float32)
    adj[dist>np.quantile(dist,.25)]=0; np.fill_diagonal(adj,1)
    t=np.arange(timesteps,dtype=np.float32); signals=[]
    for node in range(num_nodes):
        p=2*np.pi*node/max(num_nodes,1)
        signals.append(np.sin(2*np.pi*t/24+p)+.4*np.sin(2*np.pi*t/72-p)+rng.normal(0,.08,timesteps))
    return STGData(np.stack(signals,axis=1)[...,None].astype(np.float32),adj,coords,"synthetic",True)
