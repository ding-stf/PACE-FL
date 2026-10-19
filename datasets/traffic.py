from __future__ import annotations
from pathlib import Path
import numpy as np
from .stg_utils import STGData

SUPPORTED={"metr_la","pems_bay","pems04","pems08"}
def load_traffic_dataset(root: str|Path,name: str)->STGData:
    key=name.lower().replace("-","_")
    if key not in SUPPORTED: raise ValueError(f"unsupported dataset: {name}")
    directory=Path(root)/key; values_path=directory/"values.npy"; adjacency_path=directory/"adjacency.npy"
    if not values_path.exists() or not adjacency_path.exists(): raise FileNotFoundError(f"Missing {values_path} or {adjacency_path}. Run scripts/prepare_data.py")
    values=np.load(values_path).astype(np.float32)
    if values.ndim==2: values=values[...,None]
    cp=directory/"coordinates.npy"; coords=np.load(cp).astype(np.float32) if cp.exists() else None
    return STGData(values,np.load(adjacency_path).astype(np.float32),coords,key,False)

