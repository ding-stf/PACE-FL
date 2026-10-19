from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main()->None:
    p=argparse.ArgumentParser(description="Convert legally obtained traffic arrays to the normalized A3 format"); p.add_argument("--dataset",required=True,choices=["metr_la","pems_bay","pems04","pems08"]); p.add_argument("--values",type=Path,required=True); p.add_argument("--adjacency",type=Path,required=True); p.add_argument("--coordinates",type=Path); p.add_argument("--output_root",type=Path,default=Path("data")); a=p.parse_args(); out=a.output_root/a.dataset; out.mkdir(parents=True,exist_ok=True)
    values=np.load(a.values); adjacency=np.load(a.adjacency)
    if values.ndim not in {2,3}: raise ValueError("values must be [time,node] or [time,node,feature]")
    if adjacency.shape!=(values.shape[1],values.shape[1]): raise ValueError("adjacency does not match nodes")
    np.save(out/"values.npy",values.astype(np.float32)); np.save(out/"adjacency.npy",adjacency.astype(np.float32))
    if a.coordinates: np.save(out/"coordinates.npy",np.load(a.coordinates).astype(np.float32))
if __name__=="__main__": main()

