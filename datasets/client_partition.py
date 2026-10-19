from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering

@dataclass(frozen=True)
class PartitionResult:
    node_ids: tuple[np.ndarray,...]
    strategy: str
    @property
    def statistics(self) -> dict[str,float]:
        x=np.asarray([len(g) for g in self.node_ids],float)
        return {"mean":float(x.mean()),"std":float(x.std()),"min":float(x.min()),"max":float(x.max())}

def partition_nodes(adjacency: np.ndarray,num_clients: int,coordinates: np.ndarray|None=None,seed: int=42)->PartitionResult:
    n=adjacency.shape[0]
    if not 1<=num_clients<=n: raise ValueError("invalid num_clients")
    if coordinates is not None and len(coordinates)==n:
        labels=KMeans(n_clusters=num_clients,random_state=seed,n_init=10).fit_predict(coordinates); strategy="coordinate_kmeans"
    elif np.count_nonzero(adjacency)>n:
        try: labels=SpectralClustering(n_clusters=num_clients,affinity="precomputed",random_state=seed).fit_predict(adjacency); strategy="graph_community"
        except Exception: labels=None
    else: labels=None
    if labels is None:
        groups=tuple(np.asarray(x,dtype=int) for x in np.array_split(np.arange(n),num_clients)); return PartitionResult(groups,"contiguous_fallback")
    groups=tuple(np.flatnonzero(labels==i) for i in range(num_clients))
    if any(len(g)==0 for g in groups): return PartitionResult(tuple(np.asarray(x,dtype=int) for x in np.array_split(np.arange(n),num_clients)),"contiguous_fallback")
    return PartitionResult(groups,strategy)

