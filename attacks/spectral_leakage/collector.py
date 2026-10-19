from __future__ import annotations
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class UpdateRecord:
    round: int; client_id: int; layers: dict[str,torch.Tensor]; update_norm: float

class UpdateCollector:
    def __init__(self,save_full_updates: bool=False,output_dir: str|Path|None=None)->None:
        self.save_full_updates=save_full_updates; self.output_dir=Path(output_dir) if output_dir else None; self.records: list[UpdateRecord]=[]
    def collect(self,round_index: int,client_id: int,update: Mapping[str,torch.Tensor])->UpdateRecord:
        selected={k:v.detach().cpu().clone() for k,v in update.items() if self.save_full_updates or k=="graph_conv.theta" or (k.startswith("temporal_") and k.endswith("weight"))}
        norm=float(torch.sqrt(sum(v.double().square().sum() for v in update.values()))); rec=UpdateRecord(round_index,client_id,selected,norm); self.records.append(rec)
        if self.output_dir:
            self.output_dir.mkdir(parents=True,exist_ok=True); torch.save(rec,self.output_dir/f"round_{round_index:04d}_client_{client_id:03d}.pt")
        return rec
    def by_client(self)->dict[int,list[UpdateRecord]]:
        out=defaultdict(list)
        for r in self.records: out[r.client_id].append(r)
        return {k:sorted(v,key=lambda x:x.round) for k,v in out.items()}

