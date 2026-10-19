from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Mapping
import math
import time
import torch
from torch.utils.data import DataLoader
from datasets.dynamic_graph import batch_dynamic_adjacency
from .metrics import masked_l1_loss
from .aggregation import model_update

@dataclass
class ClientTrainResult:
    update: dict[str,torch.Tensor]; samples: int; client_train_loss: float; loss_samples: int=0; dp_applied: bool=False; privacy_ms: float=0.0
    @property
    def loss(self)->float:
        """Backward-compatible alias for callers using the old field name."""
        return self.client_train_loss

class STGClient:
    def __init__(self,client_id: int,model: torch.nn.Module,train_loader: DataLoader,adjacency: torch.Tensor,device: torch.device,lr: float=.01,dynamic_graph: bool=False,dynamic_alpha: float=.5,dynamic_top_k: int=8,mixed_precision: bool=False)->None:
        self.client_id=client_id; self.model=deepcopy(model).to(device); self.train_loader=train_loader; self.adjacency=adjacency.to(device); self.device=device; self.lr=lr; self.dynamic_graph=dynamic_graph; self.dynamic_alpha=dynamic_alpha; self.dynamic_top_k=dynamic_top_k; self.mixed_precision=mixed_precision and device.type=="cuda"
    def train(self,global_state: Mapping[str,torch.Tensor],local_epochs: int=1,prox_mu: float=0.0,report_loss: bool=True,release_mechanism: object|None=None,round_index: int=-1)->ClientTrainResult:
        self.model.load_state_dict(global_state); self.model.train(); opt=torch.optim.Adam(self.model.parameters(),lr=self.lr); scaler=torch.amp.GradScaler("cuda",enabled=self.mixed_precision); loss_sum=0.0; loss_samples=0
        reference={k:v.detach().to(self.device) for k,v in global_state.items()}
        for _ in range(local_epochs):
            for x,y,valid in self.train_loader:
                x=x.to(self.device); y=y.to(self.device); valid=valid.to(self.device); adjacency=batch_dynamic_adjacency(self.adjacency,x,self.dynamic_alpha,self.dynamic_top_k) if self.dynamic_graph else self.adjacency
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.mixed_precision):
                    pred=self.model(x,adjacency); loss=masked_l1_loss(pred,y,valid)
                    if prox_mu: loss=loss+.5*prox_mu*sum((p-reference[n]).square().sum() for n,p in self.model.named_parameters())
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
                if report_loss:
                    batch_samples=int(x.shape[0]); loss_sum+=float(loss.detach())*batch_samples; loss_samples+=batch_samples
        if report_loss and loss_samples==0:
            raise RuntimeError("local training produced no batches; client_train_loss is undefined")
        target_count=len(self.train_loader.dataset)*self.adjacency.shape[0]
        reported_loss=loss_sum/loss_samples if report_loss else math.nan
        update=model_update(self.model.state_dict(),global_state); dp_applied=False; privacy_ms=0.0
        if release_mechanism is not None:
            # Apply local DP before the update crosses the client/server API.
            # The coordinator receives only the returned DP tensors.
            privacy_start=time.perf_counter()
            if getattr(release_mechanism,"requires_context",False):
                released=release_mechanism.privatize(update,round_index,self.client_id)
            else:
                released=release_mechanism.privatize(update)
            privacy_ms=(time.perf_counter()-privacy_start)*1000.0
            update=released.tensors; dp_applied=bool(released.dp_applied)
        return ClientTrainResult(update,target_count,reported_loss,loss_samples,dp_applied,privacy_ms)
