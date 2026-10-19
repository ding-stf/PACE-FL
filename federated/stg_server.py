from __future__ import annotations
import random,time
from collections.abc import Sequence
import torch
from attacks.spectral_leakage.collector import UpdateCollector
from communication.error_feedback import ErrorFeedback
from communication.event_trigger import payload_l2_norm,should_upload
from communication.profiler import CommunicationRecord
from communication.serializer import deserialize_payload,serialize_payload
from communication.sparse_sync import decode,spectral_topk
from privacy.dp_fedavg import DPFedAvgMechanism,DPUpdate
from privacy.spectral_dp import SpectralDPMechanism
from privacy.central_spectral_dp import CentralSpectralDPMechanism
from .aggregation import apply_update,weighted_average_updates
from .stg_client import STGClient

class STGServer:
    def __init__(self,model: torch.nn.Module,clients: Sequence[STGClient],device: torch.device,seed: int=42,collector: UpdateCollector|None=None,privacy: DPFedAvgMechanism|SpectralDPMechanism|CentralSpectralDPMechanism|None=None,topk_ratio: float=1.0,error_feedback: bool=False,event_tau: float=0.0,skip_policy: str="zero",error_feedback_residual_norm: float|None=None,report_client_train_loss: bool=True,participation_schedule: Sequence[Sequence[int]]|None=None,record_upload_details: bool=False)->None:
        if skip_policy not in {"zero","cached"}: raise ValueError("skip_policy must be zero or cached")
        if not 0<topk_ratio<=1: raise ValueError("topk_ratio must be in (0,1]")
        if event_tau<0: raise ValueError("event_tau must be non-negative")
        self.model=model.to(device); self.clients=list(clients); self.device=device; self.rng=random.Random(seed); self.collector=collector; self.privacy=privacy; self.topk_ratio=topk_ratio; self.use_feedback=error_feedback; self.event_tau=event_tau; self.skip_policy=skip_policy; self.feedback={c.client_id:ErrorFeedback(error_feedback_residual_norm) for c in clients}; self.cache={}; self.communication=[]
        self.server_feedback=ErrorFeedback(error_feedback_residual_norm); self.record_upload_details=bool(record_upload_details); self.upload_communication=[]; self.report_client_train_loss=bool(report_client_train_loss)
        self.participation_schedule=(
            tuple(tuple(int(client_id) for client_id in row) for row in participation_schedule)
            if participation_schedule is not None else None
        )
        self.clients_by_id={client.client_id:client for client in self.clients}
        if len(self.clients_by_id)!=len(self.clients):
            raise ValueError("client_id values must be unique")
        if self.participation_schedule is not None:
            for row in self.participation_schedule:
                if len(row)!=len(set(row)):
                    raise ValueError("participation schedule cannot repeat a client within a round")
                unknown=set(row)-set(self.clients_by_id)
                if unknown:
                    raise ValueError(f"participation schedule contains unknown clients: {sorted(unknown)}")
    def run_round(self,round_index: int,clients_per_round: int,local_epochs: int=1,prox_mu: float=0.0)->dict[str,float]:
        start=time.perf_counter()
        expected_clients=min(clients_per_round,len(self.clients))
        if self.participation_schedule is None:
            selected=self.rng.sample(self.clients,expected_clients)
        else:
            if not 0<=round_index<len(self.participation_schedule):
                raise IndexError("round_index is outside the fixed participation schedule")
            selected_ids=self.participation_schedule[round_index]
            if len(selected_ids)!=expected_clients:
                raise ValueError("fixed participation row does not match clients_per_round")
            selected=[self.clients_by_id[client_id] for client_id in selected_ids]
        global_state={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}; updates=[]; weights=[]; raw_bytes=compressed_bytes=active=0; dp_ms=comp_ms=decode_ms=serialization_ms=0.0; loss_sum=0.0; loss_samples=0; payload_norms=[]
        central=getattr(self.privacy,"aggregation_level",None)=="central"; secure_updates=[]
        client_side_release=bool(
            self.privacy is not None
            and getattr(self.privacy,"client_side_release",False)
        )
        for client in selected:
            if client_side_release:
                result=client.train(
                    global_state,local_epochs,prox_mu,
                    report_loss=self.report_client_train_loss,
                    release_mechanism=self.privacy,round_index=round_index,
                )
                dp_ms+=result.privacy_ms
            else:
                result=(
                    client.train(global_state,local_epochs,prox_mu)
                    if self.report_client_train_loss
                    else client.train(global_state,local_epochs,prox_mu,report_loss=False)
                )
            if self.report_client_train_loss:
                result_loss_samples=result.loss_samples if result.loss_samples>0 else result.samples
                loss_sum+=result.client_train_loss*result_loss_samples; loss_samples+=result_loss_samples
            raw_bytes+=sum(t.numel()*t.element_size() for t in result.update.values())
            if central:
                t=time.perf_counter(); secure_updates.append(self.privacy.clip_for_secure_aggregation(result.update)); dp_ms+=(time.perf_counter()-t)*1000; weights.append(result.samples)
                # The server never receives individual values in this branch.
                continue
            t=time.perf_counter()
            if client_side_release:
                if not result.dp_applied:
                    raise RuntimeError("client-side release did not return a DP update")
                private=DPUpdate(result.update,True,{})
            elif self.privacy and getattr(self.privacy,"requires_context",False):
                private=self.privacy.privatize(result.update,round_index,client.client_id)
            else:
                private=self.privacy.privatize(result.update) if self.privacy else DPUpdate(result.update,False,{})
            if not client_side_release:
                dp_ms+=(time.perf_counter()-t)*1000
            observed=private.tensors
            compression_active=self.topk_ratio<1
            trigger_active=self.event_tau>0
            if compression_active or trigger_active:
                if not private.dp_applied: raise ValueError("compression/event trigger require a DP mechanism")
                t=time.perf_counter(); payload=self.feedback[client.client_id].compress(private,self.topk_ratio) if self.use_feedback and compression_active else spectral_topk(private,self.topk_ratio); comp_ms+=(time.perf_counter()-t)*1000; payload_norms.append(payload_l2_norm(payload))
                if should_upload(payload,self.event_tau):
                    t=time.perf_counter(); encoded=serialize_payload(payload); compressed_bytes+=len(encoded); serialization_ms+=(time.perf_counter()-t)*1000
                    if self.record_upload_details:
                        nonzero=sum(int(item.values.numel()) for item in payload.tensors.values())
                        total=sum(int(torch.tensor(item.shape).prod()) for item in payload.tensors.values())
                        self.upload_communication.append({
                            "Round":round_index,"ClientID":client.client_id,
                            "SerializedBytes":len(encoded),"NumberOfNonzeroValues":nonzero,
                            "TotalNumel":total,"Rho":self.topk_ratio,
                            "DPProtectedInput":bool(private.dp_applied),
                            "ErrorFeedback":bool(self.use_feedback),
                            "ResidualSource":"DP_PROTECTED_RELEASE_ONLY" if self.use_feedback else "NONE",
                            "SerializationFormat":"FSS1",
                        })
                    t=time.perf_counter(); observed=decode(deserialize_payload(encoded)); decode_ms+=(time.perf_counter()-t)*1000; self.cache[client.client_id]=observed; active+=1
                else: observed=self.cache.get(client.client_id,{k:torch.zeros_like(v) for k,v in result.update.items()}) if self.skip_policy=="cached" else {k:torch.zeros_like(v) for k,v in result.update.items()}
            else:
                compressed_bytes+=raw_bytes if len(selected)==1 else sum(t.numel()*t.element_size() for t in observed.values()); active+=1
            if self.collector: self.collector.collect(round_index,client.client_id,observed)
            updates.append(observed); weights.append(result.samples)
        secure_upload_bytes=dense_broadcast_bytes=sparse_broadcast_bytes=0; protocol_scope="client_upload"
        if central:
            t=time.perf_counter(); private=self.privacy.privatize_secure_mean(secure_updates); dp_ms+=(time.perf_counter()-t)*1000
            observed=private.tensors
            # DP output is compressed only after central release.  It is the
            # server-to-client synchronization payload, not a private uplink.
            dense_update_bytes=sum(t.numel()*t.element_size() for t in observed.values())
            if self.topk_ratio==1.0:
                # A sparse encoding at rho=1 unnecessarily transmits one index
                # per coefficient.  Use the dense DP release directly: this is
                # the no-compression ablation and its byte count must be exact.
                norm=float(torch.sqrt(sum(t.double().square().sum() for t in observed.values())))
                payload_norms.append(norm)
                if norm>=self.event_tau:
                    broadcast_bytes=dense_update_bytes; active=len(selected)
                else:
                    broadcast_bytes=0; observed={k:torch.zeros_like(v) for k,v in global_state.items()}; active=0
            else:
                t=time.perf_counter(); payload=self.server_feedback.compress(private,self.topk_ratio) if self.use_feedback else spectral_topk(private,self.topk_ratio); comp_ms+=(time.perf_counter()-t)*1000; payload_norms.append(payload_l2_norm(payload))
                if should_upload(payload,self.event_tau):
                    t=time.perf_counter(); encoded=serialize_payload(payload); broadcast_bytes=len(encoded); serialization_ms+=(time.perf_counter()-t)*1000
                    t=time.perf_counter(); observed=decode(deserialize_payload(encoded)); decode_ms+=(time.perf_counter()-t)*1000; active=len(selected)
                else:
                    broadcast_bytes=0; observed={k:torch.zeros_like(v) for k,v in global_state.items()}; active=0
            # Total measured protocol payload: dense secure-aggregation uploads
            # plus one broadcast to every selected client.  Keep the individual
            # directions separate for accurate reporting.
            dense_broadcast=dense_update_bytes*len(selected)
            secure_upload_bytes=raw_bytes; dense_broadcast_bytes=dense_broadcast; sparse_broadcast_bytes=broadcast_bytes*len(selected)
            compressed_bytes=secure_upload_bytes+sparse_broadcast_bytes
            raw_bytes=secure_upload_bytes+dense_broadcast_bytes
            protocol_scope="secure_aggregation_uplink_plus_central_dp_broadcast"
            updates=[observed]; weights=[1.0]
        t=time.perf_counter(); aggregate=weighted_average_updates(updates,weights); self.model.load_state_dict(apply_update(global_state,aggregate)); aggregation_ms=(time.perf_counter()-t)*1000; wall=(time.perf_counter()-start)*1000
        rec=CommunicationRecord(round_index,len(selected),active,raw_bytes,compressed_bytes,serialization_ms,dp_ms,comp_ms,decode_ms,aggregation_ms,wall,sum(payload_norms)/max(len(payload_norms),1),max(payload_norms,default=0.0),secure_upload_bytes,dense_broadcast_bytes,sparse_broadcast_bytes,protocol_scope); self.communication.append(rec)
        if self.report_client_train_loss and loss_samples<=0:
            raise RuntimeError("federated round produced no training samples; client_train_loss is undefined")
        # Empty fields are an explicit redaction, not a numerical observation.
        client_train_loss=loss_sum/loss_samples if self.report_client_train_loss else ""
        return {"client_train_loss":client_train_loss,"train_loss":client_train_loss,**rec.to_dict()}
