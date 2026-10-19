from __future__ import annotations
from dataclasses import dataclass,asdict
@dataclass
class CommunicationRecord:
    round: int; selected_clients: int; active_upload_clients: int; raw_parameter_bytes: int; compressed_bytes: int; serialization_time_ms: float; client_dp_time_ms: float; compression_time_ms: float; server_decode_time_ms: float; aggregation_time_ms: float; round_wall_clock_ms: float; mean_dp_payload_norm: float=0.0; max_dp_payload_norm: float=0.0; secure_aggregation_upload_bytes: int=0; dense_broadcast_bytes: int=0; sparse_broadcast_bytes: int=0; protocol_scope: str="client_upload"
    def to_dict(self)->dict[str,float|int]:
        x=asdict(self); x["upload_ratio"]=self.active_upload_clients/max(self.selected_clients,1); x["compression_ratio"]=self.compressed_bytes/max(self.raw_parameter_bytes,1); x["estimated_tx_10mbps_ms"]=self.compressed_bytes*8/10_000_000*1000; x["estimated_tx_100mbps_ms"]=self.compressed_bytes*8/100_000_000*1000; return x
