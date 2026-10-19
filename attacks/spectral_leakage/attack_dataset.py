from __future__ import annotations
import numpy as np

def raw_properties(client_values: list[np.ndarray],z_threshold: float=2.5,points_per_hour: int=12)->dict[str,np.ndarray]:
    if points_per_hour<=0:
        raise ValueError("points_per_hour must be positive")
    peak=[]; periodic=[]; extreme_score=[]; steps_per_day=24*points_per_hour
    for x in client_values:
        signal=np.asarray(x[...,0],dtype=np.float64)
        valid=np.isfinite(signal)&(np.abs(signal)>1e-5)
        time_counts=valid.sum(axis=1); profile=np.divide(np.where(valid,signal,0.0).sum(axis=1),time_counts,out=np.full(len(signal),np.nan),where=time_counts>0)
        fallback=float(profile[np.isfinite(profile)].mean()) if np.isfinite(profile).any() else 0.0
        profile=np.nan_to_num(profile,nan=fallback)
        hours=(np.arange(len(profile))//points_per_hour)%24
        am=profile[(hours>=6)&(hours<12)].mean(); pm=profile[(hours>=15)&(hours<21)].mean(); peak.append(int(am>pm))
        centered=profile-profile.mean(); power=np.abs(np.fft.rfft(centered))**2; idx=max(1,round(len(profile)/steps_per_day)); periodic.append(float(power[min(idx,len(power)-1)]/(power.sum()+1e-12)))
        node_counts=valid.sum(axis=0,keepdims=True); center=np.divide(np.where(valid,signal,0.0).sum(axis=0,keepdims=True),node_counts,out=np.zeros((1,signal.shape[1])),where=node_counts>0)
        variance=np.divide(np.where(valid,(signal-center)**2,0.0).sum(axis=0,keepdims=True),node_counts,out=np.zeros((1,signal.shape[1])),where=node_counts>0); scale=np.sqrt(variance); z=np.where(valid,np.abs((signal-center)/(scale+1e-12)),0.0)
        extreme_score.append(float(np.sum((z>z_threshold)&valid)/max(int(valid.sum()),1)))
    return {"peak_pattern":np.asarray(peak),"periodicity_score":np.asarray(periodic),"extreme_score":np.asarray(extreme_score)}

def label_by_train_median(scores: np.ndarray,train_indices: np.ndarray)->tuple[np.ndarray,float]:
    threshold=float(np.median(scores[train_indices])); return (scores>=threshold).astype(int),threshold

def label_periodicity(scores: np.ndarray,train_indices: np.ndarray)->tuple[np.ndarray,float]:
    return label_by_train_median(scores,train_indices)
