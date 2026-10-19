from __future__ import annotations
"""Experiment 19: exact RSR-trajectory replay with paired same-raw-update releases."""
import argparse,csv,hashlib,json,os,platform,shutil,sys,time,traceback
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import sklearn,torch,yaml
from attacks.ed_om_rsr_attack import (PublicReleaseContext,full_transform_aware_features,mixed_domain_value_features,mixed_intersection_features,public_mask_features,release_value_features)
from attacks.mask_aware_spectral_attack import FeatureVector
from attacks.spectral_leakage.collector import UpdateCollector
from communication.exact_dimension_orthogonal_mixing import GLOBAL_MIX_SEED,exact_dimension_rsr_release,group_dimensions,transform_seed_policy
from communication.implicit_mask_rsr import PublicMaskContext,encode_implicit_mask_payload,schema_from_state_dict
from communication.randomized_spectral_release import SPECTRAL_GROUPS,mask_seed_policy,randomized_spectral_release,tensor_mapping_sha256
from experiments import exp_mask_aware_adaptive_attack as e13
from experiments import exp_randomized_spectral_release_sanity as e11
from experiments import finalize_ed_om_rsr_multiseed as pf
from experiments import finalize_metrla_cross_dataset as mf
from experiments.common import save_config
from experiments.ed_om_rsr_multiseed_common import payload_pair_audit,retained_count
from experiments.exp_ellipsoidal_clipping_sanity import _build_clients_and_eval,_environment_snapshot,_prepare_data
from experiments.exp_representative_privacy_attack import extract_feature_families
from experiments.exp_spectral_norm_audit import _load_reference_config,_resolve_device,_set_seed
from federated.aggregation import apply_update,weighted_average_updates

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
P=.5; SEEDS=(42,123,2026); DATASETS=("pems08","metr_la"); PROPS=("Extreme Event","Periodicity Strength")
N=50; R=20; K=5
PEMS_CFG=ROOT/"results/pems08/pjsdp/eps2048_balanced4_pjsdp_bandp90_uniform_50c40r_seed42_20260902/seed_42/config.yaml"
METR_CFG=ROOT/"results/pems08/pjsdp/eps4096_balanced4_pjsdp_aniso_low075_high150_50c40r_seed42_20260902/seed_42/config.yaml"

def args():
 p=argparse.ArgumentParser();p.add_argument("--dataset",choices=DATASETS,required=True);p.add_argument("--seed",choices=SEEDS,type=int,required=True);p.add_argument("--device",default="cuda");return p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()).hexdigest()
def rows(p):
 with p.open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))
def csvwrite(p,x):
 x=list(x)
 if not x:raise ValueError(f"empty CSV: {p}")
 fields=list(dict.fromkeys(k for r in x for k in r))
 with p.open("w",newline="",encoding="utf8") as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(x)
def jswrite(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf8")
def archive(ds,s):
 if ds=="pems08" and s==42:return ROOT/"results/randomized_spectral_release_sanity/pems08/seed42/p050"
 if ds=="pems08":return ROOT/f"results/ed_om_rsr_multiseed/pems08/seed{s}/rsr"
 return ROOT/f"results/metrla_cross_dataset/rsr/seed{s}"
def buildcfg(ds):return PEMS_CFG if ds=="pems08" else METR_CFG
def deterministic(s):
 _set_seed(s);torch.use_deterministic_algorithms(True)
 if torch.backends.cudnn.is_available():torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def schedule(p):
 q=json.loads(p.read_text())["rounds_client_ids"];q=[[int(i) for i in x] for x in q];c=Counter(i for x in q for i in x)
 if len(q)!=R or any(len(x)!=K or len(set(x))!=K for x in q):raise RuntimeError("schedule dimensions")
 if set(c)!=set(range(N)) or any(c[i]!=2 for i in range(N)):raise RuntimeError("schedule R2")
 return q
def hashmap(a):
 q=json.loads((a/"server_visible_update_hashes.json").read_text())["records"];z={(int(x["Round"]),int(x["ClientID"])):x for x in q}
 if len(z)!=100:raise RuntimeError("archive hash coverage")
 return z
def source_gate(ds,s,a):
 cur={"communication/randomized_spectral_release.py":sha(ROOT/"communication/randomized_spectral_release.py"),"communication/implicit_mask_rsr.py":sha(ROOT/"communication/implicit_mask_rsr.py"),"communication/exact_dimension_orthogonal_mixing.py":sha(ROOT/"communication/exact_dimension_orthogonal_mixing.py")}
 m=json.loads((a/"manifest.json").read_text())
 if ds=="pems08" and s==42:exp=m["code_and_protocol_hashes"];tr="experiments/exp_randomized_spectral_release_sanity.py"
 else:exp=m["source_hashes"];tr="experiments/exp_ed_om_rsr_multiseed_train.py" if ds=="pems08" else "experiments/exp_metrla_ed_om_train.py"
 cur[tr]=sha(ROOT/tr)
 for k,v in cur.items():
  if k in exp and exp[k]!=v:raise RuntimeError(f"source changed: {k}")
 return cur
def label_archive(ds,s):
 if ds=="pems08" and s==42:return ROOT/"results/randomized_spectral_release_sanity/pems08/seed42/p050/attack/property_labels.csv"
 if ds=="pems08":return ROOT/f"results/ed_om_rsr_multiseed/pems08/summary/attack/seed{s}/property_labels_and_folds.csv"
 return ROOT/"results/metrla_cross_dataset/property_labels.csv"
def labels(ds,s):
 if ds=="pems08":lf,rec,_,_=pf.labels_and_folds(s)
 else:
  q,rec,_=mf.labels(s);lf={p:(np.asarray(q[p][2],int),np.asarray(q[p][1],int)) for p in PROPS}
 old=rows(label_archive(ds,s))
 for p in PROPS:
  z=sorted((x for x in old if x["Property"]==p and ("Seed" not in x or int(x["Seed"])==s)),key=lambda x:int(x["ClientID"]))
  if len(z)!=N:raise RuntimeError(f"archived label coverage {p}")
  if not np.array_equal(lf[p][0],[int(x["Label"]) for x in z]):raise RuntimeError(f"labels changed {p}")
  if not np.array_equal(lf[p][1],[int(x["Fold"]) for x in z]):raise RuntimeError(f"folds changed {p}")
 return lf,rec
def audit(ds,s):
 if os.getenv("CONDA_DEFAULT_ENV")!="dsl1":raise RuntimeError("must run in dsl1")
 if os.getenv("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 required")
 a=archive(ds,s);req=[a/x for x in ("manifest.json","config.yaml","final_model.pt","participation_schedule.json","server_visible_update_hashes.json","public_schema.json")]+[buildcfg(ds)];miss=[str(x) for x in req if not x.is_file()]
 if miss:raise FileNotFoundError(miss)
 c=yaml.safe_load((a/"config.yaml").read_text())
 for k,v in {"dataset":ds,"seed":s,"rounds":20,"num_clients":50,"clients_per_round":5,"releases_per_client":2,"local_epochs":1,"history":12,"horizon":12,"batch_size":64,"learning_rate":.01}.items():
  if c.get(k)!=v:raise RuntimeError(f"archived config mismatch {k}")
 sp=a/"participation_schedule.json"
 if sha(sp)!=c["schedule_hash"]:raise RuntimeError("schedule hash")
 lf,lr=labels(ds,s);return {"a":a,"sp":sp,"sched":schedule(sp),"hash":hashmap(a),"src":source_gate(ds,s,a),"lf":lf,"lr":lr,"schema":json.loads((a/"public_schema.json").read_text())}
def prepare(ds,s,dev):
 c=dict(_load_reference_config(buildcfg(ds)));c.update(dataset=ds,seed=s,rounds=20,num_clients=50,clients_per_round=5,local_epochs=1,history=12,horizon=12,lr=.01)
 if ds=="metr_la":c.update(projection_ratio=.90,projection_seed=42)
 deterministic(s);d=_resolve_device(dev);p=_prepare_data(c,ROOT/"data",ds);m,clients,_,_=_build_clients_and_eval(c,p,d);m=m.to(d)
 if len(clients)!=N:raise RuntimeError("partition count")
 return c,d,m,{x.client_id:x for x in clients}
def ident_meta():
 n=[];v=[]
 for g in SPECTRAL_GROUPS:
  n += [f"transform__{g}__dimension_retention_ratio",f"transform__{g}__padding_fraction",f"transform__{g}__release1_sign_balance",f"transform__{g}__release2_sign_balance",f"transform__{g}__same_sign_fraction"];v += [1.,0.,1.,1.,1.]
 return FeatureVector(tuple(n),np.asarray(v))
def rsr_full(std,pay,ctx,sc):
 parts=[FeatureVector(tuple(f"standard__{i:03d}" for i in range(24)),np.asarray(std)),mixed_domain_value_features(pay,ctx,sc),public_mask_features(pay,ctx,sc),mixed_intersection_features(pay,ctx,sc),ident_meta()]
 z=FeatureVector(tuple(n for x in parts for n in x.names),np.concatenate([x.values for x in parts]))
 if len(z.names)!=168 or len(set(z.names))!=168:raise RuntimeError("RSR Full dimension")
 return z
def schema_rows(u):return [{"name":k,"shape":list(v.shape),"dtype":str(v.dtype),"numel":v.numel()} for k,v in sorted(u.items())]
def attack_eval(ds,s,sc,lf,rp,ep,rr,er):
 mats={(m,a):[] for m in ("RSR","ED-OM-RSR") for a in ("Release-1","Full")};names={};fr=[]
 for i in range(N):
  a=[x[1] for x in sorted(rp[i])];ei=sorted(ep[i],key=lambda x:x[0].round_id);ctx=tuple(x[0] for x in ei);b=tuple(x[1] for x in ei)
  sr=extract_feature_families([e11.attack_record(x) for x in sorted(rr[i],key=lambda x:x.round)])["proposed_spectral"];se=extract_feature_families([e11.attack_record(x) for x in sorted(er[i],key=lambda x:x.round)])["proposed_spectral"]
  vec={("RSR","Release-1"):release_value_features(a,ctx,sc,1),("ED-OM-RSR","Release-1"):release_value_features(b,ctx,sc,1),("RSR","Full"):rsr_full(sr,a,ctx,sc),("ED-OM-RSR","Full"):full_transform_aware_features(se,b,ctx,sc)}
  if vec[("RSR","Release-1")].names!=vec[("ED-OM-RSR","Release-1")].names or vec[("RSR","Full")].names!=vec[("ED-OM-RSR","Full")].names:raise RuntimeError("paired feature schema")
  for k,z in vec.items():
   dim=28 if k[1]=="Release-1" else 168;v=np.asarray(z.values,float)
   if v.shape!=(dim,) or not np.isfinite(v).all():raise RuntimeError(f"feature vector {k}/{i}")
   if k in names and names[k]!=z.names:raise RuntimeError("client feature schema")
   names[k]=z.names;mats[k].append(v);fr += [{"Dataset":ds,"Seed":s,"Mechanism":k[0],"Attack":k[1],"ClientID":i,"FeatureIndex":j,"FeatureName":n,"FeatureValue":float(y)} for j,(n,y) in enumerate(zip(z.names,v))]
 mats={k:np.stack(v) for k,v in mats.items()};met=[];boot=[];pred=[];pair=[];res=[]
 for prop in PROPS:
  y,f=lf[prop];pr={};mm={}
  for mech in ("RSR","ED-OM-RSR"):
   for att in ("Release-1","Full"):
    k=(mech,att);common={"Dataset":ds,"Seed":s,"Mechanism":mech,"Attack":att,"FeatureDimension":mats[k].shape[1]};x,b,p,q=e13.evaluate(prop,f"{mech} {att} Matched",y,f,x=mats[k],origin=f"Experiment 19 same-raw-update {ds} seed{s}")
    met.append({**common,**x});boot.append({**common,**b});pred += [{**common,**w} for w in p];pr[k]=q;mm[k]=x
  for att in ("Release-1","Full"):
   pb=e13.paired_boot(y,pr[("ED-OM-RSR",att)],pr[("RSR",att)]);r=float(mm[("RSR",att)]["ROCAUCSeparability"]);e=float(mm[("ED-OM-RSR",att)]["ROCAUCSeparability"])
   pair.append({"Dataset":ds,"Seed":s,"Property":prop,"Attack":att,"Contrast":"ED-OM-RSR minus RSR AUC-separability",**pb});res.append({"Dataset":ds,"Seed":s,"Property":prop,"Attack":att,"RSR_S":r,"ED_OM_RSR_S":e,"PairedDelta_ED_minus_RSR":e-r})
 return {"features":fr,"schema":{f"{k[0]}/{k[1]}":list(v) for k,v in names.items()},"metrics":met,"bootstrap":boot,"predictions":pred,"paired":pair,"results":res}
def run(ds,s,dev):
 au=audit(ds,s);dest=OUT/"runs"/ds/f"seed{s}";dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists() or list(dest.parent.glob(f".seed{s}_staging_*")):raise FileExistsError(dest)
 st=dest.parent/f".seed{s}_staging_{os.getpid()}";st.mkdir();started=time.perf_counter()
 try:
  for x in ("pre_release_dense_updates","payloads/rsr","payloads/ed_om_rsr","reconstructed_updates/rsr","reconstructed_updates/ed_om_rsr"):(st/x).mkdir(parents=True)
  shutil.copy2(au["sp"],st/"participation_schedule.json");(st/"command.txt").write_text(" ".join([sys.executable,*sys.argv])+"\n");(st/"git_commit.txt").write_text("UNAVAILABLE_NOT_A_GIT_CHECKOUT\n")
  cfg,device,model,clients=prepare(ds,s,dev);(st/"environment.txt").write_text(_environment_snapshot(device))
  ec={"experiment":19,"dataset":ds,"seed":s,"p":P,"anchor_trajectory":"archived RSR","source_archive":str(au["a"]),"rounds":20,"num_clients":50,"clients_per_round":5,"releases_per_client":2,"local_optimizer":"Adam","learning_rate":.01,"batch_size":64,"local_epochs":1,"history":12,"horizon":12,"properties":list(PROPS),"observations":["Release-1","Full"],"feature_dimensions":{"Release-1":28,"Full":168},"classifier":"L2 logistic C=1, balanced; training-fold-only scaler","public_Q":True,"public_masks":True,"DP":False,"SecureAggregation":False,"encryption":False,"padding":False,"schedule_sha256":sha(au["sp"]),"CUBLAS_WORKSPACE_CONFIG":os.getenv("CUBLAS_WORKSPACE_CONFIG")}
  save_config(st/"config.yaml",ec);jswrite(st/"config.json",ec);jswrite(st/"mask_seed_policy.json",mask_seed_policy());jswrite(st/"transform_seed_policy.json",transform_seed_policy());csvwrite(st/"property_labels_and_folds.csv",au["lr"])
  raw=UpdateCollector(True,st/"pre_release_dense_updates");rc=UpdateCollector(True,st/"reconstructed_updates/rsr");edc=UpdateCollector(True,st/"reconstructed_updates/ed_om_rsr")
  cnt=defaultdict(int);rp=defaultdict(list);ep=defaultdict(list);rr=defaultdict(list);er=defaultdict(list);rel=[];rnd=[];sc=None;sch=None
  for ri,ids in enumerate(au["sched"]):
   t=time.perf_counter();gs={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};ups=[];ws=[];loss=0.;ns=0
   for ci in ids:
    z=clients[ci].train(gs,1,0.,report_loss=True)
    if z.dp_applied:raise RuntimeError("DP flag")
    li=cnt[ci];ctx=PublicMaskContext(s,ci,ri,li,P);pctx=PublicReleaseContext(ci,ri,li)
    if sc is None:
     sc=schema_from_state_dict(z.update);sch=canon(sc.to_dict());jswrite(st/"public_schema.json",sc.to_dict())
     if canon(sc.to_dict())!=canon(au["schema"]):raise RuntimeError("replayed public schema differs from archive")
    elif canon(schema_from_state_dict(z.update).to_dict())!=sch:raise RuntimeError("raw schema changed")
    dr=raw.collect(ri,ci,z.update);dh=tensor_mapping_sha256(z.update);a=randomized_spectral_release(z.update,p=P,experiment_seed=s,client_id=ci,round_id=ri,release_index=li);b=exact_dimension_rsr_release(z.update,p=P,experiment_seed=s,client_id=ci,round_id=ri,release_index=li,global_mix_seed=GLOBAL_MIX_SEED,schema=sc);pa=payload_pair_audit(a.payload,b.payload)
    if not all(pa[k] for k in ("SchemaMatch","MaskMatch","RetainedCountMatch")):raise RuntimeError("paired mask gate")
    dim=sum(group_dimensions(sc).values())
    if a.total_spectral_coefficients!=dim or b.original_coordinates!=dim or b.mixed_coordinates!=dim:raise RuntimeError("exact dimension")
    ab=encode_implicit_mask_payload(a.payload,sc,ctx)
    if len(ab)!=len(b.serialized):raise RuntimeError("serialized size")
    base=f"round_{ri:04d}_client_{ci:03d}";(st/"payloads/rsr"/f"{base}.imr").write_bytes(ab);(st/"payloads/ed_om_rsr"/f"{base}.edm").write_bytes(b.serialized)
    ar=rc.collect(ri,ci,a.reconstructed_update);br=edc.collect(ri,ci,b.reconstructed_update);expect=au["hash"][(ri,ci)]["ServerReconstructedUpdateSHA256"]
    if a.reconstructed_sha256!=expect:raise RuntimeError(f"archived RSR mismatch round={ri} client={ci}: {a.reconstructed_sha256} != {expect}")
    rp[ci].append((ri,a.payload));ep[ci].append((pctx,b.payload));rr[ci].append(ar);er[ci].append(br);ups.append(a.reconstructed_update);ws.append(z.samples);loss+=z.client_train_loss*z.loss_samples;ns+=z.loss_samples
    rel.append({"Dataset":ds,"Seed":s,"Round":ri,"ClientID":ci,"ReleaseIndex":li,"RawDenseUpdateSHA256":dh,"RSRInputDenseUpdateSHA256":dh,"EDOMInputDenseUpdateSHA256":dh,"RawUpdateNorm":dr.update_norm,"RawTensorSchemaSHA256":canon(schema_rows(z.update)),"RSRReconstructedSHA256":a.reconstructed_sha256,"ArchivedRSRReconstructedSHA256":expect,"ExactReplayMatch":True,**pa,"RSRRetainedCoordinates":retained_count(a.payload),"EDRetainedCoordinates":retained_count(b.payload),"RSRSerializedBytes":len(ab),"EDSerializedBytes":len(b.serialized),"SerializedBytesMatch":len(ab)==len(b.serialized),"RawDTypes":"|".join(sorted({str(v.dtype) for v in z.update.values()}))});cnt[ci]+=1
   model.load_state_dict(apply_update(gs,weighted_average_updates(ups,ws)));rnd.append({"Dataset":ds,"Seed":s,"Round":ri,"Clients":len(ids),"ClientTrainLoss":loss/ns,"RoundSeconds":time.perf_counter()-t,"GlobalModelSHA256":tensor_mapping_sha256(model.state_dict())});print(f"dataset={ds} seed={s} round={ri:02d} loss={loss/ns:.6f} replay_matches={len(rel)}",flush=True)
  if len(raw.records)!=100 or any(cnt[i]!=2 for i in range(N)):raise RuntimeError("release coverage")
  fh=tensor_mapping_sha256(model.state_dict());old=torch.load(au["a"]/"final_model.pt",map_location="cpu",weights_only=False);oh=tensor_mapping_sha256(old)
  if fh!=oh:raise RuntimeError(f"final trajectory mismatch {fh} != {oh}")
  torch.save(model.state_dict(),st/"final_model.pt");csvwrite(st/"release_pair_audit.csv",rel);csvwrite(st/"round_replay.csv",rnd);av=attack_eval(ds,s,sc,au["lf"],rp,ep,rr,er);csvwrite(st/"attack_features.csv",av["features"]);jswrite(st/"attack_feature_schema.json",av["schema"]);csvwrite(st/"attack_metrics.csv",av["metrics"]);csvwrite(st/"attack_bootstrap.csv",av["bootstrap"]);csvwrite(st/"attack_oof_predictions.csv",av["predictions"]);csvwrite(st/"paired_bootstrap.csv",av["paired"]);csvwrite(st/"run_results.csv",av["results"])
  man={"experiment":19,"status":"COMPLETE_EXACT_REPLAY","dataset":ds,"seed":s,"source_archive":str(au["a"]),"source_archive_manifest_sha256":sha(au["a"]/"manifest.json"),"schedule_sha256":sha(au["sp"]),"release_count":len(rel),"raw_dense_updates_archived":len(raw.records),"raw_dense_updates_used_directly_for_both_branches":True,"reconstructed_updates_used_as_raw_input":False,"all_release_replay_hashes_match":all(x["ExactReplayMatch"] for x in rel),"all_masks_match":all(x["MaskMatch"] for x in rel),"all_retained_counts_match":all(x["RetainedCountMatch"] for x in rel),"all_serialized_bytes_match":all(x["SerializedBytesMatch"] for x in rel),"replay_final_model_sha256":fh,"archived_final_model_sha256":oh,"final_model_exact_match":fh==oh,"feature_dimensions":{"Release-1":28,"Full":168},"attack_classifier":"L2 Logistic Regression C=1; client-level OOF","runtime_seconds":time.perf_counter()-started,"git_commit":"UNAVAILABLE_NOT_A_GIT_CHECKOUT","environment":{"conda_environment":os.getenv("CONDA_DEFAULT_ENV"),"python":sys.version,"python_executable":sys.executable,"platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__,"device":str(device)},"source_hashes":{**au["src"],"attacks/ed_om_rsr_attack.py":sha(ROOT/"attacks/ed_om_rsr_attack.py"),"experiments/exp_representative_privacy_attack.py":sha(ROOT/"experiments/exp_representative_privacy_attack.py"),"experiments_revision/exp19_frozen_update/run_exp19.py":sha(Path(__file__))}}
  man["output_hashes"]={str(p.relative_to(st)):sha(p) for p in sorted(st.rglob("*")) if p.is_file() and p.name!="manifest.json"};jswrite(st/"manifest.json",man);st.rename(dest);print(f"FINAL_OUTPUT={dest}",flush=True)
  for x in av["results"]:
   if x["Property"]=="Periodicity Strength":print(f"RESULT attack={x['Attack']} RSR_S={x['RSR_S']:.12f} ED_S={x['ED_OM_RSR_S']:.12f} DELTA={x['PairedDelta_ED_minus_RSR']:.12f}",flush=True)
 except Exception as e:
  jswrite(st/"failure.json",{"experiment":19,"status":"FAILED","dataset":ds,"seed":s,"error_type":type(e).__name__,"error":str(e),"traceback":traceback.format_exc(),"runtime_seconds":time.perf_counter()-started});f=OUT/"failures";f.mkdir(parents=True,exist_ok=True);d=f/f"{ds}_seed{s}_{time.strftime('%Y%m%d_%H%M%S')}";st.rename(d);print(f"FAILURE_OUTPUT={d}",flush=True);raise
if __name__=="__main__":
 a=args();run(a.dataset,a.seed,a.device)
