from __future__ import annotations
import csv,json,hashlib,os
from collections import defaultdict
from pathlib import Path
import numpy as np,torch
from attacks.ed_om_rsr_attack import PublicReleaseContext,full_transform_aware_features,release_value_features as edvals
from attacks.mask_aware_spectral_attack import build_mask_aware_feature_vector
from attacks.rsr_leakage_decomposition import release_value_features as rsrvals
from attacks.spectral_leakage.attack_dataset import raw_properties
from communication.exact_dimension_orthogonal_mixing import deserialize_ed_payload
from communication.implicit_mask_rsr import PublicMaskContext,PublicRSRSchema,decode_implicit_mask_payload
from communication.randomized_spectral_release import serialize_rsr_payload
from communication.serializer import serialize_dense_update
from datasets.client_partition import partition_nodes
from datasets.stg_utils import temporal_split
from datasets.traffic import load_traffic_dataset
from experiments import exp_metrla_cross_dataset_attack as old,exp_mask_aware_adaptive_attack as e13,exp_randomized_spectral_release_sanity as e11
from experiments.common import write_rows
from experiments.ed_om_rsr_multiseed_common import payload_pair_audit
from experiments.exp_representative_privacy_attack import extract_feature_families
from experiments.finalize_ed_om_rsr_multiseed import verify_output_hashes
from scripts.finalize_experiment14 import validate_run
R=Path(__file__).resolve().parents[1];B=R/"results/metrla_cross_dataset";S=(42,123,2026);P=("Extreme Event","Periodicity Strength")
F={42:R/"results/metrla_final_sanity/seed42/fedavg",123:R/"results/final_icassp/metr_la/fedavg/seed123",2026:R/"results/final_icassp/metr_la/fedavg/seed2026"}
def rows(p):
 with open(p,newline="",encoding="utf8") as f:return list(csv.DictReader(f))
def labels(seed):
 d=load_traffic_dataset(R/"data","metr_la");t=temporal_split(d.values).train;q=partition_nodes(d.adjacency,50,d.coordinates,seed);z=raw_properties([t[:,n] for n in q.node_ids]);out={};rec=[];thr=[]
 for p,k in (("Extreme Event","extreme_score"),("Periodicity Strength","periodicity_score")):
  score=np.asarray(z[k],float);prov,fold=old.leakage_free_stratified_folds(score,5);lab,tt=old.fold_local_labels(score,fold);out[p]=(score,fold,lab)
  rec += [{"Seed":seed,"Property":p,"ClientID":i,"Score":score[i],"Fold":fold[i],"Label":lab[i],"ProvisionalLOOStratum":prov[i],"TrainingDataOnly":True} for i in range(50)]
  thr += [{"Seed":seed,"Property":p,**x} for x in tt]
 return out,rec,thr
def load(seed):
 rd=B/"rsr"/f"seed{seed}";ed=B/"ed_om_rsr"/f"seed{seed}";rm=json.loads((rd/"manifest.json").read_text());em=json.loads((ed/"manifest.json").read_text())
 for p,m in ((rd,rm),(ed,em)):verify_output_hashes(p,m)
 if any(rm[k]!=em[k] for k in ("schedule_sha256","public_schema_sha256","mask_trajectory_sha256","retained_count_trajectory_sha256")) or rm["padding"] or em["padding"]:raise RuntimeError("paired protocol mismatch")
 sc=PublicRSRSchema.from_dict(json.loads((rd/"public_schema.json").read_text()));a={(int(x["Round"]),int(x["ClientID"])):x for x in rows(rd/"round_release_mapping.csv")};b={(int(x["Round"]),int(x["ClientID"])):x for x in rows(ed/"round_release_mapping.csv")}
 rp,ep,rr,er=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list);audit=[];explicit=0
 for key in sorted(a):
  r,c=key;i=int(a[key]["ReleaseIndex"]);ctx=PublicMaskContext(seed,c,r,i,.5);pa=rd/"server_visible_payloads"/a[key]["PayloadFile"];pb=ed/"server_visible_payloads"/b[key]["PayloadFile"];x=decode_implicit_mask_payload(pa.read_bytes(),sc,ctx);y=deserialize_ed_payload(pb.read_bytes(),sc,ctx);q=payload_pair_audit(x,y);explicit+=len(serialize_rsr_payload(x));q.update({"Seed":seed,"Round":r,"ClientID":c,"ReleaseIndex":i,"RSRImplicitBytes":pa.stat().st_size,"EDOMBytes":pb.stat().st_size});audit.append(q);rp[c].append((r,x));ep[c].append((PublicReleaseContext(c,r,i),y));rr[c].append(torch.load(rd/"server_visible_updates"/pa.with_suffix(".pt").name,map_location="cpu",weights_only=False));er[c].append(torch.load(ed/"server_visible_updates"/pb.with_suffix(".pt").name,map_location="cpu",weights_only=False))
 if len(audit)!=100 or not all(x["MaskMatch"] and x["RetainedCountMatch"] for x in audit):raise RuntimeError("INVALID PAIRED RUN")
 for c in range(50):rp[c].sort();ep[c].sort(key=lambda x:x[0].round_id);rr[c].sort(key=lambda x:x.round);er[c].sort(key=lambda x:x.round)
 return rd,ed,rm,em,sc,rp,ep,rr,er,audit,explicit
def run():
 if os.environ.get("CONDA_DEFAULT_ENV")!="dsl1":raise RuntimeError("dsl1 required")
 out=B/"summary";out.mkdir(exist_ok=True);metrics=[];boots=[];pred=[];labs=[];thrs=[];audits=[];utility=[];comm=[]
 for seed in S:
  item=validate_run(F[seed],"metr_la",seed,"FedAvg");fed=old.load_cache(F[seed]);lf,lr,tr=labels(seed);rd,ed,rm,em,sc,rp,ep,rr,er,au,explicit=load(seed);labs+=lr;thrs+=tr;audits+=au
  v={x:[] for x in ("FedAvg Standard","RSR Standard","RSR Release-1 Values","RSR Full Adaptive","ED-OM-RSR Standard","ED-OM-RSR Release-1 Values","ED-OM-RSR Full Transform-Aware")}
  for c in range(50):
   std=lambda z:extract_feature_families([e11.attack_record(x) for x in z])["proposed_spectral"];fs,rs,es=std(fed[c]),std(rr[c]),std(er[c]);pp=[x for _,x in rp[c]];cc=tuple(x for x,_ in ep[c]);ee=tuple(x for _,x in ep[c]);r1=rsrvals(pp,1);rf=build_mask_aware_feature_vector(pp,rs);e1=edvals(ee,cc,sc,1);ef=full_transform_aware_features(es,ee,cc,sc)
   for k,x in {"FedAvg Standard":fs,"RSR Standard":rs,"RSR Release-1 Values":r1.values,"RSR Full Adaptive":rf.values,"ED-OM-RSR Standard":es,"ED-OM-RSR Release-1 Values":e1.values,"ED-OM-RSR Full Transform-Aware":ef.values}.items():v[k].append(x)
  v={k:np.stack(x) for k,x in v.items()}
  for prop in P:
   score,fold,want=lf[prop]
   for cond,x in v.items():
    lab,prob,guess,_=old.evaluate_fold_local(x,score,fold)
    if not np.array_equal(lab,want):raise RuntimeError("label leakage")
    m,b,pr,_=e13.evaluate(prop,cond,lab,fold,fixed=(prob,guess),origin=f"Experiment 20 seed{seed}");method="FedAvg" if cond.startswith("FedAvg") else ("RSR" if cond.startswith("RSR") else "ED-OM-RSR");m.update({"Seed":seed,"Method":method});b.update({"Seed":seed,"Method":method})
    for z in pr:z.update({"Seed":seed,"Method":method})
    metrics.append(m);boots.append(b);pred+=pr
  for method,p in (("FedAvg",F[seed]),("RSR",rd),("ED-OM-RSR",ed)):
   x=item["metrics"] if method=="FedAvg" else json.loads((p/"utility_metrics.json").read_text());utility.append({"Seed":seed,"Method":method,**{k:float(x[k]) for k in ("MAE","RMSE","MAPE")}})
  dense=sum(len(serialize_dense_update(x.layers)) for c in range(50) for x in fed[c]);comm += [{"Seed":seed,"Method":"FedAvg","Bytes":dense},{"Seed":seed,"Method":"RSR Explicit-index","Bytes":explicit},{"Seed":seed,"Method":"RSR","Bytes":rm["serialized_bytes"]},{"Seed":seed,"Method":"ED-OM-RSR","Bytes":em["serialized_bytes"]}]
 write_rows(B/"property_labels.csv",labs);write_rows(out/"property_thresholds.csv",thrs);write_rows(B/"attack_oof_predictions.csv",pred);write_rows(B/"attack_bootstrap_ci.csv",boots);write_rows(out/"attack_metrics.csv",metrics);write_rows(B/"mask_equivalence.csv",audits);write_rows(B/"retained_count_equivalence.csv",[{k:x[k] for k in ("Seed","Round","ClientID","ReleaseIndex","RSRRetained","EDRetained","RetainedCountMatch")} for x in audits]);write_rows(B/"per_seed_utility.csv",utility);write_rows(B/"per_seed_communication.csv",comm)
 print(json.dumps({"metrics":len(metrics),"predictions":len(pred),"pair_audits":len(audits)}))
if __name__=="__main__":run()

