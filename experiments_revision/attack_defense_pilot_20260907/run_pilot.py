from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS","1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")
import json,hashlib,time,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from attacks.spectral_leakage.collector import UpdateRecord
from attacks.ed_om_rsr_attack import PublicReleaseContext,release_value_features,full_transform_aware_features
from communication.implicit_mask_rsr import PublicRSRSchema,PublicMaskContext,decode_implicit_mask_payload,encode_implicit_mask_payload
from communication.exact_dimension_orthogonal_mixing import deserialize_ed_payload,serialize_ed_payload,exact_dimension_rsr_release
from communication.randomized_spectral_release import RSRPayload,RSREntry,randomized_spectral_release,tensor_mapping_sha256,SPECTRAL_GROUPS
from spectral.spectral_groups import transform_update,inverse_groups,GROUPS
from experiments_revision.exp19_frozen_update.run_exp19 import rsr_full
from experiments.exp_randomized_spectral_release_sanity import attack_record
from experiments.exp_representative_privacy_attack import extract_feature_families,ATTACK_CV_SEED

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ARCH=ROOT/'experiments_revision/exp19_frozen_update/runs'
DATASETS=('pems08','metr_la'); SEEDS=(42,123,2026)
VARIANTS={'none':(None,False),'spectral_q90':(.9,False),'spectral_q50':(.5,False),'all_q50':(.5,True)}
FAMILIES=('Release-1','Full','Energy','Other','Direction','FullPlus')
def sha(p):
 return hashlib.sha256(p.read_bytes()).hexdigest()
def flatten(g):
 return torch.cat([g[k].reshape(-1) for k in sorted(g)]).double().numpy()
def norm(g):return float(np.linalg.norm(flatten(g)))
def recnorm(u):return float(torch.sqrt(sum(v.double().square().sum() for v in u.values())))
def sep(y,p):
 a=float(roc_auc_score(y,p));return a,max(a,1-a)
def write(p,obj):p.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def clf(fam,kind):
 parts=[StandardScaler()]
 if fam in ('Other','Direction'):parts.append(PCA(n_components=8,svd_solver='full'))
 if kind=='LR':parts.append(LogisticRegression(C=1.,class_weight='balanced',max_iter=5000,random_state=ATTACK_CV_SEED))
 else:parts.append(SVC(C=1.,gamma='scale',class_weight='balanced',probability=True,random_state=ATTACK_CV_SEED))
 return make_pipeline(*parts)
def features(evs,mech,schema):
 pay=[e['payload'] for e in evs];ctx=[e['context'] for e in evs]
 std=extract_feature_families([attack_record(e['rec']) for e in evs])['proposed_spectral']
 full=(rsr_full(std,pay,ctx,schema) if mech=='RSR' else full_transform_aware_features(std,pay,ctx,schema)).values
 r1=release_value_features(pay,ctx,schema,1).values
 energy=[];direction=[];other=[];ostat=[]
 for e in evs:
  # All attacker inputs below are transmitted values or public reconstructions.
  en=[]
  for g in GROUPS:
   en.append(np.log(norm(e['groups'][g])**2+1e-20))
  energy.append(np.array(en))
  for g in SPECTRAL_GROUPS:
   v=flatten(e['groups'][g]);direction.extend(v/(np.linalg.norm(v)+1e-12))
  other.extend(flatten(e['groups']['other']))
  row=[]
  for k in sorted(e['groups']['other']):
   v=e['groups']['other'][k].double().numpy().reshape(-1)
   row.extend([v.mean(),v.std(),np.abs(v).mean(),np.sqrt(np.mean(v*v))])
  ostat.append(np.array(row))
 en=np.concatenate([energy[0],energy[1],energy[1]-energy[0]])
 ot=np.concatenate([ostat[0],ostat[1],ostat[1]-ostat[0]])
 return {'Release-1':r1,'Full':full,'Energy':en,'Other':np.array(other),'Direction':np.array(direction),'FullPlus':np.concatenate([full,en,ot])}
def modified(e,scales):
 p=e['payload']
 pp=RSRPayload(p.p,tuple(RSREntry(x.group,x.name,x.shape,x.values*scales[x.group],x.indices) for x in p.entries))
 gs={g:{k:v*scales[g] for k,v in values.items()} for g,values in e['groups'].items()}
 rr=inverse_groups(gs,e['rec'].layers)
 rec=UpdateRecord(e['rec'].round,e['rec'].client_id,rr,recnorm(rr))
 return {**e,'payload':pp,'rec':rec,'groups':gs}
def check_direct(raw,e,scales,mech,schema,seed):
 groups=transform_update(raw.layers)
 clipped=inverse_groups({g:{k:v*scales[g] for k,v in vals.items()} for g,vals in groups.items()},raw.layers)
 ctx=e['context']
 fn=randomized_spectral_release if mech=='RSR' else exact_dimension_rsr_release
 kw=dict(p=.5,experiment_seed=seed,client_id=ctx.client_id,round_id=ctx.round_id,release_index=ctx.release_index)
 if mech!='RSR':kw['schema']=schema
 direct=fn(clipped,**kw)
 actual=e['payload']
 parts={(x.group,x.name):x for x in actual.entries}
 for x in direct.payload.entries:
  y=parts[(x.group,x.name)]
  assert (x.indices is None and y.indices is None) or torch.equal(x.indices,y.indices)
  assert torch.allclose(x.values,y.values,atol=2e-6,rtol=2e-5),(mech,x.group,x.name,'payload')
 for k,v in direct.reconstructed_update.items():
  assert torch.allclose(v,e['rec'].layers[k],atol=2e-6,rtol=2e-5),(mech,k,'reconstruction')
 return float(max((direct.reconstructed_update[k]-v).abs().max() for k,v in e['rec'].layers.items()))
def main():
 assert os.getenv('CONDA_DEFAULT_ENV')=='dsl1'
 if (OUT/'metrics.csv').exists():raise FileExistsError('results already exist')
 start=time.time();metrics=[];predictions=[];thresholds=[];dist=[];audits=[];metric_audit=[]
 hashes={'PROTOCOL.md':sha(OUT/'PROTOCOL.md'),'run_pilot.py':sha(Path(__file__))}
 for ds in DATASETS:
  for seed in SEEDS:
   run=ARCH/ds/f'seed{seed}';man=json.loads((run/'manifest.json').read_text());nverified=0
   for rel,h in man['output_hashes'].items():
    assert sha(run/rel)==h,(run,rel)
    nverified+=1
   hashes[str((run/'manifest.json').relative_to(ROOT))]=sha(run/'manifest.json')
   schema=PublicRSRSchema.from_dict(json.loads((run/'public_schema.json').read_text()))
   labels=pd.read_csv(run/'property_labels_and_folds.csv')
   lab=labels[labels.Property=='Periodicity Strength'].sort_values('ClientID')
   assert lab.ClientID.tolist()==list(range(50))
   y=lab.Label.to_numpy(int);folds=lab.Fold.to_numpy(int)
   oldpred=pd.read_csv(run/'attack_oof_predictions.csv')
   oldmet=pd.read_csv(run/'attack_metrics.csv')
   for (mech,att),q in oldpred[oldpred.Property=='Periodicity Strength'].groupby(['Mechanism','Attack']):
    a,s=sep(q.TrueLabel.to_numpy(),q.PositiveProbability.to_numpy())
    mm=oldmet[(oldmet.Property=='Periodicity Strength')&(oldmet.Mechanism==mech)&(oldmet.Attack==att)].iloc[0]
    assert abs(s-mm.ROCAUCSeparability)<1e-12
    metric_audit.append(dict(Dataset=ds,Seed=seed,Mechanism=mech,Attack=att,RecomputedAUC=a,RecomputedS=s,BalancedAccuracy=balanced_accuracy_score(q.TrueLabel,q.PredictedLabel),MatchesArchivedS=True))
   raw=defaultdict(list)
   for path in sorted((run/'pre_release_dense_updates').glob('*.pt')):
    r=torch.load(path,map_location='cpu',weights_only=False);raw[r.client_id].append(r)
   events={m:defaultdict(list) for m in ('RSR','ED-OM-RSR')}
   norms=np.zeros((50,2,5))
   for cid in range(50):
    raw[cid].sort(key=lambda z:z.round)
    assert len(raw[cid])==2
    for ri,r in enumerate(raw[cid]):
     gs=transform_update(r.layers);norms[cid,ri]=[norm(gs[g]) for g in GROUPS]
     ctx=PublicMaskContext(seed,cid,r.round,ri,.5)
     name=f'round_{r.round:04d}_client_{cid:03d}'
     for mech,sub,ext in [('RSR','rsr','imr'),('ED-OM-RSR','ed_om_rsr','edm')]:
      b=(run/'payloads'/sub/(name+'.'+ext)).read_bytes()
      pp=decode_implicit_mask_payload(b,schema,ctx) if mech=='RSR' else deserialize_ed_payload(b,schema,ctx)
      rr=torch.load(run/'reconstructed_updates'/sub/(name+'.pt'),map_location='cpu',weights_only=False)
      events[mech][cid].append(dict(payload=pp,rec=rr,groups=transform_update(rr.layers),context=PublicReleaseContext(cid,r.round,ri),maskctx=ctx,bytes=len(b)))
   baseline={m:{f:[] for f in FAMILIES} for m in events}
   for mech in events:
    for cid in range(50):
     ff=features(events[mech][cid],mech,schema)
     for fam in FAMILIES:baseline[mech][fam].append(ff[fam])
    baseline[mech]={f:np.stack(v) for f,v in baseline[mech].items()}
   # Exact fixed-LR replay is checked below against saved OOF probabilities.
   for variant,(quantile,allgroups) in VARIANTS.items():
    ppred={(m,f,c):np.zeros(50) for m in events for f in FAMILIES for c in ('LR','RBF')}
    plabel={k:np.zeros(50,dtype=int) for k in ppred}
    folds_sorted=sorted(set(folds))
    for fold in folds_sorted:
     tr=np.flatnonzero(folds!=fold);te=np.flatnonzero(folds==fold)
     assert not set(tr)&set(te)
     limits=np.full((2,5),np.inf) if quantile is None else np.quantile(norms[tr],quantile,axis=0)
     if not allgroups:limits[:,4]=np.inf
     for ri in range(2):
      for gi,g in enumerate(GROUPS):
       thresholds.append(dict(Dataset=ds,Seed=seed,Variant=variant,Fold=int(fold),ReleaseIndex=ri,Group=g,Threshold=float(limits[ri,gi]),TrainingClients=tr.tolist()))
     for mech in events:
      if quantile is None:fm=baseline[mech]
      else:
       mats={f:[] for f in FAMILIES}
       for cid in range(50):
        ev=[];inputerr=0.;out_err=0.;rawenergy=0.;clips=0
        for ri,e in enumerate(events[mech][cid]):
         factors=np.minimum(1.,limits[ri]/np.maximum(norms[cid,ri],1e-30))
         scales={g:float(factors[j]) for j,g in enumerate(GROUPS)}
         ee=modified(e,scales);ev.append(ee)
         wire=encode_implicit_mask_payload(ee['payload'],schema,e['maskctx']) if mech=='RSR' else serialize_ed_payload(ee['payload'],schema,e['maskctx'])
         assert len(wire)==e['bytes']
         for a,b in zip(e['payload'].entries,ee['payload'].entries):
          assert a.values.dtype==b.values.dtype and a.values.shape==b.values.shape
          assert (a.indices is None and b.indices is None) or torch.equal(a.indices,b.indices)
         if fold==folds_sorted[0] and cid==int(te[0]) and ri==0:
          err=check_direct(raw[cid][ri],ee,scales,mech,schema,seed)
          audits.append(dict(Dataset=ds,Seed=seed,Variant=variant,Mechanism=mech,DirectRawReleaseMaxError=err,Pass=True))
         rawenergy+=float(np.square(norms[cid,ri]).sum())
         inputerr+=float(np.sum(np.square((1-factors)*norms[cid,ri])))
         out_err+=float(sum((ee['rec'].layers[k].double()-v.double()).square().sum() for k,v in raw[cid][ri].layers.items()))
         clips+=int((factors<1).sum())
        ff=features(ev,mech,schema)
        for fam in FAMILIES:mats[fam].append(ff[fam])
        if cid in te:dist.append(dict(Dataset=ds,Seed=seed,Variant=variant,Mechanism=mech,ClientID=cid,InputRelativeL2=np.sqrt(inputerr/max(rawenergy,1e-30)),ReconstructionRelativeL2=np.sqrt(out_err/max(rawenergy,1e-30)),ClippedGroupReleases=clips))
       fm={f:np.stack(v) for f,v in mats.items()}
      for fam in FAMILIES:
       x=fm[fam];assert np.isfinite(x).all()
       for kind in ('LR','RBF'):
        model=clf(fam,kind);model.fit(x[tr],y[tr])
        ppred[(mech,fam,kind)][te]=model.predict_proba(x[te])[:,1]
        plabel[(mech,fam,kind)][te]=model.predict(x[te])
    for (mech,fam,kind),prob in ppred.items():
     a,s=sep(y,prob)
     if variant=='none' and kind=='LR' and fam in ('Full','Release-1'):
      q=oldpred[(oldpred.Property=='Periodicity Strength')&(oldpred.Mechanism==mech)&(oldpred.Attack==fam)].sort_values('ClientID')
      assert np.max(np.abs(q.PositiveProbability.to_numpy()-prob))<1e-10,(ds,seed,mech,fam,'baseline mismatch')
     metrics.append(dict(Dataset=ds,Seed=seed,Variant=variant,Mechanism=mech,Attack=fam,Classifier=kind,AUCRaw=a,S=s,BalancedAccuracy=balanced_accuracy_score(y,plabel[(mech,fam,kind)])))
     for cid in range(50):
      predictions.append(dict(Dataset=ds,Seed=seed,Variant=variant,Mechanism=mech,Attack=fam,Classifier=kind,ClientID=cid,Fold=int(folds[cid]),Label=int(y[cid]),Probability=float(prob[cid])))
    print('DONE',ds,seed,variant,'seconds',round(time.time()-start,1),flush=True)
    pd.DataFrame(metrics).to_csv(OUT/'metrics_partial.csv',index=False)
   print('RUN_VERIFIED',ds,seed,'files',nverified,flush=True)
 pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
 pd.DataFrame(predictions).to_csv(OUT/'oof_predictions.csv',index=False)
 pd.DataFrame(thresholds).to_csv(OUT/'fold_thresholds.csv',index=False)
 pd.DataFrame(dist).to_csv(OUT/'update_distortion.csv',index=False)
 pd.DataFrame(metric_audit).to_csv(OUT/'metric_definition_audit.csv',index=False)
 write(OUT/'direct_release_checks.json',audits)
 write(OUT/'manifest.json',dict(Status='COMPLETE',RuntimeSeconds=time.time()-start,ProtocolHash=hashes['PROTOCOL.md'],InputAndCodeHashes=hashes,MetricRows=len(metrics),PredictionRows=len(predictions),RawReleaseChecks=len(audits),NewVictimTraining=False,NoManuscriptChanges=True,Conda=os.getenv('CONDA_DEFAULT_ENV')))
 print('COMPLETE',OUT,round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
