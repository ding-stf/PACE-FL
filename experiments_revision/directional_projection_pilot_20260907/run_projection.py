from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from experiments_revision.attack_defense_pilot_20260907 import run_pilot as b
from communication.exact_dimension_orthogonal_mixing import exact_orthogonal_mix,exact_orthogonal_inverse,TransformContext,GLOBAL_MIX_SEED
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score

ROOT=b.ROOT; ARCH=b.ARCH; OUT=Path(__file__).resolve().parent
CONDITIONS=('none','target1','target3','random1','random3')
MECHS=('RSR','ED-OM-RSR')
FAMILIES=(*b.FAMILIES,'DirectionAware','FullPlusAware')
torch.set_num_threads(1)
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
def clf(fam,kind):
 parts=[StandardScaler()]
 if fam in ('Other','Direction','DirectionAware'):parts.append(PCA(n_components=8,svd_solver='full'))
 parts.append(LogisticRegression(C=1.,class_weight='balanced',max_iter=5000,random_state=b.ATTACK_CV_SEED) if kind=='LR' else SVC(C=1.,gamma='scale',class_weight='balanced',probability=True,random_state=b.ATTACK_CV_SEED))
 return make_pipeline(*parts)
def restore(vec,template):
 result={};start=0
 for k in sorted(template):
  t=template[k];n=t.numel()
  result[k]=torch.from_numpy(np.array(vec[start:start+n],copy=True)).to(t.dtype).reshape(t.shape)
  start+=n
 assert start==len(vec)
 return result
def fit_map(x,y,seed):
 mu=x.mean(axis=0);cols=[];yc=y-y.mean()
 for rank in range(3):
  u=np.stack(cols,axis=1) if cols else np.zeros((x.shape[1],0))
  z=x-(x-mu)@u@u.T
  a=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12);a-=a.mean(axis=0)
  v=a.T@np.linalg.solve(a@a.T+.01*np.eye(len(a)),yc)
  v-=u@(u.T@v)
  assert np.linalg.norm(v)>1e-10,'degenerate property direction'
  cols.append(v/np.linalg.norm(v))
 u=np.stack(cols,axis=1)
 rand=np.linalg.qr(np.random.default_rng(seed).normal(size=(x.shape[1],3)))[0]
 assert np.max(np.abs(u.T@u-np.eye(3)))<1e-8
 return dict(mu=mu,target=u,random=rand)
def apply_map(raw,ri,maps,condition):
 if condition=='none':return raw.layers,0.,0.
 gs=b.transform_update(raw.layers);projected={};error=0.;energy=0.
 for g in b.GROUPS:
  v=b.flatten(gs[g]);m=maps[(ri,g)];u=m['target' if condition.startswith('target') else 'random'][:,:int(condition[-1])]
  z=v-u@(u.T@(v-m['mu']))
  assert np.max(np.abs(u.T@(z-m['mu'])))<1e-8
  assert np.max(np.abs(z-(z-u@(u.T@(z-m['mu'])))))<1e-8
  projected[g]=restore(z,gs[g]);error+=float(np.sum((z-v)**2));energy+=float(v@v)
 return b.inverse_groups(projected,raw.layers),error,energy
def event_from_release(raw,ri,layers,mech,schema,seed):
 kw=dict(p=.5,experiment_seed=seed,client_id=raw.client_id,round_id=raw.round,release_index=ri)
 release=b.randomized_spectral_release(layers,**kw) if mech=='RSR' else b.exact_dimension_rsr_release(layers,schema=schema,**kw)
 ctx=b.PublicMaskContext(seed,raw.client_id,raw.round,ri,.5)
 wire=b.encode_implicit_mask_payload(release.payload,schema,ctx) if mech=='RSR' else release.serialized
 rec=b.UpdateRecord(raw.round,raw.client_id,release.reconstructed_update,b.recnorm(release.reconstructed_update))
 return dict(payload=release.payload,rec=rec,groups=b.transform_update(rec.layers),context=b.PublicReleaseContext(raw.client_id,raw.round,ri),maskctx=ctx,bytes=len(wire))
def aware_event(e,maps,condition,mech):
 # This function receives released data and public maps only.
 if condition=='none':return e,0.
 ri=e['context'].release_index;rr=e['context'].round_id
 groups={g:dict(v) for g,v in e['groups'].items()};worst=0.
 for g in b.SPECTRAL_GROUPS:
  m=maps[(ri,g)];u=m['target' if condition.startswith('target') else 'random'][:,:int(condition[-1])]
  entries=sorted([q for q in e['payload'].entries if q.group==g],key=lambda q:q.name)
  mixed=[];observed=[]
  for q in entries:
   v=np.zeros(int(np.prod(q.shape)));mask=np.zeros(len(v),bool)
   idx=q.indices.numpy().reshape(-1);v[idx]=q.values.double().numpy().reshape(-1)*e['payload'].p;mask[idx]=True
   mixed.append(v);observed.append(mask)
  v=np.concatenate(mixed);obs=np.concatenate(observed)
  ctx=TransformContext(GLOBAL_MIX_SEED,rr,ri,g)
  w=u if mech=='RSR' else np.stack([exact_orthogonal_mix(torch.from_numpy(u[:,j].copy()),ctx)[0].numpy() for j in range(u.shape[1])],axis=1)
  a=w[~obs];rhs=u.T@m['mu']-w[obs].T@v[obs]
  fill=a@np.linalg.pinv(a.T@a,rcond=1e-10)@rhs
  vv=v.copy();vv[~obs]=fill
  assert np.array_equal(vv[obs],v[obs])
  worst=max(worst,float(np.max(np.abs(w.T@vv-u.T@m['mu']))))
  assert worst<1e-6,(g,worst)
  completed=vv if mech=='RSR' else exact_orthogonal_inverse(torch.from_numpy(vv.copy()),ctx).numpy()
  groups[g]=restore(completed,groups[g])
 layers=b.inverse_groups(groups,e['rec'].layers)
 rec=b.UpdateRecord(rr,e['context'].client_id,layers,b.recnorm(layers))
 return {**e,'groups':groups,'rec':rec},worst
def load_run(ds,seed):
 run=ARCH/ds/f'seed{seed}'
 schema=b.PublicRSRSchema.from_dict(json.loads((run/'public_schema.json').read_text()))
 lab=pd.read_csv(run/'property_labels_and_folds.csv');lab=lab[lab.Property=='Periodicity Strength'].sort_values('ClientID')
 y=lab.Label.to_numpy(int);folds=lab.Fold.to_numpy(int)
 raw=defaultdict(list)
 for p in sorted((run/'pre_release_dense_updates').glob('*.pt')):
  r=torch.load(p,map_location='cpu',weights_only=False);raw[r.client_id].append(r)
 events={m:defaultdict(list) for m in MECHS}
 for cid in range(50):
  raw[cid].sort(key=lambda r:r.round);assert len(raw[cid])==2
  for ri,r in enumerate(raw[cid]):
   name=f'round_{r.round:04d}_client_{cid:03d}';ctx=b.PublicMaskContext(seed,cid,r.round,ri,.5)
   for mech,sub,ext in [('RSR','rsr','imr'),('ED-OM-RSR','ed_om_rsr','edm')]:
    wire=(run/'payloads'/sub/(name+'.'+ext)).read_bytes()
    pp=b.decode_implicit_mask_payload(wire,schema,ctx) if mech=='RSR' else b.deserialize_ed_payload(wire,schema,ctx)
    rec=torch.load(run/'reconstructed_updates'/sub/(name+'.pt'),map_location='cpu',weights_only=False)
    events[mech][cid].append(dict(payload=pp,rec=rec,groups=b.transform_update(rec.layers),context=b.PublicReleaseContext(cid,r.round,ri),maskctx=ctx,bytes=len(wire)))
 return run,schema,y,folds,raw,events
def check_release(e,old):
 assert e['bytes']==old['bytes']
 previous={(c.group,c.name):c for c in old['payload'].entries}
 assert len(previous)==len(e['payload'].entries)
 for a in e['payload'].entries:
  c=previous[(a.group,a.name)]
  assert (a.group,a.name,a.shape,a.values.dtype,a.values.shape)==(c.group,c.name,c.shape,c.values.dtype,c.values.shape)
  assert (a.indices is None and c.indices is None) or torch.equal(a.indices,c.indices)
def main():
 assert os.getenv('CONDA_DEFAULT_ENV')=='dsl1' and Path.cwd()==ROOT
 assert not (OUT/'metrics.csv').exists()
 start=time.time();metrics=[];preds=[];splits=[];dist=[];audit=[];mapsizes=[]
 for ds in b.DATASETS:
  for seed in b.SEEDS:
   run,schema,y,folds,raw,events=load_run(ds,seed)
   man=json.loads((run/'manifest.json').read_text())
   for rel,h in man['output_hashes'].items():assert b.sha(run/rel)==h,rel
   matrices={m:{f:[] for f in FAMILIES} for m in MECHS}
   for mech in MECHS:
    for cid in range(50):
     fs=b.features(events[mech][cid],mech,schema);fs.update(DirectionAware=fs['Direction'],FullPlusAware=fs['FullPlus'])
     for f in FAMILIES:matrices[mech][f].append(fs[f])
    matrices[mech]={f:np.stack(v) for f,v in matrices[mech].items()}
   predictions={(c,m,f,k):np.full(50,np.nan) for c in CONDITIONS for m in MECHS for f in FAMILIES for k in ('LR','RBF')}
   classifications={key:np.zeros(50,int) for key in predictions}
   for fold in sorted(set(folds)):
    te=np.flatnonzero(folds==fold);other=np.flatnonzero(folds!=fold)
    ia,it=next(StratifiedShuffleSplit(n_splits=1,test_size=.5,random_state=9117+seed+int(fold)).split(other,y[other]))
    aux=other[ia];tr=other[it];used=np.concatenate([tr,te])
    assert len(aux)==20 and len(tr)==20 and len(te)==10 and len(set(aux)|set(tr)|set(te))==50
    split=dict(Dataset=ds,Seed=seed,Fold=int(fold),DefenseClients=aux.tolist(),AttackTrainClients=tr.tolist(),TestClients=te.tolist());splits.append(split)
    maps={}
    for ri in range(2):
     for gi,g in enumerate(b.GROUPS):
      x=np.stack([b.flatten(b.transform_update(raw[cid][ri].layers)[g]) for cid in aux])
      maps[(ri,g)]=fit_map(x,y[aux],133713+seed*31+int(fold)*17+ri*7+gi)
    mp=OUT/'public_maps'/ds/f'seed{seed}';mp.mkdir(parents=True,exist_ok=True)
    mapfile=mp/f'fold{fold}.pt'
    if mapfile.exists():
     saved=torch.load(mapfile,weights_only=False)
     assert saved['split']==split
     for kk in maps:
      for field in maps[kk]:assert np.array_equal(saved['maps'][kk][field],maps[kk][field])
    else:torch.save(dict(maps=maps,split=split),mapfile)
    mapsizes.append(dict(Dataset=ds,Seed=seed,Fold=int(fold),SavedMapBytes=mapfile.stat().st_size,Float64BytesAllRanks=sum((v['mu'].size+v['target'].size+v['random'].size)*8 for v in maps.values())))
    for condition in CONDITIONS:
     mats=matrices if condition=='none' else {m:{f:np.zeros((50,len(matrices[m][f][0]))) for f in FAMILIES} for m in MECHS}
     nchecks=0;awaremax=0.
     if condition!='none':
      for cid in used:
       evs={m:[] for m in MECHS};aware={m:[] for m in MECHS};err=0.;energy=0.
       for ri,r in enumerate(raw[cid]):
        layers,ee,en=apply_map(r,ri,maps,condition);err+=ee;energy+=en
        for mech in MECHS:
         e=event_from_release(r,ri,layers,mech,schema,seed);check_release(e,events[mech][cid][ri]);nchecks+=1
         evs[mech].append(e);ae,res=aware_event(e,maps,condition,mech);aware[mech].append(ae);awaremax=max(awaremax,res)
       if cid in te:dist.append(dict(Dataset=ds,Seed=seed,Fold=int(fold),Condition=condition,ClientID=int(cid),InputRelativeL2=float(np.sqrt(err/max(energy,1e-30)))))
       for mech in MECHS:
        fs=b.features(evs[mech],mech,schema);afs=b.features(aware[mech],mech,schema)
        fs.update(DirectionAware=afs['Direction'],FullPlusAware=afs['FullPlus'])
        for f in FAMILIES:mats[mech][f][cid]=fs[f]
     for mech in MECHS:
      for f in FAMILIES:
       x=mats[mech][f];assert np.isfinite(x).all()
       for kind in ('LR','RBF'):
        model=clf(f,kind).fit(x[tr],y[tr]);key=(condition,mech,f,kind)
        predictions[key][te]=model.predict_proba(x[te])[:,1];classifications[key][te]=model.predict(x[te])
     audit.append(dict(Dataset=ds,Seed=seed,Fold=int(fold),Condition=condition,ReleaseChecks=nchecks,AffineCompletionMaxError=awaremax,Pass=True))
    dump(OUT/'splits.json',splits)
    print('FOLD_DONE',ds,seed,int(fold),'seconds',round(time.time()-start,1),flush=True)
   for (condition,mech,f,kind),prob in predictions.items():
    assert np.isfinite(prob).all()
    auc,s=b.sep(y,prob)
    metrics.append(dict(Dataset=ds,Seed=seed,Condition=condition,Mechanism=mech,Attack=f,Classifier=kind,AUCRaw=auc,S=s,BalancedAccuracy=balanced_accuracy_score(y,classifications[(condition,mech,f,kind)])))
    for cid in range(50):
     preds.append(dict(Dataset=ds,Seed=seed,Condition=condition,Mechanism=mech,Attack=f,Classifier=kind,ClientID=cid,Fold=int(folds[cid]),Label=int(y[cid]),Probability=float(prob[cid])))
   pd.DataFrame(metrics).to_csv(OUT/'metrics_partial.csv',index=False)
   pd.DataFrame(preds).to_csv(OUT/'oof_predictions_partial.csv',index=False)
   pd.DataFrame(dist).to_csv(OUT/'update_distortion.csv',index=False)
   dump(OUT/'release_audit.json',audit);pd.DataFrame(mapsizes).to_csv(OUT/'public_map_sizes.csv',index=False)
   print('RUN_DONE',ds,seed,flush=True)
 pd.DataFrame(metrics).to_csv(OUT/'metrics.csv',index=False)
 pd.DataFrame(preds).to_csv(OUT/'oof_predictions.csv',index=False)
 dump(OUT/'privacy_manifest.json',dict(Status='COMPLETE',RuntimeSeconds=time.time()-start,MetricRows=len(metrics),PredictionRows=len(preds),ReleaseChecks=sum(r['ReleaseChecks'] for r in audit),ProtocolSHA256=b.sha(OUT/'PROTOCOL.md'),ScriptSHA256=b.sha(Path(__file__)),NewVictimTraining=False))
 print('PRIVACY_COMPLETE',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
