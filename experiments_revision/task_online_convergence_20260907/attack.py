from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from experiments_revision.task_online_convergence_20260907 import train as t
p=t.p
PAIR_FAMILIES=('Release-1','Full','Energy','Other','Direction','FullPlus','DirectionAware','FullPlusAware','OtherAware','PublicMaps')
HISTORY_FAMILIES=('HistoryEnergy','HistoryOther','HistoryDirection','HistoryOtherAware','HistoryDirectionAware','HistoryPublicMaps')
def aware(ev,public):
 if public is None or public['alpha']==0:return ev,0.
 alpha=public['alpha'];maps=public['maps'];ri=ev['context'].release_index
 if alpha==1:
  rank=next(iter(maps.values()))['target'].shape[1]
  return p.aware_event(ev,{(ri,g):v for g,v in maps.items()},'target'+str(rank),'ED-OM-RSR')
 groups={}
 for g in p.b.GROUPS:
  x=p.b.flatten(ev['groups'][g]);m=maps[g];u=m['target']
  groups[g]=p.restore(x+alpha/(1-alpha)*u@(u.T@(x-m['mu'])),ev['groups'][g])
 layers=p.b.inverse_groups(groups,ev['rec'].layers)
 rec=p.b.UpdateRecord(ev['rec'].round,ev['rec'].client_id,layers,p.b.recnorm(layers))
 return {**ev,'groups':groups,'rec':rec},0.
def map_summary(public,ev):
 result=[float(public['alpha']) if public else 0.]
 for g in p.b.GROUPS:
  if public:
   mu=public['maps'][g]['mu'];u=public['maps'][g]['target']
   result.extend([float(np.linalg.norm(mu)),float(mu.mean()),float(mu.std()),float(u.shape[1]),float(np.abs(u).mean()) if u.size else 0.])
  else:result.extend([0.]*5)
 result.extend([float(ev['context'].round_id),float(ev['context'].release_index)])
 return np.asarray(result)
def pair_features(evs,maps,method,schema):
 mech='RSR' if method=='rsr' else 'ED-OM-RSR'
 result=p.b.features(evs,mech,schema);ae=[aware(e,m)[0] for e,m in zip(evs,maps)]
 af=p.b.features(ae,mech,schema)
 result.update(DirectionAware=af['Direction'],FullPlusAware=af['FullPlus'],OtherAware=af['Other'],PublicMaps=np.concatenate([map_summary(m,e) for m,e in zip(maps,evs)]))
 return result
def history_features(evs,maps):
 en=[];other=[];dire=[];ao=[];ad=[];meta=[]
 for ev,mp in zip(evs,maps):
  ae,_=aware(ev,mp)
  en.extend([np.log(p.b.norm(ev['groups'][g])**2+1e-20) for g in p.b.GROUPS])
  other.extend(p.b.flatten(ev['groups']['other']));ao.extend(p.b.flatten(ae['groups']['other']))
  for g in p.b.SPECTRAL_GROUPS:
   x=p.b.flatten(ev['groups'][g]);y=p.b.flatten(ae['groups'][g]);dire.extend(x/(np.linalg.norm(x)+1e-12));ad.extend(y/(np.linalg.norm(y)+1e-12))
  meta.extend(map_summary(mp,ev))
 return {k:np.asarray(v) for k,v in zip(HISTORY_FAMILIES,[en,other,dire,ao,ad,meta])}
def classifier(family,kind,n,d):
 parts=[StandardScaler()]
 if family in ['Other','Direction','DirectionAware','OtherAware','PublicMaps'] or family.startswith('History'):
  parts.append(PCA(n_components=min(8,n-1,d),svd_solver='full'))
 parts.append(LogisticRegression(C=1.,class_weight='balanced',max_iter=5000,random_state=9117) if kind=='LR' else SVC(C=1.,gamma='scale',class_weight='balanced',probability=True,random_state=9117))
 return make_pipeline(*parts)
def run_attack(dest,smoke=False):
 dest=Path(dest);split=json.loads((dest/'split.json').read_text());config=json.loads((dest/'config.json').read_text())
 ds=config['dataset'];seed=config['seed'];fold=int(dest.name.replace('fold',''))
 tr=split['AttackTrainClients'];te=split['TestClients'];used=tr+te
 assert not set(used)&set(split['DefenseClients']) and len(set(used))==30
 lab=pd.read_csv(t.ROOT/'experiments_revision/exp19_frozen_update/runs'/ds/f'seed{seed}'/'property_labels_and_folds.csv')
 labels=lab[lab.Property=='Periodicity Strength'].sort_values('ClientID').Label.to_numpy(int)
 out=dest/('attack_smoke' if smoke else 'attack');out.mkdir(exist_ok=False)
 predictions=[];metrics=[];audit=[];started=time.time()
 for method in t.METHODS:
  q=dest/method;events={cid:[] for cid in used};pub={cid:[] for cid in used}
  for f in sorted((q/'public_rounds').glob('*.pt')):
   batch=torch.load(f,map_location='cpu',weights_only=False)
   mf=q/'maps'/f.name;mp=torch.load(mf,map_location='cpu',weights_only=False) if mf.exists() else None
   if mp:assert set(mp['fit_clients'])==set(split['DefenseClients'])
   for cid,ev in batch.items():
    if cid in events:events[cid].append(ev);pub[cid].append(mp)
  counts={len(v) for v in events.values()};assert len(counts)==1
  count=counts.pop();assert count>=2 and (smoke or count>=40)
  schema=t.base.schema_from_state_dict(events[used[0]][0]['rec'].layers)
  budgets=['Early2','Late2','AllHistory'] if smoke else ['Early2','Late2','First40','AllHistory']
  for budget in budgets:
   matrices={}
   for cid in used:
    ix=[0,1] if budget=='Early2' else [count-2,count-1] if budget=='Late2' else list(range(min(count,40))) if budget=='First40' else list(range(count))
    ev=[events[cid][i] for i in ix];mp=[pub[cid][i] for i in ix]
    fs=pair_features(ev,mp,method,schema) if len(ix)==2 and budget in ['Early2','Late2'] else history_features(ev,mp)
    for f,v in fs.items():
     assert np.isfinite(v).all(),(method,budget,f,cid)
     matrices.setdefault(f,[]).append(v)
   for family,values in matrices.items():
    x=np.stack(values);assert x.shape[0]==30
    for kind in ['LR','RBF']:
     clf=classifier(family,kind,20,x.shape[1]);clf.fit(x[:20],labels[tr])
     prob=clf.predict_proba(x[20:])[:,1];pred=clf.predict(x[20:]);a=float(roc_auc_score(labels[te],prob))
     key=dict(Dataset=ds,Seed=seed,Fold=fold,Method=method,Budget=budget,Attack=family,Classifier=kind)
     metrics.append(dict(**key,RawAUC=a,S=max(a,1-a),BalancedAccuracy=float(balanced_accuracy_score(labels[te],pred)),Dimensions=x.shape[1],ReleaseCount=len(ix)))
     predictions.extend(dict(**key,ClientID=cid,Label=int(labels[cid]),Probability=float(pr),Predicted=int(pd_),ReleaseCount=len(ix)) for cid,pr,pd_ in zip(te,prob,pred))
     fitted=clf.named_steps['standardscaler'];assert np.allclose(fitted.mean_,x[:20].mean(axis=0))
    audit.append(dict(Method=method,Budget=budget,Attack=family,TrainClients=tr,TestClients=te,Dimensions=x.shape[1]))
   pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
   pd.DataFrame(metrics).to_csv(out/'fold_metrics.csv',index=False)
   print('ATTACK',ds,seed,fold,method,budget,flush=True)
 t.dump(out/'audit.json',dict(Status='SMOKE_COMPLETE' if smoke else 'COMPLETE',Rows=len(metrics),Predictions=len(predictions),Checks=audit,ElapsedSeconds=time.time()-started,ConvergenceStatus=json.loads((dest/'status.json').read_text())['Status'],RawUpdateAccess=False))
 if not smoke:
  status=json.loads((dest/'status.json').read_text());status['AttackStatus']='COMPLETE';t.dump(dest/'status.json',status)
