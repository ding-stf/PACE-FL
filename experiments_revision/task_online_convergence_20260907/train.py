from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse,json,time,traceback,random,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from experiments_revision.convergence_100r_20c_3e_20260907 import run as base
from experiments_revision.directional_projection_pilot_20260907 import run_projection as p
from datasets.dynamic_graph import batch_dynamic_adjacency
from federated.evaluation import evaluate
from federated.metrics import ForecastingTotals
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
METHODS=('rsr','ed_om_rsr','plain1','task1','task3','random1')
DEFENSES=METHODS[2:]
def dump(path,data):base.dump(path,data)
def schedule(seed,n=500):
 rng=np.random.default_rng(seed);out=[]
 for block in range(n//5):
  perm=rng.permutation(50)
  out.extend([[int(perm[(r*20+j)%50]) for j in range(20)] for r in range(5)])
 assert all(len(set(row))==20 for row in out)
 assert Counter(x for row in out for x in row)==Counter({i:n*20//50 for i in range(50)})
 assert out[:100]==base.schedule(seed)
 return out
def lr_at(rr):
 return .01 if rr<40 else .003 if rr<80 else .001 if rr<120 else .0003 if rr<200 else .0001
def plateau(values):
 a=np.asarray(values,dtype=float)
 if len(a)<100:return {'Pass':False,'Reason':'minimum100'}
 older=a[-20:-10];recent=a[-10:]
 drift=float(abs(recent.mean()/older.mean()-1));cv=float(recent.std()/recent.mean())
 improve=float(max(0.,1-a[-20:].min()/a[:-20].min()))
 return dict(Pass=bool(drift<=.005 and cv<=.02 and improve<=.001),MeanDrift=drift,CV=cv,BestImprovement=improve)
def snapshot(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def make_raw(layers,cid,rr):return p.b.UpdateRecord(rr,cid,layers,p.b.recnorm(layers))
def event(layers,cid,rr,ri,seed,mech,schema):
 return p.event_from_release(make_raw(layers,cid,rr),ri,layers,mech,schema,seed)
def project(layers,maps,alpha):
 if alpha==0:return layers
 groups=p.b.transform_update(layers);out={}
 for g in p.b.GROUPS:
  x=p.b.flatten(groups[g]);m=maps[g];u=m['target'];out[g]=p.restore(x-alpha*u@(u.T@(x-m['mu'])),groups[g])
 return p.b.inverse_groups(out,layers)
def aux_loss(env,state,gradient=False):
 model=env['model'];model.load_state_dict(state);model.eval();model.zero_grad(set_to_none=True)
 terms=[];n=0
 with torch.set_grad_enabled(gradient):
  for x,y,valid,graph in env['batches']:
   pred=model(x,graph);mask=valid.bool()&torch.isfinite(y)&torch.isfinite(pred)
   terms.append((pred[mask]-y[mask]).abs().sum());n+=int(mask.sum())
  assert n>0;loss=sum(terms)/n
  if gradient:loss.backward()
 grads={k:v.grad.detach().cpu().clone() for k,v in model.named_parameters()} if gradient else None
 return float(loss.detach())*env['std'],grads
def calibrate(env,state,raw,counts,rr,seed,method,history):
 aux=env['aux'];schema=env['schema'];weights=env['weights']
 def aggregate(maps=None,alpha=0.):
  ups=[event(project(raw[c].update,maps,alpha),c,rr,counts[c],seed,'ED-OM-RSR',schema)['rec'].layers for c in aux]
  return base.apply_update(state,base.weighted_average_updates(ups,[weights[c] for c in aux]))
 auxstate=aggregate();baseline,gradient=aux_loss(env,auxstate,method.startswith('task'))
 if gradient is not None:history=(history+[gradient])[-3:]
 groups={c:p.b.transform_update(raw[c].update) for c in aux};maps={};disabled=False
 for gi,g in enumerate(p.b.GROUPS):
  x=np.stack([p.b.flatten(groups[c][g]) for c in aux])
  try:maps[g]=p.fit_map(x,env['labels'][aux],seed*100003+rr*101+gi)
  except AssertionError:
   maps[g]=dict(mu=x.mean(0),target=np.zeros((x.shape[1],0)),random=np.zeros((x.shape[1],0)));disabled=True
 residual=0.;protected={}
 if method.startswith('task'):
  pulls={g:[] for g in p.b.GROUPS}
  for gr in history:
   for cid in aux:
    ev=event(gr,cid,rr,counts[cid],seed,'ED-OM-RSR',schema)
    for g in p.b.GROUPS:pulls[g].append(p.b.flatten(ev['groups'][g]))
  for g in p.b.GROUPS:
   a=np.stack(pulls[g]);a/=np.linalg.norm(a,axis=1,keepdims=True)+1e-20
   _,sv,vt=np.linalg.svd(a,full_matrices=False);rank=int((sv>sv[0]*1e-6).sum()) if sv[0]>0 else 0
   basis=vt[:rank].T;v=maps[g]['target']-basis@(basis.T@maps[g]['target']);cols=[]
   for j in range(v.shape[1]):
    z=v[:,j].copy()
    for q in cols:z-=q*float(q@z)
    if np.linalg.norm(z)>1e-8:cols.append(z/np.linalg.norm(z))
   maps[g]['target']=np.stack(cols,axis=1) if cols else np.zeros((len(maps[g]['mu']),0))
   if rank and cols:residual=max(residual,float(np.max(np.abs(basis.T@maps[g]['target']))))
   protected[g]=rank
 rank=3 if method=='task3' else 1
 maps={g:dict(mu=m['mu'],target=m['random' if method=='random1' else 'target'][:,:rank]) for g,m in maps.items()}
 disabled=disabled or any(m['target'].shape[1]!=rank for m in maps.values())
 trials=[];alpha=0.
 if not disabled:
  for candidate in [1.,.5,.25,0.]:
   loss,_=aux_loss(env,aggregate(maps,candidate));ratio=loss/baseline
   trials.append(dict(Alpha=candidate,MAE=loss,Ratio=ratio))
   if ratio<=1.01+1e-9:alpha=candidate;break
  assert trials[-1]['Ratio']<=1.01+1e-9
 assert residual<1e-5
 return maps,alpha,history,dict(Disabled=disabled,Trials=trials,ProtectedRank=protected,OrthogonalityError=residual,AuxiliaryBaselineMAE=baseline)
def evaluate_counts(model,partitions,device,c,prepared):
 totals=ForecastingTotals()
 for dl,adj in partitions:
  totals.merge(evaluate(model,dl,adj,device,float(prepared[-2][...,0].mean()),float(prepared[-1][...,0].mean()),bool(c['dynamic_graph']),float(c['dynamic_alpha']),int(c['dynamic_top_k'])))
 return dict(**totals.compute(),AbsError=totals.absolute_error,ValidCount=totals.valid_count)
def run(ds,seed,fold,smoke=False):
 dest=OUT/('smoke' if smoke else 'runs')/ds/f'seed{seed}'/f'fold{fold}';dest.mkdir(parents=True,exist_ok=False)
 splitfile=ROOT/'experiments_revision/directional_projection_pilot_20260907/public_maps'/ds/f'seed{seed}'/f'fold{fold}.pt'
 split=torch.load(splitfile,map_location='cpu',weights_only=False)['split']
 aux=split['DefenseClients'];tr=split['AttackTrainClients'];te=split['TestClients']
 assert len(aux)==20 and len(tr)==20 and len(te)==10 and len(set(aux+tr+te))==50
 lab=pd.read_csv(ROOT/'experiments_revision/exp19_frozen_update/runs'/ds/f'seed{seed}'/'property_labels_and_folds.csv')
 labels=lab[lab.Property=='Periodicity Strength'].sort_values('ClientID').Label.to_numpy(int)
 assert len(np.unique(labels[aux]))==2 and len(np.unique(labels[tr]))==2
 dump(dest/'split.json',split)
 c=dict(base.e._load_reference_config(base.e.buildcfg(ds)));c.update(dataset=ds,seed=seed,rounds=500,num_clients=50,clients_per_round=20,local_epochs=3,lr=.01)
 if ds=='metr_la':c.update(projection_ratio=.90,projection_seed=42)
 dump(dest/'config.json',dict(**c,MinRounds=100,MaxRounds=500,Methods=METHODS,Smoke=smoke,SchedulePolicy='balanced_five_round_blocks'))
 base.e.deterministic(seed);torch.set_num_threads(1);device=torch.device('cuda')
 prepared=base.e._prepare_data(c,ROOT/'data',ds)
 model,clients,vals,tests=base.e._build_clients_and_eval(c,prepared,device);model=model.to(device)
 sched=schedule(seed);dump(dest/'participation_schedule.json',{'rounds_client_ids':sched})
 if smoke:
  for cl in clients:cl.train_loader=DataLoader(Subset(cl.train_loader.dataset,list(range(64))),batch_size=64,shuffle=True)
  vals=[(DataLoader(Subset(dl.dataset,list(range(8))),batch_size=8),adj) for dl,adj in vals]
 batches=[]
 for cid in aux:
  dl,adj=vals[cid];ix=np.unique(np.linspace(0,len(dl.dataset)-1,8,dtype=int)).tolist()
  x,y,v=next(iter(DataLoader(Subset(dl.dataset,ix),batch_size=8)))
  x=x.to(device);y=y.to(device);v=v.to(device);adj=adj.to(device)
  graph=batch_dynamic_adjacency(adj,x,float(c['dynamic_alpha']),int(c['dynamic_top_k'])) if c['dynamic_graph'] else adj
  batches.append((x,y,v,graph))
 weights={cl.client_id:len(cl.train_loader.dataset)*int(cl.adjacency.shape[0]) for cl in clients}
 env=dict(model=model,aux=aux,labels=labels,batches=batches,std=float(prepared[-1][...,0].mean()),schema=base.schema_from_state_dict(model.state_dict()),weights=weights)
 init=snapshot(model);states={m:{k:v.clone() for k,v in init.items()} for m in METHODS}
 dump(dest/'initial_hash.json',{'Hash':base.tensor_mapping_sha256(init)})
 curves={m:[] for m in METHODS};history={m:[] for m in DEFENSES};best={m:float('inf') for m in METHODS};counts=Counter();best_round={m:0 for m in METHODS}
 for m in METHODS:
  q=dest/m;q.mkdir();(q/'public_rounds').mkdir();(q/'raw_rounds').mkdir();(q/'maps').mkdir();(q/'checkpoints').mkdir()
 dump(dest/'status.json',dict(Status='RUNNING',Round=0,Method='initializing'))
 converged=False;checks={};last_change=0;start=time.time()
 for rr,ids in enumerate(sched[:5] if smoke else sched):
  if rr and lr_at(rr)!=lr_at(rr-1):last_change=rr
  for m in METHODS:
   ts=time.time();q=dest/m;state=states[m];calibration=m in DEFENSES;raw={};loss=0.;n=0
   for cid in sorted(set(ids)|set(aux) if calibration else set(ids)):
    base.e.deterministic(seed*100000+rr*50+cid);clients[cid].lr=lr_at(rr)
    z=clients[cid].train(state,3,0.,report_loss=True);assert not z.dp_applied
    assert all(torch.isfinite(v).all() for v in z.update.values());raw[cid]=z
    if cid in ids:loss+=z.client_train_loss*z.loss_samples;n+=z.loss_samples
   caltime=time.time();maps=None;alpha=0.;audit={}
   if calibration:maps,alpha,history[m],audit=calibrate(env,state,raw,counts,rr,seed,m,history[m])
   calibration_seconds=time.time()-caltime;evs={};ups=[];transform_start=time.time()
   for cid in ids:
    ev=event(project(raw[cid].update,maps,alpha),cid,rr,counts[cid],seed,'RSR' if m=='rsr' else 'ED-OM-RSR',env['schema'])
    evs[cid]=ev;ups.append(ev['rec'].layers)
   transform_seconds=time.time()-transform_start
   new=base.apply_update(state,base.weighted_average_updates(ups,[raw[cid].samples for cid in ids]));states[m]=new
   assert all(torch.isfinite(v).all() for v in new.values())
   model.load_state_dict(new)
   # Validation timing and sample access cannot change paired training RNG streams.
   fullval=base.evaluate_fixed(model,vals,device,c,prepared)
   auxval=base.evaluate_fixed(model,[vals[cid] for cid in aux],device,c,prepared)
   heldval=base.evaluate_fixed(model,[vals[cid] for cid in te],device,c,prepared)
   row=dict(Round=rr+1,LR=lr_at(rr),TrainLoss=loss/n,AuxValMAE=auxval['MAE'],FullValMAE=fullval['MAE'],HeldoutValMAE=heldval['MAE'],Alpha=alpha,CalibrationSeconds=calibration_seconds,TransformSeconds=transform_seconds,LocalClientTrainings=len(raw),SelectedClients=len(ids),Seconds=time.time()-ts,PublicCoefficientBytes=sum(a.nbytes for mp in maps.values() for a in mp.values())+8 if maps else 0)
   assert all(np.isfinite(row[k]) for k in ['TrainLoss','AuxValMAE','FullValMAE','HeldoutValMAE'])
   curves[m].append(row);pd.DataFrame(curves[m]).to_csv(q/'training_metrics.csv',index=False)
   torch.save(evs,q/'public_rounds'/f'round_{rr:03d}.pt')
   torch.save({cid:z.update for cid,z in raw.items()},q/'raw_rounds'/f'round_{rr:03d}.pt')
   torch.save(new,q/'checkpoints'/f'round_{rr+1:03d}.pt')
   if maps:torch.save(dict(maps=maps,alpha=alpha,audit=audit,fit_clients=aux),q/'maps'/f'round_{rr:03d}.pt')
   if auxval['MAE']<best[m]:best[m]=auxval['MAE'];best_round[m]=rr+1;torch.save(new,q/'best_model.pt')
   dump(dest/'status.json',dict(Status='RUNNING',Round=rr+1,Method=m,ElapsedSeconds=time.time()-start,Latest=row))
   print(ds,seed,fold,m,json.dumps(row),flush=True)
  counts.update(ids)
  dump(dest/'round_complete.json',dict(Round=rr+1,ReleaseCounts=dict(counts)))
  if rr+1>=100 and (rr+1)%10==0 and rr+1-last_change>=20:
   checks={m:plateau([r['AuxValMAE'] for r in curves[m]]) for m in METHODS}
   dump(dest/'convergence_checks.json',dict(Round=rr+1,Methods=checks))
   if all(v['Pass'] for v in checks.values()):converged=True;break
 end=rr+1
 assert counts==Counter({i:end*20//50 for i in range(50)}) if end%5==0 else smoke
 results=[]
 if not smoke:
  for m in METHODS:
   q=dest/m;model.load_state_dict(states[m]);final=evaluate_counts(model,[tests[cid] for cid in te],device,c,prepared)
   model.load_state_dict(torch.load(q/'best_model.pt',weights_only=True,map_location=device));bst=evaluate_counts(model,[tests[cid] for cid in te],device,c,prepared)
   results.append(dict(Method=m,FinalHeldoutTest=final,BestHeldoutTest=bst,BestRound=best_round[m],FinalHeldoutValMAE=curves[m][-1]['HeldoutValMAE'],BestAuxValMAE=best[m],FinalRound=end))
  dump(dest/'utility_results.json',results)
 dump(dest/'status.json',dict(Status='SMOKE_COMPLETE' if smoke else 'CONVERGED' if converged else 'NOT_CONVERGED',Round=end,ReleaseCounts=dict(counts),Checks=checks,ElapsedSeconds=time.time()-start,AttackStatus='PENDING'))
 return dest
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--dataset',choices=['pems08','metr_la'],required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--fold',type=int,required=True);a.add_argument('--smoke',action='store_true');args=a.parse_args()
 assert Path.cwd()==ROOT and os.environ['CONDA_DEFAULT_ENV']=='dsl1'
 try:
  dest=run(args.dataset,args.seed,args.fold,args.smoke)
  if not args.smoke:
   from experiments_revision.task_online_convergence_20260907.attack import run_attack
   run_attack(dest)
 except BaseException:
  q=OUT/('smoke' if args.smoke else 'runs')/args.dataset/f'seed{args.seed}'/f'fold{args.fold}';q.mkdir(parents=True,exist_ok=True)
  dump(q/'failure.json',dict(Status='FAILED',Error=traceback.format_exc()));raise
