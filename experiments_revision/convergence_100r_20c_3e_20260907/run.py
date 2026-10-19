from __future__ import annotations
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import json,time,hashlib,traceback,random
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import torch
from experiments_revision.exp19_frozen_update import run_exp19 as e
from experiments.exp_ellipsoidal_clipping_sanity import _evaluate_partitions
from communication.randomized_spectral_release import randomized_spectral_release,tensor_mapping_sha256
from communication.exact_dimension_orthogonal_mixing import exact_dimension_rsr_release
from communication.implicit_mask_rsr import schema_from_state_dict,PublicMaskContext,encode_implicit_mask_payload
from federated.aggregation import apply_update,weighted_average_updates
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
DATASETS=('pems08','metr_la'); SEEDS=(42,123,2026); METHODS=('rsr','ed_om_rsr')
def dump(p,d):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,default=str)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def schedule(seed):
 rng=np.random.default_rng(seed);rows=[]
 for block in range(20):
  perm=rng.permutation(50)
  rows.extend([[int(perm[(r*20+j)%50]) for j in range(20)] for r in range(5)])
 assert len(rows)==100 and all(len(set(r))==20 for r in rows)
 assert Counter(x for r in rows for x in r)==Counter({i:40 for i in range(50)})
 return rows
def evaluate_fixed(model,vals,device,c,prepared):
 state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
 try:return _evaluate_partitions(model,vals,device,c,prepared[-2],prepared[-1])
 finally:
  random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])
def run(ds,seed,method):
 dest=OUT/'runs'/ds/f'seed{seed}'/method
 dest.mkdir(parents=True,exist_ok=False)
 for n in ['raw_updates','server_visible_updates','payloads','checkpoints']:(dest/n).mkdir()
 c=dict(e._load_reference_config(e.buildcfg(ds)))
 c.update(dataset=ds,seed=seed,rounds=100,num_clients=50,clients_per_round=20,local_epochs=3,history=12,horizon=12,lr=.01)
 if ds=='metr_la':c.update(projection_ratio=.90,projection_seed=42)
 sched=schedule(seed)
 dump(dest/'config.json',dict(**c,release_method=method,p=.5,releases_per_client=40,stage='convergence_training',attack_evaluated=False))
 dump(dest/'participation_schedule.json',{'rounds_client_ids':sched})
 e.deterministic(seed);torch.set_num_threads(1);device=torch.device('cuda')
 prepared=e._prepare_data(c,ROOT/'data',ds)
 model,clients,vals,tests=e._build_clients_and_eval(c,prepared,device);model=model.to(device)
 dump(dest/'data_sizes.json',{'shape':list(prepared[0].values.shape),'clients':[{'id':cl.client_id,'nodes':int(cl.adjacency.shape[0]),'train_windows':len(cl.train_loader.dataset)} for cl in clients]})
 init=tensor_mapping_sha256(model.state_dict());torch.save(model.state_dict(),dest/'checkpoints/round_000.pt')
 counts=Counter();rows=[];releases=[];best=float('inf');best_round=0;start=time.time()
 initial=evaluate_fixed(model,vals,device,c,prepared)
 dump(dest/'initial_validation.json',initial)
 for rr,ids in enumerate(sched):
  ts=time.time();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};ups=[];weights=[];loss=0.;n=0;bytecount=0
  for cid in ids:
   # Identical local shuffle streams across paired methods, regardless of evaluation/mixing RNG use.
   e.deterministic(seed*100000+rr*50+cid)
   z=clients[cid].train(state,3,0.0,report_loss=True)
   assert not z.dp_applied
   assert all(torch.isfinite(v).all() for v in z.update.values())
   ri=counts[cid];schema=schema_from_state_dict(z.update)
   if method=='rsr':
    rel=randomized_spectral_release(z.update,p=.5,experiment_seed=seed,client_id=cid,round_id=rr,release_index=ri)
    rec=rel.reconstructed_update;payload=encode_implicit_mask_payload(rel.payload,schema,PublicMaskContext(seed,cid,rr,ri,.5))
   else:
    rel=exact_dimension_rsr_release(z.update,p=.5,experiment_seed=seed,client_id=cid,round_id=rr,release_index=ri,schema=schema)
    rec=rel.reconstructed_update;payload=rel.serialized
   assert all(torch.isfinite(v).all() for v in rec.values())
   stem=f'round_{rr:03d}_client_{cid:03d}'
   torch.save({'round':rr,'client_id':cid,'release_index':ri,'layers':z.update},dest/'raw_updates'/f'{stem}.pt')
   torch.save({'round':rr,'client_id':cid,'release_index':ri,'layers':rec},dest/'server_visible_updates'/f'{stem}.pt')
   (dest/'payloads'/f'{stem}.bin').write_bytes(payload)
   releases.append(dict(Round=rr+1,ClientID=cid,ReleaseIndex=ri,RawHash=tensor_mapping_sha256(z.update),ReleasedHash=tensor_mapping_sha256(rec),Bytes=len(payload)))
   ups.append(rec);weights.append(z.samples);loss+=z.client_train_loss*z.loss_samples;n+=z.loss_samples;bytecount+=len(payload);counts[cid]+=1
  model.load_state_dict(apply_update(state,weighted_average_updates(ups,weights)))
  val=evaluate_fixed(model,vals,device,c,prepared)
  assert np.isfinite(val['MAE'])
  row=dict(Round=rr+1,ClientTrainLoss=loss/n,Val_MAE=val['MAE'],Val_RMSE=val['RMSE'],Val_MAPE=val['MAPE'],Seconds=time.time()-ts,Bytes=bytecount)
  rows.append(row);pd.DataFrame(rows).to_csv(dest/'training_metrics.csv',index=False)
  pd.DataFrame(releases).to_csv(dest/'release_manifest.csv',index=False)
  torch.save(model.state_dict(),dest/'checkpoints'/f'round_{rr+1:03d}.pt')
  if val['MAE']<best:best=val['MAE'];best_round=rr+1;torch.save(model.state_dict(),dest/'best_model.pt')
  dump(OUT/'status.json',dict(Status='RUNNING',Dataset=ds,Seed=seed,Method=method,Round=rr+1,TotalRounds=100,ValidationMAE=val['MAE'],BestRound=best_round,BestValidationMAE=best,ElapsedSeconds=time.time()-start))
  print(ds,seed,method,json.dumps(row),flush=True)
 assert len(releases)==2000 and counts==Counter({i:40 for i in range(50)})
 torch.save(model.state_dict(),dest/'final_model.pt')
 finaltest=evaluate_fixed(model,tests,device,c,prepared)
 model.load_state_dict(torch.load(dest/'best_model.pt',map_location=device,weights_only=True))
 besttest=evaluate_fixed(model,tests,device,c,prepared)
 dump(dest/'result.json',dict(Status='COMPLETE',Dataset=ds,Seed=seed,Method=method,InitialModelHash=init,InitialValidation=initial,FinalValidation=val,FinalTest=finaltest,BestRound=best_round,BestValidationMAE=best,BestTest=besttest,ReleaseCounts=dict(counts),ElapsedSeconds=time.time()-start,AttackEvaluated=False))
 dump(dest/'output_hashes.json',{str(f.relative_to(dest)):sha(f) for f in dest.rglob('*') if f.is_file()})
 return json.loads((dest/'result.json').read_text())
def main():
 assert Path.cwd()==ROOT and os.environ['CONDA_DEFAULT_ENV']=='dsl1'
 assert torch.cuda.is_available()
 source_paths=[Path(__file__),ROOT/'federated/aggregation.py',ROOT/'communication/randomized_spectral_release.py',ROOT/'communication/exact_dimension_orthogonal_mixing.py',ROOT/'experiments/exp_ellipsoidal_clipping_sanity.py']
 dump(OUT/'source_hashes.json',{str(f):sha(f) for f in source_paths})
 results=[]
 for ds in DATASETS:
  for seed in SEEDS:
   for method in METHODS:
    results.append(run(ds,seed,method));dump(OUT/'completed_results.json',results)
 dump(OUT/'status.json',dict(Status='COMPLETE',Runs=len(results),AttackEvaluated=False))
if __name__=='__main__':
 try:main()
 except BaseException:
  dump(OUT/'status.json',dict(Status='FAILED',Error=traceback.format_exc()));raise
