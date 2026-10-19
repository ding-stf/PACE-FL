from __future__ import annotations
"""Experiment 13: replay-only mask-aware adaptive attack against archived RSR."""
from collections import defaultdict
from pathlib import Path
import csv, hashlib, json, os, platform, shutil, statistics, sys, time
import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from attacks.mask_aware_spectral_attack import (
    ADAPTIVE_DIMENSION, adaptive_feature_provenance,
    build_mask_aware_feature_vector, compute_mask_only_features,
)
from communication.randomized_spectral_release import (
    deserialize_rsr_payload, reconstruct_rsr_update, tensor_mapping_sha256,
)
from experiments.exp_representative_privacy_attack import (
    ATTACK_CV_SEED, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
    auc_separability, fit_oof_predictions,
)

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/"results/randomized_spectral_release_sanity/pems08/seed42/p050"
FEDAVG=ROOT/"results/privacy_attack_locked_seed/seed42/attack_results/fedavg_r2"
OUTPUT=ROOT/"results/mask_aware_adaptive_attack/pems08/seed42/p050"
REPORT=ROOT/"docs/MASK_AWARE_ADAPTIVE_ATTACK_REPORT.md"
SPEC=ROOT/"docs/MASK_AWARE_ATTACK_FEATURE_SPEC.md"
SPEC_HASH=ROOT/"docs/MASK_AWARE_ATTACK_FEATURE_SPEC.sha256"
EXPECTED_SPEC_HASH="d88043bb4199e664f3a99f623531126dd066309b3a0f6d4cbf6d8b24ac21e259"
PROPERTIES=("Extreme Event","Periodicity Strength")
P=0.5

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1048576),b""): h.update(block)
    return h.hexdigest()

def rows(path):
    with path.open(newline="",encoding="utf-8") as f: return list(csv.DictReader(f))

def write_csv(path,data):
    if not data: raise ValueError(f"empty CSV {path}")
    with path.open("w",newline="",encoding="utf-8") as f:
        fieldnames=list(dict.fromkeys(key for row in data for key in row))
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(data)

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,sort_keys=True)+"\n",encoding="utf-8")

def labels_folds():
    table=rows(ARCHIVE/"attack/property_labels.csv")
    if len(table)!=100: raise RuntimeError("EXPERIMENT 13 INVALID: label count")
    out={}
    for prop in PROPERTIES:
        selected=sorted((r for r in table if r["Property"]==prop),key=lambda r:int(r["ClientID"]))
        clients=[int(r["ClientID"]) for r in selected]
        if clients!=list(range(50)): raise RuntimeError("EXPERIMENT 13 INVALID: client alignment")
        y=np.array([int(r["Label"]) for r in selected]); folds=np.array([int(r["Fold"]) for r in selected])
        fold_file=ARCHIVE/"attack/fold_assignments"/f"{prop.replace(' ','_')}_folds.csv"
        check={int(r["ClientID"]):int(r["Fold"]) for r in rows(fold_file)}
        if check!={i:int(folds[i]) for i in range(50)}: raise RuntimeError("EXPERIMENT 13 INVALID: folds")
        out[prop]=(y,folds)
    return out

def standard_features():
    table=rows(ARCHIVE/"attack/attack_features.csv")
    out={int(r["ClientID"]):np.array([float(r[f"proposed_spectral_{i:03d}"]) for i in range(24)]) for r in table}
    if set(out)!=set(range(50)): raise RuntimeError("EXPERIMENT 13 INVALID: standard features")
    return out

def verify_inputs():
    manifest=json.loads((ARCHIVE/"manifest.json").read_text())
    locked=(manifest["dataset"]=="PEMS08" and manifest["method"]=="RSR-FedAvg" and
            float(manifest["p"])==P and manifest["clients"]==50 and
            manifest["rounds"]==20 and manifest["releases_per_client"]==2)
    if not locked: raise RuntimeError("EXPERIMENT 13 INVALID: manifest lock")
    if sha(ARCHIVE/"attack/attack_features.csv")!=manifest["attack_feature_hash"]: raise RuntimeError("EXPERIMENT 13 INVALID: feature hash")
    if sha(ARCHIVE/"attack/attack_metrics.csv")!=manifest["attack_metrics_hash"]: raise RuntimeError("EXPERIMENT 13 INVALID: metric hash")
    doc=json.loads((ARCHIVE/"server_visible_update_hashes.json").read_text())
    expected={(int(r["Round"]),int(r["ClientID"])):r for r in doc["records"]}
    payloads=sorted((ARCHIVE/"server_visible_payloads").glob("*.rsr"))
    updates=sorted((ARCHIVE/"server_visible_updates").glob("*.pt"))
    if len(expected)!=100 or len(payloads)!=100 or len(updates)!=100: raise RuntimeError("EXPERIMENT 13 INVALID: archive count")
    out=[]; by_client=defaultdict(list)
    for path in payloads:
        p=path.stem.split("_"); key=(int(p[1]),int(p[3])); exp=expected[key]
        data=path.read_bytes(); payload=deserialize_rsr_payload(data)
        ph=hashlib.sha256(data).hexdigest(); uh=tensor_mapping_sha256(reconstruct_rsr_update(payload))
        okp=ph==exp["PayloadSHA256"]; oku=uh==exp["ServerReconstructedUpdateSHA256"]
        out.append({"round":key[0],"client":key[1],"release":int(exp["ReleaseIndex"]),
            "payload_sha256_expected":exp["PayloadSHA256"],"payload_sha256_actual":ph,"payload_hash_pass":okp,
            "server_update_sha256_expected":exp["ServerReconstructedUpdateSHA256"],
            "reconstructed_update_sha256_actual":uh,"reconstructed_update_hash_pass":oku,
            "saved_server_update_hash_pass":False})
        by_client[key[1]].append((key[0],payload))
    amap={(r["round"],r["client"]):r for r in out}
    for path in updates:
        rec=torch.load(path,map_location="cpu",weights_only=False); key=(int(rec.round),int(rec.client_id))
        amap[key]["saved_server_update_hash_pass"]=(tensor_mapping_sha256(rec.layers)==expected[key]["ServerReconstructedUpdateSHA256"])
    if not all(r["payload_hash_pass"] and r["reconstructed_update_hash_pass"] and r["saved_server_update_hash_pass"] for r in out):
        raise RuntimeError("EXPERIMENT 13 INVALID: hash gate")
    if set(by_client)!=set(range(50)) or any(len(v)!=2 for v in by_client.values()): raise RuntimeError("EXPERIMENT 13 INVALID: R2 observations")
    for value in by_client.values(): value.sort(key=lambda x:x[0])
    return sorted(out,key=lambda r:(r["round"],r["client"])),dict(by_client)

def standard_replay(features,lf):
    oldpred=rows(ARCHIVE/"attack/attack_oof_predictions.csv")
    oldmetric=rows(ARCHIVE/"attack/attack_metrics.csv"); out={}
    x=np.stack([features[i] for i in range(50)])
    for prop in PROPERTIES:
        y,folds=lf[prop]; prob,pred=fit_oof_predictions(x,y,folds,"proposed_spectral")
        old=sorted((r for r in oldpred if r["Property"]==prop and r["AttackFeature"]=="proposed_spectral"),key=lambda r:int(r["ClientID"]))
        op=np.array([float(r["PositiveProbability"]) for r in old]); od=np.array([int(r["PredictedLabel"]) for r in old])
        metric=next(r for r in oldmetric if r["Property"]==prop and r["AttackFeature"]=="proposed_spectral")
        if not np.array_equal(prob,op) or not np.array_equal(pred,od) or float(roc_auc_score(y,prob))!=float(metric["ROCAUCRaw"]):
            raise RuntimeError("EXPERIMENT 13 INVALID: standard replay")
        out[prop]=(prob,pred)
    return out

def boot(y,prob):
    rng=np.random.default_rng(BOOTSTRAP_SEED); raw=[]; sep=[]; n=len(y)
    for _ in range(BOOTSTRAP_REPLICATES):
        idx=rng.integers(0,n,size=n); yy=y[idx]
        if len(np.unique(yy))<2: continue
        auc=float(roc_auc_score(yy,prob[idx])); raw.append(auc); sep.append(auc_separability(auc))
    rci=np.percentile(raw,[2.5,97.5]); sci=np.percentile(sep,[2.5,97.5])
    return {"bootstrap_raw_auc_mean":float(np.mean(raw)),"bootstrap_auc_sep_mean":float(np.mean(sep)),
        "raw_auc_ci95_low":float(rci[0]),"raw_auc_ci95_high":float(rci[1]),
        "auc_sep_ci95_low":float(sci[0]),"auc_sep_ci95_high":float(sci[1]),
        "valid_bootstrap_replicates":len(raw),"requested_bootstrap_replicates":BOOTSTRAP_REPLICATES,
        "bootstrap_seed":BOOTSTRAP_SEED,"bootstrap_unit":"client"}

def paired_boot(y,adaptive,standard):
    rng=np.random.default_rng(BOOTSTRAP_SEED); vals=[]; n=len(y)
    for _ in range(BOOTSTRAP_REPLICATES):
        idx=rng.integers(0,n,size=n); yy=y[idx]
        if len(np.unique(yy))<2: continue
        vals.append(auc_separability(roc_auc_score(yy,adaptive[idx]))-auc_separability(roc_auc_score(yy,standard[idx])))
    ci=np.percentile(vals,[2.5,97.5])
    return {"paired_delta_mean":float(np.mean(vals)),"paired_delta_ci95_low":float(ci[0]),
        "paired_delta_ci95_high":float(ci[1]),"valid_bootstrap_replicates":len(vals),
        "requested_bootstrap_replicates":BOOTSTRAP_REPLICATES,"bootstrap_seed":BOOTSTRAP_SEED,"bootstrap_unit":"paired client"}

def evaluate(prop,condition,y,folds,x=None,fixed=None,origin=""):
    if fixed is None: prob,pred=fit_oof_predictions(x,y,folds,condition)
    else: prob,pred=fixed
    auc=float(roc_auc_score(y,prob)); b=boot(y,prob)
    metric={"Property":prop,"AttackCondition":condition,"Classifier":"Logistic Regression",
        "ROCAUCRaw":auc,"ROCAUCSeparability":auc_separability(auc),
        "Accuracy":float(accuracy_score(y,pred)),"F1":float(f1_score(y,pred,zero_division=0)),
        "AUCSeparabilityCI95Low":b["auc_sep_ci95_low"],"AUCSeparabilityCI95High":b["auc_sep_ci95_high"],
        "NumClients":len(y),"NumPositive":int(y.sum()),"NumNegative":int(len(y)-y.sum()),"EvidenceOrigin":origin}
    predictions=[{"Property":prop,"AttackCondition":condition,"ClientID":i,"Fold":int(folds[i]),
        "TrueLabel":int(y[i]),"PredictedLabel":int(pred[i]),"PositiveProbability":float(prob[i])} for i in range(len(y))]
    return metric,{"Property":prop,"AttackCondition":condition,**b},predictions,prob

def fedavg(prop,y,folds):
    metrics=rows(FEDAVG/"attack_metrics.csv"); preds=rows(FEDAVG/"oof_predictions.csv")
    row=next(r for r in metrics if r["Property"]==prop and r["AttackFeature"]=="proposed_spectral")
    selected=sorted((r for r in preds if r["Property"]==prop and r["AttackFeature"]=="proposed_spectral"),key=lambda r:int(r["ClientID"]))
    yy=np.array([int(r["TrueLabel"]) for r in selected]); prob=np.array([float(r["PositiveProbability"]) for r in selected]); pred=np.array([int(r["PredictedLabel"]) for r in selected])
    if not np.array_equal(yy,y) or float(roc_auc_score(y,prob))!=float(row["ROCAUCRaw"]): raise RuntimeError("EXPERIMENT 13 INVALID: FedAvg reference")
    return evaluate(prop,"FedAvg Standard",y,folds,fixed=(prob,pred),origin="inherited formal FedAvg R2 seed42")

def judgment(result):
    adaptive={p:float(result[p]["adaptive"]["ROCAUCSeparability"]) for p in PROPERTIES}
    fed={p:float(result[p]["fedavg"]["ROCAUCSeparability"]) for p in PROPERTIES}
    reduction={p:100*((fed[p]-.5)-(adaptive[p]-.5))/(fed[p]-.5) for p in PROPERTIES}
    if all(adaptive[p]<=.60 and adaptive[p]<fed[p] for p in PROPERTIES): return "MASK-AWARE ROBUSTNESS STRONGLY SUPPORTED","YES"
    close_to_fedavg={p:abs(adaptive[p]-fed[p])<=.01 for p in PROPERTIES}
    if sum(adaptive[p]<fed[p] and reduction[p]>=30 for p in PROPERTIES)==1 and any(close_to_fedavg.values()):
        return "MASK-AWARE ROBUSTNESS PARTIALLY SUPPORTED","NO"
    if all(adaptive[p]<fed[p] for p in PROPERTIES) and statistics.mean(reduction.values())>=30: return "MASK-AWARE ROBUSTNESS SUPPORTED","MAYBE"
    if sum(adaptive[p]<fed[p] and reduction[p]>=30 for p in PROPERTIES)==1: return "MASK-AWARE ROBUSTNESS PARTIALLY SUPPORTED","NO"
    return "MASK-AWARE ROBUSTNESS NOT SUPPORTED","NO"

def main():
    if os.environ.get("CONDA_DEFAULT_ENV")!="dsl1": raise RuntimeError("must run in dsl1")
    if OUTPUT.exists() or REPORT.exists(): raise FileExistsError("Experiment 13 output exists")
    if sha(SPEC)!=EXPECTED_SPEC_HASH or SPEC_HASH.read_text().split()[0]!=EXPECTED_SPEC_HASH: raise RuntimeError("feature spec hash")
    input_rows,payloads=verify_inputs(); lf=labels_folds(); standard=standard_features(); replay=standard_replay(standard,lf)
    adaptive={}; mask={}; fnames=mnames=None
    for client in range(50):
        pp=[x for _,x in payloads[client]]
        av=build_mask_aware_feature_vector(pp,standard[client]); mv=compute_mask_only_features(pp)
        fnames=av.names if fnames is None else fnames; mnames=mv.names if mnames is None else mnames
        if av.names!=fnames or mv.names!=mnames: raise RuntimeError("feature schema")
        adaptive[client]=av.values; mask[client]=mv.values
    ax=np.stack([adaptive[i] for i in range(50)]); mx=np.stack([mask[i] for i in range(50)])
    if ax.shape!=(50,ADAPTIVE_DIMENSION) or not np.isfinite(ax).all() or not np.isfinite(mx).all(): raise RuntimeError("feature matrix")
    stage=OUTPUT.parent/f".p050_staging_{os.getpid()}"; stage.mkdir(parents=True,exist_ok=False)
    try:
        feature_rows=[]
        for client in range(50):
            row={"ClientID":client}; row.update({n:float(v) for n,v in zip(fnames,ax[client])}); feature_rows.append(row)
        write_csv(stage/"adaptive_features.csv",feature_rows)
        provenance=adaptive_feature_provenance(fnames); write_csv(stage/"adaptive_feature_provenance.csv",provenance)
        if any(r["raw_private_update_required"] or r["label_required_at_inference"] or not r["allowed"] for r in provenance): raise RuntimeError("provenance gate")
        write_csv(stage/"input_hash_verification.csv",input_rows)
        write_json(stage/"feature_spec.json",{"status":"locked_before_auc","feature_spec_sha256":EXPECTED_SPEC_HASH,
            "adaptive_dimension":len(fnames),"mask_only_dimension":len(mnames),
            "adaptive_feature_names":list(fnames),"mask_only_feature_names":list(mnames),
            "classifier":{"name":"LogisticRegression","C":1.0,"penalty":"l2","class_weight":"balanced","max_iter":5000,"random_state":ATTACK_CV_SEED},
            "scaling":"StandardScaler fit on OOF training fold only","bootstrap":{"replicates":BOOTSTRAP_REPLICATES,"seed":BOOTSTRAP_SEED,"unit":"client"}})
        (stage/"feature_spec_hash.txt").write_text(EXPECTED_SPEC_HASH+"\n")
        metrics=[]; boots=[]; ps=[]; pm=[]; pa=[]; result={}; paired=[]
        for prop in PROPERTIES:
            y,folds=lf[prop]
            fm,fb,_,_=fedavg(prop,y,folds); metrics.append(fm); boots.append(fb)
            sm,sb,sp,sprob=evaluate(prop,"RSR Standard",y,folds,fixed=replay[prop],origin="exact Experiment-11 replay"); metrics.append(sm); boots.append(sb); ps+=sp
            mm,mb,mp,mprob=evaluate(prop,"RSR Mask-Only",y,folds,x=mx,origin="frozen public-mask control"); metrics.append(mm); boots.append(mb); pm+=mp
            am,ab,ap,aprob=evaluate(prop,"RSR Mask-Aware Adaptive",y,folds,x=ax,origin="frozen Experiment-13 vector"); metrics.append(am); boots.append(ab); pa+=ap
            pb=paired_boot(y,aprob,sprob); paired.append({"Property":prop,"AttackCondition":"Adaptive minus Standard RSR AUC-sep",**pb})
            result[prop]={"fedavg":fm,"standard":sm,"mask_only":mm,"adaptive":am,"paired":pb}
        write_csv(stage/"oof_predictions_standard.csv",ps); write_csv(stage/"oof_predictions_mask_only.csv",pm); write_csv(stage/"oof_predictions_mask_aware.csv",pa)
        write_csv(stage/"attack_metrics.csv",metrics); write_csv(stage/"bootstrap_results.csv",boots+paired)
        analysis=[]
        for prop in PROPERTIES:
            f=float(result[prop]["fedavg"]["ROCAUCSeparability"]); s=float(result[prop]["standard"]["ROCAUCSeparability"]); a=float(result[prop]["adaptive"]["ROCAUCSeparability"])
            analysis.append({"Property":prop,"FedAvgAUCsep":f,"RSRStandardAUCsep":s,"RSRAdaptiveAUCsep":a,
                "AdaptiveDeltaVsStandard":a-s,"AdaptiveRecoveryRatePct":100*(a-s)/(f-s),
                "FedAvgExcess":f-.5,"RSRStandardExcess":s-.5,"RSRAdaptiveExcess":a-.5,
                "AdaptiveExcessReductionVsFedAvgPct":100*((f-.5)-(a-.5))/(f-.5)})
        write_csv(stage/"adaptive_recovery.csv",analysis)
        control=all(float(result[p]["mask_only"]["ROCAUCSeparability"])<=.56 for p in PROPERTIES)
        verdict,next_step=judgment(result)
        deltas=[float(result[p]["adaptive"]["ROCAUCSeparability"])-float(result[p]["standard"]["ROCAUCSeparability"]) for p in PROPERTIES]
        q1="NO" if all(x<=0 for x in deltas) else ("LIMITED" if max(deltas)<.03 else "YES")
        def answer(prop):
            f=float(result[prop]["fedavg"]["ROCAUCSeparability"]); a=float(result[prop]["adaptive"]["ROCAUCSeparability"]); r=((f-.5)-(a-.5))/(f-.5)
            return "YES" if a<f and r>=.30 else ("PARTIAL" if a<f else "NO")
        answers={"Q1_additional_leakage_recovered":q1,"Q2_extreme_reduction_remains":answer(PROPERTIES[0]),
            "Q3_periodicity_reduction_remains":answer(PROPERTIES[1]),
            "Q4_public_mask_alone_leaks":"NO EVIDENCE" if control else ("POSSIBLE" if max(float(result[p]["mask_only"]["ROCAUCSeparability"]) for p in PROPERTIES)<=.60 else "POSSIBLE - ARTIFACT AUDIT REQUIRED"),
            "Q5_proceed_to_multiseed":next_step}
        main_rows=[{"AttackVictim":label,"ExtremeAUCsep":result[PROPERTIES[0]][key]["ROCAUCSeparability"],
            "PeriodicityAUCsep":result[PROPERTIES[1]][key]["ROCAUCSeparability"]} for key,label in
            [("fedavg","FedAvg Standard"),("standard","RSR Standard"),("mask_only","RSR Mask-Only"),("adaptive","RSR Mask-Aware Adaptive")]]
        write_csv(stage/"main_results.csv",main_rows)
        manifest={"experiment":13,"dataset":"PEMS08","victim_method":"RSR","victim_seed":42,"p":P,
            "new_victim_training":False,"new_mask_sampling":False,"defense_changed":False,"communication_mechanism_changed":False,
            "attacker_knows_mask":True,"attacker_knows_p":True,"attacker_knows_mechanism":True,
            "attacker_access_raw_update":False,"attacker_access_dropped_coefficients":False,
            "classifier":"LogisticRegression","properties":list(PROPERTIES),"evaluation":"client-level OOF","folds":"SAME_AS_EXP11",
            "feature_spec_sha256":EXPECTED_SPEC_HASH,"feature_spec_frozen_before_final_auc":True,
            "archived_payload_hashes_passed":100,"archived_server_update_hashes_passed":100,
            "standard_rsr_attack_replay":"PASS","same_oof_folds":True,"mask_only_negative_control_pass":control,
            "results":result,"analysis":analysis,"answers":answers,"judgment":verdict,"proceed_to_pems08_multiseed":next_step,
            "environment":{"conda_environment":os.environ.get("CONDA_DEFAULT_ENV"),"python":sys.version,"python_executable":sys.executable,
                "platform":platform.platform(),"torch":torch.__version__,"numpy":np.__version__,"sklearn":sklearn.__version__},
            "git_commit":"UNAVAILABLE_NOT_A_GIT_CHECKOUT","tests":{"status":"pending"},
            "verification":{"no_raw_dropped_coefficients_used":True,"no_private_update_used":True,"no_new_masks":True,
                "no_p_sweep":True,"no_classifier_sweep":True,"no_feature_subset_sweep":True,
                "scaler_fit_only_on_oof_training_folds":True,"standard_probabilities_bitwise_reproduced":True},
            "formal_input_hashes":{str(p.relative_to(ROOT)):sha(p) for p in [ARCHIVE/"manifest.json",ARCHIVE/"server_visible_update_hashes.json",
                ARCHIVE/"attack/attack_features.csv",ARCHIVE/"attack/attack_metrics.csv",ARCHIVE/"attack/attack_oof_predictions.csv",
                ARCHIVE/"attack/property_labels.csv",FEDAVG/"attack_metrics.csv",FEDAVG/"oof_predictions.csv",SPEC]},
            "source_hashes":{"attacks/mask_aware_spectral_attack.py":sha(ROOT/"attacks/mask_aware_spectral_attack.py"),
                "experiments/exp_mask_aware_adaptive_attack.py":sha(ROOT/"experiments/exp_mask_aware_adaptive_attack.py"),
                "tests/test_mask_aware_spectral_attack.py":sha(ROOT/"tests/test_mask_aware_spectral_attack.py")},"timestamp_unix":time.time()}
        write_json(stage/"manifest.json",manifest); (stage/"test_results.txt").write_text("PENDING\n")
        def sep(prop,key): return float(result[prop][key]["ROCAUCSeparability"])
        report=f"""# Mask-Aware Adaptive Spectral Attack Report

## 1. Objective
Replay-only audit of whether archived RSR p=0.5 leakage mitigation remains under a public-mask, mechanism-aware honest-but-curious server. This is not a defense experiment or formal privacy claim.

## 2. Threat Model
The attacker uses retained RSR values, public p/masks/schema, and cross-round overlap. Dropped coefficients, raw client updates, pre-RSR tensors, private local state, and learned reconstruction are unavailable.

## 3. Archived Victim Provenance
PEMS08/STGCN seed42 Experiment 11: 100/100 payload hashes and 100/100 server-update hashes passed. Standard RSR OOF probabilities replayed bitwise with the same clients, labels, folds, scaler, Logistic Regression, and bootstrap policy.

## 4. Adaptive Attack Construction
Observed coefficients use z=p*tilde-z only where retained. The frozen vector adds HT first-order/energy estimators, observed summaries, normalized jointly-visible cross-round statistics, and public mask statistics to the unchanged standard vector. Missing coordinates are never imputed.

## 5. Feature Specification
Frozen before AUC; SHA-256: `{EXPECTED_SPEC_HASH}`; {ADAPTIVE_DIMENSION} features; no feature/classifier subset sweep.

| Family | Uses mask | Uses retained values | Cross-round overlap | Private raw update |
|---|---:|---:|---:|---:|
| Standard | no | yes | yes | no |
| Mask-only | yes | no | yes | no |
| HT first-order | yes | yes | no | no |
| Energy | yes | yes | no | no |
| Observed-only | yes | yes | no | no |
| Intersection | yes | yes | yes | no |

## 6. Mask-Only Negative Control
Extreme AUC-sep {sep(PROPERTIES[0],"mask_only"):.3f}; Periodicity {sep(PROPERTIES[1],"mask_only"):.3f}. Status: {"PASS" if control else "REQUIRES CAUTION"}. Because masks are data-independent, elevation is treated as possible finite-sample schedule/seed association, not direct property leakage without further evidence.

## 7. Main Attack Results
| Attack / Victim | Extreme AUC-sep | Periodicity AUC-sep |
|---|---:|---:|
| FedAvg Standard | {sep(PROPERTIES[0],"fedavg"):.3f} | {sep(PROPERTIES[1],"fedavg"):.3f} |
| RSR Standard | {sep(PROPERTIES[0],"standard"):.3f} | {sep(PROPERTIES[1],"standard"):.3f} |
| RSR Mask-Only | {sep(PROPERTIES[0],"mask_only"):.3f} | {sep(PROPERTIES[1],"mask_only"):.3f} |
| RSR Mask-Aware Adaptive | {sep(PROPERTIES[0],"adaptive"):.3f} | {sep(PROPERTIES[1],"adaptive"):.3f} |

Raw AUC, accuracy, F1, and 95% client-bootstrap intervals are in the CSV artifacts.

## 8. Bootstrap Uncertainty
Extreme adaptive 95% AUC-sep CI: [{result[PROPERTIES[0]]["adaptive"]["AUCSeparabilityCI95Low"]:.3f}, {result[PROPERTIES[0]]["adaptive"]["AUCSeparabilityCI95High"]:.3f}].
Periodicity adaptive 95% AUC-sep CI: [{result[PROPERTIES[1]]["adaptive"]["AUCSeparabilityCI95Low"]:.3f}, {result[PROPERTIES[1]]["adaptive"]["AUCSeparabilityCI95High"]:.3f}]. Paired adaptive-minus-standard bootstrap results are reported without a significance claim.

## 9. Adaptive Leakage Recovery Analysis
| Property | FedAvg | RSR Standard | RSR Adaptive | Recovery |
|---|---:|---:|---:|---:|
| Extreme | {analysis[0]["FedAvgAUCsep"]:.3f} | {analysis[0]["RSRStandardAUCsep"]:.3f} | {analysis[0]["RSRAdaptiveAUCsep"]:.3f} | {analysis[0]["AdaptiveRecoveryRatePct"]:.2f}% |
| Periodicity | {analysis[1]["FedAvgAUCsep"]:.3f} | {analysis[1]["RSRStandardAUCsep"]:.3f} | {analysis[1]["RSRAdaptiveAUCsep"]:.3f} | {analysis[1]["AdaptiveRecoveryRatePct"]:.2f}% |

Q1 additional recovery: **{answers["Q1_additional_leakage_recovered"]}**. Q2 Extreme remains reduced: **{answers["Q2_extreme_reduction_remains"]}**. Q3 Periodicity remains reduced: **{answers["Q3_periodicity_reduction_remains"]}**. Q4 mask alone: **{answers["Q4_public_mask_alone_leaks"]}**. Q5 multi-seed: **{answers["Q5_proceed_to_multiseed"]}**.

## 10. Limitations
Only PEMS08 seed42 is evaluated. The mechanism-aware attacker is not claimed optimal. RSR has no formal privacy guarantee. Bootstrap uncertainty may remain substantial. Only Extreme and Periodicity are tested. No learned missing-coefficient reconstruction attack was evaluated.

## 11. Final Judgment
**{verdict}**. Proceed to PEMS08 multi-seed RSR replication: **{next_step}**. No new victim training, mask sampling, p value, defense, or classifier sweep was run.

## Verification
Dedicated and full test suites pending.
"""
        report_tmp=REPORT.parent/f".{REPORT.name}.staging_{os.getpid()}"; report_tmp.write_text(report,encoding="utf-8")
        stage.rename(OUTPUT); report_tmp.rename(REPORT)
    except Exception:
        if stage.exists(): shutil.rmtree(stage)
        raise
    print("================ EXPERIMENT 13 =================")
    print("Mask-Aware Adaptive Spectral Attack")
    print("payload hashes 100/100 PASS; update hashes 100/100 PASS; standard replay PASS; folds PASS; feature spec PASS")
    for prop in PROPERTIES: print(prop,{k:result[prop][k]["ROCAUCSeparability"] for k in ("fedavg","standard","mask_only","adaptive")})
    print(verdict); print("multiseed",next_step)

if __name__=="__main__": main()
