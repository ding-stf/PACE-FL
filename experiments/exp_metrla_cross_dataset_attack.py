from __future__ import annotations

"""Experiment 14 property protocol audit and locked METR-LA attack."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from attacks.spectral_leakage.attack_dataset import raw_properties
from attacks.spectral_leakage.collector import UpdateRecord
from datasets.client_partition import partition_nodes
from datasets.stg_utils import temporal_split
from datasets.traffic import load_traffic_dataset
from experiments import exp_locked_seed_privacy_attack as locked
from experiments.common import save_config, write_rows
from experiments.exp_representative_privacy_attack import (
    ALLOWED_LAYERS,
    ATTACK_CV_SEED,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_FEATURE_SPEC_SHA256,
    PROPERTY_DISPLAY,
    auc_separability,
    bootstrap_intervals,
    extract_feature_families,
)
import yaml

PROPERTIES = ("extreme_event", "periodicity_strength")
SCORE_KEYS = {
    "extreme_event": "extreme_score",
    "periodicity_strength": "periodicity_score",
}
FIXED_PEMS_THRESHOLDS = {
    "extreme_event": 0.00024041351123933166,
    "periodicity_strength": 0.41982064394641283,
}
SCHEDULE_HASH = "d43a6f21de12a7642d0d29dc310cf3b4307c8136ef93e645f742fbded648bc99"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--feature_spec", type=Path, required=True)
    p.add_argument("--protocol_dir", type=Path, required=True)
    p.add_argument("--audit_doc", type=Path, required=True)
    p.add_argument("--audit_only", action="store_true")
    p.add_argument("--victim", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--method")
    return p


def read_many(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_spec(path):
    digest = locked.sha256_file(path)
    if digest != EXPECTED_FEATURE_SPEC_SHA256:
        raise ValueError("R2 feature spec hash changed")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        spec["status"] != "locked_before_attack_results"
        or int(spec["observation_length"]) != 2
        or list(spec["allowed_layers"]) != list(ALLOWED_LAYERS)
        or int(spec["attack_families"]["proposed_spectral"]["expected_dimension"]) != 24
        or spec["classifier"]["main"] != "logistic_regression"
        or float(spec["classifier"]["C"]) != 1.0
        or int(spec["classifier"]["max_iter"]) != 5000
        or int(spec["cross_validation"]["seed"]) != ATTACK_CV_SEED
        or int(spec["bootstrap"]["replicates"]) != BOOTSTRAP_REPLICATES
        or int(spec["bootstrap"]["seed"]) != BOOTSTRAP_SEED
    ):
        raise ValueError("locked R2 attack protocol changed")
    return spec, digest


def load_scores(data_root):
    data = load_traffic_dataset(data_root, "metr_la")
    split = temporal_split(data.values)
    partition = partition_nodes(data.adjacency, 50, data.coordinates, 42)
    if len(partition.node_ids) != 50 or min(map(len, partition.node_ids)) < 1:
        raise RuntimeError("METR-LA partition protocol mismatch")
    values = raw_properties([split.train[:, nodes] for nodes in partition.node_ids])
    return data, partition, {name: np.asarray(values[key], dtype=float) for name, key in SCORE_KEYS.items()}


def leakage_free_stratified_folds(scores, n_splits=5):
    # LOO median labels prevent each client's own score from affecting its
    # provisional stratum. Final labels are recomputed from each training fold.
    provisional = np.asarray([
        score >= np.median(np.delete(scores, client_id))
        for client_id, score in enumerate(scores)
    ], dtype=int)
    minority = int(min(provisional.sum(), len(provisional) - provisional.sum()))
    n_splits = min(int(n_splits), minority)
    if n_splits < 3:
        raise RuntimeError("insufficient property support")
    assignment = np.full(len(scores), -1, dtype=int)
    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=ATTACK_CV_SEED
    )
    clients = np.arange(len(scores))
    for fold, (_, test) in enumerate(splitter.split(clients, provisional)):
        assignment[test] = fold
    if np.any(assignment < 0):
        raise RuntimeError("incomplete OOF assignment")
    return provisional, assignment


def fold_local_labels(scores, folds):
    labels = np.full(len(scores), -1, dtype=int)
    thresholds = []
    for fold in sorted(np.unique(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        threshold = float(np.median(scores[train]))
        labels[test] = (scores[test] >= threshold).astype(int)
        thresholds.append({
            "Fold": int(fold),
            "Threshold": threshold,
            "TrainClients": len(train),
            "TestClients": len(test),
            "ThresholdFitOnTrainingFoldOnly": True,
        })
    if np.any(labels < 0):
        raise RuntimeError("incomplete cross-fitted labels")
    return labels, thresholds


def audit(args):
    if args.protocol_dir.exists():
        raise FileExistsError(args.protocol_dir)
    if args.audit_doc.exists():
        raise FileExistsError(args.audit_doc)
    spec, feature_hash = validate_spec(args.feature_spec)
    data, partition, scores = load_scores(args.data_root)
    args.protocol_dir.mkdir(parents=True)
    fold_rows, threshold_rows, label_rows, score_rows = [], [], [], []
    audit = {}
    for name in PROPERTIES:
        values = scores[name]
        fixed = FIXED_PEMS_THRESHOLDS[name]
        fixed_labels = (values >= fixed).astype(int)
        provisional, folds = leakage_free_stratified_folds(values)
        labels, thresholds = fold_local_labels(values, folds)
        audit[name] = {
            "fixed_negative": int((fixed_labels == 0).sum()),
            "fixed_positive": int(fixed_labels.sum()),
            "negative": int((labels == 0).sum()),
            "positive": int(labels.sum()),
            "n_splits": int(len(np.unique(folds))),
            "score_min": float(values.min()),
            "score_max": float(values.max()),
        }
        for client_id in range(50):
            fold_rows.append({
                "Property": PROPERTY_DISPLAY[name],
                "ClientID": client_id,
                "Fold": int(folds[client_id]),
                "ProvisionalLOOStratum": int(provisional[client_id]),
            })
            label_rows.append({
                "Property": PROPERTY_DISPLAY[name],
                "ClientID": client_id,
                "Fold": int(folds[client_id]),
                "Label": int(labels[client_id]),
                "ThresholdSource": "attack-training-fold median only",
            })
            score_rows.append({
                "Property": PROPERTY_DISPLAY[name],
                "ClientID": client_id,
                "Score": float(values[client_id]),
            })
        for row in thresholds:
            threshold_rows.append({"Property": PROPERTY_DISPLAY[name], **row})
    write_rows(args.protocol_dir / "fold_assignments.csv", fold_rows)
    write_rows(args.protocol_dir / "fold_thresholds.csv", threshold_rows)
    write_rows(args.protocol_dir / "property_labels.csv", label_rows)
    write_rows(args.protocol_dir / "property_scores.csv", score_rows)
    save_config(args.protocol_dir / "protocol.yaml", {
        "Dataset": "METR-LA",
        "Seed": 42,
        "Nodes": int(data.adjacency.shape[0]),
        "Clients": 50,
        "PartitionStrategy": partition.strategy,
        "History": 12,
        "Horizon": 12,
        "Properties": list(PROPERTIES),
        "ScoreDefinitions": {
            "Extreme Event": "fraction of valid observations with within-node absolute z-score > 2.5",
            "Periodicity Strength": "daily-frequency FFT power divided by total non-DC profile power at 5-minute sampling",
        },
        "PEMS08FrozenThresholds": FIXED_PEMS_THRESHOLDS,
        "PEMS08ThresholdPortabilityDecision": "NOT PORTABLE: produces a single class on METR-LA",
        "METRLABinaryThresholdRule": "median fitted separately on each OOF attack-training fold",
        "FoldAssignmentRule": "client-level stratified OOF using leave-one-out-median provisional strata",
        "FeatureSpecHash": feature_hash,
        "FeatureDimension": 24,
        "Classifier": "StandardScaler(train only) + L2 LogisticRegression(C=1,max_iter=5000,class_weight=balanced)",
        "Bootstrap": f"{BOOTSTRAP_REPLICATES} client resamples, seed {BOOTSTRAP_SEED}",
        "ScheduleHash": SCHEDULE_HASH,
        "PrivateDataUsedToTuneThresholdAfterAUC": False,
    })
    lines = [
        "# METR-LA Property Protocol Audit",
        "",
        "## 1. Extreme Event definition",
        "",
        "For each client, valid observations are standardized within each sensor over the private training partition. The score is the fraction with absolute z-score greater than 2.5. This is exactly the existing PEMS08 mathematical score.",
        "",
        "## 2. Periodicity Strength definition",
        "",
        "The per-client sensor-mean profile is transformed with the existing FFT statistic. Daily-frequency power is divided by total non-DC power, using 12 samples/hour. The mathematical score is unchanged.",
        "",
        "## 3. Threshold rule",
        "",
        "The frozen PEMS08 numerical thresholds are not portable: on METR-LA they yield 50/0 Extreme labels and 0/50 Periodicity labels. Because the binary cut is distribution-dependent, each OOF attack fold fits its median on training clients only and applies it to that fold's held-out clients.",
        "",
        "## 4. Dataset independence",
        "",
        "The score definitions are dataset-independent; the numerical binary thresholds are not. No threshold was inspected or changed after observing attack AUC.",
        "",
        "## 5. Leakage controls",
        "",
        "Fold assignment uses leave-one-out-median provisional strata, so a client's own score does not determine its stratum threshold. Final thresholds and StandardScaler parameters are fitted on attack-training clients only.",
        "",
        "## 6. Positive and negative counts",
        "",
        "| Property | Fixed PEMS08 threshold (negative/positive) | Cross-fitted METR-LA labels (negative/positive) | OOF folds |",
        "|---|---:|---:|---:|",
    ]
    for name in PROPERTIES:
        item = audit[name]
        lines.append(
            f"| {PROPERTY_DISPLAY[name]} | {item['fixed_negative']}/{item['fixed_positive']} | {item['negative']}/{item['positive']} | {item['n_splits']} |"
        )
    lines += [
        "",
        "Peak Pattern is not evaluated. The attack fingerprint, dimensionality, classifier, seeds, and bootstrap protocol are unchanged.",
    ]
    args.audit_doc.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))


def load_protocol(protocol_dir):
    folds, labels, scores = {}, {}, {}
    for name in PROPERTIES:
        display = PROPERTY_DISPLAY[name]
        fold_rows = [r for r in read_many(protocol_dir / "fold_assignments.csv") if r["Property"] == display]
        label_rows = [r for r in read_many(protocol_dir / "property_labels.csv") if r["Property"] == display]
        score_rows = [r for r in read_many(protocol_dir / "property_scores.csv") if r["Property"] == display]
        for rows in (fold_rows, label_rows, score_rows):
            rows.sort(key=lambda r: int(r["ClientID"]))
            if len(rows) != 50:
                raise RuntimeError("property protocol incomplete")
        folds[name] = np.asarray([int(r["Fold"]) for r in fold_rows])
        labels[name] = np.asarray([int(r["Label"]) for r in label_rows])
        scores[name] = np.asarray([float(r["Score"]) for r in score_rows])
    return folds, labels, scores


def load_cache(victim):
    path = (victim / "server_visible_updates").resolve()
    if "private_debug_only" in path.parts or path.parent != victim.resolve():
        raise RuntimeError("attack path may only read server-visible cache")
    records = []
    for file in sorted(path.glob("*.pt")):
        record = torch.load(file, map_location="cpu", weights_only=False)
        if not isinstance(record, UpdateRecord):
            raise TypeError(file)
        missing = set(ALLOWED_LAYERS) - set(record.layers)
        if missing:
            raise RuntimeError(f"server-visible packet misses locked layers: {sorted(missing)}")
        record = UpdateRecord(
            record.round, record.client_id,
            {name: record.layers[name] for name in ALLOWED_LAYERS},
            record.update_norm,
        )
        records.append(record)
    if len(records) != 100:
        raise RuntimeError("attack requires 100 server-visible packets")
    by_client = {}
    for client_id in range(50):
        rows = sorted([r for r in records if r.client_id == client_id], key=lambda r: r.round)
        if len(rows) != 2:
            raise RuntimeError("attack requires exactly two observations per client")
        by_client[client_id] = rows
    return by_client


def evaluate_fold_local(x, scores, folds):
    probabilities = np.full(50, np.nan)
    predictions = np.full(50, -1, dtype=int)
    labels = np.full(50, -1, dtype=int)
    threshold_rows = []
    for fold in sorted(np.unique(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        threshold = float(np.median(scores[train]))
        y_train = (scores[train] >= threshold).astype(int)
        y_test = (scores[test] >= threshold).astype(int)
        scaler = StandardScaler()
        train_x = scaler.fit_transform(x[train])
        test_x = scaler.transform(x[test])
        model = LogisticRegression(
            C=1.0,
            penalty="l2",
            max_iter=5000,
            class_weight="balanced",
            random_state=ATTACK_CV_SEED,
        )
        model.fit(train_x, y_train)
        probabilities[test] = model.predict_proba(test_x)[:, 1]
        predictions[test] = model.predict(test_x)
        labels[test] = y_test
        threshold_rows.append({
            "Fold": int(fold),
            "Threshold": threshold,
            "TrainClients": len(train),
            "TestClients": len(test),
            "ScalerFitOnTrainingOnly": True,
        })
    return labels, probabilities, predictions, threshold_rows


def attack(args):
    if args.victim is None or args.output is None or not args.method:
        raise ValueError("victim, output, and method are required")
    if args.output.exists():
        raise FileExistsError(args.output)
    _, feature_hash = validate_spec(args.feature_spec)
    folds, expected_labels, scores = load_protocol(args.protocol_dir)
    records = load_cache(args.victim)
    x = np.stack([
        extract_feature_families(records[client_id])["proposed_spectral"]
        for client_id in range(50)
    ])
    if x.shape != (50, 24):
        raise RuntimeError("R2 fingerprint dimensionality changed")
    args.output.mkdir(parents=True)
    metric_rows, prediction_rows, bootstrap_rows = [], [], []
    for name in PROPERTIES:
        labels, probabilities, predictions, thresholds = evaluate_fold_local(
            x, scores[name], folds[name]
        )
        if not np.array_equal(labels, expected_labels[name]):
            raise RuntimeError("cross-fitted labels changed")
        raw_auc = float(roc_auc_score(labels, probabilities))
        ci = bootstrap_intervals(labels, probabilities, predictions)
        metric_rows.append({
            "RunID": read_many(args.victim / "metrics.csv")[0]["RunID"],
            "Dataset": "METR-LA",
            "Method": args.method,
            "Property": PROPERTY_DISPLAY[name],
            "AttackFeature": "proposed_spectral",
            "Classifier": "Logistic Regression",
            "ROCAUCRaw": raw_auc,
            "ROCAUCSeparability": auc_separability(raw_auc),
            "AUCSeparabilityCI95Low": ci["AUCSeparabilityCI95Low"],
            "AUCSeparabilityCI95High": ci["AUCSeparabilityCI95High"],
            "Accuracy": float(accuracy_score(labels, predictions)),
            "F1": float(f1_score(labels, predictions, zero_division=0)),
            "NumPositive": int(labels.sum()),
            "NumNegative": int(50 - labels.sum()),
            "PrivateDebugUsedByAttack": False,
        })
        for client_id in range(50):
            prediction_rows.append({
                "Property": PROPERTY_DISPLAY[name],
                "ClientID": client_id,
                "Fold": int(folds[name][client_id]),
                "TrueLabel": int(labels[client_id]),
                "PredictedLabel": int(predictions[client_id]),
                "PositiveProbability": float(probabilities[client_id]),
            })
        bootstrap_rows.append({
            "Property": PROPERTY_DISPLAY[name],
            "AUCSeparabilityCI95Low": ci["AUCSeparabilityCI95Low"],
            "AUCSeparabilityCI95High": ci["AUCSeparabilityCI95High"],
            "ValidBootstrapReplicates": ci["ValidBootstrapReplicates"],
            "RequestedBootstrapReplicates": BOOTSTRAP_REPLICATES,
            "BootstrapUnit": "CLIENT",
            "BootstrapSeed": BOOTSTRAP_SEED,
        })
    write_rows(args.output / "attack_metrics.csv", metric_rows)
    write_rows(args.output / "oof_predictions.csv", prediction_rows)
    write_rows(args.output / "bootstrap_ci.csv", bootstrap_rows)
    save_config(args.output / "attack_config.yaml", {
        "Dataset": "METR-LA",
        "FeatureSpecHash": feature_hash,
        "FeatureDimension": 24,
        "Properties": ["Extreme Event", "Periodicity Strength"],
        "PeakPattern": "NOT EVALUATED",
        "Classifier": "L2 LogisticRegression(C=1,max_iter=5000,class_weight=balanced)",
        "CrossValidation": "client-level stratified OOF; seed 20260903",
        "Threshold": "training-fold median only",
        "Standardization": "training-fold only",
        "Bootstrap": "2000 client replicates; seed 314159",
        "AttackReads": str((args.victim / "server_visible_updates").resolve()),
        "PrivateDebugUsedByAttack": False,
    })
    print(json.dumps(metric_rows, sort_keys=True))


def main():
    args = parser().parse_args()
    if args.audit_only:
        audit(args)
    else:
        attack(args)


if __name__ == "__main__":
    main()
