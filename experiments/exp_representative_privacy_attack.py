from __future__ import annotations

"""Experiment 9: client-level OOF property inference on five matched R2 victims."""

import argparse
import csv
import hashlib
import json
import logging
import math
import secrets
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.fft import dct as scipy_dct
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from attacks.spectral_leakage.attack_dataset import raw_properties
from attacks.spectral_leakage.collector import UpdateCollector, UpdateRecord
from attacks.spectral_leakage.fingerprint import spectral_fingerprint
from datasets.client_partition import partition_nodes
from datasets.stg_utils import temporal_split
from datasets.traffic import load_traffic_dataset
from experiments.common import configure_logging, save_config, write_rows
from experiments.exp_ellipsoidal_clipping_sanity import (
    _build_clients_and_eval,
    _environment_snapshot,
    _evaluate_partitions,
    _prepare_data,
)
from experiments.exp_matched_reduced_release_frontier import (
    DPFEDAVG_CLIP,
    DiagnosticDPFedAvg,
    calibrate_dpfedavg,
    calibrate_pjsdp,
    load_and_verify_schedule,
)
from experiments.exp_public_ellipsoid_dp_sanity import PublicGeometryReleaseMechanism
from experiments.exp_spectral_norm_audit import _load_reference_config, _resolve_device, _set_seed
from federated.stg_server import STGServer
from privacy.fixed_spectral_projection import FixedSpectralProjection
from spectral.spectral_groups import GROUPS


ROUNDS = 20
CLIENTS = 50
CLIENTS_PER_ROUND = 5
RELEASES_PER_CLIENT = 2
LOCAL_EPOCHS = 1
DELTA = 1e-5
ADJACENCY = "replacement"
ATTACK_CV_SEED = 20260903
BOOTSTRAP_SEED = 314159
BOOTSTRAP_REPLICATES = 2000
EXPECTED_SCHEDULE_SHA256 = "fc9a11ab8f15c964b01accbafeed109146be8fe4ccbc155fa01281f8d9ee5399"
EXPECTED_FEATURE_SPEC_SHA256 = "260e57f6c99c94dde01bb3b09eb27bbc888c39c70c69a454d50d1dd3dec6af7d"
ALLOWED_LAYERS = ("graph_conv.theta", "temporal_in.weight", "temporal_out.weight")
PROPERTIES = ("extreme_event", "periodicity_strength")
PROPERTY_DISPLAY = {"extreme_event": "Extreme Event", "periodicity_strength": "Periodicity Strength"}
ATTACKS = ("random_guess", "norm_stats", "cross_round_stats", "proposed_spectral")
VICTIMS = (
    ("fedavg", "FedAvg R2", None),
    ("dp_fedavg_eps2048", "DP-FedAvg R2 eps2048", 2048),
    ("pjsdp_eps2048", "PJSDP R2 eps2048", 2048),
    ("dp_fedavg_eps1024", "DP-FedAvg R2 eps1024", 1024),
    ("pjsdp_eps1024", "PJSDP R2 eps1024", 1024),
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--reference_config", type=Path, required=True)
    p.add_argument("--schedule_path", type=Path, required=True)
    p.add_argument("--exp7_root", type=Path, required=True)
    p.add_argument("--exp8_root", type=Path, required=True)
    p.add_argument("--feature_spec", type=Path, required=True)
    p.add_argument("--fingerprint_audit", type=Path, required=True)
    p.add_argument("--results_root", type=Path, default=Path("results/privacy_attack_representative"))
    p.add_argument("--report_path", type=Path, default=Path("docs/REPRESENTATIVE_PRIVACY_ATTACK_REPORT.md"))
    p.add_argument("--device", default="cuda")
    return p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_one(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}")
    return rows[0]


def _read_many(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def auc_separability(raw_auc: float) -> float:
    return max(float(raw_auc), 1.0 - float(raw_auc))


def validate_feature_spec(path: Path) -> tuple[dict[str, Any], str]:
    digest = sha256_file(path)
    if digest != EXPECTED_FEATURE_SPEC_SHA256:
        raise ValueError(f"feature specification hash mismatch: {digest}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if spec["status"] != "locked_before_attack_results" or int(spec["observation_length"]) != 2:
        raise ValueError("R2 feature specification is not locked")
    if spec["schedule_file_sha256"] != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("feature specification schedule mismatch")
    if list(spec["allowed_layers"]) != list(ALLOWED_LAYERS):
        raise ValueError("feature specification layer set changed")
    if int(spec["attack_families"]["proposed_spectral"]["round_dct_coefficients"]) != 2:
        raise ValueError("R2 proposed fingerprint must use two DCT coefficients")
    return spec, digest


def build_property_labels(data_root: Path, spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    data = load_traffic_dataset(data_root, "pems08")
    private_train = temporal_split(data.values).train
    partition = partition_nodes(data.adjacency, CLIENTS, data.coordinates, 42)
    properties = raw_properties([private_train[:, nodes] for nodes in partition.node_ids])
    result: dict[str, dict[str, Any]] = {}
    score_keys = {"extreme_event": "extreme_score", "periodicity_strength": "periodicity_score"}
    for property_name in PROPERTIES:
        property_spec = spec["properties"][property_name]
        threshold = float(property_spec["binary_threshold"])
        scores = np.asarray(properties[score_keys[property_name]], dtype=float)
        labels = (scores >= threshold).astype(int)
        minority = int(min(labels.sum(), len(labels) - labels.sum()))
        n_splits = min(5, minority)
        if n_splits < 3:
            raise RuntimeError(f"INSUFFICIENT PROPERTY SUPPORT: {property_name}")
        result[property_name] = {
            "scores": scores, "labels": labels, "threshold": threshold,
            "threshold_source": property_spec["threshold_source"], "n_splits": n_splits,
        }
    return result


def make_folds(labels: np.ndarray, n_splits: int) -> np.ndarray:
    clients = np.arange(len(labels), dtype=int)
    assignment = np.full(len(labels), -1, dtype=int)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=ATTACK_CV_SEED)
    for fold, (train, test) in enumerate(splitter.split(clients, labels)):
        if set(train).intersection(set(test)):
            raise RuntimeError("client appears in both attack train and test")
        assignment[test] = fold
    if np.any(assignment < 0):
        raise RuntimeError("not every client received an OOF fold")
    return assignment


def _record_vector(record: UpdateRecord) -> np.ndarray:
    values = [float(record.update_norm)]
    for name in ALLOWED_LAYERS:
        tensor = record.layers[name].detach().float()
        values.extend([float(tensor.norm()), float(tensor.mean()), float(tensor.std(unbiased=False))])
    return np.asarray(values, dtype=np.float64)


def extract_feature_families(records: Sequence[UpdateRecord]) -> dict[str, np.ndarray]:
    if len(records) != RELEASES_PER_CLIENT:
        raise ValueError("R2 fingerprint requires exactly two server-visible observations")
    ordered = sorted(records, key=lambda row: row.round)
    if any(set(row.layers) != set(ALLOWED_LAYERS) for row in ordered):
        raise ValueError("collector contains a non-public or inconsistent attack layer set")
    norms = np.asarray([row.update_norm for row in ordered], dtype=np.float64)
    norm_dct = scipy_dct(norms, norm="ortho")
    norm_features = np.asarray([
        norms.mean(), abs(norms[1] - norms[0]), float(np.square(norms).sum()),
        norm_dct[0], norm_dct[1],
    ], dtype=np.float64)
    raw = np.stack([_record_vector(row) for row in ordered])
    cross_round = np.concatenate([raw.mean(axis=0), raw[1] - raw[0]])
    proposed = spectral_fingerprint(ordered, round_coefficients=2).detach().cpu().numpy().astype(np.float64)
    if proposed.shape != (24,):
        raise RuntimeError(f"unexpected R2 spectral fingerprint dimension: {proposed.shape}")
    return {"norm_stats": norm_features, "cross_round_stats": cross_round, "proposed_spectral": proposed}


def fit_oof_predictions(
    x: np.ndarray | None, labels: np.ndarray, folds: np.ndarray, attack_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.full(len(labels), np.nan, dtype=float)
    predictions = np.full(len(labels), -1, dtype=int)
    for fold in sorted(np.unique(folds)):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        if set(train).intersection(set(test)):
            raise RuntimeError("client-level OOF leakage")
        if attack_name == "random_guess":
            probabilities[test] = 0.5
            predictions[test] = int(labels[train].mean() >= 0.5)
            continue
        if x is None:
            raise ValueError("non-random attack requires features")
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(x[train])
        test_scaled = scaler.transform(x[test])
        classifier = LogisticRegression(
            C=1.0, penalty="l2", max_iter=5000,
            class_weight="balanced", random_state=ATTACK_CV_SEED,
        )
        classifier.fit(train_scaled, labels[train])
        probabilities[test] = classifier.predict_proba(test_scaled)[:, 1]
        predictions[test] = classifier.predict(test_scaled)
    if not np.isfinite(probabilities).all() or np.any(predictions < 0):
        raise RuntimeError("incomplete OOF predictions")
    return probabilities, predictions


def bootstrap_intervals(
    labels: np.ndarray, probabilities: np.ndarray, predictions: np.ndarray,
) -> dict[str, float | int]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    raw_auc: list[float] = []
    auc_sep: list[float] = []
    accuracy: list[float] = []
    f1: list[float] = []
    n = len(labels)
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, n, size=n)
        y = labels[indices]
        if len(np.unique(y)) < 2:
            continue
        auc = float(roc_auc_score(y, probabilities[indices]))
        raw_auc.append(auc)
        auc_sep.append(auc_separability(auc))
        accuracy.append(float(accuracy_score(y, predictions[indices])))
        f1.append(float(f1_score(y, predictions[indices], zero_division=0)))
    if len(raw_auc) < int(0.9 * BOOTSTRAP_REPLICATES):
        raise RuntimeError("too many one-class client bootstrap samples")
    interval = lambda values: np.percentile(np.asarray(values), [2.5, 97.5])
    raw_ci, sep_ci, acc_ci, f1_ci = map(interval, (raw_auc, auc_sep, accuracy, f1))
    return {
        "ROCAUCRawCI95Low": float(raw_ci[0]), "ROCAUCRawCI95High": float(raw_ci[1]),
        "AUCSeparabilityCI95Low": float(sep_ci[0]), "AUCSeparabilityCI95High": float(sep_ci[1]),
        "AccuracyCI95Low": float(acc_ci[0]), "AccuracyCI95High": float(acc_ci[1]),
        "F1CI95Low": float(f1_ci[0]), "F1CI95High": float(f1_ci[1]),
        "ValidBootstrapReplicates": len(raw_auc), "RequestedBootstrapReplicates": BOOTSTRAP_REPLICATES,
        "BootstrapUnit": "CLIENT", "BootstrapSeed": BOOTSTRAP_SEED,
    }


def evaluate_attack(
    victim: str, property_name: str, attack_name: str, x: np.ndarray | None,
    labels: np.ndarray, folds: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    probabilities, predictions = fit_oof_predictions(x, labels, folds, attack_name)
    raw_auc = float(roc_auc_score(labels, probabilities))
    ci = bootstrap_intervals(labels, probabilities, predictions)
    metric = {
        "Property": PROPERTY_DISPLAY[property_name], "VictimMethod": victim,
        "AttackFeature": attack_name, "Classifier": "Random Guess" if attack_name == "random_guess" else "Logistic Regression",
        "Accuracy": float(accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions, zero_division=0)),
        "ROCAUCRaw": raw_auc, "ROCAUCSeparability": auc_separability(raw_auc),
        "AUCSeparabilityCI95Low": ci["AUCSeparabilityCI95Low"],
        "AUCSeparabilityCI95High": ci["AUCSeparabilityCI95High"],
        "NumClients": len(labels), "NumPositive": int(labels.sum()), "NumNegative": int(len(labels) - labels.sum()),
    }
    prediction_rows = [{
        "Property": PROPERTY_DISPLAY[property_name], "AttackFeature": attack_name,
        "ClientID": client_id, "Fold": int(folds[client_id]),
        "TrueLabel": int(labels[client_id]), "PredictedLabel": int(predictions[client_id]),
        "PositiveProbability": float(probabilities[client_id]),
    } for client_id in range(len(labels))]
    return metric, prediction_rows, {"Property": PROPERTY_DISPLAY[property_name], "AttackFeature": attack_name, **ci}


def _historical_sources(exp7_root: Path, exp8_root: Path) -> dict[str, Path]:
    return {
        "dp_fedavg_eps2048": exp8_root / "seed_42" / "dp_fedavg_r2_eps2048",
        "pjsdp_eps2048": exp7_root / "seed_42" / "r2_e",
        "dp_fedavg_eps1024": exp8_root / "seed_42" / "dp_fedavg_r2_eps1024",
        "pjsdp_eps1024": exp8_root / "seed_42" / "pjsdp_r2_eps1024",
    }


def verify_victim_sources(exp7_root: Path, exp8_root: Path) -> tuple[dict[str, Path], dict[tuple[str, int], dict[str, str]]]:
    sources = _historical_sources(exp7_root, exp8_root)
    for key, path in sources.items():
        if not (path / "config.yaml").is_file() or not (path / "metrics.csv").is_file():
            raise FileNotFoundError(f"missing victim source: {key}")
        config = yaml.safe_load((path / "config.yaml").read_text(encoding="utf-8"))
        epsilon = int(key.rsplit("eps", 1)[1])
        expected = {
            "dataset": "pems08", "seed": 42, "rounds": ROUNDS,
            "clients_per_round": CLIENTS_PER_ROUND, "local_epochs": LOCAL_EPOCHS,
        }
        for field, value in expected.items():
            if config.get(field) != value:
                raise ValueError(f"victim source {key} mismatches {field}")
        release_field = "releases_per_client" if "releases_per_client" in config else "release_count_per_client"
        if int(config[release_field]) != RELEASES_PER_CLIENT or int(float(config["target_epsilon"])) != epsilon:
            raise ValueError(f"victim source {key} has incompatible privacy composition")
        if bool(config["sampling_amplification_used"]):
            raise ValueError("subsampling amplification is forbidden")
    summary_rows = _read_many(exp8_root / "matched_frontier_summary.csv")
    utility: dict[tuple[str, int], dict[str, str]] = {}
    for row in summary_rows:
        utility[(row["Method"], int(float(row["TargetEpsilon"])))] = row
    return sources, utility


def _replay_mechanism(method: str, epsilon: int | None, model: torch.nn.Module, config: Mapping[str, Any]) -> Any:
    if method == "fedavg":
        return None
    private_seed = secrets.randbits(63)
    if method == "dp_fedavg":
        calibration = calibrate_dpfedavg(float(epsilon))
        return DiagnosticDPFedAvg(DPFEDAVG_CLIP, calibration["noise_std"], torch.Generator().manual_seed(private_seed))
    calibration = calibrate_pjsdp(config, float(epsilon))
    projection = FixedSpectralProjection(model.state_dict(), 0.90, 42)
    if projection.mask_hash != str(config["projection_mask_hash"]):
        raise RuntimeError("PJSDP projection mask changed")
    bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
    return PublicGeometryReleaseMechanism(
        "current", projection, bounds, None, calibration["group_noise_stds"],
        torch.Generator().manual_seed(private_seed),
    )


def replay_victim(
    *, key: str, display: str, epsilon: int | None, output: Path,
    reference_config: Mapping[str, Any], prepared: tuple, schedule: Sequence[Sequence[int]],
    device: torch.device,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    updates_dir = output / "updates"
    configure_logging(output / "replay_train.log")
    _set_seed(42)
    model, clients, validation_sets, test_sets = _build_clients_and_eval(reference_config, prepared, device)
    method = "fedavg" if key == "fedavg" else ("dp_fedavg" if key.startswith("dp_fedavg") else "pjsdp")
    mechanism = _replay_mechanism(method, epsilon, model, reference_config)
    collector = UpdateCollector(save_full_updates=False, output_dir=updates_dir)
    server = STGServer(
        model, clients, device, 42, collector=collector, privacy=mechanism,
        topk_ratio=1.0, error_feedback=False, event_tau=0.0, skip_policy="zero",
        report_client_train_loss=True, participation_schedule=schedule,
    )
    losses = []
    started = time.perf_counter()
    for round_index in range(ROUNDS):
        row = server.run_round(round_index, CLIENTS_PER_ROUND, LOCAL_EPOCHS, 0.0)
        loss = float(row["client_train_loss"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite victim loss: {key}")
        losses.append(loss)
        logging.info("victim=%s round=%d/%d train_loss=%.8f", key, round_index + 1, ROUNDS, loss)
    by_client = collector.by_client()
    if len(collector.records) != ROUNDS * CLIENTS_PER_ROUND or set(by_client) != set(range(CLIENTS)):
        raise RuntimeError("replay did not collect all 100 server-visible releases")
    if any(len(records) != RELEASES_PER_CLIENT for records in by_client.values()):
        raise RuntimeError("replay observation count is not exactly two per client")
    *_, mean, std = prepared
    validation = _evaluate_partitions(server.model, validation_sets, device, reference_config, mean, std)
    test = _evaluate_partitions(server.model, test_sets, device, reference_config, mean, std)
    metrics = {
        "Victim": display, "Method": method, "Epsilon": "INF / Non-Private" if epsilon is None else epsilon,
        "MAE": test["MAE"], "RMSE": test["RMSE"], "MAPE": test["MAPE"],
        "Val_MAE": validation["MAE"], "Val_RMSE": validation["RMSE"], "Val_MAPE": validation["MAPE"],
        "FinalTrainLoss": losses[-1], "TrainStable": True, "NaNInfDetected": False,
        "RuntimeSeconds": time.perf_counter() - started, "CollectedServerVisibleUpdates": len(collector.records),
    }
    write_rows(output / "replay_metrics.csv", [metrics])
    (output / "environment.txt").write_text(_environment_snapshot(device), encoding="utf-8")
    return {"metrics": metrics, "records": by_client, "updates_dir": updates_dir}


def _check_replay_utility(key: str, replay: Mapping[str, Any], historical: Mapping[tuple[str, int], Mapping[str, Any]]) -> dict[str, Any]:
    if key == "fedavg":
        return {"status": "NEW_NON_PRIVATE_REFERENCE", "mae_relative_difference": 0.0, "rmse_relative_difference": 0.0}
    method = "DP-FedAvg" if key.startswith("dp_fedavg") else "Current PJSDP"
    epsilon = int(key.rsplit("eps", 1)[1])
    source = historical[(method, epsilon)]
    replay_metrics = replay["metrics"]
    mae_diff = abs(float(replay_metrics["MAE"]) - float(source["MAE"])) / float(source["MAE"])
    rmse_diff = abs(float(replay_metrics["RMSE"]) - float(source["RMSE"])) / float(source["RMSE"])
    if mae_diff > 0.05 or rmse_diff > 0.05:
        raise RuntimeError(f"attack replay utility outside predeclared tolerance: {key}")
    return {"status": "STOCHASTIC_TRAJECTORY_CONSISTENT", "mae_relative_difference": mae_diff, "rmse_relative_difference": rmse_diff}


def _write_features(path: Path, feature_by_client: Mapping[int, Mapping[str, np.ndarray]]) -> None:
    rows = []
    for client_id in range(CLIENTS):
        row: dict[str, Any] = {"ClientID": client_id}
        for family in ("norm_stats", "cross_round_stats", "proposed_spectral"):
            values = feature_by_client[client_id][family]
            for index, value in enumerate(values):
                row[f"{family}_{index:03d}"] = float(value)
        rows.append(row)
    write_rows(path, rows)


def _property_fold_rows(property_name: str, info: Mapping[str, Any], folds: np.ndarray) -> list[dict[str, Any]]:
    return [{
        "Property": PROPERTY_DISPLAY[property_name], "ClientID": client_id,
        "Score": float(info["scores"][client_id]), "Label": int(info["labels"][client_id]),
        "Fold": int(folds[client_id]), "Threshold": float(info["threshold"]),
        "ThresholdSource": info["threshold_source"], "AttackCVSeed": ATTACK_CV_SEED,
    } for client_id in range(CLIENTS)]


def _victim_attack_config(
    key: str, display: str, epsilon: int | None, source: Path | None,
    feature_hash: str,
) -> dict[str, Any]:
    return {
        "experiment": "Representative-Point Privacy Attack Evaluation",
        "victim": display, "victim_key": key, "dataset": "pems08", "seed": 42,
        "rounds": ROUNDS, "clients": CLIENTS, "clients_per_round": CLIENTS_PER_ROUND,
        "releases_per_client": RELEASES_PER_CLIENT, "local_epochs": LOCAL_EPOCHS,
        "epsilon": "INF / Non-Private" if epsilon is None else epsilon,
        "delta": None if epsilon is None else DELTA,
        "adjacency": None if epsilon is None else ADJACENCY,
        "subsampling_amplification": "NO", "schedule_hash": EXPECTED_SCHEDULE_SHA256,
        "attack_observation": "server-received update used for aggregation",
        "passive_attack": True, "victim_model_modified": False,
        "feature_spec_sha256": feature_hash, "properties": list(PROPERTIES),
        "peak_pattern_evaluated": False, "classifier": "L2 Logistic Regression",
        "classifier_C": 1.0, "attack_cv_seed": ATTACK_CV_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "source_result_path": str(source.resolve()) if source else None,
        "whether_replayed": True,
    }


def _matched_privacy(
    summary: Sequence[Mapping[str, Any]], utility: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for property_name in PROPERTY_DISPLAY.values():
        for epsilon in (2048, 1024):
            dpf_victim = f"DP-FedAvg R2 eps{epsilon}"
            pjs_victim = f"PJSDP R2 eps{epsilon}"
            dpf = next(row for row in summary if row["Property"] == property_name and row["VictimMethod"] == dpf_victim and row["AttackFeature"] == "proposed_spectral")
            pjs = next(row for row in summary if row["Property"] == property_name and row["VictimMethod"] == pjs_victim and row["AttackFeature"] == "proposed_spectral")
            du, pu = utility[("DP-FedAvg", epsilon)], utility[("Current PJSDP", epsilon)]
            result.append({
                "Property": property_name, "Epsilon": epsilon,
                "DPFedAvg_AUCSep": dpf["ROCAUCSeparability"], "PJSDP_AUCSep": pjs["ROCAUCSeparability"],
                "Difference": float(pjs["ROCAUCSeparability"]) - float(dpf["ROCAUCSeparability"]),
                "DPFedAvg_MAE": du["MAE"], "PJSDP_MAE": pu["MAE"],
                "DPFedAvg_RMSE": du["RMSE"], "PJSDP_RMSE": pu["RMSE"],
            })
    return result


def _combined_table(
    summary: Sequence[Mapping[str, Any]], utility: Mapping[tuple[str, int], Mapping[str, Any]],
    replay_metrics: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for key, display, epsilon in VICTIMS:
        metrics = replay_metrics[key] if epsilon is None else utility[("DP-FedAvg" if key.startswith("dp_fedavg") else "Current PJSDP", int(epsilon))]
        by_property = {}
        for property_display in PROPERTY_DISPLAY.values():
            by_property[property_display] = next(row for row in summary if row["VictimMethod"] == display and row["Property"] == property_display and row["AttackFeature"] == "proposed_spectral")
        extreme, periodic = by_property["Extreme Event"], by_property["Periodicity Strength"]
        rows.append({
            "Method": display, "Epsilon": "INF / Non-Private" if epsilon is None else epsilon,
            "Delta": "N/A" if epsilon is None else DELTA, "ReleasesPerClient": RELEASES_PER_CLIENT,
            "MAE": metrics["MAE"], "RMSE": metrics["RMSE"],
            "ExtremeAUCsep": extreme["ROCAUCSeparability"],
            "ExtremeCI": f"[{float(extreme['AUCSeparabilityCI95Low']):.4f}, {float(extreme['AUCSeparabilityCI95High']):.4f}]",
            "PeriodicityAUCsep": periodic["ROCAUCSeparability"],
            "PeriodicityCI": f"[{float(periodic['AUCSeparabilityCI95Low']):.4f}, {float(periodic['AUCSeparabilityCI95High']):.4f}]",
        })
    return rows


def _leakage_reduction(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for property_display in PROPERTY_DISPLAY.values():
        fed = next(row for row in summary if row["Property"] == property_display and row["VictimMethod"] == "FedAvg R2" and row["AttackFeature"] == "proposed_spectral")
        fed_sep = float(fed["ROCAUCSeparability"])
        for _, victim, epsilon in VICTIMS[1:]:
            row = next(row for row in summary if row["Property"] == property_display and row["VictimMethod"] == victim and row["AttackFeature"] == "proposed_spectral")
            method_sep = float(row["ROCAUCSeparability"])
            excess = fed_sep - 0.5
            relative = "NOT_REPORTED_BASELINE_NEAR_RANDOM" if excess < 0.03 else 1.0 - (method_sep - 0.5) / excess
            result.append({
                "Property": property_display, "VictimMethod": victim, "Epsilon": epsilon,
                "FedAvgAUCSeparability": fed_sep, "MethodAUCSeparability": method_sep,
                "AbsoluteReduction": fed_sep - method_sep,
                "RelativeLeakageReduction": relative,
            })
    return result


def _plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _plots(summary: Sequence[Mapping[str, Any]], combined: Sequence[Mapping[str, Any]], run_root: Path) -> None:
    _plot_style()
    plot_dir = run_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=False)
    victim_order = [display for _, display, _ in VICTIMS]
    labels = ["FedAvg", "DPF\n2048", "PJSDP\n2048", "DPF\n1024", "PJSDP\n1024"]
    colors = ["#AAB2BA", "#8A949E", "#4C78A8", "#68737D", "#2F6690"]
    for property_display, filename in (("Extreme Event", "extreme_event_auc.png"), ("Periodicity Strength", "periodicity_auc.png")):
        rows = [next(row for row in summary if row["Property"] == property_display and row["VictimMethod"] == victim and row["AttackFeature"] == "proposed_spectral") for victim in victim_order]
        values = np.asarray([float(row["ROCAUCSeparability"]) for row in rows])
        low = np.asarray([float(row["AUCSeparabilityCI95Low"]) for row in rows])
        high = np.asarray([float(row["AUCSeparabilityCI95High"]) for row in rows])
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        x = np.arange(len(rows))
        ax.bar(x, values, color=colors, width=0.7)
        ax.errorbar(x, values, yerr=np.vstack([values - low, high - values]), fmt="none", ecolor="#303030", capsize=3, linewidth=1)
        ax.axhline(0.5, color="#303030", linestyle="--", linewidth=1, label="Random separability")
        ax.set_xticks(x, labels)
        ax.set_ylim(0.48, min(1.0, max(0.7, float(high.max()) + 0.04)))
        ax.set_ylabel("ROC-AUC separability")
        ax.set_title(f"{property_display}: client-level OOF attack (n=50)")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    for property_display, suffix in (("Extreme Event", "extreme"), ("Periodicity Strength", "periodicity")):
        fig, ax = plt.subplots(figsize=(6.3, 4.0))
        for row, color, label in zip(combined, colors, labels):
            auc = float(row["ExtremeAUCsep"] if property_display == "Extreme Event" else row["PeriodicityAUCsep"])
            ax.scatter(auc, float(row["MAE"]), s=65, color=color, label=label)
        ax.set_xlabel("ROC-AUC separability (lower is better)")
        ax.set_ylabel("MAE (lower is better)")
        ax.set_title(f"Privacy-utility plane: {property_display}")
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / f"privacy_utility_scatter_{suffix}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, property_display in zip(axes, PROPERTY_DISPLAY.values()):
        rows = [next(row for row in summary if row["Property"] == property_display and row["VictimMethod"] == "FedAvg R2" and row["AttackFeature"] == attack) for attack in ATTACKS]
        ax.bar(np.arange(4), [float(row["ROCAUCSeparability"]) for row in rows], color=["#C7CCD1", "#9FA8B1", "#7A8793", "#4C78A8"])
        ax.axhline(0.5, color="#303030", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(4), ["Random", "Norm", "Cross\nround", "Spectral"])
        ax.set_title(property_display)
    axes[0].set_ylabel("ROC-AUC separability")
    fig.suptitle("FedAvg R2 attack baseline comparison")
    fig.tight_layout()
    fig.savefig(plot_dir / "attack_baseline_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _comparison_status(summary: Sequence[Mapping[str, Any]], epsilon: int) -> str:
    statuses = []
    for property_display in PROPERTY_DISPLAY.values():
        dpf = next(row for row in summary if row["Property"] == property_display and row["VictimMethod"] == f"DP-FedAvg R2 eps{epsilon}" and row["AttackFeature"] == "proposed_spectral")
        pjs = next(row for row in summary if row["Property"] == property_display and row["VictimMethod"] == f"PJSDP R2 eps{epsilon}" and row["AttackFeature"] == "proposed_spectral")
        if float(pjs["AUCSeparabilityCI95Low"]) > float(dpf["AUCSeparabilityCI95High"]):
            statuses.append("WORSE")
        elif float(dpf["AUCSeparabilityCI95Low"]) > float(pjs["AUCSeparabilityCI95High"]):
            statuses.append("BETTER")
        else:
            statuses.append("COMPARABLE")
    if "WORSE" in statuses:
        return "WORSE"
    if "BETTER" in statuses:
        return "BETTER"
    return "COMPARABLE"


def decide(summary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fed = {prop: next(row for row in summary if row["Property"] == prop and row["VictimMethod"] == "FedAvg R2" and row["AttackFeature"] == "proposed_spectral") for prop in PROPERTY_DISPLAY.values()}
    baseline_outperform = {}
    leakage_reduced = {}
    for prop, spectral in fed.items():
        simple = [next(row for row in summary if row["Property"] == prop and row["VictimMethod"] == "FedAvg R2" and row["AttackFeature"] == name) for name in ("norm_stats", "cross_round_stats")]
        baseline_outperform[prop] = float(spectral["ROCAUCSeparability"]) >= max(float(row["ROCAUCSeparability"]) for row in simple) + 0.03
        leakage_reduced[prop] = all(
            float(next(row for row in summary if row["Property"] == prop and row["VictimMethod"] == f"PJSDP R2 eps{epsilon}" and row["AttackFeature"] == "proposed_spectral")["ROCAUCSeparability"]) < float(spectral["ROCAUCSeparability"])
            for epsilon in (2048, 1024)
        )
    fed_weak = all(float(row["ROCAUCSeparability"]) <= 0.55 and float(row["AUCSeparabilityCI95Low"]) <= 0.52 for row in fed.values())
    comparison = {epsilon: _comparison_status(summary, epsilon) for epsilon in (2048, 1024)}
    if "WORSE" in comparison.values():
        conclusion, next_exp = "PJSDP UTILITY ADVANTAGE DOES NOT YET MATCH EMPIRICAL PRIVACY", "Reassess Empirical Leakage Mechanism"
    elif fed_weak:
        conclusion, next_exp = "SPECTRAL ATTACK EVIDENCE REMAINS WEAK", "Strengthen Attack Evaluation Before Final Multi-Seed Runs"
    elif all(baseline_outperform.values()) and all(leakage_reduced.values()):
        conclusion, next_exp = "EMPIRICAL PRIVACY EVIDENCE SUPPORTED", "Three-Seed Final Evaluation"
    else:
        conclusion, next_exp = "EMPIRICAL PRIVACY EVIDENCE PARTIALLY SUPPORTED", "Three-Seed Final Evaluation"
    proposed_status = "YES" if all(baseline_outperform.values()) else ("MIXED" if any(baseline_outperform.values()) else "NO")
    monotonic = all(
        float(next(row for row in summary if row["Property"] == prop and row["VictimMethod"] == "PJSDP R2 eps1024" and row["AttackFeature"] == "proposed_spectral")["ROCAUCSeparability"])
        <= float(next(row for row in summary if row["Property"] == prop and row["VictimMethod"] == "PJSDP R2 eps2048" and row["AttackFeature"] == "proposed_spectral")["ROCAUCSeparability"])
        for prop in PROPERTY_DISPLAY.values()
    )
    return {
        "conclusion": conclusion, "next": next_exp, "proposed_outperforms": proposed_status,
        "leakage_reduced": leakage_reduced, "comparison": comparison,
        "empirical_monotonicity": monotonic,
    }


def _report(
    path: Path, summary: Sequence[Mapping[str, Any]], combined: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any], replay_checks: Mapping[str, Mapping[str, Any]], feature_hash: str,
) -> None:
    lines = [
        "# Representative Privacy Attack Report", "",
        "## 1-5. Motivation, victims, schedule, and threat model", "",
        "The reduced-release frontier identified epsilon 1024 as the utility-preserving candidate. This passive evaluation now asks whether an honest-but-curious server can infer client properties from exactly the updates it receives. The five points are FedAvg R2, DP-FedAvg R2 and PJSDP R2 at epsilon 2048 and 1024. Every victim uses the same public 20-round schedule, five clients per round, two observations per client, and schedule SHA-256 `" + EXPECTED_SCHEDULE_SHA256 + "`.", "",
        "Attack input is the post-mechanism update used by server aggregation. Pre-noise/pre-clip updates, raw gradients, client features, adjacency, and training data are forbidden as fingerprints. Property labels supervise evaluation only and are never fingerprint columns.", "",
        "## 6-10. Properties and locked R2 feature specification", "",
        "Extreme Event and Periodicity Strength retain the existing property-score rules and frozen historical attack-training thresholds. Peak Pattern is not evaluated. Each client has T=2 observations. Only DCT coefficients 0 (level) and 1 (contrast) exist; higher round-frequency coefficients are removed. Means, two-point differences, energies, graph-order bands, and temporal bands remain defined but have limited temporal resolution.", "",
        f"The pre-attack feature specification SHA-256 is `{feature_hash}`. The same feature dimensions and normalization policy are used for all victims.", "",
        "## 11-18. OOF protocol, classifiers, metrics, and confidence intervals", "",
        "The earlier 15-client holdout was replaced by client-level stratified OOF evaluation. Each of 50 clients is test data exactly once; all observations from one client stay in one fold. The StandardScaler is fit only on attack-training clients. The main classifier is L2 Logistic Regression with C=1.0 and fixed seed 20260903. Raw ROC-AUC and direction-independent `AUC_sep=max(AUC,1-AUC)` are both reported. Client bootstrap uses 2,000 replicates and seed 314159; one-class resamples are skipped.", "",
        "## 19-24. Property inference results", "",
        "| Property | Victim | Raw AUC | AUC separability | 95% client-bootstrap CI | Accuracy | F1 |", "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in summary:
        if row["AttackFeature"] != "proposed_spectral":
            continue
        lines.append(f"| {row['Property']} | {row['VictimMethod']} | {float(row['ROCAUCRaw']):.4f} | {float(row['ROCAUCSeparability']):.4f} | [{float(row['AUCSeparabilityCI95Low']):.4f}, {float(row['AUCSeparabilityCI95High']):.4f}] | {float(row['Accuracy']):.4f} | {float(row['F1']):.4f} |")
    lines += [
        "", f"FedAvg proposed spectral fingerprint outperforms both simple baselines: **{decision['proposed_outperforms']}**.", "",
        f"Matched PJSDP versus DP-FedAvg: epsilon 2048 **{decision['comparison'][2048]}**, epsilon 1024 **{decision['comparison'][1024]}**.", "",
        "## 25-26. Formal versus empirical privacy and combined interpretation", "",
        "Formal privacy is supplied by the accountant: replacement adjacency, delta=1e-5, two releases per client, no subsampling amplification, and epsilon 1024 < 2048. Empirical privacy is finite-sample property-inference performance. A low empirical AUC neither proves nor replaces DP. The attack is passive and therefore does not alter forecasting performance.", "",
        "The combined privacy-utility table preserves the previously completed utility metrics; replay is used only to obtain server-visible attack observations.", "",
        "## Replay equivalence qualification", "",
        "Prior private DP RNG seeds were intentionally not persisted, so the original noisy tensors cannot be recovered bit-for-bit. Replays use unchanged mechanism, accountant, schedule, model initialization, optimizer, data split, and hyperparameters, with fresh private DP randomness. A predeclared 5% relative MAE/RMSE tolerance checks stochastic trajectory consistency; this is not presented as bitwise deterministic replay.", "",
    ]
    for key, check in replay_checks.items():
        lines.append(f"- {key}: {check['status']}; relative MAE difference {float(check['mae_relative_difference']):.4%}, RMSE difference {float(check['rmse_relative_difference']):.4%}.")
    lines += [
        "", "## 27-28. Limitations and evidence strength", "",
        "Only 50 clients and one victim-training seed are available. Bootstrap quantifies client resampling uncertainty but does not replace independent victim seeds. T=2 cannot resolve higher cross-round frequencies. Frozen thresholds avoid result-driven relabeling but remain dataset-specific. Confidence-interval overlap is treated conservatively and no significance fishing is performed.", "",
        f"Overall conclusion: **{decision['conclusion']}**.", "",
        "## 29. Only one next recommendation", "", f"**{decision['next']}**",
    ]
    if path.exists():
        raise FileExistsError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.results_root.exists():
        raise FileExistsError(f"refusing to overwrite results: {args.results_root}")
    if args.report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.report_path}")
    if not args.fingerprint_audit.is_file():
        raise FileNotFoundError("R2 fingerprint audit must exist before attack execution")
    spec, feature_hash = validate_feature_spec(args.feature_spec)
    schedule, _ = load_and_verify_schedule(args.schedule_path)
    sources, historical_utility = verify_victim_sources(args.exp7_root, args.exp8_root)
    reference_config = _load_reference_config(args.reference_config)
    expected = {"dataset": "pems08", "seed": 42, "num_clients": CLIENTS, "clients_per_round": CLIENTS_PER_ROUND, "local_epochs": LOCAL_EPOCHS}
    for key, value in expected.items():
        if reference_config.get(key) != value:
            raise ValueError(f"reference config mismatch for {key}")
    property_info = build_property_labels(args.data_root, spec)
    folds = {name: make_folds(info["labels"], int(info["n_splits"])) for name, info in property_info.items()}

    run_root = args.results_root / "r2_seed42"
    run_root.mkdir(parents=True, exist_ok=False)
    folds_dir = run_root / "folds"
    folds_dir.mkdir(parents=True, exist_ok=False)
    for property_name in PROPERTIES:
        write_rows(folds_dir / f"{PROPERTY_DISPLAY[property_name].replace(' ', '_')}_folds.csv", _property_fold_rows(property_name, property_info[property_name], folds[property_name]))

    prepared = _prepare_data(reference_config, args.data_root, "pems08")
    device = _resolve_device(args.device)
    replay_root = run_root / "attack_replay"
    replay_root.mkdir(parents=True, exist_ok=False)
    replayed: dict[str, dict[str, Any]] = {}
    replay_checks: dict[str, dict[str, Any]] = {}
    manifest_entries = []
    for key, display, epsilon in VICTIMS:
        replay = replay_victim(
            key=key, display=display, epsilon=epsilon, output=replay_root / key,
            reference_config=reference_config, prepared=prepared, schedule=schedule, device=device,
        )
        replayed[key] = replay
        replay_checks[key] = _check_replay_utility(key, replay, historical_utility)
        source = sources.get(key)
        config_hash = sha256_file(source / "config.yaml") if source else None
        manifest_entries.append({
            "victim_key": key, "method": display,
            "epsilon": "INF / Non-Private" if epsilon is None else epsilon,
            "delta": "N/A" if epsilon is None else DELTA,
            "rounds": ROUNDS, "releases_per_client": RELEASES_PER_CLIENT,
            "schedule_hash": EXPECTED_SCHEDULE_SHA256,
            "source_result_path": str(source.resolve()) if source else str((replay_root / key).resolve()),
            "source_config_hash": config_hash,
            "attack_observation_source": str(replay["updates_dir"].resolve()),
            "observation_semantics": "server-received post-mechanism update used for aggregation",
            "whether_replayed": True, "replay_validation": replay_checks[key],
        })
    manifest = {
        "experiment": "Representative-Point Privacy Attack Evaluation",
        "schedule_hash": EXPECTED_SCHEDULE_SHA256, "feature_spec_sha256": feature_hash,
        "victims": manifest_entries,
    }
    (run_root / "victim_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_metrics: list[dict[str, Any]] = []
    replay_metrics = {key: value["metrics"] for key, value in replayed.items()}
    for key, display, epsilon in VICTIMS:
        victim_dir = run_root / key
        victim_dir.mkdir(parents=True, exist_ok=False)
        feature_by_client = {client_id: extract_feature_families(replayed[key]["records"][client_id]) for client_id in range(CLIENTS)}
        _write_features(victim_dir / "attack_features.csv", feature_by_client)
        label_rows = []
        for property_name in PROPERTIES:
            label_rows.extend(_property_fold_rows(property_name, property_info[property_name], folds[property_name]))
        write_rows(victim_dir / "property_labels.csv", label_rows)
        victim_predictions: list[dict[str, Any]] = []
        victim_metrics: list[dict[str, Any]] = []
        victim_bootstrap: list[dict[str, Any]] = []
        for property_name in PROPERTIES:
            labels = property_info[property_name]["labels"]
            for attack_name in ATTACKS:
                x = None if attack_name == "random_guess" else np.stack([feature_by_client[client_id][attack_name] for client_id in range(CLIENTS)])
                metric, predictions, bootstrap = evaluate_attack(display, property_name, attack_name, x, labels, folds[property_name])
                metric["Epsilon"] = "INF / Non-Private" if epsilon is None else epsilon
                victim_metrics.append(metric)
                victim_predictions.extend(predictions)
                victim_bootstrap.append(bootstrap)
        write_rows(victim_dir / "oof_predictions.csv", victim_predictions)
        write_rows(victim_dir / "attack_metrics.csv", victim_metrics)
        write_rows(victim_dir / "bootstrap_ci.csv", victim_bootstrap)
        save_config(victim_dir / "attack_config.yaml", _victim_attack_config(key, display, epsilon, sources.get(key), feature_hash))
        all_metrics.extend(victim_metrics)

    matched = _matched_privacy(all_metrics, historical_utility)
    combined = _combined_table(all_metrics, historical_utility, replay_metrics)
    leakage = _leakage_reduction(all_metrics)
    decision = decide(all_metrics)
    write_rows(run_root / "representative_attack_summary.csv", all_metrics)
    write_rows(run_root / "matched_empirical_privacy.csv", matched)
    write_rows(run_root / "privacy_utility_attack_table.csv", combined)
    write_rows(run_root / "empirical_leakage_reduction.csv", leakage)
    _plots(all_metrics, combined, run_root)
    _report(args.report_path, all_metrics, combined, decision, replay_checks, feature_hash)

    print("========================================")
    print("REPRESENTATIVE PRIVACY ATTACK EVALUATION")
    print("========================================")
    print(f"\nDataset:\nPEMS08\n\nClients:\n50\n\nRounds:\n20\n\nReleases/client:\n2\n\nSchedule hash:\n{EXPECTED_SCHEDULE_SHA256}")
    print("\nAttack evaluation:\nCLIENT-LEVEL OOF\n\nMain classifier:\nLogistic Regression\n\nProperties:\nExtreme Event\nPeriodicity Strength\n\nPeak Pattern:\nNOT EVALUATED")
    for property_display in PROPERTY_DISPLAY.values():
        print(f"\n---\n\n## {property_display.upper()}")
        for _, victim, _ in VICTIMS:
            row = next(row for row in all_metrics if row["Property"] == property_display and row["VictimMethod"] == victim and row["AttackFeature"] == "proposed_spectral")
            print(f"\n{victim}:\nRaw AUC = {float(row['ROCAUCRaw']):.6f}\nAUC Sep = {float(row['ROCAUCSeparability']):.6f}\n95% CI = [{float(row['AUCSeparabilityCI95Low']):.6f}, {float(row['AUCSeparabilityCI95High']):.6f}]")
    print("\n---\n\n## FEDAVG ATTACK BASELINES")
    for property_display in PROPERTY_DISPLAY.values():
        print(f"\n{property_display}:")
        for attack_name in ATTACKS:
            row = next(row for row in all_metrics if row["Property"] == property_display and row["VictimMethod"] == "FedAvg R2" and row["AttackFeature"] == attack_name)
            print(f"{attack_name}: AUC Sep = {float(row['ROCAUCSeparability']):.6f}")
    print(f"\nDoes proposed spectral attack outperform simple baselines:\n{decision['proposed_outperforms']}")
    print(f"\nDoes PJSDP reduce leakage vs FedAvg:\nExtreme = {'YES' if decision['leakage_reduced']['Extreme Event'] else 'NO'}\nPeriodicity = {'YES' if decision['leakage_reduced']['Periodicity Strength'] else 'NO'}")
    print(f"\nPJSDP vs matched DP-FedAvg:\n\nepsilon2048:\n{decision['comparison'][2048]}\n\nepsilon1024:\n{decision['comparison'][1024]}")
    print(f"\nFormal privacy:\nepsilon1024 < epsilon2048\n\nEmpirical monotonicity observed:\n{'YES' if decision['empirical_monotonicity'] else 'NO'}")
    print(f"\nOverall conclusion:\n{decision['conclusion']}")
    print(f"\nNext recommended experiment:\n{decision['next']}")
    return {"summary": all_metrics, "matched": matched, "combined": combined, "decision": decision}


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
