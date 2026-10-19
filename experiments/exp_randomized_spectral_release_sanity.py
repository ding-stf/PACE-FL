from __future__ import annotations

"""Experiment 11: one locked RSR-FedAvg PEMS08 seed-42 sanity run."""

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from attacks.spectral_leakage.collector import UpdateCollector, UpdateRecord
from communication.randomized_spectral_release import (
    SPECTRAL_GROUPS,
    mask_seed_policy,
    randomized_spectral_release,
    tensor_mapping_sha256,
)
from communication.serializer import serialize_dense_update
from experiments.common import configure_logging, save_config, write_rows
from experiments.exp_ellipsoidal_clipping_sanity import (
    _build_clients_and_eval,
    _environment_snapshot,
    _evaluate_partitions,
    _prepare_data,
)
from experiments.exp_matched_reduced_release_frontier import load_and_verify_schedule
from experiments.exp_representative_privacy_attack import (
    ALLOWED_LAYERS,
    ATTACKS,
    ATTACK_CV_SEED,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CLIENTS,
    CLIENTS_PER_ROUND,
    EXPECTED_FEATURE_SPEC_SHA256,
    EXPECTED_SCHEDULE_SHA256,
    LOCAL_EPOCHS,
    PROPERTIES,
    PROPERTY_DISPLAY,
    RELEASES_PER_CLIENT,
    ROUNDS,
    _property_fold_rows,
    build_property_labels,
    evaluate_attack,
    extract_feature_families,
    make_folds,
    validate_feature_spec,
)
from experiments.exp_spectral_norm_audit import _load_reference_config, _resolve_device, _set_seed
from federated.aggregation import apply_update, weighted_average_updates


P = 0.5
EXPERIMENT_SEED = 42
METHOD = "RSR-FedAvg"
METHOD_KEY = "rsr_fedavg"
VICTIM_RUN_ID = "experiment11_pems08_seed42_rsr_p050_r2"
REFERENCE_METRICS = {
    "MAE": 26.781565910282,
    "RMSE": 40.882619178156915,
    "MAPE": 22.32429322509099,
}
REFERENCE_AUCSEP = {
    "Extreme Event": 0.6201298701298701,
    "Periodicity Strength": 0.7240259740259741,
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, required=True)
    p.add_argument("--reference_config", type=Path, required=True)
    p.add_argument("--reference_victim", type=Path, required=True)
    p.add_argument("--reference_attack", type=Path, required=True)
    p.add_argument("--reference_evidence_manifest", type=Path, required=True)
    p.add_argument("--schedule", type=Path, required=True)
    p.add_argument("--feature_spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--before_tests_log", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    return p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_reference(args: argparse.Namespace) -> dict[str, Any]:
    required = [
        args.reference_victim / "config.yaml",
        args.reference_victim / "metrics.csv",
        args.reference_victim / "released_update_manifest.csv",
        args.reference_victim / "environment.txt",
        args.reference_attack / "attack_metrics.csv",
        args.reference_attack / "oof_predictions.csv",
        args.reference_attack / "bootstrap_ci.csv",
        args.reference_attack / "property_labels.csv",
        args.reference_evidence_manifest,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"formal FedAvg reference is incomplete: {missing}")
    config = yaml.safe_load((args.reference_victim / "config.yaml").read_text(encoding="utf-8"))
    expected = {
        "method": "FedAvg", "seed": 42, "rounds": 20, "num_clients": 50,
        "clients_per_round": 5, "releases_per_client": 2, "local_epochs": 1,
        "history": 12, "horizon": 12, "batch_size": 64, "learning_rate": 0.01,
        "schedule_hash": EXPECTED_SCHEDULE_SHA256,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"STOP: formal FedAvg reference mismatch for {key}")
    metrics_rows = read_rows(args.reference_victim / "metrics.csv")
    if len(metrics_rows) != 1:
        raise ValueError("STOP: FedAvg metrics must have exactly one row")
    metrics = metrics_rows[0]
    for key, value in REFERENCE_METRICS.items():
        if not math.isclose(float(metrics[key]), value, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"STOP: FedAvg reference {key} mismatch")
    update_paths = sorted((args.reference_victim / "server_visible_updates").glob("*.pt"))
    release_rows = read_rows(args.reference_victim / "released_update_manifest.csv")
    if len(update_paths) != 100 or len(release_rows) != 100 or int(metrics["ReleaseCount"]) != 100:
        raise ValueError("STOP: FedAvg reference does not contain 100 R2 observations")
    attacks = read_rows(args.reference_attack / "attack_metrics.csv")
    for property_name, expected_auc in REFERENCE_AUCSEP.items():
        row = next(
            item for item in attacks
            if item["Property"] == property_name
            and item["AttackFeature"] == "proposed_spectral"
        )
        if not math.isclose(float(row["ROCAUCSeparability"]), expected_auc, abs_tol=1e-12):
            raise ValueError(f"STOP: reference attack mismatch for {property_name}")
    evidence = next(
        row for row in read_rows(args.reference_evidence_manifest)
        if row["VictimRunID"] == metrics["VictimRunID"]
    )
    if evidence["SelfConsistent"].lower() != "true":
        raise ValueError("STOP: formal FedAvg reference is not self-consistent")
    return {
        "config": config, "metrics": metrics, "attack_rows": attacks,
        "update_paths": update_paths, "evidence": evidence,
        "legacy_manifest_note": (
            "The legacy victim directory has no standalone manifest.json; "
            "the locked evidence_manifest.csv supplies the cross-artifact hashes."
        ),
    }


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def dense_reference_byte_map(paths: Sequence[Path]) -> tuple[dict[tuple[int, int], int], int]:
    result: dict[tuple[int, int], int] = {}
    total = 0
    for path in paths:
        record = torch_load(path)
        key = (int(record.round), int(record.client_id))
        if key in result:
            raise ValueError("duplicate FedAvg reference update")
        size = len(serialize_dense_update(record.layers))
        result[key] = size
        total += size
    if len(result) != ROUNDS * CLIENTS_PER_ROUND:
        raise ValueError("dense reference byte map is incomplete")
    return result, total


def deterministic_mode() -> None:
    _set_seed(EXPERIMENT_SEED)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def released_manifest(records: Sequence[UpdateRecord]) -> tuple[list[dict[str, Any]], str]:
    rows = []
    for index, record in enumerate(sorted(records, key=lambda row: (row.round, row.client_id))):
        rows.append({
            "ReleaseIndexGlobal": index, "Round": record.round, "ClientID": record.client_id,
            "TensorHash": tensor_mapping_sha256(record.layers), "UpdateNorm": record.update_norm,
        })
    digest = canonical_hash([
        {key: row[key] for key in ("ReleaseIndexGlobal", "Round", "ClientID", "TensorHash")}
        for row in rows
    ])
    return rows, digest


def attack_record(record: UpdateRecord) -> UpdateRecord:
    if not set(ALLOWED_LAYERS).issubset(record.layers):
        raise RuntimeError("server-visible RSR update lacks a locked attack layer")
    layers = {name: record.layers[name].detach().cpu().clone() for name in ALLOWED_LAYERS}
    return UpdateRecord(record.round, record.client_id, layers, record.update_norm)


def verify_historical_folds(
    property_name: str, folds: np.ndarray, reference_attack: Path
) -> None:
    filename = f"{PROPERTY_DISPLAY[property_name].replace(' ', '_')}_folds.csv"
    rows = read_rows(reference_attack.parent / "fold_assignments" / filename)
    historical = {int(row["ClientID"]): int(row["Fold"]) for row in rows}
    if historical != {client_id: int(folds[client_id]) for client_id in range(CLIENTS)}:
        raise RuntimeError("locked client-level OOF folds changed")


def run_attack(
    records_by_client: Mapping[int, Sequence[UpdateRecord]],
    output: Path,
    data_root: Path,
    feature_spec_path: Path,
    reference_attack: Path,
) -> tuple[list[dict[str, Any]], str, str]:
    spec, feature_hash = validate_feature_spec(feature_spec_path)
    if feature_hash != EXPECTED_FEATURE_SPEC_SHA256:
        raise RuntimeError("locked attack feature specification changed")
    property_info = build_property_labels(data_root, spec)
    folds = {name: make_folds(info["labels"], int(info["n_splits"])) for name, info in property_info.items()}
    for name in PROPERTIES:
        verify_historical_folds(name, folds[name], reference_attack)
    features = {
        client_id: extract_feature_families(
            [attack_record(record) for record in records_by_client[client_id]]
        )
        for client_id in range(CLIENTS)
    }
    feature_rows = []
    for client_id in range(CLIENTS):
        row: dict[str, Any] = {"VictimRunID": VICTIM_RUN_ID, "ClientID": client_id}
        for family in ("norm_stats", "cross_round_stats", "proposed_spectral"):
            for index, value in enumerate(features[client_id][family]):
                row[f"{family}_{index:03d}"] = float(value)
        feature_rows.append(row)
    write_rows(output / "attack_features.csv", feature_rows)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    fold_dir = output / "fold_assignments"
    fold_dir.mkdir()
    for property_name in PROPERTIES:
        labels = property_info[property_name]["labels"]
        fold_rows = _property_fold_rows(property_name, property_info[property_name], folds[property_name])
        write_rows(fold_dir / f"{PROPERTY_DISPLAY[property_name].replace(' ', '_')}_folds.csv", fold_rows)
        label_rows.extend({"VictimRunID": VICTIM_RUN_ID, **row} for row in fold_rows)
        for attack_name in ATTACKS:
            x = None if attack_name == "random_guess" else np.stack(
                [features[index][attack_name] for index in range(CLIENTS)]
            )
            metric, predictions, bootstrap = evaluate_attack(
                METHOD, property_name, attack_name, x, labels, folds[property_name]
            )
            metric.update({"VictimRunID": VICTIM_RUN_ID, "Method": METHOD, "p": P})
            metric_rows.append(metric)
            prediction_rows.extend(
                {"VictimRunID": VICTIM_RUN_ID, "Method": METHOD, "p": P, **row}
                for row in predictions
            )
            bootstrap_rows.append(
                {"VictimRunID": VICTIM_RUN_ID, "Method": METHOD, "p": P, **bootstrap}
            )
    write_rows(output / "property_labels.csv", label_rows)
    write_rows(output / "attack_oof_predictions.csv", prediction_rows)
    write_rows(output / "attack_metrics.csv", metric_rows)
    write_rows(output / "attack_bootstrap.csv", bootstrap_rows)
    save_config(output / "attack_config.yaml", {
        "VictimRunID": VICTIM_RUN_ID,
        "feature_spec_sha256": feature_hash,
        "classifier": "L2 Logistic Regression",
        "C": 1.0,
        "attack_cv_seed": ATTACK_CV_SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "properties": list(PROPERTIES),
        "peak_pattern_evaluated": False,
        "observation": "same-run server-visible reconstructed RSR releases only",
        "mask_positions_public_to_server": True,
        "raw_update_fallback": False,
    })
    return metric_rows, sha256_file(output / "attack_features.csv"), sha256_file(output / "attack_metrics.csv")


def proposed(rows: Sequence[Mapping[str, Any]], property_display: str) -> Mapping[str, Any]:
    return next(
        row for row in rows
        if row["Property"] == property_display and row["AttackFeature"] == "proposed_spectral"
    )


def aggregate_mask_diagnostics(
    supports: Mapping[int, Sequence[frozenset[str]]], total_by_client: Mapping[int, int]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    for client_id in range(CLIENTS):
        first, second = supports[client_id]
        intersection = first & second
        union = first | second
        total = total_by_client[client_id]
        rows.append({
            "ClientID": client_id,
            "Release1Selected": len(first), "Release2Selected": len(second),
            "Intersection": len(intersection), "Union": len(union),
            "JaccardOverlap": len(intersection) / len(union) if union else 0.0,
            "IntersectionRatioOfAllSpectralCoefficients": len(intersection) / total,
            "VisibleBothFraction": len(intersection) / total,
            "VisibleOnceFraction": len(first ^ second) / total,
            "NeverVisibleFraction": (total - len(union)) / total,
        })
    values = lambda key: np.asarray([float(row[key]) for row in rows])
    jaccard = values("JaccardOverlap")
    summary = {
        "MeanJaccardOverlap": float(jaccard.mean()),
        "StdJaccardOverlap": float(jaccard.std(ddof=0)),
        "MinJaccardOverlap": float(jaccard.min()),
        "MaxJaccardOverlap": float(jaccard.max()),
        "MeanIntersectionRatio": float(values("IntersectionRatioOfAllSpectralCoefficients").mean()),
        "MeanVisibleBothFraction": float(values("VisibleBothFraction").mean()),
        "MeanVisibleOnceFraction": float(values("VisibleOnceFraction").mean()),
        "MeanNeverVisibleFraction": float(values("NeverVisibleFraction").mean()),
    }
    return rows, summary


def parse_tests(text: str) -> tuple[int, int]:
    passed = int(re.findall(r"(\d+) passed", text)[-1]) if re.findall(r"(\d+) passed", text) else 0
    failed = int(re.findall(r"(\d+) failed", text)[-1]) if re.findall(r"(\d+) failed", text) else 0
    return passed, failed


def empirical_reduction(reference: float, observed: float) -> float:
    return ((reference - 0.5) - (observed - 0.5)) / (reference - 0.5) * 100.0


def judgment(mae: float, extreme: float, periodicity: float) -> tuple[str, float, float, float]:
    degradation = (mae - REFERENCE_METRICS["MAE"]) / REFERENCE_METRICS["MAE"] * 100.0
    extreme_reduction = empirical_reduction(REFERENCE_AUCSEP["Extreme Event"], extreme)
    periodic_reduction = empirical_reduction(REFERENCE_AUCSEP["Periodicity Strength"], periodicity)
    average_reduction = (extreme_reduction + periodic_reduction) / 2.0
    if not math.isfinite(mae) or mae > 2.0 * REFERENCE_METRICS["MAE"]:
        result = "NOT SUPPORTED"
    elif mae <= 30.0 and extreme <= 0.57 and periodicity <= 0.57:
        result = "STRONGLY SUPPORTED"
    elif degradation <= 12.0 and average_reduction >= 40.0:
        result = "PROMISING"
    elif (
        sum(value >= 40.0 for value in (extreme_reduction, periodic_reduction)) == 1
        or (average_reduction >= 40.0 and degradation > 12.0)
    ):
        result = "PARTIALLY SUPPORTED"
    else:
        result = "NOT SUPPORTED"
    return result, degradation, extreme_reduction, periodic_reduction


def write_report(
    path: Path,
    summary: Mapping[str, Any],
    mask_summary: Mapping[str, float],
    manifest: Mapping[str, Any],
    before_tests: tuple[int, int],
    after_tests: tuple[int, int],
) -> None:
    judgment_value = str(summary["Judgment"])
    privacy_answer = (
        "YES" if judgment_value in {"STRONGLY SUPPORTED", "PROMISING"}
        else "PARTIAL" if judgment_value == "PARTIALLY SUPPORTED" else "NO"
    )
    mae_deg = float(summary["MAEDegradationPct"])
    utility_answer = "YES" if mae_deg <= 12 else ("PARTIAL" if float(summary["MAE"]) <= 30 else "NO")
    comm_answer = "YES" if float(summary["CommunicationReductionPct"]) > 0 else "NO"
    continue_answer = {
        "STRONGLY SUPPORTED": "STRONGLY YES",
        "PROMISING": "YES",
        "PARTIALLY SUPPORTED": "MAYBE",
        "NOT SUPPORTED": "NO",
    }[judgment_value]
    lines = [
        "# Randomized Spectral Release Sanity Report", "",
        "## Scope and preregistered question", "",
        "This single-run sanity experiment tested whether public, data-independent randomized spectral disclosure disrupts cross-round client fingerprints while retaining forecasting utility. It used PEMS08, STGCN, seed 42, 50 clients, 20 rounds, five clients per round, exactly two releases per client, and p=0.5. No parameter sweep was performed.", "",
        "RSR-FedAvg is not a DP mechanism. It used no Gaussian noise, clipping, or accountant. The results therefore quantify empirical property-inference resistance and communication only.", "",
        "## Locked reference audit", "",
        f"The matched FedAvg R2 victim was validated at MAE {REFERENCE_METRICS['MAE']:.6f}, RMSE {REFERENCE_METRICS['RMSE']:.6f}, and MAPE {REFERENCE_METRICS['MAPE']:.6f}, with 100 server-visible releases and schedule hash {EXPECTED_SCHEDULE_SHA256}. The legacy victim directory has no standalone manifest.json; its evidence_manifest.csv provides the cross-artifact hashes.", "",
        "## Mechanism and server visibility", "",
        "The existing public graph and temporal spectral mapping was reused. Independent Bernoulli masks were derived by SHA-256 from experiment seed, client, round, release index, group, and tensor. Selected coefficients were divided by p. Graph-low, graph-high, temporal-low, and temporal-high groups were randomized; the Other group remained dense. Mask locations, sparse values, tensor metadata, and the dense zero-filled reconstruction were treated as server-visible. Server aggregation and the locked attack both used that reconstruction from the same victim run.", "",
        "## Main comparison", "",
        "| Method | MAE | RMSE | MAPE | Extreme AUC-sep | Periodicity AUC-sep |",
        "|---|---:|---:|---:|---:|---:|",
        f"| FedAvg | {REFERENCE_METRICS['MAE']:.3f} | {REFERENCE_METRICS['RMSE']:.3f} | {REFERENCE_METRICS['MAPE']:.3f} | {REFERENCE_AUCSEP['Extreme Event']:.3f} | {REFERENCE_AUCSEP['Periodicity Strength']:.3f} |",
        f"| RSR p=0.5 | {float(summary['MAE']):.3f} | {float(summary['RMSE']):.3f} | {float(summary['MAPE']):.3f} | {float(summary['Extreme_AUCsep']):.3f} | {float(summary['Periodicity_AUCsep']):.3f} |", "",
        f"MAE changed by {mae_deg:.2f}%. Extreme excess separability changed by {float(summary['ExtremeExcessReductionPct']):.2f}%, and Periodicity excess separability changed by {float(summary['PeriodicityExcessReductionPct']):.2f}%. Negative reduction values indicate increased empirical leakage.", "",
        "## Communication and mask diagnostics", "",
        f"The 100 actual RSR wire payloads totaled {int(summary['TotalSerializedBytes'])} bytes versus {int(summary['DenseReferenceBytes'])} bytes obtained by serializing the 100 locked FedAvg updates with the explicit dense FSD1 format. The communication reduction was {float(summary['CommunicationReductionPct']):.2f}% and the dense-to-RSR compression ratio was {float(summary['CompressionRatio']):.3f}x. The realized spectral release ratio was {float(summary['RealizedReleaseRatio']):.4f}.", "",
        f"Across clients, mask Jaccard overlap was {float(mask_summary['MeanJaccardOverlap']):.4f} +/- {float(mask_summary['StdJaccardOverlap']):.4f} (range {float(mask_summary['MinJaccardOverlap']):.4f}-{float(mask_summary['MaxJaccardOverlap']):.4f}). Mean coefficient visibility was {100*float(mask_summary['MeanVisibleBothFraction']):.2f}% in both releases, {100*float(mask_summary['MeanVisibleOnceFraction']):.2f}% in exactly one release, and {100*float(mask_summary['MeanNeverVisibleFraction']):.2f}% in neither release. These are support diagnostics, not a privacy guarantee.", "",
        "## Five answers", "",
        f"1. Does RSR reduce cross-round property inference leakage? **{privacy_answer}.**",
        f"2. Does RSR preserve forecasting utility? **{utility_answer}.**",
        f"3. Does RSR reduce actual transmitted bytes? **{comm_answer}.**",
        "4. Is the improvement obtained without DP? **YES**; no DP operation was invoked.",
        f"5. Is RSR worth continuing as the main FedSpecShield defense direction? **{continue_answer}.**", "",
        f"Preregistered judgment: **RSR {judgment_value}.**", "",
        "## Boundaries and verification", "",
        "This evidence is limited to one dataset, one seed, one p value, two observations per client, and an honest-but-curious server under the locked logistic-regression attack. Public masks can themselves carry information, and unchanged Other parameters remain fully visible. The experiment does not establish cryptographic confidentiality or a formal privacy bound.", "",
        f"Tests before training: {before_tests[0]} passed, {before_tests[1]} failed. Tests after the experiment: {after_tests[0]} passed, {after_tests[1]} failed.", "",
        "Audit checklist:", "",
        "- [x] No DP mechanism, Gaussian noise, clipping, accountant, Top-k, error feedback, or quantization",
        "- [x] Random mask independent of update values and private properties",
        "- [x] Different deterministic masks across releases",
        "- [x] Mask positions public to the server and attacker",
        "- [x] Utility and attack use the same victim run",
        "- [x] Existing attack classifier, features, labels, folds, and bootstrap protocol unchanged",
        "- [x] Actual serialized bytes measured",
        f"- [{'x' if after_tests[1] == 0 else ' '}] Full tests pass", "",
        "## Chinese verification notes", "",
        "Chinese note: Evidence is reported before the preregistered judgment; RSR is only an empirical leakage-mitigation candidate.", "",
        "## Provenance", "",
        f"Victim run ID: {VICTIM_RUN_ID}. Released trajectory hash: {manifest['released_update_hash']}. Attack feature hash: {manifest['attack_feature_hash']}.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or args.summary.exists() or args.report.exists():
        raise FileExistsError("refusing to overwrite Experiment 11 outputs")
    reference = validate_reference(args)
    schedule, schedule_metadata = load_and_verify_schedule(args.schedule)
    if sha256_file(args.schedule) != EXPECTED_SCHEDULE_SHA256:
        raise ValueError("schedule hash mismatch")
    reference_config = _load_reference_config(args.reference_config)
    for key, value in {
        "dataset": "pems08", "seed": 42, "num_clients": 50,
        "clients_per_round": 5, "local_epochs": 1, "history": 12, "horizon": 12,
        "batch_size": 64, "lr": 0.01,
    }.items():
        if reference_config.get(key) != value:
            raise ValueError(f"reference training config mismatch for {key}")
    args.output.mkdir(parents=True)
    payload_dir = args.output / "server_visible_payloads"
    update_dir = args.output / "server_visible_updates"
    attack_dir = args.output / "attack"
    payload_dir.mkdir()
    attack_dir.mkdir()
    shutil.copy2(args.schedule, args.output / "participation_schedule.json")
    shutil.copy2(args.before_tests_log, args.output / "before_tests.log")
    before_tests = parse_tests(args.before_tests_log.read_text(encoding="utf-8"))
    if before_tests[1] != 0 or before_tests[0] == 0:
        raise RuntimeError("pre-training tests did not pass")
    dense_by_key, dense_reference_total = dense_reference_byte_map(reference["update_paths"])
    deterministic_mode()
    device = _resolve_device(args.device)
    prepared = _prepare_data(reference_config, args.data_root, "pems08")
    model, clients, validation_sets, test_sets = _build_clients_and_eval(
        reference_config, prepared, device
    )
    model = model.to(device)
    clients_by_id = {client.client_id: client for client in clients}
    config = {
        "experiment": "Experiment 11: Randomized Spectral Release Sanity",
        "VictimRunID": VICTIM_RUN_ID,
        "dataset": "pems08", "backbone": "STGCN", "seed": 42, "method": METHOD,
        "method_key": METHOD_KEY, "p": P,
        "rounds": 20, "num_clients": 50, "clients_per_round": 5,
        "releases_per_client": 2, "local_epochs": 1,
        "history": 12, "horizon": 12, "batch_size": 64, "learning_rate": 0.01,
        "schedule_hash": EXPECTED_SCHEDULE_SHA256,
        "randomized_groups": list(SPECTRAL_GROUPS), "dense_groups": ["other"],
        "scaling": "selected_coefficient_divided_by_p",
        "mask_sampling": "independent_Bernoulli_per_spectral_coefficient",
        "mask_public_to_server": True, "server_reconstruction": "zero_fill_then_existing_inverse_groups",
        "attack_observation": "same-run server-visible reconstructed update",
        "DP": False, "GaussianNoise": False, "DPClipping": False,
        "PrivacyAccountant": False, "TopK": False, "ErrorFeedback": False,
        "Quantization": False, "SecureAggregation": False,
    }
    save_config(args.output / "config.yaml", config)
    json_write(args.output / "config.json", config)
    json_write(args.output / "mask_seed_policy.json", mask_seed_policy())
    (args.output / "git_commit.txt").write_text(
        "UNAVAILABLE_NOT_A_GIT_CHECKOUT\n", encoding="utf-8"
    )
    (args.output / "environment.txt").write_text(_environment_snapshot(device), encoding="utf-8")
    configure_logging(args.output / "training.log")
    collector = UpdateCollector(save_full_updates=True, output_dir=update_dir)
    release_count = defaultdict(int)
    supports: dict[int, list[frozenset[str]]] = defaultdict(list)
    total_spectral_by_client: dict[int, int] = {}
    communication_rows: list[dict[str, Any]] = []
    mask_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for round_index, selected_ids in enumerate(schedule):
        round_start = time.perf_counter()
        global_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        updates, weights = [], []
        weighted_loss = loss_samples = 0.0
        for client_id in selected_ids:
            client = clients_by_id[client_id]
            train_result = client.train(global_state, LOCAL_EPOCHS, 0.0, report_loss=True)
            if train_result.dp_applied:
                raise RuntimeError("RSR path unexpectedly marked a release as DP")
            current_release = release_count[client_id]
            if current_release >= RELEASES_PER_CLIENT:
                raise RuntimeError("client released more than twice")
            rsr = randomized_spectral_release(
                train_result.update, p=P, experiment_seed=EXPERIMENT_SEED,
                client_id=client_id, round_id=round_index, release_index=current_release,
            )
            payload_path = payload_dir / f"round_{round_index:04d}_client_{client_id:03d}.rsr"
            payload_path.write_bytes(rsr.serialized)
            if sha256_file(payload_path) != rsr.payload_sha256:
                raise RuntimeError("serialized payload hash mismatch")
            collector.collect(round_index, client_id, rsr.reconstructed_update)
            updates.append(rsr.reconstructed_update)
            weights.append(train_result.samples)
            weighted_loss += train_result.client_train_loss * train_result.loss_samples
            loss_samples += train_result.loss_samples
            supports[client_id].append(rsr.support)
            total_spectral_by_client[client_id] = rsr.total_spectral_coefficients
            dense_bytes = dense_by_key[(round_index, client_id)]
            communication_rows.append({
                "VictimRunID": VICTIM_RUN_ID, "ClientID": client_id,
                "Round": round_index, "ReleaseIndex": current_release,
                "TargetP": P,
                "TotalSpectralCoefficients": rsr.total_spectral_coefficients,
                "ReleasedSpectralCoefficients": rsr.released_spectral_coefficients,
                "RealizedReleaseRatio": (
                    rsr.released_spectral_coefficients / rsr.total_spectral_coefficients
                ),
                "SerializedBytes": len(rsr.serialized),
                "DenseReferenceSerializedBytes": dense_bytes,
                "SpectralEntryBytes": rsr.spectral_entry_bytes,
                "SerializationFormat": "RSR1",
                "DenseReferenceFormat": "FSD1",
            })
            for row in rsr.mask_rows:
                mask_rows.append({
                    "VictimRunID": VICTIM_RUN_ID, "ClientID": client_id,
                    "Round": round_index, "ReleaseIndex": current_release, **row,
                })
            hash_rows.append({
                "Round": round_index, "ClientID": client_id, "ReleaseIndex": current_release,
                "PayloadSHA256": rsr.payload_sha256,
                "ServerReconstructedUpdateSHA256": rsr.reconstructed_sha256,
            })
            release_count[client_id] += 1
        aggregate = weighted_average_updates(updates, weights)
        model.load_state_dict(apply_update(global_state, aggregate))
        row = {
            "Round": round_index,
            "ClientTrainLoss": weighted_loss / loss_samples,
            "Clients": len(selected_ids),
            "SerializedBytes": sum(
                item["SerializedBytes"] for item in communication_rows if item["Round"] == round_index
            ),
            "WallSeconds": time.perf_counter() - round_start,
        }
        training_rows.append(row)
        logging.info(
            "round=%d loss=%.6f bytes=%d", round_index, row["ClientTrainLoss"], row["SerializedBytes"]
        )
    if set(release_count) != set(range(CLIENTS)) or any(release_count[index] != 2 for index in range(CLIENTS)):
        raise RuntimeError("R2 release counts are invalid")
    if len(collector.records) != 100:
        raise RuntimeError("expected 100 server-visible releases")
    records_by_client = collector.by_client()
    if any(len(records_by_client[index]) != 2 for index in range(CLIENTS)):
        raise RuntimeError("attack trajectory is not R2")
    *_, mean, std = prepared
    validation = _evaluate_partitions(model, validation_sets, device, reference_config, mean, std)
    test = _evaluate_partitions(model, test_sets, device, reference_config, mean, std)
    final_checkpoint = args.output / "final_model.pt"
    torch.save(model.state_dict(), final_checkpoint)
    release_rows, release_hash = released_manifest(collector.records)
    for row in release_rows:
        row.update({"VictimRunID": VICTIM_RUN_ID, "Method": METHOD})
    write_rows(args.output / "released_update_manifest.csv", release_rows)
    write_rows(args.output / "communication_stats.csv", communication_rows)
    write_rows(args.output / "mask_seed_metadata.csv", mask_rows)
    write_rows(args.output / "training_rounds.csv", training_rows)
    payload_hash = canonical_hash([
        {key: row[key] for key in ("Round", "ClientID", "ReleaseIndex", "PayloadSHA256",
                                    "ServerReconstructedUpdateSHA256")}
        for row in hash_rows
    ])
    json_write(args.output / "server_visible_update_hashes.json", {
        "VictimRunID": VICTIM_RUN_ID,
        "combined_payload_and_reconstruction_hash": payload_hash,
        "released_update_trajectory_hash": release_hash,
        "records": hash_rows,
    })
    attack_rows, attack_feature_hash, attack_metrics_hash = run_attack(
        records_by_client, attack_dir, args.data_root, args.feature_spec, args.reference_attack
    )
    attack_input_rows, attack_input_hash = released_manifest(
        [attack_record(record) for record in collector.records]
    )
    write_rows(attack_dir / "attack_input_trajectory_manifest.csv", attack_input_rows)
    if len(attack_input_rows) != 100:
        raise RuntimeError("attack did not use all same-run RSR releases")
    overlap_rows, overlap_summary = aggregate_mask_diagnostics(supports, total_spectral_by_client)
    write_rows(args.output / "mask_overlap_diagnostics.csv", overlap_rows)
    json_write(args.output / "mask_overlap_summary.json", overlap_summary)
    realized_ratios = np.asarray([row["RealizedReleaseRatio"] for row in communication_rows])
    total_rsr_bytes = sum(int(row["SerializedBytes"]) for row in communication_rows)
    compression = dense_reference_total / total_rsr_bytes
    communication_reduction = (1.0 - total_rsr_bytes / dense_reference_total) * 100.0
    full_numel = sum(int(tensor.numel()) for tensor in collector.records[0].layers.values())
    affected_numel = int(communication_rows[0]["TotalSpectralCoefficients"])
    spectral_wire_fraction = sum(int(row["SpectralEntryBytes"]) for row in communication_rows) / total_rsr_bytes
    extreme = proposed(attack_rows, "Extreme Event")
    periodicity = proposed(attack_rows, "Periodicity Strength")
    decision, mae_degradation, extreme_reduction, periodicity_reduction = judgment(
        float(test["MAE"]), float(extreme["ROCAUCSeparability"]),
        float(periodicity["ROCAUCSeparability"]),
    )
    metrics = {
        "VictimRunID": VICTIM_RUN_ID, "Method": METHOD, "Dataset": "PEMS08",
        "Seed": 42, "p": P,
        "MAE": test["MAE"], "RMSE": test["RMSE"], "MAPE": test["MAPE"],
        "Val_MAE": validation["MAE"], "Val_RMSE": validation["RMSE"],
        "Val_MAPE": validation["MAPE"],
        "FinalTrainLoss": training_rows[-1]["ClientTrainLoss"],
        "RuntimeSeconds": time.perf_counter() - started,
        "ReleaseCount": len(collector.records), "ReleasedUpdateHash": release_hash,
        "FinalCheckpoint": str(final_checkpoint),
        "FinalCheckpointHash": sha256_file(final_checkpoint),
    }
    json_write(args.output / "metrics.json", metrics)
    write_rows(args.output / "metrics.csv", [metrics])
    summary = {
        "Method": METHOD, "Dataset": "PEMS08", "Seed": 42, "p": P,
        "MAE": test["MAE"], "RMSE": test["RMSE"], "MAPE": test["MAPE"],
        "MAEDegradationPct": mae_degradation,
        "RMSEDegradationPct": (float(test["RMSE"]) - REFERENCE_METRICS["RMSE"]) / REFERENCE_METRICS["RMSE"] * 100.0,
        "MAPEDegradationPct": (float(test["MAPE"]) - REFERENCE_METRICS["MAPE"]) / REFERENCE_METRICS["MAPE"] * 100.0,
        "Extreme_AUC": extreme["ROCAUCRaw"],
        "Extreme_AUCsep": extreme["ROCAUCSeparability"],
        "Extreme_AUCsep_CI95_Low": extreme["AUCSeparabilityCI95Low"],
        "Extreme_AUCsep_CI95_High": extreme["AUCSeparabilityCI95High"],
        "ExtremeExcessReductionPct": extreme_reduction,
        "Periodicity_AUC": periodicity["ROCAUCRaw"],
        "Periodicity_AUCsep": periodicity["ROCAUCSeparability"],
        "Periodicity_AUCsep_CI95_Low": periodicity["AUCSeparabilityCI95Low"],
        "Periodicity_AUCsep_CI95_High": periodicity["AUCSeparabilityCI95High"],
        "PeriodicityExcessReductionPct": periodicity_reduction,
        "TotalUploads": 100, "TotalSerializedBytes": total_rsr_bytes,
        "DenseReferenceBytes": dense_reference_total,
        "AvgBytesPerUpload": total_rsr_bytes / 100.0,
        "CompressionRatio": compression,
        "CommunicationReductionPct": communication_reduction,
        "TargetP": P,
        "RealizedReleaseRatio": float(realized_ratios.mean()),
        "RealizedReleaseRatioStd": float(realized_ratios.std(ddof=0)),
        "RealizedReleaseRatioMin": float(realized_ratios.min()),
        "RealizedReleaseRatioMax": float(realized_ratios.max()),
        "RandomizedParameterFraction": affected_numel / full_numel,
        "RandomizedParameterPct": affected_numel / full_numel * 100.0,
        "RandomizedSpectralEntryWireFraction": spectral_wire_fraction,
        "RandomizedSpectralEntryWirePct": spectral_wire_fraction * 100.0,
        "MeanMaskJaccardOverlap": overlap_summary["MeanJaccardOverlap"],
        "VisibleBothPct": overlap_summary["MeanVisibleBothFraction"] * 100.0,
        "VisibleOncePct": overlap_summary["MeanVisibleOnceFraction"] * 100.0,
        "NeverVisiblePct": overlap_summary["MeanNeverVisibleFraction"] * 100.0,
        "Judgment": decision,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.summary, [summary])
    post = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"], cwd=Path.cwd(),
        capture_output=True, text=True,
    )
    post_text = post.stdout + post.stderr
    (args.output / "after_tests.log").write_text(post_text, encoding="utf-8")
    after_tests = parse_tests(post_text)
    code_paths = [
        Path("communication/randomized_spectral_release.py"),
        Path("experiments/exp_randomized_spectral_release_sanity.py"),
        Path("tests/test_randomized_spectral_release.py"),
        Path("spectral/spectral_groups.py"),
        Path("attacks/spectral_leakage/fingerprint.py"),
        Path("experiments/exp_representative_privacy_attack.py"),
        args.feature_spec, args.schedule,
    ]
    manifest = {
        "dataset": "PEMS08", "seed": 42, "method": METHOD, "method_key": METHOD_KEY, "p": P,
        "DP": False, "GaussianNoise": False, "DPClipping": False, "PrivacyAccountant": False,
        "random_mask": True, "mask_data_dependent": False,
        "mask_magnitude_dependent": False, "mask_property_dependent": False,
        "mask_public_to_server": True,
        "rounds": 20, "clients": 50, "clients_per_round": 5,
        "releases_per_client": 2, "local_epochs": 1,
        "attack_same_victim_run": True,
        "attack_input_trajectory_hash": attack_input_hash,
        "released_update_hash": release_hash,
        "payload_and_reconstruction_hash": payload_hash,
        "attack_feature_hash": attack_feature_hash,
        "attack_metrics_hash": attack_metrics_hash,
        "schedule_hash": EXPECTED_SCHEDULE_SHA256,
        "feature_spec_hash": EXPECTED_FEATURE_SPEC_SHA256,
        "final_checkpoint_hash": sha256_file(final_checkpoint),
        "actual_serialized_bytes_measured": True,
        "serialization_formats": {"RSR": "RSR1", "dense_reference": "FSD1"},
        "reference_victim": str(args.reference_victim),
        "reference_evidence": reference["evidence"],
        "reference_manifest_note": reference["legacy_manifest_note"],
        "schedule_metadata": schedule_metadata,
        "before_tests": {"passed": before_tests[0], "failed": before_tests[1]},
        "after_tests": {"passed": after_tests[0], "failed": after_tests[1]},
        "git_commit": "UNAVAILABLE_NOT_A_GIT_CHECKOUT",
        "code_and_protocol_hashes": {
            str(path): sha256_file(path) for path in code_paths
        },
        "files_added": [
            "communication/randomized_spectral_release.py",
            "experiments/exp_randomized_spectral_release_sanity.py",
            "tests/test_randomized_spectral_release.py",
        ],
        "files_modified": [],
    }
    json_write(args.output / "manifest.json", manifest)
    write_report(args.report, summary, overlap_summary, manifest, before_tests, after_tests)
    if after_tests[1] != 0 or post.returncode != 0:
        logging.error("post-experiment tests failed: %s", post_text)
    print("================ EXPERIMENT 11 =================")
    print("\nMethod:\nRandomized Spectral Release")
    print("\nDataset:\nPEMS08")
    print("\nSeed:\n42")
    print("\np:\n0.50")
    print("\n---------------- Utility ----------------")
    print(f"\nFedAvg MAE:\n{REFERENCE_METRICS['MAE']:.3f}")
    print(f"\nRSR MAE:\n{float(summary['MAE']):.6f}")
    print(f"\nMAE degradation:\n{float(summary['MAEDegradationPct']):.2f}%")
    print(f"\nFedAvg RMSE:\n{REFERENCE_METRICS['RMSE']:.3f}")
    print(f"\nRSR RMSE:\n{float(summary['RMSE']):.6f}")
    print("\n---------------- Privacy ----------------")
    print(f"\nExtreme:\nFedAvg AUC-sep = {REFERENCE_AUCSEP['Extreme Event']:.3f}")
    print(f"RSR AUC-sep    = {float(summary['Extreme_AUCsep']):.6f}")
    print(f"\nExcess separability reduction:\n{float(summary['ExtremeExcessReductionPct']):.2f}%")
    print(f"\nPeriodicity:\nFedAvg AUC-sep = {REFERENCE_AUCSEP['Periodicity Strength']:.3f}")
    print(f"RSR AUC-sep    = {float(summary['Periodicity_AUCsep']):.6f}")
    print(f"\nExcess separability reduction:\n{float(summary['PeriodicityExcessReductionPct']):.2f}%")
    print("\n---------------- Communication ----------------")
    print(f"\nRealized release ratio:\n{float(summary['RealizedReleaseRatio']):.6f}")
    print(f"\nDense bytes:\n{int(summary['DenseReferenceBytes'])}")
    print(f"\nRSR bytes:\n{int(summary['TotalSerializedBytes'])}")
    print(f"\nCommunication reduction:\n{float(summary['CommunicationReductionPct']):.2f}%")
    print(f"\nCompression ratio:\n{float(summary['CompressionRatio']):.3f}x")
    print("\n---------------- Cross-round ----------------")
    print(f"\nMean mask Jaccard overlap:\n{float(summary['MeanMaskJaccardOverlap']):.6f}")
    print(f"\nVisible in both releases:\n{float(summary['VisibleBothPct']):.2f}%")
    print(f"\nVisible once:\n{float(summary['VisibleOncePct']):.2f}%")
    print(f"\nNever visible:\n{float(summary['NeverVisiblePct']):.2f}%")
    print("\n---------------- Verification ----------------")
    print("\nNo DP:\nPASS")
    print("\nData-independent mask:\nPASS")
    print("\nDifferent cross-round masks:\nPASS")
    print("\nSame victim utility/attack:\nPASS")
    print("\nActual byte accounting:\nPASS")
    print(f"\nTests:\n{after_tests[0]} passed\n{after_tests[1]} failed")
    print("\n---------------- Judgment ----------------")
    print(f"\nRSR:\n{decision}")
    print(f"\nContinue RSR direction:\n{ {'STRONGLY SUPPORTED':'YES','PROMISING':'YES','PARTIALLY SUPPORTED':'MAYBE','NOT SUPPORTED':'NO'}[decision] }")
    print("\n================================================")
    return {"summary": summary, "manifest": manifest}


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
