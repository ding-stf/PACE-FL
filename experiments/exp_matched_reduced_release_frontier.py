from __future__ import annotations

"""Experiment 8: matched two-release PJSDP versus DP-FedAvg frontier."""

import argparse
import csv
import hashlib
import json
import logging
import math
import secrets
import time
from collections import Counter
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

from experiments.common import configure_logging, save_config, write_rows
from experiments.exp_ellipsoidal_clipping_sanity import (
    _build_clients_and_eval,
    _environment_snapshot,
    _evaluate_partitions,
    _prepare_data,
)
from experiments.exp_privacy_utility_frontier import DPFEDAVG_CLIP, DiagnosticDPFedAvg
from experiments.exp_public_ellipsoid_dp_sanity import PublicGeometryReleaseMechanism
from experiments.exp_spectral_norm_audit import _load_reference_config, _resolve_device, _set_seed
from federated.stg_server import STGServer
from privacy.accountant import RDPAccountant
from privacy.fixed_spectral_projection import FixedSpectralProjection
from privacy.pjsdp_accountant import PJSDPAccountant
from spectral.spectral_groups import GROUPS


EPSILONS = (2048, 1024)
ROUNDS = 20
CLIENTS = 50
CLIENTS_PER_ROUND = 5
RELEASES_PER_CLIENT = 2
LOCAL_EPOCHS = 1
DELTA = 1e-5
ADJACENCY = "replacement"
EXPECTED_SCHEDULE_SHA256 = "fc9a11ab8f15c964b01accbafeed109146be8fe4ccbc155fa01281f8d9ee5399"
PJS4096_MAE = 29.45579624418229
PJS4096_RMSE = 40.97786622055441


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--reference_config", type=Path, required=True)
    p.add_argument("--exp7_root", type=Path, required=True)
    p.add_argument("--pjsdp_4096_summary", type=Path, required=True)
    p.add_argument("--results_root", type=Path, default=Path("results/matched_reduced_release_frontier"))
    p.add_argument("--report_path", type=Path, default=Path("docs/MATCHED_REDUCED_RELEASE_FRONTIER_REPORT.md"))
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
        raise ValueError(f"expected exactly one row in {path}")
    return rows[0]


def _read_many(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "Mean": float(np.mean(array)),
        "Median": float(np.median(array)),
        "P90": float(np.percentile(array, 90)),
    }


def load_and_verify_schedule(path: Path) -> tuple[list[list[int]], dict[str, Any]]:
    file_hash = sha256_file(path)
    if file_hash != EXPECTED_SCHEDULE_SHA256:
        raise ValueError(f"schedule SHA-256 mismatch: {file_hash}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schedule = [[int(client_id) for client_id in row] for row in payload["rounds_client_ids"]]
    counts = Counter(client_id for row in schedule for client_id in row)
    if len(schedule) != ROUNDS or any(len(row) != CLIENTS_PER_ROUND for row in schedule):
        raise ValueError("schedule must contain 20 rounds with five distinct clients per round")
    if any(len(row) != len(set(row)) for row in schedule):
        raise ValueError("a client is repeated within a schedule round")
    if set(counts) != set(range(CLIENTS)) or any(counts[client_id] != RELEASES_PER_CLIENT for client_id in range(CLIENTS)):
        raise ValueError("schedule must contain every client exactly twice")
    if payload.get("private_data_used") is not False:
        raise ValueError("schedule is not marked data-independent")
    return schedule, payload


def calibrate_dpfedavg(target_epsilon: float) -> dict[str, float]:
    accounting_multiplier = RDPAccountant.calibrate_noise(
        float(target_epsilon), DELTA, RELEASES_PER_CLIENT, groups=1
    )
    verifier = RDPAccountant()
    verifier.step(accounting_multiplier, groups=1, steps=RELEASES_PER_CLIENT)
    achieved = verifier.epsilon(DELTA)
    if not math.isclose(achieved, float(target_epsilon), abs_tol=1e-8):
        raise RuntimeError("DP-FedAvg accountant did not match target epsilon")
    sensitivity = 2.0 * DPFEDAVG_CLIP
    return {
        "accounting_multiplier": accounting_multiplier,
        "applied_multiplier": 2.0 * accounting_multiplier,
        "noise_std": sensitivity * accounting_multiplier,
        "sensitivity": sensitivity,
        "achieved_epsilon": achieved,
    }


def calibrate_pjsdp(config: Mapping[str, Any], target_epsilon: float) -> dict[str, Any]:
    bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
    weights = {group: float(config[f"{group}_noise_weight"]) for group in GROUPS}
    sensitivities = {group: 2.0 * bounds[group] for group in GROUPS}
    result = PJSDPAccountant.calibrate_global_scale(
        float(target_epsilon), DELTA, RELEASES_PER_CLIENT, sensitivities, weights,
        sampling_rate=0.1, sampling_amplification=False,
    )
    verifier = PJSDPAccountant()
    verifier.step(
        sensitivities, result.group_noise_stds, rounds=RELEASES_PER_CLIENT,
        sampling_rate=0.1, sampling_amplification=False,
    )
    achieved = verifier.epsilon(DELTA)
    if not math.isclose(achieved, float(target_epsilon), abs_tol=1e-8):
        raise RuntimeError("PJSDP accountant did not match target epsilon")
    return {
        "global_noise_scale": result.global_scale,
        "group_noise_stds": result.group_noise_stds,
        "sensitivities": sensitivities,
        "achieved_epsilon": achieved,
    }


def _accounting_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epsilon in EPSILONS:
        pjsdp = calibrate_pjsdp(config, epsilon)
        dpf = calibrate_dpfedavg(epsilon)
        rows.extend([
            {
                "Method": "Current PJSDP", "TargetEpsilon": epsilon,
                "AchievedEpsilon": pjsdp["achieved_epsilon"], "Delta": DELTA,
                "ReleaseCount": RELEASES_PER_CLIENT, "Adjacency": ADJACENCY,
                "SubsamplingAmplification": "NO", "NoiseScale": pjsdp["global_noise_scale"],
                "Sensitivity": yaml.safe_dump(pjsdp["sensitivities"], sort_keys=False).strip().replace("\n", "; "),
                "CompositionMethod": "joint anisotropic Gaussian RDP; bounded sequential composition; no amplification",
            },
            {
                "Method": "DP-FedAvg", "TargetEpsilon": epsilon,
                "AchievedEpsilon": dpf["achieved_epsilon"], "Delta": DELTA,
                "ReleaseCount": RELEASES_PER_CLIENT, "Adjacency": ADJACENCY,
                "SubsamplingAmplification": "NO", "NoiseScale": dpf["noise_std"],
                "Sensitivity": dpf["sensitivity"],
                "CompositionMethod": "full-update Gaussian RDP; bounded sequential composition; no amplification",
            },
        ])
    return rows


def verify_exp7_reference(exp7_root: Path, pjsdp_4096_summary: Path) -> dict[str, Any]:
    schedule_path = exp7_root / "participation_schedule_2release.json"
    schedule, schedule_payload = load_and_verify_schedule(schedule_path)
    run_dir = exp7_root / "seed_42" / "r2_e"
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    metrics = _read_one(run_dir / "metrics.csv")
    privacy = _read_one(run_dir / "privacy_summary.csv")
    diagnostics = _read_many(run_dir / "dp_diagnostics.csv")
    communication = _read_one(run_dir / "communication_summary.csv")
    required = {
        "dataset": "pems08", "seed": 42, "rounds": ROUNDS,
        "clients_per_round": CLIENTS_PER_ROUND, "releases_per_client": RELEASES_PER_CLIENT,
        "local_epochs": LOCAL_EPOCHS, "sampling_amplification_used": False,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Experiment 7 reference mismatch for {key}: {config.get(key)!r}")
    if not math.isclose(float(config["target_epsilon"]), 2048.0, abs_tol=1e-12):
        raise ValueError("Experiment 7 reference epsilon mismatch")
    if not math.isclose(float(metrics["AchievedEpsilon"]), 2048.0, abs_tol=1e-8):
        raise ValueError("Experiment 7 achieved epsilon mismatch")
    if len(diagnostics) != ROUNDS * CLIENTS_PER_ROUND:
        raise ValueError("Experiment 7 diagnostic release count mismatch")
    high_rows = _read_many(pjsdp_4096_summary)
    current = [row for row in high_rows if row.get("Method") == "Current"]
    if len(current) != 1:
        raise ValueError("expected one Current PJSDP epsilon-4096 reference row")
    if not math.isclose(float(current[0]["MAE"]), PJS4096_MAE, abs_tol=1e-9):
        raise ValueError("epsilon-4096 MAE reference mismatch")
    return {
        "schedule": schedule, "schedule_payload": schedule_payload,
        "schedule_path": schedule_path, "schedule_file_sha256": EXPECTED_SCHEDULE_SHA256,
        "run_dir": run_dir, "config_path": config_path, "config": config,
        "metrics": metrics, "privacy": privacy, "diagnostics": diagnostics,
        "communication": communication, "high_utility": current[0],
    }


def _reference_summary(reference: Mapping[str, Any], validation_scale: float) -> dict[str, Any]:
    diagnostics = reference["diagnostics"]
    metrics = reference["metrics"]
    privacy = reference["privacy"]
    nsr = _stats([float(row["noise_to_signal_ratio"]) for row in diagnostics])
    alignment = _stats([float(row["dp_alignment"]) for row in diagnostics])
    distortion = _stats([float(row["clipping_distortion"]) for row in diagnostics])
    return {
        "Method": "Current PJSDP", "Rounds": ROUNDS,
        "ReleasesPerClient": RELEASES_PER_CLIENT, "LocalEpochs": LOCAL_EPOCHS,
        "TargetEpsilon": 2048, "AchievedEpsilon": float(metrics["AchievedEpsilon"]),
        "Delta": DELTA, "NoiseScale": float(privacy["PerReleaseNoiseScale"]),
        "MAE": float(metrics["MAE"]), "RMSE": float(metrics["RMSE"]), "MAPE": float(metrics["MAPE"]),
        "TrainLoss": float(metrics["FinalTrainLoss"]),
        "ValLoss": float(metrics["Val_MAE"]) / validation_scale,
        "MeanNoiseSignalRatio": nsr["Mean"], "MedianNoiseSignalRatio": nsr["Median"], "P90NoiseSignalRatio": nsr["P90"],
        "MeanDPAlignment": alignment["Mean"], "MedianDPAlignment": alignment["Median"], "P90DPAlignment": alignment["P90"],
        "MeanClipDistortion": distortion["Mean"],
        "TotalUploads": int(float(reference["communication"]["TotalUploads"])),
        "TotalBytes": int(float(reference["communication"]["TotalSerializedBytes"])),
        "TrainStable": str(metrics["TrainStable"]), "NaNInfDetected": str(metrics["NaNInfDetected"]),
        "ReferenceOnly": True,
    }


def _config_for_run(
    method: str, epsilon: int, reference_config: Mapping[str, Any],
    schedule_file_sha256: str, calibration: Mapping[str, Any],
) -> dict[str, Any]:
    is_pjsdp = method == "pjsdp"
    return {
        "experiment": "Matched Reduced-Release Privacy-Utility Sanity Test",
        "method": method, "dataset": "pems08", "seed": 42,
        "rounds": ROUNDS, "num_clients": CLIENTS, "clients_per_round": CLIENTS_PER_ROUND,
        "releases_per_client": RELEASES_PER_CLIENT, "local_epochs": LOCAL_EPOCHS,
        "batch_size": int(reference_config["batch_size"]),
        "history": int(reference_config["history"]), "horizon": int(reference_config["horizon"]),
        "hidden_channels": int(reference_config["hidden_channels"]),
        "learning_rate": float(reference_config["lr"]),
        "target_epsilon": epsilon, "achieved_epsilon": calibration["achieved_epsilon"],
        "delta": DELTA, "adjacency": ADJACENCY, "sampling_amplification_used": False,
        "participation_schedule": "Experiment 7 public deterministic fixed schedule",
        "participation_schedule_file_sha256": schedule_file_sha256,
        "projection_ratio": 0.90 if is_pjsdp else None,
        "projection_seed": 42 if is_pjsdp else None,
        "projection_mask_hash": reference_config["projection_mask_hash"] if is_pjsdp else None,
        "clipping_geometry": "current_independent_group_wise" if is_pjsdp else "full_update_l2",
        "clip_parameter": (
            {group: float(reference_config[f"{group}_clip"]) for group in GROUPS}
            if is_pjsdp else DPFEDAVG_CLIP
        ),
        "noise_scale": calibration["global_noise_scale"] if is_pjsdp else calibration["noise_std"],
        "sensitivity": calibration["sensitivities"] if is_pjsdp else calibration["sensitivity"],
        "accounting_status": "VALID", "save_full_updates": False,
        "diagnostics_scope": "private internal evaluation; excluded from DP transcript",
        "attack_executed": False,
    }


def _make_output(root: Path, method: str, epsilon: int, config: Mapping[str, Any]) -> Path:
    name = f"{'pjsdp' if method == 'pjsdp' else 'dp_fedavg'}_r2_eps{epsilon}"
    output = root / "seed_42" / name
    output.mkdir(parents=True, exist_ok=False)
    save_config(output / "config.yaml", dict(config))
    return output


def _run_one(
    *, method: str, epsilon: int, output: Path, run_config: Mapping[str, Any],
    reference_config: Mapping[str, Any], prepared: tuple, schedule: Sequence[Sequence[int]],
    device: torch.device, private_master_seed: int,
) -> dict[str, Any]:
    configure_logging(output / "train.log")
    _set_seed(42)
    model, clients, validation_sets, test_sets = _build_clients_and_eval(reference_config, prepared, device)
    if method == "pjsdp":
        calibration = calibrate_pjsdp(reference_config, epsilon)
        projection = FixedSpectralProjection(model.state_dict(), 0.90, 42)
        if projection.mask_hash != str(reference_config["projection_mask_hash"]):
            raise RuntimeError("PJSDP projection mask differs from the verified reference")
        bounds = {group: float(reference_config[f"{group}_clip"]) for group in GROUPS}
        mechanism: Any = PublicGeometryReleaseMechanism(
            "current", projection, bounds, None, calibration["group_noise_stds"],
            torch.Generator().manual_seed(private_master_seed),
        )
    else:
        calibration = calibrate_dpfedavg(epsilon)
        mechanism = DiagnosticDPFedAvg(
            DPFEDAVG_CLIP, calibration["noise_std"],
            torch.Generator().manual_seed(private_master_seed),
        )
    server = STGServer(
        model, clients, device, 42, collector=None, privacy=mechanism,
        topk_ratio=1.0, error_feedback=False, event_tau=0.0, skip_policy="zero",
        report_client_train_loss=True, participation_schedule=schedule,
    )
    losses: list[float] = []
    communication: list[dict[str, Any]] = []
    started = time.perf_counter()
    for round_index in range(ROUNDS):
        row = server.run_round(round_index, CLIENTS_PER_ROUND, LOCAL_EPOCHS, 0.0)
        loss = float(row["client_train_loss"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite train loss in {method}/eps{epsilon}")
        losses.append(loss)
        communication.append(row)
        logging.info("method=%s epsilon=%d round=%d/%d train_loss=%.8f", method, epsilon, round_index + 1, ROUNDS, loss)
    *_, mean, std = prepared
    validation = _evaluate_partitions(server.model, validation_sets, device, reference_config, mean, std)
    test = _evaluate_partitions(server.model, test_sets, device, reference_config, mean, std)
    validation_scale = float(std[..., 0].mean())
    runtime = time.perf_counter() - started
    if method == "pjsdp":
        clipping_rows = mechanism.clipping_diagnostics
        diagnostics = [{
            **row,
            "pre_clip_norm": clipping_rows[index]["raw_joint_norm"],
            "post_clip_norm": clipping_rows[index]["post_clip_joint_norm"],
            "clipping_factor": clipping_rows[index]["joint_clip_factor"],
            "clipping_distortion": clipping_rows[index]["clipping_distortion"],
        } for index, row in enumerate(mechanism.noise_signal_diagnostics)]
        group_rows = mechanism.group_diagnostics
    else:
        diagnostics = mechanism.diagnostics
        clipping_rows = [{
            "round": row["round"], "client_id": row["client_id"],
            "pre_clip_norm": row["pre_clip_norm"], "post_clip_norm": row["post_clip_norm"],
            "clipping_factor": row["clipping_factor"], "clipping_distortion": row["clipping_distortion"],
        } for row in diagnostics]
        group_rows = []
    nsr = _stats([float(row["noise_to_signal_ratio"]) for row in diagnostics])
    alignment = _stats([float(row["dp_alignment"]) for row in diagnostics])
    distortion = _stats([float(row["clipping_distortion"]) for row in diagnostics])
    metrics = {
        "Method": "Current PJSDP" if method == "pjsdp" else "DP-FedAvg",
        "Rounds": ROUNDS, "ReleasesPerClient": RELEASES_PER_CLIENT, "LocalEpochs": LOCAL_EPOCHS,
        "TargetEpsilon": epsilon, "AchievedEpsilon": calibration["achieved_epsilon"], "Delta": DELTA,
        "MAE": test["MAE"], "RMSE": test["RMSE"], "MAPE": test["MAPE"],
        "Val_MAE": validation["MAE"], "Val_RMSE": validation["RMSE"], "Val_MAPE": validation["MAPE"],
        "TrainLoss": losses[-1], "BestTrainLoss": min(losses),
        "ValLoss": validation["MAE"] / validation_scale,
        "NaNInfDetected": False, "TrainStable": True, "RuntimeSeconds": runtime,
    }
    privacy = {
        "Method": metrics["Method"], "TargetEpsilon": epsilon,
        "AchievedEpsilon": calibration["achieved_epsilon"], "Delta": DELTA,
        "ReleaseCount": RELEASES_PER_CLIENT, "Adjacency": ADJACENCY,
        "SubsamplingAmplification": "NO",
        "NoiseScale": calibration["global_noise_scale"] if method == "pjsdp" else calibration["noise_std"],
        "Sensitivity": yaml.safe_dump(calibration["sensitivities"], sort_keys=False).strip().replace("\n", "; ") if method == "pjsdp" else calibration["sensitivity"],
        "CompositionMethod": "bounded sequential RDP composition; no amplification",
    }
    comm = {
        "Method": metrics["Method"], "TotalUploads": ROUNDS * CLIENTS_PER_ROUND,
        "TotalRawBytes": sum(int(row["raw_parameter_bytes"]) for row in communication),
        "TotalSerializedBytes": sum(int(row["compressed_bytes"]) for row in communication),
        "Rounds": ROUNDS, "ClientsPerRound": CLIENTS_PER_ROUND, "CompressionEnabled": False,
    }
    write_rows(output / "metrics.csv", [metrics])
    write_rows(output / "privacy_summary.csv", [privacy])
    write_rows(output / "dp_diagnostics.csv", diagnostics)
    write_rows(output / "clipping_diagnostics.csv", clipping_rows)
    if group_rows:
        write_rows(output / "group_diagnostics.csv", group_rows)
    write_rows(output / "communication_summary.csv", [comm])
    (output / "environment.txt").write_text(_environment_snapshot(device), encoding="utf-8")
    summary = {
        "Method": metrics["Method"], "Rounds": ROUNDS,
        "ReleasesPerClient": RELEASES_PER_CLIENT, "LocalEpochs": LOCAL_EPOCHS,
        "TargetEpsilon": epsilon, "AchievedEpsilon": calibration["achieved_epsilon"], "Delta": DELTA,
        "NoiseScale": privacy["NoiseScale"], "MAE": test["MAE"], "RMSE": test["RMSE"], "MAPE": test["MAPE"],
        "TrainLoss": metrics["TrainLoss"], "ValLoss": metrics["ValLoss"],
        "MeanNoiseSignalRatio": nsr["Mean"], "MedianNoiseSignalRatio": nsr["Median"], "P90NoiseSignalRatio": nsr["P90"],
        "MeanDPAlignment": alignment["Mean"], "MedianDPAlignment": alignment["Median"], "P90DPAlignment": alignment["P90"],
        "MeanClipDistortion": distortion["Mean"], "TotalUploads": comm["TotalUploads"],
        "TotalBytes": comm["TotalSerializedBytes"], "TrainStable": True,
        "NaNInfDetected": False, "ReferenceOnly": False,
    }
    return {"summary": summary, "metrics": metrics, "privacy": privacy, "diagnostics": diagnostics}


def matched_gain(rows: Sequence[Mapping[str, Any]], epsilon: int) -> dict[str, Any]:
    selected = {str(row["Method"]): row for row in rows if int(float(row["TargetEpsilon"])) == epsilon}
    pjsdp, dpf = selected["Current PJSDP"], selected["DP-FedAvg"]
    gain = lambda metric: 100.0 * (float(dpf[metric]) - float(pjsdp[metric])) / float(dpf[metric])
    return {
        "Epsilon": epsilon,
        "DPFedAvgMAE": dpf["MAE"], "PJSDPMAE": pjsdp["MAE"], "MAEImprovementPercent": gain("MAE"),
        "DPFedAvgRMSE": dpf["RMSE"], "PJSDPRMSE": pjsdp["RMSE"], "RMSEImprovementPercent": gain("RMSE"),
        "DPFedAvgMAPE": dpf["MAPE"], "PJSDPMAPE": pjsdp["MAPE"], "MAPEImprovementPercent": gain("MAPE"),
    }


def stage_a_decision(gain: Mapping[str, Any]) -> str:
    mae_gain = float(gain["MAEImprovementPercent"])
    rmse_gain = float(gain["RMSEImprovementPercent"])
    if mae_gain <= -3.0 and rmse_gain < 0.0:
        return "NO"
    if mae_gain >= 5.0 and rmse_gain > 0.0:
        return "YES"
    return "MARGINAL"


def _headroom(rows: Sequence[Mapping[str, Any]], high: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = [{
        "Epsilon": 4096, "ReleaseCount": 4,
        "MAE": float(high["MAE"]), "RMSE": float(high["RMSE"]),
        "MAEDegradationPercent": 0.0, "RMSEDegradationPercent": 0.0,
        "ReferenceType": "existing high-utility reference",
    }]
    for row in rows:
        if row["Method"] != "Current PJSDP":
            continue
        result.append({
            "Epsilon": int(float(row["TargetEpsilon"])), "ReleaseCount": RELEASES_PER_CLIENT,
            "MAE": row["MAE"], "RMSE": row["RMSE"],
            "MAEDegradationPercent": 100.0 * (float(row["MAE"]) - PJS4096_MAE) / PJS4096_MAE,
            "RMSEDegradationPercent": 100.0 * (float(row["RMSE"]) - PJS4096_RMSE) / PJS4096_RMSE,
            "ReferenceType": "two-release candidate",
        })
    return sorted(result, key=lambda row: int(row["Epsilon"]), reverse=True)


def final_decision(stage_status: str, gains: Sequence[Mapping[str, Any]], headroom: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    if stage_status == "NO":
        return {"elbow": "not established", "conclusion": "SPECTRAL ADVANTAGE NOT SUPPORTED UNDER MATCHED RELEASE COMPOSITION", "next": "Reassess Spectral Contribution Claim"}
    gain1024 = next(row for row in gains if int(row["Epsilon"]) == 1024)
    h1024 = next(row for row in headroom if int(row["Epsilon"]) == 1024)
    spectral1024 = float(gain1024["MAEImprovementPercent"]) >= 5.0 and float(gain1024["RMSEImprovementPercent"]) > 0.0
    mae_deg = float(h1024["MAEDegradationPercent"])
    rmse_deg = float(h1024["RMSEDegradationPercent"])
    if spectral1024 and mae_deg <= 10.0 and rmse_deg <= 10.0:
        return {"elbow": "1024", "conclusion": "EPSILON 1024 REDUCED-RELEASE PJSDP STRONGLY SUPPORTED", "next": "Representative-Point Privacy Attack Evaluation"}
    if spectral1024 and max(mae_deg, rmse_deg) <= 20.0:
        return {"elbow": "2048", "conclusion": "EPSILON 1024 IS A MODERATE PRIVACY-UTILITY POINT", "next": "Representative-Point Privacy Attack Evaluation"}
    if max(mae_deg, rmse_deg) > 20.0:
        return {"elbow": "2048", "conclusion": "EPSILON 2048 IS THE REDUCED-RELEASE ELBOW", "next": "Representative-Point Privacy Attack Evaluation"}
    return {"elbow": "2048", "conclusion": "RELEASE COMPOSITION EXPLAINS MOST UTILITY GAIN", "next": "Reassess Spectral Contribution Claim"}


def _plot_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8, "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _plots(rows: Sequence[Mapping[str, Any]], headroom: Sequence[Mapping[str, Any]], root: Path) -> None:
    _plot_style()
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=False)
    colors = {"Current PJSDP": "#4C78A8", "DP-FedAvg": "#8A949E"}
    for field, filename, ylabel in (
        ("MAE", "matched_release_mae.png", "MAE (lower is better)"),
        ("RMSE", "matched_release_rmse.png", "RMSE (lower is better)"),
        ("MeanNoiseSignalRatio", "matched_release_noise_signal.png", "Mean noise / signal norm"),
    ):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for method in ("Current PJSDP", "DP-FedAvg"):
            method_rows = sorted([row for row in rows if row["Method"] == method], key=lambda row: int(float(row["TargetEpsilon"])))
            x = [int(float(row["TargetEpsilon"])) for row in method_rows]
            y = [float(row[field]) for row in method_rows]
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=6, color=colors[method], label=method)
        ax.set_xlabel("Privacy budget epsilon")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted({int(float(row["TargetEpsilon"])) for row in rows}))
        ax.legend()
        ax.set_title("Matched two-release comparison, PEMS08, seed 42")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    pjs2048 = next(row for row in rows if row["Method"] == "Current PJSDP" and int(float(row["TargetEpsilon"])) == 2048)
    dpf2048 = next(row for row in rows if row["Method"] == "DP-FedAvg" and int(float(row["TargetEpsilon"])) == 2048)
    fig, ax = plt.subplots(figsize=(6.3, 3.8))
    labels = ["PJSDP R4\n2048", "PJSDP R2\n2048", "DP-FedAvg R2\n2048"]
    values = [37.74902946890508, float(pjs2048["MAE"]), float(dpf2048["MAE"])]
    ax.bar(np.arange(3), values, color=["#B0B7BE", "#4C78A8", "#8A949E"])
    ax.set_xticks(np.arange(3), labels)
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Composition gain versus matched spectral gain")
    fig.tight_layout()
    fig.savefig(plot_dir / "composition_vs_spectral_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    h = sorted(headroom, key=lambda row: int(row["Epsilon"]), reverse=True)
    x = np.arange(len(h))
    ax.plot(x, [float(row["MAE"]) for row in h], marker="o", linewidth=2, color="#4C78A8", label="MAE")
    ax.plot(x, [float(row["RMSE"]) for row in h], marker="s", linewidth=2, color="#72A66A", label="RMSE")
    ax.set_xticks(x, [f"epsilon={row['Epsilon']}\nR{row['ReleaseCount']}" for row in h])
    ax.set_ylabel("Forecast error (lower is better)")
    ax.set_title("PJSDP reduced-release privacy headroom")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "reduced_release_privacy_headroom.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _report(
    path: Path, rows: Sequence[Mapping[str, Any]], gains: Sequence[Mapping[str, Any]],
    headroom: Sequence[Mapping[str, Any]], stage_status: str, decision: Mapping[str, str],
    schedule_hash: str,
) -> None:
    by_key = {(row["Method"], int(float(row["TargetEpsilon"]))): row for row in rows}
    lines = [
        "# Matched Reduced-Release Frontier Report", "",
        "## 1. Why release reduction was tested", "",
        "The four-release privacy frontier showed increasing noise/signal and falling DP alignment as epsilon tightened. Experiment 7 therefore reduced each client's bounded composition count from four to two.", "",
        "## 2. R4 versus R2 PJSDP", "",
        "At epsilon 2048, PJSDP MAE changed from 37.7490 (R4) to 30.6020 (R2), a composition gain of 18.93%. RMSE changed from 49.9786 to 43.4988, a gain of 12.96%.", "",
        "## 3. Why R2-E was selected", "",
        "R2-2E increased pre-clip norm and clipping distortion and was worse than R2-E. All new runs therefore retain one local epoch.", "",
        "## 4. Why matched DP-FedAvg is required", "",
        "Release reduction is not spectral. Spectral attribution requires PJSDP and DP-FedAvg to share two releases, epsilon, initialization, model, optimizer, data split, local epochs, cohort size, and the same fixed participation schedule.", "",
        "## 5-6. Fixed schedule and accounting assumptions", "",
        f"The public schedule SHA-256 is `{schedule_hash}`: 20 rounds, five clients per round, and every one of 50 clients exactly twice. Both accountants use replacement adjacency, delta=1e-5, bounded two-release composition, and **NO SUBSAMPLING AMPLIFICATION**.", "",
        "## 7-12. Matched results", "",
        "| Epsilon | Method | MAE | RMSE | MAPE | Noise/signal | DP alignment | Clip distortion | Achieved epsilon |", "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {int(float(row['TargetEpsilon']))} | {row['Method']} | {float(row['MAE']):.4f} | {float(row['RMSE']):.4f} | {float(row['MAPE']):.4f} | {float(row['MeanNoiseSignalRatio']):.4f} | {float(row['MeanDPAlignment']):.4f} | {float(row['MeanClipDistortion']):.4f} | {float(row['AchievedEpsilon']):.6f} |")
    lines += ["", f"Stage A spectral advantage status: **{stage_status}**. Stage B executed: **{'NO' if stage_status == 'NO' else 'YES'}**.", ""]
    for gain in gains:
        lines.append(f"At epsilon {int(gain['Epsilon'])}, PJSDP matched MAE advantage is {float(gain['MAEImprovementPercent']):+.2f}% and RMSE advantage is {float(gain['RMSEImprovementPercent']):+.2f}%.")
    lines += [
        "", "## 13-16. Diagnostics, clipping, and communication", "",
        "Noise/signal, DP alignment, and clipping are computed over 100 private internal release diagnostics per new run. PJSDP retains group-wise diagnostics; DP-FedAvg reports its full-update L2 clipping geometry without inventing spectral groups. Every matched run has 100 uploads. Private diagnostics are excluded from the adversary-visible DP transcript.", "",
        "## 17. Composition gain versus spectral gain", "",
        "The R4-to-R2 PJSDP difference is the composition gain. At each fixed R2 epsilon, DP-FedAvg-to-PJSDP is the matched spectral gain. These quantities are not conflated.", "",
        "## 18-19. PJSDP headroom and candidate epsilon", "",
    ]
    for row in headroom:
        lines.append(f"- epsilon {row['Epsilon']} (R{row['ReleaseCount']}): MAE degradation {float(row['MAEDegradationPercent']):+.2f}%, RMSE degradation {float(row['RMSEDegradationPercent']):+.2f}% versus the epsilon-4096 high-utility reference.")
    lines += [
        "", f"Recommended elbow epsilon: **{decision['elbow']}**.", "",
        f"Final conclusion: **{decision['conclusion']}**.", "",
        "## 20. Only one next experiment", "", f"**{decision['next']}**", "",
        "Limitations: seed 42 only; no confidence intervals or significance claim; final-only validation evaluation; mechanism-specific clipping geometries; attacks intentionally not executed.",
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
    reference_config = _load_reference_config(args.reference_config)
    expected = {
        "dataset": "pems08", "seed": 42, "num_clients": CLIENTS,
        "clients_per_round": CLIENTS_PER_ROUND, "local_epochs": LOCAL_EPOCHS,
        "projection_ratio": 0.9, "projection_seed": 42,
        "pjsdp_adjacency": ADJACENCY,
    }
    for key, expected_value in expected.items():
        actual = reference_config.get(key)
        valid = math.isclose(float(actual), expected_value, abs_tol=1e-12) if isinstance(expected_value, float) else actual == expected_value
        if not valid:
            raise ValueError(f"verified reference mismatch for {key}: {actual!r}")
    reference = verify_exp7_reference(args.exp7_root, args.pjsdp_4096_summary)
    accounting = _accounting_rows(reference_config)
    args.results_root.mkdir(parents=True, exist_ok=False)
    (args.results_root / "seed_42").mkdir(parents=True, exist_ok=False)
    manifest = {
        "reference": "PJSDP R2 epsilon=2048",
        "original_result_path": str(reference["run_dir"].resolve()),
        "copied_training_artifacts": False,
        "config_hash": sha256_file(reference["config_path"]),
        "schedule_hash": reference["schedule_file_sha256"],
        "schedule_source": str(reference["schedule_path"].resolve()),
        "privacy_summary": reference["privacy"],
        "metrics_source": str((reference["run_dir"] / "metrics.csv").resolve()),
    }
    (args.results_root / "reference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_rows(args.results_root / "matched_accounting.csv", accounting)
    prepared = _prepare_data(reference_config, args.data_root, "pems08")
    *_, std = prepared
    validation_scale = float(std[..., 0].mean())
    rows: list[dict[str, Any]] = [_reference_summary(reference, validation_scale)]
    private_master_seed = secrets.randbits(63)
    device = _resolve_device(args.device)

    dpf2048_cal = calibrate_dpfedavg(2048)
    dpf2048_config = _config_for_run("dp_fedavg", 2048, reference_config, EXPECTED_SCHEDULE_SHA256, dpf2048_cal)
    dpf2048_out = _make_output(args.results_root, "dp_fedavg", 2048, dpf2048_config)
    rows.append(_run_one(
        method="dp_fedavg", epsilon=2048, output=dpf2048_out,
        run_config=dpf2048_config, reference_config=reference_config,
        prepared=prepared, schedule=reference["schedule"], device=device,
        private_master_seed=private_master_seed,
    )["summary"])
    gains = [matched_gain(rows, 2048)]
    stage_status = stage_a_decision(gains[0])

    if stage_status != "NO":
        for method in ("pjsdp", "dp_fedavg"):
            calibration = calibrate_pjsdp(reference_config, 1024) if method == "pjsdp" else calibrate_dpfedavg(1024)
            config = _config_for_run(method, 1024, reference_config, EXPECTED_SCHEDULE_SHA256, calibration)
            output = _make_output(args.results_root, method, 1024, config)
            rows.append(_run_one(
                method=method, epsilon=1024, output=output, run_config=config,
                reference_config=reference_config, prepared=prepared,
                schedule=reference["schedule"], device=device,
                private_master_seed=private_master_seed,
            )["summary"])
        gains.append(matched_gain(rows, 1024))

    headroom = _headroom(rows, reference["high_utility"])
    decision = final_decision(stage_status, gains, headroom)
    write_rows(args.results_root / "matched_frontier_summary.csv", rows)
    write_rows(args.results_root / "matched_spectral_gain.csv", gains)
    write_rows(args.results_root / "pjsdp_reduced_release_headroom.csv", headroom)
    _plots(rows, headroom, args.results_root)
    _report(args.report_path, rows, gains, headroom, stage_status, decision, EXPECTED_SCHEDULE_SHA256)

    by_key = {(row["Method"], int(float(row["TargetEpsilon"]))): row for row in rows}
    print("========================================")
    print("MATCHED REDUCED-RELEASE FRONTIER")
    print("================================")
    print("\nDataset:\nPEMS08\n\nSeed:\n42\n\nRounds:\n20\n\nReleases/client:\n2\n\nLocal epochs:\nE\n\nDelta:\n1e-5\n\nAdjacency:\nreplacement\n\nSubsampling amplification:\nNO")
    print(f"\nSchedule hash:\n{EXPECTED_SCHEDULE_SHA256}")
    for epsilon in EPSILONS:
        print(f"\n---\n\n## EPSILON {epsilon}")
        if epsilon == 1024 and stage_status == "NO":
            print("\nExecuted:\nNO")
            continue
        if epsilon == 1024:
            print("\nExecuted:\nYES")
        for method in ("Current PJSDP", "DP-FedAvg"):
            row = by_key[(method, epsilon)]
            print(f"\n{method}:\nMAE = {float(row['MAE']):.6f}\nRMSE = {float(row['RMSE']):.6f}\nMAPE = {float(row['MAPE']):.6f}\nNoise/signal = {float(row['MeanNoiseSignalRatio']):.6f}\nDP alignment = {float(row['MeanDPAlignment']):.6f}\nAchieved epsilon = {float(row['AchievedEpsilon']):.9f}")
        gain = next(item for item in gains if int(item["Epsilon"]) == epsilon)
        print(f"\nPJSDP matched MAE advantage = {float(gain['MAEImprovementPercent']):+.3f}%")
        print(f"PJSDP matched RMSE advantage = {float(gain['RMSEImprovementPercent']):+.3f}%")
        if epsilon == 2048:
            print(f"\nSpectral advantage survives matched release:\n{stage_status}")
    composition_mae = 100.0 * (37.74902946890508 - 30.602021820412276) / 37.74902946890508
    composition_rmse = 100.0 * (49.97857964077523 - 43.498820824824165) / 49.97857964077523
    print(f"\n---\n\nComposition gain at eps2048:\nMAE {composition_mae:+.3f}%, RMSE {composition_rmse:+.3f}%")
    for gain in gains:
        print(f"\nSpectral gain at eps{int(gain['Epsilon'])}:\nMAE {float(gain['MAEImprovementPercent']):+.3f}%, RMSE {float(gain['RMSEImprovementPercent']):+.3f}%")
    print(f"\nRecommended elbow epsilon:\n{decision['elbow']}")
    print(f"\nFinal conclusion:\n{decision['conclusion']}")
    print(f"\nNext recommended experiment:\n{decision['next']}")
    return {"rows": rows, "gains": gains, "headroom": headroom, "stage_status": stage_status, "decision": decision}


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
