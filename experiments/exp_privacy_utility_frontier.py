from __future__ import annotations

"""Matched-privacy seed-42 frontier for verified DP-FedAvg and Current PJSDP."""

import argparse
import csv
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

from experiments.common import configure_logging, save_config, write_rows
from experiments.exp_ellipsoidal_clipping_sanity import (
    _build_clients_and_eval,
    _environment_snapshot,
    _evaluate_partitions,
    _global_dot,
    _prepare_data,
)
from experiments.exp_public_ellipsoid_dp_sanity import PublicGeometryReleaseMechanism
from experiments.exp_spectral_norm_audit import (
    _load_reference_config,
    _load_schedule,
    _resolve_device,
    _set_seed,
)
from federated.stg_server import STGServer
from privacy.accountant import RDPAccountant
from privacy.clipping import clip_group
from privacy.dp_fedavg import DPUpdate
from privacy.fixed_spectral_projection import FixedSpectralProjection
from privacy.pjsdp_accountant import PJSDPAccountant
from spectral.spectral_groups import GROUPS


EPSILONS = (2048, 1024, 512)
ALL_EPSILONS = (4096, 2048, 1024, 512)
METHODS = ("pjsdp", "dp_fedavg")
DELTA = 1e-5
ADJACENCY = "replacement"
RELEASES_PER_CLIENT = 4
DPFEDAVG_CLIP = 0.7197534544128077


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--pjsdp_reference_config", type=Path, required=True)
    p.add_argument("--dpfedavg_reference_config", type=Path, required=True)
    p.add_argument("--participation_schedule_csv", type=Path, required=True)
    p.add_argument("--pjsdp_4096_summary", type=Path, required=True)
    p.add_argument("--dpfedavg_4096_metrics", type=Path, required=True)
    p.add_argument("--fedavg_4096_metrics", type=Path, required=True)
    p.add_argument("--results_root", type=Path, default=Path("results/privacy_utility_frontier"))
    p.add_argument("--report_path", type=Path, default=Path("docs/PRIVACY_UTILITY_FRONTIER_SANITY_REPORT.md"))
    p.add_argument("--device", default="cuda")
    return p


def _read_one_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def calibrate_dpfedavg(target_epsilon: float) -> dict[str, float]:
    multiplier = RDPAccountant.calibrate_noise(
        float(target_epsilon), DELTA, RELEASES_PER_CLIENT, groups=1
    )
    accountant = RDPAccountant()
    accountant.step(multiplier, groups=1, steps=RELEASES_PER_CLIENT)
    achieved = accountant.epsilon(DELTA)
    return {
        "accounting_noise_multiplier": multiplier,
        "applied_noise_multiplier": 2.0 * multiplier,
        "noise_std": 2.0 * DPFEDAVG_CLIP * multiplier,
        "sensitivity": 2.0 * DPFEDAVG_CLIP,
        "achieved_epsilon": achieved,
    }


def calibrate_pjsdp(config: Mapping[str, Any], target_epsilon: float) -> dict[str, Any]:
    clip_bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
    weights = {group: float(config[f"{group}_noise_weight"]) for group in GROUPS}
    sensitivities = {group: 2.0 * clip_bounds[group] for group in GROUPS}
    result = PJSDPAccountant.calibrate_global_scale(
        float(target_epsilon),
        DELTA,
        RELEASES_PER_CLIENT,
        sensitivities,
        weights,
        sampling_rate=0.1,
        sampling_amplification=False,
    )
    verifier = PJSDPAccountant()
    verifier.step(
        sensitivities,
        result.group_noise_stds,
        rounds=RELEASES_PER_CLIENT,
        sampling_rate=0.1,
        sampling_amplification=False,
    )
    achieved = verifier.epsilon(DELTA)
    if not math.isclose(achieved, result.achieved_epsilon, abs_tol=1e-9):
        raise RuntimeError("PJSDP accountant consistency failure")
    return {
        "global_noise_scale": result.global_scale,
        "group_noise_stds": result.group_noise_stds,
        "group_sensitivities": sensitivities,
        "achieved_epsilon": achieved,
    }


def accounting_rows(pjsdp_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for epsilon in EPSILONS:
        pjsdp = calibrate_pjsdp(pjsdp_config, epsilon)
        dpf = calibrate_dpfedavg(epsilon)
        rows.extend([
            {
                "Method": "Current PJSDP",
                "TargetEpsilon": epsilon,
                "AchievedEpsilon": pjsdp["achieved_epsilon"],
                "Delta": DELTA,
                "Adjacency": ADJACENCY,
                "ReleaseCountPerClient": RELEASES_PER_CLIENT,
                "NoiseScale": pjsdp["global_noise_scale"],
                "NoiseParameter": yaml.safe_dump(pjsdp["group_noise_stds"], sort_keys=False).strip().replace("\n", "; "),
                "Sensitivity": yaml.safe_dump(pjsdp["group_sensitivities"], sort_keys=False).strip().replace("\n", "; "),
                "SubsamplingAmplificationUsed": False,
            },
            {
                "Method": "DP-FedAvg",
                "TargetEpsilon": epsilon,
                "AchievedEpsilon": dpf["achieved_epsilon"],
                "Delta": DELTA,
                "Adjacency": ADJACENCY,
                "ReleaseCountPerClient": RELEASES_PER_CLIENT,
                "NoiseScale": dpf["noise_std"],
                "NoiseParameter": f"accounting_multiplier={dpf['accounting_noise_multiplier']}; applied_multiplier={dpf['applied_noise_multiplier']}",
                "Sensitivity": dpf["sensitivity"],
                "SubsamplingAmplificationUsed": False,
            },
        ])
    return rows


def write_accounting_check(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite accounting check: {path}")
    lines = [
        "# Frontier Accounting Check", "",
        "Status: **VALID**", "",
        "All runs use replacement adjacency, delta=1e-5, a public deterministic schedule with exactly four releases per client, and no client-subsampling amplification. The existing accountant and calibrators are unchanged.", "",
        "| Method | Target epsilon | Achieved epsilon | Noise scale | Sensitivity |", "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(f"| {row['Method']} | {float(row['TargetEpsilon']):.0f} | {float(row['AchievedEpsilon']):.9f} | {float(row['NoiseScale']):.9f} | {row['Sensitivity']} |")
    lines += [
        "", "## Assumptions", "",
        "- Privacy unit: one complete client model update.",
        "- Adjacency: replacement; sensitivity is doubled relative to add/remove.",
        "- Composition: maximum four releases for any client.",
        "- Sampling rate is recorded as 0.1 but does not reduce epsilon.",
        "- **NO SUBSAMPLING AMPLIFICATION** is used for the fixed schedule.",
        "- PJSDP composes the five jointly released block sensitivities in the Gaussian Mahalanobis norm.",
        "- DP-FedAvg uses one full-update L2 bound and Gaussian noise standard deviation equal to sensitivity times the calibrated accounting multiplier.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _flat_norm(update: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float(value.detach().double().square().sum()) for value in update.values()))


def _flat_cosine(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> float:
    first_norm = _flat_norm(first)
    second_norm = _flat_norm(second)
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return 1.0
    dot = sum(float((first[name].detach().double() * second[name].detach().double()).sum()) for name in first)
    return min(1.0, max(-1.0, dot / (first_norm * second_norm + 1e-12)))


class DiagnosticDPFedAvg:
    """Exact production DP-FedAvg operation with internal-only diagnostics."""

    client_side_release = True
    requires_context = True

    def __init__(self, clip_norm: float, noise_std: float, generator: torch.Generator) -> None:
        self.clip_norm = float(clip_norm)
        self.noise_std = float(noise_std)
        self.generator = generator
        self.diagnostics: list[dict[str, Any]] = []

    def privatize(self, update: Mapping[str, torch.Tensor], round_index: int = -1, client_id: int = -1) -> DPUpdate:
        clipped, pre_norm = clip_group(update, self.clip_norm)
        noisy = {}
        noise_update = {}
        for name, value in clipped.items():
            noise = torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=self.generator) * self.noise_std
            noise_update[name] = noise
            noisy[name] = value + noise
        post_norm = _flat_norm(clipped)
        noise_norm = _flat_norm(noise_update)
        factor = min(1.0, self.clip_norm / (float(pre_norm) + 1e-12))
        self.diagnostics.append({
            "round": int(round_index) + 1,
            "client_id": int(client_id),
            "pre_clip_norm": float(pre_norm),
            "post_clip_norm": post_norm,
            "clipping_factor": factor,
            "clipping_distortion": 1.0 - factor if pre_norm > 0 else 0.0,
            "noise_std": self.noise_std,
            "noise_norm": noise_norm,
            "signal_norm": post_norm,
            "noise_to_signal_ratio": noise_norm / (post_norm + 1e-12),
            "dp_alignment": _flat_cosine(clipped, noisy),
            "diagnostic_scope": "PRIVATE_INTERNAL_NOT_PART_OF_DP_TRANSCRIPT",
        })
        return DPUpdate(noisy, True, {})


def _summary_stats(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "Mean": float(np.mean(array)),
        "Median": float(np.median(array)),
        "P50": float(np.percentile(array, 50)),
        "P90": float(np.percentile(array, 90)),
    }


def _method_dir_name(method: str, epsilon: int) -> str:
    return f"{'pjsdp' if method == 'pjsdp' else 'dp_fedavg'}_eps{epsilon}"


def prepare_run_configs(
    root: Path,
    pjsdp_config: Mapping[str, Any],
    schedule_hash: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    by_key = {(str(row["Method"]), int(row["TargetEpsilon"])): row for row in rows}
    configs = {}
    for epsilon in EPSILONS:
        for method in METHODS:
            display = "Current PJSDP" if method == "pjsdp" else "DP-FedAvg"
            accounting = by_key[(display, epsilon)]
            output = root / "seed_42" / _method_dir_name(method, epsilon)
            output.mkdir(parents=True, exist_ok=False)
            config = {
                "experiment": "Privacy-Utility Frontier Sanity Test",
                "method": method,
                "dataset": "pems08",
                "seed": 42,
                "rounds": 40,
                "num_clients": 50,
                "clients_per_round": 5,
                "release_count_per_client": RELEASES_PER_CLIENT,
                "local_epochs": int(pjsdp_config["local_epochs"]),
                "batch_size": int(pjsdp_config["batch_size"]),
                "history": int(pjsdp_config["history"]),
                "horizon": int(pjsdp_config["horizon"]),
                "hidden_channels": int(pjsdp_config["hidden_channels"]),
                "learning_rate": float(pjsdp_config["lr"]),
                "target_epsilon": epsilon,
                "achieved_epsilon": float(accounting["AchievedEpsilon"]),
                "delta": DELTA,
                "adjacency": ADJACENCY,
                "sampling_amplification_used": False,
                "participation_schedule": "existing_public_balanced_fixed",
                "participation_schedule_hash": schedule_hash,
                "model_seed": 42,
                "data_split": "existing_strict_temporal_split",
                "projection_ratio": 0.90 if method == "pjsdp" else None,
                "projection_seed": 42 if method == "pjsdp" else None,
                "clip_geometry": "independent_spectral_group_wise" if method == "pjsdp" else "full_update_l2",
                "clip_parameter": (
                    {group: float(pjsdp_config[f"{group}_clip"]) for group in GROUPS}
                    if method == "pjsdp" else DPFEDAVG_CLIP
                ),
                "noise_scale": float(accounting["NoiseScale"]),
                "noise_parameter": accounting["NoiseParameter"],
                "sensitivity": accounting["Sensitivity"],
                "dp_rng_policy": "private common master stream, independently advancing per release",
                "dp_rng_seed_disclosed": False,
                "diagnostic_scope": "private internal evaluation; not part of DP transcript",
                "save_full_updates": False,
                "accounting_status": "VALID",
                "attack_executed": False,
            }
            save_config(output / "config.yaml", config)
            configs[(method, epsilon)] = config
    return configs


def _run_one(
    *,
    method: str,
    epsilon: int,
    output: Path,
    run_config: Mapping[str, Any],
    pjsdp_reference: Mapping[str, Any],
    prepared: tuple,
    schedule: Sequence[Sequence[int]],
    device: torch.device,
    private_master_seed: int,
) -> dict[str, Any]:
    configure_logging(output / "train.log")
    _set_seed(42)
    model, clients, validation_sets, test_sets = _build_clients_and_eval(pjsdp_reference, prepared, device)
    if method == "pjsdp":
        calibration = calibrate_pjsdp(pjsdp_reference, epsilon)
        projection = FixedSpectralProjection(model.state_dict(), 0.90, 42)
        expected_hash = str(pjsdp_reference["projection_mask_hash"])
        if projection.mask_hash != expected_hash:
            raise RuntimeError("PJSDP projection mask changed from the verified reference")
        clip_bounds = {group: float(pjsdp_reference[f"{group}_clip"]) for group in GROUPS}
        mechanism: Any = PublicGeometryReleaseMechanism(
            "current", projection, clip_bounds, None,
            calibration["group_noise_stds"],
            torch.Generator().manual_seed(private_master_seed),
        )
    else:
        calibration = calibrate_dpfedavg(epsilon)
        mechanism = DiagnosticDPFedAvg(
            DPFEDAVG_CLIP,
            calibration["noise_std"],
            torch.Generator().manual_seed(private_master_seed),
        )
    server = STGServer(
        model, clients, device, 42, collector=None, privacy=mechanism,
        topk_ratio=1.0, error_feedback=False, event_tau=0.0,
        skip_policy="zero", report_client_train_loss=True,
        participation_schedule=schedule,
    )
    losses = []
    started = time.perf_counter()
    for round_index in range(40):
        round_row = server.run_round(round_index, 5, 1, 0.0)
        loss = float(round_row["client_train_loss"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss: {method}/eps{epsilon}")
        losses.append(loss)
        logging.info("method=%s epsilon=%d round=%d/40 train_loss=%.8f", method, epsilon, round_index + 1, loss)
    *_, mean, std = prepared
    validation = _evaluate_partitions(server.model, validation_sets, device, pjsdp_reference, mean, std)
    test = _evaluate_partitions(server.model, test_sets, device, pjsdp_reference, mean, std)
    runtime = time.perf_counter() - started

    if method == "pjsdp":
        raw_diagnostics = mechanism.noise_signal_diagnostics
        clipping_rows = mechanism.clipping_diagnostics
        diagnostics = [{
            **row,
            "pre_clip_norm": clipping_rows[index]["raw_joint_norm"],
            "post_clip_norm": clipping_rows[index]["post_clip_joint_norm"],
            "clipping_factor": clipping_rows[index]["joint_clip_factor"],
            "clipping_distortion": clipping_rows[index]["clipping_distortion"],
        } for index, row in enumerate(raw_diagnostics)]
        group_rows = mechanism.group_diagnostics
    else:
        diagnostics = mechanism.diagnostics
        group_rows = []
    nsr = _summary_stats([float(row["noise_to_signal_ratio"]) for row in diagnostics])
    align = _summary_stats([float(row["dp_alignment"]) for row in diagnostics])
    distortion = _summary_stats([float(row["clipping_distortion"]) for row in diagnostics])
    metrics = {
        "Method": "Current PJSDP" if method == "pjsdp" else "DP-FedAvg",
        "TargetEpsilon": epsilon,
        "AchievedEpsilon": float(run_config["achieved_epsilon"]),
        "Delta": DELTA,
        "MAE": test["MAE"], "RMSE": test["RMSE"], "MAPE": test["MAPE"],
        "Val_MAE": validation["MAE"], "Val_RMSE": validation["RMSE"], "Val_MAPE": validation["MAPE"],
        "BestValidationMAE": validation["MAE"],
        "ValidationEvaluations": 1,
        "FinalTrainLoss": losses[-1], "BestTrainLoss": min(losses),
        "TrainStable": all(math.isfinite(value) for value in losses),
        "NaNInfDetected": False,
        "RuntimeSeconds": runtime,
    }
    privacy = {
        "Method": metrics["Method"], "TargetEpsilon": epsilon,
        "AchievedEpsilon": run_config["achieved_epsilon"], "Delta": DELTA,
        "Adjacency": ADJACENCY, "ReleaseCountPerClient": RELEASES_PER_CLIENT,
        "NoiseScale": run_config["noise_scale"], "NoiseParameter": run_config["noise_parameter"],
        "Sensitivity": run_config["sensitivity"], "SubsamplingAmplificationUsed": False,
        "AccountingStatus": "VALID",
    }
    write_rows(output / "metrics.csv", [metrics])
    write_rows(output / "privacy_summary.csv", [privacy])
    write_rows(output / "dp_diagnostics.csv", diagnostics)
    if group_rows:
        write_rows(output / "group_diagnostics.csv", group_rows)
    (output / "environment.txt").write_text(_environment_snapshot(device), encoding="utf-8")
    return {"metrics": metrics, "privacy": privacy, "diagnostics": diagnostics, "groups": group_rows}


def _load_references(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    pjsdp_rows = []
    with args.pjsdp_4096_summary.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Method"] == "Current":
                pjsdp_rows.append(row)
    if len(pjsdp_rows) != 1:
        raise ValueError("expected one Current row in the existing epsilon-4096 summary")
    p = pjsdp_rows[0]
    d = _read_one_csv(args.dpfedavg_4096_metrics)
    f = _read_one_csv(args.fedavg_4096_metrics)
    rows = [
        {
            "Method": "Current PJSDP", "TargetEpsilon": 4096,
            "AchievedEpsilon": float(p["AchievedEpsilon"]), "Delta": DELTA,
            "NoiseScale": 0.040786555576922574,
            "MAE": float(p["MAE"]), "RMSE": float(p["RMSE"]), "MAPE": float(p["MAPE"]),
            "MeanNoiseSignalRatio": float(p["MeanNoiseToSignalRatio"]),
            "MedianNoiseSignalRatio": float("nan"),
            "P90NoiseSignalRatio": float("nan"),
            "MeanDPAlignment": float(p["MeanDPAlignment"]),
            "MedianDPAlignment": float("nan"), "P90DPAlignment": float("nan"),
            "MeanClipDistortion": float(p["MeanClipDistortion"]),
            "TrainStable": p["TrainStable"], "ReferenceSource": "existing_public_ellipsoid_dp_sanity",
        },
        {
            "Method": "DP-FedAvg", "TargetEpsilon": 4096,
            "AchievedEpsilon": float(d["epsilon"]), "Delta": DELTA,
            "NoiseScale": 0.045047945220757654,
            "MAE": float(d["MAE"]), "RMSE": float(d["RMSE"]), "MAPE": float(d["MAPE"]),
            "MeanNoiseSignalRatio": float("nan"), "MedianNoiseSignalRatio": float("nan"),
            "P90NoiseSignalRatio": float("nan"),
            "MeanDPAlignment": float("nan"), "MedianDPAlignment": float("nan"),
            "P90DPAlignment": float("nan"), "MeanClipDistortion": float("nan"),
            "TrainStable": True, "ReferenceSource": "existing_verified_dpfedavg_seed42",
        },
    ]
    fedavg = {"MAE": float(f["MAE"]), "RMSE": float(f["RMSE"]), "MAPE": float(f["MAPE"])}
    return rows, fedavg


def aggregate_frontier(runs: Sequence[Mapping[str, Any]], reference_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in reference_rows]
    for run in runs:
        metrics = run["metrics"]
        diagnostics = run["diagnostics"]
        nsr = _summary_stats([float(row["noise_to_signal_ratio"]) for row in diagnostics])
        alignment = _summary_stats([float(row["dp_alignment"]) for row in diagnostics])
        distortion = _summary_stats([float(row["clipping_distortion"]) for row in diagnostics])
        rows.append({
            "Method": metrics["Method"], "TargetEpsilon": metrics["TargetEpsilon"],
            "AchievedEpsilon": metrics["AchievedEpsilon"], "Delta": DELTA,
            "NoiseScale": run["privacy"]["NoiseScale"],
            "MAE": metrics["MAE"], "RMSE": metrics["RMSE"], "MAPE": metrics["MAPE"],
            "MeanNoiseSignalRatio": nsr["Mean"], "MedianNoiseSignalRatio": nsr["Median"],
            "P90NoiseSignalRatio": nsr["P90"],
            "MeanDPAlignment": alignment["Mean"], "MedianDPAlignment": alignment["Median"],
            "P90DPAlignment": alignment["P90"],
            "MeanClipDistortion": distortion["Mean"],
            "TrainStable": metrics["TrainStable"], "ReferenceSource": "new_frontier_run",
        })
    return sorted(rows, key=lambda row: (-int(row["TargetEpsilon"]), str(row["Method"])))


def matched_improvements(frontier: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(str(row["Method"]), int(row["TargetEpsilon"])): row for row in frontier}
    rows = []
    for epsilon in ALL_EPSILONS:
        p = indexed[("Current PJSDP", epsilon)]
        d = indexed[("DP-FedAvg", epsilon)]
        rows.append({
            "Epsilon": epsilon,
            "MAEImprovementPercent": 100.0 * (float(d["MAE"]) - float(p["MAE"])) / float(d["MAE"]),
            "RMSEImprovementPercent": 100.0 * (float(d["RMSE"]) - float(p["RMSE"])) / float(d["RMSE"]),
            "MAPEImprovementPercent": 100.0 * (float(d["MAPE"]) - float(p["MAPE"])) / float(d["MAPE"]),
        })
    return rows


def utility_vs_fedavg(frontier: Sequence[Mapping[str, Any]], fedavg: Mapping[str, float]) -> list[dict[str, Any]]:
    return [{
        "Method": row["Method"], "Epsilon": row["TargetEpsilon"],
        "MAEDegradationPercent": 100.0 * (float(row["MAE"]) - fedavg["MAE"]) / fedavg["MAE"],
        "RMSEDegradationPercent": 100.0 * (float(row["RMSE"]) - fedavg["RMSE"]) / fedavg["RMSE"],
        "MAPEDegradationPercent": 100.0 * (float(row["MAPE"]) - fedavg["MAPE"]) / fedavg["MAPE"],
    } for row in frontier]


def elbow_analysis(frontier: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    pjsdp = {int(row["TargetEpsilon"]): row for row in frontier if row["Method"] == "Current PJSDP"}
    reference = pjsdp[4096]
    degradation = []
    for epsilon in ALL_EPSILONS:
        row = pjsdp[epsilon]
        mae = 100.0 * (float(row["MAE"]) - float(reference["MAE"])) / float(reference["MAE"])
        rmse = 100.0 * (float(row["RMSE"]) - float(reference["RMSE"])) / float(reference["RMSE"])
        worst = max(mae, rmse)
        label = "UTILITY-PRESERVING" if worst <= 10.0 else "MODERATE DEGRADATION" if worst <= 20.0 else "SEVERE DEGRADATION"
        degradation.append({"Epsilon": epsilon, "MAEDegradationPercent": mae, "RMSEDegradationPercent": rmse, "Classification": label})
    steps = []
    for before, after in zip(ALL_EPSILONS[:-1], ALL_EPSILONS[1:]):
        first, second = pjsdp[before], pjsdp[after]
        steps.append({
            "FromEpsilon": before, "ToEpsilon": after,
            "MAEChange": float(second["MAE"]) - float(first["MAE"]),
            "MAEChangePercent": 100.0 * (float(second["MAE"]) - float(first["MAE"])) / float(first["MAE"]),
            "RMSEChange": float(second["RMSE"]) - float(first["RMSE"]),
            "RMSEChangePercent": 100.0 * (float(second["RMSE"]) - float(first["RMSE"])) / float(first["RMSE"]),
            "Log2PrivacyImprovement": 1.0,
        })
    preserving = [row for row in degradation if row["Classification"] == "UTILITY-PRESERVING"]
    moderate = [row for row in degradation if row["Classification"] == "MODERATE DEGRADATION"]
    candidate = min((int(row["Epsilon"]) for row in preserving), default=min((int(row["Epsilon"]) for row in moderate), default=4096))
    indexed_deg = {int(row["Epsilon"]): row for row in degradation}
    improvements = {int(row["Epsilon"]): row for row in matched_improvements(frontier)}
    clearly_better_2048_1024 = all(
        improvements[e]["MAEImprovementPercent"] >= 2.0 and improvements[e]["RMSEImprovementPercent"] >= 2.0
        for e in (2048, 1024)
    )
    if indexed_deg[512]["Classification"] == "UTILITY-PRESERVING":
        conclusion, next_exp = "EPSILON 512 REMAINS UTILITY-PRESERVING", "Extend Frontier Below 512"
    elif indexed_deg[1024]["Classification"] == "UTILITY-PRESERVING" and clearly_better_2048_1024:
        conclusion, next_exp = "EPSILON 1024 IS A STRONG ELBOW CANDIDATE", "Representative-Point Privacy Attack Evaluation"
    elif indexed_deg[2048]["Classification"] == "UTILITY-PRESERVING":
        conclusion, next_exp = "EPSILON 2048 IS THE ELBOW CANDIDATE", "Representative-Point Privacy Attack Evaluation"
    else:
        conclusion, next_exp = "CURRENT PJSDP HAS LIMITED PRIVACY HEADROOM", "Reassess Sensitivity/Release Composition"
    return degradation, steps, {"candidate": str(candidate), "conclusion": conclusion, "next": next_exp}


def _plot(frontier: Sequence[Mapping[str, Any]], improvements: Sequence[Mapping[str, Any]], root: Path) -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
        "axes.spines.right": False, "axes.spines.top": False,
        "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    })
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Current PJSDP": "#4C78A8", "DP-FedAvg": "#E08B52"}
    for field, filename, ylabel in (
        ("MAE", "epsilon_vs_mae.png", "MAE (lower is better)"),
        ("RMSE", "epsilon_vs_rmse.png", "RMSE (lower is better)"),
        ("MeanNoiseSignalRatio", "epsilon_vs_noise_signal.png", "Mean noise / signal norm"),
        ("MeanDPAlignment", "epsilon_vs_dp_alignment.png", "Mean DP alignment"),
    ):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        for method in ("DP-FedAvg", "Current PJSDP"):
            rows = sorted([row for row in frontier if row["Method"] == method], key=lambda row: int(row["TargetEpsilon"]))
            x = np.asarray([int(row["TargetEpsilon"]) for row in rows])
            y = np.asarray([float(row.get(field, float("nan"))) for row in rows])
            ax.plot(x, y, marker="o", linewidth=1.8, color=colors[method], label=method)
        ax.set_xscale("log", base=2)
        ax.set_xticks([512, 1024, 2048, 4096], ["512", "1024", "2048", "4096"])
        ax.set_xlabel("Privacy budget epsilon")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = [int(row["Epsilon"]) for row in improvements]
    ax.plot(x, [float(row["MAEImprovementPercent"]) for row in improvements], marker="o", label="MAE gain", color="#4C78A8")
    ax.plot(x, [float(row["RMSEImprovementPercent"]) for row in improvements], marker="s", label="RMSE gain", color="#72A66A")
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096], ["512", "1024", "2048", "4096"])
    ax.set_xlabel("Matched privacy budget epsilon")
    ax.set_ylabel("PJSDP improvement over DP-FedAvg (%)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plot_dir / "matched_privacy_utility_gain.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_report(path: Path, frontier: Sequence[Mapping[str, Any]], improvements: Sequence[Mapping[str, Any]], degradation: Sequence[Mapping[str, Any]], steps: Sequence[Mapping[str, Any]], decision: Mapping[str, str], fedavg: Mapping[str, float]) -> None:
    indexed = {(str(row["Method"]), int(row["TargetEpsilon"])): row for row in frontier}
    lines = [
        "# Privacy-Utility Frontier Sanity Report", "",
        "## 1. Motivation", "", "This seed-42 screen locates a transparent development privacy-utility elbow for Current PJSDP against matched DP-FedAvg.", "",
        "## 2. Why the ellipsoid branch was stopped", "", "At epsilon 4096, joint ellipsoids reduced clipping distortion but required larger noise and produced worse utility. No ellipsoid method is included here.", "",
        "## 3. Why Current PJSDP is used", "", "Current PJSDP was the best verified method at the matched epsilon-4096 endpoint. Its clipping, projection, covariance weights, model, and client schedule are unchanged; only target epsilon varies.", "",
        "## 4-8. Methods and privacy model", "", "The comparison contains only standard DP-FedAvg and Current PJSDP. Both use PEMS08, seed 42, replacement adjacency, delta=1e-5, 50 clients, 40 rounds, five clients per round, and exactly four releases per client. The fixed schedule receives no subsampling amplification.", "",
        "## 9. Target and achieved epsilon", "", "| Method | Target | Achieved | Noise scale |", "|---|---:|---:|---:|",
    ]
    for row in frontier:
        lines.append(f"| {row['Method']} | {int(row['TargetEpsilon'])} | {float(row['AchievedEpsilon']):.9f} | {float(row['NoiseScale']):.8f} |")
    lines += ["", "## 10-14. Frontier results", "", "| Epsilon | Method | MAE | RMSE | MAPE | Noise/signal | DP alignment | Clip distortion | Stable |", "|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for epsilon in ALL_EPSILONS:
        for method in ("Current PJSDP", "DP-FedAvg"):
            row = indexed[(method, epsilon)]
            def fmt(name: str) -> str:
                value = float(row.get(name, float("nan")))
                return "N/A" if not math.isfinite(value) else f"{value:.4f}"
            lines.append(f"| {epsilon} | {method} | {float(row['MAE']):.4f} | {float(row['RMSE']):.4f} | {float(row['MAPE']):.4f} | {fmt('MeanNoiseSignalRatio')} | {fmt('MeanDPAlignment')} | {fmt('MeanClipDistortion')} | {row['TrainStable']} |")
    lines += ["", "The epsilon-4096 DP-FedAvg reference predates the added internal diagnostics, so those cells remain N/A rather than rerunning or fabricating them. All new diagnostics are private internal evaluation artifacts and are not part of the DP transcript.", "",
        "## 15. Matched DP-FedAvg comparison", "", "| Epsilon | MAE gain | RMSE gain | MAPE gain |", "|---:|---:|---:|---:|"]
    for row in improvements:
        lines.append(f"| {row['Epsilon']} | {float(row['MAEImprovementPercent']):+.2f}% | {float(row['RMSEImprovementPercent']):+.2f}% | {float(row['MAPEImprovementPercent']):+.2f}% |")
    lines += ["", "## 16-18. Noise, alignment, and clipping diagnostics", "", "Noise/signal and DP alignment diagnose when stronger privacy overwhelms the clipped update. PJSDP group-level files retain clipping rates, retention, distortion, and group norms without using them to tune any parameter.", "",
        "## 19-21. Elbow calculation", "", "The heuristic labels an endpoint utility-preserving when both MAE and RMSE degrade no more than 10% relative to PJSDP epsilon 4096, moderate when the worse degradation is at most 20%, and severe otherwise. It selects the smallest utility-preserving epsilon, or otherwise the smallest moderate endpoint.", "", "| Epsilon | MAE degradation | RMSE degradation | Class |", "|---:|---:|---:|---|"]
    for row in degradation:
        lines.append(f"| {row['Epsilon']} | {float(row['MAEDegradationPercent']):+.2f}% | {float(row['RMSEDegradationPercent']):+.2f}% | {row['Classification']} |")
    lines += ["", f"Candidate elbow: **epsilon {decision['candidate']}**.", "", "Step costs for each one-bit (2x) privacy strengthening are saved in `epsilon_step_cost.csv`.", "",
        "## 20. Utility degradation versus non-private FedAvg", "", f"The existing seed-42 FedAvg reference is MAE={fedavg['MAE']:.4f}, RMSE={fedavg['RMSE']:.4f}, MAPE={fedavg['MAPE']:.4f}. Negative degradation values in the companion CSV are retained as genuine improvements.", "",
        "## 22. Known limitations", "", "- One seed; no uncertainty or significance claim.", "- The elbow rule is a development heuristic, not final paper model selection.", "- Epsilon 4096 is loaded from compatible historical seed-42 runs and is not rerun.", "- Internal diagnostics must not be released as part of the adversary-visible DP transcript.", "",
        "## 23. Only one next experiment", "", f"**{decision['next']}**", "", f"Final conclusion: **{decision['conclusion']}**."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.results_root.exists():
        raise FileExistsError(f"refusing to overwrite frontier results: {args.results_root}")
    if args.report_path.exists():
        raise FileExistsError(f"refusing to overwrite frontier report: {args.report_path}")
    pjsdp_config = _load_reference_config(args.pjsdp_reference_config)
    dpf_config = _load_reference_config(args.dpfedavg_reference_config)
    for config in (pjsdp_config, dpf_config):
        expected = {"dataset": "pems08", "seed": 42, "rounds": 40, "num_clients": 50, "clients_per_round": 5, "local_epochs": 1}
        for key, value in expected.items():
            if config[key] != value:
                raise ValueError(f"reference config mismatch: {key}")
        if config["participation_schedule_hash"] != pjsdp_config["participation_schedule_hash"]:
            raise ValueError("reference schedules differ")
    if dpf_config["dp_fedavg_adjacency"] != ADJACENCY or pjsdp_config["pjsdp_adjacency"] != ADJACENCY:
        raise ValueError("both methods must use replacement adjacency")
    if not math.isclose(float(dpf_config["clip_norm"]), DPFEDAVG_CLIP, abs_tol=1e-12):
        raise ValueError("DP-FedAvg clip norm changed")
    schedule, schedule_hash, counts = _load_schedule(args.participation_schedule_csv, 40, 50, 5)
    if min(counts) != 4 or max(counts) != 4 or schedule_hash != pjsdp_config["participation_schedule_hash"]:
        raise ValueError("fixed participation schedule audit failed")
    accounting = accounting_rows(pjsdp_config)
    if any(abs(float(row["AchievedEpsilon"]) - float(row["TargetEpsilon"])) > 1e-7 for row in accounting):
        raise RuntimeError("accounting mismatch; refusing training")
    accounting_path = Path("docs/FRONTIER_ACCOUNTING_CHECK.md")
    if not accounting_path.exists():
        write_accounting_check(accounting_path, accounting)
    args.results_root.mkdir(parents=True, exist_ok=False)
    write_rows(args.results_root / "accounting_check.csv", accounting)
    configs = prepare_run_configs(args.results_root, pjsdp_config, schedule_hash, accounting)
    prepared = _prepare_data(pjsdp_config, args.data_root, "pems08")
    device = _resolve_device(args.device)
    private_master_seed = secrets.randbits(63)
    runs = []
    for epsilon in EPSILONS:
        for method in METHODS:
            output = args.results_root / "seed_42" / _method_dir_name(method, epsilon)
            runs.append(_run_one(
                method=method, epsilon=epsilon, output=output,
                run_config=configs[(method, epsilon)], pjsdp_reference=pjsdp_config,
                prepared=prepared, schedule=schedule, device=device,
                private_master_seed=private_master_seed,
            ))
    references, fedavg = _load_references(args)
    frontier = aggregate_frontier(runs, references)
    improvements = matched_improvements(frontier)
    degradation_vs_fedavg = utility_vs_fedavg(frontier, fedavg)
    degradation, steps, decision = elbow_analysis(frontier)
    write_rows(args.results_root / "frontier_summary.csv", frontier)
    write_rows(args.results_root / "matched_privacy_improvement.csv", improvements)
    write_rows(args.results_root / "utility_vs_fedavg.csv", degradation_vs_fedavg)
    write_rows(args.results_root / "pjsdp_degradation_vs_eps4096.csv", degradation)
    write_rows(args.results_root / "epsilon_step_cost.csv", steps)
    _plot(frontier, improvements, args.results_root)
    _write_report(args.report_path, frontier, improvements, degradation, steps, decision, fedavg)
    indexed = {(str(row["Method"]), int(row["TargetEpsilon"])): row for row in frontier}
    improvement_index = {int(row["Epsilon"]): row for row in improvements}
    degradation_index = {int(row["Epsilon"]): row for row in degradation}
    print("========================================")
    print("PRIVACY-UTILITY FRONTIER SANITY")
    print("===============================")
    print("\nDataset:\nPEMS08\n\nSeed:\n42\n\nAdjacency:\nreplacement\n\nDelta:\n1e-5\n\nReleases per client:\n4")
    print("\nAccounting status:\nVALID\n\nSubsampling amplification used:\nNO")
    for epsilon in ALL_EPSILONS:
        print(f"\n---\n\nEPSILON {epsilon}" + ("\nExisting reference" if epsilon == 4096 else ""))
        for method in ("Current PJSDP", "DP-FedAvg"):
            row = indexed[(method, epsilon)]
            print(f"\n{method}:\nTarget epsilon = {epsilon}\nAchieved epsilon = {float(row['AchievedEpsilon']):.9f}\nMAE = {float(row['MAE']):.6f}\nRMSE = {float(row['RMSE']):.6f}\nMAPE = {float(row['MAPE']):.6f}")
        gain = improvement_index[epsilon]
        print(f"\nPJSDP MAE gain vs DP-FedAvg = {float(gain['MAEImprovementPercent']):+.3f}%")
        print(f"PJSDP RMSE gain vs DP-FedAvg = {float(gain['RMSEImprovementPercent']):+.3f}%")
    print("\n---\n\nPJSDP utility degradation vs eps4096:")
    for epsilon in EPSILONS:
        row = degradation_index[epsilon]
        print(f"\n{epsilon}:\nMAE = {float(row['MAEDegradationPercent']):+.3f}%\nRMSE = {float(row['RMSEDegradationPercent']):+.3f}%")
    print(f"\n---\n\nCandidate elbow epsilon:\n{decision['candidate']}")
    print(f"\nFinal conclusion:\n\n{decision['conclusion']}")
    print(f"\nNext recommended experiment:\n\n{decision['next']}")
    return {"frontier": frontier, "decision": decision}


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
