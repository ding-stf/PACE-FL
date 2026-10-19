from __future__ import annotations

"""Two-stage development sanity test for joint ellipsoidal spectral clipping."""

import argparse
import csv
import logging
import math
import platform
import subprocess
import sys
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
from torch.utils.data import DataLoader

from datasets.client_partition import partition_nodes
from datasets.stg_utils import WindowDataset, temporal_split
from datasets.traffic import load_traffic_dataset
from experiments.common import configure_logging, save_config, write_rows
from experiments.exp_spectral_norm_audit import (
    _load_reference_config,
    _load_schedule,
    _resolve_device,
    _set_seed,
)
from federated.evaluation import evaluate
from federated.metrics import ForecastingTotals
from federated.stg_client import STGClient
from federated.stg_server import STGServer
from models.stgcn import STGCN
from privacy.anisotropic_gaussian import JointAnisotropicGaussianMechanism
from privacy.dp_fedavg import DPUpdate
from privacy.ellipsoidal_clipping import EllipsoidalSpectralClipper
from privacy.fixed_spectral_projection import FixedSpectralProjection
from privacy.pjsdp_accountant import PJSDPAccountant
from privacy.projected_joint_spectral_dp import ProjectedJointSpectralDP
from privacy.spectral_group_clipping import SpectralGroupClipper
from spectral.spectral_groups import GROUPS, inverse_groups, transform_update


MODES = ("spherical_current", "ellipsoid_equal", "ellipsoid_anisotropic_dev")
DISPLAY = {
    "spherical_current": "Current independent",
    "ellipsoid_equal": "Equal ellipsoid",
    "ellipsoid_anisotropic_dev": "Anisotropic ellipsoid",
}
DEV_SCALES = dict(zip(GROUPS, (1.72, 0.71, 1.16, 0.52, 0.90)))
NEAR_ZERO_STD = 1e-8
JOINT_CLIP_NORM = 1.0
GATE_TOLERANCE = 0.05


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="pems08")
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--reference_config", type=Path, required=True)
    p.add_argument("--participation_schedule_csv", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--device", default=None)
    p.add_argument("--common_noise_seed", type=int, default=2042)
    p.add_argument(
        "--results_root", type=Path, default=Path("results/ellipsoid_sanity")
    )
    p.add_argument(
        "--report_path",
        type=Path,
        default=Path("docs/ELLIPSOIDAL_CLIPPING_SANITY_REPORT.md"),
    )
    return p


def _group_norm(tensors: Mapping[str, torch.Tensor]) -> float:
    return math.sqrt(
        sum(float(value.detach().double().square().sum()) for value in tensors.values())
    )


def _global_norm(groups: Mapping[str, Mapping[str, torch.Tensor]]) -> float:
    return math.sqrt(sum(_group_norm(groups[group]) ** 2 for group in GROUPS))


def _global_dot(
    first: Mapping[str, Mapping[str, torch.Tensor]],
    second: Mapping[str, Mapping[str, torch.Tensor]],
) -> float:
    return sum(
        float((value.detach().double() * second[group][name].detach().double()).sum())
        for group in GROUPS
        for name, value in first[group].items()
    )


def _relative_group_structure_error(
    raw_norms: Sequence[float], postclip_norms: Sequence[float], eps: float = 1e-12
) -> float:
    """L2 error between L1-normalized raw and clipped group-norm vectors."""

    raw = np.asarray(raw_norms, dtype=float)
    clipped = np.asarray(postclip_norms, dtype=float)
    raw_normalized = raw / (float(raw.sum()) + eps)
    clipped_normalized = clipped / (float(clipped.sum()) + eps)
    return float(np.linalg.norm(raw_normalized - clipped_normalized, ord=2))


def _environment_snapshot(device: torch.device) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    cuda_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "not_used"
    return "\n".join(
        [
            f"python={sys.version.replace(chr(10), ' ')}",
            f"platform={platform.platform()}",
            f"torch={torch.__version__}",
            f"cuda_available={torch.cuda.is_available()}",
            f"device={device}",
            f"cuda_device_name={cuda_name}",
            f"git_commit={commit}",
        ]
    ) + "\n"


class GeometryReleaseMechanism:
    """Development-only release with fixed geometry and unchanged noise stds."""

    aggregation_level = "local"
    requires_context = True
    client_side_release = True

    def __init__(
        self,
        mode: str,
        projection: FixedSpectralProjection,
        current_clip_bounds: Mapping[str, float],
        noise_stds: Mapping[str, float],
        generator: torch.Generator,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown geometry mode: {mode}")
        self.mode = mode
        self.projection = projection
        self.current_clipper = SpectralGroupClipper(current_clip_bounds)
        self.current_release = (
            ProjectedJointSpectralDP(
                projection,
                SpectralGroupClipper(current_clip_bounds),
                JointAnisotropicGaussianMechanism(
                    current_clip_bounds, noise_stds, generator
                ),
                record_private_diagnostics=False,
            )
            if mode == "spherical_current"
            else None
        )
        if mode == "ellipsoid_equal":
            scales = {group: 1.0 for group in GROUPS}
        else:
            scales = DEV_SCALES
        self.ellipsoid = EllipsoidalSpectralClipper(
            scales, JOINT_CLIP_NORM, normalization="mean1"
        )
        self.noise_stds = {group: float(noise_stds[group]) for group in GROUPS}
        self.generator = generator
        self.clipping_diagnostics: list[dict[str, Any]] = []
        self.group_diagnostics: list[dict[str, Any]] = []

    def _current_clip(
        self, groups: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any], list[dict[str, Any]]]:
        clipped, stats = self.current_clipper.clip_groups(groups)
        by_group = {row.group_name: row for row in stats}
        raw_joint = _global_norm(groups)
        post_joint = _global_norm(clipped)
        distortion_sq = sum(
            float((groups[group][name] - clipped[group][name]).detach().double().square().sum())
            for group in GROUPS
            for name in groups[group]
        )
        distortion = math.sqrt(distortion_sq) / (raw_joint + 1e-12)
        cosine = (
            _global_dot(groups, clipped) / (raw_joint * post_joint + 1e-12)
            if raw_joint > 0 and post_joint > 0
            else 1.0
        )
        normalized_sq = sum(
            (by_group[group].pre_clip_norm / by_group[group].clip_bound) ** 2
            for group in GROUPS
        )
        post_normalized_sq = sum(
            (by_group[group].post_clip_norm / by_group[group].clip_bound) ** 2
            for group in GROUPS
        )
        contribution_denominator = normalized_sq + 1e-12
        group_rows = []
        for group in GROUPS:
            row = by_group[group]
            normalized = row.pre_clip_norm / row.clip_bound
            group_rows.append(
                {
                    "group_name": group,
                    "scale": row.clip_bound,
                    "diagnostic_scale_source": "current_public_clip_bound",
                    "raw_group_norm": row.pre_clip_norm,
                    "normalized_group_norm": normalized,
                    "restored_group_norm_after_clip": row.post_clip_norm,
                    "group_retention_ratio": row.clip_factor if row.pre_clip_norm > 0 else 1.0,
                    "group_clipping_distortion": (1.0 - row.clip_factor) if row.pre_clip_norm > 0 else 0.0,
                    "ellipsoid_norm_contribution": normalized * normalized / contribution_denominator,
                    "num_coefficients": row.num_coefficients,
                }
            )
        diagnostics = {
            "normalized_joint_norm_preclip": math.sqrt(normalized_sq),
            "joint_clip_factor": post_joint / (raw_joint + 1e-12),
            "shared_joint_alpha": "",
            "normalized_joint_norm_postclip": math.sqrt(post_normalized_sq),
            "clipping_distortion": distortion,
            "cosine_similarity": min(1.0, max(-1.0, cosine)),
            "raw_joint_norm": raw_joint,
            "post_clip_joint_norm": post_joint,
            "min_group_clip_factor": min(row.clip_factor for row in stats),
            "max_group_clip_factor": max(row.clip_factor for row in stats),
            "relative_group_structure_error": _relative_group_structure_error(
                [by_group[group].pre_clip_norm for group in GROUPS],
                [by_group[group].post_clip_norm for group in GROUPS],
            ),
        }
        return clipped, diagnostics, group_rows

    def _ellipsoid_clip(
        self, groups: Mapping[str, Mapping[str, torch.Tensor]]
    ) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any], list[dict[str, Any]]]:
        clipped, result = self.ellipsoid.clip_groups(groups)
        group_rows = [
            {
                "group_name": row.group_name,
                "scale": row.scale,
                "diagnostic_scale_source": (
                    "equal_mean1" if self.mode == "ellipsoid_equal" else "DEVELOPMENT_ONLY_PRIVATE_CALIBRATION_mean1"
                ),
                "raw_group_norm": row.raw_group_norm,
                "normalized_group_norm": row.normalized_group_norm,
                "restored_group_norm_after_clip": row.restored_group_norm_after_clip,
                "group_retention_ratio": row.group_retention_ratio,
                "group_clipping_distortion": row.group_clipping_distortion,
                "ellipsoid_norm_contribution": row.ellipsoid_norm_contribution,
                "num_coefficients": row.num_coefficients,
            }
            for row in result.group_diagnostics
        ]
        diagnostics = {
            "normalized_joint_norm_preclip": result.normalized_joint_norm_preclip,
            "joint_clip_factor": result.joint_clip_factor,
            "shared_joint_alpha": result.joint_clip_factor,
            "normalized_joint_norm_postclip": result.normalized_joint_norm_postclip,
            "clipping_distortion": result.clipping_distortion,
            "cosine_similarity": result.cosine_similarity,
            "raw_joint_norm": result.raw_joint_norm,
            "post_clip_joint_norm": result.post_clip_joint_norm,
            "min_group_clip_factor": result.joint_clip_factor,
            "max_group_clip_factor": result.joint_clip_factor,
            "relative_group_structure_error": _relative_group_structure_error(
                [row.raw_group_norm for row in result.group_diagnostics],
                [row.restored_group_norm_after_clip for row in result.group_diagnostics],
            ),
        }
        return clipped, diagnostics, group_rows

    def privatize(
        self,
        update: Mapping[str, torch.Tensor],
        round_index: int = -1,
        client_id: int = -1,
    ) -> DPUpdate:
        projected = self.projection.project_groups(transform_update(update))
        if self.mode == "spherical_current":
            clipped, diagnostics, group_rows = self._current_clip(projected)
        else:
            clipped, diagnostics, group_rows = self._ellipsoid_clip(projected)

        self.clipping_diagnostics.append(
            {
                "round": int(round_index) + 1,
                "client_id": int(client_id),
                "geometry_mode": self.mode,
                **diagnostics,
            }
        )
        for row in group_rows:
            self.group_diagnostics.append(
                {
                    "round": int(round_index) + 1,
                    "client_id": int(client_id),
                    "geometry_mode": self.mode,
                    **row,
                }
            )
        if self.current_release is not None:
            # The control delegates its untouched update to the production
            # PJSDP path. The side calculation above consumes no randomness and
            # exists only to obtain development diagnostics.
            return self.current_release.privatize(update, round_index, client_id)

        noisy: dict[str, dict[str, torch.Tensor]] = {}
        for group in GROUPS:
            noisy[group] = {}
            for name, value in clipped[group].items():
                noise = torch.randn(
                    value.shape,
                    dtype=value.dtype,
                    device=value.device,
                    generator=self.generator,
                )
                noisy[group][name] = value + noise * self.noise_stds[group]
        released = self.projection.project_groups(noisy)
        return DPUpdate(inverse_groups(released, update), True, {})


def _prepare_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing sanity output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _prepare_data(config: Mapping[str, Any], data_root: Path, dataset_name: str):
    data = load_traffic_dataset(data_root, dataset_name)
    splits = temporal_split(data.values)
    mean = splits.train.mean((0, 1), keepdims=True)
    std = splits.train.std((0, 1), keepdims=True) + 1e-6
    train = (splits.train - mean) / std
    validation = (splits.validation - mean) / std
    test = (splits.test - mean) / std
    valid_train = np.isfinite(splits.train[..., 0])
    valid_validation = np.isfinite(splits.validation[..., 0])
    valid_test = np.isfinite(splits.test[..., 0])
    if bool(config["mask_zero_targets"]):
        valid_train &= np.abs(splits.train[..., 0]) > 1e-5
        valid_validation &= np.abs(splits.validation[..., 0]) > 1e-5
        valid_test &= np.abs(splits.test[..., 0]) > 1e-5
    history = int(config["history"])
    validation_context = np.concatenate([train[-history:], validation], axis=0)
    validation_valid = np.concatenate([valid_train[-history:], valid_validation], axis=0)
    test_context = np.concatenate([validation[-history:], test], axis=0)
    test_valid = np.concatenate([valid_validation[-history:], valid_test], axis=0)
    return data, train, valid_train, validation_context, validation_valid, test_context, test_valid, mean, std


def _build_clients_and_eval(
    config: Mapping[str, Any], prepared: tuple, device: torch.device
):
    data, train, valid_train, validation_context, validation_valid, test_context, test_valid, _, _ = prepared
    partition = partition_nodes(
        data.adjacency, int(config["num_clients"]), data.coordinates, int(config["seed"])
    )
    model = STGCN(data.values.shape[-1], int(config["hidden_channels"]), int(config["horizon"]))
    clients = []
    validation_sets = []
    test_sets = []
    for client_id, nodes in enumerate(partition.node_ids):
        adjacency = torch.as_tensor(data.adjacency[np.ix_(nodes, nodes)], dtype=torch.float32)
        clients.append(
            STGClient(
                client_id,
                model,
                DataLoader(
                    WindowDataset(train[:, nodes], int(config["history"]), int(config["horizon"]), valid_train[:, nodes]),
                    batch_size=int(config["batch_size"]),
                    shuffle=True,
                ),
                adjacency,
                device,
                float(config["lr"]),
                bool(config["dynamic_graph"]),
                float(config["dynamic_alpha"]),
                int(config["dynamic_top_k"]),
                bool(config["mixed_precision"]),
            )
        )
        validation_sets.append(
            (
                DataLoader(
                    WindowDataset(validation_context[:, nodes], int(config["history"]), int(config["horizon"]), validation_valid[:, nodes]),
                    batch_size=int(config["batch_size"]),
                ),
                adjacency,
            )
        )
        test_sets.append(
            (
                DataLoader(
                    WindowDataset(test_context[:, nodes], int(config["history"]), int(config["horizon"]), test_valid[:, nodes]),
                    batch_size=int(config["batch_size"]),
                ),
                adjacency,
            )
        )
    return model, clients, validation_sets, test_sets


def _evaluate_partitions(
    model: torch.nn.Module,
    partitions: Sequence,
    device: torch.device,
    config: Mapping[str, Any],
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    totals = ForecastingTotals()
    for loader, adjacency in partitions:
        totals.merge(
            evaluate(
                model,
                loader,
                adjacency,
                device,
                float(mean[..., 0].mean()),
                float(std[..., 0].mean()),
                bool(config["dynamic_graph"]),
                float(config["dynamic_alpha"]),
                int(config["dynamic_top_k"]),
            )
        )
    return totals.compute()


def _run_one(
    *,
    config: Mapping[str, Any],
    prepared: tuple,
    schedule: Sequence[Sequence[int]],
    schedule_hash: str,
    device: torch.device,
    stage: str,
    mode: str,
    rounds: int,
    noise_mode: str,
    noise_stds: Mapping[str, float],
    noise_seed: int,
    output: Path,
) -> dict[str, Any]:
    _prepare_directory(output)
    configure_logging(output / "train.log")
    _set_seed(int(config["seed"]))
    model, clients, validation_sets, test_sets = _build_clients_and_eval(config, prepared, device)
    projection = FixedSpectralProjection(
        model.state_dict(), float(config["projection_ratio"]), int(config["projection_seed"])
    )
    current_bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
    mechanism = GeometryReleaseMechanism(
        mode,
        projection,
        current_bounds,
        noise_stds,
        torch.Generator().manual_seed(noise_seed),
    )
    server = STGServer(
        model,
        clients,
        device,
        int(config["seed"]),
        collector=None,
        privacy=mechanism,
        topk_ratio=1.0,
        error_feedback=False,
        event_tau=0.0,
        skip_policy="zero",
        report_client_train_loss=True,
        participation_schedule=schedule,
    )
    started = time.perf_counter()
    losses = []
    for round_index in range(rounds):
        row = server.run_round(
            round_index,
            int(config["clients_per_round"]),
            int(config["local_epochs"]),
            0.0,
        )
        loss = float(row["client_train_loss"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss in {stage}/{mode}")
        losses.append(loss)
        logging.info(
            "stage=%s mode=%s round=%d/%d train_loss=%.8f",
            stage,
            mode,
            round_index + 1,
            rounds,
            loss,
        )
    _, _, _, _, _, _, _, mean, std = prepared
    validation_metrics = _evaluate_partitions(
        server.model, validation_sets, device, config, mean, std
    )
    test_metrics = _evaluate_partitions(server.model, test_sets, device, config, mean, std)
    runtime = time.perf_counter() - started
    actual_geometry = (
        "independent_group_wise_clipping"
        if mode == "spherical_current"
        else "joint_ellipsoidal_radial_clipping"
    )
    run_config = {
        "status": "DEVELOPMENT_ONLY_PRIVATE_CALIBRATION",
        "formal_privacy_claim_valid": False,
        "stage": stage,
        "geometry_mode": mode,
        "actual_clipping_geometry": actual_geometry,
        "dataset": config["dataset"],
        "seed": int(config["seed"]),
        "rounds": rounds,
        "num_clients": int(config["num_clients"]),
        "clients_per_round": int(config["clients_per_round"]),
        "participation_schedule": "formal_seed42_fixed_prefix",
        "participation_schedule_hash": schedule_hash,
        "projection_ratio": projection.actual_retain_ratio,
        "projection_mask_hash": projection.mask_hash,
        "noise_mode": noise_mode,
        "noise_stds": dict(noise_stds),
        "common_noise_seed_public": True,
        "common_noise_seed": noise_seed,
        "current_clip_bounds": current_bounds,
        "joint_clip_norm": JOINT_CLIP_NORM if mode != "spherical_current" else None,
        "input_group_scales": (
            {group: 1.0 for group in GROUPS}
            if mode == "ellipsoid_equal"
            else DEV_SCALES if mode == "ellipsoid_anisotropic_dev" else None
        ),
        "normalized_group_scales": mechanism.ellipsoid.scales if mode != "spherical_current" else None,
        "scale_source": (
            "DEVELOPMENT_ONLY_PRIVATE_CALIBRATION_from_seed42_norm_audit"
            if mode == "ellipsoid_anisotropic_dev"
            else "equal_fixed" if mode == "ellipsoid_equal" else "existing_current_bounds"
        ),
        "test_data_used_for_scale_selection": False,
        "attack_metrics_used": False,
        "runtime_seconds": runtime,
    }
    save_config(output / "config.yaml", run_config)
    metrics = {
        "Stage": stage,
        "Method": mode,
        "ActualGeometry": actual_geometry,
        "Rounds": rounds,
        "NoiseMode": noise_mode,
        **{f"Val_{key}": value for key, value in validation_metrics.items()},
        **test_metrics,
        "TrainLossInitial": losses[0],
        "TrainLossFinal": losses[-1],
        "TrainLossMax": max(losses),
        "TrainingStable": all(math.isfinite(value) for value in losses),
        "RuntimeSeconds": runtime,
    }
    write_rows(output / "metrics.csv", [metrics])
    write_rows(output / "clipping_diagnostics.csv", mechanism.clipping_diagnostics)
    write_rows(output / "group_diagnostics.csv", mechanism.group_diagnostics)
    (output / "environment.txt").write_text(
        _environment_snapshot(device), encoding="utf-8"
    )
    return {
        "metrics": metrics,
        "clipping": mechanism.clipping_diagnostics,
        "groups": mechanism.group_diagnostics,
        "output": output,
    }


def _aggregate(runs: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = []
    group_summary = []
    for run in runs:
        metrics = run["metrics"]
        clipping = run["clipping"]
        summary.append(
            {
                "Stage": metrics["Stage"],
                "Method": metrics["Method"],
                "Rounds": metrics["Rounds"],
                "NoiseMode": metrics["NoiseMode"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "Val_MAE": metrics["Val_MAE"],
                "Val_RMSE": metrics["Val_RMSE"],
                "Val_MAPE": metrics["Val_MAPE"],
                "MeanJointClipFactor": float(np.mean([row["joint_clip_factor"] for row in clipping])),
                "MedianJointClipFactor": float(np.median([row["joint_clip_factor"] for row in clipping])),
                "MeanClipDistortion": float(np.mean([row["clipping_distortion"] for row in clipping])),
                "MeanGroupStructureError": float(
                    np.mean([row["relative_group_structure_error"] for row in clipping])
                ),
                "MeanRawNorm": float(np.mean([row["raw_joint_norm"] for row in clipping])),
                "MeanPostClipNorm": float(np.mean([row["post_clip_joint_norm"] for row in clipping])),
                "MeanCosineSimilarity": float(np.mean([row["cosine_similarity"] for row in clipping])),
                "CosineSimilarityStd": float(np.std([row["cosine_similarity"] for row in clipping])),
                "CosineSimilarityP10": float(np.percentile([row["cosine_similarity"] for row in clipping], 10)),
                "CosineSimilarityP50": float(np.percentile([row["cosine_similarity"] for row in clipping], 50)),
                "CosineSimilarityP90": float(np.percentile([row["cosine_similarity"] for row in clipping], 90)),
                "TrainingStable": metrics["TrainingStable"],
                "TrainLossFinal": metrics["TrainLossFinal"],
            }
        )
        for group in GROUPS:
            rows = [row for row in run["groups"] if row["group_name"] == group]
            group_summary.append(
                {
                    "Stage": metrics["Stage"],
                    "Method": metrics["Method"],
                    "Group": group,
                    "RawNormMean": float(np.mean([row["raw_group_norm"] for row in rows])),
                    "RawNormP90": float(np.percentile([row["raw_group_norm"] for row in rows], 90)),
                    "NormalizedNormMean": float(np.mean([row["normalized_group_norm"] for row in rows])),
                    "RetentionRatioMean": float(np.mean([row["group_retention_ratio"] for row in rows])),
                    "ContributionMean": float(np.mean([row["ellipsoid_norm_contribution"] for row in rows])),
                    "PostClipNormMean": float(np.mean([row["restored_group_norm_after_clip"] for row in rows])),
                    "DistortionMean": float(np.mean([row["group_clipping_distortion"] for row in rows])),
                }
            )
    return summary, group_summary


def _geometry_status(
    summary: Sequence[Mapping[str, Any]], gate: Mapping[str, Any]
) -> str:
    """Apply the declared utility-plus-distortion decision rule."""

    near = {row["Method"]: row for row in summary if row["Stage"] == "near_zero_noise"}
    current = near["spherical_current"]
    anisotropic = near["ellipsoid_anisotropic_dev"]
    utility_not_worse = (
        float(anisotropic["MAE"]) <= 1.05 * float(current["MAE"])
        and float(anisotropic["RMSE"]) <= 1.05 * float(current["RMSE"])
    )
    distortion_advantage = float(anisotropic["MeanClipDistortion"]) < 0.95 * float(
        current["MeanClipDistortion"]
    )
    if bool(gate["passed"]) and utility_not_worse and distortion_advantage:
        return "SUPPORTED"
    if bool(gate["passed"]) and utility_not_worse:
        return "LIMITED"
    return "NOT SUPPORTED"


def _configure_plots() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _plot_results(runs: Sequence[Mapping[str, Any]], root: Path) -> None:
    _configure_plots()
    near = [run for run in runs if run["metrics"]["Stage"] == "near_zero_noise"]
    colors = ["#7A8793", "#5B8DB8", "#D8895B"]
    labels = [DISPLAY[run["metrics"]["Method"]] for run in near]
    for field, filename, ylabel in (
        ("joint_clip_factor", "joint_clip_factor_distribution.png", "Effective global retention / joint alpha"),
        ("clipping_distortion", "clipping_distortion.png", "Relative clipping distortion"),
        ("relative_group_structure_error", "group_structure_error.png", "Relative group-structure error"),
    ):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        values = [[float(row[field]) for row in run["clipping"]] for run in near]
        boxes = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        ax.set_ylabel(ylabel)
        ax.set_title("Near-zero-noise clipping geometry")
        ax.tick_params(axis="x", rotation=12)
        fig.tight_layout()
        fig.savefig(root / filename, dpi=240, bbox_inches="tight")
        plt.close(fig)

    x = np.arange(len(GROUPS))
    width = 0.25
    for field, filename, ylabel in (
        ("group_retention_ratio", "group_retention_ratio.png", "Mean retained norm ratio"),
        ("ellipsoid_norm_contribution", "ellipsoid_group_contribution.png", "Mean normalized-norm contribution"),
    ):
        fig, ax = plt.subplots(figsize=(7.8, 4.4))
        for index, (run, color) in enumerate(zip(near, colors)):
            means = [
                np.mean([float(row[field]) for row in run["groups"] if row["group_name"] == group])
                for group in GROUPS
            ]
            ax.bar(x + (index - 1) * width, means, width, label=DISPLAY[run["metrics"]["Method"]], color=color)
        ax.set_xticks(x, [name.replace("_", " ").title() for name in GROUPS], rotation=15)
        ax.set_ylabel(ylabel)
        ax.set_title("Near-zero-noise group diagnostics")
        ax.legend(ncols=3, loc="upper center")
        fig.tight_layout()
        fig.savefig(root / filename, dpi=240, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.8))
    labels_all = [
        ("NZ " if run["metrics"]["Stage"] == "near_zero_noise" else "DP ")
        + DISPLAY[run["metrics"]["Method"]]
        for run in runs
    ]
    plot_colors = [
        colors[MODES.index(run["metrics"]["Method"])] for run in runs
    ]
    for ax, metric in zip(axes, ("MAE", "RMSE", "MAPE")):
        values = [float(run["metrics"][metric]) for run in runs]
        ax.bar(np.arange(len(runs)), values, color=plot_colors)
        ax.set_xticks(np.arange(len(runs)), labels_all, rotation=68, ha="right")
        ax.set_title(metric)
        ax.set_ylabel("Lower is better")
    fig.suptitle("Forecasting utility by clipping geometry")
    fig.tight_layout()
    fig.savefig(root / "forecasting_metrics.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _write_report(
    path: Path,
    config: Mapping[str, Any],
    summary: Sequence[Mapping[str, Any]],
    group_summary: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
    weak_executed: bool,
) -> None:
    near = [row for row in summary if row["Stage"] == "near_zero_noise"]
    weak = [row for row in summary if row["Stage"] == "weak_dp_endpoint"]
    anisotropic = next(row for row in near if row["Method"] == "ellipsoid_anisotropic_dev")
    current = next(row for row in near if row["Method"] == "spherical_current")
    mae_delta = 100 * (float(anisotropic["MAE"]) - float(current["MAE"])) / float(current["MAE"])
    rmse_delta = 100 * (float(anisotropic["RMSE"]) - float(current["RMSE"])) / float(current["RMSE"])
    status = _geometry_status(summary, gate)
    supported = status == "SUPPORTED"
    lines = [
        "# Ellipsoidal Spectral Clipping Sanity Report",
        "",
        "## 1. Current clipping implementation audit",
        "",
        "The current PJSDP implementation performs **independent group-wise clipping**, not one global spherical clip. Each of the five projected spectral groups has its own public bound and independent factor. This can change relative group magnitudes and the full-vector direction.",
        "",
        "## 2. Norm-audit evidence",
        "",
        "The preceding training-only audit measured group P90 norms of 3.397, 1.393, 2.286, 1.019, and 1.775. The max/min ratio is 3.33 and every group was clipped at the 0.3 reference threshold.",
        "",
        "## 3. Ellipsoid definition",
        "",
        "For fixed scales `s_b`, the implementation computes `Q=sum_b ||z_b/s_b||^2`, then applies one `alpha=min(1, 1/sqrt(Q))` to every group. It accumulates squared norms group by group and never concatenates full updates.",
        "",
        "## 4. Development scale and DP status",
        "",
        "The anisotropic input scale is `[1.72, 0.71, 1.16, 0.52, 0.90]` for Graph Low, Graph High, Temporal Low, Temporal High, and Other, normalized to arithmetic mean one. It is marked `DEVELOPMENT_ONLY_PRIVATE_CALIBRATION` because it was derived from private training-update norms. It is not treated as a privacy-free public parameter and is not a formal DP result.",
        "",
        "## 5. Experiment configuration",
        "",
        f"PEMS08; seed 42; {config['rounds']} rounds; 50 clients; 5 clients/round; the completed formal fixed schedule; 90% fixed architecture-only projection; one local epoch; identical model, optimizer, data, schedule, projection, and common Gaussian noise stream across geometry modes.",
        "",
        "The gate was predeclared on validation data: anisotropic Val_MAE and Val_RMSE must each be no more than 5% worse than the current geometry, with finite training losses. Test metrics did not decide the gate.",
        "",
        "## 6. Near-zero-noise results",
        "",
        "| Mode | MAE | RMSE | MAPE | Mean clip factor | Mean distortion | Mean structure error | Mean cosine | Stable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in near:
        lines.append(
            f"| {DISPLAY[str(row['Method'])]} | {float(row['MAE']):.4f} | {float(row['RMSE']):.4f} | {float(row['MAPE']):.4f} | {float(row['MeanJointClipFactor']):.4f} | {float(row['MeanClipDistortion']):.4f} | {float(row['MeanGroupStructureError']):.6f} | {float(row['MeanCosineSimilarity']):.6f} | {row['TrainingStable']} |"
        )
    lines.extend(
        [
            "",
            f"Anisotropic versus current test-metric change: MAE {mae_delta:+.2f}%, RMSE {rmse_delta:+.2f}%.",
            "",
            "## 7. Group retention and normalized contribution",
            "",
            "| Mode | Group | Raw norm mean | Normalized norm mean | Retention mean | Contribution mean |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in [row for row in group_summary if row["Stage"] == "near_zero_noise"]:
        lines.append(
            f"| {DISPLAY[str(row['Method'])]} | {row['Group']} | {float(row['RawNormMean']):.4f} | {float(row['NormalizedNormMean']):.4f} | {float(row['RetentionRatioMean']):.4f} | {float(row['ContributionMean']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Joint ellipsoidal modes use one shared alpha, so all nonzero groups have the same retention ratio and preserve the complete projected-update direction. The current independent geometry uses different group factors and can rotate the full update; its reported `MeanJointClipFactor` is the effective global norm-retention ratio, not a shared alpha.",
            "The relative group-structure error is the L2 distance between the L1-normalized raw and clipped five-group norm vectors. A shared scalar alpha makes this error numerically zero; independent group-wise clipping does not.",
            "",
            "Direction diagnostics (near-zero-noise client updates):",
            "",
            "| Mode | Mean | Std | P10 | P50 | P90 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in near:
        lines.append(
            f"| {DISPLAY[str(row['Method'])]} | {float(row['MeanCosineSimilarity']):.6f} | {float(row['CosineSimilarityStd']):.6f} | {float(row['CosineSimilarityP10']):.6f} | {float(row['CosineSimilarityP50']):.6f} | {float(row['CosineSimilarityP90']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## 8. Geometry hypothesis decision",
            "",
            f"Validation gate: **{'PASS' if gate['passed'] else 'FAIL'}**. Val_MAE ratio={float(gate['val_mae_ratio']):.4f}; Val_RMSE ratio={float(gate['val_rmse_ratio']):.4f}.",
            f"Final geometry status under the declared utility-plus-distortion rule: **{status}**.",
            "",
            "## 9. Weak-DP endpoint",
            "",
            f"Weak-DP sanity executed: **{'YES' if weak_executed else 'NO'}**.",
        ]
    )
    if weak:
        lines.extend(
            [
                "",
                "| Mode | MAE | RMSE | MAPE | Mean clip factor | Mean distortion |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in weak:
            lines.append(
                f"| {DISPLAY[str(row['Method'])]} | {float(row['MAE']):.4f} | {float(row['RMSE']):.4f} | {float(row['MAPE']):.4f} | {float(row['MeanJointClipFactor']):.4f} | {float(row['MeanClipDistortion']):.4f} |"
            )
    lines.extend(
        [
            "",
            "The weak-DP sanity reuses the completed configuration's per-group Gaussian standard deviations and does not modify the accountant. Because the anisotropic scale was privately calibrated and its ellipsoidal sensitivity has not been formalized, this run remains mechanism/utility evidence only.",
            "",
            "## 10. Training stability",
            "",
            "All reported runs completed with finite client training losses. Per-run initial, final, and maximum losses are stored in each `metrics.csv` and full traces are in `train.log`.",
            "",
            "## 11. Known limitations",
            "",
            "- One seed; no uncertainty estimate across seeds.",
            "- Private-audit-derived scales; no formal public scale provenance.",
            "- Near-zero noise isolates clipping geometry but is not a private release.",
            "- Weak-DP ellipsoid uses unchanged noise for fairness, not a newly proven accountant calibration.",
            "- Norm scale does not establish forecasting importance.",
            "- No attack metric, epsilon sweep, new projection, or communication change was used.",
            "",
            "## 12. Exact next recommendation",
            "",
            (
                "**Public-Scale Ellipsoidal DP Calibration**"
                if supported
                else "**Do not continue ellipsoidal clipping experiments.**"
            ),
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.results_root.exists() and any(args.results_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing results: {args.results_root}")
    config = _load_reference_config(args.reference_config)
    config["dataset"] = args.dataset
    if args.rounds <= 0 or args.rounds > int(config["rounds"]):
        raise ValueError("rounds must be within the completed reference schedule")
    device = _resolve_device(args.device or str(config["device"]))
    schedule, schedule_hash, counts = _load_schedule(
        args.participation_schedule_csv,
        args.rounds,
        int(config["num_clients"]),
        int(config["clients_per_round"]),
    )
    prepared = _prepare_data(config, args.data_root, args.dataset)
    near_noise = {group: NEAR_ZERO_STD for group in GROUPS}
    runs = []
    for mode in MODES:
        runs.append(
            _run_one(
                config=config,
                prepared=prepared,
                schedule=schedule,
                schedule_hash=schedule_hash,
                device=device,
                stage="near_zero_noise",
                mode=mode,
                rounds=args.rounds,
                noise_mode="fixed_near_zero_1e-8",
                noise_stds=near_noise,
                noise_seed=args.common_noise_seed,
                output=args.results_root / "seed_42" / ("current" if mode == "spherical_current" else mode),
            )
        )

    current = next(run for run in runs if run["metrics"]["Method"] == "spherical_current")
    anisotropic = next(run for run in runs if run["metrics"]["Method"] == "ellipsoid_anisotropic_dev")
    val_mae_ratio = float(anisotropic["metrics"]["Val_MAE"]) / float(current["metrics"]["Val_MAE"])
    val_rmse_ratio = float(anisotropic["metrics"]["Val_RMSE"]) / float(current["metrics"]["Val_RMSE"])
    gate = {
        "rule": "anisotropic Val_MAE and Val_RMSE each <= 1.05 * current, all losses finite",
        "tolerance": GATE_TOLERANCE,
        "val_mae_ratio": val_mae_ratio,
        "val_rmse_ratio": val_rmse_ratio,
        "passed": (
            val_mae_ratio <= 1.0 + GATE_TOLERANCE
            and val_rmse_ratio <= 1.0 + GATE_TOLERANCE
            and bool(anisotropic["metrics"]["TrainingStable"])
        ),
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    save_config(args.results_root / "stage1_gate.yaml", gate)

    weak_executed = False
    if gate["passed"]:
        weak_noise = {
            group: float(config["pjsdp_group_noise_stds"][group]) for group in GROUPS
        }
        current_bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
        weights = {group: float(config[f"{group}_noise_weight"]) for group in GROUPS}
        factor = 2.0 if config["pjsdp_adjacency"] == "replacement" else 1.0
        sensitivities = {group: factor * current_bounds[group] for group in GROUPS}
        verified = PJSDPAccountant.calibrate_global_scale(
            float(config["target_epsilon"]),
            float(config["delta"]),
            int(config["accounted_client_releases"]),
            sensitivities,
            weights,
            sampling_rate=float(config["clients_per_round"]) / float(config["num_clients"]),
            sampling_amplification=False,
        )
        for group in GROUPS:
            if not math.isclose(
                weak_noise[group], verified.group_noise_stds[group], rel_tol=1e-10, abs_tol=1e-12
            ):
                raise RuntimeError("reference weak-DP noise does not match verified accountant calibration")
        for mode in ("spherical_current", "ellipsoid_anisotropic_dev"):
            runs.append(
                _run_one(
                    config=config,
                    prepared=prepared,
                    schedule=schedule,
                    schedule_hash=schedule_hash,
                    device=device,
                    stage="weak_dp_endpoint",
                    mode=mode,
                    rounds=args.rounds,
                    noise_mode="existing_eps4096_noise_stds",
                    noise_stds=weak_noise,
                    noise_seed=args.common_noise_seed,
                    output=args.results_root
                    / "seed_42"
                    / (
                        "weak_dp_current"
                        if mode == "spherical_current"
                        else "weak_dp_ellipsoid_anisotropic_dev"
                    ),
                )
            )
        weak_executed = True

    summary, group_summary = _aggregate(runs)
    geometry_status = _geometry_status(summary, gate)
    write_rows(args.results_root / "summary.csv", summary)
    write_rows(args.results_root / "group_summary.csv", group_summary)
    _plot_results(runs, args.results_root)
    _write_report(args.report_path, config, summary, group_summary, gate, weak_executed)
    manifest = {
        "status": "DEVELOPMENT_ONLY_PRIVATE_CALIBRATION",
        "formal_privacy_claim_valid": False,
        "dataset": args.dataset,
        "seed": int(config["seed"]),
        "rounds": args.rounds,
        "num_clients": int(config["num_clients"]),
        "clients_per_round": int(config["clients_per_round"]),
        "min_participations": min(counts),
        "max_participations": max(counts),
        "schedule_hash": schedule_hash,
        "projection_ratio": float(config["projection_ratio"]),
        "projection_seed": int(config["projection_seed"]),
        "development_scales_input": DEV_SCALES,
        "development_scales_mean1": EllipsoidalSpectralClipper(DEV_SCALES).scales,
        "stage1_gate": gate,
        "geometry_hypothesis": geometry_status,
        "weak_dp_sanity_executed": weak_executed,
        "attack_metrics_used": False,
        "epsilon_sweep_executed": False,
    }
    save_config(args.results_root / "manifest.yaml", manifest)

    near_rows = {row["Method"]: row for row in summary if row["Stage"] == "near_zero_noise"}
    weak_rows = {row["Method"]: row for row in summary if row["Stage"] == "weak_dp_endpoint"}
    print("========================================")
    print("ELLIPSOIDAL CLIPPING SANITY SUMMARY")
    print("========================================")
    print("Current clipping geometry:")
    print("INDEPENDENT GROUP-WISE")
    print("\nStage 1:\nNear-zero-noise")
    for mode in MODES:
        row = near_rows[mode]
        print(f"{DISPLAY[mode]}: MAE={float(row['MAE']):.6f} RMSE={float(row['RMSE']):.6f} MAPE={float(row['MAPE']):.6f}")
    print("\nMean clipping distortion:")
    for mode in MODES:
        print(f"{DISPLAY[mode]}: {float(near_rows[mode]['MeanClipDistortion']):.6f}")
    print("\nMean group-structure error:")
    for mode in MODES:
        print(f"{DISPLAY[mode]}: {float(near_rows[mode]['MeanGroupStructureError']):.6f}")
    print(f"\nTraining stable: {'YES' if all(bool(row['TrainingStable']) for row in summary) else 'NO'}")
    print(f"Geometry hypothesis: {geometry_status}")
    print(f"\nStage 2 weak-DP executed: {'YES' if weak_executed else 'NO'}")
    if weak_executed:
        for mode in ("spherical_current", "ellipsoid_anisotropic_dev"):
            row = weak_rows[mode]
            print(f"{DISPLAY[mode]}: MAE={float(row['MAE']):.6f} RMSE={float(row['RMSE']):.6f} MAPE={float(row['MAPE']):.6f}")
    print("\nFormal DP status: DEVELOPMENT ONLY / NOT FINAL")
    print("\nNext recommended experiment:")
    print(
        "Public-Scale Ellipsoidal DP Calibration"
        if geometry_status == "SUPPORTED"
        else "Do not continue ellipsoidal clipping experiments."
    )
    return {
        "summary": summary,
        "group_summary": group_summary,
        "gate": gate,
        "weak_executed": weak_executed,
    }


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
