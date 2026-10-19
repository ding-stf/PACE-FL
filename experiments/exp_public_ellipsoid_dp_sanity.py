from __future__ import annotations

"""Controlled public-ellipsoid comparison at the verified weak-DP endpoint.

This experiment changes only clipping geometry and its fixed public scale.  It
uses the existing block-diagonal Gaussian covariance *shape* and calibrates one
global standard-deviation multiplier per geometry so all methods meet the same
RDP endpoint.  Private diagnostics are internal evaluation artifacts and are
not part of the DP release transcript.
"""

import argparse
import csv
import hashlib
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
    JOINT_CLIP_NORM,
    _build_clients_and_eval,
    _environment_snapshot,
    _evaluate_partitions,
    _global_dot,
    _global_norm,
    _prepare_data,
    _prepare_directory,
    _relative_group_structure_error,
)
from experiments.exp_spectral_norm_audit import (
    _load_reference_config,
    _load_schedule,
    _resolve_device,
    _set_seed,
)
from federated.stg_server import STGServer
from privacy.dp_fedavg import DPUpdate
from privacy.ellipsoidal_clipping import EllipsoidalSpectralClipper
from privacy.fixed_spectral_projection import FixedSpectralProjection
from privacy.pjsdp_accountant import NoiseCalibration, PJSDPAccountant
from privacy.spectral_group_clipping import SpectralGroupClipper
from spectral.spectral_groups import GROUPS, inverse_groups, transform_update


METHODS = ("current", "equal_joint", "dimension_joint", "synthetic_joint")
DISPLAY = {
    "current": "Current",
    "equal_joint": "Equal Joint",
    "dimension_joint": "Dimension Joint",
    "synthetic_joint": "Synthetic Joint",
}
DIMENSIONS = dict(zip(GROUPS, (512, 169, 608, 198, 252)))
NOISE_WEIGHTS = dict(zip(GROUPS, (0.75, 1.50, 0.75, 1.50, 1.00)))
TARGET_EPSILON = 4096.0
DELTA = 1e-5
ADJACENCY = "replacement"
RELEASES_PER_CLIENT = 4
SYNTHETIC_DISPLAY_KEYS = {
    "graph_low": "Graph Low",
    "graph_high": "Graph High",
    "temporal_low": "Temporal Low",
    "temporal_high": "Temporal High",
    "other": "Other",
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--reference_config", type=Path, required=True)
    p.add_argument("--participation_schedule_csv", type=Path, required=True)
    p.add_argument(
        "--synthetic_scale_file",
        type=Path,
        default=Path("results/synthetic_scale_calibration/synthetic_public_scale.yaml"),
    )
    p.add_argument(
        "--results_root",
        type=Path,
        default=Path("results/public_ellipsoid_dp_sanity"),
    )
    p.add_argument(
        "--report_path",
        type=Path,
        default=Path("docs/PUBLIC_ELLIPSOID_DP_SANITY_REPORT.md"),
    )
    p.add_argument("--device", default="cuda")
    return p


def equal_scales() -> dict[str, float]:
    return {group: 1.0 for group in GROUPS}


def dimension_scales() -> dict[str, float]:
    roots = {group: math.sqrt(float(DIMENSIONS[group])) for group in GROUPS}
    mean = sum(roots.values()) / len(roots)
    return {group: roots[group] / mean for group in GROUPS}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_synthetic_scales(path: Path) -> tuple[dict[str, float], str, dict[str, Any]]:
    """Read the locked public scale without importing or invoking calibration."""

    raw_bytes = path.read_bytes()
    payload = yaml.safe_load(raw_bytes)
    if payload.get("target_private_data_used") is not False:
        raise ValueError("synthetic scale must declare target_private_data_used=false")
    if payload.get("target_dataset") not in {None, "none"}:
        raise ValueError("synthetic scale must not name a target dataset")
    scenarios = payload.get("scenarios", [])
    seeds = payload.get("seeds", [])
    if len(scenarios) != 3 or len(seeds) != 5:
        raise ValueError("expected locked 3-scenario x 5-seed calibration provenance")
    groups = payload.get("groups", {})
    scales = {
        group: float(groups[SYNTHETIC_DISPLAY_KEYS[group]]["scale"])
        for group in GROUPS
    }
    if not all(math.isfinite(value) and value > 0 for value in scales.values()):
        raise ValueError("synthetic scales must be finite and positive")
    if not math.isclose(sum(scales.values()) / len(scales), 1.0, abs_tol=1e-10):
        raise ValueError("synthetic scales must have arithmetic mean one")
    return scales, hashlib.sha256(raw_bytes).hexdigest(), payload


def ellipsoid_whitened_sensitivity_squared(
    scales: Mapping[str, float],
    noise_stds: Mapping[str, float],
    joint_clip_norm: float = JOINT_CLIP_NORM,
    adjacency: str = ADJACENCY,
) -> float:
    """Worst-case Gaussian Mahalanobis sensitivity for a radial ellipsoid.

    If E={z:sum_b ||z_b/s_b||^2 <= C^2}, replacement adjacency gives
    E-E subseteq 2E.  For block stds sigma_b, maximizing the whitened norm
    over that ellipsoid yields (2C)^2 max_b (s_b/sigma_b)^2.
    """

    if adjacency not in {"add_remove", "replacement"}:
        raise ValueError("adjacency must be add_remove or replacement")
    factor = 2.0 if adjacency == "replacement" else 1.0
    ratios = []
    for group in GROUPS:
        scale = float(scales[group])
        std = float(noise_stds[group])
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(std) or std <= 0:
            raise ValueError("scales and noise stds must be finite and positive")
        ratios.append(scale / std)
    return (factor * float(joint_clip_norm) * max(ratios)) ** 2


def calibrate_current(
    clip_bounds: Mapping[str, float],
    target_epsilon: float = TARGET_EPSILON,
    delta: float = DELTA,
    releases: int = RELEASES_PER_CLIENT,
) -> NoiseCalibration:
    sensitivities = {group: 2.0 * float(clip_bounds[group]) for group in GROUPS}
    return PJSDPAccountant.calibrate_global_scale(
        target_epsilon,
        delta,
        releases,
        sensitivities,
        NOISE_WEIGHTS,
        sampling_rate=0.1,
        sampling_amplification=False,
    )


def calibrate_joint(
    scales: Mapping[str, float],
    target_epsilon: float = TARGET_EPSILON,
    delta: float = DELTA,
    releases: int = RELEASES_PER_CLIENT,
) -> tuple[NoiseCalibration, dict[str, float], float]:
    # Keep the covariance shape fixed.  The pseudo one-block sensitivity is an
    # exact encoding of the conservative joint Mahalanobis bound for the
    # existing accountant; the accountant implementation itself is unchanged.
    effective = 2.0 * JOINT_CLIP_NORM * max(
        float(scales[group]) / NOISE_WEIGHTS[group] for group in GROUPS
    )
    calibration = PJSDPAccountant.calibrate_global_scale(
        target_epsilon,
        delta,
        releases,
        {"joint_ellipsoid": effective},
        {"joint_ellipsoid": 1.0},
        sampling_rate=0.1,
        sampling_amplification=False,
    )
    stds = {
        group: calibration.global_scale * NOISE_WEIGHTS[group] for group in GROUPS
    }
    squared = ellipsoid_whitened_sensitivity_squared(scales, stds)
    verifier = PJSDPAccountant()
    verifier.step(
        {"joint_ellipsoid": math.sqrt(squared)},
        {"joint_ellipsoid": 1.0},
        rounds=releases,
        sampling_rate=0.1,
        sampling_amplification=False,
    )
    achieved = verifier.epsilon(delta)
    if not math.isclose(achieved, target_epsilon, rel_tol=1e-10, abs_tol=1e-8):
        raise RuntimeError("joint accountant failed to match the target epsilon")
    return calibration, stds, achieved


def _cosine(
    first: Mapping[str, Mapping[str, torch.Tensor]],
    second: Mapping[str, Mapping[str, torch.Tensor]],
    eps: float = 1e-12,
) -> float:
    first_norm = _global_norm(first)
    second_norm = _global_norm(second)
    if first_norm <= eps or second_norm <= eps:
        return 1.0
    value = _global_dot(first, second) / (first_norm * second_norm + eps)
    return min(1.0, max(-1.0, value))


class PublicGeometryReleaseMechanism:
    aggregation_level = "local"
    requires_context = True
    client_side_release = True

    def __init__(
        self,
        method: str,
        projection: FixedSpectralProjection,
        current_clip_bounds: Mapping[str, float],
        scales: Mapping[str, float] | None,
        noise_stds: Mapping[str, float],
        generator: torch.Generator,
    ) -> None:
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        self.method = method
        self.projection = projection
        self.current_clipper = SpectralGroupClipper(current_clip_bounds)
        self.ellipsoid = (
            None
            if method == "current"
            else EllipsoidalSpectralClipper(scales or {}, JOINT_CLIP_NORM, normalization="none")
        )
        self.scales = None if scales is None else dict(scales)
        self.noise_stds = {group: float(noise_stds[group]) for group in GROUPS}
        self.generator = generator
        self.clipping_diagnostics: list[dict[str, Any]] = []
        self.group_diagnostics: list[dict[str, Any]] = []
        self.noise_signal_diagnostics: list[dict[str, Any]] = []

    def _clip(self, groups: Mapping[str, Mapping[str, torch.Tensor]]):
        if self.method == "current":
            clipped, stats = self.current_clipper.clip_groups(groups)
            by_group = {row.group_name: row for row in stats}
            raw_joint = _global_norm(groups)
            post_joint = _global_norm(clipped)
            distortion_sq = sum(
                float((groups[group][name] - clipped[group][name]).detach().double().square().sum())
                for group in GROUPS for name in groups[group]
            )
            normalized_sq = sum(
                (by_group[group].pre_clip_norm / by_group[group].clip_bound) ** 2
                for group in GROUPS
            )
            post_normalized_sq = sum(
                (by_group[group].post_clip_norm / by_group[group].clip_bound) ** 2
                for group in GROUPS
            )
            group_rows = []
            for group in GROUPS:
                row = by_group[group]
                group_rows.append({
                    "group_name": group,
                    "scale": row.clip_bound,
                    "raw_group_norm": row.pre_clip_norm,
                    "normalized_group_norm": row.pre_clip_norm / row.clip_bound,
                    "restored_group_norm_after_clip": row.post_clip_norm,
                    "group_retention_ratio": row.clip_factor if row.pre_clip_norm > 0 else 1.0,
                    "group_clipping_distortion": 1.0 - row.clip_factor if row.pre_clip_norm > 0 else 0.0,
                    "ellipsoid_norm_contribution": "N/A",
                    "group_clip_factor": row.clip_factor,
                    "num_coefficients": row.num_coefficients,
                })
            diagnostics = {
                "joint_norm_preclip": math.sqrt(normalized_sq),
                "joint_clip_factor": post_joint / (raw_joint + 1e-12),
                "shared_joint_alpha": "N/A",
                "joint_norm_postclip": math.sqrt(post_normalized_sq),
                "raw_joint_norm": raw_joint,
                "post_clip_joint_norm": post_joint,
                "clipping_distortion": math.sqrt(distortion_sq) / (raw_joint + 1e-12),
                "group_structure_error": _relative_group_structure_error(
                    [by_group[group].pre_clip_norm for group in GROUPS],
                    [by_group[group].post_clip_norm for group in GROUPS],
                ),
                "clip_alignment": _cosine(groups, clipped),
            }
            return clipped, diagnostics, group_rows

        assert self.ellipsoid is not None
        clipped, result = self.ellipsoid.clip_groups(groups)
        group_rows = [{
            "group_name": row.group_name,
            "scale": row.scale,
            "raw_group_norm": row.raw_group_norm,
            "normalized_group_norm": row.normalized_group_norm,
            "restored_group_norm_after_clip": row.restored_group_norm_after_clip,
            "group_retention_ratio": row.group_retention_ratio,
            "group_clipping_distortion": row.group_clipping_distortion,
            "ellipsoid_norm_contribution": row.ellipsoid_norm_contribution,
            "group_clip_factor": row.group_retention_ratio,
            "num_coefficients": row.num_coefficients,
        } for row in result.group_diagnostics]
        diagnostics = {
            "joint_norm_preclip": result.normalized_joint_norm_preclip,
            "joint_clip_factor": result.joint_clip_factor,
            "shared_joint_alpha": result.joint_clip_factor,
            "joint_norm_postclip": result.normalized_joint_norm_postclip,
            "raw_joint_norm": result.raw_joint_norm,
            "post_clip_joint_norm": result.post_clip_joint_norm,
            "clipping_distortion": result.clipping_distortion,
            "group_structure_error": _relative_group_structure_error(
                [row.raw_group_norm for row in result.group_diagnostics],
                [row.restored_group_norm_after_clip for row in result.group_diagnostics],
            ),
            "clip_alignment": result.cosine_similarity,
        }
        return clipped, diagnostics, group_rows

    def privatize(
        self,
        update: Mapping[str, torch.Tensor],
        round_index: int = -1,
        client_id: int = -1,
    ) -> DPUpdate:
        projected = self.projection.project_groups(transform_update(update))
        clipped, diagnostics, group_rows = self._clip(projected)
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
        released_groups = self.projection.project_groups(noisy)
        noise_groups = {
            group: {
                name: released_groups[group][name] - clipped[group][name]
                for name in clipped[group]
            }
            for group in GROUPS
        }
        signal_norm = _global_norm(clipped)
        noise_norm = _global_norm(noise_groups)
        dp_alignment = _cosine(clipped, released_groups)
        common = {
            "round": int(round_index) + 1,
            "client_id": int(client_id),
            "method": self.method,
        }
        self.clipping_diagnostics.append({**common, **diagnostics, "dp_alignment": dp_alignment})
        for row in group_rows:
            self.group_diagnostics.append({**common, **row})
        self.noise_signal_diagnostics.append({
            **common,
            "signal_norm": signal_norm,
            "noise_norm": noise_norm,
            "noise_to_signal_ratio": noise_norm / (signal_norm + 1e-12),
            "clip_alignment": diagnostics["clip_alignment"],
            "dp_alignment": dp_alignment,
            "diagnostic_scope": "PRIVATE_INTERNAL_NOT_PART_OF_DP_TRANSCRIPT",
        })
        reconstructed = inverse_groups(released_groups, update)
        # Release diagnostics contain no private values.  Internal diagnostics
        # stay in this trusted experiment object and are written separately.
        return DPUpdate(reconstructed, True, {})


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "Mean": float(np.mean(array)),
        "Std": float(np.std(array)),
        "P10": float(np.percentile(array, 10)),
        "P50": float(np.percentile(array, 50)),
        "P90": float(np.percentile(array, 90)),
    }


def _run_one(
    *,
    method: str,
    config: Mapping[str, Any],
    prepared: tuple,
    schedule: Sequence[Sequence[int]],
    schedule_hash: str,
    device: torch.device,
    scales: Mapping[str, float] | None,
    scale_source: str,
    scale_metadata: Mapping[str, Any],
    noise_stds: Mapping[str, float],
    achieved_epsilon: float,
    global_noise_scale: float,
    private_master_seed: int,
    output: Path,
) -> dict[str, Any]:
    _prepare_directory(output)
    configure_logging(output / "train.log")
    _set_seed(42)
    model, clients, validation_sets, test_sets = _build_clients_and_eval(config, prepared, device)
    projection = FixedSpectralProjection(model.state_dict(), 0.90, 42)
    current_bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
    # Common random numbers across methods reduce nuisance variation.  Within a
    # method, one stream advances independently over client/round/group calls.
    mechanism = PublicGeometryReleaseMechanism(
        method,
        projection,
        current_bounds,
        scales,
        noise_stds,
        torch.Generator().manual_seed(private_master_seed),
    )
    run_config = {
        "experiment": "Synthetic-Scale Ellipsoidal DP Sanity Test",
        "method": method,
        "dataset": "pems08",
        "seed": 42,
        "rounds": 40,
        "num_clients": 50,
        "clients_per_round": 5,
        "release_count_per_client": RELEASES_PER_CLIENT,
        "participation_schedule_hash": schedule_hash,
        "projection_ratio": projection.actual_retain_ratio,
        "projection_mask_hash": projection.mask_hash,
        "target_epsilon": TARGET_EPSILON,
        "achieved_epsilon": achieved_epsilon,
        "delta": DELTA,
        "adjacency": ADJACENCY,
        "noise_weights": NOISE_WEIGHTS,
        "global_noise_scale": global_noise_scale,
        "group_noise_stds": dict(noise_stds),
        "noise_calibration_mode": (
            "existing_cartesian_group_bound" if method == "current" else
            "joint_ellipsoid_mahalanobis_bound"
        ),
        "clipping_geometry": (
            "independent_group_wise" if method == "current" else "joint_ellipsoid"
        ),
        "joint_clip_norm": None if method == "current" else JOINT_CLIP_NORM,
        "scales": None if scales is None else dict(scales),
        "scale_source": scale_source,
        "target_private_data_used_for_scale": False,
        "dp_rng_policy": "private common master stream; advances independently by client/round/group call",
        "dp_rng_seed_disclosed": False,
        "private_diagnostics_scope": "internal evaluation only; excluded from DP transcript",
        "accounting_status": "VALID",
        "attack_executed": False,
        "epsilon_sweep_executed": False,
        **dict(scale_metadata),
    }
    save_config(output / "config.yaml", run_config)
    server = STGServer(
        model,
        clients,
        device,
        42,
        collector=None,
        privacy=mechanism,
        topk_ratio=1.0,
        error_feedback=False,
        event_tau=0.0,
        skip_policy="zero",
        report_client_train_loss=True,
        participation_schedule=schedule,
    )
    losses = []
    started = time.perf_counter()
    for round_index in range(40):
        row = server.run_round(round_index, 5, 1, 0.0)
        loss = float(row["client_train_loss"])
        if not math.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss in {method}")
        losses.append(loss)
        logging.info("method=%s round=%d/40 train_loss=%.8f", method, round_index + 1, loss)
    *_, mean, std = prepared
    validation = _evaluate_partitions(server.model, validation_sets, device, config, mean, std)
    test = _evaluate_partitions(server.model, test_sets, device, config, mean, std)
    runtime = time.perf_counter() - started
    metrics = {
        "Method": DISPLAY[method],
        "MAE": test["MAE"],
        "RMSE": test["RMSE"],
        "MAPE": test["MAPE"],
        "Val_MAE": validation["MAE"],
        "Val_RMSE": validation["RMSE"],
        "Val_MAPE": validation["MAPE"],
        "TrainLossInitial": losses[0],
        "TrainLossFinal": losses[-1],
        "TrainLossMax": max(losses),
        "TrainStable": all(math.isfinite(value) for value in losses),
        "RuntimeSeconds": runtime,
    }
    write_rows(output / "metrics.csv", [metrics])
    write_rows(output / "clipping_diagnostics.csv", mechanism.clipping_diagnostics)
    write_rows(output / "group_diagnostics.csv", mechanism.group_diagnostics)
    write_rows(output / "noise_signal_diagnostics.csv", mechanism.noise_signal_diagnostics)
    privacy_row = {
        "Method": DISPLAY[method],
        "Adjacency": ADJACENCY,
        "TargetEpsilon": TARGET_EPSILON,
        "AchievedEpsilon": achieved_epsilon,
        "Delta": DELTA,
        "ReleaseCountPerClient": RELEASES_PER_CLIENT,
        "NoiseCalibrationMode": run_config["noise_calibration_mode"],
        "ClippingGeometry": run_config["clipping_geometry"],
        "ScaleSource": scale_source,
        "PublicScale": "N/A" if scales is None else yaml.safe_dump(dict(scales), sort_keys=False).strip().replace("\n", "; "),
        "TargetPrivateDataUsedForScale": False,
        "GlobalNoiseScale": global_noise_scale,
        "GroupNoiseStds": yaml.safe_dump(dict(noise_stds), sort_keys=False).strip().replace("\n", "; "),
        "AccountingStatus": "VALID",
    }
    write_rows(output / "privacy_summary.csv", [privacy_row])
    (output / "environment.txt").write_text(_environment_snapshot(device), encoding="utf-8")
    return {
        "method": method,
        "metrics": metrics,
        "privacy": privacy_row,
        "clipping": mechanism.clipping_diagnostics,
        "groups": mechanism.group_diagnostics,
        "noise": mechanism.noise_signal_diagnostics,
    }


def _summarize(runs: Sequence[Mapping[str, Any]], scales_by_method: Mapping[str, Mapping[str, float] | None]):
    summary = []
    group_summary = []
    joint_factor_summary = []
    for run in runs:
        method = str(run["method"])
        clipping = run["clipping"]
        noise = run["noise"]
        clip_dist = [float(row["clipping_distortion"]) for row in clipping]
        structure = [float(row["group_structure_error"]) for row in clipping]
        factors = [float(row["joint_clip_factor"]) for row in clipping]
        clip_alignment = [float(row["clip_alignment"]) for row in noise]
        dp_alignment = [float(row["dp_alignment"]) for row in noise]
        nsr = [float(row["noise_to_signal_ratio"]) for row in noise]
        metrics = run["metrics"]
        privacy = run["privacy"]
        summary.append({
            "Method": DISPLAY[method],
            "ScaleSource": privacy["ScaleSource"],
            "TargetEpsilon": privacy["TargetEpsilon"],
            "AchievedEpsilon": privacy["AchievedEpsilon"],
            "Delta": privacy["Delta"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "MAPE": metrics["MAPE"],
            "MeanClipDistortion": float(np.mean(clip_dist)),
            "MedianClipDistortion": float(np.median(clip_dist)),
            "MeanGroupStructureError": float(np.mean(structure)),
            "MeanNoiseToSignalRatio": float(np.mean(nsr)),
            "MeanClipAlignment": float(np.mean(clip_alignment)),
            "MeanDPAlignment": float(np.mean(dp_alignment)),
            "TrainStable": metrics["TrainStable"],
        })
        stats = _percentiles(factors)
        joint_factor_summary.append({"Method": DISPLAY[method], **stats})
        for group in GROUPS:
            rows = [row for row in run["groups"] if row["group_name"] == group]
            scale = (
                float(scales_by_method[method][group])
                if scales_by_method[method] is not None else float(rows[0]["scale"])
            )
            contribution = "N/A" if method == "current" else float(np.mean([
                float(row["ellipsoid_norm_contribution"]) for row in rows
            ]))
            group_summary.append({
                "Method": DISPLAY[method],
                "Group": group,
                "Scale": scale,
                "RawNormMean": float(np.mean([float(row["raw_group_norm"]) for row in rows])),
                "NormalizedNormMean": float(np.mean([float(row["normalized_group_norm"]) for row in rows])),
                "PostClipNormMean": float(np.mean([float(row["restored_group_norm_after_clip"]) for row in rows])),
                "RetentionRatioMean": float(np.mean([float(row["group_retention_ratio"]) for row in rows])),
                "DistortionMean": float(np.mean([float(row["group_clipping_distortion"]) for row in rows])),
                "EllipsoidContributionMean": contribution,
            })
    return summary, group_summary, joint_factor_summary


def _configure_plots() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _plots(summary: Sequence[Mapping[str, Any]], root: Path) -> None:
    _configure_plots()
    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["Method"]) for row in summary]
    colors = ["#7A8793", "#77AADD", "#E0A458", "#7FB685"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    for ax, metric in zip(axes, ("MAE", "RMSE", "MAPE")):
        values = [float(row[metric]) for row in summary]
        ax.bar(x, values, color=colors)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(metric)
        ax.set_ylabel("Lower is better")
    fig.suptitle("PEMS08 utility at matched privacy budget (seed 42)")
    fig.tight_layout()
    fig.savefig(plot_dir / "utility_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for field, filename, ylabel in (
        ("MeanClipDistortion", "clipping_distortion_comparison.png", "Mean relative clipping distortion"),
        ("MeanGroupStructureError", "group_structure_error_comparison.png", "Mean group structure error"),
        ("MeanNoiseToSignalRatio", "noise_to_signal_comparison.png", "Mean noise / signal norm"),
    ):
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        ax.bar(x, [float(row[field]) for row in summary], color=colors)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title("Internal diagnostic; lower is better")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    width = 0.35
    ax.bar(x - width / 2, [float(row["MeanClipAlignment"]) for row in summary], width, label="Raw projected vs clipped", color="#5B8DB8")
    ax.bar(x + width / 2, [float(row["MeanDPAlignment"]) for row in summary], width, label="Clipped vs DP release", color="#D8895B")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(-1.0, 1.05)
    ax.legend(ncols=2, loc="lower center")
    ax.set_title("Clipping and DP-noise alignment")
    fig.tight_layout()
    fig.savefig(plot_dir / "clip_vs_dp_alignment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    eps = [float(row["AchievedEpsilon"]) for row in summary]
    mae = [float(row["MAE"]) for row in summary]
    for index, label in enumerate(labels):
        ax.scatter(eps[index], mae[index], s=55, color=colors[index], label=label)
    ax.set_xlabel("Achieved epsilon")
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Privacy-budget matching check")
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(plot_dir / "privacy_utility_endpoint.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _decision(summary: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    rows = {str(row["Method"]): row for row in summary}
    best = min(summary, key=lambda row: (float(row["MAE"]), float(row["RMSE"])))
    best_joint = min(summary[1:], key=lambda row: (float(row["MAE"]), float(row["RMSE"])))
    equal = rows["Equal Joint"]
    synthetic = rows["Synthetic Joint"]
    dimension = rows["Dimension Joint"]
    current = rows["Current"]
    syn_equal_mae_gain = 100.0 * (float(equal["MAE"]) - float(synthetic["MAE"])) / float(equal["MAE"])
    syn_equal_rmse_gain = 100.0 * (float(equal["RMSE"]) - float(synthetic["RMSE"])) / float(equal["RMSE"])
    if str(best["Method"]) == "Current":
        conclusion = "PUBLIC ELLIPSOID NOT SUPPORTED UNDER DP"
        next_exp = "Reassess Clipping Geometry Before Epsilon Sweep"
    elif str(best_joint["Method"]) == "Dimension Joint" and str(best["Method"]) == "Dimension Joint":
        conclusion = "DIMENSION JOINT PREFERRED"
        next_exp = "Dimension-Joint Epsilon Frontier Sanity Test"
    elif str(best_joint["Method"]) == "Synthetic Joint" and (syn_equal_mae_gain >= 2.0 or syn_equal_rmse_gain >= 2.0):
        conclusion = "SYNTHETIC PUBLIC SCALE SUPPORTED ON TARGET"
        next_exp = "Synthetic-Joint Epsilon Frontier Sanity Test"
    else:
        conclusion = "EQUAL JOINT PREFERRED FOR SIMPLICITY"
        next_exp = "Equal-Joint Epsilon Frontier Sanity Test"
    if syn_equal_mae_gain >= 2.0 or syn_equal_rmse_gain >= 2.0:
        beats_equal = "YES"
    elif abs(syn_equal_mae_gain) < 1.0 and abs(syn_equal_rmse_gain) < 1.0:
        beats_equal = "MARGINAL"
    else:
        beats_equal = "NO"
    justify = "YES" if conclusion == "SYNTHETIC PUBLIC SCALE SUPPORTED ON TARGET" else ("MARGINAL" if beats_equal == "MARGINAL" else "NO")
    return {
        "best_utility": str(best["Method"]),
        "best_joint": str(best_joint["Method"]),
        "beats_equal": beats_equal,
        "justify": justify,
        "conclusion": conclusion,
        "next": next_exp,
        "syn_equal_mae_gain": f"{syn_equal_mae_gain:.3f}",
        "syn_equal_rmse_gain": f"{syn_equal_rmse_gain:.3f}",
        "syn_dimension_mae_gain": f"{100.0 * (float(dimension['MAE']) - float(synthetic['MAE'])) / float(dimension['MAE']):.3f}",
        "joint_current_mae_gain": f"{100.0 * (float(current['MAE']) - float(best_joint['MAE'])) / float(current['MAE']):.3f}",
    }


def _report(path: Path, summary: Sequence[Mapping[str, Any]], privacy: Sequence[Mapping[str, Any]], scales: Mapping[str, Mapping[str, float] | None], decision: Mapping[str, str], scale_hash: str) -> None:
    rows = {str(row["Method"]): row for row in summary}
    lines = [
        "# Public Ellipsoid DP Sanity Report", "",
        "## 1. Motivation", "",
        "This controlled seed-42 experiment asks whether a fixed target-independent synthetic spectral scale improves PEMS08 forecasting utility under the same verified weak-DP endpoint.", "",
        "## 2. Four compared methods", "",
        "Current uses independent group-wise clipping. Equal, Dimension, and Synthetic use one shared joint ellipsoidal clipping factor. The model, data split, optimizer, schedule, 90% projection, initialization, covariance-weight pattern, and evaluation are fixed.", "",
        "## 3. Why Synthetic is public", "",
        f"Synthetic scales were read once from the locked calibration YAML (SHA-256 `{scale_hash}`). The file records three scenarios, five seeds, no target dataset, and `target_private_data_used=false`. Calibration was not rerun.", "",
        "## 4. Why Dimension is public", "",
        "Dimension scales equal sqrt(released group dimension) divided by its five-group mean. They depend only on architecture-defined released coordinate counts.", "",
        "## 5. Exact scale values", "",
    ]
    for method in METHODS[1:]:
        lines.append(f"- {DISPLAY[method]}: `{[round(float(scales[method][g]), 8) for g in GROUPS]}`")
    lines += [
        "", "## 6. Experiment controls", "",
        "PEMS08; seed 42; 50 clients; 40 rounds; five clients per round; every client participates exactly four times; STGCN; one local epoch; batch size 64; learning rate 0.01; identical balanced fixed schedule; identical fixed projection mask and model initialization.", "",
        "## 7. Privacy budget matching and accounting status", "",
        "All methods use replacement adjacency, delta=1e-5, four releases per protected client, no sampling amplification, and target epsilon 4096. Current uses the existing Cartesian-product bound. Joint modes use the conservative exact ellipsoid Mahalanobis bound `(2 C_joint)^2 max_b(s_b/sigma_b)^2`. The existing accountant is unchanged and verifies each achieved epsilon. **ACCOUNTING VALID**, conditional on private diagnostic files remaining internal and outside the adversary-visible DP transcript.", "",
        "| Method | Target epsilon | Achieved epsilon | Global noise scale |", "|---|---:|---:|---:|",
    ]
    for row in privacy:
        lines.append(f"| {row['Method']} | {float(row['TargetEpsilon']):.1f} | {float(row['AchievedEpsilon']):.9f} | {float(row['GlobalNoiseScale']):.8f} |")
    lines += ["", "## 8-18. Results, diagnostics, and stability", "", "| Method | MAE | RMSE | MAPE | Clip distortion | Group error | Noise/signal | Clip align. | DP align. | Stable |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summary:
        lines.append(f"| {row['Method']} | {float(row['MAE']):.4f} | {float(row['RMSE']):.4f} | {float(row['MAPE']):.4f} | {float(row['MeanClipDistortion']):.4f} | {float(row['MeanGroupStructureError']):.6f} | {float(row['MeanNoiseToSignalRatio']):.4f} | {float(row['MeanClipAlignment']):.6f} | {float(row['MeanDPAlignment']):.6f} | {row['TrainStable']} |")
    lines += [
        "", "The run-level CSV files contain mean/std/P10/P50/P90 joint factors, per-group retention/distortion/contribution, and per-release internal noise/signal and alignment diagnostics. This is a single-seed controlled comparison; no uncertainty or significance claim is made.", "",
        "## 19. Whether Synthetic beats Equal", "",
        f"Decision: **{decision['beats_equal']}**. Synthetic relative improvement over Equal is {decision['syn_equal_mae_gain']}% in MAE and {decision['syn_equal_rmse_gain']}% in RMSE (positive means Synthetic is better).", "",
        "## 20. Whether Synthetic beats Dimension", "",
        f"Synthetic relative MAE improvement over Dimension is {decision['syn_dimension_mae_gain']}% (positive means Synthetic is better).", "",
        "## 21. Whether Joint beats Current", "",
        f"Best-joint relative MAE improvement over Current is {decision['joint_current_mae_gain']}% (positive means joint is better).", "",
        "## 22. Does synthetic calibration justify its additional complexity over Equal Joint Clipping?", "",
        f"**{decision['justify']}**.", "",
        "## 23. Accounting status", "",
        "**ACCOUNTING VALID** for the DP mechanisms and logged endpoint, conditional on keeping raw/clipping/noise diagnostics private and out of the released transcript. The scale-selection parameter itself is fixed and public for all four methods.", "",
        "## 24. Only one next recommendation", "",
        f"**{decision['next']}**", "",
        f"Final conclusion: **{decision['conclusion']}**.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.results_root.exists() and any(args.results_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing results: {args.results_root}")
    if args.report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {args.report_path}")
    config = _load_reference_config(args.reference_config)
    required = {
        "dataset": "pems08", "seed": 42, "rounds": 40, "num_clients": 50,
        "clients_per_round": 5, "local_epochs": 1, "projection_ratio": 0.9,
        "projection_seed": 42, "pjsdp_adjacency": ADJACENCY,
        "accounted_client_releases": RELEASES_PER_CLIENT,
    }
    for key, expected in required.items():
        actual = config[key]
        if isinstance(expected, float):
            valid = math.isclose(float(actual), expected, abs_tol=1e-12)
        else:
            valid = actual == expected
        if not valid:
            raise ValueError(f"reference config mismatch for {key}: {actual!r} != {expected!r}")
    schedule, schedule_hash, counts = _load_schedule(
        args.participation_schedule_csv, 40, 50, 5
    )
    if min(counts) != 4 or max(counts) != 4:
        raise ValueError("every client must participate exactly four times")
    if schedule_hash != str(config["participation_schedule_hash"]):
        raise ValueError("schedule hash does not match reference config")
    synthetic, scale_hash, scale_payload = load_synthetic_scales(args.synthetic_scale_file)
    if scale_hash != _sha256(args.synthetic_scale_file):
        raise RuntimeError("synthetic scale changed while being locked")
    scales_by_method = {
        "current": None,
        "equal_joint": equal_scales(),
        "dimension_joint": dimension_scales(),
        "synthetic_joint": synthetic,
    }
    current_bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
    current_cal = calibrate_current(current_bounds)
    calibrations: dict[str, tuple[dict[str, float], float, float]] = {
        "current": (current_cal.group_noise_stds, current_cal.achieved_epsilon, current_cal.global_scale)
    }
    for method in METHODS[1:]:
        calibration, stds, achieved = calibrate_joint(scales_by_method[method] or {})
        calibrations[method] = (stds, achieved, calibration.global_scale)
    if max(abs(value[1] - TARGET_EPSILON) for value in calibrations.values()) > 1e-7:
        raise RuntimeError("privacy endpoints are not matched")

    args.results_root.mkdir(parents=True, exist_ok=False)
    lock = {
        "synthetic_scale_file": str(args.synthetic_scale_file),
        "synthetic_scale_hash": scale_hash,
        "calibration_scenarios": 3,
        "calibration_seeds": 5,
        "target_private_data_used": False,
        "scale_file_mtime_ns": args.synthetic_scale_file.stat().st_mtime_ns,
    }
    save_config(args.results_root / "scale_lock.yaml", lock)
    prepared = _prepare_data(config, args.data_root, "pems08")
    private_master_seed = secrets.randbits(63)
    runs = []
    for method in METHODS:
        noise_stds, achieved, global_scale = calibrations[method]
        metadata = lock if method == "synthetic_joint" else {}
        runs.append(_run_one(
            method=method,
            config=config,
            prepared=prepared,
            schedule=schedule,
            schedule_hash=schedule_hash,
            device=_resolve_device(args.device),
            scales=scales_by_method[method],
            scale_source={
                "current": "existing_public_clip_bounds",
                "equal_joint": "equal_public_fixed",
                "dimension_joint": "architecture_released_dimensions",
                "synthetic_joint": "locked_target_independent_synthetic_yaml",
            }[method],
            scale_metadata=metadata,
            noise_stds=noise_stds,
            achieved_epsilon=achieved,
            global_noise_scale=global_scale,
            private_master_seed=private_master_seed,
            output=args.results_root / "seed_42" / method,
        ))
    summary, group_summary, factor_summary = _summarize(runs, scales_by_method)
    write_rows(args.results_root / "summary.csv", summary)
    write_rows(args.results_root / "group_summary.csv", group_summary)
    write_rows(args.results_root / "privacy_comparison.csv", [run["privacy"] for run in runs])
    write_rows(args.results_root / "joint_clip_factor_summary.csv", factor_summary)
    _plots(summary, args.results_root)
    decision = _decision(summary)
    _report(args.report_path, summary, [run["privacy"] for run in runs], scales_by_method, decision, scale_hash)
    print("========================================")
    print("PUBLIC ELLIPSOID DP SANITY SUMMARY")
    print("========================================")
    print("\nDataset:\nPEMS08\n\nSeed:\n42\n\nRounds:\n40")
    print("\nTarget epsilon:\n4096\n\nDelta:\n1e-5")
    print("\nACCOUNTING STATUS:\nVALID")
    by_method = {str(row["Method"]): row for row in summary}
    for method in METHODS:
        row = by_method[DISPLAY[method]]
        print(f"\n---\n\n## {DISPLAY[method].upper()}")
        print(f"\nMAE = {float(row['MAE']):.6f}\nRMSE = {float(row['RMSE']):.6f}\nMAPE = {float(row['MAPE']):.6f}")
        print(f"\nAchieved epsilon = {float(row['AchievedEpsilon']):.9f}")
        print(f"\nMean clipping distortion = {float(row['MeanClipDistortion']):.6f}")
        print(f"\nMean group structure error = {float(row['MeanGroupStructureError']):.6f}")
        print(f"\nNoise/signal ratio = {float(row['MeanNoiseToSignalRatio']):.6f}")
    print(f"\n---\n\nBest utility method:\n{decision['best_utility']}")
    print(f"\nBest joint method:\n{decision['best_joint']}")
    print(f"\nDoes synthetic calibration beat Equal:\n{decision['beats_equal']}")
    print(f"\nDoes synthetic calibration justify complexity:\n{decision['justify']}")
    print(f"\nFinal conclusion:\n\n{decision['conclusion']}")
    print(f"\nNext recommended experiment:\n\n{decision['next']}")
    return {"summary": summary, "decision": decision, "scale_hash": scale_hash}


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
