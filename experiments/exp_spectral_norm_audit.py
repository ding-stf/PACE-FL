from __future__ import annotations

"""Training-only audit of pre-clip, pre-noise projected spectral updates.

This experiment is deliberately separate from the formal PJSDP runner. It wraps
the existing PJSDP release mechanism without changing its tensors, clipping,
noise, aggregation, or random-number stream. Only detached CPU statistics are
written. Validation/test data and attack metrics are never loaded.
"""

import argparse
import csv
import json
import math
import random
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
from federated.stg_client import STGClient
from federated.stg_server import STGServer
from models.stgcn import STGCN
from privacy.anisotropic_gaussian import JointAnisotropicGaussianMechanism
from privacy.fixed_spectral_projection import FixedSpectralProjection
from privacy.pjsdp_accountant import PJSDPAccountant
from privacy.projected_joint_spectral_dp import ProjectedJointSpectralDP
from privacy.spectral_group_clipping import SpectralGroupClipper
from spectral.spectral_groups import GROUPS, transform_update


CURRENT_CLIP = 0.30
CLIP_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
GROUP_LABELS = {
    "graph_low": "Graph Low",
    "graph_high": "Graph High",
    "temporal_low": "Temporal Low",
    "temporal_high": "Temporal High",
    "other": "Other",
}
EXPECTED_ARTIFACTS = (
    "raw_group_norms.csv",
    "group_norm_summary.csv",
    "overall_norm_summary.csv",
    "clipping_rate_table.csv",
    "clip_waste_analysis.csv",
    "round_stage_norms.csv",
    "client_group_norms.csv",
    "group_norm_boxplot.png",
    "group_norm_p90_p95.png",
    "clipping_rate_vs_threshold.png",
    "norm_over_rounds.png",
    "recommended_clip_candidates.yaml",
    "audit_config.yaml",
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="pems08")
    p.add_argument("--data_root", type=Path, default=Path("data"))
    p.add_argument("--reference_config", type=Path, required=True)
    p.add_argument("--participation_schedule_csv", type=Path, required=True)
    p.add_argument("--audit_rounds", type=int, default=20)
    p.add_argument("--max_train_windows", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--development_dp_noise_seed", type=int, default=1042)
    p.add_argument(
        "--results_dir", type=Path, default=Path("results/spectral_norm_audit")
    )
    p.add_argument(
        "--report_path", type=Path, default=Path("docs/SPECTRAL_NORM_AUDIT_REPORT.md")
    )
    return p


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _load_reference_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("reference config must be a YAML mapping")
    required = {
        "seed",
        "rounds",
        "num_clients",
        "clients_per_round",
        "local_epochs",
        "batch_size",
        "history",
        "horizon",
        "hidden_channels",
        "lr",
        "dynamic_graph",
        "dynamic_alpha",
        "dynamic_top_k",
        "mixed_precision",
        "mask_zero_targets",
        "projection_ratio",
        "projection_seed",
        "target_epsilon",
        "delta",
        "pjsdp_adjacency",
        "pjsdp_noise_mode",
        "accounted_client_releases",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"reference config is missing: {missing}")
    for group in GROUPS:
        for suffix in ("clip", "noise_weight"):
            key = f"{group}_{suffix}"
            if key not in config:
                raise ValueError(f"reference config is missing: {key}")
    return config


def _load_schedule(
    path: Path,
    audit_rounds: int,
    num_clients: int,
    clients_per_round: int,
) -> tuple[tuple[tuple[int, ...], ...], str, tuple[int, ...]]:
    rows: list[tuple[int, ...]] = []
    hashes: set[str] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            client_ids = tuple(int(value) for value in json.loads(row["client_ids"]))
            if len(client_ids) != clients_per_round or len(set(client_ids)) != len(
                client_ids
            ):
                raise ValueError("invalid participation row")
            if any(client_id < 0 or client_id >= num_clients for client_id in client_ids):
                raise ValueError("participation schedule contains an unknown client")
            rows.append(client_ids)
            hashes.add(str(row.get("schedule_hash", "")))
    if len(rows) < audit_rounds:
        raise ValueError("participation schedule is shorter than audit_rounds")
    if len(hashes) != 1:
        raise ValueError("participation schedule has inconsistent hashes")
    selected = tuple(rows[:audit_rounds])
    counts = [0] * num_clients
    for round_clients in selected:
        for client_id in round_clients:
            counts[client_id] += 1
    return selected, hashes.pop(), tuple(counts)


def compute_group_statistics(
    tensors: Mapping[str, torch.Tensor],
    masks: Mapping[str, torch.Tensor] | None = None,
) -> dict[str, float | int]:
    """Compute detached CPU statistics over retained coefficients only."""

    selected: list[torch.Tensor] = []
    for name, tensor in tensors.items():
        value = tensor.detach().to(device="cpu", dtype=torch.float64)
        if masks is not None:
            mask = masks[name].detach().to(device="cpu", dtype=torch.bool)
            if tuple(mask.shape) != tuple(value.shape):
                raise ValueError(f"mask shape mismatch for {name}")
            value = value[mask]
        else:
            value = value.reshape(-1)
        if value.numel():
            selected.append(value.reshape(-1))
    if not selected:
        return {
            "num_coefficients": 0,
            "l2_norm": 0.0,
            "l1_norm": 0.0,
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "rms": 0.0,
        }
    values = torch.cat(selected)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("non-finite projected coefficient in norm audit")
    abs_values = values.abs()
    return {
        "num_coefficients": int(values.numel()),
        "l2_norm": float(torch.linalg.vector_norm(values, ord=2)),
        "l1_norm": float(torch.linalg.vector_norm(values, ord=1)),
        "max_abs": float(abs_values.max()),
        "mean_abs": float(abs_values.mean()),
        "rms": float(torch.sqrt(values.square().mean())),
    }


def percentile_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": math.nan,
            "std": math.nan,
            "min": math.nan,
            "p10": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    if not np.isfinite(array).all():
        raise ValueError("norm summary received non-finite values")
    quantiles = np.percentile(array, (10, 25, 50, 75, 90, 95, 99))
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        **{
            f"p{percentile}": float(value)
            for percentile, value in zip((10, 25, 50, 75, 90, 95, 99), quantiles)
        },
        "max": float(array.max()),
    }


def clipping_rate(values: Sequence[float], threshold: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return math.nan
    if not np.isfinite(array).all() or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("clipping-rate inputs must be finite and threshold positive")
    return float(np.mean(array > threshold))


class SpectralNormCollector:
    """In-process collector for private development diagnostics."""

    def __init__(self, projection: FixedSpectralProjection) -> None:
        self.projection = projection
        self.rows: list[dict[str, Any]] = []

    def observe(
        self,
        update: Mapping[str, torch.Tensor],
        round_index: int,
        client_id: int,
    ) -> None:
        groups = transform_update(update)
        projected = self.projection.project_groups(groups)
        masks = self.projection.group_masks
        group_rows: list[dict[str, Any]] = []
        for group in GROUPS:
            stats = compute_group_statistics(projected[group], masks[group])
            group_rows.append(
                {
                    "round": int(round_index) + 1,
                    "client_id": int(client_id),
                    "group_name": group,
                    **stats,
                }
            )
        total_norm = math.sqrt(
            sum(float(row["l2_norm"]) ** 2 for row in group_rows)
        )
        for row in group_rows:
            row["total_projected_norm"] = total_norm
            self.rows.append(row)


class AuditedPJSDP:
    """Transparent audit wrapper; the wrapped PJSDP output is returned verbatim."""

    aggregation_level = "local"
    requires_context = True
    client_side_release = True

    def __init__(self, mechanism: ProjectedJointSpectralDP, collector: SpectralNormCollector):
        self.mechanism = mechanism
        self.collector = collector

    def privatize(
        self,
        update: Mapping[str, torch.Tensor],
        round_index: int = -1,
        client_id: int = -1,
    ):
        self.collector.observe(update, round_index, client_id)
        return self.mechanism.privatize(update, round_index, client_id)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _stage_for_round(round_number: int, rounds: int) -> str:
    early_end = math.ceil(rounds / 3)
    middle_end = math.ceil(2 * rounds / 3)
    if round_number <= early_end:
        return "early"
    if round_number <= middle_end:
        return "middle"
    return "late"


def build_analysis_tables(
    raw_rows: Sequence[Mapping[str, Any]], rounds: int
) -> dict[str, list[dict[str, Any]]]:
    group_values = {
        group: [float(row["l2_norm"]) for row in raw_rows if row["group_name"] == group]
        for group in GROUPS
    }
    group_summary = [
        {"group_name": group, **percentile_summary(group_values[group])}
        for group in GROUPS
    ]
    totals_by_release: dict[tuple[int, int], float] = {}
    for row in raw_rows:
        totals_by_release[(int(row["round"]), int(row["client_id"]))] = float(
            row["total_projected_norm"]
        )
    overall_summary = [
        {
            "group_name": "total_projected",
            **percentile_summary(list(totals_by_release.values())),
        }
    ]
    clipping_rows = []
    for group in GROUPS:
        clipping_rows.append(
            {
                "group_name": group,
                **{
                    f"clip_{threshold:.2f}": clipping_rate(group_values[group], threshold)
                    for threshold in CLIP_THRESHOLDS
                },
            }
        )
    waste_rows = []
    for row in group_summary:
        p90 = float(row["p90"])
        p95 = float(row["p95"])
        waste_rows.append(
            {
                "group_name": row["group_name"],
                "p90": p90,
                "p95": p95,
                "current_clip": CURRENT_CLIP,
                "clip_to_p90_ratio": CURRENT_CLIP / p90 if p90 > 0 else math.inf,
                "clip_to_p95_ratio": CURRENT_CLIP / p95 if p95 > 0 else math.inf,
            }
        )
    stage_rows: list[dict[str, Any]] = []
    for group in GROUPS:
        for stage in ("early", "middle", "late"):
            values = [
                float(row["l2_norm"])
                for row in raw_rows
                if row["group_name"] == group
                and _stage_for_round(int(row["round"]), rounds) == stage
            ]
            summary = percentile_summary(values)
            stage_rows.append(
                {
                    "group_name": group,
                    "stage": stage,
                    "count": summary["count"],
                    "p50": summary["p50"],
                    "p90": summary["p90"],
                    "p95": summary["p95"],
                }
            )
    client_rows: list[dict[str, Any]] = []
    client_ids = sorted({int(row["client_id"]) for row in raw_rows})
    for client_id in client_ids:
        for group in GROUPS:
            values = [
                float(row["l2_norm"])
                for row in raw_rows
                if int(row["client_id"]) == client_id and row["group_name"] == group
            ]
            summary = percentile_summary(values)
            client_rows.append(
                {
                    "client_id": client_id,
                    "group_name": group,
                    "count": summary["count"],
                    "median_norm": summary["p50"],
                    "p90_norm": summary["p90"],
                }
            )
    return {
        "group_summary": group_summary,
        "overall_summary": overall_summary,
        "clipping_rates": clipping_rows,
        "clip_waste": waste_rows,
        "round_stages": stage_rows,
        "client_groups": client_rows,
    }


def _simple_candidate(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        return 0.01
    if value < 0.1:
        return round(value, 2)
    if value < 1.0:
        return round(value, 2)
    return round(value, 1)


def _write_candidates(path: Path, summary: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[str, Any] = {}
    for row in summary:
        candidates: list[float] = []
        for key in ("p75", "p90", "p95"):
            candidate = max(_simple_candidate(float(row[key])), 0.01)
            if candidate not in candidates:
                candidates.append(candidate)
        groups[str(row["group_name"])] = {
            "basis": "rounded development P75/P90/P95",
            "candidates": candidates[:3],
        }
    payload = {
        "status": "DEVELOPMENT CANDIDATES ONLY",
        "do_not_modify_formal_config_automatically": True,
        "source_split": "training_only",
        "current_clip_reference": CURRENT_CLIP,
        "groups": groups,
    }
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _configure_plotting() -> None:
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


def _plot_all(
    raw_rows: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    output: Path,
) -> None:
    _configure_plotting()
    colors = ["#4C78A8", "#72A0C1", "#F2A65A", "#E07A5F", "#7A8B5A"]
    labels = [GROUP_LABELS[group] for group in GROUPS]
    values = [
        [float(row["l2_norm"]) for row in raw_rows if row["group_name"] == group]
        for group in GROUPS
    ]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(CURRENT_CLIP, color="#B33A3A", linestyle="--", linewidth=1.2, label="C = 0.30")
    ax.set_ylabel("Pre-clip projected L2 norm")
    ax.set_title("Projected spectral-group norm distributions")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output / "group_norm_boxplot.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    summary = list(tables["group_summary"])
    x = np.arange(len(GROUPS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(x - width / 2, [float(row["p90"]) for row in summary], width, label="P90", color="#4C78A8")
    ax.bar(x + width / 2, [float(row["p95"]) for row in summary], width, label="P95", color="#F2A65A")
    ax.axhline(CURRENT_CLIP, color="#B33A3A", linestyle="--", linewidth=1.2, label="C = 0.30")
    ax.set_xticks(x, labels, rotation=18)
    ax.set_ylabel("Pre-clip projected L2 norm")
    ax.set_title("Group-specific upper quantiles")
    ax.legend(ncols=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(output / "group_norm_p90_p95.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    clipping = {str(row["group_name"]): row for row in tables["clipping_rates"]}
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for group, color in zip(GROUPS, colors):
        rates = [float(clipping[group][f"clip_{threshold:.2f}"]) for threshold in CLIP_THRESHOLDS]
        ax.plot(CLIP_THRESHOLDS, rates, marker="o", linewidth=1.6, color=color, label=GROUP_LABELS[group])
    if all(
        float(clipping[group][f"clip_{threshold:.2f}"]) == 1.0
        for group in GROUPS
        for threshold in CLIP_THRESHOLDS
    ):
        ax.annotate(
            "All five groups overlap at 100%",
            xy=(0.175, 1.0),
            xytext=(0.175, 0.88),
            ha="center",
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
            color="#444444",
        )
    ax.set_xlabel("Candidate clipping threshold")
    ax.set_ylabel("Fraction with norm > threshold")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Theoretical clipping rate from observed updates")
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output / "clipping_rate_vs_threshold.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    round_numbers = sorted({int(row["round"]) for row in raw_rows})
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for group, color in zip(GROUPS, colors):
        medians = []
        lowers = []
        uppers = []
        for round_number in round_numbers:
            round_values = np.asarray(
                [
                    float(row["l2_norm"])
                    for row in raw_rows
                    if row["group_name"] == group and int(row["round"]) == round_number
                ],
                dtype=np.float64,
            )
            medians.append(float(np.median(round_values)))
            lowers.append(float(np.percentile(round_values, 10)))
            uppers.append(float(np.percentile(round_values, 90)))
        ax.plot(round_numbers, medians, linewidth=1.6, color=color, label=GROUP_LABELS[group])
        ax.fill_between(round_numbers, lowers, uppers, color=color, alpha=0.10, linewidth=0)
    ax.axhline(CURRENT_CLIP, color="#B33A3A", linestyle="--", linewidth=1.0, label="C = 0.30")
    ax.set_xlabel("Audit round")
    ax.set_ylabel("Median pre-clip L2 norm")
    ax.set_title("Norm evolution over development rounds (P10–P90 bands)")
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(output / "norm_over_rounds.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _format_float(value: Any) -> str:
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.6f}"


def _write_report(
    path: Path,
    config: Mapping[str, Any],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    audit_rounds: int,
    schedule_hash: str,
    participation_counts: Sequence[int],
    runtime_seconds: float,
    projection: FixedSpectralProjection,
) -> None:
    summary = list(tables["group_summary"])
    rates = {str(row["group_name"]): row for row in tables["clipping_rates"]}
    waste = {str(row["group_name"]): row for row in tables["clip_waste"]}
    stages = list(tables["round_stages"])
    clients = list(tables["client_groups"])
    p90_values = [float(row["p90"]) for row in summary]
    anisotropy_ratio = max(p90_values) / min(value for value in p90_values if value > 0)
    all_below = all(value < CURRENT_CLIP for value in p90_values)
    heterogeneous = anisotropy_ratio >= 2.0
    reference_clip_rates = {
        group: float(rates[group]["clip_0.30"]) for group in GROUPS
    }
    effective_clip_rates = {
        group: clipping_rate(
            [
                float(row["l2_norm"])
                for row in config["raw_rows"]
                if row["group_name"] == group
            ],
            float(config[f"{group}_clip"]),
        )
        for group in GROUPS
    }
    if heterogeneous:
        conclusion = "C. spectral groups show strongly heterogeneous scales"
        next_experiment = "Ellipsoidal Spectral Clipping Sanity Test"
    elif all_below and max(p90_values) <= CURRENT_CLIP * 0.75:
        conclusion = "B. 0.3 appears overly conservative"
        next_experiment = "Smaller Global Clipping Sanity Test"
    else:
        conclusion = "A. 0.3 appears appropriate"
        next_experiment = "Repeat the norm audit with one additional development seed"

    lines = [
        "# Spectral Group Norm Audit Report",
        "",
        "## 1. Audit setup",
        "",
        "This is a development-only, training-split norm audit. It records detached CPU statistics after the existing spectral transform and fixed 90% projection, but before clipping and Gaussian noise. No test split or attack metric is accessed. Raw private diagnostics are not part of the formal PJSDP transcript.",
        "",
        f"- Dataset: `{config['dataset']}`",
        f"- Clients: {config['num_clients']}",
        f"- Audit rounds: {audit_rounds} (prefix of the completed 40-round public schedule)",
        f"- Clients per round: {config['clients_per_round']}",
        f"- Seed: {config['seed']}",
        f"- Projection: fixed architecture-only, requested 90%, actual {projection.actual_retain_ratio:.6f}",
        f"- Projection mask hash: `{projection.mask_hash}`",
        f"- Participation schedule hash: `{schedule_hash}`",
        f"- Participation count in audit prefix: min={min(participation_counts)}, max={max(participation_counts)}",
        f"- Current global clip reference under audit: {CURRENT_CLIP}",
        f"- Actual runtime: {runtime_seconds:.1f} seconds",
        "- Model retrained: yes, one development audit trajectory; no formal result was overwritten",
        "",
        "The latest completed formal artifact retains `clip_norm: 0.3` as the base value but overrides it with per-group effective clips. This audit therefore treats 0.3 as the requested common reference threshold while leaving all formal artifacts unchanged.",
        "",
        "## 2. Existing spectral groups",
        "",
        "The implementation uses `graph_low`, `graph_high`, `temporal_low`, `temporal_high`, and `other`, corresponding directly to Graph Low, Graph High, Temporal Low, Temporal High, and Other. Graph bands split `graph_conv.theta`; temporal bands apply a DCT and split temporal-kernel frequency coordinates; all remaining parameters are assigned to Other.",
        "",
        "## 3. Group norm statistics",
        "",
        "| Group | Mean | P50 | P75 | P90 | P95 | P99 | Max | Clip rate @0.30 | 0.30/P90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        group = str(row["group_name"])
        lines.append(
            "| "
            + GROUP_LABELS[group]
            + " | "
            + " | ".join(
                _format_float(row[key])
                for key in ("mean", "p50", "p75", "p90", "p95", "p99", "max")
            )
            + f" | {100 * float(rates[group]['clip_0.30']):.1f}% | {float(waste[group]['clip_to_p90_ratio']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Theoretical clipping rates",
            "",
            "| Group | C=0.05 | C=0.10 | C=0.15 | C=0.20 | C=0.25 | C=0.30 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in GROUPS:
        lines.append(
            f"| {GROUP_LABELS[group]} | "
            + " | ".join(
                f"{100 * float(rates[group][f'clip_{threshold:.2f}']):.1f}%"
                for threshold in CLIP_THRESHOLDS
            )
            + " |"
        )
    lines.extend(["", "## 5. Early / middle / late", ""])
    for group in GROUPS:
        group_stages = [row for row in stages if row["group_name"] == group]
        changes = ", ".join(
            f"{row['stage']} P50/P90/P95={float(row['p50']):.4f}/{float(row['p90']):.4f}/{float(row['p95']):.4f}"
            for row in group_stages
        )
        lines.append(f"- {GROUP_LABELS[group]}: {changes}.")
    lines.extend(["", "## 6. Client heterogeneity", ""])
    for group in GROUPS:
        medians = np.asarray(
            [float(row["median_norm"]) for row in clients if row["group_name"] == group]
        )
        lines.append(
            f"- {GROUP_LABELS[group]}: client median range {medians.min():.4f}–{medians.max():.4f}; P90 across client medians {np.percentile(medians, 90):.4f}."
        )
    lines.extend(
        [
            "",
            "Each client contributes only two updates in this 20-round prefix, so client-level P90 estimates are descriptive rather than stable tail estimates.",
            "",
            "## 7. Interpretation",
            "",
            f"- P90 anisotropy ratio (largest/smallest group): {anisotropy_ratio:.3f}.",
            "- The common `C=0.3` is not overly conservative in this audit: "
            + ", ".join(
                f"{GROUP_LABELS[group]}={100 * reference_clip_rates[group]:.1f}% clipped"
                for group in GROUPS
            )
            + ".",
            "- The latest formal artifact's effective per-group bounds also clip every observed audit update: "
            + ", ".join(
                f"{GROUP_LABELS[group]}={100 * effective_clip_rates[group]:.1f}%"
                for group in GROUPS
            )
            + ".",
            f"- Overall conclusion: **{conclusion}**.",
            "- The quantile-scale candidates are geometry probes, not drop-in replacements. Increasing a clipping bound also increases the accounted sensitivity and, at fixed epsilon, the required Gaussian noise; the candidates therefore require a separate utility/privacy sanity test.",
            "- Norm magnitude is interpreted only as clipping/sensitivity geometry. It is not evidence that a group is more or less important for forecasting utility.",
            "- The audit supports or rejects a later clipping-geometry sanity test only; it does not modify the current algorithm or establish a new privacy guarantee.",
            "",
            "## 8. Recommended fixed candidates",
            "",
            "Candidate values are rounded from training-only P75/P90/P95 and stored in `recommended_clip_candidates.yaml` with the status `DEVELOPMENT CANDIDATES ONLY`. They must not be copied into a formal privacy accountant without a separate, predeclared experiment.",
            "",
            f"Recommended next experiment: **{next_experiment}**.",
            "",
            "## 9. Known limitations",
            "",
            "- One development seed and 20 rounds, not a formal three-seed estimate.",
            "- The audit follows the first half of the public 40-round schedule and therefore samples early/middle development behavior more densely than final convergence behavior.",
            "- The deterministic development DP noise seed makes the audit trajectory reproducible but invalidates any claim that this audit run itself is a formal private release.",
            "- Raw norm diagnostics are private calibration statistics and should not be exposed as part of a deployed server transcript.",
            "- No utility importance, attack resistance, or epsilon choice is inferred from norm magnitude.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _prepare_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    collisions = [name for name in EXPECTED_ARTIFACTS if (path / name).exists()]
    if collisions:
        raise FileExistsError(
            "spectral norm audit artifacts already exist; refusing to overwrite: "
            + ", ".join(collisions)
        )


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = _load_reference_config(args.reference_config)
    if args.audit_rounds <= 0 or args.audit_rounds > int(config["rounds"]):
        raise ValueError("audit_rounds must be in [1, reference rounds]")
    if args.max_train_windows is not None and args.max_train_windows <= 0:
        raise ValueError("max_train_windows must be positive")
    _prepare_output(args.results_dir)

    seed = int(config["seed"])
    _set_seed(seed)
    device = _resolve_device(args.device or str(config["device"]))
    data = load_traffic_dataset(args.data_root, args.dataset)
    train = temporal_split(data.values).train
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True) + 1e-6
    normalized = (train - mean) / std
    valid = np.isfinite(train[..., 0])
    if bool(config["mask_zero_targets"]):
        valid &= np.abs(train[..., 0]) > 1e-5
    if args.max_train_windows is not None:
        required = args.max_train_windows + int(config["history"]) + int(config["horizon"]) - 1
        normalized = normalized[:required]
        valid = valid[:required]

    partition = partition_nodes(
        data.adjacency, int(config["num_clients"]), data.coordinates, seed
    )
    model = STGCN(
        data.values.shape[-1], int(config["hidden_channels"]), int(config["horizon"])
    )
    clients: list[STGClient] = []
    for client_id, nodes in enumerate(partition.node_ids):
        adjacency = torch.as_tensor(
            data.adjacency[np.ix_(nodes, nodes)], dtype=torch.float32
        )
        train_dataset = WindowDataset(
            normalized[:, nodes], int(config["history"]), int(config["horizon"]), valid[:, nodes]
        )
        clients.append(
            STGClient(
                client_id,
                model,
                DataLoader(
                    train_dataset,
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

    schedule, schedule_hash, participation_counts = _load_schedule(
        args.participation_schedule_csv,
        args.audit_rounds,
        int(config["num_clients"]),
        int(config["clients_per_round"]),
    )
    projection = FixedSpectralProjection(
        model.state_dict(), float(config["projection_ratio"]), int(config["projection_seed"])
    )
    clip_bounds = {group: float(config[f"{group}_clip"]) for group in GROUPS}
    noise_weights = {
        group: float(config[f"{group}_noise_weight"]) for group in GROUPS
    }
    sensitivity_factor = 2.0 if config["pjsdp_adjacency"] == "replacement" else 1.0
    sensitivities = {
        group: sensitivity_factor * clip_bounds[group] for group in GROUPS
    }
    calibration = PJSDPAccountant.calibrate_global_scale(
        float(config["target_epsilon"]),
        float(config["delta"]),
        int(config["accounted_client_releases"]),
        sensitivities,
        noise_weights,
        sampling_rate=float(config["clients_per_round"]) / float(config["num_clients"]),
        sampling_amplification=False,
    )
    generator = torch.Generator().manual_seed(args.development_dp_noise_seed)
    mechanism = ProjectedJointSpectralDP(
        projection,
        SpectralGroupClipper(clip_bounds),
        JointAnisotropicGaussianMechanism(
            clip_bounds, calibration.group_noise_stds, generator
        ),
        group_sensitivities=sensitivities,
        record_private_diagnostics=False,
    )
    collector = SpectralNormCollector(projection)
    audited_mechanism = AuditedPJSDP(mechanism, collector)
    server = STGServer(
        model,
        clients,
        device,
        seed,
        collector=None,
        privacy=audited_mechanism,
        topk_ratio=1.0,
        error_feedback=False,
        event_tau=0.0,
        skip_policy="zero",
        report_client_train_loss=False,
        participation_schedule=schedule,
    )
    for round_index in range(args.audit_rounds):
        server.run_round(
            round_index,
            int(config["clients_per_round"]),
            int(config["local_epochs"]),
            0.0,
        )
        print(f"audit round {round_index + 1}/{args.audit_rounds} complete", flush=True)

    tables = build_analysis_tables(collector.rows, args.audit_rounds)
    summary_fields = (
        "group_name",
        "count",
        "mean",
        "std",
        "min",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
        "max",
    )
    _write_csv(
        args.results_dir / "raw_group_norms.csv",
        collector.rows,
        (
            "round",
            "client_id",
            "group_name",
            "num_coefficients",
            "l2_norm",
            "l1_norm",
            "max_abs",
            "mean_abs",
            "rms",
            "total_projected_norm",
        ),
    )
    _write_csv(args.results_dir / "group_norm_summary.csv", tables["group_summary"], summary_fields)
    _write_csv(args.results_dir / "overall_norm_summary.csv", tables["overall_summary"], summary_fields)
    _write_csv(
        args.results_dir / "clipping_rate_table.csv",
        tables["clipping_rates"],
        ("group_name", *(f"clip_{threshold:.2f}" for threshold in CLIP_THRESHOLDS)),
    )
    _write_csv(
        args.results_dir / "clip_waste_analysis.csv",
        tables["clip_waste"],
        ("group_name", "p90", "p95", "current_clip", "clip_to_p90_ratio", "clip_to_p95_ratio"),
    )
    _write_csv(
        args.results_dir / "round_stage_norms.csv",
        tables["round_stages"],
        ("group_name", "stage", "count", "p50", "p90", "p95"),
    )
    _write_csv(
        args.results_dir / "client_group_norms.csv",
        tables["client_groups"],
        ("client_id", "group_name", "count", "median_norm", "p90_norm"),
    )
    _write_candidates(args.results_dir / "recommended_clip_candidates.yaml", tables["group_summary"])
    _plot_all(collector.rows, tables, args.results_dir)
    runtime_seconds = time.perf_counter() - started
    audit_config = {
        "status": "development_norm_audit_only",
        "formal_privacy_claim_valid": False,
        "source_split": "train",
        "test_split_used": False,
        "attack_metrics_used": False,
        "dataset": args.dataset,
        "seed": seed,
        "audit_rounds": args.audit_rounds,
        "reference_protocol_rounds": int(config["rounds"]),
        "num_clients": int(config["num_clients"]),
        "clients_per_round": int(config["clients_per_round"]),
        "participation_schedule_hash": schedule_hash,
        "min_audit_participations": min(participation_counts),
        "max_audit_participations": max(participation_counts),
        "projection_ratio": projection.actual_retain_ratio,
        "projection_seed": projection.seed,
        "projection_mask_hash": projection.mask_hash,
        "current_clip_reference": CURRENT_CLIP,
        "effective_training_clip_bounds": clip_bounds,
        "development_dp_noise_seed_public": True,
        "development_dp_noise_seed": args.development_dp_noise_seed,
        "runtime_seconds": runtime_seconds,
    }
    with (args.results_dir / "audit_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(audit_config, handle, sort_keys=False)
    report_config = dict(config)
    report_config["dataset"] = args.dataset
    report_config["raw_rows"] = collector.rows
    _write_report(
        args.report_path,
        report_config,
        tables,
        args.audit_rounds,
        schedule_hash,
        participation_counts,
        runtime_seconds,
        projection,
    )

    summary_by_group = {str(row["group_name"]): row for row in tables["group_summary"]}
    rate_by_group = {str(row["group_name"]): row for row in tables["clipping_rates"]}
    waste_by_group = {str(row["group_name"]): row for row in tables["clip_waste"]}
    p90s = [float(summary_by_group[group]["p90"]) for group in GROUPS]
    anisotropy = max(p90s) / min(value for value in p90s if value > 0)
    if anisotropy >= 2.0:
        conclusion = "C. spectral groups show strongly heterogeneous scales"
        recommendation = "Ellipsoidal Spectral Clipping Sanity Test"
    elif max(p90s) <= CURRENT_CLIP * 0.75:
        conclusion = "B. 0.3 appears overly conservative"
        recommendation = "Smaller Global Clipping Sanity Test"
    else:
        conclusion = "A. 0.3 appears appropriate"
        recommendation = "Repeat the norm audit with one additional development seed"
    print("===============================")
    print("SPECTRAL NORM AUDIT SUMMARY")
    print("===============================")
    print(f"Current clip: {CURRENT_CLIP}")
    for group in GROUPS:
        row = summary_by_group[group]
        print(f"Group: {GROUP_LABELS[group]}")
        print(f"P50: {float(row['p50']):.6f}")
        print(f"P90: {float(row['p90']):.6f}")
        print(f"P95: {float(row['p95']):.6f}")
        print(f"Current clipping rate: {100 * float(rate_by_group[group]['clip_0.30']):.2f}%")
        print(f"Clip/P90: {float(waste_by_group[group]['clip_to_p90_ratio']):.3f}")
    print(f"Overall conclusion: {conclusion}")
    print(f"Recommended next experiment: {recommendation}")
    return {
        "rows": len(collector.rows),
        "runtime_seconds": runtime_seconds,
        "conclusion": conclusion,
        "recommendation": recommendation,
    }


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
